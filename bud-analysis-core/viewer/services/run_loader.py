"""Load a prepared/trained run and compute per-view predictions.

Reads a run's `index.csv`, its cached per-image embeddings, and a trained head
(`<task>/<aggregator>/head.pt`), then runs the head on **every image** to get a
per-view prediction. The viewer aggregates these client-side (per view, per
fork = MIL mean, per flower). All paths flow through `core.RunContext`.
"""

import csv
import functools
import json
import sys
from pathlib import Path

import numpy as np
import torch

# The viewer lives inside the core repo (bud-analysis-core/viewer); make the
# `core` package importable when run from source, without relying on an install.
_REPO = Path(__file__).resolve().parents[2]          # bud-analysis-core/
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from core import data, heads  # noqa: E402
from core.run_context import RunContext  # noqa: E402
from core.schemas import HeadSpec  # noqa: E402

OUTPUT_DIR = _REPO.parent / "output"                 # Bud-Analysis/output, sibling of core


def _s(v) -> str:
    """Stringify a cell, collapsing pandas NaN/float-int noise ('0.0' → '0')."""
    if v is None or (isinstance(v, float) and v != v):
        return ""
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


def list_runs() -> list[dict]:
    """Every run under output/ that has a manifest, with its aggregators.

    Each run dir is `output/<run>/` with its task results in `<run>/<task>-results/`.
    """
    runs = []
    if not OUTPUT_DIR.exists():
        return runs
    for run_dir in sorted(OUTPUT_DIR.iterdir()):
        if not (run_dir / "prep" / "info.json").is_file():
            continue
        try:
            ctx = RunContext.from_info_json(str(run_dir))
            aggs = _aggregators(ctx)
        except Exception:
            continue
        runs.append({"name": run_dir.name, "task": ctx.task, "aggregators": aggs})
    return runs


def _aggregators(ctx: RunContext) -> list[str]:
    """Trained aggregator dirs (have head.pt + metrics.json) under the task."""
    if ctx.task is None or not ctx.task_dir.exists():
        return []
    return sorted(
        d.name for d in ctx.task_dir.iterdir()
        if d.is_dir() and (d / "head.pt").exists() and (d / "metrics.json").exists()
    )


@functools.lru_cache(maxsize=8)
def load_records(run: str, aggregator: str) -> dict:
    """Per-image records for one run + aggregator head (cached in-process).

    Returns {task, aggregator, data_dir, classes, views, forks, records}, where
    each record carries flower/fork/view identity, the true target, the head's
    per-view prediction, split, and the image's relative file name.
    """
    ctx = RunContext.from_info_json(str(OUTPUT_DIR / run))
    df = data.read_index(ctx.index_csv)

    agg_dir = ctx.task_dir / aggregator
    metrics = json.loads((agg_dir / "metrics.json").read_text())
    hs = metrics["head_spec"]
    spec = HeadSpec(hs["aggregator_name"], tuple(hs["hidden_dims"]), hs["dropout"])
    head = heads.build(spec, ctx.backbone_name)
    head.load_state_dict(torch.load(agg_dir / "head.pt", map_location="cpu", weights_only=True))
    head.eval()

    rows = df.to_dict("records")
    missing = [r["image_id"] for r in rows if not ctx.embedding_path(r["image_id"]).is_file()]
    if missing:
        raise ValueError(
            f"{len(missing)}/{len(rows)} embeddings are missing for run {run!r} "
            f"(e.g. {missing[0]}). The run's emb/ cache does not match its current "
            "prep/index.csv — re-run prepare + train on one dataset so prep, "
            "embeddings, and the trained head are consistent."
        )
    embs = np.stack([np.load(ctx.embedding_path(r["image_id"])) for r in rows])
    with torch.no_grad():
        preds = head(torch.tensor(embs, dtype=torch.float32)).squeeze(-1).numpy()

    records = []
    for r, p in zip(rows, preds):
        flower_id = _s(r["flower_id"])
        # flower_id is "<class>_<seq>"; seq is the persistent bud/fork number.
        fork = flower_id.rsplit("_", 1)[-1]
        records.append({
            "image_id": _s(r["image_id"]),
            "flower_id": flower_id,
            "fork": fork,                  # real fork number (the filename <seq>)
            "round": _s(r["fork_id"]),     # 0..N repeated round (capture) of the same fork
            "view_id": int(r["view_id"]),
            "view_type": _s(r["view_type"]),
            "klass": _s(r["class"]),
            "true": float(r["target"]),
            "pred": float(p),
            "split": _s(r["split"]),
            "file_name": _s(r["file_name"]).replace("\\", "/"),
        })

    return {
        "task": ctx.task,
        "aggregator": aggregator,
        "data_dir": str(ctx.data_dir()),
        "classes": sorted({rec["klass"] for rec in records}, key=lambda c: (len(c), c)),
        "views": sorted({rec["view_type"] for rec in records},
                        key=lambda v: next(r["view_id"] for r in records if r["view_type"] == v)),
        "forks": sorted({int(rec["fork"]) for rec in records}),
        "records": records,
    }


def resolve_image(run: str, rel_file: str) -> Path | None:
    """Resolve a record's `file_name` to an absolute image path under data_dir.

    Guards against path traversal — the resolved path must stay inside data_dir.
    """
    ctx = RunContext.from_info_json(str(OUTPUT_DIR / run))
    data_dir = Path(ctx.data_dir()).resolve()
    target = (data_dir / rel_file.replace("\\", "/")).resolve()
    if data_dir not in target.parents or not target.is_file():
        return None
    return target


# ---- label corrections (relabel tool) -------------------------------------
# Human-corrected class labels for a run live in output/<run>/ripeness_changes.csv,
# one row per view (image), consumed later by prepare to remap class -> target.
_CHANGES_FIELDS = ["fileName", "flowerID", "fork", "oldClass", "newClass"]


def changes_path(run: str) -> Path:
    """Path to a run's label-corrections CSV (`<task>_changes.csv`, may not exist)."""
    ctx = RunContext.from_info_json(str(OUTPUT_DIR / run))
    return ctx.root / f"{ctx.task}_changes.csv"


def load_changes(run: str) -> dict[str, str]:
    """Return `{flower_id: new_class}` from a saved corrections CSV (empty if none)."""
    path = changes_path(run)
    if not path.is_file():
        return {}
    out: dict[str, str] = {}
    with open(path, newline="") as f:
        if not f.readline().lower().startswith("sep="):
            f.seek(0)
        for row in csv.DictReader(f, delimiter=";"):
            out[row["flowerID"]] = row["newClass"]
    return out


def save_changes(run: str, changes: list[dict]) -> dict:
    """Write per-image corrections for `changes` (a list of `{flower_id, new_class}`).

    Each changed flower is expanded to one row per image via the run's index, so
    the CSV points at concrete files with their old and new class.
    """
    new_by_flower = {str(c["flower_id"]): str(c["new_class"]) for c in changes}
    ctx = RunContext.from_info_json(str(OUTPUT_DIR / run))
    df = data.read_index(ctx.index_csv)
    rows = []
    for r in df.to_dict("records"):
        fid = _s(r["flower_id"])
        if fid in new_by_flower:
            rows.append({
                "fileName": _s(r["file_name"]).replace("/", "\\"),
                "flowerID": fid,
                "fork": fid.rsplit("_", 1)[-1],
                "oldClass": _s(r["class"]),
                "newClass": new_by_flower[fid],
            })
    path = changes_path(run)
    with open(path, "w", newline="") as f:
        f.write("sep=;\n")
        writer = csv.DictWriter(f, fieldnames=_CHANGES_FIELDS, delimiter=";")
        writer.writeheader()
        writer.writerows(rows)
    return {"flowers": len(new_by_flower), "rows": len(rows), "path": str(path)}
