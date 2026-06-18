"""Load a prepared/trained run and compute per-view predictions.

Reads a run's `index.csv`, its cached per-image embeddings, and a trained head
(`<task>-results/<variant>/head.pt`), then runs the head on **every image** to get a
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
    """Every run under output/ that has a manifest, with its trained variants.

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
            variants = _variants(ctx)
            axis = _compare_axis(ctx, variants)
        except Exception:
            continue
        runs.append({"name": run_dir.name, "task": ctx.task, "variants": variants, "axis": axis})
    return runs


def _variants(ctx: RunContext) -> list[str]:
    """Trained variant dirs (have head.pt + metrics.json) under the task.

    The dir names are the compared values — a single dim (`mse`/`huber`) or a
    composite of several (`mse-off`/`huber-on`). See `_compare_axis` for the label.
    """
    if ctx.task is None or not ctx.task_dir.exists():
        return []
    return sorted(
        d.name for d in ctx.task_dir.iterdir()
        if d.is_dir() and (d / "head.pt").exists() and (d / "metrics.json").exists()
    )


def _compare_axis(ctx: RunContext, variants: list[str]) -> str:
    """The dimension the study compared — what the viewer's variant filter is named.

    Read from any kept variant's `metrics.json` (`compare_axis`). Falls back for
    legacy runs that predate the key: `loss` if the metrics carry a loss name,
    else `aggregator` (the original compared dimension).
    """
    for name in variants:
        m = ctx.task_dir / name / "metrics.json"
        try:
            data = json.loads(m.read_text())
        except Exception:
            continue
        return data.get("compare_axis") or ("loss" if "loss" in data else "aggregator")
    return "aggregator"


@functools.lru_cache(maxsize=8)
def load_records(run: str, variant: str) -> dict:
    """Per-image records for one run + variant head (cached in-process).

    Returns {task, variant, data_dir, classes, views, forks, rounds, records}, where
    each record carries flower/fork/view identity, the true target, the head's
    per-view prediction, split, and the image's relative file name.
    """
    ctx = RunContext.from_info_json(str(OUTPUT_DIR / run))
    df = data.read_index(ctx.index_csv)

    vdir = ctx.task_dir / variant
    metrics = json.loads((vdir / "metrics.json").read_text())
    hs = metrics["head_spec"]
    spec = HeadSpec(hs["aggregator_name"], tuple(hs["hidden_dims"]), hs["dropout"])
    head = heads.build(spec, ctx.backbone_name)
    head.load_state_dict(torch.load(vdir / "head.pt", map_location="cpu", weights_only=True))
    head.eval()

    # `fork`/`cultivar` are explicit index columns for single-view tasks; multi-view
    # runs encode the fork inside flower_id ("<class>_<fork>") and carry no cultivar.
    has_fork_col = "fork" in df.columns
    has_cultivar_col = "cultivar" in df.columns
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
        # Prefer the explicit fork column; else flower_id is "<class>_<fork>".
        fork = _s(r["fork"]) if has_fork_col else flower_id.rsplit("_", 1)[-1]
        records.append({
            "image_id": _s(r["image_id"]),
            "flower_id": flower_id,
            "fork": fork,                  # tray position
            "cultivar": _s(r["cultivar"]) if has_cultivar_col else "",  # plant tag (metadata, not class)
            "round": _s(r["round_id"]) or "0",   # capture index; single-round data has none → 0
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
        "variant": variant,
        "data_dir": str(ctx.data_dir()),
        "classes": sorted({rec["klass"] for rec in records}, key=lambda c: (len(c), c)),
        "views": sorted({rec["view_type"] for rec in records},
                        key=lambda v: next(r["view_id"] for r in records if r["view_type"] == v)),
        # forks/rounds are numeric lists for the filters; skip any non-integer id so
        # single-image datasets (no real fork/round) don't crash the load.
        "forks": sorted({int(rec["fork"]) for rec in records if rec["fork"].lstrip("-").isdigit()}),
        "rounds": sorted({int(rec["round"]) for rec in records if rec["round"].lstrip("-").isdigit()}),
        # cultivars are free-text plant tags (may be absent on multi-view runs).
        "cultivars": sorted({rec["cultivar"] for rec in records if rec["cultivar"]}),
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
# Human-corrected class labels live WITH THE DATASET, at
# <data_dir>/<task>_changes.csv, one row per view (image). Keeping them beside
# the dataset (not the run) makes corrections the dataset's source of truth, so
# every run on it — any backbone / future re-prepare — inherits the
# same relabels. The dataset images are never modified; only this sidecar CSV is
# read by `core.data.apply_label_corrections` to remap class -> target at train.
_CHANGES_FIELDS = ["fileName", "flowerID", "fork", "oldClass", "newClass"]


def changes_path(run: str) -> Path:
    """Path to the dataset's label-corrections CSV (`<task>_changes.csv`, may not exist)."""
    ctx = RunContext.from_info_json(str(OUTPUT_DIR / run))
    return ctx.data_dir() / f"{ctx.task}_changes.csv"


def _read_changes_rows(path: Path) -> list[dict]:
    """All rows of a corrections CSV as dicts (empty list if the file is absent)."""
    if not path.is_file():
        return []
    with open(path, newline="") as f:
        if not f.readline().lower().startswith("sep="):
            f.seek(0)
        return list(csv.DictReader(f, delimiter=";"))


def load_changes(run: str) -> dict[str, str]:
    """Return `{flower_id: new_class}` from the dataset's corrections CSV (empty if none)."""
    return {row["flowerID"]: row["newClass"] for row in _read_changes_rows(changes_path(run))}


def save_changes(run: str, changes: list[dict]) -> dict:
    """Merge per-image corrections for `changes` (a list of `{flower_id, new_class}`)
    into the dataset's `<task>_changes.csv`.

    Each changed flower is expanded to one row per image via the run's index, so
    the CSV points at concrete files with their old and new class. Existing rows
    for flowers **not** in this request are preserved (corrections accumulate
    across runs on the same dataset); rows for flowers that **are** in this
    request are replaced with the current ones. No image file is touched.
    """
    new_by_flower = {str(c["flower_id"]): str(c["new_class"]) for c in changes}
    ctx = RunContext.from_info_json(str(OUTPUT_DIR / run))
    df = data.read_index(ctx.index_csv)
    new_rows = []
    for r in df.to_dict("records"):
        fid = _s(r["flower_id"])
        if fid in new_by_flower:
            new_rows.append({
                "fileName": _s(r["file_name"]).replace("/", "\\"),
                "flowerID": fid,
                "fork": fid.rsplit("_", 1)[-1],
                "oldClass": _s(r["class"]),
                "newClass": new_by_flower[fid],
            })
    path = changes_path(run)
    # Keep existing rows for flowers untouched by this request; replace the rest.
    kept = [row for row in _read_changes_rows(path) if row["flowerID"] not in new_by_flower]
    rows = kept + new_rows
    with open(path, "w", newline="") as f:
        f.write("sep=;\n")
        writer = csv.DictWriter(f, fieldnames=_CHANGES_FIELDS, delimiter=";")
        writer.writeheader()
        writer.writerows(rows)
    return {"flowers": len(new_by_flower), "rows": len(rows), "path": str(path)}
