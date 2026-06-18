"""prepare.py — define a ripeness-us run and freeze its manifest.

The one configurator. Picks the dataset (`--data_dir`) and backbone
(`--backbone`), scans the dataset with the task's `discover`, writes
`<run>/prep/{index.csv, info.json}`, and renders a dataset-distribution figure.
`info.json` is the full run manifest: data path, backbone + checkpoint, splits,
and a snapshot of the training config from `config.py`. After this,
`train`/`export` need only `-run <run>` — they read everything from the
manifest. Splits, CSV write, and the manifest all live in `core.data`.

Unlike `ripeness-trs/`, this dataset is SINGLE-VIEW: one crop per bud, with labels in
a `labels.csv` (not class folders). `discover` reads that CSV and emits one row
per image as a single `top` view.
"""

import argparse
import csv
import json
import shutil
from pathlib import Path

from core import data, plotting
from core.data import make_image_id
from core.run_context import RunContext

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

# CSV class labels remapped before assigning targets. Class 6 has only a couple of
# samples in this dataset, so it is merged into 5.
_CLASS_REMAP = {"6": "5"}


def _read_labels(csv_path: Path) -> dict[str, tuple[str, str]]:
    """Read `labels.csv` → `{filename: (class, cultivar)}`.

    The file is UTF-8 with a BOM and a leading `sep=;` line, then a
    `Filename;LabelIndex;Cultivar` header (the `Cultivar` column is written by
    `link_cultivars.py`). Class labels come back as strings, with `_CLASS_REMAP`
    already applied (so class 6 → 5). The cultivar is metadata — a tag of which
    plant the bud came from, used only for the viewer filter and the
    leakage-safe split, never as the target. Missing/empty cultivar is fatal:
    re-run `link_cultivars.py` against the matching split-views dataset.
    """
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        if not f.readline().lower().startswith("sep="):
            f.seek(0)
        labels: dict[str, tuple[str, str]] = {}
        for row in csv.DictReader(f, delimiter=";"):
            if not row.get("Cultivar", "").strip():
                raise ValueError(
                    f"{csv_path} row {row['Filename']!r} has no Cultivar; "
                    "run link_cultivars.py against the split-views dataset first."
                )
            name, cls = row["Filename"].strip(), row["LabelIndex"].strip()
            labels[name] = (_CLASS_REMAP.get(cls, cls), row["Cultivar"].strip())
    return labels


def _fork(file_name: str) -> str:
    """The fork/tray position from a crop filename `<date>-<time>-<fork>-<cam>...`.

    The fork is the machine position the flower was imaged at; combined with the
    cultivar it identifies one physical flower (so its rounds, if any, group
    together for the split). Raises if the name doesn't have the expected
    hyphen-delimited shape — better to fail prepare than to mis-group the split.
    """
    parts = Path(file_name).name.split("-")
    if len(parts) < 4:
        raise ValueError(f"cannot parse fork from filename {file_name!r} (expected <date>-<time>-<fork>-...)")
    return parts[2]


def discover(data_dir, targets):
    """Scan the single-view dataset → one index row per labelled crop.

    Reads `<data_dir>/labels.csv` for the `{filename: (class, cultivar)}` mapping,
    then emits one row per image: each bud is its own flower with a single `top`
    view and no rounds. `targets` maps class label → regression target in [0, 1];
    rows whose class is not in `targets` (after the remap) are skipped and
    counted, as are CSV entries missing from disk. Each row carries `cultivar`
    (the plant tag, for the viewer filter + stratified split) and `fork` (the
    tray position, parsed from the filename, for the leakage-safe grouping) —
    neither is the target. This is the only task-specific piece; core handles
    splits, the manifest, and batching.
    """
    data_dir = Path(data_dir).resolve()
    if targets is None:
        raise ValueError("ripeness-us needs an explicit TARGETS dict (got None).")
    labels = _read_labels(data_dir / "labels.csv")

    rows: list[dict] = []
    skipped_class = skipped_missing = 0
    for file_name, (cls, cultivar) in sorted(labels.items()):
        if cls not in targets:
            skipped_class += 1
            continue
        img = data_dir / file_name
        if not img.is_file() or img.suffix.lower() not in _IMAGE_EXTS:
            skipped_missing += 1
            continue
        flower_id = img.stem
        rows.append({
            "file_name": file_name.replace("/", "\\"),
            "image_id": make_image_id(flower_id, 0, ""),
            "flower_id": flower_id,
            "round_id": "",
            "view_id": 0,
            "view_type": "top",
            "class": cls,
            "target": float(targets[cls]),
            "cultivar": cultivar,
            "fork": _fork(file_name),
        })
    if skipped_class:
        print(f"skipped {skipped_class} row(s) whose class is not in TARGETS {sorted(targets)!r}")
    if skipped_missing:
        print(f"skipped {skipped_missing} CSV row(s) with no matching image file on disk")
    if not rows:
        raise ValueError(f"no labelled images discovered under {data_dir}")
    return rows


def _clear_stale_cache(ctx, data_dir: Path) -> None:
    """Drop stale embeddings + trained outputs if the dataset changed.

    A run dir is named by date+cultivar+backbone, so re-preparing the same
    identity against a *different* dataset (e.g. a newer capture) would leave
    `emb/` and the trained head pointing at the old images. When the manifest's
    recorded `data_dir` differs from the new one, wipe `emb/` and the task dir
    so prep, embeddings, and the head stay consistent.
    """
    if not ctx.prep_info_json.is_file():
        return
    old = json.loads(ctx.prep_info_json.read_text()).get("data_dir")
    if old and Path(old) != data_dir:
        for d in (ctx.embeddings_dir, ctx.task_dir):
            if d.exists():
                shutil.rmtree(d)
        print(f"dataset changed ({old} -> {data_dir}); cleared stale emb/ + {ctx.task}/")


def _config_snapshot(config) -> dict:
    """The training config frozen into the manifest for train/export to read."""
    return {
        "head_spec": {
            "aggregator_name": config.HEAD_SPEC.aggregator_name,
            "hidden_dims": list(config.HEAD_SPEC.hidden_dims),
            "dropout": config.HEAD_SPEC.dropout,
        },
        "hparams": config.HPARAMS,
        "opt_search_space": config.OPT_SEARCH_SPACE,
        "opt_n_trials": config.OPT_N_TRIALS,
    }


def _compare_grid(config, selected) -> dict[str, list]:
    """Build the frozen comparison grid `{dim: [values…]}` from `--compare`.

    A picked dim contributes all its `values`; an unpicked dim contributes only its
    `default`. The cross-product of these lists is the set of variants `train.py`
    fits. Empty selection → every dim a singleton → 1 variant.
    """
    selected = set(selected or [])
    unknown = selected - set(config.COMPARE_DIMS)
    if unknown:
        raise ValueError(f"--compare {sorted(unknown)} not in COMPARE_DIMS {sorted(config.COMPARE_DIMS)}")
    return {
        dim: (spec["values"] if dim in selected else [spec["default"]])
        for dim, spec in config.COMPARE_DIMS.items()
    }


def main(config, data_dir, cultivar, backbone=None,
         val_ratio=0.15, test_ratio=0.15, seed=None, compare=None):
    """Scan the dataset → write `<run>/prep/{index.csv, info.json}`; return the run dir.

    `data_dir` is the raw dataset folder; its absolute path is recorded in the
    manifest so downstream stages resolve it without it being passed again.
    `backbone` selects the checkpoint from `config.BACKBONE_CHECKPOINTS`
    (defaults to `config.BACKBONE_NAME`); `cultivar` names the run (required).
    The view set is fixed to `config.VIEWS` (single `top`).
    """
    backbone = backbone or config.BACKBONE_NAME
    ctx = RunContext(
        date=config.DATE,
        cultivar=cultivar,
        backbone_name=backbone,
        task=config.TASK,
        backbone_checkpoint=config.BACKBONE_CHECKPOINTS[backbone],
        output_dir=config.OUTPUT_DIR,
    )
    _clear_stale_cache(ctx, Path(data_dir).resolve())
    df = data.run(
        ctx,
        data_dir=data_dir,
        views=list(config.VIEWS),
        targets=config.TARGETS,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed if seed is not None else config.HPARAMS["seed"],
        discover=discover,
        incomplete_tolerance=config.INCOMPLETE_TOLERANCE,
        # Group a physical flower's crops (cultivar+fork) into one split so no
        # flower leaks across train/val/test; stratify by cultivar so every
        # cultivar is represented in all three splits.
        group_key=["cultivar", "fork"],
        stratify_key="cultivar",
        extra={**_config_snapshot(config), "compare": _compare_grid(config, compare)},
    )
    plotting.plot_dataset_distribution(ctx, df)
    return ctx.root


if __name__ == "__main__":
    import config as cfg

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", "-d", dest="data_dir", required=True,
                    help="path to the raw dataset folder")
    ap.add_argument(
        "--backbone", "-bkb", default=cfg.BACKBONE_NAME, choices=list(cfg.BACKBONE_CHECKPOINTS),
        help="backbone to use for this run",
    )
    ap.add_argument("--cultivar", "-c", required=True, help="run/cultivar name")
    ap.add_argument("--compare", nargs="*", default=[], choices=list(cfg.COMPARE_DIMS),
                    help="dims to compare (cross-product); omitted dims use their default")
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--test_ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int)
    a = ap.parse_args()

    run = main(
        cfg, data_dir=a.data_dir, backbone=a.backbone, cultivar=a.cultivar,
        val_ratio=a.val_ratio, test_ratio=a.test_ratio, seed=a.seed, compare=a.compare,
    )
    print(f"prepared run: {run}")
