"""Index, discovery, prep orchestrator, batching.

One file by design — each piece is small and they share types and concepts
(Coding Rule 3). See docs/core/data.md for the canonical CSV format and
schemas.
"""

import csv
import json
import random
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
import pandas as pd

CANONICAL_VIEW_TYPES: list[str] = ["top", "side_0", "side_1", "side_2", "side_3"]

# Image extensions we'll pick up when walking class folders. Anything PIL can
# read is fine in DATA_DIR; this filter only avoids hidden / sidecar files.
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".ppm", ".gif"}

# Explicit camelCase ↔ snake_case mapping for the well-known index columns.
# Optional task-added columns (e.g. captureDate) round-trip via the generic
# converters below.
_DISK_TO_PYTHON = {
    "fileName": "file_name",
    "imageID": "image_id",
    "flowerID": "flower_id",
    "forkID": "fork_id",
    "viewID": "view_id",
    "viewType": "view_type",
    # `class`, `target`, `split` pass through unchanged.
}
_PYTHON_TO_DISK = {v: k for k, v in _DISK_TO_PYTHON.items()}


def _camel_to_snake(name: str) -> str:
    """Generic camelCase → snake_case for unknown task-added columns."""
    name = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name).lower()


def _snake_to_camel(name: str) -> str:
    """Generic snake_case → camelCase, inverse of `_camel_to_snake` for
    well-formed input. Note: collapses `imageID` ↔ `image_id` ↔ `imageId`
    on the unknown path; explicit mappings above keep the canonical
    columns ID-cased on disk."""
    head, *tail = name.split("_")
    return head + "".join(p.title() for p in tail)


def _disk_to_python(col: str) -> str:
    return _DISK_TO_PYTHON.get(col, _camel_to_snake(col))


def _python_to_disk(col: str) -> str:
    return _PYTHON_TO_DISK.get(col, _snake_to_camel(col))


# -----------------------------------------------------------------------------
# index.csv
# -----------------------------------------------------------------------------


def make_image_id(flower_id: str, view_id: int, fork_id: str = "") -> str:
    """Embedding cache key. `<flower_id>_<fork_id>_<view_id>` when fork_id
    is non-empty, else `<flower_id>_<view_id>`. Does **not** rename any
    image file — only used as the .npy cache filename stem."""
    if fork_id:
        return f"{flower_id}_{fork_id}_{view_id}"
    return f"{flower_id}_{view_id}"


def build_index(rows: list[dict], csv_path: Path) -> pd.DataFrame:
    """Write `rows` (snake_case keys) to `csv_path` as the canonical
    semicolon-separated, camelCase-headered CSV with a `sep=;` first line.
    Returns the in-memory DataFrame (still snake_case)."""
    df = pd.DataFrame(rows)
    df_disk = df.rename(columns=_python_to_disk)
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w") as f:
        f.write("sep=;\n")
        df_disk.to_csv(f, sep=";", index=False)
    return df


def read_index(csv_path: Path) -> pd.DataFrame:
    """Read a canonical CSV and rename headers to snake_case."""
    df = pd.read_csv(csv_path, sep=";", skiprows=1)
    return df.rename(columns=_disk_to_python)


def apply_label_corrections(ctx, index: pd.DataFrame) -> pd.DataFrame:
    """Apply human label corrections (`<data_dir>/<task>_changes.csv`) to an index.

    Corrections live **with the dataset**, not the run, so every run on that
    dataset (any backbone / future re-prepare) inherits the same relabels. The
    viewer's relabel tool writes one row per image with a corrected
    class. Here each changed flower's `class` and `target` are remapped **on a
    copy** (the new target is taken from the index's existing class→target
    mapping), leaving `split` and the cached embeddings untouched — embeddings are
    class-independent, so retraining picks up the corrected labels without
    re-extracting anything, and **no image file is ever modified**. Returns the
    input unchanged if no file exists.
    """
    path = ctx.data_dir() / f"{ctx.task}_changes.csv"
    if not path.is_file():
        return index

    new_by_flower: dict[str, str] = {}
    with open(path, newline="") as f:
        if not f.readline().lower().startswith("sep="):
            f.seek(0)
        for row in csv.DictReader(f, delimiter=";"):
            new_by_flower[row["flowerID"]] = row["newClass"]
    if not new_by_flower:
        return index

    target_of = {str(k): float(v) for k, v in index.groupby("class")["target"].first().items()}
    index = index.copy()
    # Class folders are often numeric ("1".."6"), so the column reads back as
    # int64; corrected classes are strings. Coerce to str first so the remap is
    # dtype-safe (pandas >= 3 rejects a str into an int64 column) and class stays
    # a categorical label, not a number.
    index["class"] = index["class"].astype(str)
    remapped = 0
    for flower_id, new_class in new_by_flower.items():
        if new_class not in target_of:
            continue  # only existing classes are valid targets
        mask = index["flower_id"].astype(str) == str(flower_id)
        if not mask.any():
            continue
        index.loc[mask, "class"] = new_class
        index.loc[mask, "target"] = target_of[new_class]
        remapped += int(mask.sum())
    if remapped:
        print(f"label corrections: {len(new_by_flower)} flower(s) → {remapped} rows remapped (splits kept)")
    return index


# -----------------------------------------------------------------------------
# discovery
# -----------------------------------------------------------------------------


def discover_class_folders(
    data_dir: Path,
    targets: dict[str, float] | None = None,
) -> list[dict]:
    """Walk class folders under `data_dir` and yield one row per image.

    Each subdirectory of `data_dir` is treated as a class. Files inside
    each class folder become rows. `targets` maps folder names to floats
    in `[0, 1]`; if `None`, folder names are parsed as numbers and
    normalised to `[0, 1]` across the discovered classes (requires ≥ 2
    distinct classes).

    The default per-file row treats each image as one flower with a
    single top view (no forks). This is the trivial case; multi-view
    datasets need either pre-organisation or a custom discoverer.
    """
    data_dir = Path(data_dir).resolve()
    class_dirs = sorted(p for p in data_dir.iterdir() if p.is_dir())

    if targets is None:
        targets = _infer_targets([p.name for p in class_dirs])

    rows: list[dict] = []
    for class_dir in class_dirs:
        cls = class_dir.name
        if cls not in targets:
            raise KeyError(
                f"Class folder {cls!r} not in targets mapping (have {sorted(targets)!r})."
            )
        target = float(targets[cls])
        if not 0.0 <= target <= 1.0:
            raise ValueError(
                f"Target for class {cls!r} is {target}; must be in [0, 1]."
            )

        for img in sorted(class_dir.iterdir()):
            if not img.is_file() or img.suffix.lower() not in _IMAGE_EXTS:
                continue
            flower_id = img.stem
            fork_id = ""
            view_id = 0
            view_type = "top"
            rel = img.relative_to(data_dir)
            file_name = str(rel).replace("/", "\\")  # POSIX → Windows for disk
            rows.append(
                {
                    "file_name": file_name,
                    "image_id": make_image_id(flower_id, view_id, fork_id),
                    "flower_id": flower_id,
                    "fork_id": fork_id,
                    "view_id": view_id,
                    "view_type": view_type,
                    "class": cls,
                    "target": target,
                }
            )

    return rows


def discover_multiview(
    data_dir: Path,
    parse: Callable[[str], tuple[str, str] | None],
    views: list[str],
    targets: dict[str, float] | None = None,
) -> list[dict]:
    """Group images into multi-view flowers using a task-supplied parser.

    Each subdirectory of `data_dir` is a class. For every image, `parse` is
    called with the path relative to `data_dir` (POSIX separators) and must
    return `(flower_id, view_type)` — or `None` to skip the file. Rows whose
    `view_type` is not in `views` are dropped; `view_id` is the position of
    `view_type` within `views`; `fork_id` is empty (any rotation copies are
    encoded by the task inside `flower_id`).

    The task owns only the naming convention (`parse`). Everything reusable —
    target assignment, canonical `view_id`, and (in `run`) the flower-level
    split — stays in core, so multi-view tasks can't accidentally leak views
    across splits.
    """
    data_dir = Path(data_dir).resolve()
    class_dirs = sorted(p for p in data_dir.iterdir() if p.is_dir())
    if targets is None:
        targets = _infer_targets([p.name for p in class_dirs])

    rows: list[dict] = []
    for class_dir in class_dirs:
        cls = class_dir.name
        if cls not in targets:
            raise KeyError(
                f"Class folder {cls!r} not in targets mapping (have {sorted(targets)!r})."
            )
        target = float(targets[cls])
        if not 0.0 <= target <= 1.0:
            raise ValueError(f"Target for class {cls!r} is {target}; must be in [0, 1].")

        for img in sorted(class_dir.iterdir()):
            if not img.is_file() or img.suffix.lower() not in _IMAGE_EXTS:
                continue
            rel = img.relative_to(data_dir)
            parsed = parse(str(rel).replace("\\", "/"))
            if parsed is None:
                continue
            flower_id, view_type = parsed
            if view_type not in views:
                continue
            view_id = views.index(view_type)
            rows.append(
                {
                    "file_name": str(rel).replace("/", "\\"),
                    "image_id": make_image_id(flower_id, view_id, ""),
                    "flower_id": flower_id,
                    "fork_id": "",
                    "view_id": view_id,
                    "view_type": view_type,
                    "class": cls,
                    "target": target,
                }
            )
    return rows


def _infer_targets(class_names: list[str]) -> dict[str, float]:
    """Parse numeric folder names and normalise to `[0, 1]`. Requires ≥ 2
    distinct numeric class names."""
    try:
        values = {c: float(c) for c in class_names}
    except ValueError as e:
        raise ValueError(
            "targets=None requires numeric class folder names; "
            f"got non-numeric value: {e}. Pass targets={{...}} explicitly."
        ) from None
    if len(set(values.values())) < 2:
        raise ValueError(
            "targets=None requires ≥ 2 distinct numeric class folders for normalisation; "
            "pass targets={...} explicitly."
        )
    vmin = min(values.values())
    vmax = max(values.values())
    return {c: (v - vmin) / (vmax - vmin) for c, v in values.items()}


# -----------------------------------------------------------------------------
# splits
# -----------------------------------------------------------------------------


def _assign_splits(
    rows: list[dict], val_ratio: float, test_ratio: float, seed: int
) -> None:
    """Mutate `rows` in place, assigning `split` ∈ {train, val, test}.

    Splits are at `flower_id` level — every row of a given flower lands
    in the same split. Prevents leakage through rotation/fork copies.
    """
    flower_ids = sorted({r["flower_id"] for r in rows})
    rng = random.Random(seed)
    rng.shuffle(flower_ids)

    n = len(flower_ids)
    n_test = int(round(n * test_ratio))
    n_val = int(round(n * val_ratio))

    test_set = set(flower_ids[:n_test])
    val_set = set(flower_ids[n_test : n_test + n_val])

    for r in rows:
        fid = r["flower_id"]
        if fid in test_set:
            r["split"] = "test"
        elif fid in val_set:
            r["split"] = "val"
        else:
            r["split"] = "train"


# -----------------------------------------------------------------------------
# orchestrator
# -----------------------------------------------------------------------------


def run(
    ctx,
    data_dir: Path,
    views: list[str],
    targets: dict[str, float] | None = None,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    discover: Callable[..., list[dict]] | None = None,
    extra: dict | None = None,
) -> pd.DataFrame:
    """End-to-end prep. Discovers rows under `data_dir`, assigns splits at
    `flower_id` level, and writes `<run>/prep/{index.csv, info.json}`. Does
    **not** copy images.

    `discover(data_dir, targets=...)` overrides the row scanner. Defaults to
    the single-view `discover_class_folders`; multi-view tasks pass a binding
    of `discover_multiview` (with their filename `parse`). The flower-level
    split runs on whatever rows come back, regardless of scanner.

    `info.json` is the run manifest: it records the run identity and backbone
    (from `ctx`), the absolute `data_dir` so downstream stages
    (`embeddings.extract`) can resolve `fileName` columns against the right
    root, and any `extra` keys the task snapshots (e.g. its training config)
    so `train`/`export` can read everything from the run alone.
    """
    data_dir = Path(data_dir).resolve()
    discover = discover or discover_class_folders
    rows = discover(data_dir, targets=targets)
    _assign_splits(rows, val_ratio, test_ratio, seed)

    df = build_index(rows, ctx.index_csv)
    _write_info_json(ctx, data_dir, views, targets, df, extra)
    return df


def _write_info_json(
    ctx,
    data_dir: Path,
    views: list[str],
    targets: dict[str, float] | None,
    df: pd.DataFrame,
    extra: dict | None = None,
) -> None:
    info = {
        "date": ctx.date,
        "cultivar": ctx.cultivar,
        "backbone_name": ctx.backbone_name,
        "backbone_checkpoint": ctx.backbone_checkpoint,
        "task": ctx.task,
        "data_dir": str(data_dir),
        "created": datetime.now(timezone.utc).isoformat(),
        "views": views,
        "targets": targets,
        "class_distribution": {str(k): int(v) for k, v in df["class"].value_counts().items()},
        "per_split_counts": {str(k): int(v) for k, v in df["split"].value_counts().items()},
        **(extra or {}),
    }
    ctx.prep_info_json.parent.mkdir(parents=True, exist_ok=True)
    ctx.prep_info_json.write_text(json.dumps(info, indent=2))


# -----------------------------------------------------------------------------
# batching
# -----------------------------------------------------------------------------


def batches(
    samples: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    shuffle: bool = True,
    seed: int = 42,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """In-memory batch iterator. Embeddings already fit in RAM, so no
    `DataLoader`/`Dataset` machinery. Yields `(xb, yb)` pairs; the last
    batch may be smaller than `batch_size`."""
    n = len(samples)
    indices = np.arange(n)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

    for start in range(0, n, batch_size):
        idx = indices[start : start + batch_size]
        yield samples[idx], labels[idx]
