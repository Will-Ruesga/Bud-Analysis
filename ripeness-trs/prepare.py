"""prepare.py — define a ripeness run and freeze its manifest.

The one configurator. Picks the dataset (`--data_dir`) and backbone
(`--backbone`), scans the dataset with the task's `discover`, writes
`<run>/prep/{index.csv, info.json}`, and renders a dataset-distribution figure.
`info.json` is the full run manifest: data path, backbone + checkpoint, splits,
and a snapshot of the training config from `config.py`. After this,
`train`/`export` need only `--run <run>` — they read everything from the
manifest. Splits, CSV write, and the manifest all live in `core.data`.
"""

import argparse
import functools
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path

from core import data, plotting
from core.data import make_image_id
from core.run_context import RunContext

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def _parse_name(stem: str) -> tuple[str, str, int] | None:
    """Decode one filename stem → `(seq, timestamp, view_index)`.

    Naming: `<date>-<time>-<seq>-<sub>_<view>` (e.g.
    `2026_05_22-13_12_43_540-6-0_0`). `seq` is the bud's number within its
    class (→ physical flower), the `<date>-<time>` timestamp distinguishes
    repeated captures (rounds) of that bud, `<sub>` is a constant 0, and the
    trailing `_<n>` is the camera view (0..4). Returns None for names that
    don't match.
    """
    rest, sep, view = stem.rpartition("_")
    if not sep or not view.isdigit():
        return None
    parts = rest.rsplit("-", 2)  # [<date>-<time>, <seq>, <sub>]
    if len(parts) != 3 or not parts[1].isdigit():
        return None
    timestamp, seq, _sub = parts
    return seq, timestamp, int(view)


def parse_views(spec: str) -> dict[int, str]:
    """Parse a view spec → `{page_index: view_name}`. Same syntax as the splitter.

    `"side: 0 1 2 3, top: 4"` → `{0:'side_0',1:'side_1',2:'side_2',3:'side_3',4:'top'}`.
    Each `name: i j k` assigns those filename page indices to a group; a group
    with one index keeps the bare name, several get `<name>_<k>` suffixes.
    Indices not listed are dropped.
    """
    mapping: dict[int, str] = {}
    for part in re.split(r",\s*", spec.strip()):
        part = part.strip()
        if not part:
            continue
        m = re.match(r"(\w+)\s*:\s*([\d\s]+)$", part)
        if not m:
            raise ValueError(f"bad view group {part!r} — expected 'name: 0 1 2'")
        name, idxs = m.group(1), [int(x) for x in m.group(2).split()]
        for k, idx in enumerate(idxs):
            mapping[idx] = name if len(idxs) == 1 else f"{name}_{k}"
    if not mapping:
        raise ValueError("no view groups parsed")
    return mapping


def _infer_targets(class_names: list[str]) -> dict[str, float]:
    """Numeric class folder names → min-max normalised targets in [0, 1]."""
    try:
        values = {c: float(c) for c in class_names}
    except ValueError as e:
        raise ValueError(
            f"TARGETS=None needs numeric class folder names; got {e}. "
            "Set an explicit TARGETS dict in config.py."
        ) from None
    lo, hi = min(values.values()), max(values.values())
    if hi == lo:
        raise ValueError("TARGETS=None needs ≥ 2 distinct numeric classes.")
    return {c: (v - lo) / (hi - lo) for c, v in values.items()}


def discover(data_dir, targets, views):
    """Scan the dataset → index rows grouped flower=(class, seq), round=capture.

    Each class folder holds repeated captures of numbered buds; each capture is
    a full set of camera angles. Physical flower = `<class>_<seq>`; rounds are the
    repeated captures of that flower, ranked chronologically by timestamp
    (0, 1, ...). `views` maps each filename page index → view name
    (`{4: 'top', 0: 'side_0', ...}`); unmapped indices are dropped. This is the
    only task-specific piece — core handles splits, targets, and the manifest.
    """
    data_dir = Path(data_dir).resolve()
    class_dirs = sorted(p for p in data_dir.iterdir() if p.is_dir())
    if targets is None:
        targets = _infer_targets([p.name for p in class_dirs])
    # (class, seq) -> {timestamp -> {view_index -> rel_path}}
    by_flower: dict = defaultdict(lambda: defaultdict(dict))
    for class_dir in class_dirs:
        cls = class_dir.name
        if cls not in targets:
            raise KeyError(f"class folder {cls!r} not in TARGETS {sorted(targets)!r}")
        for img in sorted(class_dir.iterdir()):
            if not img.is_file() or img.suffix.lower() not in _IMAGE_EXTS:
                continue
            parsed = _parse_name(img.stem)
            if parsed is None:
                continue
            seq, timestamp, view_index = parsed
            if view_index not in views:
                continue
            rel = str(img.relative_to(data_dir)).replace("/", "\\")
            by_flower[(cls, seq)][timestamp][view_index] = rel

    rows: list[dict] = []
    # Emit every parsed capture; core's `_enforce_views` validates completeness
    # against the declared views and drops/raises per INCOMPLETE_TOLERANCE. Forks
    # are numbered chronologically per flower.
    for (cls, seq), captures in by_flower.items():
        target = float(targets[cls])
        flower_id = f"{cls}_{seq}"
        for round_id, timestamp in enumerate(sorted(captures)):
            for view_index, file_name in sorted(captures[timestamp].items()):
                rows.append({
                    "file_name": file_name,
                    "image_id": make_image_id(flower_id, view_index, str(round_id)),
                    "flower_id": flower_id,
                    "round_id": str(round_id),
                    "view_id": view_index,
                    "view_type": views[view_index],
                    "class": cls,
                    "target": target,
                })
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
    `default` (a single value). The cross-product of these lists is the set of
    variants `train.py` will fit. Empty selection → every dim a singleton → 1 variant.
    """
    selected = set(selected or [])
    unknown = selected - set(config.COMPARE_DIMS)
    if unknown:
        raise ValueError(f"--compare {sorted(unknown)} not in COMPARE_DIMS {sorted(config.COMPARE_DIMS)}")
    return {
        dim: (spec["values"] if dim in selected else [spec["default"]])
        for dim, spec in config.COMPARE_DIMS.items()
    }


def main(config, data_dir, cultivar, backbone=None, views=None,
         val_ratio=0.15, test_ratio=0.15, seed=None, compare=None):
    """Scan the dataset → write `<run>/prep/{index.csv, info.json}`; return the run dir.

    `data_dir` is the raw dataset folder; its absolute path is recorded in the
    manifest so downstream stages resolve it without it being passed again.
    `backbone` selects the checkpoint from `config.BACKBONE_CHECKPOINTS`
    (defaults to `config.BACKBONE_NAME`); `cultivar` names the run (required).
    `views` is a `{page_index: view_name}` map (see `parse_views`); when None it
    falls back to `config.VIEWS` (positional).
    """
    backbone = backbone or config.BACKBONE_NAME
    view_map = views if views is not None else {i: v for i, v in enumerate(config.VIEWS)}
    view_names = [view_map[i] for i in sorted(view_map)]
    ctx = RunContext(
        date=config.DATE,
        cultivar=cultivar,
        backbone_name=backbone,
        task=config.TASK,
        backbone_checkpoint=config.BACKBONE_CHECKPOINTS[backbone],
        output_dir=config.OUTPUT_DIR,
    )
    _clear_stale_cache(ctx, Path(data_dir).resolve())
    scan = functools.partial(discover, views=view_map)
    df = data.run(
        ctx,
        data_dir=data_dir,
        views=view_names,
        targets=config.TARGETS,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed if seed is not None else config.HPARAMS["seed"],
        discover=scan,
        incomplete_tolerance=config.INCOMPLETE_TOLERANCE,
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
    ap.add_argument("--views", "-v", default=None,
                    help="view groups, e.g. 'side: 0 1 2 3, top: 4'; omit to be prompted")
    ap.add_argument("--compare", nargs="*", default=[], choices=list(cfg.COMPARE_DIMS),
                    help="dims to compare (cross-product); omitted dims use their default")
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--test_ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int)
    a = ap.parse_args()

    spec = a.views
    if spec is None:  # not given on the CLI → ask, so the format is never forgotten
        print("Views — map each filename page index to a group.")
        print("  format:   name: i j k, name: i     (one-index group keeps the bare name)")
        print("  example:  side: 0 1 2 3, top: 4")
        print(f"  (empty = config.py default: {cfg.VIEWS})")
        spec = input("Views: ").strip() or None
    view_map = parse_views(spec) if spec else {i: v for i, v in enumerate(cfg.VIEWS)}
    print("views: " + ", ".join(f"{i}→{view_map[i]}" for i in sorted(view_map)))

    run = main(
        cfg, data_dir=a.data_dir, backbone=a.backbone, cultivar=a.cultivar, views=view_map,
        val_ratio=a.val_ratio, test_ratio=a.test_ratio, seed=a.seed, compare=a.compare,
    )
    print(f"prepared run: {run}")
