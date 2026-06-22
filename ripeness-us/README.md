# ripeness-us

Single-view ripeness regression on the Universalsorter `BeursTrosrozen` bud
instances — one crop per bud, labels in a `labels.csv`. Same engine as `ripeness-trs/`
(frozen DINOv3 embeddings → MLP, Optuna MSE-vs-Huber sweep, regression to [0, 1]),
adapted for a single view. Concrete task on top of `base`; expects to
sit beside the core repo (`../base`).

## Setup (once)

The env lives in the task: **core package + this task's `requirements.txt`**.
Core's `pyproject.toml` owns the shared dependency set (torch/numpy/optuna/...);
`requirements.txt` holds only ripeness-specific deps and does **not** install
core — so install core editable first, then layer the task on top:

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e /path/to/base   # core package + its full dependency set
uv pip install -r requirements.txt             # task-specific deps (currently none)
```

Editable (`-e`) so local edits to core apply without reinstalling. `pytest` and
`onnxruntime` ship with core, so the install can run the tests as-is.

## Link cultivars (once per dataset)

The bud crops carry no cultivar; their source frames live in cultivar-named
subfolders of the *split-views* (full-flower) dataset. `link_cultivars.py` tags
each crop with its cultivar by matching `<frame>_b<x>_<y>.png` → `<frame>.png` →
the split-views subfolder, and rewrites `labels.csv` with a `Cultivar` column:

```bash
python link_cultivars.py --dataset /path/to/bud-instances \
                         --split-views /path/to/...-split-views
```

Re-run it whenever the crops are regenerated. The cultivar is **metadata only**
— a plant tag for the viewer filter and the leakage-safe split; it is *not* the
class/target (which stays the ripeness `LabelIndex` 1/3/5). `prepare` then
groups the split by `(cultivar, fork)` (a physical flower's crops never split
across train/val/test) and stratifies by `cultivar` (every cultivar in all three
splits); it errors if any crop is missing a cultivar.

## Run

Edit `config.py` (targets, backbone, hparams). `prepare` freezes a run;
every later stage just points `--run` at it and reads its manifest — no config:

```bash
python prepare.py --dataset /path/to/dataset --cultivar BeursTrosrozen --compare loss
#   --dataset/-d is the dataset folder; --cultivar/-c names the run (required) — it is both
#   the run dir <DATE>-<CULTIVAR>-<BACKBONE> and the label baked into the ONNX filename.
#   --backbone/-bkb to pick one. --compare picks the grid;
#   single view, so only `loss` is comparable (omit --compare for one default variant).
#   → writes output/<run>/prep/{index.csv, info.json, distribution.png}; prints the run dir
python train.py   --run output/<run>            # Optuna sweep → output/<run>/ripeness-results/<compare_value>/ + comparison.png
python export.py  --run output/<run> --variant auto   # → output/<run>/onnx/ripeness_<variant>_<backbone>.onnx
```

`<run>` is `<DATE>-<CULTIVAR>-<BACKBONE_NAME>` (printed by `prepare`). The `output/`
base is set by `OUTPUT_DIR` in `config.py`, which defaults to `../output` — a sibling
of this task and the core checkout (`Bud-Analysis/output/`), not nested inside either.

To re-train a run prepared days ago, just `train.py --run output/<that run>` — it reads
everything from the manifest. Tweak hyperparameters per run without re-preparing:
`train.py --run output/<run> --lr 5e-4 --epochs 50`.

Discovery: `prepare.discover()` reads `<data_dir>/labels.csv` (UTF-8 BOM, a leading
`sep=;` line, then `Filename;LabelIndex;Cultivar`) and emits one `top`-view row per
crop, each carrying its `cultivar` (the plant tag) and `fork` (tray position, parsed
from the filename). Class 6 is merged into 5 (`_CLASS_REMAP`); classes not in
`config.TARGETS` are skipped. Everything else is `base`.
