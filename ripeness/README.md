# ripeness

Concrete task on top of `bud-analysis-core`. Expects to sit beside the core
repo (`../bud-analysis-core`).

## Setup (once)

The env lives in the task: **core package + this task's `requirements.txt`**.
Core's `pyproject.toml` owns the shared dependency set (torch/numpy/optuna/...);
`requirements.txt` holds only ripeness-specific deps and does **not** install
core — so install core editable first, then layer the task on top:

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e /path/to/bud-analysis-core   # core package + its full dependency set
uv pip install -r requirements.txt             # ripeness-specific deps (currently none)
```

Editable (`-e`) so local edits to core apply without reinstalling. Add `[dev]`
for the tests (`-e /path/to/bud-analysis-core[dev]`).

## Run

Edit `config.py` (targets, views, cultivar, backbones, hparams) and
`prepare.parse()` (your filename convention). `prepare` freezes a run; every
later stage just points `-run` at it and reads its manifest — no config:

```bash
python prepare.py --data_dir /path/to/dataset   # -dir for short; --backbone/-bkb to pick one
#   → writes output/<run>/prep/{index.csv, info.json, distribution.png}; prints the run dir
python train.py   -run output/<run>             # Optuna sweep → output/<run>/ripeness/<aggregator>/ + comparison.png
python export.py  -run output/<run> --aggregator mil_mean   # → output/<run>/onnx/ripeness_mil_mean_<backbone>.onnx
```

`<run>` is `<DATE>-<CULTIVAR>-<BACKBONE_NAME>` (printed by `prepare`). The `output/`
base is set by `OUTPUT_DIR` in `config.py`, which defaults to `../output` — a sibling
of this task and the core checkout (`Bud-Analysis/output/`), not nested inside either.

To re-train a run prepared days ago, just `train.py -run output/<that run>` — it reads
everything from the manifest. Tweak hyperparameters per run without re-preparing:
`train.py -run output/<run> --lr 5e-4 --epochs 50`.

Multi-view discovery: `prepare.parse()` maps a filename → `(flower_id, view_type)`;
adapt it to your camera naming. Everything else is `bud-analysis-core`.
