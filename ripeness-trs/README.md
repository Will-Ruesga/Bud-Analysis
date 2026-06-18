# ripeness-trs

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

Editable (`-e`) so local edits to core apply without reinstalling. `pytest` and
`onnxruntime` ship with core, so the install can run the tests as-is.

## Run

Edit `config.py` (targets, views, backbone, hparams, `COMPARE_DIMS`).
`prepare` freezes a run; every later stage just points `--run` at it and reads its
manifest — no config:

```bash
python prepare.py --dataset /path/to/dataset --cultivar Avalanche --compare loss consistency
#   --dataset/-d is the dataset folder; --cultivar/-c names the run (required) — it is both
#   the run dir <DATE>-<CULTIVAR>-<BACKBONE> and the label baked into the ONNX filename.
#   --backbone/-bkb to pick one; --views/-v to map camera pages (omit → prompt).
#   --compare picks the grid (cross-product); omit for one variant.
#   → writes output/<run>/prep/{index.csv, info.json, distribution.png}; prints the run dir
python train.py   --run output/<run>            # Optuna sweep → output/<run>/ripeness-results/<compare_value>/ + comparison.png
python export.py  --run output/<run> --variant auto   # → output/<run>/onnx/ripeness_<variant>_<backbone>.onnx
```

`--compare` accepts any of `config.COMPARE_DIMS` (here `loss`, `consistency`). With both,
the kept variants are `mse-off`/`mse-on`/`huber-off`/`huber-on`; with one, `mse`/`huber`.

`<run>` is `<DATE>-<CULTIVAR>-<BACKBONE_NAME>` (printed by `prepare`). The `output/`
base is set by `OUTPUT_DIR` in `config.py`, which defaults to `../output` — a sibling
of this task and the core checkout (`Bud-Analysis/output/`), not nested inside either.

To re-train a run prepared days ago, just `train.py --run output/<that run>` — it reads
everything from the manifest. Tweak hyperparameters per run without re-preparing:
`train.py --run output/<run> --lr 5e-4 --epochs 50`.

Multi-view discovery: `prepare.discover()` maps each filename → `(flower, fork, view)`;
adapt the `--views` spec to your camera naming. Everything else is `bud-analysis-core`.
