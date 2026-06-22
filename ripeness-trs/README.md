# ripeness-trs

Ripeness regression on **multi-view** tray captures (several camera angles per flower).
A task on top of [`base`](../base) — see the root [README](../README.md) for the engine.

## Setup (once)

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e ../base              # the engine + its dependencies
uv pip install -r requirements.txt    # task-specific deps (currently none)
```

## Run

```bash
python prepare.py --dataset /path/to/dataset --cultivar Avalanche --compare loss consistency
python train.py   --run output/<run>          # Optuna sweep over the head
python export.py  --run output/<run> --variant auto   # → output/<run>/onnx/
```

`prepare` prints the run dir (`<DATE>-<CULTIVAR>-<BACKBONE>`); every later step just
points `--run` at it. Edit `config.py` for targets, views, backbone, and hparams.
