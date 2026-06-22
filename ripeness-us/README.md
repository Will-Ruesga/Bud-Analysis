# ripeness-us

Ripeness regression on **single-view** Universalsorter bud crops (one crop per bud,
labels in `labels.csv`). A task on top of [`base`](../base) — see the root
[README](../README.md) for the engine.

## Setup (once)

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e ../base              # the engine + its dependencies
uv pip install -r requirements.txt    # task-specific deps (currently none)
```

## Run

```bash
# once per dataset: tag each crop with its cultivar (metadata for split + viewer)
python link_cultivars.py --dataset /path/to/bud-instances --split-views /path/to/...-split-views

python prepare.py --dataset /path/to/dataset --cultivar BeursTrosrozen --compare loss
python train.py   --run output/<run>          # Optuna sweep over the head
python export.py  --run output/<run> --variant auto   # → output/<run>/onnx/
```

`prepare` prints the run dir (`<DATE>-<CULTIVAR>-<BACKBONE>`); every later step just
points `--run` at it. Single view, so only `loss` is comparable. Edit `config.py` for
targets, backbone, and hparams.
