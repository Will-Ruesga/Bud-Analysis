# bolletje-trs

Binary-defect detection on **multi-view** tray captures: does a flower have *bolletje* —
a ball of inner petals that means the bud will not open? A task on top of
[`base`](../base) — see the root [README](../README.md) for the engine.

Two class folders: `0` (no bolletje), `1` (bolletje). The frozen DINOv3 + sigmoid-head
pipeline is unchanged from `ripeness-trs`; only the target ({0, 1}) and the loss (`bce`)
differ. The head's sigmoid output is read as **P(bolletje)** — the "percentage of
bolletje" per flower — and `mil_mean` averages it across the flower's views. Each
variant's `metrics.json` reports classification quality (accuracy / precision / recall /
F1 at 0.5, ROC-AUC, PR-AUC) next to RMSE.

## Setup (once)

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e ../base              # the engine + its dependencies (incl. scikit-learn)
uv pip install -r requirements.txt     # task-specific deps (currently none)
```

## Run — the two view-count experiments

Same folder, same code; the view count is chosen at prepare time via `--views`.

```bash
# 5 views (four sides + top)
python prepare.py -d /path/to/dataset -c Avalanche --views "side: 0 1 2 3, top: 4" --compare loss consistency
python train.py   -r output/<run>
python export.py  -r output/<run> --variant auto

# 1 view (top only) — note: consistency is a no-op at one view, so don't --compare it
python prepare.py -d /path/to/dataset -c Avalanche --views "top: 4" --compare loss
python train.py   -r output/<run>
python export.py  -r output/<run> --variant auto
```

`prepare` prints the run dir (`<DATE>-<CULTIVAR>-<BACKBONE>`); every later step just
points `--run` at it. The two runs land in **separate** run dirs only if something in the
identity differs — they share `<DATE>-<CULTIVAR>-<BACKBONE>`, so run the 1-view
experiment under a distinct `--cultivar` tag (e.g. `Avalanche-top`) to keep both side by
side without the second overwriting the first.

Edit `config.py` for targets, views, backbone, and hparams. See `ideas.txt` for the full
experiment matrix (5-view vs 1-view, and the non-DINO baselines).
