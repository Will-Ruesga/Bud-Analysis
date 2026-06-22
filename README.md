# Bud Analysis

Predict continuous attributes of a flower/bud — ripeness, defectiveness, … — as a
scalar in `[0, 1]`.

The approach is the same for every attribute: a **frozen DINOv3 backbone** turns each
image into a feature vector once (cached on disk), and a small **trainable regression
head** maps that vector to the attribute. Only the head trains; the backbone never moves.
New attributes are new heads, not new architectures.

## Layout

```
Bud-Analysis/
├── base/            # shared library (the backbone+head engine) + the viewer app
├── ripeness-trs/    # a task: ripeness on multi-view tray captures
├── ripeness-us/     # a task: ripeness on single-view bud crops
├── ...              # more tasks live here, one folder each
└── output/          # results of every run (embeddings, trained heads, plots, ONNX)
```

A **task** is one attribute on one kind of data. Each task is its own project (its own
venv) that installs `base` and supplies a `config.py` + the `prepare → train → export`
scripts. Adding a task means adding a sibling folder, not touching the others.

The flow in every task is the same: `prepare` freezes a run, `train` runs an Optuna
sweep over the head, `export` bundles the chosen head into ONNX. The `viewer` (in
`base/`) is a Flask app to explore predicted-vs-true for any trained run.

## Where to look

- A specific task → that task's `README.md` (what it is, how to run it).
- The engine, terminology, on-disk layout, design decisions → [`base/README.md`](base/README.md).
- The viewer → [`base/viewer/README.md`](base/viewer/README.md).
