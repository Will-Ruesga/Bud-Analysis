# base — Flower Attribute Regression engine

A small modular framework: train regression heads on top of a **frozen DINOv3
backbone**. Each task predicts one continuous attribute of a flower (ripeness,
defectiveness, …) as a scalar in `[0, 1]`. `base` is the shared library; tasks are
sibling projects that install it (see the root [README](../README.md)).

## Terminology (canonical — use everywhere)

| Term | Meaning |
|---|---|
| **view** | one image — a flower, in one round, on its fork (one camera angle) |
| **fork** | the machine position (tray slot) a flower sits on, numbered |
| **round** | one set of views for a flower — a single capture pass |
| **flower** | one real flower, on a fork — normally several rounds |

A **flower** (on a **fork**) is captured over several **rounds**; each **round** is a set
of **views**. In `index.csv`: `flowerID` = `<class>_<fork>` (fork = its numeric part),
`roundID` = the round, `viewType`/`viewID` = the view.

## Layout

`base/` is its own project; tasks are independent siblings, each with its own
`pyproject.toml` and venv. They depend on the *installed* `core` package, not on paths —
`pip install -e .` from `base/` makes `from core.run_context import RunContext` work in any
task sharing that environment.

```
base/
├── pyproject.toml      (package "base")
├── core/               (the Python package)
├── viewer/             (Flask predicted-vs-true explorer — see viewer/README.md)
└── tests/
```

### `core/` — task-agnostic framework

| Module | Purpose |
|---|---|
| `schemas.py` | `HeadSpec` + `TrainResult` — the dataclasses crossing the core ↔ task boundary |
| `run_context.py` | `RunContext` — single source of truth for every output path |
| `backbones.py` | Frozen DINOv3 wrapper, `eval_transform`, `feature_dim` |
| `data.py` | Discovery, prep orchestrator, `index.csv` reader/writer, batching iterator |
| `embeddings.py` | One-time DINOv3 forward pass, `.npy` cache (idempotent) |
| `aggregators.py` | View stacking `(N, V, D)` for MIL late-fusion (`heads.mil_pool`) |
| `heads.py` | `Regressor` MLP + `build(spec)` factory |
| `optimization.py` | Optuna helpers: optimiser build, keep-best-per-variant, study summary |
| `export.py` | Unified ONNX export — N heads share one backbone, filesystem-based resolution |
| `plotting.py` | Per-task variant comparison + prep distribution figure |

### Per-task files

One config module + three command scripts per task. Copy the `.py` files into a new task,
edit `config.py`, leave the rest alone.

| File | Role |
|---|---|
| `config.py` | All task settings (`DATE`, `CULTIVAR`, `HEAD_SPEC`, `HPARAMS`, `COMPARE_DIMS`, `OPT_SEARCH_SPACE`, …) |
| `prepare.py` | Freeze a run → `index.csv` + manifest (`info.json`) + `distribution.png` |
| `train.py` | Optuna study + training loop + per-variant report |
| `export.py` | One-line wrapper → `core.export.export(ctx, heads)` |

## Workflow

`prepare` freezes a run (writes the manifest, prints the run dir); every later stage takes
`--run <run>` and reads that manifest — no `config.py` needed again.

```bash
python prepare.py --dataset /path/to/dataset            # pick dataset + backbone; prints output/<run>
python train.py   --run output/<run>                    # Optuna study, best trial per variant; --lr/--epochs/--seed override
python export.py  --run output/<run>                    # bundle to ONNX; --variant huber, or --heads ripeness:auto defects:huber
```

## On-disk layout

One run dir per `(date, cultivar, backbone)`; each task's results live inside as
`<task>-results/`. The filesystem *is* the registry — each `<variant>/` carries its
`head_spec` in `metrics.json`, and `core.export` walks these dirs to find trained heads
(no separate manifest).

```
output/YYYY_MM_DD-cultivar-<backbone>/
├── prep/
│   ├── index.csv             # per-image training table
│   ├── info.json             # run manifest: identity + backbone + data_dir + compare grid + stats + config
│   └── distribution.png      # dataset overview
├── emb/<imageID>.npy         # cached frozen features (the only image-derived artifact) + meta.json
├── onnx/                     # deployable bundled models
└── ripeness-results/         # one folder per task
    ├── study_summary.json
    ├── comparison.png        # variant comparison (curves + pred-vs-true scatters)
    └── mse/ huber/ …         # best trial per variant: head.pt, metrics.json, predictions.csv
```

Image bytes are never copied: `index.csv`'s `fileName` is relative to `DATA_DIR` (recorded
absolute in `info.json`). The only thing the pipeline writes outside `output/<run>/` is
`DATA_DIR/predictions.csv` (one `pred<Variant>` column per head) — kept beside the raw
images so a file-explorer tool can use it without path remapping. `OUTPUT_DIR` in
`config.py` sets the `output/` base (defaults to `../output`).

## Conventions

| Where | Convention |
|---|---|
| CSV columns on disk | camelCase (`fileName`, `flowerID`, `viewType`) — except `class` |
| CSV columns in Python | snake_case after `read_index` (`flower_id`); `class` only via `df["class"]` |
| CSV format | `sep=;` first line, `;` separator, `fileName` = Windows-relative to `DATA_DIR` |
| JSON keys | snake_case (`head_spec`, `n_completed`) |
| Python fields / constants | snake_case attrs; `UPPERCASE_SNAKE` for `config.py` constants |
| Path math | always through `RunContext`; no string concatenation elsewhere |
| Imports | fully-qualified (`from core.run_context import RunContext`); `core/__init__.py` stays empty |

## Key design decisions

Locked. Don't relitigate without a strong reason.

- **Regression only.** Input arrives in class folders, but the classes are subjective and
  noisy — a continuous `[0, 1]` target represents between-class uncertainty and avoids
  over-fitting bin boundaries. No classifier or ordinal heads anywhere.
- **Frozen backbone.** DINOv3 (vendored in `core/dinov3/`) runs once per image, cached as
  `.npy`. Only the MLP head trains. Embedding pooling is always CLS.
- **MIL late-fusion** (`mil_mean`). All views run through the head and their *predictions*
  are averaged — never feature-pooled, so no view dominates. A view-consistency penalty
  (`+ λ·variance`, λ searched) trains toward angle-invariance.
- **Configurable compare grid** (`prepare --compare`; dims `loss`, `consistency`,
  cross-product). One kept winner per cell → `<task>-results/<value>/`; empty → one variant.
- **Unified `export()`.** N heads share one backbone in one ONNX. Input `(B, V, H, W, 3)`,
  output `(B,)` for one head / `(B, N)` for many. `auto` picks lowest `selection_score`.
- **One backbone per run dir**, named in the dir — different backbones make sibling dirs,
  no collisions.

## Getting started

1. Read `core/schemas.py` (`HeadSpec` + `TrainResult`) — the task ↔ library contract.
2. Read each `core/` module's docstring + source; the header explains purpose, inputs, outputs.
