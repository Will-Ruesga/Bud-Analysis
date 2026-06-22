# 4MT Vision — Flower Attribute Regression

A minimal modular framework for training small regression heads on top of a frozen DINOv3 backbone. Each task predicts one continuous attribute of a flower (ripeness, defectiveness, …) as a scalar in `[0, 1]`.

Two source repos (`Flower-defects` and `Flower-ripeness-v2`) shared the same architecture (DINOv3 + MLP head) but lived as forks. This project replaces them with a shared `core/` package + per-task wrappers.

## Terminology (canonical — use these everywhere: core, tasks, viewer)

This is the single source of truth for these four words. Do not redefine them.

| Term | Meaning |
|---|---|
| **view** | one image — a single flower, in one round, on its fork (one camera angle) |
| **fork** | the machine position (tray slot) a flower sits on, numbered |
| **round** | one *set of views* for a flower — a single capture pass |
| **flower** | one real flower — normally a *set of rounds* (sitting on a fork) |

Hierarchy: a **flower** (on a **fork**) is captured over several **rounds**; each **round** is a set of **views**.

Data mapping in `index.csv`: `flowerID` = the flower (`<class>_<fork>`); the **fork** number is the numeric part of `flowerID`; `roundID` = the **round** (repeated-capture index); `viewType`/`viewID` = the **view**.

## Why regression-only

Input data arrives in **class folders**, but the classes are biological / subjective and not reliable enough to train a classifier on. Treating the output as a continuous regression in `[0, 1]` makes the model's uncertainty between adjacent classes representable, and avoids over-fitting noisy bin boundaries. Do not re-introduce classification heads.

## Repository layout

The library lives in its own project directory. Tasks live in **sibling** project directories — fully independent, each with its own `pyproject.toml`. The relationship between them is "task depends on the installed `core` package", not any path indirection. Move any of them to a different folder and they keep working (re-run `pip install -e .` from the new location for editable installs).

```
<workspace>/                      ← can be anywhere on disk (in this repo it is `new_set_up/`)
├── base/           ← library project (this repo)
│   ├── pyproject.toml            (package name "base")
│   ├── README.md                 (this file)
│   ├── CLAUDE.md                 (coding rules)
│   ├── core/       (the Python package; snake_case)
│   │   └── __init__.py
│   ├── viewer/                  (Flask app: interactive predicted-vs-true explorer — see viewer/README.md)
│   └── tests/
├── ripeness-trs/                 ← task project (separate, created when you start a task)
└── ripeness-us/                  ← task project (separate)
```

`pip install -e .` from inside `base/` makes `from core.run_context import RunContext` work in every task project that uses the same Python environment.

### `core/` — task-agnostic framework

| Module | Purpose |
|---|---|
| `schemas.py` | `HeadSpec` + `TrainResult` — the two dataclasses crossing the core ↔ task boundary |
| `run_context.py` | `RunContext` — single source of truth for every output path |
| `backbones.py` | Frozen DINOv3 wrapper, `eval_transform`, `feature_dim` |
| `data.py` | Discovery, prep orchestrator, `index.csv` reader/writer, batching iterator |
| `embeddings.py` | One-time DINOv3 forward pass, `.npy` cache (idempotent) |
| `aggregators.py` | View stacking: the run's declared views stacked `(N, V, D)` for MIL late-fusion (`heads.mil_pool`) |
| `heads.py` | `Regressor` MLP + `build(spec)` factory |
| `optimization.py` | Optuna helpers: optimiser build, keep-best-per-variant, study summary |
| `export.py` | Unified ONNX export — N heads share one backbone, filesystem-based head resolution |
| `plotting.py` | Per-task variant comparison + prep dataset-distribution figure |

### Per-task template files

Three command scripts + one config module per task project. When you create a new task project (e.g. `<workspace>/ripeness-trs/`), copy these `.py` files into it, edit `config.py`, leave the rest alone.

| File | Role |
|---|---|
| `config.py` | All task settings (module constants — `DATE`, `CULTIVAR`, `HEAD_SPEC`, `HPARAMS`, `COMPARE_DIMS`, `OPT_SEARCH_SPACE`, …) |
| `prepare.py` | Configure a run (`--compare` picks the comparison grid) → write `index.csv` + the run manifest (`info.json`) + `distribution.png` |
| `train.py` | Optuna study + training loop + per-variant report |
| `export.py` | One-line wrapper → `core.export.export(ctx, heads)` |

## Workflow

`prepare` configures and freezes a run (writes the manifest, prints the run dir);
every later stage just takes `-run <run>` and reads that manifest — no `config.py`.

```bash
# 1. configure + freeze a run (pick dataset + backbone); prints output/<run>
python prepare.py --dataset /path/to/dataset             # --name sets the ONNX label; --backbone/-bkb to choose
#    also writes prep/distribution.png — the dataset overview figure

# 2. main: Optuna study, keeps best trial per variant (the compared grid, e.g. loss × consistency)
python train.py   -run output/<run>                      # --lr/--epochs/--seed override the manifest
python train.py   -run output/<run> --epochs 50          #   …same prepared data, different hparams

# 3. once trained: bundle into ONNX (deliberate variant choice)
python export.py  -run output/<run>                      # this task, auto-pick best variant
python export.py  -run output/<run> --variant huber
python export.py  -run output/<run> --heads ripeness:auto defects:huber      # multi-task bundle
python export.py  -run output/<run> --heads all                             # every task, auto each
```

## On-disk layout

One run dir per `(date, cultivar, backbone)` under `output/`; each task's training results live inside it as `<task>-results/`:

```
output/YYYY_MM_DD-cultivar-<backbone_name>/
├── prep/
│   ├── index.csv             # per-image table (training metadata)
│   ├── info.json             # run manifest: identity + backbone + image_size + abs data_dir + compare grid + stats + frozen training config
│   └── distribution.png      # dataset overview: class distribution by split + sizes by granularity
├── emb/                      # cached frozen features (the only image-derived artifact)
│   ├── <imageID>.npy         # one frozen feature per image
│   └── meta.json
├── onnx/                     # deployable bundled models
├── ripeness-results/         # one folder per task (<task>-results)
│   ├── study_summary.json
│   ├── comparison.png        # variant comparison (curves + pred-vs-true scatters)
│   ├── mse/                  # best Optuna trial for this loss variant
│   │   ├── head.pt
│   │   ├── metrics.json      # carries head_spec — the dir is self-describing
│   │   └── predictions.csv
│   └── huber/                # (same shape, different loss)
└── defects-results/          # next task, same shape
```

The `output/` base above is the default; a task sets `OUTPUT_DIR` in its `config.py` to put run dirs anywhere (e.g. a sibling of the task and core checkouts rather than inside the task). See `run_context.py`.

The filesystem under `<task>/` is the source of truth for which heads exist — no separate manifest file. `core.export` walks `<task>/<variant>/` to discover trained heads.

`_study/` scratch dirs appear under `<task>/` during a training run; `optimization.keep_best_per_variant` cleans them up afterwards.

Image bytes are **not** duplicated into the run dir. `prep/index.csv`'s `fileName` is relative to `DATA_DIR` (recorded absolute in `prep/info.json`). The only image-derived artifact in the run dir is the embedding cache (`emb/<imageID>.npy`).

Plus, in the **raw dataset folder** (the path passed as `DATA_DIR`):

```
<DATA_DIR>/
├── 0/ 1/ 2/ ...                                # original class folders (untouched)
├── predictions.csv                             # tool-friendly: <class>\<image>.png paths, one predMse/predHuber col per variant
└── ...
```

The dataset folder is the only place outside `output/<run>/` where the pipeline writes files. This is intentional so your Windows file-explorer tool can open `DATA_DIR` and use the predictions CSV directly without any path remapping.

## Conventions

| Where | Convention |
|---|---|
| CSV column names on disk | camelCase (`fileName`, `flowerID`, `roundID`, `viewType`) — except `class`, which stays as `class` |
| CSV column names in Python | snake_case after `read_index` rename (`flower_id`, `view_type`); `class` only via subscript (`df["class"]`) |
| CSV first line | `sep=;` (Excel / tool hint) |
| CSV separator | `;` |
| CSV `fileName` paths | Windows backslashes, relative to `DATA_DIR` (resolved via `info.json`) |
| JSON keys | snake_case (`best_k`, `n_completed`, `head_spec`) |
| Python attributes / dataclass fields | snake_case (`input_dim`, `aggregator_name`) |
| Python module constants in `task/config.py` | `UPPERCASE_SNAKE` (`DATA_DIR`, `HEAD_SPEC`, `HPARAMS`) |
| Path math | always through `RunContext`; no string concatenation elsewhere |
| Imports | fully-qualified from submodules (`from core.run_context import RunContext`); `core/__init__.py` stays empty — no convenience re-exports |

## Key design decisions

These are locked. Don't relitigate without a strong reason.

- **Regression only.** No classifier or ordinal heads anywhere. See "Why regression-only" at the top.
- **Frozen backbone.** DINOv3 is loaded from `core/dinov3/` (vendored), runs once per image, features are cached as `.npy`. Only the MLP head trains.
- **One view pipeline** (`mil_mean`). All 5 views run through the head and their *predictions* are averaged (`heads.mil_pool`, MIL late-fusion) — never feature-pooled, so no single view can dominate a pooled vector. A view-consistency penalty (`+ λ·variance` across views, λ searched) trains the head toward angle-invariance.
- **The study compares a configurable grid** (`prepare --compare`; dims today are `loss` and `consistency`, cross-product). One kept winner per grid cell, dir `<task>-results/<compare_value>/`. Empty `--compare` → one default variant. Embedding pooling is always CLS (the masked-mean experiment lost and was removed). No `OPT_TOP_K`, no `rank_<i>` dirs.
- **Filesystem is the registry.** Each `<task>/<variant>/` is self-describing (carries `head_spec` in its `metrics.json`). No `manifest.json`.
- **Unified `export()` function** — N heads share one backbone in a single ONNX. Input always `(B, V, H, W, 3)`. Output `(B,)` for one head, `(B, N)` for many. Variant selection is a deliberate human-in-the-loop decision; `auto` picks the lowest-`selection_score` variant per task.
- **One backbone per run dir; backbone name is in the run dir name.** Different backbones produce sibling run dirs (`<date>-<cultivar>-<backbone_A>/`, `<date>-<cultivar>-<backbone_B>/`) — no collisions, no manual date-bumping.
- **No image duplication.** `prepare.py` writes only `index.csv`, `info.json`, and `distribution.png`. Image bytes stay in `DATA_DIR`; embeddings (`emb/<imageID>.npy`) are the only image-derived cache. Whatever format PIL can open is fair game in `DATA_DIR`.
- **Predictions CSVs land in `DATA_DIR`** (the raw dataset folder). Everything else lives under `output/<run>/`.

## Plot conventions

One training plot, per task.

- **`<task>-results/comparison.png`** — per-task variant comparison (the compared grid). Top row: val-RMSE and train/val loss curves overlaid, one line per variant. Bottom row: one predicted-vs-true scatter per variant over a grey ±0.1 tolerance band, captioned with its robustness numbers (view range / fork σ / λ). (Predictions also land in one `DATA_DIR/predictions.csv` with a `pred<Variant>` column per head.)

## Getting started

1. Read [`CLAUDE.md`](CLAUDE.md) for the coding rules.
2. Read a module's docstring + source in `core/` — each module's header explains its purpose, inputs, and outputs.
3. The task ↔ library contract is `core/schemas.py` (`HeadSpec` + `TrainResult`). Read that first to see how a task plugs into the library.
