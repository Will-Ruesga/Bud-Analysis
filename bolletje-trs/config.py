"""bolletje-trs task configuration — module constants only (read by the 4 entry points).

Binary-defect variant of `ripeness-trs/`: instead of a graded ripeness scalar, the
target is whether a flower has *bolletje* — a ball of inner petals that means the bud
will not open. Two class folders, `0` (no bolletje) and `1` (bolletje). The frozen
DINOv3 + sigmoid-head pipeline is unchanged; only the target and the loss differ.

The head already ends in a sigmoid, so its output is read as **P(bolletje)** in [0, 1]
— the "percentage of bolletje" per flower — and `mil_mean` averages that probability
across the flower's views. Training to {0, 1} with `bce` (or `mse` = Brier score) keeps
this squarely inside the project's "regression to [0, 1]" design: no classifier head is
added anywhere. Classification quality (accuracy / F1 / ROC-AUC / PR-AUC at threshold
0.5) is reported alongside RMSE in each variant's `metrics.json`.

This same folder covers BOTH view-count experiments the task calls for — pick at
prepare time, no code change:
    5 views:  python prepare.py -d <data> -c <cultivar> --views "side: 0 1 2 3, top: 4"
    1 view :  python prepare.py -d <data> -c <cultivar> --views "top: 4"
"""

from datetime import datetime
from pathlib import Path

import core
from core.schemas import HeadSpec

# All run outputs land here — a sibling of base and the tasks,
# not nested inside either. Anchored to this file so it is CWD-independent.
OUTPUT_DIR = str(Path(__file__).resolve().parent.parent / "output")

# Run identity
DATE = datetime.now().strftime("%Y_%m_%d")
TASK = "bolletje-trs"  # task name → distinguishes result dirs + ONNX from the other tasks

# Raw data (dataset path is passed at runtime: `python prepare.py --dataset <path>`).
# Default view map for the multi-view (5-view) experiment. The rig captures the four
# sides first and the top-down view LAST (`_4`) — same rig and mapping as ripeness-trs.
# For the single-view (top-only) experiment, override at prepare time with
# `--views "top: 4"` (see the module docstring); no edit here is needed.
VIEWS = ["side_0", "side_1", "side_2", "side_3", "top"]

# Fraction of (flower, round) captures allowed to be missing one or more declared
# views before prepare hard-errors. A few flaky partial captures are dropped (loudly);
# a systematic shortfall means the dataset/naming doesn't match the declaration.
INCOMPLETE_TOLERANCE = 0.05

# Backbones. Checkpoints are vendored inside the installed core package, so paths are
# derived from the package location — no machine-specific path to edit. DINOv3 only.
# Add dinov3_vitb16 here to also sweep the larger backbone (richer 768-dim features).
BACKBONE_NAME = "dinov3_vits16"  # default when --backbone is omitted
BACKBONE_CHECKPOINTS = {
    "dinov3_vits16": str(
        Path(core.__file__).parent
        / "dinov3/checkpoints/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
    ),
}

# Binary target: class folder name → target in [0, 1]. Explicit (not inferred) so the
# intent is on the page — `0` = no bolletje, `1` = bolletje. The sigmoid head's output
# is then P(bolletje).
TARGETS = {"0": 0.0, "1": 1.0}

# Head spec — one fixed pipeline (mil_mean: all views, late fusion). Same head as every
# other task; for binary its sigmoid output is a probability rather than a graded score.
HEAD_SPEC = HeadSpec(aggregator_name="mil_mean", hidden_dims=(512, 256), dropout=0.2)

# Hyperparameters (task-side train.py)
HPARAMS = {
    "lr": 1e-3,
    "epochs": 50,
    "seed": 42,
    # Robustness-aware model selection: with 5 views, trials are ranked by
    # `(1 - val_pr_auc) + robustness_beta * val_view_range`, so an accurate-but-brittle
    # winner (views disagreeing on the same flower) is rejected. Inert at 1 view (no
    # view spread) — set to 0.0 for the top-only run if you want.
    "robustness_beta": 0.5,
}

# Comparison grid. `prepare --compare` picks which dims to vary; the cross-product of the
# picked dims is trained (one kept head per combination, shown side by side in
# `comparison.png`), and unpicked dims use their `default`.
#   loss        — accuracy term: "bce" (log-loss on P(bolletje)) vs "mse" (Brier score).
#   consistency — per-view consistency penalty: "off" (λ=0) vs "on" (λ searched below).
#                 Inert at a single view; only meaningful for the 5-view run.
COMPARE_DIMS = {
    "loss":        {"values": ["bce", "mse"],  "default": "bce"},
    "consistency": {"values": ["off", "on"],   "default": "on"},
}

# Optuna search space — searched *within* each compared variant, not across.
#   lambda_consistency — strength of the per-round view-variance penalty when
#                        consistency is "on"; [lo, hi] is searched ("off" pins λ=0).
OPT_N_TRIALS = 100
OPT_SEARCH_SPACE = {
    "lambda_consistency": [0.0, 0.5],
}
