"""Ripeness task configuration — module constants only (read by the 4 entry points)."""

from datetime import datetime
from pathlib import Path

import core
from core.schemas import HeadSpec

# All run outputs land here — a sibling of bud-analysis-core and ripeness,
# not nested inside either. Anchored to this file so it is CWD-independent.
OUTPUT_DIR = str(Path(__file__).resolve().parent.parent / "output")

# Run identity
DATE = datetime.now().strftime("%Y_%m_%d")
CULTIVAR = "GrootGroot-GardeniaS1"
TASK = "ripeness"

# Raw data (dataset path is passed at runtime: `python prepare.py --data_dir <path>`)
# Maps the trailing filename index `_0.._4` → view name. The rig captures the
# four sides first and the top-down view LAST (`_4`), confirmed by eye on the
# images — NOT `_0`. The mapping must be right so each view label is accurate.
VIEWS = ["side_0", "side_1", "side_2", "side_3", "top"]

# Backbones. Checkpoints are vendored inside the installed core package, so paths
# are derived from the package location — no machine-specific path to edit. Add an
# entry to make a backbone selectable; choose one per run with `prepare --backbone`.
BACKBONE_NAME = "dinov3_vits16"  # default when --backbone is omitted
BACKBONE_CHECKPOINTS = {
    "dinov3_vits16": str(
        Path(core.__file__).parent
        / "dinov3/checkpoints/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
    ),
}

# Folder-name → regression target in [0, 1]. None = infer from numeric class
# folder names, min-max normalised (so 1–5 → 0..1, 1–6 → 0..1, etc.). Set an
# explicit dict only if the classes aren't evenly-spaced numbers.
TARGETS = None

# Head spec — one fixed pipeline (mil_mean: all views, late fusion).
HEAD_SPEC = HeadSpec(aggregator_name="mil_mean", hidden_dims=(512, 256), dropout=0.2)

# Hyperparameters (task-side train.py)
HPARAMS = {
    "lr": 1e-3,
    "epochs": 50,
    "seed": 42,
    # Robustness-aware model selection: trials are ranked by
    # `val_rmse + robustness_beta * val_view_range`, so an accurate-but-brittle
    # winner is rejected automatically. 0.0 = accuracy only.
    "robustness_beta": 0.5,
}

# Optuna search space. The compared dimension is the LOSS (one kept winner per
# loss → `comparison.png` shows them side by side); everything else is searched
# within each.
#   loss                — accuracy term: "mse" vs "huber" (the consistency penalty
#                         rides on top of either).
#   lambda_consistency  — weight of the per-fork view-variance penalty added to the
#                         training loss; [lo, hi] is searched. Teaches the head to
#                         give the same ripeness across views. [0, 0] disables it.
OPT_N_TRIALS = 100
OPT_SEARCH_SPACE = {
    "loss": ["mse", "huber"],
    "lambda_consistency": [0.0, 0.5],
}
