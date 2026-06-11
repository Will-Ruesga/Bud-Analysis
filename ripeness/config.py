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
# images — NOT `_0`. `top` must be correct: the `top_only` aggregator uses it.
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

# How to turn a frozen-backbone image into its (D,) embedding.
#   "cls"          — DINOv3's CLS token (global summary).
#   "masked_mean"  — average only the patch tokens overlapping the bud, using
#                    the image's alpha mask (drops background patches).
# A/B on NancyNora: cls 0.148 vs masked_mean 0.153 test RMSE → cls wins. The
# background is already black, so CLS (a learned summary) beat a plain masked mean.
POOLING = "cls"

# Default head spec (fallback for fields Optuna does not sweep)
HEAD_SPEC = HeadSpec(aggregator_name="mil_mean", hidden_dims=(512, 256), dropout=0.2)

# Hyperparameters (task-side train.py)
HPARAMS = {
    "lr": 1e-3,
    "epochs": 50,
    "seed": 42,
    "loss": "mse",
}

# Optuna search space — aggregator_name required
OPT_N_TRIALS = 100
OPT_SEARCH_SPACE = {"aggregator_name": ["top_only", "mil_mean"]}
