"""ripeness-us task configuration — module constants only (read by the 4 entry points).

Single-view variant of `ripeness-trs/` for the Universalsorter `BeursTrosrozen` bud
instances: one crop per bud, labels in `labels.csv`, three classes {1, 3, 5}
(class 6 is merged into 5 in `prepare.discover`). Regression to [0, 1] — same as
`ripeness-trs/`; the consistency/robustness machinery is inert at one view (λ and
robustness_beta pinned to 0).
"""

from datetime import datetime
from pathlib import Path

import core
from core.schemas import HeadSpec

# All run outputs land here — a sibling of base and ripeness,
# not nested inside either. Anchored to this file so it is CWD-independent.
OUTPUT_DIR = str(Path(__file__).resolve().parent.parent / "output")

# Run identity
DATE = datetime.now().strftime("%Y_%m_%d")
TASK = "ripeness-us"  # machine-specific task name → distinguishes result dirs + ONNX from -trs

# Raw data (dataset path is passed at runtime: `python prepare.py --data_dir <path>`).
# Single-view dataset: one crop per bud, so there is exactly one view.
VIEWS = ["top"]

# Fraction of (flower, round) captures allowed to be missing a declared view before
# prepare hard-errors. Single view → a capture is either present or not a row at all,
# so this is effectively moot; 0.0 keeps it strict.
INCOMPLETE_TOLERANCE = 0.0

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

# Class label → regression target in [0, 1]. Explicit because the classes are not
# contiguous: the CSV has {1, 3, 5} (class 6 is merged into 5 in prepare.discover).
TARGETS = {"1": 0.0, "3": 0.5, "5": 1.0}

# Head spec — one fixed pipeline (mil_mean: all views, late fusion).
HEAD_SPEC = HeadSpec(aggregator_name="mil_mean", hidden_dims=(512, 256), dropout=0.2)

# Hyperparameters (task-side train.py)
HPARAMS = {
    "lr": 1e-3,
    "epochs": 50,
    "seed": 42,
    # Single view → no view spread to penalise, so selection is accuracy-only.
    "robustness_beta": 0.0,
}

# Comparison grid. `prepare --compare` picks which dims to vary (cross-product);
# unpicked dims use their `default`. Single-view, so only `loss` is comparable —
# `consistency` would be a no-op (no spread across views), so it is not offered.
COMPARE_DIMS = {
    "loss": {"values": ["mse", "huber"], "default": "huber"},
}

# Optuna search space — searched within each variant. λ pinned to 0: the per-view
# consistency penalty is a no-op with a single view.
OPT_N_TRIALS = 100
OPT_SEARCH_SPACE = {
    "lambda_consistency": [0.0, 0.0],
}
