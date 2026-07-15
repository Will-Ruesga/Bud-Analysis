"""Ripeness task configuration — module constants only (read by the 4 entry points)."""

from datetime import datetime
from pathlib import Path

import core
from core.schemas import HeadSpec

# All run outputs land here — a sibling of base and ripeness,
# not nested inside either. Anchored to this file so it is CWD-independent.
OUTPUT_DIR = str(Path(__file__).resolve().parent.parent / "output")

# Run identity
DATE = datetime.now().strftime("%Y_%m_%d")
TASK = "ripeness-trs"  # machine-specific task name → distinguishes result dirs + ONNX from -us

# Raw data (dataset path is passed at runtime: `python prepare.py --data_dir <path>`)
# Maps the trailing filename index `_0.._4` → view name. The rig captures the
# four sides first and the top-down view LAST (`_4`), confirmed by eye on the
# images — NOT `_0`. The mapping must be right so each view label is accurate.
VIEWS = ["side_0", "side_1", "side_2", "side_3", "top"]

# Fraction of (flower, round) captures allowed to be missing one or more of the
# declared views before prepare hard-errors. A few flaky partial captures out of
# hundreds are dropped (loudly); a systematic shortfall above this fraction means
# the dataset/naming doesn't match the declaration, so prepare refuses it.
INCOMPLETE_TOLERANCE = 0.05

# Backbones. Checkpoints live in `base/checkpoints/`; the path is derived from the
# installed package location — no machine-specific path to edit. Add an entry to make
# a backbone selectable; choose one per run with `prepare --backbone`.
BACKBONE_NAME = "dinov3_vits16"  # default when --backbone is omitted
BACKBONE_CHECKPOINTS = {
    "dinov3_vits16": str(
        Path(core.__file__).parent.parent
        / "checkpoints/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
    ),
}

# Folder-name → regression target in [0, 1]. None = infer from numeric class
# folder names, min-max normalised (so 1–5 → 0..1, 1–6 → 0..1, etc.). Set an
# explicit dict only if the classes aren't evenly-spaced numbers.
TARGETS = None

# Pixel crop applied before the backbone, frozen into the run manifest by
# `prepare --crop` (this is only the default when the flag is omitted).
#   none     — the image as captured.
#   adaptive — blacken everything outside the largest centred all-flower circle,
#              keeping scale and framing. Ripeness is set by the inner whorl, so
#              this drops the outer petals the model shouldn't need.
# Changing this changes the pixels, so it cannot be a COMPARE_DIMS entry: the
# embedding cache is keyed by imageID alone. Compare crops across runs instead.
CROP_MODE = "none"

# The class mix the model will actually meet on the line. Grading is subjective at
# the extremes, so this deliberately overweights them relative to production
# (5/90/5) — pushing all the way there would leave too few raw/ripe flowers to fit.
# Consumed by COMPARE_DIMS["balance"]; the mechanism lives in `core.data`.
PRODUCTION_DISTRIBUTION = {
    "raw":  {"classes": ["1", "2"],      "proportion": 0.15},
    "good": {"classes": ["3", "4", "5"], "proportion": 0.70},
    "ripe": {"classes": ["6"],           "proportion": 0.15},
}

# Head spec — one fixed pipeline (mil_mean: all views, late fusion).
HEAD_SPEC = HeadSpec(aggregator_name="mil_mean", hidden_dims=(512, 256), dropout=0.2)

# Hyperparameters (task-side train.py)
HPARAMS = {
    "lr": 1e-3,
    "epochs": 50,
    "seed": 42,
    # Robustness-aware model selection: trials are ranked by
    # `val_rmse_prod + robustness_beta * val_view_range`, so an accurate-but-brittle
    # winner is rejected automatically. 0.0 = accuracy only. The accuracy term is
    # weighted to PRODUCTION_DISTRIBUTION, so selection optimises for the line's mix.
    "robustness_beta": 0.5,
}

# Comparison grid. `prepare --compare` picks which of these dims to vary; the
# cross-product of the picked dims is trained (one kept head per combination,
# shown side by side in `comparison.png`), and unpicked dims use their `default`.
# Only dims listed here are valid for `--compare`.
#   loss        — accuracy term: "mse" vs "huber".
#   consistency — per-view consistency penalty: "off" (λ=0) vs "on" (λ searched below).
#   balance     — retarget the TRAIN split onto PRODUCTION_DISTRIBUTION:
#                   off       — train on the dataset mix as collected.
#                   reweight  — weight the loss by target/observed; keeps every flower.
#                   subsample — drop whole flowers until the mix matches; costs ~half
#                               the train split, since `ripe` is the scarce group.
#                 Val/test are never touched by this — all variants are scored on the
#                 same rows, production-weighted, so the comparison stays honest.
COMPARE_DIMS = {
    "loss":        {"values": ["mse", "huber"],                "default": "huber"},
    "consistency": {"values": ["off", "on"],                   "default": "on"},
    "balance":     {"values": ["off", "reweight", "subsample"], "default": "reweight"},
}

# Optuna search space — searched *within* each compared variant, not across.
#   lambda_consistency — strength of the per-round view-variance penalty when
#                        consistency is "on"; [lo, hi] is searched ("off" pins λ=0).
# 200, not 100: the full grid is now loss × consistency × balance = 12 cells, so
# 100 trials would leave ~8 per cell — too thin for TPE to say anything about λ.
OPT_N_TRIALS = 200
OPT_SEARCH_SPACE = {
    "lambda_consistency": [0.0, 0.5],
}
