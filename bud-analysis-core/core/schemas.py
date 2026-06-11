"""Core ↔ task contract.

`HeadSpec` and `TrainResult` are the only types that cross the boundary
between this package and a task repo. See docs/core/schemas.md for the
spec.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class HeadSpec:
    """Description of one regression head.

    `aggregator_name` also implicitly determines which view types feed the
    head — see `aggregators.VIEW_TYPES[aggregator_name]`. Input dim is
    derived from the backbone at build time, not stored here.
    """

    aggregator_name: str
    hidden_dims: tuple[int, ...]
    dropout: float


@dataclass
class TrainResult:
    """Output of one training run, handed back to core for rendering.

    `metrics` is free-form (task-computed). `head_spec` is round-tripped
    into the on-disk `metrics.json` so the aggregator dir is
    self-describing for ONNX export.
    """

    predictions: np.ndarray
    labels: np.ndarray
    metrics: dict[str, Any]
    head_state_dict: dict
    head_spec: HeadSpec
    checkpoint_path: Path
