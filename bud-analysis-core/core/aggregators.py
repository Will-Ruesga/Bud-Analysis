"""View selection: map the embedding cache + index into training samples.

Two aggregators, picked by name from `HeadSpec.aggregator_name`:

- `top_only`  — one top view per group        → samples `(N, D)`
- `mil_mean`  — all 5 views stacked, NOT pooled → samples `(N, V, D)`

`mil_mean` is MIL (late fusion): the per-view head + mean-pool of predictions
happens later in `heads.mil_pool`, not here. This module only stacks views in
canonical order; it never pools features. See docs/core/aggregators.md.
"""

from typing import Callable

import numpy as np
import pandas as pd

from core.data import CANONICAL_VIEW_TYPES

VIEW_TYPES: dict[str, list[str]] = {
    "top_only": ["top"],
    "mil_mean": list(CANONICAL_VIEW_TYPES),
}


def _groups(index: pd.DataFrame):
    """Yield `((flower_id, fork_id), view_type → image_id)` in deterministic order.

    Empty `fork_id` round-trips from CSV as NaN; normalise it to "" so groups
    aren't silently dropped by `groupby`'s NaN handling (Rule 12).
    """
    index = index.copy()
    index["fork_id"] = index["fork_id"].fillna("")
    for (flower_id, fork_id), grp in index.groupby(["flower_id", "fork_id"], sort=True):
        yield (flower_id, fork_id), dict(zip(grp["view_type"], grp["image_id"]))


def top_only(
    index: pd.DataFrame,
    emb_by_id: dict[str, np.ndarray],
) -> tuple[np.ndarray, list[tuple[str, str]]]:
    """Select the top view of each (flower, fork). Returns `(samples (N, D), keys)`."""
    samples, keys = [], []
    for key, vt_to_id in _groups(index):
        if "top" not in vt_to_id:
            raise ValueError(f"group {key} has no 'top' view (have {sorted(vt_to_id)})")
        samples.append(emb_by_id[vt_to_id["top"]])
        keys.append(key)
    return np.stack(samples), keys


def mil_mean(
    index: pd.DataFrame,
    emb_by_id: dict[str, np.ndarray],
) -> tuple[np.ndarray, list[tuple[str, str]]]:
    """Stack all 5 views of each (flower, fork) in canonical order, unpooled.

    Returns `(samples (N, V, D), keys)`. Every group must carry all of
    `VIEW_TYPES["mil_mean"]`; a missing view raises (no masking).
    """
    samples, keys = [], []
    for key, vt_to_id in _groups(index):
        views = []
        for vt in VIEW_TYPES["mil_mean"]:
            if vt not in vt_to_id:
                raise ValueError(f"group {key} missing view {vt!r} (have {sorted(vt_to_id)})")
            views.append(emb_by_id[vt_to_id[vt]])
        samples.append(np.stack(views))  # (V, D)
        keys.append(key)
    return np.stack(samples), keys  # (N, V, D)


REGISTRY: dict[str, Callable] = {
    "top_only": top_only,
    "mil_mean": mil_mean,
}


def get(name: str) -> Callable:
    """Look up an aggregator by name. Raises KeyError on unknown names (Rule 12)."""
    return REGISTRY[name]
