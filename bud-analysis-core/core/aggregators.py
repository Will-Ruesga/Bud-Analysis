"""View stacking: map the embedding cache + index into MIL training samples.

The one view pipeline: stack all views of each (flower, fork) in canonical order,
**unpooled** → samples `(N, V, D)`. The per-view head + mean-pool of *predictions*
happens later in `heads.mil_pool` (late fusion); this module only stacks views, it
never pools features. See docs/core/aggregators.md.
"""

import numpy as np
import pandas as pd

from core.data import CANONICAL_VIEW_TYPES

# The views fed to the head, in canonical order. The single source of truth for
# which view types one (flower, fork) must carry.
VIEW_TYPES: list[str] = list(CANONICAL_VIEW_TYPES)


def _groups(index: pd.DataFrame):
    """Yield `((flower_id, fork_id), view_type → image_id)` in deterministic order.

    Empty `fork_id` round-trips from CSV as NaN; normalise it to "" so groups
    aren't silently dropped by `groupby`'s NaN handling (Rule 12).
    """
    index = index.copy()
    index["fork_id"] = index["fork_id"].fillna("")
    for (flower_id, fork_id), grp in index.groupby(["flower_id", "fork_id"], sort=True):
        yield (flower_id, fork_id), dict(zip(grp["view_type"], grp["image_id"]))


def stack_views(
    index: pd.DataFrame,
    emb_by_id: dict[str, np.ndarray],
) -> tuple[np.ndarray, list[tuple[str, str]]]:
    """Stack all views of each (flower, fork) in canonical order, unpooled.

    Returns `(samples (N, V, D), keys)`. Every group must carry all of
    `VIEW_TYPES`; a missing view raises (no masking).
    """
    samples, keys = [], []
    for key, vt_to_id in _groups(index):
        views = []
        for vt in VIEW_TYPES:
            if vt not in vt_to_id:
                raise ValueError(f"group {key} missing view {vt!r} (have {sorted(vt_to_id)})")
            views.append(emb_by_id[vt_to_id[vt]])
        samples.append(np.stack(views))  # (V, D)
        keys.append(key)
    return np.stack(samples), keys  # (N, V, D)
