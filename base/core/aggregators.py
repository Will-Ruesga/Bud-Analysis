"""View stacking: map the embedding cache + index into MIL training samples.

The one view pipeline: stack all views of each (flower, round) in canonical order,
**unpooled** → samples `(N, V, D)`. The per-view head + mean-pool of *predictions*
happens later in `heads.mil_pool` (late fusion); this module only stacks views, it
never pools features.
"""

import numpy as np
import pandas as pd


def _groups(index: pd.DataFrame):
    """Yield `((flower_id, round_id), view_type → image_id)` in deterministic order.

    Empty `round_id` round-trips from CSV as NaN; normalise it to "" so groups
    aren't silently dropped by `groupby`'s NaN handling (Rule 12).
    """
    index = index.copy()
    index["round_id"] = index["round_id"].fillna("")
    for (flower_id, round_id), grp in index.groupby(["flower_id", "round_id"], sort=True):
        yield (flower_id, round_id), dict(zip(grp["view_type"], grp["image_id"]))


def stack_views(
    index: pd.DataFrame,
    emb_by_id: dict[str, np.ndarray],
    view_types: list[str],
) -> tuple[np.ndarray, list[tuple[str, str]]]:
    """Stack the declared views of each (flower, round) in order, unpooled.

    `view_types` is the run's declared view set (from `info["views"]`) — required,
    so there is no hidden default view count anywhere. Returns `(samples (N, V, D),
    keys)`; every group must carry all of `view_types` (a missing view raises — the
    backstop to `core.data._enforce_views`, which already validated at prepare time).
    A single declared view yields `V=1` bags the late-fusion head reduces to a plain MLP.
    """
    samples, keys = [], []
    for key, vt_to_id in _groups(index):
        views = []
        for vt in view_types:
            if vt not in vt_to_id:
                raise ValueError(f"group {key} missing view {vt!r} (have {sorted(vt_to_id)})")
            views.append(emb_by_id[vt_to_id[vt]])
        samples.append(np.stack(views))  # (V, D)
        keys.append(key)
    return np.stack(samples), keys  # (N, V, D)
