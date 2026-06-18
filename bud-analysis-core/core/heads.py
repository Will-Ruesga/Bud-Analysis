"""Trainable regression head on top of frozen embeddings.

One head class (`Regressor`), one factory (`build`), plus `mil_pool` — the MIL
late-fusion forward shared by the task trainer and ONNX export.
"""

import torch
from torch import nn

from core import backbones
from core.schemas import HeadSpec


class Regressor(nn.Module):
    """`Linear → ReLU → Dropout` per hidden dim, then `Linear(1) → Sigmoid`.

    `forward((B, D)) -> (B, 1)`, output in `[0, 1]`. Single-view and
    branch-free so ONNX export traces it without modification; MIL is layered
    on top by `mil_pool`, never baked into the head.
    """

    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...], dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        d = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers += [nn.Linear(d, 1), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build(spec: HeadSpec, backbone_name: str) -> nn.Module:
    """Construct the regressor described by `spec`.

    `input_dim` is derived from `backbones.feature_dim(backbone_name)` so the
    head's width matches the backbone's output; `hidden_dims` and `dropout`
    come from `spec`.
    """
    input_dim = backbones.feature_dim(backbone_name)
    return Regressor(input_dim, spec.hidden_dims, spec.dropout)


def mil_pool(head: nn.Module, x: torch.Tensor, return_views: bool = False):
    """MIL forward: per-view head, then mean over views. `(B, V, D) -> (B, 1)`.

    The single MIL implementation, called by the `mil_mean` training path and
    by `export`, so train-time and inference-time pooling are the same op.

    With `return_views=True` also returns the per-view predictions `(B, V, 1)`
    *before* the mean — the view-consistency loss and the view-spread metrics
    read these. The default (pooled-only) keeps the export/inference path and the
    ONNX trace branch-free and unchanged.
    """
    b, v, d = x.shape
    per_view = head(x.reshape(b * v, d)).reshape(b, v, -1)  # (B, V, 1)
    pooled = per_view.mean(dim=1)  # (B, 1)
    return (pooled, per_view) if return_views else pooled
