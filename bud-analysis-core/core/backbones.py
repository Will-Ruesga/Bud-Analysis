"""Backbone wrappers.

Provides a uniform `forward(image) -> (B, D)` over the vendored DINOv3
source, plus the canonical preprocessing transform shared between
extraction and ONNX export. See docs/core/backbones.md.
"""

import sys
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F


# Per-variant constants. Add a new entry to support a new backbone.
# `image_size`, `mean`, `std` are the single source of truth for
# preprocessing — read by both `eval_transform` and the ONNX-baking path
# in `export.py`.
_VARIANTS: dict[str, dict] = {
    "dinov3_vits16": {
        "feature_dim": 384,
        "image_size": 224,
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
    },
    "dinov3_vitb16": {
        "feature_dim": 768,
        "image_size": 224,
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
    },
}


def feature_dim(backbone_name: str) -> int:
    """Embedding dim for a given variant. Raises KeyError on unknown names."""
    return _VARIANTS[backbone_name]["feature_dim"]


def patch_grid(backbone_name: str) -> int:
    """Patches per side at the eval resolution (`image_size / 16`); 224 → 14.

    These are `*16` variants, so the patch size is 16. Used to align an image
    mask with the `(B, N, D)` patch tokens for mask-pooled embeddings.
    """
    return _VARIANTS[backbone_name]["image_size"] // 16


def feature_tokens(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Final-layer patch tokens `(B, N, D)` from a wrapped DINOv3.

    N = `patch_grid(name)**2` spatial tokens (CLS and storage/register tokens
    excluded). `model(x)` still returns the CLS token; this is the alternative
    used for mask-pooled embeddings.
    """
    raw = getattr(model, "model", model)
    out = raw.forward_features(x)
    out = out[0] if isinstance(out, (list, tuple)) else out
    return out["x_norm_patchtokens"]


def preprocess(images: torch.Tensor, backbone_name: str) -> torch.Tensor:
    """Resize + normalise a `(N, 3, H, W)` float batch in `[0, 255]` to
    `(N, 3, S, S)` normalised tensors.

    The **single source of truth** for pixel preprocessing — `eval_transform`
    wraps it at extraction time and `export` calls it inside the ONNX graph, so
    training and inference pixels are produced by the *same* torch op with the
    *same* constants (`image_size`, `mean`, `std` from `_VARIANTS`). No PIL
    resize anywhere, so nothing diverges between train and inference.
    """
    variant = _VARIANTS[backbone_name]
    size = variant["image_size"]
    mean = images.new_tensor(variant["mean"]).view(1, 3, 1, 1)
    std = images.new_tensor(variant["std"]).view(1, 3, 1, 1)
    # Bicubic matches DINOv3's eval transform (torchvision Resize BICUBIC); the
    # source cutouts are already square, so a direct square resize keeps the
    # whole flower instead of centre-cropping its periphery. align_corners=False
    # + no antialias keeps torch-eager and the ONNX Resize op numerically aligned.
    x = F.interpolate(images, size=(size, size), mode="bicubic", align_corners=False)
    return (x / 255.0 - mean) / std


def eval_transform(backbone_name: str) -> Callable:
    """Return the canonical PIL → tensor preprocessing for this backbone.

    The returned callable takes a PIL.Image and returns a `(3, S, S)` float
    tensor, resized + normalised via `preprocess` — the same op the ONNX graph
    bakes in, so extraction and inference pixels match by construction.
    """
    if backbone_name not in _VARIANTS:
        raise KeyError(f"Unknown backbone: {backbone_name!r}")

    def _transform(image: Image.Image) -> torch.Tensor:
        arr = np.asarray(image.convert("RGB"), dtype=np.float32)  # (H, W, 3) in [0, 255]
        chw = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
        return preprocess(chw, backbone_name).squeeze(0).contiguous()

    return _transform


class _DINOWrapper(nn.Module):
    """Frozen DINO wrapper exposing `forward(x) -> (B, D)` CLS features.

    Kept minimal — when DINOv3's hubconf returns something other than
    the CLS tensor directly, extend this wrapper.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def load_dinov3(
    backbone_name: str,
    checkpoint_path: str,
    device: str = "auto",
) -> nn.Module:
    """Load a frozen DINOv3 from the vendored source and local weights.

    Fully offline. Imports **only** the backbone builder from the vendored
    `core/dinov3/` package (not the full `hubconf`, so the detector / segmentor /
    dinotxt / depther code and their extra deps are never imported), builds the
    architecture with `pretrained=False` (no torch.hub weight download/cache),
    then loads `checkpoint_path` directly with `torch.load`. The weights you
    point at are the weights used — no network, no HuggingFace, no `~/.cache`.
    All parameters are frozen (`requires_grad=False`) and the module is set to
    eval mode.
    """
    if backbone_name not in _VARIANTS:
        raise KeyError(f"Unknown backbone: {backbone_name!r}")

    dinov3_path = Path(__file__).parent / "dinov3"
    if not dinov3_path.exists():
        raise FileNotFoundError(
            f"Vendored DINOv3 not found at {dinov3_path}. "
            "Vendor the source there before calling load_dinov3."
        )
    if not Path(checkpoint_path).is_file():
        raise FileNotFoundError(f"Backbone checkpoint not found: {checkpoint_path}")

    if str(dinov3_path) not in sys.path:
        sys.path.insert(0, str(dinov3_path))
    from dinov3.hub import backbones as _dinov3_backbones

    builder = getattr(_dinov3_backbones, backbone_name)  # e.g. dinov3_vits16
    raw = builder(pretrained=False)  # architecture only — no weights, no hubconf
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    raw.load_state_dict(state_dict, strict=True)

    for p in raw.parameters():
        p.requires_grad = False
    raw.eval()

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    return _DINOWrapper(raw).to(device).eval()
