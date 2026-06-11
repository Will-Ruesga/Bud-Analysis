"""Frozen-backbone feature extraction with an on-disk `.npy` cache.

Runs every row of `index.csv` through the frozen backbone once and caches
the per-image feature. Idempotent: rows whose `.npy` already exists are
skipped, and the backbone is never loaded when nothing is missing. See
docs/core/embeddings.md.
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from core import backbones
from core.run_context import RunContext


def _patch_keep_mask(alpha: Image.Image, n_side: int) -> np.ndarray:
    """Binary `(n_side*n_side,)` — True for patches overlapping the bud.

    The alpha mask is resized to the patch grid (`n_side` patches/side, 16 px
    each) and a patch is kept if **any** of its pixels is bud (`alpha > 0`).
    Row-major, to match `x_norm_patchtokens`. Edge patches with a sliver of bud
    count fully (binary), since boundary detail is informative.
    """
    size = n_side * 16
    m = np.asarray(alpha.resize((size, size), Image.NEAREST)) > 0
    return m.reshape(n_side, 16, n_side, 16).any(axis=(1, 3)).reshape(-1)


def _invalidate_if_pooling_changed(ctx: RunContext, pooling: str) -> None:
    """Drop the cache if it was built with a different pooling.

    The `.npy` contents depend on `pooling` but `image_id` doesn't, so existence
    checks alone can't tell — compare the sidecar's recorded pooling.
    """
    if not ctx.embeddings_meta_json.exists():
        return
    old = json.loads(ctx.embeddings_meta_json.read_text()).get("pooling", "cls")
    if old != pooling:
        for f in ctx.embeddings_dir.glob("*.npy"):
            f.unlink()
        ctx.embeddings_meta_json.unlink()


def extract(
    ctx: RunContext,
    batch_size: int = 32,
    device: str = "auto",
    pooling: str = "cls",
) -> dict[str, np.ndarray]:
    """Compute (or load) per-image embeddings, returning `{image_id: (D,)}`.

    Reads `ctx.index()`, splits rows into already-cached vs missing by the
    presence of `ctx.embedding_path(image_id)`. Missing rows are run through
    the frozen backbone (loaded once via `backbones.load_dinov3`) in batches
    of `batch_size` and written as `float32` `.npy`. If nothing is missing
    the backbone is never touched. `meta.json` is (re)written from disk facts
    whenever absent — including the case where every `.npy` exists but the
    sidecar was deleted.

    Args:
        ctx: run context; supplies the index, paths, backbone name and
            checkpoint (`ctx.backbone_checkpoint`).
        batch_size: number of images per backbone forward pass.
        device: "auto" picks CUDA when available, else CPU.
        pooling: "cls" uses the CLS token (default); "masked_mean" uses the
            image's alpha channel as a bud mask and averages only the patch
            tokens that overlap the bud (background patches dropped).

    Returns:
        Full in-memory map `{image_id: (D,) float32 np.ndarray}` for every
        row in the index.
    """
    df = ctx.index()
    image_ids = list(df["image_id"])
    file_by_id = dict(zip(df["image_id"], df["file_name"]))

    ctx.embeddings_dir.mkdir(parents=True, exist_ok=True)
    _invalidate_if_pooling_changed(ctx, pooling)
    missing = [iid for iid in image_ids if not ctx.embedding_path(iid).exists()]

    if missing:
        if ctx.backbone_checkpoint is None:
            raise ValueError(
                "ctx.backbone_checkpoint is None but embeddings are missing; "
                "the run manifest has no backbone_checkpoint — re-run prepare with a "
                "backbone whose checkpoint is set in config.BACKBONE_CHECKPOINTS."
            )
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        model = backbones.load_dinov3(ctx.backbone_name, ctx.backbone_checkpoint, device)
        transform = backbones.eval_transform(ctx.backbone_name)
        data_root = Path(ctx.data_dir())
        n_side = backbones.patch_grid(ctx.backbone_name)

        with tqdm(total=len(missing), desc=f"extracting embeddings ({pooling})", unit="img") as bar:
            for start in range(0, len(missing), batch_size):
                chunk = missing[start : start + batch_size]
                tensors, keeps = [], []
                for iid in chunk:
                    rel = file_by_id[iid].replace("\\", "/")
                    with Image.open(data_root / rel) as img:
                        if pooling == "masked_mean":
                            rgba = img.convert("RGBA")
                            tensors.append(transform(rgba.convert("RGB")))
                            keeps.append(_patch_keep_mask(rgba.getchannel("A"), n_side))
                        else:
                            tensors.append(transform(img))
                batch = torch.stack(tensors).to(device)
                with torch.no_grad():
                    if pooling == "masked_mean":
                        tokens = backbones.feature_tokens(model, batch)  # (B, N, D)
                        w = torch.as_tensor(np.stack(keeps), dtype=torch.float32, device=device)
                        w = w / w.sum(1, keepdim=True).clamp(min=1.0)    # mean over kept patches
                        feats = (tokens * w.unsqueeze(-1)).sum(1).cpu().numpy().astype(np.float32)
                    else:
                        feats = model(batch).cpu().numpy().astype(np.float32)  # (B, D) CLS
                for iid, feat in zip(chunk, feats):
                    np.save(ctx.embedding_path(iid), feat)
                bar.update(len(chunk))

    emb_by_id = {iid: np.load(ctx.embedding_path(iid)) for iid in image_ids}

    if not ctx.embeddings_meta_json.exists():
        meta = {
            "backbone_name": ctx.backbone_name,
            "feature_dim": backbones.feature_dim(ctx.backbone_name),
            "pooling": pooling,
            "created": datetime.now().strftime("%Y_%m_%dT%H:%M:%S"),
            "n_images": len(image_ids),
        }
        ctx.embeddings_meta_json.write_text(json.dumps(meta, indent=2))

    return emb_by_id
