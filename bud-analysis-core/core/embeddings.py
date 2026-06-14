"""Frozen-backbone feature extraction with an on-disk `.npy` cache.

Runs every row of `index.csv` through the frozen backbone once and caches the
per-image CLS feature. Idempotent: rows whose `.npy` already exists are skipped,
and the backbone is never loaded when nothing is missing. See
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


def extract(
    ctx: RunContext,
    batch_size: int = 32,
    device: str = "auto",
) -> dict[str, np.ndarray]:
    """Compute (or load) per-image CLS embeddings, returning `{image_id: (D,)}`.

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

    Returns:
        Full in-memory map `{image_id: (D,) float32 np.ndarray}` for every
        row in the index.
    """
    df = ctx.index()
    image_ids = list(df["image_id"])
    file_by_id = dict(zip(df["image_id"], df["file_name"]))

    ctx.embeddings_dir.mkdir(parents=True, exist_ok=True)
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

        with tqdm(total=len(missing), desc="extracting embeddings (cls)", unit="img") as bar:
            for start in range(0, len(missing), batch_size):
                chunk = missing[start : start + batch_size]
                tensors = []
                for iid in chunk:
                    rel = file_by_id[iid].replace("\\", "/")
                    with Image.open(data_root / rel) as img:
                        tensors.append(transform(img))
                batch = torch.stack(tensors).to(device)
                with torch.no_grad():
                    feats = model(batch).cpu().numpy().astype(np.float32)  # (B, D) CLS
                for iid, feat in zip(chunk, feats):
                    np.save(ctx.embedding_path(iid), feat)
                bar.update(len(chunk))

    emb_by_id = {iid: np.load(ctx.embedding_path(iid)) for iid in image_ids}

    if not ctx.embeddings_meta_json.exists():
        meta = {
            "backbone_name": ctx.backbone_name,
            "feature_dim": backbones.feature_dim(ctx.backbone_name),
            "created": datetime.now().strftime("%Y_%m_%dT%H:%M:%S"),
            "n_images": len(image_ids),
        }
        ctx.embeddings_meta_json.write_text(json.dumps(meta, indent=2))

    return emb_by_id
