"""check_crop.py — eyeball the adaptive circle before spending a run on it.

Samples random top views across classes and renders, per image: the original, the
masked result, and the mask boundary drawn over the original. Pixels come from
`core.backbones.preprocess` — the *same* op extraction and the ONNX graph call — so
what you see here is exactly what the backbone will get, not a re-implementation.

    python check_crop.py --dataset <dir>              # 3 images, one per random class
    python check_crop.py --dataset <dir> --per-class  # one from every class

Writes `<output>/crop_check.png`. Nothing else reads it; this is a human check.
"""

import argparse
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from core import backbones

_TOP_SUFFIX = "_4"  # config.VIEWS maps page index 4 -> top


def _load(path: Path) -> torch.Tensor:
    """PIL → `(1, 3, H, W)` float in [0, 255], the layout `preprocess` takes."""
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def _denorm(x: torch.Tensor, backbone_name: str) -> np.ndarray:
    """Undo `preprocess`' normalise → `(H, W, 3)` uint8, so the figure shows pixels
    as the backbone sees them rather than a separately-built approximation."""
    v = backbones._VARIANTS[backbone_name]
    mean = torch.tensor(v["mean"]).view(1, 3, 1, 1)
    std = torch.tensor(v["std"]).view(1, 3, 1, 1)
    img = ((x * std + mean) * 255.0).clamp(0, 255)
    return img.squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8)


def _resized(x: torch.Tensor, image_size: int) -> torch.Tensor:
    """The resize `preprocess` does before masking — needed to measure the radius at
    the same scale the mask runs at."""
    return torch.nn.functional.interpolate(
        x, size=(image_size, image_size), mode="bicubic", align_corners=False
    )


def _radius(x_resized: torch.Tensor) -> float:
    """The radius `_adaptive_circle_mask` picks, recovered from its own output so the
    drawn circle can't disagree with the mask actually applied."""
    kept = (backbones._adaptive_circle_mask(x_resized).sum(1) > 10).squeeze(0).numpy()
    if not kept.any():
        return 0.0
    ys, xs = np.nonzero(kept)
    h, w = kept.shape
    return float(np.sqrt((xs - (w - 1) / 2) ** 2 + (ys - (h - 1) / 2) ** 2).max())


def main(data_dir, backbone, image_size, seed, per_class, n, out):
    data_dir = Path(data_dir).resolve()
    classes = sorted(p for p in data_dir.iterdir() if p.is_dir())
    if not classes:
        raise SystemExit(f"no class folders under {data_dir}")

    rng = random.Random(seed)
    picks = classes if per_class else rng.sample(classes, min(n, len(classes)))
    rows = []
    for cdir in picks:
        tops = sorted(p for p in cdir.glob(f"*{_TOP_SUFFIX}.png"))
        if not tops:
            print(f"class {cdir.name}: no top views, skipped")
            continue
        rows.append((cdir.name, rng.choice(tops)))

    fig, axes = plt.subplots(len(rows), 3, figsize=(10.5, 3.5 * len(rows)), squeeze=False)
    for r, (cls, path) in enumerate(rows):
        x = _load(path)
        # Both panels go through the real preprocess: 'none' is the resized original,
        # 'adaptive' is the masked version. Any bug in the mask shows up here.
        plain = _denorm(backbones.preprocess(x, backbone, image_size, "none"), backbone)
        cut = _denorm(backbones.preprocess(x, backbone, image_size, "adaptive"), backbone)
        radius = _radius(_resized(x, image_size))
        kept = (cut.sum(2) > 10).mean()

        axes[r][0].imshow(plain)
        axes[r][0].set_title(f"class {cls} — original", fontsize=10)
        axes[r][1].imshow(plain)
        axes[r][1].add_patch(plt.Circle((image_size / 2 - 0.5, image_size / 2 - 0.5), radius,
                                        fill=False, color="red", lw=1.6))
        axes[r][1].set_title(f"circle r={radius:.0f}px", fontsize=10)
        axes[r][2].imshow(cut)
        axes[r][2].set_title(f"masked — {kept:.0%} flower kept", fontsize=10)
        for ax in axes[r]:
            ax.set_xticks([]); ax.set_yticks([])
        print(f"class {cls}: r={radius:6.1f}px  kept={kept:5.1%}  {path.name}")

    fig.suptitle("adaptive circle — largest centred all-flower disc (core.backbones.preprocess)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.985))  # leave the suptitle its own band
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    import config as cfg

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", "-d", dest="data_dir", required=True)
    ap.add_argument("--backbone", "-bkb", default=cfg.BACKBONE_NAME,
                    choices=list(cfg.BACKBONE_CHECKPOINTS))
    ap.add_argument("--seed", type=int, default=0, help="re-roll to see different images")
    ap.add_argument("--per-class", action="store_true", help="one image from every class")
    ap.add_argument("-n", type=int, default=3, help="number of random classes (ignored with --per-class)")
    ap.add_argument("--out", default=str(Path(cfg.OUTPUT_DIR) / "crop_check.png"))
    a = ap.parse_args()
    main(a.data_dir, a.backbone, backbones.image_size(a.backbone), a.seed, a.per_class, a.n, a.out)
