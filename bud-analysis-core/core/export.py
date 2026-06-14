"""Unified ONNX export: N trained heads sharing one frozen backbone.

Walks `<task>/<variant>/` dirs (the filesystem is the registry), rebuilds each
head from its `metrics.json` + `head.pt`, and traces a single graph: preprocess →
shared backbone (per view) → per-head `mil_pool` → concat. Every head consumes all
views (the one mil_mean late-fusion pipeline), so exported and trained pooling are
the same op. See docs/core/export.md.
"""

import json
from pathlib import Path

import torch
from torch import nn

from core import backbones, heads as heads_module
from core.heads import mil_pool
from core.data import CANONICAL_VIEW_TYPES
from core.run_context import RunContext
from core.schemas import HeadSpec


def export(
    ctx: RunContext,
    heads: list[tuple[str, str]] | None = None,
    output_path: Path | None = None,
    opset_version: int = 18,
    device: str = "auto",
) -> Path:
    """Export the requested heads into one self-contained ONNX file.

    `heads` is a list of `(task, variant)`; `variant == "auto"` resolves to the
    lowest-`selection_score` variant under `<task>/`. Defaults to
    `[(ctx.task, "auto")]`. Returns the written `.onnx` path.
    """
    if heads is None:
        heads = [(ctx.task, "auto")]

    resolved = [_resolve_head(ctx, task, agg) for task, agg in heads]
    if not resolved:
        raise ValueError("no heads to export")

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if ctx.backbone_checkpoint is None:
        raise ValueError("ctx.backbone_checkpoint is None; cannot load the backbone to export.")
    backbone = backbones.load_dinov3(ctx.backbone_name, ctx.backbone_checkpoint, device)

    variant = backbones._VARIANTS[ctx.backbone_name]

    # The one pipeline feeds every head all views (mil_mean late fusion).
    graph = _ExportGraph(backbone, resolved, ctx.backbone_name).to(device).eval()

    if output_path is None:
        output_path = ctx.onnx_dir / _default_filename(ctx, resolved)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dummy = torch.zeros(
        1, len(CANONICAL_VIEW_TYPES), variant["image_size"], variant["image_size"], 3, device=device
    )
    torch.onnx.export(
        graph,
        dummy,
        str(output_path),
        input_names=["images"],
        output_names=["scores"],
        dynamic_axes={"images": {0: "batch", 2: "height", 3: "width"}, "scores": {0: "batch"}},
        opset_version=opset_version,
        dynamo=False,  # legacy TorchScript exporter — avoids the onnxscript dependency
    )
    return output_path


def _resolve_head(ctx: RunContext, task: str, agg_name: str):
    """Resolve one (task, variant) → (task, variant, head_spec, head_module)."""
    task_dir = ctx.root / f"{task}-results"
    if agg_name == "auto":
        agg_name = _auto_select(task_dir)

    agg_dir = task_dir / agg_name
    metrics_path = agg_dir / "metrics.json"
    head_pt = agg_dir / "head.pt"
    if not metrics_path.exists() or not head_pt.exists():
        raise FileNotFoundError(
            f"head dir {agg_dir} must contain metrics.json and head.pt"
        )

    hs = json.loads(metrics_path.read_text())["head_spec"]
    head_spec = HeadSpec(
        aggregator_name=hs["aggregator_name"],
        hidden_dims=tuple(hs["hidden_dims"]),
        dropout=hs["dropout"],
    )
    head = heads_module.build(head_spec, ctx.backbone_name)
    head.load_state_dict(torch.load(head_pt, map_location="cpu"))
    head.eval()
    return task, agg_name, head_spec, head


def _auto_select(task_dir: Path) -> str:
    """Variant dir with the lowest `selection_score` (val_rmse fallback for legacy
    runs; alphabetical tie-break)."""
    candidates = []
    if task_dir.exists():
        for d in sorted(task_dir.iterdir()):
            mp = d / "metrics.json"
            if d.is_dir() and mp.exists():
                m = json.loads(mp.read_text())
                val = m.get("selection_score", m.get("val_rmse", float("inf")))
                candidates.append((val, d.name))
    if not candidates:
        raise FileNotFoundError(f"no variant dirs with metrics.json under {task_dir}")
    candidates.sort(key=lambda c: (c[0], c[1]))
    return candidates[0][1]


def _default_filename(ctx: RunContext, resolved) -> str:
    if len(resolved) == 1:
        task, agg = resolved[0][0], resolved[0][1]
        return f"{task}_{agg}_{ctx.backbone_name}.onnx"
    tasks = "_".join(sorted({task for task, _, _, _ in resolved}))
    return f"{tasks}_{ctx.backbone_name}.onnx"


class _ExportGraph(nn.Module):
    """Traceable graph: (B,V,H,W,3) → preprocess → backbone (per view) → per-head
    `mil_pool` → concat → (B,) or (B,N). Every head consumes all V views."""

    def __init__(self, backbone, resolved, backbone_name):
        super().__init__()
        self.backbone = backbone
        self.heads = nn.ModuleList([head for _, _, _, head in resolved])
        self.backbone_name = backbone_name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        b, v = x.shape[0], x.shape[1]
        x = x.permute(0, 1, 4, 2, 3)  # (B, V, 3, H, W)
        x = x.reshape(b * v, x.shape[2], x.shape[3], x.shape[4])
        x = backbones.preprocess(x, self.backbone_name)  # same op as eval_transform
        feats = self.backbone(x).reshape(b, v, -1)  # (B, V, D)

        outs = [mil_pool(head, feats) for head in self.heads]  # each (B, 1)
        out = torch.cat(outs, dim=1)  # (B, N)
        if out.shape[1] == 1:
            out = out.squeeze(1)  # (B,)
        return out
