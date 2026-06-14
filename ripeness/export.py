"""export.py — bundle trained head(s) into a single ONNX. Thin CLI wrapper."""

import argparse

from core import export as core_export
from core.run_context import RunContext


def _parse_heads(spec_list):
    """['ripeness:auto', 'defects:huber'] → [('ripeness','auto'), ('defects','huber')]."""
    out = []
    for item in spec_list:
        task, _, variant = item.partition(":")
        out.append((task, variant or "auto"))
    return out


def main(run_dir, variant=None, heads=None, output_path=None):
    """Resolve the requested head(s) for a prepared run and export them to one ONNX file."""
    ctx = RunContext.from_info_json(run_dir)
    if heads is None:
        heads = [(ctx.task, variant or "auto")]
    return core_export.export(ctx, heads=heads, output_path=output_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-run", required=True, help="run dir from prepare (e.g. output/<run>)")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--variant", help="which kept variant to export, e.g. mse / huber / auto")
    g.add_argument("--heads", nargs="+", help="multi-task: task:variant, e.g. ripeness:auto defects:huber")
    ap.add_argument("--output_path")
    a = ap.parse_args()
    heads = _parse_heads(a.heads) if a.heads else None
    print(main(a.run, variant=a.variant, heads=heads, output_path=a.output_path))
