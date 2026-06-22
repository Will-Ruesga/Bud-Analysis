"""Per-variant and per-task plots.

`report` validates one variant's history; `write_comparison` renders the
across-variant `comparison.png` (e.g. mse vs huber). Both render what the task
already produced — no metric recomputation. The per-sample `predictions.csv` and
`metrics.json` are written by the task into the trial scratch dir and moved into
place by `optimization.keep_best_per_variant` (which must read `metrics.json`
before `report` ever runs).
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write files, never open a window
import matplotlib.pyplot as plt
import numpy as np

from core import data
from core.run_context import RunContext
from core.schemas import TrainResult

_REQUIRED_HISTORY_KEYS = {"epoch", "train_loss", "val_loss", "val_rmse", "val_mae"}


def report(ctx: RunContext, result: TrainResult) -> None:
    """Validate one variant's `history` (Rule 12). No figure, no copy.

    The per-variant figure lives in `comparison.png`; the DATA_DIR predictions
    are written once for all variants by `write_predictions`.
    """
    for entry in result.metrics["history"]:
        missing = _REQUIRED_HISTORY_KEYS - entry.keys()
        if missing:
            raise ValueError(f"history entry missing keys {sorted(missing)}")


def _pred_col(variant: str) -> str:
    """Variant id → prediction column name: `mse` → `predMse`, `huber` → `predHuber`."""
    return "pred" + "".join(w.capitalize() for w in variant.split("_"))


def write_predictions(ctx: RunContext, variant_dirs: list[Path] | None = None) -> Path:
    """Merge the kept variants' predictions into one `<DATA_DIR>/predictions.csv`.

    One row per test `(flower, round)`; columns `fileName, flowerID, roundID,
    class, target`, then one `pred<Variant>` column per compared variant
    (`predMse`, `predHuber`, …). Replaces the old per-variant copies and removes
    any stale `predictions_<task>_*.csv` this pipeline wrote before.
    """
    if variant_dirs is None:
        variant_dirs = sorted(
            d for d in ctx.task_dir.iterdir()
            if d.is_dir() and (d / "predictions.csv").exists()
        )
    merged = None
    for agg_dir in variant_dirs:
        df = data.read_index(agg_dir / "predictions.csv")
        df["round_id"] = df["round_id"].fillna("").astype(str)
        col = _pred_col(agg_dir.name)
        if merged is None:
            merged = df[["file_name", "flower_id", "round_id", "class", "target"]].copy()
            merged[col] = df["prediction"].to_numpy()
        else:
            sub = df[["flower_id", "round_id", "prediction"]].rename(columns={"prediction": col})
            merged = merged.merge(sub, on=["flower_id", "round_id"], how="outer")
    if merged is None:
        raise ValueError(f"no variant predictions under {ctx.task_dir}")

    data_dir = Path(ctx.data_dir())
    for stale in data_dir.glob(f"predictions_{ctx.task}_*.csv"):
        stale.unlink()
    dest = data_dir / "predictions.csv"
    data.build_index(merged.to_dict("records"), dest)
    return dest


_PRETTY_VARIANT = {"mse": "MSE", "huber": "Huber", "mil_mean": "MIL Mean", "top_only": "Top Only"}


def _pretty_variant(name: str) -> str:
    """Variant id → display name: `mse` → "MSE", `huber` → "Huber" (legacy agg names too)."""
    return _PRETTY_VARIANT.get(name, name.replace("_", " ").title())


def _scatter_pred_true(ax, predictions, labels, classes, metrics=None, legend=True):
    """Predicted-vs-true scatter, coloured by class, over a ±0.1 tolerance band.

    The grey band marks where |pred − true| ≤ 0.1 — a quick read of "close
    enough". Filled at low opacity (with faint edges) so points stay visible
    through it but the band is still clearly there.
    """
    # Draw the band past [0,1] so the axes clip it at the plot edge instead of
    # leaving white space where it would otherwise stop short.
    xs = np.linspace(-0.15, 1.15, 200)
    ax.fill_between(xs, xs - 0.1, xs + 0.1, color="grey", alpha=0.22, lw=0, zorder=0)
    ax.plot(xs, xs - 0.1, color="grey", lw=1.2, alpha=0.5, zorder=1)
    ax.plot(xs, xs + 0.1, color="grey", lw=1.2, alpha=0.5, zorder=1)
    ax.plot([-0.15, 1.15], [-0.15, 1.15], "k--", lw=1.0, alpha=0.7, zorder=1)

    for cls in sorted(set(classes)):
        m = classes == cls
        ax.scatter(labels[m], predictions[m], s=12, alpha=0.6, zorder=2, label=cls)
    # pad the limits a touch so points at true/pred 0 or 1 aren't clipped at the edge
    ax.set(xlim=(-0.04, 1.04), ylim=(-0.04, 1.04), xlabel="true", ylabel="predicted")
    if metrics:
        rmse, mae = metrics.get("rmse"), metrics.get("mae")
        if rmse is not None and mae is not None:
            ax.text(0.03, 0.92, f"RMSE {rmse:.3f}  MAE {mae:.3f}", fontsize=9)
    if legend:
        ax.legend(fontsize=7, loc="lower right", title="class")


def write_comparison(
    ctx: RunContext,
    task: str | None = None,
    variant_dirs: list[Path] | None = None,
) -> Path:
    """Write `<task>/comparison.png`: val_rmse + loss curves and per-variant scatters.

    Top row: val_rmse and train/val loss curves across the compared variants
    (e.g. mse vs huber). Bottom row: one predicted-vs-true scatter (with ±0.1 band)
    per variant, captioned with its robustness numbers. Reads each variant dir's
    `metrics.json` (history) and `predictions.csv`. Defaults to every subdir of
    `ctx.task_dir` holding a `metrics.json`. Returns the path.
    """
    task = task or ctx.task
    if variant_dirs is None:
        variant_dirs = sorted(
            d for d in ctx.task_dir.iterdir()
            if d.is_dir() and (d / "metrics.json").exists()
        )

    n = len(variant_dirs)
    ncols = max(n, 2)
    # Wider-than-tall cells: each column ~4.5 wide, each of the 2 rows ~3.2 tall,
    # so the bottom scatter panels read landscape (x longer than y) instead of squished.
    fig = plt.figure(figsize=(4.5 * ncols, 6.5))
    gs = fig.add_gridspec(2, ncols)
    ax_rmse = fig.add_subplot(gs[0, : ncols // 2])
    ax_loss = fig.add_subplot(gs[0, ncols // 2 :])

    for i, agg_dir in enumerate(variant_dirs):
        name = agg_dir.name
        metrics = json.loads((agg_dir / "metrics.json").read_text())
        history = metrics["history"]
        epochs = [h["epoch"] for h in history]
        color = f"C{i}"

        pretty = _pretty_variant(name)
        ax_rmse.plot(epochs, [h["val_rmse"] for h in history], label=pretty, color=color)
        ax_loss.plot(epochs, [h["train_loss"] for h in history], "--", color=color)
        ax_loss.plot(epochs, [h["val_loss"] for h in history], "-", color=color, label=pretty)

        ax_sc = fig.add_subplot(gs[1, i])
        _render_scatter_panel(ax_sc, agg_dir / "predictions.csv", pretty,
                              robustness=_robustness_caption(metrics))

    ax_rmse.set(xlabel="epoch", ylabel="val_rmse")
    ax_rmse.set_title("Val RMSE", fontweight="bold")
    ax_rmse.legend(fontsize=7)
    ax_loss.set(xlabel="epoch", ylabel="loss")
    ax_loss.set_title(r"$\mathbf{Loss}$ — train (--) / val (—)")  # only "Loss" bold
    ax_loss.legend(fontsize=7)

    fig.suptitle(f"Comparison - {task.title()} - {ctx.root.name}", fontweight="bold")
    fig.tight_layout()
    out_path = ctx.task_dir / "comparison.png"
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def _robustness_caption(metrics: dict) -> str | None:
    """Compact 'view range / round σ / λ' line for the scatter panel.

    Returns None when the run predates the robustness metrics (old `metrics.json`
    without these keys), so legacy comparisons render unchanged.
    """
    parts = []
    if metrics.get("view_range") is not None:
        parts.append(f"view range {metrics['view_range']:.3f}")
    if metrics.get("round_std") is not None:
        parts.append(f"round σ {metrics['round_std']:.3f}")
    lam = metrics.get("lambda_consistency")
    if lam:
        parts.append(f"λ {lam:.3g}")
    return "  ·  ".join(parts) if parts else None


def _render_scatter_panel(ax, predictions_csv, title, robustness=None):
    """One aggregator's predicted-vs-true scatter (with ±0.1 band) from its CSV."""
    df = data.read_index(predictions_csv)
    _scatter_pred_true(
        ax,
        df["prediction"].to_numpy(),
        df["target"].to_numpy(),
        df["class"].astype(str).to_numpy(),
        legend=False,
    )
    ax.set_title(title, fontweight="bold")
    if robustness:
        ax.text(0.02, 0.98, robustness, transform=ax.transAxes, fontsize=7,
                va="top", ha="left",
                bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85))


_SPLITS = ["train", "val", "test"]


def _distribution_stats(df) -> dict:
    """Counts behind the prep distribution figure — pure, no rendering.

    Returns sizes at three granularities per split and overall, plus the
    flower count per (class, split). A *flower* is a unique `flower_id`; a
    *round* is a unique `(flower_id, round_id)` — the granularity the
    training loop actually trains on (so the two differ only when the dataset
    has round numbers). Class and split are constant within a flower.
    """
    d = df.copy()
    d["round_id"] = d["round_id"].fillna("").astype(str)
    d["class"] = d["class"].astype(str)

    def sizes(rows):
        return {
            "images": len(rows),
            "flowers": rows["flower_id"].nunique(),
            "rounds": rows.drop_duplicates(["flower_id", "round_id"]).shape[0],
        }

    sizes_by_split = {s: sizes(d[d["split"] == s]) for s in _SPLITS}
    sizes_by_split["all"] = sizes(d)

    flowers = d.drop_duplicates("flower_id")
    classes = sorted(
        flowers["class"].unique(),
        key=lambda c: flowers.loc[flowers["class"] == c, "target"].iloc[0],
    )
    flowers_per_class_split = {
        c: {s: int(((flowers["class"] == c) & (flowers["split"] == s)).sum()) for s in _SPLITS}
        for c in classes
    }
    return {
        "sizes": sizes_by_split,
        "classes": classes,
        "flowers_per_class_split": flowers_per_class_split,
    }


def plot_dataset_distribution(ctx: RunContext, df) -> Path:
    """Write `<run>/prep/distribution.png`: a dataset overview.

    Left (the visual): unique flowers per class, stacked by split.
    Right (the numbers): a counts table, split × granularity, since those scales
    differ too much to compare as bars. Display terms match the dataset's
    domain: a `(flower_id, round_id)` sample is a "flower", a `flower_id` is a
    "unique flower". `df` is the snake_case index from `data.run`. Returns path.
    """
    stats = _distribution_stats(df)
    classes, fpcs, sizes = stats["classes"], stats["flowers_per_class_split"], stats["sizes"]

    fig = plt.figure(figsize=(13, 5))
    ax_cls, ax_tbl = fig.subplots(1, 2, gridspec_kw={"width_ratios": [1.5, 1]})

    # left: unique flowers (flower_id) per class, stacked by split
    x = np.arange(len(classes))
    bottom = np.zeros(len(classes))
    for i, s in enumerate(_SPLITS):
        vals = np.array([fpcs[c][s] for c in classes], dtype=float)
        ax_cls.bar(x, vals, bottom=bottom, label=s, color=f"C{i}")
        bottom += vals
    for xi, total in zip(x, bottom):
        ax_cls.text(xi, total, str(int(total)), ha="center", va="bottom", fontsize=10)
    ax_cls.set_xticks(x)
    ax_cls.set_xticklabels(classes)
    ax_cls.set(xlabel="class", ylabel="unique flowers",
               title="Unique flowers per class (stacked by split)")
    ax_cls.legend(title="split", fontsize=9)
    ax_cls.margins(y=0.12)

    # right: exact counts as a table (scales differ too much for shared bars).
    # stat keys -> the dataset's display terms.
    gran_keys = ["flowers", "rounds", "images"]
    gran_labels = ["unique flowers", "flowers", "images"]
    rows = _SPLITS + ["all"]
    cell_text = [[f"{sizes[r][k]:,}" for k in gran_keys] for r in rows]
    ax_tbl.axis("off")
    tbl = ax_tbl.table(cellText=cell_text, rowLabels=rows, colLabels=gran_labels,
                       cellLoc="center", loc="center", bbox=[0.12, 0.18, 0.86, 0.64])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    for (r, _c), cell in tbl.get_celld().items():
        if r == 0:                               # header row
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#eaeaea")
        elif rows[r - 1] == "all":               # totals row
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#f4f4f4")
    ax_tbl.set_title("Counts by split", pad=2)
    uf, fw = sizes["all"]["flowers"], sizes["all"]["rounds"]
    im = sizes["all"]["images"]
    ax_tbl.text(0.5, 0.06, f"≈ {fw / uf:.0f} flowers / unique flower · {im / fw:.0f} views / flower",
                ha="center", fontsize=9, color="#555", transform=ax_tbl.transAxes)

    fig.suptitle(f"{ctx.cultivar} · {ctx.task} — dataset distribution", fontsize=13)
    fig.tight_layout()
    out_path = ctx.prep_distribution_png
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path
