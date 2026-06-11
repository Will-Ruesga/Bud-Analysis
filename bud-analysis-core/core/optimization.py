"""Optuna helpers shared by every task's study.

Three task-agnostic pieces; the study, search space, and objective body live
in `task/train.py`. See docs/core/optimization.md.
"""

import json
import shutil
from pathlib import Path
from typing import Iterable

import optuna
import torch
from torch import nn

from core.run_context import RunContext


def build(
    parameters: Iterable[nn.Parameter],
    hparams: dict,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    """AdamW + CosineAnnealingLR from `hparams`.

    Reads only `lr` and `epochs` (plus optional `weight_decay`). A task wanting
    a different optimiser writes one inline and skips this.
    """
    optimizer = torch.optim.AdamW(
        parameters,
        lr=hparams["lr"],
        weight_decay=hparams.get("weight_decay", 0.0),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=hparams["epochs"]
    )
    return optimizer, scheduler


def keep_best_per_aggregator(
    ctx: RunContext,
    study_dir: Path,
    metric_key: str = "val_rmse",
) -> dict[str, Path]:
    """Promote the best trial of each aggregator, delete the rest.

    Each subdir of `study_dir` is one trial with a `metrics.json` carrying
    `head_spec.aggregator_name` and `metric_key`. For each aggregator the trial
    with the lowest `metric_key` is moved to `ctx.aggregator_dir(<aggregator>)`;
    the entire scratch `study_dir` is then removed. Returns
    `{aggregator_name: kept_path}`.
    """
    study_dir = Path(study_dir)

    best: dict[str, tuple[float, Path]] = {}  # aggregator -> (value, trial_dir)
    for trial_dir in sorted(study_dir.iterdir()):
        if not trial_dir.is_dir():
            continue
        metrics = json.loads((trial_dir / "metrics.json").read_text())
        agg = metrics["head_spec"]["aggregator_name"]
        value = metrics[metric_key]
        if agg not in best or value < best[agg][0]:
            best[agg] = (value, trial_dir)

    kept: dict[str, Path] = {}
    for agg, (_, trial_dir) in best.items():
        dest = ctx.aggregator_dir(agg)
        if dest.exists():
            shutil.rmtree(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(trial_dir), str(dest))
        kept[agg] = dest

    # Winners are moved out; the scratch dir (losers + empty) is consumed whole.
    if study_dir.exists():
        shutil.rmtree(study_dir)
    return kept


def write_study_summary(
    study: optuna.Study,
    out_dir: Path,
    metric_key: str = "val_rmse",
) -> Path:
    """Write `<out_dir>/study_summary.json` (canonical schema).

    `trials[]` holds only the survivors — the best completed trial per
    aggregator — while `n_trials`/`n_completed` reflect the whole study.
    """
    minimize = study.direction == optuna.study.StudyDirection.MINIMIZE
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    best: dict[str, optuna.trial.FrozenTrial] = {}
    for t in completed:
        agg = t.params.get("aggregator_name")
        if agg is None:
            continue
        if agg not in best or (
            t.value < best[agg].value if minimize else t.value > best[agg].value
        ):
            best[agg] = t

    trials = [
        {
            "kept_as": agg,
            "trial_number": best[agg].number,
            "value": best[agg].value,
            "params": best[agg].params,
            "state": best[agg].state.name,
        }
        for agg in sorted(best)
    ]

    summary = {
        "method": "regression",
        "metric": metric_key,
        "direction": "minimize" if minimize else "maximize",
        "n_trials": len(study.trials),
        "n_completed": len(completed),
        "aggregators_kept": sorted(best),
        "best_value": study.best_value,
        "best_params": study.best_params,
        "trials": trials,
    }

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "study_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    return out_path
