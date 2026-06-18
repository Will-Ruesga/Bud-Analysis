"""Optuna helpers shared by every task's study.

Three task-agnostic pieces; the study, search space, and objective body live
in `task/train.py`.
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


def keep_best_per_variant(
    ctx: RunContext,
    study_dir: Path,
    group_key: str = "loss",
    metric_key: str = "selection_score",
) -> dict[str, Path]:
    """Promote the best trial of each variant, delete the rest.

    The study compares one dimension (`group_key`, e.g. `loss`). Each subdir of
    `study_dir` is one trial whose `metrics.json` carries `group_key` and
    `metric_key`. For each distinct `group_key` value the trial with the lowest
    `metric_key` is moved to `ctx.variant_dir(<value>)`; the entire scratch
    `study_dir` is then removed. Returns `{group_value: kept_path}`.
    """
    study_dir = Path(study_dir)

    best: dict[str, tuple[float, Path]] = {}  # group value -> (metric, trial_dir)
    for trial_dir in sorted(study_dir.iterdir()):
        if not trial_dir.is_dir():
            continue
        metrics = json.loads((trial_dir / "metrics.json").read_text())
        value = str(metrics[group_key])
        score = metrics[metric_key]
        if value not in best or score < best[value][0]:
            best[value] = (score, trial_dir)

    kept: dict[str, Path] = {}
    for value, (_, trial_dir) in best.items():
        dest = ctx.variant_dir(value)
        if dest.exists():
            shutil.rmtree(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(trial_dir), str(dest))
        kept[value] = dest

    # Winners are moved out; the scratch dir (losers + empty) is consumed whole.
    if study_dir.exists():
        shutil.rmtree(study_dir)
    return kept


def _variant_key(trial: optuna.trial.FrozenTrial, group_params: list[str]) -> str | None:
    """Composite variant id for a trial: its `group_params` values joined by '-'.

    Empty `group_params` (nothing compared) → the single `"default"` variant. A
    trial missing one of the params (shouldn't happen) → None, so it is skipped.
    """
    if not group_params:
        return "default"
    if any(p not in trial.params for p in group_params):
        return None
    return "-".join(str(trial.params[p]) for p in group_params)


def write_study_summary(
    study: optuna.Study,
    out_dir: Path,
    group_params: list[str],
    metric_key: str = "selection_score",
) -> Path:
    """Write `<out_dir>/study_summary.json` (canonical schema).

    `group_params` is the list of compared dimensions (the optuna params that vary
    across variants); trials are grouped by their composite value (see
    `_variant_key`). `trials[]` holds only the survivors — the best completed trial
    per variant — while `n_trials`/`n_completed` reflect the whole study.
    """
    minimize = study.direction == optuna.study.StudyDirection.MINIMIZE
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    best: dict[str, optuna.trial.FrozenTrial] = {}
    for t in completed:
        value = _variant_key(t, group_params)
        if value is None:
            continue
        if value not in best or (
            t.value < best[value].value if minimize else t.value > best[value].value
        ):
            best[value] = t

    trials = [
        {
            "kept_as": value,
            "trial_number": best[value].number,
            "value": best[value].value,
            "params": best[value].params,
            "state": best[value].state.name,
        }
        for value in sorted(best)
    ]

    summary = {
        "method": "regression",
        "metric": metric_key,
        "direction": "minimize" if minimize else "maximize",
        "n_trials": len(study.trials),
        "n_completed": len(completed),
        "compared": group_params,
        "variants_kept": sorted(best),
        "best_value": study.best_value,
        "best_params": study.best_params,
        "trials": trials,
    }

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "study_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    return out_path
