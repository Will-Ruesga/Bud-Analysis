"""train.py — the ripeness training loop (the only task file with real logic).

Owns losses, metrics, the Optuna objective, the loop, and per-trial bookkeeping.
Core supplies optimiser/scheduler, post-study pruning, summary, and plots.
Reads its settings from the run manifest (`--run <run>`); `--lr/--epochs/--seed`
override the frozen hparams per run. Flow: extract embeddings → Optuna study →
keep best per aggregator → report each → study summary → comparison plot.
"""

import argparse
import itertools
import json
from types import SimpleNamespace

import numpy as np
import optuna
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm

from core import aggregators, data, embeddings, heads, optimization, plotting
from core.heads import mil_pool
from core.run_context import RunContext
from core.schemas import HeadSpec, TrainResult


def _rmse(p, t):
    return float(torch.sqrt(torch.mean((p - t) ** 2)))


def _mae(p, t):
    return float(torch.mean(torch.abs(p - t)))


def _weighted_rmse(p, t, w):
    """RMSE with per-sample weights `w` (N, 1) — the error under a reweighted mix."""
    return float(torch.sqrt((w * (p - t) ** 2).sum() / w.sum()))


def _weighted_mae(p, t, w):
    return float((w * torch.abs(p - t)).sum() / w.sum())


def _loss_fn(name):
    # reduction="none": the balance dim weights the per-sample loss itself, so the
    # reduction has to happen here rather than inside the criterion.
    return nn.HuberLoss(reduction="none") if name == "huber" else nn.MSELoss(reduction="none")


def _production_mix(info):
    """`config.PRODUCTION_DISTRIBUTION` (frozen in the manifest) → the two maps
    `core.data`'s balancing helpers take: `{class: group}` and `{group: proportion}`."""
    dist = info["production_distribution"]
    group_of = {c: g for g, spec in dist.items() for c in spec["classes"]}
    target = {g: float(spec["proportion"]) for g, spec in dist.items()}
    return group_of, target


def _fit_selection(balance, classes, flower_ids, train_mask, group_of, target, seed):
    """`(fit_mask, fit_weights)` for one `balance` variant — which train rows the head
    sees and how much each counts.

    Only ever narrows the **train** split; val/test are untouched by design, so all
    three variants are scored on identical rows (see `_objective`). `off` is the
    dataset as collected; `reweight` keeps every row and reweights the loss;
    `subsample` drops whole flowers and leaves the survivors at weight 1.
    """
    if balance == "reweight":
        return train_mask, data.group_weights(classes[train_mask], group_of, target)
    if balance == "subsample":
        keep = data.subsample_groups(
            flower_ids[train_mask], classes[train_mask], group_of, target, seed
        )
        fit_mask = train_mask.copy()
        fit_mask[np.flatnonzero(train_mask)] = keep
        return fit_mask, np.ones(int(fit_mask.sum()))
    if balance != "off":
        raise ValueError(f"unknown balance {balance!r}")
    return train_mask, np.ones(int(train_mask.sum()))


def _view_spread(per_view):
    """(range_mean, std_mean) of per-view predictions, averaged over rounds.

    `per_view` is `(B, V, 1)` (the tensor from `mil_pool(..., return_views=True)`).
    Range = mean over rounds of `max - min` across views; std = mean of the
    per-round view std. Single-view aggregators (V=1) give 0 for both.
    """
    pv = per_view.reshape(per_view.shape[0], per_view.shape[1])  # (B, V)
    rng = float((pv.max(dim=1).values - pv.min(dim=1).values).mean())
    sd = float(pv.std(dim=1, unbiased=False).mean())
    return rng, sd


def _round_to_round_std(pooled, keys):
    """Mean over flowers of the std of their round predictions.

    Only flowers with >= 2 rounds contribute (a single round has no spread).
    `pooled` is a 1-D array of round-level predictions aligned with `keys`
    (`(flower_id, round_id)`). Returns None if no flower has >= 2 rounds.
    """
    by_flower: dict = {}
    for (flower_id, _round_id), p in zip(keys, pooled):
        by_flower.setdefault(flower_id, []).append(float(p))
    stds = [float(np.std(v, ddof=1)) for v in by_flower.values() if len(v) > 1]
    return float(np.mean(stds)) if stds else None


def _per_key_meta(index):
    """{(flower_id, round_id): {target, split, class, top_file}}."""
    idx = index.copy()
    idx["round_id"] = idx["round_id"].fillna("")
    meta = {}
    for (flower_id, round_id), grp in idx.groupby(["flower_id", "round_id"]):
        top = grp[grp["view_type"] == "top"]
        meta[(flower_id, round_id)] = {
            "target": float(grp["target"].iloc[0]),
            "split": grp["split"].iloc[0],
            "class": str(grp["class"].iloc[0]),
            "top_file": (top if len(top) else grp)["file_name"].iloc[0],
        }
    return meta


def _forward(head, X, return_views=False):
    """Pooled round prediction `(B, 1)`; with `return_views`, also `per_view (B, V, 1)`.

    `X` is always a stacked MIL bag `(N, V, D)` (all views, late fusion), reduced
    by `mil_pool`.
    """
    return mil_pool(head, X, return_views=return_views)


def _suggest_lambda(trial, search_space):
    """λ for the view-consistency penalty from `lambda_consistency: [lo, hi]`.

    Absent or a degenerate `[x, x]` range → the constant `x` (no search, default
    0.0), so old manifests without the key train exactly as before.
    """
    lo, hi = search_space.get("lambda_consistency", [0.0, 0.0])
    return trial.suggest_float("lambda_consistency", lo, hi) if hi > lo else float(lo)


def _objective(trial, ctx, index, emb, settings, scratch):
    # Compared dimensions (the grid) come from the manifest; everything else in
    # OPT_SEARCH_SPACE is searched within each variant. View pipeline is mil_mean.
    compare = settings.COMPARE
    chosen = {dim: trial.suggest_categorical(dim, vals) for dim, vals in compare.items()}
    compared = [dim for dim, vals in compare.items() if len(vals) > 1]
    compare_value = "-".join(str(chosen[d]) for d in compared) if compared else "default"

    loss_name = chosen["loss"]
    # consistency "on" → λ is searched; "off" (or single-view tasks without the dim) → λ=0.
    consistency_on = chosen.get("consistency", "on") == "on"
    balance = chosen.get("balance", "off")
    spec = settings.HEAD_SPEC
    lam = _suggest_lambda(trial, settings.OPT_SEARCH_SPACE) if consistency_on else 0.0
    beta = settings.HPARAMS.get("robustness_beta", 0.0)
    seed = settings.HPARAMS["seed"]

    samples, keys = aggregators.stack_views(index, emb, settings.VIEW_TYPES)
    meta = _per_key_meta(index)
    split = np.array([meta[k]["split"] for k in keys])
    classes = np.array([meta[k]["class"] for k in keys])
    flower_ids = np.array([k[0] for k in keys])
    y = torch.tensor([meta[k]["target"] for k in keys], dtype=torch.float32).unsqueeze(1)
    X = torch.tensor(samples, dtype=torch.float32)

    train_mask = split == "train"
    val_mask, test_mask = split == "val", split == "test"

    # `balance` narrows/reweights the train split only. The subsample draw is seeded
    # from the run (not the trial) so every trial of that variant fits the same
    # flowers — otherwise the draw itself would add noise to the comparison.
    group_of, target_mix = settings.PRODUCTION_MIX
    fit_mask, fit_w = _fit_selection(
        balance, classes, flower_ids, train_mask, group_of, target_mix, seed
    )
    w_fit = torch.tensor(fit_w, dtype=torch.float32).unsqueeze(1)

    # Evaluation weights are computed per split and applied to *every* variant, so
    # `off`/`reweight`/`subsample` are all scored on the same rows under the same
    # production mix. This is what keeps the three comparable.
    w_val = torch.tensor(
        data.group_weights(classes[val_mask], group_of, target_mix), dtype=torch.float32
    ).unsqueeze(1)
    w_test = torch.tensor(
        data.group_weights(classes[test_mask], group_of, target_mix), dtype=torch.float32
    ).unsqueeze(1)

    head = heads.build(spec, ctx.backbone_name)
    epochs = settings.HPARAMS["epochs"]
    opt, sched = optimization.build(head.parameters(), {**settings.HPARAMS, "epochs": epochs})
    loss_fn = _loss_fn(loss_name)

    history = []
    for epoch in range(epochs):
        head.train()
        opt.zero_grad()
        pred, per_view = _forward(head, X[fit_mask], return_views=True)
        # MSE on the round prediction + λ · mean per-round variance across views.
        # Variance (not std/range): smooth gradient, → 0 cleanly as views agree.
        consistency = per_view.var(dim=1, unbiased=False).mean()
        # Weighted mean over samples (w_fit is all-ones unless balance="reweight").
        accuracy = (w_fit * loss_fn(pred, y[fit_mask])).sum() / w_fit.sum()
        loss = accuracy + lam * consistency
        loss.backward()
        opt.step()
        sched.step()

        head.eval()
        with torch.no_grad():
            pv = _forward(head, X[val_mask])
        history.append({
            "epoch": epoch,
            "train_loss": float(loss.detach()),
            "val_loss": float(loss_fn(pv, y[val_mask]).mean()),
            "val_rmse": _rmse(pv, y[val_mask]),
            "val_mae": _mae(pv, y[val_mask]),
            "val_rmse_prod": _weighted_rmse(pv, y[val_mask], w_val),
        })

    head.eval()
    with torch.no_grad():
        pt, pv_test = _forward(head, X[test_mask], return_views=True)
        _, pv_val = _forward(head, X[val_mask], return_views=True)

    test_keys = [k for k, m in zip(keys, test_mask) if m]
    view_range, view_std = _view_spread(pv_test)            # test split, human-facing
    round_std = _round_to_round_std(pt.squeeze(1).numpy(), test_keys)
    val_view_range, _ = _view_spread(pv_val)                # selection uses val, not test
    # Robustness-aware selection: accuracy + β · view disagreement (lower is better).
    # The accuracy term is the *production-weighted* val RMSE, so Optuna optimises for
    # the mix the line will see rather than the mix that happened to be collected.
    selection_score = history[-1]["val_rmse_prod"] + beta * val_view_range

    metrics = {
        # `rmse`/`mae` stay on the dataset's own mix — unweighted and directly
        # comparable to every run predating the balance dim. `*_prod` are the same
        # errors under PRODUCTION_DISTRIBUTION and are what selection reads.
        "rmse": _rmse(pt, y[test_mask]),
        "mae": _mae(pt, y[test_mask]),
        "rmse_prod": _weighted_rmse(pt, y[test_mask], w_test),
        "mae_prod": _weighted_mae(pt, y[test_mask], w_test),
        "val_rmse": history[-1]["val_rmse"],
        "val_mae": history[-1]["val_mae"],
        "val_rmse_prod": history[-1]["val_rmse_prod"],
        "view_range": view_range,
        "view_std": view_std,
        "round_std": round_std,
        "val_view_range": val_view_range,
        "selection_score": selection_score,
        "loss": loss_name,
        "consistency": "on" if consistency_on else "off",
        "balance": balance,
        # Flowers actually fitted — makes subsample's cost visible next to its score.
        "n_fit_samples": int(fit_mask.sum()),
        "n_fit_flowers": int(len(set(flower_ids[fit_mask]))),
        "compare_value": compare_value,   # variant id = compared dims' values (kept-dir name)
        "compare_axis": ",".join(compared) if compared else "none",  # viewer reads this
        "lambda_consistency": lam,
        "robustness_beta": beta,
        "history": history,
        "head_spec": {
            "aggregator_name": spec.aggregator_name,
            "hidden_dims": list(spec.hidden_dims),
            "dropout": spec.dropout,
        },
    }

    tdir = scratch / f"trial_{trial.number}"
    tdir.mkdir(parents=True, exist_ok=True)
    torch.save(head.state_dict(), tdir / "head.pt")
    (tdir / "metrics.json").write_text(json.dumps(metrics))
    _write_predictions(tdir / "predictions.csv", keys, test_mask, meta, pt.squeeze(1).numpy())
    return selection_score


def _write_predictions(path, keys, test_mask, meta, preds):
    test_keys = [k for k, t in zip(keys, test_mask) if t]
    rows = [
        {
            "fileName": meta[k]["top_file"],
            "flowerID": k[0],
            "roundID": k[1],
            "class": meta[k]["class"],
            "target": meta[k]["target"],
            "prediction": float(pr),
        }
        for k, pr in zip(test_keys, preds)
    ]
    with open(path, "w") as f:
        f.write("sep=;\n")
        pd.DataFrame(rows).to_csv(f, sep=";", index=False)


def _load_result(agg_dir):
    metrics = json.loads((agg_dir / "metrics.json").read_text())
    pred_df = data.read_index(agg_dir / "predictions.csv")
    hs = metrics["head_spec"]
    return TrainResult(
        predictions=pred_df["prediction"].to_numpy(),
        labels=pred_df["target"].to_numpy(),
        metrics=metrics,
        head_state_dict={},
        head_spec=HeadSpec(hs["aggregator_name"], tuple(hs["hidden_dims"]), hs["dropout"]),
        checkpoint_path=agg_dir / "head.pt",
    )


def main(run_dir, overrides=None):
    """Run the full sweep for a prepared run: extract → study → keep best → report → comparison.

    Everything is read from `<run>/prep/info.json` (data, backbone, frozen
    training config, including the `compare` grid). `overrides` is a partial
    hparams dict (from the CLI flags) layered on top of the manifest's `hparams`.
    """
    ctx = RunContext.from_info_json(run_dir)
    info = ctx.info()
    hparams = {**info["hparams"], **(overrides or {})}
    hs = info["head_spec"]
    settings = SimpleNamespace(
        OPT_SEARCH_SPACE=info["opt_search_space"],
        HEAD_SPEC=HeadSpec(hs["aggregator_name"], tuple(hs["hidden_dims"]), hs["dropout"]),
        HPARAMS=hparams,
        COMPARE=info["compare"],
        # The run's declared views (any count); stack_views takes these explicitly.
        VIEW_TYPES=info["views"],
        # ({class: group}, {group: proportion}) — read from the manifest, not config,
        # so a run keeps the mix it was prepared with even if config.py later moves.
        PRODUCTION_MIX=_production_mix(info),
    )

    emb = embeddings.extract(ctx)
    index = data.apply_label_corrections(ctx, ctx.index())
    scratch = ctx.task_dir / "_study"
    n_trials = info["opt_n_trials"]

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=hparams["seed"]),
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)  # silence Optuna's own per-trial logs

    # Progress bar over trials + a status line below it, overwritten each trial.
    bar = tqdm(total=n_trials, desc="trials", unit="trial")
    status = tqdm(total=0, position=1, bar_format="{desc}")

    def _progress(study, trial):
        bar.update(1)
        val = f"{trial.value:.4f}" if trial.value is not None else "failed"
        try:
            best = f"{study.best_value:.4f}"
        except ValueError:
            best = "—"
        status.set_description_str(f"  last: trial {trial.number + 1} · score={val} · best={best}")

    # Seed every grid combination so each variant is sampled at least once.
    for combo in itertools.product(*settings.COMPARE.values()):
        study.enqueue_trial(dict(zip(settings.COMPARE.keys(), combo)))
    study.optimize(
        lambda t: _objective(t, ctx, index, emb, settings, scratch),
        n_trials=n_trials,
        callbacks=[_progress],
    )
    bar.close()
    status.close()

    compared = [dim for dim, vals in settings.COMPARE.items() if len(vals) > 1]
    kept = optimization.keep_best_per_variant(
        ctx, scratch, group_key="compare_value", metric_key="selection_score"
    )
    for variant_dir in kept.values():
        plotting.report(ctx, _load_result(variant_dir))
    optimization.write_study_summary(
        study, ctx.task_dir, group_params=compared, metric_key="selection_score"
    )
    plotting.write_comparison(ctx, task=ctx.task)
    plotting.write_predictions(ctx)  # one combined predictions.csv in DATA_DIR


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", "-r", required=True, help="run dir from prepare (e.g. output/<run>)")
    ap.add_argument("--lr", type=float)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--seed", type=int)
    a = ap.parse_args()
    overrides = {k: getattr(a, k) for k in ("lr", "epochs", "seed") if getattr(a, k) is not None}
    main(a.run, overrides)
