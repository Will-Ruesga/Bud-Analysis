"""train.py — the ripeness training loop (the only task file with real logic).

Owns losses, metrics, the Optuna objective, the loop, and per-trial bookkeeping.
Core supplies optimiser/scheduler, post-study pruning, summary, and plots.
Reads its settings from the run manifest (`-run <run>`); `--lr/--epochs/--seed`
override the frozen hparams per run. Flow: extract embeddings → Optuna study →
keep best per aggregator → report each → study summary → comparison plot.
"""

import argparse
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


def _loss_fn(name):
    return nn.HuberLoss() if name == "huber" else nn.MSELoss()


def _view_spread(per_view):
    """(range_mean, std_mean) of per-view predictions, averaged over forks.

    `per_view` is `(B, V, 1)` (the tensor from `mil_pool(..., return_views=True)`).
    Range = mean over forks of `max - min` across views; std = mean of the
    per-fork view std. Single-view aggregators (V=1) give 0 for both.
    """
    pv = per_view.reshape(per_view.shape[0], per_view.shape[1])  # (B, V)
    rng = float((pv.max(dim=1).values - pv.min(dim=1).values).mean())
    sd = float(pv.std(dim=1, unbiased=False).mean())
    return rng, sd


def _fork_to_fork_std(pooled, keys):
    """Mean over flowers of the std of their fork predictions.

    Only flowers with >= 2 forks contribute (a single fork has no spread).
    `pooled` is a 1-D array of fork-level predictions aligned with `keys`
    (`(flower_id, fork_id)`). Returns None if no flower has >= 2 forks.
    """
    by_flower: dict = {}
    for (flower_id, _fork_id), p in zip(keys, pooled):
        by_flower.setdefault(flower_id, []).append(float(p))
    stds = [float(np.std(v, ddof=1)) for v in by_flower.values() if len(v) > 1]
    return float(np.mean(stds)) if stds else None


def _per_key_meta(index):
    """{(flower_id, fork_id): {target, split, class, top_file}}."""
    idx = index.copy()
    idx["fork_id"] = idx["fork_id"].fillna("")
    meta = {}
    for (flower_id, fork_id), grp in idx.groupby(["flower_id", "fork_id"]):
        top = grp[grp["view_type"] == "top"]
        meta[(flower_id, fork_id)] = {
            "target": float(grp["target"].iloc[0]),
            "split": grp["split"].iloc[0],
            "class": str(grp["class"].iloc[0]),
            "top_file": (top if len(top) else grp)["file_name"].iloc[0],
        }
    return meta


def _forward(head, X, return_views=False):
    """Pooled fork prediction `(B, 1)`; with `return_views`, also `per_view (B, V, 1)`.

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
    # The compared dimension is the loss; the view pipeline is fixed (mil_mean).
    loss_name = trial.suggest_categorical("loss", settings.OPT_SEARCH_SPACE["loss"])
    spec = settings.HEAD_SPEC
    lam = _suggest_lambda(trial, settings.OPT_SEARCH_SPACE)
    beta = settings.HPARAMS.get("robustness_beta", 0.0)

    samples, keys = aggregators.stack_views(index, emb)
    meta = _per_key_meta(index)
    split = np.array([meta[k]["split"] for k in keys])
    y = torch.tensor([meta[k]["target"] for k in keys], dtype=torch.float32).unsqueeze(1)
    X = torch.tensor(samples, dtype=torch.float32)

    train_mask = split == "train"
    val_mask, test_mask = split == "val", split == "test"

    head = heads.build(spec, ctx.backbone_name)
    epochs = settings.HPARAMS["epochs"]
    opt, sched = optimization.build(head.parameters(), {**settings.HPARAMS, "epochs": epochs})
    loss_fn = _loss_fn(loss_name)

    history = []
    for epoch in range(epochs):
        head.train()
        opt.zero_grad()
        pred, per_view = _forward(head, X[train_mask], return_views=True)
        # MSE on the fork prediction + λ · mean per-fork variance across views.
        # Variance (not std/range): smooth gradient, → 0 cleanly as views agree.
        consistency = per_view.var(dim=1, unbiased=False).mean()
        loss = loss_fn(pred, y[train_mask]) + lam * consistency
        loss.backward()
        opt.step()
        sched.step()

        head.eval()
        with torch.no_grad():
            pv = _forward(head, X[val_mask])
        history.append({
            "epoch": epoch,
            "train_loss": float(loss.detach()),
            "val_loss": float(loss_fn(pv, y[val_mask])),
            "val_rmse": _rmse(pv, y[val_mask]),
            "val_mae": _mae(pv, y[val_mask]),
        })

    head.eval()
    with torch.no_grad():
        pt, pv_test = _forward(head, X[test_mask], return_views=True)
        _, pv_val = _forward(head, X[val_mask], return_views=True)

    test_keys = [k for k, m in zip(keys, test_mask) if m]
    view_range, view_std = _view_spread(pv_test)            # test split, human-facing
    fork_std = _fork_to_fork_std(pt.squeeze(1).numpy(), test_keys)
    val_view_range, _ = _view_spread(pv_val)                # selection uses val, not test
    # Robustness-aware selection: accuracy + β · view disagreement (lower is better).
    selection_score = history[-1]["val_rmse"] + beta * val_view_range

    metrics = {
        "rmse": _rmse(pt, y[test_mask]),
        "mae": _mae(pt, y[test_mask]),
        "val_rmse": history[-1]["val_rmse"],
        "val_mae": history[-1]["val_mae"],
        "view_range": view_range,
        "view_std": view_std,
        "fork_std": fork_std,
        "val_view_range": val_view_range,
        "selection_score": selection_score,
        "loss": loss_name,
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
            "forkID": k[1],
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
    training config). `overrides` is a partial hparams dict (from the CLI
    flags) layered on top of the manifest's `hparams`.
    """
    ctx = RunContext.from_info_json(run_dir)
    info = ctx.info()
    hparams = {**info["hparams"], **(overrides or {})}
    hs = info["head_spec"]
    settings = SimpleNamespace(
        OPT_SEARCH_SPACE=info["opt_search_space"],
        HEAD_SPEC=HeadSpec(hs["aggregator_name"], tuple(hs["hidden_dims"]), hs["dropout"]),
        HPARAMS=hparams,
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

    for loss_name in settings.OPT_SEARCH_SPACE["loss"]:
        study.enqueue_trial({"loss": loss_name})
    study.optimize(
        lambda t: _objective(t, ctx, index, emb, settings, scratch),
        n_trials=n_trials,
        callbacks=[_progress],
    )
    bar.close()
    status.close()

    kept = optimization.keep_best_per_variant(
        ctx, scratch, group_key="loss", metric_key="selection_score"
    )
    for variant_dir in kept.values():
        plotting.report(ctx, _load_result(variant_dir))
    optimization.write_study_summary(
        study, ctx.task_dir, group_param="loss", metric_key="selection_score"
    )
    plotting.write_comparison(ctx, task=ctx.task)
    plotting.write_predictions(ctx)  # one combined predictions.csv in DATA_DIR


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-run", required=True, help="run dir from prepare (e.g. output/<run>)")
    ap.add_argument("--lr", type=float)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--seed", type=int)
    a = ap.parse_args()
    overrides = {k: getattr(a, k) for k in ("lr", "epochs", "seed") if getattr(a, k) is not None}
    main(a.run, overrides)
