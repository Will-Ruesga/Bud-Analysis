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


def _forward(head, X):
    return mil_pool(head, X) if X.ndim == 3 else head(X)


def _objective(trial, ctx, index, emb, settings, scratch):
    agg_name = trial.suggest_categorical("aggregator_name", settings.OPT_SEARCH_SPACE["aggregator_name"])
    spec = HeadSpec(agg_name, settings.HEAD_SPEC.hidden_dims, settings.HEAD_SPEC.dropout)

    samples, keys = aggregators.get(agg_name)(index, emb)
    meta = _per_key_meta(index)
    split = np.array([meta[k]["split"] for k in keys])
    y = torch.tensor([meta[k]["target"] for k in keys], dtype=torch.float32).unsqueeze(1)
    X = torch.tensor(samples, dtype=torch.float32)

    train_mask = split == "train"
    val_mask, test_mask = split == "val", split == "test"

    head = heads.build(spec, ctx.backbone_name)
    epochs = settings.HPARAMS["epochs"]
    opt, sched = optimization.build(head.parameters(), {**settings.HPARAMS, "epochs": epochs})
    loss_fn = _loss_fn(settings.HPARAMS["loss"])

    history = []
    for epoch in range(epochs):
        head.train()
        opt.zero_grad()
        pred = _forward(head, X[train_mask])
        loss = loss_fn(pred, y[train_mask])
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
        pt = _forward(head, X[test_mask])
    metrics = {
        "rmse": _rmse(pt, y[test_mask]),
        "mae": _mae(pt, y[test_mask]),
        "val_rmse": history[-1]["val_rmse"],
        "val_mae": history[-1]["val_mae"],
        "history": history,
        "head_spec": {
            "aggregator_name": agg_name,
            "hidden_dims": list(spec.hidden_dims),
            "dropout": spec.dropout,
        },
    }

    tdir = scratch / f"trial_{trial.number}"
    tdir.mkdir(parents=True, exist_ok=True)
    torch.save(head.state_dict(), tdir / "head.pt")
    (tdir / "metrics.json").write_text(json.dumps(metrics))
    _write_predictions(tdir / "predictions.csv", keys, test_mask, meta, pt.squeeze(1).numpy())
    return metrics["val_rmse"]


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

    emb = embeddings.extract(ctx, pooling=info.get("pooling", "cls"))
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
        status.set_description_str(f"  last: trial {trial.number + 1} · val_rmse={val} · best={best}")

    for agg in settings.OPT_SEARCH_SPACE["aggregator_name"]:
        study.enqueue_trial({"aggregator_name": agg})
    study.optimize(
        lambda t: _objective(t, ctx, index, emb, settings, scratch),
        n_trials=n_trials,
        callbacks=[_progress],
    )
    bar.close()
    status.close()

    kept = optimization.keep_best_per_aggregator(ctx, scratch)
    for agg_dir in kept.values():
        plotting.report(ctx, _load_result(agg_dir))
    optimization.write_study_summary(study, ctx.task_dir)
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
