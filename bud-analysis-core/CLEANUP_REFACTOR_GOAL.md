# Cleanup + configurable compare-axis (two phases)

Two phases. **Phase 1** removes the experiments that lost and lands a concrete
MSE-vs-Huber comparison. **Phase 2** generalises "what we compare" so the axis is
chosen per run (loss, aggregator, lr, …) — plus a cheap β re-rank.

Phase 1 stands alone and de-risks Phase 2 (smaller surface to touch). Do 1, ship,
then 2.

> **STATUS — Phase 1 is DONE (implemented + tested).** masked_mean + top_only +
> deprecated text removed; the study now compares `loss` (mse vs huber); kept dirs
> are `<task>/<loss>/`; `comparison.png` shows MSE vs Huber with robustness
> captions; legacy runs still load in the viewer. Tests: `test_cleanup_refactor.py`
> (+ existing suites) pass. **Phase 2 is not started.**

Key idea the design hinges on — **two kinds of axes:**
- **Training axes** (`loss`, `aggregator`, `lr`, `dropout`, fixed `λ`…): a value
  change → a *different trained model*. Fits the existing sweep + keep-best-per-value.
- **Selection axis** (`robustness_beta`): changes only the post-hoc ranking
  (`val_rmse + β·view_range`), *not* training. Comparing β values = re-ranking one
  trained pool, no retraining. Cheaper, but a different mechanism — kept separate.

---

# Phase 1 — cleanup (pure deletion + loss as the compare axis) ✅ DONE

## 1.1 Remove `masked_mean` pooling → cls only
- `core/embeddings.py` — drop the `pooling` arg, `_patch_keep_mask`,
  `_invalidate_if_pooling_changed`, the `masked_mean` branch, the `"pooling"` key
  in `meta.json`. `extract(ctx)` always does CLS.
- `core/backbones.py` — delete `feature_tokens` and `patch_grid` (used **only** by
  masked_mean).
- `ripeness/config.py` — delete `POOLING` + its NancyNora A/B comment.
- `ripeness/prepare.py` — drop `--pooling`, the `pooling` param, `"pooling"` from
  `_config_snapshot`.
- `ripeness/train.py` — `embeddings.extract(ctx)` (no `pooling=`).

## 1.2 Remove `top_only` aggregator → mil_mean only
- `core/aggregators.py` — delete `top_only` + its `VIEW_TYPES`/`REGISTRY` entries.
  Collapse to a single `stack_views(index, emb) -> (N, V, D)` and a constant
  `VIEW_TYPES = CANONICAL_VIEW_TYPES`; drop `REGISTRY`/`get` if unused.
- `ripeness/train.py` — `_forward` loses the `X.ndim == 2` branch; always `mil_pool`.
- `core/export.py` — view `union` is always `CANONICAL_VIEW_TYPES`; drop the
  per-aggregator union + `top_only` single-view export path.
- `core/schemas.py` — `HeadSpec.aggregator_name` **pinned to `"mil_mean"`**
  (decided). Field stays; existing runs + viewer load unchanged. Removing it
  entirely is deferred (revisit in Phase 2 if the dead field bothers us).

## 1.3 Compare axis = loss (MSE vs Huber)
Re-point the existing per-aggregator machinery at `loss`:
- `ripeness/config.py` — `OPT_SEARCH_SPACE = {"loss": ["mse","huber"],
  "lambda_consistency": [0.0, 0.5]}`; `loss` leaves `HPARAMS`.
- `ripeness/train.py` — `_objective` suggests `loss`, builds `_loss_fn(loss)`;
  `+ λ·variance` rides on either; store `loss` in `metrics.json`.
- `core/optimization.py` — `keep_best_per_aggregator` → `keep_best_per_variant`
  with a `group_key` (reads `metrics[group_key]`); kept dirs `<task>/mse/`,
  `<task>/huber/`. Same grouping in `write_study_summary`.
- `core/run_context.py` — `aggregator_dir` → `variant_dir`.
- `core/plotting.py` — per-variant scatters titled `MSE`/`Huber`; `_pred_col`
  → `predMse`/`predHuber`; drop `_pretty_agg`'s `top_only` case. Keep the
  robustness caption.
- `core/export.py` — `"auto"` picks the lowest-`selection_score` variant.

## 1.4 Delete deprecated mentions
- `README.md` — the "Two aggregators … `all_views_max`/`all_views_mean` removed …"
  sentence and `top_only` description → single mil_mean + cls + loss-comparison story.
- Docstrings referencing pooling choice / `top_only` / `masked_mean` / old removed
  aggregators (`heads.py`, `export.py`, `aggregators.py` headers).
- **Grep gate:** `masked_mean|top_only|patch_grid|feature_tokens|all_views|POOLING`
  returns only intended references.

---

# Phase 2 — configurable compare-axis + β re-rank

## 2.1 Declare the experiment in the manifest
`prepare` already freezes the training config; extend it to also freeze *what to
compare*:
```python
COMPARE = {"axis": "loss", "values": ["mse", "huber"]}   # or "aggregator", "lr", …
```
- `_objective` suggests `COMPARE["axis"]` over `COMPARE["values"]` (type-aware:
  categorical list vs `[lo, hi]` float range), records the chosen value under a
  fixed `metrics["compare_value"]` (+ `metrics["compare_axis"]`).
- `keep_best_per_variant(group_key="compare_value")`; dirs named by the value
  (`<task>/<value>/`). `write_study_summary` + `plotting` group/label by it.
- Everything else in `OPT_SEARCH_SPACE` is searched *within* each variant (λ, etc.).
Net: one knob in the manifest swaps the comparison from loss → aggregator → lr →
dropout with no code change.

## 2.2 β as a selection axis (no retraining)
Because β only re-ranks, add a post-hoc pass over the **kept trial pool**:
- Keep all completed trials' `{val_rmse, val_view_range}` (cheap; already computed).
- For each β in a small list, recompute `val_rmse + β·val_view_range`, report which
  trial wins → a `beta_sweep.json` / a small plot of "winner vs β".
- This answers "how should β be set?" without training anything extra — the tuning
  you flagged for the current β=0.5.

## 2.3 Type-aware suggest (the one fiddly bit)
A tiny helper: `values` is a 2-float `[lo, hi]` → `suggest_float`; a list of
str/other → `suggest_categorical`. Keeps `_objective` axis-agnostic.

---

## Back-compat with existing runs (applies to both phases)
The `2026_06_09` Avalanche runs have `info.json` `"pooling"`, `metrics.json`
`head_spec.aggregator_name`, dirs `mil_mean`/`top_only`.
- **Viewer** (`run_loader.py`, `app.js`) is built around "aggregator"; it must read
  the result-dir name generically (now a loss/variant) and default a missing
  `aggregator_name` to `"mil_mean"`, ignore stray `"pooling"`. Old dirs still open.
- Cleanest: re-run Avalanche `prepare`+`train` on the new code so on-disk runs match
  the new schema; treat old dirs as read-only legacy.

## Order & done-when
1. Phase 1 (1.1 → 1.4), run the existing tests, fresh Avalanche MSE-vs-Huber run.
2. Phase 2 (2.1 → 2.3), validate by swapping the axis with no code edits + a β sweep.
- Done: grep gate clean; no pooling/aggregator flags needed; `comparison.png` shows
  the chosen axis with robustness captions; tests pass; viewer opens old + new runs;
  ONNX export works for the single mil_mean(all-views) pipeline.
