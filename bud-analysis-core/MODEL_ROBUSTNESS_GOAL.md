# Model robustness — diagnosis + improvement plan

Goal: cut the flakiness so a flower passed multiple times (different rounds / small
pose & tilt changes) gives a stable ripeness. Target: **per-flower round-to-round
std ≈ 0.05** (5 pp) or better, and per-view predictions that don't swing wildly.

---

## 0. Status — what's implemented so far

- **H (measure) — DONE.** `metrics.json` now carries test-split `view_range`,
  `view_std`, `fork_std`; `comparison.png` captions them; the viewer shows
  aggregate view/round spread. (See `VIEW_CONSISTENCY_GOAL` history.)
- **A, train-time half (consistency loss) — DONE.** Training loss is
  `MSE/Huber + λ·var_over_views(per_view)`; **λ is Optuna-searched**. Selection is
  robustness-aware: trials ranked by `val_rmse + β·val_view_range` (β=0.5 fixed).
- **D (higher resolution) — DONE.** `image_size` is hardcoded to **512** (was 224);
  all views resize to 512² before the frozen backbone.
- **Cleanup — DONE.** `masked_mean` pooling and the `top_only` aggregator were
  tried and removed (both lost). One pipeline: **cls + mil_mean (all views)**. The
  study now compares the **loss (MSE vs Huber)** instead of the aggregator.
- **Not yet:** B (TTA), C (attention-MIL), E (vitb16 sweep), F (backbone fine-tune),
  G (relabel pass). β-tuning is deferred to the cleanup goal's Phase 2 (β re-rank).

Net: the highest-leverage items (per-view consistency loss + resolution + honest
robustness metrics/selection) are live; the next real signal comes from a clean
baseline run on Avalanche.

---

## 1. What the data actually says (measured on the current runs)

Per-flower spread, MIL-mean head, all three runs (`viewer` records):

| run | round→round MIL std (mean / p90 / max) | within-flower VIEW range (mean / p90) |
|---|---|---|
| Avalanche cls      | 0.042 / 0.069 / 0.286 | **0.612 / 0.845** |
| Avalanche maskmean | 0.044 / 0.074 / 0.277 | **0.621 / 0.859** |
| GardeniaS1 (v2 onnx)| 0.060 / 0.101 / 0.167 | **0.588 / 0.859** |

**The headline:** the five views of one bud, one capture, disagree by **~0.6 on
average** (out of a 0–1 scale). The round-to-round MIL std looks smallish (~0.04–0.06)
**only because averaging 5 noisy views hides it** — but ~1 in 6 flowers still swings
> 0.15 between rounds, and worst cases hit 0.29–0.67. So:

- The instability is **per-view**, not per-capture. The model gives almost
  independent answers for different angles of the *same* bud.
- "Small movement / tilt flips the result" = the MIL mean is a near-average of 5
  high-variance numbers, so which view is extreme shifts the mean a lot.
- This is consistent with the note already in `ripeness/config.py`: the **top view
  over-predicts by ~0.26 systematically** while sides under-predict. The head was
  trained so the *mean* is calibrated, but no single view is — and the per-view
  variance is the dominant error source, not capture noise.

**Conclusion:** the lever is **per-view consistency / robustness**, not aggregation
tweaks. We must make the model give the *same* ripeness regardless of viewing angle,
tilt, and small pose changes — i.e. learn an angle-invariant ripeness.

---

## 2. Root causes (why this happens here)

1. **No training-time augmentation, and it can't be added cheaply.** Embeddings are
   extracted **once** from a single 224² bicubic resize per image
   (`core/embeddings.py`) and cached; training is a linear-ish MLP on those frozen
   vectors. So the head never sees pose/tilt/scale/crop/lighting variation — it
   memorises each frozen vector → target. Nothing teaches it that two angles of one
   bud are the same ripeness. (Literature: training-time augmentation + **consistency
   regularisation** is the standard fix for exactly this; FixMatch/Π-model family,
   and "regularising for invariance to augmentation improves *supervised* learning".)

2. **Frozen DINOv3 features are not ripeness-invariant to viewpoint.** DINOv3 was
   never trained on buds; its CLS token encodes pose/orientation/framing strongly.
   Two angles of one bud sit far apart in feature space, so a frozen-feature linear
   head *cannot* map them to the same value. (Literature: ViT features are
   resolution/tokenisation sensitive; fine-grained tasks benefit from higher input
   resolution and from tuning the backbone, not just a probe.)

3. **Single low-res view per image.** 224² bicubic of a bud that fills 16–47 % of the
   frame throws away most of the discriminative surface detail, so tiny pose changes
   move the few informative patches in/out — high sensitivity. (Already flagged in
   `ideas.txt` as "mask-crop untried".)

4. **MIL late-fusion mean is a weak aggregator for high-variance instances.** A plain
   mean lets one extreme view (e.g. the biased top) drag the fork. Attention-MIL learns
   to *down-weight* unreliable instances. (Ilse et al. 2018, gated-attention MIL — far
   more robust than mean pooling when some instances are bad.)

5. **Full-batch training, aggregator-only Optuna.** `train.py` does full-batch steps
   (no minibatch noise to regularise) and the search space only toggles the aggregator
   — lr/dropout/hidden/weight-decay are fixed (also flagged in `ideas.txt`). Underfit
   regularisation → memorisation → brittle.

---

## 3. Improvement plan (ordered: leverage / cost / confidence)

### Tier 1 — directly attack per-view variance (highest expected payoff)

**A. Train-time augmentation + consistency loss (re-extract embeddings with aug).**
Generate **K augmented embeddings per image** (random resized crop around the bud,
small rotation/flip, brightness/contrast jitter) instead of one. Train the head with:
(i) the task loss on each, and (ii) a **consistency penalty** `‖f(aug_i) − f(aug_j)‖`
that forces the head to give the *same* ripeness across augmentations of one image.
- *Why:* this is the textbook fix for "same object, different view → different
  prediction." It teaches invariance the frozen features lack, at the head level.
- *Cost:* re-extract K× embeddings (cache grows K×); head loss gains a term.
  Medium. Reuses the embedding cache machinery.
- *Confidence:* high — consistency reg is well-established for variance reduction.

**B. Test-time augmentation (TTA) — cheap, immediate, no retrain.**
At inference, embed each view K times under the same augmentations and average the
head outputs (or average embeddings). Reduces per-view variance directly.
- *Why:* TTA is the standard, training-free variance reducer; ~19 % error drop in the
  composites study, widely used for robustness.
- *Cost:* low (K× forward passes at inference only). **Do this first** as a baseline —
  it tells us how much variance is "augmentable" vs. structural.
- *Confidence:* high for variance reduction; modest for bias.

**C. Attention-MIL aggregator (replace/añadir to mean).**
Add a gated-attention pooling head (Ilse 2018) so the model learns to trust reliable
views and down-weight the biased/garbage ones (e.g. the +0.26 top).
- *Why:* directly addresses "1–3 views are astronomically off." A learned weight beats
  a fixed mean when instance quality varies — which the data shows it does.
- *Cost:* medium (new head + pooling in `core/heads.py`, add to Optuna space).
- *Confidence:* high that it helps the fork estimate; it also yields per-view weights
  we can show in the viewer.

### Tier 2 — fix the inputs (attacks causes 2 & 3)

**D. Mask-crop to the bud + higher resolution.** Crop each view to the alpha bbox and
feed a larger square (e.g. 320–448) so the bud fills the frame at more patches.
- *Why:* more invariant framing (bud always centred & scaled) + more detail = far less
  sensitivity to small pose/tilt. Cause #3 directly. Cheap-ish (re-extract).
- *Confidence:* medium-high; recommended in `ideas.txt` and the ViT-resolution lit.

**E. Bigger backbone (dinov3_vitb16).** Richer 768-d features, generally more robust.
Just add to `BACKBONE_CHECKPOINTS`. Cheap, low risk, modest gain.

### Tier 3 — higher ceiling, more work

**F. Light backbone fine-tune / LoRA on the last blocks.** Frozen DINOv3 can't be made
viewpoint-invariant for buds by a probe alone (cause #2). Unfreezing the last
block(s) or a LoRA adapter lets features themselves become ripeness-aligned and
angle-invariant. Highest ceiling; needs care with dataset size + the embedding-cache
pipeline (cache no longer valid during training).

**G. Use the relabel tool first.** Some "instability" is label noise. Correcting
mislabeled outliers (now possible) before/with these experiments removes a confound so
we measure model robustness, not label noise.

### Tier 0 — measurement (do alongside everything)

**H. Make robustness a tracked metric.** Add **per-flower round-to-round std** and
**within-flower view range** to the training report / comparison (we already compute
them in the viewer). Without this as a first-class metric we can't tell which change
actually helped. Also expand the Optuna search space (lr/dropout/hidden/weight-decay)
so trials regularise, not just re-roll inits.

---

## 4. Suggested experiment order
1. **H** (metric) + **G** (clean labels) — so we measure the right thing.
2. **B** (TTA) — free baseline; quantifies augmentable vs structural variance.
3. **A** (aug + consistency loss) — the main fix for per-view variance.
4. **C** (attention-MIL) — robust aggregation, stacks with A.
5. **D** (mask-crop + resolution) — input fix; re-extract.
6. **E** / **F** — backbone size, then fine-tune if still short.

Success check after each: per-flower round std and view range (metric H) on the test
split, plus the viewer's scatter — does the cloud per class tighten?

---

## Open questions for you
- Acceptable inference cost? (TTA / attention-MIL add forward passes; fine-tune adds
  training cost + breaks the "extract once" cache assumption.)
- Is per-view ripeness ever *meant* to differ (e.g. one side riper), or is a flower a
  single ripeness? (Determines whether per-view disagreement is error or signal — the
  +0.26 top bias says mostly error, but worth confirming.)
- Do we have alpha masks on every dataset (needed for D, mask-crop)? Avalanche/Gardenia
  are RGBA — confirm the rest.
