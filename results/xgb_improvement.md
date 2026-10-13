# XGBoost: Phase 8 (capped) vs Phase 8b (extended)

Generated automatically at the end of the Phase 8b queue (2026-10-13 03:40).
Accuracy is in %, shown as dev / test. Table: `results/tables/xgb_capped_vs_extended.csv`.
Figures: `results/figures/xgb_extended/`.

## Phase 8 (label `xgb_capped`)
- Model: gradient-boosted trees (XGBoost 3.4.1) on raw pixels, trained on the GPU (`hist` method) with the same saved
  train/dev/test splits as every other model.
- Tuning: Optuna with the TPE sampler, a form of Bayesian optimisation: it proposes new settings near the ones that
  scored well so far. Objective = dev accuracy. Each trial trained up to 3,000 rounds and stopped early after 50 rounds
  without dev log-loss improvement. The test set was used once per class size, by the best setting only.
- Stages: (1) a 2.5 h search on 10 classes (31 trials); (2) the top 4 of those settings re-run on 50 classes;
  (3) a 25 min search on 3 classes (45 trials). Searching directly on 50 classes was too slow (several seconds per round).
- GPU depth cap: trees deeper than 9 levels ran out of the 8 GB GPU memory, so max_depth was limited to 9.
- Time cap: stage 2 had to finish within 3 hours, so each 50-class trial stopped after at most 75 minutes. Only 3 of the
  4 settings ran and none reached early stopping, so the 50-class result (77.44 / 77.44) came from under-trained models.
- Several best settings sat at the edge of the search ranges (max_depth 9, reg_lambda near 10, min_child_weight near 10).

## Phase 8b (label `xgb_extended`): what changed and why
- 50 classes without a time cap (up to 5,000 rounds), so each run trains until early stopping. A run only counts as
  converged if early stopping triggered; runs stopped by the overall 11.5 h budget are recorded but never chosen.
- A new 10-class study `xgb10_wide` with wider ranges where Phase 8 hit the edges: min_child_weight 1-50 and reg_lambda 1-100
  (log scale), max_depth 6-9 (9 is still the GPU limit). Phase 8's best setting was its first trial. 3 h timeout;
  it ran 32 trials {'COMPLETE': 32, 'PRUNED': 0, 'FAIL': 0}.
- Its best setting would only have been run on 50 classes if it beat Phase 8's 10-class dev best by at least 0.3 points.
- Then the other Phase 8 top-4 settings on 50 classes, uncapped, as far as the budget allowed.
- Every run saved its dev log-loss and accuracy for every round, and a checkpoint every 200 rounds.

## Before / after
| classes | Phase 8 dev / test | Phase 8b dev / test | test change | Phase 8b choice |
|---|---|---|---|---|
| 3 | 91.65 / 90.80 | - | - | unchanged (Phase 8 setting kept) |
| 10 | 89.76 / 89.69 | 89.84 / 89.79 | +0.10 | new: xgb10_wide trial 30 |
| 50 | 77.44 / 77.44 | 77.95 / 77.89 | +0.45 | run A1 (Phase 8 xgb_50 best (stage-1 trial 24), continued from the Phase 8 capped model) |

10 classes: Phase 8 best dev 89.76; wide study best converged dev 89.84 (trial 30).

50-class runs in Phase 8b:

| run | setting | rounds trained | status | fit time | dev acc |
|---|---|---|---|---|---|
| A1 | Phase 8 xgb_50 best (stage-1 trial 24), continued from the Phase 8 capped model | 2243 (best 2193) | CONVERGED | 120 min (continued from Phase 8 round 1168: 4503 s in Phase 8 + 2688 s in Phase 8b) | 77.95 |
| A2-1 | Phase 8 stage-1 trial 16 (rank 2) | 2857 (best 2807) | CONVERGED | 2.4 h | 77.95 |
| A2-2 | Phase 8 stage-1 trial 25 (rank 3) | 4260 (best 4210) | CONVERGED | 2.7 h | 77.77 |
| A2-3 | Phase 8 stage-1 trial 29 (rank 4) | 4462 (best 4462) | CAPPED_BY_BUDGET | 2.6 h | 77.43 |

## Compute time
- Phase 8 search: 5.8 h (3 classes 26 min, 10 classes
  2.6 h, 50 classes 2.8 h).
- Phase 8b: 11.5 h (wide 10-class study 3.0 h, 50-class runs 8.4 h).
- Both phases together: 17.3 h.

## Do the best settings still sit at a range edge?
- 10 classes, wide study best: none.
- 50 classes, chosen run: max_depth=9 (high end of 6-9).
- max_depth 9 is the GPU memory limit, not a tuning result: deeper trees could not be tested on this 8 GB GPU.

## Remaining limitations
- Tree depth is capped at 9 by GPU memory, so the effect of deeper trees is unknown.
- The 50-class settings come from the 10-class search; no search ran directly on 50 classes.
- One seed per setting: differences of a few tenths of a point between settings are within run-to-run noise.
- Phase 8b used more compute than Phase 8, so part of any gain is simply more training time.
- Trees treat each pixel as an independent feature, so XGBoost is expected to stay below the CNNs.
