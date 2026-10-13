# XGBoost: first tuned run vs extended run

Accuracy is in %, validation / test. Table: `results/tables/xgb_capped_vs_extended.csv`.
Figures: `results/figures/xgboost_capped_vs_extended_test.png` and
`results/figures/xgboost_50_classes_capped_vs_converged.png`.

## First tuned run ("capped")

- Model: gradient-boosted trees (XGBoost 3.4.1) on raw pixels, trained on the GPU, with the same
  train / validation / test split as every other model.
- Tuning: Optuna with the TPE sampler, a form of Bayesian optimisation that proposes new settings near the ones
  that scored well so far. The objective was validation accuracy. Each trial trained up to 3,000 rounds and
  stopped early after 50 rounds without improvement in validation log-loss. The test set was used once per
  class size, by the best setting only.
- Searches: 31 trials on 10 classes and 45 trials on 3 classes. Searching directly on 50 classes was too slow,
  so the 4 best 10-class settings were rerun on 50 classes.
- Limits: trees deeper than 9 levels did not fit in the 8 GB of GPU memory, so tree depth was capped at 9.
  The 50-class runs had to finish within 3 hours, so each stopped after at most 75 minutes. Only 3 of the 4
  settings ran and none reached early stopping, so the 50-class result (77.4 / 77.4) came from under-trained
  models.
- Several of the best settings sat at the edge of the search ranges.

## Extended run: what changed and why

- 50 classes without a time cap (up to 5,000 rounds), so each run trains until early stopping. Only runs that
  stopped early (converged) could be chosen.
- The best 50-class model from the first run was continued from where it had stopped, instead of starting
  again.
- A new 10-class search of 32 trials with wider ranges where the first search hit the edges
  (minimum child weight 1–50, L2 regularisation 1–100, depth 6–9).
- The other top settings from the first run were also trained on 50 classes without the cap, as far as the
  11.5-hour budget allowed.

## Before / after

| Classes | First run (validation / test) | Extended run (validation / test) | Test gain (points) |
|---|---|---|---|
| 3 | 91.7 / 90.8 | unchanged (already converged) | 0.0 |
| 10 | 89.8 / 89.7 | 89.8 / 89.8 | 0.1 |
| 50 | 77.4 / 77.4 | 78.0 / 77.9 | 0.5 |

On 50 classes, every setting levelled off at about 78% validation accuracy: the continued best model reached
78.0 after 2,193 rounds, the second setting also 78.0, the third 77.8, and the fourth was stopped by the time
budget at 77.4.

## Compute time

- First run: 5.8 hours of search in total.
- Extended run: 11.5 hours (3.0 hours for the wider 10-class search, 8.4 hours of 50-class training).

## Limitations

- Tree depth is capped at 9 by GPU memory, so the effect of deeper trees is unknown.
- The 50-class settings come from the 10-class search; no search ran directly on 50 classes.
- One seed per setting: differences of a few tenths of a point are within run-to-run noise.
- The extended run used more compute than the first, so part of the small gain is simply more training time.
- Trees treat each pixel as an independent feature, so XGBoost stays below the CNNs on 50 classes.
