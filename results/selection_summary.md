# Lag and Ridge alpha selection

## Selected configuration

- Lag: **50 ms**
- Ridge alpha: **10000**
- Out-of-fold mean R2: 0.563601 (horizontal 0.565974, vertical 0.561227)
- Per-fold mean R2: 0.5543, 0.5441, 0.5887, 0.5793, 0.5460 (spread 0.0446)

## Procedure

- Split used: `train` only. Validation and held-out trials were not read.
- 123 eligible training trials in 5 contiguous groups; eligible training trials sorted by original row, then divided into contiguous groups of near-equal size with the earliest groups taking any remainder.
- 41177 shared target samples, 212 feature columns.
- Lag grid: [0, 50, 100, 150, 200] ms. Alpha grid: [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0].
- Normalization fitted on each fold's training trials, applied unchanged to its assessment trials.
- Selection metric: arithmetic mean of the per-axis coefficient of determination on concatenated out-of-fold predictions.
- Models: temporary Ridge models, discarded after selection.

## Best alpha per lag

| Lag (ms) | Best out-of-fold mean R2 |
| --- | --- |
| 0 | 0.543308 |
| 50 | 0.563601 |
| 100 | 0.558292 |
| 150 | 0.523386 |
| 200 | 0.459954 |

## Ranking

| Rank | Lag (ms) | Alpha | Out-of-fold mean R2 | Fold spread |
| --- | --- | --- | --- | --- |
| 1 | 50 | 10000 | 0.563601 | 0.0446 |
| 2 | 50 | 1000 | 0.563496 | 0.0511 |
| 3 | 50 | 100 | 0.562627 | 0.0522 |
| 4 | 50 | 10 | 0.562528 | 0.0523 |
| 5 | 50 | 1 | 0.562518 | 0.0523 |
| 6 | 50 | 0.1 | 0.562517 | 0.0523 |
| 7 | 100 | 10000 | 0.558292 | 0.0423 |
| 8 | 100 | 1000 | 0.557884 | 0.0514 |
| 9 | 100 | 100 | 0.557003 | 0.0526 |
| 10 | 100 | 10 | 0.556903 | 0.0527 |
| 11 | 100 | 1 | 0.556892 | 0.0527 |
| 12 | 100 | 0.1 | 0.556891 | 0.0527 |
| 13 | 0 | 1000 | 0.543308 | 0.0566 |
| 14 | 0 | 10000 | 0.543051 | 0.0600 |
| 15 | 0 | 100 | 0.542416 | 0.0569 |
| 16 | 0 | 10 | 0.542313 | 0.0570 |
| 17 | 0 | 1 | 0.542303 | 0.0570 |
| 18 | 0 | 0.1 | 0.542302 | 0.0570 |
| 19 | 150 | 10000 | 0.523386 | 0.0365 |
| 20 | 150 | 1000 | 0.522922 | 0.0467 |
| 21 | 150 | 100 | 0.522014 | 0.0481 |
| 22 | 150 | 10 | 0.521911 | 0.0482 |
| 23 | 150 | 1 | 0.521900 | 0.0482 |
| 24 | 150 | 0.1 | 0.521899 | 0.0482 |
| 25 | 200 | 10000 | 0.459954 | 0.0410 |
| 26 | 200 | 1000 | 0.459493 | 0.0624 |
| 27 | 200 | 100 | 0.458562 | 0.0651 |
| 28 | 200 | 10 | 0.458455 | 0.0654 |
| 29 | 200 | 1 | 0.458444 | 0.0654 |
| 30 | 200 | 0.1 | 0.458443 | 0.0654 |

## Checks

| Check | Result | Detail |
| --- | --- | --- |
| only eligible training trials were used | pass | 123 of 123 eligible training trials |
| no validation or held-out trial was read | pass | all 32 assigned validation and 33 assigned held-out trial rows remain untouched |
| every feature set is the training split | pass | split 'train' only |
| trials form contiguous groups of near-equal size | pass | 5 groups sized [25, 25, 25, 24, 24] |
| every trial is assessed exactly once | pass | 123 assessments over 123 trials |
| no trial appears in both sides of a fold | pass | fold training and assessment trials are disjoint and complete |
| each fold was standardized without its assessment trials | pass | 25 fold normalizations, one per lag and fold |
| every lag was scored on identical target samples | pass | 41177 shared target samples across 5 lags |
| the frozen grid was searched completely | pass | 5 lags x 6 alphas = 30 candidates |
| per-fold diagnostics cover every candidate and fold | pass | 150 fold results |
| out-of-fold predictions are complete and finite | pass | 41177 predictions per candidate, none unfilled |
| the selection follows the frozen tie-breaking rules | pass | lag 50 ms with alpha 10000.0 |
| the selected configuration is inside the frozen grid | pass | lag 50 ms in [0, 50, 100, 150, 200], alpha 10000.0 in [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0] |
| the winning score is not a near-tie decided by noise | pass | 1 candidate within 1e-12 of the best score |

Overall: all checks passed.

## Artifacts

- `results/selection/fold_assignments.csv`
- `results/selection/cv_results.csv`
- `results/selection/fold_metrics.csv`
- `results/selection/fold_normalization.csv`
- `results/selection/out_of_fold_predictions.npz`
- `results/selected_configuration.json`
- `results/selection_summary.json`
- `results/selection_summary.md`
