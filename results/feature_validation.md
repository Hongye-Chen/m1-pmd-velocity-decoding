# Feature validation summary

## Causal protocol

- History: [reference - 200 ms, reference), half-open, bin 0 oldest
- Bins: 4 x 50 ms, offsets [-0.2, -0.15, -0.1, -0.05, 0.0] s
- Prediction cadence: 10 ms
- Target: stored cursor velocity at reference + lag
- Candidate lags: [0, 50, 100, 150, 200] ms
- Eligibility anchored on 200 ms. Target samples are those leaving room for the largest candidate lag and a whole history before it, so every lag is scored on identical targets.
- Array layout: (n_samples, n_bins, n_units), flattened bin-major: bin 0 units 0..n, then bin 1, and so on

## Shared target samples

- Split built: `train`
- 41177 target samples across 123 trials
- Target times span 12.070 to 746.170 s

Every candidate lag is scored on these same samples in the same order; only the neural window moves.

## Per-lag features

| Lag (ms) | Samples | Feature columns | Mean spikes per sample | Normalization fitted on |
| --- | --- | --- | --- | --- |
| 0 | 41177 | 212 | 60.45 | `train` |
| 50 | 41177 | 212 | 60.72 | `train` |
| 100 | 41177 | 212 | 60.90 | `train` |
| 150 | 41177 | 212 | 61.00 | `train` |
| 200 | 41177 | 212 | 61.05 | `train` |

Validation features: Built later, once the lag is selected from training data.
Held-out features: Not constructed before the frozen evaluation.

## Checks

| Check | Result | Detail |
| --- | --- | --- |
| history window matches the configured protocol | pass | 200 ms history as 4 x 50 ms bins, 10 ms cadence |
| candidate lags built | pass | [0, 50, 100, 150, 200] ms, eligibility anchored on 200 ms |
| lag 0 ms feature and target shapes | pass | counts (41177, 4, 53), targets (41177, 2) |
| lag 0 ms counts are non-negative integers without overflow | pass | dtype uint16, maximum 10 spikes in one bin |
| lag 0 ms targets and timestamps are finite | pass | 41177 samples checked |
| lag 0 ms target sits exactly one lag after the reference | pass | maximum deviation 0.000e+00 s from 0.0 s |
| lag 0 ms history and target stay inside their own trial | pass | history margin 2.000e-01 s, target margin -0.000e+00 s |
| lag 0 ms samples are ordered and unique | pass | target indices strictly increase across the split |
| lag 0 ms uses every eligible training trial and nothing else | pass | 123 of 123 eligible training trials |
| lag 0 ms contains no validation or held-out trial | pass | 65 trials outside training remain unused |
| lag 0 ms unit order matches the session | pass | 53 units in loader order |
| lag 0 ms manifest is aligned with the arrays | pass | 41177 manifest rows for 41177 samples |
| lag 0 ms normalization is fitted on the full training set only | pass | fitted on train over 123 trials and 41177 samples |
| lag 50 ms feature and target shapes | pass | counts (41177, 4, 53), targets (41177, 2) |
| lag 50 ms counts are non-negative integers without overflow | pass | dtype uint16, maximum 10 spikes in one bin |
| lag 50 ms targets and timestamps are finite | pass | 41177 samples checked |
| lag 50 ms target sits exactly one lag after the reference | pass | maximum deviation 6.821e-14 s from 0.05 s |
| lag 50 ms history and target stay inside their own trial | pass | history margin 1.500e-01 s, target margin -0.000e+00 s |
| lag 50 ms samples are ordered and unique | pass | target indices strictly increase across the split |
| lag 50 ms uses every eligible training trial and nothing else | pass | 123 of 123 eligible training trials |
| lag 50 ms contains no validation or held-out trial | pass | 65 trials outside training remain unused |
| lag 50 ms unit order matches the session | pass | 53 units in loader order |
| lag 50 ms manifest is aligned with the arrays | pass | 41177 manifest rows for 41177 samples |
| lag 50 ms normalization is fitted on the full training set only | pass | fitted on train over 123 trials and 41177 samples |
| lag 100 ms feature and target shapes | pass | counts (41177, 4, 53), targets (41177, 2) |
| lag 100 ms counts are non-negative integers without overflow | pass | dtype uint16, maximum 10 spikes in one bin |
| lag 100 ms targets and timestamps are finite | pass | 41177 samples checked |
| lag 100 ms target sits exactly one lag after the reference | pass | maximum deviation 9.096e-14 s from 0.1 s |
| lag 100 ms history and target stay inside their own trial | pass | history margin 1.000e-01 s, target margin -0.000e+00 s |
| lag 100 ms samples are ordered and unique | pass | target indices strictly increase across the split |
| lag 100 ms uses every eligible training trial and nothing else | pass | 123 of 123 eligible training trials |
| lag 100 ms contains no validation or held-out trial | pass | 65 trials outside training remain unused |
| lag 100 ms unit order matches the session | pass | 53 units in loader order |
| lag 100 ms manifest is aligned with the arrays | pass | 41177 manifest rows for 41177 samples |
| lag 100 ms normalization is fitted on the full training set only | pass | fitted on train over 123 trials and 41177 samples |
| lag 150 ms feature and target shapes | pass | counts (41177, 4, 53), targets (41177, 2) |
| lag 150 ms counts are non-negative integers without overflow | pass | dtype uint16, maximum 10 spikes in one bin |
| lag 150 ms targets and timestamps are finite | pass | 41177 samples checked |
| lag 150 ms target sits exactly one lag after the reference | pass | maximum deviation 9.096e-14 s from 0.15 s |
| lag 150 ms history and target stay inside their own trial | pass | history margin 5.000e-02 s, target margin -0.000e+00 s |
| lag 150 ms samples are ordered and unique | pass | target indices strictly increase across the split |
| lag 150 ms uses every eligible training trial and nothing else | pass | 123 of 123 eligible training trials |
| lag 150 ms contains no validation or held-out trial | pass | 65 trials outside training remain unused |
| lag 150 ms unit order matches the session | pass | 53 units in loader order |
| lag 150 ms manifest is aligned with the arrays | pass | 41177 manifest rows for 41177 samples |
| lag 150 ms normalization is fitted on the full training set only | pass | fitted on train over 123 trials and 41177 samples |
| lag 200 ms feature and target shapes | pass | counts (41177, 4, 53), targets (41177, 2) |
| lag 200 ms counts are non-negative integers without overflow | pass | dtype uint16, maximum 10 spikes in one bin |
| lag 200 ms targets and timestamps are finite | pass | 41177 samples checked |
| lag 200 ms target sits exactly one lag after the reference | pass | maximum deviation 6.822e-14 s from 0.2 s |
| lag 200 ms history and target stay inside their own trial | pass | history margin -1.137e-13 s, target margin -0.000e+00 s |
| lag 200 ms samples are ordered and unique | pass | target indices strictly increase across the split |
| lag 200 ms uses every eligible training trial and nothing else | pass | 123 of 123 eligible training trials |
| lag 200 ms contains no validation or held-out trial | pass | 65 trials outside training remain unused |
| lag 200 ms unit order matches the session | pass | 53 units in loader order |
| lag 200 ms manifest is aligned with the arrays | pass | 41177 manifest rows for 41177 samples |
| lag 200 ms normalization is fitted on the full training set only | pass | fitted on train over 123 trials and 41177 samples |
| every lag predicts identical targets in identical order | pass | 41177 shared target samples across 5 lags |
| each lag shifts the neural window | pass | reference samples differ between lags while targets are shared |
| no validation or held-out artifact was written | pass | training artifacts only |

Overall: all checks passed.

## Artifacts

- `results/features/train_lag000ms.npz`
- `results/features/samples_train_lag000ms.csv`
- `results/features/train_lag050ms.npz`
- `results/features/samples_train_lag050ms.csv`
- `results/features/train_lag100ms.npz`
- `results/features/samples_train_lag100ms.csv`
- `results/features/train_lag150ms.npz`
- `results/features/samples_train_lag150ms.csv`
- `results/features/train_lag200ms.npz`
- `results/features/samples_train_lag200ms.csv`
- `results/feature_normalization.json`
- `results/feature_validation.json`
- `results/feature_validation.md`
