# Feature validation summary

Generated 2026-08-24T09:55:52+00:00.

## Causal protocol

- History: [reference - 200 ms, reference), half-open, bin 0 oldest
- Bins: 4 x 50 ms, offsets [-0.2, -0.15, -0.1, -0.05, 0.0] s
- Prediction cadence: 10 ms
- Target: stored cursor velocity at reference + 100 ms
- Array layout: (n_samples, n_bins, n_units), flattened bin-major: bin 0 units 0..n, then bin 1, and so on

## Samples

| Split | Samples | Feasibility reference | Difference | Trials | Feature columns |
| --- | --- | --- | --- | --- | --- |
| train | 42407 | 42407 | +0 | 123 | 212 |
| validation | 8063 | 8063 | +0 | 24 | 212 |

The held-out split is not built at this stage.

## Normalization

- Fitted on `train` using 42407 samples
- Feature means span 0.0006 to 1.5173 spikes per bin
- Feature deviations span 0.0243 to 1.5059
- Features held at the deviation floor: 0
- Target mean [0.05082662106745708, 0.46042059565616644] cm/s, deviation [7.245433872889509, 7.8269701813022525] cm/s

## Checks

| Check | Result | Detail |
| --- | --- | --- |
| feature protocol matches the frozen specification | pass | 200 ms history as 4 x 50 ms bins, 10 ms cadence, 100 ms target lag |
| train sample count matches the feasibility reference | pass | 42407 samples (reference 42407, difference +0) |
| train feature shape | pass | (42407, 4, 53) |
| train target shape | pass | (42407, 2) |
| train counts are non-negative integers without overflow | pass | dtype uint16, maximum 10 spikes in one bin |
| train targets and timestamps are finite | pass | 42407 samples checked |
| train target sits exactly one lag after the reference | pass | maximum deviation 9.096e-14 s from 0.1 s |
| train history stays inside its own trial | pass | tightest margin -1.137e-13 s before the trial start |
| train target stays inside its own trial | pass | tightest margin -0.000e+00 s before the trial stop |
| train samples are ordered and unique | pass | reference indices strictly increase across the split |
| train uses only eligible trials of its own split | pass | 123 of 123 eligible trials contributed |
| train every eligible trial contributed a sample | pass | 0 eligible trials produced no sample |
| train unit order matches the session | pass | 53 units in loader order |
| validation sample count matches the feasibility reference | pass | 8063 samples (reference 8063, difference +0) |
| validation feature shape | pass | (8063, 4, 53) |
| validation target shape | pass | (8063, 2) |
| validation counts are non-negative integers without overflow | pass | dtype uint16, maximum 9 spikes in one bin |
| validation targets and timestamps are finite | pass | 8063 samples checked |
| validation target sits exactly one lag after the reference | pass | maximum deviation 9.096e-14 s from 0.1 s |
| validation history stays inside its own trial | pass | tightest margin 0.000e+00 s before the trial start |
| validation target stays inside its own trial | pass | tightest margin -0.000e+00 s before the trial stop |
| validation samples are ordered and unique | pass | reference indices strictly increase across the split |
| validation uses only eligible trials of its own split | pass | 24 of 24 eligible trials contributed |
| validation every eligible trial contributed a sample | pass | 0 eligible trials produced no sample |
| validation unit order matches the session | pass | 53 units in loader order |
| training and validation samples share no trial | pass | 123 training and 24 validation trials |
| no held-out trial appears in any built split | pass | 33 held-out trials remain unused |
| no held-out artifact was written | pass | artifact directory contains no held-out file |
| normalization was fitted on training data only | pass | fitted on train with 42407 samples |
| standardized training features are centred and scaled | pass | largest mean 1.650e-14, largest scale error 8.860e-13 |
| validation is transformed with the unchanged training statistics | pass | validation retains a residual mean of 2.072e-01 rather than 0 |
| train manifest is aligned with the arrays | pass | 42407 manifest rows for 42407 samples |
| validation manifest is aligned with the arrays | pass | 8063 manifest rows for 8063 samples |

Overall: all checks passed.

## Artifacts

- `results/features/train.npz`
- `results/features/samples_train.csv`
- `results/features/validation.npz`
- `results/features/samples_validation.csv`
- `results/feature_normalization.json`
- `results/feature_validation.json`
- `results/feature_validation.md`
