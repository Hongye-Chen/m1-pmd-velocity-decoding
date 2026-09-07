# Official constant and Ridge baselines

## Validation results

| Model | R2 x | R2 y | R2 mean | Pearson x | Pearson y | RMSE x (cm/s) | RMSE y (cm/s) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Constant baseline | -0.000140 | -0.000276 | **-0.000208** | undefined | undefined | 7.578223 | 7.336734 |
| Ridge | 0.543588 | 0.552308 | **0.547948** | 0.745636 | 0.743826 | 5.119359 | 4.908315 |

Both models were evaluated once on the same 8183 validation samples from 24 trials.

The constant baseline predicts the mean training velocity (0.062366, 0.460252) cm/s for every sample. It uses no neural data, so its correlation is undefined rather than zero and its score is what predicting a fixed velocity earns on this validation set.

## Configuration

- Lag: **50 ms**, read from `selected_configuration.json`.
- Ridge alpha: **10000**, read from the same file.
- Neural history: 200 ms in 4 bins of 50 ms, sampled every 10 ms.
- Eligibility: target eligibility uses the selected lag, frozen for every reported model so all models share one validation sample set.
- Training: 43022 samples from 123 trials, 212 feature columns over 53 units.

## Procedure

- Normalization: fitted once on the complete training set and applied unchanged to validation.
- Constant baseline: predicts the mean training velocity for every sample and uses no neural data.
- Ridge: fresh model fitted on the complete training set at the selected alpha.
- Evaluation: each model scored once on validation; no tuning followed.
- Metric: per-axis coefficient of determination, Pearson correlation, and root-mean-square error, each reported per axis and as the arithmetic mean of the axes.

## Training-set fit

Reported for the train-to-validation gap only, not as a project result. Ridge mean R2 on the training set is 0.582208, against 0.547948 on validation.

## Checks

| Check | Result | Detail |
| --- | --- | --- |
| the configuration was read from the selection step, not chosen here | pass | lag 50 ms and alpha 10000 read from selected_configuration.json, both inside the frozen grid |
| the recording is the one the configuration was selected on | pass | checksum 83befb05b5d2... matches |
| both splits were built at the selected lag | pass | lag 50 ms for training and validation |
| target eligibility uses the selected lag | pass | eligibility anchored on 50 ms, the frozen rule for every reported model |
| every eligible training trial was fitted and no other | pass | 123 of 123 eligible training trials |
| every eligible validation trial was evaluated and no other | pass | 24 of 24 eligible validation trials |
| the training and validation samples are disjoint | pass | 123 training and 24 validation trials share no trial |
| the held-out split was not read | pass | all 33 assigned held-out trial rows remain untouched |
| normalization was fitted once on the complete training set | pass | fitted on 'train' over 43022 samples from 123 trials |
| validation was standardized with the training statistics | pass | standardized validation features retain a mean offset of up to 0.2065, so they were not re-centred on themselves |
| the constant baseline is the mean of the complete training targets | pass | baseline velocity (0.0624, 0.4603) cm/s equals the training mean |
| the constant baseline scores exactly zero on its own training targets | pass | training R2 +0.00e+00 and +0.00e+00, the arithmetic identity of predicting a set's own mean |
| the Ridge model has one coefficient per feature column and axis | pass | 2 axes x 212 columns (4 bins x 53 units) |
| the Ridge model used the selected alpha | pass | alpha 10000 |
| each model produced one finite prediction per validation sample | pass | 8183 predictions for each of 2 models |
| the saved model artifacts reproduce the predictions exactly | pass | both models reloaded from JSON and predicted identically, bit for bit |
| the reported metrics recompute from the saved per-sample predictions | pass | 8183 saved rows reproduce every reported R2 |
| every saved prediction is identified by its validation sample | pass | trial row and behaviour sample index recorded for all 8183 predictions |

Overall: all checks passed.

## Artifacts

- `results/baselines/validation_predictions.csv`
- `results/baselines/constant_baseline.json`
- `results/baselines/ridge_decoder.json`
- `results/baseline_metrics.json`
- `results/baseline_summary.json`
- `results/baseline_summary.md`
