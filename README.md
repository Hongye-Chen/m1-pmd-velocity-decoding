# Neural decoding of cursor velocity from M1 and PMd spiking activity

This project decodes two-dimensional cursor velocity from M1 and PMd sorted-unit
spiking activity in one public recording session. The implemented models use only
causal neural inputs and preserve complete trials throughout selection and
evaluation.

## Current validation results

The validation set contains 8,183 samples from 24 trials. Every decoder used the
same 123 eligible training trials and identical validation targets. Ridge and
Kalman were fixed before validation; GRU validation was used for early stopping
and configuration selection, so its scores are development results.

| Model | Validation R² (x, y) | Mean R² | Mean correlation | Mean RMSE (cm/s) |
| --- | --- | --- | --- | --- |
| Constant baseline | -0.000140, -0.000276 | -0.000208 | undefined | 7.457 |
| Ridge | 0.543588, 0.552308 | **0.547948** | 0.745 | 5.014 |
| Kalman | 0.527121, 0.473273 | **0.500197** | 0.720 | 5.267 |
| GRU, three-seed mean (development) | 0.688602, 0.664263 | **0.676432** | 0.826 | 4.239 |

The constant model predicts the mean training velocity and uses no neural data.
Compact, machine-readable evidence is available in
[`results/baseline_summary.json`](results/baseline_summary.json),
[`results/kalman_decoder_selection/`](results/kalman_decoder_selection/), and
[`results/kalman_validation/`](results/kalman_validation/). GRU selection
evidence and frozen models are under
[`results/gru_input_selection/`](results/gru_input_selection/) and
[`results/gru_architecture_selection/`](results/gru_architecture_selection/).

Held-out evaluation has not occurred; those trials remain unopened. Final model
comparisons will use that held-out result rather than the GRU development score.

## Data

The NWB recording is not included in the repository:

| Field | Value |
| --- | --- |
| Dandiset | [DANDI 000688](https://doi.org/10.48324/dandi.000688/0.250122.1735) |
| Version | `0.250122.1735` |
| License | CC BY 4.0 |
| Subject and session | subject T, random-target task, 2013-09-06 |
| File | `sub-T/sub-T_ses-RT-20130906_behavior+ecephys.nwb` |
| Size | 41,528,428 bytes |
| SHA-256 | `83befb05b5d265c92e03c544c420e2bc0b6c7d3248fa14492c17f39a8aa1f124` |

Download it to any location outside the repository.

## Setup

Python 3.11 is recommended:

```powershell
conda env create -f environment.yml
conda activate neural-decoding
```

Create `configs/local.yaml`, which is untracked so machine-specific paths never
enter version control:

```yaml
paths:
  nwb_file: "D:/path/to/sub-T_ses-RT-20130906_behavior+ecephys.nwb"
```

For a single run, the environment variable below overrides the local file:

```powershell
$env:NEURAL_DECODING_NWB_FILE = "D:/path/to/sub-T_ses-RT-20130906_behavior+ecephys.nwb"
```

CPU execution is sufficient for the classical decoders and GRU inference. A CUDA
GPU is recommended for GRU training and configuration selection.

## Methods

### Ridge

Each sample pairs 200 ms of causal spike history, as four chronological half-open
50 ms bins across all 53 units, with the cursor velocity at a later target time.
Five grouped training folds selected a 50 ms lag and Ridge alpha 10,000. The
temporary cross-validation models were discarded; the reported Ridge model was
fitted on the complete training set and evaluated once on validation.

### Kalman decoder

The training-only search compares four state definitions (`V`, `PV`, `VA`, and
`PVA`), neural count widths of 10, 20, and 50 ms, and interval-end lags of 0, 50,
and 100 ms. Each candidate uses fresh non-overlapping spike counts and is assessed
recursively without assessment velocity or acceleration. Position is
available only once at trial initialization when it belongs to the state.

The selected `PVA / 50 ms width / 50 ms lag` configuration achieved a mean
out-of-fold R² of 0.506909. It was refitted on the complete training set and
evaluated once on the same validation targets as Ridge. The frozen fitted model is
included at [`results/kalman_validation/model.json`](results/kalman_validation/model.json).
Sample-level validation predictions remain omitted; their hash is retained in the
compact validation summary.

### GRU decoder

The compact sequence-to-one GRU uses the same 53 units and target identities.
Input selection compared 100 and 200 ms histories with 10, 20, and 50 ms bins
while keeping the neural interval end 50 ms before the velocity target. The
selected input contains four chronological 50 ms bins spanning 200 ms.

Architecture selection compared one- and two-layer GRUs, hidden sizes 32, 64,
and 128, learning rates 0.001 and 0.0003, and dropout 0 or 0.2 where applicable.
Each configuration used seeds 0, 1, and 2. The selected model has two layers,
hidden size 128, dropout 0.2, and learning rate 0.001. Its complete configuration
is in
[`results/gru_architecture_selection/final_configuration.json`](results/gru_architecture_selection/final_configuration.json);
the directory also contains all three fitted seed checkpoints.

The selection tables and summaries are supporting evidence. Most users should
use the frozen configuration and checkpoints rather than rerun the selection
workflow.

## Limitations

This is an offline, within-session result from one subject and one recording.
Neither cross-session nor cross-subject generalization has been tested. The
Kalman posterior is overconfident—its nominal 95% velocity ellipse covered 86.15%
of validation targets—and its process and innovation residuals remain temporally
correlated. It should be treated as a causal linear decoder, not as a fully
calibrated probabilistic model.

GRU validation performance is tuning evidence because validation controlled both
early stopping and configuration selection. It is not an unbiased final estimate;
that comparison remains reserved for the unopened held-out trials.

## Run the tests

```powershell
pytest
```

The test suite is deliberately independent of the recording, so it runs in
seconds without the data present. Runtime checks that need the recording execute
inside the relevant data-processing commands.

## Repository layout

| Path | Contents |
| --- | --- |
| `neural_decoding/` | Reusable package: configuration, logging, seeding, session loading, trial splits, feature construction, lag selection, metrics, decoders |
| `configs/` | Tracked defaults, plus an untracked local override for paths |
| `scripts/` | Runnable entry points, invoked as `python -m scripts.<name>` |
| `tests/` | Automated tests, run with `pytest` |
| `results/` | Tracked metrics, selection evidence, validation summaries, and frozen model artifacts; sample-level predictions remain private |
| `reports/` | Generated figures and report tables |

`requirements.txt` records the dependency versions used to verify the project.
