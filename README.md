# Neural decoding of cursor velocity from M1 and PMd spiking activity

Offline decoding of two-dimensional cursor velocity from the causal history of
simultaneously recorded primary motor and dorsal premotor sorted-unit spiking
activity, within a single public recording session.

The repository currently contains the project scaffolding: configuration loading,
logging, deterministic seeding, and an environment check. Data loading, features,
decoders, and results follow.

## Setup

The project runs inside an existing conda environment named `BCI`. No new
environment is created and no packages need to be installed.

Deactivate whatever environment is currently active, then activate `BCI`:

```powershell
conda deactivate
conda activate BCI
```

Run every command below from the repository root with `BCI` active.

## Obtain the recording

The neural recording is not part of this repository. It is one public NWB file:

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

## Point the project at the recording

Create `configs/local.yaml`, which is untracked so machine-specific paths never
enter version control, and give it the absolute path to the downloaded file:

```yaml
paths:
  nwb_file: "D:/path/to/sub-T_ses-RT-20130906_behavior+ecephys.nwb"
```

A single run can instead be pointed at the file through an environment variable,
which takes precedence over the local file:

```powershell
$env:NEURAL_DECODING_NWB_FILE = "D:/path/to/sub-T_ses-RT-20130906_behavior+ecephys.nwb"
```

Configuration is merged from `configs/default.yaml`, then `configs/local.yaml`, then
`NEURAL_DECODING_NWB_FILE`.

## Verify the environment

```powershell
python -m scripts.check_environment
```

The check imports every dependency, compares the active versions against the pins
in `requirements.txt`, imports the project package, and reports whether the
configured recording is present and whether CUDA is available. It exits non-zero if
a dependency cannot be imported. A version that only differs from its pin is
reported without failing, because the environment is shared and is not modified to
satisfy this check.

## Validate the recording

```powershell
python -m scripts.validate_dataset
```

This verifies the configured file against the frozen dataset decision, then writes
the unit manifest, the trial and split manifest, and a validation summary under
`results/`. It checks the file size and SHA-256, the sorted unit count and cortical
area assignment, the cursor timebase and sample count, spike trimming to the common
neural-behavioral interval, and the trial split and eligibility counts. It exits 1
if a check fails and 2 if the recording is not configured or is a different file.

## Build the causal features

```powershell
python -m scripts.build_features
python -m scripts.build_features --lags 0 100 200
```

Each sample pairs 200 ms of causal spike history, as four chronological half-open
50 ms bins across all 53 units, with the cursor velocity a fixed lag later. Both the
history and the target must lie inside a single trial.

The neural-to-target lag is not fixed in advance. The command builds one feature set
per candidate lag from `features.candidate_lags_ms` in `configs/default.yaml`, or per
lag given on the command line. Target eligibility is anchored on the largest
candidate, so every lag is scored on an identical set of target samples in an
identical order and only the neural window moves. That keeps the later comparison
between lags fair rather than rewarding whichever lag retained the easiest samples.

The command writes the feature arrays, per-sample manifests, and the full-training
normalization for each lag, then validates the result and exits 1 if a check fails.
Only training features are built here: the lag is chosen from training data alone by
cross-validation, so no validation or held-out artifact is produced.

## Select the lag and Ridge alpha

```powershell
python -m scripts.select_lag_alpha
```

This chooses the one shared neural-to-target lag used by every later decoder, along
with a Ridge penalty, using only the eligible training trials. The 123 training
trials are sorted by original row and divided into five contiguous groups; each fold
trains on four groups and is assessed on the fifth, with normalization fitted on the
fold's training trials and applied unchanged to its assessment trials. Out-of-fold
predictions are concatenated in original sample order, and the selection score is
the mean of the per-axis coefficients of determination.

Validation and held-out trials are not read. The Ridge models fitted here are
temporary selection instruments and are discarded; the official models are fitted
separately on the complete training set. The outcome is written to
`results/selected_configuration.json`, with the full grid, per-fold diagnostics, and
ranking under `results/selection/`.

The frozen selection is a **50 ms lag** with **Ridge alpha 10000**. Its concatenated
out-of-fold mean axis-specific R2 is 0.563601. This is a training-only selection
diagnostic, not an official validation benchmark; the official constant and Ridge
models are fitted and evaluated separately in the next stage.

## Run the tests

```powershell
pytest
```

The test suite is deliberately independent of the recording, so it runs in seconds
without the data present. Checks that need the actual file live in the validation
command above.

## Repository layout

| Path | Contents |
| --- | --- |
| `neural_decoding/` | Reusable package: configuration, logging, seeding, session loading, trial splits |
| `configs/` | Tracked defaults, plus an untracked local override for paths |
| `scripts/` | Runnable entry points, invoked as `python -m scripts.<name>` |
| `tests/` | Automated tests, run with `pytest` |
| `results/` | Generated metrics, predictions, and manifests |
| `reports/` | Generated figures and report tables |

`requirements.txt` lists the direct dependencies, pinned to the versions the
project was verified against. It doubles as the record of the audited environment
and is not an installation target for the shared `BCI` environment.
