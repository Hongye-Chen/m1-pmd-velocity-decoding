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

## Run the tests

```powershell
pytest
```

## Repository layout

| Path | Contents |
| --- | --- |
| `neural_decoding/` | Reusable package: configuration, logging, seeding |
| `configs/` | Tracked defaults, plus an untracked local override for paths |
| `scripts/` | Runnable entry points, invoked as `python -m scripts.<name>` |
| `tests/` | Automated tests, run with `pytest` |
| `results/` | Generated metrics, predictions, and manifests |
| `reports/` | Generated figures and report tables |

`requirements.txt` lists the direct dependencies, pinned to the versions the
project was verified against. It doubles as the record of the audited environment
and is not an installation target for the shared `BCI` environment.
