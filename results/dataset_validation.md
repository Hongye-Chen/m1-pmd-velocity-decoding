# Dataset validation summary

## Recording

- Dandiset 000688, version `0.250122.1735`
- Subject T, session start 2013-09-06T10:19:07.391000-04:00
- File `sub-T_ses-RT-20130906_behavior+ecephys.nwb`, 41528428 bytes
- SHA-256 `83befb05b5d265c92e03c544c420e2bc0b6c7d3248fa14492c17f39a8aa1f124`

## Neural and behavioral data

- 53 sorted units: 12 M1, 41 PMd
- 107316 cursor samples over 1073.15 s at 100.00 Hz
- Cursor series sharing one timebase: cursor_pos, cursor_vel, cursor_acc
- Spikes 310001 raw, 309719 retained after trimming to [0.020, 1073.170] s
- Removed 5 spikes before the behavioral start and 277 after the behavioral end

## Trials and splits

- 216 trials, 176 eligible after the frozen rules
- Excluded artifact rows: [109, 189]

| Split | Original rows | Assigned | Retained |
| --- | --- | --- | --- |
| train | 0-150 | 151 | 123 |
| validation | 151-182 | 32 | 24 |
| test | 183-215 | 33 | 29 |

Eligibility rule pass counts:

- `is_reward`: 196
- `has_required_attempts`: 179
- `has_required_finite_cues`: 179
- `cues_within_trial_bounds`: 179
- `is_not_excluded_row`: 214

## Checks

| Check | Result | Detail |
| --- | --- | --- |
| file size matches the selected asset | pass | 41528428 bytes |
| file SHA-256 matches the selected asset | pass | 83befb05b5d265c92e03c544c420e2bc0b6c7d3248fa14492c17f39a8aa1f124 |
| sorted unit count | pass | 53 units (expected 53) |
| units per cortical area | pass | {'M1': 12, 'PMd': 41} (expected {'M1': 12, 'PMd': 41}) |
| unit order is stable and contiguous | pass | rows 0..52 |
| behavior sample count | pass | 107316 samples (expected 107316) |
| behavior timestamps are finite | pass | [0.020, 1073.170] s |
| behavior timestamps strictly increase | pass | median step 0.010000 s |
| cursor series share one timebase | pass | cursor_pos, cursor_vel, cursor_acc |
| cursor position shape | pass | (107316, 2) |
| cursor position values are finite | pass | 0 non-finite values |
| cursor velocity shape | pass | (107316, 2) |
| cursor velocity values are finite | pass | 0 non-finite values |
| spike times are finite | pass | 309719 spikes retained |
| spike times are ordered within each unit | pass | 53 units checked |
| trimmed spikes lie inside the common interval | pass | [0.020, 1073.170] s |
| trimming removed the documented neural margins | pass | 5 before start and 277 after end (expected 5 and 277) |
| trial count | pass | 216 trials (expected 216) |
| trial identifiers match their original rows | pass | NWB trial ids equal zero-based row numbers |
| splits are disjoint and cover every trial row | pass | 216 rows across 3 splits |
| split labels follow the original trial rows | pass | no trial was relabelled by filtering |
| retained trials per split | pass | {'train': 123, 'validation': 24, 'test': 29} (expected {'train': 123, 'validation': 24, 'test': 29}) |
| frozen artifact rows are excluded | pass | rows [109, 189] are not retained |
| manifests written | pass | 53 unit rows and 216 trial rows |

Overall: all checks passed.

## Artifacts

- `results/manifests/units.csv`
- `results/manifests/trials.csv`
- `results/dataset_validation.json`
- `results/dataset_validation.md`
