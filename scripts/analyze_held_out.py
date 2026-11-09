"""Analyze decoder errors on the held-out trials.

Run from the repository root with the project environment active::

    python -m scripts.analyze_held_out

Exit codes: 0 when the outputs are written, 1 when a check fails, and 2 when the
held-out evaluation outputs are unavailable.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from neural_decoding.config import ConfigError, load_config, project_root
from neural_decoding.dataset import compute_sha256
from neural_decoding.metrics import MetricError, rmse_per_axis, summarize_predictions

HELD_OUT_DIR = "held_out_evaluation"
PREDICTIONS_FILE = "held_out_predictions.csv"
HELD_OUT_SUMMARY_FILE = "summary.json"
ANALYSIS_DIR = "held_out_analysis"
TRIAL_METRICS_FILE = "trial_metrics.csv"
SUMMARY_FILE = "summary.json"
FIGURES_DIR = "figures"
FIGURE_FILES: dict[str, str] = {
    "overall": "held_out_overall.png",
    "trial_rmse": "held_out_trial_rmse.png",
    "error_by_speed_direction": "held_out_error_by_speed_direction.png",
    "lowest": "held_out_lowest_error_trial.png",
    "median": "held_out_median_trial.png",
    "highest": "held_out_highest_error_trial.png",
}

AXIS_NAMES = ("x", "y")
GRU_SEEDS: tuple[int, ...] = (0, 1, 2)
PREDICTION_NAMES: tuple[str, ...] = (
    "constant",
    "ridge",
    "kalman",
    *(f"gru_seed{seed}" for seed in GRU_SEEDS),
)
TRUTH_COLUMNS = ("target_velocity_x", "target_velocity_y")
REQUIRED_COLUMNS: tuple[str, ...] = (
    "sample_index",
    "trial_row",
    "trial_id",
    "target_index",
    "target_time",
    *TRUTH_COLUMNS,
    *(f"{name}_velocity_{axis}" for name in PREDICTION_NAMES for axis in AXIS_NAMES),
)
RMSE_KEYS = ("x", "y", "mean")
REPORTED_MODELS = ("constant", "ridge", "kalman", "gru_seed_mean")
SELECTION_MODELS = ("ridge", "kalman", "gru_seed_mean")
SELECTION_ROLES = ("lowest", "median", "highest")

N_SPEED_BINS = 4
QUANTILE_METHOD = "linear"
N_SECTORS = 8
SECTOR_WIDTH_DEG = 360.0 / N_SECTORS
TOLERANCE = 1e-12

DPI = 150
LABELS = {
    "constant": "Constant",
    "ridge": "Ridge",
    "kalman": "Kalman",
    "gru": "GRU",
    "gru_seed_mean": "GRU",
}
COLORS = {
    "constant": "#8c8c8c",
    "ridge": "#1f77b4",
    "kalman": "#ff7f0e",
    "gru": "#2ca02c",
    "gru_seed_mean": "#2ca02c",
}
GROUP_LABELS = {"x": "x velocity", "y": "y velocity", "mean": "mean of axes"}
GROUP_COLORS = {"x": "#6baed6", "y": "#fd8d3c", "mean": "#636363"}
OVERALL_PANELS = (("r2", "R²"), ("pearson_r", "Pearson r"), ("rmse_cm_s", "RMSE (cm/s)"))
ROLE_TEXT = {"lowest": "lowest error", "median": "median error", "highest": "highest error"}


class AnalysisError(RuntimeError):
    """Raised when the saved held-out outputs cannot support the analysis."""


def score_rmse(
    true: np.ndarray, predictions: Mapping[str, np.ndarray]
) -> dict[str, dict[str, float]]:
    """Calculate per-axis and mean RMSE for each model and across GRU seeds.

    Each GRU seed is scored on its own predictions; predictions are never averaged
    across seeds.
    """
    scores: dict[str, dict[str, float]] = {}
    for name in PREDICTION_NAMES:
        rmse = rmse_per_axis(true, predictions[name])
        scores[name] = {"x": float(rmse[0]), "y": float(rmse[1]), "mean": float(rmse.mean())}
    seeds = np.array(
        [[scores[f"gru_seed{seed}"][key] for key in RMSE_KEYS] for seed in GRU_SEEDS]
    )
    scores["gru_seed_mean"] = dict(zip(RMSE_KEYS, map(float, seeds.mean(axis=0))))
    scores["gru_seed_std"] = dict(zip(RMSE_KEYS, map(float, seeds.std(axis=0, ddof=1))))
    return scores


def trial_metrics(table: pd.DataFrame) -> pd.DataFrame:
    """Calculate trial-level RMSE and the score used to select example trials."""
    records: list[dict[str, Any]] = []
    for trial_row, trial in table.groupby("trial_row", sort=True):
        scores = _score_subset(trial)
        record: dict[str, Any] = {
            "trial_row": int(trial_row),
            "trial_id": int(trial["trial_id"].iloc[0]),
            "n_samples": len(trial),
        }
        for name, values in scores.items():
            record.update({f"{name}_rmse_{key}": value for key, value in values.items()})
        record["selection_score"] = float(
            np.mean([scores[name]["mean"] for name in SELECTION_MODELS])
        )
        records.append(record)
    return pd.DataFrame.from_records(records)


def select_trials(trials: pd.DataFrame) -> dict[str, dict[str, int]]:
    """Return the lowest-, median-, and highest-scoring trials, ties broken by trial row."""
    ordered = trials.sort_values(["selection_score", "trial_row"], kind="mergesort")
    rows = ordered["trial_row"].to_numpy(dtype=np.int64)
    last = rows.size - 1
    positions = {"lowest": 0, "median": last // 2, "highest": last}
    return {
        role: {"trial_row": int(rows[position]), "rank": position + 1}
        for role, position in positions.items()
    }


def speed_boundaries(speed: np.ndarray) -> np.ndarray:
    """Return the quartile boundaries of the true speed, from minimum to maximum."""
    return np.quantile(
        speed, np.linspace(0.0, 1.0, N_SPEED_BINS + 1), method=QUANTILE_METHOD
    )


def assign_speed_bins(speed: np.ndarray, boundaries: np.ndarray) -> np.ndarray:
    """Return each sample's speed bin; the first is closed, later ones are (lower, upper]."""
    return np.digitize(speed, boundaries[1:-1], right=True)


def assign_direction_sectors(velocity: np.ndarray) -> np.ndarray:
    """Return each sample's direction sector, or -1 when its speed is zero."""
    angle = np.degrees(np.arctan2(velocity[:, 1], velocity[:, 0]))
    shifted = np.mod(angle + SECTOR_WIDTH_DEG / 2.0, 360.0)
    # np.mod can round a tiny negative input up to exactly 360.
    sectors = np.minimum(np.floor(shifted / SECTOR_WIDTH_DEG).astype(np.int64), N_SECTORS - 1)
    moving = np.hypot(velocity[:, 0], velocity[:, 1]) > 0.0
    return np.where(moving, sectors, -1)


def integrate_velocity(time: np.ndarray, velocity: np.ndarray) -> np.ndarray:
    """Return the displacement from the first sample by the trapezoidal rule."""
    steps = 0.5 * (velocity[1:] + velocity[:-1]) * np.diff(time)[:, np.newaxis]
    return np.vstack([np.zeros((1, velocity.shape[1])), np.cumsum(steps, axis=0)])


def speed_summary(table: pd.DataFrame) -> dict[str, Any]:
    """Return each model's RMSE within the quartiles of true speed."""
    truth = _truth(table)
    speed = np.hypot(truth[:, 0], truth[:, 1])
    boundaries = speed_boundaries(speed)
    labels = assign_speed_bins(speed, boundaries)
    bins = [
        {
            "quartile": index + 1,
            "lower_cm_s": float(boundaries[index]),
            "upper_cm_s": float(boundaries[index + 1]),
            "n_samples": int(np.count_nonzero(labels == index)),
            "rmse_cm_s": _reported(_score_subset(table[labels == index])),
        }
        for index in range(N_SPEED_BINS)
    ]
    return {
        "definition": "speed is the norm of the true velocity; boundaries are the 0th, "
        "25th, 50th, 75th, and 100th percentiles of all held-out target speeds "
        f"(numpy '{QUANTILE_METHOD}' method); the first quartile is [b0, b1] and each "
        "later quartile is (b_k, b_k+1]",
        "boundaries_cm_s": [float(value) for value in boundaries],
        "bins": bins,
    }


def direction_summary(table: pd.DataFrame) -> dict[str, Any]:
    """Return each model's RMSE within eight equal sectors of true movement direction."""
    sectors = assign_direction_sectors(_truth(table))
    entries = []
    for sector in range(N_SECTORS):
        center = sector * SECTOR_WIDTH_DEG
        entries.append(
            {
                "sector": sector,
                "center_deg": center,
                "lower_deg": center - SECTOR_WIDTH_DEG / 2.0,
                "upper_deg": center + SECTOR_WIDTH_DEG / 2.0,
                "n_samples": int(np.count_nonzero(sectors == sector)),
                "rmse_cm_s": _reported(_score_subset(table[sectors == sector])),
            }
        )
    return {
        "definition": "direction is atan2(true vy, true vx) in degrees, counterclockwise "
        f"from +x; sector k is centred on {SECTOR_WIDTH_DEG:g}k degrees and covers "
        f"[{SECTOR_WIDTH_DEG:g}k - {SECTOR_WIDTH_DEG / 2:g}, "
        f"{SECTOR_WIDTH_DEG:g}k + {SECTOR_WIDTH_DEG / 2:g}) modulo 360; samples with "
        "zero true speed have no direction and are excluded",
        "excluded_zero_speed_samples": int(np.count_nonzero(sectors < 0)),
        "sectors": entries,
    }


def trial_distribution(trials: pd.DataFrame) -> dict[str, dict[str, dict[str, float]]]:
    """Return the minimum, quartiles, and maximum of each model's trial RMSE."""
    names = ("min", "q1", "median", "q3", "max")
    levels = (0.0, 0.25, 0.5, 0.75, 1.0)
    distribution: dict[str, dict[str, dict[str, float]]] = {}
    for model in REPORTED_MODELS:
        distribution[model] = {}
        for key in RMSE_KEYS:
            values = trials[f"{model}_rmse_{key}"].to_numpy(dtype=np.float64)
            quantiles = np.quantile(values, levels, method=QUANTILE_METHOD)
            distribution[model][key] = {
                name: float(value) for name, value in zip(names, quantiles)
            }
    return distribution


def overall_metrics(table: pd.DataFrame) -> dict[str, Any]:
    """Calculate overall metrics from the saved held-out predictions."""
    truth = _truth(table)
    predictions = _predictions(table)
    scored = {
        name: summarize_predictions(truth, predictions[name], axis_names=AXIS_NAMES)
        for name in PREDICTION_NAMES
    }
    seeds = [scored[f"gru_seed{seed}"] for seed in GRU_SEEDS]
    return {
        "constant": scored["constant"],
        "ridge": scored["ridge"],
        "kalman": scored["kalman"],
        "gru": {
            "seeds": {str(seed): scored[f"gru_seed{seed}"] for seed in GRU_SEEDS},
            "seed_mean": _reduce(seeds, lambda values: float(values.mean())),
            "seed_std": _reduce(seeds, lambda values: float(values.std(ddof=1))),
        },
    }


def input_checks(
    table: pd.DataFrame, held_out: Mapping[str, Any], predictions_sha256: str
) -> list[dict[str, object]]:
    """Check that the saved predictions and summary agree."""
    samples = held_out["samples"]
    checks = [
        _check(
            "the inputs are the frozen held-out outputs",
            predictions_sha256 == held_out["artifact_hashes"]["predictions_sha256"]
            and held_out["all_checks_passed"] is True,
            f"prediction table SHA-256 {predictions_sha256[:12]}... matches the "
            "held-out summary",
        )
    ]
    missing = [column for column in REQUIRED_COLUMNS if column not in table.columns]
    checks.append(
        _check(
            "the prediction table has every column the analysis reads",
            not missing,
            f"missing {missing}" if missing else f"{len(REQUIRED_COLUMNS)} columns",
        )
    )
    if missing:
        return checks

    trials = table.groupby("trial_row", sort=True)
    complete = (
        len(table) == samples["held_out_targets"]
        and np.array_equal(table["sample_index"].to_numpy(), np.arange(len(table)))
        and [int(row) for row in trials.groups] == list(samples["held_out_trial_rows"])
        and bool((trials["trial_id"].nunique() == 1).all())
        and bool(np.all(np.isfinite(table[list(REQUIRED_COLUMNS)].to_numpy(np.float64))))
    )
    checks.append(
        _check(
            "the table holds every held-out target once with finite values",
            complete,
            f"{len(table)} targets from {trials.ngroups} trials",
        )
    )
    consecutive = all(
        bool(np.all(np.diff(trial["target_index"].to_numpy()) == 1))
        and bool(np.all(np.diff(trial["target_time"].to_numpy()) > 0.0))
        for _, trial in trials
    )
    checks.append(
        _check(
            "each trial's targets are consecutive behavior-grid steps in time order",
            consecutive,
            "required for integrating velocity into trajectories",
        )
    )
    difference = _largest_difference(held_out["metrics"], overall_metrics(table))
    checks.append(
        _check(
            "the saved predictions reproduce the overall held-out scores",
            difference <= TOLERANCE,
            f"largest difference {difference:.1e} (tolerance {TOLERANCE:g})",
        )
    )
    return checks


def run_analysis(
    table: pd.DataFrame,
    held_out: Mapping[str, Any],
    sources: Mapping[str, Path],
    output_dir: Path,
    figures_dir: Path,
    checks: Sequence[Mapping[str, object]],
) -> dict[str, Any]:
    """Calculate error patterns and write the table, figures, and summary."""
    trials = trial_metrics(table)
    selected = select_trials(trials)
    by_speed = speed_summary(table)
    by_direction = direction_summary(table)

    trial_path = output_dir / TRIAL_METRICS_FILE
    summary_path = output_dir / SUMMARY_FILE
    figure_paths = {name: figures_dir / file for name, file in FIGURE_FILES.items()}
    trials.to_csv(trial_path, index=False, lineterminator="\n")

    plot_overall(held_out["metrics"], held_out["samples"], figure_paths["overall"])
    plot_trial_rmse(trials, figure_paths["trial_rmse"])
    plot_speed_direction(by_speed, by_direction, figure_paths["error_by_speed_direction"])
    records = trials.set_index("trial_row")
    for role in SELECTION_ROLES:
        row = selected[role]["trial_row"]
        plot_trial(
            table[table["trial_row"] == row],
            records.loc[row],
            role,
            len(trials),
            figure_paths[role],
        )

    summary: dict[str, Any] = {
        "scope": f"descriptive analysis of the {len(trials)} eligible held-out trials of "
        "one session; no model was refitted or rerun and no target was excluded",
        "inputs": {
            "predictions": _repo_relative(sources["predictions"]),
            "predictions_sha256": compute_sha256(sources["predictions"]),
            "held_out_summary": _repo_relative(sources["held_out_summary"]),
            "held_out_summary_sha256": compute_sha256(sources["held_out_summary"]),
        },
        "definitions": {
            "trial_rmse": "RMSE of each velocity axis over a trial's targets and the "
            "arithmetic mean of the two axes, in cm/s",
            "gru": "each seed is scored separately; gru_seed_mean and gru_seed_std are "
            "the mean and sample standard deviation (ddof=1) of the seed-level RMSE; "
            "predictions are never averaged across seeds",
            "selection_score": "mean of the Ridge, Kalman, and GRU seed-mean trial RMSE "
            "(mean of axes), in cm/s; the constant baseline is not included",
            "selection_rule": "trials ordered by selection score, ties by trial row; the "
            "lowest is the first, the highest the last, and the median at position "
            "(n - 1) // 2, the lower middle for an even count",
            "trajectories": "true and predicted velocity integrated over the saved target "
            "times by the trapezoidal rule, every series starting at (0, 0) cm at the "
            "trial's first target",
            "trial_figures": "true velocity with Ridge, Kalman, and each GRU seed; the "
            "constant baseline predicts one fixed velocity and is not drawn",
        },
        "trial_rmse_cm_s": trial_distribution(trials),
        "error_by_speed": by_speed,
        "error_by_direction": by_direction,
        "selected_trials": {
            role: {
                "trial_row": selected[role]["trial_row"],
                "trial_id": int(records.loc[selected[role]["trial_row"], "trial_id"]),
                "rank": selected[role]["rank"],
                "n_trials": len(trials),
                "n_samples": int(records.loc[selected[role]["trial_row"], "n_samples"]),
                "selection_score": float(
                    records.loc[selected[role]["trial_row"], "selection_score"]
                ),
                "figure": _repo_relative(figure_paths[role]),
            }
            for role in SELECTION_ROLES
        },
        "checks": list(checks),
        "all_checks_passed": all(bool(check["passed"]) for check in checks),
        "artifacts": {
            "trial_metrics": _repo_relative(trial_path),
            "summary": _repo_relative(summary_path),
            "figures": {name: _repo_relative(path) for name, path in figure_paths.items()},
        },
        "artifact_hashes": {"trial_metrics_sha256": compute_sha256(trial_path)},
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return summary


def plot_overall(
    metrics: Mapping[str, Any], samples: Mapping[str, Any], path: Path
) -> None:
    """Draw the saved overall scores per axis and as the mean of axes."""
    figure = Figure(figsize=(11.0, 3.8), layout="constrained")
    axes = figure.subplots(1, 3)
    models = ("constant", "ridge", "kalman", "gru")
    width = 0.27
    gru = metrics["gru"]
    for ax, (key, title) in zip(axes, OVERALL_PANELS):
        for position, model in enumerate(models):
            entry = gru["seed_mean"] if model == "gru" else metrics[model]
            if _metric(entry, key, "mean") is None:
                ax.text(
                    position, 0.04, "undefined", ha="center", va="bottom",
                    fontsize=9, color="0.35",
                )
                continue
            for offset, group in enumerate(RMSE_KEYS):
                x = position + (offset - 1) * width
                spread = _metric(gru["seed_std"], key, group) if model == "gru" else None
                ax.bar(
                    x, _metric(entry, key, group), width, color=GROUP_COLORS[group],
                    yerr=spread, capsize=2, error_kw={"linewidth": 0.8},
                )
                if model == "gru":
                    seeds = [_metric(gru["seeds"][str(seed)], key, group) for seed in GRU_SEEDS]
                    ax.scatter(np.full(len(seeds), x), seeds, s=6, color="black", zorder=3)
        ax.axhline(0.0, color="black", linewidth=0.6)
        ax.set_xticks(range(len(models)), [LABELS[model] for model in models])
        ax.set_title(title)
    handles: list[Any] = [
        Patch(color=GROUP_COLORS[group], label=GROUP_LABELS[group]) for group in RMSE_KEYS
    ]
    handles.append(
        Line2D([], [], marker="o", linestyle="none", color="black", markersize=3,
               label="individual GRU seed")
    )
    figure.legend(handles=handles, loc="outside lower center", ncols=len(handles),
                  frameon=False)
    figure.suptitle(
        f"Held-out scores over {samples['held_out_trials']} trials and "
        f"{samples['held_out_targets']:,} targets (GRU: mean ± SD of "
        f"{len(GRU_SEEDS)} seeds)"
    )
    figure.savefig(path, dpi=DPI)


def plot_trial_rmse(trials: pd.DataFrame, path: Path) -> None:
    """Draw each model's trial RMSE, joining the same trial across models."""
    figure = Figure(figsize=(11.0, 4.0), layout="constrained")
    axes = figure.subplots(1, 3, sharey=True)
    positions = np.arange(len(REPORTED_MODELS))
    tick_labels = [
        "GRU\n(seed mean)" if name == "gru_seed_mean" else LABELS[name]
        for name in REPORTED_MODELS
    ]
    for ax, key in zip(axes, RMSE_KEYS):
        values = trials[[f"{name}_rmse_{key}" for name in REPORTED_MODELS]].to_numpy(
            dtype=np.float64
        )
        ax.plot(positions, values.T, color="0.85", linewidth=0.6, zorder=1)
        ax.boxplot(
            values, positions=positions, widths=0.5, whis=(0, 100), showfliers=False,
            medianprops={"color": "black"}, zorder=2,
        )
        for position, name in zip(positions, REPORTED_MODELS):
            ax.scatter(
                np.full(len(values), position), values[:, position], s=10,
                color=COLORS[name], zorder=3,
            )
        ax.set_xticks(positions, tick_labels)
        ax.set_title(GROUP_LABELS[key])
    axes[0].set_ylabel("Trial RMSE (cm/s)")
    figure.suptitle(
        f"RMSE within each of the {len(trials)} held-out trials "
        "(box: quartiles; whiskers: range; lines join the same trial)"
    )
    figure.savefig(path, dpi=DPI)


def plot_speed_direction(
    by_speed: Mapping[str, Any], by_direction: Mapping[str, Any], path: Path
) -> None:
    """Draw RMSE by true-speed quartile and by true-direction sector."""
    figure = Figure(figsize=(11.0, 4.6), layout="constrained")
    speed_ax = figure.add_subplot(1, 2, 1)
    direction_ax = figure.add_subplot(1, 2, 2, projection="polar")

    bins = by_speed["bins"]
    sectors = by_direction["sectors"]
    positions = np.arange(len(bins))
    centers = [float(sector["center_deg"]) for sector in sectors]
    theta = np.radians([*centers, centers[0]])
    largest = max(
        entry["rmse_cm_s"][name]["mean"] for entry in sectors for name in REPORTED_MODELS
    )
    for name in REPORTED_MODELS:
        style = {
            "color": COLORS[name],
            "marker": "o",
            "markersize": 4,
            "linestyle": "--" if name == "constant" else "-",
            "label": LABELS[name],
        }
        spread = (
            [entry["rmse_cm_s"]["gru_seed_std"]["mean"] for entry in bins]
            if name == "gru_seed_mean"
            else None
        )
        speed_ax.errorbar(
            positions, [entry["rmse_cm_s"][name]["mean"] for entry in bins],
            yerr=spread, capsize=3, **style,
        )
        radial = [entry["rmse_cm_s"][name]["mean"] for entry in sectors]
        direction_ax.plot(theta, [*radial, radial[0]], **{**style, "label": None})

    speed_ax.set_xticks(
        positions,
        [
            f"Q{entry['quartile']}\n{entry['lower_cm_s']:.1f}–{entry['upper_cm_s']:.1f}"
            for entry in bins
        ],
    )
    speed_ax.set_xlabel("True speed quartile (cm/s)")
    speed_ax.set_ylabel("RMSE, mean of axes (cm/s)")
    speed_ax.set_ylim(bottom=0.0)
    speed_ax.set_title("By true speed")
    direction_ax.set_xticks(np.radians(centers), [f"{center:g}°" for center in centers])
    direction_ax.set_ylim(0.0, 1.1 * largest)
    direction_ax.set_title("By true movement direction (radius: RMSE, cm/s)")
    figure.legend(loc="outside lower center", ncols=len(REPORTED_MODELS), frameon=False)
    figure.suptitle(
        "Held-out RMSE (mean of axes) by true speed and direction "
        "(GRU: seed mean; error bars: seed SD)"
    )
    figure.savefig(path, dpi=DPI)


def plot_trial(
    trial: pd.DataFrame, record: pd.Series, role: str, n_trials: int, path: Path
) -> None:
    """Draw true and predicted velocity and integrated trajectories for one trial."""
    time = trial["target_time"].to_numpy(dtype=np.float64)
    predictions = _predictions(trial)
    gru_label = (
        f"GRU seeds {GRU_SEEDS[0]}–{GRU_SEEDS[-1]}, RMSE "
        f"{record['gru_seed_mean_rmse_mean']:.2f} ± {record['gru_seed_std_rmse_mean']:.2f}"
    )
    series: list[tuple[np.ndarray, dict[str, Any]]] = [
        (_truth(trial), {"color": "black", "linewidth": 1.6, "zorder": 3, "label": "True"}),
        (
            predictions["ridge"],
            {"color": COLORS["ridge"], "linewidth": 1.0,
             "label": f"Ridge, RMSE {record['ridge_rmse_mean']:.2f}"},
        ),
        (
            predictions["kalman"],
            {"color": COLORS["kalman"], "linewidth": 1.0,
             "label": f"Kalman, RMSE {record['kalman_rmse_mean']:.2f}"},
        ),
        *(
            (
                predictions[f"gru_seed{seed}"],
                {"color": COLORS["gru"], "linewidth": 0.8, "alpha": 0.8,
                 "label": gru_label if index == 0 else None},
            )
            for index, seed in enumerate(GRU_SEEDS)
        ),
    ]

    figure = Figure(figsize=(11.0, 5.6), layout="constrained")
    grid = figure.add_gridspec(2, 2, width_ratios=(1.7, 1.0))
    x_ax = figure.add_subplot(grid[0, 0])
    y_ax = figure.add_subplot(grid[1, 0], sharex=x_ax)
    path_ax = figure.add_subplot(grid[:, 1])
    elapsed = time - time[0]
    for velocity, style in series:
        x_ax.plot(elapsed, velocity[:, 0], **style)
        y_ax.plot(elapsed, velocity[:, 1], **{**style, "label": None})
        displacement = integrate_velocity(time, velocity)
        path_ax.plot(displacement[:, 0], displacement[:, 1], **{**style, "label": None})
        path_ax.plot(
            displacement[-1, 0], displacement[-1, 1], marker="o", markersize=4,
            color=style["color"], alpha=style.get("alpha", 1.0),
        )
    path_ax.plot(0.0, 0.0, marker="s", markersize=5, color="black")
    for ax in (x_ax, y_ax):
        ax.axhline(0.0, color="0.7", linewidth=0.6, zorder=0)
    x_ax.set_ylabel("x velocity (cm/s)")
    y_ax.set_ylabel("y velocity (cm/s)")
    y_ax.set_xlabel("Time from the trial's first target (s)")
    x_ax.tick_params(labelbottom=False)
    path_ax.set_aspect("equal", adjustable="datalim")
    path_ax.set_xlabel("x displacement (cm)")
    path_ax.set_ylabel("y displacement (cm)")
    path_ax.set_title("Integrated velocity from a common origin")
    figure.legend(loc="outside lower center", ncols=4, frameon=False,
                  title="Trial RMSE, mean of axes (cm/s)", title_fontsize=9)
    figure.suptitle(
        f"Held-out trial {int(record['trial_id'])}: {ROLE_TEXT[role]} of {n_trials} "
        f"trials (selection score {record['selection_score']:.2f} cm/s)"
    )
    figure.savefig(path, dpi=DPI)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output-dir",
        type=Path,
        metavar="PATH",
        help="Directory for the trial table and summary (default: "
        f"<results>/{ANALYSIS_DIR}).",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        metavar="PATH",
        help=f"Directory for the figures (default: <reports>/{FIGURES_DIR}).",
    )
    args = parser.parse_args(argv)

    try:
        config = load_config()
    except ConfigError as error:
        print(f"Cannot load the configuration: {error}")
        return 2
    held_out_dir = config.paths.results_dir / HELD_OUT_DIR
    sources = {
        "predictions": held_out_dir / PREDICTIONS_FILE,
        "held_out_summary": held_out_dir / HELD_OUT_SUMMARY_FILE,
    }
    missing = [_repo_relative(path) for path in sources.values() if not path.is_file()]
    if missing:
        print(f"Cannot analyze: the held-out evaluation outputs are missing: {missing}")
        return 2
    output_dir = (args.output_dir or config.paths.results_dir / ANALYSIS_DIR).resolve()
    figures_dir = (args.figures_dir or config.paths.reports_dir / FIGURES_DIR).resolve()

    held_out = json.loads(sources["held_out_summary"].read_text(encoding="utf-8"))
    table = pd.read_csv(sources["predictions"], float_precision="round_trip")
    try:
        checks = input_checks(table, held_out, compute_sha256(sources["predictions"]))
    except (KeyError, MetricError, ValueError) as error:
        print(f"The held-out outputs cannot be checked: {error}")
        return 1
    if not all(bool(check["passed"]) for check in checks):
        _print_checks(checks)
        print("Nothing was written.")
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    summary = run_analysis(table, held_out, sources, output_dir, figures_dir, checks)
    _print_report(summary)
    return 0


def _truth(table: pd.DataFrame) -> np.ndarray:
    return table[list(TRUTH_COLUMNS)].to_numpy(dtype=np.float64)


def _predictions(table: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        name: table[[f"{name}_velocity_{axis}" for axis in AXIS_NAMES]].to_numpy(
            dtype=np.float64
        )
        for name in PREDICTION_NAMES
    }


def _score_subset(table: pd.DataFrame) -> dict[str, dict[str, float]]:
    return score_rmse(_truth(table), _predictions(table))


def _reported(scores: Mapping[str, Mapping[str, float]]) -> dict[str, Mapping[str, float]]:
    return {name: scores[name] for name in (*REPORTED_MODELS, "gru_seed_std")}


def _metric(entry: Mapping[str, Any], key: str, group: str) -> float | None:
    return entry[f"{key}_mean"] if group == "mean" else entry[key][group]


def _reduce(entries: Sequence[Any], reducer: Callable[[np.ndarray], float]) -> Any:
    """Reduce matching metric fields while preserving undefined values."""
    first = entries[0]
    if isinstance(first, Mapping):
        return {
            key: _reduce([entry[key] for entry in entries], reducer)
            for key in first
            if key != "n_samples"
        }
    if any(value is None for value in entries):
        return None
    return reducer(np.asarray(entries, dtype=np.float64))


def _largest_difference(expected: Any, actual: Any) -> float:
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping) or set(expected) != set(actual):
            return float("inf")
        return max(
            (_largest_difference(expected[key], actual[key]) for key in expected),
            default=0.0,
        )
    if expected is None or actual is None:
        return 0.0 if expected is None and actual is None else float("inf")
    return abs(float(expected) - float(actual))


def _check(name: str, passed: bool, detail: str) -> dict[str, object]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def _print_checks(checks: Sequence[Mapping[str, object]]) -> None:
    width = max(len(str(check["name"])) for check in checks)
    print("Checks")
    for check in checks:
        status = "pass" if check["passed"] else "FAIL"
        print(f"  {str(check['name']):<{width}}  {status:<4}  {check['detail']}")


def _print_report(summary: Mapping[str, Any]) -> None:
    print("Selected trials")
    for role, entry in summary["selected_trials"].items():
        print(
            f"  {role:<8} trial {entry['trial_id']:>4}  rank {entry['rank']:>2}/"
            f"{entry['n_trials']}  score {entry['selection_score']:.4f} cm/s"
        )
    boundaries = summary["error_by_speed"]["boundaries_cm_s"]
    print(
        "Speed quartile boundaries (cm/s): "
        + ", ".join(f"{value:.4f}" for value in boundaries)
    )
    print()
    _print_checks(summary["checks"])
    print("\nArtifacts")
    artifacts = summary["artifacts"]
    for path in (artifacts["trial_metrics"], artifacts["summary"], *artifacts["figures"].values()):
        print(f"  {path}")


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(project_root()).as_posix()
    except ValueError:
        return path.name


if __name__ == "__main__":
    raise SystemExit(main())
