#!/usr/bin/env python3
"""Create audit tables and publication figures for the Qian-200 run."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import random
from statistics import fmean
import tempfile

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "retry-eval-matplotlib")
)
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
    raise SystemExit(
        'Matplotlib is required. Install the analysis extra with: '
        'python -m pip install -e ".[analysis]"'
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METHODS = (
    "gittins_full_panel_cost",
    "gittins_unit_run_cost",
    "ucb_e",
    "uniform",
)
METHOD_LABELS = {
    "gittins_full_panel_cost": "GittinsEval (full-panel mean cost)",
    "gittins_unit_run_cost": "GittinsEval (unit-run cost)",
    "ucb_e": "UCB-E",
    "uniform": "Uniform",
}
COLORS = {
    "gittins_full_panel_cost": "#0072B2",
    "gittins_unit_run_cost": "#D55E00",
    "ucb_e": "#009E73",
    "uniform": "#CC79A7",
}
LINESTYLES = {
    "gittins_full_panel_cost": "-",
    "gittins_unit_run_cost": "--",
    "ucb_e": ":",
    "uniform": "-.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "qian200"
    )
    parser.add_argument(
        "--baseline-json",
        type=Path,
        help="Held-out baseline JSON; defaults to final_held_out.json in --run-dir.",
    )
    parser.add_argument(
        "--curve-csv",
        type=Path,
        help="Matched-budget curve; defaults to the 100-seed curve when present.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "docs" / "results" / "qian200",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=PROJECT_ROOT / "docs" / "figures" / "qian200",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261005)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_curve(path: Path) -> dict[str, list[dict[str, float]]]:
    by_method: dict[str, list[dict[str, float]]] = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            by_method[row["method"]].append(
                {
                    "budget_usd": float(row["budget_usd"]),
                    "evaluations": float(row["mean_completed_run_evaluations"]),
                    "accuracy": float(row["mean_recommended_panel_accuracy"]),
                    "regret": float(row["mean_simple_regret"]),
                    "optimal_fraction": float(row["optimal_selection_fraction"]),
                    "seed_count": float(row["seed_count"]),
                }
            )
    missing = set(METHODS) - set(by_method)
    if missing:
        raise ValueError(f"Curve is missing methods: {sorted(missing)}")
    return dict(by_method)


def load_panel(path: Path) -> tuple[dict[str, list[dict[str, object]]], dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    snapshots: dict[str, object] = {}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            configuration_id = str(row["configuration_id"])
            grouped[configuration_id].append(row)
            snapshot = row.get("configuration_snapshot")
            previous = snapshots.setdefault(configuration_id, snapshot)
            if previous != snapshot:
                raise ValueError(
                    f"Inconsistent configuration snapshot on line {line_number}"
                )
    return dict(grouped), snapshots


def percentile(sorted_values: list[float], probability: float) -> float:
    position = (len(sorted_values) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def bootstrap_mean_interval(
    values: list[float], *, samples: int, seed: int
) -> tuple[float, float]:
    if samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    rng = random.Random(seed)
    count = len(values)
    estimates = sorted(
        fmean(values[rng.randrange(count)] for _ in range(count))
        for _ in range(samples)
    )
    return percentile(estimates, 0.025), percentile(estimates, 0.975)


def first_budget_at_or_above(
    points: list[dict[str, float]], threshold: float
) -> float | None:
    return next(
        (
            point["budget_usd"]
            for point in points
            if point["optimal_fraction"] + 1e-12 >= threshold
        ),
        None,
    )


def stable_budget_at_or_above(
    points: list[dict[str, float]], threshold: float
) -> float | None:
    return next(
        (
            point["budget_usd"]
            for index, point in enumerate(points)
            if all(
                later["optimal_fraction"] + 1e-12 >= threshold
                for later in points[index:]
            )
        ),
        None,
    )


def trapezoid_auc(points: list[dict[str, float]]) -> float:
    return sum(
        (left["regret"] + right["regret"])
        * 50.0
        * (right["budget_usd"] - left["budget_usd"])
        for left, right in zip(points, points[1:])
    )


def short_model(model: object) -> str:
    return {
        "openai/gpt-oss-20b": "GPT-OSS-20B",
        "mistralai/ministral-8b-2512": "Ministral-8B",
        "qwen/qwen3-32b": "Qwen3-32B",
    }.get(str(model), str(model))


def workflow_label(snapshot: object) -> str:
    if not isinstance(snapshot, dict):
        return "unknown"
    solvers = snapshot.get("solver_models", [])
    verifiers = snapshot.get("verifier_models", [])
    if len(solvers) == 2 and len(verifiers) == 1:
        return (
            f"{short_model(solvers[0])} → {short_model(verifiers[0])} → "
            f"{short_model(solvers[1])}"
        )
    return json.dumps(snapshot, sort_keys=True)


def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "figure.titlesize": 15,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#D9D9D9",
            "grid.linewidth": 0.7,
            "grid.alpha": 0.75,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
        }
    )


def save_figure(figure: plt.Figure, directory: Path, name: str) -> None:
    svg_path = directory / f"{name}.svg"
    figure.savefig(svg_path)
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_path.read_text().splitlines()) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    figure.savefig(directory / f"{name}.png", dpi=220)
    plt.close(figure)


def plot_method_lines(
    axis: plt.Axes,
    curve: dict[str, list[dict[str, float]]],
    *,
    x_key: str,
    y_key: str,
    multiplier: float,
) -> None:
    for method in METHODS:
        axis.plot(
            [point[x_key] for point in curve[method]],
            [point[y_key] * multiplier for point in curve[method]],
            color=COLORS[method],
            linestyle=LINESTYLES[method],
            linewidth=2.0,
            label=METHOD_LABELS[method],
        )


def write_anytime_figure(
    directory: Path, curve: dict[str, list[dict[str, float]]]
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12.0, 7.8), sharey="row")
    panels = (
        (axes[0, 0], "budget_usd", "accuracy", 100.0),
        (axes[0, 1], "evaluations", "accuracy", 100.0),
        (axes[1, 0], "budget_usd", "regret", 100.0),
        (axes[1, 1], "evaluations", "regret", 100.0),
    )
    for axis, x_key, y_key, multiplier in panels:
        plot_method_lines(
            axis, curve, x_key=x_key, y_key=y_key, multiplier=multiplier
        )
        axis.set_xlim(left=0)
    min_accuracy = min(
        point["accuracy"] * 100.0 for rows in curve.values() for point in rows
    )
    axes[0, 0].set_ylim(max(0, math.floor((min_accuracy - 3) / 5) * 5), 92)
    axes[1, 0].set_ylim(0, 40)
    axes[0, 0].set_title("Recommended development accuracy")
    axes[0, 1].set_title("Recommended development accuracy")
    axes[1, 0].set_title("Simple regret to brute-force optimum")
    axes[1, 1].set_title("Simple regret to brute-force optimum")
    axes[0, 0].set_ylabel("Accuracy (%)")
    axes[1, 0].set_ylabel("Simple regret (percentage points)")
    axes[1, 0].set_xlabel("Cumulative search cost (USD, logged charges)")
    axes[1, 1].set_xlabel("Mean completed workflow evaluations")
    figure.suptitle("Fixed-sequence baseline anytime performance", y=0.995)
    figure.text(
        0.5,
        0.958,
        f"100 development questions · 27 configurations · "
        f"{int(curve[METHODS[0]][0]['seed_count'])} search seeds",
        ha="center",
        color="#555555",
    )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.93),
    )
    figure.tight_layout(rect=(0, 0, 1, 0.88))
    save_figure(figure, directory, "baseline_anytime_curves")


def write_gap_figure(directory: Path, curve: dict[str, list[dict[str, float]]]) -> None:
    figure, axis = plt.subplots(figsize=(10.5, 5.8))
    for method in METHODS:
        points = [point for point in curve[method] if point["budget_usd"] <= 0.04]
        axis.plot(
            [point["budget_usd"] * 100.0 for point in points],
            [point["regret"] * 100.0 for point in points],
            color=COLORS[method],
            linestyle=LINESTYLES[method],
            linewidth=2.2,
            marker="o",
            markersize=3.2,
            label=METHOD_LABELS[method],
        )
    axis.axhline(0, color="#555555", linewidth=1.0)
    axis.set_xlim(0, 4.0)
    axis.set_ylim(0, 40)
    axis.set_xlabel("Cumulative profiling budget (US cents)")
    axis.set_ylabel("Accuracy gap (percentage points)")
    axis.set_title("Accuracy gap to brute-force search at low profiling budgets")
    axis.legend(frameon=False, ncol=2)
    figure.tight_layout()
    save_figure(figure, directory, "baseline_gap_early_budget")


def write_configuration_figure(
    directory: Path,
    configurations: list[dict[str, object]],
    selected_ids: set[str],
) -> None:
    figure, axis = plt.subplots(figsize=(10.5, 5.8))
    costs = [float(row["mean_cost_usd"]) * 100.0 for row in configurations]
    accuracies = [float(row["development_accuracy"]) * 100.0 for row in configurations]
    axis.scatter(
        costs,
        accuracies,
        color="#777777",
        alpha=0.65,
        s=38,
        label="Fixed workflow",
    )
    best_accuracy = max(float(row["development_accuracy"]) for row in configurations)
    frontier = []
    running_best = -1.0
    for row in sorted(configurations, key=lambda item: float(item["mean_cost_usd"])):
        accuracy = float(row["development_accuracy"])
        if accuracy > running_best + 1e-12:
            frontier.append(row)
            running_best = accuracy
    axis.plot(
        [float(row["mean_cost_usd"]) * 100.0 for row in frontier],
        [float(row["development_accuracy"]) * 100.0 for row in frontier],
        color="#333333",
        linestyle="--",
        linewidth=1.3,
        label="Empirical Pareto frontier",
    )
    for row in configurations:
        configuration_id = str(row["configuration_id"])
        selected = configuration_id in selected_ids
        tied = abs(float(row["development_accuracy"]) - best_accuracy) < 1e-12
        if not (selected or tied):
            continue
        x = float(row["mean_cost_usd"]) * 100.0
        y = float(row["development_accuracy"]) * 100.0
        color = "#D55E00" if selected else "#0072B2"
        label = "Selected workflow" if selected else "Tied development optimum"
        axis.scatter(
            [x],
            [y],
            color=color,
            edgecolor="white",
            linewidth=0.8,
            s=95,
            zorder=3,
            label=label,
        )
        axis.annotate(
            label,
            (x, y),
            xytext=(7, 10 if selected else 8),
            textcoords="offset points",
            color=color,
        )
    axis.set_xlabel("Mean realized deployment cost per question (US cents)")
    axis.set_ylabel("Development accuracy (%)")
    axis.set_title("Development accuracy and deployment cost across 27 workflows")
    handles, labels = axis.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    axis.legend(unique.values(), unique.keys(), frameon=False, loc="lower right")
    figure.tight_layout()
    save_figure(figure, directory, "configuration_accuracy_cost")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def wall_seconds(summary: dict[str, object]) -> float:
    return (
        datetime.fromisoformat(str(summary["finished_at_utc"]))
        - datetime.fromisoformat(str(summary["created_at_utc"]))
    ).total_seconds()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    baseline_json = (args.baseline_json or run_dir / "final_held_out.json").resolve()
    hundred_seed_curve = run_dir / "final_held_out_100_seeds_matched_budget_curve.csv"
    curve_csv = (
        args.curve_csv
        or (
            hundred_seed_curve
            if hundred_seed_curve.exists()
            else run_dir / "final_held_out_matched_budget_curve.csv"
        )
    ).resolve()
    paths = {
        "development_summary": run_dir / "development_summary.json",
        "reserved_summary": run_dir / "reserved_test_summary.json",
        "development_panel": run_dir / "development_panel.jsonl",
        "reserved_panel": run_dir / "reserved_test_panel.jsonl",
        "selected": run_dir / "selected_configurations.json",
        "baseline": baseline_json,
        "curve": curve_csv,
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise SystemExit(f"Missing required experiment files: {missing}")

    development_summary = load_json(paths["development_summary"])
    reserved_summary = load_json(paths["reserved_summary"])
    baseline = load_json(paths["baseline"])
    selected = load_json(paths["selected"])
    curve = load_curve(paths["curve"])
    development, snapshots = load_panel(paths["development_panel"])
    reserved, _ = load_panel(paths["reserved_panel"])
    if development_summary.get("status") != "completed":
        raise SystemExit("Development collection is not complete.")
    if reserved_summary.get("status") != "completed":
        raise SystemExit("Reserved-test collection is not complete.")
    if development_summary.get("execution_counts") != {"completed": 2700}:
        raise SystemExit("Expected exactly 2,700 completed development workflows.")
    if reserved_summary.get("execution_counts") != {"completed": 100}:
        raise SystemExit("Expected exactly 100 completed reserved-test workflows.")

    selected_ids = {str(value) for value in selected["configuration_ids"]}
    configuration_rows = []
    for configuration_id, records in development.items():
        configuration_rows.append(
            {
                "configuration_id": configuration_id,
                "workflow": workflow_label(snapshots[configuration_id]),
                "development_accuracy": fmean(
                    float(row["reward"]) for row in records
                ),
                "mean_cost_usd": fmean(
                    float(row["realized_cost_usd"]) for row in records
                ),
                "selected_for_reserved_test": configuration_id in selected_ids,
            }
        )
    configuration_rows.sort(
        key=lambda row: (
            -float(row["development_accuracy"]),
            float(row["mean_cost_usd"]),
            str(row["configuration_id"]),
        )
    )

    aggregates = baseline.get("held_out_aggregates", {})
    held_out_rows = {
        str(row["method"]): row for row in baseline.get("held_out_evaluations", [])
    }
    baseline_rows = []
    for method in METHODS:
        points = curve[method]
        aggregate = aggregates.get(method, {})
        baseline_rows.append(
            {
                "method": method,
                "display_name": METHOD_LABELS[method],
                "search_seeds": int(points[0]["seed_count"]),
                "initialization_cost_usd": points[0]["budget_usd"],
                "first_50pct_optimal_cost_usd": first_budget_at_or_above(
                    points, 0.50
                ),
                "first_90pct_optimal_cost_usd": first_budget_at_or_above(
                    points, 0.90
                ),
                "stable_90pct_optimal_cost_usd": stable_budget_at_or_above(
                    points, 0.90
                ),
                "final_optimal_selection_fraction": points[-1]["optimal_fraction"],
                "regret_auc_percentage_point_usd": trapezoid_auc(points),
                "final_development_accuracy": points[-1]["accuracy"],
                "final_recommended_configuration_id": held_out_rows.get(
                    method, {}
                ).get("recommended_configuration_id"),
                "held_out_evaluation_count": aggregate.get("seed_count"),
                "reserved_test_accuracy": aggregate.get(
                    "mean_reserved_test_accuracy"
                ),
                "reserved_test_mean_deployment_cost_usd": aggregate.get(
                    "mean_reserved_test_deployment_cost_usd"
                ),
            }
        )

    selected_id = sorted(selected_ids)[0]
    development_values = [float(row["reward"]) for row in development[selected_id]]
    reserved_values = [float(row["reward"]) for row in reserved[selected_id]]
    dev_ci = bootstrap_mean_interval(
        development_values, samples=args.bootstrap_samples, seed=args.bootstrap_seed
    )
    test_ci = bootstrap_mean_interval(
        reserved_values,
        samples=args.bootstrap_samples,
        seed=args.bootstrap_seed + 1,
    )

    usage: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for partition, source in (
        ("development", development_summary),
        ("reserved_test", reserved_summary),
    ):
        for role_model, values in source["usage_by_role_and_model"].items():
            row = usage[role_model]
            for key in (
                "calls",
                "input_tokens",
                "output_tokens",
                "reasoning_tokens",
                "reported_cost_usd",
                "elapsed_seconds",
            ):
                row[key] += float(values[key])
            row[f"{partition}_calls"] += float(values["calls"])
    usage_rows = []
    for role_model, values in sorted(usage.items()):
        role, model = role_model.split(":", 1)
        calls = int(values["calls"])
        usage_rows.append(
            {
                "role": role,
                "model": model,
                "calls": calls,
                "development_calls": int(values["development_calls"]),
                "reserved_test_calls": int(values["reserved_test_calls"]),
                "input_tokens": int(values["input_tokens"]),
                "output_tokens": int(values["output_tokens"]),
                "reasoning_tokens": int(values["reasoning_tokens"]),
                "reported_cost_usd": values["reported_cost_usd"],
                "mean_latency_seconds": values["elapsed_seconds"] / calls,
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "baseline_summary.csv", baseline_rows)
    write_csv(args.output_dir / "configuration_summary.csv", configuration_rows)
    write_csv(args.output_dir / "usage_summary.csv", usage_rows)
    curve_rows = [
        {"method": method, "budget_index": index, **point}
        for method in METHODS
        for index, point in enumerate(curve[method])
    ]
    write_csv(args.output_dir / "matched_budget_curve.csv", curve_rows)

    best_accuracy = max(
        float(row["development_accuracy"]) for row in configuration_rows
    )
    summary = {
        "schema_version": "qian200-analysis-v1",
        "protocol_status": "completed_single_collection_with_offline_search",
        "development_questions": 100,
        "reserved_test_questions": 100,
        "configurations": 27,
        "development_workflows": 2700,
        "search_seed_count": int(curve[METHODS[0]][0]["seed_count"]),
        "best_development_accuracy": best_accuracy,
        "development_optimum_count": sum(
            abs(float(row["development_accuracy"]) - best_accuracy) < 1e-12
            for row in configuration_rows
        ),
        "selected_configuration_id": selected_id,
        "selected_workflow": workflow_label(snapshots[selected_id]),
        "selected_development_accuracy": fmean(development_values),
        "selected_development_accuracy_bootstrap_95_ci": list(dev_ci),
        "selected_reserved_test_accuracy": fmean(reserved_values),
        "selected_reserved_test_accuracy_bootstrap_95_ci": list(test_ci),
        "selected_reserved_test_mean_deployment_cost_usd": fmean(
            float(row["realized_cost_usd"]) for row in reserved[selected_id]
        ),
        "development_collection_reported_cost_usd": development_summary[
            "reported_cost_usd"
        ],
        "reserved_test_collection_reported_cost_usd": reserved_summary[
            "reported_cost_usd"
        ],
        "total_collection_reported_cost_usd": development_summary[
            "reported_cost_usd"
        ]
        + reserved_summary["reported_cost_usd"],
        "development_executed_calls": development_summary["executed_call_count"],
        "reserved_test_executed_calls": reserved_summary["executed_call_count"],
        "development_run_id": development_summary["run_id"],
        "reserved_test_run_id": reserved_summary["run_id"],
        "development_wall_seconds": wall_seconds(development_summary),
        "reserved_test_wall_seconds": wall_seconds(reserved_summary),
        "total_input_tokens": development_summary["input_tokens"]
        + reserved_summary["input_tokens"],
        "total_output_tokens": development_summary["output_tokens"]
        + reserved_summary["output_tokens"],
        "total_reasoning_tokens": development_summary["reasoning_tokens"]
        + reserved_summary["reasoning_tokens"],
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
        "source_artifacts": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in paths.values()
        },
    }
    (args.output_dir / "experiment_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    set_plot_style()
    write_anytime_figure(args.figure_dir, curve)
    write_gap_figure(args.figure_dir, curve)
    write_configuration_figure(args.figure_dir, configuration_rows, selected_ids)
    print(
        f"analysis={args.output_dir} figures={args.figure_dir} "
        f"seeds={summary['search_seed_count']} selected={selected_id}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
