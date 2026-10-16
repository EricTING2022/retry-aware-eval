#!/usr/bin/env python3
"""Run Qian's four fixed-sequence, retry-unaware baselines on a saved panel."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import statistics
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from retry_eval.static_baselines import (  # noqa: E402
    BASELINE_NAMES,
    BalancedPanel,
    build_matched_budget_curve,
    load_panel_jsonl,
    run_static_baseline,
    split_development_reserved,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-purchases", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--seeds",
        type=int,
        default=1,
        help="Number of consecutive search seeds beginning at --seed (default: 1).",
    )
    parser.add_argument(
        "--held-out",
        action="store_true",
        help="Search on development records and evaluate frozen choices on reserved_test.",
    )
    parser.add_argument(
        "--evaluation-panel",
        type=Path,
        help="Separate reserved-test panel collected after recommendations were frozen.",
    )
    parser.add_argument("--ucb-alpha", type=float, default=1.0)
    parser.add_argument("--cost-scaling-factor", type=float, default=1e-4)
    parser.add_argument("--gittins-grid-points", type=int, default=1025)
    parser.add_argument(
        "--budget-points",
        type=int,
        default=121,
        help="Number of common realized-USD points in the matched anytime curve.",
    )
    parser.add_argument(
        "--curve-csv",
        type=Path,
        help="Matched-budget CSV; defaults beside --output.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=BASELINE_NAMES,
        default=list(BASELINE_NAMES),
    )
    args = parser.parse_args()
    if args.seeds <= 0:
        parser.error("--seeds must be positive")
    if args.budget_points < 2:
        parser.error("--budget-points must be at least 2")
    if args.held_out and args.evaluation_panel is not None:
        parser.error("use either --held-out or --evaluation-panel, not both")

    records = load_panel_jsonl(args.panel)
    if args.held_out:
        panel, reserved_panel = split_development_reserved(records)
    else:
        if args.evaluation_panel is not None and any(
            row.partition != "development" for row in records
        ):
            parser.error("search panel must contain only development records")
        panel = BalancedPanel(records)
        if args.evaluation_panel is not None:
            evaluation_records = load_panel_jsonl(args.evaluation_panel)
            if any(row.partition != "reserved_test" for row in evaluation_records):
                parser.error(
                    "--evaluation-panel must contain only reserved_test records"
                )
            reserved_panel = BalancedPanel(evaluation_records)
        else:
            reserved_panel = None
    seeds = range(args.seed, args.seed + args.seeds)
    results = [
        run_static_baseline(
            panel,
            method,
            max_purchases=args.max_purchases,
            seed=seed,
            ucb_alpha=args.ucb_alpha,
            cost_scaling_factor=args.cost_scaling_factor,
            gittins_grid_points=args.gittins_grid_points,
        )
        for seed in seeds
        for method in args.methods
    ]
    matched_budget_curve = build_matched_budget_curve(
        results, budget_points=args.budget_points
    )
    curve_csv = args.curve_csv or args.output.with_name(
        f"{args.output.stem}_matched_budget_curve.csv"
    )
    held_out_evaluations = []
    if reserved_panel is not None:
        for result in results:
            selected = result.checkpoints[-1].recommended_configuration_id
            if selected not in reserved_panel.configuration_ids:
                raise ValueError(
                    f"Reserved-test panel is missing selected configuration {selected}."
                )
            development_snapshot = panel.configuration_snapshots.get(selected)
            reserved_snapshot = reserved_panel.configuration_snapshots.get(selected)
            if development_snapshot != reserved_snapshot:
                raise ValueError(
                    f"Configuration snapshot differs across partitions for {selected}."
                )
            held_out_evaluations.append(
                {
                    "method": result.method,
                    "seed": result.seed,
                    "recommended_configuration_id": selected,
                    "development_purchases": len(result.purchases),
                    "development_accuracy": panel.panel_accuracy(selected),
                    "reserved_test_accuracy": reserved_panel.panel_accuracy(selected),
                    "reserved_test_mean_deployment_cost_usd": (
                        reserved_panel.panel_mean_cost(selected)
                    ),
                }
            )
    aggregates = {}
    for method in args.methods:
        method_rows = [
            row for row in held_out_evaluations if row["method"] == method
        ]
        if method_rows:
            aggregates[method] = {
                "seed_count": len(method_rows),
                "selection_counts": dict(
                    sorted(
                        Counter(
                            row["recommended_configuration_id"] for row in method_rows
                        ).items()
                    )
                ),
                "mean_reserved_test_accuracy": statistics.fmean(
                    row["reserved_test_accuracy"] for row in method_rows
                ),
                "mean_reserved_test_deployment_cost_usd": statistics.fmean(
                    row["reserved_test_mean_deployment_cost_usd"]
                    for row in method_rows
                ),
            }
    payload = {
        "schema_version": "static-baseline-comparison-v4",
        "panel": str(args.panel),
        "protocol": (
            "development_search_reserved_test"
            if reserved_panel is not None
            else "single_panel"
        ),
        "evaluation_panel": (
            str(args.evaluation_panel) if args.evaluation_panel is not None else None
        ),
        "seed_start": args.seed,
        "seed_count": args.seeds,
        "configuration_count": len(panel.configuration_ids),
        "configuration_snapshots": panel.configuration_snapshots,
        "development_records_per_configuration": panel.records_per_configuration,
        "reserved_test_records_per_configuration": (
            reserved_panel.records_per_configuration
            if reserved_panel is not None
            else None
        ),
        "reserved_test_used_for_search": (
            False if reserved_panel is not None else None
        ),
        "results": [result.as_dict() for result in results],
        "matched_budget_curve": matched_budget_curve,
        "matched_budget_curve_csv": str(curve_csv),
        "held_out_evaluations": held_out_evaluations,
        "held_out_aggregates": aggregates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    curve_csv.parent.mkdir(parents=True, exist_ok=True)
    with curve_csv.open("w", encoding="utf-8", newline="") as stream:
        fieldnames = (
            "method",
            "budget_index",
            "budget_usd",
            "mean_completed_run_evaluations",
            "mean_recommended_panel_accuracy",
            "mean_simple_regret",
            "optimal_selection_fraction",
            "seed_count",
        )
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for method, points in matched_budget_curve["series"].items():
            for point in points:
                writer.writerow({"method": method, **point})
    print(
        f"runs={len(results)} methods={len(args.methods)} seeds={args.seeds} "
        f"configurations={len(panel.configuration_ids)} "
        f"output={args.output} curve_csv={curve_csv}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
