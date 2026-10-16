#!/usr/bin/env python3
"""Write an inspectable four-baseline trace without network or model calls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from retry_eval.static_baselines import (  # noqa: E402
    BalancedPanel,
    PanelRecord,
    run_all_static_baselines,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "static_baselines_demo.json",
    )
    args = parser.parse_args()
    rewards = {
        "fixed-sequence-a": [1, 1, 1, 1, 0, 1],
        "fixed-sequence-b": [1, 0, 1, 0, 1, 0],
        "fixed-sequence-c": [0, 0, 1, 0, 0, 0],
    }
    costs = {
        "fixed-sequence-a": 0.03,
        "fixed-sequence-b": 0.02,
        "fixed-sequence-c": 0.01,
    }
    records = [
        PanelRecord(
            record_id=f"{configuration}-{question}",
            configuration_id=configuration,
            question_id=f"q-{question}",
            reward=float(reward),
            realized_cost_usd=costs[configuration],
        )
        for configuration, outcomes in rewards.items()
        for question, reward in enumerate(outcomes)
    ]
    results = run_all_static_baselines(
        BalancedPanel(records),
        max_purchases=12,
        seed=7,
        gittins_grid_points=257,
    )
    payload = {
        "schema_version": "static-baseline-demo-v1",
        "network_calls": 0,
        "methods": [result.as_dict() for result in results],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"methods={len(results)} network_calls=0 output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
