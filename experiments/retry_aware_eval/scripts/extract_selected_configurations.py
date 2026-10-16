#!/usr/bin/env python3
"""Extract the union of final recommendations from a baseline search result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baselines", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.baselines.read_text(encoding="utf-8"))
    results = payload.get("results")
    if not isinstance(results, list) or not results:
        parser.error("baseline file has no results")
    selected: set[str] = set()
    for result in results:
        checkpoints = result.get("checkpoints") if isinstance(result, dict) else None
        if not isinstance(checkpoints, list) or not checkpoints:
            parser.error("every baseline result must contain checkpoints")
        configuration_id = checkpoints[-1].get("recommended_configuration_id")
        if not isinstance(configuration_id, str):
            parser.error("final checkpoint is missing recommended_configuration_id")
        selected.add(configuration_id)

    output = {
        "schema_version": "selected-configurations-v1",
        "source": str(args.baselines),
        "configuration_ids": sorted(selected),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"selected_configurations={len(selected)} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
