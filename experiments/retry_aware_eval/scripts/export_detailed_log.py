#!/usr/bin/env python3
"""Export complete per-workflow solver/verifier records from SQLite to JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from retry_eval.log_export import export_detailed_log  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists; choose a new path: {args.output}")
    try:
        result = export_detailed_log(
            args.database, args.output, run_id=args.run_id
        )
    except (OSError, ValueError, sqlite3.Error) as error:
        parser.exit(1, f"Detailed log export failed: {error}\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
