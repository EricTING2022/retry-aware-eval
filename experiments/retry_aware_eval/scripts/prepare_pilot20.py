#!/usr/bin/env python3
"""Freeze the first 10 questions from each partition of the 200-item pool."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from retry_eval.benchmark import load_mathqa_benchmark  # noqa: E402


def file_hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare(parent: Path, output: Path, manifest_path: Path) -> None:
    items = load_mathqa_benchmark(parent)
    selected = []
    for partition in ("development", "reserved_test"):
        partition_items = [item for item in items if item.partition == partition]
        if len(partition_items) < 10:
            raise ValueError(f"Partition {partition!r} has fewer than 10 questions.")
        selected.extend(partition_items[:10])

    raw_by_id = {}
    with parent.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                raw_by_id[record["question_id"]] = record
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(
            json.dumps(raw_by_id[item.question_id], sort_keys=True, separators=(",", ":"))
            + "\n"
            for item in selected
        ),
        encoding="utf-8",
        newline="\n",
    )
    manifest = {
        "schema_version": "mathqa-pilot-20-v1",
        "parent_file": parent.name,
        "parent_sha256": file_hash(parent),
        "output_file": output.name,
        "output_sha256": file_hash(output),
        "selection": {
            "algorithm": "first_10_per_partition_in_parent_order",
            "development_count": 10,
            "reserved_test_count": 10,
            "question_ids": [item.question_id for item in selected],
            "source_indices": [item.source_index for item in selected],
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parent", type=Path, default=PROJECT_ROOT / "data/mathqa/test_200.jsonl"
    )
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "data/mathqa/pilot_20.jsonl"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "data/mathqa/pilot_20_manifest.json",
    )
    args = parser.parse_args()
    prepare(args.parent, args.output, args.manifest)
    print(f"questions=20 output={args.output} manifest={args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
