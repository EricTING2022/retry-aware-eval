#!/usr/bin/env python3
"""Download the official MathQA archive and build a reproducible 200-item test pool."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import urllib.request
import zipfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from retry_eval.benchmark import parse_mathqa_options  # noqa: E402


OFFICIAL_ARCHIVE_URL = "https://math-qa.github.io/math-QA/data/MathQA.zip"
DEFAULT_SEED = 20261005
EXPECTED_TEST_COUNT = 2985
EXPECTED_ARCHIVE_SHA256 = "7344f30456a7aef3176d4866cc953b35b41bec44eda6b00cdbcfde2876b2f07a"
EXPECTED_TEST_SHA256 = "dfe7bc4691caf26842ccd4cf14e8f978327bab4f2989e0b076a4e6b38a9371d1"


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "retry-aware-eval/0.1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        content = response.read()
    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.write_bytes(content)
    temporary.replace(destination)


def prepare(
    *,
    archive_path: Path,
    output_path: Path,
    manifest_path: Path,
    count: int,
    seed: int,
    force_download: bool,
) -> dict[str, object]:
    if count <= 0:
        raise ValueError("count must be positive.")
    if force_download or not archive_path.is_file():
        _download(OFFICIAL_ARCHIVE_URL, archive_path)
    archive_bytes = archive_path.read_bytes()
    archive_sha256 = _sha256_bytes(archive_bytes)
    if archive_sha256 != EXPECTED_ARCHIVE_SHA256:
        raise ValueError(
            "Official archive hash changed; review the upstream change before updating "
            "EXPECTED_ARCHIVE_SHA256."
        )
    with zipfile.ZipFile(archive_path) as archive:
        candidates = [
            name
            for name in archive.namelist()
            if Path(name).name == "test.json" and not name.startswith("__MACOSX/")
        ]
        if len(candidates) != 1:
            raise ValueError(f"Expected one test.json in the archive, found {candidates!r}.")
        test_bytes = archive.read(candidates[0])
    test_sha256 = _sha256_bytes(test_bytes)
    if test_sha256 != EXPECTED_TEST_SHA256:
        raise ValueError(
            "Official test.json hash changed; review the upstream change before updating "
            "EXPECTED_TEST_SHA256."
        )
    source_rows = json.loads(test_bytes)
    if not isinstance(source_rows, list) or len(source_rows) != EXPECTED_TEST_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_TEST_COUNT} official test rows, found {len(source_rows)}."
        )
    if count > len(source_rows):
        raise ValueError("count exceeds the official test split size.")

    selected_indices = random.Random(seed).sample(range(len(source_rows)), count)
    development_count = count // 2
    output_rows: list[dict[str, object]] = []
    for benchmark_position, source_index in enumerate(selected_indices):
        source = source_rows[source_index]
        correct = str(source["correct"]).lower()
        options = parse_mathqa_options(str(source["options"]))
        if correct not in options:
            raise ValueError(f"Source row {source_index} has invalid correct option {correct!r}.")
        output_rows.append(
            {
                "question_id": f"mathqa-test-{source_index:04d}",
                "benchmark_position": benchmark_position,
                "source_index": source_index,
                "source_split": "test",
                "partition": (
                    "development"
                    if benchmark_position < development_count
                    else "reserved_test"
                ),
                "problem": str(source["Problem"]),
                "options": options,
                "correct_option": correct,
                "category": str(source.get("category", "")),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_content = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for row in output_rows
    ).encode("utf-8")
    output_path.write_bytes(output_content)
    manifest = {
        "schema_version": "mathqa-test-pool-v1",
        "official_archive_url": OFFICIAL_ARCHIVE_URL,
        "official_homepage": "https://math-qa.github.io/math-QA/",
        "source_archive_sha256": archive_sha256,
        "source_test_member": candidates[0],
        "source_test_sha256": test_sha256,
        "source_test_count": len(source_rows),
        "selection": {
            "algorithm": "python_random_v2_sample_without_replacement",
            "seed": seed,
            "count": count,
            "development_count": development_count,
            "reserved_test_count": count - development_count,
            "source_indices": selected_indices,
        },
        "output_file": output_path.name,
        "output_sha256": _sha256_bytes(output_content),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "mathqa" / "MathQA.zip",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "mathqa" / "test_200.jsonl",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "data" / "mathqa" / "manifest.json",
    )
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()
    manifest = prepare(
        archive_path=args.archive,
        output_path=args.output,
        manifest_path=args.manifest,
        count=args.count,
        seed=args.seed,
        force_download=args.force_download,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
