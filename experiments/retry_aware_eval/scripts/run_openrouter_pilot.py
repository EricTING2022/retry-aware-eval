#!/usr/bin/env python3
"""Preview, execute, or resume an OpenRouter workflow collection."""

from __future__ import annotations

import argparse
from contextlib import closing
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from retry_eval.benchmark import load_mathqa_benchmark  # noqa: E402
from retry_eval.client import OpenRouterClient  # noqa: E402
from retry_eval.log_export import export_detailed_log  # noqa: E402
from retry_eval.pilot import (  # noqa: E402
    PILOT_PRESETS,
    configurations_for_preset,
    plan_pilot,
    run_pilot,
)
from retry_eval.profiles import QIAN_MODEL_PROFILES  # noqa: E402
from retry_eval.prompts import SOLVER_PROMPT_TEMPLATE, VERIFIER_PROMPT  # noqa: E402
from retry_eval.storage import ExperimentStore  # noqa: E402


def file_hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def git_revision() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    revision = completed.stdout.strip()
    return revision or None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=PILOT_PRESETS, default="smoke")
    parser.add_argument(
        "--partition",
        choices=("all", "development", "reserved_test"),
        default="all",
    )
    parser.add_argument(
        "--configuration-ids-file",
        type=Path,
        help="JSON file containing a configuration_ids list to collect.",
    )
    parser.add_argument(
        "--benchmark", type=Path, default=PROJECT_ROOT / "data/mathqa/pilot_20.jsonl"
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=PROJECT_ROOT / "artifacts/openrouter_pilot.sqlite3",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=PROJECT_ROOT / "artifacts/openrouter_pilot_summary.json",
    )
    parser.add_argument(
        "--detailed-log",
        type=Path,
        help="JSONL audit log; defaults beside --summary with _detailed.jsonl.",
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--resume-run-id",
        help="Resume this run in the existing --database, skipping completed workflows.",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=float, default=90.0)
    args = parser.parse_args()

    items = load_mathqa_benchmark(args.benchmark)
    if args.partition != "all":
        items = [item for item in items if item.partition == args.partition]
    configuration_ids = None
    if args.configuration_ids_file is not None:
        selected = json.loads(args.configuration_ids_file.read_text(encoding="utf-8"))
        raw_ids = selected.get("configuration_ids") if isinstance(selected, dict) else None
        if not isinstance(raw_ids, list) or not all(
            isinstance(value, str) for value in raw_ids
        ):
            parser.error(
                "--configuration-ids-file must contain a configuration_ids string list"
            )
        configuration_ids = set(raw_ids)
    try:
        plan = plan_pilot(args.preset, items, configuration_ids)
    except ValueError as error:
        parser.error(str(error))
    detailed_log = args.detailed_log or args.summary.with_name(
        f"{args.summary.stem.removesuffix('_summary')}_detailed.jsonl"
    )
    preview = {
        **plan.as_dict(),
        "benchmark": str(args.benchmark),
        "benchmark_sha256": file_hash(args.benchmark),
        "database": str(args.database),
        "summary": str(args.summary),
        "detailed_log": str(detailed_log),
        "mode": "execute" if args.execute else "preview",
        "resume_run_id": args.resume_run_id,
    }
    print(json.dumps(preview, indent=2, sort_keys=True))
    if not args.execute:
        print("preview_only=true; no API calls were made")
        return 0

    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    if args.workers <= 0:
        parser.error("--workers must be positive")
    if args.resume_run_id is None and args.database.exists():
        parser.error(f"database already exists; choose a new path: {args.database}")
    if args.resume_run_id is not None and not args.database.exists():
        parser.error("--resume-run-id requires an existing --database")
    if args.summary.exists():
        parser.error(f"summary already exists; choose a new path: {args.summary}")
    if detailed_log.exists():
        parser.error(f"detailed log already exists; choose a new path: {detailed_log}")
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key.startswith("sk-"):
        parser.error("OPENROUTER_API_KEY is missing or does not start with 'sk-'")

    client = OpenRouterClient(api_key, timeout_seconds=args.timeout_seconds)
    store = ExperimentStore(args.database)
    store.initialize()
    settings = {
        "schema_version": "openrouter-pilot-settings-v2",
        "preset": args.preset,
        "benchmark_path": str(args.benchmark.resolve()),
        "benchmark_sha256": preview["benchmark_sha256"],
        "question_ids": [item.question_id for item in items],
        "model_ids": list(plan.model_ids),
        "configuration_ids": [
            row.configuration_id
            for row in configurations_for_preset(args.preset, configuration_ids)
        ],
        "partition": args.partition,
        "model_profiles": {
            model: {
                "solver_parameters": dict(QIAN_MODEL_PROFILES[model].solver_parameters),
                "verifier_parameters": dict(
                    QIAN_MODEL_PROFILES[model].verifier_parameters
                ),
                "solver_prompt_suffix": QIAN_MODEL_PROFILES[model].solver_prompt_suffix,
            }
            for model in plan.model_ids
        },
        "prompt_templates": {
            "solver": SOLVER_PROMPT_TEMPLATE,
            "verifier": VERIFIER_PROMPT,
        },
        "code_revision": git_revision(),
        "provider_routing": "require_parameters",
        "local_call_or_cost_limit": False,
        "per_request_output_token_limit": False,
        "workers": args.workers,
        "response_cache": False,
    }
    if args.resume_run_id is not None:
        with closing(store.connect()) as connection:
            row = connection.execute(
                "SELECT settings_json FROM runs WHERE run_id=?",
                (args.resume_run_id,),
            ).fetchone()
        if row is None:
            parser.error(f"run does not exist: {args.resume_run_id}")
        frozen = json.loads(row["settings_json"])
        for field in (
            "preset",
            "benchmark_sha256",
            "question_ids",
            "model_ids",
            "configuration_ids",
            "partition",
        ):
            if frozen.get(field) != settings[field]:
                parser.error(f"resume settings differ from the original field: {field}")
        if (
            "model_profiles" in frozen
            and frozen["model_profiles"] != settings["model_profiles"]
        ):
            parser.error("resume model profiles differ from the original run")

    def progress(completed: int, total: int) -> None:
        progress_step = max(1, total // 100)
        if completed == total or completed % progress_step == 0:
            print(f"completed_workflows={completed}/{total}")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id, summary = run_pilot(
        store=store,
        client=client,
        items=items,
        preset=args.preset,
        run_name=f"openrouter-pilot-{args.preset}-{timestamp}",
        settings=settings,
        resume_run_id=args.resume_run_id,
        configuration_ids=configuration_ids,
        workers=args.workers,
        progress=progress,
    )
    summary["database"] = str(args.database)
    summary["detailed_log"] = str(detailed_log)
    export_detailed_log(args.database, detailed_log, run_id=run_id)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"run_id={run_id}")
    print(f"summary={args.summary}")
    print(f"detailed_log={detailed_log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
