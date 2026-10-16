from __future__ import annotations

from contextlib import closing
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
import unittest
from uuid import uuid4

from retry_eval.benchmark import BenchmarkItem, load_mathqa_benchmark
from retry_eval.domain import CompletionResponse
from retry_eval.log_export import export_detailed_log
from retry_eval.pilot import configurations_for_preset, plan_pilot, run_pilot
from retry_eval.storage import ExperimentStore


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def response(content: str, cost: float = 0.001) -> CompletionResponse:
    usage = {
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "completion_tokens_details": {"reasoning_tokens": 5},
        "cost": cost,
    }
    raw = {
        "id": "fake-response-id",
        "choices": [{"finish_reason": "stop", "message": {"content": content}}],
        "usage": usage,
    }
    return CompletionResponse(
        content=content,
        input_tokens=100,
        output_tokens=20,
        cost_usd=cost,
        raw_response=raw,
        returned_model="returned-model",
        provider="fake-provider",
        response_id="fake-response-id",
        http_status=200,
        finish_reason="stop",
        reasoning_tokens=5,
        usage=usage,
        response_body_text=json.dumps(raw),
    )


class ScriptedClient:
    def __init__(self, responses):
        self.responses = list(responses)

    def complete(self, request):
        if not self.responses:
            raise AssertionError("Unexpected model call.")
        return self.responses.pop(0)


class RoleClient:
    def complete(self, request):
        if request.messages[0]["role"] == "system":
            return response('{"accepted":true}')
        return response('{"calculation":"6*7=42","option":"b","value":"42"}')


class FailingClient:
    def __init__(self):
        self.calls = 0

    def complete(self, request):
        self.calls += 1
        raise RuntimeError("provider contract failure")


def item(question_id: str, partition: str) -> BenchmarkItem:
    return BenchmarkItem(
        question_id=question_id,
        source_index=1 if partition == "development" else 2,
        partition=partition,
        problem="What is 6 times 7?",
        options={"a": "40", "b": "42", "c": "44", "d": "46", "e": "48"},
        correct_option="b",
        category="general",
    )


class PilotTests(unittest.TestCase):
    def test_pilot_cli_preview_reports_detailed_log_without_writing(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "scripts/run_openrouter_pilot.py",
                "--preset",
                "smoke",
                "--benchmark",
                "data/mathqa/pilot_20.jsonl",
                "--database",
                "artifacts/unused-preview.sqlite3",
                "--summary",
                "artifacts/unused-preview_summary.json",
            ],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn(
            '"detailed_log": "artifacts\\\\unused-preview_detailed.jsonl"',
            completed.stdout,
        )
        self.assertIn("preview_only=true", completed.stdout)
        self.assertFalse((PROJECT_ROOT / "artifacts/unused-preview.sqlite3").exists())

    def test_frozen_pilot_has_ten_questions_per_partition(self) -> None:
        pilot_path = PROJECT_ROOT / "data/mathqa/pilot_20.jsonl"
        manifest = json.loads(
            (PROJECT_ROOT / "data/mathqa/pilot_20_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        items = load_mathqa_benchmark(pilot_path)
        self.assertEqual(len(items), 20)
        self.assertEqual(sum(row.partition == "development" for row in items), 10)
        self.assertEqual(sum(row.partition == "reserved_test" for row in items), 10)
        self.assertEqual(len({row.question_id for row in items}), 20)
        self.assertEqual(
            sha256(pilot_path.read_bytes()).hexdigest(), manifest["output_sha256"]
        )
        self.assertEqual(
            [row.question_id for row in items], manifest["selection"]["question_ids"]
        )

    def test_preview_counts_are_exact(self) -> None:
        items = load_mathqa_benchmark(PROJECT_ROOT / "data/mathqa/pilot_20.jsonl")
        smoke = plan_pilot("smoke", items)
        full = plan_pilot("qian-27", items)
        smoke_configuration = configurations_for_preset("smoke")[0]
        self.assertEqual(
            (smoke.configuration_count, smoke.workflow_count), (1, 20)
        )
        self.assertEqual(
            smoke_configuration.solver_models,
            ("openai/gpt-oss-20b", "mistralai/ministral-8b-2512"),
        )
        self.assertEqual(
            smoke_configuration.verifier_models, ("qwen/qwen3-32b",)
        )
        self.assertEqual((smoke.minimum_calls, smoke.maximum_calls), (40, 60))
        self.assertEqual((full.configuration_count, full.workflow_count), (27, 540))
        self.assertEqual((full.minimum_calls, full.maximum_calls), (1080, 1620))

        selected_id = configurations_for_preset("qian-27")[4].configuration_id
        selected = plan_pilot("qian-27", items, {selected_id})
        self.assertEqual(selected.configuration_count, 1)
        self.assertEqual(selected.workflow_count, 20)

    def test_fake_client_runs_workflow_ledger_and_grading_end_to_end(self) -> None:
        database = Path(__file__).parent / f"pilot-{uuid4().hex}.sqlite3"
        try:
            store = ExperimentStore(database)
            store.initialize()
            correct = '{"calculation":"6*7=42","option":"b","value":"42"}'
            wrong = '{"calculation":"6*7=40","option":"a","value":"40"}'
            client = ScriptedClient(
                [
                    response(correct),
                    response('{"accepted":true}'),
                    response(wrong),
                    response('{"accepted":false}'),
                    response(correct),
                ]
            )
            run_id, summary = run_pilot(
                store=store,
                client=client,
                items=[item("dev", "development"), item("test", "reserved_test")],
                preset="smoke",
                run_name="fake-pilot",
                settings={"test": True},
            )
            self.assertTrue(run_id)
            self.assertEqual(summary["status"], "completed")
            self.assertEqual(summary["execution_counts"], {"completed": 2})
            self.assertEqual(summary["executed_call_count"], 5)
            self.assertEqual(summary["reasoning_tokens"], 25)
            self.assertGreater(summary["workflow_elapsed_seconds"], 0)
            self.assertIn(
                "verifier:qwen/qwen3-32b", summary["usage_by_role_and_model"]
            )
            self.assertEqual(summary["stop_reason_counts"], {
                "verifier_accepted": 1,
                "retry_limit_reached": 1,
            })
            self.assertEqual(summary["grade_counts"]["development"], {
                "graded": 1,
                "correct": 1,
            })
            self.assertEqual(summary["grade_counts"]["reserved_test"], {
                "graded": 1,
                "correct": 1,
            })
        finally:
            database.unlink(missing_ok=True)

    def test_resume_skips_completed_workflows_and_archives_failed_attempt(self) -> None:
        database = Path(__file__).parent / f"pilot-resume-{uuid4().hex}.sqlite3"
        try:
            store = ExperimentStore(database)
            store.initialize()
            correct = '{"calculation":"6*7=42","option":"b","value":"42"}'
            with self.assertRaises(AssertionError):
                run_pilot(
                    store=store,
                    client=ScriptedClient(
                        [response(correct), response('{"accepted":true}')]
                    ),
                    items=[item("dev", "development"), item("test", "reserved_test")],
                    preset="smoke",
                    run_name="interrupted",
                    settings={"test": True},
                )
            with closing(store.connect()) as connection:
                run_id = connection.execute("SELECT run_id FROM runs").fetchone()[0]

            resumed_id, summary = run_pilot(
                store=store,
                client=ScriptedClient(
                    [response(correct), response('{"accepted":true}')]
                ),
                items=[item("dev", "development"), item("test", "reserved_test")],
                preset="smoke",
                run_name="ignored-on-resume",
                settings={"test": True},
                resume_run_id=run_id,
            )

            self.assertEqual(resumed_id, run_id)
            self.assertEqual(summary["execution_counts"], {"completed": 2})
            self.assertEqual(summary["archived_technical_retry_count"], 1)
            self.assertEqual(summary["executed_call_count"], 4)
        finally:
            database.unlink(missing_ok=True)

    def test_parallel_collection_keeps_a_complete_balanced_ledger(self) -> None:
        database = Path(__file__).parent / f"pilot-workers-{uuid4().hex}.sqlite3"
        try:
            store = ExperimentStore(database)
            store.initialize()
            _, summary = run_pilot(
                store=store,
                client=RoleClient(),
                items=[item("dev", "development"), item("test", "reserved_test")],
                preset="smoke",
                run_name="parallel",
                settings={"test": True},
                workers=2,
            )
            self.assertEqual(summary["execution_counts"], {"completed": 2})
            self.assertEqual(summary["executed_call_count"], 4)
        finally:
            database.unlink(missing_ok=True)

    def test_parallel_collection_does_not_queue_the_whole_panel_after_failure(self) -> None:
        database = Path(__file__).parent / f"pilot-fail-fast-{uuid4().hex}.sqlite3"
        try:
            store = ExperimentStore(database)
            store.initialize()
            client = FailingClient()
            items = [item(f"dev-{index}", "development") for index in range(20)]

            with self.assertRaisesRegex(RuntimeError, "provider contract failure"):
                run_pilot(
                    store=store,
                    client=client,
                    items=items,
                    preset="smoke",
                    run_name="fail-fast",
                    settings={"test": True},
                    workers=4,
                )

            self.assertLessEqual(client.calls, 4)
            with closing(store.connect()) as connection:
                execution_count = connection.execute(
                    "SELECT COUNT(*) FROM executions"
                ).fetchone()[0]
            self.assertLessEqual(execution_count, 4)
        finally:
            for path in (database, Path(f"{database}-wal"), Path(f"{database}-shm")):
                path.unlink(missing_ok=True)

    def test_detailed_log_contains_each_solver_and_verifier_measurement(self) -> None:
        database = Path(__file__).parent / f"pilot-log-{uuid4().hex}.sqlite3"
        output = Path(__file__).parent / f"pilot-log-{uuid4().hex}.jsonl"
        try:
            store = ExperimentStore(database)
            store.initialize()
            correct = '{"calculation":"6*7=42","option":"b","value":"42"}'
            run_id, _ = run_pilot(
                store=store,
                client=ScriptedClient(
                    [response(correct), response('{"accepted":true}')]
                ),
                items=[item("dev", "development")],
                preset="smoke",
                run_name="detailed-log",
                settings={"test": True},
            )

            result = export_detailed_log(database, output, run_id=run_id)
            records = [json.loads(line) for line in output.read_text().splitlines()]
            workflow = next(row for row in records if row["record_type"] == "workflow")
            attempt = workflow["attempts"][0]
            solver, verifier = attempt["calls"]

            self.assertEqual(result["workflow_count"], 1)
            self.assertEqual(workflow["call_totals"]["calls"], 2)
            self.assertEqual(workflow["call_totals"]["reasoning_tokens"], 10)
            self.assertGreater(workflow["execution"]["elapsed_seconds"], 0)
            self.assertEqual(attempt["parsed_answer"]["option"], "b")
            self.assertEqual(attempt["solver_option_correct"], 1)
            self.assertEqual(attempt["verifier_accepted"], 1)
            for call in (solver, verifier):
                self.assertEqual(call["status"], "executed")
                self.assertIsNotNone(call["started_at_utc"])
                self.assertIsNotNone(call["finished_at_utc"])
                self.assertEqual(call["response_id"], "fake-response-id")
                self.assertEqual(call["http_status"], 200)
                self.assertEqual(call["finish_reason"], "stop")
                self.assertEqual(call["reasoning_tokens"], 5)
                self.assertEqual(call["usage"]["prompt_tokens"], 100)
                self.assertIn("messages", call["request"])
                self.assertIn("choices", call["response"])
        finally:
            for path in (database, Path(f"{database}-wal"), Path(f"{database}-shm")):
                path.unlink(missing_ok=True)
            output.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
