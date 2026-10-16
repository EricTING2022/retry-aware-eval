from __future__ import annotations

from contextlib import closing
from decimal import Decimal
import json
from pathlib import Path
import unittest
from uuid import uuid4

from retry_eval.configuration import WorkflowConfiguration, generate_configurations
from retry_eval.costs import TokenPrice, summarize_execution_costs
from retry_eval.client import OpenRouterResponseError
from retry_eval.domain import CompletionResponse, Question
from retry_eval.grading import grade_execution
from retry_eval.profiles import (
    GEMINI_FIXED_CRITIC,
    GPT_OSS_20B,
    MINISTRAL_3_8B,
    QIAN_MODEL_IDS,
    QIAN_MODEL_PROFILES,
    QWEN3,
)
from retry_eval.storage import ExperimentStore
from retry_eval.workflow import WorkflowRunner


def response(content: str, *, cost: float = 0.001) -> CompletionResponse:
    return CompletionResponse(
        content=content,
        input_tokens=100,
        output_tokens=20,
        cost_usd=cost,
        raw_response={
            "choices": [{"message": {"content": content}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20, "cost": cost},
        },
        returned_model="returned-model",
        provider="test-provider",
    )


class ScriptedClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def complete(self, request):
        self.requests.append(request)
        if not self.responses:
            raise AssertionError("Unexpected model call.")
        return self.responses.pop(0)


class RetryAwareExperimentTests(unittest.TestCase):
    def setUp(self):
        self.database_path = Path(__file__).parent / f"test-{uuid4().hex}.sqlite3"
        self.store = ExperimentStore(self.database_path)
        self.store.initialize()
        self.question = Question(
            question_id="mathqa_test_demo",
            problem="What is 6 times 7?",
            options="a) 40, b) 42, c) 44, d) 46, e) 48",
        )

    def tearDown(self):
        self.database_path.unlink(missing_ok=True)

    def test_configuration_counts_match_qian_setup(self):
        self.assertEqual(
            QIAN_MODEL_IDS,
            (
                "openai/gpt-oss-20b",
                "mistralai/ministral-8b-2512",
                "qwen/qwen3-32b",
            ),
        )
        one_retry = list(
            generate_configurations(QIAN_MODEL_IDS, QIAN_MODEL_IDS, max_retries=1)
        )
        two_retries = list(
            generate_configurations(QIAN_MODEL_IDS, QIAN_MODEL_IDS, max_retries=2)
        )
        fixed_critic_reference = list(
            generate_configurations(
                QIAN_MODEL_IDS, (GEMINI_FIXED_CRITIC,), max_retries=2
            )
        )

        self.assertEqual(len(one_retry), 27)
        self.assertEqual(len(two_retries), 243)
        self.assertEqual(len(fixed_critic_reference), 27)
        self.assertEqual(len({item.configuration_id for item in two_retries}), 243)
        self.assertEqual(
            len(list(generate_configurations(QIAN_MODEL_IDS, (), max_retries=0))),
            3,
        )

    def test_early_accept_stops_current_question_and_marks_skipped_slots(self):
        client = ScriptedClient(
            [
                response('{"calculation":"6*7=40","option":"a","value":"40"}'),
                response('{"accepted":true}'),
            ]
        )
        configuration = WorkflowConfiguration(
            solver_models=(GPT_OSS_20B, MINISTRAL_3_8B, QWEN3),
            verifier_models=(QWEN3, GPT_OSS_20B),
            max_retries=2,
        )
        run_id = self.store.create_run("early-stop", max_retries=2)
        outcome = WorkflowRunner(
            client, self.store, QIAN_MODEL_PROFILES
        ).run_question(run_id, configuration, self.question)
        grade = grade_execution(self.store, outcome.execution_id, correct_option="b")
        record = self.store.execution_record(outcome.execution_id)

        self.assertEqual(outcome.stop_reason, "verifier_accepted")
        self.assertEqual(outcome.returned_attempt, 1)
        self.assertFalse(grade.is_correct)
        self.assertEqual(len(client.requests), 2)
        self.assertEqual(
            [call["status"] for call in record["calls"]],
            [
                "executed",
                "executed",
                "skipped_early_stop",
                "skipped_early_stop",
                "skipped_early_stop",
                "not_applicable_final",
            ],
        )
        self.assertEqual(record["calls"][1]["accepted"], 1)

    def test_retry_limit_returns_unverified_final_answer_and_records_costs(self):
        wrong = '{"calculation":"6*7=40","option":"a","value":"40"}'
        correct = '{"calculation":"6*7=42","option":"b","value":"42"}'
        client = ScriptedClient(
            [
                response(wrong),
                response('{"accepted":false}'),
                response(wrong),
                response('{"accepted":false}'),
                response(correct),
            ]
        )
        configuration = WorkflowConfiguration(
            solver_models=(GPT_OSS_20B, MINISTRAL_3_8B, QWEN3),
            verifier_models=(MINISTRAL_3_8B, QWEN3),
            max_retries=2,
        )
        run_id = self.store.create_run("retry-limit", max_retries=2)
        outcome = WorkflowRunner(
            client, self.store, QIAN_MODEL_PROFILES
        ).run_question(run_id, configuration, self.question)
        grade = grade_execution(self.store, outcome.execution_id, correct_option="b")
        record = self.store.execution_record(outcome.execution_id)
        calls = record["calls"]

        self.assertEqual(outcome.stop_reason, "retry_limit_reached")
        self.assertEqual(outcome.returned_attempt, 3)
        self.assertTrue(grade.is_correct)
        self.assertEqual(len(client.requests), 5)
        self.assertEqual(sum(call["status"] == "executed" for call in calls), 5)
        final_verifier = next(
            call
            for call in calls
            if call["attempt_index"] == 3 and call["role"] == "verifier"
        )
        self.assertEqual(final_verifier["status"], "not_applicable_final")
        self.assertIsNone(final_verifier["accepted"])
        self.assertEqual(
            [
                call["accepted"]
                for call in calls
                if call["role"] == "verifier" and call["status"] == "executed"
            ],
            [0, 0],
        )

        fixed = {model: Decimal("0.01") for model in QIAN_MODEL_IDS}
        prices = {
            model: TokenPrice(Decimal("1"), Decimal("2"))
            for model in QIAN_MODEL_IDS
        }
        costs = summarize_execution_costs(
            self.store,
            outcome.execution_id,
            fixed_cost_per_call=fixed,
            token_prices=prices,
        )
        self.assertEqual(costs.executed_call_units, 5)
        self.assertEqual(costs.model_fixed_cost_usd, Decimal("0.05"))
        self.assertEqual(costs.token_based_cost_usd, Decimal("0.0007"))
        self.assertEqual(costs.reported_actual_cost_usd, Decimal("0.005"))
        self.assertEqual(costs.calls_with_unknown_reported_cost, 0)

    def test_zero_retries_makes_one_solver_call_and_qwen3_is_non_thinking(self):
        correct = '{"calculation":"6*7=42","option":"b","value":"42"}'
        client = ScriptedClient([response(correct)])
        configuration = WorkflowConfiguration(
            solver_models=(QWEN3,), verifier_models=(), max_retries=0
        )
        run_id = self.store.create_run("zero-retry", max_retries=0)
        outcome = WorkflowRunner(
            client, self.store, QIAN_MODEL_PROFILES
        ).run_question(run_id, configuration, self.question)

        self.assertEqual(outcome.returned_attempt, 1)
        self.assertEqual(len(client.requests), 1)
        request = client.requests[0]
        self.assertEqual(request.parameters["reasoning"], {"enabled": False})
        self.assertTrue(request.messages[0]["content"].endswith("/no_think"))

    def test_solver_reasoning_controls_match_openrouter_capabilities(self):
        gpt_oss = QIAN_MODEL_PROFILES[GPT_OSS_20B]
        ministral = QIAN_MODEL_PROFILES[MINISTRAL_3_8B]
        qwen3 = QIAN_MODEL_PROFILES[QWEN3]

        self.assertEqual(gpt_oss.solver_parameters["reasoning"], {"effort": "low"})
        self.assertEqual(gpt_oss.verifier_parameters["reasoning"], {"effort": "low"})
        self.assertNotIn("reasoning", ministral.solver_parameters)
        self.assertEqual(qwen3.solver_parameters["reasoning"], {"enabled": False})
        self.assertEqual(qwen3.verifier_parameters["reasoning"], {"enabled": False})
        self.assertEqual(qwen3.solver_prompt_suffix, "\n/no_think")
        for profile in (gpt_oss, ministral, qwen3):
            for parameters in (
                profile.solver_parameters,
                profile.verifier_parameters,
            ):
                self.assertNotIn("max_tokens", parameters)
                self.assertTrue(parameters["provider"]["require_parameters"])
                self.assertEqual(parameters["usage"], {"include": True})
                self.assertEqual(parameters["response_format"]["type"], "json_schema")
        self.assertEqual(
            gpt_oss.verifier_parameters["provider"]["ignore"], ["darkbloom"]
        )
        self.assertEqual(
            gpt_oss.solver_parameters["provider"]["ignore"], ["darkbloom"]
        )
        for parameters in (
            ministral.solver_parameters,
            ministral.verifier_parameters,
            qwen3.solver_parameters,
            qwen3.verifier_parameters,
        ):
            self.assertNotIn("ignore", parameters["provider"])

    def test_unusable_api_response_is_fully_recorded_before_failure(self):
        failed_response = CompletionResponse(
            content="",
            input_tokens=123,
            output_tokens=512,
            cost_usd=0.0042,
            raw_response={
                "id": "generation-id",
                "choices": [{"finish_reason": "length", "message": {"content": None}}],
                "usage": {
                    "prompt_tokens": 123,
                    "completion_tokens": 512,
                    "completion_tokens_details": {"reasoning_tokens": 500},
                    "cost": 0.0042,
                },
            },
            returned_model=GPT_OSS_20B,
            provider="test-provider",
            response_id="generation-id",
            http_status=200,
            finish_reason="length",
            reasoning_tokens=500,
            usage={
                "prompt_tokens": 123,
                "completion_tokens": 512,
                "completion_tokens_details": {"reasoning_tokens": 500},
                "cost": 0.0042,
            },
        )

        class FailingClient:
            def complete(self, request):
                raise OpenRouterResponseError(
                    "OpenRouter response has no usable text content. finish_reason='length'.",
                    failed_response,
                )

        configuration = WorkflowConfiguration(
            solver_models=(GPT_OSS_20B,), verifier_models=(), max_retries=0
        )
        run_id = self.store.create_run("response-failure", max_retries=0)
        with self.assertRaises(OpenRouterResponseError):
            WorkflowRunner(
                FailingClient(), self.store, QIAN_MODEL_PROFILES
            ).run_question(run_id, configuration, self.question)

        with closing(self.store.connect()) as connection:
            call = connection.execute(
                "SELECT status,response_json,provider,input_tokens,output_tokens,"
                "reasoning_tokens,finish_reason,response_id,http_status,usage_json,"
                "started_at_utc,finished_at_utc,reported_cost_usd,error_type,error "
                "FROM call_slots WHERE role='solver'"
            ).fetchone()
        self.assertEqual(call["status"], "failed")
        self.assertEqual(json.loads(call["response_json"])["id"], "generation-id")
        self.assertEqual(call["provider"], "test-provider")
        self.assertEqual((call["input_tokens"], call["output_tokens"]), (123, 512))
        self.assertEqual(call["reported_cost_usd"], 0.0042)
        self.assertEqual(call["reasoning_tokens"], 500)
        self.assertEqual(call["finish_reason"], "length")
        self.assertEqual(call["response_id"], "generation-id")
        self.assertEqual(call["http_status"], 200)
        self.assertEqual(json.loads(call["usage_json"])["prompt_tokens"], 123)
        self.assertIsNotNone(call["started_at_utc"])
        self.assertIsNotNone(call["finished_at_utc"])
        self.assertEqual(call["error_type"], "OpenRouterResponseError")
        self.assertIn("finish_reason='length'", call["error"])


if __name__ == "__main__":
    unittest.main()

