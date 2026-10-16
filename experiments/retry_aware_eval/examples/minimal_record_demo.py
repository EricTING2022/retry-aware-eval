"""Create one offline, five-call Qian-style record with scripted responses."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from decimal import Decimal
import json
from pathlib import Path

from retry_eval.configuration import WorkflowConfiguration
from retry_eval.costs import TokenPrice, summarize_execution_costs
from retry_eval.domain import CompletionResponse, Question
from retry_eval.grading import grade_execution
from retry_eval.profiles import (
    GPT_OSS_20B,
    MINISTRAL_3_8B,
    QIAN_MODEL_IDS,
    QIAN_MODEL_PROFILES,
    QWEN3,
)
from retry_eval.storage import ExperimentStore
from retry_eval.workflow import WorkflowRunner


def response(content: str, cost: float) -> CompletionResponse:
    return CompletionResponse(
        content=content,
        input_tokens=100,
        output_tokens=20,
        cost_usd=cost,
        raw_response={
            "choices": [{"message": {"content": content}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20, "cost": cost},
        },
        returned_model="offline-scripted-model",
        provider="offline-scripted-provider",
    )


class ScriptedClient:
    def __init__(self, responses):
        self.responses = iter(responses)

    def complete(self, request):
        return next(self.responses)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.database.exists() or args.output.exists():
        parser.error("Choose new output paths; the demo does not overwrite records.")

    wrong = '{"calculation":"6*7=40","option":"a","value":"40"}'
    correct = '{"calculation":"6*7=42","option":"b","value":"42"}'
    client = ScriptedClient(
        [
            response(wrong, 0.0010),
            response('{"accepted":false}', 0.0004),
            response(wrong, 0.0011),
            response('{"accepted":false}', 0.0005),
            response(correct, 0.0012),
        ]
    )
    store = ExperimentStore(args.database)
    store.initialize()
    run_id = store.create_run(
        "offline-minimal-record",
        max_retries=2,
        settings={"network_calls": 0, "purpose": "record-shape demonstration"},
    )
    configuration = WorkflowConfiguration(
        solver_models=(GPT_OSS_20B, MINISTRAL_3_8B, QWEN3),
        verifier_models=(MINISTRAL_3_8B, QWEN3),
        max_retries=2,
    )
    question = Question(
        "mathqa_test_demo",
        "What is 6 times 7?",
        "a) 40, b) 42, c) 44, d) 46, e) 48",
    )
    outcome = WorkflowRunner(client, store, QIAN_MODEL_PROFILES).run_question(
        run_id, configuration, question
    )
    grade_execution(store, outcome.execution_id, correct_option="b")
    fixed = {model: Decimal("0.01") for model in QIAN_MODEL_IDS}
    prices = {
        model: TokenPrice(Decimal("1"), Decimal("2")) for model in QIAN_MODEL_IDS
    }
    costs = summarize_execution_costs(
        store,
        outcome.execution_id,
        fixed_cost_per_call=fixed,
        token_prices=prices,
    )
    record = store.execution_record(outcome.execution_id)
    record["cost_summary"] = {
        key: str(value) if isinstance(value, Decimal) else value
        for key, value in asdict(costs).items()
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"database": str(args.database), "record": str(args.output), "network_calls": 0}))


if __name__ == "__main__":
    main()

