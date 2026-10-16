"""Alternative cost views over the same executed-call ledger."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Mapping

from .storage import ExperimentStore


@dataclass(frozen=True)
class TokenPrice:
    input_usd_per_million: Decimal
    output_usd_per_million: Decimal


@dataclass(frozen=True)
class CostSummary:
    executed_call_units: int
    model_fixed_cost_usd: Decimal
    token_based_cost_usd: Decimal
    reported_actual_cost_usd: Decimal
    calls_with_unknown_reported_cost: int


def summarize_execution_costs(
    store: ExperimentStore,
    execution_id: str,
    *,
    fixed_cost_per_call: Mapping[str, Decimal],
    token_prices: Mapping[str, TokenPrice],
) -> CostSummary:
    """Compute three requested accounting schemes plus provider-reported cost.

    Fixed costs may use a role-specific ``role:model`` key, falling back to the
    model id. Token prices must be a frozen run-time snapshot.
    """

    calls = store.executed_calls(execution_id)
    fixed_total = Decimal(0)
    token_total = Decimal(0)
    actual_total = Decimal(0)
    unknown_actual = 0
    for call in calls:
        model = call["model"]
        role_key = f"{call['role']}:{model}"
        if role_key in fixed_cost_per_call:
            fixed_total += Decimal(fixed_cost_per_call[role_key])
        elif model in fixed_cost_per_call:
            fixed_total += Decimal(fixed_cost_per_call[model])
        else:
            raise ValueError(f"Missing fixed call cost for {role_key}.")

        if model not in token_prices:
            raise ValueError(f"Missing token price for {model}.")
        price = token_prices[model]
        token_total += (
            Decimal(call["input_tokens"]) * price.input_usd_per_million
            + Decimal(call["output_tokens"]) * price.output_usd_per_million
        ) / Decimal(1_000_000)

        if call["reported_cost_usd"] is None:
            unknown_actual += 1
        else:
            actual_total += Decimal(str(call["reported_cost_usd"]))

    return CostSummary(
        executed_call_units=len(calls),
        model_fixed_cost_usd=fixed_total,
        token_based_cost_usd=token_total,
        reported_actual_cost_usd=actual_total,
        calls_with_unknown_reported_cost=unknown_actual,
    )

