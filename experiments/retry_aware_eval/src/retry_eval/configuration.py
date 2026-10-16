"""Configuration generation for arbitrary retry counts."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from itertools import product
import json
from typing import Iterable, Iterator, Sequence


def _validated_models(models: Iterable[str], label: str) -> tuple[str, ...]:
    values = tuple(models)
    if not values or any(not isinstance(model, str) or not model for model in values):
        raise ValueError(f"{label} must contain nonempty model ids.")
    if len(set(values)) != len(values):
        raise ValueError(f"{label} must not contain duplicates.")
    return values


@dataclass(frozen=True)
class WorkflowConfiguration:
    """One ordered model assignment for a retry-aware workflow."""

    solver_models: tuple[str, ...]
    verifier_models: tuple[str, ...]
    max_retries: int

    def __post_init__(self) -> None:
        if type(self.max_retries) is not int or self.max_retries < 0:
            raise ValueError("max_retries must be a nonnegative integer.")
        if len(self.solver_models) != self.max_retries + 1:
            raise ValueError("A configuration needs max_retries + 1 solver models.")
        if len(self.verifier_models) != self.max_retries:
            raise ValueError("A configuration needs one verifier per retry opportunity.")
        if any(not model for model in self.solver_models + self.verifier_models):
            raise ValueError("Model ids must be nonempty.")

    def snapshot(self) -> dict[str, object]:
        return {
            "version": "retry-workflow-config-v1",
            "max_retries": self.max_retries,
            "solver_models": list(self.solver_models),
            "verifier_models": list(self.verifier_models),
        }

    @property
    def configuration_id(self) -> str:
        encoded = json.dumps(self.snapshot(), sort_keys=True, separators=(",", ":"))
        return sha256(encoded.encode("utf-8")).hexdigest()[:20]


def generate_configurations(
    solver_candidates: Sequence[str],
    verifier_candidates: Sequence[str],
    *,
    max_retries: int,
) -> Iterator[WorkflowConfiguration]:
    """Generate every ordered solver/verifier assignment.

    With S solver choices, V verifier choices, and r retries, the count is
    S ** (r + 1) * V ** r. Roles and rounds choose independently.
    """

    if type(max_retries) is not int or max_retries < 0:
        raise ValueError("max_retries must be a nonnegative integer.")
    solvers = _validated_models(solver_candidates, "solver_candidates")
    verifiers = (
        _validated_models(verifier_candidates, "verifier_candidates")
        if max_retries
        else ()
    )
    slot_candidates: list[tuple[str, ...]] = []
    for _ in range(max_retries):
        slot_candidates.extend((solvers, verifiers))
    slot_candidates.append(solvers)
    for assignment in product(*slot_candidates):
        yield WorkflowConfiguration(
            solver_models=tuple(assignment[0::2]),
            verifier_models=tuple(assignment[1::2]),
            max_retries=max_retries,
        )

