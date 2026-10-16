"""Fixed-sequence, retry-unaware outer-search baselines for balanced panels.

The finite-panel Gittins calculations follow Qian Xie et al.'s public v2
implementation. The numerical dynamic program is adapted to NumPy so this
project does not need the upstream JAX and PyTorch runtime.
"""

from __future__ import annotations

from bisect import bisect_right
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Iterable, Literal, Mapping, Sequence

import numpy as np


BaselineName = Literal[
    "gittins_full_panel_cost",
    "gittins_unit_run_cost",
    "ucb_e",
    "uniform",
]

BASELINE_NAMES: tuple[BaselineName, ...] = (
    "gittins_full_panel_cost",
    "gittins_unit_run_cost",
    "ucb_e",
    "uniform",
)


@dataclass(frozen=True)
class PanelRecord:
    record_id: str
    configuration_id: str
    question_id: str
    reward: float
    realized_cost_usd: float
    partition: str = "development"
    configuration_snapshot: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        if not self.record_id or not self.configuration_id or not self.question_id:
            raise ValueError("Panel record identifiers are required.")
        if not math.isfinite(self.reward) or not 0.0 <= self.reward <= 1.0:
            raise ValueError("reward must be finite and in [0, 1].")
        if not math.isfinite(self.realized_cost_usd) or self.realized_cost_usd < 0.0:
            raise ValueError("realized_cost_usd must be finite and nonnegative.")
        if self.partition not in ("development", "reserved_test"):
            raise ValueError("partition must be development or reserved_test.")
        if self.configuration_snapshot is not None and not isinstance(
            self.configuration_snapshot, Mapping
        ):
            raise ValueError("configuration_snapshot must be an object when supplied.")


@dataclass(frozen=True)
class Purchase:
    purchase_index: int
    record_id: str
    configuration_id: str
    question_id: str
    reward: float
    realized_cost_usd: float


@dataclass(frozen=True)
class Checkpoint:
    purchases: int
    cumulative_realized_cost_usd: float
    recommended_configuration_id: str
    recommended_panel_accuracy: float
    simple_regret: float


@dataclass(frozen=True)
class BaselineResult:
    method: BaselineName
    seed: int
    max_purchases: int
    initialization_purchases: int
    cost_scaling_factor: float | None
    cost_input_by_configuration: dict[str, float]
    natural_stop_purchase: int | None
    purchases: list[Purchase]
    checkpoints: list[Checkpoint]

    def as_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["schema_version"] = "static-baseline-trace-v1"
        return value


def build_matched_budget_curve(
    results: Sequence[BaselineResult],
    *,
    budget_points: int = 121,
    optimal_tolerance: float = 1e-12,
) -> dict[str, object]:
    """Align anytime traces at common realized-USD budgets without lookahead."""
    if budget_points < 2:
        raise ValueError("budget_points must be at least 2.")
    if not results:
        raise ValueError("At least one baseline result is required.")

    by_method: dict[str, dict[int, BaselineResult]] = defaultdict(dict)
    for result in results:
        if result.seed in by_method[result.method]:
            raise ValueError(
                f"Duplicate result for method {result.method!r}, seed {result.seed}."
            )
        if not result.checkpoints:
            raise ValueError("Every baseline result needs at least one checkpoint.")
        if result.checkpoints[0].purchases != result.initialization_purchases:
            raise ValueError("The first checkpoint must follow paid initialization.")
        by_method[result.method][result.seed] = result

    seed_sets = {tuple(sorted(rows)) for rows in by_method.values()}
    if len(seed_sets) != 1:
        raise ValueError("Every method must contain the same search seeds.")
    seeds = next(iter(seed_sets))

    lower_budget = max(
        result.checkpoints[0].cumulative_realized_cost_usd for result in results
    )
    upper_budget = min(
        result.checkpoints[-1].cumulative_realized_cost_usd for result in results
    )
    if upper_budget < lower_budget:
        raise ValueError("Baseline traces have no common post-initialization budget.")
    if math.isclose(lower_budget, upper_budget, rel_tol=0.0, abs_tol=1e-15):
        budgets = [lower_budget]
    else:
        step = (upper_budget - lower_budget) / (budget_points - 1)
        budgets = [lower_budget + index * step for index in range(budget_points)]
        budgets[-1] = upper_budget

    series: dict[str, list[dict[str, object]]] = {}
    for method, rows in sorted(by_method.items()):
        points = []
        for budget_index, budget in enumerate(budgets):
            selected_checkpoints = []
            for seed in seeds:
                checkpoints = rows[seed].checkpoints
                costs = [row.cumulative_realized_cost_usd for row in checkpoints]
                checkpoint_index = bisect_right(costs, budget + 1e-15) - 1
                if checkpoint_index < 0:
                    raise ValueError("A common budget precedes a completed initialization.")
                selected_checkpoints.append(checkpoints[checkpoint_index])
            accuracies = [
                row.recommended_panel_accuracy for row in selected_checkpoints
            ]
            regrets = [row.simple_regret for row in selected_checkpoints]
            points.append(
                {
                    "budget_index": budget_index,
                    "budget_usd": budget,
                    "mean_completed_run_evaluations": math.fsum(
                        row.purchases for row in selected_checkpoints
                    )
                    / len(selected_checkpoints),
                    "mean_recommended_panel_accuracy": math.fsum(accuracies)
                    / len(accuracies),
                    "mean_simple_regret": math.fsum(regrets) / len(regrets),
                    "optimal_selection_fraction": sum(
                        regret <= optimal_tolerance for regret in regrets
                    )
                    / len(regrets),
                    "seed_count": len(seeds),
                }
            )
        series[method] = points

    return {
        "schema_version": "matched-budget-curve-v1",
        "budget_rule": "equally spaced common realized-USD grid",
        "alignment": "last completed checkpoint at or below budget; no interpolation or lookahead",
        "initialization_policy": "budgets before every run completed paid initialization are excluded",
        "declared_budget_points": budget_points,
        "actual_budget_points": len(budgets),
        "common_budget_min_usd": lower_budget,
        "common_budget_max_usd": upper_budget,
        "seed_start": seeds[0],
        "seed_count": len(seeds),
        "optimal_tolerance": optimal_tolerance,
        "series": series,
    }


def load_panel_jsonl(path: str | Path) -> list[PanelRecord]:
    records: list[PanelRecord] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            try:
                records.append(
                    PanelRecord(
                        record_id=str(raw["record_id"]),
                        configuration_id=str(raw["configuration_id"]),
                        question_id=str(raw["question_id"]),
                        reward=float(raw["reward"]),
                        realized_cost_usd=float(raw["realized_cost_usd"]),
                        partition=str(raw.get("partition", "development")),
                        configuration_snapshot=raw.get("configuration_snapshot"),
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid panel record on line {line_number}: {exc}") from exc
    return records


class BalancedPanel:
    """Validated static response panel with identical question coverage per arm."""

    def __init__(self, records: Sequence[PanelRecord]) -> None:
        if not records:
            raise ValueError("A panel needs at least one record.")
        grouped: dict[str, list[PanelRecord]] = defaultdict(list)
        record_ids: set[str] = set()
        for record in records:
            if record.record_id in record_ids:
                raise ValueError(f"Duplicate record_id {record.record_id!r}.")
            record_ids.add(record.record_id)
            grouped[record.configuration_id].append(record)
        self.configuration_ids = tuple(sorted(grouped))
        reference = Counter(row.question_id for row in grouped[self.configuration_ids[0]])
        if not reference:
            raise ValueError("Each configuration needs at least one record.")
        for configuration_id in self.configuration_ids[1:]:
            coverage = Counter(row.question_id for row in grouped[configuration_id])
            if coverage != reference:
                raise ValueError(
                    "Static baselines require balanced question/replicate coverage; "
                    f"configuration {configuration_id!r} differs from the first arm."
                )
        self.records_by_configuration = {
            key: tuple(sorted(value, key=lambda row: (row.question_id, row.record_id)))
            for key, value in grouped.items()
        }
        self.records_per_configuration = sum(reference.values())
        self.question_ids = tuple(sorted(reference))
        self.configuration_snapshots: dict[str, Mapping[str, object]] = {}
        for key, rows in grouped.items():
            snapshots = {
                json.dumps(row.configuration_snapshot, sort_keys=True)
                for row in rows
                if row.configuration_snapshot is not None
            }
            if len(snapshots) > 1:
                raise ValueError(f"Configuration {key!r} has inconsistent snapshots.")
            if snapshots:
                self.configuration_snapshots[key] = json.loads(next(iter(snapshots)))
        self._panel_accuracy = {
            key: _equal_question_mean(rows, "reward") for key, rows in grouped.items()
        }
        self._panel_mean_cost = {
            key: _equal_question_mean(rows, "realized_cost_usd")
            for key, rows in grouped.items()
        }

    @property
    def total_records(self) -> int:
        return len(self.configuration_ids) * self.records_per_configuration

    @property
    def best_panel_accuracy(self) -> float:
        return max(self._panel_accuracy.values())

    def panel_accuracy(self, configuration_id: str) -> float:
        return self._panel_accuracy[configuration_id]

    def panel_mean_cost(self, configuration_id: str) -> float:
        return self._panel_mean_cost[configuration_id]

    def matched_queues(self, seed: int) -> dict[str, list[PanelRecord]]:
        queues: dict[str, list[PanelRecord]] = {}
        for configuration_id in self.configuration_ids:
            queue = list(self.records_by_configuration[configuration_id])
            rng = random.Random(_stable_arm_seed(seed, configuration_id))
            rng.shuffle(queue)
            queues[configuration_id] = queue
        return queues


def _equal_question_mean(
    rows: Iterable[PanelRecord], attribute: Literal["reward", "realized_cost_usd"]
) -> float:
    by_question: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_question[row.question_id].append(float(getattr(row, attribute)))
    return sum(sum(values) / len(values) for values in by_question.values()) / len(by_question)


def _stable_arm_seed(seed: int, configuration_id: str) -> int:
    digest = hashlib.sha256(f"{seed}:{configuration_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _finite_population_posterior(
    observations: Sequence[float],
    *,
    n_cells: int,
    prior_mean: float,
    prior_variance: float,
    tau_sq_cell: float,
) -> tuple[float, float]:
    count = len(observations)
    observed_sum = math.fsum(observations)
    latent_variance = 1.0 / (1.0 / prior_variance + count / tau_sq_cell)
    latent_mean = latent_variance * (
        prior_mean / prior_variance + observed_sum / tau_sq_cell
    )
    remaining = n_cells - count
    finite_mean = (observed_sum + remaining * latent_mean) / n_cells
    finite_variance = (
        remaining * remaining * latent_variance + remaining * tau_sq_cell
    ) / (n_cells * n_cells)
    return finite_mean, finite_variance


def _normal_cdf(values: np.ndarray) -> np.ndarray:
    flat = np.fromiter(
        (0.5 * (1.0 + math.erf(float(value) / math.sqrt(2.0))) for value in values.flat),
        dtype=np.float64,
        count=values.size,
    )
    return flat.reshape(values.shape)


def _fft_valid_convolution(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    full_length = len(first) + len(second) - 1
    fft_length = 1 << (full_length - 1).bit_length()
    full = np.fft.irfft(
        np.fft.rfft(first, fft_length) * np.fft.rfft(second, fft_length),
        fft_length,
    )[:full_length]
    start = min(len(first), len(second)) - 1
    length = abs(len(first) - len(second)) + 1
    return full[start : start + length]


def _gaussian_piecewise_linear_expectation(
    values: np.ndarray,
    *,
    sigma: float,
    left_bound: float,
    right_bound: float,
) -> np.ndarray:
    """FFT form of E[f(x + Normal(0, sigma^2))] used by Qian's DP."""
    count = len(values)
    grid = np.linspace(left_bound, right_bound, count, dtype=np.float64)
    spacing = grid[1] - grid[0]
    slopes = np.diff(values) / spacing
    slope_values = np.concatenate(([0.0], slopes, [1.0]))
    slope_differences = np.diff(slope_values)
    offsets = (np.arange(2 * count - 1) - (count - 1)) * spacing
    scaled = offsets / sigma
    pdf = np.exp(-0.5 * scaled * scaled) / math.sqrt(2.0 * math.pi)
    expected_improvement = offsets * _normal_cdf(scaled) + sigma * pdf
    return values[0] + _fft_valid_convolution(slope_differences, expected_improvement)


@lru_cache(maxsize=None)
def compute_gittins_root_schedule(
    *,
    n_cells: int,
    transition_cost: float,
    prior_variance: float = 0.04,
    tau_sq_cell: float = 0.25,
    grid_points: int = 1025,
) -> tuple[float, ...]:
    """Return finite-panel Gittins stopping roots for stages 0..N."""
    if n_cells <= 0:
        raise ValueError("n_cells must be positive.")
    for name, value in (
        ("transition_cost", transition_cost),
        ("prior_variance", prior_variance),
        ("tau_sq_cell", tau_sq_cell),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be positive and finite.")
    if grid_points < 129 or grid_points % 2 == 0:
        raise ValueError("grid_points must be an odd integer of at least 129.")

    transition_stds: list[float] = []
    finite_scale = 1.0 + tau_sq_cell / (n_cells * prior_variance)
    for stage in range(n_cells):
        variance = 1.0 / (1.0 / prior_variance + stage / tau_sq_cell)
        latent_std = math.sqrt(variance * variance / (variance + tau_sq_cell))
        transition_stds.append(finite_scale * latent_std)

    total_std = math.sqrt(math.fsum(value * value for value in transition_stds))
    total_cost = n_cells * transition_cost
    left_bound = -5.0 * total_std
    right_bound = 1.01 * total_cost + 5.0 * total_std
    grid = np.linspace(left_bound, right_bound, grid_points, dtype=np.float64)
    continuation = grid.copy()
    roots = [0.0] * (n_cells + 1)

    for stage in range(n_cells - 1, -1, -1):
        continuation = _gaussian_piecewise_linear_expectation(
            np.maximum(continuation, 0.0),
            sigma=transition_stds[stage],
            left_bound=left_bound,
            right_bound=right_bound,
        ) - transition_cost
        crossing = int(np.searchsorted(continuation, 0.0))
        roots[stage] = float(grid[min(crossing, grid_points - 1)])
    roots[-1] = 0.0
    return tuple(roots)


def split_development_reserved(
    records: Sequence[PanelRecord],
) -> tuple[BalancedPanel, BalancedPanel]:
    development_records = [row for row in records if row.partition == "development"]
    reserved_records = [row for row in records if row.partition == "reserved_test"]
    if not development_records or not reserved_records:
        raise ValueError(
            "Held-out evaluation requires both development and reserved_test records."
        )
    development = BalancedPanel(development_records)
    reserved = BalancedPanel(reserved_records)
    if development.configuration_ids != reserved.configuration_ids:
        raise ValueError(
            "Development and reserved-test panels must contain identical configurations."
        )
    if development.configuration_snapshots != reserved.configuration_snapshots:
        raise ValueError(
            "Development and reserved-test configuration snapshots must match."
        )
    overlap = set(development.question_ids) & set(reserved.question_ids)
    if overlap:
        raise ValueError("Development and reserved-test question IDs must be disjoint.")
    return development, reserved


def _argmax_configuration(scores: dict[str, float], candidates: Iterable[str]) -> str:
    available = tuple(sorted(candidates))
    if not available:
        raise ValueError("No configurations remain available.")
    return max(available, key=lambda key: (scores[key], -available.index(key)))


def run_static_baseline(
    panel: BalancedPanel,
    method: BaselineName,
    *,
    max_purchases: int,
    seed: int = 0,
    ucb_alpha: float = 1.0,
    prior_mean: float = 0.5,
    prior_variance: float = 0.04,
    tau_sq_cell: float = 0.25,
    cost_scaling_factor: float = 1e-4,
    gittins_grid_points: int = 1025,
) -> BaselineResult:
    """Run one fixed-sequence outer-search baseline on an offline panel."""
    if method not in BASELINE_NAMES:
        raise ValueError(f"Unknown baseline method {method!r}.")
    arm_count = len(panel.configuration_ids)
    if max_purchases < arm_count:
        raise ValueError("Budget must fund the shared one-record-per-arm initialization.")
    max_purchases = min(max_purchases, panel.total_records)
    if not math.isfinite(ucb_alpha) or ucb_alpha < 0.0:
        raise ValueError("ucb_alpha must be nonnegative and finite.")
    if not math.isfinite(cost_scaling_factor) or cost_scaling_factor <= 0.0:
        raise ValueError("cost_scaling_factor must be positive and finite.")

    queues = panel.matched_queues(seed)
    next_positions = {key: 0 for key in panel.configuration_ids}
    observations = {key: [] for key in panel.configuration_ids}
    purchases: list[Purchase] = []
    checkpoints: list[Checkpoint] = []
    cumulative_cost = 0.0
    natural_stop_purchase: int | None = None

    if method == "gittins_full_panel_cost":
        cost_inputs = {
            key: panel.panel_mean_cost(key) for key in panel.configuration_ids
        }
    elif method == "gittins_unit_run_cost":
        cost_inputs = {key: 1.0 for key in panel.configuration_ids}
    else:
        cost_inputs = {}

    root_schedules: dict[str, tuple[float, ...]] = {}
    if method.startswith("gittins_"):
        cached_by_cost: dict[float, tuple[float, ...]] = {}
        for key in panel.configuration_ids:
            effective_cost = cost_inputs[key] * cost_scaling_factor
            if effective_cost <= 0.0:
                raise ValueError("Gittins cost inputs must be strictly positive.")
            if effective_cost not in cached_by_cost:
                cached_by_cost[effective_cost] = compute_gittins_root_schedule(
                    n_cells=panel.records_per_configuration,
                    transition_cost=effective_cost,
                    prior_variance=prior_variance,
                    tau_sq_cell=tau_sq_cell,
                    grid_points=gittins_grid_points,
                )
            root_schedules[key] = cached_by_cost[effective_cost]

    def purchase(configuration_id: str) -> None:
        nonlocal cumulative_cost
        position = next_positions[configuration_id]
        row = queues[configuration_id][position]
        next_positions[configuration_id] = position + 1
        observations[configuration_id].append(row.reward)
        cumulative_cost += row.realized_cost_usd
        purchases.append(
            Purchase(
                purchase_index=len(purchases) + 1,
                record_id=row.record_id,
                configuration_id=row.configuration_id,
                question_id=row.question_id,
                reward=row.reward,
                realized_cost_usd=row.realized_cost_usd,
            )
        )

    def posterior_scores() -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
        means: dict[str, float] = {}
        variances: dict[str, float] = {}
        indices: dict[str, float] = {}
        for key in panel.configuration_ids:
            means[key], variances[key] = _finite_population_posterior(
                observations[key],
                n_cells=panel.records_per_configuration,
                prior_mean=prior_mean,
                prior_variance=prior_variance,
                tau_sq_cell=tau_sq_cell,
            )
            count = len(observations[key])
            indices[key] = means[key] - root_schedules[key][count]
        return means, variances, indices

    def recommend() -> str:
        if method.startswith("gittins_"):
            means, variances, _ = posterior_scores()
            scores = {
                key: means[key] - math.sqrt(max(0.0, variances[key]))
                for key in panel.configuration_ids
            }
        else:
            scores = {
                key: math.fsum(observations[key]) / len(observations[key])
                for key in panel.configuration_ids
            }
        return _argmax_configuration(scores, panel.configuration_ids)

    def checkpoint() -> None:
        recommended = recommend()
        accuracy = panel.panel_accuracy(recommended)
        checkpoints.append(
            Checkpoint(
                purchases=len(purchases),
                cumulative_realized_cost_usd=cumulative_cost,
                recommended_configuration_id=recommended,
                recommended_panel_accuracy=accuracy,
                simple_regret=panel.best_panel_accuracy - accuracy,
            )
        )

    def record_post_pull_natural_stop() -> None:
        nonlocal natural_stop_purchase
        if natural_stop_purchase is not None or not method.startswith("gittins_"):
            return
        _, _, indices = posterior_scores()
        best_overall = _argmax_configuration(indices, panel.configuration_ids)
        if next_positions[best_overall] == panel.records_per_configuration:
            natural_stop_purchase = len(purchases)

    # Qian's comparison gives every method the same paid initialization.
    for key in panel.configuration_ids:
        purchase(key)
    record_post_pull_natural_stop()
    checkpoint()

    while len(purchases) < max_purchases:
        incomplete = [
            key
            for key in panel.configuration_ids
            if next_positions[key] < panel.records_per_configuration
        ]
        if not incomplete:
            break
        if method == "uniform":
            selected = min(incomplete, key=lambda key: (len(observations[key]), key))
        elif method == "ucb_e":
            scores = {
                key: math.fsum(observations[key]) / len(observations[key])
                + math.sqrt(ucb_alpha / len(observations[key]))
                for key in incomplete
            }
            selected = _argmax_configuration(scores, incomplete)
        else:
            _, _, indices = posterior_scores()
            selected = _argmax_configuration(indices, incomplete)
        purchase(selected)
        record_post_pull_natural_stop()
        checkpoint()

    return BaselineResult(
        method=method,
        seed=seed,
        max_purchases=max_purchases,
        initialization_purchases=arm_count,
        cost_scaling_factor=(cost_scaling_factor if method.startswith("gittins_") else None),
        cost_input_by_configuration=cost_inputs,
        natural_stop_purchase=natural_stop_purchase,
        purchases=purchases,
        checkpoints=checkpoints,
    )


def run_all_static_baselines(
    panel: BalancedPanel,
    *,
    max_purchases: int,
    seed: int = 0,
    **kwargs: float | int,
) -> list[BaselineResult]:
    return [
        run_static_baseline(
            panel,
            method,
            max_purchases=max_purchases,
            seed=seed,
            **kwargs,
        )
        for method in BASELINE_NAMES
    ]
