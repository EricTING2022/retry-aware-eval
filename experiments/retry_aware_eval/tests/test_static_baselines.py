from __future__ import annotations

import math
import unittest

from retry_eval.static_baselines import (
    BASELINE_NAMES,
    BalancedPanel,
    PanelRecord,
    build_matched_budget_curve,
    compute_gittins_root_schedule,
    run_all_static_baselines,
    split_development_reserved,
)


def synthetic_panel() -> BalancedPanel:
    rewards = {
        "arm-a": [1, 1, 1, 1, 0, 1],
        "arm-b": [1, 0, 1, 0, 1, 0],
        "arm-c": [0, 0, 1, 0, 0, 0],
    }
    costs = {"arm-a": 0.03, "arm-b": 0.02, "arm-c": 0.01}
    records = []
    for arm, outcomes in rewards.items():
        for index, reward in enumerate(outcomes):
            records.append(
                PanelRecord(
                    record_id=f"{arm}-{index}",
                    configuration_id=arm,
                    question_id=f"q-{index}",
                    reward=float(reward),
                    realized_cost_usd=costs[arm],
                )
            )
    return BalancedPanel(records)


class StaticBaselineTests(unittest.TestCase):
    def test_all_four_baselines_run_with_matched_paid_initialization(self) -> None:
        panel = synthetic_panel()
        results = run_all_static_baselines(
            panel,
            max_purchases=12,
            seed=7,
            gittins_grid_points=257,
        )
        self.assertEqual(tuple(result.method for result in results), BASELINE_NAMES)
        first_purchases = [
            tuple(item.record_id for item in result.purchases[:3]) for result in results
        ]
        self.assertTrue(all(value == first_purchases[0] for value in first_purchases))
        for result in results:
            self.assertEqual(result.initialization_purchases, 3)
            self.assertEqual(len(result.purchases), 12)
            self.assertEqual(len({item.record_id for item in result.purchases}), 12)
            self.assertEqual(result.checkpoints[0].purchases, 3)
            self.assertGreaterEqual(result.checkpoints[-1].simple_regret, 0.0)

    def test_cost_inputs_are_exposed_only_to_the_declared_gittins_variant(self) -> None:
        panel = synthetic_panel()
        results = {
            result.method: result
            for result in run_all_static_baselines(
                panel,
                max_purchases=4,
                seed=0,
                gittins_grid_points=129,
            )
        }
        self.assertEqual(
            results["gittins_full_panel_cost"].cost_input_by_configuration,
            {"arm-a": 0.03, "arm-b": 0.02, "arm-c": 0.01},
        )
        self.assertEqual(
            results["gittins_unit_run_cost"].cost_input_by_configuration,
            {"arm-a": 1.0, "arm-b": 1.0, "arm-c": 1.0},
        )
        self.assertEqual(results["ucb_e"].cost_input_by_configuration, {})
        self.assertEqual(results["uniform"].cost_input_by_configuration, {})

    def test_matched_budget_curve_uses_only_affordable_completed_checkpoints(self) -> None:
        panel = synthetic_panel()
        results = []
        for seed in (7, 8):
            results.extend(
                run_all_static_baselines(
                    panel,
                    max_purchases=12,
                    seed=seed,
                    gittins_grid_points=129,
                )
            )
        curve = build_matched_budget_curve(results, budget_points=5)

        self.assertEqual(curve["declared_budget_points"], 5)
        self.assertEqual(curve["actual_budget_points"], 5)
        self.assertEqual(curve["seed_count"], 2)
        for method, points in curve["series"].items():
            self.assertEqual(len(points), 5, method)
            self.assertEqual([row["budget_index"] for row in points], list(range(5)))
            self.assertTrue(
                all(
                    0.0 <= row["optimal_selection_fraction"] <= 1.0
                    for row in points
                )
            )
            method_results = [row for row in results if row.method == method]
            for point in points:
                affordable = []
                for result in method_results:
                    candidates = [
                        checkpoint
                        for checkpoint in result.checkpoints
                        if checkpoint.cumulative_realized_cost_usd
                        <= point["budget_usd"] + 1e-15
                    ]
                    affordable.append(candidates[-1])
                expected = sum(
                    row.recommended_panel_accuracy for row in affordable
                ) / len(affordable)
                self.assertAlmostEqual(
                    point["mean_recommended_panel_accuracy"], expected
                )
                self.assertAlmostEqual(
                    point["mean_completed_run_evaluations"],
                    sum(row.purchases for row in affordable) / len(affordable),
                )

    def test_gittins_higher_cost_has_a_larger_stopping_root(self) -> None:
        low = compute_gittins_root_schedule(
            n_cells=6, transition_cost=1e-4, grid_points=257
        )
        high = compute_gittins_root_schedule(
            n_cells=6, transition_cost=1e-2, grid_points=257
        )
        self.assertGreater(high[0], low[0])
        self.assertEqual(low[-1], 0.0)
        self.assertEqual(high[-1], 0.0)

    def test_one_step_gittins_root_matches_closed_form_expected_improvement(self) -> None:
        cost = 0.01
        prior_variance = 0.04
        tau_sq = 0.25
        grid_points = 1025
        roots = compute_gittins_root_schedule(
            n_cells=1,
            transition_cost=cost,
            prior_variance=prior_variance,
            tau_sq_cell=tau_sq,
            grid_points=grid_points,
        )
        sigma = (1.0 + tau_sq / prior_variance) * math.sqrt(
            prior_variance * prior_variance / (prior_variance + tau_sq)
        )

        def expected_improvement(value: float) -> float:
            z = value / sigma
            cdf = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
            pdf = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
            return value * cdf + sigma * pdf

        lower, upper = -5.0 * sigma, 1.01 * cost + 5.0 * sigma
        for _ in range(80):
            midpoint = (lower + upper) / 2.0
            if expected_improvement(midpoint) < cost:
                lower = midpoint
            else:
                upper = midpoint
        grid_spacing = (1.01 * cost + 10.0 * sigma) / (grid_points - 1)
        self.assertLessEqual(abs(roots[0] - (lower + upper) / 2.0), grid_spacing)

    def test_panel_rejects_unbalanced_question_coverage(self) -> None:
        with self.assertRaisesRegex(ValueError, "balanced"):
            BalancedPanel(
                [
                    PanelRecord("a1", "a", "q1", 1.0, 0.1),
                    PanelRecord("b1", "b", "q2", 1.0, 0.1),
                ]
            )

    def test_gittins_records_post_pull_stop_at_the_end_of_budget(self) -> None:
        panel = BalancedPanel(
            [
                PanelRecord("a1", "a", "q1", 1.0, 0.1),
                PanelRecord("b1", "b", "q1", 0.0, 0.1),
            ]
        )
        results = run_all_static_baselines(
            panel,
            max_purchases=2,
            gittins_grid_points=129,
        )
        self.assertEqual(results[0].natural_stop_purchase, 2)
        self.assertEqual(results[1].natural_stop_purchase, 2)

    def test_held_out_split_keeps_reserved_outcomes_out_of_search_panel(self) -> None:
        records = []
        for partition, questions in (
            ("development", ("d1", "d2")),
            ("reserved_test", ("t1", "t2")),
        ):
            for arm, rewards in (("a", (1, 1)), ("b", (0, 1))):
                for question, reward in zip(questions, rewards):
                    records.append(
                        PanelRecord(
                            f"{partition}-{arm}-{question}",
                            arm,
                            question,
                            float(reward),
                            0.01,
                            partition,
                        )
                    )
        development, reserved = split_development_reserved(records)
        self.assertEqual(development.question_ids, ("d1", "d2"))
        self.assertEqual(reserved.question_ids, ("t1", "t2"))
        result = run_all_static_baselines(
            development, max_purchases=4, gittins_grid_points=129
        )[0]
        selected = result.checkpoints[-1].recommended_configuration_id
        self.assertEqual(reserved.panel_accuracy(selected), 1.0)


if __name__ == "__main__":
    unittest.main()
