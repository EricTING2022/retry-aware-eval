# Static baseline protocol

This module implements the four fixed-sequence, retry-unaware reference methods
specified in `experiments/main.tex`:

1. `gittins_full_panel_cost`: GittinsEval receives the frozen mean realized USD
   run cost of every configuration over the complete balanced panel.
2. `gittins_unit_run_cost`: the same GittinsEval policy receives cost `1` for
   every configuration.
3. `ucb_e`: after shared initialization, select an unfinished configuration
   maximizing `empirical_mean + sqrt(alpha / purchases)`.
4. `uniform`: purchase from a configuration with the smallest current count.

These are outer search methods. A configuration is the complete fixed sequence
`(a1, v1, ..., aR, vR, aT)`. A purchased record is one completed run of that
sequence on one question/replicate, including its actual retry behavior. The
methods never change models at retry boundaries.

## Matched offline replay

The input JSONL has one row per saved run:

```json
{"record_id":"...","configuration_id":"...","question_id":"...","reward":1,"realized_cost_usd":0.0012,"partition":"development","configuration_snapshot":{"solver_models":["..."],"verifier_models":["..."],"max_retries":1}}
```

Every configuration must have identical question/replicate coverage. Per-arm
record queues are independently shuffled from a stable seed, and all four
methods use the same queues. Every method first purchases one record per
configuration. Unpurchased rewards and individual costs are not supplied to a
policy. The full-panel Gittins condition receives only the frozen aggregate
mean cost vector as oracle side information.

The evaluator may use the complete panel only to report the recommended
configuration's equal-question-weighted accuracy and simple regret. The
horizontal cost trace always sums the realized costs of purchased records; it
never substitutes the Gittins cost input.

Every result retains every purchase and post-purchase checkpoint as its raw
anytime trace. The runner also emits a matched-budget curve on a predeclared
121-point grid spanning the common supported realized-USD range after paid
initialization. At each budget it uses the last completed checkpoint at or below
that amount: there is no interpolation, future observation, or crossing-run
credit. JSON and CSV report mean panel accuracy, mean simple regret, and the
fraction of seeds selecting a panel-optimal configuration for every method.
The USD curve also records the mean number of complete workflow evaluations at
each point, because one workflow can consume two or three model calls.

## GittinsEval details

The implementation follows the finite-population Gaussian construction in
Qian Xie, Yueli He, and Nairen Cao, arXiv:2609.25645v2, and the authors' public
repository at commit `0d9126b2a65eef229eb89d95f0fb58d46e49e312`.

- Common prior: `Normal(0.5, 0.04)`.
- Conservative binary cell noise: `tau^2 = 1/4` for one-record pulls.
- Acquisition: finite-panel posterior mean minus the precomputed stopping root.
- Anytime recommendation: maximize `M_k - sqrt(V_k)`.
- Effective DP cost: `eta * c_hat_k`; `eta` is recorded in every result.
- A completed arm reaching the largest Gittins index is recorded as the natural
  stopping point, but fixed-budget replay continues for comparable curves.

The NumPy FFT implementation is adapted from the upstream MIT-licensed dynamic
program; see `THIRD_PARTY_NOTICES.md`.

## Commands

Export a balanced panel from one completed SQLite run:

```powershell
$env:PYTHONPATH = (Resolve-Path "src")
python scripts/export_static_panel.py `
  --database artifacts/experiment.sqlite3 `
  --run-id RUN_ID `
  --benchmark data/mathqa/test_200.jsonl `
  --output artifacts/static_panel.jsonl
```

Run all four baselines:

```powershell
$env:PYTHONPATH = (Resolve-Path "src")
python scripts/run_static_baselines.py `
  --panel artifacts/static_panel.jsonl `
  --max-purchases 200 `
  --seed 0 `
  --seeds 1 `
  --budget-points 121 `
  --held-out `
  --cost-scaling-factor 1e-4 `
  --output artifacts/static_baselines.json
```

With `--held-out`, seed 0 searches only the `development` records. Its final
recommended fixed sequence is then frozen and scored on `reserved_test`; those
outcomes never enter acquisition or recommendation. The output contains each
method's trace, selected sequence, reserved-test accuracy and mean deployment
cost, plus per-method aggregates. Without `--held-out`, the command retains the
original single finite-panel reference from `main.tex`.

For a strict collect-after-freeze protocol, run the development search first,
extract the union of final recommendations, collect only those configurations
on reserved-test questions, and pass that separate file with
`--evaluation-panel`:

```powershell
python scripts/extract_selected_configurations.py `
  --baselines artifacts/development_search.json `
  --output artifacts/selected_configurations.json

python scripts/run_static_baselines.py `
  --panel artifacts/development_panel.jsonl `
  --evaluation-panel artifacts/reserved_test_panel.jsonl `
  --max-purchases 2700 `
  --seed 0 `
  --seeds 1 `
  --output artifacts/final_held_out_baselines.json
```

The search is deterministic for a fixed panel, seed, and hyperparameters,
so the second command reproduces the frozen recommendations before loading the
separate evaluation panel. It fails if a selected configuration is absent or
its saved sequence snapshot differs.

The CLI defaults to one seed, so `--seeds 1` may be omitted. This is an offline
replay and makes no model API calls.

With one seed, `optimal_selection_fraction` is necessarily zero or one. To
measure variability from the search policy/order as Thomas's comparison does,
rerun only the offline search on the same panel with multiple seeds; this does
not repeat OpenRouter collection:

```powershell
python scripts/run_static_baselines.py `
  --panel artifacts/development_panel.jsonl `
  --max-purchases 2700 `
  --seed 0 `
  --seeds 100 `
  --budget-points 121 `
  --output artifacts/development_search_100_seeds.json
```
