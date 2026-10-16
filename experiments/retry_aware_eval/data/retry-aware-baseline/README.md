# Retry-Aware Baseline Experiment Data

This directory contains the complete saved data for the fixed-sequence
retry-aware baseline experiment described in the project [README](../../README.md).
The files cover the balanced development collection, the frozen reserved-test
collection, four offline search baselines, the 100-seed search replay, detailed
request/response audit records, and the original pipeline log.

The experiment uses 100 development questions, 100 disjoint reserved-test
questions, and 27 one-retry solver-verifier-solver configurations. The
development ledger contains 2,700 completed workflows; the reserved-test ledger
contains 100 workflows for the frozen selected configuration.

All files are stored directly in Git so the dataset also works in a public
fork. The 100-seed full search trace is gzip-compressed because its original
JSON is larger than GitHub's per-file limit.

## Directory structure

```text
retry-aware-baseline/
  ledgers/      canonical SQLite experiment databases
  audit-logs/   one JSONL record per collected workflow, including raw calls
  panels/       compact records consumed by the offline search algorithms
  search/       baseline traces, recommendations, and matched-budget curves
  summaries/    compact collection-level usage and completion summaries
  logs/         original pipeline console output
```

## File catalog

### `ledgers/`

| File | Contents |
|---|---|
| `development.sqlite3` | Canonical development run: 27 configurations × 100 questions = 2,700 workflows and 6,685 executed calls. |
| `reserved_test.sqlite3` | Canonical reserved-test run: the frozen selected configuration on 100 questions and 251 executed calls. |

Both databases use the same relational schema:

| Table | Key fields and meaning |
|---|---|
| `runs` | `run_id`, run name, retry count, status, frozen settings JSON, and timestamps. |
| `configurations` | `run_id`, stable `configuration_id`, and the versioned solver/verifier snapshot. |
| `executions` | One workflow-question execution with question JSON, status, returned attempt, final answer, stop reason, error, and timing. |
| `attempts` | One row per solver attempt with model assignments, parsed answer, selected option, correctness, verifier verdict, and parse errors. |
| `call_slots` | Every planned solver/verifier call slot, including request/response JSON, provider, token counts, reported USD cost, latency, status, and errors. |
| `grades` | Independent MathQA grade keyed by `execution_id`, with correct option, returned option, and `is_correct`. |
| `execution_retries` | Archived technical retries. These completed runs contain zero archived retries. |
| `metadata` | Database-level metadata encoded as key/value strings. |

`run_id` links runs to configurations and executions. `execution_id` links an
execution to attempts, call slots, grades, and archived retries.

### `audit-logs/`

| File | Contents |
|---|---|
| `development_detailed.jsonl` | Header plus 2,700 nested development workflow records. |
| `reserved_test_detailed.jsonl` | Header plus 100 nested reserved-test workflow records. |

Line 1 is a run header with `record_type`, schema version, source database, run
metadata, workflow count, configuration count, and archived retry count. Every
later line is one workflow object:

- `configuration_id` and `configuration`: the stable ID and versioned model
  sequence;
- `execution`: question, timestamps, status, returned attempt, final answer,
  stop reason, and error;
- `attempts`: solver answer, parsed answer, selected option, verifier verdict,
  and nested solver/verifier call records for each attempt;
- `grade`: correct option, returned option, correctness, and grader version;
- `call_totals`: calls, tokens, elapsed time, reported cost, and per-role totals.

The audit logs retain raw model requests and responses. Use them for provenance
and debugging; use the compact panels for search analysis.

### `panels/`

| File | Rows | Contents |
|---|---:|---|
| `development_panel.jsonl` | 2,700 | Balanced search panel covering every development question and configuration. |
| `reserved_test_panel.jsonl` | 100 | Evaluation panel for the frozen selected configuration. |

Each line is a flat JSON object with:

| Field | Meaning |
|---|---|
| `record_id` | Stable source workflow record identifier. |
| `configuration_id` | Stable workflow configuration identifier. |
| `configuration_snapshot` | Versioned solver sequence, verifier sequence, and retry count. |
| `partition` | `development` or `reserved_test`. |
| `question_id` | Stable MathQA question identifier. |
| `reward` | `1.0` for a correct final option and `0.0` otherwise. |
| `realized_cost_usd` | Sum of provider-reported costs for calls actually executed by the workflow. |

These are the primary inputs to `scripts/run_static_baselines.py`.

### `search/`

| File | Contents |
|---|---|
| `development_search.json` | One-seed development search trace for all four methods. |
| `development_search_matched_budget_curve.csv` | One-seed development curves aligned to common realized-USD checkpoints. |
| `selected_configurations.json` | Frozen union of final development recommendations used for reserved-test collection. |
| `final_held_out.json` | One-seed search result joined to the separately collected reserved-test panel. |
| `final_held_out_matched_budget_curve.csv` | Matched-budget curve paired with `final_held_out.json`. |
| `final_held_out_100_seeds.json.gz` | Gzip-compressed full purchase histories and checkpoints for 100 offline queue-order seeds. |
| `final_held_out_100_seeds_matched_budget_curve.csv` | Compact 100-seed curve used by the committed analysis and figures. |

The search JSON objects contain:

- experiment metadata: `schema_version`, `protocol`, configuration count,
  configuration snapshots, panel descriptors, and seed range;
- `results`: one object per method and seed with initialization size, maximum
  purchases, acquisition cost inputs, every purchase, and every checkpoint;
- `matched_budget_curve`: the common realized-USD comparison grid;
- `held_out_evaluations` and `held_out_aggregates`: frozen recommendation
  accuracy and deployment cost on the reserved-test panel.

The matched-budget CSV files use the columns:

```text
method,budget_index,budget_usd,mean_completed_run_evaluations,
mean_recommended_panel_accuracy,mean_simple_regret,
optimal_selection_fraction,seed_count
```

Each row reports the latest completed checkpoint affordable at that budget;
there is no interpolation or lookahead.

Decompress the full 100-seed trace when it is needed for detailed inspection:

```powershell
$source = "data/retry-aware-baseline/search/final_held_out_100_seeds.json.gz"
$target = "artifacts/reproduced/final_held_out_100_seeds.json"
New-Item -ItemType Directory -Force (Split-Path $target) | Out-Null
$input = [IO.File]::OpenRead($source)
$output = [IO.File]::Create($target)
$gzip = [IO.Compression.GZipStream]::new(
  $input, [IO.Compression.CompressionMode]::Decompress
)
$gzip.CopyTo($output)
$gzip.Dispose(); $output.Dispose(); $input.Dispose()
```

```bash
mkdir -p artifacts/reproduced
gzip -dc data/retry-aware-baseline/search/final_held_out_100_seeds.json.gz \
  > artifacts/reproduced/final_held_out_100_seeds.json
```

The decompressed file has SHA-256
`feb5aac7acdb88543703882d5f2eefbe7ced6e135f553a956397d65a33e92cb7`.

### `summaries/`

| File | Contents |
|---|---|
| `development_summary.json` | Development completion status, run ID, workflow and call counts, token usage, cost, timing, grades, and per-role/model usage. |
| `reserved_test_summary.json` | The same collection-level fields for the reserved-test run. |

Important fields include `run_id`, `status`, `execution_counts`,
`executed_call_count`, `input_tokens`, `output_tokens`, `reasoning_tokens`,
`reported_cost_usd`, `call_elapsed_seconds`, `workflow_elapsed_seconds`,
`grade_counts`, `stop_reason_counts`, and `usage_by_role_and_model`. Embedded
file paths record the original collection location and are retained as
historical provenance.

### `logs/`

`pipeline_console.log` is the original chronological console output from the
development collection, search, recommendation freeze, and reserved-test
collection. It is useful for operational debugging; the structured files above
are authoritative for analysis.

## Reproduce the statistics

From the project directory, create the environment described in the main
README, then run the 100-seed offline replay into a separate working directory:

```powershell
New-Item -ItemType Directory -Force artifacts/reproduced | Out-Null

python scripts/run_static_baselines.py `
  --panel data/retry-aware-baseline/panels/development_panel.jsonl `
  --evaluation-panel data/retry-aware-baseline/panels/reserved_test_panel.jsonl `
  --max-purchases 2700 `
  --seed 0 `
  --seeds 100 `
  --budget-points 121 `
  --curve-csv artifacts/reproduced/final_held_out_100_seeds_matched_budget_curve.csv `
  --output artifacts/reproduced/final_held_out_100_seeds.json
```

Regenerate the committed summary tables and PNG/SVG figures directly from this
dataset:

```powershell
python scripts/analyze_qian200_results.py `
  --run-dir data/retry-aware-baseline
```

Inspect either canonical ledger without modifying it:

```powershell
python scripts/inspect_experiment.py `
  --database data/retry-aware-baseline/ledgers/development.sqlite3 `
  --recent 20
```

All three commands are offline and make no OpenRouter requests.
