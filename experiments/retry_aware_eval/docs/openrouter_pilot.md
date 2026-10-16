# OpenRouter 20-question pilot

This pilot freezes 20 questions from the existing official MathQA test pool:
the first 10 `development` records and the first 10 `reserved_test` records in
`test_200.jsonl`. The exact IDs, source indices, parent hash, and output hash are
stored in `data/mathqa/pilot_20_manifest.json`.

The runner is preview-only by default; `--execute` enables paid requests. It
does not impose a local call-count, cost, or per-request output-token limit.
OpenRouter account credits and provider limits still apply. There are no
automatic application-level retries, so every planned model call maps to one
recorded API request. Independent workflows can be collected concurrently with
`--workers`; start conservatively (for example, 4) and increase only if the
OpenRouter account is not being rate-limited.

## 1. Verify locally without an API key

From `experiments/retry_aware_eval` in PowerShell:

```powershell
$env:PYTHONPATH = (Resolve-Path "src")
python -m unittest discover -s tests -v

python scripts/run_openrouter_pilot.py `
  --preset smoke `
  --benchmark data/mathqa/pilot_20.jsonl `
  --database artifacts/openrouter_smoke_20.sqlite3 `
  --summary artifacts/openrouter_smoke_20_summary.json
```

The preview must report 20 workflows, 40 minimum calls, 60 maximum calls, and
`preview_only=true`. It does not read an API key or create the database.

## 2. Run the least-expensive end-to-end smoke test

Do not paste the secret into chat or commit it. Load it only into the current
PowerShell process:

```powershell
$secureKey = Read-Host "OpenRouter API key" -AsSecureString
$env:OPENROUTER_API_KEY = [Net.NetworkCredential]::new("", $secureKey).Password

python scripts/run_openrouter_pilot.py `
  --preset smoke `
  --benchmark data/mathqa/pilot_20.jsonl `
  --database artifacts/openrouter_smoke_20_retry.sqlite3 `
  --summary artifacts/openrouter_smoke_20_retry_summary.json `
  --workers 4 `
  --execute

Remove-Item Env:OPENROUTER_API_KEY
```

The smoke sequence is intentionally heterogeneous: gpt-oss-20b is the first
solver, Qwen3 32B is the verifier, and Ministral 3 8B is the unverified final
solver after rejection. This exercises early stop, retry exhaustion,
independent grading, and complete request/response/cost recording.

Every paid run also writes a detailed JSONL audit log next to `--summary`. For
example, `openrouter_smoke_20_retry_summary.json` produces
`openrouter_smoke_20_retry_detailed.jsonl`. Use `--detailed-log PATH` to choose
a different location.

Requests use strict JSON schemas and `provider.require_parameters=true`, as
recommended by OpenRouter for structured outputs. There is no `max_tokens`
parameter: this avoids consuming a small shared completion budget entirely on
mandatory reasoning before any visible answer is produced. If OpenRouter still
returns an unusable response, the run stops, but the raw response, provider,
usage, reported cost, finish reason, and elapsed time are retained in SQLite.
GPT-OSS verifier requests additionally exclude the `darkbloom` provider
endpoint. In a 2026-10-16 development run that endpoint returned HTTP 200 and
`finish_reason="stop"`, but placed prose only in `message.reasoning` while
leaving `message.content=null` for 31 verifier calls. Those responses do not
satisfy the declared JSON schema and are never treated as verifier verdicts.
GPT-OSS solver routing is unchanged because no solver call showed this failure.
Qwen3 uses non-reasoning mode in both roles, while mandatory-reasoning
gpt-oss-20b uses its lowest supported effort in both roles. This prevents a
Boolean verifier verdict from spending thousands of tokens on hidden reasoning.

Parallel collection keeps at most `--workers` workflows submitted at once. On
the first technical failure it stops scheduling new workflows and waits only
for the already-running calls. This prevents a failed provider contract from
silently draining the rest of a large panel queue.

Provider routing is part of the frozen run settings. A database created before
the Darkbloom exclusion must remain as audit evidence and must not be resumed
under the new profile. Start a clean run directory instead.

## 3. Inspect and export the records

```powershell
$pilot = Get-Content artifacts/openrouter_smoke_20_retry_summary.json | ConvertFrom-Json
$pilot

python scripts/inspect_experiment.py `
  --database artifacts/openrouter_smoke_20_retry.sqlite3

python scripts/export_detailed_log.py `
  --database artifacts/openrouter_smoke_20_retry.sqlite3 `
  --run-id $pilot.run_id `
  --output artifacts/openrouter_smoke_20_retry_manual_detailed.jsonl

python scripts/export_static_panel.py `
  --database artifacts/openrouter_smoke_20_retry.sqlite3 `
  --run-id $pilot.run_id `
  --benchmark data/mathqa/pilot_20.jsonl `
  --output artifacts/openrouter_smoke_20_panel.jsonl
```

The summary separates development and reserved-test grades. The SQLite ledger
is the audit source of truth and retains each planned, executed, skipped, or
not-applicable call slot. The JSONL file has one run header followed by one
self-contained record per workflow and any archived resume attempts. Each
attempt contains its solver and verifier calls, exact request and response,
visible answer, parsed fields, independent option correctness, verifier verdict,
input/output/reasoning tokens, reported USD cost, provider/model identity,
response ID, HTTP and finish status, call latency, full workflow latency, and
errors. A one-configuration smoke panel validates transport and recording, but
it cannot meaningfully compare configuration-search methods.

Requests are durably recorded before they are sent. If power fails after a
request reaches the provider but before its response is saved, the archived
retry shows an initiated call without response-side usage; local code cannot
reconstruct provider billing that was never received. Existing schema-v1/v2
runs remain readable, but newly added fields are null when the old run never
captured them.

## 4. Optional: build the complete 27-configuration one-retry panel

The same 20 questions are evaluated under all `3^3 = 27` assignments of first
solver, verifier, and final solver. This is 540 workflows and 1,080–1,620 paid
model calls. Preview it before deciding to spend:

```powershell
python scripts/run_openrouter_pilot.py `
  --preset qian-27 `
  --benchmark data/mathqa/pilot_20.jsonl `
  --database artifacts/openrouter_qian27_20.sqlite3 `
  --summary artifacts/openrouter_qian27_20_summary.json
```

Execute only after setting `OPENROUTER_API_KEY` as above:

```powershell
python scripts/run_openrouter_pilot.py `
  --preset qian-27 `
  --benchmark data/mathqa/pilot_20.jsonl `
  --database artifacts/openrouter_qian27_20.sqlite3 `
  --summary artifacts/openrouter_qian27_20_summary.json `
  --workers 4 `
  --execute
```

If the process or computer stops, use the same database and the printed
`run_id`. Completed configuration-question workflows are skipped; interrupted
or failed workflows are retried, and their previous call ledger is archived in
`execution_retries`:

```powershell
python scripts/run_openrouter_pilot.py `
  --preset qian-27 `
  --benchmark data/mathqa/pilot_20.jsonl `
  --database artifacts/openrouter_qian27_20.sqlite3 `
  --summary artifacts/openrouter_qian27_20_summary.json `
  --resume-run-id RUN_ID `
  --workers 4 `
  --execute
```

Then export the balanced panel and run all four offline baselines:

```powershell
$pilot = Get-Content artifacts/openrouter_qian27_20_summary.json | ConvertFrom-Json
python scripts/export_static_panel.py `
  --database artifacts/openrouter_qian27_20.sqlite3 `
  --run-id $pilot.run_id `
  --benchmark data/mathqa/pilot_20.jsonl `
  --output artifacts/openrouter_qian27_20_panel.jsonl

python scripts/run_static_baselines.py `
  --panel artifacts/openrouter_qian27_20_panel.jsonl `
  --max-purchases 100 `
  --seed 0 `
  --seeds 1 `
  --held-out `
  --output artifacts/openrouter_qian27_20_baselines.json
```

For the strict 100-development/100-reserved protocol, use two paid stages.
First collect all 27 configurations only on development questions, export the
panel, and run one search seed (seed 0):

```powershell
python scripts/run_openrouter_pilot.py `
  --preset qian-27 `
  --partition development `
  --benchmark data/mathqa/test_200.jsonl `
  --database artifacts/qian27_development.sqlite3 `
  --summary artifacts/qian27_development_summary.json `
  --workers 4 `
  --execute

$development = Get-Content artifacts/qian27_development_summary.json | ConvertFrom-Json
python scripts/export_static_panel.py `
  --database artifacts/qian27_development.sqlite3 `
  --run-id $development.run_id `
  --benchmark data/mathqa/test_200.jsonl `
  --output artifacts/qian27_development_panel.jsonl

python scripts/run_static_baselines.py `
  --panel artifacts/qian27_development_panel.jsonl `
  --max-purchases 2700 `
  --seed 0 `
  --seeds 1 `
  --output artifacts/qian27_development_search.json

python scripts/extract_selected_configurations.py `
  --baselines artifacts/qian27_development_search.json `
  --output artifacts/qian27_selected_configurations.json
```

Only after the recommendations are frozen, collect their union on reserved
test questions, export it, and evaluate:

```powershell
python scripts/run_openrouter_pilot.py `
  --preset qian-27 `
  --partition reserved_test `
  --configuration-ids-file artifacts/qian27_selected_configurations.json `
  --benchmark data/mathqa/test_200.jsonl `
  --database artifacts/qian27_reserved_test.sqlite3 `
  --summary artifacts/qian27_reserved_test_summary.json `
  --workers 4 `
  --execute

$test = Get-Content artifacts/qian27_reserved_test_summary.json | ConvertFrom-Json
python scripts/export_static_panel.py `
  --database artifacts/qian27_reserved_test.sqlite3 `
  --run-id $test.run_id `
  --benchmark data/mathqa/test_200.jsonl `
  --output artifacts/qian27_reserved_test_panel.jsonl

python scripts/run_static_baselines.py `
  --panel artifacts/qian27_development_panel.jsonl `
  --evaluation-panel artifacts/qian27_reserved_test_panel.jsonl `
  --max-purchases 2700 `
  --seed 0 `
  --seeds 1 `
  --output artifacts/qian27_final_held_out.json
```

The development stage has 2,700 workflows; the reserved stage has 100 times
the number of distinct configurations selected across the four methods. This
held-out extension is distinct from `main.tex`'s single finite-panel
cost-oracle reference, and the output labels preserve that distinction.

Current model identifiers should always be confirmed on the official OpenRouter
model pages before a paid run. The model pool is `openai/gpt-oss-20b`,
`mistralai/ministral-8b-2512`, and `qwen/qwen3-32b`. OpenRouter marks
gpt-oss-20b reasoning as mandatory, so both roles use its lowest supported
effort (`low`); Qwen3 explicitly disables reasoning in both roles. The separate
fixed-critic Gemini reference is not part of this pool.

Compatibility was checked against OpenRouter's public model catalog on October
16, 2026:

- <https://openrouter.ai/docs/guides/best-practices/reasoning-tokens>
- <https://openrouter.ai/docs/guides/features/structured-outputs>
- <https://openrouter.ai/openai/gpt-oss-20b>
- <https://openrouter.ai/mistralai/ministral-8b-2512>
- <https://openrouter.ai/qwen/qwen3-32b>
- <https://openrouter.ai/api/v1/models>
