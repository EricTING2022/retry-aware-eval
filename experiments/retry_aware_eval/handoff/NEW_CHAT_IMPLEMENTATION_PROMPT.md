# Role and objective

You are a software-engineering and experimental-research agent. Complete and
productionize the existing retry-aware MathQA evaluation scaffold in
`experiments/retry_aware_eval`. Preserve scientific reproducibility and make
the smallest maintainable changes needed to run Qian Xie's October 5 experiment
through OpenRouter.

# Context

Read these files before changing code:

- `meeting minutes/minutes1.md`
- `experiments/retry_aware_eval/README.md`
- `experiments/retry_aware_eval/docs/runtime_prompts.md`
- the current package and tests under `experiments/retry_aware_eval`

The three main candidates are gpt-oss-20b, Ministral 3 8B 2512, and Qwen3-32B
using the exact OpenRouter slugs frozen in `src/retry_eval/profiles.py`. Solver
and verifier roles choose independently on every eligible round. The fixed
Gemini 2.5 Flash-Lite critic remains a separate reference condition.

# Instructions

- Keep `max_retries` parameterized. For `r` retries, execute at most `r+1`
  solver calls and `r` verifier calls.
- After each non-final solver answer, call the configured verifier. Stop the
  current question only when it returns `{"accepted": true}`.
- When the retry limit is reached, return the final solver answer without a
  verifier call.
- Grade the returned answer independently against the MathQA answer key. Never
  expose the key to a solver or verifier.
- Preserve explicit round-level SQLite records for executed, failed, skipped,
  and not-applicable call slots. Retain tokens, actual reported USD cost,
  requested/returned model, provider, timing, raw evidence, and nullable verdicts.
- Support equal-call, model-fixed, and actual-token cost accounting from the
  same executed-call records. Keep provider-reported cost as an audit field.
- Freeze prompts, model/provider settings, pricing metadata, dataset identity,
  and code/config versions for each paid run.
- Keep solver models in non-thinking mode wherever it is optional. OpenRouter
  marks gpt-oss-20b reasoning as mandatory, so use its lowest supported effort.
- Do not make paid OpenRouter calls until you have produced an offline plan,
  estimated its cost and request count, and received explicit authorization.
- Prefer small domain modules and dependency injection over framework-specific
  global state. Do not add features unrelated to this experiment.

# Acceptance criteria

1. Three solver and three verifier candidates generate exactly 27 unique
   configurations for one retry and 243 for two retries, including critic
   changes across rounds.
2. Tests prove early acceptance executes no later calls and marks their slots
   as skipped rather than rejected.
3. Tests prove retry exhaustion makes no final verifier call and can receive a
   correct MathQA grade even when every earlier verifier rejected.
4. `max_retries=0`, `1`, and `2` work through the same runner without special
   fixed workflows.
5. The exported record distinguishes `accepted=false`, not called, failed, and
   final-verifier-not-applicable, while retaining all required per-call fields.
6. All cost views count only executed calls and reproduce hand-calculated test
   fixtures.
7. Existing relevant SQLite data is inspected read-only and imported only when
   its semantics match; fixed-verifier legacy data is never mixed with the new
   variable-verifier experiment.
8. The full offline test suite passes, and README commands reproduce a minimal
   inspectable record without network access.

# Deliverables

- Maintainable implementation and migrations under `experiments/retry_aware_eval`.
- Offline tests covering the acceptance criteria.
- A preview command that reports configuration counts, planned calls, model
  settings, and estimated cost without loading an API key.
- A guarded batch command and exports for the approved experiment.
- A concise README update describing how to verify offline and how to prepare,
  approve, run, and audit a paid experiment.

