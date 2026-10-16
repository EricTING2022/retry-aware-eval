#!/usr/bin/env bash
# Run the 100-development/100-reserved-test protocol for Qian's four baselines.

set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
WORKERS="${WORKERS:-4}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-300}"
RUN_DIR="${RUN_DIR:-data/retry-aware-baseline}"
BENCHMARK="${BENCHMARK:-data/mathqa/test_200.jsonl}"
LEDGER_DIR="$RUN_DIR/ledgers"
AUDIT_DIR="$RUN_DIR/audit-logs"
PANEL_DIR="$RUN_DIR/panels"
SEARCH_DIR="$RUN_DIR/search"
SUMMARY_DIR="$RUN_DIR/summaries"
LOG_DIR="$RUN_DIR/logs"
PIPELINE_LOG="$LOG_DIR/pipeline_console.log"

DEV_DB="$LEDGER_DIR/development.sqlite3"
DEV_SUMMARY="$SUMMARY_DIR/development_summary.json"
DEV_DETAIL="$AUDIT_DIR/development_detailed.jsonl"
DEV_PANEL="$PANEL_DIR/development_panel.jsonl"
DEV_SEARCH="$SEARCH_DIR/development_search.json"
DEV_CURVE="$SEARCH_DIR/development_search_matched_budget_curve.csv"
SELECTED="$SEARCH_DIR/selected_configurations.json"

TEST_DB="$LEDGER_DIR/reserved_test.sqlite3"
TEST_SUMMARY="$SUMMARY_DIR/reserved_test_summary.json"
TEST_DETAIL="$AUDIT_DIR/reserved_test_detailed.jsonl"
TEST_PANEL="$PANEL_DIR/reserved_test_panel.jsonl"
FINAL_RESULT="$SEARCH_DIR/final_held_out.json"
FINAL_CURVE="$SEARCH_DIR/final_held_out_matched_budget_curve.csv"

if [[ "${OPENROUTER_API_KEY:-}" != sk-* ]]; then
  echo "OPENROUTER_API_KEY is missing or does not start with sk-." >&2
  exit 2
fi
if [[ ! -f "$BENCHMARK" ]]; then
  echo "Benchmark not found: $BENCHMARK" >&2
  exit 2
fi
mkdir -p "$LEDGER_DIR" "$AUDIT_DIR" "$PANEL_DIR" "$SEARCH_DIR" \
  "$SUMMARY_DIR" "$LOG_DIR"

log_command() {
  echo
  echo "+ $*" | tee -a "$PIPELINE_LOG"
  "$@" 2>&1 | tee -a "$PIPELINE_LOG"
}

only_run_id() {
  "$PYTHON_BIN" -c 'import sqlite3, sys
rows = sqlite3.connect(sys.argv[1]).execute("SELECT run_id FROM runs ORDER BY created_at_utc").fetchall()
if len(rows) != 1:
    raise SystemExit(f"Expected exactly one run in {sys.argv[1]}, found {len(rows)}")
print(rows[0][0])' "$1"
}

summary_run_id() {
  "$PYTHON_BIN" -c 'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["run_id"])' "$1"
}

collect_partition() {
  local partition="$1"
  local database="$2"
  local summary="$3"
  local detailed="$4"
  shift 4
  local selection_args=("$@")

  if [[ -f "$summary" ]]; then
    echo "Skipping completed $partition collection: $summary" | tee -a "$PIPELINE_LOG"
    return
  fi

  local resume_args=()
  if [[ -f "$database" ]]; then
    resume_args=(--resume-run-id "$(only_run_id "$database")")
    if [[ -f "$detailed" ]]; then
      local backup="${detailed%.jsonl}.before_resume.$(date -u +%Y%m%dT%H%M%SZ).jsonl"
      mv -- "$detailed" "$backup"
      echo "Preserved pre-resume detailed log as $backup" | tee -a "$PIPELINE_LOG"
    fi
  fi

  log_command "$PYTHON_BIN" scripts/run_openrouter_pilot.py \
    --preset qian-27 \
    --partition "$partition" \
    --benchmark "$BENCHMARK" \
    --database "$database" \
    --summary "$summary" \
    --detailed-log "$detailed" \
    --workers "$WORKERS" \
    --timeout-seconds "$TIMEOUT_SECONDS" \
    "${selection_args[@]}" \
    "${resume_args[@]}" \
    --execute
}

echo "Qian 200-question protocol" | tee -a "$PIPELINE_LOG"
echo "development=100 reserved_test=100 workers=$WORKERS timeout_seconds=$TIMEOUT_SECONDS" | tee -a "$PIPELINE_LOG"

# Stage 1: collect every static configuration on development only.
collect_partition development "$DEV_DB" "$DEV_SUMMARY" "$DEV_DETAIL"
DEV_RUN_ID="$(summary_run_id "$DEV_SUMMARY")"

if [[ ! -f "$DEV_PANEL" ]]; then
  log_command "$PYTHON_BIN" scripts/export_static_panel.py \
    --database "$DEV_DB" \
    --run-id "$DEV_RUN_ID" \
    --benchmark "$BENCHMARK" \
    --output "$DEV_PANEL"
fi

if [[ ! -f "$DEV_SEARCH" || ! -f "$DEV_CURVE" ]]; then
  log_command "$PYTHON_BIN" scripts/run_static_baselines.py \
    --panel "$DEV_PANEL" \
    --max-purchases 2700 \
    --seed 0 \
    --seeds 1 \
    --budget-points 121 \
    --curve-csv "$DEV_CURVE" \
    --output "$DEV_SEARCH"
fi

if [[ ! -f "$SELECTED" ]]; then
  log_command "$PYTHON_BIN" scripts/extract_selected_configurations.py \
    --baselines "$DEV_SEARCH" \
    --output "$SELECTED"
fi

# Stage 2: after freezing recommendations, collect only their union on test.
collect_partition reserved_test "$TEST_DB" "$TEST_SUMMARY" "$TEST_DETAIL" \
  --configuration-ids-file "$SELECTED"
TEST_RUN_ID="$(summary_run_id "$TEST_SUMMARY")"

if [[ ! -f "$TEST_PANEL" ]]; then
  log_command "$PYTHON_BIN" scripts/export_static_panel.py \
    --database "$TEST_DB" \
    --run-id "$TEST_RUN_ID" \
    --benchmark "$BENCHMARK" \
    --output "$TEST_PANEL"
fi

if [[ ! -f "$FINAL_RESULT" || ! -f "$FINAL_CURVE" ]]; then
  log_command "$PYTHON_BIN" scripts/run_static_baselines.py \
    --panel "$DEV_PANEL" \
    --evaluation-panel "$TEST_PANEL" \
    --max-purchases 2700 \
    --seed 0 \
    --seeds 1 \
    --budget-points 121 \
    --curve-csv "$FINAL_CURVE" \
    --output "$FINAL_RESULT"
fi

cat <<EOF

Completed. Primary records:
  development SQLite: $DEV_DB
  development workflow JSONL: $DEV_DETAIL
  baseline search trace: $DEV_SEARCH
  matched-budget search curve: $DEV_CURVE
  reserved-test SQLite: $TEST_DB
  reserved-test workflow JSONL: $TEST_DETAIL
  held-out baseline result: $FINAL_RESULT
  held-out search curve: $FINAL_CURVE
  console/progress log: $PIPELINE_LOG
EOF
