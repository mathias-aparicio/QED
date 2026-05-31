#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# Run Python through uv: it auto-creates/syncs a project venv from pyproject.toml.
PY=(uv run --project "$SCRIPT_DIR" python)

# Load the project's opencode config (registers the web-search plugin) unless
# the user already set one. The plugin auto-installs on the first opencode run.
export OPENCODE_CONFIG="${OPENCODE_CONFIG:-$SCRIPT_DIR/opencode.json}"

PROBLEM_FILE="${1:-$SCRIPT_DIR/problem/problem.tex}"
OUTPUT_DIR="${2:-$SCRIPT_DIR/proof_output}"
CONFIG="${3:-$SCRIPT_DIR/config.yaml}"

echo "============================================================"
echo "  Running smoke tests..."
echo "============================================================"
"${PY[@]}" "$SCRIPT_DIR/code/smoke_test.py" --config "$CONFIG"
echo ""

# Copy global human_help into the output directory so the pipeline reads
# everything from proof_output.  Use cp -n to not overwrite files that
# already exist (e.g. edited by the UI before a resume).
mkdir -p "$OUTPUT_DIR/human_help"
for f in "$SCRIPT_DIR/human_help"/*; do
    [ -f "$f" ] && cp -n "$f" "$OUTPUT_DIR/human_help/" 2>/dev/null || true
done

echo "============================================================"
echo "  Proof Agent Pipeline"
echo "============================================================"
echo "  Problem:  $PROBLEM_FILE"
echo "  Output:   $OUTPUT_DIR"
echo "  Config:   $CONFIG"
echo ""

exec "${PY[@]}" "$SCRIPT_DIR/code/pipeline.py" \
    --input "$PROBLEM_FILE" \
    --output "$OUTPUT_DIR" \
    --config "$CONFIG"
