#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# Run Python through uv: it auto-creates/syncs a project venv from pyproject.toml.
PY=(uv run --project "$SCRIPT_DIR" python)

# Load the project's opencode config (registers the web-search plugin) unless
# the user already set one.
export OPENCODE_CONFIG="${OPENCODE_CONFIG:-$SCRIPT_DIR/opencode.json}"

OUTPUT_DIR="$SCRIPT_DIR/standalone_verifier_result"
mkdir -p "$OUTPUT_DIR"

if [ $# -eq 0 ]; then
    # No arguments: default problem + proof from standalone_verifier/
    "${PY[@]}" "$SCRIPT_DIR/verify/verify.py" \
        "$SCRIPT_DIR/standalone_verifier/problem.txt" \
        "$SCRIPT_DIR/standalone_verifier/proof.txt" \
        -o "$OUTPUT_DIR/report.md"
else
    # Pass all arguments through; argparse handles the rest
    "${PY[@]}" "$SCRIPT_DIR/verify/verify.py" \
        "$@" \
        -o "$OUTPUT_DIR/report.md"
fi
