#!/usr/bin/env bash
# Live monitor for the opencode agent driving the QED pipeline.
#
# Run this in a SECOND terminal while run.sh is going:
#     bash monitor.sh            # refresh every 6s
#     bash monitor.sh 3          # refresh every 3s
#
# It follows the newest opencode session (the agent currently running) and
# shows its tool calls (reads / web searches / file writes / bash), the text
# it has written, and how much it has just been thinking. Read-only — it never
# touches the pipeline; it only reads opencode's stored session.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INTERVAL="${1:-6}"

OC="${OPENCODE_BIN:-opencode}"
command -v "$OC" >/dev/null 2>&1 || OC="$HOME/.opencode/bin/opencode"

# --pure: skip plugins (faster, no install, no side effects). Reading sessions
# needs no provider/plugin.
while true; do
    SID="$("$OC" session list --pure </dev/null 2>/dev/null | grep -oE 'ses_[A-Za-z0-9]+' | head -1)"
    out=""
    if [ -n "$SID" ]; then
        out="$("$OC" export "$SID" --pure </dev/null 2>/dev/null \
               | python3 "$SCRIPT_DIR/code/monitor_format.py" 2>/dev/null)"
    fi
    clear 2>/dev/null || printf '\033[2J\033[H'
    echo "=== QED agent monitor — $(date '+%H:%M:%S') — session ${SID:-(none yet)} ==="
    echo "    newest opencode session · refresh ${INTERVAL}s · Ctrl-C to stop"
    echo "------------------------------------------------------------------------"
    if [ -n "$out" ]; then
        echo "$out"
    else
        echo "(waiting for agent activity...)"
    fi
    sleep "$INTERVAL"
done
