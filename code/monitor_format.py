#!/usr/bin/env python3
"""Format an `opencode export` JSON (stdin) into a compact activity feed.

Used by ../monitor.sh to show what the agent driving the pipeline is actually
doing: tool calls (reads, web searches, file writes, bash), the text it has
written, and how much it has just been thinking. Read-only; never fails loudly
(a partial/locked export just yields no output).
"""
import json
import sys


def _target(inp: dict) -> str:
    if not isinstance(inp, dict):
        return ""
    for k in ("filePath", "path", "query", "command", "pattern", "url", "description"):
        v = inp.get(k)
        if v:
            return str(v)
    return ""


def main() -> None:
    try:
        data = json.load(sys.stdin)
    except Exception:
        return  # partial / locked export — caller shows "waiting..."

    info = data.get("info", {}) if isinstance(data, dict) else {}
    toks = info.get("tokens") or {}
    cost = info.get("cost")
    if info.get("title"):
        print(f"Task: {info['title']}")
    if isinstance(toks, dict):
        line = f"tokens in/out: {toks.get('input', 0)}/{toks.get('output', 0)}"
        if isinstance(cost, (int, float)):
            line += f"   cost: ${cost:.4f}"
        print(line)
    print()

    reasoning = 0
    lines: list[str] = []
    for msg in data.get("messages", []):
        role = msg.get("info", {}).get("role")
        for p in msg.get("parts", []):
            t = p.get("type")
            if t == "tool":
                st = p.get("state", {}) if isinstance(p.get("state"), dict) else {}
                tgt = _target(st.get("input", {}))
                status = st.get("status", "")
                lines.append(f"  * {p.get('tool', 'tool'):<16} {tgt[:80]}  [{status}]")
            elif t == "text" and role == "assistant":
                txt = " ".join((p.get("text") or "").split())
                if txt:
                    lines.append(f"  > {txt[:110]}")
            elif t == "reasoning":
                reasoning += 1

    # Show the most recent activity (keeps the screen focused on "now").
    for ln in lines[-25:]:
        print(ln)
    if reasoning:
        print(f"\n  (... {reasoning} thinking steps so far)")


if __name__ == "__main__":
    main()
