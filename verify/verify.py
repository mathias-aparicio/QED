#!/usr/bin/env python3
"""Standalone proof verifier — difficulty-adaptive verification pipeline.

Two modes:
  - Problem + Proof: classifies difficulty, then routes through a 1-agent
    (Easy) or 3-agent (Hard) verification pipeline.
  - Problem only: reviews the problem statement for well-definedness and
    defects (no proof needed).

Usage:
    python verify/verify.py problem.txt proof.txt
    python verify/verify.py problem.txt --problem-only         # problem-only mode
    python verify/verify.py problem.txt proof.txt -o report.md
    python verify/verify.py problem.txt proof.txt --provider gemini
    python verify/verify.py problem.txt proof.txt --model sonnet
    python verify/verify.py problem.txt proof.txt --config /path/to/config.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile

import yaml


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
DEFAULT_CONFIG = os.path.join(REPO_ROOT, "config.yaml")

PROMPT_JUDGE = os.path.join(SCRIPT_DIR, "prompt_judge.md")
PROMPT_STRUCTURAL = os.path.join(SCRIPT_DIR, "prompt_verify_structural.md")
PROMPT_DETAILED = os.path.join(SCRIPT_DIR, "prompt_verify_detailed.md")
PROMPT_CHECK_PROBLEM = os.path.join(SCRIPT_DIR, "prompt_check_problem.md")


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def make_claude_options(claude_cfg: dict) -> dict:
    """Build options dict for the Claude CLI (mirrors pipeline.py)."""
    provider = claude_cfg.get("provider", "subscription")
    env = {}

    if provider == "subscription":
        sub_cfg = claude_cfg.get("subscription", {})
        model = sub_cfg.get("model", "opus")
    elif provider == "api_key":
        api_cfg = claude_cfg.get("api_key", {})
        model = api_cfg.get("model", "claude-opus-4-6")
        key = api_cfg.get("key", "")
        if key:
            env["ANTHROPIC_API_KEY"] = key
        # If key is empty we defer the failure: configs that don't invoke
        # Claude (e.g. all-codex setups) should still load cleanly. If Claude
        # is later invoked, the CLI will surface its own auth error.
    elif provider == "bedrock":
        bedrock_cfg = claude_cfg.get("bedrock", {})
        model = bedrock_cfg.get("model", "us.anthropic.claude-opus-4-6-v1[1m]")
        env["CLAUDE_CODE_USE_BEDROCK"] = "1"
        env["AWS_PROFILE"] = bedrock_cfg.get("aws_profile", "default")
    else:
        raise ValueError(f"Unknown claude.provider '{provider}'.")

    return {
        "cli_path": claude_cfg.get("cli_path", "claude"),
        "model": model,
        "env": env,
    }


def resolve_agent_role_cfg(config: dict, agent_name: str,
                           cli_provider: str | None,
                           cli_model: str | None) -> dict:
    """Return the per-agent role config dict for *agent_name*, with CLI overrides.

    The standalone_verifier section of config.yaml stores each agent as a dict
    of the form ``{provider: 'codex', model?: '...', ...knobs...}``. CLI flags
    ``--provider`` and ``--model`` override the corresponding fields.

    Returns ``{provider, model?, ...other knobs}``.
    """
    sv_cfg = config.get("standalone_verifier", {})
    raw = sv_cfg.get(agent_name)
    if not isinstance(raw, dict) or "provider" not in raw:
        raise ValueError(
            f"standalone_verifier.{agent_name} must be a dict with a 'provider' key "
            f"(e.g. {{provider: 'codex', model: 'gpt-5.5'}}). Got: {raw!r}"
        )
    role_cfg = dict(raw)
    if cli_provider:
        role_cfg["provider"] = cli_provider
    if cli_model:
        role_cfg["model"] = cli_model
    return role_cfg


def merge_provider_section(config: dict, role_cfg: dict) -> tuple[str, dict]:
    """Resolve a role_cfg against the global codex/gemini/claude section.

    Returns ``(provider, merged_provider_cfg)``. The merged dict has every
    knob from the global section, overlaid with per-agent overrides.

    For ``claude``, the per-agent ``model`` override slots into whichever
    auth block (subscription/api_key/bedrock) ``claude.provider`` selects.
    """
    provider = role_cfg["provider"].lower().strip()
    if provider not in ("claude", "codex", "gemini", "opencode"):
        raise ValueError(
            f"Unknown provider {provider!r}; expected 'claude', 'codex', "
            f"'gemini', or 'opencode'."
        )
    overrides = {k: v for k, v in role_cfg.items() if k != "provider"}
    global_section = config.get(provider, {})

    if provider == "claude":
        merged = dict(global_section)
        auth_mode = merged.get("provider", "subscription")
        if "model" in overrides:
            sub = dict(merged.get(auth_mode, {}))
            sub["model"] = overrides["model"]
            merged[auth_mode] = sub
        for k, v in overrides.items():
            if k == "model":
                continue
            merged[k] = v
        return provider, merged

    merged = {**global_section, **overrides}
    return provider, merged


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------

def load_prompt(path: str, **kwargs) -> str:
    with open(path) as f:
        template = f.read()
    return template.format(**kwargs)


# ---------------------------------------------------------------------------
# Model invocation (mirrors model_runner.py, synchronous for simplicity)
# ---------------------------------------------------------------------------

def run_claude(prompt: str, claude_opts: dict, model_override: str | None = None) -> str:
    """Invoke Claude CLI and return response text."""
    cli_path = claude_opts.get("cli_path", "claude")
    model = model_override or claude_opts.get("model", "opus")
    extra_env = claude_opts.get("env", {})

    cmd = [
        cli_path,
        "-p",
        "--output-format", "json",
        "--dangerously-skip-permissions",
        "--model", model,
        prompt,
    ]

    _PROVIDER_VARS = ("CLAUDE_CODE_USE_BEDROCK", "ANTHROPIC_API_KEY",
                      "AWS_PROFILE", "ANTHROPIC_MODEL")
    env = {k: v for k, v in os.environ.items() if k not in _PROVIDER_VARS}
    env.update(extra_env)

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, env=env)

    if result.returncode != 0:
        print(f"[Claude] Non-zero exit code: {result.returncode}", file=sys.stderr)
        if result.stderr.strip():
            print(f"[Claude] stderr: {result.stderr.strip()[:500]}", file=sys.stderr)

    try:
        data = json.loads(result.stdout)
        response = data.get("result", "")
    except (json.JSONDecodeError, ValueError):
        response = result.stdout.strip()

    if not response.strip():
        raise RuntimeError(f"Claude returned empty response. "
                           f"Exit code: {result.returncode}")
    return response


def run_codex(prompt: str, codex_cfg: dict, model_override: str | None = None) -> str:
    """Invoke Codex CLI and return response text."""
    cli_path = codex_cfg.get("cli_path", "codex")
    model = model_override or codex_cfg.get("model", "gpt-5.5")
    reasoning = codex_cfg.get("reasoning_effort", "xhigh")

    cmd = [
        cli_path,
        "--search",
        "-m", model,
        "-c", f'model_reasoning_effort="{reasoning}"',
        "exec",
        "--json",
        "--dangerously-bypass-approvals-and-sandbox",
        "-C", os.getcwd(),
        prompt,
    ]

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True)

    response = ""
    try:
        lines = result.stdout.strip().split("\n")
        events = [json.loads(line) for line in lines if line.strip()]
        for event in events:
            if event.get("type") == "item.completed":
                item = event.get("item", {})
                if item.get("type") == "agent_message":
                    response = item.get("text", "")
    except (json.JSONDecodeError, ValueError):
        response = result.stdout.strip()

    # Codex sometimes exits non-zero (warning) while still producing a valid
    # agent_message. Treat the response as authoritative.
    if result.returncode != 0:
        msg = (f"[Codex] Non-zero exit code: {result.returncode}"
               + (" (treating as warning since response was parsed)"
                  if response.strip() else ""))
        print(msg, file=sys.stderr)
        if result.stderr.strip():
            print(f"[Codex] stderr: {result.stderr.strip()[:500]}", file=sys.stderr)

    if not response.strip():
        raise RuntimeError(f"Codex returned empty response. "
                           f"Exit code: {result.returncode}")
    return response


def run_gemini(prompt: str, gemini_cfg: dict, model_override: str | None = None) -> str:
    """Invoke Gemini CLI and return response text."""
    cli_path = gemini_cfg.get("cli_path", "gemini")
    model = model_override or gemini_cfg.get("model", "gemini-3.1-pro-preview")
    api_key = gemini_cfg.get("api_key", "")
    approval_mode = gemini_cfg.get("approval_mode", "yolo")
    thinking_level = gemini_cfg.get("thinking_level", "")
    thinking_budget = gemini_cfg.get("thinking_budget")

    cmd = [
        cli_path,
        "-m", model,
        "--approval-mode", approval_mode,
        "-o", "json",
        "-p", prompt,
    ]

    env = os.environ.copy()
    if api_key:
        env["GEMINI_API_KEY"] = api_key

    thinking_config = {}
    if thinking_level:
        thinking_config["thinkingLevel"] = thinking_level
    if thinking_budget is not None:
        thinking_config["thinkingBudget"] = thinking_budget

    if thinking_config:
        with tempfile.TemporaryDirectory(prefix="qed-gemini-home-") as gemini_home:
            settings_dir = os.path.join(gemini_home, ".gemini")
            os.makedirs(settings_dir, exist_ok=True)
            settings_path = os.path.join(settings_dir, "settings.json")
            settings = {
                "modelConfigs": {
                    "overrides": [{
                        "match": {"model": model},
                        "modelConfig": {
                            "generateContentConfig": {
                                "thinkingConfig": thinking_config,
                            }
                        },
                    }]
                }
            }
            with open(settings_path, "w") as f:
                json.dump(settings, f)
            env["GEMINI_CLI_HOME"] = gemini_home
            result = subprocess.run(cmd, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, text=True, env=env)
    else:
        result = subprocess.run(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True, env=env)

    if result.returncode != 0:
        print(f"[Gemini] Non-zero exit code: {result.returncode}", file=sys.stderr)
        if result.stderr.strip():
            print(f"[Gemini] stderr: {result.stderr.strip()[:500]}", file=sys.stderr)

    try:
        data = json.loads(result.stdout)
        response = data.get("response", "")
    except (json.JSONDecodeError, ValueError):
        response = result.stdout.strip()

    if not response.strip():
        raise RuntimeError(f"Gemini returned empty response. "
                           f"Exit code: {result.returncode}")
    return response


def _opencode_session_id(stdout: str, stderr: str) -> str | None:
    """Recover the opencode session id from a `run` invocation's output."""
    for line in (stdout or "").strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(ev, dict):
            sid = ev.get("sessionID")
            if sid:
                return sid
            part = ev.get("part")
            if isinstance(part, dict) and part.get("sessionID"):
                return part["sessionID"]
    m = re.search(r"ses_[A-Za-z0-9]+", stderr or "")
    return m.group(0) if m else None


def run_opencode(prompt: str, opencode_cfg: dict, model_override: str | None = None) -> str:
    """Invoke opencode CLI and return response text.

    opencode doesn't flush the assistant reply to stdout when piped (only the
    ``step_start`` event, which carries the session id). So we run, recover the
    session id, then read the assistant text back via ``opencode export``.
    """
    cli_path = opencode_cfg.get("cli_path", "opencode")
    model = model_override or opencode_cfg.get("model", "google/gemini-3-flash-preview")
    variant = opencode_cfg.get("variant", "")
    agent_name = opencode_cfg.get("agent", "")
    api_key = opencode_cfg.get("api_key", "")
    timeout = opencode_cfg.get("timeout", 1800)

    cmd = [
        cli_path, "run", "--format", "json",
        "--dangerously-skip-permissions", "-m", model, "--dir", os.getcwd(),
    ]
    if variant:
        cmd += ["--variant", variant]
    if agent_name:
        cmd += ["--agent", agent_name]
    cmd.append(prompt)

    env = os.environ.copy()
    if api_key:
        env["GEMINI_API_KEY"] = api_key
    # Default to the project's opencode config (web-search plugin) if unset.
    if "OPENCODE_CONFIG" not in env:
        repo_cfg = os.path.join(REPO_ROOT, "opencode.json")
        if os.path.exists(repo_cfg):
            env["OPENCODE_CONFIG"] = repo_cfg

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            stdin=subprocess.DEVNULL, text=True, env=env, timeout=timeout)
    if result.returncode != 0 and result.stderr.strip():
        print(f"[OpenCode] stderr: {result.stderr.strip()[:500]}", file=sys.stderr)

    sid = _opencode_session_id(result.stdout, result.stderr)
    response = ""
    if sid:
        er = subprocess.run([cli_path, "export", sid], stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, stdin=subprocess.DEVNULL,
                            text=True, env=env, timeout=180)
        try:
            data = json.loads(er.stdout)
            last_text: list[str] = []
            for msg in data.get("messages", []):
                if msg.get("info", {}).get("role") != "assistant":
                    continue
                texts = [p.get("text", "") for p in msg.get("parts", [])
                         if isinstance(p, dict) and p.get("type") == "text"
                         and isinstance(p.get("text"), str)]
                if texts:
                    last_text = texts
            response = "".join(last_text).strip()
        except (json.JSONDecodeError, ValueError):
            response = ""

    if not response.strip():
        raise RuntimeError("opencode returned empty response (model produced no text; "
                           "often a transient provider HTTP 503).")
    return response


def run_model_for_role(role_cfg: dict, prompt: str, config: dict) -> str:
    """Dispatch a prompt using a per-agent role config dict.

    Merges the role's overrides into the global codex/gemini/claude section,
    then runs the chosen provider's CLI.
    """
    provider, merged_section = merge_provider_section(config, role_cfg)
    model_override = role_cfg.get("model")
    if provider == "claude":
        # Build claude_opts from the merged section (so api_key/bedrock auth
        # comes from the global section, and the model override from role_cfg).
        claude_opts = make_claude_options(merged_section)
        return run_claude(prompt, claude_opts)
    elif provider == "codex":
        return run_codex(prompt, merged_section, model_override)
    elif provider == "gemini":
        return run_gemini(prompt, merged_section, model_override)
    elif provider == "opencode":
        return run_opencode(prompt, merged_section, model_override)
    else:
        raise ValueError(f"Unknown provider: {provider!r}. "
                         f"Expected 'claude', 'codex', 'gemini', or 'opencode'.")


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

def parse_difficulty(judge_output: str) -> str:
    """Extract 'Easy' or 'Hard' from the judge output."""
    match = re.search(r'\*\*Difficulty:\*\*\s*(Easy|Hard)', judge_output, re.IGNORECASE)
    if match:
        return match.group(1).capitalize()
    # Fallback: look for the word in the first few lines
    for line in judge_output.split("\n")[:20]:
        if "easy" in line.lower() and "difficult" in line.lower():
            continue
        if re.search(r'\bEasy\b', line):
            return "Easy"
        if re.search(r'\bHard\b', line):
            return "Hard"
    # Default to Hard (safer — triggers full pipeline)
    return "Hard"


def parse_structural_verdict(structural_output: str) -> str:
    """Extract PASS/FAIL from structural verification output."""
    match = re.search(r'Overall Structural Verdict:\s*(PASS|FAIL)',
                      structural_output, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    # Fallback: look for any Overall Verdict
    match = re.search(r'Overall\s+Verdict:\s*(PASS|FAIL)',
                      structural_output, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    # Default to FAIL (safer)
    return "FAIL"


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------

def assemble_report(difficulty: str, judge_output: str,
                    structural_output: str | None = None,
                    structural_verdict: str | None = None,
                    detailed_output: str | None = None) -> str:
    """Combine agent outputs into a single final report."""
    sections = []

    if difficulty == "Easy":
        # Judge output already contains the full report
        sections.append(judge_output)
    else:
        # Hard path: combine all available outputs
        sections.append("# Standalone Proof Verification Report")
        sections.append("")
        sections.append("**Difficulty:** Hard")
        sections.append("**Pipeline:** Judge -> Structural -> Detailed")
        sections.append("")

        # Extract judge rationale
        rationale_match = re.search(
            r'\*\*Rationale:\*\*\s*(.+)', judge_output)
        if rationale_match:
            sections.append(f"**Rationale:** {rationale_match.group(1).strip()}")
            sections.append("")

        sections.append("---")
        sections.append("")

        if structural_output:
            sections.append("# Structural Verification")
            sections.append("")
            sections.append(structural_output)
            sections.append("")

        if structural_verdict == "FAIL":
            sections.append("---")
            sections.append("")
            sections.append("*Detailed verification skipped — "
                            "structural verification FAILED.*")
        elif detailed_output:
            sections.append("---")
            sections.append("")
            sections.append("# Detailed Verification")
            sections.append("")
            sections.append(detailed_output)

    return "\n".join(sections)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def write_report(report: str, output_path: str | None):
    """Write report to file or stdout."""
    if output_path:
        with open(output_path, "w") as f:
            f.write(report)
        print(f"Report written to {output_path}", file=sys.stderr)
    else:
        print(report)


def run_problem_only(args):
    """Problem-only mode: review the problem statement for defects."""
    if not os.path.isfile(args.problem):
        print(f"Error: problem file not found: {args.problem}", file=sys.stderr)
        sys.exit(1)

    with open(args.problem) as f:
        problem_text = f.read()

    config = load_config(args.config)

    role_cfg = resolve_agent_role_cfg(
        config, "problem_reviewer", args.provider, args.model)

    print(f"[Problem Review] Checking problem statement ({role_cfg['provider']})",
          file=sys.stderr)
    prompt = load_prompt(PROMPT_CHECK_PROBLEM, problem=problem_text)
    report = run_model_for_role(role_cfg, prompt, config)

    print(f"[Problem Review] Done", file=sys.stderr)
    write_report(report, args.output)


def output_dir_from_args(args) -> str | None:
    """Derive the output directory from -o path. Returns None if no -o."""
    if args.output is None:
        return None
    return os.path.dirname(args.output) or "."


def run_verification(args):
    """Full verification mode: problem + proof."""
    with open(args.problem) as f:
        problem_text = f.read()
    with open(args.proof) as f:
        proof_text = f.read()

    config = load_config(args.config)
    out_dir = output_dir_from_args(args)

    # --- Agent 1: Difficulty Judge ---
    judge_role = resolve_agent_role_cfg(
        config, "judge", args.provider, args.model)

    print(f"[Agent 1] Difficulty Judge ({judge_role['provider']})", file=sys.stderr)
    judge_prompt = load_prompt(PROMPT_JUDGE,
                               problem=problem_text, proof=proof_text)
    judge_output = run_model_for_role(judge_role, judge_prompt, config)

    difficulty = parse_difficulty(judge_output)
    print(f"[Agent 1] Difficulty: {difficulty}", file=sys.stderr)

    # --- Easy path: done ---
    if difficulty == "Easy":
        report = assemble_report("Easy", judge_output)
        write_report(report, args.output)
        return

    # --- Hard path: Agent 2 (structural) ---
    struct_role = resolve_agent_role_cfg(
        config, "structural_verifier", args.provider, args.model)

    print(f"[Agent 2] Structural Verifier ({struct_role['provider']})", file=sys.stderr)
    struct_prompt = load_prompt(PROMPT_STRUCTURAL,
                                problem=problem_text, proof=proof_text)
    structural_output = run_model_for_role(struct_role, struct_prompt, config)

    # Write structural report to its own file
    if out_dir:
        write_report(structural_output,
                     os.path.join(out_dir, "structural_report.md"))

    structural_verdict = parse_structural_verdict(structural_output)
    print(f"[Agent 2] Structural Verdict: {structural_verdict}", file=sys.stderr)

    # --- Gate: if structural FAIL, skip detailed ---
    if structural_verdict == "FAIL":
        report = assemble_report("Hard", judge_output,
                                 structural_output, structural_verdict)
        write_report(report, args.output)
        return

    # --- Hard path: Agent 3 (detailed) ---
    detail_role = resolve_agent_role_cfg(
        config, "detailed_verifier", args.provider, args.model)

    print(f"[Agent 3] Detailed Verifier ({detail_role['provider']})", file=sys.stderr)
    detail_prompt = load_prompt(PROMPT_DETAILED,
                                problem=problem_text, proof=proof_text,
                                structural_report=structural_output)
    detailed_output = run_model_for_role(detail_role, detail_prompt, config)

    print(f"[Agent 3] Done", file=sys.stderr)

    # Write detailed report to its own file
    if out_dir:
        write_report(detailed_output,
                     os.path.join(out_dir, "detailed_report.md"))

    # --- Assemble combined report ---
    report = assemble_report("Hard", judge_output,
                             structural_output, structural_verdict,
                             detailed_output)
    write_report(report, args.output)


def main():
    parser = argparse.ArgumentParser(
        description="Standalone proof verifier — "
                    "difficulty-adaptive verification pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("problem", help="Path to problem statement file")
    parser.add_argument("proof", nargs="?", default=None,
                        help="Path to proof file")
    parser.add_argument("--problem-only", action="store_true",
                        help="Review problem statement only (no proof needed)")
    parser.add_argument("-o", "--output", default=None,
                        help="Write report to file (default: stdout)")
    parser.add_argument("--provider", default=None,
                        choices=["claude", "codex", "gemini", "opencode"],
                        help="Override all agents to one provider")
    parser.add_argument("-m", "--model", default=None,
                        help="Override model name for all agents")
    parser.add_argument("-c", "--config", default=DEFAULT_CONFIG,
                        help="Path to config.yaml "
                             f"(default: {DEFAULT_CONFIG})")

    args = parser.parse_args()

    # --- Validate common inputs ---
    if not os.path.isfile(args.problem):
        print(f"Error: problem file not found: {args.problem}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(args.config):
        print(f"Error: config file not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    # --- Route to the appropriate mode ---
    if args.problem_only:
        run_problem_only(args)
    else:
        if args.proof is None:
            print("Error: proof file is required (or use --problem-only)",
                  file=sys.stderr)
            sys.exit(1)
        if not os.path.isfile(args.proof):
            print(f"Error: proof file not found: {args.proof}", file=sys.stderr)
            sys.exit(1)
        run_verification(args)


if __name__ == "__main__":
    main()
