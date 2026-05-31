#!/usr/bin/env python3
"""
Proof Agent pipeline: takes a problem.tex (LaTeX) and produces a natural-language proof.

Three-stage pipeline:
  Stage 0 — Literature Survey agent: deep-dives into the problem context and related results
  Stage 1 — Decomposition prover (see decomposition_prover.py):
    Decomposer → Single Prover → Structural Verifier → Detailed Verifier → Regulator
    with a three-level retry hierarchy (REVISE_PROOF / REVISE_PLAN / REWRITE).
  Stage 2 — Summary agent: reads all generated files and writes proof_effort_summary.md

The Stage 0 literature survey can be resumed if interrupted (detected via files
in <output>/related_info/). The decomposition prover manages its own resume
detection inside <output>/decomposition/.
"""

import asyncio
import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime

import yaml


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_prompt(prompts_dir: str, name: str, **kwargs) -> str:
    """Load a prompt template and fill placeholders."""
    path = os.path.join(prompts_dir, name)
    with open(path) as f:
        template = f.read()
    return template.format(**kwargs)


def make_claude_options(claude_cfg: dict, working_dir: str) -> dict:
    """Build options dict for the Claude CLI subprocess runner.

    Supports three providers:
      - "subscription": Claude Pro/Max subscription (no keys, shorthand model names)
      - "bedrock": AWS Bedrock (requires AWS credentials)
      - "api_key": Anthropic API key (requires ANTHROPIC_API_KEY)
    """
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
        # If key is empty we defer the failure: pipelines that don't actually
        # invoke Claude (e.g. all-codex configs) should still be able to start.
        # If Claude is later invoked, the CLI will surface its own auth error.
    elif provider == "bedrock":
        bedrock_cfg = claude_cfg.get("bedrock", {})
        model = bedrock_cfg.get("model", "us.anthropic.claude-opus-4-6-v1[1m]")
        env["CLAUDE_CODE_USE_BEDROCK"] = "1"
        env["AWS_PROFILE"] = bedrock_cfg.get("aws_profile", "default")
    else:
        raise ValueError(f"config.yaml: unknown claude.provider '{provider}'. Use 'subscription', 'bedrock', or 'api_key'.")

    return {
        "cli_path": claude_cfg.get("cli_path", "claude"),
        "model": model,
        "cwd": working_dir,
        "env": env,
    }


def check_prerequisites():
    """Check that required tools are available.

    The model backend (opencode) is validated config-aware in smoke_test.py
    (which run.sh runs first, using the configured cli_path), so here we only
    require the Python runtime.
    """
    missing = []
    for cmd in ["python3"]:
        if shutil.which(cmd) is None:
            missing.append(cmd)
    if missing:
        print(f"ERROR: Missing required tools: {', '.join(missing)}")
        print("Please install them before running the pipeline.")
        sys.exit(1)
    try:
        import yaml as _y  # noqa: F401
    except ImportError:
        missing.append("pyyaml (pip install pyyaml)")
    if missing:
        print(f"ERROR: Missing Python packages: {', '.join(missing)}")
        sys.exit(1)


def _file_nonempty(path: str) -> bool:
    """Return True if *path* exists and has non-whitespace content."""
    if not os.path.exists(path):
        return False
    with open(path) as f:
        return bool(f.read().strip())


def _check_expected_files(
    expected: list[tuple[str, str]],
    logger,
    step_name: str,
) -> None:
    """Verify that all expected output files exist after an agent call.

    Args:
        expected: List of (filepath, description) tuples.
        logger: PipelineLogger instance for logging.
        step_name: Name of the pipeline step (for error messages).

    Raises:
        RuntimeError: If any expected file is missing.
    """
    missing = []
    for path, desc in expected:
        if not os.path.exists(path):
            missing.append((path, desc))
    if missing:
        lines = [f"  - {desc}: {path}" for path, desc in missing]
        msg = f"FATAL — {step_name}: expected output file(s) missing:\n" + "\n".join(lines)
        logger.log(msg)
        logger.append_history(f"{step_name}: FATAL — {len(missing)} expected file(s) missing")
        raise RuntimeError(msg)


def _fallback_save_response(
    response: str,
    primary_files: list[str],
    error_files: list[str],
    logger=None,
    step_name: str = "",
) -> None:
    """Save agent response to missing primary output files as fallback.

    If the agent wrote the file via tool use, this is a no-op.
    If the agent didn't, the text response is saved so work isn't lost.
    When fallback triggers, error log files record that the pipeline
    (not the agent) saved the output.
    """
    fallback_used = []
    for path in primary_files:
        if not os.path.exists(path) and response.strip():
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                f.write(response)
            fallback_used.append(path)
            if logger:
                logger.log(f"  Fallback: saved response to {path}")
    # Write or append fallback notice to error files
    if fallback_used:
        notice = (f"\n\n# Pipeline Fallback Notice\n\n"
                  f"**Step:** {step_name}\n\n"
                  f"The agent did not write the following expected output file(s) "
                  f"via tool use. The pipeline saved the agent's text response "
                  f"as a fallback:\n\n")
        for fb in fallback_used:
            notice += f"- `{fb}`\n"
        notice += f"\nThe content may not be properly formatted for this file.\n"
        for path in error_files:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "a") as f:
                f.write(notice)
    else:
        for path in error_files:
            if not os.path.exists(path):
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w") as f:
                    f.write("")


def _parse_difficulty(output_dir: str) -> str:
    """Parse the difficulty classification from difficulty_evaluation.md.

    Looks for a line like '## Classification: Easy' (or Medium / Hard).
    Returns 'easy', 'medium', 'hard', or 'unknown'.
    """
    path = os.path.join(output_dir, "related_info", "difficulty_evaluation.md")
    if not os.path.exists(path):
        return "unknown"
    with open(path) as f:
        for line in f:
            if "classification" in line.lower():
                upper = line.upper()
                if "EASY" in upper:
                    return "easy"
                if "MEDIUM" in upper:
                    return "medium"
                if "HARD" in upper:
                    return "hard"
    return "unknown"


def literature_survey_complete(output_dir: str) -> bool:
    """Return True if the Stage 0 literature survey is already on disk.

    For Easy problems the survey agent writes ``proof.md`` directly and may
    skip ``related_work.md``; in that case the survey is considered complete
    once difficulty_evaluation.md and proof.md exist.
    """
    related_info_dir = os.path.join(output_dir, "related_info")
    difficulty_file = os.path.join(related_info_dir, "difficulty_evaluation.md")
    if not _file_nonempty(difficulty_file):
        return False
    if _parse_difficulty(output_dir) == "easy":
        return _file_nonempty(os.path.join(output_dir, "proof.md"))
    return _file_nonempty(os.path.join(related_info_dir, "related_work.md"))


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

class PipelineLogger:
    """Persistent logging to AUTO_RUN_STATUS.md, .history, and AUTO_RUN_LOG.txt."""

    def __init__(self, log_dir: str, phase: str):
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        self.phase = phase
        self.status_file = os.path.join(log_dir, "AUTO_RUN_STATUS.md")
        self.history_file = os.path.join(log_dir, "AUTO_RUN_STATUS.md.history")
        self.log_file = os.path.join(log_dir, "AUTO_RUN_LOG.txt")
        self.start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.pid = os.getpid()

        # Append to history (not truncate) so resumed runs preserve prior history
        self.append_history(f"{phase} started")

    def update_status(self, iteration: int, max_iter: int, step: str, state: str, details: str):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        history = ""
        if os.path.exists(self.history_file):
            with open(self.history_file) as f:
                history = f.read()
        with open(self.status_file, "w") as f:
            f.write(f"# {self.phase} - Auto Status\n\n")
            f.write("| Field | Value |\n|-------|-------|\n")
            f.write(f"| **Status** | {state} |\n")
            f.write(f"| **Current Iteration** | {iteration} / {max_iter} |\n")
            f.write(f"| **Current Step** | {step} |\n")
            f.write(f"| **Started At** | {self.start_time} |\n")
            f.write(f"| **Last Updated** | {now} |\n")
            f.write(f"| **PID** | {self.pid} |\n\n")
            f.write(f"## Current Activity\n{details}\n\n")
            f.write(f"## Progress History\n{history}\n")

    def append_history(self, msg: str):
        now = datetime.now().strftime("%H:%M:%S")
        with open(self.history_file, "a") as f:
            f.write(f"- [{now}] {msg}\n")

    def log(self, msg: str):
        print(msg)
        with open(self.log_file, "a") as f:
            f.write(msg + "\n")

    def finalize(self, iteration: int, max_iter: int, exit_state: str, details: str):
        self.update_status(iteration, max_iter, exit_state, exit_state, details)
        self.append_history(f"Process ended: {exit_state}")


# ---------------------------------------------------------------------------
# Token usage tracking
# ---------------------------------------------------------------------------

class TokenTracker:
    """Accumulates token usage across all agent calls and persists to disk
    after every update so the user can check TOKEN_USAGE.md at any time.

    Supports multi-provider tracking: each call can specify a provider
    (claude/codex/gemini) and model name. Per-provider subtotals are shown
    in TOKEN_USAGE.md when more than one provider is used.
    """

    def __init__(self, output_dir: str, model: str):
        self.output_dir = output_dir
        self.model = model  # default model label (backward compat)
        self.calls: list[dict] = []
        self.total_input = 0
        self.total_output = 0
        self.total_elapsed = 0.0
        self.per_provider: dict[str, dict] = {}  # provider → {input, output, calls, model}
        self.md_path = os.path.join(output_dir, "TOKEN_USAGE.md")
        self.json_path = os.path.join(output_dir, "token_usage.json")
        self.start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def record(self, call_name: str, input_tokens: int, output_tokens: int,
               elapsed: float, provider: str = "claude", model: str = ""):
        self.total_input += input_tokens
        self.total_output += output_tokens
        self.total_elapsed += elapsed

        # Per-provider tracking
        if provider not in self.per_provider:
            self.per_provider[provider] = {
                "input": 0, "output": 0, "calls": 0,
                "model": model or self.model,
            }
        self.per_provider[provider]["input"] += input_tokens
        self.per_provider[provider]["output"] += output_tokens
        self.per_provider[provider]["calls"] += 1

        self.calls.append({
            "call": len(self.calls) + 1,
            "name": call_name,
            "provider": provider,
            "model": model or self.model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "elapsed_s": round(elapsed, 1),
            "cumul_input": self.total_input,
            "cumul_output": self.total_output,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })
        self._save()

    def _save(self):
        """Write both TOKEN_USAGE.md and token_usage.json."""
        # --- Markdown ---
        lines = [
            "# Token Usage\n",
            f"**Primary Model:** `{self.model}`  ",
            f"**Started:** {self.start_time}  ",
            f"**Last updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n",
            "## Summary\n",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Total input tokens | {self.total_input:,} |",
            f"| Total output tokens | {self.total_output:,} |",
            f"| Total tokens | {self.total_input + self.total_output:,} |",
            f"| Total elapsed | {self.total_elapsed:.0f}s |",
            f"| Agent calls | {len(self.calls)} |\n",
        ]

        # Per-provider summary (only shown when multiple providers used)
        if len(self.per_provider) > 1:
            lines.append("## Per-Provider Summary\n")
            lines.append("| Provider | Model | Input | Output | Total | Calls |")
            lines.append("|----------|-------|------:|-------:|------:|------:|")
            for prov, stats in sorted(self.per_provider.items()):
                total = stats['input'] + stats['output']
                lines.append(
                    f"| {prov} | {stats['model']} "
                    f"| {stats['input']:,} | {stats['output']:,} "
                    f"| {total:,} | {stats['calls']} |"
                )
            lines.append("")

        lines.append("## Per-Call Breakdown\n")
        if len(self.per_provider) > 1:
            lines.append("| # | Agent | Provider | Input | Output | Time | Cumul In | Cumul Out |")
            lines.append("|---|-------|----------|------:|-------:|-----:|---------:|----------:|")
        else:
            lines.append("| # | Agent | Input | Output | Time | Cumul In | Cumul Out |")
            lines.append("|---|-------|------:|-------:|-----:|---------:|----------:|")

        for c in self.calls:
            if len(self.per_provider) > 1:
                lines.append(
                    f"| {c['call']} | {c['name']} | {c.get('provider', 'claude')} "
                    f"| {c['input_tokens']:,} | {c['output_tokens']:,} "
                    f"| {c['elapsed_s']}s "
                    f"| {c['cumul_input']:,} | {c['cumul_output']:,} |"
                )
            else:
                lines.append(
                    f"| {c['call']} | {c['name']} "
                    f"| {c['input_tokens']:,} | {c['output_tokens']:,} "
                    f"| {c['elapsed_s']}s "
                    f"| {c['cumul_input']:,} | {c['cumul_output']:,} |"
                )
        lines.append("")

        with open(self.md_path, "w") as f:
            f.write("\n".join(lines))

        # --- JSON ---
        data = {
            "model": self.model,
            "started": self.start_time,
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_input_tokens": self.total_input,
            "total_output_tokens": self.total_output,
            "total_tokens": self.total_input + self.total_output,
            "total_elapsed_s": round(self.total_elapsed, 1),
            "per_provider": self.per_provider,
            "calls": self.calls,
        }
        with open(self.json_path, "w") as f:
            json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# Agent runners
# ---------------------------------------------------------------------------

async def run_agent(
    claude_opts: dict,
    prompt: str,
    logger: PipelineLogger | None = None,
    tools: list | None = None,
    instructions: str | None = None,
    tracker: TokenTracker | None = None,
    call_name: str = "",
) -> str:
    """Run a Claude CLI call via subprocess and return the response text.

    Uses ``claude -p --output-format json`` to get structured output with
    token usage. The agent runs in the working directory specified by
    ``claude_opts["cwd"]`` and has full tool access (file read/write, bash).
    """
    cli_path = claude_opts.get("cli_path", "claude")
    model = claude_opts.get("model", "opus")
    cwd = claude_opts.get("cwd", ".")
    extra_env = claude_opts.get("env", {})

    cmd = [
        cli_path,
        "-p",
        "--output-format", "json",
        "--dangerously-skip-permissions",
        "--model", model,
    ]
    if instructions:
        cmd += ["--append-system-prompt", instructions]
    cmd.append(prompt)

    start_time = datetime.now()
    if logger:
        logger.log(f"[Claude] Starting {call_name} (model={model})")

    # Build environment with provider-specific vars (bedrock, api_key)
    env = None
    if extra_env:
        env = os.environ.copy()
        env.update(extra_env)

    def _call():
        return subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=cwd,
            env=env,
        )

    try:
        result = await asyncio.get_event_loop().run_in_executor(None, _call)
    except Exception as exc:
        elapsed = (datetime.now() - start_time).total_seconds()
        if logger:
            logger.log(f"[Claude] EXCEPTION: {type(exc).__name__}: {exc}")
        if tracker:
            tracker.record(call_name or "agent", 0, 0, elapsed)
        return ""

    elapsed = (datetime.now() - start_time).total_seconds()

    # Log stderr if present (contains error messages from CLI)
    if result.stderr and result.stderr.strip() and logger:
        logger.log(f"[Claude] stderr:\n{result.stderr.strip()}")

    # Parse JSON output
    response = ""
    input_tokens = 0
    output_tokens = 0

    try:
        data = json.loads(result.stdout)
        response = data.get("result", "")

        for _, model_stats in data.get("modelUsage", {}).items():
            input_tokens += model_stats.get("inputTokens", 0)
            output_tokens += model_stats.get("outputTokens", 0)
    except (json.JSONDecodeError, ValueError) as exc:
        if logger:
            logger.log(f"[Claude] JSON parse error: {exc}")
            if result.stdout.strip():
                logger.log(f"[Claude] Raw stdout (first 1000 chars): {result.stdout.strip()[:1000]}")
        response = result.stdout.strip()

    if result.returncode != 0 and logger:
        logger.log(f"[Claude] Non-zero exit code: {result.returncode}")

    # Log the response (truncated for readability)
    if logger and response:
        preview = response[:500] + ("..." if len(response) > 500 else "")
        for line in preview.splitlines():
            if line.strip():
                logger.log(line)

    if tracker:
        tracker.record(call_name or "agent", input_tokens, output_tokens, elapsed)

    if logger:
        logger.log(f"[Claude] Completed {call_name} in {elapsed:.0f}s "
                   f"({input_tokens} in / {output_tokens} out)")

    return response


async def run_auxiliary_agent(
    agent_role_cfg: dict,
    prompt: str,
    working_dir: str,
    config: dict,
    claude_opts: dict,
    logger: PipelineLogger | None = None,
    tracker: TokenTracker | None = None,
    call_name: str = "",
    instructions: str | None = None,
) -> str:
    """Run an auxiliary agent (literature survey, summary).

    ``agent_role_cfg`` is the per-agent config dict (e.g.
    ``config['pipeline']['literature_survey']``) of the form
    ``{provider: 'codex', model?: '...', reasoning_effort?: '...', ...}``.
    The chosen provider's global section is overlaid with these overrides
    before the call.
    """
    from model_runner import run_model_for_agent

    if logger:
        logger.log(f"[Auxiliary] Running {call_name} with role config={agent_role_cfg}")

    return await run_model_for_agent(
        agent_role_cfg=agent_role_cfg,
        prompt=prompt,
        working_dir=working_dir,
        config=config,
        claude_opts=claude_opts,
        logger=logger,
        tracker=tracker,
        call_name=call_name,
        instructions=instructions,
    )


# ---------------------------------------------------------------------------
# Literature survey
# ---------------------------------------------------------------------------

async def run_literature_survey(
    output_dir: str,
    problem_file: str,
    claude_opts: dict,
    prompts_dir: str,
    config: dict,
    math_skill: str = "",
    tracker: TokenTracker | None = None,
) -> str:
    """Run the literature survey agent before proof search.
    Returns the path to the related_info directory.
    """
    related_info_dir = os.path.join(output_dir, "related_info")
    os.makedirs(related_info_dir, exist_ok=True)
    log_dir = os.path.join(output_dir, "literature_survey_log")

    logger = PipelineLogger(log_dir, "Literature Survey")
    logger.update_status(1, 1, "Literature Survey", "RUNNING", "Running literature survey agent...")

    agent_role_cfg = config.get("pipeline", {}).get("literature_survey", {})

    proof_file = os.path.join(output_dir, "proof.md")
    survey_prompt = load_prompt(
        prompts_dir, "literature_survey.md",
        problem_file=problem_file,
        related_info_dir=related_info_dir,
        output_dir=output_dir,
        proof_file=proof_file,
        error_file=os.path.join(related_info_dir, "error_literature_survey.md"),
    )

    response = await run_auxiliary_agent(
        agent_role_cfg=agent_role_cfg,
        prompt=survey_prompt,
        working_dir=claude_opts.get("cwd", output_dir),
        config=config,
        claude_opts=claude_opts,
        logger=logger,
        tracker=tracker,
        call_name="Literature Survey",
        instructions=math_skill or None,
    )

    # Required output depends on the difficulty the agent chose. Easy → the
    # agent must produce proof.md directly and may skip related_work.md;
    # Medium/Hard → the agent must produce related_work.md and must not write
    # proof.md (Stage 1 will).
    difficulty = _parse_difficulty(output_dir)
    if difficulty == "easy":
        primary_files = [
            os.path.join(related_info_dir, "difficulty_evaluation.md"),
            proof_file,
        ]
        expected_files = [
            (os.path.join(related_info_dir, "difficulty_evaluation.md"), "difficulty evaluation"),
            (proof_file, "proof (Easy short-circuit)"),
            (os.path.join(related_info_dir, "error_literature_survey.md"), "error log"),
        ]
    else:
        primary_files = [
            os.path.join(related_info_dir, "difficulty_evaluation.md"),
            os.path.join(related_info_dir, "related_work.md"),
        ]
        expected_files = [
            (os.path.join(related_info_dir, "difficulty_evaluation.md"), "difficulty evaluation"),
            (os.path.join(related_info_dir, "related_work.md"), "related work"),
            (os.path.join(related_info_dir, "error_literature_survey.md"), "error log"),
        ]

    _fallback_save_response(
        response, primary_files,
        [os.path.join(related_info_dir, "error_literature_survey.md")],
        logger, step_name="Literature Survey",
    )
    _check_expected_files(expected_files, logger, "Literature Survey")

    logger.finalize(1, 1, "FINISHED", "Literature survey complete.")
    return related_info_dir




# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def _count_decomposition_attempts(output_dir: str) -> tuple[int, int]:
    """Count attempts and total proofs produced under ``decomposition/``.

    Returns ``(num_attempts, num_proofs)``.
    """
    decomp_dir = os.path.join(output_dir, "decomposition")
    attempts = 0
    proofs = 0
    if not os.path.isdir(decomp_dir):
        return 0, 0
    for a_name in os.listdir(decomp_dir):
        a_path = os.path.join(decomp_dir, a_name)
        if not (a_name.startswith("attempt_") and os.path.isdir(a_path)):
            continue
        attempts += 1
        for r_name in os.listdir(a_path):
            r_path = os.path.join(a_path, r_name)
            if not (r_name.startswith("revision_") and os.path.isdir(r_path)):
                continue
            for p_name in os.listdir(r_path):
                if p_name.startswith("proof_") and os.path.isdir(os.path.join(r_path, p_name)):
                    proofs += 1
    return attempts, proofs


async def main():
    parser = argparse.ArgumentParser(description="Proof Agent: natural-language proof search pipeline")
    parser.add_argument("--input", required=True, help="Path to problem.tex file")
    parser.add_argument("--output", required=True, help="Output directory for proof and logs")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    args = parser.parse_args()

    check_prerequisites()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    claude_cfg = config.get("claude", {})
    prover_mode = config.get("prover", {}).get("mode", "decomposition")

    if prover_mode != "decomposition":
        print(f"ERROR: prover.mode must be 'decomposition' (got: {prover_mode!r}). "
              f"Simple mode has been removed.")
        sys.exit(1)

    problem_file = os.path.abspath(args.input)
    output_dir = os.path.abspath(args.output)

    if not os.path.exists(problem_file):
        print(f"ERROR: Input file not found: {problem_file}")
        sys.exit(1)

    # Resolve paths relative to project root (one level up from code/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_base = os.path.dirname(script_dir)
    prompts_dir = os.path.join(project_base, "prompts")
    skill_dir = os.path.join(project_base, "skill")

    # Load math skill (used as system prompt for the literature survey agent)
    math_skill_path = os.path.join(skill_dir, "super_math_skill.md")
    proving_skill = ""
    if os.path.exists(math_skill_path):
        with open(math_skill_path) as f:
            proving_skill = f.read()

    os.makedirs(output_dir, exist_ok=True)

    problem_copy = os.path.join(output_dir, "problem.tex")
    if not os.path.exists(problem_copy):
        shutil.copy2(problem_file, problem_copy)

    # Snapshot the config actually used by this run so the UI (and the user)
    # can verify what was active. Overwrite on every invocation so resumes
    # reflect the latest config.
    shutil.copy2(args.config, os.path.join(output_dir, "config_used.yaml"))

    claude_opts = make_claude_options(claude_cfg, output_dir)
    # Label the token report with the actual backend. The project runs on
    # opencode; fall back to the claude model name only if opencode is absent.
    primary_model = config.get("opencode", {}).get("model") or claude_opts["model"]
    tracker = TokenTracker(output_dir, primary_model)

    skip_survey = literature_survey_complete(output_dir)

    decomp_cfg = config.get("decomposition", {})

    print("=" * 60)
    print("  Proof Agent Pipeline")
    print("=" * 60)
    print(f"  Problem:    {problem_file}")
    print(f"  Output:     {output_dir}")
    print(f"  Mode:       {prover_mode}")
    print(f"  Max proof attempts:    {decomp_cfg.get('max_proof_attempts', 2)}")
    print(f"  Max revisions:         {decomp_cfg.get('max_revisions', 2)}")
    print(f"  Max decompositions:    {decomp_cfg.get('max_decompositions', 2)}")
    print(f"  Token log:  {tracker.md_path}")
    if skip_survey:
        print()
        print("  RESUMING previous run:")
        print("    - Literature survey: SKIP (already complete)")
    print()

    def _print_file(title: str, path: str):
        print("=" * 60)
        print(f"  {title}")
        print(f"  ({path})")
        print("=" * 60)
        if os.path.exists(path):
            with open(path) as fh:
                content = fh.read()
            print(content if content.strip() else "(empty)")
        else:
            print("(file not found)")
        print()

    _print_file("CONFIGURATION", args.config)
    _print_file("PROBLEM STATEMENT", problem_copy)
    _print_file(
        "HUMAN GUIDANCE (prove)",
        os.path.join(output_dir, "human_help", "additional_prove_human_help_global.md"),
    )
    _print_file(
        "VERIFY RULE",
        os.path.join(output_dir, "human_help", "additional_verify_rule_global.md"),
    )

    # -------------------------------------------------------
    # Stage 0: Literature Survey
    # -------------------------------------------------------
    related_info_dir = os.path.join(output_dir, "related_info")
    if skip_survey:
        print("=" * 60)
        print("STAGE 0: Literature Survey  [SKIPPED — already complete]")
        print("=" * 60)
        print(f"  Using existing survey at: {related_info_dir}")
    else:
        print("=" * 60)
        print("STAGE 0: Literature Survey")
        print("=" * 60)
        related_info_dir = await run_literature_survey(
            output_dir, problem_file, claude_opts, prompts_dir,
            config=config, math_skill=proving_skill, tracker=tracker,
        )
        print(f"  Survey saved to: {related_info_dir}")

    difficulty = _parse_difficulty(output_dir)
    if difficulty != "unknown":
        print(f"  Difficulty: {difficulty.upper()}")
    print()

    # -------------------------------------------------------
    # Easy short-circuit: the literature survey agent has produced
    # the proof directly. Skip Stage 1 (decomposition prover) and
    # Stage 2 (proof effort summary).
    # -------------------------------------------------------
    proof_file = os.path.join(output_dir, "proof.md")
    if difficulty == "easy" and _file_nonempty(proof_file):
        print("=" * 60)
        print("EASY PATH: Survey agent produced proof directly")
        print("  Skipping Stage 1 (decomposition prover) and Stage 2 (summary).")
        print("=" * 60)
        print()
        print("=" * 60)
        print("  PIPELINE COMPLETE  [easy short-circuit]")
        print("=" * 60)
        print(f"  Proof at:    {proof_file}")
        print(f"  Token usage: {tracker.md_path}")
        print(f"  Output:      {output_dir}")
        return

    # -------------------------------------------------------
    # Stage 1: Decomposition prover
    # -------------------------------------------------------
    print("=" * 60)
    print("STAGE 1: Proof Search  [DECOMPOSITION MODE]")
    print("  (Decomposition-based proving with step-by-step verification)")
    print("=" * 60)

    from decomposition_prover import run_decomposition_prover

    related_work_file = os.path.join(related_info_dir, "related_work.md")
    # difficulty_evaluation.md is still generated by the literature survey and
    # parsed for the console banner via _parse_difficulty(); the decomposer
    # itself no longer consumes it.

    await run_decomposition_prover(
        problem_file=problem_file,
        related_work_file=related_work_file,
        output_dir=output_dir,
        config=config,
        prompts_dir=prompts_dir,
        claude_opts=claude_opts,
        logger=PipelineLogger(output_dir, "DecompositionProver"),
        tracker=tracker,
    )

    proof_file = os.path.join(output_dir, "proof.md")
    ok = os.path.exists(proof_file) and os.path.getsize(proof_file) > 0

    if ok:
        print("  Decomposition prover completed successfully")
    else:
        print("  Decomposition prover failed to produce a proof")

    # -------------------------------------------------------
    # Stage 2: Proof Effort Summary
    # -------------------------------------------------------
    summary_file = os.path.join(output_dir, "proof_effort_summary.md")

    if _file_nonempty(summary_file):
        print()
        print("=" * 60)
        print("STAGE 2: Proof Effort Summary  [SKIPPED — already exists]")
        print("=" * 60)
        print(f"  Using existing summary at: {summary_file}")
    else:
        attempts, proofs = _count_decomposition_attempts(output_dir)
        outcome = (
            "PASS — Proof verified successfully" if ok
            else "FAIL — Retry limits exhausted without a verified proof"
        )

        print()
        print("=" * 60)
        print("STAGE 2: Proof Effort Summary")
        print("=" * 60)

        summary_log_dir = os.path.join(output_dir, "summary_log")
        summary_logger = PipelineLogger(summary_log_dir, "Proof Effort Summary")

        summary_role_cfg = config.get("pipeline", {}).get("proof_summary", {})
        summary_logger.update_status(1, 1, "Summary", "RUNNING",
                                     f"Writing proof effort summary ({summary_role_cfg.get('provider', '?')})...")

        max_attempts = int(decomp_cfg.get("max_decompositions", 2))
        summary_prompt = load_prompt(
            prompts_dir, "proof_effort_summary.md",
            output_dir=output_dir,
            outcome=outcome,
            total_attempts=attempts,
            total_proofs=proofs,
            max_attempts=max_attempts,
            summary_file=summary_file,
            error_file=os.path.join(output_dir, "error_proof_effort_summary.md"),
        )
        response = await run_auxiliary_agent(
            agent_role_cfg=summary_role_cfg,
            prompt=summary_prompt,
            working_dir=claude_opts.get("cwd", output_dir),
            config=config,
            claude_opts=claude_opts,
            logger=summary_logger,
            tracker=tracker,
            call_name="Proof Effort Summary",
        )
        _fallback_save_response(response, [summary_file],
            [os.path.join(output_dir, "error_proof_effort_summary.md")],
            summary_logger, step_name="Proof Effort Summary")
        _check_expected_files([
            (summary_file, "proof effort summary"),
            (os.path.join(output_dir, "error_proof_effort_summary.md"), "error log"),
        ], summary_logger, "Proof Effort Summary")
        summary_logger.finalize(1, 1, "FINISHED", "Summary complete.")
        print(f"  Summary saved to: {summary_file}")

    print()
    print("=" * 60)
    if ok:
        print("  PIPELINE COMPLETE")
    else:
        print("  PIPELINE STOPPED — Retry limits exhausted")
    print("=" * 60)
    print(f"  Proof at:    {os.path.join(output_dir, 'proof.md')}")
    print(f"  Summary at:  {summary_file}")
    print(f"  Token usage: {tracker.md_path}")
    print(f"  Output:      {output_dir}")


if __name__ == "__main__":
    asyncio.run(main())
