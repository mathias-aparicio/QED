#!/usr/bin/env python3
"""Unified multi-model runner for Claude, Codex, and Gemini.

Provides async wrappers around each provider's CLI, returning response text
and feeding token usage into the pipeline's TokenTracker.

All three providers are invoked via their respective CLIs (subprocess),
wrapped in ``asyncio`` executors so the main event loop stays non-blocking.
"""

import asyncio
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime


async def _progress_heartbeat(label: str, start: datetime, stop: "asyncio.Event") -> None:
    """Show a live elapsed-time spinner while a model call runs.

    Answers "am I stuck or making progress?": the timer keeps climbing while
    the call is alive. Animates on a TTY (single line rewritten via ``\\r``);
    when output is piped/redirected (e.g. the UI) it emits a heartbeat line
    every ~30s instead, so liveness is still visible in logs.
    """
    frames = "|/-\\"
    tty = False
    try:
        tty = sys.stderr.isatty()
    except Exception:
        tty = False
    i = 0
    last_log = 0.0
    while not stop.is_set():
        elapsed = (datetime.now() - start).total_seconds()
        mm, ss = int(elapsed // 60), int(elapsed % 60)
        if tty:
            sys.stderr.write(f"\r  {frames[i % len(frames)]} {label} — running {mm}:{ss:02d} ...    ")
            sys.stderr.flush()
        elif elapsed - last_log >= 30:
            sys.stderr.write(f"  ... {label} still running ({mm}:{ss:02d})\n")
            sys.stderr.flush()
            last_log = elapsed
        i += 1
        try:
            await asyncio.wait_for(stop.wait(), timeout=0.4)
        except asyncio.TimeoutError:
            pass
    if tty:
        sys.stderr.write("\r" + " " * 72 + "\r")
        sys.stderr.flush()


class ModelRunnerError(Exception):
    """Raised when a model runner encounters a fatal error.

    Attributes:
        provider: The model provider (claude, codex, gemini).
        error_type: Category of error (subprocess_error, non_zero_exit, json_parse_error, empty_response).
        message: Human-readable error message.
        exit_code: Process exit code (if applicable).
        stderr: Stderr output from the CLI (if any).
        stdout: Raw stdout (for debugging).
    """

    def __init__(
        self,
        provider: str,
        error_type: str,
        message: str,
        exit_code: int | None = None,
        stderr: str = "",
        stdout: str = "",
    ):
        self.provider = provider
        self.error_type = error_type
        self.exit_code = exit_code
        self.stderr = stderr
        self.stdout = stdout
        super().__init__(message)

    def __str__(self):
        parts = [f"[{self.provider}] {self.error_type}: {self.args[0]}"]
        if self.exit_code is not None:
            parts.append(f"exit_code={self.exit_code}")
        if self.stderr:
            # Truncate stderr for display
            stderr_preview = self.stderr[:500] + ("..." if len(self.stderr) > 500 else "")
            parts.append(f"stderr={stderr_preview!r}")
        return " | ".join(parts)

    def full_details(self) -> str:
        """Return full error details for logging to file."""
        lines = [
            f"# Model Runner Error",
            f"",
            f"**Provider:** {self.provider}",
            f"**Error Type:** {self.error_type}",
            f"**Message:** {self.args[0]}",
        ]
        if self.exit_code is not None:
            lines.append(f"**Exit Code:** {self.exit_code}")
        if self.stderr:
            lines.extend(["", "## Stderr", "```", self.stderr, "```"])
        if self.stdout:
            lines.extend(["", "## Stdout (first 2000 chars)", "```", self.stdout[:2000], "```"])
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Claude wrapper
# ---------------------------------------------------------------------------

async def run_claude_agent(
    prompt: str,
    working_dir: str,
    claude_opts: dict,
    logger=None,
    tracker=None,
    call_name: str = "",
    instructions: str | None = None,
) -> str:
    """Run the Claude CLI as a proof-search agent. Returns response text.

    Args:
        prompt: The full prompt string to send.
        working_dir: Directory the agent operates in (cwd for subprocess).
        claude_opts: Dict with keys: cli_path, model, env.
        logger: Optional PipelineLogger for streaming output.
        tracker: Optional TokenTracker for recording token usage.
        call_name: Human-readable label for this call.
        instructions: Optional system prompt to append.
    """
    cli_path = claude_opts.get("cli_path", "claude")
    model = claude_opts.get("model", "opus")
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

    # Build environment: start from inherited env, strip vars that cause
    # provider cross-contamination, then add back only the configured ones.
    _PROVIDER_VARS = ("CLAUDE_CODE_USE_BEDROCK", "ANTHROPIC_API_KEY",
                      "AWS_PROFILE", "ANTHROPIC_MODEL")
    env = {k: v for k, v in os.environ.items() if k not in _PROVIDER_VARS}
    env.update(extra_env)

    def _call():
        return subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=working_dir,
            env=env,
        )

    MAX_RETRIES = 3
    RETRY_BACKOFF = [30, 60, 120]  # seconds between retries

    start = datetime.now()
    if logger:
        logger.log(f"[Claude] Starting {call_name} (model={model})")

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        attempt_start = datetime.now()

        try:
            result = await asyncio.get_event_loop().run_in_executor(None, _call)
        except Exception as exc:
            elapsed = (datetime.now() - attempt_start).total_seconds()
            if logger:
                logger.log(f"[Claude] EXCEPTION (attempt {attempt}/{MAX_RETRIES}): "
                           f"{type(exc).__name__}: {exc}")
            last_error = ModelRunnerError(
                provider="claude",
                error_type="subprocess_error",
                message=f"Failed to execute Claude CLI: {type(exc).__name__}: {exc}",
            )
            if attempt < MAX_RETRIES:
                wait = RETRY_BACKOFF[attempt - 1]
                if logger:
                    logger.log(f"[Claude] Retrying in {wait}s...")
                await asyncio.sleep(wait)
                continue
            if tracker:
                tracker.record(call_name or "claude", 0, 0,
                               (datetime.now() - start).total_seconds(),
                               provider="claude", model=model)
            raise last_error

        elapsed = (datetime.now() - attempt_start).total_seconds()

        # Log stderr if present (contains error messages from CLI)
        if result.stderr and result.stderr.strip() and logger:
            logger.log(f"[Claude] stderr:\n{result.stderr.strip()}")

        # --- Parse JSON output ---
        response = ""
        input_tokens = 0
        output_tokens = 0
        json_parse_error = None

        try:
            data = json.loads(result.stdout)
            response = data.get("result", "")

            for _, model_stats in data.get("modelUsage", {}).items():
                input_tokens += model_stats.get("inputTokens", 0)
                output_tokens += model_stats.get("outputTokens", 0)
        except (json.JSONDecodeError, ValueError) as exc:
            json_parse_error = str(exc)
            if logger:
                logger.log(f"[Claude] JSON parse error: {exc}")
                if result.stdout.strip():
                    logger.log(f"[Claude] Raw stdout (first 1000 chars): {result.stdout.strip()[:1000]}")
            response = result.stdout.strip()

        # Check for non-zero exit code (indicates CLI failure) — retryable
        if result.returncode != 0:
            if logger:
                logger.log(f"[Claude] Non-zero exit code: {result.returncode} "
                           f"(attempt {attempt}/{MAX_RETRIES})")
            last_error = ModelRunnerError(
                provider="claude",
                error_type="non_zero_exit",
                message=f"Claude CLI exited with code {result.returncode}",
                exit_code=result.returncode,
                stderr=result.stderr,
                stdout=result.stdout,
            )
            if attempt < MAX_RETRIES:
                wait = RETRY_BACKOFF[attempt - 1]
                if logger:
                    logger.log(f"[Claude] Retrying in {wait}s...")
                await asyncio.sleep(wait)
                continue
            if tracker:
                tracker.record(call_name or "claude", input_tokens, output_tokens,
                               (datetime.now() - start).total_seconds(),
                               provider="claude", model=model)
            raise last_error

        # Check for empty response (might indicate silent failure) — retryable
        if not response.strip():
            if logger:
                logger.log(f"[Claude] Empty response received "
                           f"(attempt {attempt}/{MAX_RETRIES})")
            last_error = ModelRunnerError(
                provider="claude",
                error_type="empty_response",
                message="Claude returned empty response" + (f" (JSON parse error: {json_parse_error})" if json_parse_error else ""),
                exit_code=result.returncode,
                stderr=result.stderr,
                stdout=result.stdout,
            )
            if attempt < MAX_RETRIES:
                wait = RETRY_BACKOFF[attempt - 1]
                if logger:
                    logger.log(f"[Claude] Retrying in {wait}s...")
                await asyncio.sleep(wait)
                continue
            if tracker:
                tracker.record(call_name or "claude", input_tokens, output_tokens,
                               (datetime.now() - start).total_seconds(),
                               provider="claude", model=model)
            raise last_error

        # Success — break out of retry loop
        if attempt > 1 and logger:
            logger.log(f"[Claude] Succeeded on attempt {attempt}")
        break

    total_elapsed = (datetime.now() - start).total_seconds()
    if logger:
        logger.log(f"[Claude] Completed {call_name} in {total_elapsed:.0f}s "
                    f"({input_tokens} in / {output_tokens} out)")

    if tracker:
        tracker.record(call_name or "claude", input_tokens, output_tokens,
                       total_elapsed, provider="claude", model=model)

    return response


# ---------------------------------------------------------------------------
# Codex wrapper
# ---------------------------------------------------------------------------

async def run_codex_agent(
    prompt: str,
    working_dir: str,
    codex_config: dict,
    logger=None,
    tracker=None,
    call_name: str = "",
) -> str:
    """Run the Codex CLI as a proof-search agent. Returns response text.

    Args:
        prompt: The full prompt string to send.
        working_dir: Directory the agent operates in (cwd for subprocess).
        codex_config: Dict with keys: cli_path, model, reasoning_effort.
        logger: Optional PipelineLogger for streaming output.
        tracker: Optional TokenTracker for recording token usage.
        call_name: Human-readable label for this call.
    """
    cli_path = codex_config.get("cli_path", "codex")
    model = codex_config.get("model", "gpt-5.5")
    reasoning = codex_config.get("reasoning_effort", "xhigh")

    cmd = [
        cli_path,
        "--search",
        "-m", model,
        "-c", f'model_reasoning_effort="{reasoning}"',
        "exec",
        "--json",
        "--dangerously-bypass-approvals-and-sandbox",
        "-C", working_dir,
        prompt,
    ]

    def _call():
        return subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=working_dir,
        )

    start = datetime.now()
    if logger:
        logger.log(f"[Codex] Starting {call_name} (model={model})")

    try:
        result = await asyncio.get_event_loop().run_in_executor(None, _call)
    except Exception as exc:
        elapsed = (datetime.now() - start).total_seconds()
        if logger:
            logger.log(f"[Codex] EXCEPTION: {type(exc).__name__}: {exc}")
        if tracker:
            tracker.record(call_name or "codex", 0, 0, elapsed,
                           provider="codex", model=model)
        raise ModelRunnerError(
            provider="codex",
            error_type="subprocess_error",
            message=f"Failed to execute Codex CLI: {type(exc).__name__}: {exc}",
        )

    elapsed = (datetime.now() - start).total_seconds()

    # Log stderr if present (contains error messages from CLI)
    if result.stderr and result.stderr.strip() and logger:
        logger.log(f"[Codex] stderr:\n{result.stderr.strip()}")

    # --- Parse JSONL output (adapted from test_call.py:41-67) ---
    response = ""
    input_tokens = 0
    output_tokens = 0
    json_parse_error = None

    try:
        lines = result.stdout.strip().split("\n")
        events = [json.loads(line) for line in lines if line.strip()]

        for event in events:
            if event.get("type") == "item.completed":
                item = event.get("item", {})
                if item.get("type") == "agent_message":
                    response = item.get("text", "")
            elif event.get("type") == "turn.completed":
                usage = event.get("usage", {})
                input_tokens += usage.get("input_tokens", 0)
                output_tokens += usage.get("output_tokens", 0)
    except (json.JSONDecodeError, ValueError) as exc:
        json_parse_error = str(exc)
        if logger:
            logger.log(f"[Codex] JSON parse error: {exc}")
            if result.stdout.strip():
                logger.log(f"[Codex] Raw stdout (first 1000 chars): {result.stdout.strip()[:1000]}")
        # Fall back to raw stdout as response
        response = result.stdout.strip()

    # The Codex CLI sometimes exits non-zero (e.g. "Reading additional input
    # from stdin..." warning → exit code 1) even when it produced a valid
    # agent_message. Treat the response payload as authoritative: a non-zero
    # exit is only fatal if we could not parse any usable response.
    if result.returncode != 0:
        if logger:
            logger.log(f"[Codex] Non-zero exit code: {result.returncode} "
                       f"(treating as warning since response was parsed)" if response.strip()
                       else f"[Codex] Non-zero exit code: {result.returncode}")
        if not response.strip():
            if tracker:
                tracker.record(call_name or "codex", input_tokens, output_tokens,
                               elapsed, provider="codex", model=model)
            raise ModelRunnerError(
                provider="codex",
                error_type="non_zero_exit",
                message=f"Codex CLI exited with code {result.returncode}",
                exit_code=result.returncode,
                stderr=result.stderr,
                stdout=result.stdout,
            )

    # Check for empty response (might indicate silent failure)
    if not response.strip():
        if logger:
            logger.log(f"[Codex] Empty response received")
        if tracker:
            tracker.record(call_name or "codex", input_tokens, output_tokens,
                           elapsed, provider="codex", model=model)
        raise ModelRunnerError(
            provider="codex",
            error_type="empty_response",
            message="Codex returned empty response" + (f" (JSON parse error: {json_parse_error})" if json_parse_error else ""),
            exit_code=result.returncode,
            stderr=result.stderr,
            stdout=result.stdout,
        )

    if logger:
        logger.log(f"[Codex] Completed {call_name} in {elapsed:.0f}s "
                    f"({input_tokens} in / {output_tokens} out)")

    if tracker:
        tracker.record(call_name or "codex", input_tokens, output_tokens,
                       elapsed, provider="codex", model=model)

    return response


# ---------------------------------------------------------------------------
# Gemini wrapper
# ---------------------------------------------------------------------------

async def run_gemini_agent(
    prompt: str,
    working_dir: str,
    gemini_config: dict,
    logger=None,
    tracker=None,
    call_name: str = "",
) -> str:
    """Run the Gemini CLI as a proof-search agent. Returns response text.

    Args:
        prompt: The full prompt string to send.
        working_dir: Directory the agent operates in (cwd for subprocess).
        gemini_config: Dict with keys: cli_path, model, api_key,
            approval_mode, thinking_level, thinking_budget.
        logger: Optional PipelineLogger for streaming output.
        tracker: Optional TokenTracker for recording token usage.
        call_name: Human-readable label for this call.
    """
    cli_path = gemini_config.get("cli_path", "gemini")
    model = gemini_config.get("model", "gemini-3-flash-preview")
    api_key = gemini_config.get("api_key", "")
    approval_mode = gemini_config.get("approval_mode", "yolo")
    thinking_level = gemini_config.get("thinking_level", "")
    thinking_budget = gemini_config.get("thinking_budget")

    cmd = [
        cli_path,
        "-m", model,
        "--approval-mode", approval_mode,
        "-o", "json",   # JSON output for metadata extraction
        "-p", prompt,
    ]

    def _call():
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
                        "overrides": [
                            {
                                "match": {"model": model},
                                "modelConfig": {
                                    "generateContentConfig": {
                                        "thinkingConfig": thinking_config,
                                    }
                                },
                            }
                        ]
                    }
                }
                with open(settings_path, "w", encoding="utf-8") as f:
                    json.dump(settings, f)
                env["GEMINI_CLI_HOME"] = gemini_home
                return subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    cwd=working_dir,
                    env=env,
                )

        return subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=working_dir,
            env=env,
        )

    start = datetime.now()
    if logger:
        logger.log(f"[Gemini] Starting {call_name} (model={model})")

    try:
        result = await asyncio.get_event_loop().run_in_executor(None, _call)
    except Exception as exc:
        elapsed = (datetime.now() - start).total_seconds()
        if logger:
            logger.log(f"[Gemini] EXCEPTION: {type(exc).__name__}: {exc}")
        if tracker:
            tracker.record(call_name or "gemini", 0, 0, elapsed,
                           provider="gemini", model=model)
        raise ModelRunnerError(
            provider="gemini",
            error_type="subprocess_error",
            message=f"Failed to execute Gemini CLI: {type(exc).__name__}: {exc}",
        )

    elapsed = (datetime.now() - start).total_seconds()

    # Log stderr if present (contains error messages from CLI)
    if result.stderr and result.stderr.strip() and logger:
        logger.log(f"[Gemini] stderr:\n{result.stderr.strip()}")

    # --- Parse JSON output (adapted from test_call.py:74-121) ---
    response = ""
    input_tokens = 0
    output_tokens = 0
    json_parse_error = None

    try:
        data = json.loads(result.stdout)
        response = data.get("response", "")

        for _, model_stats in data.get("stats", {}).get("models", {}).items():
            tokens = model_stats.get("tokens", {})
            input_tokens += tokens.get("input", 0)
            output_tokens += tokens.get("candidates", 0)
            output_tokens += tokens.get("thoughts", 0)  # include thinking tokens
    except (json.JSONDecodeError, ValueError) as exc:
        json_parse_error = str(exc)
        if logger:
            logger.log(f"[Gemini] JSON parse error: {exc}")
            if result.stdout.strip():
                logger.log(f"[Gemini] Raw stdout (first 1000 chars): {result.stdout.strip()[:1000]}")
        response = result.stdout.strip()

    # Check for non-zero exit code (indicates CLI failure)
    if result.returncode != 0:
        if logger:
            logger.log(f"[Gemini] Non-zero exit code: {result.returncode}")
        if tracker:
            tracker.record(call_name or "gemini", input_tokens, output_tokens,
                           elapsed, provider="gemini", model=model)
        raise ModelRunnerError(
            provider="gemini",
            error_type="non_zero_exit",
            message=f"Gemini CLI exited with code {result.returncode}",
            exit_code=result.returncode,
            stderr=result.stderr,
            stdout=result.stdout,
        )

    # Check for empty response (might indicate silent failure)
    if not response.strip():
        if logger:
            logger.log(f"[Gemini] Empty response received")
        if tracker:
            tracker.record(call_name or "gemini", input_tokens, output_tokens,
                           elapsed, provider="gemini", model=model)
        raise ModelRunnerError(
            provider="gemini",
            error_type="empty_response",
            message="Gemini returned empty response" + (f" (JSON parse error: {json_parse_error})" if json_parse_error else ""),
            exit_code=result.returncode,
            stderr=result.stderr,
            stdout=result.stdout,
        )

    if logger:
        logger.log(f"[Gemini] Completed {call_name} in {elapsed:.0f}s "
                    f"({input_tokens} in / {output_tokens} out)")

    if tracker:
        tracker.record(call_name or "gemini", input_tokens, output_tokens,
                       elapsed, provider="gemini", model=model)

    return response


# ---------------------------------------------------------------------------
# OpenCode wrapper
# ---------------------------------------------------------------------------

def _extract_opencode_session_id(stdout: str, stderr: str) -> str | None:
    """Pull the opencode session id out of a ``run`` invocation's output.

    ``opencode run --format json`` reliably flushes a ``step_start`` event to
    stdout, and that event carries ``sessionID`` (both top-level and on its
    ``part``). We fall back to scanning stderr for a ``ses_...`` token in case
    the JSON envelope changes across versions.
    """
    import re

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


def _parse_opencode_export(export_stdout: str) -> tuple[str, int, int]:
    """Parse ``opencode export`` JSON into (response_text, in_tokens, out_tokens).

    Export shape: ``{"info": {...}, "messages": [{"info": {role, tokens}, "parts":
    [{"type": "text", "text": ...}, ...]}, ...]}``. The response is the text of
    the *last* assistant message; token usage is summed across all assistant
    messages (reasoning tokens fold into output, matching the Gemini runner).
    Never raises: opencode can emit malformed/partial JSON for a session that
    ended on a provider error, so on a parse failure we fall back to a
    best-effort text salvage and let the caller surface the real provider error.
    """
    if not export_stdout.strip():
        return "", 0, 0
    try:
        data = json.loads(export_stdout)
    except (json.JSONDecodeError, ValueError):
        return _salvage_opencode_text(export_stdout), 0, 0

    messages = data.get("messages", []) if isinstance(data, dict) else []

    input_tokens = 0
    output_tokens = 0
    last_text: list[str] = []

    for msg in messages:
        info = msg.get("info", {}) if isinstance(msg, dict) else {}
        if info.get("role") != "assistant":
            continue
        toks = info.get("tokens") or {}
        input_tokens += int(toks.get("input", 0) or 0)
        output_tokens += int(toks.get("output", 0) or 0)
        output_tokens += int(toks.get("reasoning", 0) or 0)

        texts = [
            p.get("text", "")
            for p in msg.get("parts", [])
            if isinstance(p, dict) and p.get("type") == "text"
            and isinstance(p.get("text"), str)
        ]
        if texts:
            last_text = texts  # keep only the final assistant turn's text

    return "".join(last_text).strip(), input_tokens, output_tokens


def _salvage_opencode_text(export_stdout: str) -> str:
    """Best-effort recovery of the last assistant text from a malformed export.

    Scans for ``"type":"text"`` parts and decodes the following ``"text": "…"``
    JSON string literal (which handles escaping correctly). Returns the last one
    found, or "" if none — in which case the caller treats it as no response.
    """
    import re

    dec = json.JSONDecoder()
    texts: list[str] = []
    for m in re.finditer(r'"type"\s*:\s*"text"', export_stdout):
        seg = export_stdout[m.end():]
        tm = re.search(r'"text"\s*:\s*(")', seg)
        if not tm:
            continue
        try:
            val, _ = dec.raw_decode(seg[tm.start(1):])  # decode the string literal
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(val, str) and val.strip():
            texts.append(val)
    return texts[-1].strip() if texts else ""


def _extract_opencode_error(stderr: str) -> tuple[str | None, bool]:
    """Pull a concise error out of opencode's ``--print-logs`` stderr.

    Returns ``(message, is_auth_error)``. ``is_auth_error`` is True for a
    missing/unusable API key — those are not worth retrying, so the caller can
    fail fast with actionable guidance instead of silently backing off.
    """
    if not stderr:
        return None, False
    low = stderr.lower()
    if "ai_loadapikeyerror" in low or ("api key" in low and "missing" in low):
        return ("Google API key is missing/unavailable to the non-interactive "
                "opencode subprocess. Run `opencode auth login` (choose Google) so "
                "the key persists for subprocess use, then retry."), True

    # Parse the first `error={...}` JSON object (e.g. AI_APICallError) for a
    # status code + provider message.
    dec = json.JSONDecoder()
    idx = stderr.find("error=")
    while idx != -1:
        frag = stderr[idx + len("error="):].lstrip()
        try:
            obj, _ = dec.raw_decode(frag)
        except (json.JSONDecodeError, ValueError):
            obj = None
        if isinstance(obj, dict):
            e = obj.get("error", obj)
            name = e.get("name", "error")
            status = e.get("statusCode")
            rb = e.get("responseBody") or e.get("data") or {}
            if isinstance(rb, str):
                try:
                    rb = json.loads(rb)
                except (json.JSONDecodeError, ValueError):
                    rb = {}
            msg = ""
            if isinstance(rb, dict) and isinstance(rb.get("error"), dict):
                msg = rb["error"].get("message", "")
            return f"{name}" + (f" (HTTP {status})" if status else "") + (f": {msg}" if msg else ""), False
        idx = stderr.find("error=", idx + 6)
    return None, False


async def run_opencode_agent(
    prompt: str,
    working_dir: str,
    opencode_config: dict,
    logger=None,
    tracker=None,
    call_name: str = "",
) -> str:
    """Run the opencode CLI as a proof-search agent. Returns response text.

    opencode is driven non-interactively via ``opencode run``. Unlike the other
    CLIs it does not flush the assistant reply to stdout when stdout is a pipe
    (only the ``step_start`` event escapes), and it exits 0 even when the model
    call fails (e.g. an HTTP 503 from the provider). So we run, recover the
    ``sessionID`` from the run output, then read the assistant text and token
    usage back via ``opencode export <session>``. An empty reply is treated as
    a retryable error.

    Args:
        prompt: The full prompt string to send.
        working_dir: Directory the agent operates in (``--dir`` and cwd).
        opencode_config: Dict with keys: cli_path, model (``provider/model``,
            e.g. ``google/gemini-3.5-flash``), variant, agent, api_key, timeout.
        logger: Optional PipelineLogger.
        tracker: Optional TokenTracker.
        call_name: Human-readable label for this call.
    """
    cli_path = opencode_config.get("cli_path", "opencode")
    model = opencode_config.get("model", "google/gemini-3-flash-preview")
    variant = opencode_config.get("variant", "")
    agent_name = opencode_config.get("agent", "")
    api_key = opencode_config.get("api_key", "")
    timeout = opencode_config.get("timeout", 1800)

    run_cmd = [
        cli_path, "run",
        "--format", "json",
        "--print-logs", "--log-level", "ERROR",  # surface provider/auth errors on stderr
        "--dangerously-skip-permissions",
        "-m", model,
        "--dir", working_dir,
    ]
    if variant:
        run_cmd += ["--variant", variant]
    if agent_name:
        run_cmd += ["--agent", agent_name]
    run_cmd.append(prompt)

    def _env():
        env = os.environ.copy()
        if api_key:
            # opencode's google provider reads the key from GEMINI_API_KEY.
            env["GEMINI_API_KEY"] = api_key
        # Default to the project's opencode config (registers the web-search
        # plugin) when the caller hasn't set one and the file is present.
        if "OPENCODE_CONFIG" not in env:
            repo_cfg = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "opencode.json",
            )
            if os.path.exists(repo_cfg):
                env["OPENCODE_CONFIG"] = repo_cfg
        return env

    def _run(cmd, to):
        return subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,  # opencode run blocks on stdin EOF otherwise
            text=True,
            cwd=working_dir,
            env=_env(),
            timeout=to,
        )

    MAX_RETRIES = 3
    RETRY_BACKOFF = [5, 15, 30]  # seconds; provider 503s are common and transient

    start = datetime.now()
    if logger:
        logger.log(f"[OpenCode] Starting {call_name} (model={model})")

    last_error = None
    response = ""
    input_tokens = 0
    output_tokens = 0

    for attempt in range(1, MAX_RETRIES + 1):
        result = None
        attempt_label = (call_name or "opencode") + (f" (retry {attempt})" if attempt > 1 else "")
        stop_hb = asyncio.Event()
        heartbeat = asyncio.ensure_future(
            _progress_heartbeat(attempt_label, datetime.now(), stop_hb)
        )
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None, _run, run_cmd, timeout
            )
        except subprocess.TimeoutExpired:
            last_error = ModelRunnerError(
                provider="opencode", error_type="subprocess_error",
                message=f"opencode CLI timed out after {timeout}s",
            )
        except Exception as exc:
            last_error = ModelRunnerError(
                provider="opencode", error_type="subprocess_error",
                message=f"Failed to execute opencode CLI: {type(exc).__name__}: {exc}",
            )
        finally:
            stop_hb.set()
            await heartbeat

        fatal = False
        if result is not None:
            oc_err, oc_auth = _extract_opencode_error(result.stderr)
            if oc_err and logger:
                logger.log(f"[OpenCode] provider error: {oc_err}")

            session_id = _extract_opencode_session_id(result.stdout, result.stderr)
            if not session_id:
                last_error = ModelRunnerError(
                    provider="opencode", error_type="json_parse_error",
                    message="Could not determine opencode session id from run output"
                            + (f" — {oc_err}" if oc_err else ""),
                    exit_code=result.returncode, stderr=result.stderr, stdout=result.stdout,
                )
                fatal = oc_auth
            else:
                try:
                    exp = await asyncio.get_event_loop().run_in_executor(
                        None, _run, [cli_path, "export", session_id], 180
                    )
                    response, input_tokens, output_tokens = _parse_opencode_export(exp.stdout)
                except subprocess.TimeoutExpired:
                    last_error = ModelRunnerError(
                        provider="opencode", error_type="subprocess_error",
                        message="opencode export timed out",
                    )
                except (json.JSONDecodeError, ValueError) as exc:
                    last_error = ModelRunnerError(
                        provider="opencode", error_type="json_parse_error",
                        message=f"Failed to parse opencode export JSON: {exc}",
                    )

                if response.strip():
                    if attempt > 1 and logger:
                        logger.log(f"[OpenCode] Succeeded on attempt {attempt}")
                    break
                if last_error is None:
                    last_error = ModelRunnerError(
                        provider="opencode", error_type="empty_response",
                        message="opencode returned empty response — "
                                + (oc_err or "model produced no text (often a transient "
                                   "provider error such as HTTP 503)"),
                        exit_code=result.returncode, stderr=result.stderr, stdout=result.stdout,
                    )
                    fatal = oc_auth

        if logger:
            logger.log(f"[OpenCode] Attempt {attempt}/{MAX_RETRIES} failed: {last_error}")
        if fatal:
            # Auth errors won't fix themselves — stop retrying and surface now.
            if logger:
                logger.log("[OpenCode] Auth error — not retrying.")
            break
        if attempt < MAX_RETRIES:
            wait = RETRY_BACKOFF[attempt - 1]
            if logger:
                logger.log(f"[OpenCode] Retrying in {wait}s...")
            await asyncio.sleep(wait)

    total_elapsed = (datetime.now() - start).total_seconds()

    if not response.strip():
        if tracker:
            tracker.record(call_name or "opencode", input_tokens, output_tokens,
                           total_elapsed, provider="opencode", model=model)
        raise last_error or ModelRunnerError(
            provider="opencode", error_type="empty_response",
            message="opencode returned empty response",
        )

    if logger:
        logger.log(f"[OpenCode] Completed {call_name} in {total_elapsed:.0f}s "
                    f"({input_tokens} in / {output_tokens} out)")
    if tracker:
        tracker.record(call_name or "opencode", input_tokens, output_tokens,
                       total_elapsed, provider="opencode", model=model)

    return response


# ---------------------------------------------------------------------------
# Per-agent override resolution
# ---------------------------------------------------------------------------

def resolve_agent_provider_config(
    full_config: dict,
    agent_role_cfg: dict,
) -> tuple[str, dict]:
    """Resolve a per-agent role config against the global provider section.

    Each agent role in config.yaml is a dict of the form::

        { provider: "codex", model: "gpt-5.5", reasoning_effort: "xhigh" }

    The provider name picks the global section (``codex:`` / ``gemini:`` /
    ``claude:``). Any other keys override the corresponding fields from the
    global section. Knobs not set on the agent fall back to global.

    For ``claude``, the global section contains nested
    ``subscription:`` / ``api_key:`` / ``bedrock:`` blocks; the per-agent
    ``model`` override (if any) overrides whichever block ``claude.provider``
    selects (the global ``claude.provider`` value still controls auth).

    Returns ``(provider, merged_provider_cfg)`` — ``merged_provider_cfg`` is
    a shallow copy of the global section with per-agent fields overlaid.
    """
    if not isinstance(agent_role_cfg, dict):
        raise ValueError(
            f"Agent role config must be a dict like "
            f"{{provider: 'codex', model: 'gpt-5.5'}}, got: {agent_role_cfg!r}"
        )
    provider = agent_role_cfg.get("provider")
    if not provider:
        raise ValueError(
            f"Agent role config is missing required 'provider' field: {agent_role_cfg!r}"
        )
    provider = provider.lower().strip()
    if provider not in ("claude", "codex", "gemini", "opencode"):
        raise ValueError(
            f"Unknown provider {provider!r}; expected 'claude', 'codex', "
            f"'gemini', or 'opencode'."
        )

    overrides = {k: v for k, v in agent_role_cfg.items() if k != "provider"}
    global_section = full_config.get(provider, {})

    if provider == "claude":
        merged = {k: v for k, v in global_section.items()}
        auth_mode = merged.get("provider", "subscription")
        # Apply the per-agent model override into the active auth block
        if "model" in overrides:
            sub = dict(merged.get(auth_mode, {}))
            sub["model"] = overrides["model"]
            merged[auth_mode] = sub
        # Other claude-level overrides (cli_path, permission_mode) overlay directly
        for k, v in overrides.items():
            if k == "model":
                continue
            merged[k] = v
        return provider, merged

    # codex / gemini: flat dict merge
    merged = {**global_section, **overrides}
    return provider, merged


# ---------------------------------------------------------------------------
# Unified dispatcher
# ---------------------------------------------------------------------------

async def run_model(
    provider: str,
    prompt: str,
    working_dir: str,
    config: dict,
    *,
    claude_opts: dict | None = None,
    logger=None,
    tracker=None,
    call_name: str = "",
    instructions: str | None = None,
) -> str:
    """Dispatch a prompt to the specified model provider.

    Args:
        provider: One of "claude", "codex", "gemini", "opencode".
        prompt: The full prompt string.
        working_dir: Agent's working directory.
        config: Full pipeline config dict (with claude/codex/gemini sections).
        claude_opts: Claude CLI options dict (required when provider="claude").
        logger: Optional PipelineLogger.
        tracker: Optional TokenTracker.
        call_name: Human-readable label.
        instructions: System instructions (used only for Claude).

    Returns:
        The agent's response text.
    """
    if provider == "claude":
        return await run_claude_agent(
            prompt, working_dir, claude_opts or {},
            logger=logger, tracker=tracker, call_name=call_name,
            instructions=instructions,
        )
    elif provider == "codex":
        return await run_codex_agent(
            prompt, working_dir, config.get("codex", {}),
            logger=logger, tracker=tracker, call_name=call_name,
        )
    elif provider == "gemini":
        return await run_gemini_agent(
            prompt, working_dir, config.get("gemini", {}),
            logger=logger, tracker=tracker, call_name=call_name,
        )
    elif provider == "opencode":
        return await run_opencode_agent(
            prompt, working_dir, config.get("opencode", {}),
            logger=logger, tracker=tracker, call_name=call_name,
        )
    else:
        raise ValueError(f"Unknown model provider: {provider!r}. "
                         f"Expected 'claude', 'codex', 'gemini', or 'opencode'.")


async def run_model_for_agent(
    agent_role_cfg: dict,
    prompt: str,
    working_dir: str,
    config: dict,
    *,
    claude_opts: dict | None = None,
    logger=None,
    tracker=None,
    call_name: str = "",
    instructions: str | None = None,
) -> str:
    """Dispatch using a per-agent role config (the dict-with-provider format).

    Resolves ``agent_role_cfg`` against the global provider section, then
    invokes :func:`run_model` with an effective config in which the chosen
    provider's section has been overlaid with the per-agent overrides.

    For ``claude``, also overlays the per-agent ``model`` into ``claude_opts``.
    """
    provider, merged_provider_cfg = resolve_agent_provider_config(
        config, agent_role_cfg
    )
    effective_config = dict(config)
    effective_config[provider] = merged_provider_cfg

    if provider == "claude":
        effective_claude_opts = dict(claude_opts or {})
        if "model" in agent_role_cfg:
            effective_claude_opts["model"] = agent_role_cfg["model"]
    else:
        effective_claude_opts = claude_opts

    return await run_model(
        provider,
        prompt,
        working_dir,
        effective_config,
        claude_opts=effective_claude_opts,
        logger=logger,
        tracker=tracker,
        call_name=call_name,
        instructions=instructions,
    )
