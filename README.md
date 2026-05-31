# QED

## Setup (fresh install)

1. **Install uv** (Python runner — auto-manages the venv and dependencies):

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Install opencode** (the model backend):

   ```bash
   curl -fsSL https://opencode.ai/install | bash
   source ~/.bashrc
   ```

3. **Connect your Gemini API key** (one-time). Get a key from
   [Google AI Studio](https://aistudio.google.com/apikey) — it's a single
   string. Then:

   ```bash
   # Run opencode
   opencode
   # Inside opencode run /connect
   /connect

   ```

first run.

## Run

```bash
bash run.sh problem/levin.tex levin_proof
```

## Standalone verifier (optional)

Check an existing proof against a problem:

```bash
bash run_verifier.sh problem/my-problem proof/my-proof
```
