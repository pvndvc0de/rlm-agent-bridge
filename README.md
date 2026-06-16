# rlm-agent-bridge

Thin Codex CLI adapter for [`alexzhang13/rlm`](https://github.com/alexzhang13/rlm).

This package does not implement its own RLM, benchmark framework, context packer,
or agent orchestrator. It only lets upstream `rlm` create a `BaseLM`-compatible
client that calls your already authenticated local Codex CLI.

## Why

Upstream RLM normally calls model APIs directly through clients such as OpenAI,
Anthropic, or Gemini. If you want to test the same RLM loop through Codex
subscription/CLI auth instead of passing API keys to RLM, this bridge provides a
minimal backend:

```text
RLM -> rlm BaseLM client -> codex exec -> final text response
```

## Install

Install upstream RLM and this bridge in the same Python environment:

```bash
python3 -m pip install "rlms>=0.1.2"
python3 -m pip install -e .
```

If you want the latest upstream code instead of the published package:

```bash
python3 -m pip install git+https://github.com/alexzhang13/rlm.git
python3 -m pip install -e .
```

Codex CLI must already be installed and authenticated:

```bash
codex --version
```

## Use With RLM

```python
from rlm import RLM
from rlm_agent_bridge import install_codex_backend

restore_codex_backend = install_codex_backend()

try:
    rlm = RLM(
        backend="codex_agent",  # runtime backend installed by the bridge
        backend_kwargs={
            "model_name": "gpt-5",
            "cwd": "/path/to/workspace",
            "sandbox": "read-only",
            "timeout": 600,
            "extra_args": ["--ephemeral"],
        },
        environment="local",
        max_depth=1,
        max_iterations=6,
        max_concurrent_subcalls=1,
    )

    result = rlm.completion("Analyze this problem and return the final answer.")
    print(result.response)
finally:
    restore_codex_backend()
```

`install_codex_backend()` patches upstream `rlm.clients.get_client` and
`rlm.core.rlm.get_client` at runtime because RLM does not currently accept a
client instance in its public constructor.

## Direct Client Use

You can call the adapter without RLM for a quick Codex CLI smoke test:

```python
from rlm_agent_bridge import CodexRLMClient

client = CodexRLMClient(
    model_name="gpt-5",
    cwd="/path/to/workspace",
    sandbox="read-only",
    extra_args=["--ephemeral"],
)

print(client.completion("Return one sentence about this repository."))
```

Or through the helper CLI:

```bash
rlm-agent-bridge \
  --cwd /path/to/workspace \
  --model gpt-5 \
  --sandbox read-only \
  --ephemeral \
  --prompt "Return one sentence about this repository."
```

## Practical Notes

- This removes the need to pass OpenAI/Anthropic/Gemini API keys to RLM, but it
  does not make prompts private from Codex/OpenAI. Whatever RLM sends to the
  backend becomes Codex prompt content.
- Codex CLI is an agentic command, not a cheap stateless completion API. Expect
  higher latency than normal API clients.
- Token and cost accounting are approximate because Codex CLI does not expose
  API usage data to this adapter. `max_budget` is therefore not meaningful, and
  `max_tokens` should be treated as approximate.
- Start with `max_depth=1`, low `max_iterations`, and
  `max_concurrent_subcalls=1`. Increase only after a smoke test works.
- Use `sandbox="read-only"` and `extra_args=["--ephemeral"]` for evaluation runs.

## Development

Run tests:

```bash
python3 -m unittest discover -s tests
```

Run the privacy guard before pushing public changes:

```bash
python3 scripts/check_no_private_artifacts.py \
  --root . \
  --forbid "PRIVATE_PROJECT_MARKER" \
  --forbid "PRIVATE_ABSOLUTE_PATH"
```
