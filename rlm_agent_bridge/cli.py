from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from rlm_agent_bridge.codex import CodexRLMClient


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Call the Codex RLM adapter once.")
    parser.add_argument("--prompt", help="Prompt to send. Reads stdin when omitted.")
    parser.add_argument("--cwd", help="Working directory for the agent run.")
    parser.add_argument("--model", default="codex-cli", help="RLM model name and Codex model.")
    parser.add_argument("--codex-model", help="Codex CLI model override.")
    parser.add_argument("--timeout", type=float, help="Command timeout in seconds.")
    parser.add_argument(
        "--sandbox",
        choices=["read-only", "workspace-write", "danger-full-access"],
        default="read-only",
        help="Codex CLI sandbox mode.",
    )
    parser.add_argument(
        "--ephemeral",
        action="store_true",
        help="Pass --ephemeral to Codex so no session files are persisted.",
    )
    parser.add_argument("--json", action="store_true", help="Print a JSON object instead of raw text.")
    args = parser.parse_args(argv)

    prompt = args.prompt if args.prompt is not None else sys.stdin.read()
    client = CodexRLMClient(
        model_name=args.model,
        codex_model=args.codex_model,
        cwd=args.cwd,
        sandbox=args.sandbox,
        timeout=args.timeout,
        extra_args=["--ephemeral"] if args.ephemeral else None,
    )
    response = client.completion(prompt)
    if args.json:
        payload = {
            "model": client.model_name,
            "response": response,
            "usage": client.get_last_usage().__dict__,
        }
        json.dump(payload, sys.stdout, ensure_ascii=True)
        sys.stdout.write("\n")
    else:
        sys.stdout.write(response)
        if not response.endswith("\n"):
            sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
