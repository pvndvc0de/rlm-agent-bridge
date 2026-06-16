from __future__ import annotations

import asyncio
import importlib
import json
import subprocess
import tempfile
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from rlm.clients.base_lm import BaseLM
    from rlm.core.types import ModelUsageSummary, UsageSummary
except ImportError:

    class BaseLM:  # type: ignore[no-redef]
        def __init__(
            self,
            model_name: str,
            timeout: float | None = 300.0,
            sampling_args: dict[str, Any] | None = None,
            **kwargs: Any,
        ) -> None:
            self.model_name = model_name
            self.timeout = timeout
            self.sampling_args = dict(sampling_args or {})
            self.kwargs = kwargs

    @dataclass
    class ModelUsageSummary:  # type: ignore[no-redef]
        total_calls: int
        total_input_tokens: int
        total_output_tokens: int
        total_cost: float | None = None

    @dataclass
    class UsageSummary:  # type: ignore[no-redef]
        model_usage_summaries: dict[str, ModelUsageSummary]


CODEX_SANDBOXES = {"read-only", "workspace-write", "danger-full-access"}
DEFAULT_MODEL_NAME = "codex-cli"


@dataclass(frozen=True)
class CommandResult:
    exit_code: int
    stdout: str
    stderr: str


CommandRunner = Callable[[list[str], str | None, float | None], CommandResult]


class SubprocessRunner:
    def __call__(
        self,
        command: list[str],
        cwd: str | None = None,
        timeout: float | None = None,
    ) -> CommandResult:
        completed = subprocess.run(
            command,
            cwd=cwd,
            timeout=timeout,
            check=False,
            capture_output=True,
            text=True,
        )
        return CommandResult(
            exit_code=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )


class CodexRLMClient(BaseLM):
    """RLM BaseLM-compatible adapter that uses the local Codex CLI."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        *,
        executable: str = "codex",
        codex_model: str | None = None,
        cwd: str | None = None,
        sandbox: str = "read-only",
        timeout: float | None = 300.0,
        runner: CommandRunner | None = None,
        output_dir: str | Path | None = None,
        extra_args: list[str] | None = None,
        sampling_args: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if sandbox not in CODEX_SANDBOXES:
            raise ValueError(
                f"Unsupported sandbox {sandbox!r}. Expected one of: {sorted(CODEX_SANDBOXES)}"
            )

        super().__init__(
            model_name=model_name,
            timeout=timeout,
            sampling_args=sampling_args,
            **kwargs,
        )
        self.executable = executable
        self.codex_model = codex_model or (model_name if model_name != DEFAULT_MODEL_NAME else None)
        self.cwd = cwd
        self.sandbox = sandbox
        self.runner = runner or SubprocessRunner()
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.extra_args = list(extra_args or [])

        self._total_calls = 0
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._last_usage = ModelUsageSummary(
            total_calls=0,
            total_input_tokens=0,
            total_output_tokens=0,
            total_cost=None,
        )

    def completion(self, prompt: str | list[dict[str, Any]] | dict[str, Any]) -> str:
        prompt_text = prompt_to_text(prompt)
        output_path = self._new_output_path()
        command = self.build_command(prompt_text, output_path=output_path)

        try:
            result = self.runner(command, None, self.timeout)
            if result.exit_code != 0:
                message = result.stderr.strip() or result.stdout.strip() or "codex command failed"
                raise RuntimeError(message)

            response = _read_output_last_message(output_path) or result.stdout.strip()
            self._record_usage(prompt_text, response)
            return response
        finally:
            output_path.unlink(missing_ok=True)

    async def acompletion(self, prompt: str | list[dict[str, Any]] | dict[str, Any]) -> str:
        return await asyncio.to_thread(self.completion, prompt)

    def build_command(self, prompt_text: str, *, output_path: Path) -> list[str]:
        command = [
            self.executable,
            "exec",
            "-s",
            self.sandbox,
            "--color",
            "never",
            "--output-last-message",
            str(output_path),
        ]
        if self.cwd:
            command.extend(["-C", self.cwd])
        if self.codex_model:
            command.extend(["-m", self.codex_model])
        command.extend(self.extra_args)
        command.append(prompt_text)
        return command

    def get_usage_summary(self) -> UsageSummary:
        return UsageSummary(
            model_usage_summaries={
                self.model_name: ModelUsageSummary(
                    total_calls=self._total_calls,
                    total_input_tokens=self._total_input_tokens,
                    total_output_tokens=self._total_output_tokens,
                    total_cost=None,
                )
            }
        )

    def get_last_usage(self) -> ModelUsageSummary:
        return self._last_usage

    def _record_usage(self, prompt_text: str, response: str) -> None:
        input_tokens = approximate_tokens(prompt_text)
        output_tokens = approximate_tokens(response)
        self._total_calls += 1
        self._total_input_tokens += input_tokens
        self._total_output_tokens += output_tokens
        self._last_usage = ModelUsageSummary(
            total_calls=1,
            total_input_tokens=input_tokens,
            total_output_tokens=output_tokens,
            total_cost=None,
        )

    def _new_output_path(self) -> Path:
        output_dir = self.output_dir or Path(tempfile.gettempdir())
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir / f"rlm-agent-bridge-codex-{uuid.uuid4().hex}.txt"


def install_codex_backend(
    *,
    backend_name: str = "codex_agent",
    client_cls: type = CodexRLMClient,
) -> Callable[[], None]:
    """Register a Codex backend in alexzhang13/rlm without forking it.

    Upstream RLM imports ``get_client`` into ``rlm.core.rlm`` at import time, so
    both ``rlm.clients.get_client`` and ``rlm.core.rlm.get_client`` need to be
    patched. The returned callback restores the original functions.
    """
    try:
        rlm_clients = importlib.import_module("rlm.clients")
        rlm_core = importlib.import_module("rlm.core.rlm")
    except ImportError as exc:
        raise ImportError(
            "install_codex_backend() requires alexzhang13/rlm. Install it with "
            "`pip install git+https://github.com/alexzhang13/rlm.git` or an editable checkout."
        ) from exc

    original_clients_get_client = rlm_clients.get_client
    original_core_get_client = rlm_core.get_client

    def get_client(backend: str, backend_kwargs: dict[str, Any] | None = None) -> Any:
        if backend == backend_name:
            return client_cls(**(backend_kwargs or {}))
        return original_clients_get_client(backend, backend_kwargs)

    rlm_clients.get_client = get_client
    rlm_core.get_client = get_client

    def restore() -> None:
        rlm_clients.get_client = original_clients_get_client
        rlm_core.get_client = original_core_get_client

    return restore


def prompt_to_text(prompt: str | list[dict[str, Any]] | dict[str, Any]) -> str:
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        return _messages_to_text(prompt)
    if isinstance(prompt, dict):
        messages = prompt.get("messages")
        if isinstance(messages, list):
            return _messages_to_text(messages)
        return json.dumps(prompt, ensure_ascii=False, sort_keys=True)
    raise TypeError(f"Unsupported prompt type: {type(prompt)!r}")


def approximate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def _messages_to_text(messages: list[dict[str, Any]]) -> str:
    blocks = []
    for message in messages:
        role = str(message.get("role", "user"))
        content = _content_to_text(message.get("content", ""))
        blocks.append(f"{role}:\n{content}")
    return "\n\n".join(blocks)


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(str(part.get("text", "")))
            else:
                parts.append(json.dumps(part, ensure_ascii=False, sort_keys=True))
        return "\n".join(parts)
    if content is None:
        return ""
    return str(content)


def _read_output_last_message(output_path: Path) -> str:
    if not output_path.exists():
        return ""
    return output_path.read_text(encoding="utf-8").strip()


__all__ = [
    "CodexRLMClient",
    "CommandResult",
    "SubprocessRunner",
    "approximate_tokens",
    "install_codex_backend",
    "prompt_to_text",
]
