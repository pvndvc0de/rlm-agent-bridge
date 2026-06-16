import asyncio
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from rlm_agent_bridge.codex import CodexRLMClient, CommandResult, install_codex_backend


class FakeRunner:
    def __init__(self, stdout="codex log", stderr="", exit_code=0, output_text="Codex answer"):
        self.stdout = stdout
        self.stderr = stderr
        self.exit_code = exit_code
        self.output_text = output_text
        self.calls = []

    def __call__(self, command, cwd=None, timeout=None):
        self.calls.append((command, cwd, timeout))
        if "--output-last-message" in command and self.output_text is not None:
            output_path = Path(command[command.index("--output-last-message") + 1])
            output_path.write_text(self.output_text, encoding="utf-8")
        return CommandResult(exit_code=self.exit_code, stdout=self.stdout, stderr=self.stderr)


class CodexRLMClientTests(unittest.TestCase):
    def test_completion_calls_codex_exec_and_tracks_approximate_usage(self):
        with TemporaryDirectory() as tmpdir:
            runner = FakeRunner(output_text="Final answer\n")
            client = CodexRLMClient(
                model_name="gpt-5",
                cwd="/tmp/project",
                sandbox="read-only",
                timeout=17,
                runner=runner,
                output_dir=tmpdir,
                extra_args=["--ephemeral"],
            )

            response = client.completion(
                [
                    {"role": "system", "content": "Be exact."},
                    {"role": "user", "content": "Return two."},
                ]
            )

        command, cwd, timeout = runner.calls[0]
        self.assertEqual(response, "Final answer")
        self.assertEqual(command[:2], ["codex", "exec"])
        self.assertIn("-s", command)
        self.assertIn("read-only", command)
        self.assertIn("-C", command)
        self.assertIn("/tmp/project", command)
        self.assertIn("-m", command)
        self.assertIn("gpt-5", command)
        self.assertIn("--ephemeral", command)
        self.assertIn("--output-last-message", command)
        self.assertIn("system:\nBe exact.", command[-1])
        self.assertIsNone(cwd)
        self.assertEqual(timeout, 17)

        last_usage = client.get_last_usage()
        self.assertEqual(last_usage.total_calls, 1)
        self.assertGreater(last_usage.total_input_tokens, 0)
        self.assertGreater(last_usage.total_output_tokens, 0)

        summary = client.get_usage_summary()
        self.assertIn("gpt-5", summary.model_usage_summaries)

    def test_completion_falls_back_to_stdout_when_last_message_file_is_empty(self):
        runner = FakeRunner(stdout="stdout answer", output_text=None)
        client = CodexRLMClient(model_name="codex-cli", runner=runner)

        self.assertEqual(client.completion("hello"), "stdout answer")

    def test_completion_raises_when_codex_command_fails(self):
        runner = FakeRunner(stderr="auth failed", exit_code=2, output_text=None)
        client = CodexRLMClient(model_name="codex-cli", runner=runner)

        with self.assertRaisesRegex(RuntimeError, "auth failed"):
            client.completion("hello")

    def test_acompletion_runs_completion(self):
        runner = FakeRunner(output_text="async answer")
        client = CodexRLMClient(model_name="codex-cli", runner=runner)

        response = asyncio.run(client.acompletion("hello"))

        self.assertEqual(response, "async answer")

    def test_install_codex_backend_patches_rlm_client_factory_and_can_restore(self):
        def original_get_client(backend, backend_kwargs):
            return f"original:{backend}:{backend_kwargs}"

        fake_clients = types.ModuleType("rlm.clients")
        fake_clients.get_client = original_get_client
        fake_core_rlm = types.ModuleType("rlm.core.rlm")
        fake_core_rlm.get_client = original_get_client
        fake_core = types.ModuleType("rlm.core")
        fake_core.rlm = fake_core_rlm
        fake_rlm = types.ModuleType("rlm")
        fake_rlm.clients = fake_clients
        fake_rlm.core = fake_core

        class FakeClient:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        with patch.dict(
            "sys.modules",
            {
                "rlm": fake_rlm,
                "rlm.clients": fake_clients,
                "rlm.core": fake_core,
                "rlm.core.rlm": fake_core_rlm,
            },
        ):
            restore = install_codex_backend(client_cls=FakeClient)
            client = fake_core_rlm.get_client("codex_agent", {"model_name": "gpt-5"})
            delegated = fake_clients.get_client("openai", {"model_name": "gpt-5"})

            self.assertIsInstance(client, FakeClient)
            self.assertEqual(client.kwargs, {"model_name": "gpt-5"})
            self.assertEqual(delegated, "original:openai:{'model_name': 'gpt-5'}")

            restore()

            self.assertIs(fake_clients.get_client, original_get_client)
            self.assertIs(fake_core_rlm.get_client, original_get_client)


if __name__ == "__main__":
    unittest.main()
