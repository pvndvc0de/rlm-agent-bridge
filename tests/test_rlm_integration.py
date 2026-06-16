import unittest
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory

from rlm_agent_bridge import install_codex_backend
from rlm_agent_bridge.codex import CommandResult

try:
    from rlm import RLM
except ImportError:
    RLM = None


class ScriptedCodexRunner:
    def __init__(self):
        self.calls = []

    def __call__(self, command, cwd=None, timeout=None):
        self.calls.append((command, cwd, timeout))
        output_path = Path(command[command.index("--output-last-message") + 1])
        output_path.write_text(
            '```repl\nanswer["content"] = "bridge ok"\nanswer["ready"] = True\n```',
            encoding="utf-8",
        )
        return CommandResult(exit_code=0, stdout="", stderr="")


@unittest.skipIf(RLM is None, "alexzhang13/rlm is not installed")
class RLMIntegrationTests(unittest.TestCase):
    def test_upstream_rlm_can_use_codex_backend(self):
        runner = ScriptedCodexRunner()

        with TemporaryDirectory() as tmpdir:
            restore = install_codex_backend()
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=ResourceWarning)
                    rlm = RLM(
                        backend="codex_agent",
                        backend_kwargs={
                            "model_name": "codex-cli",
                            "runner": runner,
                            "output_dir": tmpdir,
                            "extra_args": ["--ephemeral"],
                        },
                        environment="local",
                        max_depth=1,
                        max_iterations=1,
                        max_concurrent_subcalls=1,
                    )

                    result = rlm.completion("Return the bridge status.")
            finally:
                restore()

        self.assertEqual(result.response, "bridge ok")
        self.assertEqual(len(runner.calls), 1)
        command, cwd, _timeout = runner.calls[0]
        self.assertEqual(command[:2], ["codex", "exec"])
        self.assertIn("--ephemeral", command)
        self.assertIsNone(cwd)


if __name__ == "__main__":
    unittest.main()
