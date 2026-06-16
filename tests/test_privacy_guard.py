import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


class PrivacyGuardTests(unittest.TestCase):
    def test_guard_fails_on_forbidden_marker(self):
        repo_root = Path(__file__).resolve().parents[1]

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "README.md").write_text("internal path: /private/project\n", encoding="utf-8")

            completed = subprocess.run(
                [
                    sys.executable,
                    str(repo_root / "scripts" / "check_no_private_artifacts.py"),
                    "--root",
                    str(root),
                    "--forbid",
                    "/private/project",
                ],
                cwd=repo_root,
                check=False,
                capture_output=True,
                text=True,
            )

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("README.md", completed.stderr)

    def test_guard_passes_when_ignored_private_dir_contains_marker(self):
        repo_root = Path(__file__).resolve().parents[1]

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / ".private").mkdir()
            (root / ".private" / "run.md").write_text("internal path: /private/project\n", encoding="utf-8")
            (root / "README.md").write_text("public docs\n", encoding="utf-8")

            completed = subprocess.run(
                [
                    sys.executable,
                    str(repo_root / "scripts" / "check_no_private_artifacts.py"),
                    "--root",
                    str(root),
                    "--forbid",
                    "/private/project",
                ],
                cwd=repo_root,
                check=False,
                capture_output=True,
                text=True,
            )

        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
