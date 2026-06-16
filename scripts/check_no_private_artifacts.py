#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEFAULT_FORBIDDEN_MARKERS = ()
SKIP_DIRS = {
    ".git",
    ".mypy_cache",
    ".private",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "benchmarks/runs",
    "build",
    "dist",
    "node_modules",
    "work",
}
TEXT_SUFFIXES = {
    "",
    ".cfg",
    ".cjs",
    ".css",
    ".html",
    ".js",
    ".json",
    ".md",
    ".py",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".yaml",
    ".yml",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Fail when public files contain private benchmark artifacts.")
    parser.add_argument("--root", default=".", help="Repository root to scan.")
    parser.add_argument(
        "--forbid",
        action="append",
        default=[],
        help="Forbidden marker. Can be repeated. Keep real private markers in local commands or CI secrets.",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    markers = tuple(args.forbid) if args.forbid else DEFAULT_FORBIDDEN_MARKERS
    violations = find_violations(root, markers)
    if violations:
        for path, marker in violations:
            print(f"{path}: contains forbidden marker {marker!r}", file=sys.stderr)
        return 1
    return 0


def find_violations(root: Path, markers: tuple[str, ...]) -> list[tuple[str, str]]:
    violations: list[tuple[str, str]] = []
    for path in _iter_public_text_files(root):
        text = _read_text(path)
        if text is None:
            continue
        for marker in markers:
            if marker and marker in text:
                violations.append((path.relative_to(root).as_posix(), marker))
    return violations


def _iter_public_text_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        if any(_is_skipped(relative, skip) for skip in SKIP_DIRS):
            continue
        if path.is_file() and path.suffix in TEXT_SUFFIXES:
            files.append(path)
    return files


def _is_skipped(relative_path: str, skip: str) -> bool:
    return relative_path == skip or relative_path.startswith(skip + "/")


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return None
    except OSError:
        return None


if __name__ == "__main__":
    raise SystemExit(main())
