#!/usr/bin/env python3
"""Audit/fix branch comments for Python scripts.

Purpose:
- Ensure conditional branches are explicitly documented in public scripts.
- Provide a lightweight, in-place fixer that inserts a rationale comment
  immediately above `if` / `elif` lines when no nearby comment exists.

Usage:
    python -B scripts/summary/enforce_python_branch_comments.py --paths scripts
    python -B scripts/summary/enforce_python_branch_comments.py --paths scripts --fix
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence


IF_HEADER_RE = re.compile(r"^(?P<indent>\s*)(?P<kw>if|elif)\s+(?P<cond>.+):\s*(?:#.*)?$")


# クラス: `Violation` の責務と境界条件を定義する。
@dataclass
class Violation:
    """Represents one missing branch-comment finding."""

    line: int
    message: str


# クラス: `FileResult` の責務と境界条件を定義する。

@dataclass
class FileResult:
    """Represents one file audit/fix result."""

    path: Path
    violations: List[Violation]
    changed: bool


# 関数: `_detect_newline_style` の入出力契約と処理意図を定義する。

def _detect_newline_style(text: str) -> str:
    """Return dominant newline style from input text."""
    # 条件分岐: `"\r\n" in text` を満たす経路を評価する。
    if "\r\n" in text:
        return "\r\n"

    # 条件分岐: `"\n" in text` を満たす経路を評価する。

    if "\n" in text:
        return "\n"

    # 条件分岐: `"\r" in text` を満たす経路を評価する。

    if "\r" in text:
        return "\r"

    return "\n"


# 関数: `_iter_python_files` の入出力契約と処理意図を定義する。

def _iter_python_files(paths: Sequence[Path]) -> Iterable[Path]:
    """Yield Python files from path list."""
    for candidate in paths:
        # 条件分岐: `candidate.is_file() and candidate.suffix == ".py"` を満たす経路を評価する。
        if candidate.is_file() and candidate.suffix == ".py":
            yield candidate
            continue

        # 条件分岐: `candidate.is_dir()` を満たす経路を評価する。

        if candidate.is_dir():
            for python_file in sorted(candidate.rglob("*.py")):
                yield python_file


# 関数: `_previous_nonempty` の入出力契約と処理意図を定義する。

def _previous_nonempty(lines: Sequence[str]) -> str | None:
    """Return previous non-empty line from built output buffer."""
    for line_text in reversed(lines):
        # 条件分岐: `line_text.strip()` を満たす経路を評価する。
        if line_text.strip():
            return line_text

    return None


# 関数: `_normalize_condition_text` の入出力契約と処理意図を定義する。

def _normalize_condition_text(condition_text: str) -> str:
    """Normalize condition snippet for compact comments."""
    compact = " ".join(condition_text.strip().split())
    # 条件分岐: `len(compact) > 80` を満たす経路を評価する。
    if len(compact) > 80:
        compact = compact[:77].rstrip() + "..."

    return compact


# 関数: `_comment_for_branch` の入出力契約と処理意図を定義する。

def _comment_for_branch(*, keyword: str, condition: str) -> str:
    """Build branch comment text."""
    condition_text = _normalize_condition_text(condition)
    # 条件分岐: `keyword == "if"` を満たす経路を評価する。
    if keyword == "if":
        return f"# 条件分岐: `{condition_text}` を満たす経路を評価する。"

    return f"# 条件分岐: 前段条件が不成立で、`{condition_text}` を追加評価する。"


# 関数: `_audit_or_fix_text` の入出力契約と処理意図を定義する。

def _audit_or_fix_text(source_text: str, *, apply_fix: bool) -> tuple[str, List[Violation], bool]:
    """Audit/fix one source text and return normalized text + findings."""
    # 条件分岐: `not source_text` を満たす経路を評価する。
    if not source_text:
        return source_text, [], False

    newline_style = _detect_newline_style(source_text)
    had_terminal_newline = source_text.endswith(("\n", "\r"))
    original_lines = source_text.splitlines()

    output_lines: List[str] = []
    violations: List[Violation] = []
    changed = False

    for line_number, line_text in enumerate(original_lines, start=1):
        match = IF_HEADER_RE.match(line_text)
        # 条件分岐: `match is not None` を満たす経路を評価する。
        if match is not None:
            previous_nonempty = _previous_nonempty(output_lines)
            has_comment = previous_nonempty is not None and previous_nonempty.lstrip().startswith("#")
            # 条件分岐: `not has_comment` を満たす経路を評価する。
            if not has_comment:
                violations.append(Violation(line=line_number, message="missing conditional-branch comment"))
                # 条件分岐: `apply_fix` を満たす経路を評価する。
                if apply_fix:
                    indent = match.group("indent")
                    keyword = match.group("kw")
                    condition = match.group("cond")
                    output_lines.append(indent + _comment_for_branch(keyword=keyword, condition=condition))
                    changed = True

        output_lines.append(line_text)

    normalized_text = newline_style.join(output_lines)
    # 条件分岐: `had_terminal_newline` を満たす経路を評価する。
    if had_terminal_newline:
        normalized_text += newline_style

    return normalized_text, violations, changed


# 関数: `_run_for_file` の入出力契約と処理意図を定義する。

def _run_for_file(path: Path, *, apply_fix: bool) -> FileResult:
    """Run audit/fix for one file."""
    source_text = path.read_text(encoding="utf-8")
    normalized_text, violations, changed = _audit_or_fix_text(source_text, apply_fix=apply_fix)

    # 条件分岐: `apply_fix and changed and normalized_text != source_text` を満たす経路を評価する。
    if apply_fix and changed and normalized_text != source_text:
        path.write_text(normalized_text, encoding="utf-8", newline="")

    return FileResult(path=path, violations=violations, changed=changed and normalized_text != source_text)


# 関数: `parse_args` の入出力契約と処理意図を定義する。

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Audit/fix conditional branch comments in Python scripts.")
    parser.add_argument(
        "--paths",
        nargs="+",
        default=["scripts"],
        help="Target files/directories (default: scripts).",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Apply fixes in-place. Without this flag, only report findings.",
    )
    return parser.parse_args()


# 関数: `main` の入出力契約と処理意図を定義する。

def main() -> int:
    """CLI entry point."""
    args = parse_args()
    target_paths = [Path(path_text) for path_text in args.paths]
    python_files = sorted(set(_iter_python_files(target_paths)))

    # 条件分岐: `not python_files` を満たす経路を評価する。
    if not python_files:
        print("[info] no python files found.")
        return 0

    total_files = 0
    total_changed = 0
    total_violations = 0

    for python_file in python_files:
        total_files += 1
        file_result = _run_for_file(python_file, apply_fix=args.fix)
        # 条件分岐: `file_result.changed` を満たす経路を評価する。
        if file_result.changed:
            total_changed += 1

        # 条件分岐: `file_result.violations and not args.fix` を満たす経路を評価する。

        if file_result.violations and not args.fix:
            for violation in file_result.violations:
                print(f"[ng] {python_file}:{violation.line}: {violation.message}")

            total_violations += len(file_result.violations)
        # 条件分岐: 前段条件が不成立で、`file_result.violations` を追加評価する。
        elif file_result.violations:
            total_violations += len(file_result.violations)

    mode_text = "fix" if args.fix else "check"
    print(
        f"[summary] mode={mode_text} files={total_files} changed={total_changed} violations={total_violations}"
    )

    # 条件分岐: `args.fix` を満たす経路を評価する。
    if args.fix:
        return 0

    return 1 if total_violations > 0 else 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())
