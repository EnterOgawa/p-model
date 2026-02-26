#!/usr/bin/env python3
"""Enforce Python block-spacing rules mapped from Java brace-style guidance.

Rules implemented (Python interpretation):
1) Add one blank line after a dedent boundary (`if/for/while/with/def/class/match` blocks)
   before the next normal statement.
2) Do not add a blank line between `if` and `elif/else`.
3) Do not add a blank line between `try` and `except/finally`.
4) If a `case` header follows a `break`, insert one blank line between them.

Usage:
    python -B scripts/summary/enforce_python_block_spacing.py --paths scripts
    python -B scripts/summary/enforce_python_block_spacing.py --paths scripts --fix
"""

from __future__ import annotations

import argparse
import io
import tokenize
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Set


CONTINUATION_PREFIXES: tuple[str, ...] = ("elif ", "elif:", "else:", "except", "finally:")

SKIP_TOKEN_TYPES: Set[int] = {
    tokenize.NL,
    tokenize.NEWLINE,
    tokenize.INDENT,
    tokenize.DEDENT,
    tokenize.COMMENT,
    tokenize.ENDMARKER,
}


# クラス: `Violation` の責務と境界条件を定義する。
@dataclass
class Violation:
    """Represents a single spacing-rule violation."""

    line: int
    reason: str


# クラス: `FileResult` の責務と境界条件を定義する。

@dataclass
class FileResult:
    """Per-file audit/fix result."""

    path: Path
    violations: List[Violation]
    changed: bool


# 関数: `_detect_newline_style` の入出力契約と処理意図を定義する。

def _detect_newline_style(source_text: str) -> str:
    """Return the dominant newline style from source text."""
    # 条件分岐: `"\r\n" in source_text` を満たす経路を評価する。
    if "\r\n" in source_text:
        return "\r\n"

    # 条件分岐: `"\n" in source_text` を満たす経路を評価する。

    if "\n" in source_text:
        return "\n"

    # 条件分岐: `"\r" in source_text` を満たす経路を評価する。

    if "\r" in source_text:
        return "\r"

    return "\n"


# 関数: `_dedent_target_lines` の入出力契約と処理意図を定義する。

def _dedent_target_lines(source_text: str) -> Set[int]:
    """Collect first code-line numbers that appear immediately after DEDENT tokens."""
    token_stream = tokenize.generate_tokens(io.StringIO(source_text).readline)
    token_list = list(token_stream)
    target_lines: Set[int] = set()

    for token_index, token_info in enumerate(token_list):
        # 条件分岐: `token_info.type != tokenize.DEDENT` を満たす経路を評価する。
        if token_info.type != tokenize.DEDENT:
            continue

        look_ahead_index = token_index + 1
        while look_ahead_index < len(token_list) and token_list[look_ahead_index].type in SKIP_TOKEN_TYPES:
            look_ahead_index += 1

        # 条件分岐: `look_ahead_index < len(token_list)` を満たす経路を評価する。

        if look_ahead_index < len(token_list):
            target_lines.add(token_list[look_ahead_index].start[0])

    return target_lines


# 関数: `_previous_nonempty_line` の入出力契約と処理意図を定義する。

def _previous_nonempty_line(lines: Sequence[str]) -> str | None:
    """Return previous non-empty line from the already-built output buffer."""
    for line_text in reversed(lines):
        # 条件分岐: `line_text.strip()` を満たす経路を評価する。
        if line_text.strip():
            return line_text

    return None


# 関数: `_is_continuation_header` の入出力契約と処理意図を定義する。

def _is_continuation_header(stripped_line: str) -> bool:
    """Return True if a line starts an else/elif/except/finally continuation."""
    for prefix in CONTINUATION_PREFIXES:
        # 条件分岐: `stripped_line.startswith(prefix)` を満たす経路を評価する。
        if stripped_line.startswith(prefix):
            return True

    return False


# 関数: `_needs_blank_line_before` の入出力契約と処理意図を定義する。

def _needs_blank_line_before(
    line_number: int,
    line_text: str,
    dedent_targets: Set[int],
    output_lines: Sequence[str],
) -> str | None:
    """Return violation reason if one blank line must exist before current line."""
    stripped_line = line_text.lstrip()

    # Rule 4 (switch/case mapping): break -> blank -> next case
    if stripped_line.startswith("case "):
        previous_nonempty = _previous_nonempty_line(output_lines)
        # 条件分岐: `previous_nonempty is not None and previous_nonempty.strip() == "break"` を満たす経路を評価する。
        if previous_nonempty is not None and previous_nonempty.strip() == "break":
            return "missing blank line between 'break' and next 'case'"

    # 条件分岐: `line_number not in dedent_targets` を満たす経路を評価する。

    if line_number not in dedent_targets:
        return None

    # Rules 2/3: no blank required between continuation constructs.

    if _is_continuation_header(stripped_line):
        return None

    return "missing blank line after block dedent"


# 関数: `_normalize_block_spacing` の入出力契約と処理意図を定義する。

def _normalize_block_spacing(source_text: str, apply_fix: bool) -> tuple[str, List[Violation], bool]:
    """Check/fix one source file and return updated text + violations + changed flag."""
    # 条件分岐: `not source_text` を満たす経路を評価する。
    if not source_text:
        return source_text, [], False

    newline_style = _detect_newline_style(source_text)
    had_terminal_newline = source_text.endswith(("\n", "\r"))
    original_lines = source_text.splitlines()
    dedent_targets = _dedent_target_lines(source_text)

    output_lines: List[str] = []
    violations: List[Violation] = []
    changed = False

    for line_number, line_text in enumerate(original_lines, start=1):
        reason = _needs_blank_line_before(
            line_number=line_number,
            line_text=line_text,
            dedent_targets=dedent_targets,
            output_lines=output_lines,
        )

        # 条件分岐: `reason is not None` を満たす経路を評価する。
        if reason is not None:
            has_blank_separator = len(output_lines) > 0 and output_lines[-1].strip() == ""
            # 条件分岐: `not has_blank_separator` を満たす経路を評価する。
            if not has_blank_separator:
                violations.append(Violation(line=line_number, reason=reason))
                # 条件分岐: `apply_fix` を満たす経路を評価する。
                if apply_fix:
                    output_lines.append("")
                    changed = True

        output_lines.append(line_text)

    normalized_text = newline_style.join(output_lines)
    # 条件分岐: `had_terminal_newline` を満たす経路を評価する。
    if had_terminal_newline:
        normalized_text += newline_style

    return normalized_text, violations, changed


# 関数: `_iter_python_files` の入出力契約と処理意図を定義する。

def _iter_python_files(paths: Sequence[Path]) -> Iterable[Path]:
    """Yield python files under target paths."""
    for candidate in paths:
        # 条件分岐: `candidate.is_file() and candidate.suffix == ".py"` を満たす経路を評価する。
        if candidate.is_file() and candidate.suffix == ".py":
            yield candidate
            continue

        # 条件分岐: `candidate.is_dir()` を満たす経路を評価する。

        if candidate.is_dir():
            for python_file in sorted(candidate.rglob("*.py")):
                yield python_file


# 関数: `_run_for_path` の入出力契約と処理意図を定義する。

def _run_for_path(path: Path, apply_fix: bool) -> FileResult:
    """Run spacing audit/fix for one file."""
    source_text = path.read_text(encoding="utf-8")
    normalized_text, violations, changed = _normalize_block_spacing(source_text, apply_fix=apply_fix)

    # 条件分岐: `apply_fix and changed and normalized_text != source_text` を満たす経路を評価する。
    if apply_fix and changed and normalized_text != source_text:
        path.write_text(normalized_text, encoding="utf-8", newline="")

    return FileResult(path=path, violations=violations, changed=changed and normalized_text != source_text)


# 関数: `parse_args` の入出力契約と処理意図を定義する。

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Audit/fix Python block-spacing rules.")
    parser.add_argument(
        "--paths",
        nargs="+",
        default=["scripts"],
        help="Target files/directories (default: scripts).",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Apply fixes in-place. Without this flag, only report violations.",
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
        file_result = _run_for_path(python_file, apply_fix=args.fix)
        # 条件分岐: `file_result.changed` を満たす経路を評価する。
        if file_result.changed:
            total_changed += 1

        # 条件分岐: `file_result.violations` を満たす経路を評価する。

        if file_result.violations:
            for violation in file_result.violations:
                print(f"[ng] {python_file}:{violation.line}: {violation.reason}")

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
