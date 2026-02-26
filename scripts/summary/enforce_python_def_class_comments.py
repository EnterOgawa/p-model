#!/usr/bin/env python3
"""Audit/fix declaration comments for Python scripts.

Purpose:
- Ensure every `def` / `class` declaration has an explicit rationale comment.
- Provide an in-place fixer that inserts a one-line comment immediately above
  the declaration block (or above the first decorator when decorators exist).

Usage:
    python -B scripts/summary/enforce_python_def_class_comments.py --paths scripts
    python -B scripts/summary/enforce_python_def_class_comments.py --paths scripts --fix
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


# クラス: `DeclarationTarget` の責務と境界条件を定義する。
@dataclass(frozen=True)
class DeclarationTarget:
    """Represents one declaration block that requires a preceding comment."""

    line: int
    indent: int
    kind: str
    name: str


# クラス: `Violation` の責務と境界条件を定義する。

@dataclass(frozen=True)
class Violation:
    """Represents one missing declaration-comment finding."""

    line: int
    message: str


# クラス: `FileResult` の責務と境界条件を定義する。

@dataclass(frozen=True)
class FileResult:
    """Represents one file audit/fix result."""

    path: Path
    violations: List[Violation]
    changed: bool


# 関数: `_detect_newline_style` の入出力契約と処理意図を定義する。

def _detect_newline_style(source_text: str) -> str:
    """Return dominant newline style from input text."""
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


# 関数: `_declaration_start_line` の入出力契約と処理意図を定義する。

def _declaration_start_line(node: ast.AST) -> int:
    """Return declaration start line, including decorators when present."""
    decorators = getattr(node, "decorator_list", [])
    # 条件分岐: `decorators` を満たす経路を評価する。
    if decorators:
        return min(decorator.lineno for decorator in decorators)

    return getattr(node, "lineno")


# 関数: `_collect_declaration_targets` の入出力契約と処理意図を定義する。

def _collect_declaration_targets(source_text: str) -> List[DeclarationTarget]:
    """Collect declaration blocks requiring comments from parsed AST."""
    module = ast.parse(source_text)
    targets: List[DeclarationTarget] = []
    for node in ast.walk(module):
        is_function = isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        is_class = isinstance(node, ast.ClassDef)
        # 条件分岐: `not (is_function or is_class)` を満たす経路を評価する。
        if not (is_function or is_class):
            continue

        kind = "class" if is_class else "function"
        target = DeclarationTarget(
            line=_declaration_start_line(node),
            indent=getattr(node, "col_offset", 0),
            kind=kind,
            name=getattr(node, "name"),
        )
        targets.append(target)

    return sorted(targets, key=lambda target: (target.line, target.indent, target.kind, target.name))


# 関数: `_previous_nonempty_line` の入出力契約と処理意図を定義する。

def _previous_nonempty_line(lines: Sequence[str]) -> str | None:
    """Return previous non-empty line from built output buffer."""
    for line_text in reversed(lines):
        # 条件分岐: `line_text.strip()` を満たす経路を評価する。
        if line_text.strip():
            return line_text

    return None


# 関数: `_comment_for_declaration` の入出力契約と処理意図を定義する。

def _comment_for_declaration(target: DeclarationTarget) -> str:
    """Build declaration comment text."""
    # 条件分岐: `target.kind == "class"` を満たす経路を評価する。
    if target.kind == "class":
        return f"# クラス: `{target.name}` の責務と境界条件を定義する。"

    return f"# 関数: `{target.name}` の入出力契約と処理意図を定義する。"


# 関数: `_audit_or_fix_text` の入出力契約と処理意図を定義する。

def _audit_or_fix_text(source_text: str, *, apply_fix: bool) -> tuple[str, List[Violation], bool]:
    """Audit/fix one source text and return normalized text + findings."""
    # 条件分岐: `not source_text` を満たす経路を評価する。
    if not source_text:
        return source_text, [], False

    newline_style = _detect_newline_style(source_text)
    had_terminal_newline = source_text.endswith(("\n", "\r"))
    original_lines = source_text.splitlines()
    declaration_targets = _collect_declaration_targets(source_text)
    targets_by_line: Dict[int, List[DeclarationTarget]] = {}
    for target in declaration_targets:
        targets_by_line.setdefault(target.line, []).append(target)

    output_lines: List[str] = []
    violations: List[Violation] = []
    changed = False

    for line_number, line_text in enumerate(original_lines, start=1):
        targets = targets_by_line.get(line_number, [])
        for target in targets:
            previous_nonempty = _previous_nonempty_line(output_lines)
            has_comment = previous_nonempty is not None and previous_nonempty.lstrip().startswith("#")
            # 条件分岐: `not has_comment` を満たす経路を評価する。
            if not has_comment:
                violations.append(
                    Violation(
                        line=target.line,
                        message=f"missing declaration comment for {target.kind} `{target.name}`",
                    )
                )
                # 条件分岐: `apply_fix` を満たす経路を評価する。
                if apply_fix:
                    output_lines.append((" " * target.indent) + _comment_for_declaration(target))
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
    parser = argparse.ArgumentParser(description="Audit/fix declaration comments in Python scripts.")
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
