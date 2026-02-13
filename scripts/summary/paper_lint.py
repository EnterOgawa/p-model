#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
paper_lint.py

Phase 8 / Step 8.2（本論文）向けの “整合チェック”。

目的：
- 草稿（doc/paper/*.md）で使っている引用キー [KEY] が参考文献に存在するか確認する
- 図表インデックス（doc/paper/01_figures_index.md）に列挙された PNG が実在するか確認する（※その草稿で参照される図のみ）
- 草稿で参照している PNG が図表インデックスに含まれるか確認する（図番号リンクの安定化）

出力：
- 標準出力にチェック結果を表示
- （任意）output/private/summary/work_history.jsonl に実行ログを追記
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Set, Tuple

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.summary import worklog  # noqa: E402


def _repo_root() -> Path:
    return _ROOT


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


_REF_KEY_RE = re.compile(r"^\s*-\s+\[([A-Za-z][A-Za-z0-9_-]{0,40})\]")
_CITE_RE = re.compile(r"\[([A-Za-z][A-Za-z0-9_-]{0,40})\]")
_PNG_CODE_RE = re.compile(r"`(output/[^`]+?\.png)`")
_FIG_INDEX_PNG_RE = re.compile(r"`(output/[^`]+?\.png)`\s*(?:（(.{0,200})）|\((.{0,200})\))?")


def _extract_reference_keys(ref_md: str) -> List[str]:
    keys: List[str] = []
    for line in ref_md.splitlines():
        m = _REF_KEY_RE.match(line)
        if not m:
            continue
        keys.append(m.group(1))
    # de-dup preserving order
    seen: Set[str] = set()
    uniq: List[str] = []
    for key in keys:
        if key in seen:
            continue
        seen.add(key)
        uniq.append(key)
    return uniq


def _extract_used_citations(md_text: str) -> Set[str]:
    return set(_CITE_RE.findall(md_text))


def _extract_png_refs(md_text: str) -> Set[str]:
    return set(_PNG_CODE_RE.findall(md_text))


def _extract_fig_index_pngs(fig_index_md: str) -> List[Tuple[str, str]]:
    """
    Return list of (png_path, caption) in appearance order (de-duplicated).
    """
    found: List[Tuple[str, str]] = []
    for line in fig_index_md.splitlines():
        m = _FIG_INDEX_PNG_RE.search(line)
        if not m:
            continue
        rel = m.group(1)
        caption = (m.group(2) or m.group(3) or "").strip()
        found.append((rel, caption))

    seen: Set[str] = set()
    uniq: List[Tuple[str, str]] = []
    for rel, caption in found:
        key = rel.lower()
        if key in seen:
            continue
        seen.add(key)
        uniq.append((rel, caption))
    return uniq


@dataclass(frozen=True)
class _LintResult:
    errors: List[str]
    warnings: List[str]

    def ok(self) -> bool:
        return not self.errors


def _lint(
    *,
    root: Path,
    manuscript_paths: Sequence[Path],
    references_path: Path,
    figures_index_path: Path,
) -> _LintResult:
    errors: List[str] = []
    warnings: List[str] = []

    # --- References / citations
    if not references_path.exists():
        errors.append(f"missing references file: {references_path}")
        return _LintResult(errors=errors, warnings=warnings)

    ref_md = _read_text(references_path)
    ref_keys = set(_extract_reference_keys(ref_md))
    if not ref_keys:
        warnings.append("no reference keys found in references (expected '- [KEY]' bullets)")

    used_keys: Set[str] = set()
    used_pngs: Set[str] = set()
    for md_path in manuscript_paths:
        if not md_path.exists():
            errors.append(f"missing manuscript file: {md_path}")
            continue
        md_text = _read_text(md_path)
        used_keys |= _extract_used_citations(md_text)
        used_pngs |= _extract_png_refs(md_text)

    missing_keys = sorted(used_keys - ref_keys)
    if missing_keys:
        errors.append(
            "missing citation keys in references: " + ", ".join(f"[{k}]" for k in missing_keys)
        )

    # --- Figures index
    if not figures_index_path.exists():
        errors.append(f"missing figures index: {figures_index_path}")
        return _LintResult(errors=errors, warnings=warnings)

    fig_idx_md = _read_text(figures_index_path)
    fig_items = _extract_fig_index_pngs(fig_idx_md)
    if not fig_items:
        warnings.append("no PNG entries found in figures index (expected backticked output/...png)")

    for rel, caption in fig_items:
        # Only enforce existence/caption for figures actually referenced by the manuscript.
        # The index may contain optional/planned figures that are not generated in a minimal build.
        if rel not in used_pngs:
            continue
        png_path = root / rel
        if not png_path.exists():
            errors.append(f"missing figure file referenced by manuscript (listed in index): {rel}")
            continue
        if not caption:
            warnings.append(f"no caption in figures index (referenced): {rel}")

    # --- Manuscript png refs must be in the figures index for stable (図N) linking
    idx_pngs = {rel for rel, _ in fig_items}

    missing_pngs = sorted(used_pngs - idx_pngs)
    if missing_pngs:
        warnings.append(
            "png referenced in manuscript but not listed in figures index (図番号リンクが不安定): "
            + ", ".join(missing_pngs)
        )

    return _LintResult(errors=errors, warnings=warnings)


def _print_block(title: str, items: Iterable[str]) -> None:
    items_list = list(items)
    if not items_list:
        return
    print(f"\n[{title}]")
    for item in items_list:
        print(f"- {item}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Phase 8 / Step 8.2 paper lint (citations / figures index)."
    )
    parser.add_argument(
        "--manuscript",
        action="append",
        default=[],
        help="manuscript markdown path (repeatable). default: doc/paper/10_manuscript.md",
    )
    parser.add_argument(
        "--references",
        default="doc/paper/30_references.md",
        help="references markdown path",
    )
    parser.add_argument(
        "--figures-index",
        default="doc/paper/01_figures_index.md",
        help="figures index markdown path",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="treat warnings as errors (non-zero exit)",
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="do not append an event to output/private/summary/work_history.jsonl",
    )

    args = parser.parse_args(list(argv) if argv is not None else None)
    root = _repo_root()

    manuscript_paths = [root / Path(p) for p in (args.manuscript or [])]
    if not manuscript_paths:
        manuscript_paths = [root / "doc" / "paper" / "10_manuscript.md"]

    references_path = root / Path(args.references)
    figures_index_path = root / Path(args.figures_index)

    result = _lint(
        root=root,
        manuscript_paths=manuscript_paths,
        references_path=references_path,
        figures_index_path=figures_index_path,
    )

    print("paper_lint:")
    print(f"- manuscript: {', '.join(str(p.relative_to(root)).replace('\\\\','/') for p in manuscript_paths)}")
    print(f"- references: {str(references_path.relative_to(root)).replace('\\\\','/')}")
    print(f"- figures_index: {str(figures_index_path.relative_to(root)).replace('\\\\','/')}")
    print(f"- errors: {len(result.errors)}")
    print(f"- warnings: {len(result.warnings)}")

    _print_block("errors", result.errors)
    _print_block("warnings", result.warnings)

    exit_code = 0
    if result.errors or (args.strict and result.warnings):
        exit_code = 1

    if not args.no_log:
        try:
            worklog.append_event(
                {
                    "tool": "paper_lint",
                    "args": {
                        "manuscript": [str(p.relative_to(root)).replace("\\", "/") for p in manuscript_paths],
                        "references": str(references_path.relative_to(root)).replace("\\", "/"),
                        "figures_index": str(figures_index_path.relative_to(root)).replace("\\", "/"),
                        "strict": bool(args.strict),
                    },
                    "result": {"errors": len(result.errors), "warnings": len(result.warnings), "exit_code": exit_code},
                }
            )
        except Exception as exc:
            print(f"\n[warn] failed to append worklog: {exc}")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
