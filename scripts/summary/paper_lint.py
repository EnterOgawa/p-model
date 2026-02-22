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


def _resolve_output_path(root: Path, rel: str) -> Path:
    rel_norm = rel.replace("\\", "/")
    if not rel_norm.startswith("output/"):
        return root / Path(rel)

    parts = Path(rel_norm).parts
    if len(parts) < 2:
        return root / Path(rel_norm)
    if parts[1] in ("private", "public"):
        return root / Path(rel_norm)

    topic = parts[1]
    tail = Path(*parts[2:]) if len(parts) > 2 else Path()
    cand_private = (root / "output" / "private" / topic / tail).resolve()
    cand_public = (root / "output" / "public" / topic / tail).resolve()

    if topic == "quantum":
        if cand_public.exists():
            return cand_public
        if cand_private.exists():
            return cand_private

    if cand_private.exists():
        return cand_private
    if cand_public.exists():
        return cand_public
    return root / Path(rel_norm)

_REF_KEY_RE = re.compile(r"^\s*-\s+\[([A-Za-z][A-Za-z0-9_-]{0,40})\]")
_CITE_RE = re.compile(r"\[([A-Za-z][A-Za-z0-9_-]{0,40})\]")
_PNG_CODE_RE = re.compile(r"`(output/[^`]+?\.png)`")
_FIG_INDEX_PNG_RE = re.compile(r"`(output/[^`]+?\.png)`\s*(?:（(.{0,200})）|\((.{0,200})\))?")
_UNFIXED_GH_MAIN_RE = re.compile(r"https://github\.com/EnterOgawa/p-model/(?:blob|tree)/main/")
_DELTA_UNIT_BLOCK_RE = re.compile(
    r"<!--\s*DELTA_UNIT:START\s+([A-Za-z0-9_.:-]+)\s*-->(.*?)<!--\s*DELTA_UNIT:END\s+\1\s*-->",
    re.DOTALL,
)
_MD_TABLE_SEP_RE = re.compile(r"^\s*\|(?:\s*:?-{3,}:?\s*\|)+\s*$", re.MULTILINE)
_HEADING_NUM_RE = re.compile(r"^\s*#{2,6}\s+([0-9]+(?:\.[0-9]+)*)\b")
_SECTION_REF_RE = re.compile(r"(?<![0-9])([0-9]+(?:\.[0-9]+)+)節")
_CROSS_PART_HINT_RE = re.compile(r"\bPart\s*(?:I|II|III|IV|1|2|3|4)\b", re.IGNORECASE)


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


def _extract_delta_units(md_text: str) -> List[Tuple[str, str]]:
    return [(m.group(1), m.group(2)) for m in _DELTA_UNIT_BLOCK_RE.finditer(md_text)]


def _count_falsification_lines(block_text: str) -> int:
    lines = block_text.splitlines()
    for idx, line in enumerate(lines):
        if line.strip().startswith("**反証条件**"):
            count = 0
            for next_line in lines[idx + 1 :]:
                text = next_line.strip()
                if not text:
                    if count > 0:
                        break
                    continue
                if text.startswith("**"):
                    break
                if text.startswith("- "):
                    count += 1
                    continue
                if count > 0:
                    break
            return count
    return 0


def _extract_heading_numbers(md_text: str) -> Set[str]:
    nums: Set[str] = set()
    for line in md_text.splitlines():
        m = _HEADING_NUM_RE.match(line)
        if not m:
            continue
        nums.add(m.group(1))
    return nums


def _find_unresolved_section_refs(md_text: str) -> List[str]:
    heading_nums = _extract_heading_numbers(md_text)
    findings: List[str] = []
    for lineno, line in enumerate(md_text.splitlines(), start=1):
        for m in _SECTION_REF_RE.finditer(line):
            ref_num = m.group(1)
            if ref_num in heading_nums:
                continue
            if _CROSS_PART_HINT_RE.search(line):
                continue
            findings.append(f"line {lineno}: {ref_num}節 (not found in manuscript headings)")
    return findings


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
        unresolved_section_refs = _find_unresolved_section_refs(md_text)
        if unresolved_section_refs:
            rel_md = str(md_path.relative_to(root)).replace("\\", "/")
            for entry in unresolved_section_refs:
                warnings.append(f"unresolved section reference: {rel_md} {entry}")
        unfixed_lines: List[int] = []
        for lineno, line in enumerate(md_text.splitlines(), start=1):
            if _UNFIXED_GH_MAIN_RE.search(line):
                unfixed_lines.append(lineno)
        if unfixed_lines:
            rel_md = str(md_path.relative_to(root)).replace("\\", "/")
            head = ", ".join(str(n) for n in unfixed_lines[:5])
            if len(unfixed_lines) > 5:
                head += ", ..."
            warnings.append(
                f"unfixed GitHub main link found (use release/tag snapshot): {rel_md} lines {head}"
            )

        # --- Delta-unit guard (Step 8.7.1)
        delta_units = _extract_delta_units(md_text)
        for unit_id, unit_body in delta_units:
            table_count = len(_MD_TABLE_SEP_RE.findall(unit_body))
            if table_count != 1:
                warnings.append(
                    f"delta-unit '{unit_id}' should contain exactly 1 markdown table (separator lines={table_count})"
                )
            png_count = len(_PNG_CODE_RE.findall(unit_body))
            if png_count != 1:
                warnings.append(
                    f"delta-unit '{unit_id}' should contain exactly 1 figure reference (png refs={png_count})"
                )
            fals_lines = _count_falsification_lines(unit_body)
            if fals_lines != 3:
                warnings.append(
                    f"delta-unit '{unit_id}' should contain exactly 3 falsification lines (found={fals_lines})"
                )

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
        png_path = _resolve_output_path(root, rel)
        if not png_path.exists():
            warnings.append(f"missing figure file referenced by manuscript (listed in index): {rel}")
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
