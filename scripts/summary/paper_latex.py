#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
paper_latex.py

Markdown 論文（doc/paper/*.md）から配布用の .tex を生成する。
（最小依存：pandoc なしで動作）
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

_ROOT = Path(__file__).resolve().parents[2]
# 条件分岐: `str(_ROOT) not in sys.path` を満たす経路を評価する。
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.summary import worklog


# 関数: `_repo_root` の入出力契約と処理意図を定義する。
def _repo_root() -> Path:
    return _ROOT


# 関数: `_escape_tex` の入出力契約と処理意図を定義する。

def _escape_tex(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


# 関数: `_safe_label` の入出力契約と処理意図を定義する。

def _safe_label(text: str) -> str:
    t = text.lower()
    t = re.sub(r"[^a-z0-9]+", "-", t).strip("-")
    return t or "sec"


_GENERIC_SECTION_LABELS = {
    "sec",
    "section",
}


# 関数: `_extract_heading_number` の入出力契約と処理意図を定義する。
def _extract_heading_number(raw_title: str) -> str:
    m = re.match(r"^\s*(\d+(?:\.\d+)*)", raw_title)
    return m.group(1) if m else ""


# 関数: `_section_label_hint` の入出力契約と処理意図を定義する。

def _section_label_hint(raw_title: str, stripped_title: str) -> str:
    raw_lower = raw_title.lower()
    stripped_lower = stripped_title.lower()
    merged = f"{raw_lower} {stripped_lower}"

    # 条件分岐: `"ベルテスト" in raw_title or "bell" in merged` を満たす経路を評価する。
    if "ベルテスト" in raw_title or "bell" in merged:
        return "bell-test"

    # 条件分岐: `"原子核" in raw_title or "nuclear" in merged` を満たす経路を評価する。

    if "原子核" in raw_title or "nuclear" in merged:
        return "nuclear"

    # 条件分岐: `"原子・分子" in raw_title or ("atomic" in merged and "molecular" in merged)` を満たす経路を評価する。

    if "原子・分子" in raw_title or ("atomic" in merged and "molecular" in merged):
        return "atomic-molecular"

    # 条件分岐: `"物性" in raw_title or "condensed" in merged` を満たす経路を評価する。

    if "物性" in raw_title or "condensed" in merged:
        return "materials"

    # 条件分岐: `"統計力学" in raw_title or "熱力学" in raw_title or "thermo" in merged` を満たす経路を評価する。

    if "統計力学" in raw_title or "熱力学" in raw_title or "thermo" in merged:
        return "stat-thermo"

    # 条件分岐: `"ddr" in merged or "distance duality" in merged` を満たす経路を評価する。

    if "ddr" in merged or "distance duality" in merged:
        return "cosmo-ddr"

    if (
        "p場" in raw_title
        or "p 場" in raw_title
        or "時間波" in raw_title
        or (" p " in f" {merged} " and ("field" in merged or "potential" in merged))
    ):
        return "tf-pfield"

    if (
        "光" in raw_title
        or "屈折" in raw_title
        or "light" in merged
        or "photon" in merged
    ):
        return "tf-light"

    # 条件分岐: `"eht" in merged` を満たす経路を評価する。

    if "eht" in merged:
        return "eht"

    # 条件分岐: `"節マップ" in raw_title or "項目対応" in raw_title` を満たす経路を評価する。

    if "節マップ" in raw_title or "項目対応" in raw_title:
        return "section-map"

    # 条件分岐: `"検証サマリ" in raw_title or "scoreboard" in merged` を満たす経路を評価する。

    if "検証サマリ" in raw_title or "scoreboard" in merged:
        return "validation-summary"

    return ""


# 関数: `_build_section_label` の入出力契約と処理意図を定義する。

def _build_section_label(
    raw_title: str,
    stripped_title: str,
    *,
    used_labels: dict[str, int],
) -> str:
    number = _extract_heading_number(raw_title)
    number_tag = number.replace(".", "-") if number else ""

    base = _section_label_hint(raw_title, stripped_title) or _safe_label(stripped_title)
    # 条件分岐: `base == "p"` を満たす経路を評価する。
    if base == "p":
        base = "tf-pfield"

    # 条件分岐: `base in _GENERIC_SECTION_LABELS` を満たす経路を評価する。

    if base in _GENERIC_SECTION_LABELS:
        raw_base = _safe_label(raw_title)
        # 条件分岐: `raw_base not in _GENERIC_SECTION_LABELS` を満たす経路を評価する。
        if raw_base not in _GENERIC_SECTION_LABELS:
            base = raw_base

    # 条件分岐: `base in _GENERIC_SECTION_LABELS` を満たす経路を評価する。

    if base in _GENERIC_SECTION_LABELS:
        base = f"sec-{number_tag}" if number_tag else "sec-topic"

    candidate = f"{base}-s{number_tag}" if number_tag and not base.endswith(f"-s{number_tag}") else base
    n = used_labels.get(candidate, 0) + 1
    used_labels[candidate] = n
    return candidate if n == 1 else f"{candidate}-{n}"


_HEADING_PREFIX_RE = re.compile(r"^\s*\d{1,2}(?:\.\d{1,2})*(?:[.)：:]|\s)\s*")


# 関数: `_strip_heading_prefix` の入出力契約と処理意図を定義する。
def _strip_heading_prefix(title: str) -> str:
    t = title.strip()
    stripped = _HEADING_PREFIX_RE.sub("", t, count=1).strip()
    return stripped or t


# 関数: `_is_abstract_heading` の入出力契約と処理意図を定義する。

def _is_abstract_heading(title: str) -> bool:
    compact = re.sub(r"[\s\u3000\(\)（）\[\]【】<>＜＞:：._\-–—・,，、/]", "", title).lower()
    return compact in {"abstract", "要旨", "要旨abstract", "abstract要旨"}


_HEADING_INLINE_MATH_RE = re.compile(r"\$(.+?)\$")
_HEADING_LATEX_CMD_RE = re.compile(r"\\[A-Za-z]+")


# 関数: `_heading_math_to_pdftext` の入出力契約と処理意図を定義する。
def _heading_math_to_pdftext(payload: str) -> str:
    text = payload
    text = text.replace(r"\theta", "theta")
    text = text.replace(r"\phi", "phi")
    text = text.replace(r"\sigma", "sigma")
    text = text.replace(r"\Delta", "Delta")
    text = text.replace(r"\beta", "beta")
    text = text.replace(r"\gamma", "gamma")
    text = re.sub(r"\\mathrm\{([^{}]+)\}", r"\1", text)
    text = text.replace(r"\_", "_")
    text = _HEADING_LATEX_CMD_RE.sub("", text)
    text = text.replace("{", "").replace("}", "")
    text = text.replace("^", "")
    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


# 関数: `_heading_pdf_text` の入出力契約と処理意図を定義する。

def _heading_pdf_text(title: str) -> str:
    text = _HEADING_INLINE_MATH_RE.sub(lambda m: _heading_math_to_pdftext(m.group(1)), title)
    greek_plain = {
        "α": "alpha",
        "β": "beta",
        "γ": "gamma",
        "δ": "delta",
        "ε": "epsilon",
        "ζ": "zeta",
        "η": "eta",
        "θ": "theta",
        "ι": "iota",
        "κ": "kappa",
        "λ": "lambda",
        "μ": "mu",
        "ν": "nu",
        "ξ": "xi",
        "π": "pi",
        "ρ": "rho",
        "σ": "sigma",
        "τ": "tau",
        "φ": "phi",
        "χ": "chi",
        "ψ": "psi",
        "ω": "omega",
        "ℓ": "ell",
    }
    text = re.sub(
        r"([α-ωℓ])_([A-Za-z0-9]+)",
        lambda m: greek_plain.get(m.group(1), m.group(1)) + " " + m.group(2),
        text,
    )
    text = text.replace("`", "")
    text = re.sub(r"\s+", " ", text).strip()
    return text or title


# 関数: `_normalize_tex_path` の入出力契約と処理意図を定義する。

def _normalize_tex_path(path_text: str) -> str:
    return path_text.replace("\\", "/")


_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".pdf", ".svg", ".bmp", ".webp")


# 関数: `_is_image_path` の入出力契約と処理意図を定義する。
def _is_image_path(path_text: str) -> bool:
    normalized = _normalize_tex_path(path_text.strip())
    lowered = normalized.lower()
    return any(lowered.endswith(ext) for ext in _IMAGE_EXTS)


# 関数: `_match_leading_image_line` の入出力契約と処理意図を定義する。

def _match_leading_image_line(line_text: str) -> Optional[tuple[str, str]]:
    s = line_text.strip()
    m = re.match(
        r"^(?:[-*+]\s+)?`?([^\s`]+\.(?:png|jpg|jpeg|pdf|svg|bmp|webp))`?(?:\s+(.*))?$",
        s,
        flags=re.IGNORECASE,
    )
    # 条件分岐: `not m` を満たす経路を評価する。
    if not m:
        return None

    path_text = m.group(1).strip()
    # 条件分岐: `not _is_image_path(path_text)` を満たす経路を評価する。
    if not _is_image_path(path_text):
        return None

    desc_text = (m.group(2) or "").strip()
    return path_text, desc_text


# 関数: `_fallback_caption_from_path` の入出力契約と処理意図を定義する。

def _fallback_caption_from_path(raw_path: str) -> str:
    stem = Path(raw_path).stem
    normalized = stem.replace("__", " ").replace("_", " ").replace("-", " ")
    normalized = re.sub(r"\s+", " ", normalized).strip()
    # 条件分岐: `not normalized` を満たす経路を評価する。
    if not normalized:
        return "観測・理論比較の結果図。"

    token_map = {
        "llr": "LLR",
        "eht": "EHT",
        "gw": "重力波",
        "cosmology": "宇宙論",
        "quantum": "量子",
        "nuclear": "核",
        "xrism": "XRISM",
        "gps": "GPS",
        "cassini": "Cassini",
        "viking": "Viking",
        "mercury": "Mercury",
        "pulsar": "連星パルサー",
        "scoreboard": "総合スコア",
        "residual": "残差",
        "constraints": "制約",
        "summary": "要約",
        "audit": "監査",
        "mapping": "写像",
        "phase": "位相",
        "interference": "干渉",
    }

    words = []
    for token in normalized.split(" "):
        words.append(token_map.get(token.lower(), token))

    text = " ".join(words).strip()
    return f"{text}の比較結果を示す。"


# 関数: `_is_image_markdown_line` の入出力契約と処理意図を定義する。

def _is_image_markdown_line(stripped: str) -> bool:
    # 条件分岐: `_match_leading_image_line(stripped)` を満たす経路を評価する。
    if _match_leading_image_line(stripped):
        return True

    return bool(re.match(r"^!\[([^\]]*)\]\(([^)]+)\)\s*$", stripped))


# 関数: `_extract_following_caption` の入出力契約と処理意図を定義する。

def _extract_following_caption(lines: list[str], start_index: int) -> tuple[str, int]:
    j = start_index
    while j < len(lines):
        raw = lines[j]
        stripped = raw.strip()
        # 条件分岐: `not stripped` を満たす経路を評価する。
        if not stripped:
            break

        # 条件分岐: `stripped.startswith("```") or stripped == "$$"` を満たす経路を評価する。

        if stripped.startswith("```") or stripped == "$$":
            break

        # 条件分岐: `re.match(r"^(#{1,6})\s+", stripped)` を満たす経路を評価する。

        if re.match(r"^(#{1,6})\s+", stripped):
            break

        # 条件分岐: `_is_image_markdown_line(stripped)` を満たす経路を評価する。

        if _is_image_markdown_line(stripped):
            break

        # 条件分岐: `re.match(r"^\s*[-*]\s+(.+)$", raw) or re.match(r"^\s*\d+[.)]\s+(.+)$", raw)` を満たす経路を評価する。

        if re.match(r"^\s*[-*]\s+(.+)$", raw) or re.match(r"^\s*\d+[.)]\s+(.+)$", raw):
            break

        # 条件分岐: `"|" in raw and (j + 1) < len(lines) and _is_table_separator(lines[j + 1])` を満たす経路を評価する。

        if "|" in raw and (j + 1) < len(lines) and _is_table_separator(lines[j + 1]):
            break

        # 条件分岐: `_is_table_separator(raw)` を満たす経路を評価する。

        if _is_table_separator(raw):
            break

        candidate = re.sub(r"\s{2,}$", "", stripped).strip()
        # 条件分岐: `candidate` を満たす経路を評価する。
        if candidate:
            return candidate, (j - start_index + 1)

        j += 1

    return "", 0


# 関数: `_resolve_image_path` の入出力契約と処理意図を定義する。

def _resolve_image_path(raw_path: str, *, root: Path) -> tuple[str, bool]:
    normalized = _normalize_tex_path(raw_path.strip())
    # 条件分岐: `normalized.startswith("http://") or normalized.startswith("https://")` を満たす経路を評価する。
    if normalized.startswith("http://") or normalized.startswith("https://"):
        return normalized, False

    candidate_paths: list[Path] = []
    candidate_norms: set[str] = set()

    # 関数: `add_candidate` の入出力契約と処理意図を定義する。
    def add_candidate(path_obj: Path) -> None:
        key = str(path_obj.resolve()) if path_obj.is_absolute() else str(path_obj)
        # 条件分岐: `key in candidate_norms` を満たす経路を評価する。
        if key in candidate_norms:
            return

        candidate_norms.add(key)
        candidate_paths.append(path_obj)

    path_obj = Path(normalized)
    # 条件分岐: `path_obj.is_absolute()` を満たす経路を評価する。
    if path_obj.is_absolute():
        add_candidate(path_obj)
    else:
        add_candidate(root / path_obj)

    # 条件分岐: `normalized.startswith("output/") and not normalized.startswith("output/public...` を満たす経路を評価する。

    if normalized.startswith("output/") and not normalized.startswith("output/public/") and not normalized.startswith("output/private/"):
        tail = normalized[len("output/") :]
        add_candidate(root / "output" / "public" / Path(tail))
        add_candidate(root / "output" / "private" / Path(tail))

    resolved_existing = next((candidate for candidate in candidate_paths if candidate.exists()), None)
    # 条件分岐: `resolved_existing is not None` を満たす経路を評価する。
    if resolved_existing is not None:
        return str(resolved_existing), True

    # 条件分岐: `candidate_paths` を満たす経路を評価する。

    if candidate_paths:
        return str(candidate_paths[0]), False

    return normalized, False


_REFERENCE_KEYS: set[str] = set()
_REFERENCE_ORDER: list[str] = []
_REFERENCE_TEXT: dict[str, str] = {}
_USED_REFERENCE_KEYS: set[str] = set()


# 関数: `_load_reference_entries` の入出力契約と処理意図を定義する。
def _load_reference_entries(references_md: Path) -> tuple[list[str], dict[str, str]]:
    # 条件分岐: `not references_md.exists()` を満たす経路を評価する。
    if not references_md.exists():
        return [], {}

    order: list[str] = []
    refs: dict[str, str] = {}
    in_internal_block = False
    for raw_line in references_md.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = raw_line.strip()
        # 条件分岐: `stripped == "<!-- INTERNAL_ONLY_START -->"` を満たす経路を評価する。
        if stripped == "<!-- INTERNAL_ONLY_START -->":
            in_internal_block = True
            continue

        # 条件分岐: `stripped == "<!-- INTERNAL_ONLY_END -->"` を満たす経路を評価する。

        if stripped == "<!-- INTERNAL_ONLY_END -->":
            in_internal_block = False
            continue

        # 条件分岐: `in_internal_block` を満たす経路を評価する。

        if in_internal_block:
            continue

        match = re.match(r"^\s*-\s+\[([A-Za-z0-9][A-Za-z0-9_.:-]*)\]\s+(.+)$", raw_line)
        # 条件分岐: `not match` を満たす経路を評価する。
        if not match:
            continue

        key = match.group(1).strip()
        text = match.group(2).strip()
        # 条件分岐: `key not in refs` を満たす経路を評価する。
        if key not in refs:
            order.append(key)

        refs[key] = text

    return order, refs


# 関数: `_render_bibliography_section` の入出力契約と処理意図を定義する。

def _render_bibliography_section() -> str:
    # 条件分岐: `not _USED_REFERENCE_KEYS` を満たす経路を評価する。
    if not _USED_REFERENCE_KEYS:
        return ""

    lines: list[str] = ["", r"\clearpage", r"\section*{References}", r"\begin{thebibliography}{99}"]
    ordered_used = [key for key in _REFERENCE_ORDER if key in _USED_REFERENCE_KEYS]
    for key in ordered_used:
        ref_text = _REFERENCE_TEXT.get(key, "").strip()
        # 条件分岐: `not ref_text` を満たす経路を評価する。
        if not ref_text:
            continue

        rendered = _convert_inline(ref_text)
        rendered = re.sub(r"\\texttt\{(https?://[^{}]+)\}", r"\\url{\1}", rendered)
        lines.append(r"\bibitem{" + key + "} " + rendered)

    lines += [r"\end{thebibliography}", ""]
    return "\n".join(lines)


# 関数: `_render_figure_block` の入出力契約と処理意図を定義する。

def _render_figure_block(
    *,
    raw_path: str,
    caption: str,
    root: Path,
    outdir: Path,
    figures_dir: Path,
    staged_assets: dict[str, str],
    used_figure_names: set[str],
    used_figure_labels: dict[str, int],
) -> list[str]:
    resolved_path, exists = _resolve_image_path(raw_path, root=root)
    # 条件分岐: `resolved_path.startswith("http://") or resolved_path.startswith("https://")` を満たす経路を評価する。
    if resolved_path.startswith("http://") or resolved_path.startswith("https://"):
        return [r"\noindent\href{" + _escape_tex(resolved_path) + "}{" + _convert_inline(caption or raw_path) + "}", ""]

    resolved_obj = Path(resolved_path)
    tex_path = Path(raw_path).name or "missing_figure.png"

    # 条件分岐: `exists` を満たす経路を評価する。
    if exists:
        try:
            source_key = str(resolved_obj.resolve())
        except Exception:
            source_key = str(resolved_obj)

        # 条件分岐: `source_key in staged_assets` を満たす経路を評価する。

        if source_key in staged_assets:
            tex_path = staged_assets[source_key]
        else:
            base_name = resolved_obj.name
            stem = Path(base_name).stem
            suffix = Path(base_name).suffix
            candidate = base_name
            serial = 2
            while candidate.lower() in used_figure_names:
                candidate = f"{stem}__{serial}{suffix}"
                serial += 1

            used_figure_names.add(candidate.lower())
            dst = figures_dir / candidate
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(resolved_obj, dst)
            tex_path = candidate
            staged_assets[source_key] = candidate

    normalized_caption = caption.strip()
    normalized_caption = re.sub(r"[:：]\s*$", "", normalized_caption).strip()
    # 条件分岐: `not normalized_caption` を満たす経路を評価する。
    if not normalized_caption:
        normalized_caption = _fallback_caption_from_path(raw_path)

    caption_text = _convert_inline(normalized_caption)
    # 条件分岐: `not exists` を満たす経路を評価する。
    if not exists:
        caption_text = _convert_inline(f"{normalized_caption} (missing file: {raw_path})")

    label_source = Path(tex_path).stem or Path(raw_path).stem or "figure"
    label_base = f"fig:{_safe_label(label_source)}"
    label_count = used_figure_labels.get(label_base, 0) + 1
    used_figure_labels[label_base] = label_count
    figure_label = label_base if label_count == 1 else f"{label_base}-{label_count}"

    return [
        r"\begin{figure}[H]",
        r"\centering",
        r"\includegraphics[width=\linewidth]{" + _normalize_tex_path(tex_path) + "}",
        r"\caption{" + caption_text + "}",
        r"\label{" + _escape_tex(figure_label) + "}",
        r"\end{figure}",
        "",
    ]


_GREEK_UNICODE_TO_LATEX = {
    "α": r"\alpha",
    "β": r"\beta",
    "γ": r"\gamma",
    "δ": r"\delta",
    "ε": r"\epsilon",
    "ζ": r"\zeta",
    "η": r"\eta",
    "θ": r"\theta",
    "ι": r"\iota",
    "κ": r"\kappa",
    "λ": r"\lambda",
    "μ": r"\mu",
    "ν": r"\nu",
    "ξ": r"\xi",
    "π": r"\pi",
    "ρ": r"\rho",
    "σ": r"\sigma",
    "τ": r"\tau",
    "φ": r"\phi",
    "χ": r"\chi",
    "ψ": r"\psi",
    "ω": r"\omega",
    "Γ": r"\Gamma",
    "Δ": r"\Delta",
    "Θ": r"\Theta",
    "Λ": r"\Lambda",
    "Ξ": r"\Xi",
    "Π": r"\Pi",
    "Σ": r"\Sigma",
    "Φ": r"\Phi",
    "Ψ": r"\Psi",
    "Ω": r"\Omega",
}

_MATH_UNICODE_TO_LATEX = {
    "−": "-",
    "–": "-",
    "—": "-",
    "≒": r"\approx",
    "≈": r"\approx",
    "≃": r"\simeq",
    "≠": r"\neq",
    "≤": r"\le",
    "≥": r"\ge",
    "→": r"\to",
    "⇒": r"\Rightarrow",
    "⇔": r"\Leftrightarrow",
    "∝": r"\propto",
    "∇": r"\nabla",
    "□": r"\Box",
    "∞": r"\infty",
    "×": r"\times",
    "·": r"\cdot",
    "⋅": r"\cdot",
    "∂": r"\partial",
    "∫": r"\int",
    "≡": r"\equiv",
    "±": r"\pm",
    "∥": r"\parallel",
    "⊥": r"\perp",
    "ℓ": r"\ell",
}

_SUPERSCRIPT_TO_ASCII = {
    "⁰": "^0",
    "¹": "^1",
    "²": "^2",
    "³": "^3",
    "⁴": "^4",
    "⁵": "^5",
    "⁶": "^6",
    "⁷": "^7",
    "⁸": "^8",
    "⁹": "^9",
}

_SUBSCRIPT_TO_ASCII = {
    "₀": "_0",
    "₁": "_1",
    "₂": "_2",
    "₃": "_3",
    "₄": "_4",
    "₅": "_5",
    "₆": "_6",
    "₇": "_7",
    "₈": "_8",
    "₉": "_9",
}

_CODE_FILE_EXT_RE = re.compile(
    r"\.(?:json|csv|png|jpg|jpeg|pdf|svg|bmp|webp|txt|md|tex|html|docx|py|bat|sh|yaml|yml|toml|ini|log|gz|zip|tar|tgz)$",
    flags=re.IGNORECASE,
)

_MATH_GREEK_OR_SYMBOL_RE = re.compile(r"[α-ωΑ-ΩΔΘΛΞΠΣΦΨΩℓ∇∂∫≠≤≥≈≃→⇒⇔∝∞×⋅·±≡∥⊥□]")
_PUNCT_ONLY_RE = re.compile(r"^[\s,，、。.:：;；!！?？()\[\]{}<>＜＞「」『』【】/／・\-+*|]+$")
_CITATION_BLOCK_RE = re.compile(
    r"\[(?P<keys>[A-Za-z0-9][A-Za-z0-9_.:-]*(?:\s*[,;]\s*[A-Za-z0-9][A-Za-z0-9_.:-]*)*)\]"
)
_GREEK_CMD_GLUE_RE = re.compile(
    r"\\(alpha|beta|gamma|delta|epsilon|zeta|eta|theta|iota|kappa|lambda|mu|nu|xi|pi|rho|sigma|tau|phi|chi|psi|omega|Gamma|Delta|Theta|Lambda|Xi|Pi|Sigma|Phi|Psi|Omega)(?=[A-Za-z])"
)
_GREEK_NAME_TOKEN = (
    "alpha|beta|gamma|delta|epsilon|zeta|eta|theta|iota|kappa|lambda|mu|nu|xi|pi|rho|sigma|tau|phi|chi|psi|omega|"
    "Gamma|Delta|Theta|Lambda|Xi|Pi|Sigma|Phi|Psi|Omega|ell"
)
_GREEK_CMD_SUBSCRIPT_RE = re.compile(
    r"\\(?P<cmd>"
    + _GREEK_NAME_TOKEN
    + r")_(?P<sub>[A-Za-z][A-Za-z0-9]*)\b"
)
_LATIN_VAR_SUBSCRIPT_RE = re.compile(
    r"(?<!\\)\b(?P<base>[A-Za-z])_(?P<sub>[A-Za-z][A-Za-z0-9]*)\b"
)
_PHYSICS_SINGLE_LHS_RE = re.compile(
    r"^(?:"
    r"[A-Za-z]"
    r"|[α-ωΑ-ΩΔΘΛΞΠΣΦΨΩℓ]"
    r"|\\[A-Za-z]+"
    r")(?:_[A-Za-z0-9]+)?(?:\([^()]*\))?$"
)
_PHYSICS_ASCII_GREEK_TOKEN_RE = re.compile(
    r"^(?:"
    + _GREEK_NAME_TOKEN
    + r")(?:_[A-Za-z0-9]+)?(?:\([^()]*\))?$"
)


# 関数: `_looks_like_artifact_code` の入出力契約と処理意図を定義する。
def _looks_like_artifact_code(s: str) -> bool:
    candidate = s.strip()
    # 条件分岐: `not candidate` を満たす経路を評価する。
    if not candidate:
        return False

    low = candidate.lower()
    # 条件分岐: `"://" in candidate` を満たす経路を評価する。
    if "://" in candidate:
        return True

    # 条件分岐: `re.match(r"^[A-Za-z]:[\\/]", candidate)` を満たす経路を評価する。

    if re.match(r"^[A-Za-z]:[\\/]", candidate):
        return True

    # 条件分岐: `low.startswith(("output/", "scripts/", "data/", "doc/", "./", "../", ".\\", "...` を満たす経路を評価する。

    if low.startswith(("output/", "scripts/", "data/", "doc/", "./", "../", ".\\", "..\\")):
        return True

    # 条件分岐: `_CODE_FILE_EXT_RE.search(low)` を満たす経路を評価する。

    if _CODE_FILE_EXT_RE.search(low):
        return True

    # 条件分岐: `re.fullmatch(r"[A-Za-z0-9_.-]+/", candidate)` を満たす経路を評価する。

    if re.fullmatch(r"[A-Za-z0-9_.-]+/", candidate):
        return True

    # 条件分岐: `_MATH_GREEK_OR_SYMBOL_RE.search(candidate)` を満たす経路を評価する。

    if _MATH_GREEK_OR_SYMBOL_RE.search(candidate):
        return False

    # 条件分岐: `"%" in candidate` を満たす経路を評価する。

    if "%" in candidate:
        return True

    # 条件分岐: `re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*(?:\.[A-Za-z0-9_]+)+", candidate)` を満たす経路を評価する。

    if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*(?:\.[A-Za-z0-9_]+)+", candidate):
        return True

    if re.fullmatch(
        r"[A-Za-z][A-Za-z0-9_.]*(?:_[A-Za-z0-9_.]+)+\s*(?:<=|>=|==|!=|<|>)\s*[-+]?(?:\d+(?:\.\d+)?|true|false)",
        low,
    ):
        return True

    if re.fullmatch(
        r"(?:max|min)\s*\|?.+\|?\s*(?:<=|>=|<|>)\s*[-+]?\d+(?:\.\d+)?",
        candidate,
    ) and "\\" not in candidate:
        return True

    # 条件分岐: `"=" in candidate and " " in candidate and "\\" not in candidate and not re.se...` を満たす経路を評価する。

    if "=" in candidate and " " in candidate and "\\" not in candidate and not re.search(r"[{}^]", candidate):
        lhs = candidate.split("=", 1)[0].strip()
        # 条件分岐: `len(lhs) >= 4 and re.search(r"[A-Za-z]", lhs)` を満たす経路を評価する。
        if len(lhs) >= 4 and re.search(r"[A-Za-z]", lhs):
            return True

    # 条件分岐: `candidate.count("_") >= 2 and "\\" not in candidate` を満たす経路を評価する。

    if candidate.count("_") >= 2 and "\\" not in candidate:
        return True

    # 条件分岐: `re.search(r"[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+){2,}", candidate)` を満たす経路を評価する。

    if re.search(r"[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+){2,}", candidate):
        return True

    m_snake = re.fullmatch(r"[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+", candidate)
    # 条件分岐: `m_snake` を満たす経路を評価する。
    if m_snake:
        parts = candidate.split("_")
        # 条件分岐: `len(parts) >= 3` を満たす経路を評価する。
        if len(parts) >= 3:
            return True

        # 条件分岐: `len(parts) == 2 and (len(parts[0]) > 1 or len(parts[1]) > 1)` を満たす経路を評価する。

        if len(parts) == 2 and (len(parts[0]) > 1 or len(parts[1]) > 1):
            return True

    # 条件分岐: `re.match(r"^(?:--?)[A-Za-z0-9][A-Za-z0-9_.-]*(?:=[^\\s]+)?$", candidate)` を満たす経路を評価する。

    if re.match(r"^(?:--?)[A-Za-z0-9][A-Za-z0-9_.-]*(?:=[^\\s]+)?$", candidate):
        return True

    m_keyval = re.match(
        r"^(?P<lhs>[A-Za-z][A-Za-z0-9_]*(?:\.[A-Za-z0-9_]+)*)\s*=\s*(?P<rhs>[^=]+)$",
        candidate,
    )
    # 条件分岐: `m_keyval` を満たす経路を評価する。
    if m_keyval:
        lhs = m_keyval.group("lhs")
        rhs = m_keyval.group("rhs")
        lhs_low = lhs.lower()
        rhs_low = rhs.lower()
        # 条件分岐: `re.search(r"[\\α-ωΑ-ΩΔΘΛΞΠΣΦΨΩ^{}()|]", rhs)` を満たす経路を評価する。
        if re.search(r"[\\α-ωΑ-ΩΔΘΛΞΠΣΦΨΩ^{}()|]", rhs):
            return False

        # 条件分岐: `re.search(r"[A-Z]", lhs) or len(lhs) <= 3` を満たす経路を評価する。

        if re.search(r"[A-Z]", lhs) or len(lhs) <= 3:
            return False

        # 条件分岐: `lhs_low.count("_") >= 1 and len(lhs_low) >= 6` を満たす経路を評価する。

        if lhs_low.count("_") >= 1 and len(lhs_low) >= 6:
            return True

        # 条件分岐: `lhs_low in {"event", "event_counter", "next", "source", "selected", "target",...` を満たす経路を評価する。

        if lhs_low in {"event", "event_counter", "next", "source", "selected", "target", "without", "shift"}:
            return True

        # 条件分岐: `re.search(r"[A-Za-z0-9]+_[A-Za-z0-9_]+", rhs)` を満たす経路を評価する。

        if re.search(r"[A-Za-z0-9]+_[A-Za-z0-9_]+", rhs):
            return True

        # 条件分岐: `rhs_low in {"pass", "watch", "reject", "true", "false", "none"}` を満たす経路を評価する。

        if rhs_low in {"pass", "watch", "reject", "true", "false", "none"}:
            return True

    if low in {
        "pass",
        "watch",
        "reject",
        "true",
        "false",
        "derived",
        "inconclusive",
        "a_continue",
        "a_reject",
        "b_continue",
        "b_reject",
        "no_change",
    }:
        return True

    return False


# 関数: `_looks_like_math_code` の入出力契約と処理意図を定義する。

def _looks_like_math_code(s: str) -> bool:
    candidate = s.strip()
    # 条件分岐: `not candidate` を満たす経路を評価する。
    if not candidate:
        return False

    # 条件分岐: `_looks_like_artifact_code(candidate)` を満たす経路を評価する。

    if _looks_like_artifact_code(candidate):
        return False

    # 条件分岐: `_MATH_GREEK_OR_SYMBOL_RE.search(candidate)` を満たす経路を評価する。

    if _MATH_GREEK_OR_SYMBOL_RE.search(candidate):
        return True

    # 条件分岐: `re.search(r"\\[A-Za-z]+", candidate)` を満たす経路を評価する。

    if re.search(r"\\[A-Za-z]+", candidate):
        return True

    # 条件分岐: `re.search(r"[A-Za-z][_^][A-Za-z0-9\\{(]", candidate)` を満たす経路を評価する。

    if re.search(r"[A-Za-z][_^][A-Za-z0-9\\{(]", candidate):
        return True

    # 条件分岐: `re.search(r"[A-Za-z]\([A-Za-z0-9_,+\-*/ ]+\)", candidate)` を満たす経路を評価する。

    if re.search(r"[A-Za-z]\([A-Za-z0-9_,+\-*/ ]+\)", candidate):
        return True

    # 条件分岐: `" " not in candidate and re.search(r"[+\-*/]", candidate) and re.search(r"[A-...` を満たす経路を評価する。

    if " " not in candidate and re.search(r"[+\-*/]", candidate) and re.search(r"[A-Za-zα-ωΑ-Ω]", candidate):
        return True

    # 条件分岐: `candidate in {"ln", "exp", "sqrt()", "sin", "cos", "tan", "max", "min"}` を満たす経路を評価する。

    if candidate in {"ln", "exp", "sqrt()", "sin", "cos", "tan", "max", "min"}:
        return True

    # 条件分岐: `re.search(r"[=<>|]", candidate)` を満たす経路を評価する。

    if re.search(r"[=<>|]", candidate):
        return True

    # 条件分岐: `re.fullmatch(r"[A-Za-z](?:/[A-Za-z0-9_]+)+", candidate)` を満たす経路を評価する。

    if re.fullmatch(r"[A-Za-z](?:/[A-Za-z0-9_]+)+", candidate):
        return True

    # 条件分岐: `re.fullmatch(r"[A-Za-z][0-9]+", candidate)` を満たす経路を評価する。

    if re.fullmatch(r"[A-Za-z][0-9]+", candidate):
        return True

    # 条件分岐: `re.fullmatch(r"[A-Za-z](?:_[A-Za-z0-9]+)?", candidate)` を満たす経路を評価する。

    if re.fullmatch(r"[A-Za-z](?:_[A-Za-z0-9]+)?", candidate):
        return True

    # 条件分岐: `" " in candidate and _MATH_GREEK_OR_SYMBOL_RE.search(candidate)` を満たす経路を評価する。

    if " " in candidate and _MATH_GREEK_OR_SYMBOL_RE.search(candidate):
        return True

    return False


# 関数: `_format_subscript_token` の入出力契約と処理意図を定義する。

def _format_subscript_token(sub: str) -> str:
    # 条件分岐: `re.fullmatch(r"[A-Za-z0-9]", sub)` を満たす経路を評価する。
    if re.fullmatch(r"[A-Za-z0-9]", sub):
        return sub

    # 条件分岐: `"_" in sub` を満たす経路を評価する。

    if "_" in sub:
        return r"\mathrm{" + sub.replace("_", r"\_") + "}"

    return r"\mathrm{" + sub + "}"


# 関数: `_normalize_word_subscripts` の入出力契約と処理意図を定義する。

def _normalize_word_subscripts(text: str) -> str:
    normalized = text
    normalized = _GREEK_CMD_SUBSCRIPT_RE.sub(
        lambda m: rf"\{m.group('cmd')}_{{{_format_subscript_token(m.group('sub'))}}}",
        normalized,
    )
    normalized = _LATIN_VAR_SUBSCRIPT_RE.sub(
        lambda m: rf"{m.group('base')}_{{{_format_subscript_token(m.group('sub'))}}}",
        normalized,
    )
    return normalized


# 関数: `_looks_like_physics_equation_code` の入出力契約と処理意図を定義する。

def _looks_like_physics_equation_code(s: str) -> bool:
    candidate = s.strip()
    # 条件分岐: `not candidate` を満たす経路を評価する。
    if not candidate:
        return False

    low = candidate.lower()
    # 条件分岐: `"://" in candidate` を満たす経路を評価する。
    if "://" in candidate:
        return False

    # 条件分岐: `re.match(r"^[A-Za-z]:[\\/]", candidate)` を満たす経路を評価する。

    if re.match(r"^[A-Za-z]:[\\/]", candidate):
        return False

    # 条件分岐: `low.startswith(("output/", "scripts/", "data/", "doc/", "./", "../", ".\\", "...` を満たす経路を評価する。

    if low.startswith(("output/", "scripts/", "data/", "doc/", "./", "../", ".\\", "..\\")):
        return False

    # 条件分岐: `_CODE_FILE_EXT_RE.search(low)` を満たす経路を評価する。

    if _CODE_FILE_EXT_RE.search(low):
        return False

    # 条件分岐: `not re.search(r"(=|<=|>=|<|>|≈|≃|≡|∝)", candidate)` を満たす経路を評価する。

    if not re.search(r"(=|<=|>=|<|>|≈|≃|≡|∝)", candidate):
        return False

    lhs = re.split(r"(?:<=|>=|=|<|>|≈|≃|≡|∝)", candidate, maxsplit=1)[0].strip()
    lhs = lhs.replace(r"\_", "_")
    lhs_plain = lhs
    # 条件分岐: `lhs_plain.startswith("|") and lhs_plain.endswith("|") and len(lhs_plain) >= 2` を満たす経路を評価する。
    if lhs_plain.startswith("|") and lhs_plain.endswith("|") and len(lhs_plain) >= 2:
        lhs_plain = lhs_plain[1:-1].strip()

    # 条件分岐: `_PHYSICS_SINGLE_LHS_RE.fullmatch(lhs)` を満たす経路を評価する。

    if _PHYSICS_SINGLE_LHS_RE.fullmatch(lhs):
        return True

    # 条件分岐: `_PHYSICS_ASCII_GREEK_TOKEN_RE.fullmatch(lhs)` を満たす経路を評価する。

    if _PHYSICS_ASCII_GREEK_TOKEN_RE.fullmatch(lhs):
        return True

    # 条件分岐: `_PHYSICS_SINGLE_LHS_RE.fullmatch(lhs_plain)` を満たす経路を評価する。

    if _PHYSICS_SINGLE_LHS_RE.fullmatch(lhs_plain):
        return True

    # 条件分岐: `_PHYSICS_ASCII_GREEK_TOKEN_RE.fullmatch(lhs_plain)` を満たす経路を評価する。

    if _PHYSICS_ASCII_GREEK_TOKEN_RE.fullmatch(lhs_plain):
        return True

    if re.fullmatch(
        r"(?:[A-Za-z](?:_[A-Za-z0-9]+)?|\\[A-Za-z]+(?:_[A-Za-z0-9]+)?)"
        r"/"
        r"(?:[A-Za-z](?:_[A-Za-z0-9]+)?|\\[A-Za-z]+(?:_[A-Za-z0-9]+)?)"
        r"(?:\([^()]*\))?",
        lhs_plain,
    ):
        return True

    if re.fullmatch(
        r"(?:[A-Za-z](?:_[A-Za-z0-9]+)?|\\[A-Za-z]+(?:_[A-Za-z0-9]+)?)"
        r"/"
        r"(?:[A-Za-z](?:_[A-Za-z0-9]+)?|\\[A-Za-z]+(?:_[A-Za-z0-9]+)?)"
        r"/"
        r"(?:[A-Za-z](?:_[A-Za-z0-9]+)?|\\[A-Za-z]+(?:_[A-Za-z0-9]+)?)"
        r"(?:\([^()]*\))?",
        lhs_plain,
    ):
        return True

    # 条件分岐: `re.fullmatch(r"[A-Z][A-Za-z0-9]{0,4}(?:_[A-Za-z0-9]+)?(?:\([^()]*\))?", lhs)` を満たす経路を評価する。

    if re.fullmatch(r"[A-Z][A-Za-z0-9]{0,4}(?:_[A-Za-z0-9]+)?(?:\([^()]*\))?", lhs):
        return True

    # 条件分岐: `re.fullmatch(r"[A-Z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)*(?:/[A-Za-z0-9_,]+)?(?:\([^...` を満たす経路を評価する。

    if re.fullmatch(r"[A-Z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)*(?:/[A-Za-z0-9_,]+)?(?:\([^()]*\))?", lhs):
        return True

    # 条件分岐: `_MATH_GREEK_OR_SYMBOL_RE.search(candidate) or re.search(r"\\[A-Za-z]+", candi...` を満たす経路を評価する。

    if _MATH_GREEK_OR_SYMBOL_RE.search(candidate) or re.search(r"\\[A-Za-z]+", candidate):
        return True

    return False


# 関数: `_looks_like_physics_symbol_code` の入出力契約と処理意図を定義する。

def _looks_like_physics_symbol_code(s: str) -> bool:
    candidate = s.strip().replace(r"\_", "_")
    # 条件分岐: `not candidate` を満たす経路を評価する。
    if not candidate:
        return False

    # 条件分岐: `re.search(r"[=<>]", candidate)` を満たす経路を評価する。

    if re.search(r"[=<>]", candidate):
        return False

    # 条件分岐: `_PHYSICS_SINGLE_LHS_RE.fullmatch(candidate)` を満たす経路を評価する。

    if _PHYSICS_SINGLE_LHS_RE.fullmatch(candidate):
        return True

    # 条件分岐: `_PHYSICS_ASCII_GREEK_TOKEN_RE.fullmatch(candidate)` を満たす経路を評価する。

    if _PHYSICS_ASCII_GREEK_TOKEN_RE.fullmatch(candidate):
        return True

    # 条件分岐: `re.fullmatch(r"[A-Z][A-Za-z0-9]{0,4}(?:_[A-Za-z0-9]+)?", candidate)` を満たす経路を評価する。

    if re.fullmatch(r"[A-Z][A-Za-z0-9]{0,4}(?:_[A-Za-z0-9]+)?", candidate):
        return True

    return False


# 関数: `_replace_plain_symbolic_tokens` の入出力契約と処理意図を定義する。

def _replace_plain_symbolic_tokens(text: str, make_token) -> str:
    # 関数: `repl_unicode_greek_sub` の入出力契約と処理意図を定義する。
    def repl_unicode_greek_sub(match: re.Match[str]) -> str:
        sym = match.group("sym")
        sub = match.group("sub")
        sym_tex = _normalize_inline_math_payload(sym)
        sub_tex = _format_subscript_token(sub)
        return make_token(f"${sym_tex}_{{{sub_tex}}}$")

    # 関数: `repl_latin_sub` の入出力契約と処理意図を定義する。

    def repl_latin_sub(match: re.Match[str]) -> str:
        base = match.group("base")
        sub = match.group("sub")
        sub_tex = _format_subscript_token(sub)
        return make_token(rf"${base}_{{{sub_tex}}}$")

    converted = re.sub(
        r"(?<![\\$])\b(?P<sym>[α-ωΑ-ΩΔΘΛΞΠΣΦΨΩℓ])(?:\\_|_)(?P<sub>[A-Za-z][A-Za-z0-9]*)\b",
        repl_unicode_greek_sub,
        text,
    )
    converted = re.sub(
        r"(?<![\\$])\b(?P<base>[A-Za-z])(?:\\_|_)(?P<sub>[A-Za-z][A-Za-z0-9]*)\b",
        repl_latin_sub,
        converted,
    )
    return converted


# 関数: `_normalize_inline_math_payload` の入出力契約と処理意図を定義する。

def _normalize_inline_math_payload(code_text: str) -> str:
    normalized = code_text.strip()
    normalized = re.sub(r"\\\\(?=[A-Za-z])", r"\\", normalized)
    for src, dst in _SUPERSCRIPT_TO_ASCII.items():
        normalized = normalized.replace(src, dst)

    for src, dst in _SUBSCRIPT_TO_ASCII.items():
        normalized = normalized.replace(src, dst)

    for src, dst in _MATH_UNICODE_TO_LATEX.items():
        normalized = normalized.replace(src, dst)

    for src, dst in _GREEK_UNICODE_TO_LATEX.items():
        normalized = normalized.replace(src, dst)

    normalized = re.sub(r"\s+", " ", normalized)
    normalized = re.sub(r"\^\(([^()]+)\)", r"^{(\1)}", normalized)
    normalized = re.sub(r"\^(-?\d+)\b", r"^{\1}", normalized)
    normalized = re.sub(r"([A-Za-z])\\mu\\nu", r"\1_{\\mu\\nu}", normalized)
    normalized = re.sub(r"([A-Za-z])\\mu", r"\1_{\\mu}", normalized)
    normalized = re.sub(
        r"(?<![A-Za-z\\])(?P<name>"
        + _GREEK_NAME_TOKEN
        + r")_(?P<sub>[A-Za-z][A-Za-z0-9_]*)\b",
        lambda m: "\\" + m.group("name") + "_{" + _format_subscript_token(m.group("sub")) + "}",
        normalized,
    )
    normalized = re.sub(
        r"(?<![A-Za-z\\])(?P<name>"
        + _GREEK_NAME_TOKEN
        + r")(?P<idx>[0-9]+)\b",
        lambda m: "\\" + m.group("name") + "_{" + m.group("idx") + "}",
        normalized,
    )
    normalized = re.sub(r"\bP([0-9]+)\b", r"P_{\1}", normalized)
    normalized = re.sub(r"\bJ([0-9]+)\b", r"J_{\1}", normalized)
    normalized = re.sub(r"\bf\\sigma([0-9]+)\b", r"f\\sigma_{\1}", normalized)
    normalized = re.sub(r"\s*=\s*", "=", normalized)
    normalized = normalized.replace("<<", r"\ll")
    normalized = normalized.replace(">>", r"\gg")
    normalized = _normalize_math_command_spacing(normalized)
    normalized = _normalize_word_subscripts(normalized)
    normalized = re.sub(
        r"(?<!\\)\b([A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+){2,})\b",
        lambda m: r"\mathrm{" + m.group(1).replace("_", r"\_") + "}",
        normalized,
    )
    normalized = re.sub(r"(?<!\\)\bln\b", r"\\ln", normalized)
    return normalized.strip()


# 関数: `_normalize_math_command_spacing` の入出力契約と処理意図を定義する。

def _normalize_math_command_spacing(text: str) -> str:
    normalized = text
    normalized = _GREEK_CMD_GLUE_RE.sub(r"\\\1 ", normalized)
    normalized = re.sub(r"\\(Rightarrow|Leftrightarrow|equiv|Box|nabla|propto|times|approx)(?=[A-Za-z])", r"\\\1 ", normalized)
    normalized = re.sub(r"\\cdot(?!s)(?=[A-Za-z])", r"\\cdot ", normalized)
    normalized = normalized.replace(r"\proptocos", r"\propto \cos")
    normalized = normalized.replace(r"\proptosin", r"\propto \sin")
    normalized = normalized.replace(r"\approxx", r"\approx x")
    normalized = re.sub(r"(?<=[A-Za-z0-9\)])\\to(?=[A-Za-z])", r"\\to ", normalized)
    normalized = re.sub(r"\\partial_([A-Za-z])(?=[A-Za-z])", r"\\partial_\1 ", normalized)
    normalized = re.sub(r"\\partial_((?>\\[A-Za-z]+))(?=[A-Za-z])", r"\\partial_\1 ", normalized)
    return normalized


# 関数: `_postprocess_latex_body` の入出力契約と処理意図を定義する。

def _postprocess_latex_body(body: str) -> str:
    normalized = _normalize_math_command_spacing(body)
    normalized = re.sub(
        r"\\href\{(?!(?:https?|mailto):)([^{}]+?\.(?:html?|HTML?)(?:[?#][^{}]*)?)\}\{([^{}]*)\}",
        lambda m: m.group(2),
        normalized,
        flags=re.IGNORECASE,
    )

    # 関数: `repl_texttt_math` の入出力契約と処理意図を定義する。
    def repl_texttt_math(match: re.Match[str]) -> str:
        raw_payload = match.group(1)
        payload = (
            raw_payload.replace(r"\_", "_")
            .replace(r"\textasciicircum{}", "^")
            .replace(r"\textbackslash{}", "\\")
            .replace(r"\%", "%")
            .replace(r"\&", "&")
            .replace(r"\$", "$")
            .replace(r"\#", "#")
            .replace(r"\{", "{")
            .replace(r"\}", "}")
        )
        payload = re.sub(r"\s+", " ", payload).strip()
        # 条件分岐: `_looks_like_artifact_code(payload)` を満たす経路を評価する。
        if _looks_like_artifact_code(payload):
            return match.group(0)

        # 条件分岐: `_looks_like_physics_equation_code(payload) or _looks_like_physics_symbol_code...` を満たす経路を評価する。

        if _looks_like_physics_equation_code(payload) or _looks_like_physics_symbol_code(payload):
            return "$" + _normalize_inline_math_payload(payload) + "$"

        return match.group(0)

    normalized = re.sub(r"\\texttt\{([^{}]+)\}", repl_texttt_math, normalized)
    normalized = re.sub(
        r"(?<![$\\])(?P<sym>[α-ωΑ-ΩΔΘΛΞΠΣΦΨΩℓ])\\_(?P<sub>[A-Za-z][A-Za-z0-9]*)",
        lambda m: "$"
        + _normalize_inline_math_payload(m.group("sym"))
        + "_{"
        + _format_subscript_token(m.group("sub"))
        + "}$",
        normalized,
    )
    normalized = _normalize_word_subscripts(normalized)

    normalized = re.sub(
        r"(?<!\\)\$(event|selected|target|source)=([A-Za-z0-9_\\-]+)\$",
        lambda m: r"\texttt{" + m.group(1) + "=" + m.group(2).replace("_", r"\_") + "}",
        normalized,
    )
    normalized = re.sub(
        r"(?<!\\)step(?:_[A-Za-z0-9]+){2,}",
        lambda m: m.group(0).replace("_", r"\_"),
        normalized,
    )
    normalized = re.sub(
        r"\$next=([A-Za-z0-9\\_]+)\$",
        lambda m: r"\texttt{next=" + m.group(1) + "}",
        normalized,
    )
    normalized = re.sub(
        r"\$([A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+){2,})\$",
        lambda m: r"\texttt{" + m.group(1).replace("_", r"\_") + "}",
        normalized,
    )
    # 関数: `_texttt_allowbreak` の入出力契約と処理意図を定義する。
    def _texttt_allowbreak(match: re.Match[str]) -> str:
        payload = match.group(1)
        # 条件分岐: `len(payload) < 28 or r"\_" not in payload` を満たす経路を評価する。
        if len(payload) < 28 or r"\_" not in payload:
            return match.group(0)

        return r"\texttt{" + payload.replace(r"\_", r"\_\allowbreak ") + "}"

    normalized = re.sub(r"\\texttt\{([^{}]+)\}", _texttt_allowbreak, normalized)
    return normalized


# 関数: `_convert_inline` の入出力契約と処理意図を定義する。

def _convert_inline(text: str) -> str:
    token_map: dict[str, str] = {}
    token_index = 0

    # 関数: `make_token` の入出力契約と処理意図を定義する。
    def make_token(rendered: str) -> str:
        nonlocal token_index
        key = f"@@TOK{token_index}@@"
        token_map[key] = rendered
        token_index += 1
        return key

    # inline code

    def repl_inline_code(match: re.Match[str]) -> str:
        payload_raw = match.group(1)
        payload = payload_raw.strip()
        # 条件分岐: `not payload` を満たす経路を評価する。
        if not payload:
            return ""

        # 条件分岐: `_PUNCT_ONLY_RE.fullmatch(payload)` を満たす経路を評価する。

        if _PUNCT_ONLY_RE.fullmatch(payload):
            return make_token(_escape_tex(payload))

        # 条件分岐: `re.search(r"[\u3040-\u30ff\u3400-\u9fff]", payload) and not _looks_like_artif...` を満たす経路を評価する。

        if re.search(r"[\u3040-\u30ff\u3400-\u9fff]", payload) and not _looks_like_artifact_code(payload):
            return make_token(_escape_tex(payload))

        # 条件分岐: `_looks_like_physics_equation_code(payload) or _looks_like_physics_symbol_code...` を満たす経路を評価する。

        if _looks_like_physics_equation_code(payload) or _looks_like_physics_symbol_code(payload):
            return make_token("$" + _normalize_inline_math_payload(payload) + "$")

        # 条件分岐: `_looks_like_math_code(payload)` を満たす経路を評価する。

        if _looks_like_math_code(payload):
            return make_token("$" + _normalize_inline_math_payload(payload) + "$")

        return make_token(r"\texttt{" + _escape_tex(payload) + r"}")

    text = re.sub(r"`([^`]+)`", repl_inline_code, text)
    # inline math
    def repl_inline_math(match: re.Match[str]) -> str:
        payload_raw = match.group(1)
        payload = payload_raw.strip()
        # 条件分岐: `not payload` を満たす経路を評価する。
        if not payload:
            return ""

        if _looks_like_artifact_code(payload) and not (
            _looks_like_physics_equation_code(payload) or _looks_like_physics_symbol_code(payload)
        ):
            return make_token(r"\texttt{" + _escape_tex(payload) + r"}")

        # 条件分岐: `re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*(?:\.[A-Za-z0-9_]+)+", payload)` を満たす経路を評価する。

        if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*(?:\.[A-Za-z0-9_]+)+", payload):
            return make_token(r"\texttt{" + _escape_tex(payload) + r"}")

        return make_token("$" + _normalize_inline_math_payload(payload) + "$")

    text = re.sub(
        r"(?<!\\)\$(.+?)(?<!\\)\$",
        repl_inline_math,
        text,
    )
    # links
    def repl_link(match: re.Match[str]) -> str:
        label = match.group(1)
        target = match.group(2).strip()
        target_norm = target.lower()

        no_scheme = re.match(r"^[a-z][a-z0-9+.\-]*:", target_norm) is None
        target_core = re.split(r"[?#]", target_norm, maxsplit=1)[0]
        # 条件分岐: `no_scheme and (target_core.endswith(".html") or target_core.endswith(".htm"))` を満たす経路を評価する。
        if no_scheme and (target_core.endswith(".html") or target_core.endswith(".htm")):
            return make_token(_escape_tex(label))

        return make_token(r"\href{" + _escape_tex(target) + "}{" + _escape_tex(label) + "}")

    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", repl_link, text)

    # 関数: `repl_citation` の入出力契約と処理意図を定義する。
    def repl_citation(match: re.Match[str]) -> str:
        keys = [k.strip() for k in re.split(r"\s*[,;]\s*", match.group("keys")) if k.strip()]
        # 条件分岐: `not keys` を満たす経路を評価する。
        if not keys:
            return match.group(0)

        # 条件分岐: `not all(key in _REFERENCE_KEYS for key in keys)` を満たす経路を評価する。

        if not all(key in _REFERENCE_KEYS for key in keys):
            return match.group(0)

        for key in keys:
            _USED_REFERENCE_KEYS.add(key)

        return make_token(r"\cite{" + ",".join(keys) + "}")

    text = _CITATION_BLOCK_RE.sub(repl_citation, text)
    text = _replace_plain_symbolic_tokens(text, make_token)
    # bold / italic
    text = re.sub(r"(?<!\\)\*\*([^*]+)\*\*", lambda m: make_token(r"\textbf{" + _escape_tex(m.group(1)) + "}"), text)
    text = re.sub(r"(?<!\\)\*([^*]+)\*", lambda m: make_token(r"\emph{" + _escape_tex(m.group(1)) + "}"), text)
    text = text.replace(r"\*", "*")

    escaped = _escape_tex(text)
    # Resolve nested placeholder expansions (e.g., bold that contains inline-code tokens).
    for _ in range(len(token_map) + 1):
        changed = False
        for key, rendered in token_map.items():
            escaped_key = _escape_tex(key)
            # 条件分岐: `escaped_key in escaped` を満たす経路を評価する。
            if escaped_key in escaped:
                escaped = escaped.replace(escaped_key, rendered)
                changed = True

        # 条件分岐: `not changed` を満たす経路を評価する。

        if not changed:
            break

    return escaped


# 関数: `_is_table_separator` の入出力契約と処理意図を定義する。

def _is_table_separator(line: str) -> bool:
    s = line.strip()
    # 条件分岐: `"|" not in s` を満たす経路を評価する。
    if "|" not in s:
        return False

    core = s.replace("|", "").replace(":", "").replace(" ", "")
    return bool(core) and set(core) <= {"-"}


# 関数: `_parse_table_row` の入出力契約と処理意図を定義する。

def _parse_table_row(line: str) -> list[str]:
    s = line.strip()
    # 条件分岐: `s.startswith("|")` を満たす経路を評価する。
    if s.startswith("|"):
        s = s[1:]

    # 条件分岐: `s.endswith("|")` を満たす経路を評価する。

    if s.endswith("|"):
        s = s[:-1]

    cells: list[str] = []
    buf: list[str] = []
    in_code = False
    in_math = False
    escaped = False

    for ch in s:
        # 条件分岐: `escaped` を満たす経路を評価する。
        if escaped:
            buf.append(ch)
            escaped = False
            continue

        # 条件分岐: `ch == "\\"` を満たす経路を評価する。

        if ch == "\\":
            buf.append(ch)
            escaped = True
            continue

        # 条件分岐: `ch == "`" and not in_math` を満たす経路を評価する。

        if ch == "`" and not in_math:
            in_code = not in_code
            buf.append(ch)
            continue

        # 条件分岐: `ch == "$" and not in_code` を満たす経路を評価する。

        if ch == "$" and not in_code:
            in_math = not in_math
            buf.append(ch)
            continue

        # 条件分岐: `ch == "|" and not in_code and not in_math` を満たす経路を評価する。

        if ch == "|" and not in_code and not in_math:
            cells.append("".join(buf).strip())
            buf = []
            continue

        buf.append(ch)

    cells.append("".join(buf).strip())
    return cells


# 関数: `_render_table` の入出力契約と処理意図を定義する。

def _render_table(block_lines: list[str]) -> list[str]:
    # 条件分岐: `len(block_lines) < 2` を満たす経路を評価する。
    if len(block_lines) < 2:
        return [_convert_inline(block_lines[0])] if block_lines else []

    header = _parse_table_row(block_lines[0])
    body_lines = block_lines[2:] if _is_table_separator(block_lines[1]) else block_lines[1:]
    rows = [_parse_table_row(line) for line in body_lines]
    ncols = max(1, len(header))
    width = max(0.08, min(0.42, round((0.97 / ncols) - 0.03, 3)))
    colspec = "".join(r">{\raggedright\arraybackslash}p{" + f"{width}\\linewidth" + "}" for _ in range(ncols))

    compact_table = ncols >= 4
    table_font = r"\normalsize"
    # 条件分岐: `ncols >= 7` を満たす経路を評価する。
    if ncols >= 7:
        table_font = r"\tiny"
    # 条件分岐: 前段条件が不成立で、`ncols >= 5` を追加評価する。
    elif ncols >= 5:
        table_font = r"\scriptsize"
    # 条件分岐: 前段条件が不成立で、`ncols >= 4` を追加評価する。
    elif ncols >= 4:
        table_font = r"\footnotesize"

    out: list[str] = []
    # 条件分岐: `compact_table` を満たす経路を評価する。
    if compact_table:
        out += [
            r"\begingroup",
            table_font,
            r"\setlength{\tabcolsep}{2pt}",
            r"\renewcommand{\arraystretch}{1.05}",
        ]

    out += [r"\begin{longtable}{" + colspec + "}", r"\toprule"]
    out.append(" & ".join(_convert_inline(c) for c in header) + r" \\")
    out.append(r"\midrule")
    for row in rows:
        padded = row + [""] * (ncols - len(row))
        out.append(" & ".join(_convert_inline(c) for c in padded[:ncols]) + r" \\")

    out += [r"\bottomrule", r"\end{longtable}"]
    # 条件分岐: `compact_table` を満たす経路を評価する。
    if compact_table:
        out.append(r"\endgroup")

    out.append("")
    return out


# 関数: `_markdown_to_latex` の入出力契約と処理意図を定義する。

def _markdown_to_latex(
    md_text: str,
    *,
    root: Path,
    outdir: Path,
    figures_dir: Path,
    profile: str = "",
) -> str:
    lines = md_text.splitlines()
    out: list[str] = []
    used_labels: dict[str, int] = {}
    staged_assets: dict[str, str] = {}
    used_figure_names: set[str] = set()
    used_figure_labels: dict[str, int] = {}

    paragraph: list[str] = []
    in_code = False
    code_listing_open = False
    in_math = False
    list_mode: Optional[str] = None  # "itemize" | "enumerate"
    chapter_started = False
    top_h1_seen = False
    appendix_started = False
    i = 0

    # 関数: `flush_paragraph` の入出力契約と処理意図を定義する。
    def flush_paragraph() -> None:
        nonlocal paragraph
        # 条件分岐: `paragraph` を満たす経路を評価する。
        if paragraph:
            out.append(_convert_inline(" ".join(s.strip() for s in paragraph if s.strip())))
            out.append("")
            paragraph = []

    # 関数: `close_list` の入出力契約と処理意図を定義する。

    def close_list() -> None:
        nonlocal list_mode
        # 条件分岐: `list_mode` を満たす経路を評価する。
        if list_mode:
            out.append(r"\end{" + list_mode + "}")
            out.append("")
            list_mode = None

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # 条件分岐: `in_code` を満たす経路を評価する。
        if in_code:
            # 条件分岐: `stripped.startswith("```")` を満たす経路を評価する。
            if stripped.startswith("```"):
                # 条件分岐: `code_listing_open` を満たす経路を評価する。
                if code_listing_open:
                    out.append(r"\end{lstlisting}")
                else:
                    out.append(r"\end{verbatim}")

                out.append("")
                in_code = False
                code_listing_open = False
            else:
                out.append(line.rstrip("\n"))

            i += 1
            continue

        # 条件分岐: `in_math` を満たす経路を評価する。

        if in_math:
            # 条件分岐: `stripped == "$$"` を満たす経路を評価する。
            if stripped == "$$":
                out.append(r"\]")
                out.append("")
                in_math = False
            # 条件分岐: 前段条件が不成立で、`stripped.endswith("$$")` を追加評価する。
            elif stripped.endswith("$$"):
                body_end = line.rsplit("$$", 1)[0].strip()
                # 条件分岐: `body_end` を満たす経路を評価する。
                if body_end:
                    out.append(body_end)

                out.append(r"\]")
                out.append("")
                in_math = False
            else:
                out.append(line)

            i += 1
            continue

        # block starts

        if stripped.startswith("```"):
            flush_paragraph()
            close_list()
            out.append(r"\begin{lstlisting}[breaklines=true]")
            in_code = True
            code_listing_open = True
            i += 1
            continue

        # 条件分岐: `stripped == "$$"` を満たす経路を評価する。

        if stripped == "$$":
            flush_paragraph()
            close_list()
            out.append(r"\[")
            in_math = True
            i += 1
            continue

        # 条件分岐: `stripped.startswith("$$") and stripped.endswith("$$") and len(stripped) > 4` を満たす経路を評価する。

        if stripped.startswith("$$") and stripped.endswith("$$") and len(stripped) > 4:
            flush_paragraph()
            close_list()
            math_inline = stripped[2:-2].strip()
            out.append(r"\[")
            # 条件分岐: `math_inline` を満たす経路を評価する。
            if math_inline:
                out.append(math_inline)

            out.append(r"\]")
            out.append("")
            i += 1
            continue

        # 条件分岐: `stripped.startswith("$$") and len(stripped) > 2` を満たす経路を評価する。

        if stripped.startswith("$$") and len(stripped) > 2:
            flush_paragraph()
            close_list()
            out.append(r"\[")
            body_start = line.split("$$", 1)[1].strip()
            # 条件分岐: `body_start` を満たす経路を評価する。
            if body_start:
                out.append(body_start)

            in_math = True
            i += 1
            continue

        m_leading_image = _match_leading_image_line(stripped)
        # 条件分岐: `m_leading_image` を満たす経路を評価する。
        if m_leading_image:
            path_text, inline_desc = m_leading_image
            caption_text = ""
            consumed_after_caption = 0
            # 条件分岐: `paragraph` を満たす経路を評価する。
            if paragraph:
                last_line = re.sub(r"\s{2,}$", "", paragraph[-1]).strip()
                # 条件分岐: `re.match(r"^(図|Figure|Fig\.?)", last_line, flags=re.IGNORECASE)` を満たす経路を評価する。
                if re.match(r"^(図|Figure|Fig\.?)", last_line, flags=re.IGNORECASE):
                    paragraph = paragraph[:-1]
                    flush_paragraph()
                    caption_text = last_line
                else:
                    flush_paragraph()

            # 条件分岐: `not caption_text and inline_desc` を満たす経路を評価する。

            if not caption_text and inline_desc:
                caption_text = inline_desc

            # 条件分岐: `not caption_text` を満たす経路を評価する。

            if not caption_text:
                next_caption, consumed = _extract_following_caption(lines, i + 1)
                # 条件分岐: `next_caption` を満たす経路を評価する。
                if next_caption:
                    caption_text = next_caption
                    consumed_after_caption = consumed

            # 条件分岐: `not caption_text` を満たす経路を評価する。

            if not caption_text:
                caption_text = _fallback_caption_from_path(path_text)

            close_list()
            out.extend(
                _render_figure_block(
                    raw_path=path_text,
                    caption=caption_text,
                    root=root,
                    outdir=outdir,
                    figures_dir=figures_dir,
                    staged_assets=staged_assets,
                    used_figure_names=used_figure_names,
                    used_figure_labels=used_figure_labels,
                )
            )
            i += 1 + consumed_after_caption
            continue

        m_caption_code_path = re.match(r"^(.*?)[\s　]*`([^`]+)`\s*$", stripped)
        if (
            m_caption_code_path
            and m_caption_code_path.group(1).strip()
            and _is_image_path(m_caption_code_path.group(2))
            and re.match(r"^(図|Figure|Fig\.?)", m_caption_code_path.group(1).strip(), flags=re.IGNORECASE)
        ):
            flush_paragraph()
            close_list()
            out.extend(
                _render_figure_block(
                    raw_path=m_caption_code_path.group(2).strip(),
                    caption=m_caption_code_path.group(1).strip(),
                    root=root,
                    outdir=outdir,
                    figures_dir=figures_dir,
                    staged_assets=staged_assets,
                    used_figure_names=used_figure_names,
                    used_figure_labels=used_figure_labels,
                )
            )
            i += 1
            continue

        # 条件分岐: `stripped == ""` を満たす経路を評価する。

        if stripped == "":
            flush_paragraph()
            close_list()
            i += 1
            continue

        # table block

        if "|" in line and (i + 1) < len(lines) and _is_table_separator(lines[i + 1]):
            flush_paragraph()
            close_list()
            block = [line]
            i += 1
            while i < len(lines):
                # 条件分岐: `lines[i].strip() == ""` を満たす経路を評価する。
                if lines[i].strip() == "":
                    break

                # 条件分岐: `"|" not in lines[i]` を満たす経路を評価する。

                if "|" not in lines[i]:
                    break

                block.append(lines[i])
                i += 1

            out.extend(_render_table(block))
            continue

        # headings

        m_head = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        # 条件分岐: `m_head` を満たす経路を評価する。
        if m_head:
            flush_paragraph()
            close_list()
            level = len(m_head.group(1))
            raw_title = m_head.group(2).strip()
            title = _strip_heading_prefix(raw_title)
            heading_number = _extract_heading_number(raw_title)
            # 条件分岐: `level == 1 and not top_h1_seen` を満たす経路を評価する。
            if level == 1 and not top_h1_seen:
                top_h1_seen = True
                i += 1
                continue

            effective_level = 1 if level == 1 else max(1, level - 1)
            # 条件分岐: `effective_level == 1 and title.startswith("付録")` を満たす経路を評価する。
            if effective_level == 1 and title.startswith("付録"):
                # 条件分岐: `not appendix_started` を満たす経路を評価する。
                if not appendix_started:
                    out.append(r"\appendix")
                    out.append("")
                    appendix_started = True

                appendix_title = re.sub(r"^付録\s*[A-Za-zＡ-Ｚ0-9一二三四五六七八九十]*\s*[\.．:：]?\s*", "", title).strip()
                # 条件分岐: `appendix_title` を満たす経路を評価する。
                if appendix_title:
                    title = appendix_title

            # 条件分岐: `effective_level == 1` を満たす経路を評価する。

            if effective_level == 1:
                # 条件分岐: `_is_abstract_heading(title)` を満たす経路を評価する。
                if _is_abstract_heading(title):
                    out.append(r"\section*{" + _convert_inline(title) + "}")
                    out.append("")
                    chapter_started = True
                    i += 1
                    continue

                # 条件分岐: `chapter_started` を満たす経路を評価する。

                if chapter_started:
                    out.append(r"\clearpage")
                    out.append("")

                chapter_started = True

            # 条件分岐: `effective_level == 1` を満たす経路を評価する。

            if effective_level == 1:
                cmd = "section"
            # 条件分岐: 前段条件が不成立で、`effective_level == 2` を追加評価する。
            elif effective_level == 2:
                cmd = "subsection"
            # 条件分岐: 前段条件が不成立で、`effective_level == 3` を追加評価する。
            elif effective_level == 3:
                cmd = "subsubsection"
            # 条件分岐: 前段条件が不成立で、`effective_level == 4` を追加評価する。
            elif effective_level == 4:
                cmd = "paragraph"
            # 条件分岐: 前段条件が不成立で、`effective_level == 5` を追加評価する。
            elif effective_level == 5:
                cmd = "subparagraph"
            else:
                out.append(r"\textbf{" + _convert_inline(title) + "}")
                out.append("")
                i += 1
                continue

            force_subsection_pagebreak = False
            force_heading_pagebreak = False
            # 条件分岐: `cmd == "subsection"` を満たす経路を評価する。
            if cmd == "subsection":
                if (
                    profile == "part2_astrophysics"
                    and heading_number.startswith("4.")
                    and heading_number != "4.1"
                ):
                    force_subsection_pagebreak = True
                elif (
                    profile == "part3_quantum"
                    and heading_number.startswith("4.")
                    and heading_number != "4.1"
                ):
                    force_subsection_pagebreak = True
                elif (
                    profile == "part4_verification"
                    and heading_number.startswith("2.")
                    and heading_number != "2.0"
                ):
                    force_subsection_pagebreak = True

            # 条件分岐: `profile == "part2_astrophysics" and "項目対応（節マップ）" in title` を満たす経路を評価する。

            if profile == "part2_astrophysics" and "項目対応（節マップ）" in title:
                force_heading_pagebreak = True

            # 条件分岐: `profile == "part3_quantum" and "項目対応（節マップ）" in title` を満たす経路を評価する。

            if profile == "part3_quantum" and "項目対応（節マップ）" in title:
                force_heading_pagebreak = True

            # 条件分岐: `profile == "part3_quantum"` を満たす経路を評価する。

            if profile == "part3_quantum":
                # 条件分岐: `heading_number in {"4.10.2", "4.10.3", "4.11.2", "4.11.3"}` を満たす経路を評価する。
                if heading_number in {"4.10.2", "4.10.3", "4.11.2", "4.11.3"}:
                    force_heading_pagebreak = True
                # 条件分岐: 前段条件が不成立で、`("黒体放射の基準量" in title) or ("黒体：エントロピーと第2法則整合" in title)` を追加評価する。
                elif ("黒体放射の基準量" in title) or ("黒体：エントロピーと第2法則整合" in title):
                    force_heading_pagebreak = True

            # 条件分岐: `force_subsection_pagebreak or force_heading_pagebreak` を満たす経路を評価する。

            if force_subsection_pagebreak or force_heading_pagebreak:
                # If a markdown horizontal rule was emitted just before this heading,
                # remove it so the heading itself can start at the very top of the page.
                while out and out[-1] == "":
                    out.pop()

                # 条件分岐: `len(out) >= 3 and out[-3:] == [r"\medskip", r"\hrule", r"\medskip"]` を満たす経路を評価する。

                if len(out) >= 3 and out[-3:] == [r"\medskip", r"\hrule", r"\medskip"]:
                    del out[-3:]

                while out and out[-1] == "":
                    out.pop()

                out.append(r"\clearpage")
                out.append("")

            heading_tex = _convert_inline(title)
            heading_pdf = _escape_tex(_heading_pdf_text(title))
            unnumbered_subsection = (
                profile == "paper"
                and cmd == "subsection"
                and heading_number == "2.0"
                and "統一解釈" in title
            )
            # 条件分岐: `unnumbered_subsection` を満たす経路を評価する。
            if unnumbered_subsection:
                out.append(rf"\{cmd}*{{\texorpdfstring{{{heading_tex}}}{{{heading_pdf}}}}}")
            else:
                out.append(rf"\{cmd}{{\texorpdfstring{{{heading_tex}}}{{{heading_pdf}}}}}")

            section_label = _build_section_label(raw_title, title, used_labels=used_labels)
            out.append(rf"\label{{sec:{section_label}}}")
            out.append("")
            i += 1
            continue

        # horizontal rule

        if re.match(r"^[-*_]{3,}\s*$", stripped):
            flush_paragraph()
            close_list()
            out += [r"\medskip", r"\hrule", r"\medskip", ""]
            i += 1
            continue

        # image-only line

        m_img = re.match(r"^!\[([^\]]*)\]\(([^)]+)\)\s*$", stripped)
        # 条件分岐: `m_img` を満たす経路を評価する。
        if m_img:
            flush_paragraph()
            close_list()
            alt = m_img.group(1).strip()
            path = m_img.group(2).strip()
            caption_text = alt
            consumed_after_caption = 0
            # 条件分岐: `not caption_text` を満たす経路を評価する。
            if not caption_text:
                next_caption, consumed = _extract_following_caption(lines, i + 1)
                # 条件分岐: `next_caption` を満たす経路を評価する。
                if next_caption:
                    caption_text = next_caption
                    consumed_after_caption = consumed

            # 条件分岐: `not caption_text` を満たす経路を評価する。

            if not caption_text:
                caption_text = _fallback_caption_from_path(path)

            out.extend(
                _render_figure_block(
                    raw_path=path,
                    caption=caption_text,
                    root=root,
                    outdir=outdir,
                    figures_dir=figures_dir,
                    staged_assets=staged_assets,
                    used_figure_names=used_figure_names,
                    used_figure_labels=used_figure_labels,
                )
            )
            i += 1 + consumed_after_caption
            continue

        # blockquote

        if stripped.startswith(">"):
            flush_paragraph()
            close_list()
            q_lines: list[str] = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                q_lines.append(lines[i].strip()[1:].strip())
                i += 1

            out.append(r"\begin{quote}")
            for q in q_lines:
                # 条件分岐: `q` を満たす経路を評価する。
                if q:
                    out.append(_convert_inline(q) + r"\\")

            out.append(r"\end{quote}")
            out.append("")
            continue

        # lists

        m_ul = re.match(r"^\s*[-*]\s+(.+)$", line)
        # 条件分岐: `m_ul` を満たす経路を評価する。
        if m_ul:
            flush_paragraph()
            # 条件分岐: `list_mode != "itemize"` を満たす経路を評価する。
            if list_mode != "itemize":
                close_list()
                out.append(r"\begin{itemize}[leftmargin=2em]")
                list_mode = "itemize"

            out.append(r"\item " + _convert_inline(m_ul.group(1).strip()))
            i += 1
            continue

        m_ol = re.match(r"^\s*\d+[.)]\s+(.+)$", line)
        # 条件分岐: `m_ol` を満たす経路を評価する。
        if m_ol:
            flush_paragraph()
            # 条件分岐: `list_mode != "enumerate"` を満たす経路を評価する。
            if list_mode != "enumerate":
                close_list()
                out.append(r"\begin{enumerate}[leftmargin=2em]")
                list_mode = "enumerate"

            out.append(r"\item " + _convert_inline(m_ol.group(1).strip()))
            i += 1
            continue

        # default paragraph line

        paragraph.append(line)
        i += 1

    flush_paragraph()
    close_list()

    # 条件分岐: `in_code` を満たす経路を評価する。
    if in_code:
        # 条件分岐: `code_listing_open` を満たす経路を評価する。
        if code_listing_open:
            out.append(r"\end{lstlisting}")
        else:
            out.append(r"\end{verbatim}")

    # 条件分岐: `in_math` を満たす経路を評価する。

    if in_math:
        out.append(r"\]")

    return "\n".join(out).strip() + "\n"


# 関数: `main` の入出力契約と処理意図を定義する。

def main(argv: Optional[Sequence[str]] = None) -> int:
    global _REFERENCE_KEYS, _REFERENCE_ORDER, _REFERENCE_TEXT, _USED_REFERENCE_KEYS

    ap = argparse.ArgumentParser(description="Generate LaTeX paper from markdown manuscript.")
    ap.add_argument(
        "--profile",
        choices=["paper", "part2_astrophysics", "part3_quantum", "part4_verification"],
        default="paper",
        help="paper profile",
    )
    ap.add_argument("--manuscript", default=None, help="input markdown path (default by profile)")
    ap.add_argument("--outdir", default=None, help="output directory (default: output/private/summary)")
    ap.add_argument("--out-name", default=None, help="output .tex name (default by profile)")
    args = ap.parse_args(argv)

    root = _repo_root()
    profile = str(args.profile)
    references_md = root / "doc" / "paper" / "30_references.md"
    _REFERENCE_ORDER, _REFERENCE_TEXT = _load_reference_entries(references_md)
    _REFERENCE_KEYS = set(_REFERENCE_TEXT.keys())
    _USED_REFERENCE_KEYS = set()

    # 条件分岐: `args.manuscript` を満たす経路を評価する。
    if args.manuscript:
        manuscript_md = Path(args.manuscript)
    else:
        # 条件分岐: `profile == "paper"` を満たす経路を評価する。
        if profile == "paper":
            manuscript_md = root / "doc" / "paper" / "10_part1_core_theory.md"
        # 条件分岐: 前段条件が不成立で、`profile == "part2_astrophysics"` を追加評価する。
        elif profile == "part2_astrophysics":
            manuscript_md = root / "doc" / "paper" / "11_part2_astrophysics.md"
        # 条件分岐: 前段条件が不成立で、`profile == "part3_quantum"` を追加評価する。
        elif profile == "part3_quantum":
            manuscript_md = root / "doc" / "paper" / "12_part3_quantum.md"
        else:
            manuscript_md = root / "doc" / "paper" / "13_part4_verification.md"

    # 条件分岐: `not manuscript_md.exists()` を満たす経路を評価する。

    if not manuscript_md.exists():
        print(f"[error] manuscript not found: {manuscript_md}")
        return 1

    # 条件分岐: `args.outdir` を満たす経路を評価する。

    if args.outdir:
        outdir = Path(args.outdir)
    else:
        outdir = root / "output" / "private" / "summary"

    outdir.mkdir(parents=True, exist_ok=True)
    figures_dir = outdir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 条件分岐: `args.out_name` を満たす経路を評価する。
    if args.out_name:
        out_name = str(args.out_name)
    else:
        # 条件分岐: `profile == "paper"` を満たす経路を評価する。
        if profile == "paper":
            out_name = "pmodel_paper.tex"
        # 条件分岐: 前段条件が不成立で、`profile == "part2_astrophysics"` を追加評価する。
        elif profile == "part2_astrophysics":
            out_name = "pmodel_paper_part2_astrophysics.tex"
        # 条件分岐: 前段条件が不成立で、`profile == "part3_quantum"` を追加評価する。
        elif profile == "part3_quantum":
            out_name = "pmodel_paper_part3_quantum.tex"
        else:
            out_name = "pmodel_paper_part4_verification.tex"

    md_text = manuscript_md.read_text(encoding="utf-8", errors="replace")
    body = _markdown_to_latex(
        md_text,
        root=root,
        outdir=outdir,
        figures_dir=figures_dir,
        profile=profile,
    )
    body = _postprocess_latex_body(body)
    bibliography_section = _render_bibliography_section()

    title_map = {
        "paper": "P-model Paper Part I",
        "part2_astrophysics": "P-model Paper Part II (Astrophysics)",
        "part3_quantum": "P-model Paper Part III (Quantum)",
        "part4_verification": "P-model Paper Part IV (Verification Materials)",
    }
    title = title_map.get(profile, "P-model Paper")
    author_tex = (
        r"\author{" "\n"
        r"  Shunji Ogawa \\" "\n"
        r"  \vspace{0.2em}" "\n"
        r"  \small ENTERSYSTEM Co., Ltd., Osaka, Japan" "\n"
        r"}"
    )

    tex = (
        r"% !TeX program = lualatex" "\n"
        r"\documentclass[11pt,a4paper]{article}" "\n"
        r"\usepackage{iftex}" "\n"
        r"\ifPDFTeX" "\n"
        r"  \usepackage[utf8]{inputenc}" "\n"
        r"  \usepackage[T1]{fontenc}" "\n"
        r"  \usepackage{lmodern}" "\n"
        r"  \usepackage{CJKutf8}" "\n"
        r"\else" "\n"
        r"  \usepackage{fontspec}" "\n"
        r"\fi" "\n"
        r"\ifXeTeX" "\n"
        r"  \usepackage{xeCJK}" "\n"
        r"\fi" "\n"
        r"\ifLuaTeX" "\n"
        r"  \usepackage{luatexja}" "\n"
        r"  % avoid lltjp-microtype warning under LuaTeX-ja" "\n"
        r"\else" "\n"
        r"  \usepackage{microtype}" "\n"
        r"\fi" "\n"
        r"\usepackage{geometry}" "\n"
        r"\geometry{margin=20mm}" "\n"
        r"\usepackage{hyperref}" "\n"
        r"\usepackage{graphicx}" "\n"
        r"\usepackage{longtable}" "\n"
        r"\usepackage{booktabs}" "\n"
        r"\usepackage{array}" "\n"
        r"\usepackage{enumitem}" "\n"
        r"\usepackage{amsmath,amssymb}" "\n"
        r"\usepackage{float}" "\n"
        r"\usepackage{xcolor}" "\n"
        r"\usepackage{listings}" "\n"
        r"\usepackage{setspace}" "\n"
        r"\lstset{breaklines=true,breakatwhitespace=false,columns=fullflexible,keepspaces=true,basicstyle=\ttfamily\small}" "\n"
        r"\setstretch{1.1}" "\n"
        r"\setlength{\parskip}{0.4em}" "\n"
        r"\setlength{\parindent}{0pt}" "\n"
        r"\setcounter{secnumdepth}{3}" "\n"
        r"\urlstyle{same}" "\n\n"
        r"\setlength{\tabcolsep}{3pt}" "\n"
        r"\setlength{\LTleft}{0pt}" "\n"
        r"\setlength{\LTright}{0pt}" "\n"
        r"\setlength{\emergencystretch}{4em}" "\n"
        r"\sloppy" "\n\n"
        r"\graphicspath{{figures/}}" "\n\n"
        r"% --- convenience macros (avoid undefined control sequences) ---" "\n"
        r"\newcommand{\sigmaV}{\sigma_V}" "\n"
        r"\newcommand{\Deltax}{\Delta x}" "\n"
        r"\newcommand{\DeltaA}{\Delta A}" "\n"
        r"\newcommand{\Deltat}{\Delta t}" "\n"
        r"\newcommand{\Deltaz}{\Delta z}" "\n"
        r"\newcommand{\DeltaAIC}{\Delta \mathrm{AIC}}" "\n\n"
        + r"\title{" + _escape_tex(title) + "}\n"
        + author_tex + "\n"
        + r"\date{" + _escape_tex(datetime.now(timezone.utc).strftime("%Y-%m-%d UTC")) + "}\n\n"
        + r"\begin{document}" + "\n"
        + r"\ifPDFTeX\begin{CJK}{UTF8}{min}\fi" + "\n"
        + r"\maketitle" + "\n\n"
        + body
        + bibliography_section
        + "\n"
        + r"\ifPDFTeX\end{CJK}\fi" + "\n"
        + r"\end{document}" + "\n"
    )

    out_tex = outdir / out_name
    out_tex.write_text(tex, encoding="utf-8")
    print(f"[ok] wrote: {out_tex}")

    try:
        worklog.append_event(
            {
                "event_type": "paper_latex",
                "profile": profile,
                "manuscript": manuscript_md,
                "output_tex": out_tex,
            }
        )
    except Exception:
        pass

    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())
