#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
paper_latex.py

Markdown 論文（doc/paper/*.md）から配布用の .tex を生成する。
（最小依存：pandoc なしで動作）
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Optional, Sequence

_ROOT = Path(__file__).resolve().parents[2]

_PMODEL_VERSION_STYLE_DEFAULT = r"""\NeedsTeXFormat{LaTeX2e}
\ProvidesPackage{pmodel_version}[2026/03/09 P-model shared version info]

% === ここだけ更新すれば全パートに反映される ===
\newcommand{\PmodelDocVersion}{v1.0}
\newcommand{\PmodelDate}{2026-03-09 UTC}

% 組み立て済みマクロ
\newcommand{\PmodelFullDate}{%
  Document \PmodelDocVersion \quad \PmodelDate%
}
"""
# 条件分岐: `str(_ROOT) not in sys.path` を満たす経路を評価する。
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.summary import worklog

try:
    from sitecustomize import _translate_wavep_text_to_japanese as _translate_wavep_figure_text_to_japanese
except Exception:
    _translate_wavep_figure_text_to_japanese = None


# 関数: `_repo_root` の入出力契約と処理意図を定義する。
def _repo_root() -> Path:
    return _ROOT


# 関数: `_ensure_pmodel_version_style` の入出力契約と処理意図を定義する。
def _ensure_pmodel_version_style(*, root: Path, outdir: Path) -> Path:
    src = root / "pmodel_version.sty"
    dst = outdir / "pmodel_version.sty"

    # 条件分岐: `not src.exists()` を満たす経路を評価する。
    if not src.exists():
        src.write_text(_PMODEL_VERSION_STYLE_DEFAULT, encoding="utf-8")

    style_text = src.read_text(encoding="utf-8")
    dst.write_text(style_text, encoding="utf-8")
    return dst


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


# 関数: `_compact_label` の入出力契約と処理意図を定義する。
def _compact_label(label: str, *, max_len: int = 64) -> str:
    # 条件分岐: `len(label) <= max_len` を満たす経路を評価する。
    if len(label) <= max_len:
        return label

    digest = hashlib.sha1(label.encode("utf-8")).hexdigest()[:8]
    keep = max(8, max_len - 9)
    return f"{label[:keep]}-{digest}"


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

    if "最低条件" in raw_title and ("schr" in merged or "kg" in merged):
        return "p-schr-kg"

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

    # 条件分岐: Part4 の「Part III 検証サマリ表 行ラベル整合」見出しを固定ラベルへ写像する。
    if "検証サマリ表" in raw_title and "行ラベル整合" in raw_title:
        return "part-iii-scoreboard-label-parity"

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
    profile: str = "",
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
    # Part別の重複回避: 将来の統合コンパイルで衝突しやすい汎用ラベルのみ接頭辞を付与する。
    profile_prefix = ""
    if profile == "part2_astrophysics":
        profile_prefix = "p2"
    elif profile == "part3_quantum":
        profile_prefix = "p3"
    if profile_prefix and candidate in {"2-2-s2-2", "conclusion-s7"}:
        candidate = f"{profile_prefix}-{candidate}"
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
_LITERAL_REF_RE = re.compile(r"\\textbackslash\{\}ref\\\{((?:sec|fig):[^{}]+)\\\}")


# 関数: `_heading_math_to_pdftext` の入出力契約と処理意図を定義する。
def _heading_math_to_pdftext(payload: str) -> str:
    text = payload
    text = text.replace(r"\theta", "theta")
    text = text.replace(r"\phi", "phi")
    text = text.replace(r"\sigma", "sigma")
    text = text.replace(r"\Delta", "Delta")
    text = text.replace(r"\beta", "beta")
    text = text.replace(r"\gamma", "gamma")
    text = text.replace(r"\eta", "η")
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
        "η": "η",
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


# 関数: `_restore_literal_refs` の入出力契約と処理意図を定義する。
def _restore_literal_refs(tex_body: str) -> str:
    return _LITERAL_REF_RE.sub(r"\\ref{\1}", tex_body)


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

def _should_localize_figure_caption_to_japanese() -> bool:
    figure_lang = os.environ.get("WAVEP_FIGURE_LANG", "").strip().lower()
    if figure_lang == "en":
        return False

    raw_force = os.environ.get("WAVEP_MPL_FORCE_JA_TEXT", "").strip().lower()
    if raw_force in {"1", "true", "yes", "on"}:
        return True

    return figure_lang == "ja"


# 関数: `_fallback_caption_from_path` の入出力契約と処理意図を定義する。

def _fallback_caption_from_path(raw_path: str) -> str:
    stem = Path(raw_path).stem
    normalized = stem.replace("__", " ").replace("_", " ").replace("-", " ")
    normalized = re.sub(r"\s+", " ", normalized).strip()
    # 条件分岐: `not normalized` を満たす経路を評価する。
    if not normalized:
        if _should_localize_figure_caption_to_japanese():
            return "観測・理論比較の結果図。"

        return "Observation-theory comparison figure."

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
    if _should_localize_figure_caption_to_japanese() and _translate_wavep_figure_text_to_japanese is not None:
        text = _translate_wavep_figure_text_to_japanese(text)
        return f"{text}の比較結果を示す。"

    return f"Comparison result for {text}."


# 関数: `_normalize_figure_caption` の入出力契約と処理意図を定義する。
def _normalize_figure_caption(caption: str, raw_path: str) -> str:
    text = (caption or "").strip()
    text = re.sub(r"[:：]\s*$", "", text).strip()
    text = re.sub(r"^\s*(?:図|Figure|Fig\.?)\s*\((.+)\)\s*$", r"\1", text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*(?:図|Figure|Fig\.?)\s*（(.+)）\s*$", r"\1", text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*(?:図|Figure|Fig\.?)\s*[:：]\s*(.+)$", r"\1", text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*(?:図|Figure|Fig\.?)\s*$", "", text, flags=re.IGNORECASE).strip()

    if text in {"", "/"}:
        return _fallback_caption_from_path(raw_path)

    return text


# 関数: `_is_image_markdown_line` の入出力契約と処理意図を定義する。

def _is_image_markdown_line(stripped: str) -> bool:
    # 条件分岐: `_match_leading_image_line(stripped)` を満たす経路を評価する。
    if _match_leading_image_line(stripped):
        return True

    return bool(re.match(r"^!\[([^\]]*)\]\(([^)]+)\)\s*$", stripped))


# 関数: `_count_prose_lines_between_headings` の入出力契約と処理意図を定義する。
def _count_prose_lines_between_headings(lines: list[str], start_index: int, end_index: int) -> int:
    """
    Count prose-like markdown lines between headings.
    We exclude blank lines and structural lines (heading/list/table/code/math/image/rule),
    because the page-break rule should follow the amount of actual explanatory text.
    """
    start = max(0, int(start_index))
    end = min(len(lines), max(start, int(end_index)))
    in_code = False
    in_math = False
    prose_count = 0
    i = start
    while i < end:
        raw = lines[i]
        stripped = raw.strip()
        # 条件分岐: `in_code` を満たす経路を評価する。
        if in_code:
            if stripped.startswith("```"):
                in_code = False

            i += 1
            continue

        # 条件分岐: `in_math` を満たす経路を評価する。
        if in_math:
            if stripped == "$$" or stripped.endswith("$$"):
                in_math = False

            i += 1
            continue

        # 条件分岐: `not stripped` を満たす経路を評価する。
        if not stripped:
            i += 1
            continue

        # 条件分岐: `stripped.startswith("```")` を満たす経路を評価する。
        if stripped.startswith("```"):
            in_code = True
            i += 1
            continue

        # 条件分岐: `stripped == "$$"` を満たす経路を評価する。
        if stripped == "$$":
            in_math = True
            i += 1
            continue

        # 条件分岐: `stripped.startswith("$$")` を満たす経路を評価する。
        if stripped.startswith("$$"):
            if not (stripped.endswith("$$") and len(stripped) > 4):
                in_math = True

            i += 1
            continue

        # 条件分岐: `re.match(r"^(#{1,6})\\s+", stripped)` を満たす経路を評価する。
        if re.match(r"^(#{1,6})\s+", stripped):
            i += 1
            continue

        # 条件分岐: `_is_image_markdown_line(stripped)` を満たす経路を評価する。
        if _is_image_markdown_line(stripped):
            i += 1
            continue

        # 条件分岐: `re.match(r"^\\s*[-*]\\s+(.+)$", raw) or re.match(r"^\\s*\\d+[.)]\\s+(.+)$", raw)` を満たす経路を評価する。
        if re.match(r"^\s*[-*]\s+(.+)$", raw) or re.match(r"^\s*\d+[.)]\s+(.+)$", raw):
            i += 1
            continue

        # 条件分岐: `"|" in raw and (i + 1) < end and _is_table_separator(lines[i + 1])` を満たす経路を評価する。
        if "|" in raw and (i + 1) < end and _is_table_separator(lines[i + 1]):
            i += 1
            continue

        # 条件分岐: `_is_table_separator(raw)` を満たす経路を評価する。
        if _is_table_separator(raw):
            i += 1
            continue

        # 条件分岐: `re.match(r"^[-*_]{3,}\\s*$", stripped)` を満たす経路を評価する。
        if re.match(r"^[-*_]{3,}\s*$", stripped):
            i += 1
            continue

        # 条件分岐: `stripped in {"<!-- LATEX_CLEARPAGE -->", r"\\clearpage", r"\\newpage"}` を満たす経路を評価する。
        if stripped in {"<!-- LATEX_CLEARPAGE -->", r"\clearpage", r"\newpage"}:
            i += 1
            continue

        prose_count += 1
        i += 1

    return prose_count


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

# Prefer vector/pdf assets globally whenever available.
_FIGURE_EXT_SEARCH_ORDER = (".pdf", ".png", ".jpg", ".jpeg")
_PDF_PREFERRED_PROFILES = {"paper", "part2_astrophysics", "part3_quantum", "part4_verification", "part5_future_predictions"}


# 関数: `_copy_private_asset_to_public_if_needed` の入出力契約と処理意図を定義する。
def _copy_private_asset_to_public_if_needed(private_path: Path, public_path: Path) -> bool:
    # 条件分岐: `not private_path.exists()` を満たす経路を評価する。
    if not private_path.exists():
        return False

    needs_copy = not public_path.exists()
    # 条件分岐: `not needs_copy` を満たす経路を評価する。
    if not needs_copy:
        try:
            private_stat = private_path.stat()
            public_stat = public_path.stat()
            needs_copy = (
                private_stat.st_mtime_ns > public_stat.st_mtime_ns
                or private_stat.st_size != public_stat.st_size
            )
        except Exception:
            needs_copy = True

    # 条件分岐: `needs_copy` を満たす経路を評価する。
    if needs_copy:
        public_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(private_path, public_path)

    return public_path.exists()


# 関数: `_sync_public_mirror_for_output_reference` の入出力契約と処理意図を定義する。
def _sync_public_mirror_for_output_reference(raw_path: str, *, root: Path) -> None:
    normalized = _normalize_tex_path(raw_path.strip())
    # 条件分岐: `not normalized.startswith("output/")` を満たす経路を評価する。
    if not normalized.startswith("output/"):
        return

    # 条件分岐: `normalized.startswith("output/private/")` を満たす経路を評価する。
    if normalized.startswith("output/private/"):
        return

    pairs: list[tuple[Path, Path]] = []
    path_obj = Path(normalized)
    suffixes = list(_FIGURE_EXT_SEARCH_ORDER)
    # 条件分岐: `path_obj.suffix and path_obj.suffix.lower() not in suffixes` を満たす経路を評価する。
    if path_obj.suffix and path_obj.suffix.lower() not in suffixes:
        suffixes.append(path_obj.suffix.lower())

    # 条件分岐: `normalized.startswith("output/public/")` を満たす経路を評価する。
    explicit_public = normalized.startswith("output/public/")
    if explicit_public:
        tail = normalized[len("output/public/") :]
        private_base = root / "output" / "private" / Path(tail)
        public_base = root / "output" / "public" / Path(tail)
    else:
        tail = normalized[len("output/") :]
        private_base = root / "output" / "private" / Path(tail)
        public_base = root / "output" / "public" / Path(tail)

    for ext in suffixes:
        pairs.append((private_base.with_suffix(ext), public_base.with_suffix(ext)))

    pairs.append((private_base, public_base))

    seen: set[str] = set()
    for private_path, public_path in pairs:
        key = f"{private_path}|{public_path}"
        # 条件分岐: `key in seen` を満たす経路を評価する。
        if key in seen:
            continue

        seen.add(key)
        # 明示的に output/public/... を参照している場合は public を正本として扱う。
        # public が欠けているときだけ private から補完し、既存 public を private で上書きしない。
        if explicit_public and public_path.exists():
            continue

        _copy_private_asset_to_public_if_needed(private_path, public_path)


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

    # 関数: `add_candidate_variants` の入出力契約と処理意図を定義する。
    def add_candidate_variants(path_obj: Path) -> None:
        # Prefer vector/pdf assets first, then bitmap fallbacks.
        base = path_obj.with_suffix("") if path_obj.suffix else path_obj
        for ext in _FIGURE_EXT_SEARCH_ORDER:
            add_candidate(base.with_suffix(ext))

        add_candidate(path_obj)

    # 関数: `add_candidate_variants_across_roots` の入出力契約と処理意図を定義する。
    def add_candidate_variants_across_roots(path_objs: list[Path]) -> None:
        bases = [(p.with_suffix("") if p.suffix else p) for p in path_objs]
        # Keep extension priority global across roots: any PDF is preferred over any PNG/JPG.
        for ext in _FIGURE_EXT_SEARCH_ORDER:
            for base in bases:
                add_candidate(base.with_suffix(ext))

        for p in path_objs:
            add_candidate(p)

    path_obj = Path(normalized)
    # 条件分岐: `path_obj.is_absolute()` を満たす経路を評価する。
    if path_obj.is_absolute():
        add_candidate_variants(path_obj)
    else:
        _sync_public_mirror_for_output_reference(normalized, root=root)
        # Legacy "output/<topic>/..." references are resolved only via modern roots.
        # This eliminates accidental mixing with stale assets under legacy "output/<topic>".
        is_legacy_output = normalized.startswith("output/") and not normalized.startswith(
            "output/public/"
        ) and not normalized.startswith("output/private/")
        if is_legacy_output:
            tail = normalized[len("output/") :]
            # Canonical precedence for paper builds: public (published mirror) -> private (local fallback).
            # Extension priority is evaluated across both roots so PDF wins even if PNG exists in another root.
            add_candidate_variants_across_roots(
                [root / "output" / "public" / Path(tail), root / "output" / "private" / Path(tail)]
            )
        else:
            add_candidate_variants(root / path_obj)

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

def _render_bibliography_section(profile: str = "") -> str:
    # 条件分岐: `not _USED_REFERENCE_KEYS` を満たす経路を評価する。
    if not _USED_REFERENCE_KEYS:
        return ""

    lines: list[str] = [""]
    # Part3 は「章（section）直前のみ改ページ」を運用固定とするため、
    # 文献節直前の強制改ページを入れない。
    if profile != "part3_quantum":
        lines.append(r"\clearpage")

    lines += [r"\section*{References}", r"\begin{thebibliography}{99}"]
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


# 関数: `_convert_raster_image_to_pdf` の入出力契約と処理意図を定義する。
def _convert_raster_image_to_pdf(src_image: Path, dst_pdf: Path) -> bool:
    """
    Convert a raster image to PDF for TeX inclusion.
    Returns True when conversion succeeded.
    """
    try:
        from PIL import Image
    except Exception:
        Image = None

    # 条件分岐: `Image is not None` を満たす経路を評価する。
    if Image is not None:
        try:
            with Image.open(src_image) as im:
                if im.mode in {"RGBA", "LA"}:
                    bg = Image.new("RGB", im.size, (255, 255, 255))
                    alpha = im.getchannel("A") if "A" in im.getbands() else None
                    bg.paste(im.convert("RGB"), mask=alpha)
                    out = bg
                else:
                    out = im.convert("RGB")

                dst_pdf.parent.mkdir(parents=True, exist_ok=True)
                out.save(dst_pdf, "PDF", resolution=300.0)

            return True
        except Exception:
            pass

    try:
        import matplotlib.image as mpimg
        import matplotlib.pyplot as plt

        arr = mpimg.imread(str(src_image))
        if hasattr(arr, "shape") and len(arr.shape) >= 2:
            h_px = int(arr.shape[0]) if int(arr.shape[0]) > 0 else 1000
            w_px = int(arr.shape[1]) if int(arr.shape[1]) > 0 else 1000
        else:
            h_px, w_px = 1000, 1000

        dpi = 200.0
        fig = plt.figure(figsize=(w_px / dpi, h_px / dpi), dpi=dpi)
        ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
        ax.axis("off")
        ax.imshow(arr)
        dst_pdf.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(dst_pdf, format="pdf", bbox_inches="tight", pad_inches=0.0)
        plt.close(fig)
        return True
    except Exception:
        return False


# 関数: `_render_figure_block` の入出力契約と処理意図を定義する。

def _render_figure_block(
    *,
    raw_path: str,
    caption: str,
    profile: str,
    root: Path,
    outdir: Path,
    figures_dir: Path,
    staged_assets: dict[str, str],
    used_figure_names: set[str],
    used_figure_labels: dict[str, int],
) -> list[str]:
    # 既に出力済みの図数から、今回の図番号（1始まり）を決定する。
    figure_index = sum(used_figure_labels.values()) + 1
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
            desired_suffix = suffix
            if profile in _PDF_PREFERRED_PROFILES and suffix.lower() in {".png", ".jpg", ".jpeg"}:
                desired_suffix = ".pdf"

            candidate = f"{stem}{desired_suffix}"
            serial = 2
            while candidate.lower() in used_figure_names:
                candidate = f"{stem}__{serial}{desired_suffix}"
                serial += 1

            used_figure_names.add(candidate.lower())
            dst = figures_dir / candidate
            dst.parent.mkdir(parents=True, exist_ok=True)
            staged_ok = False
            if profile in _PDF_PREFERRED_PROFILES and desired_suffix == ".pdf" and suffix.lower() in {".png", ".jpg", ".jpeg"}:
                staged_ok = _convert_raster_image_to_pdf(resolved_obj, dst)

            if not staged_ok:
                if desired_suffix != suffix:
                    fallback_candidate = f"{stem}{suffix}"
                    serial = 2
                    while fallback_candidate.lower() in used_figure_names:
                        fallback_candidate = f"{stem}__{serial}{suffix}"
                        serial += 1

                    candidate = fallback_candidate
                    dst = figures_dir / candidate
                    used_figure_names.add(candidate.lower())

                skip_copy_same_file = False
                try:
                    skip_copy_same_file = resolved_obj.resolve() == dst.resolve()
                except Exception:
                    skip_copy_same_file = str(resolved_obj) == str(dst)

                if not skip_copy_same_file:
                    try:
                        shutil.copy2(resolved_obj, dst)
                    except PermissionError:
                        # Windows ではPDFプレビュー等で一時ロックが残る場合がある。
                        # 既に同名ファイルが配置済みならコピーを省略して続行する。
                        if not dst.exists():
                            raise

            tex_path = candidate
            staged_assets[source_key] = candidate

    normalized_caption = _normalize_figure_caption(caption, raw_path)

    caption_text = _convert_inline(normalized_caption)
    # 条件分岐: `not exists` を満たす経路を評価する。
    if not exists:
        caption_text = _convert_inline(f"{normalized_caption} (missing file: {raw_path})")

    tex_path_obj = Path(tex_path)
    image_stem = tex_path_obj.stem.lower()

    # Part4 の検証サマリ表ラベル整合監査図は、本文要件に合わせてキャプション/ラベルを固定する。
    custom_label_base = ""
    if profile == "part4_verification" and image_stem == "table1_part4_label_parity_audit":
        caption_text = _convert_inline("検証サマリ表 Part IV ラベル整合監査の比較結果を示す。")
        custom_label_base = "fig:p4:scoreboard-part4-label-parity-audit"
    if profile == "part2_astrophysics" and image_stem == "delta_saturation_constraints":
        custom_label_base = "fig:p2:delta-saturation-constraints"

    label_source = Path(tex_path).stem or Path(raw_path).stem or "figure"
    label_slug = _compact_label(_safe_label(label_source), max_len=52)
    if custom_label_base:
        label_base = custom_label_base
    elif profile == "part4_verification":
        label_base = f"fig:p4:{label_slug}"
    else:
        label_base = f"fig:{label_slug}"
    label_count = used_figure_labels.get(label_base, 0) + 1
    used_figure_labels[label_base] = label_count
    figure_label = label_base if label_count == 1 else f"{label_base}-{label_count}"
    # Keep the resolved extension explicitly so PDF preference stays deterministic.
    includegraphics_target = _normalize_tex_path(tex_path)
    includegraphics_opts = r"width=\linewidth"
    if profile == "part3_quantum":
        includegraphics_opts = r"width=\linewidth"
        # Part3 指定図の可読性改善（図内フォントをさらに少し大きく見せる）。
        part3_mild_upscale_indices = {
            1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
            29, 30, 31, 33, 35, 36, 37, 38, 39, 40, 41, 42, 43, 45,
            52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67,
            72, 73, 74, 76,
        }
        if figure_index in part3_mild_upscale_indices:
            includegraphics_opts = r"width=1.04\linewidth"
        # Part3 の凡例/注記が小さい図は、追加で表示スケールを上げる。
        part3_legend_note_upscale_indices = {
            3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
            29, 30, 31, 33, 35, 36, 37, 38, 39, 40, 41,
            54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67,
            72, 76,
        }
        if figure_index in part3_legend_note_upscale_indices:
            includegraphics_opts = r"width=1.08\linewidth"
        # ユーザー指定図は、フォント可読性を優先してさらに拡大する。
        part3_target_font_upscale_indices = {
            2, 3, 4, 5, 9, 13, 16, 17, 18, 19,
            29, 30, 31, 33, 35, 36, 37, 38, 39, 40, 41,
            61, 73, 74, 76,
        }
        if figure_index in part3_target_font_upscale_indices:
            includegraphics_opts = r"width=1.12\linewidth"
    if profile == "part2_astrophysics":
        # Part II default: keep readability while still preventing page overflow.
        includegraphics_opts = r"width=\linewidth,height=0.62\paperheight,keepaspectratio"
    elif profile == "part4_verification":
        # Part IV default: 余白維持を優先し、本文幅内に収める。
        includegraphics_opts = r"width=\linewidth,height=0.78\paperheight,keepaspectratio"

    part2_height_overrides = {
        # Extra cap for figures that tend to overflow or crowd captions.
        "validation_scoreboard": r"width=\linewidth,height=0.68\paperheight,keepaspectratio",
        "cassini_pds_vs_digitized": r"width=\linewidth,height=0.58\paperheight,keepaspectratio",
        "solar_light_deflection": r"width=\linewidth,height=0.68\paperheight,keepaspectratio",
        "viking_p_model_vs_measured_no_arrow": r"width=\linewidth,height=0.66\paperheight,keepaspectratio",
        "mercury_orbit": r"width=\linewidth,height=0.80\paperheight,keepaspectratio",
        "fek_relativistic_broadening_isco_constraints": r"width=\linewidth,height=0.66\paperheight,keepaspectratio",
        "sparc_rotation_curve_pmodel_audit": r"width=\linewidth,height=0.64\paperheight,keepaspectratio",
        "cosmology_bao_xi_multipole_peakfit": r"width=\linewidth,height=0.77\paperheight,keepaspectratio",
        "cosmology_cmb_polarization_phase_audit": r"width=\linewidth,height=0.80\paperheight,keepaspectratio",
        "cosmology_fsigma8_growth_mapping": r"width=\linewidth,height=0.64\paperheight,keepaspectratio",
    }
    if profile == "part2_astrophysics" and image_stem in part2_height_overrides:
        includegraphics_opts = part2_height_overrides[image_stem]

    part4_size_overrides = {
        # Part4 図1/図2は本文幅に揃え、余白超過を避ける。
        "validation_scoreboard": r"width=\linewidth,height=0.72\paperheight,keepaspectratio",
        "quantum_scoreboard": r"width=\linewidth,height=0.72\paperheight,keepaspectratio",
        # Part4 図3も本文幅に固定する。
        "table1_part4_label_parity_audit": r"width=\linewidth,height=0.74\paperheight,keepaspectratio",
        # Part4 図11/95/99: 1列化した縦長図のページ内収まりを安定化。
        "lagrangian_noether_rotational_closure_audit": r"width=\linewidth,height=0.86\paperheight,keepaspectratio",
        "llr_operational_metrics_audit": r"width=\linewidth,height=0.86\paperheight,keepaspectratio",
        "sparc_rotation_curve_pmodel_audit": r"width=\linewidth,height=0.86\paperheight,keepaspectratio",
    }
    if profile == "part4_verification" and image_stem in part4_size_overrides:
        includegraphics_opts = part4_size_overrides[image_stem]

    tall_part3_figures = {
        "nuclear_effective_potential_two_range_fit_as_rs",
        "nuclear_effective_potential_pion_constrained_barrier_tail_kq_scan",
        "nuclear_effective_potential_pion_constrained_barrier_tail_channel_split_kq_scan_triplet_barrier_fraction_scan",
    }
    # Part3 図23-28（eq18/eq19）サイズを統一し、図23のみ縦方向を追加拡大する。
    part3_height_overrides = {
        "nuclear_effective_potential_two_range_fit_as_rs_eq18": r"width=\linewidth,height=0.98\textheight,keepaspectratio",
        "nuclear_effective_potential_two_range_fit_as_rs_eq19": r"width=\linewidth,height=0.98\textheight,keepaspectratio",
        "nuclear_effective_potential_pion_constrained_barrier_tail_kq_scan_eq18": r"width=\linewidth,height=0.98\textheight,keepaspectratio",
        "nuclear_effective_potential_pion_constrained_barrier_tail_kq_scan_eq19": r"width=\linewidth,height=0.98\textheight,keepaspectratio",
        "nuclear_effective_potential_pion_constrained_barrier_tail_channel_split_kq_scan_triplet_barrier_fraction_scan_eq18": r"width=\linewidth,height=0.98\textheight,keepaspectratio",
        "nuclear_effective_potential_pion_constrained_barrier_tail_channel_split_kq_scan_triplet_barrier_fraction_scan_eq19": r"width=\linewidth,height=0.98\textheight,keepaspectratio",
        # 図71: 横幅は本文幅に揃え、縮小率を抑えて可読性を優先する。
        "systematics_decomposition_15items": r"width=\linewidth,height=0.88\textheight,keepaspectratio",
    }
    if profile == "part3_quantum" and image_stem in part3_height_overrides:
        includegraphics_opts = part3_height_overrides[image_stem]
    # Backward compatibility for non-eq split filenames.
    elif image_stem in tall_part3_figures:
        includegraphics_opts = r"width=\linewidth"

    # Part4 図1/図2は最終段でも本文幅に固定し、見た目と余白を一致させる。
    if profile == "part4_verification" and (
        "validation_scoreboard" in image_stem or "quantum_scoreboard" in image_stem
    ):
        includegraphics_opts = r"width=\linewidth,height=0.72\paperheight,keepaspectratio"

    # Keep strict in-text ordering for figure-adjacent explanations/supplements.
    figure_float_spec = "H"

    includegraphics_line = r"\includegraphics[" + includegraphics_opts + "]{" + includegraphics_target + "}"
    if profile in {"part3_quantum", "part4_verification"} and "width=1." in includegraphics_opts:
        includegraphics_line = r"\makebox[\linewidth][c]{" + includegraphics_line + "}"

    return [
        r"\begin{figure}[" + figure_float_spec + "]",
        r"\centering",
        includegraphics_line,
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


# 関数: `_looks_like_wrappable_code_literal` の入出力契約と処理意図を定義する。
def _looks_like_wrappable_code_literal(s: str) -> bool:
    candidate = s.strip()
    if not candidate:
        return False

    if any(token in candidate for token in ("\\", "{", "}", "$")):
        return False

    if _looks_like_artifact_code(candidate):
        return True

    if len(candidate) >= 20 and candidate.count("_") >= 2:
        return True

    return bool(
        len(candidate) >= 24
        and re.fullmatch(r"[A-Za-z0-9_.:/=+\-\[\],()%]+", candidate)
    )


# 関数: `_render_code_literal` の入出力契約と処理意図を定義する。
def _render_code_literal(payload: str) -> str:
    code = payload.strip()
    if not code:
        return ""

    if "://" in code:
        return r"\url{" + code + "}"

    if _looks_like_wrappable_code_literal(code):
        return r"\nolinkurl{" + code + "}"

    return r"\texttt{" + _escape_tex(code) + r"}"


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


# 関数: `_compact_display_math` の入出力契約と処理意図を定義する。
def _compact_display_math(math_body: str) -> str:
    compact = re.sub(r"\s+", "", math_body)
    compact = compact.replace(r"\,", "")
    compact = compact.replace(r"\!", "")
    return compact


_PART2_REQUIRED_EQUATION_RULES: tuple[tuple[str, tuple[str, ...], bool], ...] = (
    # A. 理論的枠組み
    ("a_phi_def", (r"\phi\equiv-c^2\ln\left(\frac{P}{P_0}\right)",), False),
    ("a_motion_eq", (r"\mathbf{a}=-\nabla\phi=c^2\nabla\ln\left(\frac{P}{P_0}\right)",), False),
    ("a_clock_map", (r"\frac{d\tau}{dt}=\frac{P_0}{P}\left(\frac{d\tau}{dt}\right)_v",), False),
    ("a_velocity_term", (r"\left(\frac{d\tau}{dt}\right)_v=\sqrt{1-\frac{v^2}{c^2}}",), False),
    ("a_refraction", (r"n(P)=\left(\frac{P}{P_0}\right)^{2\beta}",), False),
    # B. 弱場検証
    ("b_alpha_b", (r"\alpha(b)\approx\frac{4\betaGM}{c^2b}",), False),
    ("b_gamma_pred", (r"\gamma_{\rmpred}=2\beta-1",), False),
    ("b_shapiro", (r"\Deltat_{\mathrm{Shapiro}}\approx",), False),
    ("b_mercury_precession", (r"\Delta\omega\approx\frac{6\piGM_{\odot}}{c^2a(1-e^2)}",), False),
    ("b_gps_rel", (r"\Deltat_{\mathrm{rel}}=\frac{2\mathbf{r}\cdot\mathbf{v}}{c^2}",), False),
    # C. EHT強場
    ("c_theta_ring", (r"\theta_{\rmring}=\kappa\theta_{\rmsh}",), True),
    ("c_kappa_def", (r"\kappa\equiv\frac{b_{\mathrm{peak}}}{b_{\mathrm{sh}}(P)}",), False),
    ("c_coeff_diff", (r"C_{\rmP}=2e\beta", r"C_{\rmref}=\sqrt{27}"), False),
    ("c_gamma_max", (r"\gamma_{\max}\simeq\frac{1}{\sqrt{\delta_0}}",), False),
    # D. 連星重力波
    ("d_dipole_forbidden", (r"D_{P}=\kappaMX_{\mathrm{cm}},\quadX_{\mathrm{cm}}=0\Rightarrow\ddot{D}_P=0",), False),
    ("d_quad_formula", (r"\dot{P}_b^{\rmquad}",), False),
    ("d_orbital_ratio", (r"R\equiv\frac{\dot{P}_{b,\rmint}}{\dot{P}_b^{\rmquad}}",), False),
    ("d_chirp_eq", (r"\frac{df}{dt}=\frac{96}{5}\pi^{8/3}",), False),
    # E. フレームドラッグ
    ("e_p_phi", (r"P_\phi(r,\theta)=\frac{g_PJ_*}{2c^2}\frac{\sin^2\theta}{r^2}",), False),
    ("e_omega_pred", (r"\Omega_{\mathrm{pred}}^{(P)}", r"=\frac{|g_{P}\xi_{\mathrm{rot}}J_*|}{4c^2r^3}"), False),
    ("e_mu_ratio", (r"\mu\equiv\frac{|\Omega_{\rmobs}|}{|\Omega_{\rmpred}|}",), False),
    # F. 銀河回転曲線
    ("f_g_p", (r"g_{\rmP}=\frac{g_{\rmbar}}{1-\exp\left(-\sqrt{g_{\rmbar}/a_0}\right)}",), False),
    # G. 銀河団衝突
    ("g_delay_response", (r"\tau\frac{dX_P}{dt}+X_{P}=",), False),
    ("g_gamma_path", (r"\Gamma_{\mathrm{path}}", r"=\Gamma_{\mathrm{adv}}+\Gamma_{\mathrm{coll}}"), False),
    # H. 宇宙論DDR
    ("h_eta_p_def", (r"\eta^{(P)}(z)\equiv\frac{D_{L}}{(1+z)D_{A}}",), False),
    ("h_eta_std_def", (r"\eta(z)\equiv\frac{D_{L}}{(1+z)^2D_{A}}",), False),
    ("h_eta_p_eps", (r"\eta^{(P)}(z)\equiv\frac{D_{L}}{(1+z)D_{A}}=(1+z)\eta(z)=(1+z)^{1+\epsilon_0}",), False),
    ("h_dl_p", (r"D_{L}^{(P)}=(1+z)D_{A}^{(P)}",), False),
    ("h_reconnect", (r"\Delta\mu(z)=5\log_{10}((1+z)^{\Delta\epsilon}),", r"\tau(z)=2\alpha\ln(1+z)"), False),
    # I. CMB音響ピーク
    ("i_driven_osc", (r"\Theta_{k}''+\Gamma_{P}\Theta_{k}'+c_{s,P}^2k^2\Theta_{k}=-k^2\Psi_{P}",), False),
    ("i_peak_pred", (r"\ell_{n}^{(P)}\approx(n-\phi_{P})\ell_{A}^{(P)},", r"A_{n}^{(P)}=A_0e^{-\delta_{P}(n-1)}"), False),
    ("i_dm_free_damping", (r"\frac{A_3}{A_1}=", r"<1", r"(\delta_{P}>0"), False),
    ("i_phase_shift", (r"\Deltax_{EE-TT}=\frac{\pi}{2},", r"\Deltax_{TE-TT}=\frac{\pi}{4}"), False),
    # J. 構造形成
    ("j_growth_eq", (r"\delta^{(2)}+\Gamma_{\mathrm{eff}}\delta^{(1)}+\left(c_{s}^2k^2-4\piG\bar\rho\right)\delta=0",), False),
)

_DISPLAY_MATH_BLOCK_RE = re.compile(r"\\\[\n(?P<body>.*?)\n\\\]", flags=re.DOTALL)


# 関数: `_apply_part2_selective_equation_numbering` の入出力契約と処理意図を定義する。
def _apply_part2_selective_equation_numbering(tex_body: str) -> str:
    used_rule_keys: set[str] = set()

    def repl(match: re.Match[str]) -> str:
        body = match.group("body")
        compact = _compact_display_math(body)
        for rule_key, required_tokens, exact_match in _PART2_REQUIRED_EQUATION_RULES:
            if rule_key in used_rule_keys:
                continue
            if exact_match:
                if compact != required_tokens[0]:
                    continue
            elif not all(token in compact for token in required_tokens):
                continue

            used_rule_keys.add(rule_key)
            eq_label = f"eq:part2-{rule_key}"
            return (
                r"\begin{equation}\label{"
                + eq_label
                + "}"
                + "\n"
                + body
                + "\n"
                + r"\end{equation}"
            )

        return match.group(0)

    return _DISPLAY_MATH_BLOCK_RE.sub(repl, tex_body)


_PART3_REQUIRED_EQUATION_RULES: tuple[tuple[str, tuple[str, ...], bool], ...] = (
    # 2 理論的枠組み
    ("eq:phi-def", (r"\phi\equiv-c^2\ln\left(\frac{P}{P_0}\right)",), False),
    ("eq:clock-map", (r"\frac{d\tau}{dt}=\frac{P_0}{P}\left(\frac{d\tau}{dt}\right)_v",), False),
    ("eq:refraction", (r"n(P)=\left(\frac{P}{P_0}\right)^{2\beta}",), False),
    ("eq:schrodinger-p", (r"i\hbar\frac{\partial\psi}{\partialt}=\left(-\frac{\hbar^2}{2m}\nabla^2+m\phi\right)\psi",), False),
    ("eq:kg-p", (r"-\frac{1}{c^2}\frac{\partial^2}{\partialt^2}+\nabla^2-\frac{m^2c^2}{\hbar^2}", r"\Psi=0"), False),
    ("eq:born-rule", (r"p(x)\propto|\psi(x)|^2",), False),
    ("eq:self-gravity-scale", (r"\frac{E_{G}}{\hbar}T\sim\frac{Gm^2}{\hbarR}T\gtrsim1",), False),
    ("eq:selection-weight", (r"P_{\rmobs}(x,y\mida,b)=", r"w_{ab}(\lambda)"), False),
    ("eq:continuous-measurement", (r"d|\psi\rangle=\left[-iHdt-\frac{\gamma_{m}}{2}\left(M-\langleM\rangle\right)^2dt+\sqrt{\gamma_{m}}\left(M-\langleM\rangle\right)dW_t\right]|\psi\rangle",), False),
    ("eq:lagrangian-em", (r"\mathcal{L}=\lvertD_\muP\rvert^2-V(\lvertP\rvert)-\frac{1}{4}F_{\mu\nu}F^{\mu\nu}",), False),
    ("eq:schrodinger-em", (r"\frac{(-i\hbar\nabla-qA)^2}{2m}+qA_0+m\phi",), False),
    ("eq:total-action", (r"\begin{aligned}S_{\mathrm{total}}", r"\mathcal{L}_{\mathrm{total}}^{\mathrm{vec}}"), False),
    # 4 結果
    ("eq:clock-comparison", (r"\ln\left(\frac{P}{P_0}\right)=x", r"\left(\frac{d\tau}{dt}\right)_{\rmGR}=\sqrt{1-2x}"), False),
    ("eq:delta-z-weak", (r"\frac{\deltaz}{z_{\rmGR}}\approx-(x_{\rmlow}+x_{\rmhigh})",), False),
    ("eq:ere", (r"k\cot\delta_{t,s}(k)=-\frac{1}{a_{t,s}}+\frac{1}{2}r_{t,s}k^2+v_{2,t,s}k^4+\cdots",), False),
    ("eq:delta2r", (r"\Delta^2r(x)=r(x+2)-2r(x)+r(x-2)",), False),
    ("eq:nu-sat", (r"\nu_{\rmbase}=\frac{2(A-1)}{A},\quad\nu_{\rmeff}=\min(\nu_{\rmbase},\nu_{\rmsat}),\quadC_{\rmeff}=\frac{\nu_{\rmeff}A}{2}",), False),
    ("eq:collapse-sim", (r"d|\psi\rangle=", r"H=\frac{\omega}{2}\sigma_{x}+\frac{\omega_{\mathrm{env}}}{2}E", r"M=\sigma_{z}"), False),
    # 5 差分予測
    ("eq:r-sel", (r"\Delta_{\mathrm{sys}}(T)\equiv\max_kT(k)-\min_kT(k),\qquadR_{\mathrm{sel}}\equiv\frac{\Delta_{\mathrm{sys}}(T)}{\sigma_{\mathrm{stat,med}}}",), False),
    ("eq:chsh", (r"E(a,b)\equiv\frac{N_{++}+N_{--}-N_{+-}-N_{-+}}", r"S\equivE(a,b)-E(a,b')+E(a',b)+E(a',b')"), False),
    ("eq:ch-prob", (r"J_{\mathrm{prob}}\equivP_{++}(a,b)-P_{++}(a,b')+P_{++}(a',b)+P_{++}(a',b')-P_{+}(a')-P_{+}(b)",), False),
    ("eq:z-delay", (r"dt\equivt_{B}-t_{A}-\mathrm{offset},\qquadz_{\mathrm{delay}}\equiv\frac{\lvert\Delta\mathrm{median}\rvert}{\sigma_{\Delta\mathrm{median}}}",), False),
    ("eq:bbn-kinetics", (r"\frac{dX_n}{dt}=-\Gamma_{np}(T)\left[X_{n}-X_{n}^{\mathrm{eq}}(T)\right]",), False),
    ("eq:bbn-freeze", (r"A_{w}T_{F}^5=C_{F}(T_{F})\frac{q_{B}}{t_{B}}\left(\frac{T_{F}}{T_{B}}\right)^{1/q_{B}}",), False),
    ("eq:yp", (r"Y_{p}=\frac{2(n/p)_N}{1+(n/p)_N}",), False),
)


# 関数: `_apply_part3_selective_equation_numbering` の入出力契約と処理意図を定義する。
def _apply_part3_selective_equation_numbering(tex_body: str) -> str:
    used_labels: set[str] = set()

    def repl(match: re.Match[str]) -> str:
        body = match.group("body")
        compact = _compact_display_math(body)
        for eq_label, required_tokens, exact_match in _PART3_REQUIRED_EQUATION_RULES:
            if eq_label in used_labels:
                continue
            if exact_match:
                if compact != required_tokens[0]:
                    continue
            elif not all(token in compact for token in required_tokens):
                continue

            used_labels.add(eq_label)
            return (
                r"\begin{equation}\label{"
                + eq_label
                + "}"
                + "\n"
                + body
                + "\n"
                + r"\end{equation}"
            )

        return match.group(0)

    return _DISPLAY_MATH_BLOCK_RE.sub(repl, tex_body)


# 関数: `_postprocess_latex_body` の入出力契約と処理意図を定義する。

def _postprocess_latex_body(body: str) -> str:
    # 本文全体へ数式コマンド補正をかけると \multicolumn などの
    # 構造コマンドまで壊すため、ここでは生テキストを起点にする。
    normalized = body
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
    normalized, verbatim_token_map = _protect_verbatim_tex_commands(
        normalized,
        commands=("nolinkurl", "url", "texttt"),
    )

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
        lambda m: _render_code_literal(m.group(1) + "=" + m.group(2)),
        normalized,
    )
    normalized = re.sub(
        r"(?<!\\)step(?:_[A-Za-z0-9]+){2,}",
        lambda m: m.group(0).replace("_", r"\_"),
        normalized,
    )
    normalized = re.sub(
        r"\$next=([A-Za-z0-9\\_]+)\$",
        lambda m: _render_code_literal("next=" + m.group(1).replace(r"\_", "_")),
        normalized,
    )
    normalized = re.sub(
        r"\$([A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+){2,})\$",
        lambda m: _render_code_literal(m.group(1)),
        normalized,
    )
    for token, rendered in verbatim_token_map.items():
        normalized = normalized.replace(token, rendered)

    # 関数: `_texttt_allowbreak` の入出力契約と処理意図を定義する。
    def _texttt_allowbreak(match: re.Match[str]) -> str:
        payload = match.group(1)
        if r"\allowbreak" in payload or len(payload) < 20:
            return match.group(0)

        rendered_payload = payload
        for source, target in (
            (r"\_", r"\_\allowbreak "),
            ("/", r"/\allowbreak "),
            (".", r".\allowbreak "),
            ("=", r"=\allowbreak "),
            (",", r",\allowbreak "),
            (":", r":\allowbreak "),
            ("-", r"-\allowbreak "),
        ):
            rendered_payload = rendered_payload.replace(source, target)

        return r"\texttt{" + rendered_payload + "}"

    normalized = re.sub(r"\\texttt\{([^{}]+)\}", _texttt_allowbreak, normalized)
    # 同一行に連結された「補足」見出しを段落分離する。
    normalized = re.sub(
        r"([^\n])[ \t]+\\textbf\{補足\}：",
        r"\1\n\n\\textbf{補足}：",
        normalized,
    )
    return normalized


# 関数: `_protect_verbatim_tex_commands` の入出力契約と処理意図を定義する。
def _protect_verbatim_tex_commands(text: str, *, commands: tuple[str, ...]) -> tuple[str, dict[str, str]]:
    token_map: dict[str, str] = {}
    protected = text
    token_index = 0
    for command in commands:
        pattern = re.compile(rf"\\{command}\{{([^{{}}]*)\}}")

        def repl(match: re.Match[str]) -> str:
            nonlocal token_index
            key = f"@@VERBATIM{token_index}@@"
            token_map[key] = match.group(0)
            token_index += 1
            return key

        protected = pattern.sub(repl, protected)

    return protected, token_map


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

        # backtick code spans should preserve machine-readable artifacts before any math heuristics run.
        if _looks_like_artifact_code(payload):
            return make_token(_render_code_literal(payload))

        # 条件分岐: `_looks_like_physics_equation_code(payload) or _looks_like_physics_symbol_code...` を満たす経路を評価する。

        if _looks_like_physics_equation_code(payload) or _looks_like_physics_symbol_code(payload):
            return make_token("$" + _normalize_inline_math_payload(payload) + "$")

        # 条件分岐: `_looks_like_math_code(payload)` を満たす経路を評価する。

        if _looks_like_math_code(payload):
            return make_token("$" + _normalize_inline_math_payload(payload) + "$")

        return make_token(_render_code_literal(payload))

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
            return make_token(_render_code_literal(payload))

        # 条件分岐: `re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*(?:\.[A-Za-z0-9_]+)+", payload)` を満たす経路を評価する。

        if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*(?:\.[A-Za-z0-9_]+)+", payload):
            return make_token(_render_code_literal(payload))

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
    # 図下説明の基準フォーム: 「補足：」を太字見出しとして統一する。
    text = text.replace("補足：", "**補足**：")
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

def _render_table(
    block_lines: list[str],
    *,
    caption_text: str = "",
    label_text: str = "",
    profile: str = "",
) -> list[str]:
    # 条件分岐: `len(block_lines) < 2` を満たす経路を評価する。
    if len(block_lines) < 2:
        return [_convert_inline(block_lines[0])] if block_lines else []

    header = _parse_table_row(block_lines[0])
    body_lines = block_lines[2:] if _is_table_separator(block_lines[1]) else block_lines[1:]
    rows = [_parse_table_row(line) for line in body_lines]
    ncols = max(1, len(header))
    score_kind_table = (
        ncols == 5
        and any("score_kind" in c.lower() for c in header)
        and any("判定変数" in c for c in header)
    )
    parameter_policy_table = (
        ncols == 5
        and any("代表パラメータ" in c for c in header)
        and any("再調整可否" in c for c in header)
        and any("追跡先" in c for c in header)
    )
    header_join = " || ".join(header)
    header_join_lower = header_join.lower()
    eht_kappa_main_result_table = (
        ncols == 7
        and ("ring,obs" in header_join_lower or "theta_{\\mathrm{ring,obs}}" in header_join_lower)
        and ("kappa" in header_join_lower or "\\kappa" in header_join_lower)
    )
    eht_kappa_fp_result_table = ncols == 6 and "kappa_{\\mathrm{fp}}" in header_join_lower
    cosmology_ddr_result_table = (
        ncols == 7
        and (
            "uses_bao" in header_join_lower
            or "uses\\_bao" in header_join_lower
            or "uses bao" in header_join_lower
        )
        and ("eta^{(p)}" in header_join_lower or "\\eta" in header_join_lower)
    )
    part3_diff_summary_table = (
        ncols == 7
        and any("テーマ" in c for c in header)
        and any("標準QM予測" in c for c in header)
        and any("棄却条件" in c for c in header)
    )
    part3_born_result_table = (
        ncols == 7
        and any("シナリオ" in c for c in header)
        and ("m_{\\mathrm{crit}}" in header_join_lower or "m_{\\rm crit}" in header_join_lower)
    )
    part3_bell_selection_sweep_table = (
        ncols >= 7
        and any("dataset" in c.lower() for c in header)
        and any("selection凍結" in c for c in header)
        and any("sweep" in c.lower() for c in header)
        and (
            "Δs" in header_join
            or "\\Delta S" in header_join
            or "Δj" in header_join_lower
            or "j_sweep" in header_join_lower
        )
    )
    part3_bell_frozen_table = (
        ncols == 8
        and any("dataset" in c.lower() for c in header)
        and any("selection凍結" in c for c in header)
        and ("s_{\\mathrm{frozen}}" in header_join_lower or "s_frozen" in header_join_lower)
        and ("Δs" in header_join or "\\Delta S" in header_join)
    )
    molecular_constants_table = (
        ncols == 7
        and any("分子" in c for c in header)
        and ("\\omega" in header_join_lower or "ω" in header_join_lower)
        and ("b_{e}" in header_join_lower or "b_e" in header_join_lower)
        and ("r_{e}" in header_join_lower or "r_e" in header_join_lower)
    )
    part2_cluster_decomposition_table = (
        ncols == 6
        and any("lw_current_term_signed_kpc" in c.lower() or "current term" in c.lower() for c in header)
        and any("sim_abs_offset_kpc" in c.lower() or "sim. abs. offset" in c.lower() for c in header)
    )
    # 条件分岐: `score_kind_table` を満たす経路を評価する。
    if score_kind_table:
        # 4.1.1「項目対応（節マップ + 判定変数）」向け:
        # score_kind 列を広げ、短い列（判定/対応節）を圧縮して可読性を確保する。
        widths = [0.150, 0.090, 0.250, 0.230, 0.120]
        colspec = "".join(r">{\raggedright\arraybackslash}p{" + f"{w}\\linewidth" + "}" for w in widths)
    # 条件分岐: 前段条件が不成立で、`parameter_policy_table` を追加評価する。
    elif part3_diff_summary_table:
        widths = [0.105, 0.085, 0.145, 0.155, 0.120, 0.110, 0.170]
        colspec = "".join(r">{\raggedright\arraybackslash}p{" + f"{w}\\linewidth" + "}" for w in widths)
    elif parameter_policy_table:
        # 3章「パラメータ運用区分（可動パラメータ許可リスト）」向け:
        # 代表パラメータ列を広げ、可否列を圧縮して長い変数名の折返しを減らす。
        widths = [0.110, 0.265, 0.205, 0.135, 0.215]
        colspec = "".join(r">{\raggedright\arraybackslash}p{" + f"{w}\\linewidth" + "}" for w in widths)
    elif part3_bell_frozen_table:
        widths = [0.105, 0.170, 0.115, 0.125, 0.080, 0.080, 0.060, 0.095]
        colspec = "".join(r">{\raggedright\arraybackslash}p{" + f"{w}\\linewidth" + "}" for w in widths)
    elif molecular_constants_table:
        # Part3 4.9.1「分子定数」表は読みやすさ優先で列幅を明示し、極小フォント化を避ける。
        widths = [0.105, 0.128, 0.128, 0.128, 0.128, 0.128, 0.128]
        colspec = "".join(r">{\raggedright\arraybackslash}p{" + f"{w}\\linewidth" + "}" for w in widths)
    elif part2_cluster_decomposition_table:
        widths = [0.125, 0.145, 0.145, 0.155, 0.155, 0.135]
        colspec = "".join(r">{\raggedright\arraybackslash}p{" + f"{w}\\linewidth" + "}" for w in widths)
    elif cosmology_ddr_result_table:
        widths = [0.180, 0.085, 0.110, 0.135, 0.115, 0.090, 0.135]
        colspec = "".join(r">{\raggedright\arraybackslash}p{" + f"{w}\\linewidth" + "}" for w in widths)
    else:
        width = max(0.08, min(0.42, round((0.97 / ncols) - 0.03, 3)))
        colspec = "".join(r">{\raggedright\arraybackslash}p{" + f"{width}\\linewidth" + "}" for _ in range(ncols))

    compact_table = ncols >= 4
    table_font = r"\normalsize"
    if part3_diff_summary_table or part3_born_result_table:
        table_font = r"\footnotesize"
    elif part3_bell_frozen_table or part3_bell_selection_sweep_table:
        # 4.2 ベルテストの sweep 結果表は tiny だと可読性が落ちるため一段上げる。
        table_font = r"\scriptsize"
    elif molecular_constants_table:
        # Part3 4.9.1「分子定数」表は、本文スクリーンショット可読性を優先して一段拡大する。
        table_font = r"\small"
    elif eht_kappa_fp_result_table:
        table_font = r"\footnotesize"
    elif eht_kappa_main_result_table or cosmology_ddr_result_table:
        table_font = r"\scriptsize"
    # 条件分岐: `ncols >= 7` を満たす経路を評価する。
    elif ncols >= 7:
        table_font = r"\tiny"
    # 条件分岐: 前段条件が不成立で、`ncols >= 5` を追加評価する。
    elif ncols >= 5:
        table_font = r"\scriptsize"
    # 条件分岐: 前段条件が不成立で、`ncols >= 4` を追加評価する。
    elif ncols >= 4:
        table_font = r"\footnotesize"

    compact_tabcolsep = r"\setlength{\tabcolsep}{2pt}"
    compact_arraystretch = r"\renewcommand{\arraystretch}{1.05}"
    if part3_diff_summary_table or part3_born_result_table:
        compact_tabcolsep = r"\setlength{\tabcolsep}{3pt}"
        compact_arraystretch = r"\renewcommand{\arraystretch}{1.12}"
    elif part3_bell_frozen_table:
        compact_tabcolsep = r"\setlength{\tabcolsep}{2.0pt}"
        compact_arraystretch = r"\renewcommand{\arraystretch}{1.08}"
    elif part3_bell_selection_sweep_table:
        compact_tabcolsep = r"\setlength{\tabcolsep}{2.4pt}"
        compact_arraystretch = r"\renewcommand{\arraystretch}{1.08}"
    elif molecular_constants_table:
        compact_tabcolsep = r"\setlength{\tabcolsep}{2.8pt}"
        compact_arraystretch = r"\renewcommand{\arraystretch}{1.12}"

    out: list[str] = []
    # 条件分岐: `compact_table` を満たす経路を評価する。
    if compact_table:
        out += [
            r"\begingroup",
            table_font,
            compact_tabcolsep,
            compact_arraystretch,
        ]

    header_row = " & ".join(_convert_inline(c) for c in header) + r" \\"
    out += [r"\begin{longtable}{" + colspec + "}"]
    normalized_caption = re.sub(r"[:：]\s*$", "", caption_text.strip())
    if normalized_caption:
        table_label = label_text.strip()
        if not table_label:
            table_label = "tab:" + _compact_label(_safe_label(normalized_caption), max_len=52)
        if profile == "part4_verification" and not table_label.startswith("p4:"):
            table_label = f"p4:{table_label}"
        out.append(r"\caption{" + _convert_inline(normalized_caption) + "}")
        out.append(r"\label{" + table_label + r"} \\")
    out += [r"\toprule"]
    out.append(header_row)
    out += [
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        header_row,
        r"\midrule",
        r"\endhead",
        r"\midrule",
        r"\multicolumn{" + str(ncols) + r"}{r}{\footnotesize 続きは次ページ} \\",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
    ]
    for row in rows:
        padded = row + [""] * (ncols - len(row))
        out.append(" & ".join(_convert_inline(c) for c in padded[:ncols]) + r" \\")

    out += [r"\end{longtable}"]
    # 条件分岐: `compact_table` を満たす経路を評価する。
    if compact_table:
        out.append(r"\endgroup")

    out.append("")
    return out


# 関数: `_split_protocol_triplet` の入出力契約と処理意図を定義する。
def _split_protocol_triplet(text: str) -> Optional[tuple[str, str, str]]:
    compact = " ".join(text.strip().split())
    m = re.match(
        r"^\*\*凍結\*\*：\s*(.+?)\s+\*\*棄却\*\*：\s*(.+?)\s+\*\*次\*\*：\s*(.+)$",
        compact,
        flags=re.DOTALL,
    )
    if not m:
        return None

    return m.group(1).strip(), m.group(2).strip(), m.group(3).strip()


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
    used_table_labels: dict[str, int] = {}

    paragraph: list[str] = []
    pending_table_caption = ""
    pending_table_label = ""
    in_code = False
    code_listing_open = False
    in_math = False
    numbered_display_math = profile == "paper"
    display_math_open = r"\begin{equation}" if numbered_display_math else r"\["
    display_math_close = r"\end{equation}" if numbered_display_math else r"\]"
    list_mode: Optional[str] = None  # "itemize" | "enumerate"
    chapter_started = False
    top_h1_seen = False
    appendix_started = False
    last_heading_effective_level: Optional[int] = None
    last_heading_source_index: Optional[int] = None
    i = 0

    # 関数: `flush_paragraph` の入出力契約と処理意図を定義する。
    def flush_paragraph() -> None:
        nonlocal paragraph
        # 条件分岐: `paragraph` を満たす経路を評価する。
        if paragraph:
            joined = " ".join(s.strip() for s in paragraph if s.strip())
            if profile == "part3_quantum":
                triplet = _split_protocol_triplet(joined)
                if triplet:
                    frozen_text, reject_text, next_text = triplet
                    out.extend([
                        r"\vspace{0.8em}",
                        r"\noindent\fbox{",
                        r"\begin{minipage}{0.97\linewidth}",
                        r"\textbf{凍結}：" + _convert_inline(frozen_text) + r"\\",
                        r"\textbf{棄却}：" + _convert_inline(reject_text) + r"\\",
                        r"\textbf{次}：" + _convert_inline(next_text),
                        r"\end{minipage}",
                        r"}",
                        r"\vspace{0.8em}",
                    ])
                    out.append("")
                    paragraph = []
                    return

            out.append(_convert_inline(joined))
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
        m_ul_preview = re.match(r"^\s*[-*]\s+(.+)$", line)
        m_ol_preview = re.match(r"^\s*\d+[.)]\s+(.+)$", line)

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
                out.append(display_math_close)
                out.append("")
                in_math = False
            # 条件分岐: 前段条件が不成立で、`stripped.endswith("$$")` を追加評価する。
            elif stripped.endswith("$$"):
                body_end = line.rsplit("$$", 1)[0].strip()
                # 条件分岐: `body_end` を満たす経路を評価する。
                if body_end:
                    out.append(body_end)

                out.append(display_math_close)
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

        # 箇条書き中の式や継続行は list を維持し、非継続行でのみ list を閉じる。
        if (
            list_mode
            and stripped
            and not m_ul_preview
            and not m_ol_preview
            and not line[:1].isspace()
            and not stripped.startswith("$$")
        ):
            close_list()

        # 条件分岐: `stripped in {"<!-- LATEX_CLEARPAGE -->", r"\clearpage", r"\newpage"}` を満たす経路を評価する。
        if stripped in {"<!-- LATEX_CLEARPAGE -->", r"\clearpage", r"\newpage"}:
            flush_paragraph()
            close_list()
            # Part2/Part3 は「章（section）直前のみ改ページ」を運用固定とするため、
            # 明示マーカーによる追加改ページは無効化する。
            if profile not in {"part2_astrophysics", "part3_quantum"}:
                out.append(r"\clearpage" if stripped == "<!-- LATEX_CLEARPAGE -->" else stripped)
                out.append("")
            i += 1
            continue

        # Generic HTML comments in markdown (e.g., sync markers) should not appear in PDF.
        if stripped.startswith("<!--") and stripped.endswith("-->"):
            flush_paragraph()
            close_list()
            marker_body = stripped[4:-3].strip()
            if marker_body:
                out.append("% " + marker_body)
            i += 1
            continue

        # 条件分岐: `stripped.startswith(">")` を満たす経路を評価する。
        if stripped.startswith(">"):
            flush_paragraph()
            close_list()
            quote_lines: list[str] = []
            while i < len(lines):
                raw_quote = lines[i].strip()
                if not raw_quote.startswith(">"):
                    break
                payload = re.sub(r"^>\s?", "", raw_quote).strip()
                if payload:
                    quote_lines.append(_convert_inline(payload))
                i += 1

            if quote_lines:
                out.append(r"\begin{quote}")
                out.extend(quote_lines)
                out.append(r"\end{quote}")
                out.append("")

            continue

        # 条件分岐: `stripped == "$$"` を満たす経路を評価する。

        if stripped == "$$":
            flush_paragraph()
            if not list_mode:
                close_list()
            out.append(display_math_open)
            in_math = True
            i += 1
            continue

        # 条件分岐: `stripped.startswith("$$") and stripped.endswith("$$") and len(stripped) > 4` を満たす経路を評価する。

        if stripped.startswith("$$") and stripped.endswith("$$") and len(stripped) > 4:
            flush_paragraph()
            if not list_mode:
                close_list()
            math_inline = stripped[2:-2].strip()
            out.append(display_math_open)
            # 条件分岐: `math_inline` を満たす経路を評価する。
            if math_inline:
                out.append(math_inline)

            out.append(display_math_close)
            out.append("")
            i += 1
            continue

        # 条件分岐: `stripped.startswith("$$") and len(stripped) > 2` を満たす経路を評価する。

        if stripped.startswith("$$") and len(stripped) > 2:
            flush_paragraph()
            if not list_mode:
                close_list()
            out.append(display_math_open)
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
                    profile=profile,
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
                    profile=profile,
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

        m_table_caption = re.match(
            r"^(?:Table|表)\s*\d*\s*[:：]\s*(.+?)\s*(?:\{#([A-Za-z0-9:_-]+)\})?\s*$",
            stripped,
            flags=re.IGNORECASE,
        )
        if m_table_caption:
            flush_paragraph()
            close_list()
            pending_table_caption = m_table_caption.group(1).strip()
            explicit_label = (m_table_caption.group(2) or "").strip()
            if explicit_label:
                pending_table_label = explicit_label
            else:
                pending_table_label = "tab:" + _compact_label(_safe_label(pending_table_caption), max_len=52)
            i += 1
            continue

        # 条件分岐: `stripped == ""` を満たす経路を評価する。

        if stripped == "":
            flush_paragraph()
            if not list_mode:
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

            table_caption = pending_table_caption
            table_label = pending_table_label
            if table_label:
                label_count = used_table_labels.get(table_label, 0) + 1
                used_table_labels[table_label] = label_count
                if label_count > 1:
                    table_label = f"{table_label}-{label_count}"

            out.extend(_render_table(block, caption_text=table_caption, label_text=table_label, profile=profile))
            pending_table_caption = ""
            pending_table_label = ""
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
                # Part4/Part5 固定: 「この文書の目的」の直前にのみ区切り線を入れる。
                if profile in {"part4_verification", "part5_future_predictions"}:
                    compact_title = re.sub(r"[\s\u3000\(\)（）\[\]【】<>＜＞:：._\-–—・,，、/]", "", title)
                    if compact_title in {"この文書の目的", "この文書目的"}:
                        out += [r"\medskip", r"\hrule", r"\medskip", ""]
                # 条件分岐: `_is_abstract_heading(title)` を満たす経路を評価する。
                if _is_abstract_heading(title):
                    out += [r"\medskip", r"\hrule", r"\medskip", ""]
                    out.append(r"\section*{" + _convert_inline(title) + "}")
                    out.append("")
                    chapter_started = True
                    last_heading_effective_level = effective_level
                    last_heading_source_index = i
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
                # Part1-4 共通運用: 節（subsection）の直前で改ページする。
                force_subsection_pagebreak = True
                if last_heading_effective_level in {1, 2} and last_heading_source_index is not None:
                    prose_line_count = _count_prose_lines_between_headings(
                        lines,
                        last_heading_source_index + 1,
                        i,
                    )
                    # 章/節の後の本文が短い場合（<=10行）は改ページしない。
                    if prose_line_count <= 10:
                        force_subsection_pagebreak = False

            # 条件分岐: `profile == "part2_astrophysics" and "項目対応（節マップ）" in title` を満たす経路を評価する。

            # 条件分岐: `profile == "part3_quantum" and "項目対応（節マップ）" in title` を満たす経路を評価する。

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
            out.append(rf"\{cmd}{{\texorpdfstring{{{heading_tex}}}{{{heading_pdf}}}}}")

            section_label = _build_section_label(raw_title, title, used_labels=used_labels, profile=profile)
            if profile == "part4_verification":
                out.append(rf"\label{{sec:p4:{section_label}}}")
            else:
                out.append(rf"\label{{sec:{section_label}}}")
            out.append("")
            last_heading_effective_level = effective_level
            last_heading_source_index = i
            i += 1
            continue

        # horizontal rule

        if re.match(r"^[-*_]{3,}\s*$", stripped):
            flush_paragraph()
            close_list()
            # Part1-4 共通運用: Markdown 水平線は本文へ出力しない。
            # 必要な区切り線は、明示ルール（例: Abstract の直前）でのみ挿入する。
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
                    profile=profile,
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

    body = "\n".join(out).strip() + "\n"
    body = _restore_literal_refs(body)
    return body


# 関数: `main` の入出力契約と処理意図を定義する。

def main(argv: Optional[Sequence[str]] = None) -> int:
    global _REFERENCE_KEYS, _REFERENCE_ORDER, _REFERENCE_TEXT, _USED_REFERENCE_KEYS

    ap = argparse.ArgumentParser(description="Generate LaTeX paper from markdown manuscript.")
    ap.add_argument(
        "--profile",
        choices=["paper", "part2_astrophysics", "part3_quantum", "part4_verification", "part5_future_predictions"],
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
        elif profile == "part4_verification":
            manuscript_md = root / "doc" / "paper" / "13_part4_verification.md"
        else:
            manuscript_md = root / "doc" / "paper" / "14_part5_future_predictions.md"

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
    _ensure_pmodel_version_style(root=root, outdir=outdir)

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
        elif profile == "part4_verification":
            out_name = "pmodel_paper_part4_verification.tex"
        else:
            out_name = "pmodel_paper_part5_future_predictions.tex"

    md_text = manuscript_md.read_text(encoding="utf-8", errors="replace")
    body = _markdown_to_latex(
        md_text,
        root=root,
        outdir=outdir,
        figures_dir=figures_dir,
        profile=profile,
    )
    body = _postprocess_latex_body(body)
    if profile == "part2_astrophysics":
        body = _apply_part2_selective_equation_numbering(body)
    elif profile == "part3_quantum":
        body = _apply_part3_selective_equation_numbering(body)

    bibliography_section = _render_bibliography_section(profile=profile)

    title_tex_map = {
        "paper": r"時間波ダイナミクスに基づく統一理論 (The P-model)\\[0.5em]Part I: 理論的基礎と写像原理",
        "part2_astrophysics": r"時間波ダイナミクスに基づく統一理論 (The P-model)\\[0.5em]Part II: 宇宙物理学および宇宙論的検証\\[1em]\large",
        "part3_quantum": r"時間波ダイナミクスに基づく統一理論 (The P-model)\\[0.5em]Part III: 微視的および量子的現象の再評価",
        "part4_verification": r"時間波ダイナミクスに基づく統一理論 (The P-model)\\[0.5em]Part IV: 再現性監査と未来の差分予測\\[1em]\large",
        "part5_future_predictions": r"時間波ダイナミクスに基づく統一理論 (The P-model)\\[0.5em]Part V: 未来への予測（Predictions for Future Observations）\\[1em]\large",
    }
    title_tex = title_tex_map.get(profile, r"P-model Paper")
    author_tex = (
        r"\author{" "\n"
        r"  Shunji Ogawa \\" "\n"
        r"  \vspace{0.2em}" "\n"
        r"  \small ENTERSYSTEM Co., Ltd." "\n"
        r"}"
    )
    figure_locale_tex = ""
    if profile == "paper":
        figure_locale_tex = r"\renewcommand{\figurename}{図}" "\n"
    elif profile == "part2_astrophysics":
        figure_locale_tex = (
            r"\renewcommand{\figurename}{図}" "\n"
            r"\renewcommand{\tablename}{表}" "\n"
        )
    elif profile == "part3_quantum":
        figure_locale_tex = r"\renewcommand{\figurename}{図}" "\n"
    elif profile == "part4_verification":
        figure_locale_tex = r"\renewcommand{\figurename}{図}" "\n"
    elif profile == "part5_future_predictions":
        figure_locale_tex = r"\renewcommand{\figurename}{図}" "\n"

    engine_tex = (
        r"\usepackage{fontspec}" "\n"
        r"\usepackage{luatexja}" "\n"
        r"% avoid lltjp-microtype warning under LuaTeX-ja" "\n"
    )

    cjk_wrap_begin = ""
    cjk_wrap_end = ""
    if profile != "paper":
        engine_tex = (
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
        )
        cjk_wrap_begin = r"\ifPDFTeX\begin{CJK}{UTF8}{min}\fi" + "\n"
        cjk_wrap_end = r"\ifPDFTeX\end{CJK}\fi" + "\n"

    lstset_tex = ""
    if profile != "paper":
        lstset_tex = (
            r"\lstset{breaklines=true,breakatwhitespace=false,columns=fullflexible,keepspaces=true,basicstyle=\ttfamily\small}"
            "\n"
        )

    tex = (
        r"% !TeX program = lualatex" "\n"
        r"\documentclass[11pt,a4paper]{article}" "\n"
        + engine_tex
        + r"\usepackage{geometry}" "\n"
        + r"\geometry{margin=20mm}" "\n"
        + r"\usepackage{hyperref}" "\n"
        + r"\usepackage{pmodel_version}" "\n"
        + r"\usepackage{graphicx}" "\n"
        + r"\usepackage{longtable}" "\n"
        + r"\usepackage{booktabs}" "\n"
        + r"\usepackage{array}" "\n"
        + r"\usepackage{enumitem}" "\n"
        + r"\usepackage{amsmath,amssymb}" "\n"
        + r"\usepackage{float}" "\n"
        + r"\usepackage{xcolor}" "\n"
        + r"\usepackage{listings}" "\n"
        + r"\usepackage{setspace}" "\n"
        + lstset_tex
        + r"\setstretch{1.1}" "\n"
        + r"\setlength{\parskip}{0.4em}" "\n"
        + r"\setlength{\parindent}{0pt}" "\n"
        + r"\setcounter{secnumdepth}{3}" "\n"
        + r"\urlstyle{same}" "\n\n"
        + figure_locale_tex
        + "\n"
        + r"\setlength{\tabcolsep}{3pt}" "\n"
        + r"\setlength{\LTleft}{0pt}" "\n"
        + r"\setlength{\LTright}{0pt}" "\n"
        + r"\setlength{\emergencystretch}{4em}" "\n"
        + "\n"
        + r"\graphicspath{{figures/}}" "\n\n"
        + r"\DeclareGraphicsExtensions{.pdf,.png,.jpg,.jpeg}" "\n\n"
        + r"% --- convenience macros (avoid undefined control sequences) ---" "\n"
        + r"\newcommand{\sigmaV}{\sigma_V}" "\n"
        + r"\newcommand{\Deltax}{\Delta x}" "\n"
        + r"\newcommand{\DeltaA}{\Delta A}" "\n"
        + r"\newcommand{\Deltat}{\Delta t}" "\n"
        + r"\newcommand{\Deltaz}{\Delta z}" "\n"
        + r"\newcommand{\DeltaAIC}{\Delta \mathrm{AIC}}" "\n\n"
        + r"\title{" + title_tex + "}\n"
        + author_tex + "\n"
        + r"\date{\PmodelFullDate}"
        + "\n\n"
        + r"\begin{document}" + "\n"
        + cjk_wrap_begin
        + r"\maketitle" + "\n\n"
        + body
        + bibliography_section
        + "\n"
        + cjk_wrap_end
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
