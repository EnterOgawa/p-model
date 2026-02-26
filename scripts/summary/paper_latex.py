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
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.summary import worklog


def _repo_root() -> Path:
    return _ROOT


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


def _safe_label(text: str) -> str:
    t = text.lower()
    t = re.sub(r"[^a-z0-9]+", "-", t).strip("-")
    return t or "sec"


_GENERIC_SECTION_LABELS = {
    "sec",
    "section",
}


def _extract_heading_number(raw_title: str) -> str:
    m = re.match(r"^\s*(\d+(?:\.\d+)*)", raw_title)
    return m.group(1) if m else ""


def _section_label_hint(raw_title: str, stripped_title: str) -> str:
    raw_lower = raw_title.lower()
    stripped_lower = stripped_title.lower()
    merged = f"{raw_lower} {stripped_lower}"

    if "ベルテスト" in raw_title or "bell" in merged:
        return "bell-test"

    if "原子核" in raw_title or "nuclear" in merged:
        return "nuclear"

    if "原子・分子" in raw_title or ("atomic" in merged and "molecular" in merged):
        return "atomic-molecular"

    if "物性" in raw_title or "condensed" in merged:
        return "materials"

    if "統計力学" in raw_title or "熱力学" in raw_title or "thermo" in merged:
        return "stat-thermo"

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

    if "eht" in merged:
        return "eht"

    if "節マップ" in raw_title or "項目対応" in raw_title:
        return "section-map"

    if "検証サマリ" in raw_title or "scoreboard" in merged:
        return "validation-summary"

    return ""


def _build_section_label(
    raw_title: str,
    stripped_title: str,
    *,
    used_labels: dict[str, int],
) -> str:
    number = _extract_heading_number(raw_title)
    number_tag = number.replace(".", "-") if number else ""

    base = _section_label_hint(raw_title, stripped_title) or _safe_label(stripped_title)
    if base == "p":
        base = "tf-pfield"

    if base in _GENERIC_SECTION_LABELS:
        raw_base = _safe_label(raw_title)
        if raw_base not in _GENERIC_SECTION_LABELS:
            base = raw_base

    if base in _GENERIC_SECTION_LABELS:
        base = f"sec-{number_tag}" if number_tag else "sec-topic"

    candidate = f"{base}-s{number_tag}" if number_tag and not base.endswith(f"-s{number_tag}") else base
    n = used_labels.get(candidate, 0) + 1
    used_labels[candidate] = n
    return candidate if n == 1 else f"{candidate}-{n}"


_HEADING_PREFIX_RE = re.compile(r"^\s*\d{1,2}(?:\.\d{1,2})*(?:[.)：:]|\s)\s*")


def _strip_heading_prefix(title: str) -> str:
    t = title.strip()
    stripped = _HEADING_PREFIX_RE.sub("", t, count=1).strip()
    return stripped or t


def _is_abstract_heading(title: str) -> bool:
    compact = re.sub(r"[\s\u3000\(\)（）\[\]【】<>＜＞:：._\-–—・,，、/]", "", title).lower()
    return compact in {"abstract", "要旨", "要旨abstract", "abstract要旨"}


_HEADING_INLINE_MATH_RE = re.compile(r"\$(.+?)\$")
_HEADING_LATEX_CMD_RE = re.compile(r"\\[A-Za-z]+")


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


def _normalize_tex_path(path_text: str) -> str:
    return path_text.replace("\\", "/")


_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".pdf", ".svg", ".bmp", ".webp")


def _is_image_path(path_text: str) -> bool:
    normalized = _normalize_tex_path(path_text.strip())
    lowered = normalized.lower()
    return any(lowered.endswith(ext) for ext in _IMAGE_EXTS)


def _match_leading_image_line(line_text: str) -> Optional[tuple[str, str]]:
    s = line_text.strip()
    m = re.match(
        r"^(?:[-*+]\s+)?`?([^\s`]+\.(?:png|jpg|jpeg|pdf|svg|bmp|webp))`?(?:\s+(.*))?$",
        s,
        flags=re.IGNORECASE,
    )
    if not m:
        return None

    path_text = m.group(1).strip()
    if not _is_image_path(path_text):
        return None

    desc_text = (m.group(2) or "").strip()
    return path_text, desc_text


def _fallback_caption_from_path(raw_path: str) -> str:
    stem = Path(raw_path).stem
    normalized = stem.replace("__", " ").replace("_", " ").replace("-", " ")
    normalized = re.sub(r"\s+", " ", normalized).strip()
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


def _is_image_markdown_line(stripped: str) -> bool:
    if _match_leading_image_line(stripped):
        return True

    return bool(re.match(r"^!\[([^\]]*)\]\(([^)]+)\)\s*$", stripped))


def _extract_following_caption(lines: list[str], start_index: int) -> tuple[str, int]:
    j = start_index
    while j < len(lines):
        raw = lines[j]
        stripped = raw.strip()
        if not stripped:
            break

        if stripped.startswith("```") or stripped == "$$":
            break

        if re.match(r"^(#{1,6})\s+", stripped):
            break

        if _is_image_markdown_line(stripped):
            break

        if re.match(r"^\s*[-*]\s+(.+)$", raw) or re.match(r"^\s*\d+[.)]\s+(.+)$", raw):
            break

        if "|" in raw and (j + 1) < len(lines) and _is_table_separator(lines[j + 1]):
            break

        if _is_table_separator(raw):
            break

        candidate = re.sub(r"\s{2,}$", "", stripped).strip()
        if candidate:
            return candidate, (j - start_index + 1)

        j += 1

    return "", 0


def _resolve_image_path(raw_path: str, *, root: Path) -> tuple[str, bool]:
    normalized = _normalize_tex_path(raw_path.strip())
    if normalized.startswith("http://") or normalized.startswith("https://"):
        return normalized, False

    candidate_paths: list[Path] = []
    candidate_norms: set[str] = set()

    def add_candidate(path_obj: Path) -> None:
        key = str(path_obj.resolve()) if path_obj.is_absolute() else str(path_obj)
        if key in candidate_norms:
            return

        candidate_norms.add(key)
        candidate_paths.append(path_obj)

    path_obj = Path(normalized)
    if path_obj.is_absolute():
        add_candidate(path_obj)
    else:
        add_candidate(root / path_obj)

    if normalized.startswith("output/") and not normalized.startswith("output/public/") and not normalized.startswith("output/private/"):
        tail = normalized[len("output/") :]
        add_candidate(root / "output" / "public" / Path(tail))
        add_candidate(root / "output" / "private" / Path(tail))

    resolved_existing = next((candidate for candidate in candidate_paths if candidate.exists()), None)
    if resolved_existing is not None:
        return str(resolved_existing), True

    if candidate_paths:
        return str(candidate_paths[0]), False

    return normalized, False


_REFERENCE_KEYS: set[str] = set()
_REFERENCE_ORDER: list[str] = []
_REFERENCE_TEXT: dict[str, str] = {}
_USED_REFERENCE_KEYS: set[str] = set()


def _load_reference_entries(references_md: Path) -> tuple[list[str], dict[str, str]]:
    if not references_md.exists():
        return [], {}

    order: list[str] = []
    refs: dict[str, str] = {}
    in_internal_block = False
    for raw_line in references_md.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = raw_line.strip()
        if stripped == "<!-- INTERNAL_ONLY_START -->":
            in_internal_block = True
            continue

        if stripped == "<!-- INTERNAL_ONLY_END -->":
            in_internal_block = False
            continue

        if in_internal_block:
            continue

        match = re.match(r"^\s*-\s+\[([A-Za-z0-9][A-Za-z0-9_.:-]*)\]\s+(.+)$", raw_line)
        if not match:
            continue

        key = match.group(1).strip()
        text = match.group(2).strip()
        if key not in refs:
            order.append(key)

        refs[key] = text

    return order, refs


def _render_bibliography_section() -> str:
    if not _USED_REFERENCE_KEYS:
        return ""

    lines: list[str] = ["", r"\clearpage", r"\section*{References}", r"\begin{thebibliography}{99}"]
    ordered_used = [key for key in _REFERENCE_ORDER if key in _USED_REFERENCE_KEYS]
    for key in ordered_used:
        ref_text = _REFERENCE_TEXT.get(key, "").strip()
        if not ref_text:
            continue

        rendered = _convert_inline(ref_text)
        rendered = re.sub(r"\\texttt\{(https?://[^{}]+)\}", r"\\url{\1}", rendered)
        lines.append(r"\bibitem{" + key + "} " + rendered)

    lines += [r"\end{thebibliography}", ""]
    return "\n".join(lines)


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
    if resolved_path.startswith("http://") or resolved_path.startswith("https://"):
        return [r"\noindent\href{" + _escape_tex(resolved_path) + "}{" + _convert_inline(caption or raw_path) + "}", ""]

    resolved_obj = Path(resolved_path)
    tex_path = Path(raw_path).name or "missing_figure.png"

    if exists:
        try:
            source_key = str(resolved_obj.resolve())
        except Exception:
            source_key = str(resolved_obj)

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
    if not normalized_caption:
        normalized_caption = _fallback_caption_from_path(raw_path)

    caption_text = _convert_inline(normalized_caption)
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


def _looks_like_artifact_code(s: str) -> bool:
    candidate = s.strip()
    if not candidate:
        return False

    low = candidate.lower()
    if "://" in candidate:
        return True

    if re.match(r"^[A-Za-z]:[\\/]", candidate):
        return True

    if low.startswith(("output/", "scripts/", "data/", "doc/", "./", "../", ".\\", "..\\")):
        return True

    if _CODE_FILE_EXT_RE.search(low):
        return True

    if re.fullmatch(r"[A-Za-z0-9_.-]+/", candidate):
        return True

    if _MATH_GREEK_OR_SYMBOL_RE.search(candidate):
        return False

    if "%" in candidate:
        return True

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

    if "=" in candidate and " " in candidate and "\\" not in candidate and not re.search(r"[{}^]", candidate):
        lhs = candidate.split("=", 1)[0].strip()
        if len(lhs) >= 4 and re.search(r"[A-Za-z]", lhs):
            return True

    if candidate.count("_") >= 2 and "\\" not in candidate:
        return True

    if re.search(r"[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+){2,}", candidate):
        return True

    m_snake = re.fullmatch(r"[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+", candidate)
    if m_snake:
        parts = candidate.split("_")
        if len(parts) >= 3:
            return True

        if len(parts) == 2 and (len(parts[0]) > 1 or len(parts[1]) > 1):
            return True

    if re.match(r"^(?:--?)[A-Za-z0-9][A-Za-z0-9_.-]*(?:=[^\\s]+)?$", candidate):
        return True

    m_keyval = re.match(
        r"^(?P<lhs>[A-Za-z][A-Za-z0-9_]*(?:\.[A-Za-z0-9_]+)*)\s*=\s*(?P<rhs>[^=]+)$",
        candidate,
    )
    if m_keyval:
        lhs = m_keyval.group("lhs")
        rhs = m_keyval.group("rhs")
        lhs_low = lhs.lower()
        rhs_low = rhs.lower()
        if re.search(r"[\\α-ωΑ-ΩΔΘΛΞΠΣΦΨΩ^{}()|]", rhs):
            return False

        if re.search(r"[A-Z]", lhs) or len(lhs) <= 3:
            return False

        if lhs_low.count("_") >= 1 and len(lhs_low) >= 6:
            return True

        if lhs_low in {"event", "event_counter", "next", "source", "selected", "target", "without", "shift"}:
            return True

        if re.search(r"[A-Za-z0-9]+_[A-Za-z0-9_]+", rhs):
            return True

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


def _looks_like_math_code(s: str) -> bool:
    candidate = s.strip()
    if not candidate:
        return False

    if _looks_like_artifact_code(candidate):
        return False

    if _MATH_GREEK_OR_SYMBOL_RE.search(candidate):
        return True

    if re.search(r"\\[A-Za-z]+", candidate):
        return True

    if re.search(r"[A-Za-z][_^][A-Za-z0-9\\{(]", candidate):
        return True

    if re.search(r"[A-Za-z]\([A-Za-z0-9_,+\-*/ ]+\)", candidate):
        return True

    if " " not in candidate and re.search(r"[+\-*/]", candidate) and re.search(r"[A-Za-zα-ωΑ-Ω]", candidate):
        return True

    if candidate in {"ln", "exp", "sqrt()", "sin", "cos", "tan", "max", "min"}:
        return True

    if re.search(r"[=<>|]", candidate):
        return True

    if re.fullmatch(r"[A-Za-z](?:/[A-Za-z0-9_]+)+", candidate):
        return True

    if re.fullmatch(r"[A-Za-z][0-9]+", candidate):
        return True

    if re.fullmatch(r"[A-Za-z](?:_[A-Za-z0-9]+)?", candidate):
        return True

    if " " in candidate and _MATH_GREEK_OR_SYMBOL_RE.search(candidate):
        return True

    return False


def _format_subscript_token(sub: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9]", sub):
        return sub

    if "_" in sub:
        return r"\mathrm{" + sub.replace("_", r"\_") + "}"

    return r"\mathrm{" + sub + "}"


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


def _looks_like_physics_equation_code(s: str) -> bool:
    candidate = s.strip()
    if not candidate:
        return False

    low = candidate.lower()
    if "://" in candidate:
        return False

    if re.match(r"^[A-Za-z]:[\\/]", candidate):
        return False

    if low.startswith(("output/", "scripts/", "data/", "doc/", "./", "../", ".\\", "..\\")):
        return False

    if _CODE_FILE_EXT_RE.search(low):
        return False

    if not re.search(r"(=|<=|>=|<|>|≈|≃|≡|∝)", candidate):
        return False

    lhs = re.split(r"(?:<=|>=|=|<|>|≈|≃|≡|∝)", candidate, maxsplit=1)[0].strip()
    lhs = lhs.replace(r"\_", "_")
    lhs_plain = lhs
    if lhs_plain.startswith("|") and lhs_plain.endswith("|") and len(lhs_plain) >= 2:
        lhs_plain = lhs_plain[1:-1].strip()

    if _PHYSICS_SINGLE_LHS_RE.fullmatch(lhs):
        return True

    if _PHYSICS_ASCII_GREEK_TOKEN_RE.fullmatch(lhs):
        return True

    if _PHYSICS_SINGLE_LHS_RE.fullmatch(lhs_plain):
        return True

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

    if re.fullmatch(r"[A-Z][A-Za-z0-9]{0,4}(?:_[A-Za-z0-9]+)?(?:\([^()]*\))?", lhs):
        return True

    if re.fullmatch(r"[A-Z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)*(?:/[A-Za-z0-9_,]+)?(?:\([^()]*\))?", lhs):
        return True

    if _MATH_GREEK_OR_SYMBOL_RE.search(candidate) or re.search(r"\\[A-Za-z]+", candidate):
        return True

    return False


def _looks_like_physics_symbol_code(s: str) -> bool:
    candidate = s.strip().replace(r"\_", "_")
    if not candidate:
        return False

    if re.search(r"[=<>]", candidate):
        return False

    if _PHYSICS_SINGLE_LHS_RE.fullmatch(candidate):
        return True

    if _PHYSICS_ASCII_GREEK_TOKEN_RE.fullmatch(candidate):
        return True

    if re.fullmatch(r"[A-Z][A-Za-z0-9]{0,4}(?:_[A-Za-z0-9]+)?", candidate):
        return True

    return False


def _replace_plain_symbolic_tokens(text: str, make_token) -> str:
    def repl_unicode_greek_sub(match: re.Match[str]) -> str:
        sym = match.group("sym")
        sub = match.group("sub")
        sym_tex = _normalize_inline_math_payload(sym)
        sub_tex = _format_subscript_token(sub)
        return make_token(f"${sym_tex}_{{{sub_tex}}}$")

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


def _postprocess_latex_body(body: str) -> str:
    normalized = _normalize_math_command_spacing(body)
    normalized = re.sub(
        r"\\href\{(?!(?:https?|mailto):)([^{}]+?\.(?:html?|HTML?)(?:[?#][^{}]*)?)\}\{([^{}]*)\}",
        lambda m: m.group(2),
        normalized,
        flags=re.IGNORECASE,
    )

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
        if _looks_like_artifact_code(payload):
            return match.group(0)

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
    def _texttt_allowbreak(match: re.Match[str]) -> str:
        payload = match.group(1)
        if len(payload) < 28 or r"\_" not in payload:
            return match.group(0)

        return r"\texttt{" + payload.replace(r"\_", r"\_\allowbreak ") + "}"

    normalized = re.sub(r"\\texttt\{([^{}]+)\}", _texttt_allowbreak, normalized)
    return normalized


def _convert_inline(text: str) -> str:
    token_map: dict[str, str] = {}
    token_index = 0

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
        if not payload:
            return ""

        if _PUNCT_ONLY_RE.fullmatch(payload):
            return make_token(_escape_tex(payload))

        if re.search(r"[\u3040-\u30ff\u3400-\u9fff]", payload) and not _looks_like_artifact_code(payload):
            return make_token(_escape_tex(payload))

        if _looks_like_physics_equation_code(payload) or _looks_like_physics_symbol_code(payload):
            return make_token("$" + _normalize_inline_math_payload(payload) + "$")

        if _looks_like_math_code(payload):
            return make_token("$" + _normalize_inline_math_payload(payload) + "$")

        return make_token(r"\texttt{" + _escape_tex(payload) + r"}")

    text = re.sub(r"`([^`]+)`", repl_inline_code, text)
    # inline math
    def repl_inline_math(match: re.Match[str]) -> str:
        payload_raw = match.group(1)
        payload = payload_raw.strip()
        if not payload:
            return ""

        if _looks_like_artifact_code(payload) and not (
            _looks_like_physics_equation_code(payload) or _looks_like_physics_symbol_code(payload)
        ):
            return make_token(r"\texttt{" + _escape_tex(payload) + r"}")

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
        if no_scheme and (target_core.endswith(".html") or target_core.endswith(".htm")):
            return make_token(_escape_tex(label))

        return make_token(r"\href{" + _escape_tex(target) + "}{" + _escape_tex(label) + "}")

    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", repl_link, text)

    def repl_citation(match: re.Match[str]) -> str:
        keys = [k.strip() for k in re.split(r"\s*[,;]\s*", match.group("keys")) if k.strip()]
        if not keys:
            return match.group(0)

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
            if escaped_key in escaped:
                escaped = escaped.replace(escaped_key, rendered)
                changed = True

        if not changed:
            break

    return escaped


def _is_table_separator(line: str) -> bool:
    s = line.strip()
    if "|" not in s:
        return False

    core = s.replace("|", "").replace(":", "").replace(" ", "")
    return bool(core) and set(core) <= {"-"}


def _parse_table_row(line: str) -> list[str]:
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]

    if s.endswith("|"):
        s = s[:-1]

    cells: list[str] = []
    buf: list[str] = []
    in_code = False
    in_math = False
    escaped = False

    for ch in s:
        if escaped:
            buf.append(ch)
            escaped = False
            continue

        if ch == "\\":
            buf.append(ch)
            escaped = True
            continue

        if ch == "`" and not in_math:
            in_code = not in_code
            buf.append(ch)
            continue

        if ch == "$" and not in_code:
            in_math = not in_math
            buf.append(ch)
            continue

        if ch == "|" and not in_code and not in_math:
            cells.append("".join(buf).strip())
            buf = []
            continue

        buf.append(ch)

    cells.append("".join(buf).strip())
    return cells


def _render_table(block_lines: list[str]) -> list[str]:
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
    if ncols >= 7:
        table_font = r"\tiny"
    elif ncols >= 5:
        table_font = r"\scriptsize"
    elif ncols >= 4:
        table_font = r"\footnotesize"

    out: list[str] = []
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
    if compact_table:
        out.append(r"\endgroup")

    out.append("")
    return out


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

    def flush_paragraph() -> None:
        nonlocal paragraph
        if paragraph:
            out.append(_convert_inline(" ".join(s.strip() for s in paragraph if s.strip())))
            out.append("")
            paragraph = []

    def close_list() -> None:
        nonlocal list_mode
        if list_mode:
            out.append(r"\end{" + list_mode + "}")
            out.append("")
            list_mode = None

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if in_code:
            if stripped.startswith("```"):
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

        if in_math:
            if stripped == "$$":
                out.append(r"\]")
                out.append("")
                in_math = False
            elif stripped.endswith("$$"):
                body_end = line.rsplit("$$", 1)[0].strip()
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

        if stripped == "$$":
            flush_paragraph()
            close_list()
            out.append(r"\[")
            in_math = True
            i += 1
            continue

        if stripped.startswith("$$") and stripped.endswith("$$") and len(stripped) > 4:
            flush_paragraph()
            close_list()
            math_inline = stripped[2:-2].strip()
            out.append(r"\[")
            if math_inline:
                out.append(math_inline)

            out.append(r"\]")
            out.append("")
            i += 1
            continue

        if stripped.startswith("$$") and len(stripped) > 2:
            flush_paragraph()
            close_list()
            out.append(r"\[")
            body_start = line.split("$$", 1)[1].strip()
            if body_start:
                out.append(body_start)

            in_math = True
            i += 1
            continue

        m_leading_image = _match_leading_image_line(stripped)
        if m_leading_image:
            path_text, inline_desc = m_leading_image
            caption_text = ""
            consumed_after_caption = 0
            if paragraph:
                last_line = re.sub(r"\s{2,}$", "", paragraph[-1]).strip()
                if re.match(r"^(図|Figure|Fig\.?)", last_line, flags=re.IGNORECASE):
                    paragraph = paragraph[:-1]
                    flush_paragraph()
                    caption_text = last_line
                else:
                    flush_paragraph()

            if not caption_text and inline_desc:
                caption_text = inline_desc

            if not caption_text:
                next_caption, consumed = _extract_following_caption(lines, i + 1)
                if next_caption:
                    caption_text = next_caption
                    consumed_after_caption = consumed

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
                if lines[i].strip() == "":
                    break

                if "|" not in lines[i]:
                    break

                block.append(lines[i])
                i += 1

            out.extend(_render_table(block))
            continue

        # headings

        m_head = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        if m_head:
            flush_paragraph()
            close_list()
            level = len(m_head.group(1))
            raw_title = m_head.group(2).strip()
            title = _strip_heading_prefix(raw_title)
            heading_number = _extract_heading_number(raw_title)
            if level == 1 and not top_h1_seen:
                top_h1_seen = True
                i += 1
                continue

            effective_level = 1 if level == 1 else max(1, level - 1)
            if effective_level == 1 and title.startswith("付録"):
                if not appendix_started:
                    out.append(r"\appendix")
                    out.append("")
                    appendix_started = True

                appendix_title = re.sub(r"^付録\s*[A-Za-zＡ-Ｚ0-9一二三四五六七八九十]*\s*[\.．:：]?\s*", "", title).strip()
                if appendix_title:
                    title = appendix_title

            if effective_level == 1:
                if _is_abstract_heading(title):
                    out.append(r"\section*{" + _convert_inline(title) + "}")
                    out.append("")
                    chapter_started = True
                    i += 1
                    continue

                if chapter_started:
                    out.append(r"\clearpage")
                    out.append("")

                chapter_started = True

            if effective_level == 1:
                cmd = "section"
            elif effective_level == 2:
                cmd = "subsection"
            elif effective_level == 3:
                cmd = "subsubsection"
            elif effective_level == 4:
                cmd = "paragraph"
            elif effective_level == 5:
                cmd = "subparagraph"
            else:
                out.append(r"\textbf{" + _convert_inline(title) + "}")
                out.append("")
                i += 1
                continue

            force_subsection_pagebreak = False
            force_heading_pagebreak = False
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

            if profile == "part2_astrophysics" and "項目対応（節マップ）" in title:
                force_heading_pagebreak = True

            if profile == "part3_quantum" and "項目対応（節マップ）" in title:
                force_heading_pagebreak = True

            if profile == "part3_quantum":
                if heading_number in {"4.10.2", "4.10.3", "4.11.2", "4.11.3"}:
                    force_heading_pagebreak = True
                elif ("黒体放射の基準量" in title) or ("黒体：エントロピーと第2法則整合" in title):
                    force_heading_pagebreak = True

            if force_subsection_pagebreak or force_heading_pagebreak:
                # If a markdown horizontal rule was emitted just before this heading,
                # remove it so the heading itself can start at the very top of the page.
                while out and out[-1] == "":
                    out.pop()

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
        if m_img:
            flush_paragraph()
            close_list()
            alt = m_img.group(1).strip()
            path = m_img.group(2).strip()
            caption_text = alt
            consumed_after_caption = 0
            if not caption_text:
                next_caption, consumed = _extract_following_caption(lines, i + 1)
                if next_caption:
                    caption_text = next_caption
                    consumed_after_caption = consumed

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
                if q:
                    out.append(_convert_inline(q) + r"\\")

            out.append(r"\end{quote}")
            out.append("")
            continue

        # lists

        m_ul = re.match(r"^\s*[-*]\s+(.+)$", line)
        if m_ul:
            flush_paragraph()
            if list_mode != "itemize":
                close_list()
                out.append(r"\begin{itemize}[leftmargin=2em]")
                list_mode = "itemize"

            out.append(r"\item " + _convert_inline(m_ul.group(1).strip()))
            i += 1
            continue

        m_ol = re.match(r"^\s*\d+[.)]\s+(.+)$", line)
        if m_ol:
            flush_paragraph()
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

    if in_code:
        if code_listing_open:
            out.append(r"\end{lstlisting}")
        else:
            out.append(r"\end{verbatim}")

    if in_math:
        out.append(r"\]")

    return "\n".join(out).strip() + "\n"


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

    if args.manuscript:
        manuscript_md = Path(args.manuscript)
    else:
        if profile == "paper":
            manuscript_md = root / "doc" / "paper" / "10_part1_core_theory.md"
        elif profile == "part2_astrophysics":
            manuscript_md = root / "doc" / "paper" / "11_part2_astrophysics.md"
        elif profile == "part3_quantum":
            manuscript_md = root / "doc" / "paper" / "12_part3_quantum.md"
        else:
            manuscript_md = root / "doc" / "paper" / "13_part4_verification.md"

    if not manuscript_md.exists():
        print(f"[error] manuscript not found: {manuscript_md}")
        return 1

    if args.outdir:
        outdir = Path(args.outdir)
    else:
        outdir = root / "output" / "private" / "summary"

    outdir.mkdir(parents=True, exist_ok=True)
    figures_dir = outdir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    if args.out_name:
        out_name = str(args.out_name)
    else:
        if profile == "paper":
            out_name = "pmodel_paper.tex"
        elif profile == "part2_astrophysics":
            out_name = "pmodel_paper_part2_astrophysics.tex"
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


if __name__ == "__main__":
    raise SystemExit(main())
