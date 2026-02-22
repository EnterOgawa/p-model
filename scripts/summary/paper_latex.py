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


_HEADING_PREFIX_RE = re.compile(r"^\s*\d{1,2}(?:\.\d{1,2})*(?:[.)：:]|\s)\s*")


def _strip_heading_prefix(title: str) -> str:
    t = title.strip()
    stripped = _HEADING_PREFIX_RE.sub("", t, count=1).strip()
    return stripped or t


def _is_abstract_heading(title: str) -> bool:
    compact = re.sub(r"[\s\u3000\(\)（）\[\]【】<>＜＞:：._\-–—・,，、/]", "", title).lower()
    return compact in {"abstract", "要旨", "要旨abstract", "abstract要旨"}


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


def _render_figure_block(
    *,
    raw_path: str,
    caption: str,
    root: Path,
    outdir: Path,
) -> list[str]:
    resolved_path, exists = _resolve_image_path(raw_path, root=root)
    if resolved_path.startswith("http://") or resolved_path.startswith("https://"):
        return [r"\noindent\href{" + _escape_tex(resolved_path) + "}{" + _convert_inline(caption or raw_path) + "}", ""]

    resolved_obj = Path(resolved_path)
    try:
        tex_rel = _normalize_tex_path(os.path.relpath(resolved_obj, outdir))
    except Exception:
        tex_rel = _normalize_tex_path(resolved_path)

    normalized_caption = caption.strip()
    normalized_caption = re.sub(r"[:：]\s*$", "", normalized_caption).strip() or Path(raw_path).name

    caption_text = _convert_inline(normalized_caption)
    if not exists:
        caption_text = _convert_inline(f"{normalized_caption} (missing file: {raw_path})")

    return [
        r"\begin{figure}[H]",
        r"\centering",
        r"\includegraphics[width=\linewidth]{" + _escape_tex(tex_rel) + "}",
        r"\caption{" + caption_text + "}",
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

_MATH_GREEK_OR_SYMBOL_RE = re.compile(r"[α-ωΑ-ΩΔΘΛΞΠΣΦΨΩ∇∂∫≠≤≥≈≃→⇒⇔∝∞×⋅·±≡∥⊥□]")
_PUNCT_ONLY_RE = re.compile(r"^[\s,，、。.:：;；!！?？()\[\]{}<>＜＞「」『』【】/／・\-+*|]+$")


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


def _normalize_inline_math_payload(code_text: str) -> str:
    normalized = code_text.strip()
    for src, dst in _SUPERSCRIPT_TO_ASCII.items():
        normalized = normalized.replace(src, dst)
    for src, dst in _SUBSCRIPT_TO_ASCII.items():
        normalized = normalized.replace(src, dst)
    for src, dst in _MATH_UNICODE_TO_LATEX.items():
        normalized = normalized.replace(src, dst)
    for src, dst in _GREEK_UNICODE_TO_LATEX.items():
        normalized = normalized.replace(src, dst)

    normalized = re.sub(r"\s+", " ", normalized)
    normalized = re.sub(r"([A-Za-z])\\mu\\nu", r"\1_{\\mu\\nu}", normalized)
    normalized = re.sub(r"([A-Za-z])\\mu", r"\1_{\\mu}", normalized)
    normalized = re.sub(r"\bP([0-9]+)\b", r"P_{\1}", normalized)
    normalized = re.sub(r"\bJ([0-9]+)\b", r"J_{\1}", normalized)
    normalized = re.sub(r"\bf\\sigma([0-9]+)\b", r"f\\sigma_{\1}", normalized)
    normalized = re.sub(r"\s*=\s*", "=", normalized)
    normalized = normalized.replace("<<", r"\ll")
    normalized = normalized.replace(">>", r"\gg")
    return normalized.strip()


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
        if _looks_like_math_code(payload):
            return make_token("$" + _normalize_inline_math_payload(payload) + "$")
        return make_token(r"\texttt{" + _escape_tex(payload) + r"}")

    text = re.sub(r"`([^`]+)`", repl_inline_code, text)
    # inline math
    text = re.sub(
        r"(?<!\\)\$(.+?)(?<!\\)\$",
        lambda m: make_token("$" + _normalize_inline_math_payload(m.group(1)) + "$"),
        text,
    )
    # links
    text = re.sub(
        r"\[([^\]]+)\]\(([^)]+)\)",
        lambda m: make_token(r"\href{" + _escape_tex(m.group(2)) + "}{" + _escape_tex(m.group(1)) + "}"),
        text,
    )
    # bold / italic
    text = re.sub(r"\*\*([^*]+)\*\*", lambda m: make_token(r"\textbf{" + _escape_tex(m.group(1)) + "}"), text)
    text = re.sub(r"\*([^*]+)\*", lambda m: make_token(r"\emph{" + _escape_tex(m.group(1)) + "}"), text)

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
    s = line.strip().strip("|")
    return [cell.strip() for cell in s.split("|")]


def _render_table(block_lines: list[str]) -> list[str]:
    if len(block_lines) < 2:
        return [_convert_inline(block_lines[0])] if block_lines else []
    header = _parse_table_row(block_lines[0])
    body_lines = block_lines[2:] if _is_table_separator(block_lines[1]) else block_lines[1:]
    rows = [_parse_table_row(line) for line in body_lines]
    ncols = max(1, len(header))
    width = max(0.08, min(0.45, round(0.95 / ncols, 3)))
    colspec = "".join(f"p{{{width}\\linewidth}}" for _ in range(ncols))

    out: list[str] = [r"\begin{longtable}{" + colspec + "}", r"\toprule"]
    out.append(" & ".join(_convert_inline(c) for c in header) + r" \\")
    out.append(r"\midrule")
    for row in rows:
        padded = row + [""] * (ncols - len(row))
        out.append(" & ".join(_convert_inline(c) for c in padded[:ncols]) + r" \\")
    out += [r"\bottomrule", r"\end{longtable}", ""]
    return out


def _markdown_to_latex(md_text: str, *, root: Path, outdir: Path) -> str:
    lines = md_text.splitlines()
    out: list[str] = []

    paragraph: list[str] = []
    in_code = False
    in_math = False
    list_mode: Optional[str] = None  # "itemize" | "enumerate"
    chapter_started = False
    top_h1_seen = False
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
                out.append(r"\end{verbatim}")
                out.append("")
                in_code = False
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
            out.append(r"\begin{verbatim}")
            in_code = True
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
            caption_text = Path(path_text).name
            if paragraph:
                last_line = re.sub(r"\s{2,}$", "", paragraph[-1]).strip()
                if re.match(r"^(図|Figure|Fig\.?)", last_line, flags=re.IGNORECASE):
                    paragraph = paragraph[:-1]
                    flush_paragraph()
                    caption_text = last_line
                else:
                    flush_paragraph()
            close_list()
            out.extend(_render_figure_block(raw_path=path_text, caption=caption_text, root=root, outdir=outdir))
            if inline_desc:
                out.append(_convert_inline(inline_desc))
                out.append("")
            i += 1
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
            title = _strip_heading_prefix(m_head.group(2).strip())
            if level == 1 and not top_h1_seen:
                top_h1_seen = True
                i += 1
                continue

            effective_level = 1 if level == 1 else max(1, level - 1)
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
            out.append(rf"\{cmd}{{{_convert_inline(title)}}}")
            out.append(rf"\label{{sec:{_safe_label(title)}}}")
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
            out.extend(_render_figure_block(raw_path=path, caption=(alt or path), root=root, outdir=outdir))
            i += 1
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
        out.append(r"\end{verbatim}")
    if in_math:
        out.append(r"\]")

    return "\n".join(out).strip() + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
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
    body = _markdown_to_latex(md_text, root=root, outdir=outdir)

    title_map = {
        "paper": "P-model Paper Part I",
        "part2_astrophysics": "P-model Paper Part II (Astrophysics)",
        "part3_quantum": "P-model Paper Part III (Quantum)",
        "part4_verification": "P-model Paper Part IV (Verification Materials)",
    }
    title = title_map.get(profile, "P-model Paper")

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
        r"\usepackage{setspace}" "\n"
        r"\setstretch{1.1}" "\n"
        r"\setlength{\parskip}{0.4em}" "\n"
        r"\setlength{\parindent}{0pt}" "\n"
        r"\setcounter{secnumdepth}{3}" "\n"
        r"\urlstyle{same}" "\n\n"
        + r"\title{" + _escape_tex(title) + "}\n"
        + r"\author{P-model Project}" + "\n"
        + r"\date{" + _escape_tex(datetime.now(timezone.utc).strftime("%Y-%m-%d UTC")) + "}\n\n"
        + r"\begin{document}" + "\n"
        + r"\ifPDFTeX\begin{CJK}{UTF8}{min}\fi" + "\n"
        + r"\maketitle" + "\n\n"
        + body
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
