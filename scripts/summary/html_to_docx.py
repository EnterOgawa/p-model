#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
html_to_docx.py

Convert a local HTML file to DOCX (for sharing/editing) using Microsoft Word automation.

Requirements:
  - Windows
  - Microsoft Word installed (COM: Word.Application)

Outputs:
  - output DOCX file (path given by --out)
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import os
import re
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from typing import List, Optional, Sequence, Tuple
from urllib.parse import unquote, urlparse

_ROOT = Path(__file__).resolve().parents[2]
# 条件分岐: `str(_ROOT) not in sys.path` を満たす経路を評価する。
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.summary import paper_html as _paper_html, worklog


def _repo_root() -> Path:
    return _ROOT


def _mm_to_points(mm: float) -> float:
    return float(mm) * 72.0 / 25.4


def _choose_tmp_docx_path(target: Path) -> Path:
    """
    Pick a writable temporary DOCX path next to the target.

    Rationale:
    - The target DOCX may be open/locked by Word. In that case, we still want to produce an updated
      DOCX (as __tmp) rather than failing the whole build.
    - We keep the name stable (stem__tmp.docx) when possible to make it easy for users to find.
    """

    def _candidate(tag: str) -> Path:
        return target.with_name(f"{target.stem}{tag}{target.suffix}")

    # Prefer a stable name first.

    for tag in ["__tmp", "__tmp2", "__tmp3", "__tmp4", f"__tmp_{int(time.time())}"]:
        cand = _candidate(tag)
        # 条件分岐: `not cand.exists()` を満たす経路を評価する。
        if not cand.exists():
            return cand

        try:
            cand.unlink()
            return cand
        except Exception:
            continue
    # Last resort: PID-tagged path.

    return _candidate(f"__tmp_{os.getpid()}_{int(time.time())}")


def _promote_tmp_docx(tmp_path: Path, target_path: Path) -> Tuple[Path, str]:
    """
    Try to replace target with tmp. If target is locked/open, keep tmp and return a warning message.
    """
    try:
        # 条件分岐: `tmp_path.resolve() == target_path.resolve()` を満たす経路を評価する。
        if tmp_path.resolve() == target_path.resolve():
            return target_path, ""
    except Exception:
        pass

    try:
        tmp_path.replace(target_path)
        return target_path, ""
    except Exception as e:
        # If the target is open in Word (and thus locked), try to update it in-place via the running Word instance.
        # This makes "build_materials.bat" able to refresh the canonical filename even when the DOCX is open.
        try:
            updated, msg = _try_update_open_word_docx_from_tmp(target_path=target_path, tmp_path=tmp_path)
            # 条件分岐: `updated` を満たす経路を評価する。
            if updated:
                try:
                    tmp_path.unlink()
                except Exception:
                    pass

                return target_path, msg
        except Exception:
            pass

        return tmp_path, (
            f"Output DOCX is locked/open; wrote: {tmp_path} "
            f"(close it and re-run to overwrite: {target_path}) ({e})"
        )


def _try_update_open_word_docx_from_tmp(*, target_path: Path, tmp_path: Path) -> Tuple[bool, str]:
    """
    If `target_path` is currently open in Microsoft Word (thus file-locked),
    replace its document content with the contents of `tmp_path` and save.

    Returns (updated, message).
    """
    # 条件分岐: `os.name != "nt"` を満たす経路を評価する。
    if os.name != "nt":
        return False, "non-Windows platform"

    try:
        import win32com.client  # type: ignore
    except Exception as e:
        return False, f"pywin32 (win32com) not available: {e}"

    try:
        word = win32com.client.GetActiveObject("Word.Application")
    except Exception as e:
        return False, f"no running Word instance: {e}"

    target_doc = None
    try:
        count = int(word.Documents.Count)
    except Exception as e:
        return False, f"cannot enumerate Word documents: {e}"

    target_norm = None
    try:
        target_norm = str(target_path.resolve()).lower()
    except Exception:
        target_norm = str(target_path).lower()

    for i in range(1, count + 1):
        try:
            doc = word.Documents(i)
            full = str(doc.FullName)
        except Exception:
            continue

        try:
            full_norm = str(Path(full).resolve()).lower()
        except Exception:
            full_norm = full.lower()

        # 条件分岐: `full_norm == target_norm` を満たす経路を評価する。

        if full_norm == target_norm:
            target_doc = doc
            break

    # 条件分岐: `target_doc is None` を満たす経路を評価する。

    if target_doc is None:
        return False, "target DOCX is not open in Word"

    try:
        # 条件分岐: `bool(getattr(target_doc, "ReadOnly", False))` を満たす経路を評価する。
        if bool(getattr(target_doc, "ReadOnly", False)):
            return False, "target DOCX is open read-only in Word"
    except Exception:
        pass

    # Best-effort: suppress prompts during automation, but restore the user's setting.

    prev_alerts = None
    try:
        prev_alerts = getattr(word, "DisplayAlerts")
        # 0 = wdAlertsNone
        setattr(word, "DisplayAlerts", 0)
    except Exception:
        prev_alerts = None

    prev_screen = None
    try:
        prev_screen = getattr(word, "ScreenUpdating")
        setattr(word, "ScreenUpdating", False)
    except Exception:
        prev_screen = None

    try:
        # Replace document body with the freshly built tmp DOCX.
        try:
            rng = target_doc.Range(0, target_doc.Content.End)
            rng.Delete()
        except Exception:
            pass

        try:
            target_doc.Range(0, 0).InsertFile(str(tmp_path))
        except Exception as e:
            return False, f"InsertFile failed: {e}"

        try:
            target_doc.Save()
        except Exception as e:
            return False, f"Save failed: {e}"
    finally:
        try:
            # 条件分岐: `prev_screen is not None` を満たす経路を評価する。
            if prev_screen is not None:
                setattr(word, "ScreenUpdating", prev_screen)
        except Exception:
            pass

        try:
            # 条件分岐: `prev_alerts is not None` を満たす経路を評価する。
            if prev_alerts is not None:
                setattr(word, "DisplayAlerts", prev_alerts)
        except Exception:
            pass

    return True, "updated open Word document in-place"


_MATH_BLOCK_RE = re.compile(r"\$\$(.+?)\$\$", flags=re.DOTALL)
_INLINE_MATH_RE = re.compile(r"(?<!\\)(?<!\$)\$(?!\$)([^$\n]+?)(?<!\\)\$(?!\$)")
_IMG_SRC_RE = re.compile(r'(<img\b[^>]*\bsrc\s*=\s*)(["\'])([^"\']+)\2', flags=re.IGNORECASE)
_IMG_TAG_RE = re.compile(r"<img\b[^>]*>", flags=re.IGNORECASE | re.DOTALL)
_ATTR_RE = re.compile(r'([A-Za-z_:][A-Za-z0-9_.:-]*)\s*=\s*(["\'])(.*?)\2', flags=re.DOTALL)
_INTERNAL_BLOCK_RE = re.compile(
    r"<!--\s*INTERNAL_ONLY_START\s*-->.*?<!--\s*INTERNAL_ONLY_END\s*-->",
    flags=re.DOTALL,
)


def _mime_from_ext(ext: str) -> str:
    e = (ext or "").lower()
    # 条件分岐: `e == ".png"` を満たす経路を評価する。
    if e == ".png":
        return "image/png"

    # 条件分岐: `e in (".jpg", ".jpeg")` を満たす経路を評価する。

    if e in (".jpg", ".jpeg"):
        return "image/jpeg"

    # 条件分岐: `e == ".gif"` を満たす経路を評価する。

    if e == ".gif":
        return "image/gif"

    # 条件分岐: `e == ".bmp"` を満たす経路を評価する。

    if e == ".bmp":
        return "image/bmp"

    # 条件分岐: `e == ".svg"` を満たす経路を評価する。

    if e == ".svg":
        return "image/svg+xml"

    return "application/octet-stream"


def _resolve_local_image_path(html_path: Path, src: str) -> Optional[Path]:
    s = (src or "").strip()
    # 条件分岐: `not s` を満たす経路を評価する。
    if not s:
        return None

    s_low = s.lower()
    # 条件分岐: `s_low.startswith("data:")` を満たす経路を評価する。
    if s_low.startswith("data:"):
        return None

    # 条件分岐: `s_low.startswith("http:") or s_low.startswith("https:")` を満たす経路を評価する。

    if s_low.startswith("http:") or s_low.startswith("https:"):
        return None

    # 条件分岐: `s_low.startswith("file:")` を満たす経路を評価する。

    if s_low.startswith("file:"):
        try:
            u = urlparse(s)
            p = unquote(u.path)
            # file:///C:/... comes through as /C:/...
            if p.startswith("/") and len(p) >= 3 and p[2] == ":":
                p = p[1:]

            return Path(p)
        except Exception:
            return None

    p = unquote(s.split("?", 1)[0].split("#", 1)[0])
    # 条件分岐: `not p` を満たす経路を評価する。
    if not p:
        return None

    try:
        return (html_path.parent / p).resolve()
    except Exception:
        return None


def _inline_local_images_for_word(html_path: Path) -> Tuple[Path, int, Optional[Path]]:
    """
    Word tends to link (not embed) images referenced by relative paths in HTML.
    To make the resulting DOCX self-contained, inline local images as data URIs.

    Returns (path_to_open, n_inlined, temp_path_to_delete).
    """
    html = html_path.read_text(encoding="utf-8", errors="replace")

    n_inlined = 0

    def _repl(m: re.Match[str]) -> str:
        nonlocal n_inlined
        prefix, quote, src = m.group(1), m.group(2), m.group(3)
        img_path = _resolve_local_image_path(html_path, src)
        # 条件分岐: `img_path is None or not img_path.exists() or not img_path.is_file()` を満たす経路を評価する。
        if img_path is None or not img_path.exists() or not img_path.is_file():
            return m.group(0)

        try:
            data = img_path.read_bytes()
        except Exception:
            return m.group(0)

        mime = _mime_from_ext(img_path.suffix)
        b64 = base64.b64encode(data).decode("ascii")
        n_inlined += 1
        return f"{prefix}{quote}data:{mime};base64,{b64}{quote}"

    html2 = _IMG_SRC_RE.sub(_repl, html)
    # 条件分岐: `n_inlined <= 0 or html2 == html` を満たす経路を評価する。
    if n_inlined <= 0 or html2 == html:
        return html_path, 0, None

    tmp_dir = Path(tempfile.gettempdir()) / "waveP_html_to_docx"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / f"{html_path.stem}_inlined_{int(time.time())}.html"
    tmp_path.write_text(html2, encoding="utf-8")
    return tmp_path, n_inlined, tmp_path


def _normalize_latex(latex: str) -> str:
    # Same normalization used in scripts/summary/paper_html.py for equation PNG rendering.
    latex_norm = latex.strip().replace("\r\n", "\n")
    latex_norm = latex_norm.replace("\\\\", "\\")
    latex_norm = (
        latex_norm.replace("\\lvert", "|")
        .replace("\\rvert", "|")
        .replace("\\lVert", "\\|")
        .replace("\\rVert", "\\|")
    )
    latex_norm = " ".join(latex_norm.split())
    return latex_norm


def _extract_math_blocks(md_text: str) -> List[str]:
    out: List[str] = []
    for m in _MATH_BLOCK_RE.finditer(md_text):
        s = (m.group(1) or "").strip()
        # 条件分岐: `not s` を満たす経路を評価する。
        if not s:
            continue

        out.append(_normalize_latex(s))

    return out


def _extract_inline_math(md_text: str) -> List[str]:
    out: List[str] = []

    def _looks_like_inline_math(expr: str) -> bool:
        s = (expr or "").strip()
        # 条件分岐: `not s` を満たす経路を評価する。
        if not s:
            return False

        # 条件分岐: `re.fullmatch(r"[0-9]+(?:[.,][0-9]+)?%?", s)` を満たす経路を評価する。

        if re.fullmatch(r"[0-9]+(?:[.,][0-9]+)?%?", s):
            return False

        return True

    for m in _INLINE_MATH_RE.finditer(md_text):
        s = (m.group(1) or "").strip()
        # 条件分岐: `not _looks_like_inline_math(s)` を満たす経路を評価する。
        if not _looks_like_inline_math(s):
            continue

        out.append(_normalize_latex(s))

    return out


def _extract_latex_from_html_images(*, html_path: Path, alt_text: str) -> List[str]:
    out: List[str] = []
    # 条件分岐: `not html_path.exists()` を満たす経路を評価する。
    if not html_path.exists():
        return out

    try:
        text = html_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return out

    for m in _IMG_TAG_RE.finditer(text):
        tag = m.group(0)
        attrs: dict[str, str] = {}
        for am in _ATTR_RE.finditer(tag):
            k = (am.group(1) or "").strip().lower()
            v = html.unescape((am.group(3) or "").strip())
            attrs[k] = v

        # 条件分岐: `attrs.get("alt") != alt_text` を満たす経路を評価する。

        if attrs.get("alt") != alt_text:
            continue

        latex = (attrs.get("data-latex") or "").strip()
        # 条件分岐: `not latex` を満たす経路を評価する。
        if not latex:
            continue

        out.append(_normalize_latex(latex))

    return out


def _strip_internal_blocks_for_publish(md_text: str) -> str:
    return _INTERNAL_BLOCK_RE.sub("", md_text)


def _default_mml2omml_xsl() -> Optional[Path]:
    # Microsoft Word ships MathML→OMML converter XSL with Office installation.
    candidates = [
        Path(r"C:\Program Files\Microsoft Office\root\Office16\MML2OMML.XSL"),
        Path(r"C:\Program Files (x86)\Microsoft Office\root\Office16\MML2OMML.XSL"),
        Path(r"C:\Program Files\Microsoft Office\Office16\MML2OMML.XSL"),
        Path(r"C:\Program Files (x86)\Microsoft Office\Office16\MML2OMML.XSL"),
    ]
    for p in candidates:
        # 条件分岐: `p.exists()` を満たす経路を評価する。
        if p.exists():
            return p

    return None


def _replace_equation_images_with_word_equations(
    *,
    root: Path,
    out_dir: Path,
    docx_path: Path,
    mml2omml_xsl: Optional[Path],
    alt_text: str,
    equation_kind: str = "block",
    html_source: Optional[Path] = None,
) -> Tuple[int, int]:
    """
    Post-process the generated DOCX:
    - Find equation PNGs inserted from paper HTML (identified by descr/alt_text).
    - Replace each with native Word equation (OMML) converted from LaTeX blocks in doc/paper/*.md.

    Returns (n_equations_found_in_docx, n_equations_replaced).
    """
    try:
        from latex2mathml.converter import convert  # type: ignore
    except Exception as e:
        raise RuntimeError(f"latex2mathml is required for native equations: {e}")

    # 条件分岐: `not mml2omml_xsl or not mml2omml_xsl.exists()` を満たす経路を評価する。

    if not mml2omml_xsl or not mml2omml_xsl.exists():
        raise RuntimeError("MML2OMML.XSL not found (Microsoft Word/Office installation required).")

    try:
        from lxml import etree  # type: ignore
    except Exception as e:
        raise RuntimeError(f"lxml is required for native equations: {e}")

    kind = (equation_kind or "block").strip().lower()
    # 条件分岐: `kind not in ("block", "inline")` を満たす経路を評価する。
    if kind not in ("block", "inline"):
        raise RuntimeError(f"unknown equation_kind: {equation_kind}")

    def _extract_math(paths: List[Path]) -> List[str]:
        items: List[str] = []
        for p in paths:
            # 条件分岐: `not p.exists()` を満たす経路を評価する。
            if not p.exists():
                continue

            try:
                t = p.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue

            t = _strip_internal_blocks_for_publish(t)
            try:
                t = _paper_html._normalize_markdown_for_tex_docx_parity(t)
            except Exception:
                pass

            # 条件分岐: `kind == "inline"` を満たす経路を評価する。

            if kind == "inline":
                items.extend(_extract_inline_math(t))
            else:
                items.extend(_extract_math_blocks(t))

        return items

    transform = etree.XSLT(etree.parse(str(mml2omml_xsl)))

    ns = {
        "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
        "wp": "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
        "m": "http://schemas.openxmlformats.org/officeDocument/2006/math",
    }

    with zipfile.ZipFile(docx_path, "r") as zin:
        xml_bytes = zin.read("word/document.xml")

    doc = etree.fromstring(xml_bytes)
    # 条件分岐: `kind == "inline"` を満たす経路を評価する。
    if kind == "inline":
        targets = doc.xpath(f'.//w:r[w:drawing//wp:docPr[@descr=\"{alt_text}\"]]', namespaces=ns)
    else:
        targets = doc.xpath(f'.//w:p[.//wp:docPr[@descr=\"{alt_text}\"]]', namespaces=ns)

    n_found = len(targets)
    # 条件分岐: `n_found == 0` を満たす経路を評価する。
    if n_found == 0:
        return 0, 0

    # Build candidate markdown source sets (Part I/II/III/Part IV + legacy), and pick the one
    # whose LaTeX block count matches the DOCX equation-image count. This keeps the mapping stable
    # even when manuscripts are split/renamed.

    sources_md = root / "doc" / "paper" / "20_data_sources.md"
    refs_md = root / "doc" / "paper" / "30_references.md"
    candidates: List[Tuple[str, List[Path]]] = [
        (
            "part1_core",
            [
                root / "doc" / "paper" / "10_part1_core_theory.md",
                root / "doc" / "paper" / "05_definitions.md",
                sources_md,
                refs_md,
            ],
        ),
        (
            "part2_astrophysics",
            [
                out_dir / "paper_table1_results.md",
                root / "doc" / "paper" / "11_part2_astrophysics.md",
                root / "doc" / "paper" / "07_llr_appendix.md",
                sources_md,
                refs_md,
            ],
        ),
        (
            "part3_quantum",
            [
                out_dir / "paper_table1_quantum_results.md",
                root / "doc" / "paper" / "12_part3_quantum.md",
                sources_md,
                refs_md,
            ],
        ),
        (
            "part4_verification",
            [
                root / "doc" / "paper" / "13_part4_verification.md",
                sources_md,
                refs_md,
            ],
        ),
        (
            "legacy_manuscript",
            [
                out_dir / "paper_table1_results.md",
                root / "doc" / "paper" / "05_definitions.md",
                root / "doc" / "paper" / "06_uncertainty.md",
                root / "doc" / "paper" / "10_manuscript.md",
                sources_md,
                refs_md,
            ],
        ),
    ]

    html_latex_items: List[str] = []
    # 条件分岐: `html_source is not None` を満たす経路を評価する。
    if html_source is not None:
        try:
            html_latex_items = _extract_latex_from_html_images(html_path=html_source, alt_text=alt_text)
        except Exception:
            html_latex_items = []

    # 条件分岐: `html_latex_items and len(html_latex_items) == n_found` を満たす経路を評価する。

    if html_latex_items and len(html_latex_items) == n_found:
        matches: List[Tuple[str, List[Path], List[str]]] = [("html_data_latex", [], html_latex_items)]
    else:
        matches = []

    # 条件分岐: `not matches` を満たす経路を評価する。

    if not matches:
        for label, paths in candidates:
            items = _extract_math(paths)
            # 条件分岐: `len(items) == n_found` を満たす経路を評価する。
            if len(items) == n_found:
                matches.append((label, paths, items))

    # 条件分岐: `not matches` を満たす経路を評価する。

    if not matches:
        # Provide a helpful diagnostic (counts per candidate).
        counts = []
        # 条件分岐: `html_latex_items` を満たす経路を評価する。
        if html_latex_items:
            counts.append(f"html_data_latex={len(html_latex_items)}")

        for label, paths in candidates:
            items = _extract_math(paths)
            counts.append(f"{label}={len(items)}")

        raise RuntimeError(f"equation count mismatch: docx={n_found}, candidates: " + ", ".join(counts))

    # If multiple match (unlikely), prefer non-legacy.

    matches.sort(key=lambda t: (t[0] == "legacy_manuscript", t[0]))
    picked_label, picked_paths, latex_items = matches[0]

    # 条件分岐: `n_found != len(latex_items)` を満たす経路を評価する。
    if n_found != len(latex_items):
        raise RuntimeError(
            f"equation count mismatch after pick ({picked_label}): docx={n_found}, latex_items={len(latex_items)}"
        )

    n_replaced = 0
    for target, latex in zip(targets, latex_items):
        try:
            mathml = convert(latex)
            omml_doc = transform(etree.fromstring(mathml.encode("utf-8")))
            omml_str = str(omml_doc)
            omml_str = re.sub(r"^<\\?xml[^>]*>\\s*", "", omml_str).strip()
            omml_elem = etree.fromstring(omml_str.encode("utf-8"))
        except Exception:
            continue

        # 条件分岐: `kind == "inline"` を満たす経路を評価する。

        if kind == "inline":
            run = target
            parent = run.getparent()
            # 条件分岐: `parent is None` を満たす経路を評価する。
            if parent is None:
                continue

            idx = parent.index(run)
            new_run = etree.Element(f"{{{ns['w']}}}r")
            new_run.append(omml_elem)
            parent.remove(run)
            parent.insert(idx, new_run)
        else:
            para = target
            ppr = None
            for child in list(para):
                # 条件分岐: `child.tag == f"{{{ns['w']}}}pPr"` を満たす経路を評価する。
                if child.tag == f"{{{ns['w']}}}pPr":
                    ppr = child
                    break

            for child in list(para):
                # 条件分岐: `child is ppr` を満たす経路を評価する。
                if child is ppr:
                    continue

                para.remove(child)

            omath_para = etree.Element(f"{{{ns['m']}}}oMathPara")
            omath_para.append(omml_elem)
            para.append(omath_para)

        n_replaced += 1

    new_xml = etree.tostring(doc, xml_declaration=True, encoding="UTF-8", standalone="yes")

    buf = io.BytesIO()
    with zipfile.ZipFile(docx_path, "r") as zin:
        with zipfile.ZipFile(buf, "w") as zout:
            for item in zin.infolist():
                data = zin.read(item.filename)
                # 条件分岐: `item.filename == "word/document.xml"` を満たす経路を評価する。
                if item.filename == "word/document.xml":
                    data = new_xml

                zout.writestr(item, data)

    docx_path.write_bytes(buf.getvalue())
    return n_found, n_replaced


def _try_start_word() -> Tuple[Optional[object], str]:
    # 条件分岐: `os.name != "nt"` を満たす経路を評価する。
    if os.name != "nt":
        return None, "non-Windows platform"

    try:
        import win32com.client  # type: ignore
    except Exception as e:
        return None, f"pywin32 (win32com) not available: {e}"

    try:
        # DispatchEx creates a new instance and is less likely to interfere with user's active Word session.
        word = win32com.client.DispatchEx("Word.Application")
    except Exception as e:
        return None, f"Word COM not available: {e}"

    try:
        setattr(word, "Visible", False)
    except Exception:
        pass

    try:
        # 0 = wdAlertsNone
        setattr(word, "DisplayAlerts", 0)
    except Exception:
        pass

    try:
        # Avoid normal.dotm save prompts.
        setattr(word.Options, "SaveNormalPrompt", False)
    except Exception:
        pass

    return word, ""


def _apply_page_margins(doc: object, *, margin_mm: float) -> None:
    margin_mm = max(0.0, float(margin_mm))
    points = _mm_to_points(margin_mm)

    try:
        count = int(doc.Sections.Count)
    except Exception:
        count = 0

    for i in range(1, count + 1):
        try:
            sec = doc.Sections(i)
            ps = sec.PageSetup
            ps.TopMargin = points
            ps.BottomMargin = points
            ps.LeftMargin = points
            ps.RightMargin = points
            try:
                ps.Gutter = 0
            except Exception:
                pass

            try:
                ps.HeaderDistance = 0
                ps.FooterDistance = 0
            except Exception:
                pass
        except Exception:
            continue


def _apply_page_orientation(doc: object, *, orientation: str) -> None:
    o = (orientation or "").strip().lower()
    # WdOrientation: wdOrientPortrait=0, wdOrientLandscape=1
    target = 1 if o == "landscape" else 0

    try:
        count = int(doc.Sections.Count)
    except Exception:
        count = 0

    for i in range(1, count + 1):
        try:
            sec = doc.Sections(i)
            sec.PageSetup.Orientation = target
        except Exception:
            continue


def _max_content_width_points(doc: object) -> Optional[float]:
    try:
        sec_count = int(doc.Sections.Count)
    except Exception:
        sec_count = 0

    widths: List[float] = []
    for i in range(1, sec_count + 1):
        try:
            ps = doc.Sections(i).PageSetup
            w = float(ps.PageWidth) - float(ps.LeftMargin) - float(ps.RightMargin)
            # 条件分岐: `w > 0.0` を満たす経路を評価する。
            if w > 0.0:
                widths.append(w)
        except Exception:
            continue

    # 条件分岐: `not widths` を満たす経路を評価する。

    if not widths:
        return None

    return min(widths)


def _fit_inline_shapes_to_page(doc: object, *, max_width_pt: float) -> None:
    def _safe_pos_points(v: object) -> float:
        try:
            x = float(v)
        except Exception:
            return 0.0
        # Word sometimes uses extreme sentinel values for "undefined".

        if x <= 0.0 or abs(x) > 10000.0:
            return 0.0

        return x

    try:
        count = int(doc.InlineShapes.Count)
    except Exception:
        count = 0

    # Leave a safety margin to avoid rounding/border overflow at the page edge.
    # (Word UI/print sometimes shows slight right-edge overflow when it's too tight.)

    safety_pt = 54.0

    for i in range(1, count + 1):
        try:
            shp = doc.InlineShapes(i)
            w = float(shp.Width)

            avail = float(max_width_pt)
            r = None
            try:
                r = shp.Range
            except Exception:
                r = None

            # If the picture is inside a table, use the containing cell width as the frame.

            if r is not None:
                in_table = False
                try:
                    # wdWithInTable = 12
                    in_table = bool(r.Information(12))
                except Exception:
                    in_table = False

                # 条件分岐: `in_table` を満たす経路を評価する。

                if in_table:
                    try:
                        cell = r.Cells(1)
                        cell_w = float(cell.Width)
                        try:
                            cell_w -= float(cell.LeftPadding) + float(cell.RightPadding)
                        except Exception:
                            pass

                        # 条件分岐: `cell_w > 0.0` を満たす経路を評価する。

                        if cell_w > 0.0:
                            avail = min(avail, cell_w)
                    except Exception:
                        pass

            # Account for paragraph indents (HTML import sometimes indents figure paragraphs).

            if r is not None:
                try:
                    # 条件分岐: `int(r.Paragraphs.Count) >= 1` を満たす経路を評価する。
                    if int(r.Paragraphs.Count) >= 1:
                        p = r.Paragraphs(1)
                        pf = p.Range.ParagraphFormat
                        avail -= _safe_pos_points(pf.LeftIndent)
                        avail -= _safe_pos_points(pf.RightIndent)
                        avail -= _safe_pos_points(pf.FirstLineIndent)
                except Exception:
                    pass

            avail = max(36.0, float(avail) - float(safety_pt))
            # 条件分岐: `w <= avail` を満たす経路を評価する。
            if w <= avail:
                continue

            try:
                shp.LockAspectRatio = True
            except Exception:
                pass

            shp.Width = float(avail)
        except Exception:
            continue


def _fit_floating_shapes_to_page(doc: object, *, max_width_pt: float) -> None:
    """
    Word's HTML import sometimes creates floating Shapes (not InlineShapes), especially when
    images are placed side-by-side. Resize picture-like Shapes to avoid page overflow.
    """

    def _safe_pos_points(v: object) -> float:
        try:
            x = float(v)
        except Exception:
            return 0.0

        # 条件分岐: `x <= 0.0 or abs(x) > 10000.0` を満たす経路を評価する。

        if x <= 0.0 or abs(x) > 10000.0:
            return 0.0

        return x

    try:
        count = int(doc.Shapes.Count)
    except Exception:
        count = 0

    safety_pt = 54.0

    for i in range(1, count + 1):
        try:
            shp = doc.Shapes(i)

            # Only touch picture-like shapes. (msoPicture=13, msoLinkedPicture=11)
            try:
                t = int(shp.Type)
                # 条件分岐: `t not in (11, 13)` を満たす経路を評価する。
                if t not in (11, 13):
                    continue
            except Exception:
                continue

            w = float(shp.Width)
            avail = float(max_width_pt)

            anchor = None
            try:
                anchor = shp.Anchor
            except Exception:
                anchor = None

            # If the picture is anchored inside a table cell, use the cell width as the frame.

            if anchor is not None:
                in_table = False
                try:
                    in_table = bool(anchor.Information(12))  # wdWithInTable
                except Exception:
                    in_table = False

                # 条件分岐: `in_table` を満たす経路を評価する。

                if in_table:
                    try:
                        cell = anchor.Cells(1)
                        cell_w = float(cell.Width)
                        try:
                            cell_w -= float(cell.LeftPadding) + float(cell.RightPadding)
                        except Exception:
                            pass

                        # 条件分岐: `cell_w > 0.0` を満たす経路を評価する。

                        if cell_w > 0.0:
                            avail = min(avail, cell_w)
                    except Exception:
                        pass

                # Account for paragraph indents (HTML import sometimes indents figure paragraphs).

                try:
                    # 条件分岐: `int(anchor.Paragraphs.Count) >= 1` を満たす経路を評価する。
                    if int(anchor.Paragraphs.Count) >= 1:
                        p = anchor.Paragraphs(1)
                        pf = p.Range.ParagraphFormat
                        avail -= _safe_pos_points(pf.LeftIndent)
                        avail -= _safe_pos_points(pf.RightIndent)
                        avail -= _safe_pos_points(pf.FirstLineIndent)
                except Exception:
                    pass

            avail = max(36.0, float(avail) - float(safety_pt))
            # 条件分岐: `w <= avail` を満たす経路を評価する。
            if w <= avail:
                continue

            try:
                shp.LockAspectRatio = True
            except Exception:
                pass

            try:
                shp.Width = float(avail)
            except Exception:
                continue
        except Exception:
            continue


def _table_width_points(table: object) -> Optional[float]:
    try:
        cols = int(table.Columns.Count)
    except Exception:
        return None

    # 条件分岐: `cols <= 0` を満たす経路を評価する。

    if cols <= 0:
        return None

    total = 0.0
    for j in range(1, cols + 1):
        try:
            total += float(table.Columns(j).Width)
        except Exception:
            pass

    return total


def _fit_tables_to_page(doc: object, *, max_width_pt: float) -> None:
    try:
        count = int(doc.Tables.Count)
    except Exception:
        count = 0

    # Word may not update table geometry until the document is (re)paginated.
    # Without it, AutoFitBehavior can appear to succeed yet widths remain unchanged.

    def _repaginate() -> None:
        try:
            doc.Repaginate()
        except Exception:
            pass

    def _normalize_table_frame(t: object) -> None:
        try:
            pf = t.Range.ParagraphFormat
            pf.LeftIndent = 0
            pf.RightIndent = 0
            pf.FirstLineIndent = 0
        except Exception:
            pass

        try:
            t.Rows.LeftIndent = 0
        except Exception:
            pass

        try:
            t.Rows.WrapAroundText = False
        except Exception:
            pass

    def _insert_break_opportunities_in_table_text(t: object) -> None:
        """
        Some table cells contain long unbroken tokens (paths/ids). Insert zero-width spaces after
        common separators so Word can wrap without widening the table.
        """
        try:
            rng = t.Range
            # Exclude the end-of-cell/end-of-table marker.
            rng.End = rng.End - 1
        except Exception:
            return

        zws = "\u200b"

        # Remove previous ZWS insertions to keep the operation idempotent.
        try:
            f = rng.Find
            f.ClearFormatting()
            f.Replacement.ClearFormatting()
            # Execute(FindText, MatchCase, MatchWholeWord, MatchWildcards, MatchSoundsLike, MatchAllWordForms,
            #         Forward, Wrap, Format, ReplaceWith, Replace)
            f.Execute(zws, False, False, False, False, False, True, 1, False, "", 2)
        except Exception:
            pass

        for sep in ("/", "\\", "_", ";", ":"):
            try:
                f = rng.Find
                f.ClearFormatting()
                f.Replacement.ClearFormatting()
                # Execute(FindText, MatchCase, MatchWholeWord, MatchWildcards, MatchSoundsLike, MatchAllWordForms,
                #         Forward, Wrap, Format, ReplaceWith, Replace)
                f.Execute(sep, False, False, False, False, False, True, 1, False, sep + zws, 2)
            except Exception:
                continue

    def _freeze_table_layout(t: object) -> None:
        # Freeze widths so Word won't re-expand the table when opened interactively.
        try:
            # wdAutoFitFixed = 0
            t.AutoFitBehavior(0)
        except Exception:
            pass

        try:
            t.AllowAutoFit = False
        except Exception:
            pass

        try:
            # Keep table within the frame (avoid edge overflow in UI/print due to rounding).
            # wdPreferredWidthPoints = 3
            t.PreferredWidthType = 3
            t.PreferredWidth = float(max_width_pt) - 54.0
        except Exception:
            pass

    # First pass: normalize and ask Word to auto-fit all tables to the window width.

    for i in range(1, count + 1):
        try:
            t = doc.Tables(i)
            _normalize_table_frame(t)
            _insert_break_opportunities_in_table_text(t)
            try:
                t.AllowAutoFit = True
            except Exception:
                pass

            try:
                # wdAutoFitWindow = 2 (fit to page width)
                t.AutoFitBehavior(2)
            except Exception:
                pass
        except Exception:
            continue

    _repaginate()

    # Second pass: scale columns proportionally if any table is wider than our target frame.
    target_w = float(max_width_pt) - 54.0
    for i in range(1, count + 1):
        try:
            t = doc.Tables(i)
            w = _table_width_points(t)
            # 条件分岐: `w is None or w <= target_w + 0.5` を満たす経路を評価する。
            if w is None or w <= target_w + 0.5:
                continue

            # 条件分岐: `w <= 0.0` を満たす経路を評価する。

            if w <= 0.0:
                continue

            ratio = float(target_w) / float(w)
            # 条件分岐: `ratio <= 0.0 or ratio >= 1.0` を満たす経路を評価する。
            if ratio <= 0.0 or ratio >= 1.0:
                continue

            for j in range(1, int(t.Columns.Count) + 1):
                try:
                    t.Columns(j).Width = float(t.Columns(j).Width) * ratio
                except Exception:
                    continue
        except Exception:
            continue

    _repaginate()

    # Freeze table layouts to keep them stable in Word UI.
    for i in range(1, count + 1):
        try:
            _freeze_table_layout(doc.Tables(i))
        except Exception:
            continue

    _repaginate()


def _disable_table_borders(doc: object) -> int:
    """
    Remove table borders in the exported DOCX.

    Paper style requirement: no visible table borders (枠線なし).
    """
    try:
        count = int(doc.Tables.Count)
    except Exception:
        count = 0

    # 条件分岐: `count <= 0` を満たす経路を評価する。

    if count <= 0:
        return 0

    n = 0
    for i in range(1, count + 1):
        try:
            t = doc.Tables(i)
        except Exception:
            continue

        try:
            t.Borders.Enable = False
            n += 1
        except Exception:
            continue

    return n


def _normalize_heading_style_sizes(doc: object) -> int:
    """
    Normalize Heading 4/5 sizes so subitems don't become unreadably small.
    """

    def _get_style(name: str) -> Optional[object]:
        try:
            return doc.Styles(name)
        except Exception:
            return None

    def _first_style(names: Sequence[str]) -> Optional[object]:
        for name in names:
            st = _get_style(name)
            # 条件分岐: `st is not None` を満たす経路を評価する。
            if st is not None:
                return st

        return None

    base_size: Optional[float]
    try:
        base_size = float(doc.Styles("Normal").Font.Size)
    except Exception:
        base_size = None

    # 条件分岐: `base_size is not None and (not (base_size > 0.0) or base_size > 300.0)` を満たす経路を評価する。

    if base_size is not None and (not (base_size > 0.0) or base_size > 300.0):
        base_size = None

    h4 = _first_style(["Heading 4", "見出し 4"])
    h5 = _first_style(["Heading 5", "見出し 5"])

    touched = 0

    def _safe_size(val: object) -> Optional[float]:
        try:
            x = float(val)
        except Exception:
            return None

        # 条件分岐: `not (x > 0.0) or x > 300.0` を満たす経路を評価する。

        if not (x > 0.0) or x > 300.0:
            return None

        return x

    desired_h4: Optional[float] = None
    # 条件分岐: `base_size is not None` を満たす経路を評価する。
    if base_size is not None:
        desired_h4 = base_size + 2.0

    h4_size: Optional[float] = None
    # 条件分岐: `h4 is not None` を満たす経路を評価する。
    if h4 is not None:
        try:
            h4_size = _safe_size(h4.Font.Size)
        except Exception:
            h4_size = None

        # 条件分岐: `desired_h4 is not None and (h4_size is None or h4_size < desired_h4 - 1e-9)` を満たす経路を評価する。

        if desired_h4 is not None and (h4_size is None or h4_size < desired_h4 - 1e-9):
            try:
                h4.Font.Size = desired_h4
                h4_size = desired_h4
                touched += 1
            except Exception:
                pass

    desired_h5: Optional[float] = None
    # 条件分岐: `h4_size is not None` を満たす経路を評価する。
    if h4_size is not None:
        desired_h5 = h4_size
    # 条件分岐: 前段条件が不成立で、`desired_h4 is not None` を追加評価する。
    elif desired_h4 is not None:
        desired_h5 = desired_h4
    # 条件分岐: 前段条件が不成立で、`base_size is not None` を追加評価する。
    elif base_size is not None:
        desired_h5 = base_size + 2.0

    # 条件分岐: `h5 is not None and desired_h5 is not None` を満たす経路を評価する。

    if h5 is not None and desired_h5 is not None:
        try:
            h5_size = _safe_size(h5.Font.Size)
        except Exception:
            h5_size = None

        # 条件分岐: `h5_size is None or h5_size < desired_h5 - 1e-9` を満たす経路を評価する。

        if h5_size is None or h5_size < desired_h5 - 1e-9:
            try:
                h5.Font.Size = desired_h5
                touched += 1
            except Exception:
                pass

    return touched


def _tighten_equation_paragraph_spacing(doc: object) -> int:
    """
    Reduce "looks like missing equations" whitespace around Word native equations (OMML).
    """
    try:
        count = int(doc.OMaths.Count)
    except Exception:
        count = 0

    # 条件分岐: `count <= 0` を満たす経路を評価する。

    if count <= 0:
        return 0

    touched = 0
    for i in range(1, count + 1):
        try:
            om = doc.OMaths(i)
            rng = om.Range
        except Exception:
            continue

        try:
            pcount = int(rng.Paragraphs.Count)
        except Exception:
            pcount = 0

        for j in range(1, pcount + 1):
            try:
                p = rng.Paragraphs(j)
                pf = p.Range.ParagraphFormat
                pf.SpaceBefore = 0
                pf.SpaceAfter = 0
                # wdLineSpaceSingle = 0
                pf.LineSpacingRule = 0
                touched += 1
            except Exception:
                continue

    return touched


def _tighten_spacing_around_equations(doc: object) -> int:
    """
    Tighten whitespace in the *adjacent* paragraphs around OMML blocks.

    Even if the equation paragraph itself has SpaceBefore/After=0, a large SpaceAfter in the
    previous paragraph (or SpaceBefore in the next) can create a "blank line" impression.
    """
    try:
        paras = doc.Paragraphs
        n = int(paras.Count)
    except Exception:
        return 0

    def _is_any_heading(style_name: str) -> bool:
        for level in (1, 2, 3, 4, 5, 6):
            # 条件分岐: `_is_heading(style_name, level)` を満たす経路を評価する。
            if _is_heading(style_name, level):
                return True

        return False

    touched = 0
    for i in range(1, n + 1):
        try:
            p = paras(i)
        except Exception:
            continue

        try:
            has_math = int(p.Range.OMaths.Count) > 0
        except Exception:
            has_math = False

        # 条件分岐: `not has_math` を満たす経路を評価する。

        if not has_math:
            continue

        try:
            pf = p.Range.ParagraphFormat
            pf.SpaceBefore = 0
            pf.SpaceAfter = 0
            # wdLineSpaceSingle = 0
            pf.LineSpacingRule = 0
            touched += 1
        except Exception:
            pass

        # Previous paragraph: reduce SpaceAfter so the equation doesn't look "detached".

        if i > 1:
            try:
                prev = paras(i - 1)
                prev_style = _style_name_local(prev)
                prev_text = str(prev.Range.Text).replace("\r", "").replace("\n", "").strip()
                # 条件分岐: `prev_text and (not _is_any_heading(prev_style))` を満たす経路を評価する。
                if prev_text and (not _is_any_heading(prev_style)):
                    prev_pf = prev.Range.ParagraphFormat
                    prev_pf.SpaceAfter = 0
                    touched += 1
            except Exception:
                pass

        # Next paragraph: reduce SpaceBefore for the same reason.

        if i < n:
            try:
                nxt = paras(i + 1)
                nxt_style = _style_name_local(nxt)
                nxt_text = str(nxt.Range.Text).replace("\r", "").replace("\n", "").strip()
                # 条件分岐: `nxt_text and (not _is_any_heading(nxt_style))` を満たす経路を評価する。
                if nxt_text and (not _is_any_heading(nxt_style)):
                    nxt_pf = nxt.Range.ParagraphFormat
                    nxt_pf.SpaceBefore = 0
                    touched += 1
            except Exception:
                pass

    return touched


def _patch_docx_callout_punctuation(docx_path: Path) -> int:
    """
    Patch DOCX XML directly to restore fullwidth punctuation for key callouts.

    Word's HTML import may normalize fullwidth punctuation (e.g. "："→":") and we can't reliably
    prevent that via Word automation. Post-save XML patching is deterministic.
    """
    # 条件分岐: `not docx_path.exists()` を満たす経路を評価する。
    if not docx_path.exists():
        return 0

    try:
        with zipfile.ZipFile(docx_path, "r") as zin:
            try:
                xml_bytes = zin.read("word/document.xml")
            except Exception:
                return 0

            try:
                xml = xml_bytes.decode("utf-8")
            except Exception:
                xml = xml_bytes.decode("utf-8", errors="ignore")

            n = 0
            # If the arrow got lost, restore it.
            if "要請（P内部）帰結" in xml:
                xml = xml.replace("要請（P内部）帰結", "要請（P内部）→帰結")
                n += 1

            def _replace_colon_after_marker(text: str, marker: str) -> Tuple[str, int]:
                touched = 0
                pos = 0
                # Replace the first <w:t>:</w:t> that appears after each marker occurrence.
                while True:
                    idx = text.find(marker, pos)
                    # 条件分岐: `idx < 0` を満たす経路を評価する。
                    if idx < 0:
                        break

                    window_end = min(len(text), idx + 2000)
                    window = text[idx:window_end]
                    m = re.search(r"(<w:t[^>]*>)(:)(</w:t>)", window)
                    # 条件分岐: `not m` を満たす経路を評価する。
                    if not m:
                        pos = idx + len(marker)
                        continue
                    # Replace only the ':' character (keep tag/attrs unchanged).

                    abs_start = idx + m.start(2)
                    abs_end = idx + m.end(2)
                    text = text[:abs_start] + "：" + text[abs_end:]
                    touched += 1
                    pos = idx + m.end(3)

                return text, touched

            # Restore "：" after specific callout labels.

            for marker in ("要請（P内部）→帰結", "要請（自由波の応答）"):
                xml, dn = _replace_colon_after_marker(xml, marker)
                n += dn

            # 条件分岐: `n <= 0` を満たす経路を評価する。

            if n <= 0:
                return 0

            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w") as zout:
                for info in zin.infolist():
                    data = zin.read(info.filename)
                    # 条件分岐: `info.filename == "word/document.xml"` を満たす経路を評価する。
                    if info.filename == "word/document.xml":
                        data = xml.encode("utf-8")

                    zout.writestr(info, data)

            docx_path.write_bytes(buf.getvalue())
            return n
    except Exception:
        return 0


def _patch_docx_force_white_background(docx_path: Path) -> int:
    """
    Force a white background in the exported DOCX.

    Word's HTML import may translate HTML/CSS background colors into DOCX shading (<w:shd>),
    which can lead to gray paragraph/table backgrounds in the output. For paper sharing, we
    want a clean white page with no shaded blocks.

    This patch:
    - normalizes *unwanted* gray-ish shading fills (e.g. FAFAFA) to white
      while preserving intentional colors (e.g. table header shading, heatmap cells).
    - removes any <w:highlight .../> tags
    """
    # 条件分岐: `not docx_path.exists()` を満たす経路を評価する。
    if not docx_path.exists():
        return 0

    # Only rewrite these typical "HTML import" gray backgrounds.
    # Keep other colors (e.g., header shading / metric heatmaps).

    unwanted_fills = {
        "FAFAFA",
        "F8F8F8",
        "F5F5F5",
        "F2F2F2",
        "F0F0F0",
        "EEEEEE",
        "EDEDED",
        "E3E3E3",
        # old dark code-block background (paper_html used to set it)
        "0B1020",
    }

    targets = ("word/document.xml", "word/styles.xml")
    try:
        with zipfile.ZipFile(docx_path, "r") as zin:
            patched: dict[str, bytes] = {}
            n_total = 0

            for name in targets:
                try:
                    xml_bytes = zin.read(name)
                except Exception:
                    continue

                try:
                    xml = xml_bytes.decode("utf-8")
                except Exception:
                    xml = xml_bytes.decode("utf-8", errors="ignore")

                def _repl_shd(m: re.Match[str]) -> str:
                    fill_raw = m.group(2) or ""
                    fill = fill_raw.strip().upper()
                    # 条件分岐: `fill == "AUTO" or fill in unwanted_fills` を満たす経路を評価する。
                    if fill == "AUTO" or fill in unwanted_fills:
                        return m.group(1) + "FFFFFF" + m.group(3)
                    # Preserve intentional colors.

                    return m.group(0)

                xml2, n1 = re.subn(
                    r'(<w:shd\b[^>]*?\bw:fill=")([^"]+)(")',
                    _repl_shd,
                    xml,
                    flags=re.IGNORECASE,
                )
                xml2, n2 = re.subn(r"<w:highlight\b[^/>]*/>", "", xml2)

                # 条件分岐: `xml2 != xml` を満たす経路を評価する。
                if xml2 != xml:
                    patched[name] = xml2.encode("utf-8")
                    n_total += int(n1) + int(n2)

            # 条件分岐: `n_total <= 0` を満たす経路を評価する。

            if n_total <= 0:
                return 0

            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w") as zout:
                for info in zin.infolist():
                    data = zin.read(info.filename)
                    # 条件分岐: `info.filename in patched` を満たす経路を評価する。
                    if info.filename in patched:
                        data = patched[info.filename]

                    zout.writestr(info, data)

            docx_path.write_bytes(buf.getvalue())
            return n_total
    except Exception:
        return 0


def _style_name_local(paragraph: object) -> str:
    try:
        style = paragraph.Range.Style
        try:
            return str(style.NameLocal)
        except Exception:
            return str(style)
    except Exception:
        return ""


def _is_heading(style_name: str, level: int) -> bool:
    s = (style_name or "").strip()
    # 条件分岐: `not s` を満たす経路を評価する。
    if not s:
        return False

    # 条件分岐: `s == f"Heading {level}"` を満たす経路を評価する。

    if s == f"Heading {level}":
        return True

    # 条件分岐: `s == f"見出し {level}"` を満たす経路を評価する。

    if s == f"見出し {level}":
        return True

    return s.endswith(f"見出し {level}")


def _apply_page_breaks_for_cards(doc: object) -> None:
    """
    Public report readability:
    - Start each section (Heading 2, after TOC) on a new page.
    - Start each validation card (Heading 3) on a new page, except the first card right after its section heading.
    """
    try:
        paras = doc.Paragraphs
        n = int(paras.Count)
    except Exception:
        return

    records: List[Tuple[int, str, str]] = []
    for i in range(1, n + 1):
        try:
            p = paras(i)
            text = str(p.Range.Text).replace("\r", "").replace("\n", "").strip()
            style = _style_name_local(p)
            records.append((i, text, style))
        except Exception:
            records.append((i, "", ""))

    toc_seen = False
    after_toc = False
    prev_nonempty_style = ""
    break_before: List[int] = []

    for idx, text, style in records:
        # 条件分岐: `_is_heading(style, 2) and text == "目次"` を満たす経路を評価する。
        if _is_heading(style, 2) and text == "目次":
            toc_seen = True
        # 条件分岐: 前段条件が不成立で、`toc_seen and _is_heading(style, 2) and text` を追加評価する。
        elif toc_seen and _is_heading(style, 2) and text:
            after_toc = True

        # 条件分岐: `after_toc` を満たす経路を評価する。

        if after_toc:
            # 条件分岐: `_is_heading(style, 2) and text` を満たす経路を評価する。
            if _is_heading(style, 2) and text:
                break_before.append(idx)
            # 条件分岐: 前段条件が不成立で、`_is_heading(style, 3) and text` を追加評価する。
            elif _is_heading(style, 3) and text:
                # 条件分岐: `not _is_heading(prev_nonempty_style, 2)` を満たす経路を評価する。
                if not _is_heading(prev_nonempty_style, 2):
                    break_before.append(idx)

        # 条件分岐: `text` を満たす経路を評価する。

        if text:
            prev_nonempty_style = style

    for idx in break_before:
        try:
            paras(idx).Range.ParagraphFormat.PageBreakBefore = True
        except Exception:
            continue


def _apply_page_breaks_for_headings(doc: object, *, levels: Sequence[int]) -> None:
    """
    Paper readability:
    - Insert a page break before specified heading levels (e.g. chapter/section).
    - If a TOC ("目次") exists, start applying after the TOC.
    """
    try:
        paras = doc.Paragraphs
        n = int(paras.Count)
    except Exception:
        return

    levels_norm = [int(x) for x in levels if int(x) >= 1]
    # 条件分岐: `not levels_norm` を満たす経路を評価する。
    if not levels_norm:
        return

    records: List[Tuple[int, str, str]] = []
    for i in range(1, n + 1):
        try:
            p = paras(i)
            text = str(p.Range.Text).replace("\r", "").replace("\n", "").strip()
            style = _style_name_local(p)
            records.append((i, text, style))
        except Exception:
            records.append((i, "", ""))

    has_toc = any(_is_heading(style, 2) and text == "目次" for _, text, style in records)
    toc_seen = False
    after_toc = not has_toc
    break_before: List[int] = []

    # Fallback targets for headings that may arrive with non-heading styles after HTML import.
    # (kept narrow to avoid unintended breaks)
    forced_prefixes = ("4.10.2 ", "4.10.3 ")

    for idx, text, style in records:
        # 条件分岐: `not text` を満たす経路を評価する。
        if not text:
            continue

        # 条件分岐: `has_toc` を満たす経路を評価する。

        if has_toc:
            # 条件分岐: `_is_heading(style, 2) and text == "目次"` を満たす経路を評価する。
            if _is_heading(style, 2) and text == "目次":
                toc_seen = True
                after_toc = False
                continue

            # 条件分岐: `toc_seen and _is_heading(style, 2)` を満たす経路を評価する。

            if toc_seen and _is_heading(style, 2):
                after_toc = True

        # 条件分岐: `not after_toc` を満たす経路を評価する。

        if not after_toc:
            continue

        # 条件分岐: `any(text.startswith(pref) for pref in forced_prefixes)` を満たす経路を評価する。

        if any(text.startswith(pref) for pref in forced_prefixes):
            break_before.append(idx)
            continue

        for level in levels_norm:
            # 条件分岐: `_is_heading(style, level)` を満たす経路を評価する。
            if _is_heading(style, level):
                break_before.append(idx)
                break

    for idx in break_before:
        try:
            paras(idx).Range.ParagraphFormat.PageBreakBefore = True
        except Exception:
            continue


def _inlineize_picture_shapes(doc: object) -> int:
    """
    Word's HTML import sometimes creates floating Shapes for pictures (especially when
    layout gets complicated). Floating Shapes can overflow the right edge even if each
    picture width looks "reasonable". Convert picture-like Shapes to InlineShapes so the
    later fitting logic can keep them inside the text frame.
    """
    try:
        count = int(doc.Shapes.Count)
    except Exception:
        return 0

    n = 0
    for i in range(count, 0, -1):
        try:
            shp = doc.Shapes(i)
            try:
                # msoPicture=13, msoLinkedPicture=11
                t = int(shp.Type)
                # 条件分岐: `t not in (11, 13)` を満たす経路を評価する。
                if t not in (11, 13):
                    continue
            except Exception:
                continue

            try:
                shp.ConvertToInlineShape()
                n += 1
            except Exception:
                continue
        except Exception:
            continue

    return n


def _scale_equation_images(doc: object, *, alt_text: str, scale: float) -> int:
    """
    Word の HTML 取り込みは、PNG の DPI を尊重して「物理サイズ」を決めるため、
    publish で生成した “数式画像” が Word では小さく見えがち。

    ここでは alt/descr が `alt_text`（既定: "数式"）の画像だけを対象に、幅/高さを
    一律で拡大する（後段の fit 処理でページ幅に収まるよう補正される）。
    """
    s = float(scale)
    # 条件分岐: `not (s > 0.0) or abs(s - 1.0) < 1e-9` を満たす経路を評価する。
    if not (s > 0.0) or abs(s - 1.0) < 1e-9:
        return 0

    def _has_alt(v: object) -> bool:
        try:
            t = str(v).strip()
        except Exception:
            return False

        return bool(t) and (alt_text in t)

    n = 0

    # InlineShapes（通常の <img>）
    try:
        count = int(doc.InlineShapes.Count)
    except Exception:
        count = 0

    for i in range(1, count + 1):
        try:
            shp = doc.InlineShapes(i)
            try:
                # 条件分岐: `not _has_alt(getattr(shp, "AlternativeText", ""))` を満たす経路を評価する。
                if not _has_alt(getattr(shp, "AlternativeText", "")):
                    continue
            except Exception:
                continue

            try:
                w = float(shp.Width)
                h = float(shp.Height)
            except Exception:
                continue

            # 条件分岐: `w <= 0.0 or h <= 0.0` を満たす経路を評価する。

            if w <= 0.0 or h <= 0.0:
                continue

            try:
                shp.LockAspectRatio = True
            except Exception:
                pass

            try:
                shp.Width = w * s
                shp.Height = h * s
            except Exception:
                # Fallback: width only
                try:
                    shp.Width = w * s
                except Exception:
                    continue

            n += 1
        except Exception:
            continue

    # Floating Shapes（HTML 取り込みで稀に発生）

    try:
        count2 = int(doc.Shapes.Count)
    except Exception:
        count2 = 0

    for i in range(1, count2 + 1):
        try:
            shp = doc.Shapes(i)
            try:
                # msoPicture=13, msoLinkedPicture=11
                t = int(shp.Type)
                # 条件分岐: `t not in (11, 13)` を満たす経路を評価する。
                if t not in (11, 13):
                    continue
            except Exception:
                continue

            try:
                # 条件分岐: `not _has_alt(getattr(shp, "AlternativeText", ""))` を満たす経路を評価する。
                if not _has_alt(getattr(shp, "AlternativeText", "")):
                    continue
            except Exception:
                continue

            try:
                w = float(shp.Width)
                h = float(shp.Height)
            except Exception:
                continue

            # 条件分岐: `w <= 0.0 or h <= 0.0` を満たす経路を評価する。

            if w <= 0.0 or h <= 0.0:
                continue

            try:
                shp.LockAspectRatio = True
            except Exception:
                pass

            try:
                shp.Width = w * s
                shp.Height = h * s
            except Exception:
                try:
                    shp.Width = w * s
                except Exception:
                    continue

            n += 1
        except Exception:
            continue

    return n


def _scale_word_equations(doc: object, *, scale: float) -> int:
    """
    Word のネイティブ数式（OMML）のフォントサイズを一律スケールする。
    """
    s = float(scale)
    # 条件分岐: `not (s > 0.0) or abs(s - 1.0) < 1e-9` を満たす経路を評価する。
    if not (s > 0.0) or abs(s - 1.0) < 1e-9:
        return 0

    try:
        count = int(doc.OMaths.Count)
    except Exception:
        return 0

    base_size: Optional[float] = None
    try:
        base_size = float(doc.Styles("Normal").Font.Size)
    except Exception:
        base_size = None

    # 条件分岐: `base_size is not None and (not (base_size > 0.0) or base_size > 300.0)` を満たす経路を評価する。

    if base_size is not None and (not (base_size > 0.0) or base_size > 300.0):
        base_size = None

    n = 0
    for i in range(1, count + 1):
        try:
            om = doc.OMaths(i)
            r = om.Range
            try:
                size = float(r.Font.Size)
            except Exception:
                size = -1.0
            # Guard against sentinel / invalid values.

            if (not (size > 0.0) or size > 300.0) and base_size is not None:
                size = base_size

            # 条件分岐: `not (size > 0.0) or size > 300.0` を満たす経路を評価する。

            if not (size > 0.0) or size > 300.0:
                continue

            r.Font.Size = float(size) * s
            n += 1
        except Exception:
            continue

    return n


def _postprocess_docx(
    word: object,
    docx_path: Path,
    *,
    orientation: str,
    margin_mm: float,
    pagebreak_cards: bool,
    pagebreak_headings: bool,
) -> None:
    doc = None
    try:
        doc = word.Documents.Open(
            str(docx_path),
            ConfirmConversions=False,
            ReadOnly=False,
            AddToRecentFiles=False,
        )
        _apply_page_orientation(doc, orientation=str(orientation))
        _apply_page_margins(doc, margin_mm=float(margin_mm))
        # 条件分岐: `pagebreak_cards` を満たす経路を評価する。
        if pagebreak_cards:
            _apply_page_breaks_for_cards(doc)

        # 条件分岐: `pagebreak_headings` を満たす経路を評価する。

        if pagebreak_headings:
            # Paper readability:
            # - Heading 2 = 章
            # - Heading 3 = 項
            # - Heading 4 = 小項（例：5.3.1）
            # - Heading 5 = 小項の細分（例：4.2.7.2）
            _apply_page_breaks_for_headings(doc, levels=(2, 3, 4, 5))

        try:
            doc.Repaginate()
        except Exception:
            pass

        try:
            _normalize_heading_style_sizes(doc)
        except Exception:
            pass

        # Reduce layout surprises: convert picture-like floating Shapes to InlineShapes.

        try:
            _inlineize_picture_shapes(doc)
        except Exception:
            pass

        # Keep equation image size close to HTML baseline in Word.
        # (historical 3.0x scaling made inline equations too large)

        try:
            _scale_equation_images(doc, alt_text="数式", scale=1.0)
        except Exception:
            pass

        try:
            _scale_equation_images(doc, alt_text="数式-inline", scale=1.0)
        except Exception:
            pass

        max_w = _max_content_width_points(doc)
        # 条件分岐: `max_w is not None and max_w > 0.0` を満たす経路を評価する。
        if max_w is not None and max_w > 0.0:
            _fit_tables_to_page(doc, max_width_pt=float(max_w))
            _fit_inline_shapes_to_page(doc, max_width_pt=float(max_w))
            _fit_floating_shapes_to_page(doc, max_width_pt=float(max_w))
            try:
                doc.Repaginate()
            except Exception:
                pass

        # Paper style: remove visible table borders.

        try:
            _disable_table_borders(doc)
        except Exception:
            pass

        # Make native Word equations readable.

        try:
            _scale_word_equations(doc, scale=1.5)
        except Exception:
            pass

        try:
            _tighten_equation_paragraph_spacing(doc)
        except Exception:
            pass

        try:
            _tighten_spacing_around_equations(doc)
        except Exception:
            pass

        doc.Save()
        doc.Close(False)
        doc = None
    finally:
        try:
            # 条件分岐: `doc is not None` を満たす経路を評価する。
            if doc is not None:
                doc.Close(False)
        except Exception:
            pass


def _convert_html_to_docx(
    word: object,
    html_in: Path,
    docx_out: Path,
    *,
    timeout_s: int,
) -> None:
    docx_out.parent.mkdir(parents=True, exist_ok=True)

    # Best-effort: remove existing file to avoid SaveAs prompts / stale outputs.
    if docx_out.exists():
        try:
            docx_out.unlink()
        except Exception as e:
            raise RuntimeError(f"Output DOCX is locked/open; close it and retry: {docx_out} ({e})") from e

    started = time.perf_counter()
    doc = None
    try:
        doc = word.Documents.Open(
            str(html_in),
            ConfirmConversions=False,
            ReadOnly=False,
            AddToRecentFiles=False,
        )
        # 16 = wdFormatDocumentDefault (DOCX)
        doc.SaveAs2(str(docx_out), FileFormat=16)
        doc.Close(False)
        doc = None
    finally:
        try:
            # 条件分岐: `doc is not None` を満たす経路を評価する。
            if doc is not None:
                doc.Close(False)
        except Exception:
            pass

    # Word sometimes returns before the file is fully flushed. Poll a little.

    while True:
        # 条件分岐: `docx_out.exists()` を満たす経路を評価する。
        if docx_out.exists():
            try:
                # 条件分岐: `docx_out.stat().st_size > 0` を満たす経路を評価する。
                if docx_out.stat().st_size > 0:
                    return
            except Exception:
                pass

        # 条件分岐: `(time.perf_counter() - started) > float(timeout_s)` を満たす経路を評価する。

        if (time.perf_counter() - started) > float(timeout_s):
            break

        time.sleep(0.1)

    raise RuntimeError("DOCX creation timed out (file not materialized)")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Convert a local HTML file to DOCX (Microsoft Word automation).")
    ap.add_argument("--in", dest="html_in", required=True, help="Input HTML file path.")
    ap.add_argument("--out", dest="docx_out", required=True, help="Output DOCX file path.")
    ap.add_argument(
        "--orientation",
        choices=["portrait", "landscape"],
        default="portrait",
        help="Page orientation (default: portrait).",
    )
    ap.add_argument("--margin-mm", type=float, default=5.0, help="Page margins in mm (default: 5).")
    ap.add_argument(
        "--pagebreak-validations",
        action="store_true",
        help="Insert page breaks so each validation card starts on a new page (public report).",
    )
    ap.add_argument(
        "--pagebreak-headings",
        action="store_true",
        help="Insert page breaks before chapter/section headings (Heading 2/3; paper).",
    )
    ap.add_argument(
        "--paper-equations",
        action="store_true",
        help="Replace paper equation images with native Word equations (OMML) using doc/paper/*.md LaTeX blocks.",
    )
    ap.add_argument(
        "--mml2omml-xsl",
        default=None,
        help="Path to MML2OMML.XSL (MathML→OMML converter). Default: from Office install if found.",
    )
    ap.add_argument("--timeout-s", type=int, default=180, help="Timeout seconds (default: 180).")
    args = ap.parse_args(argv)

    root = _repo_root()
    html_in = Path(args.html_in)
    docx_out = Path(args.docx_out)

    # 条件分岐: `not html_in.is_absolute()` を満たす経路を評価する。
    if not html_in.is_absolute():
        html_in = (root / html_in).resolve()

    # 条件分岐: `not docx_out.is_absolute()` を満たす経路を評価する。

    if not docx_out.is_absolute():
        docx_out = (root / docx_out).resolve()

    # 条件分岐: `not html_in.exists()` を満たす経路を評価する。

    if not html_in.exists():
        print(f"[err] input HTML not found: {html_in}", file=sys.stderr)
        return 2

    docx_target = docx_out
    docx_out = _choose_tmp_docx_path(docx_target)

    word, reason = _try_start_word()
    # 条件分岐: `not word` を満たす経路を評価する。
    if not word:
        print(f"[warn] no supported Word backend for HTML→DOCX; skipping ({reason}).", file=sys.stderr)
        return 3

    equations_found = 0
    equations_replaced = 0
    equations_error = ""
    html_open, inline_images_count, temp_html = _inline_local_images_for_word(html_in)
    try:
        _convert_html_to_docx(word, html_open, docx_out, timeout_s=int(args.timeout_s))
    except Exception as e:
        print(f"[err] HTML→DOCX failed: {e}", file=sys.stderr)
        return 1
    finally:
        try:
            word.Quit()
        except Exception:
            pass

        # 条件分岐: `temp_html is not None` を満たす経路を評価する。

        if temp_html is not None:
            try:
                temp_html.unlink()
            except Exception:
                pass

    # 条件分岐: `bool(args.paper_equations)` を満たす経路を評価する。

    if bool(args.paper_equations):
        try:
            xsl_path = Path(args.mml2omml_xsl) if args.mml2omml_xsl else _default_mml2omml_xsl()
            b_found, b_replaced = _replace_equation_images_with_word_equations(
                root=root,
                out_dir=docx_out.parent,
                docx_path=docx_out,
                mml2omml_xsl=xsl_path,
                alt_text="数式",
                equation_kind="block",
                html_source=html_in,
            )
            i_found, i_replaced = _replace_equation_images_with_word_equations(
                root=root,
                out_dir=docx_out.parent,
                docx_path=docx_out,
                mml2omml_xsl=xsl_path,
                alt_text="数式-inline",
                equation_kind="inline",
                html_source=html_in,
            )
            equations_found = int(b_found) + int(i_found)
            equations_replaced = int(b_replaced) + int(i_replaced)
        except Exception as e:
            equations_error = str(e)
            print(f"[warn] equation conversion skipped: {e}", file=sys.stderr)

    # Post-process in a fresh Word instance (AutoFit is more reliable after reopening).

    word2, reason2 = _try_start_word()
    # 条件分岐: `not word2` を満たす経路を評価する。
    if not word2:
        print(f"[warn] cannot postprocess DOCX; leaving as-is ({reason2}).", file=sys.stderr)
    else:
        try:
            _postprocess_docx(
                word2,
                docx_out,
                orientation=str(args.orientation),
                margin_mm=float(args.margin_mm),
                pagebreak_cards=bool(args.pagebreak_validations),
                pagebreak_headings=bool(args.pagebreak_headings) or bool(args.paper_equations),
            )
        except Exception as e:
            print(f"[err] DOCX postprocess failed: {e}", file=sys.stderr)
            return 1
        finally:
            try:
                word2.Quit()
            except Exception:
                pass

    # Final deterministic patch (after Word postprocess/save).

    try:
        _patch_docx_callout_punctuation(docx_out)
    except Exception:
        pass

    try:
        _patch_docx_force_white_background(docx_out)
    except Exception:
        pass

    final_docx, promote_warn = _promote_tmp_docx(docx_out, docx_target)
    # 条件分岐: `promote_warn` を満たす経路を評価する。
    if promote_warn:
        print(f"[warn] {promote_warn}", file=sys.stderr)

    docx_out = final_docx

    try:
        worklog.append_event(
            {
                "event_type": "html_to_docx",
                "inputs": {"html": html_in, "html_for_word": html_open},
                "params": {
                    "timeout_s": int(args.timeout_s),
                    "orientation": str(args.orientation),
                    "margin_mm": float(args.margin_mm),
                    "pagebreak_validations": bool(args.pagebreak_validations),
                    "pagebreak_headings": bool(args.pagebreak_headings) or bool(args.paper_equations),
                    "inline_images_count": int(inline_images_count),
                    "paper_equations": bool(args.paper_equations),
                    "equations_found": int(equations_found),
                    "equations_replaced": int(equations_replaced),
                    **({"equations_error": equations_error} if equations_error else {}),
                },
                "outputs": {"docx": docx_out, **({"docx_target": docx_target} if docx_target != docx_out else {})},
            }
        )
    except Exception:
        pass

    print(f"[ok] docx: {docx_out}")
    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())
