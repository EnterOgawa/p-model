#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
paper_html.py

Phase 8 向けに、Markdown草稿（doc/paper/*.md）をHTMLで読める形に整形して出力する。

生成物（既定）:
  - profile=paper: output/private/summary/pmodel_paper.html
  - profile=part2_astrophysics: output/private/summary/pmodel_paper_part2_astrophysics.html
  - profile=part3_quantum: output/private/summary/pmodel_paper_part3_quantum.html
  - profile=part4_verification: output/private/summary/pmodel_paper_part4_verification.html

方針:
  - 公開レポート（pmodel_public_report.html）と同じ “カード” スタイルの単一HTMLにまとめる。
  - Markdown変換は python-markdown があれば使用（無い場合は最低限のフォールバック）。
  - 公開（publish）モードでは “ファイルパス/リンク” を紙面から排除し、本文は「説明と式」「説明と図」に寄せる。
    - 画像（図・式）はHTMLに埋め込み（data URI）で自己完結させる。
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import html
import io
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_ROOT = Path(__file__).resolve().parents[2]
# 条件分岐: `str(_ROOT) not in sys.path` を満たす経路を評価する。
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.summary import paper_latex as _paper_latex, worklog  # noqa: E402

FigureAnchorMap = Dict[str, Tuple[str, str]]


def _repo_root() -> Path:
    return _ROOT


@dataclass(frozen=True)
class FigureItem:
    section: str
    rel: str
    path: Path
    caption: str


def _resolve_repo_asset(rel: str, *, root: Path) -> Path:
    """
    Resolve repo-relative paths, remapping legacy output/<topic>/... to
    output/private/<topic>/... (or output/public for quantum) when needed.
    """
    rel_norm = rel.replace("\\", "/")
    # 条件分岐: `not rel_norm.startswith("output/")` を満たす経路を評価する。
    if not rel_norm.startswith("output/"):
        return root / Path(rel)

    parts = Path(rel_norm).parts
    # 条件分岐: `len(parts) < 2` を満たす経路を評価する。
    if len(parts) < 2:
        return root / Path(rel_norm)

    # 条件分岐: `parts[1] in ("private", "public")` を満たす経路を評価する。

    if parts[1] in ("private", "public"):
        return root / Path(rel_norm)

    topic = parts[1]
    tail = Path(*parts[2:]) if len(parts) > 2 else Path()
    cand_public = (root / "output" / "public" / topic / tail).resolve()
    cand_private = (root / "output" / "private" / topic / tail).resolve()

    # 条件分岐: `topic == "quantum"` を満たす経路を評価する。
    if topic == "quantum":
        # 条件分岐: `cand_public.exists()` を満たす経路を評価する。
        if cand_public.exists():
            return cand_public

        # 条件分岐: `cand_private.exists()` を満たす経路を評価する。

        if cand_private.exists():
            return cand_private

    # 条件分岐: `cand_private.exists()` を満たす経路を評価する。

    if cand_private.exists():
        return cand_private

    # 条件分岐: `cand_public.exists()` を満たす経路を評価する。

    if cand_public.exists():
        return cand_public

    return root / Path(rel_norm)

def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _rel_url(from_dir: Path, target: Path) -> str:
    try:
        rel = os.path.relpath(target, start=from_dir)
    except ValueError:
        rel = str(target)

    return rel.replace("\\", "/")


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _markdown_to_html(md_text: str) -> Tuple[str, str]:
    """
    Return (html, toc_html). toc_html may be "".
    """
    try:
        import markdown as md  # type: ignore

        engine = md.Markdown(
            extensions=["toc", "fenced_code", "tables"],
            extension_configs={"toc": {"permalink": True}},
        )
        body = engine.convert(md_text)
        toc = getattr(engine, "toc", "") or ""
        return body, toc
    except Exception:
        escaped = html.escape(md_text)
        return f"<pre>{escaped}</pre>", ""


_CODE_RE = re.compile(r"<code>([^<]{1,500})</code>")
_CITE_RE = re.compile(r"\[([A-Za-z][A-Za-z0-9_-]{0,40})\]")
# NOTE: use backreference \1 (not a literal "\1") to match the closing tag.
_HEADING_RE = re.compile(r"<h([1-6])\s+id=\"([^\"]+)\"([^>]*)>(.*?)</h\1>", re.S)
_HEADING_NUM_PREFIX_RE = re.compile(r"^\s*(\d+)(?:\.(\d+))?(?:\.(\d+))?(?:\.)?\s")
_TABLE_OPEN_NO_BORDER_RE = re.compile(r"<table(?![^>]*\bborder=)([^>]*)>", flags=re.IGNORECASE)
_ASSET_URL_ATTR_RE = re.compile(
    r"""(?P<attr>\b(?:src|href))=(?P<q>["'])(?P<url>[^"']+)(?P=q)"""
)
_INLINE_FIGURE_LABEL_P_RE = re.compile(
    r"<p>\s*<strong>(?:図|要約図)</strong>[：:]\s*(<figure class='inline-figure'.*?</figure>)\s*</p>",
    flags=re.S,
)
_INLINE_FIGURE_LABEL_RE = re.compile(
    r"<strong>(?:図|要約図)</strong>[：:]\s*(?=<figure class='inline-figure')",
    flags=re.S,
)
_INLINE_FIGURE_P_RE = re.compile(
    r"<p>\s*(<figure class='inline-figure'.*?</figure>)\s*</p>",
    flags=re.S,
)


def _rewrite_repo_relative_asset_urls(rendered_html: str, *, root: Path, out_dir: Path) -> str:
    """
    HTML 内の src/href に含まれる `output/...` 等の “repo-root 相対パス” を、
    HTML 出力先（out_dir）からの相対パスに変換する。

    例：out_dir=output/private/summary のとき
      src="output/cosmology/foo.png" → src="../cosmology/foo.png"

    Markdown 草稿側は `output/...` 表記を維持したまま、生成物だけを表示可能にする。
    """

    def repl(m: re.Match[str]) -> str:
        attr = m.group("attr")
        q = m.group("q")
        url = m.group("url")

        # 条件分岐: `not url` を満たす経路を評価する。
        if not url:
            return m.group(0)

        # External / non-file URLs

        if url.startswith(("http://", "https://", "data:", "mailto:", "javascript:", "#")):
            return m.group(0)

        norm = url.replace("\\", "/")
        # 条件分岐: `norm.startswith(("./", "../"))` を満たす経路を評価する。
        if norm.startswith(("./", "../")):
            return m.group(0)

        # Split off query / fragment suffix (rare for local files, but keep safe).

        base = norm
        suffix = ""
        for sep in ("#", "?"):
            # 条件分岐: `sep in base` を満たす経路を評価する。
            if sep in base:
                base, tail = base.split(sep, 1)
                suffix = sep + tail
                break

        # 条件分岐: `not re.match(r"^(output|doc|data|scripts)/", base)` を満たす経路を評価する。

        if not re.match(r"^(output|doc|data|scripts)/", base):
            return m.group(0)

        target = _resolve_repo_asset(base, root=root)
        rel = _rel_url(out_dir, target)
        return f"{attr}={q}{rel}{suffix}{q}"

    return _ASSET_URL_ATTR_RE.sub(repl, rendered_html)


def _clamp(x: float, lo: float, hi: float) -> float:
    try:
        xf = float(x)
    except Exception:
        return lo

    # 条件分岐: `xf < lo` を満たす経路を評価する。

    if xf < lo:
        return lo

    # 条件分岐: `xf > hi` を満たす経路を評価する。

    if xf > hi:
        return hi

    return xf


def _hsl_to_hex(h: float, s: float, l: float) -> str:
    """
    Convert an HSL color to #RRGGBB.

    - h: [0, 360)
    - s: [0, 1]
    - l: [0, 1]
    """
    h = float(h) % 360.0
    s = _clamp(s, 0.0, 1.0)
    l = _clamp(l, 0.0, 1.0)

    c = (1.0 - abs(2.0 * l - 1.0)) * s
    x = c * (1.0 - abs(((h / 60.0) % 2.0) - 1.0))
    m = l - c / 2.0

    r1 = g1 = b1 = 0.0
    # 条件分岐: `0.0 <= h < 60.0` を満たす経路を評価する。
    if 0.0 <= h < 60.0:
        r1, g1, b1 = c, x, 0.0
    # 条件分岐: 前段条件が不成立で、`60.0 <= h < 120.0` を追加評価する。
    elif 60.0 <= h < 120.0:
        r1, g1, b1 = x, c, 0.0
    # 条件分岐: 前段条件が不成立で、`120.0 <= h < 180.0` を追加評価する。
    elif 120.0 <= h < 180.0:
        r1, g1, b1 = 0.0, c, x
    # 条件分岐: 前段条件が不成立で、`180.0 <= h < 240.0` を追加評価する。
    elif 180.0 <= h < 240.0:
        r1, g1, b1 = 0.0, x, c
    # 条件分岐: 前段条件が不成立で、`240.0 <= h < 300.0` を追加評価する。
    elif 240.0 <= h < 300.0:
        r1, g1, b1 = x, 0.0, c
    else:
        r1, g1, b1 = c, 0.0, x

    r = int(round((r1 + m) * 255.0))
    g = int(round((g1 + m) * 255.0))
    b = int(round((b1 + m) * 255.0))
    r = max(0, min(255, r))
    g = max(0, min(255, g))
    b = max(0, min(255, b))
    return f"#{r:02X}{g:02X}{b:02X}"


def _score_to_bg_hex(score_norm_0_to_3: float) -> str:
    """
    Map discrepancy score to a readable pastel color.

    0 (best) -> green
    1.5      -> yellow
    3 (worst)-> red
    """
    t = _clamp(score_norm_0_to_3 / 3.0, 0.0, 1.0)
    hue = 120.0 * (1.0 - t)  # 120=green, 0=red
    return _hsl_to_hex(hue, 0.65, 0.90)


_INTERNAL_SOURCE_TOKEN_RE = re.compile(r"(?:\s*/\s*)?source=[^\s/;,)]+", flags=re.IGNORECASE)


def _strip_internal_source_tokens(text: str) -> str:
    """
    Remove internal provenance tokens like "source=...metrics.json" from paper-facing text.

    These are useful for repo debugging but are not meaningful to third-party readers.
    """
    s = (text or "").strip()
    # 条件分岐: `not s` を満たす経路を評価する。
    if not s:
        return ""

    s = _INTERNAL_SOURCE_TOKEN_RE.sub("", s)
    # cleanup: repeated separators/spaces after removal
    s = re.sub(r"\s{2,}", " ", s).strip()
    s = re.sub(r"(?:\s*/\s*)+$", "", s).strip()
    return s


def _table1_metric_score_norm_astrophysics(metric_public: str, metric_fallback: str) -> Optional[float]:
    """
    Return a normalized discrepancy score in [0,3] (smaller is better) for Part II Table 1.
    """
    text = (metric_public or "").strip() or (metric_fallback or "").strip()
    # 条件分岐: `not text` を満たす経路を評価する。
    if not text:
        return 1.5

    # Special-case: Viking is a coarse literature-range sanity check.

    if "文献代表レンジ(200" in text and "μs" in text and "内" in text:
        return 0.5

    sigma_m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*σ", text)
    # 条件分岐: `sigma_m` を満たす経路を評価する。
    if sigma_m:
        try:
            abs_sigma = abs(float(sigma_m.group(1)))
            return _clamp(abs_sigma, 0.0, 3.0)
        except Exception:
            pass

    corr_m = re.search(r"corr\s*=\s*([0-9]+(?:\.[0-9]+)?)", text, flags=re.IGNORECASE)
    # 条件分岐: `corr_m` を満たす経路を評価する。
    if corr_m:
        try:
            corr = float(corr_m.group(1))
        except Exception:
            corr = None

        # 条件分岐: `corr is not None` を満たす経路を評価する。

        if corr is not None:
            # 1.00 => 0, 0.95 => 1, 0.90 => 2, 0.70 => 3 (clip)
            if corr >= 0.95:
                return _clamp((1.0 - corr) / (1.0 - 0.95), 0.0, 1.0)

            # 条件分岐: `corr >= 0.90` を満たす経路を評価する。

            if corr >= 0.90:
                return 1.0 + _clamp((0.95 - corr) / (0.95 - 0.90), 0.0, 1.0)

            return 2.0 + _clamp((0.90 - corr) / (0.90 - 0.70), 0.0, 1.0)

    # Use best detector's R^2 as a coarse agreement metric.

    r2_vals: List[float] = []
    for m in re.finditer(r"R\^2\s*=\s*([0-9]+(?:\.[0-9]+)?)", metric_fallback or ""):
        try:
            r2_vals.append(float(m.group(1)))
        except Exception:
            continue

    # 条件分岐: `r2_vals` を満たす経路を評価する。

    if r2_vals:
        best = max(r2_vals)
        # 1.00 => 0, 0.90 => 1, 0.60 => 2, 0.00 => 3 (clip)
        if best >= 0.90:
            return _clamp((1.0 - best) / (1.0 - 0.90), 0.0, 1.0)

        # 条件分岐: `best >= 0.60` を満たす経路を評価する。

        if best >= 0.60:
            return 1.0 + _clamp((0.90 - best) / (0.90 - 0.60), 0.0, 1.0)

        return 2.0 + _clamp((0.60 - best) / 0.60, 0.0, 1.0)

    pct_m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*%", text)
    # 条件分岐: `pct_m` を満たす経路を評価する。
    if pct_m:
        try:
            abs_pct = abs(float(pct_m.group(1)))
        except Exception:
            abs_pct = None

        # 条件分岐: `abs_pct is not None` を満たす経路を評価する。

        if abs_pct is not None:
            # 0.0% => 0, 0.1% => 1, 1.0% => 2, 5.0% => 3 (clip)
            if abs_pct <= 0.1:
                return _clamp(abs_pct / 0.1, 0.0, 1.0)

            # 条件分岐: `abs_pct <= 1.0` を満たす経路を評価する。

            if abs_pct <= 1.0:
                return 1.0 + _clamp((abs_pct - 0.1) / (1.0 - 0.1), 0.0, 1.0)

            return 2.0 + _clamp((abs_pct - 1.0) / (5.0 - 1.0), 0.0, 1.0)

    meter_m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*m[（(]", text)
    # 条件分岐: `meter_m` を満たす経路を評価する。
    if meter_m:
        try:
            meters = float(meter_m.group(1))
        except Exception:
            meters = None

        # 条件分岐: `meters is not None` を満たす経路を評価する。

        if meters is not None:
            # 0m => 0, 1m => 1, 2m => 2, 5m => 3 (clip)
            if meters <= 1.0:
                return _clamp(meters / 1.0, 0.0, 1.0)

            # 条件分岐: `meters <= 2.0` を満たす経路を評価する。

            if meters <= 2.0:
                return 1.0 + _clamp((meters - 1.0) / 1.0, 0.0, 1.0)

            return 2.0 + _clamp((meters - 2.0) / 3.0, 0.0, 1.0)

    return 1.5


def _table1_metric_score_norm_quantum(metric_public: str, metric_fallback: str, pmodel: str) -> Optional[float]:
    """
    Return a normalized discrepancy score in [0,3] (smaller is better) for Part III Table 1.

    Quantum rows are heterogeneous: some have explicit σ / z-scores, while others are
    "same-in-weak-field" mappings or "entrance constraints". For heatmap usability, we:
    - use |σ| / |z| when available
    - use max(Δ...) when the metric summarizes procedure sensitivity (e.g., Bell selection sweep)
    - fall back to a coarse categorical score inferred from the P-model column (green for "same",
      yellow for "entrance/target/constraint")
    """
    text = (metric_public or "").strip() or (metric_fallback or "").strip()
    # 条件分岐: `not text` を満たす経路を評価する。
    if not text:
        text = ""

    sigma_m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*σ", text)
    # 条件分岐: `sigma_m` を満たす経路を評価する。
    if sigma_m:
        try:
            abs_sigma = abs(float(sigma_m.group(1)))
            return _clamp(abs_sigma, 0.0, 3.0)
        except Exception:
            pass

    # Accept "z=..." when it represents a z-score (not cosmological redshift in this Part).

    z_m = re.search(r"\bz\s*=\s*([+-]?[0-9]+(?:\.[0-9]+)?)\b", text)
    # 条件分岐: `z_m` を満たす経路を評価する。
    if z_m:
        try:
            abs_z = abs(float(z_m.group(1)))
            return _clamp(abs_z, 0.0, 3.0)
        except Exception:
            pass

    # Procedure sensitivity (e.g., "Δ|S|≈...").

    delta_vals: List[float] = []
    for m in re.finditer(r"Δ[^0-9+\-]*([+-]?[0-9]+(?:\.[0-9]+)?)", text):
        try:
            delta_vals.append(abs(float(m.group(1))))
        except Exception:
            continue

    # 条件分岐: `delta_vals` を満たす経路を評価する。

    if delta_vals:
        # Roughly: 0.33 -> 1, 0.66 -> 2, 1.0 -> 3 (clip)
        return _clamp(max(delta_vals) / 0.33, 0.0, 3.0)

    # Coarse categorical fallback (keep "N/A" from looking identical to "mismatch").

    pm = (pmodel or "").strip()
    # 条件分岐: `pm` を満たす経路を評価する。
    if pm:
        # 条件分岐: `any(k in pm for k in ("同", "同スケール", "弱場写像", "整合", "ε=0"))` を満たす経路を評価する。
        if any(k in pm for k in ("同", "同スケール", "弱場写像", "整合", "ε=0")):
            return 0.5

        # 条件分岐: `any(k in pm for k in ("入口", "基準値", "ターゲット", "固定", "再導出", "必要条件", "制約"))` を満たす経路を評価する。

        if any(k in pm for k in ("入口", "基準値", "ターゲット", "固定", "再導出", "必要条件", "制約")):
            return 1.5

    return 1.5


def _render_table1_html_from_json(table1_json: Path, *, profile: str) -> str:
    """
    Render Table 1 (validation summary) as an HTML table with:
    - colored header row
    - colored "差/指標" cells based on discrepancy magnitude
    """
    try:
        payload = json.loads(table1_json.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return "<p class='muted'>[err] Table 1 JSON not readable.</p>"

    table1 = payload.get("table1") if isinstance(payload.get("table1"), dict) else {}
    rows = table1.get("rows") if isinstance(table1.get("rows"), list) else []
    notes = table1.get("notes") if isinstance(table1.get("notes"), list) else []

    headers = ["テーマ", "観測量/指標", "データ", "N", "参照", "P-model", "差/指標"]
    parts: List[str] = []
    parts.append("<div class='table-wrap'>")
    parts.append("<table class='table1'>")
    parts.append("<thead><tr>")
    for h in headers:
        parts.append(f"<th>{html.escape(h)}</th>")

    parts.append("</tr></thead>")
    parts.append("<tbody>")

    for r in rows:
        # 条件分岐: `not isinstance(r, dict)` を満たす経路を評価する。
        if not isinstance(r, dict):
            continue

        topic = str(r.get("topic") or "")
        observable = str(r.get("observable") or "")
        data = str(r.get("data") or "")
        n = "" if r.get("n") in (None, "") else str(r.get("n"))
        reference = str(r.get("reference") or "")
        pmodel = str(r.get("pmodel") or "")
        metric = str(r.get("metric") or "")
        metric_public = str(r.get("metric_public") or "").strip()

        metric_display = _strip_internal_source_tokens(metric_public or metric)

        score_norm: Optional[float]
        # 条件分岐: `profile == "part3_quantum"` を満たす経路を評価する。
        if profile == "part3_quantum":
            score_norm = _table1_metric_score_norm_quantum(metric_public, metric, pmodel)
        else:
            score_norm = _table1_metric_score_norm_astrophysics(metric_public, metric)

        style = ""
        # 条件分岐: `score_norm is not None` を満たす経路を評価する。
        if score_norm is not None:
            bg = _score_to_bg_hex(score_norm)
            style = f" style='background-color:{html.escape(bg)}'"

        parts.append("<tr>")
        parts.append(f"<td>{html.escape(topic)}</td>")
        parts.append(f"<td>{html.escape(observable)}</td>")
        parts.append(f"<td>{html.escape(data)}</td>")
        parts.append(f"<td style='text-align:right'>{html.escape(n)}</td>")
        parts.append(f"<td>{html.escape(reference)}</td>")
        parts.append(f"<td>{html.escape(pmodel)}</td>")
        parts.append(f"<td{style}>{html.escape(metric_display)}</td>")
        parts.append("</tr>")

    parts.append("</tbody></table></div>")

    # 条件分岐: `notes` を満たす経路を評価する。
    if notes:
        parts.append("<h4>注記</h4>")
        parts.append("<ul>")
        for nline in notes:
            parts.append(f"<li>{html.escape(str(nline))}</li>")

        parts.append("</ul>")

    return "\n".join(parts)


def _inject_table1_after_h3(body_html: str, *, insert_html: str) -> str:
    """
    Insert Table 1 right after "4.1 Table 1（検証サマリ）" heading inside the manuscript HTML.
    """
    # 条件分岐: `not body_html or not insert_html` を満たす経路を評価する。
    if not body_html or not insert_html:
        return body_html

    pat = re.compile(
        r'(<h3[^>]*id=\"section-4-1\"[^>]*>.*?Table 1（検証サマリ）.*?</h3>)',
        flags=re.S,
    )
    m = pat.search(body_html)
    # 条件分岐: `not m` を満たす経路を評価する。
    if not m:
        return body_html

    return body_html[: m.end(1)] + "\n" + insert_html + "\n" + body_html[m.end(1) :]


def _inject_pagebreak_before_heading_ids(body_html: str, *, heading_ids: Sequence[str]) -> str:
    """
    Insert explicit page-break markers before selected heading ids.
    Used to keep DOCX pagination deterministic for specific subsections.
    """
    # 条件分岐: `not body_html` を満たす経路を評価する。
    if not body_html:
        return body_html

    ids = [str(x).strip() for x in heading_ids if str(x).strip()]
    # 条件分岐: `not ids` を満たす経路を評価する。
    if not ids:
        return body_html

    out = body_html
    for hid in ids:
        pat = re.compile(rf'(<h[2-5][^>]*id="{re.escape(hid)}"[^>]*>)', flags=re.S)
        out = pat.sub(r"<div class='pb-before'></div>\n\1", out, count=1)

    return out


def _standardize_numbered_heading_ids(*, body_html: str, toc_html: str) -> Tuple[str, str]:
    """
    Normalize heading anchors like "#11" or "#1-introduction" into "#section-1-1", etc.

    This is intended to stabilize intra-doc navigation (especially for numbered Japanese headings).
    """

    # 条件分岐: `not body_html` を満たす経路を評価する。
    if not body_html:
        return body_html, toc_html

    def _heading_text(inner_html: str) -> str:
        t = re.sub(r"<[^>]+>", "", inner_html or "")
        t = html.unescape(t)
        t = t.replace("¶", "").strip()
        return t

    mapping: Dict[str, str] = {}
    used_new: set[str] = set()

    for m in _HEADING_RE.finditer(body_html):
        old_id = m.group(2)
        text = _heading_text(m.group(4))
        num = _HEADING_NUM_PREFIX_RE.match(text)
        # 条件分岐: `not num` を満たす経路を評価する。
        if not num:
            continue

        nums = [g for g in num.groups() if g is not None]
        # 条件分岐: `not nums` を満たす経路を評価する。
        if not nums:
            continue

        base = "section-" + "-".join(nums)
        new_id = base
        k = 2
        while new_id in used_new:
            new_id = f"{base}-{k}"
            k += 1

        mapping[old_id] = new_id
        used_new.add(new_id)

    # 条件分岐: `not mapping` を満たす経路を評価する。

    if not mapping:
        return body_html, toc_html

    def _replace(s: str) -> str:
        # 条件分岐: `not s` を満たす経路を評価する。
        if not s:
            return s

        for old, new in mapping.items():
            s = s.replace(f'id="{old}"', f'id="{new}"')
            s = s.replace(f'href="#{old}"', f'href="#{new}"')
            s = s.replace(f"href='#{old}'", f"href='#{new}'")

        return s

    return _replace(body_html), _replace(toc_html)


def _linkify_repo_paths(
    rendered_html: str,
    *,
    root: Path,
    out_dir: Path,
    fig_anchor_map: Optional[FigureAnchorMap] = None,
) -> str:
    """
    Convert <code>output/..</code> style snippets into clickable links when the target exists.
    Avoid touching code that doesn't look like a repo path.
    """

    def _candidate(s: str) -> Optional[str]:
        s = html.unescape(s).strip()
        s = s.replace("\\", "/")
        # 条件分岐: `s.startswith(("output/", "doc/", "scripts/", "data/"))` を満たす経路を評価する。
        if s.startswith(("output/", "doc/", "scripts/", "data/")):
            return s

        return None

    def repl(m: re.Match[str]) -> str:
        raw = m.group(1)
        cand = _candidate(raw)
        # 条件分岐: `not cand` を満たす経路を評価する。
        if not cand:
            return m.group(0)

        # 条件分岐: `" " in cand or "\n" in cand or "\r" in cand` を満たす経路を評価する。

        if " " in cand or "\n" in cand or "\r" in cand:
            return m.group(0)

        target = _resolve_repo_asset(cand, root=root)
        # 条件分岐: `not target.exists()` を満たす経路を評価する。
        if not target.exists():
            return m.group(0)
        # Stable figures: link to the in-page anchor (図番号) rather than opening the raw image.

        if fig_anchor_map and cand.endswith(".png") and cand in fig_anchor_map:
            anchor, label = fig_anchor_map[cand]
            return (
                f"<a href='{html.escape(anchor)}' title='{html.escape(label)}'><code>{html.escape(cand)}</code></a>"
                f"<span class='muted'>（{html.escape(label)}）</span>"
            )

        href = _rel_url(out_dir, target)
        return f"<a href='{html.escape(href)}'><code>{html.escape(cand)}</code></a>"

    return _CODE_RE.sub(repl, rendered_html)


_REF_KEY_RE = re.compile(r"^\s*-\s+\[([A-Za-z][A-Za-z0-9_-]{0,40})\]")


def _extract_reference_keys(md_text: str) -> List[str]:
    """
    Extract citation keys like [Will2014] from the references markdown.
    """
    keys: List[str] = []
    for line in md_text.splitlines():
        m = _REF_KEY_RE.match(line)
        # 条件分岐: `not m` を満たす経路を評価する。
        if not m:
            continue

        keys.append(m.group(1))
    # de-dup preserving order

    seen: set[str] = set()
    uniq: List[str] = []
    for k in keys:
        # 条件分岐: `k in seen` を満たす経路を評価する。
        if k in seen:
            continue

        seen.add(k)
        uniq.append(k)

    return uniq


_H2_SECTION_RE = re.compile(r"^\s*##\s+")


def _filter_md_h2_sections_by_ref_keys(md_text: str, *, keep_keys: set[str]) -> str:
    """
    Keep only H2 (## ...) sections that contain at least one '- [KEY]' bullet whose KEY is in keep_keys.

    This is used to tailor "データ出典（一次ソース）" / "参考文献" per Part, so Part II/III
    don't ship the full project-wide lists in publish output.
    """
    # 条件分岐: `not keep_keys` を満たす経路を評価する。
    if not keep_keys:
        return md_text

    lines = (md_text or "").splitlines()
    preamble: List[str] = []
    sections: List[Tuple[List[str], set[str]]] = []

    cur_lines: List[str] = []
    cur_keys: set[str] = set()

    def _flush() -> None:
        nonlocal cur_lines, cur_keys
        # 条件分岐: `not cur_lines` を満たす経路を評価する。
        if not cur_lines:
            return

        sections.append((cur_lines, cur_keys))
        cur_lines = []
        cur_keys = set()

    for line in lines:
        # 条件分岐: `_H2_SECTION_RE.match(line)` を満たす経路を評価する。
        if _H2_SECTION_RE.match(line):
            _flush()
            cur_lines = [line]
            cur_keys = set()
            continue

        # 条件分岐: `not cur_lines` を満たす経路を評価する。

        if not cur_lines:
            preamble.append(line)
            continue

        cur_lines.append(line)
        m = _REF_KEY_RE.match(line)
        # 条件分岐: `m` を満たす経路を評価する。
        if m:
            cur_keys.add(m.group(1))

    _flush()

    out: List[str] = list(preamble)
    for sec_lines, sec_keys in sections:
        # 条件分岐: `sec_keys & keep_keys` を満たす経路を評価する。
        if sec_keys & keep_keys:
            # 条件分岐: `out and out[-1].strip()` を満たす経路を評価する。
            if out and out[-1].strip():
                out.append("")

            out.extend(sec_lines)

    return "\n".join(out).rstrip() + "\n"


def _inject_reference_anchors(md_text: str) -> str:
    """
    Add stable in-page anchors so citations can link to the References section.

    - "[Will2014]" becomes "<a id='ref-Will2014'></a>[Will2014]" for bullet list items.
    """

    line_re = re.compile(r"^(\s*)-\s+\[([A-Za-z][A-Za-z0-9_-]{0,40})\]")

    def repl(m: re.Match[str]) -> str:
        indent = m.group(1)
        key = m.group(2)
        return f"{indent}- <a id='ref-{key}'></a>[{key}]"

    # Preserve indentation (nested bullets are common).

    out_lines: List[str] = []
    for line in md_text.splitlines():
        out_lines.append(line_re.sub(repl, line))

    return "\n".join(out_lines)


def _linkify_citations(rendered_html: str, *, ref_keys: Sequence[str]) -> str:
    """
    Convert "[Will2014]" into links to "#ref-Will2014" when the key exists in References.
    Avoid touching content inside <code>...</code>.
    """
    # 条件分岐: `not ref_keys` を満たす経路を評価する。
    if not ref_keys:
        return rendered_html

    ref_set = set(ref_keys)

    def repl(m: re.Match[str]) -> str:
        key = m.group(1)
        # 条件分岐: `key not in ref_set` を満たす経路を評価する。
        if key not in ref_set:
            return m.group(0)

        safe_key = html.escape(key)
        return f"<a href='#ref-{safe_key}'>[{safe_key}]</a>"

    parts: List[str] = []
    last = 0
    for cm in _CODE_RE.finditer(rendered_html):
        before = rendered_html[last : cm.start()]
        parts.append(_CITE_RE.sub(repl, before))
        parts.append(cm.group(0))
        last = cm.end()

    parts.append(_CITE_RE.sub(repl, rendered_html[last:]))
    return "".join(parts)


def _extract_png_paths_from_figures_index(root: Path) -> List[FigureItem]:
    """
    Pull stable figure PNG paths (and optional captions) from doc/paper/01_figures_index.md.

    Expected format (examples):
      - `output/foo/bar.png`（説明）
      - `output/foo/bar.png` (desc)
    """
    idx = root / "doc" / "paper" / "01_figures_index.md"
    # 条件分岐: `not idx.exists()` を満たす経路を評価する。
    if not idx.exists():
        return []

    text = _read_text(idx)
    found: List[FigureItem] = []
    current_section = ""

    def _strip_internal_step_refs(s: str) -> str:
        """
        Drop internal roadmap tokens like "Step 7.13.19.12" from captions.

        These IDs are useful in internal notes but should not leak into publish output.
        """
        # 条件分岐: `not s` を満たす経路を評価する。
        if not s:
            return s

        s = re.sub(r"（\s*Step\s+[0-9][0-9.\-]{0,40}\s*）", "", s)
        s = re.sub(r"\(\s*Step\s+[0-9][0-9.\-]{0,40}\s*\)", "", s)
        s = re.sub(r"\bStep\s+[0-9][0-9.\-]{0,40}\b", "", s)
        s = re.sub(r"[；;,:]\s*）", "）", s)
        s = re.sub(r"\s{2,}", " ", s)
        return s.strip()

    for line in text.splitlines():
        h = re.match(r"^\s*##\s+(.+?)\s*$", line)
        # 条件分岐: `h` を満たす経路を評価する。
        if h:
            current_section = h.group(1).strip()
            continue
        # Captions may themselves contain parentheses (e.g. "...（inlierのみ）"),
        # so avoid patterns that stop at the first closing bracket.

        m = re.search(r"`(output/[^`]+?\.png)`\s*(?:（(.{0,200})）|\((.{0,200})\))?", line)
        # 条件分岐: `not m` を満たす経路を評価する。
        if not m:
            continue

        rel = m.group(1)
        caption = _strip_internal_step_refs((m.group(2) or m.group(3) or "").strip())
        p = _resolve_repo_asset(rel, root=root)
        # 条件分岐: `not p.exists()` を満たす経路を評価する。
        if not p.exists():
            continue

        found.append(FigureItem(section=current_section, rel=rel, path=p, caption=caption))

    # de-dup while preserving order (keep first caption)

    uniq: List[FigureItem] = []
    seen: set[str] = set()
    for fig in found:
        key = fig.rel.lower()
        # 条件分岐: `key in seen` を満たす経路を評価する。
        if key in seen:
            continue

        seen.add(key)
        uniq.append(fig)

    return uniq


def _build_fig_anchor_map(figs: List[FigureItem], *, root: Path) -> FigureAnchorMap:
    """
    Map `output/...png` (repo-relative, POSIX-style) -> ("#fig-001", "図1").
    """
    fig_map: FigureAnchorMap = {}
    fig_no = 0
    for fig in figs:
        fig_no += 1
        fig_id = f"fig-{fig_no:03d}"
        fig_label = f"図{fig_no}"
        fig_map[fig.rel] = (f"#{fig_id}", fig_label)

    return fig_map


_PNG_REL_RE = re.compile(r"(output/[A-Za-z0-9_./-]+?\.png)")


def _extract_output_png_relpaths_in_order(md_text: str) -> List[str]:
    """
    Extract repo-relative `output/...png` paths in order of first appearance.
    """
    out: List[str] = []
    seen: set[str] = set()
    for m in _PNG_REL_RE.finditer(md_text or ""):
        rel = (m.group(1) or "").strip().replace("\\", "/")
        # 条件分岐: `not rel` を満たす経路を評価する。
        if not rel:
            continue

        # 条件分岐: `rel in seen` を満たす経路を評価する。

        if rel in seen:
            continue

        seen.add(rel)
        out.append(rel)

    return out


def _reorder_figs_by_reference_order(
    figs: List[FigureItem],
    *,
    root: Path,
    reference_relpaths: Sequence[str],
) -> List[FigureItem]:
    """
    Reorder figures so that numbering becomes ascending in the paper reading order.

    - Primary order: first appearance in the paper markdown ("reference_relpaths").
    - Remaining figures from the index are appended in the original index order.
    - If a referenced PNG exists on disk but is missing from the figures index, include it (caption="").
    """
    by_rel: Dict[str, FigureItem] = {}
    for fig in figs:
        # 条件分岐: `fig.rel in by_rel` を満たす経路を評価する。
        if fig.rel in by_rel:
            continue

        by_rel[fig.rel] = fig

    ordered: List[FigureItem] = []
    used: set[str] = set()

    for rel in reference_relpaths:
        # 条件分岐: `rel in used` を満たす経路を評価する。
        if rel in used:
            continue

        # 条件分岐: `rel in by_rel` を満たす経路を評価する。

        if rel in by_rel:
            ordered.append(by_rel[rel])
            used.add(rel)
            continue

        p = _resolve_repo_asset(rel, root=root)
        # 条件分岐: `p.exists() and p.is_file()` を満たす経路を評価する。
        if p.exists() and p.is_file():
            ordered.append(FigureItem(section="", rel=rel, path=p, caption=""))
            used.add(rel)

    for fig in figs:
        # 条件分岐: `fig.rel in used` を満たす経路を評価する。
        if fig.rel in used:
            continue

        ordered.append(fig)
        used.add(fig.rel)

    return ordered


_INTERNAL_BLOCK_RE = re.compile(
    r"<!--\s*INTERNAL_ONLY_START\s*-->.*?<!--\s*INTERNAL_ONLY_END\s*-->",
    flags=re.DOTALL,
)


def _strip_internal_blocks(md_text: str, *, mode: str) -> str:
    """
    publish モードでは、Markdown中の “内部向けブロック” を紙面から除外する。
    """
    # 条件分岐: `mode != "publish"` を満たす経路を評価する。
    if mode != "publish":
        return md_text

    return _INTERNAL_BLOCK_RE.sub("", md_text)


_INLINE_MATH_PARITY_RE = re.compile(r"(?<!\\)(?<!\$)\$(?!\$)([^$\n]+?)(?<!\\)\$(?!\$)")
_CJK_CHAR_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")


def _normalize_inline_markdown_segment_for_parity(text: str) -> str:
    """
    Apply the same inline math heuristics used by paper_latex so publish HTML/DOCX
    stays aligned with TeX output for Part I-IV.
    """

    token_map: Dict[str, str] = {}
    token_index = 0

    def make_token(rendered: str) -> str:
        nonlocal token_index
        key = f"@@PHTMLTOK{token_index}@@"
        token_map[key] = rendered
        token_index += 1
        return key

    def repl_inline_code(match: re.Match[str]) -> str:
        payload = (match.group(1) or "").strip()
        # 条件分岐: `not payload` を満たす経路を評価する。
        if not payload:
            return make_token(match.group(0))

        # 条件分岐: `_paper_latex._PUNCT_ONLY_RE.fullmatch(payload)` を満たす経路を評価する。

        if _paper_latex._PUNCT_ONLY_RE.fullmatch(payload):
            return make_token(match.group(0))

        # 条件分岐: `_CJK_CHAR_RE.search(payload) and not _paper_latex._looks_like_artifact_code(p...` を満たす経路を評価する。

        if _CJK_CHAR_RE.search(payload) and not _paper_latex._looks_like_artifact_code(payload):
            return make_token(match.group(0))

        if (
            _paper_latex._looks_like_physics_equation_code(payload)
            or _paper_latex._looks_like_physics_symbol_code(payload)
            or _paper_latex._looks_like_math_code(payload)
        ):
            return make_token("$" + _paper_latex._normalize_inline_math_payload(payload) + "$")

        return make_token(match.group(0))

    normalized = re.sub(r"`([^`]+)`", repl_inline_code, text)

    def repl_inline_math(match: re.Match[str]) -> str:
        payload = (match.group(1) or "").strip()
        # 条件分岐: `not payload` を満たす経路を評価する。
        if not payload:
            return match.group(0)

        if _paper_latex._looks_like_artifact_code(payload) and not (
            _paper_latex._looks_like_physics_equation_code(payload)
            or _paper_latex._looks_like_physics_symbol_code(payload)
        ):
            return make_token(match.group(0))

        return make_token("$" + _paper_latex._normalize_inline_math_payload(payload) + "$")

    normalized = _INLINE_MATH_PARITY_RE.sub(repl_inline_math, normalized)
    normalized = _paper_latex._replace_plain_symbolic_tokens(normalized, make_token)

    for _ in range(len(token_map) + 1):
        changed = False
        for key, rendered in token_map.items():
            # 条件分岐: `key in normalized` を満たす経路を評価する。
            if key in normalized:
                normalized = normalized.replace(key, rendered)
                changed = True

        # 条件分岐: `not changed` を満たす経路を評価する。

        if not changed:
            break

    return normalized


def _normalize_markdown_for_tex_docx_parity(md_text: str) -> str:
    """
    Normalize inline pseudo-math in markdown prose so HTML/DOCX follows the same
    math tokenization rules as TeX generation.
    """

    lines = md_text.splitlines()
    out_lines: List[str] = []
    in_code_fence = False
    in_math_block = False

    for line in lines:
        stripped = line.strip()
        # 条件分岐: `stripped.startswith("```")` を満たす経路を評価する。
        if stripped.startswith("```"):
            in_code_fence = not in_code_fence
            out_lines.append(line)
            continue

        # 条件分岐: `in_code_fence` を満たす経路を評価する。

        if in_code_fence:
            out_lines.append(line)
            continue

        parts = line.split("$$")
        # 条件分岐: `len(parts) == 1` を満たす経路を評価する。
        if len(parts) == 1:
            # 条件分岐: `in_math_block` を満たす経路を評価する。
            if in_math_block:
                out_lines.append(line)
            else:
                out_lines.append(_normalize_inline_markdown_segment_for_parity(line))

            continue

        cur_in_math = in_math_block
        rewritten_parts: List[str] = []
        for idx, part in enumerate(parts):
            # 条件分岐: `cur_in_math` を満たす経路を評価する。
            if cur_in_math:
                rewritten_parts.append(part)
            else:
                rewritten_parts.append(_normalize_inline_markdown_segment_for_parity(part))

            # 条件分岐: `idx != len(parts) - 1` を満たす経路を評価する。

            if idx != len(parts) - 1:
                cur_in_math = not cur_in_math

        out_lines.append("$$".join(rewritten_parts))
        in_math_block = cur_in_math

    return "\n".join(out_lines)


_MATH_BLOCK_RE = re.compile(r"\$\$(.+?)\$\$", flags=re.DOTALL)
_INLINE_MATH_RE = re.compile(r"(?<!\\)(?<!\$)\$(?!\$)([^$\n]+?)(?<!\\)\$(?!\$)")


def _read_bytes(path: Path) -> bytes:
    return path.read_bytes()


def _data_uri_png(png_bytes: bytes) -> str:
    b64 = base64.b64encode(png_bytes).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _render_equation_png(*, latex: str, eq_dir: Path, inline: bool = False) -> Path:
    """
    Render a LaTeX math string to a PNG file (white background, tight bbox).

    - matplotlib mathtext を使用（外部LaTeX不要）
    - 添付例（式.png）の見た目に近づけるため、CMフォント＋白背景＋余白少なめ
    """
    eq_dir.mkdir(parents=True, exist_ok=True)
    # NOTE:
    # 本プロジェクトの Markdown は、LaTeX の `\phi` を `\\phi` のように
    # “バックスラッシュを二重化” して書いている箇所がある。
    # matplotlib mathtext は `\\phi` を改行コマンド `\\` と解釈して壊れるため、
    # ここで正規化してから描画する。
    latex_norm = latex.strip().replace("\r\n", "\n")
    # Normalize only duplicated macro-introducing backslashes (e.g. "\\phi" -> "\phi").
    # Keep row-separator style "\\ " untouched for matrix-like expressions.
    latex_norm = re.sub(r"\\\\(?=[A-Za-z])", r"\\", latex_norm)
    # matplotlib mathtext does not support \begin{...}\end{...} environments.
    # Convert common matrix environments to bracketed lists.
    def _matrix_env_repl(m: re.Match[str]) -> str:
        env = str(m.group(1) or "").strip().lower()
        body = str(m.group(2) or "")
        body = body.replace("\n", " ")
        body = re.sub(r"\\\\+", " ; ", body)
        body = body.replace("&", " , ")
        body = re.sub(r"\s+", " ", body).strip(" ;,")
        # 条件分岐: `env == "pmatrix"` を満たす経路を評価する。
        if env == "pmatrix":
            return rf"\left({body}\right)"

        return rf"\left[{body}\right]"

    latex_norm = re.sub(
        r"\\begin\{(bmatrix|pmatrix|matrix|array)\}(?:\{[^{}]*\})?(.*?)\\end\{\1\}",
        _matrix_env_repl,
        latex_norm,
        flags=re.DOTALL,
    )
    # matplotlib mathtext does not support some LaTeX delimiter aliases.
    # Normalize common variants to equivalent tokens that mathtext can render.
    latex_norm = (
        latex_norm.replace("\\lvert", "|")
        .replace("\\rvert", "|")
        .replace("\\lVert", "\\|")
        .replace("\\rVert", "\\|")
    )
    # matplotlib mathtext has limited macro support and can fail on
    # vector/bold commands often used in the manuscript.
    # Normalize unsupported wrappers to plain symbols while preserving semantics.
    for macro in ("mathbf", "boldsymbol", "bm"):
        latex_norm = re.sub(rf"\\{macro}\s*\{{([^{{}}]+)\}}", r"{\1}", latex_norm)
        latex_norm = re.sub(rf"\\{macro}\s+([A-Za-z])", r"{\1}", latex_norm)
        latex_norm = re.sub(rf"\\{macro}(?=\\[A-Za-z]+)", "", latex_norm)

    latex_norm = " ".join(latex_norm.split())

    # NOTE: Rendering parameters must be part of the cache key, otherwise changing
    # font size / DPI would not invalidate existing PNGs.
    eq_dpi = 110 if inline else 300
    eq_fontsize = 4.5 if inline else 6.5
    pad_inches = 0.0 if inline else 0.08
    render_mode = "inline" if inline else "block"
    style_version = f"eqpng_v4|mode={render_mode}|dpi={eq_dpi}|fs={eq_fontsize}|pad={pad_inches}"

    key = hashlib.sha1(f"{style_version}\n{latex_norm}".encode("utf-8")).hexdigest()[:12]
    out = eq_dir / f"eq_{key}.png"
    # 条件分岐: `out.exists()` を満たす経路を評価する。
    if out.exists():
        return out

    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import rcParams

    rcParams["mathtext.fontset"] = "cm"
    rcParams["font.family"] = "serif"

    import matplotlib.pyplot as plt  # type: ignore

    fig = plt.figure(figsize=(0.01, 0.01), dpi=eq_dpi)
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.set_facecolor("white")
    ax.text(
        0.5,
        0.5,
        f"${latex_norm}$",
        fontsize=eq_fontsize,
        ha="center",
        va="center",
        color="black",
    )

    buf = io.BytesIO()
    try:
        fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=pad_inches, facecolor="white")
        plt.close(fig)
    except Exception as exc:
        plt.close(fig)
        print(f"[warn] equation render fallback (mathtext): {exc!r}")
        fig = plt.figure(figsize=(0.01, 0.01), dpi=eq_dpi)
        fig.patch.set_facecolor("white")
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")
        ax.set_facecolor("white")
        fallback_text = " ".join(latex_norm.split())
        ax.text(0.5, 0.5, fallback_text, fontsize=eq_fontsize, ha="center", va="center", color="black")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=pad_inches, facecolor="white")
        plt.close(fig)

    out.write_bytes(buf.getvalue())
    return out


def _replace_math_blocks_with_images(
    md_text: str,
    *,
    out_dir: Path,
    mode: str,
    embed_images: bool,
) -> str:
    """
    $$ ... $$ のブロック数式と $ ... $ のインライン数式を “数式画像” に置換する（publishモードのみ）。
    """
    # 条件分岐: `mode != "publish"` を満たす経路を評価する。
    if mode != "publish":
        return md_text

    eq_dir = out_dir / "equations"

    def repl(m: re.Match[str]) -> str:
        latex = m.group(1).strip()
        # 条件分岐: `not latex` を満たす経路を評価する。
        if not latex:
            return ""

        png_path = _render_equation_png(latex=latex, eq_dir=eq_dir, inline=False)
        # 条件分岐: `embed_images` を満たす経路を評価する。
        if embed_images:
            src = _data_uri_png(_read_bytes(png_path))
        else:
            src = _rel_url(out_dir, png_path)
        # LaTeX のバックスラッシュ表記（例: `\\gamma`）は本文外の属性でも目立つため、publish では汎用の代替文言にする。

        alt = "数式"
        latex_attr = html.escape(latex, quote=True)
        return (
            "<div class='equation-block'>"
            f"<img class='equation-img' src='{src}' alt='{alt}' data-latex='{latex_attr}'>"
            "</div>"
        )

    replaced = _MATH_BLOCK_RE.sub(repl, md_text)

    def _looks_like_inline_math(expr: str) -> bool:
        s = (expr or "").strip()
        # 条件分岐: `not s` を満たす経路を評価する。
        if not s:
            return False
        # Avoid replacing plain currency-like values (e.g., "$5", "$12.3%").

        if re.fullmatch(r"[0-9]+(?:[.,][0-9]+)?%?", s):
            return False
        # Treat almost all remaining `$...$` as math.
        # This intentionally includes simple variables such as `$e$`, `$R$`, `$m1$`
        # so TeX-like markers never leak into Word as raw text.

        return True

    def repl_inline(m: re.Match[str]) -> str:
        latex = (m.group(1) or "").strip()
        # 条件分岐: `not _looks_like_inline_math(latex)` を満たす経路を評価する。
        if not _looks_like_inline_math(latex):
            return m.group(0)

        png_path = _render_equation_png(latex=latex, eq_dir=eq_dir, inline=True)
        # 条件分岐: `embed_images` を満たす経路を評価する。
        if embed_images:
            src = _data_uri_png(_read_bytes(png_path))
        else:
            src = _rel_url(out_dir, png_path)

        latex_attr = html.escape(latex, quote=True)
        return (
            "<span class='equation-inline'>"
            f"<img class='equation-inline-img' src='{src}' alt='数式-inline' data-latex='{latex_attr}' style='height:0.95em;width:auto;vertical-align:middle;'>"
            "</span>"
        )

    return _INLINE_MATH_RE.sub(repl_inline, replaced)


def _inline_png_code_snippets(
    rendered_html: str,
    *,
    root: Path,
    out_dir: Path,
    figs: List[FigureItem],
    mode: str,
    embed_images: bool,
    fig_map: Optional[Dict[str, Dict[str, Any]]] = None,
    inlined: Optional[set[str]] = None,
    img_cache: Optional[Dict[str, str]] = None,
) -> str:
    """
    publish モード向け：
    本文中の `<code>output/...png</code>` を “図（画像）” としてインライン挿入し、
    以後の参照は図番号（アンカー）に変換する。

    これにより、紙面からファイルパス文字列を除去できる。
    """
    # 条件分岐: `mode != "publish"` を満たす経路を評価する。
    if mode != "publish":
        return _rewrite_repo_relative_asset_urls(rendered_html, root=root, out_dir=out_dir)

    # 条件分岐: `fig_map is None` を満たす経路を評価する。

    if fig_map is None:
        fig_map = {}
        # 条件分岐: `figs` を満たす経路を評価する。
        if figs:
            fig_no = 0
            for fig in figs:
                fig_no += 1
                fig_id = f"fig-{fig_no:03d}"
                fig_label = f"図{fig_no}"
                fig_map[fig.rel] = {
                    "id": fig_id,
                    "label": fig_label,
                    "caption": fig.caption or "",
                    "path": fig.path,
                }

    # 条件分岐: `inlined is None` を満たす経路を評価する。

    if inlined is None:
        inlined = set()

    # 条件分岐: `img_cache is None` を満たす経路を評価する。

    if img_cache is None:
        img_cache = {}

    def _candidate(s: str) -> Optional[str]:
        s = html.unescape(s).strip().replace("\\", "/")
        # 条件分岐: `s.startswith("output/") and s.endswith(".png")` を満たす経路を評価する。
        if s.startswith("output/") and s.endswith(".png"):
            return s

        return None

    def repl(m: re.Match[str]) -> str:
        raw = m.group(1)
        cand = _candidate(raw)
        # 条件分岐: `not cand` を満たす経路を評価する。
        if not cand:
            return m.group(0)

        info = fig_map.get(cand)
        # 条件分岐: `not info` を満たす経路を評価する。
        if not info:
            return m.group(0)

        fig_id = str(info["id"])
        fig_label = str(info["label"])
        caption = str(info["caption"])
        p: Path = info["path"]

        # 2回目以降は図番号参照のみ
        if cand in inlined:
            return f"<a href='#{html.escape(fig_id)}'>{html.escape(fig_label)}</a>"

        inlined.add(cand)

        # 条件分岐: `embed_images` を満たす経路を評価する。
        if embed_images:
            cache_key = str(p).lower()
            # 条件分岐: `cache_key not in img_cache` を満たす経路を評価する。
            if cache_key not in img_cache:
                img_cache[cache_key] = _data_uri_png(_read_bytes(p))

            src = img_cache[cache_key]
        else:
            src = _rel_url(out_dir, p)

        cap = html.escape(caption) if caption else ""
        # 条件分岐: `cap` を満たす経路を評価する。
        if cap:
            figcap = f"<figcaption><strong>{html.escape(fig_label)}:</strong> {cap}</figcaption>"
        else:
            figcap = f"<figcaption><strong>{html.escape(fig_label)}</strong></figcaption>"

        return (
            f"<figure class='inline-figure' id='{html.escape(fig_id)}'>"
            f"{figcap}"
            f"<a class='fig-link' href='{src}' target='_blank' rel='noopener noreferrer'>"
            f"<img src='{src}' loading='lazy'>"
            f"</a>"
             f"</figure>"
         )

    out = _CODE_RE.sub(repl, rendered_html)

    # Word HTML import and some browsers get confused when <figure> is nested in a <p>.
    # Also, "図：図1:" is visually redundant (figcaption already contains the numbering).
    out = _INLINE_FIGURE_LABEL_P_RE.sub(r"\1", out)
    out = _INLINE_FIGURE_LABEL_RE.sub("", out)
    out = _INLINE_FIGURE_P_RE.sub(r"\1", out)

    # Word import sometimes ignores CSS padding; add HTML attributes so tables keep spacing.
    # (Paper style requirement: no visible borders.)
    out = _TABLE_OPEN_NO_BORDER_RE.sub(r"<table border='0' cellspacing='0' cellpadding='4'\1>", out)

    return _rewrite_repo_relative_asset_urls(out, root=root, out_dir=out_dir)


def _render_html(
    *,
    out_dir: Path,
    out_name: str,
    title: str,
    subtitle: str,
    header_badge: str,
    manuscript_link: str,
    sections: List[Dict[str, Any]],
    mode: str,
    toc_mode: str = "section_details",
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    html_path = out_dir / str(out_name)

    parts: List[str] = []
    parts.append("<!doctype html>")
    parts.append("<html lang='ja'><head>")
    parts.append("<meta charset='utf-8'>")
    parts.append("<meta name='viewport' content='width=device-width, initial-scale=1'>")
    parts.append(f"<title>{html.escape(title)}</title>")
    parts.append("<style>")
    parts.append(
        "body{font-family:Yu Gothic,Meiryo,BIZ UDGothic,MS Gothic,system-ui,sans-serif;margin:24px;max-width:1200px;background:#fff}"
        "h1{margin:0 0 6px 0;font-size:24px;line-height:1.25}"
        "h2{margin:22px 0 10px 0;font-size:20px;line-height:1.3}"
        "h3{margin:18px 0 6px 0;font-size:18px;line-height:1.35}"
        "h4{margin:16px 0 6px 0;font-size:16px;line-height:1.35}"
        "h5{margin:14px 0 6px 0;font-size:14px;line-height:1.35}"
        "h6{margin:12px 0 6px 0;font-size:13px;line-height:1.35}"
        ".muted{color:#666;font-size:13px}"
        ".card{border:none;border-radius:10px;padding:14px 16px;margin:14px 0;background:#fff}"
        ".meta{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin:4px 0 10px 0}"
        ".badge{font-size:12px;padding:2px 8px;border-radius:999px;background:transparent;color:#333}"
        "ul{margin:6px 0 10px 18px}"
        "code{font-family:ui-monospace,Consolas,Menlo,monospace;font-size:0.95em}"
        "pre{background:#fff;color:#111;border-radius:0;padding:8px 10px;overflow:auto}"
        "pre code{color:inherit}"
        "img{max-width:100%;height:auto;border:none;border-radius:8px}"
        "figure{margin:14px 0}"
        "figcaption{margin:0 0 6px 0}"
        ".equation-block{overflow-x:auto;overflow-y:hidden;margin:6px 0;padding:2px 0}"
        ".equation-img{border:none;border-radius:0;display:block;margin:0 auto;max-width:none;height:auto}"
        ".equation-inline{display:inline-block;vertical-align:middle;line-height:1}"
        ".equation-inline-img{border:none;border-radius:0;display:inline-block;vertical-align:middle;max-width:none;height:1.25em}"
        ".inline-figure{margin:14px 0 18px 0}"
        ".inline-figure img{width:100%}"
        ".fig-link{display:block}"
        ".pb-before{break-before:page;page-break-before:always;height:0;margin:0;padding:0}"
        ".table-wrap{overflow-x:auto;margin:8px 0 10px 0}"
        "table{border-collapse:separate;border-spacing:0;width:100%;font-size:12px}"
        "th,td{border:none;padding:6px 8px;vertical-align:top}"
        "th{background:#f2f7ff;text-align:left}"
        "a{color:#0b65c2;text-decoration:none}"
        "a:hover{text-decoration:underline}"
        ".toc{font-size:13px}"
        ".toc ul{margin-left:18px}"
        "@media print{body{margin:0;max-width:none} .card{break-inside:avoid-page;page-break-inside:avoid} .equation-block{overflow-x:visible} .equation-img{max-width:100%} .pb-before{break-before:page;page-break-before:always} details{display:none}}"
    )
    parts.append("</style>")
    parts.append("</head><body>")

    parts.append(f"<h1>{html.escape(title)}</h1>")
    parts.append(f"<div class='muted'>{html.escape(subtitle)}</div>")

    parts.append("<div class='card'>")
    parts.append("<div class='meta'>")
    parts.append(f"<span class='badge'>{html.escape(header_badge)}</span>")
    parts.append(f"<span class='muted'>生成（UTC）: {html.escape(_iso_utc_now())}</span>")
    parts.append("</div>")
    # 条件分岐: `mode != "publish"` を満たす経路を評価する。
    if mode != "publish":
        parts.append("<ul>")
        parts.append("<li>一般向け統一レポート: <a href='pmodel_public_report.html'><code>output/private/summary/pmodel_public_report.html</code></a></li>")
        parts.append(
            "<li>Markdown草稿: "
            f"<a href='../../{html.escape(manuscript_link)}'><code>{html.escape(manuscript_link)}</code></a>"
            "</li>"
        )
        parts.append("</ul>")

    parts.append("</div>")

    # TOC
    parts.append("<h2>目次</h2><ul>")
    for sec in sections:
        sid = str(sec.get("id") or "")
        stitle = str(sec.get("title") or "")
        toc_html = str(sec.get("toc_html") or "")

        # Preserve legacy behavior: in per-section-TOC mode, "本文" is navigated via the card-local toc.
        if toc_mode != "global_only" and sid == "manuscript":
            continue

        parts.append(f"<li><a href='#{html.escape(sid)}'>{html.escape(stitle)}</a>")
        # 条件分岐: `toc_mode == "global_only" and sid == "manuscript" and toc_html` を満たす経路を評価する。
        if toc_mode == "global_only" and sid == "manuscript" and toc_html:
            parts.append(f"{toc_html}")

        parts.append("</li>")

    parts.append("</ul>")

    for sec in sections:
        sid = str(sec.get("id") or "")
        stitle = str(sec.get("title") or "")
        badge = str(sec.get("badge") or "")
        body_html = str(sec.get("body_html") or "")
        toc_html = str(sec.get("toc_html") or "")

        parts.append(f"<div class='card' id='{html.escape(sid)}'>")
        parts.append("<div class='meta'>")
        # 条件分岐: `badge` を満たす経路を評価する。
        if badge:
            parts.append(f"<span class='badge'>{html.escape(badge)}</span>")

        parts.append(f"<h2 style='border:none;margin:0'>{html.escape(stitle)}</h2>")
        parts.append("</div>")

        # 条件分岐: `toc_html and toc_mode != "global_only"` を満たす経路を評価する。
        if toc_html and toc_mode != "global_only":
            parts.append("<details>")
            parts.append("<summary class='muted'>この章の目次</summary>")
            parts.append(f"<div class='toc'>{toc_html}</div>")
            parts.append("</details>")

        parts.append(body_html)
        parts.append("</div>")

    parts.append("</body></html>")

    html_path.write_text("\n".join(parts), encoding="utf-8")
    return html_path


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Render paper (Markdown draft) as a single HTML page.")
    ap.add_argument("--outdir", default=None, help="Output directory (default: output/private/summary).")
    ap.add_argument(
        "--profile",
        choices=["paper", "part2_astrophysics", "part3_quantum", "part4_verification"],
        default="paper",
        help="render profile: paper (Part I) / part2_astrophysics / part3_quantum / part4_verification",
    )
    ap.add_argument(
        "--mode",
        choices=["publish", "internal"],
        default="publish",
        help="render mode: publish (no file links, equations as images) or internal",
    )
    ap.add_argument(
        "--manuscript",
        default=None,
        help="manuscript markdown path (default depends on --profile)",
    )
    ap.add_argument(
        "--out-name",
        default=None,
        help="output HTML filename (default depends on --profile)",
    )
    ap.add_argument(
        "--no-embed-images",
        action="store_true",
        help="do not embed images as data URI (publish mode). use relative file paths instead.",
    )
    ap.add_argument("--open", action="store_true", help="Open the HTML after generation (Windows only).")
    args = ap.parse_args(argv)

    root = _repo_root()
    out_dir = Path(args.outdir) if args.outdir else (root / "output" / "private" / "summary")
    mode = str(args.mode)
    profile = str(args.profile)
    embed_images = (mode == "publish") and (not bool(args.no_embed_images))

    # 条件分岐: `args.manuscript` を満たす経路を評価する。
    if args.manuscript:
        manuscript_md = (root / Path(str(args.manuscript))).resolve()
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
        # 条件分岐: 前段条件が不成立で、`profile == "part4_verification"` を追加評価する。
        elif profile == "part4_verification":
            manuscript_md = root / "doc" / "paper" / "13_part4_verification.md"

    quantum_appendix_a_md = root / "doc" / "paper" / "12_part3_quantum_appendix_a.md"
    definitions_md = root / "doc" / "paper" / "05_definitions.md"
    uncertainty_md = root / "doc" / "paper" / "06_uncertainty.md"
    llr_appendix_md = root / "doc" / "paper" / "07_llr_appendix.md"
    table1_astrophysics_md = root / "output" / "private" / "summary" / "paper_table1_results.md"
    table1_quantum_md = root / "output" / "private" / "summary" / "paper_table1_quantum_results.md"
    table1_md = table1_quantum_md if profile == "part3_quantum" else table1_astrophysics_md
    sources_md = root / "doc" / "paper" / "20_data_sources.md"
    refs_md = root / "doc" / "paper" / "30_references.md"

    ref_keys: List[str] = []
    # 条件分岐: `refs_md.exists()` を満たす経路を評価する。
    if refs_md.exists():
        try:
            ref_keys = _extract_reference_keys(_read_text(refs_md))
        except Exception:
            ref_keys = []

    # Figure numbering policy:
    # Assign "図1, 図2, ..." in the first-reference order (reading order) so numbers are ascending in the manuscript.
    # Captions are taken from doc/paper/01_figures_index.md when available.

    include_table1 = profile in {"part2_astrophysics", "part3_quantum"}
    include_definitions = profile == "paper"
    include_uncertainty = False
    include_llr_appendix = profile == "part2_astrophysics"
    # Part III の Si α(T) ansatzログ（旧：補遺A）は本文 4.2.13 に統合したため、補遺セクションは生成しない。
    include_quantum_appendix_a = False
    enable_citation_links = profile != "paper"
    # Keep anchors stable across all paper profiles so intra-doc links and TOC are consistent.
    standardize_numbered_anchors = profile in {"paper", "part2_astrophysics", "part3_quantum", "part4_verification"}

    # Used to tailor "データ出典/参考文献" per Part in publish mode.
    used_cite_keys: set[str] = set()
    cite_scan_paths: List[Path] = []
    # 条件分岐: `include_table1` を満たす経路を評価する。
    if include_table1:
        cite_scan_paths.append(table1_md)

    # 条件分岐: `include_definitions` を満たす経路を評価する。

    if include_definitions:
        cite_scan_paths.append(definitions_md)

    # 条件分岐: `include_uncertainty` を満たす経路を評価する。

    if include_uncertainty:
        cite_scan_paths.append(uncertainty_md)

    cite_scan_paths.append(manuscript_md)
    # 条件分岐: `include_quantum_appendix_a` を満たす経路を評価する。
    if include_quantum_appendix_a:
        cite_scan_paths.append(quantum_appendix_a_md)

    # 条件分岐: `include_llr_appendix` を満たす経路を評価する。

    if include_llr_appendix:
        cite_scan_paths.append(llr_appendix_md)

    for p in cite_scan_paths:
        # 条件分岐: `not p.exists()` を満たす経路を評価する。
        if not p.exists():
            continue

        try:
            t = _read_text(p)
            t = _strip_internal_blocks(t, mode=mode)
            used_cite_keys |= set(_CITE_RE.findall(t))
        except Exception:
            continue

    scan_paths: List[Path] = [table1_md] if include_table1 else []
    # 条件分岐: `include_definitions` を満たす経路を評価する。
    if include_definitions:
        scan_paths.append(definitions_md)

    # 条件分岐: `include_uncertainty` を満たす経路を評価する。

    if include_uncertainty:
        scan_paths.append(uncertainty_md)

    scan_paths.append(manuscript_md)
    # 条件分岐: `include_quantum_appendix_a` を満たす経路を評価する。
    if include_quantum_appendix_a:
        scan_paths.append(quantum_appendix_a_md)

    # 条件分岐: `include_llr_appendix` を満たす経路を評価する。

    if include_llr_appendix:
        scan_paths.append(llr_appendix_md)

    scan_paths += [sources_md, refs_md]
    scan_texts: List[str] = []
    for p in scan_paths:
        # 条件分岐: `not p.exists()` を満たす経路を評価する。
        if not p.exists():
            continue

        try:
            t = _read_text(p)
            t = _strip_internal_blocks(t, mode=mode)
            scan_texts.append(t)
        except Exception:
            continue

    reference_relpaths = _extract_output_png_relpaths_in_order("\n".join(scan_texts))

    figs_idx = _extract_png_paths_from_figures_index(root)
    # 条件分岐: `figs_idx` を満たす経路を評価する。
    if figs_idx:
        figs = _reorder_figs_by_reference_order(figs_idx, root=root, reference_relpaths=reference_relpaths)
    else:
        # Fallback: still inline referenced figures in publish mode even if the index is missing.
        figs = []
        seen_rel: set[str] = set()
        for rel in reference_relpaths:
            # 条件分岐: `rel in seen_rel` を満たす経路を評価する。
            if rel in seen_rel:
                continue

            seen_rel.add(rel)
            p = _resolve_repo_asset(rel, root=root)
            # 条件分岐: `p.exists() and p.is_file()` を満たす経路を評価する。
            if p.exists() and p.is_file():
                figs.append(FigureItem(section="", rel=rel, path=p, caption=""))

    fig_anchor_map = _build_fig_anchor_map(figs, root=root) if figs else None

    publish_fig_map: Optional[Dict[str, Dict[str, Any]]] = None
    publish_inlined: Optional[set[str]] = None
    publish_img_cache: Optional[Dict[str, str]] = None
    # 条件分岐: `mode == "publish" and figs` を満たす経路を評価する。
    if mode == "publish" and figs:
        publish_fig_map = {}
        fig_no = 0
        for fig in figs:
            fig_no += 1
            fig_id = f"fig-{fig_no:03d}"
            fig_label = f"図{fig_no}"
            publish_fig_map[fig.rel] = {
                "id": fig_id,
                "label": fig_label,
                "caption": fig.caption or "",
                "path": fig.path,
            }

        publish_inlined = set()
        publish_img_cache = {}

    sections: List[Dict[str, Any]] = []

    # NOTE:
    # For Part II / Part III, Table 1 is injected under "4.1 Table 1（検証サマリ）" inside the manuscript
    # to avoid duplicating the title as a standalone section.

    if include_uncertainty and uncertainty_md.exists():
        uncertainty_text = _read_text(uncertainty_md)
        uncertainty_text = _strip_internal_blocks(uncertainty_text, mode=mode)
        uncertainty_text = _normalize_markdown_for_tex_docx_parity(uncertainty_text)
        uncertainty_text = _replace_math_blocks_with_images(
            uncertainty_text, out_dir=out_dir, mode=mode, embed_images=embed_images
        )
        body, toc = _markdown_to_html(uncertainty_text)
        # 条件分岐: `mode != "publish"` を満たす経路を評価する。
        if mode != "publish":
            body = _linkify_repo_paths(body, root=root, out_dir=out_dir, fig_anchor_map=fig_anchor_map)

        # 条件分岐: `standardize_numbered_anchors` を満たす経路を評価する。

        if standardize_numbered_anchors:
            body, toc = _standardize_numbered_heading_ids(body_html=body, toc_html=toc)

        # 条件分岐: `enable_citation_links` を満たす経路を評価する。

        if enable_citation_links:
            body = _linkify_citations(body, ref_keys=ref_keys)

        body = _inline_png_code_snippets(
            body,
            root=root,
            out_dir=out_dir,
            figs=figs,
            mode=mode,
            embed_images=embed_images,
            fig_map=publish_fig_map,
            inlined=publish_inlined,
            img_cache=publish_img_cache,
        )
        sections.append({"id": "uncertainty", "title": "不確かさ（統計＋系統）", "badge": "付録", "body_html": body, "toc_html": toc})

    # 条件分岐: `manuscript_md.exists()` を満たす経路を評価する。

    if manuscript_md.exists():
        manuscript_text = _read_text(manuscript_md)
        manuscript_text = _strip_internal_blocks(manuscript_text, mode=mode)
        manuscript_text = _normalize_markdown_for_tex_docx_parity(manuscript_text)
        manuscript_text = _replace_math_blocks_with_images(
            manuscript_text, out_dir=out_dir, mode=mode, embed_images=embed_images
        )
        body, toc = _markdown_to_html(manuscript_text)
        # 条件分岐: `mode != "publish"` を満たす経路を評価する。
        if mode != "publish":
            body = _linkify_repo_paths(body, root=root, out_dir=out_dir, fig_anchor_map=fig_anchor_map)

        # 条件分岐: `standardize_numbered_anchors` を満たす経路を評価する。

        if standardize_numbered_anchors:
            body, toc = _standardize_numbered_heading_ids(body_html=body, toc_html=toc)

        # 条件分岐: `enable_citation_links` を満たす経路を評価する。

        if enable_citation_links:
            body = _linkify_citations(body, ref_keys=ref_keys)

        # 条件分岐: `include_table1 and table1_md.exists()` を満たす経路を評価する。

        if include_table1 and table1_md.exists():
            table1_json = table1_md.with_suffix(".json")
            # 条件分岐: `table1_json.exists()` を満たす経路を評価する。
            if table1_json.exists():
                table1_html = _render_table1_html_from_json(table1_json, profile=profile)
                body = _inject_table1_after_h3(body, insert_html=table1_html)

        # 条件分岐: `profile == "part3_quantum"` を満たす経路を評価する。

        if profile == "part3_quantum":
            body = _inject_pagebreak_before_heading_ids(
                body,
                heading_ids=("section-4-10-2", "section-4-10-3"),
            )

        body = _inline_png_code_snippets(
            body,
            root=root,
            out_dir=out_dir,
            figs=figs,
            mode=mode,
            embed_images=embed_images,
            fig_map=publish_fig_map,
            inlined=publish_inlined,
            img_cache=publish_img_cache,
        )
        sections.append({"id": "manuscript", "title": "本文", "badge": "", "body_html": body, "toc_html": toc})
    else:
        sections.append(
            {
                "id": "manuscript",
                "title": "本文",
                "badge": "",
                "body_html": f"<p class='muted'>Missing: <code>{html.escape(str(manuscript_md))}</code></p>",
                "toc_html": "",
            }
        )

    # 条件分岐: `include_quantum_appendix_a and quantum_appendix_a_md.exists()` を満たす経路を評価する。

    if include_quantum_appendix_a and quantum_appendix_a_md.exists():
        appendix_text = _read_text(quantum_appendix_a_md)
        appendix_text = _strip_internal_blocks(appendix_text, mode=mode)
        appendix_text = _normalize_markdown_for_tex_docx_parity(appendix_text)
        appendix_text = _replace_math_blocks_with_images(appendix_text, out_dir=out_dir, mode=mode, embed_images=embed_images)
        body, toc = _markdown_to_html(appendix_text)
        # 条件分岐: `mode != "publish"` を満たす経路を評価する。
        if mode != "publish":
            body = _linkify_repo_paths(body, root=root, out_dir=out_dir, fig_anchor_map=fig_anchor_map)

        # 条件分岐: `standardize_numbered_anchors` を満たす経路を評価する。

        if standardize_numbered_anchors:
            body, toc = _standardize_numbered_heading_ids(body_html=body, toc_html=toc)

        # 条件分岐: `enable_citation_links` を満たす経路を評価する。

        if enable_citation_links:
            body = _linkify_citations(body, ref_keys=ref_keys)

        body = _inline_png_code_snippets(
            body,
            root=root,
            out_dir=out_dir,
            figs=figs,
            mode=mode,
            embed_images=embed_images,
            fig_map=publish_fig_map,
            inlined=publish_inlined,
            img_cache=publish_img_cache,
        )
        sections.append(
            {
                "id": "quantum_appendix_a",
                "title": "補遺A：Si α(T) ansatzログ",
                "badge": "付録",
                "body_html": body,
                "toc_html": toc,
            }
        )

    # 条件分岐: `include_definitions and definitions_md.exists()` を満たす経路を評価する。

    if include_definitions and definitions_md.exists():
        definitions_text = _read_text(definitions_md)
        definitions_text = _strip_internal_blocks(definitions_text, mode=mode)
        definitions_text = _normalize_markdown_for_tex_docx_parity(definitions_text)
        definitions_text = _replace_math_blocks_with_images(
            definitions_text, out_dir=out_dir, mode=mode, embed_images=embed_images
        )
        body, toc = _markdown_to_html(definitions_text)
        # 条件分岐: `mode != "publish"` を満たす経路を評価する。
        if mode != "publish":
            body = _linkify_repo_paths(body, root=root, out_dir=out_dir, fig_anchor_map=fig_anchor_map)

        # 条件分岐: `standardize_numbered_anchors` を満たす経路を評価する。

        if standardize_numbered_anchors:
            body, toc = _standardize_numbered_heading_ids(body_html=body, toc_html=toc)

        # 条件分岐: `enable_citation_links` を満たす経路を評価する。

        if enable_citation_links:
            body = _linkify_citations(body, ref_keys=ref_keys)

        body = _inline_png_code_snippets(
            body,
            root=root,
            out_dir=out_dir,
            figs=figs,
            mode=mode,
            embed_images=embed_images,
            fig_map=publish_fig_map,
            inlined=publish_inlined,
            img_cache=publish_img_cache,
        )
        sections.append({"id": "definitions", "title": "記号・定義", "badge": "付録", "body_html": body, "toc_html": toc})

    # 条件分岐: `include_llr_appendix and llr_appendix_md.exists()` を満たす経路を評価する。

    if include_llr_appendix and llr_appendix_md.exists():
        llr_text = _read_text(llr_appendix_md)
        llr_text = _strip_internal_blocks(llr_text, mode=mode)
        llr_text = _normalize_markdown_for_tex_docx_parity(llr_text)
        llr_text = _replace_math_blocks_with_images(llr_text, out_dir=out_dir, mode=mode, embed_images=embed_images)
        body, toc = _markdown_to_html(llr_text)
        # 条件分岐: `mode != "publish"` を満たす経路を評価する。
        if mode != "publish":
            body = _linkify_repo_paths(body, root=root, out_dir=out_dir, fig_anchor_map=fig_anchor_map)

        # 条件分岐: `standardize_numbered_anchors` を満たす経路を評価する。

        if standardize_numbered_anchors:
            body, toc = _standardize_numbered_heading_ids(body_html=body, toc_html=toc)

        # 条件分岐: `enable_citation_links` を満たす経路を評価する。

        if enable_citation_links:
            body = _linkify_citations(body, ref_keys=ref_keys)

        body = _inline_png_code_snippets(
            body,
            root=root,
            out_dir=out_dir,
            figs=figs,
            mode=mode,
            embed_images=embed_images,
            fig_map=publish_fig_map,
            inlined=publish_inlined,
            img_cache=publish_img_cache,
        )
        sections.append({"id": "llr_appendix", "title": "付録：LLR 全グラフ", "badge": "付録", "body_html": body, "toc_html": toc})

    # Figures (fixed paths) – show as an appendix gallery for paper reading.

    if figs and mode != "publish":
        fig_parts: List[str] = []
        fig_parts.append("<p class='muted'>固定パス（doc/paper/01_figures_index.md）から .png を抽出して一覧表示します。</p>")
        fig_no = 0
        current_section = None
        for fig in figs:
            # 条件分岐: `fig.section and fig.section != current_section` を満たす経路を評価する。
            if fig.section and fig.section != current_section:
                current_section = fig.section
                fig_parts.append(f"<h3>{html.escape(current_section)}</h3>")

            fig_no += 1
            fig_id = f"fig-{fig_no:03d}"
            fig_label = f"図{fig_no}"
            rel = _rel_url(out_dir, fig.path)
            fig_parts.append(f"<figure id='{html.escape(fig_id)}'>")
            cap = html.escape(fig.caption) if fig.caption else ""
            path_code = html.escape(fig.rel)
            # 条件分岐: `cap` を満たす経路を評価する。
            if cap:
                fig_parts.append(
                    f"<figcaption><strong>{html.escape(fig_label)}: {cap}</strong>"
                    f"<div class='muted'><code>{path_code}</code></div></figcaption>"
                )
            else:
                fig_parts.append(f"<figcaption><strong>{html.escape(fig_label)}</strong> <span class='muted'><code>{path_code}</code></span></figcaption>")

            fig_parts.append(f"<a href='{html.escape(rel)}'><img src='{html.escape(rel)}' loading='lazy'></a>")
            fig_parts.append("</figure>")

        sections.append({"id": "figures", "title": "図表（固定パス一覧）", "badge": "付録", "body_html": "\n".join(fig_parts), "toc_html": ""})

    # 条件分岐: `sources_md.exists()` を満たす経路を評価する。

    if sources_md.exists():
        sources_text = _read_text(sources_md)
        sources_text = _strip_internal_blocks(sources_text, mode=mode)
        sources_text = _normalize_markdown_for_tex_docx_parity(sources_text)
        # 条件分岐: `mode == "publish"` を満たす経路を評価する。
        if mode == "publish":
            sources_text = _filter_md_h2_sections_by_ref_keys(sources_text, keep_keys=used_cite_keys)

        sources_text = _replace_math_blocks_with_images(
            sources_text, out_dir=out_dir, mode=mode, embed_images=embed_images
        )
        body, toc = _markdown_to_html(sources_text)
        # 条件分岐: `mode != "publish"` を満たす経路を評価する。
        if mode != "publish":
            body = _linkify_repo_paths(body, root=root, out_dir=out_dir, fig_anchor_map=fig_anchor_map)

        # 条件分岐: `standardize_numbered_anchors` を満たす経路を評価する。

        if standardize_numbered_anchors:
            body, toc = _standardize_numbered_heading_ids(body_html=body, toc_html=toc)

        # 条件分岐: `enable_citation_links` を満たす経路を評価する。

        if enable_citation_links:
            body = _linkify_citations(body, ref_keys=ref_keys)

        body = _inline_png_code_snippets(
            body,
            root=root,
            out_dir=out_dir,
            figs=figs,
            mode=mode,
            embed_images=embed_images,
            fig_map=publish_fig_map,
            inlined=publish_inlined,
            img_cache=publish_img_cache,
        )
        sections.append({"id": "sources", "title": "データ出典（一次ソース）", "badge": "付録", "body_html": body, "toc_html": toc})

    # 条件分岐: `refs_md.exists()` を満たす経路を評価する。

    if refs_md.exists():
        refs_text = _read_text(refs_md)
        refs_text = _strip_internal_blocks(refs_text, mode=mode)
        refs_text = _normalize_markdown_for_tex_docx_parity(refs_text)
        # 条件分岐: `mode == "publish"` を満たす経路を評価する。
        if mode == "publish":
            refs_text = _filter_md_h2_sections_by_ref_keys(refs_text, keep_keys=used_cite_keys)

        refs_text = _inject_reference_anchors(refs_text)
        refs_text = _replace_math_blocks_with_images(
            refs_text, out_dir=out_dir, mode=mode, embed_images=embed_images
        )
        body, toc = _markdown_to_html(refs_text)
        # 条件分岐: `mode != "publish"` を満たす経路を評価する。
        if mode != "publish":
            body = _linkify_repo_paths(body, root=root, out_dir=out_dir, fig_anchor_map=fig_anchor_map)

        sections.append({"id": "references", "title": "参考文献", "badge": "付録", "body_html": body, "toc_html": toc})

    # 条件分岐: `args.out_name` を満たす経路を評価する。

    if args.out_name:
        out_name = str(args.out_name)
    else:
        # 条件分岐: `profile == "paper"` を満たす経路を評価する。
        if profile == "paper":
            out_name = "pmodel_paper.html"
        # 条件分岐: 前段条件が不成立で、`profile == "part2_astrophysics"` を追加評価する。
        elif profile == "part2_astrophysics":
            out_name = "pmodel_paper_part2_astrophysics.html"
        # 条件分岐: 前段条件が不成立で、`profile == "part3_quantum"` を追加評価する。
        elif profile == "part3_quantum":
            out_name = "pmodel_paper_part3_quantum.html"
        # 条件分岐: 前段条件が不成立で、`profile == "part4_verification"` を追加評価する。
        elif profile == "part4_verification":
            out_name = "pmodel_paper_part4_verification.html"
        else:  # pragma: no cover (guarded by argparse choices)
            raise ValueError(f"unknown profile: {profile}")

    # 条件分岐: `profile == "part4_verification"` を満たす経路を評価する。

    if profile == "part4_verification":
        title = "P-model Part IV（検証資料）"
        subtitle = "検証方法と公開成果物への参照先（GitHub）"
        header_badge = "Part IV"
    # 条件分岐: 前段条件が不成立で、`profile == "part2_astrophysics"` を追加評価する。
    elif profile == "part2_astrophysics":
        title = "P-model Part II（宇宙物理編）"
        subtitle = "応用検証：宇宙物理（公開体裁）"
        header_badge = "Part II"
    # 条件分岐: 前段条件が不成立で、`profile == "part3_quantum"` を追加評価する。
    elif profile == "part3_quantum":
        title = "P-model Part III（量子物理編）"
        subtitle = "応用検証：量子物理（公開体裁）"
        header_badge = "Part III"
    else:
        title = "P-model Part I（コア理論）"
        subtitle = "記号規約・最小仮定・写像（公開体裁）"
        header_badge = "Part I"

    manuscript_link = str(manuscript_md.relative_to(root)).replace("\\", "/")
    html_path = _render_html(
        out_dir=out_dir,
        out_name=out_name,
        title=title,
        subtitle=subtitle,
        header_badge=header_badge,
        manuscript_link=manuscript_link,
        sections=sections,
        mode=mode,
        toc_mode="global_only",
    )
    print(f"[ok] html: {html_path}")

    try:
        worklog.append_event(
            {
                "event_type": "paper_html",
                "argv": list(argv) if argv is not None else None,
                "outputs": {"paper_html": html_path},
                "profile": profile,
                "manuscript": manuscript_link,
                "out_name": out_name,
                "mode": mode,
                "embed_images": bool(embed_images),
            }
        )
    except Exception:
        pass

    # 条件分岐: `args.open and os.name == "nt"` を満たす経路を評価する。

    if args.open and os.name == "nt":
        try:
            os.startfile(str(html_path))  # type: ignore[attr-defined]
        except Exception:
            pass

    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())
