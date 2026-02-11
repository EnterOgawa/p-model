from __future__ import annotations

import html
import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.summary import worklog  # noqa: E402

LLR_LONG_NAME = "月レーザー測距（LLR: Lunar Laser Ranging）"
LLR_SHORT_NAME = "月レーザー測距（LLR）"

# NOTE: BepiColombo(MORE) is still in a "data not publicly available" phase.
# Keep scripts/data for readiness, but do not include it in the public report by default.
INCLUDE_BEPICOLOMBO_IN_PUBLIC_REPORT = os.environ.get("WAVEP_INCLUDE_BEPICOLOMBO", "0").strip() == "1"


def _repo_root() -> Path:
    return _ROOT


def _remap_output_path(path: Path) -> Path:
    """Map legacy output paths to the new output layout.

    New layout:
      - output/public/<topic>/...  (tracked public artifacts)
      - output/private/<topic>/... (local-only / intermediate artifacts)

    Many scripts historically wrote to output/<topic>/..., so the dashboard accepts legacy
    paths and remaps them if the target exists.
    """
    try:
        rel = path.resolve().relative_to(_ROOT)
    except Exception:
        return path

    parts = list(rel.parts)
    if len(parts) < 2:
        return path
    if parts[0] != "output":
        return path
    if parts[1] in ("public", "private"):
        return path

    topic = parts[1]
    tail = Path(*parts[2:]) if len(parts) > 2 else Path()
    # Quantum artifacts are now tracked under output/public/quantum.
    if topic == "quantum":
        cand = (_ROOT / "output" / "public" / topic / tail).resolve()
        if cand.exists():
            return cand

    # Default: treat as local-only under output/private/<topic>/.
    cand = (_ROOT / "output" / "private" / topic / tail).resolve()
    if cand.exists():
        return cand

    return path


def _set_japanese_font() -> None:
    try:
        import matplotlib as mpl
        import matplotlib.font_manager as fm

        preferred = [
            "Yu Gothic",
            "Meiryo",
            "BIZ UDGothic",
            "MS Gothic",
            "Yu Mincho",
            "MS Mincho",
        ]
        available = {f.name for f in fm.fontManager.ttflist}
        chosen = [name for name in preferred if name in available]
        if not chosen:
            return

        mpl.rcParams["font.family"] = chosen + ["DejaVu Sans"]
        mpl.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass


def _try_read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        path = _remap_output_path(path)
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _try_read_text(path: Path) -> Optional[str]:
    try:
        path = _remap_output_path(path)
        return path.read_text(encoding="utf-8")
    except Exception:
        return None


def _try_read_csv_rows(path: Path, *, max_rows: int = 200) -> List[Dict[str, str]]:
    try:
        import csv

        path = _remap_output_path(path)
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            rows: List[Dict[str, str]] = []
            for i, r in enumerate(reader):
                if i >= int(max_rows):
                    break
                rows.append({str(k): ("" if v is None else str(v)) for k, v in dict(r).items()})
            return rows
    except Exception:
        return []


def _try_compute_llr_inlier_rms(points_csv: Path) -> Optional[Dict[str, Any]]:
    """Compute global RMS (inlier-only) for key LLR residual columns.

    Keep this dependency-free (no pandas). The input CSV is produced by
    scripts/llr/llr_batch_eval.py and is small enough to stream.
    """
    points_csv = _remap_output_path(points_csv)
    if not points_csv.exists():
        return None

    try:
        import csv
        import math

        cols = {
            "rms_sr_ns": "residual_sr_ns",
            "rms_sr_tropo_ns": "residual_sr_tropo_ns",
            "rms_sr_tropo_tide_ns": "residual_sr_tropo_tide_ns",
        }
        sums_sq: Dict[str, float] = {k: 0.0 for k in cols.keys()}
        counts: Dict[str, int] = {k: 0 for k in cols.keys()}

        n_total = 0
        n_inlier = 0
        with open(points_csv, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                n_total += 1
                inlier = (row.get("inlier_best") or "").strip().lower()
                if inlier not in ("1", "true", "yes"):
                    continue
                n_inlier += 1
                for out_k, col in cols.items():
                    v_raw = (row.get(col) or "").strip()
                    if not v_raw:
                        continue
                    try:
                        v = float(v_raw)
                    except Exception:
                        continue
                    if not math.isfinite(v):
                        continue
                    sums_sq[out_k] += v * v
                    counts[out_k] += 1

        def _rms(k: str) -> Optional[float]:
            c = counts.get(k, 0)
            if not c:
                return None
            return math.sqrt(sums_sq[k] / float(c))

        out: Dict[str, Any] = {"n_total": n_total, "n_inlier": n_inlier}
        for k in cols.keys():
            out[k] = _rms(k)
        return out
    except Exception:
        return None


def _format_num(x: Any, *, digits: int = 3) -> str:
    if isinstance(x, bool) or x is None:
        return str(x)
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        return f"{x:.{digits}g}"
    return str(x)


def _as_str(x: Any) -> str:
    return "" if x is None else str(x)


def _extract_cassini_metrics(root: Path) -> Dict[str, Any]:
    metrics_csv = root / "output" / "cassini" / "cassini_fig2_metrics.csv"
    txt = _try_read_text(metrics_csv)
    if not txt:
        return {}

    # Minimal CSV parse (no pandas dependency)
    lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
    if len(lines) < 2:
        return {}
    header = lines[0].split(",")
    rows = [dict(zip(header, ln.split(","))) for ln in lines[1:]]
    row_all = next((r for r in rows if (r.get("window") or "").startswith("all")), rows[0] if rows else None)
    if not row_all:
        return {}

    out: Dict[str, Any] = {}
    for k in ("n", "rmse", "corr", "beta"):
        v = row_all.get(k)
        if v is None:
            continue
        try:
            out[k] = float(v) if "." in v or "e" in v.lower() else int(v)
        except Exception:
            out[k] = v
    return out


def _extract_viking_peak(root: Path) -> Dict[str, Any]:
    csv_path = _remap_output_path(root / "output" / "viking" / "viking_shapiro_result.csv")
    if not csv_path.exists():
        return {}

    try:
        import csv

        peak_us = None
        peak_time = None
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                try:
                    v = float((r.get("shapiro_delay_us") or "").strip())
                except Exception:
                    continue
                if peak_us is None or v > peak_us:
                    peak_us = v
                    peak_time = (r.get("time_utc") or "").strip()

        if peak_us is None:
            return {}
        return {"peak_us": float(peak_us), "peak_time_utc": peak_time}
    except Exception:
        return {}


def _format_sci(x: Any, *, digits: int = 2) -> str:
    try:
        v = float(x)
    except Exception:
        return str(x)
    if v == 0.0:
        return "0"
    if not (v == v):  # NaN
        return "n/a"
    return f"{v:.{digits}e}"


def _panel_text(title: str, lines: List[str]) -> str:
    s = title
    for ln in lines:
        s += "\n" + ln
    return s


def _rel_url(from_dir: Path, target: Path) -> str:
    try:
        rel = os.path.relpath(target, start=from_dir)
    except ValueError:
        rel = str(target)
    return rel.replace("\\", "/")


def _rel_repo_path(root: Path, target: Path) -> str:
    try:
        rel = target.relative_to(root)
    except ValueError:
        return str(target).replace("\\", "/")
    return str(rel).replace("\\", "/")


_REPO_PATH_RE = re.compile(r"(?P<path>(?:output|doc|scripts|data)/[^\s<>'\"`]+)")
_TRAIL_TRIM = set(".,;:)]}>）】」』、。")


def _render_text_with_links(text: str, *, root: Path, out_dir: Path) -> str:
    """Render plain text as HTML with repo-path linkification (safe)."""
    s = "" if text is None else str(text)
    parts: List[str] = []
    last = 0

    for m in _REPO_PATH_RE.finditer(s):
        start, end = m.span("path")
        cand = m.group("path")
        parts.append(html.escape(s[last:start]))

        trimmed = cand
        suffix = ""
        # Split explanatory suffix like "（...）" or "(...)" that sometimes sticks to paths.
        split_chars = ("（", "(", "[", "{", "<")
        split_idx = None
        for ch in split_chars:
            i = trimmed.find(ch)
            if i >= 0:
                split_idx = i if split_idx is None else min(split_idx, i)
        if split_idx is not None and split_idx > 0:
            suffix = trimmed[split_idx:]
            trimmed = trimmed[:split_idx]

        trailer = ""
        while trimmed and not (root / trimmed).exists():
            if trimmed[-1] in _TRAIL_TRIM:
                trailer = trimmed[-1] + trailer
                trimmed = trimmed[:-1]
                continue
            break

        target = (root / trimmed) if trimmed else None
        if trimmed and target and target.exists():
            href = _rel_url(out_dir, target)
            parts.append(f"<a href='{html.escape(href)}'><code>{html.escape(trimmed)}</code></a>")
            if trailer or suffix:
                parts.append(html.escape(trailer + suffix))
        else:
            parts.append(html.escape(cand))

        last = end

    parts.append(html.escape(s[last:]))
    return "".join(parts)


def _extract_cassini_best_beta(root: Path) -> Dict[str, Any]:
    # Prefer the latest run metadata so we don't accidentally show stale sweep results
    # when the most recent Cassini run was executed with --no-sweep.
    meta = _try_read_json(root / "output" / "cassini" / "cassini_fig2_run_metadata.json") or {}
    best_beta = None
    if isinstance(meta, dict):
        outputs = meta.get("outputs")
        if isinstance(outputs, dict):
            best_beta = outputs.get("best_beta_by_rmse10")

    if best_beta is None:
        return {}

    try:
        best_beta_f = float(best_beta)
    except Exception:
        return {}

    out: Dict[str, Any] = {"best_beta_by_rmse10": best_beta_f}

    # Optional: if sweep CSV exists, also extract the corresponding RMSE(±10d).
    sweep_csv = root / "output" / "cassini" / "cassini_beta_sweep_rmse.csv"
    txt = _try_read_text(sweep_csv)
    if not txt:
        return out

    lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
    if len(lines) < 2:
        return out
    header = lines[0].split(",")

    def _f(r: Dict[str, str], k: str) -> Optional[float]:
        v = r.get(k)
        if v is None:
            return None
        try:
            return float(v)
        except Exception:
            return None

    best_rmse10 = None
    for ln in lines[1:]:
        row = dict(zip(header, ln.split(",")))
        beta = _f(row, "beta")
        rmse10 = _f(row, "rmse_10")
        if beta is None or rmse10 is None:
            continue
        if abs(beta - best_beta_f) <= 1e-12:
            best_rmse10 = rmse10
            break

    if best_rmse10 is not None:
        out["best_rmse10"] = best_rmse10
    return out


def _extract_bepicolombo_more_psa_status(root: Path) -> Dict[str, Any]:
    p = root / "output" / "bepicolombo" / "more_psa_status.json"
    j = _try_read_json(p)
    return j if isinstance(j, dict) else {}


def _extract_bepicolombo_spice_psa_status(root: Path) -> Dict[str, Any]:
    p = root / "output" / "bepicolombo" / "spice_psa_status.json"
    j = _try_read_json(p)
    return j if isinstance(j, dict) else {}


def _extract_bepicolombo_shapiro_predict(root: Path) -> Dict[str, Any]:
    p = root / "output" / "bepicolombo" / "bepicolombo_shapiro_geometry_summary.json"
    j = _try_read_json(p)
    return j if isinstance(j, dict) else {}


def _extract_bepicolombo_conjunction_catalog(root: Path) -> Dict[str, Any]:
    p = root / "output" / "bepicolombo" / "bepicolombo_conjunction_catalog_summary.json"
    j = _try_read_json(p)
    return j if isinstance(j, dict) else {}


def _extract_bepicolombo_more_document_catalog(root: Path) -> Dict[str, Any]:
    p = root / "output" / "bepicolombo" / "more_document_catalog.json"
    j = _try_read_json(p)
    return j if isinstance(j, dict) else {}


def _llr_residual_vs_elevation_cards(root: Path) -> List[Dict[str, Any]]:
    out_llr = root / "output" / "llr" / "batch"
    diag = _try_read_json(out_llr / "llr_station_diagnostics.json") or {}
    diag_map: Dict[str, Any] = {}
    stations: List[str] = []
    if isinstance(diag, dict):
        sd = diag.get("station_diagnostics")
        if isinstance(sd, dict):
            diag_map = sd
            stations = sorted(str(k) for k in sd.keys())

    if not stations:
        # Fallback: include typical stations if the corresponding figures exist.
        for st in ("GRSM", "APOL", "MATM", "WETL"):
            fname = "llr_grsm_residual_vs_elevation.png" if st == "GRSM" else f"llr_{st.lower()}_residual_vs_elevation.png"
            if (out_llr / fname).exists():
                stations.append(st)

    cards: List[Dict[str, Any]] = []
    for st in stations:
        st_u = str(st).strip().upper()
        fname = "llr_grsm_residual_vs_elevation.png" if st_u == "GRSM" else f"llr_{st_u.lower()}_residual_vs_elevation.png"
        path = out_llr / fname
        if not path.exists():
            continue

        meta = diag_map.get(st_u) if isinstance(diag_map, dict) else None
        n = meta.get("n") if isinstance(meta, dict) else None
        rms = meta.get("rms_sr_tropo_tide_ns") if isinstance(meta, dict) else None
        corr = meta.get("corr_res_vs_elev_deg") if isinstance(meta, dict) else None

        summary_lines = [
            "仰角（高度角）と残差の相関を見ることで、大気（対流圏）や低仰角データの影響を確認する。",
        ]
        if n is not None:
            summary_lines.append(f"点数 n={_format_num(n)}")
        if rms is not None:
            summary_lines.append(f"RMS={_format_num(rms, digits=4)} ns（SR+Tropo+Tide）")
        if corr is not None:
            summary_lines.append(f"相関 corr(残差,仰角)={_format_num(corr, digits=4)}")

        cards.append(
            {
                "id": f"llr_batch_{st_u.lower()}_residual_vs_elevation",
                "title": f"{LLR_SHORT_NAME}：{st_u} 残差 vs 仰角（SR+Tropo+Tide）",
                "kind": "バッチ（原因切り分け：低仰角依存）",
                "path": path,
                "summary_lines": summary_lines,
                "explain_lines": [
                    "低仰角ほど大気の通過量が増え、対流圏モデル誤差が残差に出やすい。",
                    "反射器ごとに色分けし、特定ターゲットだけの依存か/局全体の依存かを切り分ける。",
                ],
                "detail_lines": [
                    "横軸は平均仰角（上り/下りの平均）、縦軸は残差（SR+Tropo+Tide, 定数オフセット整列後）。",
                    "低仰角側で残差の散らばりが増える、あるいは片側に偏るなら、対流圏モデル（マッピング関数・湿度モデル）や低仰角データ編集の影響を疑う。",
                    "仰角依存が弱いのに期間依存スパイクが残る場合は、局ログ（座標/装置遅延）更新や外れ値混入の可能性が高い。",
                    "点ごとの診断CSVは output/llr/batch/llr_batch_points.csv に保存される（オフラインで再解析可能）。",
                    f"再現: scripts/llr/llr_batch_eval.py → output/llr/batch/{fname}",
                ],
            }
        )
    return cards


def _pick_llr_stem(root: Path) -> str:
    out_llr = root / "output" / "llr"
    for stem in ("llr_primary", "demo_llr_like"):
        if (out_llr / f"{stem}_summary.json").exists():
            return stem

    candidates = sorted(out_llr.glob("*_summary.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for p in candidates:
        if p.stem.startswith("merged"):
            continue
        if p.name.endswith("_summary.json"):
            return p.name[: -len("_summary.json")]
    return "demo_llr_like"


def _render_public_html(
    *,
    root: Path,
    out_dir: Path,
    title: str,
    subtitle: str,
    sections: List[Tuple[str, List[Dict[str, Any]]]],
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    html_path = out_dir / "pmodel_public_report.html"

    parts: List[str] = []
    parts.append("<!doctype html>")
    parts.append("<html lang='ja'><head>")
    parts.append("<meta charset='utf-8'>")
    parts.append(f"<title>{html.escape(title)}</title>")
    parts.append(
        "<style>"
        "body{font-family:Yu Gothic,Meiryo,BIZ UDGothic,MS Gothic,system-ui,sans-serif;margin:24px;max-width:1100px}"
        "h1{margin:0 0 6px 0;font-size:24px;line-height:1.25}"
        "h2{margin:28px 0 10px 0;font-size:20px;line-height:1.3;border-bottom:1px solid #ddd;padding-bottom:6px}"
        "h3{margin:0 0 6px 0;font-size:18px;line-height:1.35}"
        ".muted{color:#666;font-size:13px}"
        ".card{border:1px solid #e3e3e3;border-radius:10px;padding:14px 16px;margin:14px 0}"
        ".meta{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin:4px 0 10px 0}"
        ".badge{font-size:12px;padding:2px 8px;border-radius:999px;background:#f2f2f2;color:#333}"
        "ul{margin:6px 0 10px 18px}"
        "img{max-width:100%;height:auto;border:1px solid #ddd;border-radius:8px}"
        ".table-wrap{overflow-x:auto;margin:8px 0 10px 0}"
        "table{border-collapse:collapse;width:100%;font-size:12px}"
        "th,td{border:1px solid #ddd;padding:4px 6px;white-space:nowrap}"
        "th{background:#f6f6f6;text-align:left}"
        "a{color:#0b65c2;text-decoration:none}"
        "a:hover{text-decoration:underline}"
        "</style>"
    )
    parts.append("</head><body>")
    parts.append(f"<h1>{html.escape(title)}</h1>")
    parts.append(f"<div class='muted'>{html.escape(subtitle)}</div>")

    # Stable anchor ids (avoid whitespace / special chars in id attributes).
    sec_ids: Dict[str, str] = {}
    used: set[str] = set()
    for sec_title, _graphs in sections:
        base = f"sec-{len(used) + 1}"
        sec_id = base
        n = 2
        while sec_id in used:
            sec_id = f"{base}-{n}"
            n += 1
        used.add(sec_id)
        sec_ids[sec_title] = sec_id

    # TOC
    parts.append("<h2>目次</h2><ul>")
    for sec_title, graphs in sections:
        sec_id = sec_ids.get(sec_title, "")
        parts.append(f"<li><a href='#{html.escape(sec_id)}'>{html.escape(sec_title)}</a></li>")
        for g in graphs:
            gid = g.get("id") or ""
            gt = g.get("title") or ""
            if gid and gt:
                parts.append(f"<li style='margin-left:18px'><a href='#{html.escape(gid)}'>{html.escape(gt)}</a></li>")
    parts.append("</ul>")

    for sec_title, graphs in sections:
        sec_id = sec_ids.get(sec_title, "")
        parts.append(f"<h2 id='{html.escape(sec_id)}'>{html.escape(sec_title)}</h2>")
        for g in graphs:
            gid = str(g.get("id") or "")
            gt = str(g.get("title") or "")
            kind = str(g.get("kind") or "")
            path = g.get("path")
            detail_href = str(g.get("detail_href") or "")
            summary_lines = g.get("summary_lines") or []
            explain_lines = g.get("explain_lines") or []
            detail_lines = g.get("detail_lines") or []
            table = g.get("table") or None

            parts.append(f"<div class='card' id='{html.escape(gid)}'>")
            parts.append(f"<h3>{html.escape(gt)}</h3>")

            parts.append("<div class='meta'>")
            if kind:
                parts.append(f"<span class='badge'>{html.escape(kind)}</span>")
            if detail_href:
                parts.append(f"<a class='badge' href='{html.escape(detail_href)}'>詳細</a>")
            parts.append("</div>")

            if path is None or path == "":
                pass
            elif isinstance(path, Path) and path.exists():
                rel = _rel_url(out_dir, path)
                parts.append(f"<a href='{html.escape(rel)}'><img src='{html.escape(rel)}' alt='{html.escape(gt)}'></a>")
                parts.append("<div class='muted'>（クリックで拡大）</div>")
            else:
                parts.append(f"<div class='muted'>Missing: <code>{html.escape(str(path))}</code></div>")

            if summary_lines:
                parts.append("<div class='muted'>概要</div><ul>")
                for ln in summary_lines:
                    parts.append(f"<li>{_render_text_with_links(str(ln), root=root, out_dir=out_dir)}</li>")
                parts.append("</ul>")

            if explain_lines:
                parts.append("<div class='muted'>解説</div><ul>")
                for ln in explain_lines:
                    parts.append(f"<li>{_render_text_with_links(str(ln), root=root, out_dir=out_dir)}</li>")
                parts.append("</ul>")

            if isinstance(table, dict) and table.get("rows"):
                headers = table.get("headers") or []
                rows = table.get("rows") or []
                caption = str(table.get("caption") or "")
                parts.append("<div class='muted'>一覧</div>")
                if caption:
                    parts.append(f"<div class='muted'>{html.escape(caption)}</div>")
                parts.append("<div class='table-wrap'><table><thead><tr>")
                for h in headers:
                    parts.append(f"<th>{html.escape(str(h))}</th>")
                parts.append("</tr></thead><tbody>")
                for r in rows:
                    parts.append("<tr>")
                    for cell in (r or []):
                        parts.append(f"<td>{html.escape(str(cell))}</td>")
                    parts.append("</tr>")
                parts.append("</tbody></table></div>")

            if detail_lines:
                parts.append("<div class='muted'>詳細解説</div><ul>")
                for ln in detail_lines:
                    parts.append(f"<li>{_render_text_with_links(str(ln), root=root, out_dir=out_dir)}</li>")
                parts.append("</ul>")

            parts.append("</div>")

    parts.append("</body></html>")
    html_path.write_text("\n".join(parts), encoding="utf-8")
    return html_path


def _extract_roadmap_table(root: Path) -> Dict[str, Any]:
    status_path = root / "doc" / "STATUS.md"
    text = _try_read_text(status_path) or ""

    updated_utc: Optional[str] = None
    m = re.search(r"最終更新（UTC）：\s*([0-9]{4}-[0-9]{2}-[0-9]{2})", text)
    if m:
        updated_utc = m.group(1).strip()

    rows: List[List[str]] = []
    phases: Dict[int, Tuple[str, str, str]] = {}
    for line in text.splitlines():
        m = re.match(r"\s*-\s*Phase\s*(\d+)（([^）]+)）:\s*(.+)\s*$", line)
        if not m:
            continue
        phase = int(m.group(1))
        theme = m.group(2).strip()
        status_full = m.group(3).strip()
        status_short = status_full.split("（", 1)[0].strip()
        phases[phase] = (theme, status_short, status_full)

    for phase in sorted(phases.keys()):
        theme, status_short, _status_full = phases[phase]
        rows.append([f"Phase {phase}", theme, status_short])

    focus_phase: Optional[int] = None
    focus_theme: Optional[str] = None
    for phase in sorted(phases.keys()):
        theme, status_short, _status_full = phases[phase]
        if status_short not in ("完了", "一旦完了"):
            focus_phase = phase
            focus_theme = theme
            break

    return {
        "updated_utc": updated_utc,
        "rows": rows,
        "focus_phase": focus_phase,
        "focus_theme": focus_theme,
        "status_path": status_path,
    }


def _extract_paper_table1_card(root: Path) -> Dict[str, Any]:
    json_path = root / "output" / "private" / "summary" / "paper_table1_results.json"
    md_path = root / "output" / "private" / "summary" / "paper_table1_results.md"
    csv_path = root / "output" / "private" / "summary" / "paper_table1_results.csv"
    paper_html = root / "output" / "private" / "summary" / "pmodel_paper.html"

    j = _try_read_json(json_path)
    if not isinstance(j, dict):
        detail_href = None
        if paper_html.exists():
            # public report lives in output/private/summary, so link by filename.
            detail_href = "pmodel_paper.html"
        return {
            "id": "paper_table1",
            "title": "検証サマリ（Table 1）",
            "kind": "論文化（Phase 8 / Step 8.2）",
            "summary_lines": [
                "未生成: cmd /c scripts\\summary\\build_materials.bat quick-nodocx を実行してください。",
            ],
            **({"detail_href": detail_href} if detail_href else {}),
            "detail_lines": [
                f"期待する出力: {_rel_repo_path(root, json_path)} / {_rel_repo_path(root, md_path)} / {_rel_repo_path(root, csv_path)}",
                f"論文HTML（任意）: {_rel_repo_path(root, paper_html)}",
                "生成（HTMLのみ）: python -B scripts/summary/paper_build.py --mode publish --outdir output/private/summary --skip-docx",
            ],
        }

    generated_utc = str(j.get("generated_utc") or "")
    table1 = j.get("table1") or {}
    rows_dict = table1.get("rows") or []

    headers = ["テーマ", "観測量/指標", "データ", "N", "参照", "P-model", "差/指標（技術）", "かんたん解釈"]
    rows: List[List[str]] = []
    for r in rows_dict:
        if not isinstance(r, dict):
            continue
        n = r.get("n")
        rows.append(
            [
                _as_str(r.get("topic")),
                _as_str(r.get("observable")),
                _as_str(r.get("data")),
                "" if n is None else _as_str(n),
                _as_str(r.get("reference")),
                _as_str(r.get("pmodel")),
                _as_str(r.get("metric")),
                _as_str(r.get("metric_public")),
            ]
        )

    notes = table1.get("notes") or []
    summary_lines = [
        f"生成（UTC）: {generated_utc}" if generated_utc else "生成時刻: （不明）",
        "論文化（Phase 8 / Step 8.2）用の最小サマリ。output/ の確定結果から集計（再計算なし）。",
    ]
    if isinstance(notes, list):
        for ln in notes[:3]:
            if ln:
                summary_lines.append(str(ln))

    return {
        "id": "paper_table1",
        "title": "検証サマリ（Table 1）",
        "kind": "論文化（Phase 8 / Step 8.2）",
        **({"detail_href": "pmodel_paper.html"} if paper_html.exists() else {}),
        "summary_lines": summary_lines,
        "detail_lines": [
            "生成: scripts/summary/paper_tables.py（paper_build/build_materials に統合済み）",
            f"出力: {_rel_repo_path(root, md_path)} / {_rel_repo_path(root, csv_path)} / {_rel_repo_path(root, json_path)}",
            "本文: doc/paper/10_manuscript.md（固定パスの図表に言及）",
            f"論文HTML: {_rel_repo_path(root, paper_html)}",
        ],
        "table": {"headers": headers, "rows": rows, "caption": "検証サマリ（Table 1, 自動生成）"} if rows else None,
    }


def _extract_decisive_scoreboard_card(root: Path) -> Dict[str, Any]:
    json_path = root / "output" / "private" / "summary" / "decisive_scoreboard.json"
    png_path = root / "output" / "private" / "summary" / "decisive_scoreboard.png"
    frozen_path = root / "output" / "theory" / "frozen_parameters.json"

    j = _try_read_json(json_path)
    if not isinstance(j, dict) or not png_path.exists():
        return {
            "id": "decisive_scoreboard",
            "title": "決定的スコアボード（β固定）",
            "kind": "決定的検証（Phase 7）",
            "summary_lines": [
                "未生成: python -B scripts/summary/decisive_scoreboard.py を実行してください。",
                "β凍結: python -B scripts/theory/freeze_parameters.py",
            ],
        }

    beta = j.get("beta")
    beta_sigma = j.get("beta_sigma")
    beta_source = _as_str(j.get("beta_source"))
    gamma_p = j.get("gamma_pmodel")

    rows = j.get("rows") or []
    if not isinstance(rows, list):
        rows = []

    # Summary: show the worst |z| among predict rows + a short list.
    predict_rows = [r for r in rows if isinstance(r, dict) and str(r.get("kind") or "") == "predict"]
    max_abs_z = None
    if predict_rows:
        try:
            max_abs_z = max(float(r.get("abs_z")) for r in predict_rows if r.get("abs_z") is not None)
        except Exception:
            max_abs_z = None

    summary_lines: List[str] = []
    if beta is not None:
        if beta_sigma is not None:
            summary_lines.append(
                f"β固定: β={_format_num(float(beta), digits=10)} ± {_format_num(float(beta_sigma), digits=3)}（source={beta_source or 'unknown'}）"
            )
        else:
            summary_lines.append(f"β固定: β={_format_num(float(beta), digits=10)}（source={beta_source or 'unknown'}）")
    if gamma_p is not None:
        summary_lines.append(f"P-model: γ=2β-1 → γ={_format_num(float(gamma_p), digits=10)}")
    if max_abs_z is not None:
        summary_lines.append(f"最大|z|（predictのみ）={max_abs_z:.2f}（小さいほど整合）")

    # Table: full z-score list
    table_rows: List[List[str]] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        label = _as_str(r.get("label"))
        kind = _as_str(r.get("kind"))
        obs = _as_str(r.get("observed"))
        pred = _as_str(r.get("predicted"))
        sig = _as_str(r.get("sigma"))
        z = r.get("z")
        try:
            z_txt = f"{float(z):+.2f}" if z is not None else ""
        except Exception:
            z_txt = _as_str(z)
        table_rows.append([label, kind, obs, pred, sig, z_txt])

    table = {
        "headers": ["項目", "役割", "観測", "予測（β固定）", "σ", "z"],
        "rows": table_rows,
        "caption": "Cassiniでβを固定（fit）し、他の検証は予測（predict）としてzスコアで整合性を集約する。",
    }

    frozen_note = "βは凍結済み"
    if frozen_path.exists():
        frozen_note = "βは凍結済み（frozen_parameters.json）"

    return {
        "id": "decisive_scoreboard",
        "title": "決定的スコアボード（fit/predict分離, β固定）",
        "kind": "決定的検証（Phase 7）",
        "path": png_path,
        "summary_lines": summary_lines,
        "explain_lines": [
            "『決定的』にするため、β（光伝播の係数）を一次ソース（CassiniのPPN γ拘束）で固定し、他の検証を予測として評価する。",
            "zスコアは（予測-観測）/σ（|z|が小さいほど整合）。",
            "EHTはリング直径≒シャドウ直径（κ=1）という近似を含むため、現時点では整合性チェックの入口として扱う。",
        ],
        "detail_lines": [
            frozen_note,
            "Cassini行はβの“決め方”なので、z≈0は定義上の一致（独立検証ではない）。",
            "独立検証（predict）としては、太陽光偏向のPPN γ、重力赤方偏移 ε、EHT（κ仮定）が入る。",
            "将来的にLLR/Cassiniの一次データをさらに詰め、βの固定源と予測側の分離を強化する。",
            "再現: scripts/theory/freeze_parameters.py → output/theory/frozen_parameters.json",
            "再現: scripts/summary/decisive_scoreboard.py → output/private/summary/decisive_scoreboard.(png|json)",
        ],
        "table": table,
    }


def _extract_validation_scoreboard_card(root: Path) -> Dict[str, Any]:
    json_path = root / "output" / "private" / "summary" / "validation_scoreboard.json"
    png_path = root / "output" / "private" / "summary" / "validation_scoreboard.png"

    j = _try_read_json(json_path)
    if not isinstance(j, dict) or not png_path.exists():
        return {
            "id": "validation_scoreboard",
            "title": "総合スコアボード（全検証）",
            "kind": "決定的検証（Phase 7）",
            "summary_lines": [
                "未生成: python -B scripts/summary/validation_scoreboard.py を実行してください。",
                "（Table 1 と各章の図を俯瞰する1枚要約）",
            ],
        }

    sigma_stats = j.get("sigma_stats") if isinstance(j.get("sigma_stats"), dict) else None
    status_counts = j.get("status_counts") if isinstance(j.get("status_counts"), dict) else {}
    table1_counts = j.get("table1_status_counts") if isinstance(j.get("table1_status_counts"), dict) else {}
    table1_summary = j.get("table1_status_summary") if isinstance(j.get("table1_status_summary"), dict) else None
    table1_breakdown = j.get("table1_breakdown") if isinstance(j.get("table1_breakdown"), list) else []
    rows = j.get("rows") if isinstance(j.get("rows"), list) else []

    def find_metric(row_id: str) -> Optional[str]:
        for r in rows:
            if not isinstance(r, dict):
                continue
            if str(r.get("id") or "") == row_id:
                return _as_str(r.get("metric"))
        return None

    summary_lines: List[str] = []
    if sigma_stats and sigma_stats.get("n"):
        n_sigma = int(sigma_stats.get("n") or 0)
        n1 = int(sigma_stats.get("n_within_1sigma") or 0)
        n2 = int(sigma_stats.get("n_within_2sigma") or 0)
        rate2 = sigma_stats.get("rate_within_2sigma")
        rate2_txt = ""
        try:
            rate2_txt = f"（{float(rate2)*100:.1f}%）" if rate2 is not None else ""
        except Exception:
            rate2_txt = ""
        summary_lines.append(f"σ評価可能: 2σ以内 {n2}/{n_sigma} {rate2_txt}、1σ以内 {n1}/{n_sigma}")

    ok = int(status_counts.get("ok") or 0)
    mixed = int(status_counts.get("mixed") or 0)
    ng = int(status_counts.get("ng") or 0)
    info = int(status_counts.get("info") or 0)
    summary_lines.append(f"全体（要約行）: OK={ok}, 要改善={mixed}, 不一致={ng}, 参考={info}")

    if table1_counts:
        t_ok = int(table1_counts.get("ok") or 0)
        t_mixed = int(table1_counts.get("mixed") or 0)
        t_ng = int(table1_counts.get("ng") or 0)
        t_info = int(table1_counts.get("info") or 0)
        suffix = ""
        if isinstance(table1_summary, dict):
            try:
                ok_rate = float(table1_summary.get("ok_rate"))
                ok_mixed_rate = float(table1_summary.get("ok_or_mixed_rate"))
                suffix = f" / OK率={ok_rate*100:.1f}% / OK+要改善={ok_mixed_rate*100:.1f}%"
            except Exception:
                suffix = ""
        summary_lines.append(f"Table 1（全27行, 目安）: OK={t_ok}, 要改善={t_mixed}, 不一致={t_ng}, 参考={t_info}{suffix}")

    gw_metric = find_metric("gw")
    if gw_metric:
        summary_lines.append(f"重力波（要約）: {gw_metric}")

    cosmo_ddr = find_metric("cosmo_ddr")
    if cosmo_ddr:
        summary_lines.append(f"宇宙論（距離二重性）: {cosmo_ddr}")

    explain_lines = [
        "検証結果（Table 1）を『1枚で俯瞰』するための要約スコアボード。二重パルサーと重力波も含む。",
        "OK/要改善/不一致 は zスコア（|z|<=1/2）や相関・RMSなどの暫定しきい値に基づく“目安”。",
        "詳細は Table 1 と、各章の図（LLR/Cassini/GW/EHT等）を参照。",
    ]
    policy = j.get("policy") if isinstance(j.get("policy"), dict) else {}
    exceptions = policy.get("exceptions") if isinstance(policy.get("exceptions"), dict) else {}
    if exceptions:
        if isinstance(exceptions.get("cosmo_ddr"), str):
            explain_lines.append("注意: 宇宙論（DDR/Tolman）は距離指標の前提（標準光源/標準定規/進化/不透明度）が強く効く。")
        if isinstance(exceptions.get("eht"), str):
            explain_lines.append("注意: EHTは κ（リング/影）や散乱・スピン系統が支配的で、κ=1のzは入口。")

    table = None
    if table1_breakdown:
        table_rows: List[List[str]] = []
        for r in table1_breakdown:
            if not isinstance(r, dict):
                continue
            metric_public = _as_str(r.get("metric_public"))
            metric = _as_str(r.get("metric"))
            table_rows.append(
                [
                    _as_str(r.get("topic")),
                    _as_str(r.get("observable")),
                    _as_str(r.get("status_label")),
                    metric_public or metric,
                ]
            )
        if table_rows:
            table = {
                "headers": ["テーマ", "観測量/指標", "判定（目安）", "かんたん解釈"],
                "rows": table_rows,
                "caption": "Table 1（全検証）の各行を、簡易しきい値で OK/要改善/不一致/参考 に分類した一覧（目安）。",
            }

    return {
        "id": "validation_scoreboard",
        "title": "総合スコアボード（全検証）",
        "kind": "決定的検証（Phase 7）",
        "path": png_path,
        "summary_lines": summary_lines,
        "explain_lines": explain_lines,
        "detail_lines": [
            "再現: scripts/summary/validation_scoreboard.py → output/private/summary/validation_scoreboard.(png|json)",
            "補足: σ評価可能な項目の集計は Table 1（output/private/summary/paper_table1_results.json）を元に計算。",
        ],
        **({"table": table} if table else {}),
    }


def _extract_decisive_falsification_card(root: Path) -> Dict[str, Any]:
    json_path = root / "output" / "private" / "summary" / "decisive_falsification.json"
    png_path = root / "output" / "private" / "summary" / "decisive_falsification.png"

    j = _try_read_json(json_path)
    if not isinstance(j, dict) or not png_path.exists():
        return {
            "id": "decisive_falsification",
            "title": "反証条件パック（必要精度と棄却条件）",
            "kind": "決定的検証（Phase 7）",
            "summary_lines": [
                "未生成: python -B scripts/summary/decisive_falsification.py を実行してください。",
            ],
            "detail_lines": [
                "目的: 『どの観測がどの精度に到達すれば、P-modelとGRの差を3σで判別できるか』を数値で示す。",
            ],
        }

    policy = j.get("policy") or {}
    if not isinstance(policy, dict):
        policy = {}
    beta = policy.get("beta")
    delta_adopted = policy.get("delta")

    eht = j.get("eht") or {}
    if not isinstance(eht, dict):
        eht = {}
    ratio = eht.get("shadow_diameter_coeff_ratio_p_over_gr")
    diff_percent = eht.get("shadow_diameter_coeff_diff_percent")
    rel_sigma_needed = eht.get("rel_sigma_needed_3sigma_percent")
    eht_rows = eht.get("rows") or []
    if not isinstance(eht_rows, list):
        eht_rows = []

    delta = j.get("delta") or {}
    if not isinstance(delta, dict):
        delta = {}
    gamma_max = delta.get("gamma_max_for_delta_adopted")
    delta_rows = delta.get("rows") or []
    if not isinstance(delta_rows, list):
        delta_rows = []

    summary_lines: List[str] = []
    if ratio is not None and diff_percent is not None:
        summary_lines.append(
            f"EHT（影直径係数）: 係数比 P/GR={_format_num(float(ratio), digits=10)}（差={_format_num(float(diff_percent), digits=4)}%）"
        )
    if rel_sigma_needed is not None:
        summary_lines.append(f"3σ判別に必要な総合1σ（相対）≈{_format_num(float(rel_sigma_needed), digits=4)}%")

    # EHT: per-target gap summary
    for r in eht_rows:
        if not isinstance(r, dict):
            continue
        name = _as_str(r.get("name"))
        sigma_now = r.get("sigma_obs_now_uas")
        sigma_now_kappa = r.get("sigma_obs_now_with_kappa_uas")
        sigma_now_kappa_scatt = r.get("sigma_obs_now_with_kappa_scattering_uas")
        sigma_need = r.get("sigma_obs_needed_3sigma_uas")
        gap = r.get("gap_factor_now_over_needed")
        gap_kappa = r.get("gap_factor_now_over_needed_with_kappa")
        gap_kappa_scatt = r.get("gap_factor_now_over_needed_with_kappa_scattering")
        theta_rel_now_pct = r.get("theta_unit_rel_sigma_now_pct")
        theta_rel_need_pct = r.get("theta_unit_rel_sigma_needed_pct")
        if not name:
            continue
        if sigma_now is None:
            continue
        try:
            kappa_hint = ""
            gap_hint = ""
            if sigma_now_kappa is not None:
                kappa_hint += f"（参考:+κ={_format_num(float(sigma_now_kappa), digits=4)} μas）"
            if sigma_now_kappa_scatt is not None:
                kappa_hint += f"（参考:+κ+散乱={_format_num(float(sigma_now_kappa_scatt), digits=4)} μas）"
            if gap_kappa is not None:
                gap_hint += f"（参考×{float(gap_kappa):.2f}）"
            if gap_kappa_scatt is not None:
                gap_hint += f"（参考×{float(gap_kappa_scatt):.2f}）"

            if sigma_need is None or not math.isfinite(float(sigma_need)):
                if theta_rel_now_pct is not None and theta_rel_need_pct is not None:
                    summary_lines.append(
                        f"{name}: 必要σ_obs=n/a（θ_unit相対誤差={float(theta_rel_now_pct):.1f}% > 要求={float(theta_rel_need_pct):.1f}%）"
                    )
                else:
                    summary_lines.append(f"{name}: 必要σ_obs=n/a（質量/距離の不確かさが支配）")
            else:
                if gap is None or not math.isfinite(float(gap)):
                    summary_lines.append(
                        f"{name}: 現状σ_obs={_format_num(float(sigma_now), digits=4)} μas{kappa_hint} → 必要σ_obs={_format_num(float(sigma_need), digits=4)} μas"
                    )
                else:
                    summary_lines.append(
                        f"{name}: 現状σ_obs={_format_num(float(sigma_now), digits=4)} μas{kappa_hint} → 必要σ_obs={_format_num(float(sigma_need), digits=4)} μas（ギャップ×{float(gap):.2f}）{gap_hint}"
                    )
        except Exception:
            continue

    # δ: compact summary (kept minimal because δ is a constraint, not a measured fit)
    if delta_adopted is not None and gamma_max is not None:
        try:
            summary_lines.append(
                f"速度飽和: 採用δ={float(delta_adopted):.0e} → γ_max≈{float(gamma_max):.0e}（既知観測からはまだ未到達）"
            )
        except Exception:
            pass

    # Table: EHT gaps
    table_rows: List[List[str]] = []
    for r in eht_rows:
        if not isinstance(r, dict):
            continue
        name = _as_str(r.get("name"))
        if not name:
            continue
        diff_uas = r.get("diff_uas")
        sigma_now = r.get("sigma_obs_now_uas")
        sigma_now_kappa = r.get("sigma_obs_now_with_kappa_uas")
        sigma_now_kappa_scatt = r.get("sigma_obs_now_with_kappa_scattering_uas")
        sigma_need = r.get("sigma_obs_needed_3sigma_uas")
        gap = r.get("gap_factor_now_over_needed")
        gap_kappa = r.get("gap_factor_now_over_needed_with_kappa")
        gap_kappa_scatt = r.get("gap_factor_now_over_needed_with_kappa_scattering")
        z_now = r.get("z_separation_now_sigma")
        z_now_k = r.get("z_separation_now_with_kappa_sigma")
        z_now_ks = r.get("z_separation_now_with_kappa_scattering_sigma")
        src = _as_str(r.get("source_keys"))

        def _f(v: Any, digits: int = 4) -> str:
            try:
                return _format_num(float(v), digits=digits) if v is not None else ""
            except Exception:
                return _as_str(v)

        table_rows.append(
            [
                name,
                _f(diff_uas),
                _f(sigma_now),
                _f(sigma_now_kappa),
                _f(sigma_now_kappa_scatt),
                _f(sigma_need),
                "" if gap is None else f"{float(gap):.2f}",
                "" if gap_kappa is None else f"{float(gap_kappa):.2f}",
                "" if gap_kappa_scatt is None else f"{float(gap_kappa_scatt):.2f}",
                "" if z_now is None else f"{float(z_now):.2f}",
                "" if z_now_k is None else f"{float(z_now_k):.2f}",
                "" if z_now_ks is None else f"{float(z_now_ks):.2f}",
                src,
            ]
        )

    eht_table = None
    if table_rows:
        eht_table = {
            "headers": [
                "対象",
                "差Δθ[μas]",
                "現状σ_obs[μas]",
                "参考σ_obs[μas](+κ)",
                "参考σ_obs[μas](+κ+散乱)",
                "必要σ_obs[μas]（3σ）",
                "ギャップ",
                "ギャップ(+κ)",
                "ギャップ(+κ+散乱)",
                "Δ/σ（現状）",
                "Δ/σ（+κ）",
                "Δ/σ（+κ+散乱）",
                "出典キー",
            ],
            "rows": table_rows,
            "caption": "EHT（リング直径）から『影直径係数』を推定し、P-modelとGRの差を3σで判別するための必要精度（σ_obs）を評価（κ/散乱は参考）。",
        }

    delta_hint_lines: List[str] = []
    if delta_rows:
        # Show only the tightest (smallest) delta upper bound as a short hint.
        try:
            tightest = min((r for r in delta_rows if isinstance(r, dict)), key=lambda x: float(x.get("delta_upper_from_gamma")))
            label = _as_str(tightest.get("label"))
            du = tightest.get("delta_upper_from_gamma")
            if label and du is not None:
                delta_hint_lines.append(f"既知観測の上限（例）: {label} → δ < {float(du):.0e}（概算）")
        except Exception:
            pass

    return {
        "id": "decisive_falsification",
        "title": "反証条件パック（必要精度と棄却条件）",
        "kind": "決定的検証（Phase 7）",
        "path": png_path,
        "summary_lines": summary_lines,
        "explain_lines": [
            "『決定的』にするため、P-modelとGRの“差が出る観測量”について、必要精度と棄却条件を数値で示す。",
            "上段（EHT）は影直径係数の数%差を前提に、3σで判別するのに必要な観測誤差 σ_obs（影直径換算）の目安を計算する。",
            "下段（δ）は速度飽和がある場合の上限を、既知の高γ観測（概算）と突き合わせる（δは“測定値”ではなく制約）。",
        ],
        "detail_lines": [
            *(
                [f"β固定: β={_format_num(float(beta), digits=10)}（source={_as_str(policy.get('beta_source')) or 'unknown'}）"]
                if beta is not None
                else []
            ),
            *delta_hint_lines,
            "棄却条件（EHT）: β固定後、観測から得た影直径係数 r_obs と総合誤差 σ_r に対して、|r_obs - r_P| > 3σ_r なら P-model（β固定）は棄却。|r_obs - r_GR| > 3σ_r なら GR（Schwarzschild近似）は棄却。",
            "注意: EHTはリング直径→影直径の変換係数 κ、Kerrスピン、散乱などの系統誤差を含むため、σ_r はそれらも含めた“総合誤差”で評価する。",
            "棄却条件（δ）: 観測で γ_obs > 1/√δ が確定すれば、その δ は棄却（観測更新により δ < 1/γ_obs^2 の上限が下がる）。",
            "再現: scripts/summary/decisive_falsification.py → output/private/summary/decisive_falsification.(png|json)",
        ],
        "table": eht_table,
    }


def _extract_decisive_candidates_card(root: Path) -> Dict[str, Any]:
    json_path = root / "output" / "private" / "summary" / "decisive_candidates.json"
    png_path = root / "output" / "private" / "summary" / "decisive_candidates.png"

    j = _try_read_json(json_path)
    if not isinstance(j, dict) or not png_path.exists():
        return {
            "id": "decisive_candidates",
            "title": "差分予測候補（Phase 8.1）",
            "kind": "差分予測の拡張（Phase 8）",
            "summary_lines": [
                "未生成: python -B scripts/summary/decisive_candidates.py を実行してください。",
            ],
            "detail_lines": [
                "目的: GRとP-modelの差が“必ず出る”観測量を候補として棚卸しし、必要精度と支配誤差（系統）でスクリーニングする。",
            ],
        }

    candidates = j.get("candidates") if isinstance(j.get("candidates"), list) else []
    summary_lines: List[str] = []

    eht_gaps: List[str] = []
    delta_gap = None
    for c in candidates:
        if not isinstance(c, dict):
            continue
        topic = _as_str(c.get("topic"))
        if topic.startswith("EHT"):
            tgt = _as_str(c.get("target"))
            gap = c.get("gap_factor_now_over_needed_with_kappa")
            if gap is None:
                gap = c.get("gap_factor_now_over_needed")
            if tgt and gap is not None:
                try:
                    eht_gaps.append(f"{tgt}: ギャップ≈{_format_num(float(gap), digits=3)}×")
                except Exception:
                    pass
        if topic.startswith("速度飽和"):
            delta_gap = c.get("gap_gamma_needed_over_obs")

    if eht_gaps:
        summary_lines.append("EHT（影直径係数）の必要精度ギャップ（参考:+κ）: " + " / ".join(eht_gaps))
    if delta_gap is not None:
        try:
            summary_lines.append(f"速度飽和δ: γ_needed/γ_obs_max≈{_format_sci(delta_gap, digits=2)}×（概算）")
        except Exception:
            pass

    table_rows: List[List[str]] = []
    for c in candidates:
        if not isinstance(c, dict):
            continue
        topic = _as_str(c.get("topic"))
        target = _as_str(c.get("target"))
        observable = _as_str(c.get("observable"))
        status = _as_str(c.get("status"))

        need = ""
        now = ""
        gap = ""

        if topic.startswith("EHT"):
            try:
                need = f"σ_total≤{_format_num(float(c.get('sigma_needed_3sigma_uas')), digits=3)} μas"
            except Exception:
                need = ""
            try:
                now_val = c.get("sigma_now_uas")
                now = f"σ_total≈{_format_num(float(now_val), digits=3)} μas" if now_val is not None else ""
            except Exception:
                now = ""
            try:
                now_k = c.get("sigma_now_with_kappa_uas")
                if now_k is not None:
                    now += f"（参考:+κ={_format_num(float(now_k), digits=3)}）"
            except Exception:
                pass
            try:
                g = c.get("gap_factor_now_over_needed_with_kappa")
                if g is None:
                    g = c.get("gap_factor_now_over_needed")
                if g is not None:
                    gap = f"{_format_num(float(g), digits=3)}×"
            except Exception:
                gap = ""

        if topic.startswith("速度飽和"):
            try:
                need = f"γ≳{_format_sci(c.get('gamma_needed_to_probe_delta_adopted'), digits=2)}"
            except Exception:
                need = ""
            try:
                now = f"γ_max(既知)≈{_format_sci(c.get('gamma_obs_max_in_sources'), digits=2)}"
            except Exception:
                now = ""
            try:
                g = c.get("gap_gamma_needed_over_obs")
                if g is not None:
                    gap = f"{_format_sci(g, digits=2)}×"
            except Exception:
                gap = ""

        table_rows.append([topic, target, observable, need, now, gap, status])

    table = {
        "headers": ["候補", "対象", "観測量（要約）", "必要精度（3σ）", "現状精度（代表）", "ギャップ", "状態"],
        "rows": table_rows,
        "caption": "Phase 8.1 の棚卸し（第一版）。差分予測の中心（EHT）と、飽和δの反証条件（概算）を同じ枠で俯瞰する。",
    }

    return {
        "id": "decisive_candidates",
        "title": "差分予測候補（Phase 8.1）",
        "kind": "差分予測の拡張（Phase 8）",
        "path": png_path,
        "summary_lines": summary_lines or ["差分予測候補の棚卸し（第一版）を生成しました。"],
        "explain_lines": [
            "Phase 7（決定的検証パック）で作った『反証条件（必要精度＋棄却条件）』を土台に、Phase 8 では“決定打”になり得る候補を拡張する。",
            "ここでは候補ごとに『必要精度（3σ判別）』『現状精度（代表）』『ギャップ（倍率）』を同じフォーマットで並べる。",
        ],
        "detail_lines": [
            "EHTは κ（リング/シャドウ比）・スピン/傾斜・散乱などの系統が支配しやすいため、現状σだけでなく『参考:+κ』の誤差予算も併記する。",
            "速度飽和δは“測定値”ではなく、既知の高γ観測と矛盾しないための制約（概算）として扱う（粒子質量仮定に依存）。",
            "再現: scripts/summary/decisive_candidates.py → output/private/summary/decisive_candidates.(png|json)",
        ],
        "table": table,
    }


def _extract_paper_html_card(root: Path) -> Dict[str, Any]:
    paper_html = root / "output" / "private" / "summary" / "pmodel_paper.html"
    manuscript_md = root / "doc" / "paper" / "10_manuscript.md"
    sources_md = root / "doc" / "paper" / "20_data_sources.md"

    if paper_html.exists():
        return {
            "id": "paper_html",
            "title": "論文（HTML版）",
            "kind": "論文化（Phase 8 / Step 8.2）",
            "detail_href": "pmodel_paper.html",
            "summary_lines": [
                "論文本体（Markdown）を、レポートと同じカード形式の単一HTMLにまとめたもの。",
                "Table 1（検証サマリ）・本文・図表一覧・一次ソース一覧へすぐ辿れる。",
            ],
            "detail_lines": [
                f"出力: {_rel_repo_path(root, paper_html)}",
                f"本文: {_rel_repo_path(root, manuscript_md)}",
                f"一次ソース: {_rel_repo_path(root, sources_md)}",
                "生成（推奨）: cmd /c scripts\\summary\\build_materials.bat quick-nodocx",
                "生成（Full）: cmd /c scripts\\summary\\build_materials.bat",
            ],
        }

    return {
        "id": "paper_html",
        "title": "論文（HTML版）",
        "kind": "論文化（Phase 8 / Step 8.2）",
        "summary_lines": [
            "未生成: cmd /c scripts\\summary\\build_materials.bat quick-nodocx を実行してください。",
        ],
        "detail_lines": [
            f"期待する出力: {_rel_repo_path(root, paper_html)}",
            f"本文: {_rel_repo_path(root, manuscript_md)}",
            f"一次ソース: {_rel_repo_path(root, sources_md)}",
            "生成（HTMLのみ）: python -B scripts/summary/paper_build.py --mode publish --outdir output/private/summary --skip-docx",
        ],
    }


def _extract_recent_worklog_table(root: Path, *, n: int = 10) -> Optional[Dict[str, Any]]:
    path = root / "output" / "private" / "summary" / "work_history.jsonl"
    if not path.exists():
        return None

    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    tail = lines[-max(1, int(n)) :]

    rows: List[List[str]] = []
    for ln in reversed(tail):
        try:
            ev = json.loads(ln)
        except Exception:
            continue
        ts = str(ev.get("generated_utc") or "")
        event_type = str(ev.get("event_type") or "")

        # Public report: avoid mentioning blocked topics (e.g. BepiColombo) even if scripts were run internally.
        if "bepicolombo" in event_type.lower():
            continue

        outputs = ev.get("outputs") or {}
        out_hint = ""
        if isinstance(outputs, dict):
            if any("bepicolombo" in str(v).lower() for v in outputs.values() if isinstance(v, str)):
                continue
            preferred = [
                "public_report_html",
                "dashboard_png",
                "png",
                "png_differential",
                "csv",
                "json",
                "metrics_json",
                "status_path",
            ]
            for k in preferred:
                v = outputs.get(k)
                if isinstance(v, str) and v:
                    out_hint = v
                    break
            if not out_hint:
                out_hint = ", ".join(sorted(str(k) for k in outputs.keys())[:6])
        rows.append([ts, event_type, out_hint])

    return {
        "headers": ["UTC", "event", "代表出力（例）"],
        "rows": rows,
        "caption": "機械可読ログ output/private/summary/work_history.jsonl の直近イベント（重複作業の防止用）。",
    }


def _extract_run_all_status_card(root: Path) -> Optional[Dict[str, Any]]:
    status_path = root / "output" / "private" / "summary" / "run_all_status.json"
    st = _try_read_json(status_path)
    if not isinstance(st, dict):
        return None

    generated = str(st.get("generated_utc") or "")
    mode = str(st.get("mode") or "")
    include_blocked = bool(st.get("include_blocked") or False)

    ok_n = 0
    skipped_n = 0
    fail_n = 0
    rows: List[List[str]] = []
    for rec in st.get("tasks") or []:
        if not isinstance(rec, dict):
            continue
        key = str(rec.get("key") or "")

        # Public report: do not surface blocked topics.
        if "bepicolombo" in key.lower():
            continue

        if rec.get("skipped"):
            skipped_n += 1
            reason = str(rec.get("reason") or "")
            rows.append([key, "skipped", "", reason])
            continue

        if rec.get("ok") is True:
            ok_n += 1
            continue

        fail_n += 1
        elapsed = rec.get("elapsed_s")
        elapsed_s = "" if elapsed is None else _format_num(elapsed, digits=2)
        log = str(rec.get("log") or "")
        rows.append([key, "fail", elapsed_s, log])

    summary_lines: List[str] = []
    if generated:
        summary_lines.append(f"最終実行（UTC）: {generated}")
    if mode:
        summary_lines.append(f"モード: {mode}")
    summary_lines.append(f"タスク: ok={ok_n}, skipped={skipped_n}, fail={fail_n}")
    if include_blocked:
        summary_lines.append("include_blocked=1（ブロック中テーマは公開レポートでは非表示）")

    table = {
        "headers": ["task", "status", "elapsed_s", "note"],
        "rows": rows[:30],
        "caption": "失敗/スキップの要点のみ（全ログは output/private/summary/logs/ と run_all_status.json）。",
    }

    return {
        "id": "run_all_status",
        "title": "再現バッチ（run_all）の状態",
        "kind": "再現性（Phase 2）",
        "summary_lines": summary_lines,
        "detail_lines": [
            "再現: python -B scripts/summary/run_all.py --offline",
            "状況: output/private/summary/run_all_status.json",
            "ログ: output/private/summary/logs/",
        ],
        "table": table,
    }


def _render_llr_detail_html(
    *,
    out_dir: Path,
    graphs: List[Dict[str, Any]],
    llr_stem: str,
    llr_summary: Dict[str, Any],
    llr_source: Dict[str, Any],
    llr_batch_summary: Dict[str, Any],
) -> Path:
    details_dir = out_dir / "details"
    details_dir.mkdir(parents=True, exist_ok=True)
    html_path = details_dir / "llr.html"

    title = f"{LLR_LONG_NAME}詳細解説"
    subtitle = f"一覧ページの{LLR_SHORT_NAME}グラフを、図ごとに「何を見ているか」「どう解釈するか」を結びつけて説明します。"

    def _as_str(x: Any) -> str:
        return "" if x is None else str(x)

    def _h(text: Any) -> str:
        return html.escape(_as_str(text))

    selected_info: List[str] = []
    if llr_stem == "llr_primary":
        src = " / ".join(
            [x for x in [llr_source.get("source"), llr_source.get("target"), llr_source.get("filename")] if x]
        )
        if src:
            selected_info.append(f"入力（一次データ）: {src}")
        else:
            selected_info.append("入力（一次データ）: EDC公開データ（data/llr にキャッシュ）")
    else:
        selected_info.append("入力（デモ）: demo_llr_like.crd（動作確認用）")
    if llr_summary:
        if llr_summary.get("station"):
            selected_info.append(f"観測局: {llr_summary.get('station')}")
        if llr_summary.get("target"):
            selected_info.append(f"反射器: {llr_summary.get('target')}")
        if llr_summary.get("n_npt11") is not None:
            selected_info.append(f"点数（CRD Normal Point, record 11）: {llr_summary.get('n_npt11')}")
    if llr_batch_summary:
        selected_info.append(
            f"バッチ: {llr_batch_summary.get('n_files')}ファイル / {llr_batch_summary.get('n_points_total')}点（station×reflector）"
        )

    # Per-figure details (kept compact but more explicit than the public page).
    detail: Dict[str, List[str]] = {
        "llr_time_tag_selection": [
            "観測ファイルに書かれた時刻が tx/rx/mid のどれか不明確な場合、モデル評価時刻がズレて残差が大きく崩れる。",
            "局ごとに tx/rx/mid を総当たりし、station×reflector 単位で定数オフセット整列した残差RMSが最小のモードを採用する。",
            "以降の図（バッチ集計を含む）は、この選ばれた time-tag を既定として固定し、再現性を優先する。",
        ],
        "llr_tof_timeseries": [
            "月レーザー測距（LLR: Lunar Laser Ranging）は、地上局→月面反射器→地上局の「往復」レーザーの飛行時間（TOF）を測る観測。",
            "Normal Point（NP）は、短時間に得た多数のショットを平均化してノイズを下げたデータ（CRD record 11）。",
            "TOFの変化は主に「地球自転」「地上局の位置」「月の位置・回転」「反射器の位置」による幾何で決まる。",
            "ns精度で詰めると、1 ns（往復）は光路長で約0.30 m、片道距離で約0.15 mに相当。",
        ],
        "llr_range_timeseries": [
            "TOFを光速で換算して片道距離（range）にした参考図（概ね range = c*TOF/2）。",
            "厳密には大気遅延や局内部遅延などが入るため、絶対値（オフセット）よりも変動の形を見る用途が中心。",
        ],
        "llr_overlay": [
            "観測TOFとモデルTOFを同一軸に重ね、形（時間変化）が合うかを確認する図。",
            "比較の前に定数オフセットを整列して、装置遅延・基準系の定数差を吸収し、変動成分に注目する。",
            "モデル側は HORIZONS（観測局のtopocentric幾何/EOPを含む）とSPICE（月回転 MOON_PA_DE421）で反射器位置を扱う。",
            "ここで合わない場合は、(1)座標系の不整合 (2)月回転モデル (3)大気遅延 (4)反射器座標の出典差 などが疑わしい。",
        ],
        "llr_residual": [
            "残差 = 観測 - モデル（定数オフセット整列後）。0に近いほどモデルが観測の変動を説明している。",
            "残差の形が周期的なら、未補正の地球自転/章動/大気/潮汐などの系統誤差の可能性が高い。",
            "バラつき（RMS）でns級に入るかをチェックする。",
        ],
        "llr_residual_compare": [
            "モデルを段階的に精密化したときの効果を比較する図。",
            "地球中心（geocenter）→観測局（topocentric）→反射器（reflector）と進めるほど、幾何の誤差が減り残差RMSが下がるのが理想。",
            "LLRは「観測局」と「月面の固定点」を正しく扱うことが支配的に効く。",
        ],
        "llr_batch_improvement": [
            "複数局×複数反射器のデータをまとめて評価し、モデル改善が統計的に効いているかを見る図。",
            "グループ（station×reflector）ごとに定数オフセット整列を行い、変動成分の残差RMSを比較する。",
        ],
        "llr_batch_station_target": [
            "局×反射器の組み合わせごとの残差RMSをヒートマップ化。",
            "特定の局・反射器だけ悪い場合、局座標/ログ情報/データ品質の問題を切り分けやすい。",
        ],
        "llr_batch_station_month": [
            "局ごとの残差RMSを月別に集計し、期間依存（運用差・データ品質差）があるかを確認する。",
            "特定月だけ悪化する場合は、局ログ更新・装置変更・天候/大気条件・データ編集方針などの影響が疑わしい。",
        ],
        "llr_batch_grsm_target_month": [
            "残差が大きい局（例：GRSM）について、反射器別×月別に分解して原因を切り分ける。",
            "反射器間で同時に悪化する月は局側、反射器ごとに悪化するなら幾何/月回転/仰角分布などを疑う。",
        ],
        "llr_batch_ablations": [
            "太陽Shapiro（ON/OFF）と月回転モデル（IAU近似 vs SPICE）の影響を比較する切り分け図。",
            "LLRでは月回転モデル（反射器の向き・位置）が支配的に効くことが多く、Shapiroは補正として上乗せされる。",
        ],
        "llr_batch_shapiro": [
            "太陽ShapiroのON/OFF（必要なら地球Shapiroも）で、重力遅延が残差RMSにどれだけ寄与するかを定量化する。",
            "対流圏など他の誤差源を共通にONにして比較し、『Shapiroだけの差』を見える化する狙い。",
        ],
        "llr_batch_tide": [
            "潮汐補正（観測局潮汐/反射器側の月体潮汐）をON/OFFして寄与を定量化する。",
            "ns級では潮汐の寄与も効き始めるため、どちらが支配的かを切り分ける。",
        ],
        "llr_batch_outliers": [
            "バッチ評価では、稀な外れ値（スパイク）がRMSを破壊するため、station×reflectorごとにMADベースで外れ値をゲートしている。",
            "この図は、その“除外された点”がいつ/どこで起きたか（時刻・仰角・局・月）を可視化し、原因切り分けの入口を作る。",
            "Δは (観測TOF - モデルTOF) を反射器ごとの中央値で中心化した値（定数オフセット除去後のズレ）。",
            "ms級のΔは大気遅延やShapiroでは説明できないため、観測データ（一次ソース）側の異常値・編集・ログ整合を疑う。",
            "表の出典（file:line）から一次データを開き、該当レコードの品質/補正欄と整合を確認する。",
            "再現: scripts/llr/llr_batch_eval.py → output/llr/batch/llr_outliers_overview.png / llr_outliers.csv",
        ],
        "llr_batch_outliers_diagnosis": [
            "外れ値（スパイク）について、time-tag（tx/rx/mid）を切り替えたとき |Δ| がどれだけ変わるかを比較する。",
            "もし別のtime-tagで |Δ| が大きく小さくなるなら、時刻タグの解釈ミスや局ログ/時刻系の整合問題が疑わしい。",
            "どのtime-tagでも ms級の |Δ| が残る場合、物理補正では説明できないため一次データ側の異常（記録混入など）を優先して疑う。",
            "再現: scripts/llr/llr_batch_eval.py → output/llr/batch/llr_outliers_time_tag_sensitivity.png / llr_outliers_diagnosis.csv",
        ],
    }

    parts: List[str] = []
    parts.append("<!doctype html>")
    parts.append("<html lang='ja'><head>")
    parts.append("<meta charset='utf-8'>")
    parts.append(f"<title>{_h(title)}</title>")
    parts.append(
        "<style>"
        "body{font-family:Yu Gothic,Meiryo,BIZ UDGothic,MS Gothic,system-ui,sans-serif;margin:24px;max-width:1100px}"
        "h1{margin:0 0 6px 0;font-size:24px;line-height:1.25}"
        "h2{margin:24px 0 10px 0;font-size:20px;line-height:1.3;border-bottom:1px solid #ddd;padding-bottom:6px}"
        "h3{margin:0 0 8px 0;font-size:16px;line-height:1.35}"
        ".muted{color:#666;font-size:13px}"
        ".card{border:1px solid #e3e3e3;border-radius:10px;padding:14px 16px;margin:14px 0}"
        ".meta{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin:4px 0 10px 0}"
        ".badge{font-size:12px;padding:2px 8px;border-radius:999px;background:#f2f2f2;color:#333}"
        "ul{margin:6px 0 10px 18px}"
        "img{max-width:100%;height:auto;border:1px solid #ddd;border-radius:8px}"
        ".table-wrap{overflow-x:auto;margin:8px 0 10px 0}"
        "table{border-collapse:collapse;width:100%;font-size:12px}"
        "th,td{border:1px solid #ddd;padding:4px 6px;white-space:nowrap}"
        "th{background:#f6f6f6;text-align:left}"
        "a{color:#0b65c2;text-decoration:none}"
        "a:hover{text-decoration:underline}"
        "code{background:#f6f6f6;padding:1px 4px;border-radius:4px}"
        "</style>"
    )
    parts.append("</head><body>")

    parts.append("<div class='muted'>")
    parts.append("<a href='../pmodel_public_report.html'>← 一覧ページへ戻る</a>")
    parts.append("</div>")

    parts.append(f"<h1 id='top'>{_h(title)}</h1>")
    parts.append(f"<div class='muted'>{_h(subtitle)}</div>")

    parts.append("<div class='card'>")
    parts.append("<h3>このページの目的</h3><ul>")
    parts.append("<li>LLRグラフが「何の観測量」で「何を比較」しているのかを、図ごとに明確化する。</li>")
    parts.append("<li>モデルの正否は、残差（観測 - モデル）をns精度まで詰めて判断する。</li>")
    parts.append("</ul>")
    if selected_info:
        parts.append("<div class='muted'>使用データ</div><ul>")
        for ln in selected_info:
            parts.append(f"<li>{_h(ln)}</li>")
        parts.append("</ul>")
    parts.append("<div class='muted'>関連ドキュメント</div><ul>")
    parts.append("<li><a href='../../doc/llr/README.md'>doc/llr/README.md</a></li>")
    parts.append("<li><a href='../../doc/llr/MODEL_SPEC.md'>doc/llr/MODEL_SPEC.md</a></li>")
    parts.append("</ul>")
    parts.append("</div>")

    parts.append("<h2>図の一覧</h2><ul>")
    for g in graphs:
        gid = _as_str(g.get("id"))
        gt = _as_str(g.get("title"))
        if gid and gt:
            parts.append(f"<li><a href='#{_h(gid)}'>{_h(gt)}</a></li>")
    parts.append("</ul>")

    for g in graphs:
        gid = _as_str(g.get("id"))
        gt = _as_str(g.get("title"))
        kind = _as_str(g.get("kind"))
        path = g.get("path")
        summary_lines = g.get("summary_lines") or []
        explain_lines = g.get("explain_lines") or []
        table = g.get("table") or None

        parts.append(f"<h2 id='{_h(gid)}'>{_h(gt)}</h2>")
        parts.append("<div class='card'>")
        parts.append("<div class='meta'>")
        if kind:
            parts.append(f"<span class='badge'>{_h(kind)}</span>")
        parts.append("<a class='badge' href='#top'>ページ上部へ</a>")
        parts.append("</div>")

        # Image first (so text appears under the graph as requested).
        if isinstance(path, Path) and path.exists():
            rel = _rel_url(details_dir, path)
            parts.append(f"<a href='{_h(rel)}'><img src='{_h(rel)}' alt='{_h(gt)}'></a>")
            parts.append(f"<div class='muted'>画像: <code>{_h(rel)}</code>（クリックで拡大）</div>")
        else:
            parts.append(f"<div class='muted'>Missing: <code>{_h(path)}</code></div>")

        if detail.get(gid):
            parts.append("<div class='muted'>詳細解説</div><ul>")
            for ln in detail[gid]:
                parts.append(f"<li>{_h(ln)}</li>")
            parts.append("</ul>")

        if summary_lines:
            parts.append("<div class='muted'>一覧ページの概要</div><ul>")
            for ln in summary_lines:
                parts.append(f"<li>{_h(ln)}</li>")
            parts.append("</ul>")
        if explain_lines:
            parts.append("<div class='muted'>一覧ページの簡易解説</div><ul>")
            for ln in explain_lines:
                parts.append(f"<li>{_h(ln)}</li>")
            parts.append("</ul>")

        if isinstance(table, dict) and table.get("rows"):
            headers = table.get("headers") or []
            rows = table.get("rows") or []
            caption = _as_str(table.get("caption") or "")
            parts.append("<div class='muted'>一覧（抜粋）</div>")
            if caption:
                parts.append(f"<div class='muted'>{_h(caption)}</div>")
            parts.append("<div class='table-wrap'><table><thead><tr>")
            for h in headers:
                parts.append(f"<th>{_h(h)}</th>")
            parts.append("</tr></thead><tbody>")
            for r in rows:
                parts.append("<tr>")
                for cell in (r or []):
                    parts.append(f"<td>{_h(cell)}</td>")
                parts.append("</tr>")
            parts.append("</tbody></table></div>")

        parts.append("</div>")

    parts.append("</body></html>")
    html_path.write_text("\n".join(parts), encoding="utf-8")
    return html_path


def _extract_quantum_public_cards(root: Path) -> List[Dict[str, Any]]:
    out_q = root / "output" / "public" / "quantum"

    cards: List[Dict[str, Any]] = []

    # Positioning / scope note (public-facing)
    cards.append(
        {
            "id": "quantum_positioning",
            "title": "量子（位置づけ）",
            "kind": "Phase 7（量子）",
            "summary_lines": [
                "本プロジェクトは量子現象の“結論”を現段階で主張しない。",
                "ここでは公開データ（time-tag / trial log）に対して、観測手続き（選別条件）の前提がどこで入るかを明確化し、反証へ接続できる入口を整備する。",
            ],
            "detail_lines": [
                "関連: doc/quantum/04_bell_time_tag_primary_reanalysis.md（一次データ再解析の狙い）",
                "関連: doc/quantum/03_entanglement_local_p.md（未主張・反証条件）",
            ],
        }
    )

    def _safe_float(x: Any) -> Optional[float]:
        try:
            v = float(x)
        except Exception:
            return None
        if not math.isfinite(v):
            return None
        return v

    # Bell tests (time-tag / selection)
    nist_png = out_q / "nist_belltest_time_tag_bias.png"
    nist_m = _try_read_json(out_q / "nist_belltest_time_tag_bias_metrics.json") or {}
    nist_summary: List[str] = []
    if isinstance(nist_m, dict):
        ds = nist_m.get("delay_stats_ns") if isinstance(nist_m.get("delay_stats_ns"), dict) else {}
        a = ds.get("alice") if isinstance(ds.get("alice"), dict) else {}
        b = ds.get("bob") if isinstance(ds.get("bob"), dict) else {}
        ksa = _safe_float(a.get("ks_setting0_vs_1"))
        ksb = _safe_float(b.get("ks_setting0_vs_1"))
        if ksa is not None or ksb is not None:
            nist_summary.append(
                "設定依存のclick delay（KS距離）: "
                f"A={_format_num(ksa, digits=3) if ksa is not None else 'n/a'}, "
                f"B={_format_num(ksb, digits=3) if ksb is not None else 'n/a'}"
            )
        am0 = _safe_float(a.get("setting0_median"))
        am1 = _safe_float(a.get("setting1_median"))
        bm0 = _safe_float(b.get("setting0_median"))
        bm1 = _safe_float(b.get("setting1_median"))
        if am0 is not None and am1 is not None:
            nist_summary.append(f"A median delay: s0={am0:.1f} ns, s1={am1:.1f} ns（Δ={am0-am1:+.1f} ns）")
        if bm0 is not None and bm1 is not None:
            nist_summary.append(f"B median delay: s0={bm0:.1f} ns, s1={bm1:.1f} ns（Δ={bm0-bm1:+.1f} ns）")
    if not nist_summary:
        nist_summary = ["time-tag（生ログ）から、設定依存の時間構造が現れるかを可視化（KS距離＋window掃引）。"]
    if not nist_png.exists():
        nist_summary = ["未生成: python -B scripts/quantum/nist_belltest_time_tag_reanalysis.py"]
    cards.append(
        {
            "id": "quantum_bell_nist",
            "title": "ベルテスト（NIST）：time-tag 選別の入口",
            "kind": "量子（ベルテスト；time-tag）",
            **({"path": nist_png} if nist_png.exists() else {}),
            "summary_lines": nist_summary,
            "explain_lines": [
                "time-tag に基づく coincidence window / pairing が入ると、採用される事象集合（統計母集団）が変わり得る。",
                "設定ごとに click delay 分布が異なるなら、選別が“中立”という前提は自明ではない。",
            ],
            "detail_lines": [
                "取得: scripts/quantum/fetch_nist_belltestdata.py",
                "再解析: scripts/quantum/nist_belltest_time_tag_reanalysis.py",
                "補助（trial-based）: scripts/quantum/nist_belltest_trial_based_reanalysis.py",
            ],
        }
    )

    delft_png = out_q / "delft_hensen2015_chsh.png"
    delft_m = _try_read_json(out_q / "delft_hensen2015_chsh_metrics.json") or {}
    delft_summary: List[str] = []
    if isinstance(delft_m, dict):
        base = delft_m.get("baseline") if isinstance(delft_m.get("baseline"), dict) else {}
        s = _safe_float(base.get("S"))
        se = _safe_float(base.get("S_err"))
        nt = base.get("n_trials")
        pv = _safe_float(base.get("p_value"))
        if s is not None and se is not None:
            delft_summary.append(f"baseline CHSH: S={s:.3f}±{se:.3f}")
        if nt is not None:
            delft_summary.append(f"trial数（baseline）: n={int(nt)}")
        if pv is not None:
            delft_summary.append(f"p-value（baseline）={pv:.3g}")
    if not delft_summary:
        delft_summary = ["event-ready（trial-based）データで、window start の選択に対する CHSH S の感度を可視化。"]
    if not delft_png.exists():
        delft_summary = ["未生成: python -B scripts/quantum/delft_hensen2015_chsh_reanalysis.py"]
    cards.append(
        {
            "id": "quantum_bell_delft",
            "title": "ベルテスト（Delft）：event-ready window と CHSH",
            "kind": "量子（ベルテスト；trial-based）",
            **({"path": delft_png} if delft_png.exists() else {}),
            "summary_lines": delft_summary,
            "explain_lines": [
                "event-ready window の開始位置を動かすと、採用trial数や S が動き得る（selectionの入口）。",
            ],
            "detail_lines": [
                "取得: scripts/quantum/fetch_delft_hensen2015.py",
                "再解析: scripts/quantum/delft_hensen2015_chsh_reanalysis.py",
            ],
        }
    )

    delft2_png = out_q / "delft_hensen2016_srep30289_chsh.png"
    delft2_m = _try_read_json(out_q / "delft_hensen2016_srep30289_chsh_metrics.json") or {}
    delft2_summary: List[str] = []
    if isinstance(delft2_m, dict):
        base = delft2_m.get("baseline") if isinstance(delft2_m.get("baseline"), dict) else {}
        combined = base.get("combined") if isinstance(base.get("combined"), dict) else {}
        s = _safe_float(combined.get("S"))
        se = _safe_float(combined.get("S_err"))
        nt = base.get("n_trials_total")
        pv = _safe_float(base.get("p_value"))
        if s is not None and se is not None:
            delft2_summary.append(f"baseline combined CHSH: S={s:.3f}±{se:.3f}")
        if nt is not None:
            delft2_summary.append(f"trial数（baseline）: n={int(nt)}")
        if pv is not None:
            delft2_summary.append(f"p-value（baseline）={pv:.3g}")
    if not delft2_summary:
        delft2_summary = ["event-ready（trial-based）データで、旧/新 detector を統合し CHSH を評価。"]
    if not delft2_png.exists():
        delft2_summary = ["未生成: python -B scripts/quantum/delft_hensen2015_chsh_reanalysis.py --profile hensen2016_srep30289"]
    cards.append(
        {
            "id": "quantum_bell_delft_2016",
            "title": "ベルテスト（Delft 2016）：旧/新 detector 統合と CHSH",
            "kind": "量子（ベルテスト；trial-based）",
            **({"path": delft2_png} if delft2_png.exists() else {}),
            "summary_lines": delft2_summary,
            "explain_lines": [
                "旧/新 detector の2サブセットを合わせても、event-ready window start の選択で trial数や S が動き得る（selectionの入口）。",
                "psi+ と psi- を分けて評価し、最後に trial数重みで combined を報告（公開サンプルに合わせる）。",
            ],
            "detail_lines": [
                "取得: scripts/quantum/fetch_delft_hensen2016_srep30289.py",
                "再解析: scripts/quantum/delft_hensen2015_chsh_reanalysis.py --profile hensen2016_srep30289",
            ],
        }
    )

    weihs_png = out_q / "weihs1998_chsh_sweep_summary__multi_subdirs.png"
    weihs_m = _try_read_json(out_q / "weihs1998_chsh_sweep_summary__multi_subdirs_metrics.json") or {}
    weihs_summary: List[str] = []
    if isinstance(weihs_m, dict):
        series = weihs_m.get("series") if isinstance(weihs_m.get("series"), list) else []
        for s in series:
            if not isinstance(s, dict):
                continue
            subdir = _as_str(s.get("subdir"))
            run = _as_str(s.get("run"))
            mS = _safe_float(s.get("max_abs_S_fixed"))
            if subdir and run:
                weihs_summary.append(f"{subdir}/{run}: max|S|={_format_num(mS, digits=4) if mS is not None else 'n/a'}")
    if not weihs_summary:
        weihs_summary = ["複数条件（subdir）で coincidence window sweep の感度を横断比較。"]
    if not weihs_png.exists():
        weihs_summary = ["未生成: python -B scripts/quantum/weihs1998_time_tag_reanalysis.py（+ summary）"]
    cards.append(
        {
            "id": "quantum_bell_weihs",
            "title": "ベルテスト（Weihs 1998）：coincidence window sweep（複数条件）",
            "kind": "量子（ベルテスト；photon time-tag）",
            **({"path": weihs_png} if weihs_png.exists() else {}),
            "summary_lines": weihs_summary,
            "explain_lines": [
                "同一の time-tag 生ログでも、coincidence window の選び方で CHSH |S| が大きく変わり得る。",
            ],
            "detail_lines": [
                "取得: scripts/quantum/fetch_weihs1998_zenodo_7185335.py",
                "再解析: scripts/quantum/weihs1998_time_tag_reanalysis.py",
                "まとめ: scripts/quantum/weihs1998_chsh_sweep_summary.py",
            ],
        }
    )

    sel_png = out_q / "bell_selection_sensitivity_summary.png"
    sel_m = _try_read_json(out_q / "bell_selection_sensitivity_summary.json") or {}
    sel_summary: List[str] = []
    if isinstance(sel_m, dict):
        w = sel_m.get("weihs") if isinstance(sel_m.get("weihs"), dict) else {}
        w_min = _safe_float(w.get("abs_S_min"))
        w_max = _safe_float(w.get("abs_S_max"))
        if w_min is not None and w_max is not None:
            sel_summary.append(f"Weihs |S| range: {_format_num(w_min, digits=3)} → {_format_num(w_max, digits=3)}")

        n = sel_m.get("nist") if isinstance(sel_m.get("nist"), dict) else {}
        jmin = _safe_float(n.get("j_prob_sweep_min"))
        jmax = _safe_float(n.get("j_prob_sweep_max"))
        jtb = _safe_float(n.get("j_prob_trial_best"))
        if jmin is not None and jmax is not None:
            sel_summary.append(f"NIST J_prob (coincidence): {_format_num(jmin, digits=4)} → {_format_num(jmax, digits=4)}")
        if jtb is not None:
            sel_summary.append(f"NIST J_prob (trial-based best): {_format_num(jtb, digits=4)}")

        k = sel_m.get("kwiat2013") if isinstance(sel_m.get("kwiat2013"), dict) else {}
        kjmin = _safe_float(k.get("j_prob_sweep_min"))
        kjmax = _safe_float(k.get("j_prob_sweep_max"))
        kjtb = _safe_float(k.get("j_prob_trial_best"))
        if kjmin is not None and kjmax is not None:
            sel_summary.append(f"Kwiat2013 J_prob: {_format_num(kjmin, digits=4)} → {_format_num(kjmax, digits=4)}")
        if kjtb is not None:
            sel_summary.append(f"Kwiat2013 J_prob (baseline): {_format_num(kjtb, digits=4)}")

        d15 = sel_m.get("delft2015") if isinstance(sel_m.get("delft2015"), dict) else {}
        dsmin = _safe_float(d15.get("sweep_S_min"))
        dsmax = _safe_float(d15.get("sweep_S_max"))
        if dsmin is not None and dsmax is not None:
            sel_summary.append(f"Delft 2015 CHSH S (sweep): {_format_num(dsmin, digits=3)} → {_format_num(dsmax, digits=3)}")

        d16 = sel_m.get("delft2016") if isinstance(sel_m.get("delft2016"), dict) else {}
        d16min = _safe_float(d16.get("sweep_combined_S_min"))
        d16max = _safe_float(d16.get("sweep_combined_S_max"))
        if d16min is not None and d16max is not None:
            sel_summary.append(
                f"Delft 2016 CHSH S (combined sweep): {_format_num(d16min, digits=3)} → {_format_num(d16max, digits=3)}"
            )

    if not sel_summary:
        sel_summary = ["複数データセットで、selection（window/offset）が統計量をどれだけ動かすかを横断可視化。"]
    if not sel_png.exists():
        sel_summary = ["未生成: python -B scripts/quantum/bell_selection_sensitivity_summary.py"]
    cards.append(
        {
            "id": "quantum_bell_selection_sensitivity",
            "title": "ベル：selection感度の横断まとめ（NIST / Kwiat / Weihs / Delft）",
            "kind": "量子（ベルテスト；まとめ）",
            **({"path": sel_png} if sel_png.exists() else {}),
            "summary_lines": sel_summary,
            "explain_lines": [
                "同一の一次データでも、観測手続き（time-tagのwindow/pairing、event-ready window start）の選択が結果へ入る位置を明示する。",
            ],
            "detail_lines": [
                "生成: scripts/quantum/bell_selection_sensitivity_summary.py",
                "入力: output/public/quantum/bell/*（window_sweep/offset_sweep/covariance；固定出力）",
            ],
        }
    )

    # Photon interference observables (mapping-like summaries)
    qopt_png = out_q / "photon_quantum_interference.png"
    qopt_m = _try_read_json(out_q / "photon_quantum_interference_metrics.json") or {}
    qopt_summary: List[str] = []
    if isinstance(qopt_m, dict):
        sp = qopt_m.get("single_photon_interference") if isinstance(qopt_m.get("single_photon_interference"), dict) else {}
        sigL = _safe_float(sp.get("sigma_path_nm_from_visibility"))
        if sigL is not None:
            qopt_summary.append(f"単一光子: V≥0.8 → 等価 path noise σL≈{sigL:.1f} nm（簡易モデル）")
        sq = qopt_m.get("squeezing") if isinstance(qopt_m.get("squeezing"), dict) else {}
        eta = _safe_float(sq.get("eta_lower_if_perfect_intrinsic"))
        if eta is not None:
            qopt_summary.append(f"スクイーズド光: 10 dB → loss-only bound η≥{eta:.3f}")
    if not qopt_summary:
        qopt_summary = ["単一光子・HOM・スクイーズド光の観測量を一次ソースで固定し、簡易な対応関係（mapping）を提示。"]
    if not qopt_png.exists():
        qopt_summary = ["未生成: python -B scripts/quantum/photon_quantum_interference.py"]
    cards.append(
        {
            "id": "quantum_photon_interference",
            "title": "光の量子干渉：観測量の固定（visibility / HOM / squeezing）",
            "kind": "量子（光学；観測量）",
            **({"path": qopt_png} if qopt_png.exists() else {}),
            "summary_lines": qopt_summary,
            "explain_lines": [
                "装置遅延・time-tag・不可識別性が、どの観測量へ効くか（効かないか）を整理する入口。",
            ],
            "detail_lines": [
                "取得: scripts/quantum/fetch_photon_interference_sources.py",
                "実装: scripts/quantum/photon_quantum_interference.py",
            ],
        }
    )

    # Vacuum + QED precision (Casimir, Lamb)
    qed_png = out_q / "qed_vacuum_precision.png"
    qed_m = _try_read_json(out_q / "qed_vacuum_precision_metrics.json") or {}
    qed_summary: List[str] = []
    if isinstance(qed_m, dict):
        # Casimir: prefer the primary source's stated relative precision if present.
        rel_prec = None
        try:
            sources = qed_m.get("sources") if isinstance(qed_m.get("sources"), list) else []
            if sources and isinstance(sources[0], dict):
                abs_v = (sources[0].get("abstract_value") or {}) if isinstance(sources[0].get("abstract_value"), dict) else {}
                rel_prec = _safe_float(abs_v.get("relative_precision_at_closest_separation"))
        except Exception:
            rel_prec = None
        if rel_prec is not None:
            qed_summary.append(f"Casimir: closest-separation precision ≈{rel_prec*100:.1f}%（一次ソース）")
        qed_summary.append("Lamb shift: 代表スケーリング（Z^4）＋高次（Z^6）を整理（定義と系統の入口）")
    if not qed_summary:
        qed_summary = ["Casimir/Lamb の一次ソースと観測量を固定し、量子解釈の最低要件（再現できなければ棄却）を明文化。"]
    if not qed_png.exists():
        qed_summary = ["未生成: python -B scripts/quantum/qed_vacuum_precision.py"]
    cards.append(
        {
            "id": "quantum_qed_vacuum_precision",
            "title": "真空・QED精密：Casimir / Lamb shift",
            "kind": "量子（真空/QED；精密）",
            **({"path": qed_png} if qed_png.exists() else {}),
            "summary_lines": qed_summary,
            "explain_lines": [
                "“量子の解釈”は、Casimir/Lamb の精密現象学と矛盾すれば棄却される。",
                "ここでは導出主張ではなく、観測量と一次ソースの固定を優先する。",
            ],
            "detail_lines": [
                "取得: scripts/quantum/fetch_qed_vacuum_precision_sources.py",
                "実装: scripts/quantum/qed_vacuum_precision.py",
            ],
        }
    )

    return cards


def main() -> int:
    root = _repo_root()
    out_dir = root / "output" / "private" / "summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Source images (already produced by run_all)
    items: List[Tuple[str, Path]] = [
        ("太陽重力による光の偏向（観測γ vs P-model）", root / "output" / "theory" / "solar_light_deflection.png"),
        ("GPS（時計残差：観測IGS vs P-model）", root / "output" / "gps" / "gps_rms_compare.png"),
        ("Cassini ドップラー y（PDS一次データ vs P-model）", root / "output" / "cassini" / "cassini_fig2_overlay_zoom10d.png"),
        ("Viking シャピロ遅延（代表値 vs P-model）", root / "output" / "viking" / "viking_p_model_vs_measured_no_arrow.png"),
    ]

    # Metrics for annotations
    solar_meta = _try_read_json(root / "output" / "theory" / "solar_light_deflection_metrics.json") or {}
    gps_cmp = _try_read_json(root / "output" / "gps" / "gps_compare_metrics.json") or {}
    cassini_m = _extract_cassini_metrics(root)

    solar_m = (solar_meta.get("metrics") or {}) if isinstance(solar_meta, dict) else {}
    gps_m = (gps_cmp.get("metrics") or {}) if isinstance(gps_cmp, dict) else {}
    viking_peak = _extract_viking_peak(root)

    viking_notes: List[str] = ["最大遅延: 文献で約200-250 マイクロ秒"]
    if viking_peak.get("peak_us") is not None:
        peak_us = float(viking_peak["peak_us"])
        peak_t = _as_str(viking_peak.get("peak_time_utc"))
        if peak_t:
            viking_notes.insert(0, f"P-model 最大（往復）={peak_us:.2f} μs（{peak_t}）")
        else:
            viking_notes.insert(0, f"P-model 最大（往復）={peak_us:.2f} μs")
        viking_notes.append("文献代表値: 約250 μs")

    solar_best_label = _as_str(solar_m.get("observed_best_label")) or _as_str(solar_m.get("observed_best_id"))

    panel_notes: Dict[str, List[str]] = {
        "太陽重力による光の偏向（観測γ vs P-model）": [
            f"観測（{solar_best_label}）: γ={_format_num(solar_m.get('observed_gamma_best'), digits=6)}±{_format_num(solar_m.get('observed_gamma_best_sigma'), digits=6)}",
            f"P-model: γ={_format_num(solar_m.get('gamma_pmodel'), digits=6)}（β={_format_num(solar_m.get('beta'), digits=6)}）",
            f"太陽縁: α_obs={_format_num(solar_m.get('observed_alpha_arcsec_limb_best'), digits=7)}±{_format_num(solar_m.get('observed_alpha_sigma_arcsec_limb_best'), digits=7)} 角秒",
            f"z={_format_num(solar_m.get('observed_z_score_best'), digits=3)}（γの差/σ）",
        ],
        "GPS（時計残差：観測IGS vs P-model）": [
            f"観測: IGS Final CLK/SP3（準実測）",
            f"中央値RMS: BRDC={_format_num(gps_m.get('brdc_rms_ns_median'), digits=4)} ns",
            f"中央値RMS: P-model={_format_num(gps_m.get('pmodel_rms_ns_median'), digits=4)} ns",
            f"P-model優位={_format_num(gps_m.get('pmodel_better_count'), digits=3)}/{_format_num(gps_m.get('n_sats'), digits=3)} 衛星",
        ],
        "Cassini ドップラー y（PDS一次データ vs P-model）": [
            f"RMSE={_format_num(cassini_m.get('rmse'), digits=4)}",
            f"相関={_format_num(cassini_m.get('corr'), digits=6)}",
        ],
        "Viking シャピロ遅延（代表値 vs P-model）": viking_notes,
    }

    panel_explain: Dict[str, List[str]] = {
        "太陽重力による光の偏向（観測γ vs P-model）": [
            "光はPが高い側へ曲がる（最短時間経路）。",
            "最近接距離 b が小さいほど偏向角 α が大きくなる。",
            "右図はVLBI等の観測 γ±σ（一次ソース）と、P-model予測 γ=2β-1 の比較。",
        ],
        "GPS（時計残差：観測IGS vs P-model）": [
            "IGS（観測プロダクト）に対するクロック残差RMSを比較。",
            "棒が低いほど観測に近い（バイアス＋ドリフトは除去）。",
            "IGSの慣例に合わせ、P-model側も dt_rel（近日点効果）を除去。",
        ],
        "Cassini ドップラー y（PDS一次データ vs P-model）": [
            "太陽会合時の電波ドップラー y（周波数比）の時間変化。",
            "PDS一次データ（TDF）から処理して得た観測 y(t) と、P-model を比較。",
            "形状の一致度を RMSE / 相関 で要約。",
        ],
        "Viking シャピロ遅延（代表値 vs P-model）": [
            "地球-火星の往復通信で生じる Shapiro 遅延の時間変化。",
            "太陽会合で遅延が最大になり、離れると減少する。",
            "赤点は文献のピーク代表値（約250 µs）。",
        ],
    }

    try:
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
        from matplotlib import gridspec
    except Exception as e:
        print(f"[err] matplotlib not available: {e}")
        return 2

    _set_japanese_font()

    dashboard_title_overrides: Dict[str, str] = {
        "太陽重力による光の偏向（観測γ vs P-model）": "太陽重力による光の偏向",
        "GPS（時計残差：観測IGS vs P-model）": "GPS（時計残差）",
        "Cassini ドップラー y（PDS一次データ vs P-model）": "Cassini ドップラー y",
        "Viking シャピロ遅延（代表値 vs P-model）": "Viking シャピロ遅延",
    }

    # 2x2 layout, but each panel has a dedicated "note area" above the image,
    # so the summary never overlaps and hides the figure.
    fig = plt.figure(figsize=(16.4, 13.2))
    outer = gridspec.GridSpec(nrows=2, ncols=2, figure=fig, wspace=0.12, hspace=0.22)

    for idx, (title, path) in enumerate(items):
        r = idx // 2
        c = idx % 2
        inner = gridspec.GridSpecFromSubplotSpec(
            nrows=3,
            ncols=1,
            subplot_spec=outer[r, c],
            height_ratios=[2.7, 8.4, 2.2],
            hspace=0.06,
        )
        ax_note = fig.add_subplot(inner[0, 0])
        ax_img = fig.add_subplot(inner[1, 0])
        ax_explain = fig.add_subplot(inner[2, 0])

        ax_note.axis("off")
        ax_img.axis("off")
        ax_explain.axis("off")

        display_title = dashboard_title_overrides.get(title, title)
        ax_note.text(
            0.0,
            0.98,
            display_title,
            transform=ax_note.transAxes,
            va="top",
            ha="left",
            fontsize=13.5,
            fontweight="bold",
        )

        notes = panel_notes.get(title, [])
        if notes:
            ax_note.text(
                0.0,
                0.02,
                _panel_text("概要", notes),
                transform=ax_note.transAxes,
                va="bottom",
                ha="left",
                fontsize=10.2,
                linespacing=1.12,
            )

        expl = panel_explain.get(title, [])
        if expl:
            ax_explain.text(
                0.0,
                1.0,
                _panel_text("解説", expl),
                transform=ax_explain.transAxes,
                va="top",
                ha="left",
                fontsize=10.2,
                linespacing=1.12,
            )

        if not path.exists():
            ax_img.text(0.03, 0.97, f"Missing:\n{path}", transform=ax_img.transAxes, va="top")
            continue
        try:
            img = mpimg.imread(str(path))
            ax_img.imshow(img)
        except Exception as e:
            ax_img.text(0.03, 0.97, f"Failed to load:\n{path}\n{e}", transform=ax_img.transAxes, va="top")
            continue

    fig.suptitle("Pモデル 比較（一般向け）\n観測（実測/デジタイズ/代表値）・標準値 vs シミュレーション", fontsize=18, y=0.985)
    # tight_layout is unreliable with nested GridSpec; use fixed margins instead.
    fig.subplots_adjust(left=0.04, right=0.99, bottom=0.04, top=0.91)

    png_path = out_dir / "pmodel_public_dashboard.png"
    fig.savefig(png_path, dpi=220)
    plt.close(fig)

    # Public report (all graphs, scrollable)
    cassini_best = _extract_cassini_best_beta(root)
    cassini_pds_vs_dig: Dict[str, Any] = {}
    try:
        p = root / "output" / "cassini" / "cassini_pds_vs_digitized_metrics.csv"
        txt = _try_read_text(p)
        if txt:
            lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
            if len(lines) >= 2:
                header = lines[0].split(",")
                vals = lines[1].split(",")
                row = dict(zip(header, vals))
                for k in (
                    "n",
                    "rmse",
                    "corr",
                    "shift_days",
                    "rmse_zero_shift",
                    "corr_zero_shift",
                    "n_zero_shift",
                ):
                    v = row.get(k)
                    if v is None:
                        continue
                    try:
                        cassini_pds_vs_dig[k] = float(v) if (("." in v) or ("e" in v.lower())) else int(v)
                    except Exception:
                        cassini_pds_vs_dig[k] = v
    except Exception:
        cassini_pds_vs_dig = {}

    cassini_run_meta = _try_read_json(root / "output" / "cassini" / "cassini_fig2_run_metadata.json") or {}
    cassini_effective_source = ""
    cassini_obs_label = ""
    cassini_fallback_reason = ""
    if isinstance(cassini_run_meta, dict):
        cassini_effective_source = str(cassini_run_meta.get("effective_source") or "")
        cassini_fallback_reason = str(cassini_run_meta.get("fallback_reason") or "")
        labels = cassini_run_meta.get("labels")
        if isinstance(labels, dict):
            cassini_obs_label = str(labels.get("obs_label") or "")

    def _cassini_obs_kind() -> str:
        if cassini_effective_source.startswith("pds_tdf"):
            return "観測（一次データ: PDS TDF）"
        if cassini_effective_source.startswith("pds_odf"):
            return "観測（一次データ: PDS ODF）"
        if cassini_effective_source.startswith("digitized"):
            return "観測（論文図をデジタイズ）"
        return "観測（ソース不明）"

    cassini_obs_kind = _cassini_obs_kind()
    if cassini_fallback_reason == "pds_cache_missing":
        cassini_obs_kind += "（PDSキャッシュ未取得のため代替）"

    cassini_source_tag = "観測"
    if cassini_effective_source.startswith("pds_tdf_processed"):
        cassini_source_tag = "一次データ（PDS TDF, 処理後）"
    elif cassini_effective_source.startswith("pds_tdf_raw"):
        cassini_source_tag = "一次データ（PDS TDF, 生値）"
    elif cassini_effective_source.startswith("pds_odf_raw"):
        cassini_source_tag = "一次データ（PDS ODF, 生値）"
    elif cassini_effective_source.startswith("digitized"):
        cassini_source_tag = "論文図デジタイズ"
    bepi_more_psa = _extract_bepicolombo_more_psa_status(root)
    bepi_spice_psa = _extract_bepicolombo_spice_psa_status(root)
    bepi_shapiro_pred = _extract_bepicolombo_shapiro_predict(root)
    bepi_conj_catalog = _extract_bepicolombo_conjunction_catalog(root)
    bepi_more_docs = _extract_bepicolombo_more_document_catalog(root)
    llr_stem = _pick_llr_stem(root)
    llr_summary = _try_read_json(root / "output" / "llr" / f"{llr_stem}_summary.json") or {}
    llr_source = _try_read_json(root / "data" / "llr" / "llr_primary_source.json") or {}
    llr_model_metrics = _try_read_json(root / "output" / "llr" / "out_llr" / f"{llr_stem}_metrics.json") or {}
    llr_batch_summary = _try_read_json(root / "output" / "llr" / "batch" / "llr_batch_summary.json") or {}
    llr_outliers_summary = _try_read_json(root / "output" / "llr" / "batch" / "llr_outliers_summary.json") or {}
    llr_outliers_diag_summary = _try_read_json(root / "output" / "llr" / "batch" / "llr_outliers_diagnosis_summary.json") or {}
    llr_time_tag_best = _try_read_json(root / "output" / "llr" / "batch" / "llr_time_tag_best_by_station.json") or {}
    llr_inlier_rms = _try_compute_llr_inlier_rms(root / "output" / "llr" / "batch" / "llr_batch_points.csv") or {}
    rel_corr = gps_m.get("rel_corr")
    rel_rmse_ns = gps_m.get("rel_rmse_ns")

    llr_outliers_table: Optional[Dict[str, Any]] = None
    try:
        outliers_csv = root / "output" / "llr" / "batch" / "llr_outliers.csv"
        rows = _try_read_csv_rows(outliers_csv, max_rows=50) if outliers_csv.exists() else []
        if rows:
            table_rows: List[List[str]] = []
            for r in rows:
                t = r.get("epoch_utc", "")
                st = r.get("station", "")
                tgt = r.get("target", "")
                delta = r.get("delta_best_centered_ns", "")
                elev = r.get("elev_mean_deg", "")
                tag = r.get("time_tag_mode", "")
                src = f"{r.get('source_file','')}:{r.get('lineno','')}".strip(":")
                try:
                    delta_s = _format_num(float(delta), digits=6)
                except Exception:
                    delta_s = str(delta)
                try:
                    elev_s = _format_num(float(elev), digits=4)
                except Exception:
                    elev_s = str(elev)
                table_rows.append([t, st, tgt, delta_s, elev_s, tag, src])

            llr_outliers_table = {
                "headers": ["UTC時刻", "局", "反射器", "Δ[ns]（中心化）", "平均仰角[deg]", "time-tag", "出典（file:line）"],
                "rows": table_rows,
                "caption": "Δは (観測-モデル) を反射器ごとの中央値で中心化した値。外れ値は |Δ| の大きい順。",
            }
    except Exception:
        llr_outliers_table = None

    llr_outliers_diag_table: Optional[Dict[str, Any]] = None
    try:
        diag_csv = root / "output" / "llr" / "batch" / "llr_outliers_diagnosis.csv"
        rows = _try_read_csv_rows(diag_csv, max_rows=50) if diag_csv.exists() else []
        if rows:
            table_rows = []
            for r in rows:
                t = r.get("epoch_utc", "")
                st = r.get("station", "")
                tgt = r.get("target", "")
                cause = r.get("cause_hint", "")
                best = r.get("best_time_tag_mode", "")
                d0 = r.get("abs_delta_centered_ns", "")
                dtx = r.get("abs_delta_centered_tx_ns", "")
                drx = r.get("abs_delta_centered_rx_ns", "")
                dmid = r.get("abs_delta_centered_mid_ns", "")
                src = f"{r.get('source_file','')}:{r.get('lineno','')}".strip(":")

                def _fmt_float(v: str, digits: int = 4) -> str:
                    try:
                        return _format_num(float(v), digits=digits)
                    except Exception:
                        return str(v)

                table_rows.append(
                    [
                        t,
                        st,
                        tgt,
                        _fmt_float(d0, digits=4),
                        _fmt_float(dtx, digits=4),
                        _fmt_float(drx, digits=4),
                        _fmt_float(dmid, digits=4),
                        str(best),
                        str(cause),
                        src,
                    ]
                )

            llr_outliers_diag_table = {
                "headers": [
                    "UTC時刻",
                    "局",
                    "反射器",
                    "|Δ|[ns]（中心化）",
                    "|Δ|_tx[ns]",
                    "|Δ|_rx[ns]",
                    "|Δ|_mid[ns]",
                    "best",
                    "原因（暫定）",
                    "出典（file:line）",
                ],
                "rows": table_rows,
                "caption": "外れ値の |Δ| を time-tag（tx/rx/mid）別に比較。|Δ| は反射器ごとの中央値で中心化（定数オフセット除去）。",
            }
    except Exception:
        llr_outliers_diag_table = None

    def _fmt_ns_to_us(x: Any) -> str:
        if not isinstance(x, (int, float)):
            return "n/a"
        return _format_num(float(x) / 1e3, digits=4)

    def _llr_best_time_tag_summary_lines() -> List[str]:
        if not llr_time_tag_best:
            return []
        best_by_station = llr_time_tag_best.get("best_mode_by_station") or {}
        if not isinstance(best_by_station, dict) or not best_by_station:
            return []

        ordered = [(str(k), str(v)) for k, v in best_by_station.items()]
        ordered.sort(key=lambda kv: kv[0])
        s = ", ".join([f"{k}={v}" for k, v in ordered])

        metric = llr_time_tag_best.get("selection_metric")
        if metric:
            return [f"最適time-tag（局別）: {s}", f"評価指標: {metric}"]
        return [f"最適time-tag（局別）: {s}"]

    def _llr_outliers_diag_summary_lines() -> List[str]:
        if not llr_outliers_diag_summary:
            return ["外れ値診断（output/llr/batch/llr_outliers_diagnosis_summary.json）が見つかりません。"]
        tt = llr_outliers_diag_summary.get("time_tag_sensitivity") if isinstance(llr_outliers_diag_summary, dict) else None
        tm = llr_outliers_diag_summary.get("target_mixing_sensitivity") if isinstance(llr_outliers_diag_summary, dict) else None
        by_cause = llr_outliers_diag_summary.get("by_cause_hint") if isinstance(llr_outliers_diag_summary, dict) else None
        lines: List[str] = []
        if isinstance(tt, dict):
            lines.append(
                "time-tag感度: computed="
                + str(tt.get("computed"))
                + " modes="
                + ",".join([str(x) for x in (tt.get("computed_modes") or [])])
            )
            lines.append(
                "best!=current: "
                + str(tt.get("n_best_mode_differs"))
                + "/"
                + str(llr_outliers_diag_summary.get("n_outliers"))
            )
        if isinstance(tm, dict):
            lines.append(
                "ターゲット混入推定: "
                + str(tm.get("n_suspected"))
                + "/"
                + str(llr_outliers_diag_summary.get("n_outliers"))
            )
            try:
                if int(tm.get("n_suspected") or 0) > 0:
                    lines.append("注記: 混入疑いは統計から除外（再割当しない）")
            except Exception:
                pass
        if isinstance(by_cause, dict) and by_cause:
            parts = [f"{k}={v}" for k, v in by_cause.items()]
            lines.append("原因分類: " + ", ".join(parts))
        lines.append("一覧: output/llr/batch/llr_outliers_diagnosis.csv")
        return lines

    def _llr_outliers_target_mixing_summary_lines() -> List[str]:
        if not llr_outliers_diag_summary:
            return ["外れ値診断（output/llr/batch/llr_outliers_diagnosis_summary.json）が見つかりません。"]
        tm = llr_outliers_diag_summary.get("target_mixing_sensitivity") if isinstance(llr_outliers_diag_summary, dict) else None
        if not isinstance(tm, dict):
            return ["ターゲット混入診断（target_mixing_sensitivity）が見つかりません。"]
        n_outliers = llr_outliers_diag_summary.get("n_outliers")
        n_sus = tm.get("n_suspected")
        lines = [f"ターゲット混入推定: {_format_num(n_sus)}/{_format_num(n_outliers)}", "注記: 疑いがあっても自動再割当しない（統計から除外・一次データ行確認）"]
        crit = tm.get("criteria") if isinstance(tm.get("criteria"), dict) else None
        if isinstance(crit, dict):
            try:
                lines.append(f"判定基準: |Δ_raw|≥{_format_num(crit.get('abs_delta_raw_ge_ns'))} ns かつ best |Δ_raw|≤{_format_num(crit.get('best_abs_delta_raw_le_ns'))} ns")
            except Exception:
                pass
        return lines

    def _bepicolombo_more_psa_summary_lines() -> List[str]:
        if not bepi_more_psa:
            return [
                "一次ソース（ESA PSA）に公開されているかを確認し、検証の入口とする。",
                "出力（PNG/JSON）が未生成です。まず scripts/bepicolombo/more_psa_status.py を実行してください。",
            ]

        downloaded = bepi_more_psa.get("document_downloaded")
        total = bepi_more_psa.get("document_total")
        has_data_dirs = bepi_more_psa.get("has_data_dirs")
        errs = bepi_more_psa.get("errors") or []
        listing = bepi_more_psa.get("listing") or {}

        lines = [
            f"ドキュメント: {downloaded}/{total} 件をローカル保持",
            f"データ本体ディレクトリ公開: {'あり' if has_data_dirs else 'なし（404の可能性）'}",
        ]
        if isinstance(listing, dict):
            base_entries = listing.get("base_entries") if isinstance(listing.get("base_entries"), list) else []
            if base_entries:
                n_dir = sum(1 for e in base_entries if isinstance(e, dict) and e.get("is_dir") == "1")
                n_file = sum(1 for e in base_entries if isinstance(e, dict) and e.get("is_dir") == "0")
                lines.append(f"bc_mpo_more/ 直下: dir={n_dir}, file={n_file}")
        if errs:
            lines.append(f"注意: エラー {len(errs)} 件（details: output/bepicolombo/more_psa_status.json）")
        return lines

    def _bepicolombo_spice_psa_summary_lines() -> List[str]:
        if not bepi_spice_psa:
            return [
                "幾何（軌道/座標系）の一次ソースとして、SPICE kernels の公開状況を確認する。",
                "出力（PNG/JSON）が未生成です。まず scripts/bepicolombo/spice_psa_status.py を実行してください。",
            ]

        inv = bepi_spice_psa.get("inventory_latest") if isinstance(bepi_spice_psa, dict) else None
        inv = inv if isinstance(inv, dict) else {}
        n_total = inv.get("n_total")
        inv_name = inv.get("name")
        craft = inv.get("craft_counts") if isinstance(inv.get("craft_counts"), dict) else {}

        lines: List[str] = []
        if inv_name:
            lines.append(f"最新inventory: {inv_name}")
        if n_total is not None:
            lines.append(f"inventory件数: {n_total}")
        if craft:
            lines.append(
                f"spacecraft内訳: MPO={craft.get('mpo')}, MMO={craft.get('mmo')}, MTM={craft.get('mtm')}, other={craft.get('other')}"
            )
        return lines

    def _bepicolombo_shapiro_predict_summary_lines() -> List[str]:
        if not bepi_shapiro_pred:
            return [
                "SPICE（一次ソース）で幾何を計算し、太陽会合での Shapiro y(t) を予測する。",
                "出力（PNG/JSON）が未生成です。scripts/bepicolombo/bepicolombo_shapiro_predict.py を実行してください。",
            ]

        t0 = bepi_shapiro_pred.get("conjunction_center_utc")
        bmin = bepi_shapiro_pred.get("b_min_rsun")
        ypk = bepi_shapiro_pred.get("y_peak_eq2")
        dt_rng = bepi_shapiro_pred.get("dt_roundtrip_us_range")
        min_b = bepi_shapiro_pred.get("min_b_rsun")
        raw_b = bepi_shapiro_pred.get("b_min_raw_rsun_in_window")

        lines: List[str] = []
        if t0:
            lines.append(f"会合中心（b_min）: {t0}")
        if bmin is not None:
            lines.append(f"b_min={_format_num(bmin, digits=4)} R_sun（閾値={_format_num(min_b, digits=3)} R_sun）")
        if ypk is not None:
            lines.append(f"y_peak（Eq2, |y|最大）={_format_num(ypk, digits=4)}")
        if isinstance(dt_rng, list) and len(dt_rng) == 2:
            lines.append(f"往復Δt（μs, 非遮蔽のみ）={_format_num(dt_rng[0], digits=2)} .. {_format_num(dt_rng[1], digits=2)}")
        if raw_b is not None:
            lines.append(f"参考: 窓内の最小b（無制限）={_format_num(raw_b, digits=4)} R_sun")
        return lines

    def _bepicolombo_conjunction_catalog_summary_lines() -> List[str]:
        if not bepi_conj_catalog:
            return [
                "SPICE（一次ソース）で会合イベントを抽出し、Shapiro信号の強さを一覧化する。",
                "出力（PNG/JSON）が未生成です。scripts/bepicolombo/bepicolombo_conjunction_catalog.py を実行してください。",
            ]

        summ = bepi_conj_catalog.get("summary") if isinstance(bepi_conj_catalog, dict) else None
        if not isinstance(summ, dict):
            summ = {}

        lines: List[str] = []
        span = summ.get("span_utc") if isinstance(summ.get("span_utc"), dict) else {}
        if isinstance(span, dict):
            a = span.get("start")
            b = span.get("stop")
            if a and b:
                lines.append(f"期間: {a} 〜 {b}")

        n_events = summ.get("n_events")
        n_usable = summ.get("n_usable_events")
        if n_events is not None:
            lines.append(f"イベント数: {n_events}（usable: {n_usable}）")

        bmin = summ.get("b_min_nonocculted_rsun_min")
        if bmin is not None:
            try:
                lines.append(f"最小 b（非遮蔽）: {float(bmin):.6f} R_sun")
            except Exception:
                pass

        dtmax = summ.get("dt_roundtrip_us_max")
        if dtmax is not None:
            try:
                lines.append(f"最大 Δt（非遮蔽）: {float(dtmax):.2f} μs")
            except Exception:
                pass

        step = (
            (bepi_conj_catalog.get("scan") or {}).get("refine_step_sec")
            if isinstance(bepi_conj_catalog.get("scan"), dict)
            else None
        )
        if step is not None:
            try:
                lines.append(f"refine_step={float(step):.0f} s（時刻精度の目安）")
            except Exception:
                pass

        return lines

    bepi_docs_table: Optional[Dict[str, Any]] = None
    try:
        docs = bepi_more_docs.get("documents") if isinstance(bepi_more_docs, dict) else None
        generated = bepi_more_docs.get("generated_utc") if isinstance(bepi_more_docs, dict) else None
        if isinstance(docs, list) and docs:
            headers = ["タイトル", "文書番号", "版", "出版年", "発行日", "ファイル"]
            rows: List[List[str]] = []
            for d in docs:
                if not isinstance(d, dict):
                    continue
                rows.append(
                    [
                        _as_str(d.get("title")),
                        _as_str(d.get("document_name")),
                        _as_str(d.get("revision_id")),
                        _as_str(d.get("publication_year")),
                        _as_str(d.get("publication_date")),
                        _as_str(d.get("file_name")),
                    ]
                )
            bepi_docs_table = {
                "headers": headers,
                "rows": rows,
                "caption": f"PSA document/*.lblx から抽出（generated_utc={generated}）",
            }
    except Exception:
        bepi_docs_table = None

    mercury_metrics = _try_read_json(root / "output" / "mercury" / "mercury_precession_metrics.json") or {}
    mercury_m = (mercury_metrics.get("simulation_physical") or {}) if isinstance(mercury_metrics, dict) else {}
    mercury_p = (mercury_m.get("pmodel") or {}) if isinstance(mercury_m, dict) else {}
    mercury_ref = (
        float(mercury_metrics.get("reference_arcsec_century"))
        if isinstance(mercury_metrics, dict) and mercury_metrics.get("reference_arcsec_century") is not None
        else None
    )
    mercury_p_century = (
        float(mercury_p.get("arcsec_per_century")) if mercury_p.get("arcsec_per_century") is not None else None
    )
    mercury_summary_lines: List[str] = [
        "左は可視化のため近日点移動を誇張（cを小さくした表示）。",
        "右は実Cでの近日点移動（角秒）を周回ごとに累積。",
    ]
    if mercury_p_century is not None:
        mercury_summary_lines.insert(0, f"P-model 推定={mercury_p_century:.3f} 角秒/世紀（実C）")
    if mercury_ref is not None:
        mercury_summary_lines.insert(1, f"参照（観測残差の代表）≈{mercury_ref:.2f} 角秒/世紀")

    eht_cmp = _try_read_json(root / "output" / "eht" / "eht_shadow_compare.json") or {}
    eht_rows = eht_cmp.get("rows") if isinstance(eht_cmp, dict) else []
    eht_rows = eht_rows if isinstance(eht_rows, list) else []
    eht_beta = None
    try:
        eht_beta = float((eht_cmp.get("pmodel") or {}).get("beta")) if isinstance(eht_cmp, dict) else None
    except Exception:
        eht_beta = None

    eht_summary_lines: List[str] = []
    eht_morph_summary_lines: List[str] = []
    eht_diff_summary_lines: List[str] = []
    eht_sys_summary_lines: List[str] = []
    eht_zscore_summary_lines: List[str] = []
    eht_kappa_precision_summary_lines: List[str] = []
    eht_kappa_tradeoff_summary_lines: List[str] = []
    eht_delta_precision_summary_lines: List[str] = []
    eht_paper5_m3_rescue_summary_lines: List[str] = []
    eht_kerr_grid_summary_lines: List[str] = []
    eht_kerr_def_sens_summary_lines: List[str] = []
    eht_m87_multi_epoch_summary_lines: List[str] = []

    eht_m87_epoch = (
        _try_read_json(root / "output" / "eht" / "eht_m87_persistent_shadow_ring_diameter_metrics.json") or {}
    )
    if isinstance(eht_m87_epoch, dict) and eht_m87_epoch:
        rms = (
            eht_m87_epoch.get("ring_measurements")
            if isinstance(eht_m87_epoch.get("ring_measurements"), list)
            else []
        )
        parts: List[str] = []
        for rm in rms:
            if not isinstance(rm, dict):
                continue
            epoch = str(rm.get("epoch") or "")
            try:
                d = float(rm.get("diameter_uas"))
                sm = float(rm.get("sigma_minus_uas"))
                sp = float(rm.get("sigma_plus_uas"))
            except Exception:
                continue
            if epoch:
                if abs(sm - sp) < 1e-12:
                    parts.append(f"{epoch}: {d:.1f}±{sm:.1f} µas")
                else:
                    parts.append(f"{epoch}: {d:.1f}(-{sm:.1f}/+{sp:.1f}) µas")
        if parts:
            eht_m87_multi_epoch_summary_lines.append(" / ".join(parts))
        try:
            d = float(eht_m87_epoch.get("delta_2018_minus_2017_uas"))
            s = float(eht_m87_epoch.get("delta_sigma_uas_avg_sym"))
            z = float(eht_m87_epoch.get("delta_z_avg_sym"))
            eht_m87_multi_epoch_summary_lines.append(f"Δ(2018−2017)={d:+.1f} µas（avg_sym; z≈{z:.2f}, 1σ≈{s:.2f} µas）")
        except Exception:
            pass
    else:
        eht_m87_multi_epoch_summary_lines = [
            "出力未生成: scripts/eht/eht_m87_persistent_shadow_metrics.py を実行してください。"
        ]

    eht_kerr_grid = _try_read_json(root / "output" / "eht" / "eht_kerr_shadow_coeff_grid_metrics.json") or {}
    eht_kerr_def = (
        _try_read_json(root / "output" / "eht" / "eht_kerr_shadow_coeff_definition_sensitivity_metrics.json") or {}
    )
    if isinstance(eht_kerr_grid, dict) and eht_kerr_grid:
        try:
            c = (eht_kerr_grid.get("coefficients") or {}) if isinstance(eht_kerr_grid.get("coefficients"), dict) else {}
            d0 = float(c.get("delta_p_over_kerr_min_percent"))
            d1 = float(c.get("delta_p_over_kerr_max_percent"))
            eht_kerr_grid_summary_lines.append(f"P-model係数は Kerr係数レンジより +{d0:.2f}%〜+{d1:.2f}%（avg(width,height)）")
        except Exception:
            pass
        ovs = eht_kerr_grid.get("object_overlays") if isinstance(eht_kerr_grid.get("object_overlays"), list) else []
        for ov in ovs:
            if not isinstance(ov, dict):
                continue
            name = str(ov.get("name") or ov.get("key") or "")
            try:
                d0 = float(ov.get("delta_p_over_kerr_min_percent"))
                d1 = float(ov.get("delta_p_over_kerr_max_percent"))
                if name:
                    eht_kerr_grid_summary_lines.append(f"{name}: +{d0:.2f}%〜+{d1:.2f}%（制約下）")
            except Exception:
                continue
    else:
        eht_kerr_grid_summary_lines = ["出力未生成: scripts/eht/eht_kerr_shadow_coeff_grid.py を実行してください。"]

    if isinstance(eht_kerr_def, dict) and eht_kerr_def:
        try:
            smax = float(eht_kerr_def.get("definition_spread_rel_max")) * 100.0
            smed = float(eht_kerr_def.get("definition_spread_rel_median")) * 100.0
            eht_kerr_def_sens_summary_lines.append(f"定義依存 spread: max≈{smax:.2f}% / median≈{smed:.3f}%")
        except Exception:
            pass
        try:
            rg = (eht_kerr_def.get("ranges_global") or {}) if isinstance(eht_kerr_def.get("ranges_global"), dict) else {}
            env = (rg.get("envelope") or {}) if isinstance(rg.get("envelope"), dict) else {}
            cmin = float(env.get("coeff_min"))
            cmax = float(env.get("coeff_max"))
            eht_kerr_def_sens_summary_lines.append(f"係数レンジ（definition envelope）={cmin:.3f}–{cmax:.3f}")
        except Exception:
            pass
    else:
        eht_kerr_def_sens_summary_lines = ["出力未生成: scripts/eht/eht_kerr_shadow_coeff_definition_sensitivity.py を実行してください。"]
    if eht_rows:
        if eht_beta is not None:
            eht_summary_lines.append(f"P-model は β={eht_beta:g}（Cassini拘束により概ね1）で計算。")
        for r in eht_rows:
            if not isinstance(r, dict):
                continue
            name = str(r.get("name") or r.get("key") or "")
            obs = r.get("ring_diameter_obs_uas")
            obs_s = r.get("ring_diameter_obs_uas_sigma")
            p = r.get("shadow_diameter_pmodel_uas")
            p_s = r.get("shadow_diameter_pmodel_uas_sigma")
            kfit = r.get("kappa_ring_over_shadow_fit_pmodel")
            kfit_s = r.get("kappa_ring_over_shadow_fit_pmodel_sigma")
            try:
                eht_summary_lines.append(
                    f"{name}: 観測={float(obs):.1f}±{float(obs_s):.1f} µas, "
                    f"P-model={float(p):.1f}±{float(p_s):.1f} µas, "
                    f"κ_fit={float(kfit):.3f}±{float(kfit_s):.3f}"
                )
            except Exception:
                continue

        # Phase 4: differential prediction (P-model vs GR)
        try:
            p4 = (eht_cmp.get("phase4") or {}) if isinstance(eht_cmp, dict) else {}
            ratio = float(p4.get("shadow_diameter_coeff_ratio_p_over_gr"))
            eht_diff_summary_lines.append(f"差分予測（係数比 P/GR）={ratio:.4f}（差 {(ratio-1)*100:.2f}%）")
        except Exception:
            pass
        for r in eht_rows:
            if not isinstance(r, dict):
                continue
            name = str(r.get("name") or r.get("key") or "")
            try:
                d = float(r.get("shadow_diameter_diff_p_minus_gr_uas"))
                ds = float(r.get("shadow_diameter_diff_p_minus_gr_uas_sigma"))
                need = float(r.get("shadow_diameter_sigma_obs_needed_3sigma_uas"))
            except Exception:
                continue
            if not (name and (d == d) and (ds == ds)):
                continue
            if need == need and need > 0:
                eht_diff_summary_lines.append(f"{name}: 差={d:+.2f}±{ds:.2f} μas、3σ判別に必要な観測誤差(1σ)<{need:.2f} μas")
            else:
                extra = ""
                try:
                    rel = float(r.get("theta_unit_rel_sigma", float("nan")))
                    rel_req = float(r.get("theta_unit_rel_sigma_required_3sigma", float("nan")))
                    if math.isfinite(rel) and math.isfinite(rel_req) and rel_req > 0:
                        extra = f"（θ_unit相対誤差={rel*100:.1f}% > 要求={rel_req*100:.1f}%）"
                except Exception:
                    extra = ""
                eht_diff_summary_lines.append(
                    f"{name}: 差={d:+.2f}±{ds:.2f} μas、3σ判別に必要精度=n/a（質量/距離の不確かさが支配）{extra}"
                )

        # Systematics: ring diameter vs shadow diameter (kappa), and Kerr coefficient range (reference)
        try:
            kerr_def = (
                (eht_cmp.get("reference_gr_kerr_definition_sensitivity") or {}) if isinstance(eht_cmp, dict) else {}
            )
            if isinstance(kerr_def, dict) and kerr_def.get("coeff_min") is not None and kerr_def.get("coeff_max") is not None:
                kmin = float(kerr_def.get("coeff_min"))
                kmax = float(kerr_def.get("coeff_max"))
                spread = kerr_def.get("definition_spread_rel_max")
                if spread is not None:
                    eht_sys_summary_lines.append(
                        f"参考: GR（Kerr）係数レンジ={kmin:.3f}–{kmax:.3f}（definition envelope; max spread≈{float(spread)*100:.2f}%）"
                    )
                else:
                    eht_sys_summary_lines.append(f"参考: GR（Kerr）係数レンジ={kmin:.3f}–{kmax:.3f}（definition envelope）")
            else:
                kerr = (eht_cmp.get("reference_gr_kerr") or {}) if isinstance(eht_cmp, dict) else {}
                kmin = float(kerr.get("coeff_min"))
                kmax = float(kerr.get("coeff_max"))
                method = str(kerr.get("method") or "")
                eht_sys_summary_lines.append(f"参考: GR（Kerr）係数レンジ={kmin:.3f}–{kmax:.3f}（定義={method}）")
        except Exception:
            pass
        for r in eht_rows:
            if not isinstance(r, dict):
                continue
            name = str(r.get("name") or r.get("key") or "")
            try:
                kp = float(r.get("kappa_ring_over_shadow_fit_pmodel"))
                kps = float(r.get("kappa_ring_over_shadow_fit_pmodel_sigma"))
                kg = float(r.get("kappa_ring_over_shadow_fit_gr"))
                kgs = float(r.get("kappa_ring_over_shadow_fit_gr_sigma"))
            except Exception:
                continue
            if name and (kp == kp) and (kg == kg):
                eht_sys_summary_lines.append(f"{name}: κ(P-model)={kp:.3f}±{kps:.3f}, κ(GR)={kg:.3f}±{kgs:.3f}")

        # Kappa precision requirement (3σ discrimination) summary.
        eht_kappa_precision_summary_lines.append("κ（リング/シャドウ比）の必要精度（相対, 1σ；3σ判別の目安）。")
        for r in eht_rows:
            if not isinstance(r, dict):
                continue
            name = str(r.get("name") or r.get("key") or "")
            if not name:
                continue
            try:
                ring = float(r.get("ring_diameter_obs_uas", float("nan")))
                ring_s = float(r.get("ring_diameter_obs_uas_sigma", float("nan")))
                ring_rel = ring_s / ring if ring > 0 else float("nan")
            except Exception:
                ring_rel = float("nan")
            try:
                req0 = float(r.get("kappa_sigma_required_3sigma_if_ring_sigma_zero", float("nan")))
                req_cur = float(r.get("kappa_sigma_required_3sigma_if_ring_sigma_current", float("nan")))
                kerr_s = float(r.get("kappa_sigma_assumed_kerr", float("nan")))
                tu = float(r.get("theta_unit_rel_sigma", float("nan")))
                tu_req = float(r.get("theta_unit_rel_sigma_required_3sigma", float("nan")))
            except Exception:
                continue

            parts: List[str] = []
            if math.isfinite(req0) and req0 > 0:
                parts.append(f"κ要求<{req0*100:.2f}%（ringσ→0）")
            else:
                if math.isfinite(tu) and math.isfinite(tu_req) and tu_req > 0 and tu > tu_req:
                    parts.append(f"κ要求=n/a（θ_unit相対誤差={tu*100:.1f}% > 要求={tu_req*100:.1f}%）")
                else:
                    parts.append("κ要求=n/a")

            if math.isfinite(req_cur) and req_cur > 0:
                parts.append(f"κ要求<{req_cur*100:.2f}%（ringσ=現状）")
            else:
                if math.isfinite(ring_rel):
                    parts.append(f"ringσ={ring_rel*100:.1f}%")

            if math.isfinite(kerr_s) and kerr_s > 0:
                parts.append(f"参考: Kerr系統σ≈{kerr_s*100:.2f}%")

            eht_kappa_precision_summary_lines.append(f"{name}: " + " / ".join(parts))

        # Kappa tradeoff summary (ring σ vs κσ).
        eht_kappa_tradeoff_summary_lines.append("3σ判別の誤差予算：ring σ（統計）と κσ（系統）のトレードオフ。")
        for r in eht_rows:
            if not isinstance(r, dict):
                continue
            name = str(r.get("name") or r.get("key") or "")
            if not name:
                continue
            try:
                tu = float(r.get("theta_unit_rel_sigma", float("nan")))
                tu_req = float(r.get("theta_unit_rel_sigma_required_3sigma", float("nan")))
                tu_factor = float(r.get("theta_unit_rel_sigma_improvement_factor_to_3sigma", float("nan")))
                ring = float(r.get("ring_diameter_obs_uas", float("nan")))
                ring_s = float(r.get("ring_diameter_obs_uas_sigma", float("nan")))
                ring_rel = ring_s / ring if ring > 0 else float("nan")
                ring_factor = float(r.get("ring_diameter_sigma_improvement_factor_to_3sigma_if_kappa1", float("nan")))
                kerr_s = float(r.get("kappa_sigma_assumed_kerr", float("nan")))
                kreq0 = float(r.get("kappa_sigma_required_3sigma_if_ring_sigma_zero", float("nan")))
            except Exception:
                continue

            if math.isfinite(tu) and math.isfinite(tu_req) and tu_req > 0 and tu > tu_req:
                extra = ""
                if math.isfinite(tu_factor) and tu_factor > 1:
                    extra = f"（~{tu_factor:.1f}×改善）"
                eht_kappa_tradeoff_summary_lines.append(
                    f"{name}: θ_unit={tu*100:.1f}% > 要求={tu_req*100:.1f}%{extra}（まず質量/距離が律速）"
                )
                continue

            parts: List[str] = []
            if math.isfinite(ring_rel):
                parts.append(f"ringσ={ring_rel*100:.1f}%")
                if math.isfinite(ring_factor) and ring_factor > 1:
                    parts.append(f"ring精度は~{ring_factor:.1f}×改善が目安（κ=1仮定）")
            if math.isfinite(kerr_s) and kerr_s > 0:
                parts.append(f"参考: κ系統（Kerr）σ≈{kerr_s*100:.2f}%")
            if math.isfinite(kreq0) and kreq0 > 0:
                parts.append(f"κ要求<{kreq0*100:.2f}%（ringσ→0）")
            eht_kappa_tradeoff_summary_lines.append(f"{name}: " + " / ".join(parts))

        # Delta precision summary (reference).
        eht_delta_precision_summary_lines.append("δ（Schwarzschild shadow deviation）の必要精度（参考; δはモデル依存）。")
        for r in eht_rows:
            if not isinstance(r, dict):
                continue
            name = str(r.get("name") or r.get("key") or "")
            if not name:
                continue
            try:
                req = float(r.get("delta_sigma_required_3sigma", float("nan")))
                vlti = float(r.get("delta_schwarzschild_vlti_sigma_sym", float("nan")))
                vlti_factor = float(r.get("delta_schwarzschild_vlti_improvement_factor_to_3sigma", float("nan")))
                keck = float(r.get("delta_schwarzschild_keck_sigma_sym", float("nan")))
                keck_factor = float(r.get("delta_schwarzschild_keck_improvement_factor_to_3sigma", float("nan")))
            except Exception:
                continue

            parts: List[str] = []
            if math.isfinite(req) and req > 0:
                parts.append(f"要求≈{req*100:.2f}%（3σ）")
            if math.isfinite(vlti) and vlti > 0:
                if math.isfinite(vlti_factor) and vlti_factor > 1:
                    parts.append(f"VLTI: δσ≈{vlti*100:.1f}%（~{vlti_factor:.1f}×改善）")
                else:
                    parts.append(f"VLTI: δσ≈{vlti*100:.1f}%")
            if math.isfinite(keck) and keck > 0:
                if math.isfinite(keck_factor) and keck_factor > 1:
                    parts.append(f"Keck: δσ≈{keck*100:.1f}%（~{keck_factor:.1f}×改善）")
                else:
                    parts.append(f"Keck: δσ≈{keck*100:.1f}%")
            if parts:
                eht_delta_precision_summary_lines.append(f"{name}: " + " / ".join(parts))

        # Z-score summary (obs - pred, normalized by combined uncertainty), under κ=1 assumption.
        eht_zscore_summary_lines.append("zスコア=(観測-予測)/σ_total（σ_total=√(σ_obs^2+σ_pred^2), κ=1 の仮定）")
        for r in eht_rows:
            if not isinstance(r, dict):
                continue
            name = str(r.get("name") or r.get("key") or "")
            try:
                zp = float(r.get("zscore_pmodel"))
                zg = float(r.get("zscore_gr"))
            except Exception:
                continue
            if name and (zp == zp) and (zg == zg):
                eht_zscore_summary_lines.append(f"{name}: z(P-model)={zp:+.2f}, z(GR)={zg:+.2f}")

        # Morphology: additional observables that inform systematics (width/asymmetry).
        for r in eht_rows:
            if not isinstance(r, dict):
                continue
            name = str(r.get("name") or r.get("key") or "")
            try:
                w_min = float(r.get("ring_fractional_width_min", float("nan")))
                w_max = float(r.get("ring_fractional_width_max", float("nan")))
                a_min = float(r.get("ring_brightness_asymmetry_min", float("nan")))
                a_max = float(r.get("ring_brightness_asymmetry_max", float("nan")))
                scatt = float(r.get("scattering_kernel_fwhm_uas", float("nan")))
                scatt_a = float(r.get("scattering_kernel_fwhm_major_uas", float("nan")))
                scatt_b = float(r.get("scattering_kernel_fwhm_minor_uas", float("nan")))
            except Exception:
                continue
            if not name:
                continue

            parts: List[str] = []
            if (w_min == w_min) and (w_max == w_max):
                parts.append(f"幅(W/d)={w_min:.2f}–{w_max:.2f}")
            elif w_max == w_max:
                parts.append(f"幅(W/d)≤{w_max:.2f}")
            elif w_min == w_min:
                parts.append(f"幅(W/d)≥{w_min:.2f}")

            if (a_min == a_min) and (a_max == a_max):
                parts.append(f"非対称A={a_min:.2f}–{a_max:.2f}")
            elif a_max == a_max:
                parts.append(f"非対称A≤{a_max:.2f}")
            elif a_min == a_min:
                parts.append(f"非対称A≥{a_min:.2f}")

            if (scatt_a == scatt_a) and (scatt_b == scatt_b) and (scatt_a > 0) and (scatt_b > 0):
                parts.append(f"散乱blur FWHM≈{scatt_a:.0f}×{scatt_b:.0f} μas（参考）")
            elif scatt == scatt and scatt > 0:
                parts.append(f"散乱blur FWHM≈{scatt:.0f} μas（参考）")

            if parts:
                eht_morph_summary_lines.append(f"{name}: " + ", ".join(parts) + "（一次ソース/参考レンジ）")
    else:
        eht_summary_lines = ["出力未生成: scripts/eht/eht_shadow_compare.py を実行してください。"]
        eht_morph_summary_lines = ["出力未生成: scripts/eht/eht_shadow_compare.py を実行してください。"]
        eht_diff_summary_lines = ["出力未生成: scripts/eht/eht_shadow_compare.py を実行してください。"]
        eht_sys_summary_lines = ["出力未生成: scripts/eht/eht_shadow_compare.py を実行してください。"]
        eht_zscore_summary_lines = ["出力未生成: scripts/eht/eht_shadow_compare.py を実行してください。"]
        eht_kappa_precision_summary_lines = ["出力未生成: scripts/eht/eht_shadow_compare.py を実行してください。"]

    # EHT (Sgr A* Paper V): near-passing rescue conditions (M3 / 2.2 μm / KS sensitivity)
    paper5_rescue = (
        _try_read_json(root / "output" / "eht" / "eht_sgra_paper5_m3_nir_reconnection_conditions_metrics.json") or {}
    )
    try:
        derived = paper5_rescue.get("derived") if isinstance(paper5_rescue, dict) else None
        m3 = (derived.get("m3") or {}) if isinstance(derived, dict) else {}
        ks = (m3.get("historical_distribution_values_ks") or {}) if isinstance(m3, dict) else {}
        salvage = (derived.get("near_passing_salvage") or {}) if isinstance(derived, dict) else {}

        if isinstance(ks, dict) and ks.get("ok") is True:
            alpha = float(ks.get("alpha")) if ks.get("alpha") is not None else None
            d = float(ks.get("d")) if ks.get("d") is not None else None
            p = float(ks.get("p_asymptotic")) if ks.get("p_asymptotic") is not None else None
            n2017 = int(ks.get("n_2017")) if ks.get("n_2017") is not None else None
            nh = int(ks.get("n_historical")) if ks.get("n_historical") is not None else None

            if (d is not None) and (p is not None) and (alpha is not None) and (n2017 is not None) and (nh is not None):
                eht_paper5_m3_rescue_summary_lines.append(
                    f"KS（2017 n={n2017} vs historical n={nh}）: D={d:.6f}, p={p:.6f}（α={alpha:g}）"
                )

                margin = ks.get("margin") or {}
                if isinstance(margin, dict) and margin.get("p_minus_alpha") is not None:
                    try:
                        pma = float(margin.get("p_minus_alpha"))
                        eht_paper5_m3_rescue_summary_lines.append(f"閾値余裕: p−α={pma:+.6f}（near-passing）")
                    except Exception:
                        pass

            dd = ks.get("d_discreteness") or {}
            if isinstance(dd, dict) and dd.get("d_step") is not None:
                try:
                    d_step = float(dd.get("d_step"))
                    d_up = float(dd.get("d_next_up")) if dd.get("d_next_up") is not None else None
                    p_up = float(dd.get("p_next_up_asymptotic")) if dd.get("p_next_up_asymptotic") is not None else None
                    if (d_up is not None) and (p_up is not None):
                        eht_paper5_m3_rescue_summary_lines.append(
                            f"Dは離散（step={d_step:.6f}）。1 step上（D={d_up:.6f}）で p≈{p_up:.6f}"
                        )
                    else:
                        eht_paper5_m3_rescue_summary_lines.append(f"Dは離散（step={d_step:.6f}）")
                except Exception:
                    pass

            ds = ks.get("digitize_scale_thresholds") or {}
            if isinstance(ds, dict) and ds.get("scale_up_to_decrease_hist_le_count_by_1") is not None:
                try:
                    scale_up = float(ds.get("scale_up_to_decrease_hist_le_count_by_1"))
                    rel = float(ds.get("delta_rel_to_move_hist_le_max_to_t")) if ds.get("delta_rel_to_move_hist_le_max_to_t") is not None else None
                    if rel is not None:
                        eht_paper5_m3_rescue_summary_lines.append(
                            f"digitize/系統の目安: mi3 を一様に +{rel*100:.2f}%（×{scale_up:.6f}）で D が 1 step 変化"
                        )
                    else:
                        eht_paper5_m3_rescue_summary_lines.append(
                            f"digitize/系統の目安: mi3 を一様に ×{scale_up:.6f} で D が 1 step 変化"
                        )
                except Exception:
                    pass

        if isinstance(salvage, dict) and isinstance(salvage.get("constraints_ranked_by_count"), list):
            top = salvage.get("constraints_ranked_by_count")[:2]
            parts: List[str] = []
            for row in top:
                if not isinstance(row, dict):
                    continue
                c = str(row.get("constraint") or "")
                f = row.get("fraction")
                try:
                    parts.append(f"{c}={float(f)*100:.0f}%")
                except Exception:
                    continue
            total_n = salvage.get("rows_total_n")
            if parts and total_n is not None:
                try:
                    eht_paper5_m3_rescue_summary_lines.append(
                        f"near-passing（Paper V）残り制約（n={int(total_n)}）: " + ", ".join(parts)
                    )
                except Exception:
                    pass
    except Exception:
        eht_paper5_m3_rescue_summary_lines = []

    if not eht_paper5_m3_rescue_summary_lines:
        eht_paper5_m3_rescue_summary_lines = [
            "出力未生成: scripts/eht/eht_sgra_paper5_m3_nir_reconnection_conditions.py を実行してください。"
        ]

    # Strong-field: binary pulsars (orbital decay) and GW150914 chirp phase (GWOSC)
    pulsar_metrics = _try_read_json(root / "output" / "pulsar" / "binary_pulsar_orbital_decay_metrics.json") or {}
    pulsar_summary_lines: List[str] = []
    if isinstance(pulsar_metrics, dict) and isinstance(pulsar_metrics.get("metrics"), list):
        for m in pulsar_metrics.get("metrics") or []:
            if not isinstance(m, dict):
                continue
            name = str(m.get("name") or m.get("id") or "")
            R = m.get("R")
            sigma_1 = m.get("sigma_1")
            conf = str(m.get("sigma_note") or "")
            non_gr_3s = m.get("non_gr_fraction_upper_3sigma")
            try:
                msg = f"{name}: R={float(R):.6f} ±{_format_sci(float(sigma_1), digits=2)}（{conf}）"
                if non_gr_3s is not None:
                    msg += f" / 非GR成分上限（概算, 3σ）<{float(non_gr_3s)*100:.2f}%"
                pulsar_summary_lines.append(msg)
            except Exception:
                continue
        if not pulsar_summary_lines:
            pulsar_summary_lines = ["出力の読み取りに失敗: output/pulsar/binary_pulsar_orbital_decay_metrics.json"]
    else:
        pulsar_summary_lines = ["出力未生成: scripts/pulsar/binary_pulsar_orbital_decay.py を実行してください。"]

    # GW: summarize multiple events succinctly (public-friendly)
    gw_summary_lines: List[str] = []
    gw_events = [
        ("GW150914", "gw150914"),
        ("GW151226", "gw151226"),
        ("GW170104", "gw170104"),
        ("GW170817", "gw170817"),
        ("GW190425", "gw190425"),
    ]
    any_gw = False
    gw_summary_lines.append("公開データ: GWOSC（複数イベント, chirp整合の指標 R^2）")
    for ev_name, slug in gw_events:
        path = root / "output" / "gw" / f"{slug}_chirp_phase_metrics.json"
        if not path.exists():
            continue
        gw_metrics = _try_read_json(path) or {}
        if not (isinstance(gw_metrics, dict) and isinstance(gw_metrics.get("detectors"), list)):
            continue

        any_gw = True
        dets = gw_metrics.get("detectors") or []
        best: Optional[Tuple[float, str, str]] = None  # (r2, detector, preprocess)
        for d in dets:
            if not isinstance(d, dict):
                continue
            det = str(d.get("detector") or "")
            preprocess = str(d.get("preprocess") or "")
            fit = d.get("fit") if isinstance(d.get("fit"), dict) else {}
            r2 = fit.get("r2")
            try:
                r2f = float(r2)
            except Exception:
                continue
            if best is None or r2f > best[0]:
                best = (r2f, det, preprocess)

        if best is None:
            continue

        r2f, det, preprocess = best
        suffix = f" ({det})"
        if preprocess:
            suffix = f" ({det}, {preprocess})"
        gw_summary_lines.append(f"{ev_name}: best R^2={_format_num(r2f, digits=4)}{suffix}")

    if not any_gw:
        gw_summary_lines = ["出力未生成: scripts/gw/gw150914_chirp_phase.py を実行してください。"]

    # Phase 4 (differential): saturation parameter δ consistency
    delta_sat = _try_read_json(root / "output" / "theory" / "delta_saturation_constraints.json") or {}
    delta_sat_summary_lines: List[str] = []
    delta_sat_table: Optional[Dict[str, Any]] = None
    if isinstance(delta_sat, dict) and delta_sat.get("rows"):
        try:
            delta_adopted = float(delta_sat.get("delta_adopted"))
            gamma_max = float(delta_sat.get("gamma_max_for_delta_adopted"))
            delta_sat_summary_lines.append(
                f"採用δ={_format_sci(delta_adopted, digits=1)} → γ_max≈{_format_sci(gamma_max, digits=1)}"
            )

            strictest: Optional[Tuple[str, float, float]] = None  # (label, gamma_obs, delta_upper)
            table_rows: List[List[str]] = []
            for r in delta_sat.get("rows") or []:
                if not isinstance(r, dict):
                    continue
                label = str(r.get("label") or r.get("key") or "")
                gamma_obs = float(r.get("gamma_obs"))
                delta_upper = float(r.get("delta_upper_from_gamma"))
                table_rows.append([label, _format_sci(gamma_obs, digits=1), _format_sci(delta_upper, digits=1)])
                if strictest is None or (delta_upper == delta_upper and delta_upper < strictest[2]):
                    strictest = (label, gamma_obs, delta_upper)

            if strictest is not None:
                label, gamma_obs, delta_upper = strictest
                ok = delta_adopted < delta_upper
                delta_sat_summary_lines.append(
                    f"最も厳しい上限: {label}（γ≈{_format_sci(gamma_obs, digits=1)}）→ δ<{_format_sci(delta_upper, digits=1)}"
                    + ("（整合）" if ok else "（要見直し）")
                )

            delta_sat_table = {
                "headers": ["例", "γ（概算）", "δ上限（概算）"],
                "rows": table_rows,
                "caption": "δは既知の高γ観測と矛盾しないよう十分小さくする必要がある（γ_max≈1/√δ）。",
            }
        except Exception:
            delta_sat_summary_lines = ["出力の読み取りに失敗: output/theory/delta_saturation_constraints.json"]
            delta_sat_table = None
    else:
        delta_sat_summary_lines = ["出力未生成: scripts/theory/delta_saturation_constraints.py を実行してください。"]

    # Theory: gravitational redshift (observed deviation epsilon vs prediction epsilon=0)
    redshift = _try_read_json(root / "output" / "theory" / "gravitational_redshift_experiments.json") or {}
    redshift_summary_lines: List[str] = []
    redshift_table: Optional[Dict[str, Any]] = None
    redshift_definition_lines: List[str] = []
    if isinstance(redshift, dict) and redshift.get("rows"):
        try:
            definition = redshift.get("definition") or {}
            if isinstance(definition, dict):
                eps_def = str(definition.get("epsilon") or "").strip()
                pred_def = str(definition.get("pmodel_prediction") or "").strip()
                if eps_def:
                    redshift_definition_lines.append(f"εの定義: {eps_def}")
                if pred_def:
                    redshift_definition_lines.append(f"P-model予測: {pred_def}")

            rows = redshift.get("rows") or []
            table_rows: List[List[str]] = []
            max_abs_z: Optional[float] = None
            for r in rows:
                if not isinstance(r, dict):
                    continue
                label = str(r.get("short_label") or r.get("id") or "")
                eps = float(r.get("epsilon") or 0.0)
                sig = float(r.get("sigma") or 0.0)
                z = r.get("z_score")
                zf = None if z is None else float(z)
                if zf is not None:
                    az = abs(zf)
                    if max_abs_z is None or az > max_abs_z:
                        max_abs_z = az

                src = r.get("source") or {}
                if not isinstance(src, dict):
                    src = {}
                doi = str(src.get("doi") or "")
                year = str(src.get("year") or "")
                src_txt = doi or (str(src.get("url") or "") if src.get("url") else "")
                if year and doi:
                    src_txt = f"{year}, {doi}"
                elif year and not src_txt:
                    src_txt = year

                eps_txt = _format_sci(eps, digits=3)
                sig_txt = _format_sci(sig, digits=3)
                z_txt = "" if zf is None else f"{zf:+.2f}"

                redshift_summary_lines.append(f"{label}: ε={eps_txt} ± {sig_txt}（z={z_txt}）")
                sigma_note = str(r.get("sigma_note") or "").strip()
                if sigma_note:
                    redshift_summary_lines.append(f"  注: {label} のσは {sigma_note}")

                table_rows.append([label, eps_txt, sig_txt, z_txt, src_txt])

            if max_abs_z is not None:
                redshift_summary_lines.insert(0, f"最大|z|={max_abs_z:.2f}（小さいほど整合）")

            redshift_table = {
                "headers": ["実験", "ε", "σ", "z", "一次ソース（年, DOI）"],
                "rows": table_rows,
                "caption": "εは重力赤方偏移の偏差パラメータ（GRは ε=0）。P-model（弱場・静止時計）は ε=0 を予測する。",
            }
        except Exception:
            redshift_summary_lines = ["出力の読み取りに失敗: output/theory/gravitational_redshift_experiments.json"]
            redshift_table = None
    else:
        redshift_summary_lines = ["出力未生成: scripts/theory/gravitational_redshift_experiments.py を実行してください。"]

    # Theory: frame dragging (observed ratio μ vs prediction μ=1)
    frame_drag = _try_read_json(root / "output" / "theory" / "frame_dragging_experiments.json") or {}
    frame_drag_summary_lines: List[str] = []
    frame_drag_table: Optional[Dict[str, Any]] = None
    frame_drag_definition_lines: List[str] = []
    if isinstance(frame_drag, dict) and frame_drag.get("rows"):
        try:
            definition = frame_drag.get("definition") or {}
            if isinstance(definition, dict):
                obs_def = str(definition.get("observable") or "").strip()
                mu_def = str(definition.get("mu") or "").strip()
                pred_def = str(definition.get("pmodel_prediction") or "").strip()
                if obs_def:
                    frame_drag_definition_lines.append(f"観測量: {obs_def}")
                if mu_def:
                    frame_drag_definition_lines.append(f"μの定義: {mu_def}")
                if pred_def:
                    frame_drag_definition_lines.append(f"P-model予測: {pred_def}")

            rows = frame_drag.get("rows") or []
            table_rows: List[List[str]] = []
            max_abs_z: Optional[float] = None
            for r in rows:
                if not isinstance(r, dict):
                    continue
                label = str(r.get("short_label") or r.get("id") or "")
                mu = float(r.get("mu") or 0.0)
                sig = float(r.get("mu_sigma") or 0.0)
                z = r.get("z_score")
                zf = None if z is None else float(z)
                if zf is not None:
                    az = abs(zf)
                    if max_abs_z is None or az > max_abs_z:
                        max_abs_z = az

                src = r.get("source") or {}
                if not isinstance(src, dict):
                    src = {}
                doi = str(src.get("doi") or "")
                year = str(src.get("year") or "")
                src_txt = doi or (str(src.get("url") or "") if src.get("url") else "")
                if year and doi:
                    src_txt = f"{year}, {doi}"
                elif year and not src_txt:
                    src_txt = year

                mu_txt = f"{mu:.3f}"
                sig_txt = f"{sig:.3f}"
                z_txt = "" if zf is None else f"{zf:+.2f}"

                extra = ""
                omega_obs = r.get("omega_obs_mas_per_yr")
                omega_sig = r.get("omega_obs_sigma_mas_per_yr")
                omega_pred = r.get("omega_pred_mas_per_yr")
                try:
                    if omega_obs is not None and omega_pred is not None:
                        if omega_sig is not None:
                            extra = f" / |Ω|={abs(float(omega_obs)):.1f}±{abs(float(omega_sig)):.1f} mas/yr（予測 {abs(float(omega_pred)):.1f}）"
                        else:
                            extra = f" / |Ω|={abs(float(omega_obs)):.1f} mas/yr（予測 {abs(float(omega_pred)):.1f}）"
                except Exception:
                    extra = ""

                frame_drag_summary_lines.append(f"{label}: μ={mu_txt} ± {sig_txt}（z={z_txt}）{extra}")
                sigma_note = str(r.get("sigma_note") or "").strip()
                if sigma_note:
                    frame_drag_summary_lines.append(f"  注: {label} のσは {sigma_note}")

                table_rows.append([label, mu_txt, sig_txt, z_txt, src_txt])

            if max_abs_z is not None:
                frame_drag_summary_lines.insert(0, f"最大|z|={max_abs_z:.2f}（小さいほど整合）")

            frame_drag_table = {
                "headers": ["実験", "μ", "σ", "z", "一次ソース（年, DOI）"],
                "rows": table_rows,
                "caption": "μはフレームドラッグ歳差の比（μ=|Ω_obs|/|Ω_pred|）。GRとP-model（弱場・回転項の最小拡張）は μ=1 を予測する。",
            }
        except Exception:
            frame_drag_summary_lines = ["出力の読み取りに失敗: output/theory/frame_dragging_experiments.json"]
            frame_drag_table = None
    else:
        frame_drag_summary_lines = ["出力未生成: scripts/theory/frame_dragging_experiments.py を実行してください。"]

    # Cosmology: distance duality (DDR) observational constraint (epsilon0)
    cosmo_ddr = (
        _try_read_json(
            root / "output" / "private" / "cosmology" / "cosmology_distance_duality_constraints_metrics.json"
        )
        or {}
    )
    cosmo_ddr_summary_lines: List[str] = []
    cosmo_ddr_table: Optional[Dict[str, Any]] = None
    cosmo_ddr_definition_lines: List[str] = []
    if isinstance(cosmo_ddr, dict) and cosmo_ddr.get("rows"):
        try:
            definition = cosmo_ddr.get("definition") or {}
            if isinstance(definition, dict):
                eps_def = str(definition.get("epsilon0") or "").strip()
                eta_def = str(definition.get("eta") or "").strip()
                pred_def = str(definition.get("pbg_static_prediction") or "").strip()
                if eps_def:
                    cosmo_ddr_definition_lines.append(f"ε0の定義: {eps_def}")
                if eta_def:
                    cosmo_ddr_definition_lines.append(f"ηの定義: {eta_def}")
                if pred_def:
                    cosmo_ddr_definition_lines.append(f"背景P（静的）予測: {pred_def}")

            rows = cosmo_ddr.get("rows") or []
            table_rows: List[List[str]] = []
            max_abs_z_pbg: Optional[float] = None
            for r in rows:
                if not isinstance(r, dict):
                    continue
                label = str(r.get("short_label") or r.get("id") or "")
                eps = float(r.get("epsilon0_obs") or 0.0)
                sig = float(r.get("epsilon0_sigma") or 0.0)
                z_pbg = r.get("z_pbg_static")
                z_frw = r.get("z_frw")
                z_pbg_f = None if z_pbg is None else float(z_pbg)
                z_frw_f = None if z_frw is None else float(z_frw)
                if z_pbg_f is not None:
                    az = abs(z_pbg_f)
                    if max_abs_z_pbg is None or az > max_abs_z_pbg:
                        max_abs_z_pbg = az

                src = r.get("source") or {}
                if not isinstance(src, dict):
                    src = {}
                doi = str(src.get("doi") or "")
                year = str(src.get("year") or "")
                src_txt = doi or (str(src.get("url") or "") if src.get("url") else "")
                if year and doi:
                    src_txt = f"{year}, {doi}"
                elif year and not src_txt:
                    src_txt = year

                need = r.get("sigma_needed_to_not_reject_pbg_static_3sigma")
                need_f = None if need is None else float(need)
                need_txt = "" if need_f is None else f"{need_f:.3f}"

                z_pbg_txt = "" if z_pbg_f is None else f"{z_pbg_f:+.1f}"
                z_frw_txt = "" if z_frw_f is None else f"{z_frw_f:+.2f}"

                cosmo_ddr_summary_lines.append(
                    f"{label}: ε0={eps:+.3f} ± {sig:.3f}（z: 標準={z_frw_txt}, 背景P静的={z_pbg_txt}）"
                )

                # Reconnection hint for the "static infinite space" interpretation:
                # show the required effective correction at z=1 if present in metrics JSON.
                try:
                    delta_eps = r.get("epsilon0_extra_needed_to_match_obs")
                    dl_fac = r.get("extra_dl_factor_needed_z1")
                    dmu = r.get("delta_distance_modulus_mag_z1")
                    flux_dim = r.get("flux_dimming_factor_needed_z1")
                    if delta_eps is not None and dl_fac is not None:
                        extra = f"  必要補正: Δε≈{float(delta_eps):.3f}（z=1でD_L×{float(dl_fac):.2f}）"
                        if dmu is not None and flux_dim is not None:
                            extra += f"（Δμ≈{float(dmu):.2f} mag, flux≈1/{float(flux_dim):.2f}）"
                        cosmo_ddr_summary_lines.append(extra)
                except Exception:
                    pass
                if need_txt:
                    cosmo_ddr_summary_lines.append(
                        f"  参考: 背景P（静的, ε0=-1）を3σで“棄却しない”には σ≳{need_txt} が必要（現状より大きい）"
                    )

                table_rows.append([label, f"{eps:+.3f}", f"{sig:.3f}", z_frw_txt, z_pbg_txt, need_txt, src_txt])

            if max_abs_z_pbg is not None:
                cosmo_ddr_summary_lines.insert(
                    0, f"背景P（静的, ε0=-1）の棄却度: 最大|z|={max_abs_z_pbg:.1f}（大きいほど不整合）"
                )

            cosmo_ddr_table = {
                "headers": ["制約", "ε0", "σ", "z(標準)", "z(背景P静的)", "σ_min(棄却回避,3σ)", "一次ソース（年, DOI）"],
                "rows": table_rows,
                "caption": "距離二重性: d_L=(1+z)^(2+ε0) d_A（標準はε0=0）。背景P（静的）の最小モデルは ε0=-1 を予測する。",
            }
        except Exception:
            cosmo_ddr_summary_lines = [
                "出力の読み取りに失敗: output/cosmology/cosmology_distance_duality_constraints_metrics.json"
            ]
            cosmo_ddr_table = None
    else:
        cosmo_ddr_summary_lines = [
            "出力未生成: scripts/cosmology/cosmology_distance_duality_constraints.py を実行してください。"
        ]

    # Cosmology: Tolman surface brightness (SB dimming) observational constraint (n exponent)
    cosmo_tolman = (
        _try_read_json(
            root / "output" / "private" / "cosmology" / "cosmology_tolman_surface_brightness_constraints_metrics.json"
        )
        or {}
    )
    cosmo_tolman_summary_lines: List[str] = []
    cosmo_tolman_table: Optional[Dict[str, Any]] = None
    cosmo_tolman_definition_lines: List[str] = []
    if isinstance(cosmo_tolman, dict) and cosmo_tolman.get("rows"):
        try:
            definition = cosmo_tolman.get("definition") or {}
            if isinstance(definition, dict):
                obs_def = str(definition.get("observable") or "").strip()
                preds = definition.get("predictions") or {}
                note = str(definition.get("note") or "").strip()
                if obs_def:
                    cosmo_tolman_definition_lines.append(f"nの定義: {obs_def}")
                if isinstance(preds, dict):
                    frw_def = str(preds.get("FRW") or "").strip()
                    pbg_def = str(preds.get("P_bg_static") or "").strip()
                    if frw_def:
                        cosmo_tolman_definition_lines.append(f"標準（FRW）: {frw_def}")
                    if pbg_def:
                        cosmo_tolman_definition_lines.append(f"背景P（静的）: {pbg_def}")
                if note:
                    cosmo_tolman_definition_lines.append(f"注: {note}")

            rows = cosmo_tolman.get("rows") or []
            table_rows: List[List[str]] = []
            max_abs_z_pbg: Optional[float] = None
            for r in rows:
                if not isinstance(r, dict):
                    continue
                label = str(r.get("short_label") or r.get("id") or "")
                n_obs = float(r.get("n_obs") or 0.0)
                sig = float(r.get("n_sigma") or 0.0)
                z_pbg = r.get("z_pbg_static")
                z_frw = r.get("z_frw")
                evol_pbg = r.get("evolution_exponent_needed_pbg_static")
                z_pbg_f = None if z_pbg is None else float(z_pbg)
                z_frw_f = None if z_frw is None else float(z_frw)
                evol_pbg_f = None if evol_pbg is None else float(evol_pbg)
                if z_pbg_f is not None:
                    az = abs(z_pbg_f)
                    if max_abs_z_pbg is None or az > max_abs_z_pbg:
                        max_abs_z_pbg = az

                src = r.get("source") or {}
                if not isinstance(src, dict):
                    src = {}
                doi = str(src.get("doi") or "")
                year = str(src.get("year") or "")
                src_txt = doi or (str(src.get("url") or "") if src.get("url") else "")
                if year and doi:
                    src_txt = f"{year}, {doi}"
                elif year and not src_txt:
                    src_txt = year

                need = r.get("sigma_needed_to_not_reject_pbg_static_3sigma")
                need_f = None if need is None else float(need)
                need_txt = "" if need_f is None else f"{need_f:.3f}"

                z_pbg_txt = "" if z_pbg_f is None else f"{z_pbg_f:+.1f}"
                z_frw_txt = "" if z_frw_f is None else f"{z_frw_f:+.1f}"
                evol_pbg_txt = "" if evol_pbg_f is None else f"{evol_pbg_f:+.2f}"

                # Keep the summary short: emphasize background-P mismatch and the sign of required evolution.
                evol_note = ""
                if evol_pbg_f is not None and evol_pbg_f < 0:
                    evol_note = "（整合には進化補正が逆符号）"
                cosmo_tolman_summary_lines.append(
                    f"{label}: n={n_obs:.2f} ± {sig:.2f}（z: 標準={z_frw_txt}, 背景P静的={z_pbg_txt}）{evol_note}"
                )
                if need_txt:
                    cosmo_tolman_summary_lines.append(
                        f"  参考: 背景P（静的, n=2）を3σで“棄却しない”には σ?{need_txt} が必要（進化の系統は別）"
                    )

                table_rows.append([label, f"{n_obs:.2f}", f"{sig:.2f}", z_frw_txt, z_pbg_txt, evol_pbg_txt, need_txt, src_txt])

            if max_abs_z_pbg is not None:
                cosmo_tolman_summary_lines.insert(
                    0,
                    f"背景P（静的, n=2）の差: 最大|z|={max_abs_z_pbg:.1f}（進化が系統）",
                )

            cosmo_tolman_table = {
                "headers": ["制約", "n", "σ", "z(標準)", "z(背景P静的)", "evol_needed(Pbg)", "σ_min(棄却回避,3σ)", "一次ソース（年, DOI）"],
                "rows": table_rows,
                "caption": "Tolman表面輝度: SB(z)/SB(0)∝(1+z)^(-n)。標準（FRW）はn=4、背景P（静的）の最小モデルはn=2。観測のnには光度進化が混入する。",
            }
        except Exception:
            cosmo_tolman_summary_lines = [
                "出力の読み取りに失敗: output/cosmology/cosmology_tolman_surface_brightness_constraints_metrics.json"
            ]
            cosmo_tolman_table = None
    else:
        cosmo_tolman_summary_lines = [
            "出力未生成: scripts/cosmology/cosmology_tolman_surface_brightness_constraints.py を実行してください。"
        ]

    # Cosmology: tension attribution (Step 16.5 entrance)
    cosmo_tension_attr = (
        _try_read_json(root / "output" / "private" / "cosmology" / "cosmology_tension_attribution_metrics.json")
        or {}
    )
    cosmo_tension_attr_summary_lines: List[str] = []
    cosmo_tension_attr_detail_lines: List[str] = []
    if isinstance(cosmo_tension_attr, dict) and isinstance(cosmo_tension_attr.get("results"), dict):
        try:
            res = cosmo_tension_attr.get("results") or {}
            ddr_rep = res.get("ddr_representative") or {}
            bao_relax = res.get("bao_sigma_relax_needed") or {}
            indep = res.get("independent_probes") or {}
            next_actions = res.get("next_actions") or []

            ddr_z_abs = None
            try:
                z = ddr_rep.get("z_pbg_static")
                if z is not None:
                    ddr_z_abs = abs(float(z))
            except Exception:
                ddr_z_abs = None

            ddr_sig_mult = None
            try:
                v = ddr_rep.get("sigma_multiplier_to_not_reject_pbg_static_3sigma")
                if v is not None:
                    ddr_sig_mult = float(v)
            except Exception:
                ddr_sig_mult = None

            bao_f = None
            try:
                v = bao_relax.get("rep_bao_f_for_max_abs_z_le_1sigma")
                if v is not None:
                    bao_f = float(v)
            except Exception:
                bao_f = None

            sn = indep.get("sn_time_dilation") or {}
            cmb = indep.get("cmb_temperature") or {}
            sn_z = None
            cmb_z = None
            try:
                if sn.get("z_std") is not None:
                    sn_z = float(sn.get("z_std"))
                if cmb.get("z_std") is not None:
                    cmb_z = float(cmb.get("z_std"))
            except Exception:
                sn_z = cmb_z = None

            cosmo_tension_attr_summary_lines.append(
                "要旨：張力は距離指標依存（DDR/BAO/Tolman）に集中し、独立プローブ（時間伸長/T(z)）は整合。"
            )
            if ddr_z_abs is not None:
                msg = f"DDR（SNIa+BAO代表）: |z|≈{ddr_z_abs:.1f}"
                if ddr_sig_mult is not None:
                    msg += f" / 3σ回避にσ×{ddr_sig_mult:.1f}"
                cosmo_tension_attr_summary_lines.append(msg)
            if bao_f is not None:
                cosmo_tension_attr_summary_lines.append(
                    f"候補探索: 代表（BAO含むDDR）を1σに入れるにはBAOのσを f≈{bao_f:.2f} 倍に緩める必要"
                )
            if sn_z is not None and cmb_z is not None:
                cosmo_tension_attr_summary_lines.append(
                    f"独立プローブ: SN time dilation（z={sn_z:+.2f}）, CMB T(z)（z={cmb_z:+.2f}）"
                )

            if isinstance(next_actions, list) and next_actions:
                cosmo_tension_attr_detail_lines.append("次の検証（要約）:")
                for line in next_actions[:6]:
                    if not isinstance(line, str):
                        continue
                    cosmo_tension_attr_detail_lines.append(f"- {line}")
            cosmo_tension_attr_detail_lines.append(
                "再現: scripts/cosmology/cosmology_tension_attribution.py → output/cosmology/cosmology_tension_attribution.png"
            )
        except Exception:
            cosmo_tension_attr_summary_lines = [
                "出力の読み取りに失敗: output/cosmology/cosmology_tension_attribution_metrics.json"
            ]
            cosmo_tension_attr_detail_lines = []
    else:
        cosmo_tension_attr_summary_lines = [
            "出力未生成: scripts/cosmology/cosmology_tension_attribution.py を実行してください。"
        ]

    # Cosmology: DESI DR1 BAO promotion check (multi-tracer; screening→確証)
    cosmo_desi_promo = (
        _try_read_json(root / "output" / "private" / "cosmology" / "cosmology_desi_dr1_bao_promotion_check.json")
        or {}
    )
    cosmo_desi_promo_summary_lines: List[str] = []
    cosmo_desi_promo_detail_lines: List[str] = []
    if isinstance(cosmo_desi_promo, dict) and isinstance(cosmo_desi_promo.get("result"), dict):
        try:
            res = cosmo_desi_promo.get("result") or {}
            params = cosmo_desi_promo.get("params") or {}
            gate = cosmo_desi_promo.get("gate_by_tracer") or {}

            promoted = bool(res.get("promoted", False))
            passing = res.get("passing_tracers") or []
            passing = [str(x) for x in passing if str(x).strip()]
            min_tracers = params.get("min_tracers")
            thr = params.get("threshold_abs")
            z_field = str(params.get("z_field") or "")
            target_dist = str(params.get("target_dist") or "")

            if promoted:
                cosmo_desi_promo_summary_lines.append("DESI DR1 BAO: multi-tracer 昇格（screening→確証）= promoted")
            else:
                cosmo_desi_promo_summary_lines.append("DESI DR1 BAO: multi-tracer 昇格（screening→確証）= not promoted")

            if z_field and target_dist and thr is not None:
                cosmo_desi_promo_summary_lines.append(f"判定: dist={target_dist}, z={z_field}, stable |z|≥{thr:g}")
            if min_tracers is not None:
                cosmo_desi_promo_summary_lines.append(f"条件: min_tracers={int(min_tracers)}")
            if passing:
                cosmo_desi_promo_summary_lines.append("passing_tracers=" + ", ".join(passing))

            # Per-tracer z-range across methods (min/max over λ and methods).
            if isinstance(gate, dict) and passing:
                for tracer in passing[:6]:
                    g = gate.get(tracer) if isinstance(gate, dict) else None
                    ranges = g.get("ranges") if isinstance(g, dict) else None
                    stable = bool(g.get("stable_all_methods")) if isinstance(g, dict) else False
                    if not (isinstance(ranges, list) and ranges):
                        continue
                    try:
                        zmin = min(float(r.get("z_min")) for r in ranges if r.get("z_min") is not None)
                        zmax = max(float(r.get("z_max")) for r in ranges if r.get("z_max") is not None)
                        cosmo_desi_promo_summary_lines.append(
                            f"{tracer}: z∈[{zmin:.2f},{zmax:.2f}]（stable={'yes' if stable else 'no'}）"
                        )
                    except Exception:
                        continue

            cosmo_desi_promo_detail_lines.append(
                "このカードは『P-modelが正しい/誤り』を直接結論するものではなく、距離指標（BAO）の前提依存で張力が出ることを示す。"
            )
            cosmo_desi_promo_detail_lines.append(
                "張力は Step 4.7（DDR再接続条件）側で『距離指標前提に対する必要補正』として定量化する。"
            )
            cosmo_desi_promo_detail_lines.append(
                "再現: scripts/cosmology/cosmology_desi_dr1_bao_promotion_check.py（入力CSVは output/cosmology/cosmology_desi_dr1_bao_promotion_check.json の inputs.csv を参照）"
            )
        except Exception:
            cosmo_desi_promo_summary_lines = [
                "出力の読み取りに失敗: output/cosmology/cosmology_desi_dr1_bao_promotion_check.json"
            ]
            cosmo_desi_promo_detail_lines = []
    else:
        cosmo_desi_promo_summary_lines = [
            "出力未生成: scripts/cosmology/cosmology_desi_dr1_bao_promotion_check.py を実行してください。"
        ]

    # JWST/MAST spectra (x1d): cache/availability summary (distance-indicator independent).
    jwst_mast_summary_lines: List[str] = []
    jwst_mast_detail_lines: List[str] = []
    jwst_mast_table: Optional[Dict[str, Any]] = None
    jwst_waitlist_path = root / "output" / "private" / "cosmology" / "jwst_spectra_release_waitlist.json"
    jwst_waitlist_by_slug: Dict[str, Dict[str, Any]] = {}
    try:
        if jwst_waitlist_path.exists():
            wl = _try_read_json(jwst_waitlist_path) or {}
            blocked = wl.get("blocked_targets") if isinstance(wl, dict) else None
            if isinstance(blocked, list):
                for b in blocked:
                    if not isinstance(b, dict):
                        continue
                    slug = str(b.get("target_slug") or "").strip()
                    if slug:
                        jwst_waitlist_by_slug[slug] = dict(b)
    except Exception:
        jwst_waitlist_by_slug = {}

    jwst_manifest_all_path = root / "data" / "cosmology" / "mast" / "jwst_spectra" / "manifest_all.json"
    if jwst_manifest_all_path.exists():
        jwst_obj = _try_read_json(jwst_manifest_all_path) or {}
        try:
            jwst_items = jwst_obj.get("items") or {}
            if not isinstance(jwst_items, dict):
                jwst_items = {}
            table_rows: List[List[str]] = []
            ok_count = 0
            for slug, info in sorted(jwst_items.items(), key=lambda kv: str(kv[0])):
                if not isinstance(info, dict):
                    continue
                manifest_rel = _as_str(info.get("manifest") or "")
                manifest = _try_read_json(root / manifest_rel) if manifest_rel else None
                manifest = manifest if isinstance(manifest, dict) else {}
                target_name = _as_str(manifest.get("target_name") or "") or str(slug)
                obs_n = len(manifest.get("obs") or []) if isinstance(manifest.get("obs"), list) else 0
                qc = manifest.get("qc") if isinstance(manifest.get("qc"), dict) else {}
                x1d_local_n = int(qc.get("spectra_plotted") or 0) if isinstance(qc, dict) else 0
                if x1d_local_n > 0:
                    ok_count += 1

                # Release date (earliest)
                rel_utc = ""
                try:
                    rels = []
                    for o in manifest.get("obs") or []:
                        if not isinstance(o, dict):
                            continue
                        s = _as_str(o.get("t_obs_release_utc") or "")
                        if s:
                            rels.append(s)
                    rel_utc = min(rels) if rels else ""
                except Exception:
                    rel_utc = ""

                # Access state
                state = "ok" if x1d_local_n > 0 else "unknown"
                try:
                    dls = manifest.get("downloads") or []
                    if isinstance(dls, list) and any(
                        isinstance(d, dict) and int(d.get("status_code") or 0) == 401 for d in dls
                    ):
                        state = "proprietary/401"
                    elif str(slug) in jwst_waitlist_by_slug:
                        # The waitlist is the source of truth for future release dates (no download attempt needed).
                        state = "not_released_yet"
                    elif x1d_local_n <= 0 and obs_n > 0:
                        state = "no_local_x1d"
                    elif obs_n <= 0:
                        state = "not_found"
                except Exception:
                    state = "unknown"

                # Best z (if available)
                z_txt = ""
                nm_txt = ""
                try:
                    z_est = manifest.get("z_estimate") or info.get("z_estimate") or {}
                    if isinstance(z_est, dict) and bool(z_est.get("ok")):
                        best = z_est.get("best") or {}
                        if isinstance(best, dict):
                            z_mean = best.get("z_mean_from_matches")
                            z_raw = best.get("z")
                            z_val = z_mean if z_mean is not None else z_raw
                            if z_val is not None:
                                z_txt = f"{float(z_val):.3f}"
                            nm_txt = str(int(best.get("n_matches") or 0))
                except Exception:
                    z_txt = ""
                    nm_txt = ""

                table_rows.append([target_name, str(obs_n), str(x1d_local_n), rel_utc, z_txt, nm_txt, state])

            jwst_mast_summary_lines = [
                f"MAST（JWST）から x1d（1D spectrum）を取得し、ローカルへキャッシュ（対象 {len(table_rows)}）。",
                f"x1d 取得OK={ok_count}/{len(table_rows)}（残りは proprietary/未取得 等）。",
            ]
            if jwst_waitlist_by_slug:
                jwst_mast_summary_lines.append(
                    f"公開待ち（release日が未来）={len(jwst_waitlist_by_slug)}/{len(table_rows)}（waitlist固定）。"
                )
            jwst_mast_detail_lines = [
                "距離指標（ΛCDM距離など）の二次産物ではなく、スペクトル一次データから z（輝線のズレ）を直接扱う入口。",
                "保存: data/cosmology/mast/jwst_spectra/<target>/raw/*.fits（x1d）と manifest.json（取得条件/一覧）。",
                "再現（online取得）: python -B scripts/cosmology/fetch_mast_jwst_spectra.py --download-missing --max-obs 1",
                "再現（offline QC+z推定）: python -B scripts/cosmology/fetch_mast_jwst_spectra.py --offline --estimate-z",
            ]
            if jwst_waitlist_by_slug:
                jwst_mast_detail_lines.append(
                    "公開待ち（release日が未来）の定量化: python -B scripts/cosmology/jwst_spectra_release_waitlist.py"
                )
                jwst_mast_detail_lines.append("出力: output/cosmology/jwst_spectra_release_waitlist.json")
            jwst_mast_table = {
                "headers": ["対象", "obs数", "local x1d", "release(UTC)", "z(best)", "matches", "状態"],
                "rows": table_rows,
                "caption": "状態は取得可能性の目安（not_released_yet は waitlist に基づく）。z(best) は半自動候補（線同定は手動検証が必要）。",
            }
        except Exception:
            jwst_mast_summary_lines = ["出力の読み取りに失敗: data/cosmology/mast/jwst_spectra/manifest_all.json"]
            jwst_mast_detail_lines = []
            jwst_mast_table = None
    else:
        jwst_mast_summary_lines = [
            "未生成: python -B scripts/cosmology/fetch_mast_jwst_spectra.py --download-missing --max-obs 1",
        ]

    recent_worklog_table = _extract_recent_worklog_table(root, n=10)
    run_all_status_card = _extract_run_all_status_card(root)
    paper_table1_card = _extract_paper_table1_card(root)
    paper_html_card = _extract_paper_html_card(root)
    quantum_cards = _extract_quantum_public_cards(root)

    sections: List[Tuple[str, List[Dict[str, Any]]]] = [
        (
            "検証サマリ / 更新情報",
            [
                paper_table1_card,
                paper_html_card,
                run_all_status_card
                if isinstance(run_all_status_card, dict)
                else {
                    "id": "run_all_status",
                    "title": "再現バッチ（run_all）の状態",
                    "kind": "再現性（Phase 2）",
                    "summary_lines": [
                        "未生成: python -B scripts/summary/run_all.py --offline を実行してください。",
                    ],
                },
                {
                    "id": "recent_worklog",
                    "title": "最近の作業履歴（自動ログ）",
                    "kind": "履歴",
                    "summary_lines": ["直近の主要イベントを抜粋（スクリプト実行履歴）。"],
                    "detail_lines": ["機械可読ログ: output/private/summary/work_history.jsonl（手動編集しない）"],
                    "table": recent_worklog_table,
                }
                if isinstance(recent_worklog_table, dict)
                else {
                    "id": "recent_worklog",
                    "title": "最近の作業履歴（自動ログ）",
                    "kind": "履歴",
                    "summary_lines": ["ログ未生成: 先に python -B scripts/summary/run_all.py --offline を実行してください。"],
                    "detail_lines": ["機械可読ログ: output/private/summary/work_history.jsonl（手動編集しない）"],
                },
            ],
        ),
        (
            "理論（標準値との比較）",
            [
                {
                    "id": "theory_solar_deflection",
                    "title": "太陽重力による光の偏向（観測γ vs P-model）",
                    "kind": "観測（PPN γ）との比較",
                    "path": root / "output" / "theory" / "solar_light_deflection.png",
                    "summary_lines": panel_notes.get("太陽重力による光の偏向（観測γ vs P-model）", []),
                    "explain_lines": panel_explain.get("太陽重力による光の偏向（観測γ vs P-model）", []),
                    "detail_lines": [
                        "横軸は最近接距離 b（太陽半径 R_sun で規格化）、縦軸は偏向角 α（角秒）。",
                        "弱重力近似では、太陽縁（b=1 R_sun）で αは約1.75 角秒がよく引用される。",
                        "理想的には偏向角は b が大きいほど小さくなり、遠方では0に近づく（弱重力では概ね 1/b に比例）。",
                        "P-modelでは「質量近傍でPが増える」「光は高P側へ屈折する（最短時間経路）」という仮定から α(b) を計算する。",
                        "右図はVLBI等の観測から推定された PPNパラメータ γ（一次ソース）を用い、P-modelの予測 γ=2β-1 と比較する。",
                        "実観測では太陽コロナ（プラズマ）による電波の屈折/遅延が混ざるため、観測側は周波数や解析で補正している。",
                        "ここでは高次項（太陽の自転・四極子・強重力）などは無視した簡易モデルである点に注意。",
                        "再現: scripts/theory/solar_light_deflection.py → output/theory/solar_light_deflection.png",
                    ],
                },
                {
                    "id": "theory_gps_time_dilation",
                    "title": "GPSの時間補正（重力＋速度の内訳）",
                    "kind": "教科書値（代表）との比較",
                    "path": root / "output" / "theory" / "gps_time_dilation.png",
                    "summary_lines": [
                        "地上とGPS軌道での「1日あたり時間差」を、重力と速度に分解して比較。",
                        "P-model（近似）は、教科書で引用される代表値（+45.7, -7.2, 合計+38.5 µs/日）と整合。",
                    ],
                    "explain_lines": [
                        "重力が弱い場所（高高度）ほど時間は速く進む。",
                        "一方、速度が速いほど時間は遅く進む。",
                        "GPSは合計として地上より約+38 µs/日だけ速い（補正が必要）。",
                    ],
                    "detail_lines": [
                        "棒の値は「地上の時計」に対する「GPS衛星の時計」の進み方（1日あたり、マイクロ秒/日）。",
                        "重力項（正）は“高高度ほど重力が弱い→時間が速い”を表す（重力赤方偏移）。",
                        "速度項（負）は“高速運動→時間が遅い”を表す（特殊相対論的な時間遅れ）。",
                        "合計（net）がプラスなので、GPS衛星の時計は地上より速く進み、補正なしでは時刻がずれていく。",
                        "実運用では衛星搭載クロックの周波数を事前に少しずらして打ち上げ、地上受信時に合うようにする（平均の補正）。",
                        "この図は“平均の差”だけを示しており、離心軌道による周期成分 dt_rel は別項で扱われる（下のdt_rel図）。",
                        "P-modelではP分布から時間の進みを評価し、同じ単位に換算して標準的な代表値と一致するかを確認する。",
                        "再現: scripts/theory/gps_time_dilation.py → output/theory/gps_time_dilation.png",
                    ],
                },
                {
                    "id": "theory_gravitational_redshift",
                    "title": "重力赤方偏移（観測の偏差 ε vs P-model）",
                    "kind": "観測（一次ソース）との比較",
                    "path": root / "output" / "theory" / "gravitational_redshift_experiments.png",
                    "summary_lines": redshift_summary_lines,
                    "explain_lines": [
                        "重力が強い（低高度）ほど時計は遅く進み、周波数は赤方偏移する。",
                        "観測では z_obs=(1+ε)ΔU/c^2 と書き、一般相対論は ε=0 を予測する。",
                        "P-model（弱場・静止時計）も ε=0 を予測し、観測の ε が0付近かを確認する。",
                    ],
                    "detail_lines": [
                        *redshift_definition_lines,
                        "この図は一次ソースで公表された ε と σ（誤差）を、予測（0線）に対して誤差棒で示す。",
                        "zスコアは ε/σ（符号付き）で、|z|が小さいほど整合が良い。",
                        "再現: scripts/theory/gravitational_redshift_experiments.py → output/theory/gravitational_redshift_experiments.png",
                    ],
                    "table": redshift_table,
                },
                {
                    "id": "theory_frame_dragging",
                    "title": "回転（フレームドラッグ）：観測比 μ vs P-model",
                    "kind": "観測（一次ソース）との比較",
                    "path": root / "output" / "theory" / "frame_dragging_experiments.png",
                    "summary_lines": frame_drag_summary_lines,
                    "explain_lines": [
                        "自転する天体の周りでは、空間（局所慣性系）がわずかに引きずられて歳差する（フレームドラッグ）。",
                        "人工衛星やジャイロ（GP-B）の歳差を使い、予測と観測が一致するかを検証する。",
                        "ここでは比 μ=|Ω_obs|/|Ω_pred| を用い、μ=1 が一致を意味する。",
                    ],
                    "detail_lines": [
                        *frame_drag_definition_lines,
                        "この図は一次ソースで公表された μ と σ（誤差）を、予測（1線）に対して誤差棒で示す。",
                        "zスコアは (μ-1)/σ（符号付き）で、|z|が小さいほど整合が良い。",
                        "再現: scripts/theory/frame_dragging_experiments.py → output/theory/frame_dragging_experiments.png",
                    ],
                    "table": frame_drag_table,
                },
            ],
        ),
        (
            "決定的検証（Phase 7）",
            [
                _extract_validation_scoreboard_card(root),
                _extract_decisive_falsification_card(root),
            ],
        ),
        (
            "量子（Phase 7）",
            quantum_cards,
        ),
        (
            "差分予測の拡張（Phase 8）",
            [
                _extract_decisive_candidates_card(root),
            ],
        ),
        (
            "GPS（観測IGSとの比較）",
            [
                {
                    "id": "gps_rms_all",
                    "title": "GPS：観測IGSに対する残差RMS（全衛星）",
                    "kind": "観測（準実測）= IGS Final CLK/SP3",
                    "path": root / "output" / "gps" / "gps_rms_compare.png",
                    "summary_lines": panel_notes.get("GPS（時計残差：観測IGS vs P-model）", []),
                    "explain_lines": panel_explain.get("GPS（時計残差：観測IGS vs P-model）", []),
                    "detail_lines": [
                        "基準は IGS Final の精密クロック/精密軌道（CLK/SP3）で、ここでは準実測として扱う。",
                        "比較対象は、(青)放送暦（BRDC）と、(橙)P-model推定の衛星クロック（または補正）の2つ。",
                        "各衛星ごとに IGS に対する残差時系列を作り、バイアス＋ドリフト（一次）を除去した後のRMSを計算する。",
                        "棒が低いほど IGS に近い＝観測整合が良い（ただしIGS自体にも誤差はある）。",
                        "このRMSは“1日の中の変動成分”の比較で、平均の周波数オフセット（定数差）は評価対象に入れていない。",
                        "P-model側はGNSS運用慣例に合わせ、離心による dt_rel（近日点効果）を別扱いにしている。",
                        "衛星ごとに差が大きい場合は、軌道/時刻の入力、補正項の不足、データ品質などの要因が考えられる。",
                        "再現: scripts/gps/plot.py → output/gps/gps_rms_compare.png",
                    ],
                },
                {
                    "id": "gps_residual_g01",
                    "title": "GPS 時計残差：G01（観測IGSに対する比較）",
                    "kind": "観測（準実測）= IGS Final CLK",
                    "path": root / "output" / "gps" / "gps_residual_compare_G01.png",
                    "summary_lines": [
                        "1衛星（G01）の残差の時系列例。",
                        "バイアス＋ドリフト（一次）を除去した後の揺らぎを比較。",
                    ],
                    "explain_lines": [
                        "青は放送暦（BRDC）、橙はP-modelの推定クロック。",
                        "線が0付近で安定するほど観測（IGS）と整合している。",
                    ],
                    "detail_lines": [
                        "上のRMS図の“中身”を、1衛星（G01）で時系列として示した例。",
                        "縦軸は残差（ns）、横軸はUTC時刻。残差は0付近が一致（観測=IGSに近い）。",
                        "バイアス＋ドリフト（一次）を除去しているため、ここでは周期的な揺らぎや形の一致度が主役。",
                        "主な周期成分は軌道周期（約12時間）や、その高調波として現れやすい。",
                        "P-model線（橙）がBRDC（青）より0付近に長く留まれば、P-modelが観測の変動をより説明できている。",
                        "逆に位相ずれや振幅差が見える場合、dt_rel扱い・軌道入力・補正項（例：相対補正/地球自転関連）を疑う。",
                        "短時間の尖りは観測側（IGS推定）のノイズやイベントの可能性もあるため、他衛星や別日で再現性を見る。",
                        "再現: scripts/gps/plot.py → output/gps/gps_residual_compare_G01.png",
                    ],
                },
                {
                    "id": "gps_brdc_all",
                    "title": "GPS 放送暦 時計残差（BRDC - IGS, 全衛星）",
                    "kind": "観測（準実測）= IGS Final CLK",
                    "path": root / "output" / "gps" / "gps_clock_residuals_all_31.png",
                    "summary_lines": ["全衛星の BRDC - IGS 残差を重ね描き（参考）。"],
                    "explain_lines": [
                        "放送暦（BRDC）は実用のための簡易モデルで、精密プロダクト（IGS）と差が出る。",
                        "衛星ごとの差や、時間帯による揺らぎの雰囲気を把握するための図。",
                    ],
                    "detail_lines": [
                        "各色の線が各衛星の BRDC - IGS 残差（ns）で、全衛星分を同一図に重ねている。",
                        "IGSを“精密側の基準”として見たとき、放送暦（BRDC）がどの程度ばらつくかの全体像を示す。",
                        "線の太い帯の幅が、BRDCモデルの典型的な誤差スケール（この日/この期間）と考えられる。",
                        "衛星ごとの段差や外れ値は、衛星固有の事情（運用、機器、データ欠損）や推定の切替で起き得る。",
                        "P-model検証の本体は「IGSに対して残差が減るか」なので、この図は“ベースライン”の参考。",
                        "比較を公平にするため、残差の算出ではバイアス＋ドリフト（一次）を除去している。",
                        "再現: scripts/gps/plot.py → output/gps/gps_clock_residuals_all_31.png",
                    ],
                },
                {
                    "id": "gps_dt_rel",
                    "title": "GPS：相対補正（近日点効果） 標準式 vs P-model（同じ周期成分）",
                    "kind": "補助図（dt_rel の扱いの確認）",
                    "path": root / "output" / "gps" / "gps_relativistic_correction_G02.png",
                    "summary_lines": [
                        f"相関={_format_num(rel_corr, digits=6)}",
                        f"RMSE={_format_num(rel_rmse_ns, digits=4)} ns（周期成分）",
                        "IGSの慣例に合わせ、P-model比較では dt_rel を別扱いにしている。",
                    ],
                    "explain_lines": [
                        "GNSSでは、軌道離心率に由来する相対補正 dt_rel を別項で扱う慣例がある。",
                        "P-model側の周期成分が標準式の dt_rel と一致することを確認する図。",
                    ],
                    "detail_lines": [
                        "標準式の dt_rel は、離心軌道で生じる周期的な相対論補正（近日点効果）を表す。",
                        "GNSSでよく使われる形は dt_rel = -2 (r·v) / c^2（r:地心位置, v:速度）で、ほぼ正弦波になる。",
                        "この項は軌道周期で変動し、近地点/遠地点で符号や傾きが入れ替わるのが特徴。",
                        "この図は“周期成分だけ”を取り出して比較するため、定数オフセットやドリフトは除去している。",
                        "相関が高くRMSEが小さいほど、P-modelが標準式が表す周期補正を再現できている。",
                        "IGS比較（残差RMS）では、運用慣例に合わせて dt_rel を除去した状態で評価している。",
                        "もしここで一致しない場合、P-modelの時間率の定義（P→時間）や、軌道入力の整合を見直す必要がある。",
                        "この図は“dt_relを別項にする妥当性”の確認であり、P-model全体の優劣を直接示すものではない。",
                        "再現: scripts/gps/plot.py → output/gps/gps_relativistic_correction_G02.png",
                    ],
                },
            ],
        ),
        (
            "太陽会合（Cassini / Viking）",
            [
                {
                    "id": "cassini_overlay",
                    "title": f"Cassini：ドップラー y（±10日, {cassini_source_tag} vs P-model）",
                    "kind": cassini_obs_kind,
                    "path": root / "output" / "cassini" / "cassini_fig2_overlay_zoom10d.png",
                    "summary_lines": (
                        [
                            (f"観測系列: {cassini_obs_label}" if cassini_obs_label else f"観測ソース: {cassini_effective_source or 'unknown'}"),
                            f"RMSE={_format_num(cassini_m.get('rmse'), digits=4)}",
                            f"相関={_format_num(cassini_m.get('corr'), digits=6)}",
                        ]
                        if cassini_m
                        else ["（未生成）scripts/cassini/cassini_fig2_overlay.py を実行してください。"]
                    ),
                    "explain_lines": [
                        "太陽会合時の電波ドップラー y（周波数比）の時間変化。",
                        "観測系列（一次データTDFから処理、または論文図デジタイズ）と、P-model を比較する。",
                        "形状の一致度を RMSE / 相関 で要約する。",
                    ],
                    "detail_lines": [
                        "縦軸 y はドップラーの周波数比（fractional frequency, 無次元）で、電波の伝搬時間変化が周波数に現れたもの。",
                        "太陽会合では視線が太陽近傍を通り、重力による Shapiro 遅延が最大になる。",
                        "y(t) は遅延の“時間微分”に近いため、会合中心付近でS字に大きく変化し、前後で符号が変わる形になる。",
                        "点は観測系列。PDS TDFは Doppler pseudo-residual（観測-予測）なので、参照Shapiro y_ref(t) を足し戻して y(t) を復元し、平滑化＋デトレンドしている（論文図スケール）。",
                        "線はP-modelで計算した y(t)（太陽Shapiroによる寄与）。",
                        "PDS由来の処理ログ/中間生成物（PDS使用時）: output/cassini/cassini_sce1_tdf_extracted.csv, output/cassini/cassini_sce1_tdf_paperlike.csv",
                        "βやδなどのパラメータは y(t) の振幅/形に効くため、残差やRMSEで感度を見る。",
                        "会合中心時刻や最近接距離 b_min の誤差は、位相ずれとして現れやすい（曲線の左右シフト）。",
                        "再現: scripts/cassini/cassini_fig2_overlay.py → output/cassini/cassini_fig2_overlay_zoom10d.png",
                    ],
                },
                {
                    "id": "cassini_pds_vs_digitized",
                    "title": "Cassini：PDS一次データ（処理後） vs 論文図デジタイズ",
                    "kind": "整合チェック（一次データ→論文図スケール）",
                    "path": root / "output" / "cassini" / "cassini_pds_vs_digitized.png",
                    "summary_lines": (
                        [
                            f"最適時間シフト={_format_num(cassini_pds_vs_dig.get('shift_days'), digits=4)} 日",
                            f"RMSE={_format_num(cassini_pds_vs_dig.get('rmse'), digits=4)}",
                            f"相関={_format_num(cassini_pds_vs_dig.get('corr'), digits=6)}",
                            f"点数={_format_num(cassini_pds_vs_dig.get('n'), digits=6)}",
                        ]
                        + (
                            [
                                "（シフトなし: "
                                f"RMSE={_format_num(cassini_pds_vs_dig.get('rmse_zero_shift'), digits=4)}, "
                                f"相関={_format_num(cassini_pds_vs_dig.get('corr_zero_shift'), digits=6)}）"
                            ]
                            if cassini_pds_vs_dig.get("rmse_zero_shift") is not None
                            else []
                        )
                        if cassini_pds_vs_dig
                        else ["（未生成）cassini_fig2_overlay.py 実行時に自動生成されます。"]
                    ),
                    "explain_lines": [
                        "PDS一次データ（TDF）から復元した y(t) が、論文図のスケール感と整合するかを確認する図。",
                        "ここが大きくズレる場合、TDF復元（固定小数点/符号）や処理（ビン幅/デトレンド）を見直す。",
                        "論文図の t=0 と、モデル側の b_min（最近接）定義の差を吸収するため、最適な時間シフトも併記する。",
                    ],
                    "detail_lines": [
                        "青: PDS一次データ（TDF, 疑似残差）に参照Shapiroを足し戻して復元した y(t)（平滑化＋デトレンド後）。",
                        "橙点: 公開論文の図からデジタイズした y(t)。",
                        "一致度（RMSE/相関）と、最適時間シフトを output/cassini/cassini_pds_vs_digitized_metrics.csv に保存している。",
                        "再現: scripts/cassini/cassini_fig2_overlay.py → output/cassini/cassini_pds_vs_digitized.png",
                    ],
                },
                {
                    "id": "cassini_residual",
                    "title": "Cassini：残差（観測 - P-model）",
                    "kind": cassini_obs_kind,
                    "path": root / "output" / "cassini" / "cassini_fig2_residuals.png",
                    "summary_lines": [
                        "観測系列とモデルの差（残差）を可視化。",
                        "モデルの傾向・ズレの形を確認する。",
                    ],
                    "explain_lines": [
                        "残差が0に近いほど、観測点列とモデルが一致。",
                        "太陽会合中心付近でのズレが最も効きやすい。",
                    ],
                    "detail_lines": [
                        "残差 = 観測 - P-model。0に近いほど一致。",
                        "残差が会合中心付近で大きい場合、モデルの強さ（β）や b_min の扱い、またはプラズマ補正の不足が疑わしい。",
                        "残差が全体的に片側へ寄る場合は、定数オフセット（基準レベル）や符号/基準の取り方の可能性がある。",
                        "左右で非対称な残差は、会合中心時刻のずれや、地上局/探査機の幾何がずれているサインになり得る。",
                        "残差の“形”は次の改善（β最適化、幾何の再計算、データの一次化）を決めるための診断材料。",
                        "最終的に一次ソースに置き換えた後は、残差が観測ノイズ水準に近づくかを評価する。",
                        "再現: scripts/cassini/cassini_fig2_overlay.py → output/cassini/cassini_fig2_residuals.png",
                    ],
                },
                *(
                    [
                        {
                            "id": "cassini_beta_sweep",
                            "title": "Cassini：βスイープ（RMSE）",
                            "kind": "パラメータ感度（最適βの探索）",
                            "path": root / "output" / "cassini" / "cassini_beta_sweep_rmse.png",
                            "summary_lines": [
                                f"最小RMSE(±10日)のβ={_format_num(cassini_best.get('best_beta_by_rmse10'), digits=8)}",
                                f"最小RMSE(±10日)={_format_num(cassini_best.get('best_rmse10'), digits=4)}",
                            ],
                            "explain_lines": [
                                "βを微小に変えると、モデル y(t) の形が変わる。",
                                "RMSEが最も小さいβが、観測（PDS処理後）に最も合う値。",
                            ],
                            "detail_lines": [
                                "βはP-modelの効果の強さ（Shapiro/屈折の係数）を調整するパラメータ。",
                                "GRで現れる係数2（PPNの(1+γ)=2）は、P-modelでは β=1 として光伝播側が担う（重力ポテンシャルφの定義は固定）。",
                                "βを掃引し、観測（PDS処理後）とのRMSEが最小になる値を探索している。",
                                "横軸がβ、縦軸がRMSE（小さいほど一致）。",
                                "最小RMSEのβが1付近なら、標準値（GRでのγ=1付近）と同等の強さになる、という読み方ができる。",
                                "ただしRMSEは評価窓（例：±10日）や処理条件（ビン幅/デトレンド）に依存するため、最小値そのものより“傾向”を重視する。",
                                "曲線が鋭いほどβに敏感で、曲線が平坦ならβの同定が難しい（別のデータで拘束が必要）。",
                                "Cassiniだけで決めたβが他の検証（Viking/LLR/光偏向など）でも整合するかが重要。",
                                "再現: scripts/cassini/cassini_fig2_overlay.py（--no-sweep なし）→ output/cassini/cassini_beta_sweep_rmse.png",
                            ],
                        }
                    ]
                    if cassini_best
                    else []
                ),
                *(
                    [
                {
                    "id": "bepicolombo_more_psa_status",
                    "title": "BepiColombo（MORE）：PSA公開状況（一次ソース確認）",
                    "kind": "データ準備（一次ソースの公開状況）",
                    "path": root / "output" / "bepicolombo" / "more_psa_status.png",
                    "summary_lines": _bepicolombo_more_psa_summary_lines(),
                    "explain_lines": [
                        "BepiColombo（MPO/MORE）の一次データが公開されれば、Cassiniと同様にShapiro（重力遅延）の高精度検証が可能になる。",
                        "まずは ESA PSA を一次ソースとして『どのディレクトリが公開されているか』を機械的に記録し、以後の再現性を担保する。",
                    ],
                    "table": bepi_docs_table,
                    "detail_lines": [
                        "この図は『観測データそのもの』の比較ではなく、検証に使える一次データが公開されているかを確認するための“入口”である。",
                        "PSA上の `bc_mpo_more/` が bundle/document だけ公開で data_raw などが404の場合、現時点では時系列検証（y(t)/range）ができない。",
                        "公開が進んだら、bundleが指す collection（data_raw 等）を `data/bepicolombo/` に取得し、Cassiniと同じバッチ/レポート形式へ拡張する。",
                        "再現: scripts/bepicolombo/more_psa_status.py → output/bepicolombo/more_psa_status.png",
                        "PDFも取得: scripts/bepicolombo/more_psa_status.py --download-pdfs",
                        "詳細: doc/bepicolombo/README.md",
                        "索引: scripts/bepicolombo/more_document_catalog.py → output/bepicolombo/more_document_catalog.csv",
                        "取得: scripts/bepicolombo/more_fetch_collections.py → output/bepicolombo/more_fetch_collections.json",
                        "準備メモ: doc/bepicolombo/MORE_DATA_NOTES.md",
                    ],
                },
                {
                    "id": "bepicolombo_spice_psa_status",
                    "title": "BepiColombo（SPICE）：PSA公開状況（一次ソース確認）",
                    "kind": "幾何準備（SPICE kernels / inventory）",
                    "path": root / "output" / "bepicolombo" / "spice_psa_status.png",
                    "summary_lines": _bepicolombo_spice_psa_summary_lines(),
                    "explain_lines": [
                        "BepiColombo（MPO/MORE）の検証では、探査機・惑星・地球の幾何（位置/速度）が支配的になる。",
                        "その基礎となるSPICE kernels（軌道/姿勢/時刻/定数など）が一次ソース（ESA PSA）で公開されているかを確認する。",
                    ],
                    "detail_lines": [
                        "この図は観測値（レンジ/ドップラー）の比較ではなく、“幾何モデルの一次ソースが揃うか”のチェックである。",
                        "inventory（CSV）は、公開されているSPICE製品（例：SPK/CK/LSK/PCK/FK/IKなど）の一覧を示す。",
                        "この段階での完了条件は、最新inventoryの取得と、MPO/MMO/MTM等の内訳が追跡できる状態にすること。",
                        "次工程では、SPICE kernels を `data/bepicolombo/` に取得し、Cassiniと同じ形式で y(t)/range の検証へ進む。",
                        "再現: scripts/bepicolombo/spice_psa_status.py → output/bepicolombo/spice_psa_status.png",
                        "詳細: doc/bepicolombo/README.md",
                    ],
                },
                {
                    "id": "bepicolombo_shapiro_predict",
                    "title": "BepiColombo（水星探査）：太陽会合 Shapiro 予測（SPICE幾何）",
                    "kind": "予測（y(t)/Δt：観測前の幾何チェック）",
                    "path": root / "output" / "bepicolombo" / "bepicolombo_shapiro_geometry.png",
                    "summary_lines": _bepicolombo_shapiro_predict_summary_lines(),
                    "explain_lines": [
                        "BepiColomboの軌道（SPICE一次ソース）から、太陽会合の“理想的な”Shapiro信号の強さと形を予測する。",
                        "観測（MORE）が公開された後は、この予測曲線を基準にレンジ/ドップラー時系列を比較する。",
                    ],
                    "detail_lines": [
                        "この図は観測（レンジ/ドップラー）との比較ではなく、一次ソース（SPICE）から計算した“予測曲線”である。",
                        "上段: 太陽への最接近距離 b(t)（太陽半径 R_sun 単位）。会合中心は b が最小になる時刻。",
                        "中段: P-model（β=1）での往復Shapiro遅延 Δt(t) の予測（μs）。",
                        "下段: Δt の時間微分に相当する y(t) の予測（Eq.2）。ドップラーで見える成分。",
                        "b<R_sun（太陽円盤内）は遮蔽扱いとして y/Δt を NaN にし、可視領域（min_b_rsun 以上）のみでピーク等を評価する。",
                        "再現（オンラインでカーネル取得）: python -B scripts/bepicolombo/fetch_spice_kernels_psa.py",
                        "再現（予測図生成）: python -B scripts/bepicolombo/bepicolombo_shapiro_predict.py --min-b-rsun 1.0",
                        "詳細: doc/bepicolombo/README.md",
                    ],
                },
                {
                    "id": "bepicolombo_conjunction_catalog",
                    "title": "BepiColombo（水星探査）：太陽会合イベント一覧（Shapiro予測, SPICE幾何）",
                    "kind": "カタログ（観測前の準備）",
                    "path": root / "output" / "bepicolombo" / "bepicolombo_conjunction_catalog.png",
                    "summary_lines": _bepicolombo_conjunction_catalog_summary_lines(),
                    "explain_lines": [
                        "SPICE幾何（一次ソース）から、太陽会合（bが小さい）イベントを抽出して一覧化する。",
                        "MOREの一次データが公開されたら、この一覧のイベント窓で観測 y(t) を重ねて検証する。",
                    ],
                    "detail_lines": [
                        "この図は観測（レンジ/ドップラー）との比較ではなく、SPICE幾何だけで作る『会合イベントのカレンダー』。",
                        "上段: 各イベントの最小インパクトパラメータ b（R_sun）。小さいほど太陽に近い。",
                        "下段: 往復 Shapiro 遅延 Δt（μs, 非遮蔽側）。大きいほど信号が強い（ただし太陽コロナの影響も増える）。",
                        "赤点は『rawの最小点が太陽円盤内（遮蔽）』があるイベント（観測には不利）。",
                        "時刻は refine_step_sec の分解能に依存（中心時刻を詰めたい場合は --refine-step-sec を下げて再生成）。",
                        "再現: python -B scripts/bepicolombo/bepicolombo_conjunction_catalog.py --min-b-rsun 1.0 --max-b-rsun 10.0",
                        "出力: output/bepicolombo/bepicolombo_conjunction_catalog.csv / .png / _summary.json",
                        "詳細: doc/bepicolombo/README.md",
                    ],
                },
                    ]
                    if INCLUDE_BEPICOLOMBO_IN_PUBLIC_REPORT
                    else []
                ),
                {
                    "id": "viking_shapiro",
                    "title": "Viking シャピロ遅延（代表値 vs P-model）",
                    "kind": "文献代表値（ピーク約250 µs）",
                    "path": root / "output" / "viking" / "viking_p_model_vs_measured_no_arrow.png",
                    "summary_lines": panel_notes.get("Viking シャピロ遅延（代表値 vs P-model）", []),
                    "explain_lines": panel_explain.get("Viking シャピロ遅延（代表値 vs P-model）", []),
                    "detail_lines": [
                        "Vikingでは地球-火星の電波通信で、太陽近傍を通ると往復時間が増える（Shapiro遅延）。",
                        "縦軸は遅延時間（マイクロ秒）、横軸は日付。太陽会合付近でピークになり、離れると小さくなる（山形）。",
                        "赤点は文献で引用されるピーク代表値（約200-250マイクロ秒）。",
                        "この図は“実測の時系列”ではなく、代表値とのオーダー比較（sanity check）が目的。",
                        "一次ソースに置き換える段階では、レンジ/ドップラーの時系列で、時間軸・幾何・補正項まで含めて評価する。",
                        "太陽コロナ（プラズマ）や惑星暦の誤差が混ざるため、観測側の前処理（周波数多重等）が重要になる。",
                        "βが普遍なら、Cassiniで整えたβと同程度の強さになるはず（ずれる場合はモデルの前提や補正を再点検）。",
                        "再現: scripts/viking/update_slides.py → output/viking/viking_p_model_vs_measured_no_arrow.png",
                    ],
                },
            ],
        ),
        (
            LLR_LONG_NAME,
            [
                {
                    "id": "llr_final_summary",
                    "title": f"{LLR_SHORT_NAME}：決定版（現状まとめ）",
                    "kind": "まとめ（再現条件 + 主要数値）",
                    "path": root / "output" / "llr" / "batch" / "llr_rms_improvement_overall.png",
                    "summary_lines": (
                        [
                            f"生成UTC: {llr_batch_summary.get('generated_utc')}",
                            f"バッチ: {llr_batch_summary.get('n_files')} ファイル / {llr_batch_summary.get('n_points_total')} 点（重複除去後）",
                            f"評価単位: {llr_batch_summary.get('n_groups')} グループ（station×reflector, n≥{llr_batch_summary.get('min_points_per_group')}）",
                            f"β={_format_num(llr_batch_summary.get('beta'), digits=4)} / time-tag={llr_batch_summary.get('time_tag_mode')} / 局座標={llr_batch_summary.get('station_coords_mode')}",
                            (
                                "pos+eop適用局: "
                                + f"{len(llr_batch_summary.get('station_coord_summary') or {})}"
                                + f"/{len(llr_batch_summary.get('stations') or [])}"
                            ),
                            (
                                "中央値RMS（SR+Tropo+Tide）="
                                + f"{_format_num(llr_batch_summary.get('median_rms_ns', {}).get('station_reflector_tropo_tide'), digits=4)} ns"
                            ),
                            "目安: 1 ns（往復）≈0.30 m、片道距離では約0.15 m",
                        ]
                        if llr_batch_summary
                        else [
                            "LLRバッチ結果（output/llr/batch/llr_batch_summary.json）が見つからないため、現状まとめを表示できません。",
                            "先に scripts/llr/llr_batch_eval.py を実行してください。",
                        ]
                    ),
                    "explain_lines": [
                        "この章は「月レーザー測距（LLR）」を、観測（EDC公開データ）とモデル（HORIZONS/Spice + P-model補正）で比較する。",
                        "まず幾何モデルが支配的なので、ns級の残差に入るまで座標系・時刻タグ・大気・潮汐などを優先して固める。",
                        "この“現状まとめ”は、再現条件と主要数値を固定し、改善の前後で変化を追えるようにするためのカード。",
                    ],
                    "detail_lines": [
                        "LLR（Lunar Laser Ranging）は、地上局→月面反射器→地上局の往復レーザー飛行時間（TOF）を測る観測。",
                        "P-modelの検証としては、まず“光路が高P側へ曲がる/遅れる”という仮説が、時系列の形と残差で破綻しないかを見る。",
                        "一方でLLRは、地球自転・月回転・局座標・反射器座標・大気遅延が支配的で、これらが崩れるとP-model以前に一致しない。",
                        "ここでのRMSは TOF（往復時間）の残差で評価している。1 ns（往復）は光路長で約0.30 m、片道距離で約0.15 mに相当。",
                        "再現（オフライン）: python -B scripts/llr/llr_batch_eval.py --out-dir output/llr/batch --offline --time-tag-mode auto --chunk 50 --min-points 30",
                        "改善候補（次工程）: pos+eop（日次）キャッシュ拡充→対流圏（MET欠損の扱い）→潮汐/荷重（より精密なモデル）を段階的に追加し、RMSの低下と系統残差の消失を確認する。",
                    ],
                },
                {
                    "id": "llr_time_tag_selection",
                    "title": f"{LLR_SHORT_NAME}：time-tag 最適化（局ごとに tx/rx/mid を選択）",
                    "kind": "バッチ（時刻タグの切り分け）",
                    "path": root / "output" / "llr" / "batch" / "llr_time_tag_selection_by_station.png",
                    "summary_lines": _llr_best_time_tag_summary_lines(),
                    "explain_lines": [
                        "LLRの観測時刻は、送信(tx)/受信(rx)/中点(mid)のどれを指すかがデータ系統で混在しうる。",
                        "各局について、残差RMSが最小になる time-tag モードを選ぶ。",
                    ],
                    "detail_lines": [
                        "LLRは 2.5秒程度の往復飛行時間（TOF）を扱うが、ns級で詰めるには「時刻タグの定義」が重要になる。",
                        "tx/rx/mid は、観測ファイルに書かれた時刻を『送信時刻』『受信時刻』『往復の中点時刻（反射時刻近傍）』のどれとして解釈するかの仮説。",
                        "同じ観測でも time-tag の解釈がずれると、幾何（特に観測局の自転・月の運動）が違う時刻で評価され、残差が大きく悪化する。",
                        "この図は、各局で tx/rx/mid を総当たりし、station×reflector ごとの定数オフセット整列後の残差RMS（ns）で比較したもの。",
                        "結果として多くの局で tx が最も良いなら、以降の評価は tx を既定として固定できる（再現性のため）。",
                        "再現: scripts/llr/llr_batch_eval.py（time-tag 探索）→ output/llr/batch/llr_time_tag_selection_by_station.png",
                    ],
                },
                {
                    "id": "llr_tof_timeseries",
                    "title": f"{LLR_SHORT_NAME}（CRD Normal Point）：TOF 時系列",
                    "kind": "観測（CRD record 11）",
                    "path": root / "output" / "llr" / f"{llr_stem}_tof_timeseries.png",
                    "summary_lines": [
                        f"点数 n={_format_num(llr_summary.get('n_npt11'), digits=3)}",
                        f"station={llr_summary.get('station')}",
                        f"target={llr_summary.get('target')}",
                        (
                            f"出典: {llr_source.get('source')} / {llr_source.get('target')} / {llr_source.get('filename')}"
                            if (llr_stem == "llr_primary" and llr_source)
                            else "入力: デモ（demo_llr_like.crd）"
                        ),
                    ],
                    "explain_lines": [
                        "LLRの観測データ（Normal Point）から抽出した往復飛行時間（TOF）。",
                        (
                            "公開データ（EDC）を data/llr にキャッシュして再現できる形にしている。"
                            if llr_stem == "llr_primary"
                            else "このリポジトリのデモ入力（demo_llr_like.crd）で動作確認している。"
                        ),
                    ],
                    "detail_lines": [
                        "月レーザー測距（LLR: Lunar Laser Ranging）は、地上局→月面反射器→地上局の「往復」レーザーの飛行時間（TOF）を測る観測。",
                        "Normal Point（NP）は、短時間に得た多数のショットを平均化してノイズを下げたデータ（CRD record 11）。",
                        "TOFの変化は主に「地球自転」「地上局の位置」「月の位置・回転」「反射器の位置」による幾何で決まる。",
                        "ns精度で詰めると、1 ns（往復）は光路長で約0.30 m、片道距離で約0.15 mに相当。",
                        "縦軸のTOFは秒オーダー（約2.56 s）で、往復距離（約76万 km）を光速で割った値に相当する。",
                        "観測値には幾何以外に、局内遅延・大気遅延・相対論補正などが加わるため、精密化には補正モデルが必要。",
                        "この図は「データの読み取り（時刻・TOF）の妥当性」をまず確認するための“入口”の図。",
                        "モデルの正否は、重ね合わせ図と残差図で“変動成分”がnsレベルまで一致するかで判断する。",
                        "再現: scripts/llr/llr_crd_quicklook.py（CRD/NP2）→ output/llr/*_tof_timeseries.png",
                    ],
                },
                {
                    "id": "llr_range_timeseries",
                    "title": f"{LLR_SHORT_NAME}（CRD Normal Point）：片道距離 時系列",
                    "kind": "観測（CRD record 11）",
                    "path": root / "output" / "llr" / f"{llr_stem}_range_timeseries.png",
                    "summary_lines": [],
                    "explain_lines": [
                        "TOFから換算した片道距離（参考）。",
                        "本来は局・反射器・補正項を含めた厳密処理が必要。",
                    ],
                    "detail_lines": [
                        "TOFを光速で換算して片道距離（range）にした参考図（概ね range = c*TOF/2）。",
                        "厳密には大気遅延や局内部遅延などが入るため、絶対値（オフセット）よりも変動の形を見る用途が中心。",
                        "rangeは地上局-月面反射器の距離で、平均は約38万 km（図の縦軸はその近傍）になる。",
                        "1 ns（TOF）に相当する距離は約0.15 m（片道）なので、ns精度は“10 cm台”の距離精度に相当する。",
                        "絶対値を合わせるには、反射器座標・局座標・時間系（UTC/TT/TDB）・大気遅延・潮汐などの整合が必要。",
                        "本検証ではまず、定数オフセットを除いた“変動成分”が一致するかを重視する（残差図）。",
                        "再現: scripts/llr/llr_crd_quicklook.py（CRD/NP2）→ output/llr/*_range_timeseries.png",
                    ],
                },
                {
                    "id": "llr_overlay",
                    "title": f"{LLR_SHORT_NAME}：観測 vs P-model（太陽Shapiro含む）",
                    "kind": "観測（CRD） + 幾何モデル（HORIZONS + SPICE）",
                    "path": root / "output" / "llr" / "out_llr" / f"{llr_stem}_overlay_tof.png",
                    "summary_lines": [
                        "観測局→月/反射器の往復TOFを、HORIZONS（EOP適用の幾何）とSPICE（月回転 MOON_PA_DE421）で計算。",
                        "定数オフセット整列で、局・系遅延など「定数の差」を吸収して変動成分を比較する。",
                    ],
                    "explain_lines": [
                        "観測TOFの大部分は「局位置・地球自転・反射器位置」などの幾何で決まる。",
                        "線が重なって見える場合は、差がns級で、TOF変動成分がms級（=数千万ns）だから。",
                        "月回転はSPICEのDE421 Principal Axes（MOON_PA_DE421）を使用し、反射器座標（Moon PA）と整合させる。",
                        "次の段階：反射器座標/局座標の一次ソース確定、追加補正（大気等）の導入。",
                    ],
                    "detail_lines": [
                        "観測TOFとモデルTOFを同一軸に重ね、形（時間変化）が合うかを確認する図。",
                        "比較の前に定数オフセットを整列して、装置遅延・基準系の定数差を吸収し、変動成分に注目する。",
                        "モデル側は HORIZONS（観測局のtopocentric幾何/EOPを含む）とSPICE（月回転 MOON_PA_DE421）で反射器位置を扱う。",
                        "ここで合わない場合は、(1)座標系の不整合 (2)月回転モデル (3)大気遅延 (4)反射器座標の出典差 などが疑わしい。",
                        "LLRのTOFは、幾何（地球自転・局位置・月軌道・月回転）が支配的なので、まず“形が合うか”が最重要。",
                        "月面反射器は月固定座標で与えられるため、月回転モデル（MOON_PA_DE421）と反射器座標のフレーム整合が鍵になる。",
                        "太陽Shapiroは会合ほど大きくないが、ns精度で詰める段階では無視できない補正項になる。",
                        "定数オフセット整列後でもズレが残る場合、未補正の大気遅延（対流圏）や、局座標/時刻系の不整合が疑わしい。",
                        "線の重なりでは差が見えにくいので、次の残差図（観測 - モデル）で“ns級のズレ”を読む。",
                        "次の読み方は、同じ条件で残差（観測 - モデル）がnsレベルに収束するかを見ること。",
                        "再現: scripts/llr/llr_pmodel_overlay_horizons_noargs.py → output/llr/out_llr/*_overlay_tof.png",
                    ],
                },
                {
                    "id": "llr_residual",
                    "title": f"{LLR_SHORT_NAME}：残差（観測 - モデル, 定数オフセット整列後）",
                    "kind": "観測（CRD） + 幾何モデル（HORIZONS + SPICE）",
                    "path": root / "output" / "llr" / "out_llr" / f"{llr_stem}_residual.png",
                    "summary_lines": [],
                    "explain_lines": [
                        "観測TOFとモデルTOFの差（残差）。",
                        "定数オフセット整列後の残差で、幾何モデルの変動成分の一致度を見る。",
                        "次の段階：大気遅延・局/反射器座標の一次ソース確定などを加える。",
                    ],
                    "detail_lines": [
                        "残差 = 観測 - モデル（定数オフセット整列後）。0に近いほどモデルが観測の変動を説明している。",
                        "残差の形が周期的なら、未補正の地球自転/章動/大気/潮汐などの系統誤差の可能性が高い。",
                        "バラつき（RMS）でns級に入るかをチェックする（1 nsは片道距離で約0.15 m）。",
                        "残差がゆっくりドリフトする場合、局座標・反射器座標・時間系の定数差が残っている可能性がある。",
                        "残差が鋭いスパイクになる場合、観測データ側の外れ値や、特定の補正（例：大気）が外れている可能性がある。",
                        "この段階では“P-modelの重力時間（Shapiro）”以前に、幾何と座標系の整合が支配的に効く。",
                        "ns精度に近づいた後に、P-model由来の微小効果（δなど）が観測で必要かどうかを検証するのが順序として安全。",
                        "再現: scripts/llr/llr_pmodel_overlay_horizons_noargs.py → output/llr/out_llr/*_residual.png",
                    ],
                },
                {
                    "id": "llr_residual_distribution",
                    "title": f"{LLR_SHORT_NAME}：残差分布（観測 - P-model, inlierのみ）",
                    "kind": "バッチ（ヒストグラム + |残差|累積）",
                    "path": root / "output" / "llr" / "batch" / "llr_residual_distribution.png",
                    "summary_lines": (
                        [
                            f"inlier n={_format_num(llr_outliers_summary.get('n_inliers'))} / total n={_format_num(llr_outliers_summary.get('n_total'))}",
                            f"外れ値 n={_format_num(llr_outliers_summary.get('n_outliers'))}（ゲート後は分布から除外）",
                        ]
                        + (
                            [
                                f"RMS（SR）={_format_num(llr_inlier_rms.get('rms_sr_ns'), digits=4)} ns",
                                f"RMS（SR+Tropo）={_format_num(llr_inlier_rms.get('rms_sr_tropo_ns'), digits=4)} ns",
                                f"RMS（SR+Tropo+Tide）={_format_num(llr_inlier_rms.get('rms_sr_tropo_tide_ns'), digits=4)} ns",
                            ]
                            if llr_inlier_rms.get("rms_sr_ns") is not None
                            else []
                        )
                        if llr_outliers_summary
                        else []
                    ),
                    "explain_lines": [
                        "残差（観測 - モデル）を“点ごと”に集め、全体の散らばりを分布として可視化する。",
                        "左：残差のヒストグラム（符号付き）。右：|残差|の累積分布（小さいほど良い）。",
                        "対流圏・潮汐などを追加したときに、分布が狭くなる（RMSが下がる）かを確認する。",
                    ],
                    "detail_lines": [
                        "この図は「Phase 3 の LLRで、P-modelとの差がどの程度か」を直感的に読むための図。",
                        "定数オフセット整列後の残差を使い、装置遅延などの定数差は吸収した上で、変動成分の一致度を評価する。",
                        "外れ値（ms級）は物理補正では説明できないため、MADベースのゲートで除外し、inlierのみで分布を作る。",
                        "右の累積分布は、例えば『|残差|が5 ns以下の割合』のように“何割がどの精度か”を読み取れる。",
                        "再現: scripts/llr/llr_batch_eval.py → output/llr/batch/llr_residual_distribution.png",
                    ],
                },
                {
                    "id": "llr_residual_compare",
                    "title": f"{LLR_SHORT_NAME}：モデル改善（地球中心 → 観測局 → 反射器）の効果",
                    "kind": "残差比較（定数オフセット整列後）",
                    "path": root / "output" / "llr" / "out_llr" / f"{llr_stem}_residual_compare.png",
                    "summary_lines": [
                        (
                            "例: "
                            + f"{llr_model_metrics.get('station_code') or llr_summary.get('station')}"
                            + " → "
                            + f"{llr_model_metrics.get('target') or llr_summary.get('target')}"
                        ),
                        (
                            "最良モデルの残差RMS="
                            + (
                                f"{_format_num(float(llr_model_metrics.get('rms_residual_station_reflector_ns')), digits=4)} ns"
                                + f"（片道距離で約{_format_num(float(llr_model_metrics.get('rms_residual_station_reflector_ns')) * 0.149896229, digits=3)} m）"
                                if isinstance(llr_model_metrics.get("rms_residual_station_reflector_ns"), (int, float))
                                else "n/a"
                            )
                            + " / 目安: 1 ns（往復）≈0.30 m、片道距離では約0.15 m"
                        ),
                    ],
                    "explain_lines": [
                        "観測局（topocentric）をHORIZONSで扱い、EOP（UT1-UTC/極運動）を自前実装せず整合させる。",
                        "月回転はSPICE（MOON_PA_DE421）を使い、反射器座標のフレームと一致させる。",
                        "次の段階：反射器座標/局座標の一次ソース確定、大気遅延等の追加補正。",
                    ],
                    "detail_lines": [
                        "モデルを段階的に精密化したときの効果を比較する図。",
                        "地球中心（geocenter）→観測局（topocentric）→反射器（reflector）と進めるほど、幾何の誤差が減り残差RMSが下がるのが理想。",
                        "LLRは「観測局」と「月面の固定点」を正しく扱うことが支配的に効く。",
                        "地球中心モデルは“地上局の位置”を無視するため、日周変化（地球自転）の成分が合わず、残差が大きくなりやすい。",
                        "観測局モデルでは、局をtopocentricで扱い、地球自転/EOPの影響を取り込むことで形が大きく改善する。",
                        "反射器モデルでは、月回転（MOON_PA_DE421）で月面固定点を正しく回し、さらに一致度が上がるのが期待。",
                        "この図の目的は“P-model効果”を見る前に、幾何モデルがns精度に届く土台になっているかを確認すること。",
                        "残差RMSが下がらない場合は、座標系の不整合（例：反射器座標のフレーム違い）が最優先で疑われる。",
                        "再現: scripts/llr/llr_pmodel_overlay_horizons_noargs.py → output/llr/out_llr/*_residual_compare.png",
                    ],
                },
                {
                    "id": "llr_batch_improvement",
                    "title": f"{LLR_SHORT_NAME}（バッチ）：モデル改善の効果（全体）",
                    "kind": "複数局×複数反射器（EDC月次）",
                    "path": root / "output" / "llr" / "batch" / "llr_rms_improvement_overall.png",
                    "summary_lines": (
                        [
                            f"入力: {llr_batch_summary.get('n_files')} ファイル / {llr_batch_summary.get('n_points_total')} 点",
                            f"評価: {llr_batch_summary.get('n_groups')} グループ（station×reflector, n≥{llr_batch_summary.get('min_points_per_group')}）",
                            f"中央値RMS（観測局→反射器）={_format_num(llr_batch_summary.get('median_rms_ns', {}).get('station_reflector'), digits=4)} ns",
                        ]
                        if llr_batch_summary
                        else ["バッチ集計が未生成（scripts/llr/llr_batch_eval.py を実行）。"]
                    ),
                    "explain_lines": [
                        "全データをまとめて評価し、幾何モデルを段階的に精密化したときの残差RMSの改善を比較。",
                        "定数オフセットを除いた“変動成分”の一致度（ログスケール）を表示している。",
                    ],
                    "detail_lines": [
                        "複数局×複数反射器のデータをまとめて評価し、モデル改善が統計的に効いているかを見る図。",
                        "グループ（station×reflector）ごとに定数オフセット整列を行い、変動成分の残差RMSを比較する。",
                        "縦軸のRMSはns（ナノ秒）で、距離に直すと 1 ns ≒ 0.15 m（片道）に相当する。",
                        "全データを一括で見ることで、特定の局/反射器だけに効く“偶然の当たり”を避け、再現性を重視する。",
                        "RMSが大きく改善するなら、まず幾何（局・反射器・月回転）の取り込みが正しく効いている可能性が高い。",
                        "改善が小さい場合は、(1)データ品質 (2)補正不足（大気/潮汐）(3)座標系の混在 がボトルネックになりやすい。",
                        "この図は“平均の傾向”なので、個別の外れは station×reflector のヒートマップで切り分ける。",
                        "再現: scripts/llr/fetch_llr_edc_batch.py + scripts/llr/llr_batch_eval.py → output/llr/batch/*.png",
                    ],
                },
                {
                    "id": "llr_batch_station_target",
                    "title": f"{LLR_SHORT_NAME}（バッチ）：残差RMS（観測局→反射器） station×reflector",
                    "kind": "ヒートマップ（定数オフセット整列後）",
                    "path": root / "output" / "llr" / "batch" / "llr_rms_by_station_target.png",
                    "summary_lines": [],
                    "explain_lines": [
                        "観測局（station）と反射器（reflector）の組み合わせごとの残差RMSを一覧化。",
                        "局・反射器・データ期間の違いで残差の傾向が変わるかを俯瞰する。",
                    ],
                    "detail_lines": [
                        "局×反射器の組み合わせごとの残差RMSをヒートマップ化し、どこが難しいかを俯瞰する。",
                        "特定の局・反射器だけ悪い場合、局座標/ログ情報/データ品質の問題を切り分けやすい。",
                        "縦（station）と横（reflector）のセルが小さいほど、その組み合わせでモデルが観測に合っている。",
                        "同じ反射器でも局によって難易度が変わる場合、局の高度角分布や大気条件、局座標の誤差が疑わしい。",
                        "同じ局で反射器によって差が出る場合、反射器座標や月回転モデルの整合が疑わしい。",
                        "外れのセルを優先して個別に時系列（残差）を見て、原因（系統か外れ値か）を判断するのが効率的。",
                        "再現: scripts/llr/llr_batch_eval.py → output/llr/batch/llr_rms_by_station_target.png",
                    ],
                },
                {
                    "id": "llr_batch_station_month",
                    "title": f"{LLR_SHORT_NAME}（バッチ）：残差RMS（局別・月別）",
                    "kind": "バッチ（期間依存の診断）",
                    "path": root / "output" / "llr" / "batch" / "llr_rms_by_station_month.png",
                    "summary_lines": [
                        "局ごとの残差RMSを月別に集計し、期間依存（データ品質/運用差）を可視化。",
                    ],
                    "explain_lines": [
                        "同じ局でも月（期間）によって残差RMSが変動することがある。",
                        "スパイクがある月は、観測の品質・解析前処理・局ログの整合を疑う。",
                    ],
                    "detail_lines": [
                        "LLRの残差は『物理モデルの不足』だけでなく、『観測データの品質』『局の装置・運用』『ログ情報（座標/遅延）の更新』に強く依存する。",
                        "月別にRMSを集計すると、モデルが一定でも“その月だけ悪い”という現象が見える（原因切り分けの入口）。",
                        "典型的な原因: 局座標/時刻系の更新、装置遅延の変更、天候（大気）条件、観測の仰角分布の偏り、データ編集の方針差。",
                        "この図を使って『悪い期間』を特定し、該当期間の site log（slrlog）や運用記録を一次ソースで確認する。",
                        "再現: scripts/llr/llr_batch_eval.py → output/llr/batch/llr_rms_by_station_month.png",
                    ],
                },
                {
                    "id": "llr_batch_station_month_models",
                    "title": f"{LLR_SHORT_NAME}（バッチ）：残差RMS（局別・月別）モデル切り分け",
                    "kind": "バッチ（SR / +対流圏 / +潮汐 の比較）",
                    "path": root / "output" / "llr" / "batch" / "llr_rms_by_station_month_models.png",
                    "summary_lines": [
                        "同じ月別集計を、モデル項（対流圏/潮汐）のON/OFFで比較して原因を切り分ける。",
                    ],
                    "explain_lines": [
                        "SR→SR+Tropoで改善するなら、大気（仰角依存）の未補正が支配的。",
                        "SR+Tropo→SR+Tropo+Tideで改善するなら、潮汐（局/反射器の動的補正）が効いている可能性。",
                    ],
                    "detail_lines": [
                        "Step2/3/4（原因切り分け）のための診断図。",
                        "同じデータに対して、モデル項を段階的に追加したときの『期間依存のRMS』がどう変わるかを見る。",
                        "SR: 観測局→反射器（SPICE月回転 + 太陽Shapiro）の基本モデル（定数オフセット整列後）。",
                        "SR+Tropo: CRD record 20（気象: 気圧/気温/湿度）を優先し、Saastamoinen（ZHD/ZWD）+ Niell mapping function（NMF）で対流圏遅延を加える。",
                        "SR+Tropo+Tide: 固体地球潮汐（Moon+Sun, 水平成分を含む簡易）＋月体潮汐（Earth, 簡易）を追加。",
                        "月ごとのスパイクが『どの追加項で消える/残るか』で、原因（大気/潮汐/データ品質）を絞り込める。",
                        "再現: scripts/llr/llr_batch_eval.py → output/llr/batch/llr_rms_by_station_month_models.png",
                    ],
                },
                {
                    "id": "llr_batch_grsm_target_month",
                    "title": f"{LLR_SHORT_NAME}：GRSM 残差RMS（反射器別の期間依存）",
                    "kind": "バッチ（局の深掘り）",
                    "path": root / "output" / "llr" / "batch" / "llr_grsm_rms_by_target_month.png",
                    "summary_lines": [
                        "残差が大きくなりやすい局の例（GRSM）を、反射器別×月別に分解して可視化。",
                    ],
                    "explain_lines": [
                        "局が同じでも反射器や期間でRMSが変わるなら、観測の仰角分布やデータ品質の影響が疑わしい。",
                        "反射器間で同時に悪化する月は、局側（時刻系/座標/装置）の問題を疑う。",
                    ],
                    "detail_lines": [
                        "この図は Step2（残差が大きい局の原因切り分け）のための診断図。",
                        "反射器別に分解することで『特定反射器だけ悪い』のか『全反射器で同時に悪い』のかが分かる。",
                        "前者なら反射器座標系・月回転・観測幾何（仰角）依存など、後者なら局の座標/時計/装置遅延/気象条件などが疑わしい。",
                        "ns級では troposphere（対流圏）や潮汐、EOPなどの補正も効き始めるため、追加項のON/OFFで系統残差を狙い撃ちする。",
                        "再現: scripts/llr/llr_batch_eval.py → output/llr/batch/llr_grsm_rms_by_target_month.png",
                    ],
                },
                {
                    "id": "llr_batch_grsm_month_models",
                    "title": f"{LLR_SHORT_NAME}：GRSM 残差RMS（モデル切り分け）",
                    "kind": "バッチ（原因切り分けの要約）",
                    "path": root / "output" / "llr" / "batch" / "llr_grsm_rms_by_month_models.png",
                    "summary_lines": [
                        "GRSMの月別RMSを、SR / SR+Tropo / SR+Tropo+Tide で比較。",
                    ],
                    "explain_lines": [
                        "GRSMで残差が大きい月が、対流圏や潮汐で改善するかを見る。",
                        "改善しない月は、局ログ/観測品質（外れ値・編集方針）の影響が強い可能性。",
                    ],
                    "detail_lines": [
                        "反射器別の図（上）より情報量は減るが、『原因が大気寄りか/潮汐寄りか/データ品質寄りか』を直感的に把握できる。",
                        "SR→SR+Tropoで大きく改善する月は、仰角分布や気象（record 20）の寄与が残差を支配している可能性が高い。",
                        "SR+Tropo→SR+Tropo+Tideで改善する月は、動的な地球/月の変形（潮汐）が効いている可能性がある。",
                        "いずれでも改善しない月は、局座標（site log）や装置遅延の更新、あるいは外れ値の混入を疑う。",
                        "再現: scripts/llr/llr_batch_eval.py → output/llr/batch/llr_grsm_rms_by_month_models.png",
                    ],
                },
                {
                    "id": "llr_batch_outliers",
                    "title": f"{LLR_SHORT_NAME}（バッチ）：外れ値（スパイク）の一覧と影響",
                    "kind": "品質診断（外れ値）",
                    "path": root / "output" / "llr" / "batch" / "llr_outliers_overview.png",
                    "summary_lines": (
                        [
                            f"外れ値 n={_format_num(llr_outliers_summary.get('n_outliers'))} / 使用点 n={_format_num((llr_outliers_summary.get('n_inliers') or 0) + (llr_outliers_summary.get('n_outliers') or 0))}",
                            f"最大 |Δ|（中心化）={_format_num(llr_outliers_summary.get('max_abs_delta_centered_ns'), digits=4)} ns",
                            (
                                "ゲート条件: max(clip_ns, clip_sigma*MAD), "
                                + f"clip_sigma={_format_num((llr_outliers_summary.get('clip') or {}).get('sigma'))}, "
                                + f"clip_ns={_format_num((llr_outliers_summary.get('clip') or {}).get('min_ns'))} ns"
                            ),
                            "一覧: output/llr/batch/llr_outliers.csv",
                        ]
                        if llr_outliers_summary
                        else ["外れ値要約（output/llr/batch/llr_outliers_summary.json）が見つかりません。"]
                    ),
                    "explain_lines": [
                        "外れ値（スパイク）が混ざるとRMSが破壊されるため、station×reflectorごとにMADベースで外れ値をゲートしている。",
                        "この図は除外された点がいつ/どこで起きたか（時刻・仰角・局・月）を可視化し、原因切り分けの入口にする。",
                    ],
                    "detail_lines": [
                        "Δは (観測TOF - モデルTOF) を反射器ごとの中央値で中心化した値（定数オフセット除去後のズレ）。",
                        "ns級（数m以下）なら大気/潮汐/EOP等の未補正が疑わしいが、ms級（数百km相当）のΔは物理補正では説明できない。",
                        "その場合は観測データ（一次ソース）側の異常値・編集・ログ整合を疑い、表の出典（file:line）から原データを確認する。",
                        "再現: scripts/llr/llr_batch_eval.py → output/llr/batch/llr_outliers_overview.png / llr_outliers.csv",
                    ],
                    "table": llr_outliers_table,
                },
                {
                    "id": "llr_batch_outliers_diagnosis",
                    "title": f"{LLR_SHORT_NAME}：外れ値の原因切り分け（time-tag 感度）",
                    "kind": "品質診断（外れ値）",
                    "path": root / "output" / "llr" / "batch" / "llr_outliers_time_tag_sensitivity.png",
                    "summary_lines": _llr_outliers_diag_summary_lines(),
                    "explain_lines": [
                        "外れ値ごとに time-tag（tx/rx/mid）の仮定を切り替えてモデルTOFを再計算し、|Δ|（中心化）の大小を比較する図。",
                        "棒が低い mode ほど、その仮定で観測とモデルが近い（= time-tag解釈ミスの可能性を示唆）。",
                    ],
                    "detail_lines": [
                        "この図は、外れ値の“原因切り分け”の補助であり、ここだけで物理的結論は出さない。",
                        "best!=current が多い場合は、局ごとの time-tag 混在やログ整合の問題が疑わしい。",
                        "どの mode でも ms級の |Δ| が残る場合は、一次データ側の異常（記録混入/編集/TOF値の破綻）を優先して疑う。",
                        "再現: scripts/llr/llr_batch_eval.py → output/llr/batch/llr_outliers_time_tag_sensitivity.png / llr_outliers_diagnosis.csv",
                    ],
                    "table": llr_outliers_diag_table,
                },
                {
                    "id": "llr_batch_outliers_target_mixing",
                    "title": f"{LLR_SHORT_NAME}：外れ値の原因切り分け（ターゲット混入感度）",
                    "kind": "品質診断（外れ値）",
                    "path": root / "output" / "llr" / "batch" / "llr_outliers_target_mixing_sensitivity.png",
                    "summary_lines": _llr_outliers_target_mixing_summary_lines(),
                    "explain_lines": [
                        "外れ値ごとに「ターゲット（反射器）ラベル」を入れ替えてモデルTOFを再計算し、|Δ_raw|（観測-モデル）が劇的に減るかを調べる図。",
                        "巨大スパイクが別反射器でns〜μs級へ落ちる場合、観測側の「反射器取り違え/混入」の可能性が高い。",
                    ],
                    "detail_lines": [
                        "この図は診断目的であり、疑いがあっても観測データのターゲットラベルは自動では書き換えない（統計からは除外）。",
                        "疑いの点は `output/llr/batch/llr_outliers_diagnosis.csv` の source_file:lineno から一次データ行へ追跡し、手で確認する。",
                        "再現: scripts/llr/llr_batch_eval.py → output/llr/batch/llr_outliers_target_mixing_sensitivity.png / llr_outliers_diagnosis.csv",
                    ],
                },
                *_llr_residual_vs_elevation_cards(root),
                {
                    "id": "llr_station_coord_delta_pos_eop",
                    "title": f"{LLR_SHORT_NAME}：局座標差分（pos+eop vs site log）",
                    "kind": "一次ソース確認（局座標）",
                    "path": root / "output" / "llr" / "batch" / "llr_station_coord_delta_pos_eop.png",
                    "summary_lines": [
                        "EDCの一次ソース（pos+eop/SINEX）で得られる局座標と、ILRS site log の座標の差を可視化。",
                    ],
                    "explain_lines": [
                        "局座標が数十cmずれると、往復TOFでns級の残差（=数十cm〜m級の距離差）に繋がりうる。",
                        "ns級を目指す段階では、局座標の出典（ITRF/EOP）を一次ソースに固定する価値が高い。",
                    ],
                    "detail_lines": [
                        "縦軸は ||Δr||（pos+eop - site log）[m]。",
                        "差が大きい局は、site log の基準点（monument / o.c.）や更新時期の違い、座標系の扱い（永久潮の有無等）を疑う。",
                        "本リポジトリでは、EDCの pos+eop（SINEX）を“局座標の一次ソース”としてキャッシュし、LLRモデル側で優先的に使用する。",
                        "取得: scripts/llr/fetch_pos_eop_edc.py → data/llr/pos_eop/snx/<YYYY>/<YYMMDD>/pos_eop_<YYMMDD>.snx.gz",
                        "再現: scripts/llr/llr_batch_eval.py → output/llr/batch/llr_station_coord_delta_pos_eop.png",
                    ],
                },
                {
                    "id": "llr_station_coords_rms_compare",
                    "title": f"{LLR_SHORT_NAME}：局座標ソース比較（RMS）",
                    "kind": "原因切り分け（局座標の影響を定量化）",
                    "path": root / "output" / "llr" / "coord_compare" / "llr_station_coords_rms_compare.png",
                    "summary_lines": [
                        "同じ観測（station×target）に対して、局座標ソースを変えたときの残差RMSの変化を比較。",
                        "左：散布図（slrlog vs pos+eop/auto）、右：局別の中央値RMS。",
                    ],
                    "explain_lines": [
                        "pos+eop（SINEX, EDC一次ソース）に寄せることで、局座標が原因の系統残差が減るかを確認できる。",
                        "散布図で y<x に偏れば、pos+eop/auto の方がRMSが小さくなっている（改善）。",
                    ],
                    "detail_lines": [
                        "この比較は、モデル式そのものを変えずに（P-modelのβは固定）、入力側（局座標）の品質差だけを見たいときに有効。",
                        "再現: scripts/llr/llr_batch_eval.py（--out-dir を分けて2回）→ scripts/llr/llr_station_coords_compare.py",
                    ],
                },
                {
                    "id": "llr_grsm_monthly_rms_station_coords",
                    "title": f"{LLR_SHORT_NAME}：GRSM 月別RMS（局座標ソース比較）",
                    "kind": "原因切り分け（GRSM・期間依存）",
                    "path": root / "output" / "llr" / "coord_compare" / "llr_grsm_monthly_rms_pos_eop_vs_slrlog.png",
                    "summary_lines": [
                        "GRSMの月別RMSを、局座標ソース（slrlog vs pos+eop/auto）で比較。",
                        "月内 n>=30 のみ表示（サンプル不足による見かけのスパイクを回避）。",
                    ],
                    "explain_lines": [
                        "期間依存の残差が「座標ソースの選び方」で改善するなら、局ログの更新時期/基準点/座標系の差が主因の可能性が高い。",
                        "改善しない月は、time-tag混在、外れ値、観測品質（低仰角）など他要因を疑う。",
                    ],
                    "detail_lines": [
                        "再現: scripts/llr/llr_station_coords_compare.py → output/llr/coord_compare/llr_grsm_monthly_rms_pos_eop_vs_slrlog.png",
                    ],
                },
                {
                    "id": "llr_batch_ablations",
                    "title": f"{LLR_SHORT_NAME}（バッチ）：切り分け（Shapiro / 対流圏 / 月回転モデル）",
                    "kind": "反射器モデルの要素別比較（全体中央値）",
                    "path": root / "output" / "llr" / "batch" / "llr_rms_ablations_overall.png",
                    "summary_lines": (
                        [
                            f"中央値RMS（SPICE+Shapiro）={_format_num(llr_batch_summary.get('median_rms_ns', {}).get('station_reflector'), digits=4)} ns",
                            f"中央値RMS（+対流圏）={_format_num(llr_batch_summary.get('median_rms_ns', {}).get('station_reflector_tropo'), digits=4)} ns",
                            f"中央値RMS（+対流圏+潮汐(固体+月)）={_format_num(llr_batch_summary.get('median_rms_ns', {}).get('station_reflector_tropo_tide'), digits=4)} ns",
                            f"中央値RMS（Shapiro OFF）={_format_num(llr_batch_summary.get('median_rms_ns', {}).get('station_reflector_no_shapiro'), digits=4)} ns",
                            f"中央値RMS（IAU近似）={_format_num(llr_batch_summary.get('median_rms_ns', {}).get('station_reflector_iau'), digits=4)} ns",
                        ]
                        if llr_batch_summary
                        else []
                    ),
                    "explain_lines": [
                        "太陽ShapiroのON/OFF、対流圏（Saastamoinen+Niell）の有無、および月回転の IAU近似 vs SPICE(MOON_PA_DE421) の影響を比較。",
                        "月固定点（反射器）の扱いでは、月回転モデルの精度が支配的に効くことが分かる。",
                    ],
                    "detail_lines": [
                        "太陽Shapiro（ON/OFF）、対流圏、月回転モデル（IAU近似 vs SPICE）の影響を比較する切り分け図。",
                        "LLRでは月回転モデル（反射器の向き・位置）が支配的に効くことが多く、Shapiroは補正として上乗せされる。",
                        "“月回転モデル”を粗くすると反射器位置がずれ、TOFの変動が合わなくなり残差RMSが増えやすい。",
                        "“対流圏”は局の高度角によって変化し、局依存の系統残差を減らせる可能性がある（すでに観測で補正済みなら二重計上になる）。",
                        "この実装では CRD record 20（気象: 気圧/気温/湿度）を優先してSaastamoinenでZHD/ZWDを計算し、Niell mapping function（NMF）で仰角方向に写像する（無い場合は標準大気で補う）。",
                        "“潮汐(固体+月)”は固体地球潮（Moon+Sun, 水平成分含む簡易）と月体潮（Earth, 簡易）を追加したもの。",
                        "観測側で補正済みなら効果は出ず、二重計上で悪化する可能性があるため、ON/OFFで寄与を確認する。",
                        "“Shapiro OFF”との差が小さければ、そのデータ/期間では太陽Shapiroは二次的で、他が支配的という示唆になる。",
                        "逆にShapiro差が大きいなら、ns精度の段階で相対論補正が効いている（モデルの追加価値が出やすい）。",
                        "ここでの目的は、P-modelの検証以前に「どの補正が残差を支配しているか」を定量的に掴むこと。",
                        "再現: scripts/llr/llr_batch_eval.py → output/llr/batch/llr_rms_ablations_overall.png",
                    ],
                },
                {
                    "id": "llr_batch_shapiro",
                    "title": f"{LLR_SHORT_NAME}（バッチ）：重力遅延の寄与（Shapiroの拡張）",
                    "kind": "対流圏ありで Shapiro だけ比較",
                    "path": root / "output" / "llr" / "batch" / "llr_shapiro_ablations_overall.png",
                    "summary_lines": (
                        [
                            f"中央値RMS（対流圏のみ）={_format_num(llr_batch_summary.get('median_rms_ns', {}).get('station_reflector_tropo_no_shapiro'), digits=4)} ns",
                            f"中央値RMS（対流圏+太陽Shapiro）={_format_num(llr_batch_summary.get('median_rms_ns', {}).get('station_reflector_tropo'), digits=4)} ns",
                            f"中央値RMS（対流圏+太陽+地球Shapiro）={_format_num(llr_batch_summary.get('median_rms_ns', {}).get('station_reflector_tropo_earth_shapiro'), digits=4)} ns",
                        ]
                        if llr_batch_summary
                        else []
                    ),
                    "explain_lines": [
                        "同じ対流圏モデルの上で、Shapiro（太陽/地球）のON/OFFを比較して寄与を定量化。",
                        "地球Shapiroは規模が小さいため、ns級の段階でも差が小さい可能性が高い。",
                    ],
                    "detail_lines": [
                        "この図は「相対論補正（Shapiro遅延）の差」だけを見たいので、対流圏（Saastamoinen+Niell）を共通にONにして比較している。",
                        "“対流圏のみ”と“対流圏+太陽Shapiro”の差が、太陽Shapiroが残差にどれだけ効くかの目安になる。",
                        "“+地球Shapiro”はさらに小さい補正で、理論上は追加できるが、現状の誤差源（大気・データ品質等）に埋もれやすい。",
                        "目的は「追加した補正の寄与が、統計的に有意に残差RMSを減らすか」を段階的に確認すること。",
                        "再現: scripts/llr/llr_batch_eval.py → output/llr/batch/llr_shapiro_ablations_overall.png",
                    ],
                },
                {
                    "id": "llr_batch_tide",
                    "title": f"{LLR_SHORT_NAME}（バッチ）：潮汐の寄与（固体潮汐/海洋荷重/月体潮汐）",
                    "kind": "対流圏ありで 潮汐 のON/OFF比較",
                    "path": root / "output" / "llr" / "batch" / "llr_tide_ablations_overall.png",
                    "summary_lines": (
                        [
                            f"中央値RMS（対流圏のみ）={_format_num(llr_batch_summary.get('median_rms_ns', {}).get('station_reflector_tropo'), digits=4)} ns",
                            f"中央値RMS（+観測局潮汐）={_format_num(llr_batch_summary.get('median_rms_ns', {}).get('station_reflector_tropo_station_tide'), digits=4)} ns",
                            f"中央値RMS（+月体潮汐）={_format_num(llr_batch_summary.get('median_rms_ns', {}).get('station_reflector_tropo_moon_tide'), digits=4)} ns",
                            f"中央値RMS（+潮汐=固体+月）={_format_num(llr_batch_summary.get('median_rms_ns', {}).get('station_reflector_tropo_tide_no_ocean'), digits=4)} ns",
                            f"中央値RMS（+潮汐=固体+海洋+月）={_format_num(llr_batch_summary.get('median_rms_ns', {}).get('station_reflector_tropo_tide'), digits=4)} ns",
                        ]
                        if llr_batch_summary
                        else []
                    ),
                    "explain_lines": [
                        "同じ対流圏モデルの上で、潮汐（固体地球潮汐/海洋荷重/月体潮汐）の寄与を段階的に比較して定量化。",
                        "“固体+月”と“固体+海洋+月”の差分が、海洋荷重（TOC）がどれだけ効くかの目安になる。",
                    ],
                    "detail_lines": [
                        "この図は『潮汐補正』を2つに分けて比較する：",
                        "  - 観測局潮汐（固体地球潮汐）：Moon+Sun による観測局座標の変位（簡易）。",
                        "  - 月体潮汐（反射器側）：Earth による反射器位置の変位（簡易）。",
                        "さらに、観測局側には『海洋荷重（TOC: tidal ocean loading）』による変位があり、ns級では無視できない場合がある。",
                        "海洋荷重は IMLS（International Mass Loading Service）の HARPOS（調和定数）から、UTC→TT変換の上で UEN 変位を合成している。",
                        "“+潮汐=固体+月”と“+潮汐=固体+海洋+月”の差が、海洋荷重の追加で残差RMSがどれだけ動くかを示す。",
                        "ns級で“改善が出ない/悪化する”場合、観測側で既に補正済みで二重計上になっている可能性もあるため注意（CRDの適用補正一覧と整合を取る）。",
                        "再現: scripts/llr/llr_batch_eval.py → output/llr/batch/llr_tide_ablations_overall.png",
                    ],
                },
            ],
        ),
        (
            "惑星軌道（Mercury）",
            [
                {
                    "id": "mercury_orbit",
                    "title": "水星の近日点移動（誇張表示）と線形性",
                    "kind": "数値シミュレーション",
                    "path": root / "output" / "mercury" / "mercury_orbit.png",
                    "summary_lines": mercury_summary_lines,
                    "explain_lines": [
                        "左：ニュートン軌道とP-model軌道の違い（誇張表示で近日点がずれていくのが見える）。",
                        "右：実Cでの近日点移動（角秒）が周回ごとにほぼ線形に増えることを確認。",
                    ],
                    "detail_lines": [
                        "近日点移動（歳差）は、ニュートン力学だけでは説明できない補正が現れる代表的な検証項目。",
                        "左図は軌道を重ねて、近日点が少しずつ回転していく様子を可視化している（見やすさのため c を小さくして効果を誇張）。",
                        "右図は実際の光速 c（実C）で計算した近日点角の累積で、ほぼ直線なら「周回あたり一定の歳差」になっていることを示す。",
                        "実Cでの推定値（角秒/世紀）は、図中の注記と `output/mercury/mercury_precession_metrics.json` に保存される。",
                        "水星の観測値（約43 角秒/世紀）は、他惑星摂動などを差し引いた“残差”として議論されるため、厳密比較には多体摂動を入れる必要がある。",
                        "本図の役割はまず「P-modelの力学で、ニュートンには無い歳差が自然に出るか」を定性的に確認すること。",
                        "補助出力：周回ごとの累積値は `output/mercury/mercury_perihelion_shifts.csv` に保存する（再集計/可視化に利用可能）。",
                        "再現: scripts/mercury/mercury_precession_v3.py → output/mercury/mercury_orbit.png",
                    ],
                },
            ],
        ),
        (
            "強重力（ブラックホール/EHT）",
            [
                {
                    "id": "eht_shadow_compare",
                    "title": "EHT：観測リング直径はモデルシャドウ予測の何倍？（κ）",
                    "kind": "強重力テスト（公開値）",
                    "path": root / "output" / "eht" / "eht_kappa_fit_public.png",
                    "summary_lines": eht_summary_lines,
                    "explain_lines": [
                        "EHTはブラックホール近傍のリング状構造の角直径（µas）を推定している。",
                        "ここでは、観測リング直径 θ_ring とモデルのシャドウ直径 θ_shadow の比 κ=θ_ring/θ_shadow を可視化する。",
                        "κ が 1 付近なら、モデルのシャドウ予測が観測リングのスケールと整合している目安になる（放射/散乱/スピン等の系統を含む）。",
                    ],
                    "detail_lines": [
                        "P-modelの最小モデルでは、点質量解 P/P0=exp(GM/(c^2 r)) と、光の屈折率 n=(P/P0)^(2β) を仮定する。",
                        "球対称屈折率では b=n(r)r が保存され、b(r) の最小が捕獲境界（影の縁）に対応する。",
                        "この仮定からシャドウ直径は θ_shadow = 4 e β (GM/(c^2 D)) になる（詳細は doc/eht/README.md）。",
                        "標準理論（Schwarzschild近似）では係数が 2√27 ≈ 10.392、P-model（β=1）の係数 4e ≈ 10.873（数%差）。",
                        "EHTのリング直径は放射モデル/散乱/スピン等に依存するため、厳密比較には後段で精密化が必要。",
                        "再現: scripts/eht/eht_shadow_compare.py → output/eht/eht_kappa_fit_public.png",
                    ],
                }
                ,
                {
                    "id": "eht_m87_ring_diameter_multi_epoch",
                    "title": "EHT（M87*）：リング直径のmulti-epoch整合（2017/2018）",
                    "kind": "頑健性チェック（公開値）",
                    "path": root / "output" / "eht" / "eht_m87_persistent_shadow_ring_diameter.png",
                    "summary_lines": eht_m87_multi_epoch_summary_lines,
                    "explain_lines": [
                        "M87* は 2017 と 2018 の独立epochでリング直径が報告されている。",
                        "epoch 間の差が統計誤差の範囲に収まるかを確認し、入力（θ_ring）の頑健性をチェックする。",
                    ],
                    "detail_lines": [
                        "この図は『P-model vs GR の判別』ではなく、EHTの入力値（リング直径）が観測epochに依存して大きく揺れないかの確認。",
                        "差分は Δθ_ring=θ_2018−θ_2017 とし、単純化のため非対称誤差は平均（avg_sym）または保守的最大（max_sym）で対称化して z を算出する。",
                        "multi-epoch で整合していれば、以降の κ/シャドウ比較で『観測入力が揺れているだけ』という反論を弱められる。",
                        "再現: scripts/eht/eht_m87_persistent_shadow_metrics.py → output/eht/eht_m87_persistent_shadow_ring_diameter.png",
                    ],
                },
                {
                    "id": "eht_shadow_zscores",
                    "title": "EHT：観測とモデルのずれ（zスコア, κ=1 の仮定）",
                    "kind": "強重力テスト（整合性）",
                    "path": root / "output" / "eht" / "eht_shadow_zscores_public.png",
                    "summary_lines": eht_zscore_summary_lines,
                    "explain_lines": [
                        "各モデルの予測と観測の差を、合成誤差で規格化した zスコアで表示（|z|が小さいほど一致）。",
                        "ここではリング直径≒シャドウ直径（κ=1）を仮定しているため、κの系統誤差は別図で扱う。",
                    ],
                    "detail_lines": [
                        "定義：z=(観測-予測)/σ_total、σ_total=√(σ_obs^2+σ_pred^2)。",
                        "点質量+最小延長の近似に基づく「整合性チェック」として、P-modelとGR（Schwarzschild）が観測レンジ内に入るかを俯瞰する。",
                        "|z|?1 は“概ね一致”、|z|?2 は“ややズレ”、|z|?3 は“有意なズレ”の目安（ただし κ の系統誤差が支配する可能性あり）。",
                        "κ（リング/シャドウ）とKerrスピン依存は `output/eht/eht_shadow_systematics_public.png` を参照。",
                        "再現: scripts/eht/eht_shadow_compare.py → output/eht/eht_shadow_zscores_public.png",
                    ],
                },
                {
                    "id": "eht_shadow_systematics",
                    "title": "EHT：系統誤差（κ）とGRスピン依存（参考）",
                    "kind": "強重力テスト（不確かさ/感度）",
                    "path": root / "output" / "eht" / "eht_shadow_systematics_public.png",
                    "summary_lines": eht_sys_summary_lines,
                    "explain_lines": [
                        "EHTの公表値は『リング直径』であり、『シャドウ直径』とは必ずしも同一ではない（κが必要）。",
                        "GR側もKerr（スピン）で形が歪むため、数%差の判定には系統誤差の管理が重要。",
                    ],
                    "detail_lines": [
                        "左図：P-model係数（4eβ）とGR係数（Schwarzschild）を比較し、参考としてKerrシャドウの等価直径係数レンジ（spin/inc + 定義依存の envelope）を陰影で表示。",
                        "右図：リング直径 ≈ κ×シャドウ直径 とおき、観測からκを逆算した値（P-model β固定/GR Schwarzschild）を表示。",
                        "κ（放射モデル/散乱/スピン起因）が数%あると、P-model vs GR の数%差と同程度になり、判定が難しくなる可能性がある。",
                        "将来的には放射モデル（GRMHD等）/散乱/スピンの同時推定でκの系統誤差を詰める必要がある。",
                        "再現: scripts/eht/eht_shadow_compare.py → output/eht/eht_shadow_systematics_public.png",
                    ],
                },
                {
                    "id": "eht_kerr_shadow_coeff_grid",
                    "title": "EHT：Kerr shadow 係数の (a*,inc) 依存（reference systematic）",
                    "kind": "強重力テスト（不確かさ/感度）",
                    "path": root / "output" / "eht" / "eht_kerr_shadow_coeff_grid_public.png",
                    "summary_lines": eht_kerr_grid_summary_lines,
                    "explain_lines": [
                        "Kerr（回転BH）では、シャドウは非円形になり、等価直径係数がスピンa*と視線角incに依存する。",
                        "ここでは『GRジオデシック由来の reference systematic』として、(a*,inc) グリッドで係数の変化を可視化する。",
                    ],
                    "detail_lines": [
                        "上段：Schwarzschild（a*=0）の係数 2√27 からの変化率。",
                        "下段：P-model係数（4eβ）と Kerr 係数の差（%）。",
                        "M87*/Sgr A* は一次ソース由来の inc 制約を矩形で表示（スピンは広い範囲）。",
                        "再現: scripts/eht/eht_kerr_shadow_coeff_grid.py → output/eht/eht_kerr_shadow_coeff_grid_public.png",
                    ],
                },
                {
                    "id": "eht_kerr_shadow_definition_sensitivity",
                    "title": "EHT：Kerr shadow 係数の定義依存（effective diameter）感度",
                    "kind": "強重力テスト（不確かさ/感度）",
                    "path": root / "output" / "eht" / "eht_kerr_shadow_coeff_definition_sensitivity_public.png",
                    "summary_lines": eht_kerr_def_sens_summary_lines,
                    "explain_lines": [
                        "Kerr shadow は非円形であり、『単一の直径』への写像は定義依存になる。",
                        "ここでは複数定義で係数レンジを比較し、定義依存を系統誤差として扱う方針を固定する。",
                    ],
                    "detail_lines": [
                        "定義依存は spin/inc と別の系統として扱い得るが、保守的に definition envelope として吸収する（κ予算）。",
                        "eht_shadow_compare は κ の Kerr 系統を envelope(across methods) で計算し、spin/incのみ（avg(width,height)）は *_spin_only として併記。",
                        "再現: scripts/eht/eht_kerr_shadow_coeff_definition_sensitivity.py → output/eht/eht_kerr_shadow_coeff_definition_sensitivity_public.png",
                    ],
                },
                {
                    "id": "eht_kappa_precision_required",
                    "title": "EHT：3σ判別に必要な κ 精度（相対不確かさの目安）",
                    "kind": "強重力テスト（必要精度）",
                    "path": root / "output" / "eht" / "eht_kappa_precision_required_public.png",
                    "summary_lines": eht_kappa_precision_summary_lines,
                    "explain_lines": [
                        "P-model vs GR の数%差を3σで判別するには、リング→シャドウ変換（κ）の不確かさを詰める必要がある。",
                        "図は κ の相対不確かさ（1σ）の目安と、どの誤差源が支配的かの指針を示す。",
                    ],
                    "detail_lines": [
                        "棒: 3σ判別に必要な κ 精度（best-case: ringσ→0）。点: 現状のリング直径相対誤差（1σ）。",
                        "菱形: 参考として、Kerrスピン/傾斜レンジ由来の κ 系統（概算, 1σ）。",
                        "M87*は質量/距離の不確かさ（θ_unit）が支配的で、κ以前にθ_unitの精度改善が必要になる場合がある。",
                        "再現: scripts/eht/eht_shadow_compare.py → output/eht/eht_kappa_precision_required_public.png",
                    ],
                },
                {
                    "id": "eht_kappa_tradeoff",
                    "title": "EHT：ring σ と κσ のトレードオフ（3σ判別の許容域）",
                    "kind": "強重力テスト（必要精度）",
                    "path": root / "output" / "eht" / "eht_kappa_tradeoff_public.png",
                    "summary_lines": eht_kappa_tradeoff_summary_lines,
                    "explain_lines": [
                        "3σ判別に向けて、リング直径の統計誤差（ring σ）と、リング→シャドウ変換の系統（κσ）の“どちらをどれだけ詰める必要があるか”を可視化する。",
                        "青領域に入る改善（ring σ↓とκσ↓）ができれば、係数差（約4.6%）の判別が視野に入る。",
                    ],
                    "detail_lines": [
                        "左（M87*）：まず質量/距離の不確かさ（θ_unit）が支配的になるため、ring/κ以前にM/Dの改善が必要になる場合がある。",
                        "右（Sgr A*）：θ_unitは概ね要求を満たす一方、現状は ring σ が大きく、κ系統も数%級になり得るため、両方の改善が必要。",
                        "再現: scripts/eht/eht_shadow_compare.py → output/eht/eht_kappa_tradeoff_public.png",
                    ],
                },
                {
                    "id": "eht_delta_precision_required",
                    "title": "EHT：δ（Schwarzschild shadow deviation）の必要精度（参考）",
                    "kind": "強重力テスト（必要精度/参考）",
                    "path": root / "output" / "eht" / "eht_delta_precision_required_public.png",
                    "summary_lines": eht_delta_precision_summary_lines,
                    "explain_lines": [
                        "EHT論文側の δ は“GRの解析枠組みで定義された派生量”であり、放射/散乱/再構成などのモデル依存を含み得る（参考指標）。",
                        "ここでは、係数差（約4.6%）を3σで判別するために必要な δ 精度の“目安”を示す。",
                    ],
                    "detail_lines": [
                        "VLTI/Keck の δ 制約は一次ソース表記（既存のGR整合チェック）として扱う。",
                        "Kerrレンジ由来の δ 系統（参考）も併記し、影の非円形性が δ 解釈へ与えるスケール感を示す。",
                        "再現: scripts/eht/eht_shadow_compare.py → output/eht/eht_delta_precision_required_public.png",
                    ],
                },
                {
                    "id": "eht_sgra_paper5_m3_nir_reconnection_conditions",
                    "title": "EHT（Sgr A*）：Paper V near-passing と M3/2.2 μm 制約の救済条件",
                    "kind": "強重力テスト（判定基準/系統）",
                    "path": root / "output" / "eht" / "eht_sgra_paper5_m3_nir_reconnection_conditions.png",
                    "summary_lines": eht_paper5_m3_rescue_summary_lines,
                    "explain_lines": [
                        "Sgr A* Paper V では、モデルが『ほぼ通る』near-passing ケースが多く、最後に落ちる制約がどれかを整理できる。",
                        "ここでは特に、mm の M3 と 2.2 μm flux（F_2um）が残り制約になりやすいこと、",
                        "さらに KS 検定が閾値付近にあり、離散性/digitize 系統で判定が動く条件を明文化する。",
                    ],
                    "detail_lines": [
                        "判定基準案：α=0.01 で『棄却/保留』を決めるには、mi3 の系統誤差（digitize/抽出等）が“一様スケール換算で”≲2.8% 程度に抑えられていることを示す必要がある（1 step反転の閾値）。",
                        "補図（ECDF/離散性）：`output/eht/eht_sgra_paper5_m3_historical_distribution_values_ecdf.png`",
                        "詳細metrics：`output/eht/eht_sgra_paper5_m3_nir_reconnection_conditions_metrics.json`",
                        "再現: scripts/eht/eht_sgra_paper5_m3_nir_reconnection_conditions.py → output/eht/eht_sgra_paper5_m3_nir_reconnection_conditions.png",
                    ],
                },
                {
                    "id": "eht_ring_morphology",
                    "title": "EHT：リングの形状指標（幅 W/d と非対称 A）の一次ソースレンジ",
                    "kind": "強重力テスト（形状/系統の手がかり）",
                    "path": root / "output" / "eht" / "eht_ring_morphology_public.png",
                    "summary_lines": eht_morph_summary_lines,
                    "explain_lines": [
                        "リング直径（サイズ）だけでなく、リングの幅や明るさの偏りも観測から推定される。",
                        "これらは放射モデル/散乱/スピンなどの系統誤差の“手がかり”になり、κ（リング/シャドウ比）の不確かさを詰める議論に接続できる。",
                    ],
                    "detail_lines": [
                        "W/d：リングの相対的な幅（fractional width）。A：リングの明るさの非対称（brightness asymmetry）。",
                        "値は解析手法に依存し得るため、ここでは一次ソースで公表されたレンジをそのまま記録し、比較に用いる。",
                        "将来的には、リング形状（非円形性/偏心など）も含めて κ とスピン/傾斜を同時に拘束できるかが鍵になる。",
                        "再現: scripts/eht/eht_shadow_compare.py → output/eht/eht_ring_morphology_public.png",
                    ],
                },
            ],
        ),
        (
            "強重力（連星パルサー/重力波）",
            [
                {
                    "id": "pulsar_orbital_decay",
                    "title": "二重パルサー：軌道減衰（観測/P-model）と追加放射（双極など）の制約",
                    "kind": "強重力テスト（放射: 四重極）",
                    "path": root / "output" / "pulsar" / "binary_pulsar_orbital_decay_public.png",
                    "summary_lines": pulsar_summary_lines,
                    "explain_lines": [
                        "二重パルサーでは軌道周期が時間とともに短くなる（エネルギー損失）。",
                        "観測は『四重極放射（弱場のP-model。一次ソース表記は観測/GR）』と高精度で一致し、代替理論で出やすい双極放射（dipole）を強く制限する。",
                        "P-modelが相対論の置換候補であるためには、この“放射の制約”とも整合する必要がある（棄却条件の一部）。",
                    ],
                    "detail_lines": [
                        "図は一致度 R=Ṗ_b(obs)/Ṗ_b(P-model quad) を示し、R=1 が四重極則と一致を意味する（一次ソース表記は観測/GR）。",
                        "観測が R≒1 からズレるほど、四重極以外の追加放射（双極放射など）が混入している可能性が高くなる。",
                        "ここでは最小の整理として、|R-1| と誤差（1σ/95%）から『追加放射成分の上限（概算）』を併記している。",
                        "一次ソース（DOI/参照日）は data/pulsar/binary_pulsar_orbital_decay.json に記録している。",
                        "再現: scripts/pulsar/binary_pulsar_orbital_decay.py → output/pulsar/binary_pulsar_orbital_decay_public.png",
                    ],
                },
                {
                    "id": "gw_multi_event_summary",
                    "title": "重力波：複数イベントの要約（R^2 と match）",
                    "kind": "強重力テスト（放射: 要約）",
                    "path": root / "output" / "gw" / "gw_multi_event_summary_public.png",
                    "summary_lines": gw_summary_lines,
                    "explain_lines": [
                        "連星のエネルギー損失（四重極放射）により周波数が上がる（chirp）。",
                        "公開データから chirp の整合（R^2）と、波形の一致度（match, 参考）を同じ手順で複数イベントに適用して要約する。",
                    ],
                    "detail_lines": [
                        "入力: GWOSC公開 strain（32秒, 4kHz, 検出器はイベントにより異なる）。",
                        "処理: bandpass（必要ならwhiten）→ 周波数トラック抽出 → 四重極チャープ則（t=t_c-A f^{-8/3}）の当てはめ → 波形テンプレート生成。",
                        "match窓: 周波数帯 70..300 Hz に対応する時刻範囲を求め、その範囲で match を計算する（イベント間で比較しやすくするため）。",
                        "要約図: 各イベントの R^2 と match を一覧化（検出器ごとの差も表示）。",
                        "注: 本スクリプトの M_c は“簡易推定”であり、LIGO公式の厳密推定（波形テンプレート/事前分布込み）ではない。",
                        "再現: scripts/gw/gw_multi_event_summary.py（要約） + scripts/gw/gw150914_chirp_phase.py（各イベントの生成）",
                    ],
                },
                {
                    "id": "gw150914_chirp_phase",
                    "title": "重力波（代表例: GW150914）：観測波形と単純モデル（四重極）の比較",
                    "kind": "強重力テスト（放射: 波形）",
                    "path": root / "output" / "gw" / "gw150914_waveform_compare_public.png",
                    "summary_lines": ["代表例：GW150914（波形の重ね合わせ, 窓内）"],
                    "explain_lines": [
                        "連星のエネルギー損失（四重極放射）により周波数が上がる（chirp）。",
                        "公開データの波形（bandpass後）と、四重極チャープ則（Newton近似）から作る単純モデル波形を重ね合わせて比較する。",
                    ],
                    "detail_lines": [
                        "入力: GWOSC公開 strain（GW150914, 32秒, 4kHz, H1/L1）。",
                        "処理: bandpass → 周波数トラック抽出 → 四重極チャープ則（t=t_c-A f^{-8/3}）の当てはめ → 波形テンプレート生成。",
                        "波形比較: 周波数帯 70..300 Hz に対応する時刻範囲で振幅/位相を最小二乗で合わせ、match（正規化内積）を併記する。",
                        "一次データのURL/ハッシュは data/gw/gw150914/gw150914_sources.json に保存している。",
                        "再現: scripts/gw/gw150914_chirp_phase.py（オンライン）/ scripts/gw/gw150914_chirp_phase.py --offline（キャッシュ後）",
                    ],
                },
            ],
        ),
        (
            "差分予測（Phase 4）",
            [
                {
                    "id": "theory_delta_saturation",
                    "title": "速度項の飽和 δ：既知の高γ観測との整合",
                    "kind": "差分予測（Phase 4）",
                    "path": root / "output" / "theory" / "delta_saturation_constraints.png",
                    "summary_lines": delta_sat_summary_lines,
                    "explain_lines": [
                        "P-modelの速度項では v→c で dτ/dt が0にならず、ローレンツ因子 γ に上限 γ_max≈1/√δ が生じる。",
                        "既知の高エネルギー粒子（加速器/宇宙線/ニュートリノ等）が示す γ を上回るよう、δ を十分小さく選ぶ必要がある。",
                    ],
                    "detail_lines": [
                        "ここでの γ は概算として γ≈E/(mc^2) を用いる（超相対論では運動エネルギーとの差は無視できる）。",
                        "観測から得た γ_obs に対し、P-modelの上限 γ_max が γ_obs を下回ると矛盾するため、δ は δ < 1/(γ_obs^2 - 1) を満たす必要がある。",
                        "右図は各例から得られる δ の上限（概算）で、採用δ（破線）が十分小さければ整合する。",
                        "νの質量は未確定なので、νの行は『仮定した質量』に応じて上限がスケールする点に注意（m が小さいほど γ が大きくなり、δ上限はより厳しくなる）。",
                        "この図は δ を“測定した”ものではなく、P-modelが既存観測と矛盾しないように δ をどこまで小さくすべきかの指標を与える。",
                        "再現: scripts/theory/delta_saturation_constraints.py → output/theory/delta_saturation_constraints.png",
                    ],
                    "table": delta_sat_table,
                },
                {
                    "id": "eht_shadow_differential",
                    "title": "EHT：差分予測（P-model − GR のシャドウ直径）",
                    "kind": "差分予測（Phase 4）",
                    "path": root / "output" / "eht" / "eht_shadow_differential_public.png",
                    "summary_lines": eht_diff_summary_lines,
                    "explain_lines": [
                        "Cassini等で β を固定しても、P-model最小延長のシャドウ直径係数はGR（Schwarzschild）と数%ずれる。",
                        "左は差（µas）、右は3σで判別するのに必要な観測精度（1σ, µas）の目安。",
                    ],
                    "detail_lines": [
                        "P-model（最小）: θ_shadow = 4 e β (GM/(c^2 D))。",
                        "GR（Schwarzschild）: θ_shadow = 2√27 (GM/(c^2 D))。",
                        "係数比（P/GR）は β に比例し、β=1 では約 +4.6% の差が残る。",
                        "右図の『必要精度』は、質量/距離の不確かさ（パラメータ誤差）を含めた上で、差が3σで判別できる条件から逆算した目安。",
                        "リング直径≒シャドウ直径の近似には系統誤差があり、厳密判定には放射モデル/散乱/スピン依存の精密化が必要。",
                        "再現: scripts/eht/eht_shadow_compare.py → output/eht/eht_shadow_differential_public.png",
                    ],
                },
            ],
        ),
        (
            "宇宙論（赤方偏移：膨張なし）",
            [
                {
                    "id": "cosmology_redshift_pbg",
                    "title": "宇宙論：宇宙膨張なしで赤方偏移を P で説明（背景Pの時間変化）",
                    "kind": "機構の提示（差分予測候補）",
                    "path": root / "output" / "private" / "cosmology" / "cosmology_redshift_pbg.png",
                    "summary_lines": [
                        "P(x,t)=P_bg(t)·P_local(x) を仮定し、1+z=P_em/P_obs を用いる。",
                        "低zでは H0^(P)=-(d/dt ln P_bg)|t0 により z≈H0^(P)·D/c（Hubble則）となる。",
                    ],
                    "explain_lines": [
                        "赤方偏移は「空間の膨張」だけでなく「時計刻みの宇宙規模の変化」でも説明できる可能性がある。",
                        "この図は“膨張なしの機構”を最小モデルで可視化したもの。",
                    ],
                    "detail_lines": [
                        "観測で定義される赤方偏移は 1+z = λ_obs/λ_em = ν_em/ν_obs。",
                        "P-modelの宇宙論的拡張として、背景の時間波密度を P_bg(t) とし、P(x,t)=P_bg(t)P_local(x) と分離する。",
                        "周波数比を ν_obs/ν_em = P_obs/P_em と置くと、直ちに 1+z = P_em/P_obs が得られる。",
                        "P_bg(t) が時間とともに減少するなら、過去ほど P_em>P_obs となり z>0（赤方偏移）が出る。",
                        "低zでは H0^(P)≡-(d/dt ln P_bg)|t0 を導入して z≈H0^(P)Δt≈H0^(P)D/c（静的近似 D≈cΔt）となる。",
                        "注意：この図は“赤方偏移だけから宇宙膨張や特異点を唯一の説明として結論づけられない”ことを示す機構の例示であり、宇宙論全体（CMB/元素合成など）の置換は別途検証が必要。",
                        "再現: scripts/cosmology/cosmology_redshift_pbg.py → output/cosmology/cosmology_redshift_pbg.png",
                    ],
                },
                {
                    "id": "cosmology_observable_scalings",
                    "title": "宇宙論：観測量スケーリング（距離二重性 / Tolman表面輝度）",
                    "kind": "差分予測（観測量への接続）",
                    "path": root / "output" / "private" / "cosmology" / "cosmology_observable_scalings.png",
                    "summary_lines": [
                        "距離二重性 η(z)=D_L/((1+z)^2 D_A) を比較：FRWではη=1、静的背景Pではη=1/(1+z)。",
                        "Tolman表面輝度：FRWは(1+z)^-4、静的背景Pは(1+z)^-2（最小仮定）。",
                    ],
                    "explain_lines": [
                        "“赤方偏移の起源”を議論するには、距離・表面輝度などの観測量で差が出るかを明示する必要がある。",
                        "この図は、観測データに当てる前の差分予測（定義からのスケーリング）をまとめたもの。",
                    ],
                    "detail_lines": [
                        "距離二重性（光子保存＋幾何学的膨張）: D_L=(1+z)^2 D_A → η(z)=1。",
                        "背景P（膨張なし・静的幾何）: D_L=(1+z) D_A → η(z)=1/(1+z)。",
                        "Tolman表面輝度: FRWではSB∝(1+z)^-4、背景P（静的）ではSB∝(1+z)^-2。",
                        "注：ここでは時間伸長 Δt_obs=(1+z)Δt_em を両モデルに置いた（最小仮定）。",
                        "再現: scripts/cosmology/cosmology_observable_scalings.py → output/cosmology/cosmology_observable_scalings.png",
                    ],
                },
                {
                    "id": "cosmology_distance_duality_constraints",
                    "title": "宇宙論：距離二重性ηの観測制約（棄却条件）",
                    "kind": "観測制約（一次ソース）",
                    "path": root / "output" / "private" / "cosmology" / "cosmology_distance_duality_constraints.png",
                    "summary_lines": cosmo_ddr_summary_lines,
                    "explain_lines": [
                        "距離二重性（DDR）は、光度距離 D_L と角径距離 D_A の間の関係で、観測から検証できる。",
                        "標準（FRW + 光子数保存）では η(z)=D_L/((1+z)^2 D_A)=1（ε0=0）。",
                        "背景P（膨張なし・静的幾何）の最小モデルは η(z)=1/(1+z)（ε0=-1）となり、観測制約で判別できる。",
                    ],
                    "detail_lines": [
                        *cosmo_ddr_definition_lines,
                        "この図は一次ソースで公表された ε0 の制約（1σ）を η(z) の帯として可視化し、背景P（静的）の予測と比較する。",
                        "棄却条件（目安）：|ε_obs - ε_model| > 3σ ならそのモデルは棄却（統計的な目安）。",
                        "再現: scripts/cosmology/cosmology_distance_duality_constraints.py → output/cosmology/cosmology_distance_duality_constraints.png",
                    ],
                    "table": cosmo_ddr_table,
                },
                {
                    "id": "cosmology_distance_duality_source_sensitivity",
                    "title": "宇宙論：DDR一次ソース依存（距離指標で結論がどれだけ変わるか）",
                    "kind": "現状整理（Step 16.5.1）",
                    "path": root / "output" / "private" / "cosmology" / "cosmology_distance_duality_source_sensitivity.png",
                    "summary_lines": [
                        "同じ静的最小（ε0=-1）でも、一次ソース（距離指標）の採り方で棄却度 |z| が大きく変わる。",
                        "BAOを含む最も強い制約は強く棄却する一方、BAOなしの一部制約は棄却が弱い。",
                    ],
                    "detail_lines": [
                        "これは DDR の全一次ソース行（Table 1 の元データ）を並べ、ε0=-1 からの外れ度 |z| を比較した俯瞰図。",
                        "次の焦点：どの仮定（校正/共分散/進化補正/幾何モデル）が |z| を支配しているかを個別に切り分ける。",
                        "再現: scripts/cosmology/cosmology_distance_duality_source_sensitivity.py → output/cosmology/cosmology_distance_duality_source_sensitivity.png",
                    ],
                },
                {
                    "id": "cosmology_tolman_surface_brightness_constraints",
                    "title": "宇宙論：Tolman表面輝度の一次ソース制約（参考）",
                    "kind": "観測制約（一次ソース, 進化が系統）",
                    "path": root / "output" / "private" / "cosmology" / "cosmology_tolman_surface_brightness_constraints.png",
                    "summary_lines": cosmo_tolman_summary_lines,
                    "explain_lines": [
                        "Tolman表面輝度（SB dimming）は、膨張（FRW）と“膨張なし”の差が観測量として現れる候補。",
                        "一方で銀河の光度進化が混入するため、このカードの数値は「差の符号/スケール」の参考として扱う。",
                    ],
                    "detail_lines": [
                        *cosmo_tolman_definition_lines,
                        "背景P（静的）の最小モデル（n=2）に合わせるには、しばしば進化補正が逆符号（過去が暗い側）になる点が問題になる。",
                        "再現: scripts/cosmology/cosmology_tolman_surface_brightness_constraints.py → output/cosmology/cosmology_tolman_surface_brightness_constraints.png",
                    ],
                    "table": cosmo_tolman_table,
                },
                {
                    "id": "cosmology_tension_attribution",
                    "title": "宇宙論：張力の原因切り分け（現状）",
                    "kind": "現状整理（Step 16.5）",
                    "path": root / "output" / "private" / "cosmology" / "cosmology_tension_attribution.png",
                    "summary_lines": cosmo_tension_attr_summary_lines,
                    "explain_lines": [
                        "距離指標依存の張力（DDR/BAO/Tolman）と、独立プローブ（時間伸長/T(z)）の整合を並べて表示する。",
                        "現状は『距離指標の定義・校正・進化補正』が結論を支配しているため、次に何を詰めるべきかを明確にする入口。",
                    ],
                    "detail_lines": cosmo_tension_attr_detail_lines,
                },
                {
                    "id": "cosmology_desi_dr1_bao_promotion_check",
                    "title": "宇宙論：DESI DR1 BAO（multi-tracer 昇格判定）",
                    "kind": "距離指標依存の強い制約（Step 4.5）",
                    "path": root
                    / "output"
                    / "cosmology"
                    / "cosmology_desi_dr1_bao_promotion_check_public.png",
                    "summary_lines": cosmo_desi_promo_summary_lines,
                    "explain_lines": [
                        "DESI DR1 の raw/VAC から dv=[ξ0,ξ2]+cov を自前生成し、ε_expected との cross-check が λ と cov 推定法に対して stable かを確認する。",
                        "stable |z|≥3 を満たす tracer が2つ以上なら、DESI を screening→確証へ昇格（距離指標依存の張力として Step 4.7 へ接続）。",
                    ],
                    "detail_lines": cosmo_desi_promo_detail_lines,
                },
                {
                    "id": "cosmology_jwst_mast_x1d",
                    "title": "宇宙論：JWST スペクトル一次データ（MAST; x1d）",
                    "kind": "一次データ（距離指標非依存）",
                    "path": root / "output" / "private" / "cosmology" / "jwst_spectra__jades_gs_z14_0__x1d_qc.png",
                    "summary_lines": jwst_mast_summary_lines,
                    "explain_lines": [
                        "赤方偏移 z は本質的に「スペクトル線のズレ」で決まるため、距離指標の前提に依らず一次データから扱える。",
                        "本カードは JWST の x1d をローカルにキャッシュし、offline で QC と z候補推定（半自動）まで回す入口。",
                    ],
                    "detail_lines": jwst_mast_detail_lines,
                    "table": jwst_mast_table,
                },
            ],
        ),
    ]

    llr_graphs_detail: Optional[List[Dict[str, Any]]] = None
    llr_section_idx: Optional[int] = None
    try:
        for idx, (sec_title, graphs) in enumerate(sections):
            if sec_title != LLR_LONG_NAME or not isinstance(graphs, list) or not graphs:
                continue
            llr_section_idx = idx
            llr_graphs_detail = graphs
            break
    except Exception:
        llr_graphs_detail = None
        llr_section_idx = None

    # Detailed LLR page (anchors per graph), so readers can jump from the public report.
    llr_detail_html: Optional[Path] = None
    try:
        llr_graphs = llr_graphs_detail
        if not (isinstance(llr_graphs, list) and llr_graphs):
            llr_graphs = next(
                (
                    gs
                    for _sec, gs in sections
                    if isinstance(gs, list) and any(str(g.get("id") or "").startswith("llr_") for g in gs)
                ),
                None,
            )
        if isinstance(llr_graphs, list) and llr_graphs:
            llr_detail_html = _render_llr_detail_html(
                out_dir=out_dir,
                graphs=llr_graphs,
                llr_stem=llr_stem,
                llr_summary=llr_summary,
                llr_source=llr_source,
                llr_batch_summary=llr_batch_summary,
            )
            rel_llr_detail = _rel_url(out_dir, llr_detail_html)
            for g in llr_graphs:
                gid = str(g.get("id") or "")
                if gid:
                    g["detail_href"] = f"{rel_llr_detail}#{gid}"
    except Exception as e:
        print(f"[warn] failed to build LLR detail page: {e}")
        llr_detail_html = None

    # Public report should keep LLR minimal: show only the residual (obs - model) card.
    # Detailed diagnostics stay in output/private/summary/details/llr.html.
    try:
        if llr_section_idx is not None and isinstance(llr_graphs_detail, list) and llr_graphs_detail:
            llr_public_ids = {"llr_residual"}
            llr_graphs_public: List[Dict[str, Any]] = []
            for g in llr_graphs_detail:
                if str(g.get("id") or "") not in llr_public_ids:
                    continue
                g_pub = dict(g)
                if str(g_pub.get("id") or "") == "llr_residual":
                    rms_ns = llr_model_metrics.get("rms_residual_station_reflector_ns")
                    st = llr_model_metrics.get("station_code") or llr_summary.get("station")
                    tgt = llr_model_metrics.get("target") or llr_summary.get("target")
                    summary_lines: List[str] = []
                    if st or tgt:
                        summary_lines.append(f"代表例: {st} → {tgt}".strip())
                    if isinstance(rms_ns, (int, float)):
                        one_way_m = float(rms_ns) * 0.149896229
                        summary_lines.append(
                            f"残差RMS={_format_num(float(rms_ns), digits=4)} ns（片道距離で約{_format_num(one_way_m, digits=3)} m）"
                        )
                    if summary_lines:
                        g_pub["summary_lines"] = summary_lines
                    g_pub["explain_lines"] = [
                        "残差（観測 - モデル）。0に近いほど一致している。",
                        "定数オフセット整列後なので、装置遅延などの定数差は吸収している。",
                    ]
                    g_pub["detail_lines"] = []
                llr_graphs_public.append(g_pub)
            if llr_graphs_public:
                sections[llr_section_idx] = (LLR_LONG_NAME, llr_graphs_public)
    except Exception:
        pass

    public_html = _render_public_html(
        root=root,
        out_dir=out_dir,
        title="Pモデル 比較レポート（一般向け）",
        subtitle="観測（実測/準実測/デジタイズ/代表値）・標準値 と P-model の比較図を、解説付きで一覧表示します。",
        sections=sections,
    )

    out = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "outputs": {
            "png": str(png_path),
            "html": str(public_html),
            "llr_detail_html": str(llr_detail_html) if llr_detail_html else None,
        },
        "inputs": {title: str(path) for title, path in items},
        "panel_metrics": {
            "solar_light_deflection": solar_m,
            "gps_compare": gps_m,
            "cassini_fig2": cassini_m,
        },
        "notes": [
            "このダッシュボードは各トピックの既存図を再利用し、読みやすい指標を追記しています。",
            "詳細は output/<topic>/ と output/private/summary/pmodel_report.html を参照してください。",
            "一般向けに全グラフを一覧表示するHTMLは output/private/summary/pmodel_public_report.html。",
        ],
    }
    json_path = out_dir / "pmodel_public_dashboard_metrics.json"
    json_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[ok] png : {png_path}")
    print(f"[ok] html: {public_html}")
    print(f"[ok] json: {json_path}")

    try:
        worklog.append_event(
            {
                "event_type": "public_dashboard",
                "argv": sys.argv,
                "outputs": {
                    "dashboard_png": png_path,
                    "public_report_html": public_html,
                    "metrics_json": json_path,
                    "llr_detail_html": llr_detail_html,
                },
            }
        )
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
