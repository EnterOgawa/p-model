from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# クラス: `Section` の責務と境界条件を定義する。
@dataclass(frozen=True)
class Section:
    key: str
    title: str
    runbook_path: Path
    images: List[Tuple[str, Path]]  # (caption, path)
    metrics: Dict[str, Any]
    notes: List[str]


# 関数: `_repo_root` の入出力契約と処理意図を定義する。

def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


# 関数: `_set_japanese_font` の入出力契約と処理意図を定義する。

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
        # 条件分岐: `not chosen` を満たす経路を評価する。
        if not chosen:
            return

        mpl.rcParams["font.family"] = chosen + ["DejaVu Sans"]
        mpl.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass


# 関数: `_rel_url` の入出力契約と処理意図を定義する。

def _rel_url(from_dir: Path, target: Path) -> str:
    try:
        rel = os.path.relpath(target, start=from_dir)
    except ValueError:
        rel = str(target)

    return rel.replace("\\", "/")


# 関数: `_read_csv_dicts` の入出力契約と処理意図を定義する。

def _read_csv_dicts(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        return list(r)


# 関数: `_read_json` の入出力契約と処理意図を定義する。

def _read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# 関数: `_pick_newest` の入出力契約と処理意図を定義する。

def _pick_newest(glob_paths: List[Path]) -> Optional[Path]:
    items = [p for p in glob_paths if p.exists()]
    # 条件分岐: `not items` を満たす経路を評価する。
    if not items:
        return None

    return max(items, key=lambda p: p.stat().st_mtime)


# 関数: `_try_float` の入出力契約と処理意図を定義する。

def _try_float(x: str) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


# 関数: `_cassini_section` の入出力契約と処理意図を定義する。

def _cassini_section(root: Path) -> Section:
    out_dir = root / "output" / "private" / "cassini"
    runbook = root / "doc" / "cassini" / "README.md"

    overlay_zoom = out_dir / "cassini_fig2_overlay_zoom10d.png"
    overlay_full = out_dir / "cassini_fig2_overlay_full.png"
    residuals = out_dir / "cassini_fig2_residuals.png"
    metrics_csv = out_dir / "cassini_fig2_metrics.csv"

    overlay = overlay_zoom if overlay_zoom.exists() else overlay_full

    metrics: Dict[str, Any] = {}
    notes: List[str] = []
    # 条件分岐: `metrics_csv.exists()` を満たす経路を評価する。
    if metrics_csv.exists():
        rows = _read_csv_dicts(metrics_csv)
        row_all = next((r for r in rows if (r.get("window") or "").startswith("all")), rows[0] if rows else None)
        # 条件分岐: `row_all` を満たす経路を評価する。
        if row_all:
            metrics["n"] = int(float(row_all.get("n", "nan")))
            rmse = _try_float(row_all.get("rmse", ""))
            corr = _try_float(row_all.get("corr", ""))
            # 条件分岐: `rmse is not None` を満たす経路を評価する。
            if rmse is not None:
                metrics["rmse"] = rmse

            # 条件分岐: `corr is not None` を満たす経路を評価する。

            if corr is not None:
                metrics["corr"] = corr
    else:
        notes.append(f"Missing: {metrics_csv}")

    images: List[Tuple[str, Path]] = []
    # 条件分岐: `overlay.exists()` を満たす経路を評価する。
    if overlay.exists():
        images.append(("重ね合わせ（拡大）", overlay))
    else:
        notes.append(f"Missing: {overlay_zoom} / {overlay_full}")

    # 条件分岐: `residuals.exists()` を満たす経路を評価する。

    if residuals.exists():
        images.append(("残差", residuals))

    return Section(
        key="cassini",
        title="カッシーニ（Shapiro / ドップラー y）",
        runbook_path=runbook,
        images=images,
        metrics=metrics,
        notes=notes,
    )


# 関数: `_viking_section` の入出力契約と処理意図を定義する。

def _viking_section(root: Path) -> Section:
    out_dir = root / "output" / "private" / "viking"
    runbook = root / "doc" / "viking" / "README.md"

    plot = out_dir / "viking_p_model_vs_measured_no_arrow.png"
    csv_path = out_dir / "viking_shapiro_result.csv"

    metrics: Dict[str, Any] = {}
    notes: List[str] = []
    # 条件分岐: `csv_path.exists()` を満たす経路を評価する。
    if csv_path.exists():
        peak_us = None
        peak_t = None
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                t = row.get("time_utc")
                v = _try_float(row.get("shapiro_delay_us", ""))
                # 条件分岐: `t is None or v is None` を満たす経路を評価する。
                if t is None or v is None:
                    continue

                # 条件分岐: `peak_us is None or v > peak_us` を満たす経路を評価する。

                if peak_us is None or v > peak_us:
                    peak_us = v
                    peak_t = t

        # 条件分岐: `peak_us is not None` を満たす経路を評価する。

        if peak_us is not None:
            metrics["peak_delay_us"] = peak_us

        # 条件分岐: `peak_t is not None` を満たす経路を評価する。

        if peak_t is not None:
            metrics["peak_time_utc"] = peak_t
    else:
        notes.append(f"Missing: {csv_path}")

    images: List[Tuple[str, Path]] = []
    # 条件分岐: `plot.exists()` を満たす経路を評価する。
    if plot.exists():
        images.append(("シミュレーション vs 観測（代表値）", plot))
    else:
        notes.append(f"Missing: {plot}")

    return Section(
        key="viking",
        title="バイキング（Shapiro遅延・往復）",
        runbook_path=runbook,
        images=images,
        metrics=metrics,
        notes=notes,
    )


# 関数: `_gps_section` の入出力契約と処理意図を定義する。

def _gps_section(root: Path) -> Section:
    out_dir = root / "output" / "private" / "gps"
    runbook = root / "doc" / "gps" / "README.md"

    rms_plot = out_dir / "gps_rms_compare.png"
    g01_plot = out_dir / "gps_residual_compare_G01.png"
    plot_all = out_dir / "gps_clock_residuals_all_31.png"
    summary_csv = out_dir / "summary_batch.csv"

    metrics: Dict[str, Any] = {}
    notes: List[str] = []
    # 条件分岐: `summary_csv.exists()` を満たす経路を評価する。
    if summary_csv.exists():
        rms_b_ns: List[float] = []
        rms_p_ns: List[float] = []
        with open(summary_csv, "r", newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                rms_m = _try_float(row.get("RMS_BRDC_m", ""))
                # 条件分岐: `rms_m is None` を満たす経路を評価する。
                if rms_m is None:
                    continue

                rms_b_ns.append((rms_m / 299_792_458.0) * 1e9)

                rms_p_m = _try_float(row.get("RMS_PMODEL_m", ""))
                # 条件分岐: `rms_p_m is not None` を満たす経路を評価する。
                if rms_p_m is not None:
                    rms_p_ns.append((rms_p_m / 299_792_458.0) * 1e9)

        # 条件分岐: `rms_b_ns` を満たす経路を評価する。

        if rms_b_ns:
            b_sorted = sorted(rms_b_ns)
            metrics["n_sats"] = len(b_sorted)
            metrics["brdc_rms_ns_median"] = b_sorted[len(b_sorted) // 2]
            metrics["brdc_rms_ns_max"] = max(b_sorted)

        # 条件分岐: `rms_p_ns` を満たす経路を評価する。

        if rms_p_ns:
            p_sorted = sorted(rms_p_ns)
            metrics["pmodel_rms_ns_median"] = p_sorted[len(p_sorted) // 2]
            metrics["pmodel_rms_ns_max"] = max(p_sorted)

            # If both are present per-satellite, count which is better.
            try:
                # Re-read once to align by PRN reliably.
                rows = _read_csv_dicts(summary_csv)
                better = 0
                worse = 0
                for row in rows:
                    b_m = _try_float(row.get("RMS_BRDC_m", ""))
                    p_m = _try_float(row.get("RMS_PMODEL_m", ""))
                    # 条件分岐: `b_m is None or p_m is None` を満たす経路を評価する。
                    if b_m is None or p_m is None:
                        continue

                    # 条件分岐: `p_m < b_m` を満たす経路を評価する。

                    if p_m < b_m:
                        better += 1
                    # 条件分岐: 前段条件が不成立で、`p_m > b_m` を追加評価する。
                    elif p_m > b_m:
                        worse += 1

                metrics["pmodel_better_count"] = better
                metrics["brdc_better_count"] = worse
            except Exception:
                pass
    else:
        notes.append(f"Missing: {summary_csv}")

    images: List[Tuple[str, Path]] = []
    # 条件分岐: `rms_plot.exists()` を満たす経路を評価する。
    if rms_plot.exists():
        images.append(("残差RMS（BRDC vs P-model）, 全衛星", rms_plot))

    # 条件分岐: `g01_plot.exists()` を満たす経路を評価する。

    if g01_plot.exists():
        images.append(("時系列例（G01）", g01_plot))

    # 条件分岐: `plot_all.exists()` を満たす経路を評価する。

    if plot_all.exists():
        images.append(("残差（BRDC - IGS）, 全衛星", plot_all))

    # 条件分岐: `not images` を満たす経路を評価する。

    if not images:
        notes.append(f"Missing: {rms_plot} / {g01_plot} / {plot_all}")

    return Section(
        key="gps",
        title="GPS（時計残差, 観測=IGS Final）",
        runbook_path=runbook,
        images=images,
        metrics=metrics,
        notes=notes,
    )


# 関数: `_llr_section` の入出力契約と処理意図を定義する。

def _llr_section(root: Path) -> Section:
    out_root = root / "output" / "private" / "llr"
    runbook = root / "doc" / "llr" / "README.md"

    out_dir = out_root / "out_llr"
    overlay = _pick_newest(list(out_dir.glob("*_overlay_tof.png")))
    residual = _pick_newest(list(out_dir.glob("*_residual.png")))
    table = _pick_newest(list(out_dir.glob("*_table.csv")))

    metrics: Dict[str, Any] = {}
    notes: List[str] = []
    # 条件分岐: `table and table.exists()` を満たす経路を評価する。
    if table and table.exists():
        res_ns: List[float] = []
        with open(table, "r", newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                obs = _try_float(row.get("tof_obs_s", ""))
                mod = _try_float(row.get("tof_model_s", ""))
                # 条件分岐: `obs is None or mod is None` を満たす経路を評価する。
                if obs is None or mod is None:
                    continue

                res_ns.append((obs - mod) * 1e9)

        # 条件分岐: `res_ns` を満たす経路を評価する。

        if res_ns:
            mean = sum(res_ns) / len(res_ns)
            rms = math.sqrt(sum((x - mean) ** 2 for x in res_ns) / len(res_ns))
            metrics["n_points"] = len(res_ns)
            metrics["residual_rms_ns"] = rms
    else:
        notes.append(f"Missing: {out_dir}/*_table.csv")

    images: List[Tuple[str, Path]] = []
    # 条件分岐: `overlay` を満たす経路を評価する。
    if overlay:
        images.append(("観測 vs モデル（平均除去）", overlay))
    else:
        notes.append(f"Missing: {out_dir}/*_overlay_tof.png")

    # 条件分岐: `residual` を満たす経路を評価する。

    if residual:
        images.append(("残差（観測 - モデル）", residual))

    return Section(
        key="llr",
        title="LLR（CRD Normal Point 重ね合わせ）",
        runbook_path=runbook,
        images=images,
        metrics=metrics,
        notes=notes,
    )


# 関数: `_mercury_section` の入出力契約と処理意図を定義する。

def _mercury_section(root: Path) -> Section:
    out_dir = root / "output" / "private" / "mercury"
    runbook = root / "doc" / "mercury" / "README.md"

    plot = out_dir / "mercury_orbit.png"

    notes: List[str] = []
    images: List[Tuple[str, Path]] = []
    # 条件分岐: `plot.exists()` を満たす経路を評価する。
    if plot.exists():
        images.append(("軌道 / 近日点移動（概念図）", plot))
    else:
        notes.append(f"Missing: {plot}")

    return Section(
        key="mercury",
        title="水星（近日点移動の概念図）",
        runbook_path=runbook,
        images=images,
        metrics={},
        notes=notes,
    )


# 関数: `_theory_section` の入出力契約と処理意図を定義する。

def _theory_section(root: Path) -> Section:
    out_dir = root / "output" / "private" / "theory"
    runbook = root / "doc" / "theory" / "README.md"

    notes: List[str] = []
    images: List[Tuple[str, Path]] = []
    metrics: Dict[str, Any] = {}

    # Solar light deflection
    solar_png = out_dir / "solar_light_deflection.png"
    solar_json = out_dir / "solar_light_deflection_metrics.json"
    # 条件分岐: `solar_png.exists()` を満たす経路を評価する。
    if solar_png.exists():
        images.append(("太陽重力による光の偏向（α vs b）", solar_png))
    else:
        notes.append(f"Missing: {solar_png}")

    # 条件分岐: `solar_json.exists()` を満たす経路を評価する。

    if solar_json.exists():
        try:
            j = _read_json(solar_json)
            m = j.get("metrics", {})
            metrics["solar_alpha_limb_arcsec"] = m.get("alpha_pmodel_arcsec_limb")
            metrics["solar_ref_arcsec_limb"] = m.get("reference_arcsec_limb")
            metrics["solar_abs_error_arcsec"] = m.get("abs_error_arcsec")
        except Exception as e:
            notes.append(f"Failed to read {solar_json}: {e}")
    else:
        notes.append(f"Missing: {solar_json}")

    # GPS time dilation breakdown

    gps_png = out_dir / "gps_time_dilation.png"
    gps_json = out_dir / "gps_time_dilation_metrics.json"
    # 条件分岐: `gps_png.exists()` を満たす経路を評価する。
    if gps_png.exists():
        images.append(("GPSの時間補正（区間あたり）", gps_png))
    else:
        notes.append(f"Missing: {gps_png}")

    # 条件分岐: `gps_json.exists()` を満たす経路を評価する。

    if gps_json.exists():
        try:
            j = _read_json(gps_json)
            m = j.get("metrics", {})
            metrics["gps_grav_us"] = m.get("grav_approx_us")
            metrics["gps_sr_us"] = m.get("sr_approx_us")
            metrics["gps_net_us"] = m.get("net_approx_us")
            metrics["gps_ref_net_us"] = m.get("ref_net_us_day")
            metrics["gps_abs_error_net_us"] = m.get("abs_error_net_us_vs_ref")
        except Exception as e:
            notes.append(f"Failed to read {gps_json}: {e}")
    else:
        notes.append(f"Missing: {gps_json}")

    return Section(
        key="theory",
        title="理論チェック（光の偏向 / 時間補正）",
        runbook_path=runbook,
        images=images,
        metrics=metrics,
        notes=notes
        + [
            "注：ここでの標準値は文献・教科書の代表値（このリポジトリ内の生データではない）です。"
        ],
    )


# 関数: `_run_all_status_section` の入出力契約と処理意図を定義する。

def _run_all_status_section(root: Path, out_dir: Path) -> Section:
    status_path = root / "output" / "private" / "summary" / "run_all_status.json"
    runbook = root / "doc" / "summary" / "README.md"

    metrics: Dict[str, Any] = {
        "status_json": _rel_url(out_dir, status_path),
    }
    notes: List[str] = []

    # 条件分岐: `not status_path.exists()` を満たす経路を評価する。
    if not status_path.exists():
        notes.append(f"Missing: {status_path}")
        return Section(
            key="run_all",
            title="実行ステータス（run_all）",
            runbook_path=runbook,
            images=[],
            metrics=metrics,
            notes=notes,
        )

    try:
        status = _read_json(status_path)
    except Exception as e:
        notes.append(f"Failed to read {status_path}: {e}")
        return Section(
            key="run_all",
            title="実行ステータス（run_all）",
            runbook_path=runbook,
            images=[],
            metrics=metrics,
            notes=notes,
        )

    metrics["generated_utc"] = status.get("generated_utc")
    metrics["mode"] = status.get("mode")

    tasks = status.get("tasks", [])
    ok = 0
    failed = 0
    skipped = 0
    for t in tasks:
        # 条件分岐: `not isinstance(t, dict)` を満たす経路を評価する。
        if not isinstance(t, dict):
            continue

        key = str(t.get("key", ""))
        # 条件分岐: `t.get("skipped", False)` を満たす経路を評価する。
        if t.get("skipped", False):
            skipped += 1
            reason = t.get("reason", "")
            notes.append(f"{key}: スキップ（{reason}）")
            continue

        # 条件分岐: `t.get("ok", False)` を満たす経路を評価する。

        if t.get("ok", False):
            ok += 1
            rc = t.get("returncode", 0)
            elapsed = t.get("elapsed_s", None)
            # 条件分岐: `isinstance(elapsed, (int, float))` を満たす経路を評価する。
            if isinstance(elapsed, (int, float)):
                notes.append(f"{key}: OK（rc={rc}, {elapsed:.2f}秒）")
            else:
                notes.append(f"{key}: OK（rc={rc}）")

            continue

        failed += 1
        rc = t.get("returncode", None)
        log = t.get("log", "")
        elapsed = t.get("elapsed_s", None)
        # 条件分岐: `isinstance(elapsed, (int, float))` を満たす経路を評価する。
        if isinstance(elapsed, (int, float)):
            notes.append(f"{key}: 失敗（rc={rc}, {elapsed:.2f}秒） log={log}")
        else:
            notes.append(f"{key}: 失敗（rc={rc}） log={log}")

    metrics["tasks_total"] = len([t for t in tasks if isinstance(t, dict)])
    metrics["tasks_ok"] = ok
    metrics["tasks_failed"] = failed
    metrics["tasks_skipped"] = skipped

    return Section(
        key="run_all",
        title="実行ステータス（run_all）",
        runbook_path=runbook,
        images=[],
        metrics=metrics,
        notes=notes,
    )


# 関数: `_public_section` の入出力契約と処理意図を定義する。

def _public_section(root: Path) -> Section:
    out_dir = root / "output" / "private" / "summary"
    runbook = root / "doc" / "summary" / "README.md"

    dash_png = out_dir / "pmodel_public_dashboard.png"
    dash_json = out_dir / "pmodel_public_dashboard_metrics.json"

    notes: List[str] = []
    images: List[Tuple[str, Path]] = []
    metrics: Dict[str, Any] = {}

    # 条件分岐: `dash_png.exists()` を満たす経路を評価する。
    if dash_png.exists():
        images.append(("一般向けダッシュボード（観測/標準値 vs シミュレーション）", dash_png))
    else:
        notes.append(f"Missing: {dash_png}")

    # 条件分岐: `dash_json.exists()` を満たす経路を評価する。

    if dash_json.exists():
        try:
            j = _read_json(dash_json)
            metrics["generated_utc"] = j.get("generated_utc")
        except Exception as e:
            notes.append(f"Failed to read {dash_json}: {e}")
    else:
        notes.append(f"Missing: {dash_json}")

    return Section(
        key="public",
        title="一般向け（比較ダッシュボード）",
        runbook_path=runbook,
        images=images,
        metrics=metrics,
        notes=notes,
    )


# 関数: `_render_html` の入出力契約と処理意図を定義する。

def _render_html(sections: List[Section], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    html_path = out_dir / "pmodel_report.html"

    now = datetime.now(timezone.utc).isoformat()

    parts: List[str] = []
    parts.append("<!doctype html>")
    parts.append('<html lang="ja">')
    parts.append("<head>")
    parts.append('<meta charset="utf-8">')
    parts.append('<meta name="viewport" content="width=device-width, initial-scale=1">')
    parts.append("<title>P-model 比較レポート</title>")
    parts.append(
        "<style>"
        "body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;"
        "margin:24px;line-height:1.5;color:#111;}"
        "h1{margin:0 0 8px 0;} .muted{color:#666;font-size:0.95em;}"
        "section{margin:24px 0;padding-top:8px;border-top:1px solid #eee;}"
        "figure{margin:12px 0 18px 0;}"
        "img{max-width:100%;height:auto;border:1px solid #ddd;border-radius:6px;}"
        "code{background:#f6f6f6;padding:2px 6px;border-radius:4px;}"
        "table{border-collapse:collapse;margin:8px 0;}"
        "th,td{border:1px solid #ddd;padding:6px 10px;font-size:0.95em;}"
        "th{background:#fafafa;text-align:left;}"
        "</style>"
    )
    parts.append("</head>")
    parts.append("<body>")
    parts.append("<h1>P-model 比較レポート</h1>")
    parts.append(f'<div class="muted">生成: <code>{html.escape(now)}</code></div>')

    for sec in sections:
        parts.append(f"<section id='{html.escape(sec.key)}'>")
        parts.append(f"<h2>{html.escape(sec.title)}</h2>")
        parts.append(
            f'<div class="muted">手順: <code>{html.escape(_rel_url(out_dir, sec.runbook_path))}</code></div>'
        )

        # 条件分岐: `sec.metrics` を満たす経路を評価する。
        if sec.metrics:
            parts.append("<table>")
            parts.append("<tr><th>指標</th><th>値</th></tr>")
            for k, v in sec.metrics.items():
                parts.append(
                    "<tr>"
                    f"<td><code>{html.escape(str(k))}</code></td>"
                    f"<td><code>{html.escape(str(v))}</code></td>"
                    "</tr>"
                )

            parts.append("</table>")

        # 条件分岐: `sec.notes` を満たす経路を評価する。

        if sec.notes:
            parts.append("<ul>")
            for n in sec.notes:
                parts.append(f"<li class='muted'>{html.escape(n)}</li>")

            parts.append("</ul>")

        # 条件分岐: `not sec.images` を満たす経路を評価する。

        if not sec.images:
            parts.append("<p class='muted'>画像が見つかりません。</p>")
        else:
            for caption, img_path in sec.images:
                rel = _rel_url(out_dir, img_path)
                parts.append("<figure>")
                parts.append(f"<figcaption class='muted'>{html.escape(caption)}: <code>{html.escape(rel)}</code></figcaption>")
                parts.append(f"<a href='{html.escape(rel)}'><img src='{html.escape(rel)}' loading='lazy'></a>")
                parts.append("</figure>")

        parts.append("</section>")

    parts.append("</body></html>")

    html_path.write_text("\n".join(parts), encoding="utf-8")
    return html_path


# 関数: `_render_dashboard_png` の入出力契約と処理意図を定義する。

def _render_dashboard_png(sections: List[Section], out_dir: Path) -> Optional[Path]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None

    _set_japanese_font()

    panels: List[Tuple[str, Path]] = []
    for sec in sections:
        # 条件分岐: `sec.images` を満たす経路を評価する。
        if sec.images:
            panels.append((sec.title, sec.images[0][1]))

    panels = [(t, p) for t, p in panels if p.exists()]
    # 条件分岐: `not panels` を満たす経路を評価する。
    if not panels:
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "pmodel_dashboard.png"

    n = len(panels)
    ncols = 2
    nrows = int(math.ceil(n / ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 5.2 * nrows))
    axes_flat = list(fig.axes)

    for ax in axes_flat:
        ax.axis("off")

    for i, (title, img_path) in enumerate(panels):
        ax = axes_flat[i]
        ax.set_title(title, fontsize=12)
        try:
            img = plt.imread(str(img_path))
            ax.imshow(img)
        except Exception as e:
            ax.text(0.02, 0.98, f"Failed to load: {img_path}\n{e}", transform=ax.transAxes, va="top")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


# 関数: `main` の入出力契約と処理意図を定義する。

def main() -> int:
    root = _repo_root()
    default_out = root / "output" / "private" / "summary"

    ap = argparse.ArgumentParser(description="Generate a single-page report of P-model comparisons.")
    ap.add_argument(
        "--outdir",
        type=str,
        default=str(default_out),
        help="Output directory (default: output/private/summary)",
    )
    ap.add_argument("--open", action="store_true", help="Open the HTML report after generation (Windows only).")
    ap.add_argument("--no-dashboard", action="store_true", help="Do not generate dashboard PNG.")
    args = ap.parse_args()

    out_dir = Path(args.outdir)
    sections = [
        _run_all_status_section(root, out_dir),
        _public_section(root),
        _theory_section(root),
        _cassini_section(root),
        _viking_section(root),
        _llr_section(root),
        _gps_section(root),
        _mercury_section(root),
    ]

    html_path = _render_html(sections, out_dir)
    print(f"[ok] html: {html_path}")

    dashboard = None
    # 条件分岐: `not args.no_dashboard` を満たす経路を評価する。
    if not args.no_dashboard:
        dashboard = _render_dashboard_png(sections, out_dir)
        # 条件分岐: `dashboard` を満たす経路を評価する。
        if dashboard:
            print(f"[ok] dashboard: {dashboard}")
        else:
            print("[skip] dashboard: matplotlib not available or no images found")

    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "html": str(html_path),
        "dashboard": str(dashboard) if dashboard else None,
        "sections": [
            {
                "key": s.key,
                "title": s.title,
                "runbook": str(s.runbook_path),
                "images": [{"caption": c, "path": str(p)} for c, p in s.images],
                "metrics": s.metrics,
                "notes": s.notes,
            }
            for s in sections
        ],
    }
    json_path = out_dir / "pmodel_report_summary.json"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] summary: {json_path}")

    # 条件分岐: `args.open` を満たす経路を評価する。
    if args.open:
        # 条件分岐: `os.name == "nt"` を満たす経路を評価する。
        if os.name == "nt":
            try:
                os.startfile(str(html_path))  # type: ignore[attr-defined]
            except Exception as e:
                print(f"[warn] failed to open: {e}", file=sys.stderr)
        else:
            print("[warn] --open is supported on Windows only.", file=sys.stderr)

    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())
