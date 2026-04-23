"""
目的: GPS topic の plot に対応する公開図・表・監査指標を再生成する。
入力: script 内の既定パラメータと必要な公開データまたは基準値を用いる。
出力: output/public と output/private の canonical artifact を更新する。
前提: 論文本文と README はこの script が出力する公開成果物を正として参照する。
"""

from __future__ import annotations

import csv
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


C = 299_792_458.0


# 関数: `_set_japanese_font` の入出力契約と処理意図を定義する。
def _set_japanese_font() -> None:
    try:
        import matplotlib as mpl
        import matplotlib.font_manager as fm

        resolved = resolve_wavep_cjk_font_family()
        preferred = [
            *( [resolved] if resolved else [] ),
            "Noto Sans CJK JP",
            "Noto Sans JP",
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
        mpl.rcParams["font.sans-serif"] = chosen + ["DejaVu Sans"]
        mpl.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass


ROOT = Path(__file__).resolve().parents[2]
# 条件分岐: `str(ROOT) not in sys.path` を満たす経路を評価する。
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.summary import worklog  # noqa: E402
from scripts.utils.figure_locale_paths import localize_figure_output_path  # noqa: E402
from scripts.utils.plot_style import apply_paper_style, apply_wavep_figure_layout, get_wavep_font_size, resolve_wavep_cjk_font_family  # noqa: E402
from scripts.quantum.figure_japanese_localizer import get_figure_language  # noqa: E402

OUT_DIR = ROOT / "output" / "private" / "gps"
OUT_PUBLIC_DIR = ROOT / "output" / "public" / "gps"


# 関数: `_plot_text` の入出力契約と処理意図を定義する。
def _plot_text(ja: str, en: str, *, lang: str) -> str:
    return ja if lang == "ja" else en


# 関数: `_to_ns_from_m` の入出力契約と処理意図を定義する。

def _to_ns_from_m(rms_m: float) -> float:
    return (rms_m / C) * 1e9


# 関数: `_save_dual_figure` の入出力契約と処理意図を定義する。

def _save_dual_figure(fig: plt.Figure, *, stem: str, dpi_png: int) -> tuple[Path, Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
    figure_lang = get_figure_language(default="ja")
    out_png = localize_figure_output_path(OUT_DIR / f"{stem}.png", root=ROOT, locale=figure_lang)
    out_pdf = localize_figure_output_path(OUT_DIR / f"{stem}.pdf", root=ROOT, locale=figure_lang)
    public_png = localize_figure_output_path(OUT_PUBLIC_DIR / out_png.name, root=ROOT, locale=figure_lang)
    public_pdf = localize_figure_output_path(OUT_PUBLIC_DIR / out_pdf.name, root=ROOT, locale=figure_lang)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    public_png.parent.mkdir(parents=True, exist_ok=True)
    public_pdf.parent.mkdir(parents=True, exist_ok=True)

    with plt.rc_context({"savefig.bbox": None, "savefig.pad_inches": 0.0}):
        fig.savefig(out_png, dpi=int(dpi_png))
        fig.savefig(out_pdf)

    shutil.copy2(out_png, public_png)
    shutil.copy2(out_pdf, public_pdf)
    return (out_png, out_pdf)


# 関数: `load_summary` の入出力契約と処理意図を定義する。

def load_summary(summary_csv: Path) -> List[Dict[str, str]]:
    with open(summary_csv, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        return list(r)


# 関数: `plot_all_residuals_brdc` の入出力契約と処理意図を定義する。

def plot_all_residuals_brdc(sats: List[str]) -> Path:
    figure_lang = get_figure_language(default="ja")
    if figure_lang == "ja":
        _set_japanese_font()

    apply_paper_style()
    fig, ax = plt.subplots()
    layout_template = "part2_single_panel_legend_bottom" if figure_lang == "en" else "part2_single_panel_tall"
    apply_wavep_figure_layout(fig, template=layout_template)
    title_font = get_wavep_font_size("title", name="part2_astrophysics") * (0.98 if figure_lang == "en" else 1.0)
    axis_font = get_wavep_font_size("axis", name="part2_astrophysics")
    tick_font = get_wavep_font_size("tick", name="part2_astrophysics")
    legend_font = get_wavep_font_size("legend", name="part2_astrophysics")

    count = 0
    for sat in sats:
        filename = OUT_DIR / f"residual_precise_{sat}.csv"
        # 条件分岐: `not filename.exists()` を満たす経路を評価する。
        if not filename.exists():
            continue

        try:
            df = pd.read_csv(filename)
            df["time_utc"] = pd.to_datetime(df["time_utc"])
            ax.plot(df["time_utc"], df["res_brdc_s"] * 1e9, label=sat, linewidth=1.00, alpha=0.68)
            count += 1
        except Exception as e:
            print(f"[warn] failed to read {filename}: {e}")

    ax.set_title(
        _plot_text(
            "GPS 放送暦時計残差（BRDC - IGS, 全衛星）",
            "GPS broadcast-clock residuals\n(BRDC - IGS, all satellites)",
            lang=figure_lang,
        ),
        fontsize=title_font,
        pad=10.0,
    )
    ax.set_ylabel(_plot_text("時計残差 [ns]", "Clock residual [ns]", lang=figure_lang), fontsize=axis_font)
    ax.axhline(0, color="black", linestyle="-", linewidth=0.8)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.tick_params(axis="both", labelsize=tick_font)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    for tick_label in ax.get_xticklabels():
        tick_label.set_rotation(20)
        tick_label.set_ha("right")

    if figure_lang == "en":
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.16),
            ncol=8,
            frameon=True,
            framealpha=0.95,
            fontsize=legend_font,
            columnspacing=0.9,
            handlelength=1.4,
            handletextpad=0.4,
            borderaxespad=0.0,
            labelspacing=0.35,
        )

    fig.subplots_adjust(left=0.115, right=0.985, top=0.88, bottom=(0.285 if figure_lang == "en" else 0.13))

    out_png, out_pdf = _save_dual_figure(fig, stem="gps_clock_residuals_all_31", dpi_png=300)

    plt.close(fig)
    print(f"[ok] {out_png} (plotted {count} sats)")
    print(f"[ok] {out_pdf} (plotted {count} sats)")
    return out_png


# 関数: `plot_residual_compare_g01` の入出力契約と処理意図を定義する。

def plot_residual_compare_g01() -> Optional[Path]:
    path = OUT_DIR / "residual_precise_G01.csv"
    # 条件分岐: `not path.exists()` を満たす経路を評価する。
    if not path.exists():
        return None

    df = pd.read_csv(path)
    df["time_utc"] = pd.to_datetime(df["time_utc"])

    apply_paper_style()
    _set_japanese_font()
    fig, ax = plt.subplots()
    apply_wavep_figure_layout(fig, template="part2_single_panel")
    ax.plot(df["time_utc"], df["res_brdc_s"] * 1e9, label="放送暦（BRDC）- IGS", linewidth=2.1)
    # 条件分岐: `"res_pmodel_s" in df.columns` を満たす経路を評価する。
    if "res_pmodel_s" in df.columns:
        ax.plot(
            df["time_utc"],
            df["res_pmodel_s"] * 1e9,
            label="P-model（dt_rel除去）- IGS",
            linewidth=2.1,
        )

    ax.set_title("GPS 時計残差: G01（観測 IGS に対する比較）", pad=6.0)
    ax.set_xlabel("UTC時刻")
    ax.set_ylabel("残差 [ns]（バイアス＋ドリフト除去後）")
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    fig.autofmt_xdate()
    ax.legend(
        loc="upper right",
        frameon=True,
        borderpad=0.6,
        labelspacing=0.35,
        handlelength=2.1,
    )
    out_png, out_pdf = _save_dual_figure(fig, stem="gps_residual_compare_G01", dpi_png=220)
    plt.close(fig)
    print(f"[ok] {out_png}")
    print(f"[ok] {out_pdf}")
    return out_png


# 関数: `plot_rms_compare` の入出力契約と処理意図を定義する。

def plot_rms_compare(summary_rows: List[Dict[str, str]]) -> Tuple[Optional[Path], Dict[str, float]]:
    rms_b_ns: List[Tuple[str, float]] = []
    rms_p_ns: List[Tuple[str, float]] = []

    for row in summary_rows:
        prn = (row.get("PRN") or "").strip()
        # 条件分岐: `not prn` を満たす経路を評価する。
        if not prn:
            continue

        try:
            rms_b_m = float(row.get("RMS_BRDC_m") or "nan")
        except Exception:
            continue

        rms_b_ns.append((prn, _to_ns_from_m(rms_b_m)))

        rms_p_m_raw = row.get("RMS_PMODEL_m")
        # 条件分岐: `rms_p_m_raw is not None and str(rms_p_m_raw).strip() != ""` を満たす経路を評価する。
        if rms_p_m_raw is not None and str(rms_p_m_raw).strip() != "":
            try:
                rms_p_m = float(rms_p_m_raw)
                rms_p_ns.append((prn, _to_ns_from_m(rms_p_m)))
            except Exception:
                pass

    rms_b_ns.sort(key=lambda x: x[0])
    rms_p_ns.sort(key=lambda x: x[0])

    metrics: Dict[str, float] = {
        "n_sats": float(len(rms_b_ns)),
    }
    # 条件分岐: `rms_b_ns` を満たす経路を評価する。
    if rms_b_ns:
        b_vals = [v for _, v in rms_b_ns]
        b_sorted = sorted(b_vals)
        metrics["brdc_rms_ns_median"] = b_sorted[len(b_sorted) // 2]
        metrics["brdc_rms_ns_max"] = max(b_sorted)

    # 条件分岐: `rms_p_ns` を満たす経路を評価する。

    if rms_p_ns:
        p_vals = [v for _, v in rms_p_ns]
        p_sorted = sorted(p_vals)
        metrics["pmodel_rms_ns_median"] = p_sorted[len(p_sorted) // 2]
        metrics["pmodel_rms_ns_max"] = max(p_sorted)

        p_map = {k: v for k, v in rms_p_ns}
        b_map = {k: v for k, v in rms_b_ns}
        better = 0
        worse = 0
        for prn, b in b_map.items():
            p = p_map.get(prn)
            # 条件分岐: `p is None` を満たす経路を評価する。
            if p is None:
                continue

            # 条件分岐: `p < b` を満たす経路を評価する。

            if p < b:
                better += 1
            # 条件分岐: 前段条件が不成立で、`p > b` を追加評価する。
            elif p > b:
                worse += 1

        metrics["pmodel_better_count"] = float(better)
        metrics["brdc_better_count"] = float(worse)

    # 条件分岐: `not rms_b_ns or not rms_p_ns` を満たす経路を評価する。

    if not rms_b_ns or not rms_p_ns:
        return None, metrics

    labels = [k for k, _ in rms_b_ns]
    b_vals = [v for _, v in rms_b_ns]
    p_map = {k: v for k, v in rms_p_ns}
    p_vals = [p_map.get(k, float("nan")) for k in labels]

    apply_paper_style()
    _set_japanese_font()
    fig, ax = plt.subplots()
    apply_wavep_figure_layout(fig, template="part2_single_panel")
    x = range(len(labels))
    w = 0.42
    ax.bar([i - w / 2 for i in x], b_vals, width=w, label="放送暦（BRDC）- IGS（RMS）")
    ax.bar([i + w / 2 for i in x], p_vals, width=w, label="P-model（dt_rel除去）- IGS（RMS）")
    ax.set_title("GPS: 観測 IGS に対する残差 RMS（全衛星）", pad=6.0)
    ax.set_ylabel("RMS [ns]（バイアス＋ドリフト除去後）")
    ax.set_xlabel("衛星PRN")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=60, ha="right")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper right", framealpha=0.95)
    out_png, out_pdf = _save_dual_figure(fig, stem="gps_rms_compare", dpi_png=220)
    plt.close(fig)
    print(f"[ok] {out_png}")
    print(f"[ok] {out_pdf}")
    return out_png, metrics


# 関数: `_detrend_affine` の入出力契約と処理意図を定義する。

def _detrend_affine(t_s: "np.ndarray", y: "np.ndarray") -> "np.ndarray":
    # y - (a + b t)
    if len(t_s) < 2:
        return y.copy()

    t0 = t_s[0]
    tt = t_s - t0
    A = np.vstack([np.ones_like(tt), tt]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    a, b = coef
    return y - (a + b * tt)


# 関数: `_dt_rel_from_r` の入出力契約と処理意図を定義する。

def _dt_rel_from_r(t_s: "np.ndarray", r_m: "np.ndarray") -> "np.ndarray":
    # Standard GNSS relativistic correction (eccentricity term) can be written as:
    #   dt_rel = -2 (r·v)/c^2
    # and r·v = r * dr/dt (since dr/dt = (r·v)/r). We estimate dr/dt by finite differences.
    n = len(r_m)
    # 条件分岐: `n < 2` を満たす経路を評価する。
    if n < 2:
        return np.zeros_like(r_m)

    drdt = np.zeros_like(r_m, dtype=float)
    drdt[0] = (r_m[1] - r_m[0]) / (t_s[1] - t_s[0])
    drdt[-1] = (r_m[-1] - r_m[-2]) / (t_s[-1] - t_s[-2])
    # 条件分岐: `n >= 3` を満たす経路を評価する。
    if n >= 3:
        drdt[1:-1] = (r_m[2:] - r_m[:-2]) / (t_s[2:] - t_s[:-2])

    rv = r_m * drdt
    return (-2.0 * rv) / (C * C)


# 関数: `plot_relativistic_correction_example` の入出力契約と処理意図を定義する。

def plot_relativistic_correction_example(prn: str = "G02") -> Tuple[Optional[Path], Dict[str, float]]:
    path = OUT_DIR / f"residual_precise_{prn}.csv"
    # 条件分岐: `not path.exists()` を満たす経路を評価する。
    if not path.exists():
        return None, {}

    df = pd.read_csv(path)
    df["time_utc"] = pd.to_datetime(df["time_utc"])

    # 条件分岐: `"pmodel_clk_s" not in df.columns or "r_m" not in df.columns or "tsec" not in...` を満たす経路を評価する。
    if "pmodel_clk_s" not in df.columns or "r_m" not in df.columns or "tsec" not in df.columns:
        return None, {}

    tsec = df["tsec"].to_numpy(dtype=float)
    r_m = df["r_m"].to_numpy(dtype=float)
    pmodel_clk = df["pmodel_clk_s"].to_numpy(dtype=float)

    dt_rel = _dt_rel_from_r(tsec, r_m)

    # Compare periodic components (remove bias+drift).
    p_det = _detrend_affine(tsec, pmodel_clk)
    rel_det = _detrend_affine(tsec, dt_rel)

    # Metrics
    def _corr(a: "np.ndarray", b: "np.ndarray") -> float:
        # 条件分岐: `len(a) < 2` を満たす経路を評価する。
        if len(a) < 2:
            return float("nan")

        aa = a - float(np.mean(a))
        bb = b - float(np.mean(b))
        da = float(np.sqrt(np.sum(aa * aa)))
        db = float(np.sqrt(np.sum(bb * bb)))
        # 条件分岐: `da == 0.0 or db == 0.0` を満たす経路を評価する。
        if da == 0.0 or db == 0.0:
            return float("nan")

        return float(np.sum(aa * bb) / (da * db))

    rmse_s = float(np.sqrt(np.mean((p_det - rel_det) ** 2)))
    metrics = {
        "prn": prn,
        "corr": _corr(p_det, rel_det),
        "rmse_ns": rmse_s * 1e9,
        "pmodel_peak_to_peak_ns": float((np.max(p_det) - np.min(p_det)) * 1e9),
        "dt_rel_peak_to_peak_ns": float((np.max(rel_det) - np.min(rel_det)) * 1e9),
    }

    apply_paper_style()
    figure_lang = get_figure_language(default="ja")
    if figure_lang == "ja":
        _set_japanese_font()

    fig, ax = plt.subplots()
    apply_wavep_figure_layout(fig, template="part2_single_panel")
    ax.plot(df["time_utc"], rel_det * 1e9, label=_plot_text("標準式 δt_rel（-2 r·v / c^2）", "Standard formula δt_rel (-2 r·v / c^2)", lang=figure_lang), linewidth=2.0)
    ax.plot(df["time_utc"], p_det * 1e9, label=_plot_text("P-model（dτ/dt を積分, バイアス＋ドリフト除去）", "P-model (integrated dτ/dt,\nbias+drift removed)", lang=figure_lang), linewidth=2.0)
    ax.set_title(_plot_text(f"GPS: 相対補正（近日点効果） {prn}", f"GPS: relativistic correction\n(perigee effect) {prn}", lang=figure_lang), pad=6.0)
    ax.set_xlabel(_plot_text("UTC時刻", "UTC time", lang=figure_lang))
    ax.set_ylabel(_plot_text("時間補正 [ns]（バイアス＋ドリフト除去後）", "Time correction [ns]\n(after bias+drift removal)", lang=figure_lang))
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", framealpha=0.95)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    fig.autofmt_xdate()
    out_png, _ = _save_dual_figure(fig, stem=f"gps_relativistic_correction_{prn}", dpi_png=220)
    plt.close(fig)
    print(f"[ok] {out_png}")
    return out_png, metrics


# 関数: `main` の入出力契約と処理意図を定義する。

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Determine satellites from residual files (fallback to G01.. patterns).
    sats = sorted([p.stem.replace("residual_precise_", "") for p in OUT_DIR.glob("residual_precise_G*.csv")])
    # 条件分岐: `not sats` を満たす経路を評価する。
    if not sats:
        sats = [
            "G01",
            "G02",
            "G03",
            "G04",
            "G05",
            "G06",
            "G07",
            "G08",
            "G09",
            "G10",
            "G11",
            "G12",
            "G13",
            "G14",
            "G15",
            "G16",
            "G17",
            "G18",
            "G19",
            "G21",
            "G22",
            "G23",
            "G24",
            "G25",
            "G26",
            "G27",
            "G28",
            "G29",
            "G30",
            "G31",
            "G32",
        ]

    plot_all_residuals_brdc(sats)
    plot_residual_compare_g01()

    summary_csv = OUT_DIR / "summary_batch.csv"
    summary_rows = load_summary(summary_csv) if summary_csv.exists() else []
    rms_png, metrics = plot_rms_compare(summary_rows)
    rel_png, rel_metrics = plot_relativistic_correction_example("G02")
    # 条件分岐: `rel_metrics` を満たす経路を評価する。
    if rel_metrics:
        metrics.update({f"rel_{k}": v for k, v in rel_metrics.items() if k != "prn"})

    out = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "outputs": {
            "gps_clock_residuals_all_31_png": str(OUT_DIR / "gps_clock_residuals_all_31.png"),
            "gps_residual_compare_G01_png": str(OUT_DIR / "gps_residual_compare_G01.png"),
            "gps_rms_compare_png": str(rms_png) if rms_png else None,
            "gps_relativistic_correction_G02_png": str(rel_png) if rel_png else None,
        },
        "metrics": metrics,
        "notes": [
            "IGS Final CLK/SP3 を観測（準実測）として使用。",
            "各系列はバイアス＋ドリフト（一次）を最小二乗で除去した残差を表示。",
            "IGSの衛星クロックは慣例的に相対補正（-2 r·v / c^2）を別扱いにするため、P-model側も dt_rel を除去して比較しています。",
            "dt_rel 自体（標準式）と P-model の周期成分が一致することは別図で確認できます。",
        ],
    }
    json_path = OUT_DIR / "gps_compare_metrics.json"
    json_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] {json_path}")

    try:
        worklog.append_event(
            {
                "event_type": "gps_plot",
                "argv": sys.argv,
                "metrics": {
                    "n_sats": float(metrics.get("n_sats") or 0.0),
                    "brdc_rms_ns_median": float(metrics.get("brdc_rms_ns_median") or 0.0),
                    "pmodel_rms_ns_median": float(metrics.get("pmodel_rms_ns_median") or 0.0),
                },
                "outputs": {
                    "gps_compare_metrics_json": json_path,
                    "gps_rms_compare_png": rms_png,
                    "gps_clock_residuals_all_png": OUT_DIR / "gps_clock_residuals_all_31.png",
                    "gps_residual_compare_g01_png": OUT_DIR / "gps_residual_compare_G01.png",
                    "gps_relativistic_correction_g02_png": rel_png,
                },
            }
        )
    except Exception:
        pass


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    main()
