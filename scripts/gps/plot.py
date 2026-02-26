from __future__ import annotations

import csv
import json
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


ROOT = Path(__file__).resolve().parents[2]
# 条件分岐: `str(ROOT) not in sys.path` を満たす経路を評価する。
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.summary import worklog  # noqa: E402

OUT_DIR = ROOT / "output" / "private" / "gps"


# 関数: `_to_ns_from_m` の入出力契約と処理意図を定義する。
def _to_ns_from_m(rms_m: float) -> float:
    return (rms_m / C) * 1e9


# 関数: `load_summary` の入出力契約と処理意図を定義する。

def load_summary(summary_csv: Path) -> List[Dict[str, str]]:
    with open(summary_csv, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        return list(r)


# 関数: `plot_all_residuals_brdc` の入出力契約と処理意図を定義する。

def plot_all_residuals_brdc(sats: List[str]) -> Path:
    _set_japanese_font()
    plt.figure(figsize=(12, 7))

    count = 0
    for sat in sats:
        filename = OUT_DIR / f"residual_precise_{sat}.csv"
        # 条件分岐: `not filename.exists()` を満たす経路を評価する。
        if not filename.exists():
            continue

        try:
            df = pd.read_csv(filename)
            df["time_utc"] = pd.to_datetime(df["time_utc"])
            plt.plot(df["time_utc"], df["res_brdc_s"] * 1e9, label=sat, linewidth=0.8, alpha=0.6)
            count += 1
        except Exception as e:
            print(f"[warn] failed to read {filename}: {e}")

    plt.title("GPS 放送暦 時計残差（BRDC - IGS, 全衛星）", fontsize=16)
    plt.xlabel("UTC時刻", fontsize=12)
    plt.ylabel("時計残差 [ns]", fontsize=12)
    plt.axhline(0, color="black", linestyle="-", linewidth=0.8)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize="small", ncol=2, borderaxespad=0.0)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    plt.gcf().autofmt_xdate()
    plt.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_png = OUT_DIR / "gps_clock_residuals_all_31.png"
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"[ok] {out_png} (plotted {count} sats)")
    return out_png


# 関数: `plot_residual_compare_g01` の入出力契約と処理意図を定義する。

def plot_residual_compare_g01() -> Optional[Path]:
    path = OUT_DIR / "residual_precise_G01.csv"
    # 条件分岐: `not path.exists()` を満たす経路を評価する。
    if not path.exists():
        return None

    df = pd.read_csv(path)
    df["time_utc"] = pd.to_datetime(df["time_utc"])

    _set_japanese_font()
    plt.figure(figsize=(10.5, 5.2))
    plt.plot(df["time_utc"], df["res_brdc_s"] * 1e9, label="放送暦（BRDC）- IGS", linewidth=1.6)
    # 条件分岐: `"res_pmodel_s" in df.columns` を満たす経路を評価する。
    if "res_pmodel_s" in df.columns:
        plt.plot(
            df["time_utc"],
            df["res_pmodel_s"] * 1e9,
            label="P-model（dt_rel除去）- IGS",
            linewidth=1.6,
        )

    plt.title("GPS 時計残差：G01（観測IGSに対する比較）", fontsize=14)
    plt.xlabel("UTC時刻")
    plt.ylabel("残差 [ns]（バイアス＋ドリフト除去後）")
    plt.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
    plt.grid(True, alpha=0.3)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    plt.gcf().autofmt_xdate()
    plt.legend()
    plt.tight_layout()

    out_png = OUT_DIR / "gps_residual_compare_G01.png"
    plt.savefig(out_png, dpi=220)
    plt.close()
    print(f"[ok] {out_png}")
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

    _set_japanese_font()
    fig, ax = plt.subplots(figsize=(13.0, 5.2))
    x = range(len(labels))
    w = 0.42
    ax.bar([i - w / 2 for i in x], b_vals, width=w, label="放送暦（BRDC）- IGS（RMS）")
    ax.bar([i + w / 2 for i in x], p_vals, width=w, label="P-model（dt_rel除去）- IGS（RMS）")
    ax.set_title("GPS：観測IGSに対する残差RMS（全衛星）", fontsize=14)
    ax.set_ylabel("RMS [ns]（バイアス＋ドリフト除去後）")
    ax.set_xlabel("衛星PRN")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8.5)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()

    out_png = OUT_DIR / "gps_rms_compare.png"
    fig.savefig(out_png, dpi=220)
    plt.close(fig)
    print(f"[ok] {out_png}")
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

    _set_japanese_font()
    fig, ax = plt.subplots(figsize=(10.8, 5.2))
    ax.plot(df["time_utc"], rel_det * 1e9, label="標準式 δt_rel（-2 r·v / c^2）", linewidth=2.0)
    ax.plot(df["time_utc"], p_det * 1e9, label="P-model（dτ/dt を積分, バイアス＋ドリフト除去）", linewidth=2.0)
    ax.set_title(f"GPS：相対補正（近日点効果） {prn}\n標準式 vs P-model（同じ周期成分）", fontsize=14)
    ax.set_xlabel("UTC時刻")
    ax.set_ylabel("時間補正 [ns]（バイアス＋ドリフト除去後）")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    fig.autofmt_xdate()
    fig.tight_layout()

    out_png = OUT_DIR / f"gps_relativistic_correction_{prn}.png"
    fig.savefig(out_png, dpi=220)
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
