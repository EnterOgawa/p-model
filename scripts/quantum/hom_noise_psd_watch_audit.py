from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sys
import zipfile
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
# 条件分岐: `str(ROOT) not in sys.path` を満たす経路を評価する。
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.summary import worklog


# 関数: `_read_json` の入出力契約と処理意図を定義する。
def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


# 関数: `_write_json` の入出力契約と処理意図を定義する。

def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


# 関数: `_fmt_float` の入出力契約と処理意図を定義する。

def _fmt_float(x: float, digits: int = 6) -> str:
    # 条件分岐: `x == 0.0` を満たす経路を評価する。
    if x == 0.0:
        return "0"

    ax = abs(x)
    # 条件分岐: `ax >= 1e4 or ax < 1e-3` を満たす経路を評価する。
    if ax >= 1e4 or ax < 1e-3:
        return f"{x:.{digits}g}"

    return f"{x:.{digits}f}".rstrip("0").rstrip(".")


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
        return


# 関数: `_read_psd_from_zip` の入出力契約と処理意図を定義する。

def _read_psd_from_zip(zip_path: Path, csv_name: str) -> Tuple[np.ndarray, np.ndarray]:
    freq: List[float] = []
    psd: List[float] = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        with zf.open(csv_name, "r") as f:
            lines = f.read().decode("utf-8", "replace").splitlines()

    for line in lines[1:]:
        # 条件分岐: `not line.strip()` を満たす経路を評価する。
        if not line.strip():
            continue

        parts = line.split(",")
        # 条件分岐: `len(parts) != 2` を満たす経路を評価する。
        if len(parts) != 2:
            continue

        try:
            freq.append(float(parts[0]))
            psd.append(float(parts[1]))
        except Exception:
            continue

    return np.asarray(freq, dtype=float), np.asarray(psd, dtype=float)


# 関数: `_interp_log_psd` の入出力契約と処理意図を定義する。

def _interp_log_psd(freq_hz: np.ndarray, psd: np.ndarray, target_hz: float) -> Optional[float]:
    mask = (freq_hz > 0) & (psd > 0)
    # 条件分岐: `np.count_nonzero(mask) < 3 or target_hz <= 0` を満たす経路を評価する。
    if np.count_nonzero(mask) < 3 or target_hz <= 0:
        return None

    xf = np.log10(freq_hz[mask])
    yf = np.log10(psd[mask])
    xt = math.log10(float(target_hz))
    val = float(10.0 ** np.interp(xt, xf, yf))
    # 条件分岐: `not math.isfinite(val) or val <= 0` を満たす経路を評価する。
    if not math.isfinite(val) or val <= 0:
        return None

    return val


# 関数: `_fit_loglog_trend` の入出力契約と処理意図を定義する。

def _fit_loglog_trend(freq_hz: np.ndarray, psd: np.ndarray, fmin: float, fmax: float) -> Optional[Tuple[float, float]]:
    mask = (freq_hz >= fmin) & (freq_hz <= fmax) & (freq_hz > 0) & (psd > 0)
    # 条件分岐: `np.count_nonzero(mask) < 10` を満たす経路を評価する。
    if np.count_nonzero(mask) < 10:
        return None

    x = np.log10(freq_hz[mask])
    y = np.log10(psd[mask])
    slope, intercept = np.polyfit(x, y, 1)
    # 条件分岐: `not (math.isfinite(float(slope)) and math.isfinite(float(intercept)))` を満たす経路を評価する。
    if not (math.isfinite(float(slope)) and math.isfinite(float(intercept))):
        return None

    return float(slope), float(intercept)


# 関数: `_trend_value` の入出力契約と処理意図を定義する。

def _trend_value(target_hz: float, trend: Tuple[float, float]) -> Optional[float]:
    # 条件分岐: `target_hz <= 0` を満たす経路を評価する。
    if target_hz <= 0:
        return None

    slope, intercept = trend
    y = slope * math.log10(float(target_hz)) + intercept
    v = float(10.0 ** y)
    # 条件分岐: `not math.isfinite(v) or v <= 0` を満たす経路を評価する。
    if not math.isfinite(v) or v <= 0:
        return None

    return v


# 関数: `_band_median` の入出力契約と処理意図を定義する。

def _band_median(freq_hz: np.ndarray, psd: np.ndarray, fmin: float, fmax: float) -> Optional[float]:
    mask = (freq_hz >= fmin) & (freq_hz <= fmax) & (psd > 0)
    # 条件分岐: `np.count_nonzero(mask) < 3` を満たす経路を評価する。
    if np.count_nonzero(mask) < 3:
        return None

    val = float(np.median(psd[mask]))
    return val if math.isfinite(val) and val > 0 else None


# 関数: `_run_audit` の入出力契約と処理意図を定義する。

def _run_audit(
    *,
    hom_metrics_json: Path,
    psd_zip: Path,
    psd_csv_name: str,
    ratio_threshold: float,
) -> Dict[str, Any]:
    hom = _read_json(hom_metrics_json)
    freq_hz, psd = _read_psd_from_zip(psd_zip, psd_csv_name)

    p10 = _interp_log_psd(freq_hz, psd, 1.0e4)
    p100 = _interp_log_psd(freq_hz, psd, 1.0e5)
    baseline_ratio = None if p10 is None or p100 is None else float(p10 / p100)

    trend = _fit_loglog_trend(freq_hz, psd, 3.0e4, 3.0e5)
    detrended_ratio = None
    # 条件分岐: `trend is not None and p10 is not None and p100 is not None` を満たす経路を評価する。
    if trend is not None and p10 is not None and p100 is not None:
        t10 = _trend_value(1.0e4, trend)
        t100 = _trend_value(1.0e5, trend)
        # 条件分岐: `t10 is not None and t100 is not None and t10 > 0 and t100 > 0` を満たす経路を評価する。
        if t10 is not None and t100 is not None and t10 > 0 and t100 > 0:
            detrended_ratio = float((p10 / t10) / (p100 / t100))

    low_centers = [8.0e3, 1.0e4, 1.2e4, 1.5e4]
    high_centers = [8.0e4, 1.0e5, 1.2e5, 1.5e5]
    pair_rows: List[Dict[str, Any]] = []
    for low in low_centers:
        for high in high_centers:
            pl = _interp_log_psd(freq_hz, psd, low)
            ph = _interp_log_psd(freq_hz, psd, high)
            ratio_raw = None if pl is None or ph is None or ph <= 0 else float(pl / ph)
            ratio_detrended = None
            # 条件分岐: `trend is not None and pl is not None and ph is not None and ph > 0` を満たす経路を評価する。
            if trend is not None and pl is not None and ph is not None and ph > 0:
                tl = _trend_value(low, trend)
                th = _trend_value(high, trend)
                # 条件分岐: `tl is not None and th is not None and tl > 0 and th > 0` を満たす経路を評価する。
                if tl is not None and th is not None and tl > 0 and th > 0:
                    ratio_detrended = float((pl / tl) / (ph / th))

            pair_rows.append(
                {
                    "low_hz": float(low),
                    "high_hz": float(high),
                    "ratio_raw": ratio_raw,
                    "ratio_detrended": ratio_detrended,
                    "raw_ge_threshold": (ratio_raw is not None and ratio_raw >= ratio_threshold),
                    "detrended_ge_threshold": (
                        ratio_detrended is not None and ratio_detrended >= ratio_threshold
                    ),
                }
            )

    raw_vals = [float(r["ratio_raw"]) for r in pair_rows if isinstance(r.get("ratio_raw"), (int, float))]
    det_vals = [float(r["ratio_detrended"]) for r in pair_rows if isinstance(r.get("ratio_detrended"), (int, float))]

    raw_med = float(np.median(raw_vals)) if raw_vals else None
    det_med = float(np.median(det_vals)) if det_vals else None
    raw_pass_n = sum(1 for r in pair_rows if r.get("raw_ge_threshold") is True)
    det_pass_n = sum(1 for r in pair_rows if r.get("detrended_ge_threshold") is True)
    n_pairs = len(pair_rows)

    low_band = _band_median(freq_hz, psd, 8.0e3, 1.2e4)
    high_band = _band_median(freq_hz, psd, 8.0e4, 1.2e5)
    band_median_ratio = None if low_band is None or high_band is None else float(low_band / high_band)

    hard_gate_applicable = False
    decision = "keep_watch_nonhard_gate"
    # 条件分岐: `baseline_ratio is not None and baseline_ratio >= ratio_threshold and det_med...` を満たす経路を評価する。
    if baseline_ratio is not None and baseline_ratio >= ratio_threshold and det_med is not None and det_med >= ratio_threshold:
        decision = "pass_after_drift_and_band_audit"

    hom_rows = hom.get("rows") if isinstance(hom.get("rows"), list) else []
    old_noise_row = next(
        (r for r in hom_rows if isinstance(r, dict) and str(r.get("channel") or "") == "noise_psd_shape"),
        {},
    )
    old_ratio = old_noise_row.get("metric_value")
    old_ratio = float(old_ratio) if isinstance(old_ratio, (int, float)) else None

    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "schema": "wavep.quantum.hom_noise_psd_watch_audit.v1",
        "inputs": {
            "hom_squeezed_light_unified_audit_metrics_json": str(hom_metrics_json).replace("\\", "/"),
            "psd_zip": str(psd_zip).replace("\\", "/"),
            "psd_csv_name": psd_csv_name,
        },
        "thresholds": {
            "lf_to_hf_ratio_min": float(ratio_threshold),
            "trend_fit_band_hz": [3.0e4, 3.0e5],
            "pair_low_centers_hz": low_centers,
            "pair_high_centers_hz": high_centers,
            "band_median_low_hz": [8.0e3, 1.2e4],
            "band_median_high_hz": [8.0e4, 1.2e5],
        },
        "baseline": {
            "ratio_from_hom_unified_metric": old_ratio,
            "ratio_interp_10k_over_100k": baseline_ratio,
            "ratio_detrended_10k_over_100k": detrended_ratio,
            "trend_slope_log10": trend[0] if trend is not None else None,
            "trend_intercept_log10": trend[1] if trend is not None else None,
            "band_median_ratio_low8to12k_over_high80to120k": band_median_ratio,
        },
        "pair_sensitivity": pair_rows,
        "summary": {
            "pairs_n": n_pairs,
            "raw_ratio_median": raw_med,
            "detrended_ratio_median": det_med,
            "raw_ratio_pass_n": int(raw_pass_n),
            "detrended_ratio_pass_n": int(det_pass_n),
            "raw_ratio_pass_fraction": (float(raw_pass_n) / float(n_pairs)) if n_pairs > 0 else None,
            "detrended_ratio_pass_fraction": (float(det_pass_n) / float(n_pairs)) if n_pairs > 0 else None,
            "hard_gate_applicable": hard_gate_applicable,
            "decision": decision,
            "decision_note": (
                "single-point ratio is band-sensitive; keep as non-hard watch diagnostic."
                if decision != "pass_after_drift_and_band_audit"
                else "drift/band audit supports pass at operational threshold."
            ),
        },
    }


# 関数: `_write_csv` の入出力契約と処理意図を定義する。

def _write_csv(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = payload.get("pair_sensitivity") if isinstance(payload.get("pair_sensitivity"), list) else []
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "low_hz",
                "high_hz",
                "ratio_raw",
                "ratio_detrended",
                "raw_ge_threshold",
                "detrended_ge_threshold",
            ]
        )
        for row in rows:
            w.writerow(
                [
                    _fmt_float(float(row["low_hz"]), 6),
                    _fmt_float(float(row["high_hz"]), 6),
                    "" if row.get("ratio_raw") is None else _fmt_float(float(row["ratio_raw"]), 6),
                    "" if row.get("ratio_detrended") is None else _fmt_float(float(row["ratio_detrended"]), 6),
                    row.get("raw_ge_threshold"),
                    row.get("detrended_ge_threshold"),
                ]
            )


# 関数: `_plot` の入出力契約と処理意図を定義する。

def _plot(path: Path, payload: Dict[str, Any]) -> None:
    _set_japanese_font()
    thr = float(payload.get("thresholds", {}).get("lf_to_hf_ratio_min", 1.0))
    rows = payload.get("pair_sensitivity") if isinstance(payload.get("pair_sensitivity"), list) else []

    raw_low = [float(r["low_hz"]) for r in rows if isinstance(r.get("ratio_raw"), (int, float))]
    raw_high = [float(r["high_hz"]) for r in rows if isinstance(r.get("ratio_raw"), (int, float))]
    raw_ratio = [float(r["ratio_raw"]) for r in rows if isinstance(r.get("ratio_raw"), (int, float))]
    det_ratio = [float(r["ratio_detrended"]) for r in rows if isinstance(r.get("ratio_detrended"), (int, float))]

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.8), dpi=170)

    ax0 = axes[0]
    # 条件分岐: `raw_ratio` を満たす経路を評価する。
    if raw_ratio:
        x = np.arange(len(raw_ratio), dtype=float)
        ax0.plot(x, raw_ratio, marker="o", linewidth=1.6, label="raw")
        # 条件分岐: `len(det_ratio) == len(raw_ratio)` を満たす経路を評価する。
        if len(det_ratio) == len(raw_ratio):
            ax0.plot(x, det_ratio, marker="s", linewidth=1.4, label="detrended")

        labels = [f"{int(l/1000)}k/{int(h/1000)}k" for l, h in zip(raw_low, raw_high)]
        ax0.set_xticks(x)
        ax0.set_xticklabels(labels, rotation=70, ha="right", fontsize=8)

    ax0.axhline(thr, color="#333333", linestyle="--", linewidth=1.1)
    ax0.set_ylabel("PSD ratio (low/high)")
    ax0.set_title("Band-pair sensitivity sweep")
    ax0.grid(True, alpha=0.25)
    ax0.legend(loc="upper right")

    ax1 = axes[1]
    baseline = payload.get("baseline", {}) if isinstance(payload.get("baseline"), dict) else {}
    bars = ["raw 10k/100k", "detrended 10k/100k", "band-median ratio"]
    vals = [
        baseline.get("ratio_interp_10k_over_100k"),
        baseline.get("ratio_detrended_10k_over_100k"),
        baseline.get("band_median_ratio_low8to12k_over_high80to120k"),
    ]
    vals_num = [float(v) if isinstance(v, (int, float)) else float("nan") for v in vals]
    ax1.bar(bars, vals_num, color=["#1f77b4", "#ff7f0e", "#2ca02c"], alpha=0.92)
    ax1.axhline(thr, color="#333333", linestyle="--", linewidth=1.1)
    ax1.set_ylabel("ratio")
    ax1.set_title("Operational ratios")
    ax1.grid(True, axis="y", alpha=0.25)
    ax1.tick_params(axis="x", rotation=25)

    decision = str(payload.get("summary", {}).get("decision") or "unknown")
    fig.suptitle(f"HOM noise-PSD watch audit ({decision})", y=1.02)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# 関数: `main` の入出力契約と処理意図を定義する。

def main(argv: Optional[Sequence[str]] = None) -> int:
    default_hom_metrics = ROOT / "output" / "public" / "quantum" / "hom_squeezed_light_unified_audit_metrics.json"
    default_psd_zip = ROOT / "data" / "quantum" / "sources" / "zenodo_6371310" / "DataExfig3.zip"
    default_outdir = ROOT / "output" / "private" / "quantum"
    default_public_outdir = ROOT / "output" / "public" / "quantum"

    ap = argparse.ArgumentParser(description="Audit HOM noise-PSD watch item with drift correction and band-definition sensitivity.")
    ap.add_argument("--hom-metrics-json", type=str, default=str(default_hom_metrics), help="Input HOM unified metrics JSON.")
    ap.add_argument("--psd-zip", type=str, default=str(default_psd_zip), help="Input PSD ZIP.")
    ap.add_argument("--psd-csv-name", type=str, default="DataExfig3b.csv", help="PSD CSV name inside ZIP.")
    ap.add_argument("--ratio-threshold", type=float, default=1.0, help="Threshold for low/high PSD ratio.")
    ap.add_argument("--outdir", type=str, default=str(default_outdir), help="Private output directory.")
    ap.add_argument("--public-outdir", type=str, default=str(default_public_outdir), help="Public output directory.")
    ap.add_argument("--no-public-copy", action="store_true", help="Do not copy outputs to public directory.")
    args = ap.parse_args(list(argv) if argv is not None else None)

    hom_metrics = Path(args.hom_metrics_json)
    psd_zip = Path(args.psd_zip)
    outdir = Path(args.outdir)
    public_outdir = Path(args.public_outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    public_outdir.mkdir(parents=True, exist_ok=True)

    payload = _run_audit(
        hom_metrics_json=hom_metrics,
        psd_zip=psd_zip,
        psd_csv_name=str(args.psd_csv_name),
        ratio_threshold=float(args.ratio_threshold),
    )

    out_json = outdir / "hom_noise_psd_watch_audit.json"
    out_csv = outdir / "hom_noise_psd_watch_audit.csv"
    out_png = outdir / "hom_noise_psd_watch_audit.png"

    _write_json(out_json, payload)
    _write_csv(out_csv, payload)
    _plot(out_png, payload)

    copied: List[Path] = []
    # 条件分岐: `not args.no_public_copy` を満たす経路を評価する。
    if not args.no_public_copy:
        for src in (out_json, out_csv, out_png):
            dst = public_outdir / src.name
            shutil.copy2(src, dst)
            copied.append(dst)

    try:
        worklog.append_event(
            {
                "event_type": "quantum_hom_noise_psd_watch_audit",
                "argv": sys.argv,
                "inputs": {
                    "hom_metrics_json": hom_metrics,
                    "psd_zip": psd_zip,
                    "psd_csv_name": args.psd_csv_name,
                },
                "outputs": {
                    "json": out_json,
                    "csv": out_csv,
                    "png": out_png,
                    "public_copies": copied,
                },
                "metrics": payload.get("summary"),
            }
        )
    except Exception:
        pass

    print(f"[ok] json : {out_json}")
    print(f"[ok] csv  : {out_csv}")
    print(f"[ok] png  : {out_png}")
    # 条件分岐: `copied` を満たす経路を評価する。
    if copied:
        print(f"[ok] public copies: {len(copied)} files -> {public_outdir}")

    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())
