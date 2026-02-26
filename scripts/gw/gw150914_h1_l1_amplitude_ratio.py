from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.signal import hilbert  # noqa: E402

_ROOT = Path(__file__).resolve().parents[2]
# 条件分岐: `str(_ROOT) not in sys.path` を満たす経路を評価する。
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.gw.gw150914_chirp_phase import _bandpass, _fetch_inputs, _parse_gwosc_txt_gz, _whiten_fft
from scripts.summary import worklog


def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def _slugify(s: str) -> str:
    out = "".join(ch.lower() if ch.isalnum() else "_" for ch in (s or "").strip())
    while "__" in out:
        out = out.replace("__", "_")

    return out.strip("_") or "event"


def _corrcoef_1d(x: np.ndarray, y: np.ndarray) -> float:
    # 条件分岐: `x.size != y.size or x.size < 8` を満たす経路を評価する。
    if x.size != y.size or x.size < 8:
        return float("nan")

    x0 = x - float(np.mean(x))
    y0 = y - float(np.mean(y))
    nx = float(np.linalg.norm(x0))
    ny = float(np.linalg.norm(y0))
    den = nx * ny
    # 条件分岐: `not math.isfinite(den) or den <= 0.0` を満たす経路を評価する。
    if not math.isfinite(den) or den <= 0.0:
        return float("nan")

    return float(np.dot(x0, y0) / den)


def _scan_lag_corr(x_first: np.ndarray, x_second: np.ndarray, max_lag_samples: int) -> Tuple[np.ndarray, np.ndarray, int, float]:
    n = int(min(x_first.size, x_second.size))
    # 条件分岐: `n <= 16` を満たす経路を評価する。
    if n <= 16:
        return np.array([0], dtype=np.int32), np.array([float("nan")], dtype=np.float64), 0, float("nan")

    h = np.asarray(x_first[:n], dtype=np.float64)
    l = np.asarray(x_second[:n], dtype=np.float64)
    max_lag = int(max(0, min(int(max_lag_samples), n // 4)))
    lags = np.arange(-max_lag, max_lag + 1, dtype=np.int32)
    corrs = np.full(lags.shape, np.nan, dtype=np.float64)

    best_lag = 0
    best_corr = float("nan")
    best_abs = -1.0
    for i, lag in enumerate(lags):
        lg = int(lag)
        # 条件分岐: `lg > 0` を満たす経路を評価する。
        if lg > 0:
            a = h[lg:]
            b = l[:-lg]
        # 条件分岐: 前段条件が不成立で、`lg < 0` を追加評価する。
        elif lg < 0:
            a = h[:lg]
            b = l[-lg:]
        else:
            a = h
            b = l

        c = _corrcoef_1d(a, b)
        corrs[i] = c
        score = abs(float(c)) if math.isfinite(float(c)) else -1.0
        # 条件分岐: `score > best_abs` を満たす経路を評価する。
        if score > best_abs:
            best_abs = score
            best_lag = lg
            best_corr = float(c)

    return lags, corrs, best_lag, best_corr


def _apply_lag_to_first(
    t: np.ndarray, x_first: np.ndarray, x_second: np.ndarray, lag_samples: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    lag = int(lag_samples)
    n = int(min(x_first.size, x_second.size, t.size))
    t0 = np.asarray(t[:n], dtype=np.float64)
    h0 = np.asarray(x_first[:n], dtype=np.float64)
    l0 = np.asarray(x_second[:n], dtype=np.float64)
    # 条件分岐: `lag > 0` を満たす経路を評価する。
    if lag > 0:
        return t0[lag:], h0[lag:], l0[:-lag]

    # 条件分岐: `lag < 0` を満たす経路を評価する。

    if lag < 0:
        return t0[:lag], h0[:lag], l0[-lag:]

    return t0, h0, l0


def _align_on_common_grid(
    t_first: np.ndarray,
    x_first: np.ndarray,
    fs_first: float,
    t_second: np.ndarray,
    x_second: np.ndarray,
    fs_second: float,
    *,
    detector_first: str,
    detector_second: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    fs_ref = float(min(fs_first, fs_second))
    # 条件分岐: `not (math.isfinite(fs_ref) and fs_ref > 0)` を満たす経路を評価する。
    if not (math.isfinite(fs_ref) and fs_ref > 0):
        raise ValueError("invalid sampling rate")

    t_min = float(max(float(t_first[0]), float(t_second[0])))
    t_max = float(min(float(t_first[-1]), float(t_second[-1])))
    # 条件分岐: `not math.isfinite(t_min) or not math.isfinite(t_max) or t_max <= t_min` を満たす経路を評価する。
    if not math.isfinite(t_min) or not math.isfinite(t_max) or t_max <= t_min:
        raise ValueError(f"no overlap between {detector_first}/{detector_second} time ranges")

    n = int(math.floor((t_max - t_min) * fs_ref))
    # 条件分岐: `n < 128` を満たす経路を評価する。
    if n < 128:
        raise ValueError("overlap window too short")

    t = t_min + np.arange(n, dtype=np.float64) / fs_ref

    xh = np.interp(t, t_first, x_first)
    xl = np.interp(t, t_second, x_second)
    return t, xh, xl, fs_ref


def _prepare_detector_series(
    strain_path: Path,
    *,
    gps_event: float,
    preprocess: str,
    f_lo: float,
    f_hi: float,
    whiten_nperseg: int,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    gps_start, fs, strain = _parse_gwosc_txt_gz(strain_path)
    x = np.asarray(strain, dtype=np.float64)
    x = x - float(np.mean(x))
    # 条件分岐: `preprocess == "whiten"` を満たす経路を評価する。
    if preprocess == "whiten":
        xf = _whiten_fft(x, fs, f_lo=float(f_lo), f_hi=float(f_hi), welch_nperseg=int(whiten_nperseg))
    else:
        xf = _bandpass(x, fs, f_lo=float(f_lo), f_hi=float(f_hi), order=4)

    t = (float(gps_start) + np.arange(xf.size, dtype=np.float64) / float(fs)) - float(gps_event)
    return t, xf, float(fs), float(gps_start)


def _fmt_float(v: float, ndigits: int = 6) -> str:
    # 条件分岐: `not math.isfinite(float(v))` を満たす経路を評価する。
    if not math.isfinite(float(v)):
        return ""

    x = float(v)
    # 条件分岐: `x == 0.0` を満たす経路を評価する。
    if x == 0.0:
        return "0"

    ax = abs(x)
    # 条件分岐: `ax >= 1e4 or ax < 1e-3` を満たす経路を評価する。
    if ax >= 1e4 or ax < 1e-3:
        return f"{x:.{ndigits}g}"

    return f"{x:.{ndigits}f}".rstrip("0").rstrip(".")


def _build_metrics(
    *,
    t_aligned: np.ndarray,
    x_h1: np.ndarray,
    x_l1: np.ndarray,
    ratio_signal_percentile: float,
    min_ratio_points: int,
) -> Dict[str, Any]:
    env_h1 = np.abs(hilbert(x_h1))
    env_l1 = np.abs(hilbert(x_l1))
    envelope_strength = np.sqrt(np.maximum(env_h1 * env_l1, 0.0))
    pct = float(min(99.9, max(0.0, ratio_signal_percentile)))
    threshold = float(np.percentile(envelope_strength, pct))
    eps = float(max(np.percentile(env_l1, 5.0) * 1e-6, np.finfo(np.float64).tiny))
    mask = (envelope_strength >= threshold) & (env_l1 > eps)

    min_pts = int(max(32, min_ratio_points))
    # 条件分岐: `int(np.sum(mask)) < min_pts` を満たす経路を評価する。
    if int(np.sum(mask)) < min_pts:
        k = int(min(max(min_pts, int(0.2 * env_l1.size)), env_l1.size))
        idx = np.argsort(envelope_strength)[::-1]
        mask2 = np.zeros(env_l1.size, dtype=bool)
        mask2[idx[:k]] = True
        mask = mask2 & (env_l1 > eps)

    ratio = env_h1[mask] / env_l1[mask]
    t_ratio = t_aligned[mask]
    # 条件分岐: `ratio.size < 8` を満たす経路を評価する。
    if ratio.size < 8:
        raise ValueError("ratio sample too small after envelope selection")

    q16, q25, q50, q75, q84 = np.percentile(ratio, [16, 25, 50, 75, 84]).tolist()
    ratio_log_std = float(np.std(np.log(np.maximum(ratio, 1e-15))))
    iqr = float(q75 - q25)
    iqr_over_med = float(iqr / (q50 + 1e-15))
    p16p84_halfspan_over_med = float((q84 - q16) / (2.0 * (q50 + 1e-15)))

    slope = float("nan")
    slope_window_frac = float("nan")
    # 条件分岐: `ratio.size >= 10` を満たす経路を評価する。
    if ratio.size >= 10:
        slope = float(np.polyfit(t_ratio, ratio, 1)[0])
        dt = float(np.max(t_ratio) - np.min(t_ratio))
        # 条件分岐: `dt > 0` を満たす経路を評価する。
        if dt > 0:
            slope_window_frac = float(abs(slope) * dt / (abs(q50) + 1e-15))

    return {
        "envelope_h1": env_h1,
        "envelope_l1": env_l1,
        "envelope_strength": envelope_strength,
        "selection_mask": mask,
        "selection_threshold": threshold,
        "ratio": ratio,
        "ratio_time_s": t_ratio,
        "ratio_stats": {
            "median": float(q50),
            "p16": float(q16),
            "p84": float(q84),
            "q25": float(q25),
            "q75": float(q75),
            "iqr": float(iqr),
            "iqr_over_median": float(iqr_over_med),
            "p16_p84_halfspan_over_median": float(p16p84_halfspan_over_med),
            "log_std": float(ratio_log_std),
            "slope_per_s": float(slope),
            "slope_window_fraction": float(slope_window_frac),
            "selected_points": int(ratio.size),
            "total_points": int(env_h1.size),
        },
    }


def _gate_status(
    *,
    abs_corr: float,
    corr_min: float,
    ratio_points: int,
    min_ratio_points: int,
    iqr_over_median: float,
    iqr_warn: float,
    slope_window_fraction: float,
    slope_warn: float,
) -> Tuple[str, List[str]]:
    reasons: List[str] = []
    status = "pass"

    # 条件分岐: `not math.isfinite(abs_corr) or abs_corr < float(corr_min)` を満たす経路を評価する。
    if not math.isfinite(abs_corr) or abs_corr < float(corr_min):
        status = "reject"
        reasons.append("low_cross_correlation")

    # 条件分岐: `int(ratio_points) < int(min_ratio_points)` を満たす経路を評価する。

    if int(ratio_points) < int(min_ratio_points):
        status = "reject"
        reasons.append("insufficient_ratio_samples")

    # 条件分岐: `status != "reject"` を満たす経路を評価する。

    if status != "reject":
        # 条件分岐: `math.isfinite(iqr_over_median) and iqr_over_median > float(iqr_warn)` を満たす経路を評価する。
        if math.isfinite(iqr_over_median) and iqr_over_median > float(iqr_warn):
            status = "watch"
            reasons.append("ratio_spread_large")

        # 条件分岐: `math.isfinite(slope_window_fraction) and slope_window_fraction > float(slope_...` を満たす経路を評価する。

        if math.isfinite(slope_window_fraction) and slope_window_fraction > float(slope_warn):
            status = "watch"
            reasons.append("ratio_time_drift_large")

    # 条件分岐: `not reasons` を満たす経路を評価する。

    if not reasons:
        reasons.append("all_gates_passed")

    return status, reasons


def _plot(
    *,
    out_png: Path,
    title: str,
    detector_first: str,
    detector_second: str,
    t_window: np.ndarray,
    x_first: np.ndarray,
    x_second: np.ndarray,
    lags_ms: np.ndarray,
    corrs: np.ndarray,
    best_lag_ms: float,
    best_corr: float,
    ratio_time_s: np.ndarray,
    ratio: np.ndarray,
    ratio_stats: Dict[str, float],
) -> None:
    _set_japanese_font()
    fig = plt.figure(figsize=(13.6, 8.4))

    ax1 = fig.add_subplot(2, 2, 1)
    std_h = float(np.std(x_first)) or 1.0
    std_l = float(np.std(x_second)) or 1.0
    ax1.plot(
        t_window,
        x_first / std_h,
        label=f"{detector_first} (aligned, normalized)",
        linewidth=1.5,
        color="#1f77b4",
    )
    ax1.plot(t_window, x_second / std_l, label=f"{detector_second} (normalized)", linewidth=1.5, color="#ff7f0e")
    ax1.set_title("Aligned strain in chirp window")
    ax1.set_xlabel("t relative to event time [s]")
    ax1.set_ylabel("normalized strain")
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="upper left")

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(lags_ms, corrs, color="#2ca02c", linewidth=2.0)
    ax2.axvline(best_lag_ms, color="#d62728", linestyle="--", linewidth=1.2, label=f"best lag={best_lag_ms:.3f} ms")
    ax2.axhline(0.0, color="#555555", linewidth=1.0)
    ax2.set_title(f"{detector_first}/{detector_second} normalized cross-correlation vs lag")
    ax2.set_xlabel(f"lag applied to {detector_first} [ms]")
    ax2.set_ylabel("correlation")
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="upper right")

    ax3 = fig.add_subplot(2, 1, 2)
    ax3.plot(ratio_time_s, ratio, ".", color="#9467bd", alpha=0.55, markersize=4.0, label="selected envelope ratio")
    med = float(ratio_stats.get("median", float("nan")))
    p16 = float(ratio_stats.get("p16", float("nan")))
    p84 = float(ratio_stats.get("p84", float("nan")))
    # 条件分岐: `math.isfinite(med)` を満たす経路を評価する。
    if math.isfinite(med):
        ax3.axhline(med, color="#111111", linewidth=1.8, label=f"median={med:.3f}")

    # 条件分岐: `math.isfinite(p16)` を満たす経路を評価する。

    if math.isfinite(p16):
        ax3.axhline(p16, color="#555555", linestyle="--", linewidth=1.0, label=f"p16={p16:.3f}")

    # 条件分岐: `math.isfinite(p84)` を満たす経路を評価する。

    if math.isfinite(p84):
        ax3.axhline(p84, color="#555555", linestyle="--", linewidth=1.0, label=f"p84={p84:.3f}")

    ax3.set_title(
        f"Envelope amplitude ratio |{detector_first}|/|{detector_second}| "
        + f"(best corr={best_corr:+.3f}, half-span/med={ratio_stats.get('p16_p84_halfspan_over_median', float('nan')):.3f})"
    )
    ax3.set_xlabel("t relative to event time [s]")
    ax3.set_ylabel(f"|{detector_first}| / |{detector_second}|")
    ax3.grid(True, alpha=0.25)
    ax3.legend(loc="upper right", ncol=2, fontsize=9)

    fig.suptitle(title, fontsize=15)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def _write_summary_csv(path: Path, summary_row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = [
        "event",
        "preprocess",
        "window_start_s",
        "window_end_s",
        "f_lo_hz",
        "f_hi_hz",
        "best_lag_ms",
        "best_corr",
        "abs_best_corr",
        "ratio_median",
        "ratio_p16",
        "ratio_p84",
        "ratio_log_std",
        "ratio_iqr_over_median",
        "ratio_slope_window_fraction",
        "status",
        "status_reason",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys)
        w.writerow([summary_row.get(k, "") for k in keys])


def _write_lag_scan_csv(path: Path, lags_ms: np.ndarray, corrs: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["lag_ms", "corr"])
        for lag_ms, corr in zip(lags_ms.tolist(), corrs.tolist()):
            w.writerow([_fmt_float(float(lag_ms), 6), _fmt_float(float(corr), 8)])


def _write_ratio_samples_csv(
    path: Path,
    t_aligned: np.ndarray,
    env_first: np.ndarray,
    env_second: np.ndarray,
    ratio: np.ndarray,
    ratio_time: np.ndarray,
    selection_mask: np.ndarray,
    *,
    detector_first: str,
    detector_second: str,
) -> None:
    selected_lookup = {float(t): float(r) for t, r in zip(ratio_time.tolist(), ratio.tolist())}
    path.parent.mkdir(parents=True, exist_ok=True)
    det_first_slug = _slugify(detector_first)
    det_second_slug = _slugify(detector_second)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "t_s",
                f"env_{det_first_slug}",
                f"env_{det_second_slug}",
                "selected",
                f"ratio_{det_first_slug}_{det_second_slug}_selected",
            ]
        )
        for i in range(int(t_aligned.size)):
            ti = float(t_aligned[i])
            is_selected = bool(selection_mask[i])
            ratio_val = selected_lookup.get(ti, float("nan"))
            w.writerow(
                [
                    _fmt_float(ti, 9),
                    _fmt_float(float(env_first[i]), 9),
                    _fmt_float(float(env_second[i]), 9),
                    "1" if is_selected else "0",
                    _fmt_float(float(ratio_val), 9) if is_selected and math.isfinite(float(ratio_val)) else "",
                ]
            )


def main(argv: Optional[Sequence[str]] = None) -> int:
    root = _ROOT
    ap = argparse.ArgumentParser(description="GW detector-pair amplitude-ratio consistency audit.")
    ap.add_argument("--event", type=str, default="GW150914")
    ap.add_argument("--catalog", type=str, default="GWTC-1-confident")
    ap.add_argument("--version", type=str, default="auto")
    ap.add_argument("--detectors", type=str, default="H1,L1")
    ap.add_argument("--prefer-duration-s", type=int, default=32)
    ap.add_argument("--prefer-sampling-rate-hz", type=int, default=4096)
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--force-download", action="store_true")
    ap.add_argument("--preprocess", type=str, default="bandpass", choices=["bandpass", "whiten"])
    ap.add_argument("--f-lo", type=float, default=30.0)
    ap.add_argument("--f-hi", type=float, default=350.0)
    ap.add_argument("--whiten-nperseg", type=int, default=4096)
    ap.add_argument("--window", type=str, default="-0.12,-0.01", help="Analysis window [s], e.g. -0.12,-0.01.")
    ap.add_argument("--max-lag-ms", type=float, default=10.0)
    ap.add_argument("--ratio-signal-percentile", type=float, default=70.0)
    ap.add_argument("--min-ratio-points", type=int, default=80)
    ap.add_argument("--corr-min", type=float, default=0.6)
    ap.add_argument("--ratio-iqr-warn", type=float, default=0.35)
    ap.add_argument("--ratio-slope-window-frac-warn", type=float, default=0.5)
    ap.add_argument("--data-dir", type=str, default="", help="Optional override for GW cache directory.")
    ap.add_argument("--outdir", type=str, default=str(root / "output" / "private" / "gw"))
    ap.add_argument("--public-outdir", type=str, default=str(root / "output" / "public" / "gw"))
    ap.add_argument("--no-public-copy", action="store_true")
    args = ap.parse_args(list(argv) if argv is not None else None)

    event_name = str(args.event).strip() or "GW150914"
    event_slug = _slugify(event_name)
    detectors_in = [d.strip().upper() for d in str(args.detectors).split(",") if d.strip()]
    # 条件分岐: `len(detectors_in) < 2` を満たす経路を評価する。
    if len(detectors_in) < 2:
        print("[err] --detectors must include two detector names (e.g. H1,L1 or H1,V1)")
        return 2

    order = {"H1": 0, "L1": 1, "V1": 2, "K1": 3}
    pair = sorted(detectors_in[:2], key=lambda d: (order.get(d, 999), d))
    detector_first, detector_second = pair[0], pair[1]

    try:
        ws, we = [float(x.strip()) for x in str(args.window).split(",", 1)]
    except Exception:
        print("[err] invalid --window (expected \"start,end\")")
        return 2

    # 条件分岐: `not (math.isfinite(ws) and math.isfinite(we) and ws < we)` を満たす経路を評価する。

    if not (math.isfinite(ws) and math.isfinite(we) and ws < we):
        print("[err] invalid --window range")
        return 2

    # 条件分岐: `args.data_dir` を満たす経路を評価する。

    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = root / "data" / "gw" / event_slug

    outdir = Path(args.outdir)
    public_outdir = Path(args.public_outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    public_outdir.mkdir(parents=True, exist_ok=True)

    try:
        fetch = _fetch_inputs(
            data_dir,
            event=event_name,
            catalog=str(args.catalog),
            version=str(args.version),
            detectors=[detector_first, detector_second],
            prefer_duration_s=int(args.prefer_duration_s),
            prefer_sampling_rate_hz=int(args.prefer_sampling_rate_hz),
            offline=bool(args.offline),
            force=bool(args.force_download),
        )
    except Exception as e:
        print(f"[err] fetch inputs failed: {e}")
        return 2

    event_info = (fetch.get("event_info") or {}) if isinstance(fetch, dict) else {}
    try:
        gps_event = float(event_info.get("GPS"))
    except Exception:
        print("[err] event GPS not available in GWOSC metadata")
        return 2

    strain_paths = ((fetch.get("paths") or {}).get("strain") or {}) if isinstance(fetch, dict) else {}
    path_first = Path(strain_paths.get(detector_first) or "")
    path_second = Path(strain_paths.get(detector_second) or "")
    # 条件分岐: `not path_first.exists() or not path_second.exists()` を満たす経路を評価する。
    if not path_first.exists() or not path_second.exists():
        print(f"[err] {detector_first}/{detector_second} strain file missing")
        return 2

    try:
        t_first, x_first, fs_first, gps_start_first = _prepare_detector_series(
            path_first,
            gps_event=gps_event,
            preprocess=str(args.preprocess),
            f_lo=float(args.f_lo),
            f_hi=float(args.f_hi),
            whiten_nperseg=int(args.whiten_nperseg),
        )
        t_second, x_second, fs_second, gps_start_second = _prepare_detector_series(
            path_second,
            gps_event=gps_event,
            preprocess=str(args.preprocess),
            f_lo=float(args.f_lo),
            f_hi=float(args.f_hi),
            whiten_nperseg=int(args.whiten_nperseg),
        )
        t_common, xh_common, xl_common, fs_ref = _align_on_common_grid(
            t_first,
            x_first,
            fs_first,
            t_second,
            x_second,
            fs_second,
            detector_first=detector_first,
            detector_second=detector_second,
        )
    except Exception as e:
        print(f"[err] preprocess/alignment failed: {e}")
        return 2

    mask_window = (t_common >= ws) & (t_common <= we)
    # 条件分岐: `int(np.sum(mask_window)) < 128` を満たす経路を評価する。
    if int(np.sum(mask_window)) < 128:
        print("[err] analysis window too short after alignment")
        return 2

    t_win = t_common[mask_window]
    h_win = xh_common[mask_window]
    l_win = xl_common[mask_window]

    max_lag_samples = int(round(float(args.max_lag_ms) * 1e-3 * fs_ref))
    lags, corrs, best_lag_samples, best_corr = _scan_lag_corr(h_win, l_win, max_lag_samples)
    best_lag_ms = float(best_lag_samples / fs_ref * 1e3)
    abs_best_corr = abs(float(best_corr)) if math.isfinite(float(best_corr)) else float("nan")

    t_al, h_al, l_al = _apply_lag_to_first(t_win, h_win, l_win, best_lag_samples)
    # 条件分岐: `t_al.size < 128` を満たす経路を評価する。
    if t_al.size < 128:
        print("[err] aligned window too short")
        return 2

    try:
        m = _build_metrics(
            t_aligned=t_al,
            x_h1=h_al,
            x_l1=l_al,
            ratio_signal_percentile=float(args.ratio_signal_percentile),
            min_ratio_points=int(args.min_ratio_points),
        )
    except Exception as e:
        print(f"[err] ratio metric computation failed: {e}")
        return 2

    ratio_stats = m["ratio_stats"]
    status, status_reasons = _gate_status(
        abs_corr=abs_best_corr,
        corr_min=float(args.corr_min),
        ratio_points=int(ratio_stats.get("selected_points", 0)),
        min_ratio_points=int(args.min_ratio_points),
        iqr_over_median=float(ratio_stats.get("iqr_over_median", float("nan"))),
        iqr_warn=float(args.ratio_iqr_warn),
        slope_window_fraction=float(ratio_stats.get("slope_window_fraction", float("nan"))),
        slope_warn=float(args.ratio_slope_window_frac_warn),
    )

    det_first_slug = _slugify(detector_first)
    det_second_slug = _slugify(detector_second)
    stem = f"{event_slug}_{det_first_slug}_{det_second_slug}_amplitude_ratio"
    out_png = outdir / f"{stem}.png"
    out_json = outdir / f"{stem}_metrics.json"
    out_summary_csv = outdir / f"{stem}_summary.csv"
    out_lag_csv = outdir / f"{stem}_lag_scan.csv"
    out_samples_csv = outdir / f"{stem}_samples.csv"

    _plot(
        out_png=out_png,
        title=f"{event_name}: {detector_first}/{detector_second} amplitude-ratio consistency audit",
        detector_first=detector_first,
        detector_second=detector_second,
        t_window=t_al,
        x_first=h_al,
        x_second=l_al,
        lags_ms=lags.astype(np.float64) / fs_ref * 1e3,
        corrs=corrs,
        best_lag_ms=best_lag_ms,
        best_corr=float(best_corr),
        ratio_time_s=m["ratio_time_s"],
        ratio=m["ratio"],
        ratio_stats=ratio_stats,
    )

    summary_row = {
        "event": event_name,
        "preprocess": str(args.preprocess),
        "window_start_s": _fmt_float(ws, 6),
        "window_end_s": _fmt_float(we, 6),
        "f_lo_hz": _fmt_float(float(args.f_lo), 6),
        "f_hi_hz": _fmt_float(float(args.f_hi), 6),
        "best_lag_ms": _fmt_float(best_lag_ms, 6),
        "best_corr": _fmt_float(float(best_corr), 7),
        "abs_best_corr": _fmt_float(float(abs_best_corr), 7),
        "ratio_median": _fmt_float(float(ratio_stats.get("median", float("nan"))), 7),
        "ratio_p16": _fmt_float(float(ratio_stats.get("p16", float("nan"))), 7),
        "ratio_p84": _fmt_float(float(ratio_stats.get("p84", float("nan"))), 7),
        "ratio_log_std": _fmt_float(float(ratio_stats.get("log_std", float("nan"))), 7),
        "ratio_iqr_over_median": _fmt_float(float(ratio_stats.get("iqr_over_median", float("nan"))), 7),
        "ratio_slope_window_fraction": _fmt_float(float(ratio_stats.get("slope_window_fraction", float("nan"))), 7),
        "status": status,
        "status_reason": ";".join(status_reasons),
    }
    _write_summary_csv(out_summary_csv, summary_row)
    _write_lag_scan_csv(out_lag_csv, lags.astype(np.float64) / fs_ref * 1e3, corrs)
    _write_ratio_samples_csv(
        out_samples_csv,
        t_aligned=t_al,
        env_first=m["envelope_h1"],
        env_second=m["envelope_l1"],
        ratio=m["ratio"],
        ratio_time=m["ratio_time_s"],
        selection_mask=m["selection_mask"],
        detector_first=detector_first,
        detector_second=detector_second,
    )

    payload: Dict[str, Any] = {
        "generated_utc": _iso_utc_now(),
        "schema": "wavep.gw.detector_pair_amplitude_ratio_audit.v1",
        "event": {
            "name": event_name,
            "slug": event_slug,
            "gps_event": gps_event,
            "catalog": str(args.catalog),
            "version": str(((fetch.get("meta") or {}).get("selection") or {}).get("api_version") or ""),
        },
        "inputs": {
            "detector_first": detector_first,
            "detector_second": detector_second,
            "strain_first": str(path_first).replace("\\", "/"),
            "strain_second": str(path_second).replace("\\", "/"),
            "gps_start_first": gps_start_first,
            "gps_start_second": gps_start_second,
            "fs_first_hz": fs_first,
            "fs_second_hz": fs_second,
            "analysis_fs_hz": fs_ref,
            "window_s": [ws, we],
            "preprocess": str(args.preprocess),
            "bandpass_hz": [float(args.f_lo), float(args.f_hi)],
            "ratio_signal_percentile": float(args.ratio_signal_percentile),
            "max_lag_ms": float(args.max_lag_ms),
        },
        "metrics": {
            "best_lag_samples_apply_to_first": int(best_lag_samples),
            "best_lag_ms_apply_to_first": float(best_lag_ms),
            "best_corr": float(best_corr),
            "abs_best_corr": float(abs_best_corr),
            "ratio": ratio_stats,
        },
        "gate": {
            "corr_min": float(args.corr_min),
            "ratio_iqr_warn": float(args.ratio_iqr_warn),
            "ratio_slope_window_frac_warn": float(args.ratio_slope_window_frac_warn),
            "min_ratio_points": int(args.min_ratio_points),
            "status": status,
            "status_reasons": status_reasons,
        },
        "sources": fetch.get("meta") if isinstance(fetch, dict) else {},
        "outputs": {
            "plot_png": str(out_png).replace("\\", "/"),
            "summary_csv": str(out_summary_csv).replace("\\", "/"),
            "lag_scan_csv": str(out_lag_csv).replace("\\", "/"),
            "samples_csv": str(out_samples_csv).replace("\\", "/"),
            "metrics_json": str(out_json).replace("\\", "/"),
        },
        "notes": {
            "lag_definition": f"positive lag means {detector_first} is advanced by this lag for alignment against {detector_second}",
            "consistency_definition": "high |corr| and stable envelope ratio across the chirp window",
        },
    }
    # 条件分岐: `detector_first == "H1"` を満たす経路を評価する。
    if detector_first == "H1":
        payload["metrics"]["best_lag_samples_apply_to_h1"] = int(best_lag_samples)
        payload["metrics"]["best_lag_ms_apply_to_h1"] = float(best_lag_ms)

    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    copied: List[Path] = []
    # 条件分岐: `not args.no_public_copy` を満たす経路を評価する。
    if not args.no_public_copy:
        for src in [out_png, out_json, out_summary_csv, out_lag_csv, out_samples_csv]:
            dst = public_outdir / src.name
            shutil.copy2(src, dst)
            copied.append(dst)

    try:
        worklog.append_event(
            {
                "event_type": f"{event_slug}_{det_first_slug}_{det_second_slug}_amplitude_ratio",
                "argv": sys.argv,
                "inputs": {"data_dir": data_dir},
                "outputs": {
                    "plot_png": out_png,
                    "summary_csv": out_summary_csv,
                    "lag_scan_csv": out_lag_csv,
                    "samples_csv": out_samples_csv,
                    "metrics_json": out_json,
                    "public_copies": copied,
                },
                "metrics": {
                    "detector_first": detector_first,
                    "detector_second": detector_second,
                    "status": status,
                    "best_lag_ms": best_lag_ms,
                    "abs_best_corr": abs_best_corr,
                    "ratio_median": float(ratio_stats.get("median", float("nan"))),
                    "ratio_iqr_over_median": float(ratio_stats.get("iqr_over_median", float("nan"))),
                },
            }
        )
    except Exception:
        pass

    print(f"[ok] png        : {out_png}")
    print(f"[ok] metrics    : {out_json}")
    print(f"[ok] summary csv: {out_summary_csv}")
    print(f"[ok] lag csv    : {out_lag_csv}")
    print(f"[ok] samples csv: {out_samples_csv}")
    # 条件分岐: `copied` を満たす経路を評価する。
    if copied:
        print(f"[ok] public copies: {len(copied)} files -> {public_outdir}")

    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())
