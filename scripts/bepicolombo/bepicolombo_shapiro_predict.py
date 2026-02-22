#!/usr/bin/env python3
"""
bepicolombo_shapiro_predict.py

BepiColombo（MPO）と地球の太陽会合幾何（impact parameter b）から、
Shapiro 由来の y(t)（Cassini Eq(2) 近似）を P-model（β）で予測して可視化する。

前提：
  - 観測データ（MORE: range/Doppler）がPSA上で未公開の可能性があるため、
    現段階では「一次ソース（SPICE）に基づく幾何の確立＋予測曲線」を作る。
  - 以後、観測データが公開されたら同じ形式（y(t) vs 観測）へ拡張する。

入力（オフライン再現）：
  - data/bepicolombo/kernels/psa/kernels_meta.json
  - kernels_meta.json の selected_paths を SPICE でロードして使用する

出力（固定）：
  - output/private/bepicolombo/bepicolombo_shapiro_geometry.csv
  - output/private/bepicolombo/bepicolombo_shapiro_geometry_summary.json
  - output/private/bepicolombo/bepicolombo_shapiro_geometry.png
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    import spiceypy as sp  # type: ignore
except Exception as e:  # pragma: no cover
    raise SystemExit(f"[err] spiceypy is required: {e}")

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.summary import worklog  # noqa: E402

C = 299792458.0
MU_SUN = 1.3271244e20  # m^3/s^2
R_SUN = 6.957e8  # m


@dataclass(frozen=True)
class SeriesRow:
    et: float
    t_utc: datetime
    t_days: float
    r1_m: float
    r2_m: float
    b_m: float
    occulted: bool
    bdot_mps: float
    r1dot_mps: float
    r2dot_mps: float
    dt_roundtrip_s: float
    y_eq2: float
    y_full: float


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _set_japanese_font() -> None:
    try:
        import matplotlib as mpl
        import matplotlib.font_manager as fm

        preferred = [
            "Yu Gothic",
            "Yu Gothic UI",
            "Meiryo",
            "BIZ UDGothic",
            "MS Gothic",
            "Noto Sans CJK JP",
            "IPAexGothic",
        ]
        available = {f.name for f in fm.fontManager.ttflist}
        chosen = [name for name in preferred if name in available]
        if not chosen:
            return
        mpl.rcParams["font.family"] = chosen + ["DejaVu Sans"]
        mpl.rcParams["axes.unicode_minus"] = False
    except Exception:
        return


def _load_kernels_from_meta(kernels_dir: Path) -> Dict[str, Any]:
    meta_path = kernels_dir / "kernels_meta.json"
    if not meta_path.exists():
        raise RuntimeError(
            f"Missing kernels meta: {meta_path}\n"
            "Run: python -B scripts/bepicolombo/fetch_spice_kernels_psa.py"
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if not isinstance(meta, dict):
        raise RuntimeError(f"Invalid kernels meta JSON: {meta_path}")

    selected = meta.get("selected_paths")
    if not isinstance(selected, list) or not selected:
        raise RuntimeError(f"Missing selected_paths in: {meta_path}")

    # Reset kernel pool for reproducibility
    sp.kclear()
    loaded: List[str] = []
    for rel in selected:
        p = kernels_dir / str(rel)
        if not p.exists():
            raise RuntimeError(f"Missing kernel file: {p}")
        sp.furnsh(str(p))
        loaded.append(str(p))
    return {"meta_path": str(meta_path), "meta": meta, "loaded": loaded}


def _wn_intervals(window) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []
    n = int(sp.wncard(window))
    for i in range(n):
        a, b = sp.wnfetd(window, i)
        out.append((float(a), float(b)))
    return out


def _spk_time_bounds(spk_path: Path, obj_id: int) -> Tuple[float, float]:
    cov = sp.spkcov(str(spk_path), int(obj_id))
    intervals = _wn_intervals(cov)
    if not intervals:
        raise RuntimeError(f"SPK coverage empty: {spk_path} (obj={obj_id})")
    return min(a for a, _ in intervals), max(b for _, b in intervals)


def _impact_b_and_bdot(rE_m: np.ndarray, vE_mps: np.ndarray, rS_m: np.ndarray, vS_mps: np.ndarray) -> Tuple[float, float]:
    """
    Compute the closest-approach distance b(t) [m] from the Sun (origin) to the
    *finite* line segment Earth->spacecraft, and its time derivative db/dt.

    Notes:
    - The common formula |rE×rS|/|rS-rE| gives the distance to the *infinite* line.
      For inferior conjunction geometry (Earth and spacecraft on the same side of the Sun),
      the closest point to the infinite line can lie *outside* the segment, which would
      incorrectly yield b≈0. This implementation clamps the closest point to the segment.
    """
    dr = rS_m - rE_m
    dv = vS_mps - vE_mps

    d2 = float(np.dot(dr, dr))
    if d2 == 0.0:
        return float("nan"), float("nan")

    # Closest point on the infinite line: w = rE + t*dr
    a = float(np.dot(rE_m, dr))  # rE·dr
    t_unclamped = -a / d2

    # If the closest point is outside the segment, clamp to the endpoint.
    if t_unclamped <= 0.0:
        w = rE_m
        wdot = vE_mps
        b = float(np.linalg.norm(w))
        if b == 0.0:
            return float("nan"), float("nan")
        return b, float(np.dot(w, wdot)) / b

    if t_unclamped >= 1.0:
        w = rS_m
        wdot = vS_mps
        b = float(np.linalg.norm(w))
        if b == 0.0:
            return float("nan"), float("nan")
        return b, float(np.dot(w, wdot)) / b

    # Interior case: differentiate b = |w| with w = rE + t*dr.
    t = float(t_unclamped)
    a_dot = float(np.dot(vE_mps, dr) + np.dot(rE_m, dv))  # d/dt (rE·dr)
    d2_dot = 2.0 * float(np.dot(dr, dv))  # d/dt (dr·dr)
    t_dot = -(a_dot * d2 - a * d2_dot) / (d2 * d2)

    w = rE_m + t * dr
    wdot = vE_mps + t * dv + t_dot * dr
    b = float(np.linalg.norm(w))
    if b == 0.0:
        return float("nan"), float("nan")
    bdot = float(np.dot(w, wdot)) / b
    return b, bdot


def _shapiro_dt_roundtrip(r1_m: float, r2_m: float, b_m: float, *, beta: float) -> float:
    # Round-trip delay (b approximation). PPN gamma = 2β-1  => (1+gamma)=2β
    one_plus_gamma = 2.0 * float(beta)
    return 2.0 * one_plus_gamma * MU_SUN / (C**3) * math.log((4.0 * r1_m * r2_m) / (b_m * b_m))


def _y_eq2(b_m: float, bdot_mps: float, *, beta: float) -> float:
    # Eq(2) approximation used in Cassini analyses:
    #   y ≈ 4(1+gamma) GM/c^3 * (1/b) db/dt
    one_plus_gamma = 2.0 * float(beta)
    return 4.0 * one_plus_gamma * MU_SUN / (C**3) * (bdot_mps / b_m)


def _y_full(r1_m: float, r1dot_mps: float, r2_m: float, r2dot_mps: float, b_m: float, bdot_mps: float, *, beta: float) -> float:
    # Round-trip Doppler observable (sign convention): y = - d(Δt)/dt
    one_plus_gamma = 2.0 * float(beta)
    coef = 2.0 * one_plus_gamma * MU_SUN / (C**3)
    return -coef * ((r1dot_mps / r1_m) + (r2dot_mps / r2_m) - 2.0 * (bdot_mps / b_m))


def _state_m(target: str, et: float, *, observer: str = "SUN") -> Tuple[np.ndarray, np.ndarray]:
    # SPICE state is km and km/s by default.
    st, _ = sp.spkezr(target, float(et), "J2000", "NONE", observer)
    r_km = np.array(st[:3], dtype=float)
    v_kmps = np.array(st[3:], dtype=float)
    return r_km * 1000.0, v_kmps * 1000.0


def _find_conjunction_center(
    target: str,
    *,
    et_start: float,
    et_stop: float,
    coarse_step_s: float,
    refine_half_window_s: float,
    refine_step_s: float,
    min_b_m: float,
) -> Tuple[float, float]:
    # Coarse scan
    ets = np.arange(et_start, et_stop + 1.0, float(coarse_step_s))
    b_list: List[float] = []
    for et in ets:
        rE, vE = _state_m("EARTH", float(et), observer="SUN")
        rS, vS = _state_m(target, float(et), observer="SUN")
        b, _ = _impact_b_and_bdot(rE, vE, rS, vS)
        if math.isfinite(min_b_m) and min_b_m > 0.0 and b < min_b_m:
            b = float("nan")
        b_list.append(b)
    if not np.isfinite(np.nanmin(np.array(b_list))):
        raise RuntimeError("No valid conjunction candidates in coarse scan (min_b filter too strict?).")
    idx = int(np.nanargmin(np.array(b_list)))
    et0 = float(ets[idx])

    # Refine around coarse min
    et_a = et0 - float(refine_half_window_s)
    et_b = et0 + float(refine_half_window_s)
    ets2 = np.arange(et_a, et_b + 1.0, float(refine_step_s))
    b_list2: List[float] = []
    for et in ets2:
        rE, vE = _state_m("EARTH", float(et), observer="SUN")
        rS, vS = _state_m(target, float(et), observer="SUN")
        b, _ = _impact_b_and_bdot(rE, vE, rS, vS)
        if math.isfinite(min_b_m) and min_b_m > 0.0 and b < min_b_m:
            b = float("nan")
        b_list2.append(b)
    if not np.isfinite(np.nanmin(np.array(b_list2))):
        raise RuntimeError("No valid conjunction candidates in refine scan (min_b filter too strict?).")
    idx2 = int(np.nanargmin(np.array(b_list2)))
    et_min = float(ets2[idx2])
    b_min = float(b_list2[idx2])
    return et_min, b_min


def _build_series(
    target: str,
    *,
    et_center: float,
    half_window_days: float,
    step_s: float,
    beta: float,
    min_b_m: float,
) -> List[SeriesRow]:
    half_s = float(half_window_days) * 86400.0
    ets = np.arange(et_center - half_s, et_center + half_s + 1.0, float(step_s))
    out: List[SeriesRow] = []
    for et in ets:
        rE, vE = _state_m("EARTH", float(et), observer="SUN")
        rS, vS = _state_m(target, float(et), observer="SUN")
        b, bdot = _impact_b_and_bdot(rE, vE, rS, vS)
        occulted = bool(math.isfinite(min_b_m) and min_b_m > 0.0 and b < min_b_m)
        r1 = float(np.linalg.norm(rE))
        r2 = float(np.linalg.norm(rS))
        r1dot = float(np.dot(rE, vE) / r1)
        r2dot = float(np.dot(rS, vS) / r2)
        if occulted:
            dt_rt = float("nan")
            y2 = float("nan")
            yf = float("nan")
        else:
            dt_rt = _shapiro_dt_roundtrip(r1, r2, b, beta=beta)
            y2 = _y_eq2(b, bdot, beta=beta)
            yf = _y_full(r1, r1dot, r2, r2dot, b, bdot, beta=beta)
        t_utc = datetime.fromisoformat(sp.et2utc(float(et), "ISOC", 3)).replace(tzinfo=timezone.utc)
        t_days = (float(et) - float(et_center)) / 86400.0
        out.append(
            SeriesRow(
                et=float(et),
                t_utc=t_utc,
                t_days=t_days,
                r1_m=r1,
                r2_m=r2,
                b_m=b,
                occulted=occulted,
                bdot_mps=bdot,
                r1dot_mps=r1dot,
                r2dot_mps=r2dot,
                dt_roundtrip_s=dt_rt,
                y_eq2=y2,
                y_full=yf,
            )
        )
    return out


def _write_csv(path: Path, rows: Iterable[SeriesRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "time_utc",
                "t_days",
                "et_s",
                "b_m",
                "b_over_rsun",
                "occulted",
                "bdot_mps",
                "r1_m",
                "r2_m",
                "r1dot_mps",
                "r2dot_mps",
                "delta_t_roundtrip_s",
                "y_eq2",
                "y_full",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r.t_utc.isoformat(),
                    f"{r.t_days:.6f}",
                    f"{r.et:.3f}",
                    f"{r.b_m:.6e}",
                    f"{(r.b_m / R_SUN):.6f}",
                    "1" if r.occulted else "0",
                    f"{r.bdot_mps:.6e}",
                    f"{r.r1_m:.6e}",
                    f"{r.r2_m:.6e}",
                    f"{r.r1dot_mps:.6e}",
                    f"{r.r2dot_mps:.6e}",
                    f"{r.dt_roundtrip_s:.12e}",
                    f"{r.y_eq2:.12e}",
                    f"{r.y_full:.12e}",
                ]
            )


def _plot(path: Path, rows: List[SeriesRow], *, title: str, min_b_rsun: float) -> None:
    _set_japanese_font()
    import matplotlib.pyplot as plt

    xs = [r.t_days for r in rows]
    b_rsun = [r.b_m / R_SUN for r in rows]
    dt_us = [r.dt_roundtrip_s * 1e6 for r in rows]
    y2 = [r.y_eq2 for r in rows]

    fig, axes = plt.subplots(3, 1, figsize=(10.5, 7.2), sharex=True, constrained_layout=True)

    axes[0].plot(xs, b_rsun, color="#1f77b4")
    axes[0].axvline(0.0, color="0.5", linestyle="--", linewidth=1.0)
    if math.isfinite(min_b_rsun) and min_b_rsun > 0.0:
        axes[0].axhline(min_b_rsun, color="0.4", linestyle=":", linewidth=1.0, label=f"閾値 b={min_b_rsun:g} R_sun")
    axes[0].set_ylabel("b / R_sun")
    axes[0].set_title("幾何：インパクトパラメータ b（太陽中心からの最短距離）")
    axes[0].grid(True, alpha=0.3)
    if math.isfinite(min_b_rsun) and min_b_rsun > 0.0:
        axes[0].legend(loc="best")

    axes[1].plot(xs, dt_us, color="#2ca02c")
    axes[1].axvline(0.0, color="0.5", linestyle="--", linewidth=1.0)
    axes[1].set_ylabel("往復遅延 Δt [μs]")
    axes[1].set_title("Shapiro遅延（往復, b近似）")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(xs, y2, color="#ff7f0e", label="y(t) Eq(2) 近似")
    axes[2].axhline(0.0, color="0.3", linewidth=1.0)
    axes[2].axvline(0.0, color="0.5", linestyle="--", linewidth=1.0)
    axes[2].set_ylabel("y（周波数比）")
    axes[2].set_title("Doppler y(t)（Shapiro由来, P-model β）")
    axes[2].set_xlabel("t（b_min からの相対日数）")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc="best")

    fig.suptitle(title, fontsize=14)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description="BepiColombo (MPO) Shapiro geometry prediction from SPICE kernels.")
    ap.add_argument("--beta", type=float, default=1.0, help="P-model beta (default: 1.0)")
    ap.add_argument(
        "--min-b-rsun",
        type=float,
        default=1.0,
        help="Minimum b/R_sun to accept as a conjunction center (default: 1.0; set 0 to allow b<R_sun).",
    )
    ap.add_argument("--half-window-days", type=float, default=10.0, help="Plot window half-width in days (default: 10)")
    ap.add_argument("--step-sec", type=float, default=3600.0, help="Time step seconds for the final series (default: 3600)")
    ap.add_argument("--coarse-step-days", type=float, default=1.0, help="Coarse scan step days (default: 1)")
    ap.add_argument("--refine-window-days", type=float, default=30.0, help="Refine window half-width days (default: 30)")
    ap.add_argument("--refine-step-sec", type=float, default=3600.0, help="Refine step seconds (default: 3600)")
    ap.add_argument("--target-id", type=int, default=None, help="Override target NAIF ID (default: auto from SPK)")
    args = ap.parse_args()

    root = _ROOT
    kernels_dir = root / "data" / "bepicolombo" / "kernels" / "psa"
    out_dir = root / "output" / "private" / "bepicolombo"
    out_dir.mkdir(parents=True, exist_ok=True)

    load_info = _load_kernels_from_meta(kernels_dir)
    meta: Dict[str, Any] = load_info["meta"]

    spk_rel = next((p for p in (meta.get("selected_paths") or []) if isinstance(p, str) and p.startswith("spk/") and "bc_mpo_fcp_" in p), None)
    if not spk_rel:
        raise RuntimeError("Could not find MPO SPK in kernels_meta.json selected_paths.")
    spk_path = kernels_dir / spk_rel

    obj_ids = list(sp.spkobj(str(spk_path)))
    if not obj_ids:
        raise RuntimeError(f"SPK has no objects: {spk_path}")

    if args.target_id is not None:
        target_id = int(args.target_id)
    else:
        # Heuristic: pick the first negative ID (spacecraft) if present, else the first one.
        neg = [i for i in obj_ids if int(i) < 0]
        target_id = int(neg[0] if neg else obj_ids[0])

    target = str(target_id)
    et_start, et_stop = _spk_time_bounds(spk_path, target_id)

    et_center, b_min = _find_conjunction_center(
        target,
        et_start=et_start,
        et_stop=et_stop,
        coarse_step_s=float(args.coarse_step_days) * 86400.0,
        refine_half_window_s=float(args.refine_window_days) * 86400.0,
        refine_step_s=float(args.refine_step_sec),
        min_b_m=float(args.min_b_rsun) * R_SUN,
    )

    rows = _build_series(
        target,
        et_center=et_center,
        half_window_days=float(args.half_window_days),
        step_s=float(args.step_sec),
        beta=float(args.beta),
        min_b_m=float(args.min_b_rsun) * R_SUN,
    )

    bmin_rsun = float(b_min / R_SUN)
    b_min_raw_rsun = float(min((r.b_m for r in rows), default=float("nan")) / R_SUN)
    y_peak = max((abs(r.y_eq2) for r in rows if not r.occulted and math.isfinite(r.y_eq2)), default=float("nan"))
    y_peak_full = max((abs(r.y_full) for r in rows if not r.occulted and math.isfinite(r.y_full)), default=float("nan"))
    dt_vals = [r.dt_roundtrip_s for r in rows if not r.occulted and math.isfinite(r.dt_roundtrip_s)]
    dt_min_us = (min(dt_vals) if dt_vals else float("nan")) * 1e6
    dt_max_us = (max(dt_vals) if dt_vals else float("nan")) * 1e6

    t0_utc = datetime.fromisoformat(sp.et2utc(float(et_center), "ISOC", 3)).replace(tzinfo=timezone.utc)

    csv_path = out_dir / "bepicolombo_shapiro_geometry.csv"
    png_path = out_dir / "bepicolombo_shapiro_geometry.png"
    json_path = out_dir / "bepicolombo_shapiro_geometry_summary.json"

    _write_csv(csv_path, rows)
    _plot(
        png_path,
        rows,
        title=f"BepiColombo（MPO, NAIF={target_id}）: 太陽会合のShapiro予測（β={float(args.beta):.6f}）",
        min_b_rsun=float(args.min_b_rsun),
    )

    summary: Dict[str, Any] = {
        "generated_utc": _utc_now_iso(),
        "beta": float(args.beta),
        "min_b_rsun": float(args.min_b_rsun),
        "target_id": int(target_id),
        "spk_path": str(spk_path),
        "coverage_et": {"start": float(et_start), "stop": float(et_stop)},
        "conjunction_center_utc": t0_utc.isoformat(),
        "b_min_rsun": bmin_rsun,
        "b_min_raw_rsun_in_window": b_min_raw_rsun,
        "y_peak_eq2": float(y_peak),
        "y_peak_full": float(y_peak_full),
        "dt_roundtrip_us_range": [float(dt_min_us), float(dt_max_us)],
        "series": {
            "half_window_days": float(args.half_window_days),
            "step_sec": float(args.step_sec),
            "n": len(rows),
        },
        "inputs": {
            "kernels_meta_json": str(kernels_dir / "kernels_meta.json"),
            "loaded_kernels": load_info.get("loaded", []),
        },
        "outputs": {
            "csv": str(csv_path),
            "png": str(png_path),
            "summary_json": str(json_path),
        },
        "notes": [
            "この出力は『観測データとの比較』ではなく、一次ソース（SPICE）に基づく幾何＋P-model（β）での予測曲線。",
            "MORE の range/Doppler が PSA 等で公開され次第、同じ時系列軸で観測 y(t) を重ねて評価する。",
        ],
    }
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[ok] wrote:", csv_path)
    print("[ok] wrote:", png_path)
    print("[ok] wrote:", json_path)
    print("conjunction_center_utc:", t0_utc.isoformat())
    print("b_min / R_sun:", f"{bmin_rsun:.6f}")
    print("y_peak Eq(2):", f"{y_peak:.6e}")
    print("dt_roundtrip range (us):", f"{dt_min_us:.3f} .. {dt_max_us:.3f}")

    try:
        worklog.append_event(
            {
                "event_type": "bepicolombo_shapiro_predict",
                "argv": sys.argv,
                "params": {
                    "beta": float(args.beta),
                    "min_b_rsun": float(args.min_b_rsun),
                    "half_window_days": float(args.half_window_days),
                    "step_sec": float(args.step_sec),
                },
                "outputs": {
                    "png": png_path,
                    "csv": csv_path,
                    "summary_json": json_path,
                },
            }
        )
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
