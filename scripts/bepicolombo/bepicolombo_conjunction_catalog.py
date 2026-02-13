#!/usr/bin/env python3
"""
bepicolombo_conjunction_catalog.py

BepiColombo（MPO）×地球の太陽会合（solar conjunction）イベントを、SPICE幾何から一覧化する。

目的：
  - BepiColombo（MORE）の一次データ（range/Doppler）が未公開/未取得でも、
    「いつ・どれくらい強い Shapiro 信号が期待されるか」を先に固定する。
  - 後で観測データが入手できた瞬間に、同じイベント軸で “観測 vs P-model” へ接続する。

入力（オフライン再現）：
  - data/bepicolombo/kernels/psa/kernels_meta.json（selected_paths をロード）

出力（固定）：
  - output/bepicolombo/bepicolombo_conjunction_catalog.csv
  - output/bepicolombo/bepicolombo_conjunction_catalog.json
  - output/bepicolombo/bepicolombo_conjunction_catalog_summary.json
  - output/bepicolombo/bepicolombo_conjunction_catalog.png
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
from typing import Any, Dict, List, Optional, Tuple

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
class EventRow:
    event_index: int
    t_min_raw_utc: datetime
    b_min_raw_rsun: float
    occulted_raw: bool
    t_min_nonocculted_utc: Optional[datetime]
    b_min_nonocculted_rsun: Optional[float]
    dt_roundtrip_us_at_min_nonocculted: Optional[float]
    y_peak_eq2: Optional[float]


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

    sp.kclear()
    loaded: List[str] = []
    for rel in selected:
        p = kernels_dir / str(rel)
        if not p.exists():
            raise RuntimeError(f"Missing kernel file: {p}")
        sp.furnsh(str(p))
        loaded.append(str(p))
    return {"meta_path": str(meta_path), "meta": meta, "loaded": loaded}


def _state_m(target: str, et: float, *, observer: str = "SUN") -> Tuple[np.ndarray, np.ndarray]:
    st, _ = sp.spkezr(target, float(et), "J2000", "NONE", observer)
    r_km = np.array(st[:3], dtype=float)
    v_kmps = np.array(st[3:], dtype=float)
    return r_km * 1000.0, v_kmps * 1000.0


def _impact_b_and_bdot(rE_m: np.ndarray, vE_mps: np.ndarray, rS_m: np.ndarray, vS_mps: np.ndarray) -> Tuple[float, float]:
    """
    Closest-approach distance b(t) [m] from the Sun (origin) to the *segment* Earth->spacecraft,
    and its time derivative db/dt.
    """
    dr = rS_m - rE_m
    dv = vS_mps - vE_mps

    d2 = float(np.dot(dr, dr))
    if d2 == 0.0:
        return float("nan"), float("nan")

    a = float(np.dot(rE_m, dr))
    t_unclamped = -a / d2

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

    t = float(t_unclamped)
    a_dot = float(np.dot(vE_mps, dr) + np.dot(rE_m, dv))
    d2_dot = 2.0 * float(np.dot(dr, dv))
    t_dot = -(a_dot * d2 - a * d2_dot) / (d2 * d2)

    w = rE_m + t * dr
    wdot = vE_mps + t * dv + t_dot * dr
    b = float(np.linalg.norm(w))
    if b == 0.0:
        return float("nan"), float("nan")
    bdot = float(np.dot(w, wdot)) / b
    return b, bdot


def _shapiro_dt_roundtrip(r1_m: float, r2_m: float, b_m: float, *, beta: float) -> float:
    one_plus_gamma = 2.0 * float(beta)  # PPN: (1+gamma)=2β
    return 2.0 * one_plus_gamma * MU_SUN / (C**3) * math.log((4.0 * r1_m * r2_m) / (b_m * b_m))


def _y_eq2(b_m: float, bdot_mps: float, *, beta: float) -> float:
    one_plus_gamma = 2.0 * float(beta)
    return 4.0 * one_plus_gamma * MU_SUN / (C**3) * (bdot_mps / b_m)


def _local_minima_indices(y: np.ndarray) -> List[int]:
    idx: List[int] = []
    n = int(len(y))
    for i in range(1, n - 1):
        yi = float(y[i])
        if not math.isfinite(yi):
            continue
        yp = float(y[i - 1])
        yn = float(y[i + 1])
        if not (math.isfinite(yp) and math.isfinite(yn)):
            continue
        if yi <= yp and yi <= yn:
            idx.append(i)
    return idx


def _select_event_guesses(ets: np.ndarray, b_m: np.ndarray, *, max_b_m: float, merge_window_s: float) -> List[Tuple[float, float]]:
    mins = _local_minima_indices(b_m)
    cands: List[Tuple[float, float]] = []
    for i in mins:
        bi = float(b_m[i])
        if not math.isfinite(bi):
            continue
        if bi > float(max_b_m):
            continue
        cands.append((float(ets[i]), bi))

    cands.sort(key=lambda x: x[1])
    picked: List[Tuple[float, float]] = []
    for et, b in cands:
        if all(abs(et - et2) > float(merge_window_s) for et2, _ in picked):
            picked.append((et, b))

    picked.sort(key=lambda x: x[0])
    return picked


def _refine_event(
    target: str,
    et_guess: float,
    *,
    beta: float,
    min_b_m: float,
    refine_half_window_s: float,
    refine_step_s: float,
) -> Tuple[EventRow, Dict[str, Any]]:
    ets = np.arange(et_guess - float(refine_half_window_s), et_guess + float(refine_half_window_s) + 1.0, float(refine_step_s))

    best_raw: Optional[Tuple[float, float]] = None  # (b, et)
    best_nonocc: Optional[Tuple[float, float, float, float]] = None  # (b, et, r1, r2)
    y_peak: Optional[float] = None

    for et in ets:
        rE, vE = _state_m("EARTH", float(et), observer="SUN")
        rS, vS = _state_m(target, float(et), observer="SUN")

        r1 = float(np.linalg.norm(rE))
        r2 = float(np.linalg.norm(rS))
        if r1 == 0.0 or r2 == 0.0:
            continue

        b, bdot = _impact_b_and_bdot(rE, vE, rS, vS)
        if not math.isfinite(b):
            continue

        if best_raw is None or b < best_raw[0]:
            best_raw = (float(b), float(et))

        if b >= float(min_b_m):
            if best_nonocc is None or b < best_nonocc[0]:
                best_nonocc = (float(b), float(et), r1, r2)
            y = _y_eq2(float(b), float(bdot), beta=float(beta))
            y_abs = abs(float(y))
            if y_peak is None or y_abs > abs(float(y_peak)):
                y_peak = float(y)

    if best_raw is None:
        raise RuntimeError("No valid b in refine window.")

    b_raw, et_raw = best_raw
    t_raw = datetime.fromisoformat(sp.et2utc(float(et_raw), "ISOC", 3)).replace(tzinfo=timezone.utc)
    occulted_raw = b_raw < R_SUN

    t_non: Optional[datetime] = None
    b_non_rsun: Optional[float] = None
    dt_us: Optional[float] = None
    if best_nonocc is not None:
        b_non, et_non, r1_non, r2_non = best_nonocc
        t_non = datetime.fromisoformat(sp.et2utc(float(et_non), "ISOC", 3)).replace(tzinfo=timezone.utc)
        b_non_rsun = float(b_non) / R_SUN
        dt_us = 1e6 * _shapiro_dt_roundtrip(float(r1_non), float(r2_non), float(b_non), beta=float(beta))

    row = EventRow(
        event_index=-1,
        t_min_raw_utc=t_raw,
        b_min_raw_rsun=float(b_raw) / R_SUN,
        occulted_raw=bool(occulted_raw),
        t_min_nonocculted_utc=t_non,
        b_min_nonocculted_rsun=b_non_rsun,
        dt_roundtrip_us_at_min_nonocculted=dt_us,
        y_peak_eq2=y_peak,
    )

    extra = {
        "et_guess": float(et_guess),
        "refine": {
            "half_window_days": float(refine_half_window_s) / 86400.0,
            "step_sec": float(refine_step_s),
        },
    }
    return row, extra


def _write_csv(path: Path, rows: List[EventRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "event_index",
                "t_min_raw_utc",
                "b_min_raw_rsun",
                "occulted_raw",
                "t_min_nonocculted_utc",
                "b_min_nonocculted_rsun",
                "dt_roundtrip_us_at_min_nonocculted",
                "y_peak_eq2",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r.event_index,
                    r.t_min_raw_utc.isoformat(),
                    f"{r.b_min_raw_rsun:.9f}",
                    int(bool(r.occulted_raw)),
                    "" if r.t_min_nonocculted_utc is None else r.t_min_nonocculted_utc.isoformat(),
                    "" if r.b_min_nonocculted_rsun is None else f"{float(r.b_min_nonocculted_rsun):.9f}",
                    "" if r.dt_roundtrip_us_at_min_nonocculted is None else f"{float(r.dt_roundtrip_us_at_min_nonocculted):.6f}",
                    "" if r.y_peak_eq2 is None else f"{float(r.y_peak_eq2):.12e}",
                ]
            )


def _plot(path: Path, rows: List[EventRow], *, title: str) -> None:
    _set_japanese_font()
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    xs: List[datetime] = []
    b_vals: List[float] = []
    dt_vals: List[float] = []
    occult_flags: List[bool] = []

    for r in rows:
        t = r.t_min_nonocculted_utc or r.t_min_raw_utc
        xs.append(t)
        b_vals.append(float(r.b_min_nonocculted_rsun or r.b_min_raw_rsun))
        dt_vals.append(float(r.dt_roundtrip_us_at_min_nonocculted or float("nan")))
        occult_flags.append(bool(r.occulted_raw))

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={"hspace": 0.18})
    ax1, ax2 = axes

    # b_min
    c1 = ["tab:red" if occ else "tab:blue" for occ in occult_flags]
    ax1.scatter(xs, b_vals, s=30, c=c1)
    ax1.set_ylabel("最小インパクトパラメータ b [R_sun]")
    ax1.grid(True, alpha=0.3)

    # dt
    ax2.scatter(xs, dt_vals, s=30, c=c1)
    ax2.set_ylabel("往復 Shapiro 遅延 Δt [μs]（非遮蔽側）")
    ax2.set_xlabel("UTC 時刻")
    ax2.grid(True, alpha=0.3)

    locator = mdates.AutoDateLocator(minticks=6, maxticks=10)
    ax2.xaxis.set_major_locator(locator)
    ax2.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

    fig.suptitle(title)

    # Small legend hint
    from matplotlib.lines import Line2D

    legend_items = [
        Line2D([0], [0], marker="o", color="w", label="非遮蔽（min_b>=設定）", markerfacecolor="tab:blue", markersize=8),
        Line2D([0], [0], marker="o", color="w", label="遮蔽あり（raw min が太陽円盤内）", markerfacecolor="tab:red", markersize=8),
    ]
    ax1.legend(handles=legend_items, loc="upper right")

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description="BepiColombo (MPO) conjunction catalog from SPICE kernels.")
    ap.add_argument("--beta", type=float, default=1.0, help="P-model β (PPN: (1+γ)=2β).")
    ap.add_argument("--target-id", type=int, default=-121, help="NAIF ID of MPO (default: -121).")
    ap.add_argument("--kernels-dir", type=str, default="", help="Override kernels cache dir (default: data/bepicolombo/kernels/psa).")
    ap.add_argument("--offline", action="store_true", help="Offline mode (no network; kept for symmetry).")

    ap.add_argument("--coarse-step-hours", type=float, default=6.0, help="Coarse scan step [hours] over coverage.")
    ap.add_argument("--max-b-rsun", type=float, default=10.0, help="Detect conjunction candidates with b <= max_b [R_sun].")
    ap.add_argument("--merge-days", type=float, default=30.0, help="Merge event guesses closer than this [days].")

    ap.add_argument("--refine-half-window-days", type=float, default=5.0, help="Refine each event within +/- window [days].")
    ap.add_argument("--refine-step-sec", type=float, default=600.0, help="Refine step [sec].")
    ap.add_argument("--min-b-rsun", type=float, default=1.0, help="Non-occulted minimum b threshold [R_sun].")
    args = ap.parse_args()

    root = _ROOT
    out_dir = root / "output" / "private" / "bepicolombo"
    out_dir.mkdir(parents=True, exist_ok=True)

    kernels_dir = Path(args.kernels_dir) if str(args.kernels_dir).strip() else (root / "data" / "bepicolombo" / "kernels" / "psa")
    loaded = _load_kernels_from_meta(kernels_dir)
    meta = loaded.get("meta") if isinstance(loaded.get("meta"), dict) else {}

    target_id = int(args.target_id)
    target = str(target_id)

    # Determine MPO SPK (prefer the one selected in kernels_meta.json for reproducibility)
    spk_path: Optional[Path] = None
    sel = meta.get("selected_paths") if isinstance(meta, dict) else None
    if isinstance(sel, list):
        spk_rel = next((p for p in sel if isinstance(p, str) and p.startswith("spk/") and "bc_mpo_fcp_" in p), None)
        if spk_rel:
            spk_path = kernels_dir / str(spk_rel)

    if spk_path is None:
        for p in sorted((kernels_dir / "spk").glob("*.bsp")):
            if "bc_mpo" in p.name.lower():
                spk_path = p
                break

    if spk_path is None or not spk_path.exists():
        raise RuntimeError("Could not find MPO SPK (bc_mpo_fcp_*.bsp). Re-run fetch_spice_kernels_psa.py.")

    et_start, et_stop = _spk_time_bounds(spk_path, target_id)

    coarse_step_s = float(args.coarse_step_hours) * 3600.0
    ets = np.arange(float(et_start), float(et_stop) + 1.0, coarse_step_s)
    b_list: List[float] = []
    for et in ets:
        rE, vE = _state_m("EARTH", float(et), observer="SUN")
        rS, vS = _state_m(target, float(et), observer="SUN")
        b, _ = _impact_b_and_bdot(rE, vE, rS, vS)
        b_list.append(float(b))

    b_arr = np.array(b_list, dtype=float)

    max_b_m = float(args.max_b_rsun) * R_SUN
    merge_window_s = float(args.merge_days) * 86400.0
    guesses = _select_event_guesses(ets, b_arr, max_b_m=max_b_m, merge_window_s=merge_window_s)

    min_b_m = float(args.min_b_rsun) * R_SUN
    refine_half_s = float(args.refine_half_window_days) * 86400.0
    refine_step_s = float(args.refine_step_sec)

    events: List[EventRow] = []
    extras: List[Dict[str, Any]] = []
    for i, (et_guess, _b_guess) in enumerate(guesses, start=1):
        row, extra = _refine_event(
            target,
            float(et_guess),
            beta=float(args.beta),
            min_b_m=min_b_m,
            refine_half_window_s=refine_half_s,
            refine_step_s=refine_step_s,
        )
        row = EventRow(
            event_index=i,
            t_min_raw_utc=row.t_min_raw_utc,
            b_min_raw_rsun=row.b_min_raw_rsun,
            occulted_raw=row.occulted_raw,
            t_min_nonocculted_utc=row.t_min_nonocculted_utc,
            b_min_nonocculted_rsun=row.b_min_nonocculted_rsun,
            dt_roundtrip_us_at_min_nonocculted=row.dt_roundtrip_us_at_min_nonocculted,
            y_peak_eq2=row.y_peak_eq2,
        )
        events.append(row)
        extra["event_index"] = i
        extras.append(extra)

    csv_path = out_dir / "bepicolombo_conjunction_catalog.csv"
    json_path = out_dir / "bepicolombo_conjunction_catalog.json"
    summary_path = out_dir / "bepicolombo_conjunction_catalog_summary.json"
    png_path = out_dir / "bepicolombo_conjunction_catalog.png"

    _write_csv(csv_path, events)
    json_path.write_text(
        json.dumps([r.__dict__ for r in events], ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    # Summary
    usable = [r for r in events if r.b_min_nonocculted_rsun is not None]
    b_min = min((float(r.b_min_nonocculted_rsun) for r in usable), default=float("nan"))
    dt_max = max((float(r.dt_roundtrip_us_at_min_nonocculted or float("nan")) for r in usable), default=float("nan"))
    span_utc = {
        "start": (usable[0].t_min_nonocculted_utc or usable[0].t_min_raw_utc).isoformat() if usable else None,
        "stop": (usable[-1].t_min_nonocculted_utc or usable[-1].t_min_raw_utc).isoformat() if usable else None,
    }

    summary: Dict[str, Any] = {
        "generated_utc": _utc_now_iso(),
        "beta": float(args.beta),
        "min_b_rsun": float(args.min_b_rsun),
        "max_b_rsun": float(args.max_b_rsun),
        "target_id": int(target_id),
        "spk_path": str(spk_path),
        "coverage_et": {"start": float(et_start), "stop": float(et_stop)},
        "scan": {
            "coarse_step_hours": float(args.coarse_step_hours),
            "merge_days": float(args.merge_days),
            "refine_half_window_days": float(args.refine_half_window_days),
            "refine_step_sec": float(args.refine_step_sec),
        },
        "summary": {
            "n_events": int(len(events)),
            "n_usable_events": int(len(usable)),
            "b_min_nonocculted_rsun_min": b_min,
            "dt_roundtrip_us_max": dt_max,
            "span_utc": span_utc,
        },
        "events": [
            {
                "event_index": r.event_index,
                "t_min_raw_utc": r.t_min_raw_utc.isoformat(),
                "b_min_raw_rsun": float(r.b_min_raw_rsun),
                "occulted_raw": bool(r.occulted_raw),
                "t_min_nonocculted_utc": None if r.t_min_nonocculted_utc is None else r.t_min_nonocculted_utc.isoformat(),
                "b_min_nonocculted_rsun": None if r.b_min_nonocculted_rsun is None else float(r.b_min_nonocculted_rsun),
                "dt_roundtrip_us_at_min_nonocculted": None
                if r.dt_roundtrip_us_at_min_nonocculted is None
                else float(r.dt_roundtrip_us_at_min_nonocculted),
                "y_peak_eq2": None if r.y_peak_eq2 is None else float(r.y_peak_eq2),
            }
            for r in events
        ],
        "debug": {"event_refine_meta": extras},
        "inputs": {
            "kernels_meta_json": loaded.get("meta_path"),
            "loaded_kernels": loaded.get("loaded", []),
        },
        "outputs": {
            "csv": str(csv_path),
            "json": str(json_path),
            "summary_json": str(summary_path),
            "png": str(png_path),
        },
        "notes": [
            "この出力は観測データとの比較ではなく、一次ソース（SPICE）に基づく会合イベントの抽出と Shapiro 予測。",
            "MORE の range/Doppler が公開されたら、同じイベントの時刻窓で観測 y(t) を重ねて評価する。",
        ],
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    title = "BepiColombo（MPO）: 太陽会合イベント一覧（Shapiro予測, SPICE幾何）"
    _plot(png_path, events, title=title)

    # Log machine history
    try:
        worklog.append_event(
            {
                "event_type": "bepicolombo_conjunction_catalog",
                "argv": sys.argv,
                "params": {
                    "beta": float(args.beta),
                    "min_b_rsun": float(args.min_b_rsun),
                    "max_b_rsun": float(args.max_b_rsun),
                    "coarse_step_hours": float(args.coarse_step_hours),
                    "merge_days": float(args.merge_days),
                    "refine_half_window_days": float(args.refine_half_window_days),
                    "refine_step_sec": float(args.refine_step_sec),
                },
                "outputs": {
                    "png": png_path,
                    "csv": csv_path,
                    "summary_json": summary_path,
                },
            }
        )
    except Exception:
        pass

    print(f"[ok] wrote: {csv_path}")
    print(f"[ok] wrote: {json_path}")
    print(f"[ok] wrote: {summary_path}")
    print(f"[ok] wrote: {png_path}")
    print(f"n_events: {len(events)} (usable={len(usable)})")
    if usable:
        print(f"min b (non-occulted) [R_sun]: {b_min:.6f}")
        print(f"max dt_roundtrip [us]: {dt_max:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
