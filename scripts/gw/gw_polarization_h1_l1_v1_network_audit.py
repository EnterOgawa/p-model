from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.gw.gw_polarization_antenna_pattern_audit import (  # noqa: E402
    _C,
    _DETECTOR_SITES,
    _build_network_geometry,
    _canonical_pair,
    _estimate_delay_tolerance_s,
    _fibonacci_sphere,
    _fmt,
    _interval_overlap,
    _load_lag_scan,
    _load_metrics,
    _min_rel_mismatch,
    _range_clip,
    _response_grid_for_direction,
    _safe_float,
    _set_japanese_font,
    _slugify,
)
from scripts.summary import worklog  # noqa: E402


def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _detector_order(det: str) -> int:
    order = {"H1": 0, "L1": 1, "V1": 2, "K1": 3}
    return int(order.get(str(det).upper(), 999))


def _ring_dirs_from_constraints(
    *,
    constraints: List[Dict[str, Any]],
    geom: Dict[str, Dict[str, Any]],
    sky_dirs: np.ndarray,
    min_ring_directions: int,
    geometry_relax_factor: float,
    geometry_delay_floor_s: float,
) -> Tuple[np.ndarray, str]:
    if not constraints:
        return sky_dirs[:0], "none"

    mask = np.ones(sky_dirs.shape[0], dtype=bool)
    for c in constraints:
        baseline = geom[c["first"]]["position_m"] - geom[c["second"]]["position_m"]
        dt_pred = -(sky_dirs @ baseline) / _C
        mask &= np.abs(dt_pred - float(c["delay_s"])) <= float(c["delay_tol_s"])
    ring_strict = sky_dirs[mask]
    if int(ring_strict.shape[0]) >= int(min_ring_directions):
        return ring_strict, "strict"

    mask_relaxed = np.ones(sky_dirs.shape[0], dtype=bool)
    for c in constraints:
        baseline = geom[c["first"]]["position_m"] - geom[c["second"]]["position_m"]
        dt_pred = -(sky_dirs @ baseline) / _C
        tol = max(float(c["delay_tol_s"]), float(geometry_delay_floor_s)) * float(geometry_relax_factor)
        mask_relaxed &= np.abs(dt_pred - float(c["delay_s"])) <= tol
    ring_relaxed = sky_dirs[mask_relaxed]
    if int(ring_relaxed.shape[0]) >= int(min_ring_directions):
        return ring_relaxed, "relaxed"
    return ring_relaxed, "relaxed_insufficient"


def _select_pruned_constraints(
    *,
    constraints: List[Dict[str, Any]],
    geom: Dict[str, Dict[str, Any]],
    sky_dirs: np.ndarray,
    min_ring_directions: int,
    geometry_relax_factor: float,
    geometry_delay_floor_s: float,
) -> Optional[Dict[str, Any]]:
    if len(constraints) < 3:
        return None

    best_candidate: Optional[Dict[str, Any]] = None
    for subset_size in range(len(constraints) - 1, 1, -1):
        size_best: Optional[Dict[str, Any]] = None
        for subset in combinations(constraints, subset_size):
            subset_list = list(subset)
            ring_dirs, geometry_mode = _ring_dirs_from_constraints(
                constraints=subset_list,
                geom=geom,
                sky_dirs=sky_dirs,
                min_ring_directions=int(min_ring_directions),
                geometry_relax_factor=float(geometry_relax_factor),
                geometry_delay_floor_s=float(geometry_delay_floor_s),
            )
            ring_n = int(ring_dirs.shape[0])
            if ring_n < int(min_ring_directions):
                continue
            corr_sum = float(sum(_safe_float(c.get("abs_corr")) for c in subset_list))
            candidate = {
                "constraints": subset_list,
                "ring_dirs": ring_dirs,
                "ring_n": ring_n,
                "geometry_mode": geometry_mode,
                "corr_sum": corr_sum,
            }
            if size_best is None:
                size_best = candidate
                continue
            if int(candidate["ring_n"]) > int(size_best["ring_n"]):
                size_best = candidate
            elif int(candidate["ring_n"]) == int(size_best["ring_n"]) and float(candidate["corr_sum"]) > float(
                size_best["corr_sum"]
            ):
                size_best = candidate
        if size_best is not None:
            best_candidate = size_best
            break
    if best_candidate is None:
        return None

    used_pairs = {str(c["pair"]) for c in best_candidate["constraints"]}
    dropped_pairs = sorted([str(c["pair"]) for c in constraints if str(c["pair"]) not in used_pairs])
    best_candidate["dropped_pairs"] = dropped_pairs
    return best_candidate


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    headers = [
        "event",
        "quality",
        "available_pair_count",
        "usable_pair_count",
        "pair_names",
        "pair_pruning_applied",
        "dropped_pairs",
        "geometry_mode",
        "pair_tensor_overlap_count",
        "pair_scalar_overlap_count",
        "status",
        "status_reason",
        "scalar_overlap_fraction",
        "tensor_mismatch_max",
        "scalar_mismatch_max",
        "ring_directions_used",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for row in rows:
            vals: List[Any] = []
            for h in headers:
                v = row.get(h, "")
                vals.append(_fmt(v) if isinstance(v, float) else v)
            w.writerow(vals)


def _plot(rows: List[Dict[str, Any]], out_png: Path) -> None:
    _set_japanese_font()
    labels = [str(r.get("event", "")) for r in rows]
    x = np.arange(len(rows), dtype=float)
    scalar_frac = np.array([_safe_float(r.get("scalar_overlap_fraction")) for r in rows], dtype=float)
    pair_tensor = np.array([_safe_float(r.get("pair_tensor_overlap_count")) for r in rows], dtype=float)
    pair_used = np.array([max(_safe_float(r.get("usable_pair_count")), 1.0) for r in rows], dtype=float)
    tensor_frac = np.clip(pair_tensor / pair_used, 0.0, 1.0)

    t_mis = np.array([_safe_float(r.get("tensor_mismatch_max")) for r in rows], dtype=float)
    s_mis = np.array([_safe_float(r.get("scalar_mismatch_max")) for r in rows], dtype=float)

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(12.8, 8.0), sharex=True)
    width = 0.38
    ax0.bar(x - width / 2.0, tensor_frac, width=width, color="#2ca02c", alpha=0.85, label="tensor overlap fraction")
    ax0.bar(x + width / 2.0, scalar_frac, width=width, color="#f58518", alpha=0.85, label="scalar overlap fraction")
    ax0.set_ylim(0.0, 1.05)
    ax0.set_ylabel("overlap fraction")
    ax0.grid(True, axis="y", alpha=0.25)
    ax0.legend(loc="upper right", fontsize=9)
    ax0.set_title("Three-detector polarization overlap diagnostics")

    ax1.bar(x - width / 2.0, t_mis, width=width, color="#1f77b4", alpha=0.9, label="tensor max mismatch")
    ax1.bar(x + width / 2.0, s_mis, width=width, color="#d62728", alpha=0.9, label="scalar max mismatch")
    ax1.set_ylabel("max relative mismatch")
    ax1.set_xlabel("event")
    ax1.grid(True, axis="y", alpha=0.25)
    ax1.legend(loc="upper right", fontsize=9)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=0, fontsize=9)

    fig.suptitle("GW polarization network audit (H1/L1/V1; Step 8.7.19.3)", fontsize=14)
    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.96])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _pair_factor_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        if str(row.get("quality", "")) != "usable":
            continue
        event_status = str(row.get("status", ""))
        if not event_status.startswith(("reject_", "watch_", "pass_")):
            continue
        event = str(row.get("event", ""))
        pair_results = row.get("pair_results")
        if not isinstance(pair_results, list):
            continue
        for pr in pair_results:
            if not isinstance(pr, dict):
                continue
            delay_s = _safe_float(pr.get("delay_s"))
            delay_tol_s = _safe_float(pr.get("delay_tol_s"))
            lag_ms = float("nan")
            lag_tol_ms = float("nan")
            lag_over_tol = float("nan")
            if math.isfinite(delay_s):
                lag_ms = float(-delay_s * 1.0e3)
            if math.isfinite(delay_tol_s):
                lag_tol_ms = float(delay_tol_s * 1.0e3)
            if math.isfinite(lag_ms) and math.isfinite(lag_tol_ms) and lag_tol_ms > 0.0:
                lag_over_tol = float(abs(lag_ms) / lag_tol_ms)

            obs_p16 = _safe_float(pr.get("obs_p16"))
            obs_med = _safe_float(pr.get("obs_med"))
            obs_p84 = _safe_float(pr.get("obs_p84"))
            obs_iqr_rel = float("nan")
            if math.isfinite(obs_p16) and math.isfinite(obs_p84) and math.isfinite(obs_med) and abs(obs_med) > 0.0:
                obs_iqr_rel = float(abs(obs_p84 - obs_p16) / abs(obs_med))

            out.append(
                {
                    "event": event,
                    "pair": str(pr.get("pair", "")),
                    "event_status": event_status,
                    "event_is_tensor_reject": int(event_status == "reject_tensor_response"),
                    "tensor_overlap": int(bool(pr.get("tensor_overlap"))),
                    "scalar_overlap": int(bool(pr.get("scalar_overlap"))),
                    "tensor_mismatch": _safe_float(pr.get("tensor_mismatch")),
                    "scalar_mismatch": _safe_float(pr.get("scalar_mismatch")),
                    "abs_corr": _safe_float(pr.get("abs_corr")),
                    "best_lag_ms_apply_to_first": lag_ms,
                    "lag_tolerance_ms": lag_tol_ms,
                    "abs_lag_over_tolerance": lag_over_tol,
                    "obs_ratio_p16": obs_p16,
                    "obs_ratio_median": obs_med,
                    "obs_ratio_p84": obs_p84,
                    "obs_ratio_iqr_rel": obs_iqr_rel,
                }
            )
    return out


def _median(values: List[float]) -> float:
    arr = np.asarray([v for v in values if math.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    return float(np.median(arr))


def _pair_factor_summary(detail_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_pair: Dict[str, List[Dict[str, Any]]] = {}
    for row in detail_rows:
        pair = str(row.get("pair", ""))
        by_pair.setdefault(pair, []).append(row)

    reject_event_sum: Dict[str, float] = {}
    for row in detail_rows:
        if int(row.get("event_is_tensor_reject", 0)) != 1:
            continue
        event = str(row.get("event", ""))
        value = _safe_float(row.get("tensor_mismatch"))
        if not math.isfinite(value):
            continue
        reject_event_sum[event] = reject_event_sum.get(event, 0.0) + max(0.0, float(value))

    summary_rows: List[Dict[str, Any]] = []
    for pair, rows_pair in by_pair.items():
        n_rows = int(len(rows_pair))
        n_reject_rows = int(sum(int(r.get("event_is_tensor_reject", 0)) for r in rows_pair))
        n_tensor_fail = int(sum(1 for r in rows_pair if int(r.get("tensor_overlap", 0)) == 0))
        n_scalar_overlap = int(sum(1 for r in rows_pair if int(r.get("scalar_overlap", 0)) == 1))

        reject_share_vals: List[float] = []
        for r in rows_pair:
            if int(r.get("event_is_tensor_reject", 0)) != 1:
                continue
            event = str(r.get("event", ""))
            denom = float(reject_event_sum.get(event, 0.0))
            num = _safe_float(r.get("tensor_mismatch"))
            if denom > 0.0 and math.isfinite(num):
                reject_share_vals.append(float(max(0.0, num) / denom))

        summary_rows.append(
            {
                "pair": pair,
                "n_rows": n_rows,
                "n_reject_event_rows": n_reject_rows,
                "n_tensor_fail": n_tensor_fail,
                "tensor_fail_rate": float(n_tensor_fail / max(n_rows, 1)),
                "scalar_overlap_rate": float(n_scalar_overlap / max(n_rows, 1)),
                "mean_reject_tensor_mismatch_share": float(np.mean(reject_share_vals)) if reject_share_vals else float("nan"),
                "median_tensor_mismatch": _median([_safe_float(r.get("tensor_mismatch")) for r in rows_pair]),
                "median_scalar_mismatch": _median([_safe_float(r.get("scalar_mismatch")) for r in rows_pair]),
                "median_abs_corr": _median([_safe_float(r.get("abs_corr")) for r in rows_pair]),
                "median_abs_lag_over_tolerance": _median([_safe_float(r.get("abs_lag_over_tolerance")) for r in rows_pair]),
                "median_obs_ratio_iqr_rel": _median([_safe_float(r.get("obs_ratio_iqr_rel")) for r in rows_pair]),
            }
        )

    summary_rows.sort(
        key=lambda r: (
            -_safe_float(r.get("mean_reject_tensor_mismatch_share")),
            -_safe_float(r.get("tensor_fail_rate")),
            str(r.get("pair", "")),
        )
    )
    for idx, row in enumerate(summary_rows, start=1):
        row["bottleneck_rank"] = int(idx)
    return summary_rows


def _write_table_csv(path: Path, rows: List[Dict[str, Any]], headers: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for row in rows:
            values: List[Any] = []
            for h in headers:
                value = row.get(h, "")
                values.append(_fmt(value) if isinstance(value, float) else value)
            w.writerow(values)


def _plot_pair_factor_summary(rows: List[Dict[str, Any]], out_png: Path) -> None:
    if not rows:
        fig, ax = plt.subplots(figsize=(10.0, 4.2))
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "No usable pair-factor rows",
            ha="center",
            va="center",
            fontsize=14,
            color="#444444",
        )
        fig.tight_layout()
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return
    _set_japanese_font()
    labels = [str(r.get("pair", "")) for r in rows]
    x = np.arange(len(rows), dtype=float)
    tensor_fail = np.asarray([_safe_float(r.get("tensor_fail_rate")) for r in rows], dtype=float)
    reject_share = np.asarray([_safe_float(r.get("mean_reject_tensor_mismatch_share")) for r in rows], dtype=float)
    tensor_mis = np.asarray([_safe_float(r.get("median_tensor_mismatch")) for r in rows], dtype=float)
    lag_ratio = np.asarray([_safe_float(r.get("median_abs_lag_over_tolerance")) for r in rows], dtype=float)

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(11.8, 7.6), sharex=True)
    width = 0.38
    ax0.bar(x - width / 2.0, tensor_fail, width=width, color="#d62728", alpha=0.85, label="tensor fail rate")
    ax0.bar(x + width / 2.0, reject_share, width=width, color="#9467bd", alpha=0.85, label="reject mismatch share")
    ax0.set_ylim(0.0, 1.05)
    ax0.set_ylabel("fraction")
    ax0.grid(True, axis="y", alpha=0.25)
    ax0.legend(loc="upper right", fontsize=9)
    ax0.set_title("Network reject-factor decomposition by detector pair")

    ax1.bar(x - width / 2.0, tensor_mis, width=width, color="#1f77b4", alpha=0.9, label="median tensor mismatch")
    ax1.bar(x + width / 2.0, lag_ratio, width=width, color="#ff7f0e", alpha=0.9, label="median |lag| / tolerance")
    ax1.set_ylabel("normalized score")
    ax1.set_xlabel("detector pair")
    ax1.grid(True, axis="y", alpha=0.25)
    ax1.legend(loc="upper right", fontsize=9)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=0, fontsize=10)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Step 8.7.19.3: H1/L1/V1 polarization network audit.")
    ap.add_argument("--events", type=str, default="GW200115_042309,GW200129_065458,GW200311_115853")
    ap.add_argument("--detectors", type=str, default="H1,L1,V1")
    ap.add_argument("--corr-use-min", type=float, default=0.05)
    ap.add_argument("--sky-samples", type=int, default=5000)
    ap.add_argument("--psi-samples", type=int, default=36)
    ap.add_argument("--cosi-samples", type=int, default=41)
    ap.add_argument("--response-floor-frac", type=float, default=0.1)
    ap.add_argument("--min-ring-directions", type=int, default=8)
    ap.add_argument("--geometry-relax-factor", type=float, default=2.0)
    ap.add_argument("--geometry-delay-floor-ms", type=float, default=0.25)
    ap.add_argument(
        "--allow-pair-pruning",
        action="store_true",
        help="When all usable pairs give inconsistent sky rings, try the largest pair subset that restores ring directions.",
    )
    ap.add_argument("--outdir", type=str, default=str(_ROOT / "output" / "private" / "gw"))
    ap.add_argument("--public-outdir", type=str, default=str(_ROOT / "output" / "public" / "gw"))
    ap.add_argument("--prefix", type=str, default="gw_polarization_h1_l1_v1_network_audit")
    args = ap.parse_args(list(argv) if argv is not None else None)

    events = [s.strip() for s in str(args.events).split(",") if s.strip()]
    if not events:
        print("[err] --events is empty")
        return 2
    detectors = sorted(
        [d.strip().upper() for d in str(args.detectors).split(",") if d.strip().upper() in _DETECTOR_SITES],
        key=_detector_order,
    )
    if len(detectors) < 3:
        print("[err] need at least 3 detectors (H1,L1,V1)")
        return 2

    pair_list = [_canonical_pair(a, b) for a, b in combinations(detectors, 2)]
    geom_all = _build_network_geometry()
    geom = {d: geom_all[d] for d in detectors}
    sky_dirs = _fibonacci_sphere(int(args.sky_samples))
    psi_grid = np.linspace(0.0, math.pi, int(max(8, args.psi_samples)), endpoint=False, dtype=np.float64)
    cosi_grid = np.linspace(-1.0, 1.0, int(max(9, args.cosi_samples)), dtype=np.float64)
    min_ring_directions = int(max(4, int(args.min_ring_directions)))
    geometry_relax_factor = float(max(1.0, float(args.geometry_relax_factor)))
    geometry_delay_floor_s = float(max(0.0, float(args.geometry_delay_floor_ms))) * 1.0e-3

    rows: List[Dict[str, Any]] = []
    for event in events:
        slug = _slugify(event)
        constraints: List[Dict[str, Any]] = []
        missing: List[str] = []
        low_corr: List[str] = []
        bad_fields: List[str] = []

        for first, second in pair_list:
            pair = f"{first}-{second}"
            payload = _load_metrics(event, slug, first, second)
            if payload is None:
                missing.append(pair)
                continue
            metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
            ratio = metrics.get("ratio") if isinstance(metrics.get("ratio"), dict) else {}
            inputs = payload.get("inputs") if isinstance(payload.get("inputs"), dict) else {}
            best_lag_ms = _safe_float(metrics.get("best_lag_ms_apply_to_first"))
            if not math.isfinite(best_lag_ms):
                best_lag_ms = _safe_float(metrics.get("best_lag_ms_apply_to_h1"))
            abs_corr = _safe_float(metrics.get("abs_best_corr"))
            obs_p16 = _safe_float(ratio.get("p16"))
            obs_med = _safe_float(ratio.get("median"))
            obs_p84 = _safe_float(ratio.get("p84"))
            fs_hz = _safe_float(inputs.get("analysis_fs_hz"))
            lag_scan = _load_lag_scan(slug, first, second)
            delay_tol_s = _estimate_delay_tolerance_s(
                best_lag_ms=best_lag_ms, best_abs_corr=abs_corr, fs_hz=fs_hz, lag_scan=lag_scan
            )
            delay_s = -best_lag_ms * 1e-3 if math.isfinite(best_lag_ms) else float("nan")
            if not (math.isfinite(abs_corr) and abs_corr >= float(args.corr_use_min)):
                low_corr.append(pair)
                continue
            if not (math.isfinite(delay_s) and math.isfinite(obs_p16) and math.isfinite(obs_med) and math.isfinite(obs_p84)):
                bad_fields.append(pair)
                continue
            constraints.append(
                {
                    "pair": pair,
                    "first": first,
                    "second": second,
                    "delay_s": delay_s,
                    "delay_tol_s": delay_tol_s,
                    "abs_corr": abs_corr,
                    "obs_p16": obs_p16,
                    "obs_med": obs_med,
                    "obs_p84": obs_p84,
                }
            )

        if len(constraints) < 2:
            reason = []
            if missing:
                reason.append(f"missing={','.join(missing)}")
            if low_corr:
                reason.append(f"low_corr={','.join(low_corr)}")
            if bad_fields:
                reason.append(f"bad_fields={','.join(bad_fields)}")
            rows.append(
                {
                    "event": event,
                    "quality": "missing" if not constraints else "low_corr",
                    "available_pair_count": int(len(constraints)),
                    "usable_pair_count": int(len(constraints)),
                    "pair_names": ";".join(c["pair"] for c in constraints),
                    "pair_pruning_applied": 0,
                    "dropped_pairs": "",
                    "geometry_mode": "none",
                    "pair_tensor_overlap_count": 0,
                    "pair_scalar_overlap_count": 0,
                    "status": "inconclusive_insufficient_pairs",
                    "status_reason": " | ".join(reason) if reason else "insufficient_pair_constraints",
                    "scalar_overlap_fraction": float("nan"),
                    "tensor_mismatch_max": float("nan"),
                    "scalar_mismatch_max": float("nan"),
                    "ring_directions_used": 0,
                }
            )
            continue

        effective_constraints = list(constraints)
        pair_pruning_applied = 0
        dropped_pairs: List[str] = []
        ring_dirs, geometry_mode = _ring_dirs_from_constraints(
            constraints=effective_constraints,
            geom=geom,
            sky_dirs=sky_dirs,
            min_ring_directions=min_ring_directions,
            geometry_relax_factor=geometry_relax_factor,
            geometry_delay_floor_s=geometry_delay_floor_s,
        )
        if ring_dirs.shape[0] < min_ring_directions and bool(args.allow_pair_pruning):
            pruned = _select_pruned_constraints(
                constraints=constraints,
                geom=geom,
                sky_dirs=sky_dirs,
                min_ring_directions=min_ring_directions,
                geometry_relax_factor=geometry_relax_factor,
                geometry_delay_floor_s=geometry_delay_floor_s,
            )
            if pruned is not None:
                effective_constraints = list(pruned["constraints"])
                pair_pruning_applied = 1
                dropped_pairs = list(pruned.get("dropped_pairs") or [])
                ring_dirs = pruned["ring_dirs"]
                geometry_mode = str(pruned.get("geometry_mode") or geometry_mode)

        if ring_dirs.shape[0] < min_ring_directions:
            rows.append(
                {
                    "event": event,
                    "quality": "usable",
                    "available_pair_count": int(len(constraints)),
                    "usable_pair_count": int(len(effective_constraints)),
                    "pair_names": ";".join(c["pair"] for c in effective_constraints),
                    "pair_pruning_applied": int(pair_pruning_applied),
                    "dropped_pairs": ";".join(dropped_pairs),
                    "geometry_mode": geometry_mode,
                    "pair_tensor_overlap_count": 0,
                    "pair_scalar_overlap_count": 0,
                    "status": "inconclusive_geometry",
                    "status_reason": "insufficient_ring_directions",
                    "scalar_overlap_fraction": float("nan"),
                    "tensor_mismatch_max": float("nan"),
                    "scalar_mismatch_max": float("nan"),
                    "ring_directions_used": int(ring_dirs.shape[0]),
                }
            )
            continue

        pair_results: List[Dict[str, Any]] = []
        for c in effective_constraints:
            tensor_first = geom[c["first"]]["tensor"]
            tensor_second = geom[c["second"]]["tensor"]
            tensor_all: List[float] = []
            scalar_all: List[float] = []
            for n in ring_dirs:
                tensor_r, scalar_r = _response_grid_for_direction(
                    n=n,
                    tensor_h=tensor_first,
                    tensor_l=tensor_second,
                    psi_grid=psi_grid,
                    cosi_grid=cosi_grid,
                    response_floor_frac=float(args.response_floor_frac),
                )
                if tensor_r.size > 0:
                    tensor_all.extend(tensor_r.tolist())
                if scalar_r.size > 0:
                    scalar_all.extend(scalar_r.tolist())
            tensor_arr = np.asarray(tensor_all, dtype=np.float64)
            scalar_arr = np.asarray(scalar_all, dtype=np.float64)
            t_lo, t_hi = _range_clip(tensor_arr, 0.5, 99.5)
            s_lo, s_hi = _range_clip(scalar_arr, 0.5, 99.5)
            obs_lo = min(float(c["obs_p16"]), float(c["obs_p84"]))
            obs_hi = max(float(c["obs_p16"]), float(c["obs_p84"]))
            pair_results.append(
                {
                    "pair": c["pair"],
                    "tensor_overlap": _interval_overlap(obs_lo, obs_hi, t_lo, t_hi),
                    "scalar_overlap": _interval_overlap(obs_lo, obs_hi, s_lo, s_hi),
                    "tensor_mismatch": _min_rel_mismatch(tensor_arr, float(c["obs_med"])),
                    "scalar_mismatch": _min_rel_mismatch(scalar_arr, float(c["obs_med"])),
                    "abs_corr": float(c["abs_corr"]),
                    "delay_s": float(c["delay_s"]),
                    "delay_tol_s": float(c["delay_tol_s"]),
                    "obs_p16": float(c["obs_p16"]),
                    "obs_med": float(c["obs_med"]),
                    "obs_p84": float(c["obs_p84"]),
                }
            )

        t_cnt = int(sum(1 for pr in pair_results if bool(pr["tensor_overlap"])))
        s_cnt = int(sum(1 for pr in pair_results if bool(pr["scalar_overlap"])))
        t_reject = t_cnt < len(pair_results)
        s_disfavored = s_cnt == 0
        status = "reject_tensor_response" if t_reject else ("pass_scalar_only_disfavored" if s_disfavored else "watch_scalar_not_excluded")
        reason = (
            "at_least_one_pair_outside_tensor_range"
            if t_reject
            else ("all_pairs_outside_scalar_range" if s_disfavored else "tensor_and_scalar_overlap_in_some_pairs")
        )
        rows.append(
            {
                "event": event,
                "quality": "usable",
                "available_pair_count": int(len(constraints)),
                "usable_pair_count": int(len(pair_results)),
                "pair_names": ";".join(pr["pair"] for pr in pair_results),
                "pair_pruning_applied": int(pair_pruning_applied),
                "dropped_pairs": ";".join(dropped_pairs),
                "geometry_mode": geometry_mode,
                "pair_tensor_overlap_count": t_cnt,
                "pair_scalar_overlap_count": s_cnt,
                "status": status,
                "status_reason": reason,
                "scalar_overlap_fraction": float(s_cnt / max(len(pair_results), 1)),
                "tensor_mismatch_max": float(np.nanmax([_safe_float(pr["tensor_mismatch"]) for pr in pair_results])),
                "scalar_mismatch_max": float(np.nanmax([_safe_float(pr["scalar_mismatch"]) for pr in pair_results])),
                "ring_directions_used": int(ring_dirs.shape[0]),
                "pair_results": pair_results,
            }
        )

    usable = [r for r in rows if str(r.get("quality")) == "usable" and str(r.get("status", "")).startswith(("reject_", "watch_", "pass_"))]
    pair_pruned_events = [r for r in rows if int(_safe_float(r.get("pair_pruning_applied"))) == 1]
    if not usable:
        overall_status, overall_reason = "inconclusive", "no_usable_events"
    elif any(str(r.get("status")) == "reject_tensor_response" for r in usable):
        overall_status, overall_reason = "reject", "tensor_response_failed_for_some_events"
    elif all(str(r.get("status")) == "pass_scalar_only_disfavored" for r in usable):
        overall_status, overall_reason = "pass", "scalar_only_disfavored_in_all_usable_events"
    else:
        overall_status, overall_reason = "watch", "scalar_not_excluded_with_current_three_detector_constraints"
    scalar_proxy = float(max((_safe_float(r.get("scalar_overlap_fraction")) for r in usable), default=1.0 if usable else 0.0))

    outdir = Path(str(args.outdir))
    public_outdir = Path(str(args.public_outdir))
    outdir.mkdir(parents=True, exist_ok=True)
    public_outdir.mkdir(parents=True, exist_ok=True)
    out_json = outdir / f"{args.prefix}.json"
    out_csv = outdir / f"{args.prefix}.csv"
    out_png = outdir / f"{args.prefix}.png"
    out_reject_detail_csv = outdir / f"{args.prefix}_reject_factor_details.csv"
    out_reject_summary_csv = outdir / f"{args.prefix}_reject_factor_decomposition.csv"
    out_reject_summary_json = outdir / f"{args.prefix}_reject_factor_decomposition.json"
    out_reject_summary_png = outdir / f"{args.prefix}_reject_factor_decomposition.png"
    _write_csv(out_csv, rows)
    _plot(rows, out_png)
    reject_factor_detail_rows = _pair_factor_rows(rows)
    reject_factor_summary_rows = _pair_factor_summary(reject_factor_detail_rows)
    _write_table_csv(
        out_reject_detail_csv,
        reject_factor_detail_rows,
        [
            "event",
            "pair",
            "event_status",
            "event_is_tensor_reject",
            "tensor_overlap",
            "scalar_overlap",
            "tensor_mismatch",
            "scalar_mismatch",
            "abs_corr",
            "best_lag_ms_apply_to_first",
            "lag_tolerance_ms",
            "abs_lag_over_tolerance",
            "obs_ratio_p16",
            "obs_ratio_median",
            "obs_ratio_p84",
            "obs_ratio_iqr_rel",
        ],
    )
    _write_table_csv(
        out_reject_summary_csv,
        reject_factor_summary_rows,
        [
            "pair",
            "bottleneck_rank",
            "n_rows",
            "n_reject_event_rows",
            "n_tensor_fail",
            "tensor_fail_rate",
            "scalar_overlap_rate",
            "mean_reject_tensor_mismatch_share",
            "median_tensor_mismatch",
            "median_scalar_mismatch",
            "median_abs_corr",
            "median_abs_lag_over_tolerance",
            "median_obs_ratio_iqr_rel",
        ],
    )
    _plot_pair_factor_summary(reject_factor_summary_rows, out_reject_summary_png)

    focus_pairs = ["H1-V1", "L1-V1"]
    focus_summary_rows = [r for r in reject_factor_summary_rows if str(r.get("pair")) in focus_pairs]
    reject_factor_payload: Dict[str, Any] = {
        "generated_utc": _iso_utc_now(),
        "schema": "wavep.gw.polarization.network_reject_factor_decomposition.v1",
        "source_network_audit_json": str(out_json).replace("\\", "/"),
        "focus_pairs": focus_pairs,
        "focus_pair_rows": focus_summary_rows,
        "pair_summary_rows": reject_factor_summary_rows,
        "detail_rows_count": int(len(reject_factor_detail_rows)),
    }
    out_reject_summary_json.write_text(json.dumps(reject_factor_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    payload: Dict[str, Any] = {
        "generated_utc": _iso_utc_now(),
        "schema": "wavep.gw.polarization.network_audit.v1",
        "phase": 8,
        "step": "8.7.19.3",
        "inputs": {
            "events": events,
            "detectors": detectors,
            "detector_pairs": [f"{a}-{b}" for a, b in pair_list],
            "corr_use_min": float(args.corr_use_min),
            "sky_samples": int(args.sky_samples),
            "psi_samples": int(args.psi_samples),
            "cosi_samples": int(args.cosi_samples),
            "response_floor_frac": float(args.response_floor_frac),
            "min_ring_directions": int(min_ring_directions),
            "geometry_relax_factor": float(geometry_relax_factor),
            "geometry_delay_floor_ms": float(args.geometry_delay_floor_ms),
            "allow_pair_pruning": bool(args.allow_pair_pruning),
        },
        "summary": {
            "n_events_requested": int(len(events)),
            "n_rows": int(len(rows)),
            "n_usable_events": int(len(usable)),
            "n_pair_pruned_events": int(len(pair_pruned_events)),
            "overall_status": overall_status,
            "overall_reason": overall_reason,
            "scalar_only_mode_global_upper_bound_proxy": scalar_proxy,
            "reject_factor_focus_pairs": focus_pairs,
            "reject_factor_focus_summary": focus_summary_rows,
        },
        "rows": rows,
        "outputs": {
            "audit_json": str(out_json).replace("\\", "/"),
            "audit_csv": str(out_csv).replace("\\", "/"),
            "audit_png": str(out_png).replace("\\", "/"),
            "reject_factor_detail_csv": str(out_reject_detail_csv).replace("\\", "/"),
            "reject_factor_summary_csv": str(out_reject_summary_csv).replace("\\", "/"),
            "reject_factor_summary_json": str(out_reject_summary_json).replace("\\", "/"),
            "reject_factor_summary_png": str(out_reject_summary_png).replace("\\", "/"),
        },
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    copied: List[str] = []
    for src in [
        out_json,
        out_csv,
        out_png,
        out_reject_detail_csv,
        out_reject_summary_csv,
        out_reject_summary_json,
        out_reject_summary_png,
    ]:
        dst = public_outdir / src.name
        shutil.copy2(src, dst)
        copied.append(str(dst).replace("\\", "/"))
    payload["outputs"]["public_copies"] = copied
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copy2(out_json, public_outdir / out_json.name)

    try:
        worklog.append_event(
            {
                "event_type": "gw_polarization_h1_l1_v1_network_audit",
                "argv": list(sys.argv),
                "outputs": {"audit_json": out_json, "audit_csv": out_csv, "audit_png": out_png},
                "metrics": {
                    "overall_status": overall_status,
                    "n_usable_events": int(len(usable)),
                    "scalar_only_mode_global_upper_bound_proxy": scalar_proxy,
                },
            }
        )
    except Exception:
        pass

    print(f"[ok] json: {out_json}")
    print(f"[ok] csv : {out_csv}")
    print(f"[ok] png : {out_png}")
    print(f"[ok] public copies: {len(copied)} files -> {public_outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
