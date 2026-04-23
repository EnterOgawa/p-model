#!/usr/bin/env python3
"""Cassini ODF sign convention record-level re-audit for Step 8.7.44.

Purpose:
- Decompose sign candidates (same/invert/offset-removed) into record-level groups
  defined by station, band, and arc.
- Re-evaluate convergence under fixed window/quality rules.
- Output machine-readable artifacts for Part II/IV synchronization.

Inputs:
- output/cassini/cassini_odf_beta_direct_fit_points.csv
- output/cassini/cassini_sce1_odf_observed_raw.csv
- output/cassini/cassini_beta_direct_fit_cross_source_metrics.json

Outputs:
- output/cassini/cassini_odf_sign_recordlevel_reaudit.json
- output/cassini/cassini_odf_sign_recordlevel_groups.csv
- output/cassini/cassini_odf_sign_recordlevel_window_quality.csv
- synced copies in output/public/cassini/
"""

from __future__ import annotations

import csv
import json
import math
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Dict, List, Optional, Sequence, Tuple


# クラス: `FitResult` の責務と境界条件を定義する。
@dataclass(frozen=True)
class FitResult:
    candidate: str
    fit_intercept: bool
    delta_beta: float
    delta_beta_sigma_proxy: float
    intercept: float
    corr_obs_fit: float
    rmse: float
    weighted_rms: float
    n_points: int
    zscore_vs_tdf_plasma: Optional[float]
    zscore_vs_tdf_raw: Optional[float]
    zscore_min_vs_tdf: Optional[float]


# 関数: `_repo_root` の入出力契約と処理意図を定義する。

def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


# 関数: `_safe_float` の入出力契約と処理意図を定義する。

def _safe_float(value: object) -> Optional[float]:
    try:
        parsed = float(value)
    except Exception:
        return None

    if not math.isfinite(parsed):
        return None

    return parsed


# 関数: `_safe_int` の入出力契約と処理意図を定義する。

def _safe_int(value: object, default: int = -1) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


# 関数: `_read_json` の入出力契約と処理意図を定義する。

def _read_json(path: Path) -> Dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


# 関数: `_arc_from_source_file` の入出力契約と処理意図を定義する。

def _arc_from_source_file(source_file: str) -> str:
    match = re.search(r"(sce1_\d+)", source_file, flags=re.IGNORECASE)
    if match:
        return match.group(1).lower()

    return "unknown_arc"


# 関数: `_load_points_with_metadata` の入出力契約と処理意図を定義する。

def _load_points_with_metadata(points_csv: Path, observed_csv: Path) -> List[Dict[str, object]]:
    if not points_csv.exists():
        raise FileNotFoundError(f"Missing points CSV: {points_csv}")

    if not observed_csv.exists():
        raise FileNotFoundError(f"Missing observed CSV: {observed_csv}")

    rows: List[Dict[str, object]] = []
    with points_csv.open("r", encoding="utf-8", newline="") as fp, observed_csv.open("r", encoding="utf-8", newline="") as fo:
        rp = csv.DictReader(fp)
        ro = csv.DictReader(fo)
        idx = 0
        for p, o in zip(rp, ro):
            idx += 1
            y_obs = _safe_float(p.get("y_obs"))
            x_unit = _safe_float(p.get("y_unit_beta1"))
            weight = _safe_float(p.get("weight"))
            t_days = _safe_float(p.get("t_days"))
            y_obs_meta = _safe_float(o.get("y_obs"))
            t_days_meta = _safe_float(o.get("t_days"))
            if y_obs is None or x_unit is None or t_days is None:
                continue

            if y_obs_meta is None or t_days_meta is None:
                continue

            if abs(y_obs - y_obs_meta) > 1.0e-18 or abs(t_days - t_days_meta) > 1.0e-12:
                raise RuntimeError(f"points/observed alignment mismatch at row={idx}")

            if weight is None or weight <= 0.0:
                weight = 1.0

            source_file = str(o.get("source_file") or "")
            row = {
                "t_days": float(t_days),
                "y_obs": float(y_obs),
                "x_unit": float(x_unit),
                "weight": float(weight),
                "station_rx": _safe_int(o.get("station_rx"), default=-1),
                "station_tx": _safe_int(o.get("station_tx"), default=-1),
                "downlink_band_id": _safe_int(o.get("downlink_band_id"), default=-1),
                "uplink_band_id": _safe_int(o.get("uplink_band_id"), default=-1),
                "source_file": source_file,
                "arc_id": _arc_from_source_file(source_file),
            }
            rows.append(row)

    if not rows:
        raise RuntimeError("No aligned rows loaded from points/observed CSV")

    return rows


# 関数: `_weighted_fit` の入出力契約と処理意図を定義する。

def _weighted_fit(
    y_obs: Sequence[float],
    x_unit: Sequence[float],
    weights: Sequence[float],
    *,
    fit_intercept: bool,
) -> Tuple[float, float, float, float, float, float]:
    n = len(y_obs)
    if n <= 1 or n != len(x_unit) or n != len(weights):
        raise ValueError("Invalid input lengths for weighted fit")

    if fit_intercept:
        s00 = sum(weights)
        s01 = sum(w * x for w, x in zip(weights, x_unit))
        s11 = sum(w * x * x for w, x in zip(weights, x_unit))
        b0 = sum(w * y for w, y in zip(weights, y_obs))
        b1 = sum(w * x * y for w, x, y in zip(weights, x_unit, y_obs))
        det = s00 * s11 - s01 * s01
        if abs(det) <= 0.0:
            raise RuntimeError("Singular normal matrix in weighted fit")

        intercept = (b0 * s11 - b1 * s01) / det
        delta = (s00 * b1 - s01 * b0) / det
        inv11 = s00 / det
        dof = max(1, n - 2)
    else:
        s11 = sum(w * x * x for w, x in zip(weights, x_unit))
        b1 = sum(w * x * y for w, x, y in zip(weights, x_unit, y_obs))
        if abs(s11) <= 0.0:
            raise RuntimeError("Singular denominator in zero-intercept fit")

        intercept = 0.0
        delta = b1 / s11
        inv11 = 1.0 / s11
        dof = max(1, n - 1)

    y_fit = [intercept + delta * x for x in x_unit]
    residuals = [y - yf for y, yf in zip(y_obs, y_fit)]
    wrss = sum(w * r * r for w, r in zip(weights, residuals))
    sigma2 = wrss / float(dof)
    delta_sigma = math.sqrt(max(0.0, sigma2 * inv11))
    rmse = math.sqrt(sum(r * r for r in residuals) / float(n))
    weighted_rms = math.sqrt(wrss / sum(weights))

    y_mean = sum(y_obs) / float(n)
    f_mean = sum(y_fit) / float(n)
    num = sum((y - y_mean) * (f - f_mean) for y, f in zip(y_obs, y_fit))
    den_y = sum((y - y_mean) ** 2 for y in y_obs)
    den_f = sum((f - f_mean) ** 2 for f in y_fit)
    corr = num / math.sqrt(den_y * den_f) if den_y > 0.0 and den_f > 0.0 else math.nan
    return float(delta), float(delta_sigma), float(intercept), float(corr), float(rmse), float(weighted_rms)


# 関数: `_zscore` の入出力契約と処理意図を定義する。

def _zscore(delta_a: float, sigma_a: float, delta_b: float, sigma_b: float) -> Optional[float]:
    if not (math.isfinite(delta_a) and math.isfinite(delta_b) and sigma_a > 0.0 and sigma_b > 0.0):
        return None

    pooled = math.sqrt(sigma_a * sigma_a + sigma_b * sigma_b)
    if pooled <= 0.0:
        return None

    return abs(delta_a - delta_b) / pooled


# 関数: `_rank_key` の入出力契約と処理意図を定義する。

def _rank_key(result: FitResult) -> Tuple[float, float]:
    z_min = result.zscore_min_vs_tdf if result.zscore_min_vs_tdf is not None and math.isfinite(result.zscore_min_vs_tdf) else float("inf")
    corr = result.corr_obs_fit if math.isfinite(result.corr_obs_fit) else -float("inf")
    return (z_min, -corr)


# 関数: `_transform_candidate` の入出力契約と処理意図を定義する。

def _transform_candidate(y_values: Sequence[float], candidate: str) -> Tuple[List[float], bool]:
    med = float(median(y_values))
    if candidate == "same_sign":
        return list(y_values), True

    if candidate == "invert_sign":
        return ([-y for y in y_values], True)

    if candidate == "same_sign_offset_removed":
        return ([y - med for y in y_values], False)

    if candidate == "invert_sign_offset_removed":
        return ([-(y - med) for y in y_values], False)

    raise ValueError(f"Unsupported candidate: {candidate}")


# 関数: `_fit_candidates` の入出力契約と処理意図を定義する。

def _fit_candidates(
    rows: Sequence[Dict[str, object]],
    ref_plasma: Tuple[float, float],
    ref_tdf_raw: Tuple[float, float],
) -> List[FitResult]:
    y_base = [float(r["y_obs"]) for r in rows]
    x_base = [float(r["x_unit"]) for r in rows]
    w_base = [float(r["weight"]) for r in rows]
    candidates = [
        "same_sign",
        "invert_sign",
        "same_sign_offset_removed",
        "invert_sign_offset_removed",
    ]
    out: List[FitResult] = []
    for name in candidates:
        y_cand, fit_intercept = _transform_candidate(y_base, name)
        delta, sig, intercept, corr, rmse, wrms = _weighted_fit(y_cand, x_base, w_base, fit_intercept=fit_intercept)
        z_plasma = _zscore(delta, sig, ref_plasma[0], ref_plasma[1])
        z_raw = _zscore(delta, sig, ref_tdf_raw[0], ref_tdf_raw[1])
        z_values = [z for z in [z_plasma, z_raw] if z is not None and math.isfinite(z)]
        z_min = min(z_values) if z_values else None
        out.append(
            FitResult(
                candidate=name,
                fit_intercept=fit_intercept,
                delta_beta=float(delta),
                delta_beta_sigma_proxy=float(sig),
                intercept=float(intercept),
                corr_obs_fit=float(corr),
                rmse=float(rmse),
                weighted_rms=float(wrms),
                n_points=len(rows),
                zscore_vs_tdf_plasma=z_plasma,
                zscore_vs_tdf_raw=z_raw,
                zscore_min_vs_tdf=z_min,
            )
        )

    return out


# 関数: `_group_key` の入出力契約と処理意図を定義する。

def _group_key(row: Dict[str, object]) -> Tuple[int, int, int, str]:
    return (
        int(row["station_rx"]),
        int(row["downlink_band_id"]),
        int(row["uplink_band_id"]),
        str(row["arc_id"]),
    )


# 関数: `_arc_number` の入出力契約と処理意図を定義する。

def _arc_number(arc_id: str) -> Optional[int]:
    text = str(arc_id or "").strip()
    if not text:
        return None

    m = re.search(r"(\d+)$", text)
    if m is None:
        return None

    try:
        return int(m.group(1))
    except Exception:
        return None


# 関数: `_build_arc_stability_intervals` の入出力契約と処理意図を定義する。

def _build_arc_stability_intervals(
    group_rows: Sequence[Dict[str, object]],
    *,
    total_points: int,
) -> List[Dict[str, object]]:
    pass_rows: List[Dict[str, object]] = []
    for row in group_rows:
        if not bool(row.get("group_gate_pass")):
            continue

        arc_id = str(row.get("arc_id") or "")
        arc_num = _arc_number(arc_id)
        if arc_num is None:
            continue

        pass_rows.append(
            {
                "arc_id": arc_id,
                "arc_num": int(arc_num),
                "best_candidate": str(row.get("best_candidate") or ""),
                "n_points": int(row.get("n_points") or 0),
            }
        )

    pass_rows_sorted = sorted(pass_rows, key=lambda r: (int(r["arc_num"]), str(r["arc_id"])))
    intervals: List[Dict[str, object]] = []
    current: Optional[Dict[str, object]] = None
    idx = 0
    while idx < len(pass_rows_sorted):
        row = pass_rows_sorted[idx]
        if current is None:
            current = {
                "best_candidate": str(row["best_candidate"]),
                "arc_numbers": [int(row["arc_num"])],
                "arc_ids": [str(row["arc_id"])],
                "n_points": int(row["n_points"]),
            }
            idx += 1
            continue

        prev_arc_num = int(current["arc_numbers"][-1])
        same_candidate = str(row["best_candidate"]) == str(current["best_candidate"])
        contiguous = int(row["arc_num"]) == prev_arc_num + 1
        if contiguous and same_candidate:
            current["arc_numbers"].append(int(row["arc_num"]))
            current["arc_ids"].append(str(row["arc_id"]))
            current["n_points"] = int(current["n_points"]) + int(row["n_points"])
            idx += 1
            continue

        intervals.append(current)
        current = None

    if current is not None:
        intervals.append(current)

    out: List[Dict[str, object]] = []
    for one in intervals:
        n_points = int(one["n_points"])
        cov_points = (float(n_points) / float(total_points)) if total_points > 0 else 0.0
        out.append(
            {
                "best_candidate": str(one["best_candidate"]),
                "arc_start": int(one["arc_numbers"][0]),
                "arc_end": int(one["arc_numbers"][-1]),
                "n_arcs": int(len(one["arc_numbers"])),
                "arc_ids": [str(v) for v in one["arc_ids"]],
                "n_points": int(n_points),
                "coverage_ratio_points": float(cov_points),
            }
        )

    out_sorted = sorted(out, key=lambda r: (-float(r["coverage_ratio_points"]), -int(r["n_arcs"]), int(r["arc_start"])))
    return out_sorted


# 関数: `_evaluate_arc_stability_terminal_gate` の入出力契約と処理意図を定義する。

def _evaluate_arc_stability_terminal_gate(
    group_rows: Sequence[Dict[str, object]],
    *,
    total_points: int,
) -> Dict[str, object]:
    cfg = {
        "min_contiguous_arcs": 2,
        "min_coverage_ratio_points": 0.25,
    }
    intervals = _build_arc_stability_intervals(group_rows, total_points=total_points)
    selected = intervals[0] if intervals else None
    selected_n_arcs = int(selected["n_arcs"]) if isinstance(selected, dict) else 0
    selected_cov_points = float(selected["coverage_ratio_points"]) if isinstance(selected, dict) else 0.0
    gate_pass = bool(
        selected is not None
        and selected_n_arcs >= int(cfg["min_contiguous_arcs"])
        and selected_cov_points >= float(cfg["min_coverage_ratio_points"])
    )
    reasons: List[str] = []
    if selected is None:
        reasons.append("stable_arc_interval_not_found")
    else:
        if selected_n_arcs < int(cfg["min_contiguous_arcs"]):
            reasons.append("stable_arc_count_below_gate")

        if selected_cov_points < float(cfg["min_coverage_ratio_points"]):
            reasons.append("stable_arc_coverage_below_gate")

    return {
        "config": cfg,
        "stable_intervals": intervals,
        "selected_interval": selected,
        "gate_pass": bool(gate_pass),
        "status": "pass" if gate_pass else "watch",
        "reasons": reasons,
    }


# 関数: `_quantile` の入出力契約と処理意図を定義する。

def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return math.nan

    ordered = sorted(values)
    if q <= 0.0:
        return float(ordered[0])

    if q >= 1.0:
        return float(ordered[-1])

    idx = q * (len(ordered) - 1)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return float(ordered[lo])

    frac = idx - lo
    return float((1.0 - frac) * ordered[lo] + frac * ordered[hi])


# 関数: `_write_group_csv` の入出力契約と処理意図を定義する。

def _write_group_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "station_rx",
                "downlink_band_id",
                "uplink_band_id",
                "arc_id",
                "n_points",
                "best_candidate",
                "best_delta_beta",
                "best_delta_beta_sigma_proxy",
                "best_corr_obs_fit",
                "best_zscore_min_vs_tdf",
                "group_gate_pass",
            ]
        )
        for r in rows:
            writer.writerow(
                [
                    r["station_rx"],
                    r["downlink_band_id"],
                    r["uplink_band_id"],
                    r["arc_id"],
                    r["n_points"],
                    r["best_candidate"],
                    r["best_delta_beta"],
                    r["best_delta_beta_sigma_proxy"],
                    r["best_corr_obs_fit"],
                    r["best_zscore_min_vs_tdf"],
                    "yes" if bool(r["group_gate_pass"]) else "no",
                ]
            )


# 関数: `_write_window_quality_csv` の入出力契約と処理意図を定義する。

def _write_window_quality_csv(path: Path, rows: Sequence[Dict[str, object]], best_idx: int) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "window_days",
                "outlier_trim_quantile",
                "n_points",
                "best_candidate",
                "best_delta_beta",
                "best_delta_beta_sigma_proxy",
                "best_corr_obs_fit",
                "best_zscore_min_vs_tdf",
                "gate_pass",
                "is_best_scenario",
            ]
        )
        for idx, r in enumerate(rows):
            writer.writerow(
                [
                    r["window_days"],
                    r["outlier_trim_quantile"],
                    r["n_points"],
                    r["best_candidate"],
                    r["best_delta_beta"],
                    r["best_delta_beta_sigma_proxy"],
                    r["best_corr_obs_fit"],
                    r["best_zscore_min_vs_tdf"],
                    "yes" if bool(r["gate_pass"]) else "no",
                    "yes" if idx == best_idx else "no",
                ]
            )


# 関数: `_sync_public` の入出力契約と処理意図を定義する。

def _sync_public(root: Path, names: Sequence[str]) -> None:
    src_dir = root / "output" / "cassini"
    dst_dir = root / "output" / "public" / "cassini"
    dst_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        src = src_dir / name
        if src.exists():
            shutil.copy2(src, dst_dir / name)


# 関数: `main` の入出力契約と処理意図を定義する。

def main() -> None:
    root = _repo_root()
    out_dir = root / "output" / "cassini"
    out_dir.mkdir(parents=True, exist_ok=True)

    points_csv = out_dir / "cassini_odf_beta_direct_fit_points.csv"
    observed_csv = out_dir / "cassini_sce1_odf_observed_raw.csv"
    cross_metrics_json = out_dir / "cassini_beta_direct_fit_cross_source_metrics.json"
    cross = _read_json(cross_metrics_json)
    rows_cross = cross.get("rows") if isinstance(cross.get("rows"), list) else []
    by_scenario = {str(r.get("scenario") or ""): r for r in rows_cross if isinstance(r, dict)}
    tdf_plasma = by_scenario.get("tdf_plasma", {})
    tdf_raw = by_scenario.get("tdf_raw", {})
    d_plasma = _safe_float(tdf_plasma.get("delta_beta")) or math.nan
    s_plasma = _safe_float(tdf_plasma.get("delta_beta_sigma_proxy")) or math.nan
    d_raw = _safe_float(tdf_raw.get("delta_beta")) or math.nan
    s_raw = _safe_float(tdf_raw.get("delta_beta_sigma_proxy")) or math.nan
    ref_plasma = (float(d_plasma), float(s_plasma))
    ref_tdf_raw = (float(d_raw), float(s_raw))

    rows = _load_points_with_metadata(points_csv, observed_csv)
    global_candidates = _fit_candidates(rows, ref_plasma, ref_tdf_raw)
    global_best = min(global_candidates, key=_rank_key)
    global_gate_pass = bool(
        global_best.zscore_min_vs_tdf is not None
        and math.isfinite(global_best.zscore_min_vs_tdf)
        and global_best.zscore_min_vs_tdf <= 3.0
        and math.isfinite(global_best.corr_obs_fit)
        and global_best.corr_obs_fit >= 0.25
    )

    grouped: Dict[Tuple[int, int, int, str], List[Dict[str, object]]] = {}
    for row in rows:
        key = _group_key(row)
        grouped.setdefault(key, []).append(row)

    group_rows: List[Dict[str, object]] = []
    for key, chunk in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1], item[0][2], item[0][3])):
        if len(chunk) < 30:
            continue

        group_candidates = _fit_candidates(chunk, ref_plasma, ref_tdf_raw)
        best_group = min(group_candidates, key=_rank_key)
        group_gate_pass = bool(
            best_group.zscore_min_vs_tdf is not None
            and math.isfinite(best_group.zscore_min_vs_tdf)
            and best_group.zscore_min_vs_tdf <= 3.0
            and math.isfinite(best_group.corr_obs_fit)
            and best_group.corr_obs_fit >= 0.25
        )
        group_rows.append(
            {
                "station_rx": key[0],
                "downlink_band_id": key[1],
                "uplink_band_id": key[2],
                "arc_id": key[3],
                "n_points": len(chunk),
                "best_candidate": best_group.candidate,
                "best_delta_beta": best_group.delta_beta,
                "best_delta_beta_sigma_proxy": best_group.delta_beta_sigma_proxy,
                "best_corr_obs_fit": best_group.corr_obs_fit,
                "best_zscore_min_vs_tdf": best_group.zscore_min_vs_tdf,
                "group_gate_pass": bool(group_gate_pass),
            }
        )

    group_n = len(group_rows)
    group_pass_n = sum(1 for r in group_rows if bool(r["group_gate_pass"]))
    group_pass_ratio = (float(group_pass_n) / float(group_n)) if group_n > 0 else 0.0
    arc_stability_gate = _evaluate_arc_stability_terminal_gate(group_rows, total_points=len(rows))

    window_grid = [10.0, 8.0, 6.0, 4.0]
    trim_grid = [0.0, 0.005, 0.01]
    window_quality_rows: List[Dict[str, object]] = []
    for window_days in window_grid:
        window_rows = [r for r in rows if abs(float(r["t_days"])) <= window_days]
        if len(window_rows) < 30:
            continue

        y_window = [float(r["y_obs"]) for r in window_rows]
        x_window = [float(r["x_unit"]) for r in window_rows]
        w_window = [float(r["weight"]) for r in window_rows]
        d0, _, a0, _, _, _ = _weighted_fit(y_window, x_window, w_window, fit_intercept=True)
        residual_abs = [abs(y - (a0 + d0 * x)) for y, x in zip(y_window, x_window)]
        for trim_q in trim_grid:
            filtered_rows = window_rows
            threshold = None
            if trim_q > 0.0 and residual_abs:
                threshold = _quantile(residual_abs, 1.0 - trim_q)
                filtered_rows = [r for r, abs_res in zip(window_rows, residual_abs) if abs_res <= threshold]

            if len(filtered_rows) < 30:
                continue

            candidates = _fit_candidates(filtered_rows, ref_plasma, ref_tdf_raw)
            best = min(candidates, key=_rank_key)
            gate_pass = bool(
                best.zscore_min_vs_tdf is not None
                and math.isfinite(best.zscore_min_vs_tdf)
                and best.zscore_min_vs_tdf <= 3.0
                and math.isfinite(best.corr_obs_fit)
                and best.corr_obs_fit >= 0.25
            )
            window_quality_rows.append(
                {
                    "window_days": float(window_days),
                    "outlier_trim_quantile": float(trim_q),
                    "n_points": len(filtered_rows),
                    "residual_abs_threshold": threshold,
                    "best_candidate": best.candidate,
                    "best_delta_beta": best.delta_beta,
                    "best_delta_beta_sigma_proxy": best.delta_beta_sigma_proxy,
                    "best_corr_obs_fit": best.corr_obs_fit,
                    "best_zscore_min_vs_tdf": best.zscore_min_vs_tdf,
                    "gate_pass": bool(gate_pass),
                }
            )

    if not window_quality_rows:
        raise RuntimeError("Window/quality reassessment produced no valid scenarios")

    best_window_idx = min(
        range(len(window_quality_rows)),
        key=lambda i: (
            float(window_quality_rows[i]["best_zscore_min_vs_tdf"])
            if window_quality_rows[i]["best_zscore_min_vs_tdf"] is not None
            else float("inf"),
            -float(window_quality_rows[i]["best_corr_obs_fit"]),
            -int(window_quality_rows[i]["n_points"]),
        ),
    )
    best_window = window_quality_rows[best_window_idx]
    window_gate_pass = bool(best_window.get("gate_pass"))

    sign_closed = bool(global_gate_pass or window_gate_pass)
    reasons: List[str] = []
    if not sign_closed:
        reasons.append("sign_hypothesis_not_closed")

    if group_pass_ratio < 1.0:
        reasons.append("sign_not_closed_across_record_levels")

    if not window_gate_pass:
        reasons.append("window_quality_reassessment_no_improvement_to_gate")

    for reason in arc_stability_gate.get("reasons") if isinstance(arc_stability_gate.get("reasons"), list) else []:
        reason_text = str(reason).strip()
        if reason_text and reason_text not in reasons:
            reasons.append(reason_text)

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "points_csv": str(points_csv),
            "observed_csv": str(observed_csv),
            "cross_source_metrics_json": str(cross_metrics_json),
        },
        "reference_tdf": {
            "delta_beta_tdf_plasma": ref_plasma[0],
            "delta_beta_sigma_tdf_plasma": ref_plasma[1],
            "delta_beta_tdf_raw": ref_tdf_raw[0],
            "delta_beta_sigma_tdf_raw": ref_tdf_raw[1],
        },
        "global_baseline": {
            "candidate_fit_results": [
                {
                    "candidate": r.candidate,
                    "fit_intercept": bool(r.fit_intercept),
                    "delta_beta": r.delta_beta,
                    "delta_beta_sigma_proxy": r.delta_beta_sigma_proxy,
                    "corr_obs_fit": r.corr_obs_fit,
                    "zscore_min_vs_tdf": r.zscore_min_vs_tdf,
                }
                for r in global_candidates
            ],
            "best_candidate": {
                "candidate": global_best.candidate,
                "fit_intercept": bool(global_best.fit_intercept),
                "delta_beta": global_best.delta_beta,
                "delta_beta_sigma_proxy": global_best.delta_beta_sigma_proxy,
                "corr_obs_fit": global_best.corr_obs_fit,
                "zscore_min_vs_tdf": global_best.zscore_min_vs_tdf,
                "gate_pass": bool(global_gate_pass),
            },
        },
        "record_level_sign_audit": {
            "group_key_definition": ["station_rx", "downlink_band_id", "uplink_band_id", "arc_id"],
            "groups_n": int(group_n),
            "groups_pass_n": int(group_pass_n),
            "groups_pass_ratio": float(group_pass_ratio),
            "groups": group_rows,
            "arc_stability_terminal_gate": arc_stability_gate,
        },
        "window_quality_reassessment": {
            "window_days_grid": window_grid,
            "outlier_trim_quantile_grid": trim_grid,
            "scenarios": window_quality_rows,
            "best_scenario_index": int(best_window_idx),
            "best_scenario": best_window,
        },
        "recommended_status": "pass_candidate" if sign_closed and group_pass_ratio == 1.0 else "watch",
        "promotion_gates": {
            "sign_closed_global_baseline": bool(global_gate_pass),
            "sign_closed_best_window_quality": bool(window_gate_pass),
            "sign_closed_all_record_levels": bool(group_pass_ratio == 1.0),
        },
        "reasons": reasons,
    }

    out_json = out_dir / "cassini_odf_sign_recordlevel_reaudit.json"
    out_groups_csv = out_dir / "cassini_odf_sign_recordlevel_groups.csv"
    out_window_csv = out_dir / "cassini_odf_sign_recordlevel_window_quality.csv"
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_group_csv(out_groups_csv, group_rows)
    _write_window_quality_csv(out_window_csv, window_quality_rows, best_window_idx)
    _sync_public(
        root,
        [
            out_json.name,
            out_groups_csv.name,
            out_window_csv.name,
        ],
    )
    print("Wrote:", out_json)
    print("Wrote:", out_groups_csv)
    print("Wrote:", out_window_csv)
    print("Synced:", root / "output" / "public" / "cassini")


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    main()
