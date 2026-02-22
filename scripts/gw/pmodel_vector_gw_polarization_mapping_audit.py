from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
from datetime import datetime, timezone
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
    _build_network_geometry,
    _direction_basis,
    _fibonacci_sphere,
    _response_grid_for_direction,
    _safe_float,
)
from scripts.summary import worklog  # noqa: E402

_C = 299_792_458.0


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
        available = {font.name for font in fm.fontManager.ttflist}
        chosen = [name for name in preferred if name in available]
        if not chosen:
            return
        mpl.rcParams["font.family"] = chosen + ["DejaVu Sans"]
        mpl.rcParams["axes.unicode_minus"] = False
    except Exception:
        return


def _slugify(value: str) -> str:
    out = "".join(ch.lower() if ch.isalnum() else "_" for ch in (value or "").strip())
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_") or "event"


def _fmt(value: float, digits: int = 7) -> str:
    if not math.isfinite(float(value)):
        return ""
    x = float(value)
    if x == 0.0:
        return "0"
    abs_x = abs(x)
    if abs_x >= 1e4 or abs_x < 1e-3:
        return f"{x:.{digits}g}"
    return f"{x:.{digits}f}".rstrip("0").rstrip(".")


def _canonical_pair(detector_first: str, detector_second: str) -> Tuple[str, str]:
    first = str(detector_first).upper().strip()
    second = str(detector_second).upper().strip()
    if first <= second:
        return first, second
    return second, first


def _load_pair_metrics(event: str, detector_first: str, detector_second: str) -> Optional[Dict[str, Any]]:
    event_slug = _slugify(event)
    first_slug = _slugify(detector_first)
    second_slug = _slugify(detector_second)
    stem = f"{event_slug}_{first_slug}_{second_slug}_amplitude_ratio_metrics.json"
    legacy_stem = f"{event_slug}_h1_l1_amplitude_ratio_metrics.json"
    candidates = [
        _ROOT / "output" / "public" / "gw" / stem,
        _ROOT / "output" / "private" / "gw" / stem,
        _ROOT / "output" / "public" / "gw" / legacy_stem,
        _ROOT / "output" / "private" / "gw" / legacy_stem,
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["_metrics_path"] = str(path).replace("\\", "/")
            return payload
        except Exception:
            continue
    return None


def _load_lag_scan(event: str, detector_first: str, detector_second: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    event_slug = _slugify(event)
    first_slug = _slugify(detector_first)
    second_slug = _slugify(detector_second)
    stem = f"{event_slug}_{first_slug}_{second_slug}_amplitude_ratio_lag_scan.csv"
    legacy_stem = f"{event_slug}_h1_l1_amplitude_ratio_lag_scan.csv"
    candidates = [
        _ROOT / "output" / "public" / "gw" / stem,
        _ROOT / "output" / "private" / "gw" / stem,
        _ROOT / "output" / "public" / "gw" / legacy_stem,
        _ROOT / "output" / "private" / "gw" / legacy_stem,
    ]
    for path in candidates:
        if not path.exists():
            continue
        lags: List[float] = []
        corrs: List[float] = []
        try:
            with path.open("r", encoding="utf-8", newline="") as handle:
                next(handle, None)
                for line in handle:
                    cells = [c.strip() for c in line.split(",")]
                    if len(cells) < 2:
                        continue
                    lag_ms = _safe_float(cells[0])
                    corr = _safe_float(cells[1])
                    if math.isfinite(lag_ms) and math.isfinite(corr):
                        lags.append(float(lag_ms))
                        corrs.append(float(corr))
            if lags:
                return np.asarray(lags, dtype=np.float64), np.asarray(corrs, dtype=np.float64)
        except Exception:
            continue
    return None


def _estimate_delay_tolerance_ms(
    *,
    abs_corr: float,
    analysis_fs_hz: float,
    lag_scan: Optional[Tuple[np.ndarray, np.ndarray]],
) -> float:
    sample_floor_ms = 1000.0 / analysis_fs_hz if math.isfinite(analysis_fs_hz) and analysis_fs_hz > 0.0 else 0.25
    tol_floor_ms = max(0.25, 2.0 * sample_floor_ms)
    if lag_scan is None or not math.isfinite(abs_corr):
        return float(tol_floor_ms)
    lags_ms, corrs = lag_scan
    if lags_ms.size < 3:
        return float(tol_floor_ms)
    threshold = 0.95 * float(abs_corr)
    mask = np.abs(corrs) >= threshold
    if int(np.sum(mask)) < 2:
        return float(tol_floor_ms)
    span_ms = float(np.max(lags_ms[mask]) - np.min(lags_ms[mask]))
    return float(max(tol_floor_ms, 0.5 * span_ms))


def _range_clip(values: np.ndarray, q_lo: float = 0.5, q_hi: float = 99.5) -> Tuple[float, float]:
    if values.size == 0:
        return float("nan"), float("nan")
    return float(np.percentile(values, q_lo)), float(np.percentile(values, q_hi))


def _interval_overlap(a_lo: float, a_hi: float, b_lo: float, b_hi: float) -> bool:
    if not (math.isfinite(a_lo) and math.isfinite(a_hi) and math.isfinite(b_lo) and math.isfinite(b_hi)):
        return False
    return not (a_hi < b_lo or b_hi < a_lo)


def _min_rel_mismatch(values: np.ndarray, target: float) -> float:
    if values.size == 0 or not math.isfinite(target):
        return float("nan")
    denom = abs(float(target)) + 1e-15
    return float(np.min(np.abs(values - float(target)) / denom))


def _vector_ratio_grid_for_direction(
    *,
    direction: np.ndarray,
    tensor_first: np.ndarray,
    tensor_second: np.ndarray,
    psi_grid: np.ndarray,
    cosi_grid: np.ndarray,
    response_floor_frac: float,
) -> np.ndarray:
    e_theta, e_phi = _direction_basis(direction)
    cpsi = np.cos(psi_grid)
    spsi = np.sin(psi_grid)
    p = cpsi[:, None] * e_theta[None, :] + spsi[:, None] * e_phi[None, :]
    q = -spsi[:, None] * e_theta[None, :] + cpsi[:, None] * e_phi[None, :]

    e_vec_x = np.einsum("i,aj->aij", direction, p) + np.einsum("ai,j->aij", p, direction)
    e_vec_y = np.einsum("i,aj->aij", direction, q) + np.einsum("ai,j->aij", q, direction)

    fv_x_first = np.einsum("ij,aij->a", tensor_first, e_vec_x)
    fv_y_first = np.einsum("ij,aij->a", tensor_first, e_vec_y)
    fv_x_second = np.einsum("ij,aij->a", tensor_second, e_vec_x)
    fv_y_second = np.einsum("ij,aij->a", tensor_second, e_vec_y)

    floor_frac = float(max(0.0, min(0.5, response_floor_frac)))
    ratios: List[float] = []
    for cosi in cosi_grid:
        h_x = math.sqrt(max(0.0, 1.0 - float(cosi) * float(cosi)))
        h_y = 1.0
        amp_first = np.sqrt((fv_x_first * h_x) ** 2 + (fv_y_first * h_y) ** 2)
        amp_second = np.sqrt((fv_x_second * h_x) ** 2 + (fv_y_second * h_y) ** 2)
        floor_first = floor_frac * float(np.max(amp_first)) if amp_first.size else 0.0
        floor_second = floor_frac * float(np.max(amp_second)) if amp_second.size else 0.0
        usable = (amp_first > max(1e-10, floor_first)) & (amp_second > max(1e-10, floor_second))
        if np.any(usable):
            ratios.extend((amp_first[usable] / amp_second[usable]).tolist())
    return np.asarray(ratios, dtype=np.float64)


def _run_network_scalar_gate(
    *,
    enabled: bool,
    events_csv: str,
    detectors_csv: str,
    corr_use_min: float,
    sky_samples: int,
    psi_samples: int,
    cosi_samples: int,
    response_floor_frac: float,
    min_ring_directions: int,
    geometry_relax_factor: float,
    geometry_delay_floor_ms: float,
    allow_pair_pruning: bool,
    outdir: Path,
    public_outdir: Path,
    prefix: str,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "enabled": bool(enabled),
        "ran": False,
        "status": "disabled",
        "reason": "network_gate_disabled",
        "n_usable_events": 0,
        "scalar_overlap_proxy_max": float("nan"),
        "scalar_overlap_proxy_median": float("nan"),
        "scalar_reduction_pass": False,
        "scalar_exclusion_pass": False,
        "tensor_consistency_pass": False,
        "json_path": "",
        "reject_factor_focus_summary": [],
        "reject_factor_summary_json_path": "",
    }
    if not enabled:
        return result

    script = _ROOT / "scripts" / "gw" / "gw_polarization_h1_l1_v1_network_audit.py"
    if not script.exists():
        result["status"] = "inconclusive"
        result["reason"] = "network_audit_script_missing"
        return result

    cmd: List[str] = [
        sys.executable,
        "-B",
        str(script),
        "--events",
        str(events_csv),
        "--detectors",
        str(detectors_csv),
        "--corr-use-min",
        str(float(corr_use_min)),
        "--sky-samples",
        str(int(sky_samples)),
        "--psi-samples",
        str(int(psi_samples)),
        "--cosi-samples",
        str(int(cosi_samples)),
        "--response-floor-frac",
        str(float(response_floor_frac)),
        "--min-ring-directions",
        str(int(min_ring_directions)),
        "--geometry-relax-factor",
        str(float(geometry_relax_factor)),
        "--geometry-delay-floor-ms",
        str(float(geometry_delay_floor_ms)),
        "--outdir",
        str(outdir),
        "--public-outdir",
        str(public_outdir),
        "--prefix",
        str(prefix),
    ]
    if allow_pair_pruning:
        cmd.append("--allow-pair-pruning")

    proc = subprocess.run(cmd, cwd=str(_ROOT), capture_output=True, text=True)
    result["ran"] = True
    result["returncode"] = int(proc.returncode)
    if proc.returncode != 0:
        result["status"] = "inconclusive"
        result["reason"] = "network_audit_subprocess_failed"
        result["stderr_tail"] = (proc.stderr or "")[-2000:]
        result["stdout_tail"] = (proc.stdout or "")[-2000:]
        return result

    json_path = outdir / f"{prefix}.json"
    if not json_path.exists():
        result["status"] = "inconclusive"
        result["reason"] = "network_audit_json_missing"
        return result

    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        result["status"] = "inconclusive"
        result["reason"] = "network_audit_json_parse_failed"
        return result

    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    rows = payload.get("rows") if isinstance(payload.get("rows"), list) else []
    scalar_fracs: List[float] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        quality = str(row.get("quality", ""))
        status = str(row.get("status", ""))
        value = _safe_float(row.get("scalar_overlap_fraction"))
        if quality == "usable" and status.startswith(("reject_", "watch_", "pass_")) and math.isfinite(value):
            scalar_fracs.append(float(value))

    if scalar_fracs:
        scalar_arr = np.asarray(scalar_fracs, dtype=np.float64)
        scalar_max = float(np.max(scalar_arr))
        scalar_med = float(np.median(scalar_arr))
    else:
        scalar_max = float("nan")
        scalar_med = float("nan")

    network_status = str(summary.get("overall_status", "inconclusive"))
    network_reason = str(summary.get("overall_reason", "no_reason"))
    n_usable_events = int(_safe_float(summary.get("n_usable_events")))
    scalar_proxy = _safe_float(summary.get("scalar_only_mode_global_upper_bound_proxy"))
    if math.isfinite(scalar_proxy):
        scalar_max = float(scalar_proxy)
        if not math.isfinite(scalar_med):
            scalar_med = float(scalar_proxy)

    tensor_consistency_pass = bool(network_status == "pass")
    scalar_exclusion_pass = bool(tensor_consistency_pass and math.isfinite(scalar_max) and scalar_max <= 0.0)
    scalar_reduction_pass = bool(math.isfinite(scalar_med) and scalar_med < 1.0)

    result.update(
        {
            "status": network_status,
            "reason": network_reason,
            "n_usable_events": n_usable_events,
            "scalar_overlap_proxy_max": scalar_max,
            "scalar_overlap_proxy_median": scalar_med,
            "scalar_reduction_pass": scalar_reduction_pass,
            "scalar_exclusion_pass": scalar_exclusion_pass,
            "tensor_consistency_pass": tensor_consistency_pass,
            "json_path": str(json_path).replace("\\", "/"),
            "reject_factor_focus_summary": summary.get("reject_factor_focus_summary")
            if isinstance(summary.get("reject_factor_focus_summary"), list)
            else [],
            "reject_factor_summary_json_path": str(
                (
                    (payload.get("outputs") or {}).get("reject_factor_summary_json")
                    if isinstance(payload.get("outputs"), dict)
                    else ""
                )
                or ""
            ),
        }
    )
    return result


def _event_audit_row(
    *,
    event: str,
    detector_first: str,
    detector_second: str,
    corr_min: float,
    sky_samples: int,
    psi_samples: int,
    cosi_samples: int,
    response_floor_frac: float,
    min_ring_directions: int,
    tensor_equiv_ratio_max: float,
) -> Dict[str, Any]:
    payload = _load_pair_metrics(event, detector_first, detector_second)
    base = {
        "event": event,
        "detector_pair": f"{detector_first}-{detector_second}",
        "quality": "missing",
        "status": "inconclusive_missing_metrics",
        "status_reason": "metrics_not_found",
        "metrics_path": "",
    }
    if payload is None:
        return base

    metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
    ratio = metrics.get("ratio") if isinstance(metrics.get("ratio"), dict) else {}
    inputs = payload.get("inputs") if isinstance(payload.get("inputs"), dict) else {}
    best_lag_ms = _safe_float(metrics.get("best_lag_ms_apply_to_first"))
    if not math.isfinite(best_lag_ms):
        best_lag_ms = _safe_float(metrics.get("best_lag_ms_apply_to_h1"))
    abs_corr = _safe_float(metrics.get("abs_best_corr"))
    ratio_p16 = _safe_float(ratio.get("p16"))
    ratio_med = _safe_float(ratio.get("median"))
    ratio_p84 = _safe_float(ratio.get("p84"))
    analysis_fs_hz = _safe_float(inputs.get("analysis_fs_hz"))
    lag_scan = _load_lag_scan(event, detector_first, detector_second)
    delay_tol_ms = _estimate_delay_tolerance_ms(
        abs_corr=abs_corr,
        analysis_fs_hz=analysis_fs_hz,
        lag_scan=lag_scan,
    )

    base["metrics_path"] = str(payload.get("_metrics_path") or "")
    base["abs_corr"] = abs_corr
    base["best_lag_ms_apply_to_first"] = best_lag_ms
    base["delay_obs_ms"] = -best_lag_ms if math.isfinite(best_lag_ms) else float("nan")
    base["delay_tolerance_ms"] = delay_tol_ms
    base["ratio_obs_p16"] = ratio_p16
    base["ratio_obs_median"] = ratio_med
    base["ratio_obs_p84"] = ratio_p84

    if not (math.isfinite(abs_corr) and abs_corr >= float(corr_min)):
        base["quality"] = "low_corr"
        base["status"] = "inconclusive_low_corr"
        base["status_reason"] = f"abs_corr<{corr_min}"
        return base
    if not (
        math.isfinite(best_lag_ms)
        and math.isfinite(ratio_p16)
        and math.isfinite(ratio_med)
        and math.isfinite(ratio_p84)
        and math.isfinite(analysis_fs_hz)
        and analysis_fs_hz > 0.0
    ):
        base["quality"] = "bad_fields"
        base["status"] = "inconclusive_bad_fields"
        base["status_reason"] = "required_fields_missing"
        return base

    detector_a, detector_b = _canonical_pair(detector_first, detector_second)
    geometry = _build_network_geometry()
    if detector_a not in geometry or detector_b not in geometry:
        base["quality"] = "bad_detector"
        base["status"] = "inconclusive_detector_missing"
        base["status_reason"] = "detector_geometry_missing"
        return base

    position_a = geometry[detector_a]["position_m"]
    position_b = geometry[detector_b]["position_m"]
    tensor_a = geometry[detector_a]["tensor"]
    tensor_b = geometry[detector_b]["tensor"]
    baseline_max_delay_ms = float(np.linalg.norm(position_a - position_b) / _C * 1e3)

    psi_grid = np.linspace(0.0, math.pi, int(max(8, psi_samples)), endpoint=False, dtype=np.float64)
    cosi_grid = np.linspace(-1.0, 1.0, int(max(9, cosi_samples)), dtype=np.float64)
    sky_dirs = _fibonacci_sphere(int(max(256, sky_samples)))
    delay_obs_ms = float(-best_lag_ms)

    ring_dirs: List[np.ndarray] = []
    for direction in sky_dirs:
        delay_model_ms = float(np.dot(position_a - position_b, direction) / _C * 1e3)
        if abs(delay_model_ms - delay_obs_ms) <= delay_tol_ms:
            ring_dirs.append(direction)

    ring_array = np.asarray(ring_dirs, dtype=np.float64)
    base["baseline_max_delay_ms"] = baseline_max_delay_ms
    base["ring_directions_used"] = int(ring_array.shape[0])
    base["ring_directions_fraction"] = float(ring_array.shape[0] / max(int(sky_dirs.shape[0]), 1))

    if ring_array.shape[0] < int(max(1, min_ring_directions)):
        base["quality"] = "insufficient_geometry"
        base["status"] = "inconclusive_geometry"
        base["status_reason"] = "insufficient_ring_directions"
        return base

    tensor_all: List[float] = []
    vector_all: List[float] = []
    scalar_all: List[float] = []
    for direction in ring_array:
        tensor_ratio, scalar_ratio = _response_grid_for_direction(
            n=direction,
            tensor_h=tensor_a,
            tensor_l=tensor_b,
            psi_grid=psi_grid,
            cosi_grid=cosi_grid,
            response_floor_frac=float(response_floor_frac),
        )
        vector_ratio = _vector_ratio_grid_for_direction(
            direction=direction,
            tensor_first=tensor_a,
            tensor_second=tensor_b,
            psi_grid=psi_grid,
            cosi_grid=cosi_grid,
            response_floor_frac=float(response_floor_frac),
        )
        if tensor_ratio.size > 0:
            tensor_all.extend(tensor_ratio.tolist())
        if vector_ratio.size > 0:
            vector_all.extend(vector_ratio.tolist())
        if scalar_ratio.size > 0:
            scalar_all.extend(scalar_ratio.tolist())

    tensor_arr = np.asarray(tensor_all, dtype=np.float64)
    vector_arr = np.asarray(vector_all, dtype=np.float64)
    scalar_arr = np.asarray(scalar_all, dtype=np.float64)

    obs_lo = min(float(ratio_p16), float(ratio_p84))
    obs_hi = max(float(ratio_p16), float(ratio_p84))
    tensor_lo, tensor_hi = _range_clip(tensor_arr, 0.5, 99.5)
    vector_lo, vector_hi = _range_clip(vector_arr, 0.5, 99.5)
    scalar_lo, scalar_hi = _range_clip(scalar_arr, 0.5, 99.5)

    tensor_overlap = _interval_overlap(obs_lo, obs_hi, tensor_lo, tensor_hi)
    vector_overlap = _interval_overlap(obs_lo, obs_hi, vector_lo, vector_hi)
    scalar_overlap = _interval_overlap(obs_lo, obs_hi, scalar_lo, scalar_hi)
    tensor_mismatch = _min_rel_mismatch(tensor_arr, float(ratio_med))
    vector_mismatch = _min_rel_mismatch(vector_arr, float(ratio_med))
    scalar_mismatch = _min_rel_mismatch(scalar_arr, float(ratio_med))
    equivalence_ratio = (
        float(vector_mismatch / (tensor_mismatch + 1e-12))
        if math.isfinite(vector_mismatch) and math.isfinite(tensor_mismatch)
        else float("nan")
    )

    gate_corr = bool(math.isfinite(abs_corr) and abs_corr >= float(corr_min))
    gate_lag_causal = bool(abs(delay_obs_ms) <= baseline_max_delay_ms + delay_tol_ms)
    gate_vector_overlap = bool(vector_overlap)
    gate_tensor_equiv = bool(math.isfinite(equivalence_ratio) and equivalence_ratio <= float(tensor_equiv_ratio_max))
    hard_pass = gate_corr and gate_lag_causal and gate_vector_overlap and gate_tensor_equiv

    status = "reject_vector_mapping_failed"
    reason = []
    if hard_pass:
        status = "pass_tensor_equivalent_vector_mapping"
        reason.append("all_hard_gates_passed")
        if scalar_overlap:
            reason.append("scalar_overlap_remains_for_two_detector_case")
            status = "watch_scalar_not_excluded_two_detector_degeneracy"
    else:
        if not gate_corr:
            reason.append("low_corr")
        if not gate_lag_causal:
            reason.append("lag_outside_causal_baseline_bound")
        if not gate_vector_overlap:
            reason.append("vector_ratio_interval_no_overlap")
        if not gate_tensor_equiv:
            reason.append("vector_tensor_equivalence_broken")

    base.update(
        {
            "quality": "usable",
            "status": status,
            "status_reason": ";".join(reason) if reason else "no_reason",
            "gate_corr_pass": int(gate_corr),
            "gate_lag_causal_pass": int(gate_lag_causal),
            "gate_vector_overlap_pass": int(gate_vector_overlap),
            "gate_tensor_equiv_pass": int(gate_tensor_equiv),
            "scalar_overlap_flag": int(bool(scalar_overlap)),
            "tensor_overlap_flag": int(bool(tensor_overlap)),
            "vector_overlap_flag": int(bool(vector_overlap)),
            "tensor_ratio_lo": tensor_lo,
            "tensor_ratio_hi": tensor_hi,
            "vector_ratio_lo": vector_lo,
            "vector_ratio_hi": vector_hi,
            "scalar_ratio_lo": scalar_lo,
            "scalar_ratio_hi": scalar_hi,
            "tensor_min_rel_mismatch": tensor_mismatch,
            "vector_min_rel_mismatch": vector_mismatch,
            "scalar_min_rel_mismatch": scalar_mismatch,
            "vector_tensor_equiv_ratio": equivalence_ratio,
            "hard_pass_flag": int(hard_pass),
        }
    )
    return base


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    headers = [
        "event",
        "detector_pair",
        "quality",
        "status",
        "status_reason",
        "abs_corr",
        "best_lag_ms_apply_to_first",
        "delay_obs_ms",
        "delay_tolerance_ms",
        "baseline_max_delay_ms",
        "ring_directions_used",
        "ring_directions_fraction",
        "ratio_obs_p16",
        "ratio_obs_median",
        "ratio_obs_p84",
        "tensor_ratio_lo",
        "tensor_ratio_hi",
        "vector_ratio_lo",
        "vector_ratio_hi",
        "scalar_ratio_lo",
        "scalar_ratio_hi",
        "tensor_min_rel_mismatch",
        "vector_min_rel_mismatch",
        "scalar_min_rel_mismatch",
        "vector_tensor_equiv_ratio",
        "gate_corr_pass",
        "gate_lag_causal_pass",
        "gate_vector_overlap_pass",
        "gate_tensor_equiv_pass",
        "hard_pass_flag",
        "scalar_overlap_flag",
        "metrics_path",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        for row in rows:
            out: List[Any] = []
            for key in headers:
                value = row.get(key, "")
                if isinstance(value, float):
                    out.append(_fmt(value))
                else:
                    out.append(value)
            writer.writerow(out)


def _plot(rows: List[Dict[str, Any]], out_png: Path, detector_pair_label: str) -> None:
    _set_japanese_font()
    usable = [row for row in rows if str(row.get("quality")) == "usable"]
    labels = [str(row.get("event", "")) for row in usable]
    if not labels:
        labels = [str(row.get("event", "")) for row in rows]
    idx = np.arange(len(labels), dtype=float)

    lag_ratio = np.asarray(
        [
            abs(float(row.get("delay_obs_ms", float("nan"))))
            / max(float(row.get("baseline_max_delay_ms", float("nan"))) + float(row.get("delay_tolerance_ms", float("nan"))), 1e-12)
            for row in usable
        ],
        dtype=np.float64,
    )
    tensor_mismatch = np.asarray([float(row.get("tensor_min_rel_mismatch", float("nan"))) for row in usable], dtype=np.float64)
    vector_mismatch = np.asarray([float(row.get("vector_min_rel_mismatch", float("nan"))) for row in usable], dtype=np.float64)
    scalar_mismatch = np.asarray([float(row.get("scalar_min_rel_mismatch", float("nan"))) for row in usable], dtype=np.float64)
    equiv_ratio = np.asarray([float(row.get("vector_tensor_equiv_ratio", float("nan"))) for row in usable], dtype=np.float64)

    fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(12.8, 9.0), sharex=True)
    fig.suptitle(f"P_μ transverse-wave mapping audit (GW {detector_pair_label})")

    if usable:
        ax0.bar(idx, lag_ratio, color="#1f77b4", alpha=0.9)
    ax0.axhline(1.0, color="#333333", linestyle="--", linewidth=1.0, label="causal bound")
    ax0.set_ylabel("|delay_obs| / (delay_max+tol)")
    ax0.set_title("Phase-delay causal consistency")
    ax0.grid(True, axis="y", alpha=0.25)
    ax0.legend(loc="best", fontsize=9)

    width = 0.24
    if usable:
        ax1.bar(idx - width, tensor_mismatch, width=width, color="#2ca02c", alpha=0.9, label="tensor mismatch")
        ax1.bar(idx, vector_mismatch, width=width, color="#1f77b4", alpha=0.9, label="vector mismatch")
        ax1.bar(idx + width, scalar_mismatch, width=width, color="#ff7f0e", alpha=0.9, label="scalar mismatch")
    ax1.set_ylabel("min rel mismatch")
    ax1.set_title("Observed ratio to model families")
    ax1.grid(True, axis="y", alpha=0.25)
    ax1.legend(loc="best", fontsize=9)

    if usable:
        ax2.bar(idx, equiv_ratio, color="#9467bd", alpha=0.9, label="vector/tensor mismatch ratio")
    ax2.axhline(1.0, color="#333333", linestyle="--", linewidth=1.0, label="equivalent")
    ax2.set_ylabel("equivalence ratio")
    ax2.set_xlabel("event")
    ax2.set_title("Tensor-equivalence indicator")
    ax2.grid(True, axis="y", alpha=0.25)
    ax2.legend(loc="best", fontsize=9)

    ax2.set_xticks(idx)
    ax2.set_xticklabels(labels, rotation=0, fontsize=9)
    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.95])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Step 8.7.32.3: GW polarization tensor-equivalence mapping audit (pair + H1/L1/V1 network gate)."
    )
    parser.add_argument("--events", type=str, default="GW150914", help="Comma-separated event names.")
    parser.add_argument("--detector-first", type=str, default="H1")
    parser.add_argument("--detector-second", type=str, default="L1")
    parser.add_argument("--corr-min", type=float, default=0.6)
    parser.add_argument("--sky-samples", type=int, default=3000)
    parser.add_argument("--psi-samples", type=int, default=48)
    parser.add_argument("--cosi-samples", type=int, default=41)
    parser.add_argument("--response-floor-frac", type=float, default=0.1)
    parser.add_argument("--min-ring-directions", type=int, default=16)
    parser.add_argument("--tensor-equiv-ratio-max", type=float, default=5.0)
    parser.add_argument("--skip-network-gate", action="store_true")
    parser.add_argument(
        "--network-events",
        type=str,
        default="GW200129_065458,GW200224_222234,GW200115_042309,GW200311_115853",
        help="Comma-separated events used for H1/L1/V1 scalar-degeneracy gate.",
    )
    parser.add_argument("--network-detectors", type=str, default="H1,L1,V1")
    parser.add_argument("--network-corr-use-min", type=float, default=0.05)
    parser.add_argument("--network-sky-samples", type=int, default=5000)
    parser.add_argument("--network-psi-samples", type=int, default=36)
    parser.add_argument("--network-cosi-samples", type=int, default=41)
    parser.add_argument("--network-response-floor-frac", type=float, default=0.1)
    parser.add_argument("--network-min-ring-directions", type=int, default=8)
    parser.add_argument("--network-geometry-relax-factor", type=float, default=4.0)
    parser.add_argument("--network-geometry-delay-floor-ms", type=float, default=0.25)
    network_pruning_group = parser.add_mutually_exclusive_group()
    network_pruning_group.add_argument("--network-allow-pair-pruning", dest="network_allow_pair_pruning", action="store_true")
    network_pruning_group.add_argument("--network-no-pair-pruning", dest="network_allow_pair_pruning", action="store_false")
    parser.set_defaults(network_allow_pair_pruning=False)
    parser.add_argument(
        "--network-prefix",
        type=str,
        default="gw_polarization_h1_l1_v1_network_audit_for_step87323",
    )
    parser.add_argument("--outdir", type=str, default=str(_ROOT / "output" / "private" / "gw"))
    parser.add_argument("--public-outdir", type=str, default=str(_ROOT / "output" / "public" / "gw"))
    parser.add_argument("--prefix", type=str, default="pmodel_vector_gw_polarization_mapping_audit")
    args = parser.parse_args(list(argv) if argv is not None else None)

    events = [value.strip() for value in str(args.events).split(",") if value.strip()]
    if not events:
        print("[err] --events is empty")
        return 2

    rows = [
        _event_audit_row(
            event=event,
            detector_first=str(args.detector_first).upper(),
            detector_second=str(args.detector_second).upper(),
            corr_min=float(args.corr_min),
            sky_samples=int(args.sky_samples),
            psi_samples=int(args.psi_samples),
            cosi_samples=int(args.cosi_samples),
            response_floor_frac=float(args.response_floor_frac),
            min_ring_directions=int(args.min_ring_directions),
            tensor_equiv_ratio_max=float(args.tensor_equiv_ratio_max),
        )
        for event in events
    ]

    outdir = Path(str(args.outdir))
    public_outdir = Path(str(args.public_outdir))
    outdir.mkdir(parents=True, exist_ok=True)
    public_outdir.mkdir(parents=True, exist_ok=True)

    network_gate = _run_network_scalar_gate(
        enabled=not bool(args.skip_network_gate),
        events_csv=str(args.network_events),
        detectors_csv=str(args.network_detectors),
        corr_use_min=float(args.network_corr_use_min),
        sky_samples=int(args.network_sky_samples),
        psi_samples=int(args.network_psi_samples),
        cosi_samples=int(args.network_cosi_samples),
        response_floor_frac=float(args.network_response_floor_frac),
        min_ring_directions=int(args.network_min_ring_directions),
        geometry_relax_factor=float(args.network_geometry_relax_factor),
        geometry_delay_floor_ms=float(args.network_geometry_delay_floor_ms),
        allow_pair_pruning=bool(args.network_allow_pair_pruning),
        outdir=outdir,
        public_outdir=public_outdir,
        prefix=str(args.network_prefix),
    )

    usable = [row for row in rows if str(row.get("quality")) == "usable"]
    reject_rows = [row for row in usable if str(row.get("status", "")).startswith("reject")]
    watch_rows = [row for row in usable if str(row.get("status", "")).startswith("watch")]
    pass_rows = [row for row in usable if str(row.get("status", "")).startswith("pass")]

    if not usable:
        overall_status = "inconclusive"
        overall_reason = "no_usable_events"
    elif reject_rows:
        overall_status = "reject"
        overall_reason = "at_least_one_event_failed_hard_mapping_gate"
    elif watch_rows:
        if bool(network_gate.get("scalar_exclusion_pass")) and bool(network_gate.get("tensor_consistency_pass")):
            overall_status = "pass"
            overall_reason = "pairwise_scalar_overlap_resolved_by_h1_l1_v1_network_gate"
        elif bool(network_gate.get("scalar_reduction_pass")):
            overall_status = "watch"
            overall_reason = "pairwise_scalar_overlap_reduced_by_h1_l1_v1_network_gate_but_not_fully_excluded"
        else:
            overall_status = "watch"
            overall_reason = "vector_mapping_passed_but_scalar_not_excluded_for_some_events"
    else:
        overall_status = "pass"
        overall_reason = "vector_mapping_tensor_equivalent_for_all_usable_events"
    out_json = outdir / f"{args.prefix}.json"
    out_csv = outdir / f"{args.prefix}.csv"
    out_png = outdir / f"{args.prefix}.png"
    _write_csv(out_csv, rows)
    _plot(rows, out_png, detector_pair_label=f"{str(args.detector_first).upper()}/{str(args.detector_second).upper()}")

    payload: Dict[str, Any] = {
        "generated_utc": _iso_utc_now(),
        "schema": "wavep.gw.pmodel_vector_gw_polarization_mapping_audit.v2",
        "phase": 8,
        "step": "8.7.32.3",
        "inputs": {
            "events": events,
            "detector_first": str(args.detector_first).upper(),
            "detector_second": str(args.detector_second).upper(),
            "corr_min": float(args.corr_min),
            "sky_samples": int(args.sky_samples),
            "psi_samples": int(args.psi_samples),
            "cosi_samples": int(args.cosi_samples),
            "response_floor_frac": float(args.response_floor_frac),
            "min_ring_directions": int(args.min_ring_directions),
            "tensor_equiv_ratio_max": float(args.tensor_equiv_ratio_max),
            "network_gate_enabled": bool(not args.skip_network_gate),
            "network_events": [value.strip() for value in str(args.network_events).split(",") if value.strip()],
            "network_detectors": [value.strip().upper() for value in str(args.network_detectors).split(",") if value.strip()],
            "network_corr_use_min": float(args.network_corr_use_min),
            "network_sky_samples": int(args.network_sky_samples),
            "network_psi_samples": int(args.network_psi_samples),
            "network_cosi_samples": int(args.network_cosi_samples),
            "network_response_floor_frac": float(args.network_response_floor_frac),
            "network_min_ring_directions": int(args.network_min_ring_directions),
            "network_geometry_relax_factor": float(args.network_geometry_relax_factor),
            "network_geometry_delay_floor_ms": float(args.network_geometry_delay_floor_ms),
            "network_allow_pair_pruning": bool(args.network_allow_pair_pruning),
            "network_prefix": str(args.network_prefix),
        },
        "equations": {
            "vector_transverse_polarizations": "e_x^V = n⊗p + p⊗n,  e_y^V = n⊗q + q⊗n",
            "detector_response": "h_d = D_d : (h_x e_x^V + h_y e_y^V)",
            "mapping_target": "vector branch must reproduce observed H1/L1 amplitude-ratio interval under causal delay ring",
            "network_scalar_gate": "three-detector network (H1/L1/V1) checks scalar overlap reduction from pairwise degeneracy",
        },
        "summary": {
            "n_events_requested": int(len(events)),
            "n_rows": int(len(rows)),
            "n_usable_events": int(len(usable)),
            "n_pass_events": int(len(pass_rows)),
            "n_watch_events": int(len(watch_rows)),
            "n_reject_events": int(len(reject_rows)),
            "overall_status": overall_status,
            "overall_reason": overall_reason,
            "hard_pass_condition": "corr_pass && lag_causal_pass && vector_overlap_pass && tensor_equiv_pass",
            "failure_conditions": [
                "low_corr",
                "lag_outside_causal_baseline_bound",
                "vector_ratio_interval_no_overlap",
                "vector_tensor_equivalence_broken",
            ],
            "network_gate": network_gate,
        },
        "rows": rows,
        "outputs": {
            "audit_json": str(out_json).replace("\\", "/"),
            "audit_csv": str(out_csv).replace("\\", "/"),
            "audit_png": str(out_png).replace("\\", "/"),
        },
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    copied: List[str] = []
    for src in [out_json, out_csv, out_png]:
        dst = public_outdir / src.name
        shutil.copy2(src, dst)
        copied.append(str(dst).replace("\\", "/"))
    payload["outputs"]["public_copies"] = copied
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copy2(out_json, public_outdir / out_json.name)

    try:
        worklog.append_event(
            {
                "event_type": "pmodel_vector_gw_polarization_mapping_audit",
                "argv": list(sys.argv),
                "outputs": {"audit_json": out_json, "audit_csv": out_csv, "audit_png": out_png},
                "metrics": {
                    "overall_status": overall_status,
                    "n_usable_events": int(len(usable)),
                    "n_reject_events": int(len(reject_rows)),
                    "n_watch_events": int(len(watch_rows)),
                    "network_status": str(network_gate.get("status", "")),
                    "network_scalar_reduction_pass": int(bool(network_gate.get("scalar_reduction_pass"))),
                },
            }
        )
    except Exception:
        pass

    print(f"[ok] json: {out_json}")
    print(f"[ok] csv : {out_csv}")
    print(f"[ok] png : {out_png}")
    print(f"[ok] public copies: {len(copied)} files -> {public_outdir}")
    print(
        "[ok] network_gate="
        f"{network_gate.get('status')} reduction={int(bool(network_gate.get('scalar_reduction_pass')))} "
        f"exclusion={int(bool(network_gate.get('scalar_exclusion_pass')))}"
    )
    print(f"[ok] overall_status={overall_status} reason={overall_reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
