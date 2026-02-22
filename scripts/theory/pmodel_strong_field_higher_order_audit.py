#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pmodel_strong_field_higher_order_audit.py

Step 8.7.27（優先度S3）:
強場高次項（非線形）を最小1自由度で監査し、EHTとGWの同時拘束を固定する。

固定出力:
- output/public/theory/pmodel_strong_field_higher_order_audit.json
- output/public/theory/pmodel_strong_field_higher_order_audit.csv
- output/public/theory/pmodel_strong_field_higher_order_audit.png
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from scripts.summary import worklog  # type: ignore  # noqa: E402
except Exception:  # pragma: no cover
    worklog = None

try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover
    plt = None


@dataclass(frozen=True)
class LambdaObs:
    channel: str
    source: str
    label: str
    lambda_obs: float
    sigma_lambda: float
    note: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except Exception:
        return path.as_posix()


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _sha256_file(path: Path) -> Optional[str]:
    h = hashlib.sha256()
    try:
        with path.open("rb") as f:
            while True:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    break
                h.update(chunk)
    except Exception:
        return None
    return h.hexdigest().upper()


def _file_signature(path: Optional[Path]) -> Dict[str, Any]:
    if path is None:
        return {"exists": False, "path": None}
    payload: Dict[str, Any] = {"exists": bool(path.exists()), "path": _rel(path)}
    if not path.exists():
        return payload
    try:
        stat = path.stat()
        payload["size_bytes"] = int(stat.st_size)
        payload["mtime_utc"] = datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat()
    except Exception:
        pass
    sha256 = _sha256_file(path)
    if sha256 is not None:
        payload["sha256"] = sha256
    return payload


def _load_previous_watchpack(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = _read_json(path)
    except Exception:
        return {}
    diagnostics = payload.get("diagnostics")
    if not isinstance(diagnostics, dict):
        return {}
    watchpack = diagnostics.get("eht_kappa_update_watchpack")
    if not isinstance(watchpack, dict):
        return {}
    return watchpack


def _load_first_existing(paths: Sequence[Path]) -> Tuple[Dict[str, Any], Path]:
    for path in paths:
        if path.exists():
            return _read_json(path), path
    raise FileNotFoundError(f"No input found among: {[str(p) for p in paths]}")


def _load_optional_first_existing(paths: Sequence[Path]) -> Tuple[Dict[str, Any], Optional[Path]]:
    for path in paths:
        if path.exists():
            return _read_json(path), path
    return {}, None


def _load_optional_csv_first_existing(paths: Sequence[Path]) -> Tuple[List[Dict[str, str]], Optional[Path]]:
    for path in paths:
        if path.exists():
            return _read_csv_rows(path), path
    return [], None


def _to_float(v: Any) -> Optional[float]:
    try:
        val = float(v)
    except Exception:
        return None
    if not np.isfinite(val):
        return None
    return val


def _resolve_existing_path(path_like: Any) -> Optional[Path]:
    if path_like is None:
        return None
    raw = str(path_like).strip()
    if not raw:
        return None
    p = Path(raw)
    candidates: List[Path] = []
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.extend([ROOT / p, Path(raw)])
    for c in candidates:
        try:
            if c.exists():
                return c.resolve()
        except Exception:
            continue
    return None


def _load_planck_tt_binned_sigma(path: Path) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    try:
        arr = np.loadtxt(path)
    except Exception:
        return None
    if arr.ndim != 2 or arr.shape[1] < 4:
        return None
    ell = np.asarray(arr[:, 0], dtype=float)
    err_lo = np.asarray(arr[:, 2], dtype=float)
    err_hi = np.asarray(arr[:, 3], dtype=float)
    sigma = 0.5 * (np.abs(err_lo) + np.abs(err_hi))
    mask = np.isfinite(ell) & np.isfinite(sigma) & (sigma > 0.0)
    if not np.any(mask):
        return None
    return ell[mask], sigma[mask]


def _nearest_planck_sigma(ell_grid: np.ndarray, sigma_grid: np.ndarray, ell_target: float) -> Optional[float]:
    if ell_grid.size == 0 or sigma_grid.size == 0:
        return None
    idx = int(np.argmin(np.abs(ell_grid - float(ell_target))))
    sig = float(sigma_grid[idx])
    if not np.isfinite(sig) or sig <= 0.0:
        return None
    return sig


def _sigma_from_p16_p84(block: Dict[str, Any]) -> Optional[float]:
    p16_p84 = block.get("p16_p84")
    if not isinstance(p16_p84, list) or len(p16_p84) != 2:
        return None
    lo = _to_float(p16_p84[0])
    hi = _to_float(p16_p84[1])
    if lo is None or hi is None:
        return None
    if hi < lo:
        lo, hi = hi, lo
    sig = 0.5 * (hi - lo)
    if sig <= 0.0:
        return None
    return float(sig)


def _extract_eht_observables(payload: Dict[str, Any]) -> List[LambdaObs]:
    rows = payload.get("rows")
    if not isinstance(rows, list):
        return []
    out: List[LambdaObs] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        key = str(row.get("key", "")).strip().lower()
        if key not in {"m87", "sgra"}:
            continue
        obs = _to_float(row.get("shadow_diameter_obs_uas"))
        pred = _to_float(row.get("shadow_diameter_pmodel_uas"))
        sig = _to_float(row.get("shadow_diameter_obs_uas_sigma"))
        if obs is None or pred is None or sig is None or pred <= 0.0 or sig <= 0.0:
            continue
        lam = (obs / pred) - 1.0
        sig_lam = sig / pred
        if sig_lam <= 0.0:
            continue
        out.append(
            LambdaObs(
                channel="EHT",
                source=key,
                label=f"EHT {key.upper()} shadow",
                lambda_obs=float(lam),
                sigma_lambda=float(sig_lam),
                note="λ_H inferred from fractional shadow-size mismatch (obs/pred - 1).",
            )
        )
    return out


def _extract_gw_observables(payload: Dict[str, Any]) -> List[LambdaObs]:
    out: List[LambdaObs] = []
    gr = payload.get("gr_prediction_from_pe") if isinstance(payload.get("gr_prediction_from_pe"), dict) else {}
    cons = payload.get("consistency") if isinstance(payload.get("consistency"), dict) else {}
    ring_mc = cons.get("ringdown_inferred_final_mc") if isinstance(cons.get("ringdown_inferred_final_mc"), dict) else {}
    ring_1d = cons.get("ringdown_inferred_final") if isinstance(cons.get("ringdown_inferred_final"), dict) else {}

    m_gr_block = gr.get("final_mass_det_msun") if isinstance(gr.get("final_mass_det_msun"), dict) else {}
    a_gr_block = gr.get("final_spin") if isinstance(gr.get("final_spin"), dict) else {}
    m_gr = _to_float(m_gr_block.get("median"))
    a_gr = _to_float(a_gr_block.get("median"))
    m_gr_sig = _sigma_from_p16_p84(m_gr_block)
    a_gr_sig = _sigma_from_p16_p84(a_gr_block)

    m_ring_block = ring_mc.get("final_mass_det_msun") if isinstance(ring_mc.get("final_mass_det_msun"), dict) else {}
    a_ring_block = ring_mc.get("final_spin") if isinstance(ring_mc.get("final_spin"), dict) else {}
    m_ring = _to_float(m_ring_block.get("median"))
    a_ring = _to_float(a_ring_block.get("median"))
    m_ring_sig = _sigma_from_p16_p84(m_ring_block)
    a_ring_sig = _sigma_from_p16_p84(a_ring_block)

    if m_ring is None:
        m_ring = _to_float(ring_1d.get("final_mass_det_msun"))
    if a_ring is None:
        a_ring = _to_float(ring_1d.get("final_spin"))

    if m_gr is not None and m_ring is not None and m_gr > 0:
        sig_mass = None
        if m_gr_sig is not None and m_ring_sig is not None:
            sig_mass = math.sqrt(m_gr_sig * m_gr_sig + m_ring_sig * m_ring_sig)
        elif m_gr_sig is not None:
            sig_mass = m_gr_sig
        elif m_ring_sig is not None:
            sig_mass = m_ring_sig
        if sig_mass is not None and sig_mass > 0:
            out.append(
                LambdaObs(
                    channel="GW",
                    source="gw250114_mass",
                    label="GW250114 IMR mass",
                    lambda_obs=float((m_ring / m_gr) - 1.0),
                    sigma_lambda=float(sig_mass / m_gr),
                    note="λ_H from ringdown-inferred final mass vs IMR posterior median.",
                )
            )
    if a_gr is not None and a_ring is not None and a_gr > 0:
        sig_spin = None
        if a_gr_sig is not None and a_ring_sig is not None:
            sig_spin = math.sqrt(a_gr_sig * a_gr_sig + a_ring_sig * a_ring_sig)
        elif a_gr_sig is not None:
            sig_spin = a_gr_sig
        elif a_ring_sig is not None:
            sig_spin = a_ring_sig
        if sig_spin is not None and sig_spin > 0:
            out.append(
                LambdaObs(
                    channel="GW",
                    source="gw250114_spin",
                    label="GW250114 IMR spin",
                    lambda_obs=float((a_ring / a_gr) - 1.0),
                    sigma_lambda=float(sig_spin / a_gr),
                    note="λ_H from ringdown-inferred final spin vs IMR posterior median.",
                )
            )
    return out


def _extract_gw_premger_overlap(payload: Dict[str, Any]) -> Dict[str, Any]:
    dets = payload.get("detectors")
    if not isinstance(dets, list):
        return {"median_overlap": float("nan"), "n_detectors": 0}
    overlaps: List[float] = []
    for det in dets:
        if not isinstance(det, dict):
            continue
        wave = det.get("waveform_fit") if isinstance(det.get("waveform_fit"), dict) else {}
        ov = _to_float(wave.get("overlap"))
        if ov is None:
            continue
        overlaps.append(float(ov))
    if not overlaps:
        return {"median_overlap": float("nan"), "n_detectors": 0}
    arr = np.asarray(overlaps, dtype=float)
    return {
        "median_overlap": float(np.median(arr)),
        "mean_overlap": float(np.mean(arr)),
        "min_overlap": float(np.min(arr)),
        "max_overlap": float(np.max(arr)),
        "n_detectors": int(arr.size),
    }


def _extract_gw_primary_homology(payload: Dict[str, Any]) -> Dict[str, Any]:
    metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
    ratio = metrics.get("ratio") if isinstance(metrics.get("ratio"), dict) else {}
    gate = payload.get("gate") if isinstance(payload.get("gate"), dict) else {}

    corr = _to_float(metrics.get("abs_best_corr"))
    if corr is None:
        raw_corr = _to_float(metrics.get("best_corr"))
        corr = abs(raw_corr) if raw_corr is not None else None

    out: Dict[str, Any] = {
        "homology_metric": "abs_best_corr(H1,L1; primary strain)",
        "homology_value": corr,
        "lag_ms_apply_to_first": _to_float(metrics.get("best_lag_ms_apply_to_first")),
        "ratio_iqr_over_median": _to_float(ratio.get("iqr_over_median")),
        "ratio_slope_window_fraction": _to_float(ratio.get("slope_window_fraction")),
        "ratio_selected_points": _to_float(ratio.get("selected_points")),
        "ratio_total_points": _to_float(ratio.get("total_points")),
        "gate_status": str(gate.get("status")) if gate.get("status") is not None else None,
        "ok": corr is not None,
    }
    return out


def _extract_gw_multi_event_homology(payload: Dict[str, Any]) -> Dict[str, Any]:
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    n_usable = _to_float(summary.get("n_usable_events"))
    median_corr = _to_float(summary.get("median_abs_corr_usable"))
    event_cv = _to_float(summary.get("event_dependency_index_ratio_cv"))
    status = summary.get("overall_status")
    return {
        "homology_metric": "median_abs_corr_usable(H1/L1 multi-event)",
        "n_usable_events": int(n_usable) if n_usable is not None else 0,
        "homology_value": median_corr,
        "event_dependency_index_ratio_cv": event_cv,
        "overall_status": str(status) if status is not None else None,
        "ok": median_corr is not None,
    }


def _extract_gw_area_theorem(payload: Dict[str, Any]) -> Dict[str, Any]:
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    sigma_ref = summary.get("sigma_ref") if isinstance(summary.get("sigma_ref"), dict) else {}
    sigma_combined = _to_float(sigma_ref.get("sigma_gaussian_combined"))
    sigma_min = _to_float(summary.get("sigma_min_over_times_combined"))
    t_ref = _to_float(sigma_ref.get("reference_inspiral_time"))
    first_ge5 = _to_float(summary.get("first_time_sigma_ge_5_combined"))
    return {
        "sigma_metric": "sigma_gaussian_combined(area theorem)",
        "sigma_gaussian_combined": sigma_combined,
        "sigma_min_over_times_combined": sigma_min,
        "reference_inspiral_time": t_ref,
        "first_time_sigma_ge_5_combined": first_ge5,
        "ok": sigma_combined is not None,
    }


def _extract_gw_polarization_stage_readiness(payload: Dict[str, Any]) -> Dict[str, Any]:
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    rows = payload.get("rows") if isinstance(payload.get("rows"), list) else []
    locked = _to_float(summary.get("locked_corr_use_min"))
    locked_row: Dict[str, Any] = {}
    if locked is not None:
        for row in rows:
            if not isinstance(row, dict):
                continue
            corr = _to_float(row.get("corr_use_min"))
            if corr is not None and abs(corr - locked) <= 1.0e-9:
                locked_row = row
                break
    if not locked_row and rows:
        for row in rows:
            if isinstance(row, dict):
                locked_row = row
                break

    n_usable_locked = int(locked_row.get("n_usable_events", 0)) if isinstance(locked_row, dict) else 0
    status_locked = str(locked_row.get("overall_status") or "").lower() if isinstance(locked_row, dict) else ""
    n_candidate_thresholds = int(_to_float(summary.get("n_candidate_thresholds")) or 0)

    high_tension_candidate_ready = bool(
        (status_locked == "reject") and (n_usable_locked >= 1) and (n_candidate_thresholds >= 1)
    )
    return {
        "ok": bool(summary or rows),
        "locked_corr_use_min": locked,
        "locked_overall_status": status_locked if status_locked else None,
        "locked_n_usable_events": n_usable_locked,
        "n_candidate_thresholds": n_candidate_thresholds,
        "selection_signature_stable_across_candidates": bool(summary.get("selection_signature_stable_across_candidates"))
        if summary.get("selection_signature_stable_across_candidates") is not None
        else None,
        "high_tension_candidate_ready": high_tension_candidate_ready,
        "note": (
            "GW polarization stage audit is treated as candidate-channel readiness only; "
            "it is not yet injected into lambda_H fit until stable multi-event usable set is available."
        ),
    }


def _derive_aic_support_recovery_target(fit_joint: Dict[str, Any]) -> Dict[str, Any]:
    chi2_base = _to_float(fit_joint.get("chi2_baseline"))
    chi2_fit = _to_float(fit_joint.get("chi2_fit"))
    if chi2_base is None or chi2_fit is None:
        return {
            "ok": False,
            "note": "chi2_baseline/chi2_fit missing.",
        }

    target_delta_chi2_gain = 4.0  # for delta_aic <= -2 with k=1: chi2_base - chi2_fit >= 4
    current_delta_chi2_gain = float(chi2_base - chi2_fit)
    missing_delta_chi2_gain = float(max(0.0, target_delta_chi2_gain - current_delta_chi2_gain))

    required_abs_z_single_if_zero_residual = float(math.sqrt(missing_delta_chi2_gain))
    required_abs_z_single_if_half_sigma_residual = float(math.sqrt(missing_delta_chi2_gain + 0.25))
    required_abs_z_single_if_one_sigma_residual = float(math.sqrt(missing_delta_chi2_gain + 1.0))

    def _n_needed(avg_abs_z: float) -> int:
        if avg_abs_z <= 0.0:
            return 0
        gain_per_obs = avg_abs_z * avg_abs_z
        if gain_per_obs <= 0.0:
            return 0
        return int(math.ceil(missing_delta_chi2_gain / gain_per_obs)) if missing_delta_chi2_gain > 0 else 0

    return {
        "ok": True,
        "formula": "delta_AIC = chi2_fit + 2*k - chi2_baseline; support requires delta_AIC <= -2 (k=1).",
        "target_delta_chi2_gain_for_support_k1": target_delta_chi2_gain,
        "current_delta_chi2_gain": current_delta_chi2_gain,
        "missing_delta_chi2_gain": missing_delta_chi2_gain,
        "required_abs_z_single_new_channel_if_fitted_residual_zero": required_abs_z_single_if_zero_residual,
        "required_abs_z_single_new_channel_if_fitted_residual_half_sigma": required_abs_z_single_if_half_sigma_residual,
        "required_abs_z_single_new_channel_if_fitted_residual_one_sigma": required_abs_z_single_if_one_sigma_residual,
        "min_new_channels_if_each_abs_z_1p0_and_well_fitted": _n_needed(1.0),
        "min_new_channels_if_each_abs_z_1p5_and_well_fitted": _n_needed(1.5),
        "min_new_channels_if_each_abs_z_2p0_and_well_fitted": _n_needed(2.0),
        "note": (
            "z is the baseline standardized mismatch of additional independent channel(s). "
            "This is a planning estimator under additive-chi2 approximation."
        ),
    }


def _extract_gw_polarization_high_tension_candidates(
    payload_entries: Sequence[Tuple[str, Dict[str, Any]]],
    support_recovery_target: Dict[str, Any],
) -> Dict[str, Any]:
    required_abs_z = _to_float(support_recovery_target.get("required_abs_z_single_new_channel_if_fitted_residual_zero"))
    rows: List[Dict[str, Any]] = []
    for source_id, payload in payload_entries:
        summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
        event_rows = payload.get("rows") if isinstance(payload.get("rows"), list) else []

        n_usable_events = int(_to_float(summary.get("n_usable_events")) or 0)
        overall_status = str(summary.get("overall_status") or "").lower()
        overall_reason = summary.get("overall_reason")

        tensor_mismatch_vals: List[float] = []
        scalar_mismatch_vals: List[float] = []
        n_reject_tensor_events = 0
        for event_row in event_rows:
            if not isinstance(event_row, dict):
                continue
            status = str(event_row.get("status") or "").lower()
            if status == "reject_tensor_response":
                n_reject_tensor_events += 1
            tmis = _to_float(event_row.get("tensor_mismatch_max"))
            smis = _to_float(event_row.get("scalar_mismatch_max"))
            if tmis is not None:
                tensor_mismatch_vals.append(abs(float(tmis)))
            if smis is not None:
                scalar_mismatch_vals.append(abs(float(smis)))

        tensor_arr = np.array(tensor_mismatch_vals, dtype=float) if tensor_mismatch_vals else np.array([], dtype=float)
        scalar_arr = np.array(scalar_mismatch_vals, dtype=float) if scalar_mismatch_vals else np.array([], dtype=float)
        tensor_median = float(np.median(tensor_arr)) if tensor_arr.size > 0 else None
        tensor_p75 = float(np.percentile(tensor_arr, 75.0)) if tensor_arr.size > 0 else None
        tensor_max = float(np.max(tensor_arr)) if tensor_arr.size > 0 else None
        scalar_median = float(np.median(scalar_arr)) if scalar_arr.size > 0 else None

        eligible = bool(overall_status == "reject" and n_usable_events >= 2 and tensor_median is not None)
        clears_required_if_unit_scale = bool(
            eligible and required_abs_z is not None and tensor_median is not None and tensor_median >= required_abs_z
        )

        rows.append(
            {
                "source_id": source_id,
                "overall_status": overall_status if overall_status else None,
                "overall_reason": overall_reason,
                "n_events": len([r for r in event_rows if isinstance(r, dict)]),
                "n_usable_events": n_usable_events,
                "n_reject_tensor_events": n_reject_tensor_events,
                "tensor_mismatch_median": tensor_median,
                "tensor_mismatch_p75": tensor_p75,
                "tensor_mismatch_max": tensor_max,
                "scalar_mismatch_median": scalar_median,
                "eligible_as_high_tension_candidate": eligible,
                "would_clear_support_if_unit_scale": clears_required_if_unit_scale,
            }
        )

    eligible_rows = [r for r in rows if bool(r.get("eligible_as_high_tension_candidate"))]
    best_row: Optional[Dict[str, Any]] = None
    if eligible_rows:
        best_row = max(
            eligible_rows,
            key=lambda r: float(_to_float(r.get("tensor_mismatch_median")) or float("-inf")),
        )

    return {
        "ok": len(rows) > 0,
        "required_abs_z_for_single_new_channel": required_abs_z,
        "n_loaded_candidate_audits": len(rows),
        "n_eligible_candidates": len(eligible_rows),
        "best_candidate_source_id": best_row.get("source_id") if best_row else None,
        "best_candidate_tensor_mismatch_median": _to_float(best_row.get("tensor_mismatch_median")) if best_row else None,
        "best_candidate_would_clear_support_if_unit_scale": (
            bool(best_row.get("would_clear_support_if_unit_scale")) if best_row else None
        ),
        "rows": rows,
        "note": (
            "tensor_mismatch is a relative mismatch from GW polarization geometry audit, not a standardized z. "
            "Comparison against required_abs_z is a planning-only unit-scale proxy until a formal z-mapping is fixed."
        ),
    }


def _build_gw_polarization_injected_observables(
    registry: Dict[str, Any],
    base_observables: Sequence[LambdaObs],
    fit_gw: Dict[str, Any],
    fit_joint: Dict[str, Any],
) -> Dict[str, Any]:
    valid_sigmas_all = [float(o.sigma_lambda) for o in base_observables if float(o.sigma_lambda) > 0.0]
    valid_sigmas_gw = [
        float(o.sigma_lambda) for o in base_observables if o.channel == "GW" and float(o.sigma_lambda) > 0.0
    ]

    sigma_ref_all = float(np.median(np.asarray(valid_sigmas_all, dtype=float))) if valid_sigmas_all else None
    sigma_ref_gw = float(np.median(np.asarray(valid_sigmas_gw, dtype=float))) if valid_sigmas_gw else None
    sigma_ref_fit_gw = _to_float(fit_gw.get("lambda_sigma"))
    sigma_ref_candidates = [sigma_ref_fit_gw, sigma_ref_gw, sigma_ref_all]
    sigma_ref = next((float(v) for v in sigma_ref_candidates if v is not None and float(v) > 0.0), None)
    if sigma_ref is None:
        return {
            "ok": False,
            "note": "base sigma reference unavailable.",
            "rows": [],
            "observables": [],
        }

    lambda_gw = _to_float(fit_gw.get("lambda_fit"))
    lambda_joint = _to_float(fit_joint.get("lambda_fit"))
    sign_reference = 1.0
    if lambda_gw is not None and lambda_joint is not None and abs(lambda_gw - lambda_joint) > 0.0:
        sign_reference = 1.0 if (lambda_gw - lambda_joint) >= 0.0 else -1.0
    elif lambda_gw is not None and abs(lambda_gw) > 0.0:
        sign_reference = 1.0 if lambda_gw >= 0.0 else -1.0

    rows_in = registry.get("rows") if isinstance(registry.get("rows"), list) else []
    rows_out: List[Dict[str, Any]] = []
    observables: List[LambdaObs] = []
    required_abs_z = _to_float(registry.get("required_abs_z_for_single_new_channel"))

    for row in rows_in:
        if not isinstance(row, dict):
            continue
        if not bool(row.get("eligible_as_high_tension_candidate")):
            continue
        source_id = str(row.get("source_id") or "").strip()
        tensor_mismatch = _to_float(row.get("tensor_mismatch_median"))
        scalar_mismatch = _to_float(row.get("scalar_mismatch_median"))
        n_reject_tensor = max(0, int(_to_float(row.get("n_reject_tensor_events")) or 0))
        n_usable = max(0, int(_to_float(row.get("n_usable_events")) or 0))
        n_events = max(1, int(_to_float(row.get("n_events")) or 1))
        if not source_id or tensor_mismatch is None:
            continue

        if scalar_mismatch is None or scalar_mismatch <= 0.0:
            scalar_mismatch = float(tensor_mismatch)

        coverage_ratio = float(max(0.0, min(1.0, n_usable / n_events)))
        reject_ratio_within_usable = float(max(0.0, min(1.0, n_reject_tensor / max(1, n_usable))))
        sensitivity_factor = float(math.sqrt(reject_ratio_within_usable))
        correlation_factor = float(math.sqrt(coverage_ratio))
        contrast_den = max(float(tensor_mismatch), float(scalar_mismatch), 1.0e-12)
        contrast_factor = float(math.sqrt(max(0.0, min(1.0, float(tensor_mismatch) / contrast_den))))
        z_mapped_raw = float(tensor_mismatch * sensitivity_factor * correlation_factor * contrast_factor)
        z_mapped = float(max(0.0, min(5.0, z_mapped_raw)))

        lambda_obs = float(sign_reference * z_mapped * sigma_ref)
        sigma_lambda = sigma_ref
        observables.append(
            LambdaObs(
                channel="GW_POL",
                source=f"gw_pol::{source_id}",
                label=f"GW polarization candidate {source_id}",
                lambda_obs=lambda_obs,
                sigma_lambda=sigma_lambda,
                note=(
                    "Projected pseudo-observable from tensor/scalar mismatch contrast with sensitivity "
                    "(reject ratio) and correlation (usable coverage) damping."
                ),
            )
        )
        rows_out.append(
            {
                "source_id": source_id,
                "tensor_mismatch_median": tensor_mismatch,
                "scalar_mismatch_median": scalar_mismatch,
                "n_reject_tensor_events": n_reject_tensor,
                "n_usable_events": n_usable,
                "n_events": n_events,
                "coverage_ratio": coverage_ratio,
                "reject_ratio_within_usable": reject_ratio_within_usable,
                "sensitivity_factor": sensitivity_factor,
                "correlation_factor": correlation_factor,
                "contrast_factor": contrast_factor,
                "z_mapped_raw": z_mapped_raw,
                "z_mapped_capped": z_mapped,
                "required_abs_z_reference": required_abs_z,
                "z_mapped_over_required_abs_z": (
                    float(z_mapped / required_abs_z) if required_abs_z is not None and required_abs_z > 0.0 else None
                ),
                "sigma_ref": sigma_ref,
                "sign_reference": sign_reference,
                "lambda_obs_injected": lambda_obs,
                "sigma_lambda_injected": sigma_lambda,
            }
        )

    return {
        "ok": len(observables) > 0,
        "mapping_formula": (
            "z_mapped = clip(tensor_mismatch_median * sqrt(n_reject_tensor_events / n_usable_events) "
            "* sqrt(n_usable_events / n_events) "
            "* sqrt(tensor_mismatch_median / max(tensor_mismatch_median, scalar_mismatch_median)), 0, 5)"
        ),
        "sign_policy": "sign(lambda_fit_gw_only - lambda_fit_joint_baseline) if available else sign(lambda_fit_gw_only)",
        "sigma_reference_policy": (
            "lambda_sigma(gw_only) -> median(sigma_lambda over GW channels) -> median(sigma_lambda over all channels)"
        ),
        "sigma_reference_value": sigma_ref,
        "sign_reference_value": sign_reference,
        "required_abs_z_reference": required_abs_z,
        "rows": rows_out,
        "observables": observables,
        "note": (
            "This mapping freezes a physics-calibrated proxy (sign/sensitivity/correlation/contrast) "
            "for Step 8.7.27.14 injection trial."
        ),
    }


def _extract_cross_domain_high_tension_candidates(
    *,
    frame_dragging_scalar_limit: Dict[str, Any],
    cmb_peak_uplift: Dict[str, Any],
    cluster_collision_offset: Dict[str, Any],
    support_recovery_target: Dict[str, Any],
) -> Dict[str, Any]:
    required_abs_z = _to_float(support_recovery_target.get("required_abs_z_single_new_channel_if_fitted_residual_zero"))
    rows: List[Dict[str, Any]] = []

    frame_rows = frame_dragging_scalar_limit.get("rows") if isinstance(frame_dragging_scalar_limit.get("rows"), list) else []
    z_frame: List[float] = []
    frame_reject_experiments: List[str] = []
    for row in frame_rows:
        if not isinstance(row, dict):
            continue
        observable = str(row.get("observable") or "").strip().lower()
        status = str(row.get("status") or "").strip().lower()
        z_scalar = _to_float(row.get("z_scalar"))
        if observable != "frame_dragging" or status != "reject" or z_scalar is None:
            continue
        z_frame.append(abs(float(z_scalar)))
        exp = str(row.get("experiment") or "").strip()
        if exp:
            frame_reject_experiments.append(exp)
    if z_frame:
        z_raw = float(max(z_frame))
        rows.append(
            {
                "source_id": "frame_dragging_scalar_limit",
                "domain": "theory.frame_dragging",
                "z_proxy_raw": z_raw,
                "n_reject_channels": int(len(z_frame)),
                "summary_metric": "max_abs_z_scalar(frame_dragging_reject)",
                "summary_value": z_raw,
                "experiments": sorted(set(frame_reject_experiments)),
                "note": (
                    "Scalar-only frame-dragging branch is rejected by GP-B/LAGEOS; "
                    "use max |z_scalar| as cross-domain tension proxy."
                ),
            }
        )

    models = cmb_peak_uplift.get("models") if isinstance(cmb_peak_uplift.get("models"), list) else []
    thresholds = cmb_peak_uplift.get("thresholds") if isinstance(cmb_peak_uplift.get("thresholds"), dict) else {}
    baryon_model: Dict[str, Any] = {}
    for model in models:
        if isinstance(model, dict) and str(model.get("key") or "").strip().lower() == "baryon_only":
            baryon_model = model
            break
    if baryon_model:
        metrics = baryon_model.get("metrics") if isinstance(baryon_model.get("metrics"), dict) else {}
        ratios = baryon_model.get("ratios") if isinstance(baryon_model.get("ratios"), dict) else {}
        amp_err = _to_float(metrics.get("max_abs_delta_amp_rel"))
        ratio_err = _to_float(ratios.get("a3_a1_abs_rel_error"))
        amp_pass = _to_float(thresholds.get("amplitude_pass"))
        ratio_pass = _to_float(thresholds.get("ratio_pass"))
        z_amp = (abs(float(amp_err)) / amp_pass) if amp_err is not None and amp_pass is not None and amp_pass > 0.0 else None
        z_ratio = (
            (abs(float(ratio_err)) / ratio_pass) if ratio_err is not None and ratio_pass is not None and ratio_pass > 0.0 else None
        )
        z_candidates = [v for v in [z_amp, z_ratio] if v is not None and np.isfinite(v)]
        if z_candidates:
            z_raw = float(max(z_candidates))
            rows.append(
                {
                    "source_id": "cmb_peak_uplift_baryon_gap",
                    "domain": "cosmology.cmb",
                    "z_proxy_raw": z_raw,
                    "n_reject_channels": int(len(z_candidates)),
                    "summary_metric": "max(normalized baryon-only mismatch)",
                    "summary_value": z_raw,
                    "components": {
                        "z_amp_over_pass": z_amp,
                        "z_a3a1_over_pass": z_ratio,
                        "amp_abs_rel_error": amp_err,
                        "a3a1_abs_rel_error": ratio_err,
                    },
                    "note": (
                        "CMB baryon-only branch fails amplitude/ratio gates; "
                        "normalize mismatches by pass thresholds to build a cross-domain z proxy."
                    ),
                }
            )

    cluster_models = cluster_collision_offset.get("models") if isinstance(cluster_collision_offset.get("models"), dict) else {}
    baryon = cluster_models.get("baryon_only") if isinstance(cluster_models.get("baryon_only"), dict) else {}
    comparison = cluster_models.get("comparison") if isinstance(cluster_models.get("comparison"), dict) else {}
    z_lens = _to_float(baryon.get("max_abs_z_p_lens"))
    delta_chi2 = _to_float(comparison.get("delta_chi2_baryon_minus_pmodel"))
    if z_lens is not None and np.isfinite(z_lens):
        z_raw = abs(float(z_lens))
        rows.append(
            {
                "source_id": "cluster_collision_baryon_gap",
                "domain": "cosmology.cluster_collision",
                "z_proxy_raw": z_raw,
                "n_reject_channels": 1,
                "summary_metric": "max_abs_z_p_lens(baryon_only)",
                "summary_value": z_raw,
                "components": {
                    "delta_chi2_baryon_minus_pmodel": delta_chi2,
                },
                "note": (
                    "Bullet proxy audit rejects baryon-only offset branch; "
                    "use lens-anchor |z| mismatch as cross-domain tension proxy."
                ),
            }
        )

    eligible_rows: List[Dict[str, Any]] = []
    for row in rows:
        z_raw = _to_float(row.get("z_proxy_raw"))
        z_capped = float(max(0.0, min(5.0, float(z_raw)))) if z_raw is not None else None
        eligible = bool(z_capped is not None and (required_abs_z is None or z_capped >= required_abs_z))
        row["z_proxy_capped"] = z_capped
        row["required_abs_z_reference"] = required_abs_z
        row["z_proxy_over_required_abs_z"] = (
            float(z_capped / required_abs_z) if z_capped is not None and required_abs_z is not None and required_abs_z > 0.0 else None
        )
        row["eligible_as_high_tension_candidate"] = eligible
        if eligible:
            eligible_rows.append(row)

    best_row: Optional[Dict[str, Any]] = None
    if eligible_rows:
        best_row = max(
            eligible_rows,
            key=lambda r: float(_to_float(r.get("z_proxy_capped")) or float("-inf")),
        )

    return {
        "ok": len(rows) > 0,
        "required_abs_z_for_single_new_channel": required_abs_z,
        "n_loaded_candidate_audits": len(rows),
        "n_eligible_candidates": len(eligible_rows),
        "best_candidate_source_id": best_row.get("source_id") if best_row else None,
        "best_candidate_domain": best_row.get("domain") if best_row else None,
        "best_candidate_z_proxy_capped": _to_float(best_row.get("z_proxy_capped")) if best_row else None,
        "rows": rows,
        "note": (
            "Cross-domain candidates are tension proxies derived from already-fixed primary audits "
            "(frame-dragging scalar-limit, CMB uplift, Bullet offset). They are not direct lambda_H observables."
        ),
    }


def _build_cross_domain_injected_observables(
    registry: Dict[str, Any],
    base_observables: Sequence[LambdaObs],
    fit_joint: Dict[str, Any],
) -> Dict[str, Any]:
    valid_sigmas_all = [float(o.sigma_lambda) for o in base_observables if float(o.sigma_lambda) > 0.0]
    sigma_ref_all = float(np.median(np.asarray(valid_sigmas_all, dtype=float))) if valid_sigmas_all else None
    sigma_ref_joint = _to_float(fit_joint.get("lambda_sigma"))
    sigma_ref_candidates = [sigma_ref_joint, sigma_ref_all]
    sigma_ref = next((float(v) for v in sigma_ref_candidates if v is not None and float(v) > 0.0), None)
    if sigma_ref is None:
        return {
            "ok": False,
            "note": "base sigma reference unavailable.",
            "rows": [],
            "observables_plus": [],
            "observables_minus": [],
        }

    rows_in = registry.get("rows") if isinstance(registry.get("rows"), list) else []
    rows_out: List[Dict[str, Any]] = []
    observables_plus: List[LambdaObs] = []
    observables_minus: List[LambdaObs] = []

    for row in rows_in:
        if not isinstance(row, dict):
            continue
        if not bool(row.get("eligible_as_high_tension_candidate")):
            continue
        source_id = str(row.get("source_id") or "").strip()
        domain = str(row.get("domain") or "").strip()
        z_capped = _to_float(row.get("z_proxy_capped"))
        if not source_id or z_capped is None:
            continue
        lambda_plus = float(z_capped * sigma_ref)
        lambda_minus = float(-z_capped * sigma_ref)

        observables_plus.append(
            LambdaObs(
                channel="XDOM",
                source=f"xdom::{source_id}",
                label=f"Cross-domain candidate {source_id}",
                lambda_obs=lambda_plus,
                sigma_lambda=sigma_ref,
                note="Projected pseudo-observable from cross-domain high-tension audit (plus-sign envelope).",
            )
        )
        observables_minus.append(
            LambdaObs(
                channel="XDOM",
                source=f"xdom::{source_id}",
                label=f"Cross-domain candidate {source_id}",
                lambda_obs=lambda_minus,
                sigma_lambda=sigma_ref,
                note="Projected pseudo-observable from cross-domain high-tension audit (minus-sign envelope).",
            )
        )
        rows_out.append(
            {
                "source_id": source_id,
                "domain": domain,
                "z_proxy_raw": _to_float(row.get("z_proxy_raw")),
                "z_proxy_capped": z_capped,
                "required_abs_z_reference": _to_float(row.get("required_abs_z_reference")),
                "z_proxy_over_required_abs_z": _to_float(row.get("z_proxy_over_required_abs_z")),
                "sigma_ref": sigma_ref,
                "lambda_obs_plus": lambda_plus,
                "lambda_obs_minus": lambda_minus,
                "summary_metric": row.get("summary_metric"),
                "summary_value": _to_float(row.get("summary_value")),
            }
        )

    return {
        "ok": len(observables_plus) > 0,
        "mapping_formula": "z_mapped = clip(z_proxy_raw, 0, 5); lambda_obs = +/- z_mapped * sigma_ref",
        "sign_policy": "dual-envelope (plus/minus); report both and select lower delta_aic projection",
        "sigma_reference_policy": "lambda_sigma(joint) -> median(sigma_lambda over all baseline channels)",
        "sigma_reference_value": sigma_ref,
        "rows": rows_out,
        "observables_plus": observables_plus,
        "observables_minus": observables_minus,
        "note": (
            "Projection-only envelope to estimate whether non-GW independent high-tension evidence "
            "could close missing delta-chi2 for AIC support."
        ),
    }


def _extract_cross_domain_direct_observables(
    *,
    frame_dragging_scalar_limit: Dict[str, Any],
    cmb_peak_uplift: Dict[str, Any],
    cluster_collision_offset: Dict[str, Any],
) -> Dict[str, Any]:
    observables: List[LambdaObs] = []
    rows: List[Dict[str, Any]] = []
    sigma_source_counts: Dict[str, int] = {}
    covariance_pairs: List[Dict[str, Any]] = []

    def _count_sigma_source(source: str) -> None:
        sigma_source_counts[source] = int(sigma_source_counts.get(source, 0) + 1)

    def _add_covariance_pair(source_i: str, source_j: str, cov_lambda: float, origin: str) -> None:
        if not source_i or not source_j or source_i == source_j:
            return
        cov_val = _to_float(cov_lambda)
        if cov_val is None:
            return
        covariance_pairs.append(
            {
                "source_i": source_i,
                "source_j": source_j,
                "cov_lambda": float(cov_val),
                "origin": origin,
            }
        )

    frame_rows = frame_dragging_scalar_limit.get("rows") if isinstance(frame_dragging_scalar_limit.get("rows"), list) else []
    for row in frame_rows:
        if not isinstance(row, dict):
            continue
        observable = str(row.get("observable") or "").strip().lower()
        if observable != "frame_dragging":
            continue
        source_id = str(row.get("id") or "").strip()
        experiment = str(row.get("experiment") or "").strip()
        observed = _to_float(row.get("observed"))
        observed_sigma = _to_float(row.get("observed_sigma"))
        value_domain = str(row.get("value_domain") or "").strip().lower()
        reference_prediction = _to_float(row.get("reference_prediction"))
        if observed is None or observed_sigma is None or observed_sigma <= 0.0:
            continue

        lambda_obs = None
        sigma_lambda = None
        mapping_detail = None
        sigma_source = "primary_observed_sigma"
        if value_domain == "ratio_to_gr":
            lambda_obs = float(observed - 1.0)
            sigma_lambda = float(observed_sigma)
            mapping_detail = "lambda = mu_obs - 1 (mu is ratio-to-GR observable)"
        elif reference_prediction is not None and abs(reference_prediction) > 0.0:
            lambda_obs = float((observed / reference_prediction) - 1.0)
            sigma_lambda = float(observed_sigma / abs(reference_prediction))
            mapping_detail = "lambda = obs/reference_prediction - 1"

        if lambda_obs is None or sigma_lambda is None or sigma_lambda <= 0.0 or not np.isfinite(sigma_lambda):
            continue
        _count_sigma_source(sigma_source)
        obs = LambdaObs(
            channel="XDOM_DIR",
            source=f"xdom_dir::frame::{source_id or experiment}",
            label=f"Cross-domain frame-dragging {experiment or source_id}",
            lambda_obs=lambda_obs,
            sigma_lambda=sigma_lambda,
            note="Direct mapped from frame-dragging observable against GR-reference scale.",
        )
        observables.append(obs)
        rows.append(
            {
                "domain": "theory.frame_dragging",
                "source_id": source_id or experiment,
                "observable": observable,
                "value_domain": value_domain,
                "mapping": mapping_detail,
                "observed": observed,
                "observed_sigma": observed_sigma,
                "reference_prediction": reference_prediction,
                "lambda_obs_direct": lambda_obs,
                "sigma_lambda_direct": sigma_lambda,
                "sigma_source": sigma_source,
                "z_direct": float(lambda_obs / sigma_lambda),
            }
        )

    models = cmb_peak_uplift.get("models") if isinstance(cmb_peak_uplift.get("models"), list) else []
    thresholds = cmb_peak_uplift.get("thresholds") if isinstance(cmb_peak_uplift.get("thresholds"), dict) else {}
    amp_sigma_proxy = _to_float(thresholds.get("amplitude_pass"))
    ratio_sigma_proxy = _to_float(thresholds.get("ratio_pass"))
    cmb_primary_input = _resolve_existing_path(cmb_peak_uplift.get("input"))
    cmb_primary_sigma_grid = _load_planck_tt_binned_sigma(cmb_primary_input) if cmb_primary_input is not None else None
    cmb_model: Dict[str, Any] = {}
    for model in models:
        if not isinstance(model, dict):
            continue
        key = str(model.get("key") or "").strip().lower()
        if key == "pressure_ruler":
            cmb_model = model
            break
    if not cmb_model:
        for model in models:
            if isinstance(model, dict) and str(model.get("key") or "").strip().lower() == "pressure":
                cmb_model = model
                break

    peaks = cmb_model.get("peaks") if isinstance(cmb_model.get("peaks"), list) else []
    peak_obs_amp_by_n: Dict[int, float] = {}
    peak_obs_sigma_by_n: Dict[int, float] = {}
    peak_obs_entry_by_n: Dict[int, Dict[str, Any]] = {}
    for peak in peaks:
        if not isinstance(peak, dict):
            continue
        label = str(peak.get("label") or "").strip()
        peak_n = int(_to_float(peak.get("n")) or 0)
        observed_block = peak.get("observed") if isinstance(peak.get("observed"), dict) else {}
        predicted_block = peak.get("predicted") if isinstance(peak.get("predicted"), dict) else {}
        observed_amp = _to_float(observed_block.get("amplitude"))
        predicted_amp = _to_float(predicted_block.get("amplitude"))
        if observed_amp is None or predicted_amp is None or predicted_amp <= 0.0:
            continue
        peak_obs_amp_by_n[peak_n] = float(observed_amp)

        sigma_amp = _to_float(observed_block.get("sigma"))
        sigma_source = "primary_observed_peak_sigma"
        if sigma_amp is None:
            sigma_amp = _to_float(observed_block.get("amplitude_sigma"))
        if sigma_amp is None:
            obs_ell = _to_float(observed_block.get("ell"))
            if obs_ell is not None and cmb_primary_sigma_grid is not None:
                sigma_amp = _nearest_planck_sigma(cmb_primary_sigma_grid[0], cmb_primary_sigma_grid[1], obs_ell)
                if sigma_amp is not None:
                    sigma_source = "primary_planck_binned_sigma"
        sigma_lambda = float(sigma_amp / abs(predicted_amp)) if sigma_amp is not None and sigma_amp > 0.0 else None
        if sigma_lambda is None and amp_sigma_proxy is not None and amp_sigma_proxy > 0.0:
            sigma_lambda = float(amp_sigma_proxy)
            sigma_source = "threshold_proxy_amplitude_pass"
        if sigma_lambda is None:
            continue
        if sigma_amp is not None and sigma_amp > 0.0:
            peak_obs_sigma_by_n[peak_n] = float(sigma_amp)
        _count_sigma_source(sigma_source)
        lambda_obs = float((observed_amp / predicted_amp) - 1.0)
        source_id = f"xdom_dir::cmb::{label or 'peak'}"
        obs = LambdaObs(
            channel="XDOM_DIR",
            source=source_id,
            label=f"Cross-domain CMB {label or 'peak'} amplitude",
            lambda_obs=lambda_obs,
            sigma_lambda=float(sigma_lambda),
            note="Direct mapped from CMB pass-branch peak amplitude mismatch (obs/pred - 1).",
        )
        observables.append(obs)
        rows.append(
            {
                "domain": "cosmology.cmb",
                "source_id": label or "peak",
                "observable": "peak_amplitude",
                "model_key": str(cmb_model.get("key") or ""),
                "mapping": "lambda = obs_amp/pred_amp - 1 ; sigma uses primary amplitude uncertainty (Planck binned/observed) with threshold fallback",
                "observed": observed_amp,
                "predicted": predicted_amp,
                "observed_sigma": sigma_amp,
                "lambda_obs_direct": lambda_obs,
                "sigma_lambda_direct": float(sigma_lambda),
                "sigma_source": sigma_source,
                "z_direct": float(lambda_obs / sigma_lambda),
            }
        )
        peak_obs_entry_by_n[peak_n] = {
            "source": source_id,
            "label": label or f"peak_{peak_n}",
            "obs_amp": float(observed_amp),
            "pred_amp": float(predicted_amp),
            "obs_sigma": float(sigma_amp) if sigma_amp is not None and sigma_amp > 0.0 else None,
            "sigma_source": sigma_source,
        }

    ratios = cmb_model.get("ratios") if isinstance(cmb_model.get("ratios"), dict) else {}
    ratio_obs = _to_float(ratios.get("a3_a1_obs"))
    ratio_pred = _to_float(ratios.get("a3_a1_pred"))
    ratio_sigma_obs = _to_float(ratios.get("a3_a1_obs_sigma"))
    ratio_sigma_source = "primary_observed_ratio_sigma"
    ratio_source: Optional[str] = None
    if ratio_sigma_obs is None:
        a1_obs = peak_obs_amp_by_n.get(1)
        a3_obs = peak_obs_amp_by_n.get(3)
        s1_obs = peak_obs_sigma_by_n.get(1)
        s3_obs = peak_obs_sigma_by_n.get(3)
        if (
            a1_obs is not None
            and a3_obs is not None
            and s1_obs is not None
            and s3_obs is not None
            and a1_obs > 0.0
            and a3_obs > 0.0
            and ratio_obs is not None
        ):
            ratio_sigma_obs = abs(ratio_obs) * math.sqrt((s3_obs / a3_obs) ** 2 + (s1_obs / a1_obs) ** 2)
            ratio_sigma_source = "primary_propagated_from_peak_sigmas"
    if ratio_sigma_obs is None and ratio_sigma_proxy is not None and ratio_sigma_proxy > 0.0:
        ratio_sigma_obs = float(ratio_sigma_proxy)
        ratio_sigma_source = "threshold_proxy_ratio_pass"
    if ratio_obs is not None and ratio_pred is not None and ratio_pred > 0.0 and ratio_sigma_obs is not None and ratio_sigma_obs > 0.0:
        lambda_obs = float((ratio_obs / ratio_pred) - 1.0)
        sigma_lambda = float(ratio_sigma_obs / abs(ratio_pred))
        _count_sigma_source(ratio_sigma_source)
        ratio_source = "xdom_dir::cmb::a3a1"
        obs = LambdaObs(
            channel="XDOM_DIR",
            source=ratio_source,
            label="Cross-domain CMB A3/A1 ratio",
            lambda_obs=lambda_obs,
            sigma_lambda=sigma_lambda,
            note="Direct mapped from CMB A3/A1 pass-branch ratio mismatch (obs/pred - 1).",
        )
        observables.append(obs)
        rows.append(
            {
                "domain": "cosmology.cmb",
                "source_id": "a3_a1",
                "observable": "peak_ratio_a3_a1",
                "model_key": str(cmb_model.get("key") or ""),
                "mapping": "lambda = ratio_obs/ratio_pred - 1 ; sigma uses primary ratio uncertainty (observed or propagated from peak sigmas) with threshold fallback",
                "observed": ratio_obs,
                "predicted": ratio_pred,
                "observed_sigma": ratio_sigma_obs,
                "lambda_obs_direct": lambda_obs,
                "sigma_lambda_direct": sigma_lambda,
                "sigma_source": ratio_sigma_source,
                "z_direct": float(lambda_obs / sigma_lambda),
            }
        )

    cmb_covariance_summary: Dict[str, Any] = {
        "enabled": False,
        "pair_count": 0,
        "method": "delta-method from shared peak amplitudes (A1, A3) -> lambda covariance for {l1,l3,a3a1}",
        "assumptions": [
            "CMB peak observables use explicit amplitude-level mapping from pass branch.",
            "If external a3_a1 sigma exists, infer Cov(A1,A3) and clamp to physical |rho|<=1.",
            "If ratio sigma is propagated from peak sigmas, Cov(A1,A3)=0 is used.",
        ],
        "inferred_amp_cov13": None,
        "inferred_amp_corr13": None,
        "ratio_sigma_source": ratio_sigma_source,
    }
    peak1 = peak_obs_entry_by_n.get(1)
    peak3 = peak_obs_entry_by_n.get(3)
    if (
        ratio_source is not None
        and ratio_obs is not None
        and ratio_pred is not None
        and ratio_pred > 0.0
        and peak1 is not None
        and peak3 is not None
    ):
        a1 = _to_float(peak1.get("obs_amp"))
        p1 = _to_float(peak1.get("pred_amp"))
        s1 = _to_float(peak1.get("obs_sigma"))
        a3 = _to_float(peak3.get("obs_amp"))
        p3 = _to_float(peak3.get("pred_amp"))
        s3 = _to_float(peak3.get("obs_sigma"))
        cov13 = 0.0
        cov13_mode = "assumed_zero"
        if (
            ratio_sigma_source == "primary_observed_ratio_sigma"
            and ratio_sigma_obs is not None
            and a1 is not None
            and s1 is not None
            and s3 is not None
            and a1 > 0.0
            and abs(float(ratio_obs)) > 0.0
        ):
            var_r = float(ratio_sigma_obs) * float(ratio_sigma_obs)
            numerator = float((s3 * s3) + (ratio_obs * ratio_obs * s1 * s1) - (var_r * a1 * a1))
            denominator = float(2.0 * ratio_obs)
            if abs(denominator) > 0.0:
                cov13_candidate = numerator / denominator
                bound = abs(float(s1 * s3))
                cov13 = float(max(-bound, min(bound, cov13_candidate)))
                cov13_mode = "inferred_from_ratio_sigma"
        cmb_covariance_summary["inferred_amp_cov13"] = float(cov13)
        if s1 is not None and s3 is not None and s1 > 0.0 and s3 > 0.0:
            cmb_covariance_summary["inferred_amp_corr13"] = float(cov13 / (s1 * s3))
        cmb_covariance_summary["amp_cov13_mode"] = cov13_mode

        if (
            a1 is not None
            and a1 > 0.0
            and p1 is not None
            and p1 > 0.0
            and p3 is not None
            and p3 > 0.0
            and s1 is not None
            and s1 > 0.0
            and s3 is not None
            and s3 > 0.0
        ):
            cov_l1_l3 = float(cov13 / (p1 * p3))
            cov_a1_ratio = float(((-ratio_obs / a1) * (s1 * s1)) + ((1.0 / a1) * cov13))
            cov_a3_ratio = float(((-ratio_obs / a1) * cov13) + ((1.0 / a1) * (s3 * s3)))
            cov_l1_ratio = float(cov_a1_ratio / (p1 * ratio_pred))
            cov_l3_ratio = float(cov_a3_ratio / (p3 * ratio_pred))

            _add_covariance_pair(str(peak1.get("source") or ""), str(peak3.get("source") or ""), cov_l1_l3, "cmb_amp_cov13")
            _add_covariance_pair(str(peak1.get("source") or ""), ratio_source, cov_l1_ratio, "cmb_ratio_coupling")
            _add_covariance_pair(str(peak3.get("source") or ""), ratio_source, cov_l3_ratio, "cmb_ratio_coupling")

    cmb_covariance_summary["enabled"] = len(covariance_pairs) > 0
    cmb_covariance_summary["pair_count"] = len(covariance_pairs)

    cluster_rows = cluster_collision_offset.get("cluster_rows")
    if isinstance(cluster_rows, list):
        for row in cluster_rows:
            if not isinstance(row, dict):
                continue
            model_name = str(row.get("model") or "").strip().lower()
            if model_name != "pmodel_corrected":
                continue
            cluster_id = str(row.get("cluster_id") or "").strip()
            observed = _to_float(row.get("obs_lens_gas_offset_kpc"))
            observed_sigma = _to_float(row.get("obs_lens_gas_sigma_kpc"))
            predicted = _to_float(row.get("pred_delta_x_p_gas_kpc"))
            if (
                not cluster_id
                or observed is None
                or observed_sigma is None
                or predicted is None
                or observed_sigma <= 0.0
                or abs(predicted) <= 0.0
            ):
                continue
            lambda_obs = float((observed / predicted) - 1.0)
            sigma_lambda = float(observed_sigma / abs(predicted))
            if sigma_lambda <= 0.0 or not np.isfinite(sigma_lambda):
                continue
            sigma_source = "primary_observed_sigma"
            _count_sigma_source(sigma_source)
            obs = LambdaObs(
                channel="XDOM_DIR",
                source=f"xdom_dir::cluster::{cluster_id}",
                label=f"Cross-domain cluster offset {cluster_id}",
                lambda_obs=lambda_obs,
                sigma_lambda=sigma_lambda,
                note="Direct mapped from cluster offset pass-branch (obs_lens_gas_offset/pred_delta_x_p_gas - 1).",
            )
            observables.append(obs)
            rows.append(
                {
                    "domain": "cosmology.cluster_collision",
                    "source_id": cluster_id,
                    "observable": "lens_gas_offset",
                    "mapping": "lambda = obs_lens_gas_offset/pred_delta_x_p_gas - 1",
                    "observed": observed,
                    "predicted": predicted,
                    "observed_sigma": observed_sigma,
                    "lambda_obs_direct": lambda_obs,
                    "sigma_lambda_direct": sigma_lambda,
                    "sigma_source": sigma_source,
                    "z_direct": float(lambda_obs / sigma_lambda),
                }
            )

    return {
        "ok": len(observables) > 0,
        "n_observables": len(observables),
        "channels_present": sorted(set(o.channel for o in observables)),
        "rows": rows,
        "observables": observables,
        "mapping_policy": {
            "frame_dragging": "absolute: obs/reference-1 ; ratio_to_gr: mu-1",
            "cmb": "pass-branch pressure(+ruler) amplitudes/ratio mapped as obs/pred-1 with primary sigma (Planck binned or observed), threshold proxy only fallback",
            "cluster": "pmodel-corrected offset branch mapped as obs/pred-1 with sigma=obs_sigma/pred",
        },
        "covariance_pairs": covariance_pairs,
        "covariance_summary": cmb_covariance_summary,
        "sigma_source_counts": sigma_source_counts,
        "cmb_primary_sigma_input": _rel(cmb_primary_input) if cmb_primary_input is not None else None,
        "cmb_primary_sigma_available": bool(cmb_primary_sigma_grid is not None),
        "note": (
            "Decision-grade direct mapping candidate set. Unlike proxy z injection, each row is represented as "
            "(lambda_obs, sigma_lambda) from explicit observable-level formulas; CMB off-diagonal covariance is "
            "added for shared A1/A3-derived ratio coupling."
        ),
    }


def _extract_pulsar_observables(payload: Dict[str, Any]) -> List[LambdaObs]:
    rows = payload.get("metrics")
    if not isinstance(rows, list):
        return []
    out: List[LambdaObs] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        source_id = str(row.get("id", "")).strip()
        label = str(row.get("name", "")).strip() or source_id
        delta = _to_float(row.get("delta"))
        sigma = _to_float(row.get("sigma_1"))
        if not source_id or delta is None or sigma is None or sigma <= 0.0:
            continue
        out.append(
            LambdaObs(
                channel="PULSAR",
                source=source_id,
                label=f"Pulsar {label}",
                lambda_obs=float(delta),
                sigma_lambda=float(sigma),
                note="λ_H proxy from binary-pulsar orbital-decay ratio R-1 (intrinsic Pbdot corrected).",
            )
        )
    return out


def _extract_xray_isco_observables(rows: Sequence[Dict[str, Any]]) -> List[LambdaObs]:
    out: List[LambdaObs] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        constrained = str(row.get("proxy_isco_constrained", "")).strip()
        if constrained != "1":
            continue
        if str(row.get("r_in_bound", "")).strip():
            continue
        rin = _to_float(row.get("r_in_rg"))
        sig_tot = _to_float(row.get("sigma_total_rg"))
        if rin is None or sig_tot is None or rin <= 0.0 or sig_tot <= 0.0:
            continue
        source = str(row.get("obsid", "")).strip() or str(row.get("target_name", "")).strip()
        mission = str(row.get("mission", "")).strip().lower() or "xray"
        if not source:
            continue
        lam = (rin / 6.0) - 1.0
        sig_lam = sig_tot / 6.0
        if sig_lam <= 0.0:
            continue
        out.append(
            LambdaObs(
                channel="XRAY",
                source=source,
                label=f"Fe-Kα {mission.upper()} {source}",
                lambda_obs=float(lam),
                sigma_lambda=float(sig_lam),
                note="λ_H proxy from Fe-Kα broad-line inner-radius vs GR ISCO baseline (proxy fit, systematics folded).",
            )
        )
    return out


def _extract_eht_kappa_precision(payload: Dict[str, Any]) -> Dict[str, Any]:
    rows = payload.get("rows") if isinstance(payload.get("rows"), dict) else {}
    row = rows.get("sgra") if isinstance(rows.get("sgra"), dict) else {}

    required = _to_float(row.get("kappa_sigma_required_3sigma_if_ring_sigma_zero"))
    conservative = _to_float(row.get("kappa_sigma_adopted_for_falsification"))

    compressed_candidates: List[Tuple[str, float]] = []
    for key in (
        "kappa_sigma_method_scatter_median",
        "kappa_sigma_from_ring_published",
        "kappa_sigma_assumed_kerr",
    ):
        v = _to_float(row.get(key))
        if v is not None and np.isfinite(v) and v > 0.0:
            compressed_candidates.append((key, float(v)))

    if compressed_candidates:
        mode = "max(method_scatter_median, ring_published, kerr_floor)"
        compressed = float(max(v for _, v in compressed_candidates))
    else:
        mode = "fallback_to_conservative"
        compressed = float(conservative) if conservative is not None else float("nan")

    ratio_conservative = float("nan")
    ratio_compressed = float("nan")
    if required is not None and required > 0.0:
        if conservative is not None and conservative > 0.0:
            ratio_conservative = float(conservative / required)
        if np.isfinite(compressed) and compressed > 0.0:
            ratio_compressed = float(compressed / required)

    compression_factor = float("nan")
    if conservative is not None and conservative > 0.0 and np.isfinite(compressed) and compressed > 0.0:
        compression_factor = float(conservative / compressed)

    return {
        "required_sigma": required,
        "conservative_sigma": conservative,
        "compressed_sigma": compressed,
        "compressed_mode": mode,
        "compressed_candidates": [{"key": k, "sigma": v} for k, v in compressed_candidates],
        "ratio_conservative_over_required": ratio_conservative,
        "ratio_compressed_over_required": ratio_compressed,
        "compression_factor_conservative_over_compressed": compression_factor,
    }


def _derive_eht_kappa_readiness(eht_kappa_precision: Dict[str, Any]) -> Dict[str, Any]:
    required = _to_float(eht_kappa_precision.get("required_sigma"))
    conservative = _to_float(eht_kappa_precision.get("conservative_sigma"))
    compressed = _to_float(eht_kappa_precision.get("compressed_sigma"))

    ratio_current = _to_float(eht_kappa_precision.get("ratio_compressed_over_required"))
    ratio_excess = float("nan")
    sigma_gap = float("nan")
    sigma_reduction_fraction = float("nan")
    sigma_reduction_percent = float("nan")
    sigma_from_conservative_gain = float("nan")
    info_multiplier_if_sigma_inv_sqrt_n = float("nan")
    gate_ready = False

    if required is not None and compressed is not None and required > 0.0 and compressed > 0.0:
        ratio_current = compressed / required
        ratio_excess = ratio_current - 1.0
        sigma_gap = compressed - required
        sigma_reduction_fraction = sigma_gap / compressed
        sigma_reduction_percent = 100.0 * sigma_reduction_fraction
        info_multiplier_if_sigma_inv_sqrt_n = ratio_current * ratio_current
        gate_ready = ratio_current <= 1.0

    if conservative is not None and compressed is not None and conservative > 0.0 and compressed > 0.0:
        sigma_from_conservative_gain = (conservative - compressed) / conservative

    return {
        "gate_target_ratio": 1.0,
        "ratio_current": ratio_current,
        "ratio_excess_over_gate": ratio_excess,
        "sigma_target_for_gate": required,
        "sigma_current_compressed": compressed,
        "sigma_current_conservative": conservative,
        "sigma_gap_to_gate": sigma_gap,
        "sigma_reduction_fraction_needed": sigma_reduction_fraction,
        "sigma_reduction_percent_needed": sigma_reduction_percent,
        "compression_gain_fraction_from_conservative": sigma_from_conservative_gain,
        "independent_info_multiplier_if_sigma_inv_sqrt_n": info_multiplier_if_sigma_inv_sqrt_n,
        "gate_ready_now": gate_ready,
        "note": (
            "Approximation uses sigma ~ 1/sqrt(N_indep). "
            "Required information multiplier is indicative for planning, not a strict experimental forecast."
        ),
    }


def _derive_eht_kappa_update_watchpack(
    *,
    eht_kappa_readiness: Dict[str, Any],
    current_input_signature: Dict[str, Any],
    previous_watchpack: Dict[str, Any],
) -> Dict[str, Any]:
    ratio_current = _to_float(eht_kappa_readiness.get("ratio_current"))
    ratio_previous = _to_float(previous_watchpack.get("ratio_current"))
    ratio_delta_vs_previous = float("nan")
    if ratio_current is not None and ratio_previous is not None:
        ratio_delta_vs_previous = float(ratio_current - ratio_previous)

    previous_signature = (
        previous_watchpack.get("input_signature")
        if isinstance(previous_watchpack.get("input_signature"), dict)
        else {}
    )
    curr_sha = str(current_input_signature.get("sha256", "")).strip().upper()
    prev_sha = str(previous_signature.get("sha256", "")).strip().upper()
    curr_exists = bool(current_input_signature.get("exists"))
    prev_exists = bool(previous_signature.get("exists"))
    hash_changed = curr_exists and prev_exists and bool(curr_sha) and bool(prev_sha) and curr_sha != prev_sha

    curr_mtime = str(current_input_signature.get("mtime_utc", "")).strip()
    prev_mtime = str(previous_signature.get("mtime_utc", "")).strip()
    curr_size = _to_float(current_input_signature.get("size_bytes"))
    prev_size = _to_float(previous_signature.get("size_bytes"))
    metadata_changed_without_hash_change = False
    if curr_exists and prev_exists and not hash_changed:
        if (curr_mtime and prev_mtime and curr_mtime != prev_mtime) or (
            curr_size is not None and prev_size is not None and curr_size != prev_size
        ):
            metadata_changed_without_hash_change = True

    baseline_initialized_now = curr_exists and not prev_exists
    update_event_detected = hash_changed
    if baseline_initialized_now:
        update_event_type = "baseline_initialized"
    elif hash_changed:
        update_event_type = "input_hash_changed"
    elif metadata_changed_without_hash_change:
        update_event_type = "metadata_changed_hash_same"
    else:
        update_event_type = "no_change"

    event_counter_prev = int(previous_watchpack.get("event_counter", 0)) if previous_watchpack else 0
    event_counter = event_counter_prev + 1 if update_event_detected else event_counter_prev

    gate_ready_now = bool(eht_kappa_readiness.get("gate_ready_now"))
    next_action = (
        "run_support_recheck_now"
        if gate_ready_now
        else "wait_for_eht_primary_precision_update_then_rerun_step_8_7_27"
    )

    return {
        "gate_target_ratio": eht_kappa_readiness.get("gate_target_ratio"),
        "ratio_current": ratio_current,
        "ratio_previous": ratio_previous,
        "ratio_delta_vs_previous": ratio_delta_vs_previous,
        "gate_ready_now": gate_ready_now,
        "sigma_gap_to_gate": eht_kappa_readiness.get("sigma_gap_to_gate"),
        "sigma_reduction_percent_needed": eht_kappa_readiness.get("sigma_reduction_percent_needed"),
        "info_multiplier_if_sigma_inv_sqrt_n": eht_kappa_readiness.get("independent_info_multiplier_if_sigma_inv_sqrt_n"),
        "input_signature": current_input_signature,
        "previous_input_signature": previous_signature,
        "update_event_detected": update_event_detected,
        "update_event_type": update_event_type,
        "input_hash_changed": hash_changed,
        "input_metadata_changed_without_hash_change": metadata_changed_without_hash_change,
        "baseline_initialized_now": baseline_initialized_now,
        "event_counter": event_counter,
        "next_action": next_action,
        "note": (
            "kappa_ratio<=1 gate is rerun-triggered by primary-input hash updates. "
            "metadata-only changes are logged but do not increment the event counter."
        ),
    }


def _weighted_fit(observables: Sequence[LambdaObs]) -> Dict[str, Any]:
    if not observables:
        return {
            "ok": False,
            "n_obs": 0,
            "lambda_fit": float("nan"),
            "lambda_sigma": float("nan"),
            "chi2_baseline": float("nan"),
            "chi2_fit": float("nan"),
            "chi2_dof_baseline": float("nan"),
            "chi2_dof_fit": float("nan"),
            "improvement_ratio": float("nan"),
            "delta_aic_fit_minus_baseline": float("nan"),
        }

    y = np.asarray([o.lambda_obs for o in observables], dtype=float)
    s = np.asarray([o.sigma_lambda for o in observables], dtype=float)
    w = 1.0 / np.square(s)

    denom = float(np.sum(w))
    if denom <= 0.0:
        return {
            "ok": False,
            "n_obs": int(y.size),
            "lambda_fit": float("nan"),
            "lambda_sigma": float("nan"),
            "chi2_baseline": float("nan"),
            "chi2_fit": float("nan"),
            "chi2_dof_baseline": float("nan"),
            "chi2_dof_fit": float("nan"),
            "improvement_ratio": float("nan"),
            "delta_aic_fit_minus_baseline": float("nan"),
        }

    lam = float(np.sum(w * y) / denom)
    sig = float(math.sqrt(1.0 / denom))

    chi2_baseline = float(np.sum(np.square(y / s)))
    chi2_fit = float(np.sum(np.square((y - lam) / s)))

    n = int(y.size)
    dof_base = max(1, n)  # k=0
    dof_fit = max(1, n - 1)  # k=1
    chi2_dof_base = chi2_baseline / float(dof_base)
    chi2_dof_fit = chi2_fit / float(dof_fit)
    improvement = chi2_fit / max(chi2_baseline, 1.0e-12)

    aic_base = chi2_baseline + 2.0 * 0.0
    aic_fit = chi2_fit + 2.0 * 1.0
    delta_aic = float(aic_fit - aic_base)

    return {
        "ok": True,
        "n_obs": n,
        "lambda_fit": lam,
        "lambda_sigma": sig,
        "chi2_baseline": chi2_baseline,
        "chi2_fit": chi2_fit,
        "chi2_dof_baseline": chi2_dof_base,
        "chi2_dof_fit": chi2_dof_fit,
        "improvement_ratio": float(improvement),
        "delta_aic_fit_minus_baseline": delta_aic,
    }


def _weighted_fit_with_covariance(
    observables: Sequence[LambdaObs],
    *,
    covariance_pairs: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    base = _weighted_fit(observables)
    if not observables:
        return base

    n = len(observables)
    y = np.asarray([o.lambda_obs for o in observables], dtype=float)
    s = np.asarray([o.sigma_lambda for o in observables], dtype=float)
    cov = np.diag(np.square(s))
    index_by_source: Dict[str, int] = {}
    for idx, row in enumerate(observables):
        source = str(row.source)
        if source and source not in index_by_source:
            index_by_source[source] = idx

    applied_pairs: List[Dict[str, Any]] = []
    requested_pairs = 0
    if covariance_pairs is not None:
        for pair in covariance_pairs:
            if not isinstance(pair, dict):
                continue
            requested_pairs += 1
            src_i = str(pair.get("source_i") or "").strip()
            src_j = str(pair.get("source_j") or "").strip()
            cov_ij = _to_float(pair.get("cov_lambda"))
            if not src_i or not src_j or src_i == src_j or cov_ij is None:
                continue
            i = index_by_source.get(src_i)
            j = index_by_source.get(src_j)
            if i is None or j is None:
                continue
            cov[i, j] = float(cov_ij)
            cov[j, i] = float(cov_ij)
            applied_pairs.append(
                {
                    "source_i": src_i,
                    "source_j": src_j,
                    "cov_lambda": float(cov_ij),
                    "origin": str(pair.get("origin") or "").strip(),
                }
            )

    cov = 0.5 * (cov + cov.T)
    min_eig_before = float("nan")
    reg_jitter = 0.0
    diag_scale = float(np.max(np.diag(cov))) if cov.size > 0 else 0.0
    singular_eps = max(1.0e-18, diag_scale * 1.0e-10)
    prefer_pinv = False
    try:
        eigvals = np.linalg.eigvalsh(cov)
        min_eig_before = float(np.min(eigvals))
    except Exception:
        eigvals = np.asarray([], dtype=float)
    if np.isfinite(min_eig_before):
        if min_eig_before < -singular_eps:
            reg_jitter = float(abs(min_eig_before) + singular_eps)
            cov = cov + np.eye(n, dtype=float) * reg_jitter
        elif min_eig_before <= singular_eps:
            prefer_pinv = True

    try:
        cond_number_pre = float(np.linalg.cond(cov))
    except Exception:
        cond_number_pre = float("nan")
    if np.isfinite(cond_number_pre) and cond_number_pre > 1.0e10:
        prefer_pinv = True

    if prefer_pinv:
        cov_inv = np.linalg.pinv(cov, rcond=1.0e-12)
        solver = "pinv_singular"
    else:
        try:
            cov_inv = np.linalg.inv(cov)
            solver = "inv"
        except np.linalg.LinAlgError:
            cov_inv = np.linalg.pinv(cov, rcond=1.0e-12)
            solver = "pinv"

    ones = np.ones(n, dtype=float)
    denom = float(ones.T @ cov_inv @ ones)
    if not np.isfinite(denom) or denom <= 0.0:
        out = dict(base)
        out.update(
            {
                "ok": False,
                "covariance_mode": "full_covariance_failed",
                "covariance_solver": solver,
                "covariance_pairs_requested": int(requested_pairs),
                "covariance_pairs_applied": int(len(applied_pairs)),
                "covariance_applied_rows": applied_pairs,
                "covariance_regularization_jitter": reg_jitter,
                "covariance_min_eigenvalue_before_regularization": min_eig_before,
                "note": "covariance fit denominator <=0; fallback to diagonal metrics only.",
            }
        )
        return out

    lam = float((ones.T @ cov_inv @ y) / denom)
    sig = float(math.sqrt(1.0 / denom))
    residual_fit = y - lam
    chi2_baseline = float(y.T @ cov_inv @ y)
    chi2_fit = float(residual_fit.T @ cov_inv @ residual_fit)
    dof_base = max(1, n)
    dof_fit = max(1, n - 1)
    chi2_dof_base = chi2_baseline / float(dof_base)
    chi2_dof_fit = chi2_fit / float(dof_fit)
    improvement = chi2_fit / max(chi2_baseline, 1.0e-12)
    delta_aic = float((chi2_fit + 2.0 * 1.0) - chi2_baseline)
    cond_number = float(np.linalg.cond(cov)) if cov.size > 0 else float("nan")

    return {
        "ok": True,
        "n_obs": n,
        "lambda_fit": lam,
        "lambda_sigma": sig,
        "chi2_baseline": chi2_baseline,
        "chi2_fit": chi2_fit,
        "chi2_dof_baseline": chi2_dof_base,
        "chi2_dof_fit": chi2_dof_fit,
        "improvement_ratio": float(improvement),
        "delta_aic_fit_minus_baseline": delta_aic,
        "covariance_mode": "full_covariance" if applied_pairs else "diagonal_only",
        "covariance_solver": solver,
        "covariance_pairs_requested": int(requested_pairs),
        "covariance_pairs_applied": int(len(applied_pairs)),
        "covariance_applied_rows": applied_pairs,
        "covariance_regularization_jitter": reg_jitter,
        "covariance_min_eigenvalue_before_regularization": min_eig_before,
        "covariance_singular_threshold": singular_eps,
        "covariance_condition_number": cond_number,
    }


def _build_design_matrix(observables: Sequence[LambdaObs], model_id: str) -> Tuple[List[str], np.ndarray]:
    channels = [str(o.channel).upper() for o in observables]
    n = len(channels)
    one = np.ones(n, dtype=float)
    is_eht = np.asarray([1.0 if ch == "EHT" else 0.0 for ch in channels], dtype=float)
    is_gw = np.asarray([1.0 if ch == "GW" else 0.0 for ch in channels], dtype=float)
    is_pulsar = np.asarray([1.0 if ch == "PULSAR" else 0.0 for ch in channels], dtype=float)
    is_xray = np.asarray([1.0 if ch == "XRAY" else 0.0 for ch in channels], dtype=float)
    is_imaging = np.asarray([1.0 if ch in {"EHT", "XRAY"} else 0.0 for ch in channels], dtype=float)

    if model_id == "shared_lambda":
        return ["lambda0"], one[:, None]
    if model_id == "split_eht_offset":
        return ["lambda0", "delta_eht"], np.column_stack([one, is_eht])
    if model_id == "split_gw_offset":
        return ["lambda0", "delta_gw"], np.column_stack([one, is_gw])
    if model_id == "split_imaging_vs_wave":
        return ["lambda0", "delta_imaging"], np.column_stack([one, is_imaging])
    if model_id == "channel_offsets_4":
        return ["lambda_eht", "lambda_gw", "lambda_pulsar", "lambda_xray"], np.column_stack([is_eht, is_gw, is_pulsar, is_xray])
    raise ValueError(f"Unknown model_id: {model_id}")


def _weighted_linear_fit(
    observables: Sequence[LambdaObs],
    *,
    model_id: str,
    model_label: str,
    param_names: Sequence[str],
    X: np.ndarray,
) -> Dict[str, Any]:
    if not observables:
        return {"ok": False, "model_id": model_id, "model_label": model_label, "n_obs": 0, "k_params": int(X.shape[1])}
    y = np.asarray([o.lambda_obs for o in observables], dtype=float)
    s = np.asarray([o.sigma_lambda for o in observables], dtype=float)
    n = int(y.size)
    k = int(X.shape[1])
    if X.shape[0] != n or k <= 0:
        return {"ok": False, "model_id": model_id, "model_label": model_label, "n_obs": n, "k_params": k}

    w = 1.0 / np.square(s)
    xtwx = X.T @ (w[:, None] * X)
    xtwy = X.T @ (w * y)
    try:
        cov = np.linalg.inv(xtwx)
        solver = "inv"
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(xtwx)
        solver = "pinv"

    theta = cov @ xtwy
    y_hat = X @ theta
    residual = y - y_hat
    chi2 = float(np.sum(np.square(residual / s)))
    dof = max(1, n - k)
    chi2_dof = chi2 / float(dof)
    aic = chi2 + 2.0 * float(k)
    bic = chi2 + float(k) * math.log(float(max(2, n)))

    diag = np.diag(cov) if cov.ndim == 2 else np.full(k, float("nan"))
    param_sigma = [float(math.sqrt(max(0.0, float(v)))) if np.isfinite(v) else float("nan") for v in diag]
    params = {str(name): float(val) for name, val in zip(param_names, theta.tolist())}
    params_sigma = {str(name): float(val) for name, val in zip(param_names, param_sigma)}

    return {
        "ok": True,
        "model_id": model_id,
        "model_label": model_label,
        "solver": solver,
        "n_obs": n,
        "k_params": k,
        "design_rank": int(np.linalg.matrix_rank(X)),
        "param_names": list(param_names),
        "params": params,
        "params_sigma": params_sigma,
        "chi2": chi2,
        "chi2_dof": chi2_dof,
        "aic": float(aic),
        "bic": float(bic),
    }


def _ansatz_extension_model_matrix(observables: Sequence[LambdaObs], fit_joint: Dict[str, Any]) -> Dict[str, Any]:
    chi2_baseline = _to_float(fit_joint.get("chi2_baseline"))
    chi2_shared = _to_float(fit_joint.get("chi2_fit"))
    if chi2_baseline is None or chi2_shared is None or not np.isfinite(chi2_baseline) or not np.isfinite(chi2_shared):
        return {
            "ok": False,
            "note": "baseline/shared chi2 is unavailable; ansatz extension matrix skipped.",
            "rows": [],
        }

    aic_baseline = float(chi2_baseline)
    aic_shared = float(chi2_shared + 2.0)

    definitions: List[Tuple[str, str]] = [
        ("shared_lambda", "single shared lambda_H"),
        ("split_eht_offset", "shared lambda_H + EHT offset"),
        ("split_gw_offset", "shared lambda_H + GW offset"),
        ("split_imaging_vs_wave", "shared lambda_H + (EHT/XRAY) imaging offset"),
        ("channel_offsets_4", "independent lambda per channel (EHT/GW/PULSAR/XRAY)"),
    ]

    rows: List[Dict[str, Any]] = []
    for model_id, model_label in definitions:
        names, X = _build_design_matrix(observables, model_id)
        fit = _weighted_linear_fit(
            observables,
            model_id=model_id,
            model_label=model_label,
            param_names=names,
            X=X,
        )
        if not bool(fit.get("ok")):
            rows.append(
                {
                    "model_id": model_id,
                    "model_label": model_label,
                    "ok": False,
                    "n_obs": fit.get("n_obs"),
                    "k_params": fit.get("k_params"),
                    "note": "fit failed",
                }
            )
            continue

        k = int(fit.get("k_params", 0))
        chi2_fit = _to_float(fit.get("chi2"))
        aic_fit = _to_float(fit.get("aic"))
        if chi2_fit is None or aic_fit is None:
            continue
        delta_aic_vs_baseline = float(aic_fit - aic_baseline)
        delta_aic_vs_shared = float(aic_fit - aic_shared)
        delta_aic_theoretical_min = float(2.0 * k - chi2_baseline)
        required_chi2_for_support = float(chi2_baseline - 2.0 * k - 2.0)
        support_possible_if_perfect = bool(required_chi2_for_support >= 0.0)

        rows.append(
            {
                "model_id": model_id,
                "model_label": model_label,
                "ok": True,
                "n_obs": int(fit.get("n_obs", 0)),
                "k_params": k,
                "chi2": chi2_fit,
                "chi2_dof": fit.get("chi2_dof"),
                "aic": aic_fit,
                "bic": fit.get("bic"),
                "delta_aic_vs_baseline": delta_aic_vs_baseline,
                "delta_aic_vs_shared": delta_aic_vs_shared,
                "aic_support_pass": bool(delta_aic_vs_baseline <= -2.0),
                "delta_aic_theoretical_min_if_chi2_zero": delta_aic_theoretical_min,
                "required_chi2_for_support_threshold": required_chi2_for_support,
                "support_possible_if_perfect_fit": support_possible_if_perfect,
                "params": fit.get("params"),
                "params_sigma": fit.get("params_sigma"),
            }
        )

    valid = [r for r in rows if bool(r.get("ok"))]
    if not valid:
        return {
            "ok": False,
            "baseline_chi2": chi2_baseline,
            "baseline_aic": aic_baseline,
            "shared_aic": aic_shared,
            "rows": rows,
            "note": "No valid ansatz extension fits were produced.",
        }

    best_by_aic = min(valid, key=lambda r: float(r.get("aic", float("inf"))))
    best_by_delta = min(valid, key=lambda r: float(r.get("delta_aic_vs_baseline", float("inf"))))
    return {
        "ok": True,
        "baseline_chi2": chi2_baseline,
        "baseline_aic": aic_baseline,
        "shared_aic": aic_shared,
        "aic_support_threshold": -2.0,
        "best_model_id_by_aic": best_by_aic.get("model_id"),
        "best_model_id_by_delta_aic": best_by_delta.get("model_id"),
        "best_delta_aic_vs_baseline": best_by_delta.get("delta_aic_vs_baseline"),
        "support_passed_by_any_model": any(bool(r.get("aic_support_pass")) for r in valid),
        "support_possible_under_current_data_if_perfect_fit": any(bool(r.get("support_possible_if_perfect_fit")) for r in valid),
        "rows": rows,
        "note": (
            "AIC support threshold is strict (<=-2 vs baseline k=0). "
            "If required_chi2_for_support_threshold < 0, no perfect fit can clear support for that parameter count."
        ),
    }


def _aic_support_scenario_matrix(
    *,
    obs_eht: Sequence[LambdaObs],
    obs_gw: Sequence[LambdaObs],
    obs_pulsar: Sequence[LambdaObs],
    obs_xray: Sequence[LambdaObs],
) -> Dict[str, Any]:
    pools: Dict[str, List[LambdaObs]] = {
        "EHT": list(obs_eht),
        "GW": list(obs_gw),
        "PULSAR": list(obs_pulsar),
        "XRAY": list(obs_xray),
    }
    definitions: List[Tuple[str, Tuple[str, ...]]] = [
        ("eht_only", ("EHT",)),
        ("gw_only", ("GW",)),
        ("pulsar_only", ("PULSAR",)),
        ("xray_only", ("XRAY",)),
        ("eht_gw", ("EHT", "GW")),
        ("eht_gw_pulsar", ("EHT", "GW", "PULSAR")),
        ("eht_gw_xray", ("EHT", "GW", "XRAY")),
        ("all_channels", ("EHT", "GW", "PULSAR", "XRAY")),
    ]

    rows: List[Dict[str, Any]] = []
    for scenario_id, channels in definitions:
        obs: List[LambdaObs] = []
        for channel in channels:
            obs.extend(pools.get(channel, []))
        fit = _weighted_fit(obs)
        delta_aic = _to_float(fit.get("delta_aic_fit_minus_baseline"))
        rows.append(
            {
                "scenario_id": scenario_id,
                "channels": list(channels),
                "n_obs": int(fit.get("n_obs", 0)),
                "lambda_fit": fit.get("lambda_fit"),
                "lambda_sigma": fit.get("lambda_sigma"),
                "chi2_dof_baseline": fit.get("chi2_dof_baseline"),
                "chi2_dof_fit": fit.get("chi2_dof_fit"),
                "delta_aic_fit_minus_baseline": delta_aic,
                "aic_support_pass": bool(delta_aic is not None and delta_aic <= -2.0),
            }
        )

    valid = [r for r in rows if _to_float(r.get("delta_aic_fit_minus_baseline")) is not None]
    if not valid:
        return {
            "ok": False,
            "required_threshold": -2.0,
            "support_possible_under_current_channels": False,
            "best_scenario_id": None,
            "best_delta_aic": None,
            "rows": rows,
        }

    best = min(valid, key=lambda r: float(r["delta_aic_fit_minus_baseline"]))
    support_possible = any(bool(r.get("aic_support_pass")) for r in valid)
    return {
        "ok": True,
        "required_threshold": -2.0,
        "support_possible_under_current_channels": bool(support_possible),
        "best_scenario_id": best.get("scenario_id"),
        "best_delta_aic": best.get("delta_aic_fit_minus_baseline"),
        "rows": rows,
        "note": "Current one-parameter lambda_H ansatz with fixed channels; if best_delta_aic > -2, support-gate watch cannot clear without new information/model change.",
    }


def _status_from_gate(passed: bool, gate_level: str) -> str:
    if passed:
        return "pass"
    return "reject" if gate_level == "hard" else "watch"


def _score_from_status(status: str) -> float:
    if status == "pass":
        return 1.0
    if status == "watch":
        return 0.5
    return 0.0


def _build_checks(
    *,
    fit_joint: Dict[str, Any],
    fit_eht: Dict[str, Any],
    fit_gw: Dict[str, Any],
    fit_pulsar: Dict[str, Any],
    fit_xray: Dict[str, Any],
    eht_kappa_precision: Dict[str, Any],
    gw_primary_homology: Dict[str, Any],
    gw_multi_event_homology: Dict[str, Any],
    gw_area_theorem: Dict[str, Any],
    premerger_overlap: Dict[str, Any],
) -> List[Dict[str, Any]]:
    checks: List[Dict[str, Any]] = []

    def add_check(check_id: str, metric: str, value: Any, expected: str, passed: bool, gate_level: str, note: str) -> None:
        status = _status_from_gate(bool(passed), gate_level)
        checks.append(
            {
                "id": check_id,
                "metric": metric,
                "value": value,
                "expected": expected,
                "pass": bool(passed),
                "gate_level": gate_level,
                "status": status,
                "score": _score_from_status(status),
                "note": note,
            }
        )

    n_joint = int(fit_joint.get("n_obs", 0))
    add_check(
        "strong_field::input_channels",
        "joint_observables_n",
        n_joint,
        ">=4",
        n_joint >= 4,
        "hard",
        "EHT(2) + GW(2) の最小同時拘束チャネルを要求する。",
    )

    lam_e = _to_float(fit_eht.get("lambda_fit"))
    lam_g = _to_float(fit_gw.get("lambda_fit"))
    sig_e = _to_float(fit_eht.get("lambda_sigma"))
    sig_g = _to_float(fit_gw.get("lambda_sigma"))
    z_consistency = float("nan")
    consistency_pass = False
    if lam_e is not None and lam_g is not None and sig_e is not None and sig_g is not None:
        sig_comb = math.sqrt(sig_e * sig_e + sig_g * sig_g)
        if sig_comb > 0.0:
            z_consistency = abs(lam_e - lam_g) / sig_comb
            consistency_pass = z_consistency <= 3.0
    add_check(
        "strong_field::single_lambda_consistency",
        "abs(lambda_EHT-lambda_GW)/sigma_combined",
        z_consistency,
        "<=3.0",
        consistency_pass,
        "hard",
        "EHT と GW で推定される高次係数が 3σ 以内で整合すること。",
    )

    delta_aic = _to_float(fit_joint.get("delta_aic_fit_minus_baseline"))
    aic_not_overfit = delta_aic is not None and delta_aic <= 6.0
    add_check(
        "strong_field::aic_overfit_gate",
        "delta_aic(fit-baseline)",
        delta_aic,
        "<=6.0",
        aic_not_overfit,
        "hard",
        "高次項導入が過剰自由度（AIC悪化）として即棄却されないこと。",
    )
    add_check(
        "strong_field::aic_support_gate",
        "delta_aic(fit-baseline)",
        delta_aic,
        "<=-2.0",
        delta_aic is not None and delta_aic <= -2.0,
        "watch",
        "高次項を採択するにはAICで有意改善（<=-2）を要求する。",
    )

    kappa_ratio = _to_float(eht_kappa_precision.get("ratio_compressed_over_required"))
    add_check(
        "strong_field::eht_kappa_precision_gate",
        "kappa_sigma_compressed/required",
        kappa_ratio,
        "<=1.0",
        np.isfinite(kappa_ratio) and kappa_ratio <= 1.0,
        "watch",
        "EHT κ系統を圧縮再評価した後でも 3σ 判別必要精度に到達していること。",
    )

    gw_homology = _to_float(gw_primary_homology.get("homology_value"))
    gw_metric = "abs_best_corr(H1,L1; primary strain)"
    gw_expected = ">=0.60"
    gw_note = "GW150914 の一次 strain 同型監査（H1/L1 相関）で pre-merger 同型を判定する。"
    gw_pass = gw_homology is not None and gw_homology >= 0.60
    if gw_homology is None:
        overlap = _to_float(premerger_overlap.get("median_overlap"))
        gw_metric = "median_overlap(GW150914 H1/L1; chirp fallback)"
        gw_expected = ">=0.20"
        gw_note = "一次 strain 同型指標が欠損のため chirp overlap にフォールバック。"
        gw_homology = overlap
        gw_pass = overlap is not None and overlap >= 0.20
    add_check(
        "strong_field::gw_primary_homology_gate",
        gw_metric,
        gw_homology,
        gw_expected,
        gw_pass,
        "watch",
        gw_note,
    )

    gw_multi_value = _to_float(gw_multi_event_homology.get("homology_value"))
    gw_multi_n = int(gw_multi_event_homology.get("n_usable_events", 0))
    gw_multi_status = str(gw_multi_event_homology.get("overall_status") or "")
    gw_multi_pass = gw_multi_n >= 2 and gw_multi_value is not None and gw_multi_value >= 0.60
    if gw_multi_status.lower() == "reject":
        gw_multi_pass = False
    add_check(
        "strong_field::gw_multi_event_homology_gate",
        "median_abs_corr_usable(H1/L1 multi-event)",
        gw_multi_value,
        ">=0.60 with n_usable_events>=2",
        gw_multi_pass,
        "watch",
        "GW multi-event H1/L1 振幅比監査で、単一イベント依存でない同型性を確認する。",
    )

    area_sigma = _to_float(gw_area_theorem.get("sigma_gaussian_combined"))
    add_check(
        "strong_field::gw_area_theorem_gate",
        "sigma_gaussian_combined(area theorem)",
        area_sigma,
        ">=3.0",
        area_sigma is not None and area_sigma >= 3.0,
        "watch",
        "GW250114 面積定理の合成有意度が 3σ 以上であること。",
    )

    n_pulsar = int(fit_pulsar.get("n_obs", 0))
    add_check(
        "strong_field::pulsar_channel_gate",
        "pulsar_observables_n",
        n_pulsar,
        ">=2",
        n_pulsar >= 2,
        "watch",
        "binary-pulsar 軌道崩壊（R-1）の独立チャネルが2系以上あること。",
    )

    lam_p = _to_float(fit_pulsar.get("lambda_fit"))
    sig_p = _to_float(fit_pulsar.get("lambda_sigma"))
    lam_g = _to_float(fit_gw.get("lambda_fit"))
    sig_g = _to_float(fit_gw.get("lambda_sigma"))
    z_pg = float("nan")
    pass_pg = False
    if lam_p is not None and sig_p is not None and sig_p > 0.0 and lam_g is not None and sig_g is not None and sig_g > 0.0:
        sig_comb = math.sqrt(sig_p * sig_p + sig_g * sig_g)
        if sig_comb > 0.0:
            z_pg = abs(lam_p - lam_g) / sig_comb
            pass_pg = z_pg <= 3.0
    add_check(
        "strong_field::gw_pulsar_consistency_gate",
        "abs(lambda_GW-lambda_PULSAR)/sigma_combined",
        z_pg,
        "<=3.0",
        pass_pg,
        "watch",
        "GW（合体）と pulsar（軌道崩壊）で λ_H proxy が 3σ 以内で整合すること。",
    )

    n_xray = int(fit_xray.get("n_obs", 0))
    add_check(
        "strong_field::xray_isco_channel_gate",
        "xray_isco_observables_n",
        n_xray,
        ">=3",
        n_xray >= 3,
        "watch",
        "Fe-Kα broad line の ISCO proxy で独立チャネルが3点以上あること。",
    )

    lam_x = _to_float(fit_xray.get("lambda_fit"))
    sig_x = _to_float(fit_xray.get("lambda_sigma"))
    z_xg = float("nan")
    pass_xg = False
    if lam_x is not None and sig_x is not None and sig_x > 0.0 and lam_g is not None and sig_g is not None and sig_g > 0.0:
        sig_comb = math.sqrt(sig_x * sig_x + sig_g * sig_g)
        if sig_comb > 0.0:
            z_xg = abs(lam_x - lam_g) / sig_comb
            pass_xg = z_xg <= 3.0
    add_check(
        "strong_field::gw_xray_consistency_gate",
        "abs(lambda_GW-lambda_XRAY)/sigma_combined",
        z_xg,
        "<=3.0",
        pass_xg,
        "watch",
        "GW（合体）と Fe-Kα/ISCO proxy で λ_H proxy が 3σ 以内で整合すること。",
    )
    return checks


def _decision_from_checks(checks: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    hard_fail_ids = [str(c["id"]) for c in checks if c.get("gate_level") == "hard" and c.get("pass") is not True]
    watch_ids = [str(c["id"]) for c in checks if c.get("gate_level") == "watch" and c.get("pass") is not True]
    if hard_fail_ids:
        overall = "reject"
        decision = "strong_field_higher_order_reject"
    elif watch_ids:
        overall = "watch"
        decision = "strong_field_higher_order_watch"
    else:
        overall = "pass"
        decision = "strong_field_higher_order_pass"
    return {
        "overall_status": overall,
        "decision": decision,
        "hard_fail_ids": hard_fail_ids,
        "watch_ids": watch_ids,
        "rule": "Reject if any hard gate fails; watch if only watch gates fail; otherwise pass.",
    }


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[LambdaObs], lambda_joint: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "channel",
        "source",
        "label",
        "lambda_obs",
        "sigma_lambda",
        "z_baseline_lambda0",
        "z_joint_fit",
        "note",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "channel": row.channel,
                    "source": row.source,
                    "label": row.label,
                    "lambda_obs": row.lambda_obs,
                    "sigma_lambda": row.sigma_lambda,
                    "z_baseline_lambda0": row.lambda_obs / row.sigma_lambda,
                    "z_joint_fit": (row.lambda_obs - lambda_joint) / row.sigma_lambda,
                    "note": row.note,
                }
            )


def _set_japanese_font() -> None:
    if plt is None:
        return
    try:
        import matplotlib as mpl
        import matplotlib.font_manager as fm

        preferred = ["Yu Gothic", "Meiryo", "BIZ UDGothic", "MS Gothic", "Yu Mincho", "MS Mincho"]
        available = {f.name for f in fm.fontManager.ttflist}
        chosen = [name for name in preferred if name in available]
        if not chosen:
            return
        mpl.rcParams["font.family"] = chosen + ["DejaVu Sans"]
        mpl.rcParams["axes.unicode_minus"] = False
    except Exception:
        return


def _plot(
    path: Path,
    *,
    rows: Sequence[LambdaObs],
    fit_joint: Dict[str, Any],
    fit_eht: Dict[str, Any],
    fit_gw: Dict[str, Any],
    fit_pulsar: Dict[str, Any],
    fit_xray: Dict[str, Any],
    kappa_ratio: float,
    gw_homology: float,
    gw_homology_expected: float,
    gw_multi_homology: float,
    gw_area_sigma: float,
) -> None:
    if plt is None:
        return
    _set_japanese_font()
    path.parent.mkdir(parents=True, exist_ok=True)

    labels = [r.label for r in rows]
    y = np.arange(len(rows), dtype=float)
    lam = np.asarray([r.lambda_obs for r in rows], dtype=float)
    sig = np.asarray([r.sigma_lambda for r in rows], dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.8), constrained_layout=True)
    ax0, ax1, ax2 = axes

    color_map = {"EHT": "#1f77b4", "GW": "#ff7f0e", "PULSAR": "#2ca02c"}
    for i, row in enumerate(rows):
        ax0.errorbar(
            x=lam[i],
            y=y[i],
            xerr=sig[i],
            fmt="o",
            color=color_map.get(row.channel, "#666666"),
            capsize=3,
        )
    lam_joint = float(fit_joint.get("lambda_fit", 0.0))
    sig_joint = float(fit_joint.get("lambda_sigma", float("nan")))
    lam_eht = float(fit_eht.get("lambda_fit", float("nan")))
    lam_gw = float(fit_gw.get("lambda_fit", float("nan")))
    lam_pulsar = float(fit_pulsar.get("lambda_fit", float("nan")))
    lam_xray = float(fit_xray.get("lambda_fit", float("nan")))
    ax0.axvline(0.0, color="#666666", linestyle="--", linewidth=1.0, label="baseline λ=0")
    ax0.axvline(lam_joint, color="#2ca02c", linestyle="-", linewidth=1.8, label="joint λ fit")
    if np.isfinite(sig_joint) and sig_joint > 0:
        ax0.axvspan(lam_joint - sig_joint, lam_joint + sig_joint, color="#2ca02c", alpha=0.15, lw=0)
    if np.isfinite(lam_eht):
        ax0.axvline(lam_eht, color="#1f77b4", linestyle=":", linewidth=1.2, label="EHT-only fit")
    if np.isfinite(lam_gw):
        ax0.axvline(lam_gw, color="#ff7f0e", linestyle=":", linewidth=1.2, label="GW-only fit")
    if np.isfinite(lam_pulsar):
        ax0.axvline(lam_pulsar, color="#2ca02c", linestyle=":", linewidth=1.2, label="Pulsar-only fit")
    if np.isfinite(lam_xray):
        ax0.axvline(lam_xray, color="#8c564b", linestyle=":", linewidth=1.2, label="X-ray ISCO-only fit")
    ax0.set_yticks(y)
    ax0.set_yticklabels(labels)
    ax0.set_xlabel("λ_H estimate")
    ax0.set_title("Channel-level λ_H constraints")
    ax0.grid(alpha=0.25)
    ax0.legend(loc="best", fontsize=8.8)

    metrics = ["chi2/dof baseline", "chi2/dof fit", "ΔAIC (fit-baseline)"]
    vals = [
        float(fit_joint.get("chi2_dof_baseline", float("nan"))),
        float(fit_joint.get("chi2_dof_fit", float("nan"))),
        float(fit_joint.get("delta_aic_fit_minus_baseline", float("nan"))),
    ]
    bars = ax1.bar(np.arange(len(metrics)), vals, color=["#7f7f7f", "#2ca02c", "#d62728"])
    ax1.axhline(0.0, color="#666666", linewidth=1.0)
    ax1.set_xticks(np.arange(len(metrics)))
    ax1.set_xticklabels(metrics, rotation=12, ha="right")
    ax1.set_title("Joint fit merit")
    ax1.grid(alpha=0.25, axis="y")
    for b in bars:
        h = b.get_height()
        if np.isfinite(h):
            ax1.text(b.get_x() + b.get_width() * 0.5, h, f"{h:.3f}", ha="center", va="bottom", fontsize=8)

    diag_labels = ["κ ratio (compressed)", "GW primary homology", "GW multi-event homology", "GW area σ"]
    diag_vals = [kappa_ratio, gw_homology, gw_multi_homology, gw_area_sigma]
    ax2.bar([0, 1, 2, 3], diag_vals, color=["#9467bd", "#17becf", "#bcbd22", "#7f7f7f"])
    ax2.axhline(1.0, color="#333333", linestyle="--", linewidth=1.0, label="κ ratio gate")
    ax2.axhline(gw_homology_expected, color="#17becf", linestyle=":", linewidth=1.0, label="GW homology gate")
    ax2.axhline(0.60, color="#bcbd22", linestyle=":", linewidth=1.0, label="GW multi-event gate")
    ax2.axhline(3.0, color="#7f7f7f", linestyle=":", linewidth=1.0, label="GW area gate")
    ax2.set_xticks([0, 1, 2, 3])
    ax2.set_xticklabels(diag_labels, rotation=12, ha="right")
    ax2.set_title("Watch diagnostics")
    ax2.grid(alpha=0.25, axis="y")
    ax2.legend(loc="best", fontsize=8.8)

    fig.suptitle("Phase 8 / Step 8.7.27: strong-field higher-order closure audit")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 8.7.27 strong-field higher-order audit")
    parser.add_argument(
        "--step-tag",
        default="8.7.27.24",
        help="Step tag stored in output JSON/worklog (e.g. 8.7.27.30).",
    )
    parser.add_argument(
        "--eht-shadow-json",
        type=Path,
        default=ROOT / "output" / "public" / "eht" / "eht_shadow_compare.json",
        help="EHT shadow comparison JSON (public).",
    )
    parser.add_argument(
        "--eht-shadow-json-fallback",
        type=Path,
        default=ROOT / "output" / "private" / "eht" / "eht_shadow_compare.json",
        help="EHT shadow comparison JSON (fallback).",
    )
    parser.add_argument(
        "--eht-kappa-budget-json",
        type=Path,
        default=ROOT / "output" / "public" / "eht" / "eht_kappa_error_budget.json",
        help="EHT kappa budget JSON (public).",
    )
    parser.add_argument(
        "--eht-kappa-budget-json-fallback",
        type=Path,
        default=ROOT / "output" / "private" / "eht" / "eht_kappa_error_budget.json",
        help="EHT kappa budget JSON (fallback).",
    )
    parser.add_argument(
        "--gw-imr-json",
        type=Path,
        default=ROOT / "output" / "public" / "gw" / "gw250114_imr_consistency.json",
        help="GW IMR consistency JSON (public).",
    )
    parser.add_argument(
        "--gw-imr-json-fallback",
        type=Path,
        default=ROOT / "output" / "private" / "gw" / "gw250114_imr_consistency.json",
        help="GW IMR consistency JSON (fallback).",
    )
    parser.add_argument(
        "--gw-chirp-json",
        type=Path,
        default=ROOT / "output" / "public" / "gw" / "gw150914_chirp_phase_metrics.json",
        help="GW150914 chirp metrics JSON (public; fallback homology source).",
    )
    parser.add_argument(
        "--gw-chirp-json-fallback",
        type=Path,
        default=ROOT / "output" / "private" / "gw" / "gw150914_chirp_phase_metrics.json",
        help="GW150914 chirp metrics JSON (private fallback).",
    )
    parser.add_argument(
        "--gw-primary-homology-json",
        type=Path,
        default=ROOT / "output" / "public" / "gw" / "gw150914_h1_l1_amplitude_ratio_metrics.json",
        help="GW150914 H1/L1 primary-strain homology metrics JSON (public).",
    )
    parser.add_argument(
        "--gw-primary-homology-json-fallback",
        type=Path,
        default=ROOT / "output" / "private" / "gw" / "gw150914_h1_l1_amplitude_ratio_metrics.json",
        help="GW150914 H1/L1 primary-strain homology metrics JSON (private fallback).",
    )
    parser.add_argument(
        "--gw-multi-event-homology-json",
        type=Path,
        default=ROOT / "output" / "public" / "gw" / "gw_h1_l1_multi_event_amplitude_audit.json",
        help="GW multi-event H1/L1 homology audit JSON (public).",
    )
    parser.add_argument(
        "--gw-multi-event-homology-json-fallback",
        type=Path,
        default=ROOT / "output" / "private" / "gw" / "gw_h1_l1_multi_event_amplitude_audit.json",
        help="GW multi-event H1/L1 homology audit JSON (private fallback).",
    )
    parser.add_argument(
        "--gw-area-theorem-json",
        type=Path,
        default=ROOT / "output" / "public" / "gw" / "gw250114_area_theorem_test.json",
        help="GW250114 area-theorem metrics JSON (public).",
    )
    parser.add_argument(
        "--gw-area-theorem-json-fallback",
        type=Path,
        default=ROOT / "output" / "private" / "gw" / "gw250114_area_theorem_test.json",
        help="GW250114 area-theorem metrics JSON (private fallback).",
    )
    parser.add_argument(
        "--gw-polarization-stage-json",
        type=Path,
        default=ROOT / "output" / "public" / "gw" / "gw_polarization_corr_gate_stage_audit.json",
        help="GW polarization corr-gate stage audit JSON (public).",
    )
    parser.add_argument(
        "--gw-polarization-stage-json-fallback",
        type=Path,
        default=ROOT / "output" / "private" / "gw" / "gw_polarization_corr_gate_stage_audit.json",
        help="GW polarization corr-gate stage audit JSON (private fallback).",
    )
    parser.add_argument(
        "--gw-polarization-network-candidate-json",
        type=Path,
        default=ROOT
        / "output"
        / "public"
        / "gw"
        / "gw_polarization_h1_l1_v1_network_audit_corr005_ext4_sky50k.json",
        help="GW polarization network audit JSON candidate #1 (public).",
    )
    parser.add_argument(
        "--gw-polarization-network-candidate-json-fallback",
        type=Path,
        default=ROOT
        / "output"
        / "private"
        / "gw"
        / "gw_polarization_h1_l1_v1_network_audit_corr005_ext4_sky50k.json",
        help="GW polarization network audit JSON candidate #1 (private fallback).",
    )
    parser.add_argument(
        "--gw-polarization-network-candidate-pruned-json",
        type=Path,
        default=ROOT
        / "output"
        / "public"
        / "gw"
        / "gw_polarization_h1_l1_v1_network_audit_corr005_pruned_sky50k.json",
        help="GW polarization network audit JSON candidate #2 (public).",
    )
    parser.add_argument(
        "--gw-polarization-network-candidate-pruned-json-fallback",
        type=Path,
        default=ROOT
        / "output"
        / "private"
        / "gw"
        / "gw_polarization_h1_l1_v1_network_audit_corr005_pruned_sky50k.json",
        help="GW polarization network audit JSON candidate #2 (private fallback).",
    )
    parser.add_argument(
        "--gw-polarization-network-candidate-pruned-ext4-json",
        type=Path,
        default=ROOT
        / "output"
        / "public"
        / "gw"
        / "gw_polarization_h1_l1_v1_network_audit_corr005_pruned_ext4_sky50k.json",
        help="GW polarization network audit JSON candidate #3 (public).",
    )
    parser.add_argument(
        "--gw-polarization-network-candidate-pruned-ext4-json-fallback",
        type=Path,
        default=ROOT
        / "output"
        / "private"
        / "gw"
        / "gw_polarization_h1_l1_v1_network_audit_corr005_pruned_ext4_sky50k.json",
        help="GW polarization network audit JSON candidate #3 (private fallback).",
    )
    parser.add_argument(
        "--gw-polarization-network-candidate-whitenrefresh-json",
        type=Path,
        default=ROOT
        / "output"
        / "public"
        / "gw"
        / "gw_polarization_h1_l1_v1_network_audit_corr005_whitenrefresh_sky50k.json",
        help="GW polarization network audit JSON candidate #4 (public).",
    )
    parser.add_argument(
        "--gw-polarization-network-candidate-whitenrefresh-json-fallback",
        type=Path,
        default=ROOT
        / "output"
        / "private"
        / "gw"
        / "gw_polarization_h1_l1_v1_network_audit_corr005_whitenrefresh_sky50k.json",
        help="GW polarization network audit JSON candidate #4 (private fallback).",
    )
    parser.add_argument(
        "--gw-polarization-network-candidate-corr003-ext4-json",
        type=Path,
        default=ROOT
        / "output"
        / "public"
        / "gw"
        / "gw_polarization_h1_l1_v1_network_audit_corr003_ext4.json",
        help="GW polarization network audit JSON candidate #5 (public).",
    )
    parser.add_argument(
        "--gw-polarization-network-candidate-corr003-ext4-json-fallback",
        type=Path,
        default=ROOT
        / "output"
        / "private"
        / "gw"
        / "gw_polarization_h1_l1_v1_network_audit_corr003_ext4.json",
        help="GW polarization network audit JSON candidate #5 (private fallback).",
    )
    parser.add_argument(
        "--pulsar-orbital-decay-json",
        type=Path,
        default=ROOT / "output" / "public" / "pulsar" / "binary_pulsar_orbital_decay_metrics.json",
        help="Binary pulsar orbital-decay metrics JSON (public).",
    )
    parser.add_argument(
        "--pulsar-orbital-decay-json-fallback",
        type=Path,
        default=ROOT / "output" / "private" / "pulsar" / "binary_pulsar_orbital_decay_metrics.json",
        help="Binary pulsar orbital-decay metrics JSON (private fallback).",
    )
    parser.add_argument(
        "--xray-isco-csv",
        type=Path,
        default=ROOT / "output" / "public" / "xrism" / "fek_relativistic_broadening_isco_constraints.csv",
        help="Fe-Kα broad-line ISCO constraints CSV (public).",
    )
    parser.add_argument(
        "--xray-isco-csv-fallback",
        type=Path,
        default=ROOT / "output" / "private" / "xrism" / "fek_relativistic_broadening_isco_constraints.csv",
        help="Fe-Kα broad-line ISCO constraints CSV (private fallback).",
    )
    parser.add_argument(
        "--frame-dragging-scalar-limit-json",
        type=Path,
        default=ROOT / "output" / "public" / "theory" / "frame_dragging_scalar_limit_combined_audit.json",
        help="Frame-dragging scalar-limit combined audit JSON (public).",
    )
    parser.add_argument(
        "--frame-dragging-scalar-limit-json-fallback",
        type=Path,
        default=ROOT / "output" / "private" / "theory" / "frame_dragging_scalar_limit_combined_audit.json",
        help="Frame-dragging scalar-limit combined audit JSON (private fallback).",
    )
    parser.add_argument(
        "--cmb-peak-uplift-json",
        type=Path,
        default=ROOT / "output" / "public" / "cosmology" / "cosmology_cmb_peak_uplift_audit.json",
        help="CMB peak uplift audit JSON (public).",
    )
    parser.add_argument(
        "--cmb-peak-uplift-json-fallback",
        type=Path,
        default=ROOT / "output" / "private" / "cosmology" / "cosmology_cmb_peak_uplift_audit.json",
        help="CMB peak uplift audit JSON (private fallback).",
    )
    parser.add_argument(
        "--cluster-collision-offset-json",
        type=Path,
        default=ROOT / "output" / "public" / "cosmology" / "cosmology_cluster_collision_p_peak_offset_audit.json",
        help="Cluster collision offset audit JSON (public).",
    )
    parser.add_argument(
        "--cluster-collision-offset-json-fallback",
        type=Path,
        default=ROOT / "output" / "private" / "cosmology" / "cosmology_cluster_collision_p_peak_offset_audit.json",
        help="Cluster collision offset audit JSON (private fallback).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "output" / "public" / "theory",
        help="Output directory.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    step_tag = str(args.step_tag).strip() or "8.7.27.24"
    step_slug = step_tag.replace(".", "_")

    out_json = out_dir / "pmodel_strong_field_higher_order_audit.json"
    out_csv = out_dir / "pmodel_strong_field_higher_order_audit.csv"
    out_png = out_dir / "pmodel_strong_field_higher_order_audit.png"
    previous_watchpack = _load_previous_watchpack(out_json)

    eht_shadow, eht_shadow_path = _load_first_existing([args.eht_shadow_json, args.eht_shadow_json_fallback])
    eht_budget, eht_budget_path = _load_first_existing([args.eht_kappa_budget_json, args.eht_kappa_budget_json_fallback])
    gw_imr, gw_imr_path = _load_first_existing([args.gw_imr_json, args.gw_imr_json_fallback])
    gw_chirp, gw_chirp_path = _load_first_existing([args.gw_chirp_json, args.gw_chirp_json_fallback])
    gw_primary_homology, gw_primary_homology_path = _load_optional_first_existing(
        [args.gw_primary_homology_json, args.gw_primary_homology_json_fallback]
    )
    gw_multi_event_homology, gw_multi_event_homology_path = _load_optional_first_existing(
        [args.gw_multi_event_homology_json, args.gw_multi_event_homology_json_fallback]
    )
    gw_area_theorem, gw_area_theorem_path = _load_optional_first_existing(
        [args.gw_area_theorem_json, args.gw_area_theorem_json_fallback]
    )
    gw_polarization_stage, gw_polarization_stage_path = _load_optional_first_existing(
        [args.gw_polarization_stage_json, args.gw_polarization_stage_json_fallback]
    )
    gw_pol_network_candidate_1, gw_pol_network_candidate_1_path = _load_optional_first_existing(
        [args.gw_polarization_network_candidate_json, args.gw_polarization_network_candidate_json_fallback]
    )
    gw_pol_network_candidate_2, gw_pol_network_candidate_2_path = _load_optional_first_existing(
        [
            args.gw_polarization_network_candidate_pruned_json,
            args.gw_polarization_network_candidate_pruned_json_fallback,
        ]
    )
    gw_pol_network_candidate_3, gw_pol_network_candidate_3_path = _load_optional_first_existing(
        [
            args.gw_polarization_network_candidate_pruned_ext4_json,
            args.gw_polarization_network_candidate_pruned_ext4_json_fallback,
        ]
    )
    gw_pol_network_candidate_4, gw_pol_network_candidate_4_path = _load_optional_first_existing(
        [
            args.gw_polarization_network_candidate_whitenrefresh_json,
            args.gw_polarization_network_candidate_whitenrefresh_json_fallback,
        ]
    )
    gw_pol_network_candidate_5, gw_pol_network_candidate_5_path = _load_optional_first_existing(
        [
            args.gw_polarization_network_candidate_corr003_ext4_json,
            args.gw_polarization_network_candidate_corr003_ext4_json_fallback,
        ]
    )
    pulsar_metrics, pulsar_metrics_path = _load_optional_first_existing(
        [args.pulsar_orbital_decay_json, args.pulsar_orbital_decay_json_fallback]
    )
    xray_isco_rows, xray_isco_path = _load_optional_csv_first_existing([args.xray_isco_csv, args.xray_isco_csv_fallback])
    frame_dragging_scalar_limit, frame_dragging_scalar_limit_path = _load_optional_first_existing(
        [args.frame_dragging_scalar_limit_json, args.frame_dragging_scalar_limit_json_fallback]
    )
    cmb_peak_uplift, cmb_peak_uplift_path = _load_optional_first_existing(
        [args.cmb_peak_uplift_json, args.cmb_peak_uplift_json_fallback]
    )
    cluster_collision_offset, cluster_collision_offset_path = _load_optional_first_existing(
        [args.cluster_collision_offset_json, args.cluster_collision_offset_json_fallback]
    )

    obs_eht = _extract_eht_observables(eht_shadow)
    obs_gw = _extract_gw_observables(gw_imr)
    obs_pulsar = _extract_pulsar_observables(pulsar_metrics)
    obs_xray = _extract_xray_isco_observables(xray_isco_rows)
    obs_all = [*obs_eht, *obs_gw, *obs_pulsar, *obs_xray]

    fit_eht = _weighted_fit(obs_eht)
    fit_gw = _weighted_fit(obs_gw)
    fit_pulsar = _weighted_fit(obs_pulsar)
    fit_xray = _weighted_fit(obs_xray)
    fit_joint = _weighted_fit(obs_all)
    ansatz_models = _ansatz_extension_model_matrix(obs_all, fit_joint)
    aic_scenarios = _aic_support_scenario_matrix(
        obs_eht=obs_eht,
        obs_gw=obs_gw,
        obs_pulsar=obs_pulsar,
        obs_xray=obs_xray,
    )
    lam_joint = float(fit_joint.get("lambda_fit", 0.0)) if bool(fit_joint.get("ok")) else 0.0

    eht_kappa_precision = _extract_eht_kappa_precision(eht_budget)
    eht_kappa_readiness = _derive_eht_kappa_readiness(eht_kappa_precision)
    eht_kappa_input_signature = _file_signature(eht_budget_path)
    eht_kappa_update_watchpack = _derive_eht_kappa_update_watchpack(
        eht_kappa_readiness=eht_kappa_readiness,
        current_input_signature=eht_kappa_input_signature,
        previous_watchpack=previous_watchpack,
    )
    kappa_required_sigma = _to_float(eht_kappa_precision.get("required_sigma"))
    kappa_budget_sigma = _to_float(eht_kappa_precision.get("conservative_sigma"))
    kappa_compressed_sigma = _to_float(eht_kappa_precision.get("compressed_sigma"))
    kappa_ratio_conservative = _to_float(eht_kappa_precision.get("ratio_conservative_over_required"))
    kappa_ratio = _to_float(eht_kappa_precision.get("ratio_compressed_over_required"))
    if kappa_ratio is None:
        kappa_ratio = float("nan")

    premerger_overlap = _extract_gw_premger_overlap(gw_chirp)
    gw_primary_homology_diag = _extract_gw_primary_homology(gw_primary_homology)
    gw_multi_event_homology_diag = _extract_gw_multi_event_homology(gw_multi_event_homology)
    gw_area_theorem_diag = _extract_gw_area_theorem(gw_area_theorem)
    gw_polarization_stage_diag = _extract_gw_polarization_stage_readiness(gw_polarization_stage)
    support_recovery_target = _derive_aic_support_recovery_target(fit_joint)
    cross_domain_high_tension_candidates = _extract_cross_domain_high_tension_candidates(
        frame_dragging_scalar_limit=frame_dragging_scalar_limit,
        cmb_peak_uplift=cmb_peak_uplift,
        cluster_collision_offset=cluster_collision_offset,
        support_recovery_target=support_recovery_target,
    )
    cross_domain_direct_mapping = _extract_cross_domain_direct_observables(
        frame_dragging_scalar_limit=frame_dragging_scalar_limit,
        cmb_peak_uplift=cmb_peak_uplift,
        cluster_collision_offset=cluster_collision_offset,
    )
    gw_pol_candidate_entries: List[Tuple[str, Dict[str, Any]]] = []
    if gw_pol_network_candidate_1_path is not None:
        gw_pol_candidate_entries.append(("corr005_ext4_sky50k", gw_pol_network_candidate_1))
    if gw_pol_network_candidate_2_path is not None:
        gw_pol_candidate_entries.append(("corr005_pruned_sky50k", gw_pol_network_candidate_2))
    if gw_pol_network_candidate_3_path is not None:
        gw_pol_candidate_entries.append(("corr005_pruned_ext4_sky50k", gw_pol_network_candidate_3))
    if gw_pol_network_candidate_4_path is not None:
        gw_pol_candidate_entries.append(("corr005_whitenrefresh_sky50k", gw_pol_network_candidate_4))
    if gw_pol_network_candidate_5_path is not None:
        gw_pol_candidate_entries.append(("corr003_ext4", gw_pol_network_candidate_5))
    gw_polarization_high_tension_candidates = _extract_gw_polarization_high_tension_candidates(
        gw_pol_candidate_entries, support_recovery_target
    )
    gw_polarization_injected_trial = _build_gw_polarization_injected_observables(
        gw_polarization_high_tension_candidates, obs_all, fit_gw, fit_joint
    )
    obs_with_gw_pol_trial = list(obs_all) + list(gw_polarization_injected_trial.get("observables", []))
    fit_joint_with_gw_pol_trial = _weighted_fit(obs_with_gw_pol_trial)
    delta_aic_without = _to_float(fit_joint.get("delta_aic_fit_minus_baseline"))
    delta_aic_with = _to_float(fit_joint_with_gw_pol_trial.get("delta_aic_fit_minus_baseline"))
    projected_support_gate_cleared = bool(delta_aic_with is not None and delta_aic_with <= -2.0)
    projected_delta_aic_shift = (
        float(delta_aic_with - delta_aic_without)
        if (delta_aic_with is not None and delta_aic_without is not None)
        else None
    )
    gw_polarization_injection_projection = {
        "ok": bool(gw_polarization_injected_trial.get("ok")),
        "n_injected_observables": int(len(gw_polarization_injected_trial.get("observables", []))),
        "delta_aic_without_injection": delta_aic_without,
        "delta_aic_with_injection": delta_aic_with,
        "delta_aic_shift_with_injection": projected_delta_aic_shift,
        "support_gate_cleared_with_injection": projected_support_gate_cleared,
        "fit_joint_with_injection": fit_joint_with_gw_pol_trial,
        "note": (
            "Injection projection uses physics-calibrated mismatch->z mapping; "
            "support decision remains based on baseline channels until physics-level z calibration is locked."
        ),
    }
    cross_domain_injected_trial = _build_cross_domain_injected_observables(
        cross_domain_high_tension_candidates, obs_all, fit_joint
    )
    obs_with_cross_domain_plus = list(obs_all) + list(cross_domain_injected_trial.get("observables_plus", []))
    obs_with_cross_domain_minus = list(obs_all) + list(cross_domain_injected_trial.get("observables_minus", []))
    fit_joint_with_cross_domain_plus = _weighted_fit(obs_with_cross_domain_plus)
    fit_joint_with_cross_domain_minus = _weighted_fit(obs_with_cross_domain_minus)
    delta_aic_with_cross_domain_plus = _to_float(fit_joint_with_cross_domain_plus.get("delta_aic_fit_minus_baseline"))
    delta_aic_with_cross_domain_minus = _to_float(fit_joint_with_cross_domain_minus.get("delta_aic_fit_minus_baseline"))

    selected_sign = None
    selected_fit: Dict[str, Any] = {}
    delta_aic_with_cross_domain_selected = None
    if delta_aic_with_cross_domain_plus is not None and delta_aic_with_cross_domain_minus is not None:
        if delta_aic_with_cross_domain_plus <= delta_aic_with_cross_domain_minus:
            selected_sign = "plus"
            selected_fit = fit_joint_with_cross_domain_plus
            delta_aic_with_cross_domain_selected = float(delta_aic_with_cross_domain_plus)
        else:
            selected_sign = "minus"
            selected_fit = fit_joint_with_cross_domain_minus
            delta_aic_with_cross_domain_selected = float(delta_aic_with_cross_domain_minus)
    elif delta_aic_with_cross_domain_plus is not None:
        selected_sign = "plus"
        selected_fit = fit_joint_with_cross_domain_plus
        delta_aic_with_cross_domain_selected = float(delta_aic_with_cross_domain_plus)
    elif delta_aic_with_cross_domain_minus is not None:
        selected_sign = "minus"
        selected_fit = fit_joint_with_cross_domain_minus
        delta_aic_with_cross_domain_selected = float(delta_aic_with_cross_domain_minus)

    cross_domain_delta_aic_shift_selected = (
        float(delta_aic_with_cross_domain_selected - delta_aic_without)
        if (delta_aic_with_cross_domain_selected is not None and delta_aic_without is not None)
        else None
    )
    cross_domain_support_gate_cleared = bool(
        delta_aic_with_cross_domain_selected is not None and delta_aic_with_cross_domain_selected <= -2.0
    )
    cross_domain_best_z_proxy_capped = _to_float(
        cross_domain_high_tension_candidates.get("best_candidate_z_proxy_capped")
    )
    cross_domain_injection_projection = {
        "ok": bool(cross_domain_injected_trial.get("ok")),
        "n_injected_observables": int(len(cross_domain_injected_trial.get("observables_plus", []))),
        "delta_aic_without_injection": delta_aic_without,
        "delta_aic_with_injection_plus": delta_aic_with_cross_domain_plus,
        "delta_aic_with_injection_minus": delta_aic_with_cross_domain_minus,
        "selected_sign_envelope": selected_sign,
        "delta_aic_with_injection_selected": delta_aic_with_cross_domain_selected,
        "delta_aic_shift_with_injection_selected": cross_domain_delta_aic_shift_selected,
        "support_gate_cleared_with_selected": cross_domain_support_gate_cleared,
        "fit_joint_with_injection_plus": fit_joint_with_cross_domain_plus,
        "fit_joint_with_injection_minus": fit_joint_with_cross_domain_minus,
        "fit_joint_with_injection_selected": selected_fit,
        "best_z_proxy_capped": cross_domain_best_z_proxy_capped,
        "note": (
            "Dual-sign envelope projection for non-GW cross-domain candidates. "
            "Use only as planning proxy until direct lambda_H mapping is fixed per channel."
        ),
    }

    obs_with_cross_domain_direct = list(obs_all) + list(cross_domain_direct_mapping.get("observables", []))
    fit_joint_with_cross_domain_direct_diag = _weighted_fit(obs_with_cross_domain_direct)
    fit_joint_with_cross_domain_direct = _weighted_fit_with_covariance(
        obs_with_cross_domain_direct,
        covariance_pairs=cross_domain_direct_mapping.get("covariance_pairs"),
    )
    delta_aic_with_cross_domain_direct = _to_float(fit_joint_with_cross_domain_direct.get("delta_aic_fit_minus_baseline"))
    delta_aic_with_cross_domain_direct_diag = _to_float(
        fit_joint_with_cross_domain_direct_diag.get("delta_aic_fit_minus_baseline")
    )
    delta_aic_shift_with_cross_domain_direct = (
        float(delta_aic_with_cross_domain_direct - delta_aic_without)
        if (delta_aic_with_cross_domain_direct is not None and delta_aic_without is not None)
        else None
    )
    delta_aic_cov_minus_diag = (
        float(delta_aic_with_cross_domain_direct - delta_aic_with_cross_domain_direct_diag)
        if (
            delta_aic_with_cross_domain_direct is not None
            and delta_aic_with_cross_domain_direct_diag is not None
        )
        else None
    )
    support_gate_cleared_with_cross_domain_direct = bool(
        delta_aic_with_cross_domain_direct is not None and delta_aic_with_cross_domain_direct <= -2.0
    )
    cross_domain_direct_projection = {
        "ok": bool(cross_domain_direct_mapping.get("ok")),
        "n_injected_observables": int(len(cross_domain_direct_mapping.get("observables", []))),
        "delta_aic_without_injection": delta_aic_without,
        "delta_aic_with_direct_mapping": delta_aic_with_cross_domain_direct,
        "delta_aic_with_direct_mapping_diag_only": delta_aic_with_cross_domain_direct_diag,
        "delta_aic_cov_minus_diag": delta_aic_cov_minus_diag,
        "delta_aic_shift_with_direct_mapping": delta_aic_shift_with_cross_domain_direct,
        "support_gate_cleared_with_direct_mapping": support_gate_cleared_with_cross_domain_direct,
        "covariance_mode": fit_joint_with_cross_domain_direct.get("covariance_mode"),
        "covariance_pairs_applied": fit_joint_with_cross_domain_direct.get("covariance_pairs_applied"),
        "covariance_regularization_jitter": fit_joint_with_cross_domain_direct.get("covariance_regularization_jitter"),
        "fit_joint_with_direct_mapping": fit_joint_with_cross_domain_direct,
        "fit_joint_with_direct_mapping_diag_only": fit_joint_with_cross_domain_direct_diag,
        "note": (
            "Decision-grade path: explicit per-observable mappings (lambda_obs, sigma_lambda) "
            "from cross-domain audits without z-proxy envelope; CMB peak/ratio off-diagonal covariance included."
        ),
    }
    gw_homology_value = _to_float(gw_primary_homology_diag.get("homology_value"))
    gw_multi_homology_value = _to_float(gw_multi_event_homology_diag.get("homology_value"))
    gw_area_sigma_value = _to_float(gw_area_theorem_diag.get("sigma_gaussian_combined"))
    gw_homology_expected = 0.60
    if gw_homology_value is None:
        fallback_overlap = _to_float(premerger_overlap.get("median_overlap"))
        gw_homology_value = fallback_overlap
        gw_homology_expected = 0.20

    checks = _build_checks(
        fit_joint=fit_joint,
        fit_eht=fit_eht,
        fit_gw=fit_gw,
        fit_pulsar=fit_pulsar,
        fit_xray=fit_xray,
        eht_kappa_precision=eht_kappa_precision,
        gw_primary_homology=gw_primary_homology_diag,
        gw_multi_event_homology=gw_multi_event_homology_diag,
        gw_area_theorem=gw_area_theorem_diag,
        premerger_overlap=premerger_overlap,
    )
    decision = _decision_from_checks(checks)

    _write_csv(out_csv, obs_all, lambda_joint=lam_joint)
    _plot(
        out_png,
        rows=obs_all,
        fit_joint=fit_joint,
        fit_eht=fit_eht,
        fit_gw=fit_gw,
        fit_pulsar=fit_pulsar,
        fit_xray=fit_xray,
        kappa_ratio=kappa_ratio,
        gw_homology=float(gw_homology_value) if gw_homology_value is not None else float("nan"),
        gw_homology_expected=gw_homology_expected,
        gw_multi_homology=float(gw_multi_homology_value) if gw_multi_homology_value is not None else float("nan"),
        gw_area_sigma=float(gw_area_sigma_value) if gw_area_sigma_value is not None else float("nan"),
    )

    payload: Dict[str, Any] = {
        "generated_utc": _utc_now(),
        "phase": {"phase": 8, "step": step_tag, "name": "strong-field higher-order closure audit"},
        "intent": (
            "Constrain a minimal higher-order coefficient lambda_H from EHT+GW+Pulsar+Fe-Kα channels "
            "under one shared parameter and evaluate whether strong-field data currently support the extension "
            "after GW homology updates, GW area-theorem consistency, independent X-ray ISCO proxy integration, "
            "AIC-penalized ansatz-extension matrix diagnostics, and GW polarization high-tension candidate registry/"
            "physics-calibrated injection mapping, plus non-GW cross-domain high-tension integration with both "
            "proxy envelope and direct observable-level mappings, including CMB off-diagonal covariance and "
            "kappa-gate update-event watchpack."
        ),
        "model": {
            "baseline": "lambda_H = 0 (no higher-order correction)",
            "higher_order_ansatz": "O_pred = (1 + lambda_H) * O_baseline",
            "fit_policy": "single-parameter weighted least squares across EHT, GW, pulsar and X-ray ISCO channels",
        },
        "inputs": {
            "eht_shadow_compare_json": _rel(eht_shadow_path),
            "eht_kappa_error_budget_json": _rel(eht_budget_path),
            "gw_imr_consistency_json": _rel(gw_imr_path),
            "gw150914_chirp_phase_metrics_json": _rel(gw_chirp_path),
            "gw150914_h1_l1_primary_homology_json": (
                _rel(gw_primary_homology_path) if gw_primary_homology_path is not None else None
            ),
            "gw_h1_l1_multi_event_homology_json": (
                _rel(gw_multi_event_homology_path) if gw_multi_event_homology_path is not None else None
            ),
            "gw250114_area_theorem_json": _rel(gw_area_theorem_path) if gw_area_theorem_path is not None else None,
            "gw_polarization_corr_gate_stage_audit_json": (
                _rel(gw_polarization_stage_path) if gw_polarization_stage_path is not None else None
            ),
            "gw_polarization_network_candidate_json": (
                _rel(gw_pol_network_candidate_1_path) if gw_pol_network_candidate_1_path is not None else None
            ),
            "gw_polarization_network_candidate_pruned_json": (
                _rel(gw_pol_network_candidate_2_path) if gw_pol_network_candidate_2_path is not None else None
            ),
            "gw_polarization_network_candidate_pruned_ext4_json": (
                _rel(gw_pol_network_candidate_3_path) if gw_pol_network_candidate_3_path is not None else None
            ),
            "gw_polarization_network_candidate_whitenrefresh_json": (
                _rel(gw_pol_network_candidate_4_path) if gw_pol_network_candidate_4_path is not None else None
            ),
            "gw_polarization_network_candidate_corr003_ext4_json": (
                _rel(gw_pol_network_candidate_5_path) if gw_pol_network_candidate_5_path is not None else None
            ),
            "pulsar_orbital_decay_metrics_json": (
                _rel(pulsar_metrics_path) if pulsar_metrics_path is not None else None
            ),
            "xray_fek_isco_constraints_csv": _rel(xray_isco_path) if xray_isco_path is not None else None,
            "frame_dragging_scalar_limit_combined_audit_json": (
                _rel(frame_dragging_scalar_limit_path) if frame_dragging_scalar_limit_path is not None else None
            ),
            "cosmology_cmb_peak_uplift_audit_json": (
                _rel(cmb_peak_uplift_path) if cmb_peak_uplift_path is not None else None
            ),
            "cosmology_cluster_collision_p_peak_offset_audit_json": (
                _rel(cluster_collision_offset_path) if cluster_collision_offset_path is not None else None
            ),
        },
        "channel_observables": [
            {
                "channel": r.channel,
                "source": r.source,
                "label": r.label,
                "lambda_obs": r.lambda_obs,
                "sigma_lambda": r.sigma_lambda,
                "z_baseline_lambda0": r.lambda_obs / r.sigma_lambda,
                "z_joint_fit": (r.lambda_obs - lam_joint) / r.sigma_lambda,
                "note": r.note,
            }
            for r in obs_all
        ],
        "fits": {
            "eht_only": fit_eht,
            "gw_only": fit_gw,
            "pulsar_only": fit_pulsar,
            "xray_isco_only": fit_xray,
            "joint": fit_joint,
            "joint_with_gw_polarization_injection_trial": fit_joint_with_gw_pol_trial,
            "joint_with_cross_domain_direct_mapping": fit_joint_with_cross_domain_direct,
        },
        "diagnostics": {
            "eht_kappa_precision": {
                "kappa_sigma_required_sgra": kappa_required_sigma,
                "kappa_sigma_budget_conservative_sgra": kappa_budget_sigma,
                "kappa_sigma_budget_compressed_sgra": kappa_compressed_sigma,
                "kappa_precision_ratio_conservative_over_required": kappa_ratio_conservative,
                "kappa_precision_ratio_compressed_over_required": kappa_ratio,
                "compression_mode": eht_kappa_precision.get("compressed_mode"),
                "compression_factor_conservative_over_compressed": eht_kappa_precision.get(
                    "compression_factor_conservative_over_compressed"
                ),
                "compressed_candidates": eht_kappa_precision.get("compressed_candidates"),
            },
            "eht_kappa_readiness": eht_kappa_readiness,
            "eht_kappa_update_watchpack": eht_kappa_update_watchpack,
            "gw150914_primary_homology": gw_primary_homology_diag,
            "gw_h1_l1_multi_event_homology": gw_multi_event_homology_diag,
            "gw250114_area_theorem": gw_area_theorem_diag,
            "gw_polarization_channel_readiness": gw_polarization_stage_diag,
            "gw150914_premerger_overlap_fallback": premerger_overlap,
            "aic_support_recovery_target": support_recovery_target,
            "gw_polarization_high_tension_candidate_registry": gw_polarization_high_tension_candidates,
            "gw_polarization_injected_trial_mapping": {
                "mapping_formula": gw_polarization_injected_trial.get("mapping_formula"),
                "sign_policy": gw_polarization_injected_trial.get("sign_policy"),
                "sigma_reference_policy": gw_polarization_injected_trial.get("sigma_reference_policy"),
                "sigma_reference_value": gw_polarization_injected_trial.get("sigma_reference_value"),
                "sign_reference_value": gw_polarization_injected_trial.get("sign_reference_value"),
                "rows": gw_polarization_injected_trial.get("rows"),
                "ok": gw_polarization_injected_trial.get("ok"),
            },
            "gw_polarization_injection_projection": gw_polarization_injection_projection,
            "cross_domain_high_tension_candidate_registry": cross_domain_high_tension_candidates,
            "cross_domain_injected_trial_mapping": {
                "mapping_formula": cross_domain_injected_trial.get("mapping_formula"),
                "sign_policy": cross_domain_injected_trial.get("sign_policy"),
                "sigma_reference_policy": cross_domain_injected_trial.get("sigma_reference_policy"),
                "sigma_reference_value": cross_domain_injected_trial.get("sigma_reference_value"),
                "rows": cross_domain_injected_trial.get("rows"),
                "ok": cross_domain_injected_trial.get("ok"),
            },
            "cross_domain_injection_projection": cross_domain_injection_projection,
            "cross_domain_direct_mapping": {
                "ok": cross_domain_direct_mapping.get("ok"),
                "n_observables": cross_domain_direct_mapping.get("n_observables"),
                "mapping_policy": cross_domain_direct_mapping.get("mapping_policy"),
                "sigma_source_counts": cross_domain_direct_mapping.get("sigma_source_counts"),
                "cmb_primary_sigma_input": cross_domain_direct_mapping.get("cmb_primary_sigma_input"),
                "cmb_primary_sigma_available": cross_domain_direct_mapping.get("cmb_primary_sigma_available"),
                "covariance_pairs": cross_domain_direct_mapping.get("covariance_pairs"),
                "covariance_summary": cross_domain_direct_mapping.get("covariance_summary"),
                "rows": cross_domain_direct_mapping.get("rows"),
                "note": cross_domain_direct_mapping.get("note"),
            },
            "cross_domain_direct_projection": cross_domain_direct_projection,
            "pulsar_orbital_decay_channels": {
                "n_obs": fit_pulsar.get("n_obs", 0),
                "lambda_fit": fit_pulsar.get("lambda_fit"),
                "lambda_sigma": fit_pulsar.get("lambda_sigma"),
                "chi2_dof_baseline": fit_pulsar.get("chi2_dof_baseline"),
                "chi2_dof_fit": fit_pulsar.get("chi2_dof_fit"),
            },
            "xray_isco_channels": {
                "n_obs": fit_xray.get("n_obs", 0),
                "lambda_fit": fit_xray.get("lambda_fit"),
                "lambda_sigma": fit_xray.get("lambda_sigma"),
                "chi2_dof_baseline": fit_xray.get("chi2_dof_baseline"),
                "chi2_dof_fit": fit_xray.get("chi2_dof_fit"),
            },
            "ansatz_extension_model_matrix": ansatz_models,
            "aic_support_scenario_matrix": aic_scenarios,
        },
        "checks": checks,
        "decision": decision,
        "outputs": {
            "audit_json": _rel(out_json),
            "audit_csv": _rel(out_csv),
            "audit_png": _rel(out_png),
        },
        "falsification_gate": {
            "reject_if": [
                "Any hard gate fails (missing channels, inconsistent lambda across EHT/GW, or severe AIC overfit).",
            ],
            "watch_if": [
                "Only watch-level gates fail (current precision/overlap limits).",
            ],
            "pass_if": [
                "All hard and watch gates pass with one shared lambda_H.",
            ],
        },
    }
    _write_json(out_json, payload)

    if worklog is not None:
        try:
            worklog.append_event(
                f"step_{step_slug}_strong_field_higher_order_audit",
                {
                    "decision": decision.get("decision"),
                    "overall_status": decision.get("overall_status"),
                    "hard_fail_ids": decision.get("hard_fail_ids"),
                    "watch_ids": decision.get("watch_ids"),
                    "lambda_joint": fit_joint.get("lambda_fit"),
                    "lambda_joint_sigma": fit_joint.get("lambda_sigma"),
                    "delta_aic": fit_joint.get("delta_aic_fit_minus_baseline"),
                    "kappa_precision_ratio_compressed": kappa_ratio,
                    "kappa_precision_ratio_conservative": kappa_ratio_conservative,
                    "gw150914_primary_homology": gw_primary_homology_diag.get("homology_value"),
                    "gw_multi_event_homology": gw_multi_event_homology_diag.get("homology_value"),
                    "gw_multi_event_usable": gw_multi_event_homology_diag.get("n_usable_events"),
                    "gw250114_area_sigma": gw_area_theorem_diag.get("sigma_gaussian_combined"),
                    "gw_polarization_locked_corr_use_min": gw_polarization_stage_diag.get("locked_corr_use_min"),
                    "gw_polarization_locked_status": gw_polarization_stage_diag.get("locked_overall_status"),
                    "gw_polarization_locked_n_usable_events": gw_polarization_stage_diag.get("locked_n_usable_events"),
                    "gw_polarization_high_tension_candidate_ready": gw_polarization_stage_diag.get(
                        "high_tension_candidate_ready"
                    ),
                    "gw_pol_candidate_registry_loaded_n": gw_polarization_high_tension_candidates.get(
                        "n_loaded_candidate_audits"
                    ),
                    "gw_pol_candidate_registry_eligible_n": gw_polarization_high_tension_candidates.get(
                        "n_eligible_candidates"
                    ),
                    "gw_pol_candidate_best_source_id": gw_polarization_high_tension_candidates.get(
                        "best_candidate_source_id"
                    ),
                    "gw_pol_candidate_best_tensor_mismatch_median": gw_polarization_high_tension_candidates.get(
                        "best_candidate_tensor_mismatch_median"
                    ),
                    "gw_pol_candidate_best_clear_support_if_unit_scale": gw_polarization_high_tension_candidates.get(
                        "best_candidate_would_clear_support_if_unit_scale"
                    ),
                    "gw_pol_injected_trial_n_obs": gw_polarization_injection_projection.get("n_injected_observables"),
                    "gw_pol_injected_trial_delta_aic_with": gw_polarization_injection_projection.get(
                        "delta_aic_with_injection"
                    ),
                    "gw_pol_injected_trial_delta_aic_shift": gw_polarization_injection_projection.get(
                        "delta_aic_shift_with_injection"
                    ),
                    "gw_pol_injected_trial_support_gate_cleared": gw_polarization_injection_projection.get(
                        "support_gate_cleared_with_injection"
                    ),
                    "cross_domain_candidate_registry_loaded_n": cross_domain_high_tension_candidates.get(
                        "n_loaded_candidate_audits"
                    ),
                    "cross_domain_candidate_registry_eligible_n": cross_domain_high_tension_candidates.get(
                        "n_eligible_candidates"
                    ),
                    "cross_domain_candidate_best_source_id": cross_domain_high_tension_candidates.get(
                        "best_candidate_source_id"
                    ),
                    "cross_domain_candidate_best_z_proxy_capped": cross_domain_high_tension_candidates.get(
                        "best_candidate_z_proxy_capped"
                    ),
                    "cross_domain_injected_trial_n_obs": cross_domain_injection_projection.get("n_injected_observables"),
                    "cross_domain_injected_trial_selected_sign": cross_domain_injection_projection.get(
                        "selected_sign_envelope"
                    ),
                    "cross_domain_injected_trial_delta_aic_selected": cross_domain_injection_projection.get(
                        "delta_aic_with_injection_selected"
                    ),
                    "cross_domain_injected_trial_delta_aic_shift_selected": cross_domain_injection_projection.get(
                        "delta_aic_shift_with_injection_selected"
                    ),
                    "cross_domain_injected_trial_support_gate_cleared": cross_domain_injection_projection.get(
                        "support_gate_cleared_with_selected"
                    ),
                    "cross_domain_direct_mapping_n_obs": cross_domain_direct_mapping.get("n_observables"),
                    "cross_domain_direct_mapping_ok": cross_domain_direct_mapping.get("ok"),
                    "cross_domain_direct_sigma_source_counts": cross_domain_direct_mapping.get("sigma_source_counts"),
                    "cross_domain_direct_covariance_pair_count": (
                        cross_domain_direct_mapping.get("covariance_summary", {}) or {}
                    ).get("pair_count"),
                    "cross_domain_direct_covariance_mode": fit_joint_with_cross_domain_direct.get("covariance_mode"),
                    "cross_domain_direct_covariance_pairs_applied": fit_joint_with_cross_domain_direct.get(
                        "covariance_pairs_applied"
                    ),
                    "cross_domain_direct_cmb_primary_sigma_available": cross_domain_direct_mapping.get(
                        "cmb_primary_sigma_available"
                    ),
                    "cross_domain_direct_trial_delta_aic_with": cross_domain_direct_projection.get(
                        "delta_aic_with_direct_mapping"
                    ),
                    "cross_domain_direct_trial_delta_aic_with_diag_only": cross_domain_direct_projection.get(
                        "delta_aic_with_direct_mapping_diag_only"
                    ),
                    "cross_domain_direct_trial_delta_aic_cov_minus_diag": cross_domain_direct_projection.get(
                        "delta_aic_cov_minus_diag"
                    ),
                    "cross_domain_direct_trial_delta_aic_shift": cross_domain_direct_projection.get(
                        "delta_aic_shift_with_direct_mapping"
                    ),
                    "cross_domain_direct_trial_support_gate_cleared": cross_domain_direct_projection.get(
                        "support_gate_cleared_with_direct_mapping"
                    ),
                    "pulsar_n_obs": fit_pulsar.get("n_obs"),
                    "pulsar_lambda_fit": fit_pulsar.get("lambda_fit"),
                    "pulsar_lambda_sigma": fit_pulsar.get("lambda_sigma"),
                    "xray_isco_n_obs": fit_xray.get("n_obs"),
                    "xray_isco_lambda_fit": fit_xray.get("lambda_fit"),
                    "xray_isco_lambda_sigma": fit_xray.get("lambda_sigma"),
                    "aic_best_scenario_id": aic_scenarios.get("best_scenario_id"),
                    "aic_best_delta_aic": aic_scenarios.get("best_delta_aic"),
                    "ansatz_best_model_id": ansatz_models.get("best_model_id_by_delta_aic"),
                    "ansatz_best_delta_aic": ansatz_models.get("best_delta_aic_vs_baseline"),
                    "ansatz_support_passed_any": ansatz_models.get("support_passed_by_any_model"),
                    "ansatz_support_possible_if_perfect": ansatz_models.get(
                        "support_possible_under_current_data_if_perfect_fit"
                    ),
                    "support_recovery_missing_delta_chi2_gain": support_recovery_target.get("missing_delta_chi2_gain"),
                    "support_recovery_required_abs_z_single": support_recovery_target.get(
                        "required_abs_z_single_new_channel_if_fitted_residual_zero"
                    ),
                    "kappa_gate_ready_now": eht_kappa_readiness.get("gate_ready_now"),
                    "kappa_sigma_gap_to_gate": eht_kappa_readiness.get("sigma_gap_to_gate"),
                    "kappa_sigma_reduction_percent_needed": eht_kappa_readiness.get("sigma_reduction_percent_needed"),
                    "kappa_info_multiplier_if_sigma_inv_sqrt_n": eht_kappa_readiness.get(
                        "independent_info_multiplier_if_sigma_inv_sqrt_n"
                    ),
                    "kappa_update_event_detected": eht_kappa_update_watchpack.get("update_event_detected"),
                    "kappa_update_event_type": eht_kappa_update_watchpack.get("update_event_type"),
                    "kappa_update_event_counter": eht_kappa_update_watchpack.get("event_counter"),
                    "gw150914_overlap_fallback": premerger_overlap.get("median_overlap"),
                    "output_json": _rel(out_json),
                },
            )
        except Exception:
            pass

    print(f"[ok] wrote {out_json}")
    print(f"[ok] wrote {out_csv}")
    print(f"[ok] wrote {out_png}")
    print(
        "[summary] decision={0} status={1} lambda_joint={2:.6f}+/-{3:.6f} delta_aic={4:.3f} kappa_ratio={5:.3f} gw_homology={6:.3f} gw_multi_homology={7:.3f} gw_area_sigma={8:.3f} xray_n={9}".format(
            decision.get("decision"),
            decision.get("overall_status"),
            float(fit_joint.get("lambda_fit", float("nan"))),
            float(fit_joint.get("lambda_sigma", float("nan"))),
            float(fit_joint.get("delta_aic_fit_minus_baseline", float("nan"))),
            float(kappa_ratio),
            float(gw_homology_value) if gw_homology_value is not None else float("nan"),
            float(gw_multi_homology_value) if gw_multi_homology_value is not None else float("nan"),
            float(gw_area_sigma_value) if gw_area_sigma_value is not None else float("nan"),
            int(fit_xray.get("n_obs", 0)),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
