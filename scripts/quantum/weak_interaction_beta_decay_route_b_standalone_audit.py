#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.quantum import weak_interaction_beta_decay_route_ab_audit as route_ab  # noqa: E402
from scripts.summary import worklog  # noqa: E402


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except Exception:
        return path.as_posix()


def _to_float(v: Any) -> float:
    try:
        out = float(v)
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _as_bool(v: Any) -> Optional[bool]:
    if isinstance(v, bool):
        return v
    t = str(v).strip().lower()
    if t in {"true", "1", "yes", "y"}:
        return True
    if t in {"false", "0", "no", "n"}:
        return False
    return None


def _finite_quantile(values: List[float], q: float) -> float:
    arr = np.array([float(v) for v in values if math.isfinite(float(v))], dtype=float)
    if arr.size == 0:
        return float("nan")
    qq = float(max(0.0, min(1.0, q)))
    return float(np.quantile(arr, qq))


def _channel_sign(row: Dict[str, Any], fallback: float) -> float:
    channel = str(row.get("channel", "")).strip().lower()
    if channel == "beta_minus":
        return 1.0
    if channel == "beta_plus":
        return -1.0
    return 1.0 if fallback >= 0.0 else -1.0


def _hflavor_transform_value(
    *,
    row: Dict[str, Any],
    before_mix: float,
    sat_scale_mev: float,
    branch_pivot_mev: float,
    branch_gain: float,
    sign_blend: float,
) -> float:
    q_after = _to_float(row.get("q_pred_after_MeV"))
    q_before = _to_float(row.get("q_pred_before_MeV"))
    if not math.isfinite(q_after):
        return float("nan")
    if not math.isfinite(q_before):
        q_before = q_after

    q_mix = q_after + float(before_mix) * (q_before - q_after)
    sat = max(1.0e-9, float(sat_scale_mev))
    q_sat = sat * math.tanh(q_mix / sat)

    pivot = max(0.0, float(branch_pivot_mev))
    if abs(q_sat) < pivot:
        q_sat = float(branch_gain) * q_sat

    sgn = _channel_sign(row, q_sat)
    q_sign = sgn * abs(q_sat)
    alpha = max(0.0, min(1.0, float(sign_blend)))
    return (1.0 - alpha) * q_sat + alpha * q_sign


def _apply_hflavor_v1(
    *,
    rows: List[Dict[str, Any]],
    before_mix: float,
    sat_scale_mev: float,
    branch_pivot_mev: float,
    branch_gain: float,
    sign_blend: float,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    mapped_rows: List[Dict[str, Any]] = []
    n_transformed = 0
    n_sign_mismatch_before = 0
    n_sign_mismatch_after = 0
    delta_abs: List[float] = []

    for row in rows:
        row_new = dict(row)
        q_before_map = _to_float(row.get("q_pred_after_MeV"))
        q_after_map = _hflavor_transform_value(
            row=row,
            before_mix=before_mix,
            sat_scale_mev=sat_scale_mev,
            branch_pivot_mev=branch_pivot_mev,
            branch_gain=branch_gain,
            sign_blend=sign_blend,
        )
        if math.isfinite(q_before_map) and math.isfinite(q_after_map):
            n_transformed += 1
            delta_abs.append(abs(q_after_map - q_before_map))
            s_expect = _channel_sign(row, q_before_map)
            if q_before_map != 0.0 and math.copysign(1.0, q_before_map) != math.copysign(1.0, s_expect):
                n_sign_mismatch_before += 1
            if q_after_map != 0.0 and math.copysign(1.0, q_after_map) != math.copysign(1.0, s_expect):
                n_sign_mismatch_after += 1
            row_new["q_pred_after_MeV"] = f"{q_after_map:.16g}"
            row_new["hflavor_q_pred_after_raw_MeV"] = f"{q_before_map:.16g}"
            row_new["hflavor_q_pred_after_v1_MeV"] = f"{q_after_map:.16g}"
        mapped_rows.append(row_new)

    mapping_meta = {
        "mode": "hflavor_v1",
        "params_frozen": {
            "before_mix": float(before_mix),
            "sat_scale_mev": float(sat_scale_mev),
            "branch_pivot_mev": float(branch_pivot_mev),
            "branch_gain": float(branch_gain),
            "sign_blend": float(sign_blend),
        },
        "transform_counts": {
            "n_rows": len(rows),
            "n_transformed": int(n_transformed),
            "n_sign_mismatch_before": int(n_sign_mismatch_before),
            "n_sign_mismatch_after": int(n_sign_mismatch_after),
        },
        "transform_delta_abs_MeV": {
            "median": _finite_quantile(delta_abs, 0.50),
            "p95": _finite_quantile(delta_abs, 0.95),
            "max": (float(max(delta_abs)) if delta_abs else float("nan")),
        },
    }
    return mapped_rows, mapping_meta


def _sigma_band_thresholds_from_rows(rows: List[Dict[str, Any]]) -> Tuple[float, float]:
    sigma_vals = [
        _to_float(r.get("q_obs_sigma_MeV"))
        for r in rows
        if math.isfinite(_to_float(r.get("q_obs_sigma_MeV"))) and _to_float(r.get("q_obs_sigma_MeV")) > 0.0
    ]
    s33 = _finite_quantile(sigma_vals, 0.33)
    s67 = _finite_quantile(sigma_vals, 0.67)
    if not (math.isfinite(s33) and s33 > 0.0):
        s33 = 1.0e-3
    if not (math.isfinite(s67) and s67 > s33):
        s67 = max(s33 * 10.0, 1.0e-2)
    return float(s33), float(s67)


def _sigma_band_label(sig: float, s33: float, s67: float) -> str:
    if not math.isfinite(sig) or sig <= 0.0:
        return "sigma_unknown"
    if sig <= s33:
        return "sigma_low"
    if sig <= s67:
        return "sigma_mid"
    return "sigma_high"


def _mode_tag_from_row(row: Dict[str, Any]) -> str:
    mode_consistent = _as_bool(row.get("mode_consistent"))
    if mode_consistent is True:
        return "mode_consistent"
    if mode_consistent is False:
        return "mode_inconsistent"
    return "mode_unknown"


def _transition_class_from_row(row: Dict[str, Any], *, s33: float, s67: float) -> str:
    channel = str(row.get("channel", "unknown") or "unknown")
    q_sigma = _to_float(row.get("q_obs_sigma_MeV"))
    band = _sigma_band_label(q_sigma, s33, s67)
    mode_tag = _mode_tag_from_row(row)
    return f"{channel}|{band}|{mode_tag}"


def _apply_transition_class_local_correction(
    *,
    rows: List[Dict[str, Any]],
    target_classes: Set[str],
    gain: float,
    sat_mev: float,
    blend: float,
    sign_blend: float,
    class_gain_scales: Optional[Dict[str, float]] = None,
    class_sat_scales: Optional[Dict[str, float]] = None,
    class_blend_profiles: Optional[Dict[str, Dict[str, float]]] = None,
    class_sign_scales: Optional[Dict[str, float]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    s33, s67 = _sigma_band_thresholds_from_rows(rows)
    g = float(gain)
    sat = max(1.0e-9, float(sat_mev))
    alpha = max(0.0, min(1.0, float(blend)))
    alpha_sign = max(0.0, min(1.0, float(sign_blend)))
    gain_scales = class_gain_scales if isinstance(class_gain_scales, dict) else {}
    sat_scales = class_sat_scales if isinstance(class_sat_scales, dict) else {}
    blend_profiles = class_blend_profiles if isinstance(class_blend_profiles, dict) else {}
    sign_scales = class_sign_scales if isinstance(class_sign_scales, dict) else {}

    mapped_rows: List[Dict[str, Any]] = []
    n_target_rows = 0
    n_changed = 0
    n_sign_mismatch_before = 0
    n_sign_mismatch_after = 0
    delta_abs: List[float] = []

    for row in rows:
        row_new = dict(row)
        tclass = _transition_class_from_row(row, s33=s33, s67=s67)
        row_new["localcorr_transition_class"] = tclass
        q_before = _to_float(row.get("q_pred_after_MeV"))
        if tclass in target_classes:
            n_target_rows += 1
            if math.isfinite(q_before):
                gain_scale = _to_float(gain_scales.get(tclass, 1.0))
                if not math.isfinite(gain_scale) or gain_scale <= 0.0:
                    gain_scale = 1.0
                sat_scale = _to_float(sat_scales.get(tclass, 1.0))
                if not math.isfinite(sat_scale) or sat_scale <= 0.0:
                    sat_scale = 1.0
                blend_eff = float(alpha)
                blend_prof = blend_profiles.get(tclass) if isinstance(blend_profiles.get(tclass), dict) else {}
                if blend_prof:
                    z_ref = _to_float(blend_prof.get("z_ref", 5.0))
                    if not math.isfinite(z_ref) or z_ref <= 0.0:
                        z_ref = 5.0
                    min_scale = _to_float(blend_prof.get("min_scale", 1.0))
                    min_scale = max(0.0, min(1.0, min_scale)) if math.isfinite(min_scale) else 1.0
                    q_obs = _to_float(row.get("q_obs_MeV"))
                    q_sig = _to_float(row.get("q_obs_sigma_MeV"))
                    z_abs = float("nan")
                    if math.isfinite(q_before) and math.isfinite(q_obs) and math.isfinite(q_sig) and q_sig > 0.0:
                        z_abs = abs((q_before - q_obs) / q_sig)
                    if math.isfinite(z_abs):
                        z_ratio = max(0.0, min(1.0, z_abs / float(z_ref)))
                        blend_scale = float(min_scale + (1.0 - min_scale) * z_ratio)
                        blend_eff = float(alpha * blend_scale)
                    else:
                        blend_eff = float(alpha * min_scale)
                g_eff = float(g * gain_scale)
                sat_eff = max(1.0e-9, float(sat * sat_scale))
                q_sat = sat_eff * math.tanh(q_before / sat_eff)
                q_target = g_eff * q_sat
                q_blend = (1.0 - blend_eff) * q_before + blend_eff * q_target
                sign_scale = _to_float(sign_scales.get(tclass, 1.0))
                if not math.isfinite(sign_scale) or sign_scale < 0.0:
                    sign_scale = 1.0
                alpha_sign_eff = max(0.0, min(1.0, float(alpha_sign) * float(sign_scale)))
                sgn = _channel_sign(row, q_blend)
                q_sign = sgn * abs(q_blend)
                q_after = (1.0 - alpha_sign_eff) * q_blend + alpha_sign_eff * q_sign
                if q_before != 0.0 and math.copysign(1.0, q_before) != math.copysign(1.0, sgn):
                    n_sign_mismatch_before += 1
                if q_after != 0.0 and math.copysign(1.0, q_after) != math.copysign(1.0, sgn):
                    n_sign_mismatch_after += 1
                if q_after != q_before:
                    n_changed += 1
                    delta_abs.append(abs(q_after - q_before))
                row_new["q_pred_after_MeV"] = f"{q_after:.16g}"
                row_new["localcorr_q_pred_before_MeV"] = f"{q_before:.16g}"
                row_new["localcorr_q_pred_after_MeV"] = f"{q_after:.16g}"
        mapped_rows.append(row_new)

    meta = {
        "status": "applied",
        "sigma_band_thresholds_mev": {"q33": float(s33), "q67": float(s67)},
        "target_classes": sorted([str(x) for x in target_classes]),
        "params_frozen": {
            "gain": g,
            "sat_mev": sat,
            "blend": alpha,
            "sign_blend": alpha_sign,
            "class_gain_scales": {
                str(k): float(_to_float(v))
                for k, v in sorted(gain_scales.items(), key=lambda kv: str(kv[0]))
                if str(k) in target_classes and math.isfinite(_to_float(v))
            },
            "class_sat_scales": {
                str(k): float(_to_float(v))
                for k, v in sorted(sat_scales.items(), key=lambda kv: str(kv[0]))
                if str(k) in target_classes and math.isfinite(_to_float(v))
            },
            "class_blend_profiles": {
                str(k): {
                    "z_ref": float(_to_float(v.get("z_ref", float("nan")))),
                    "min_scale": float(_to_float(v.get("min_scale", float("nan")))),
                }
                for k, v in sorted(blend_profiles.items(), key=lambda kv: str(kv[0]))
                if str(k) in target_classes and isinstance(v, dict)
            },
            "class_sign_scales": {
                str(k): float(_to_float(v))
                for k, v in sorted(sign_scales.items(), key=lambda kv: str(kv[0]))
                if str(k) in target_classes and math.isfinite(_to_float(v))
            },
        },
        "transform_counts": {
            "n_rows": int(len(rows)),
            "n_target_rows": int(n_target_rows),
            "n_changed_rows": int(n_changed),
            "n_sign_mismatch_before": int(n_sign_mismatch_before),
            "n_sign_mismatch_after": int(n_sign_mismatch_after),
        },
        "transform_delta_abs_MeV": {
            "median": _finite_quantile(delta_abs, 0.50),
            "p95": _finite_quantile(delta_abs, 0.95),
            "max": (float(max(delta_abs)) if delta_abs else float("nan")),
        },
    }
    return mapped_rows, meta


def _build_candidate_class_gain_scales(
    *,
    candidate: Dict[str, Any],
    target_classes: List[str],
) -> Dict[str, float]:
    tlist = [str(x) for x in target_classes if str(x).strip()]
    if not tlist:
        return {}
    top_class = str(candidate.get("top_priority_class_targeted", "")).strip()
    if top_class not in tlist:
        top_class = tlist[0]
    boost = _to_float(candidate.get("top_class_gain_boost", 1.0))
    if not math.isfinite(boost) or boost <= 0.0:
        boost = 1.0
    scales = {str(tc): 1.0 for tc in tlist}
    scales[str(top_class)] = float(boost)
    return scales


def _build_candidate_class_sat_scales(
    *,
    candidate: Dict[str, Any],
    target_classes: List[str],
) -> Dict[str, float]:
    tlist = [str(x) for x in target_classes if str(x).strip()]
    if not tlist:
        return {}
    sat_scale = _to_float(candidate.get("sigma_low_mode_inconsistent_sat_scale", 1.0))
    if not math.isfinite(sat_scale) or sat_scale <= 0.0:
        sat_scale = 1.0
    scales = {str(tc): 1.0 for tc in tlist}
    for tc in tlist:
        if "|sigma_low|mode_inconsistent" in str(tc):
            scales[str(tc)] = float(sat_scale)
    return scales


def _build_candidate_class_blend_profiles(
    *,
    candidate: Dict[str, Any],
    target_classes: List[str],
) -> Dict[str, Dict[str, float]]:
    tlist = [str(x) for x in target_classes if str(x).strip()]
    if not tlist:
        return {}
    z_ref = _to_float(candidate.get("sigma_low_mode_inconsistent_zref", 5.0))
    if not math.isfinite(z_ref) or z_ref <= 0.0:
        z_ref = 5.0
    min_scale = _to_float(candidate.get("sigma_low_mode_inconsistent_min_blend_scale", 1.0))
    if not math.isfinite(min_scale):
        min_scale = 1.0
    min_scale = max(0.0, min(1.0, float(min_scale)))

    out: Dict[str, Dict[str, float]] = {}
    for tc in tlist:
        if "|sigma_low|mode_inconsistent" in str(tc):
            out[str(tc)] = {
                "z_ref": float(z_ref),
                "min_scale": float(min_scale),
            }
    return out


def _build_candidate_class_sign_scales(
    *,
    candidate: Dict[str, Any],
    target_classes: List[str],
) -> Dict[str, float]:
    tlist = [str(x) for x in target_classes if str(x).strip()]
    if not tlist:
        return {}
    sign_scale = _to_float(candidate.get("sigma_low_mode_inconsistent_sign_blend_scale", 1.0))
    if not math.isfinite(sign_scale):
        sign_scale = 1.0
    sign_scale = max(0.0, float(sign_scale))
    top_class = str(candidate.get("top_priority_class_targeted", "")).strip()
    if top_class not in tlist:
        top_class = tlist[0]
    top_sign_scale = _to_float(candidate.get("top_class_sign_blend_scale", 1.0))
    if not math.isfinite(top_sign_scale):
        top_sign_scale = 1.0
    top_sign_scale = max(0.0, float(top_sign_scale))
    scales: Dict[str, float] = {}
    for tc in tlist:
        scale = 1.0
        if str(tc) == str(top_class):
            scale *= float(top_sign_scale)
        if "|sigma_low|mode_inconsistent" in str(tc):
            scale *= float(sign_scale)
        scales[str(tc)] = float(scale)
    return scales


def _sha256(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(8 * 1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _file_signature(path: Path) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"path": _rel(path), "exists": bool(path.exists())}
    if not path.exists():
        return payload
    stat = path.stat()
    payload["size_bytes"] = int(stat.st_size)
    payload["mtime_utc"] = datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat()
    payload["sha256"] = _sha256(path)
    return payload


def _load_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _load_route_ab_transition(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"exists": False}
    payload = json.loads(path.read_text(encoding="utf-8"))
    decision = payload.get("decision") if isinstance(payload.get("decision"), dict) else {}
    return {
        "exists": True,
        "path": _rel(path),
        "sha256": _sha256(path),
        "transition": decision.get("transition"),
        "route_a_hard_pass": decision.get("route_a_hard_pass"),
        "route_b_hard_pass": decision.get("route_b_hard_pass"),
        "route_b_watch_pass": decision.get("route_b_watch_pass"),
    }


def _evaluate_route_b(
    *,
    route_b_eval: Dict[str, Any],
    closure_gate: Dict[str, Any],
    z_gate: float,
    overfit_gap_gate: float,
) -> Dict[str, Any]:
    holdout_all = route_b_eval.get("holdout_all") if isinstance(route_b_eval.get("holdout_all"), dict) else {}
    train_all = route_b_eval.get("train_all") if isinstance(route_b_eval.get("train_all"), dict) else {}
    overfit_guard = route_b_eval.get("overfit_guard") if isinstance(route_b_eval.get("overfit_guard"), dict) else {}
    dof_penalty = route_b_eval.get("dof_penalty") if isinstance(route_b_eval.get("dof_penalty"), dict) else {}

    p95_q = _to_float(holdout_all.get("p95_abs_z_qbeta"))
    p95_logft = _to_float(holdout_all.get("p95_abs_z_logft_proxy"))
    max_q = _to_float(holdout_all.get("max_abs_z_qbeta"))
    max_logft = _to_float(holdout_all.get("max_abs_z_logft_proxy"))
    gap_q = _to_float(overfit_guard.get("p95_gap_qbeta"))
    gap_logft = _to_float(overfit_guard.get("p95_gap_logft_proxy"))

    gate_qbeta = bool(math.isfinite(p95_q) and p95_q <= z_gate)
    gate_logft = bool(math.isfinite(p95_logft) and p95_logft <= z_gate)
    overfit_pass = bool(_as_bool(overfit_guard.get("pass")) is True)
    dof_equalized = bool(int(dof_penalty.get("k_params") or 0) == 2)
    closure_hard = bool(_as_bool(closure_gate.get("hard_pass")) is True)
    closure_watch = bool(_as_bool(closure_gate.get("watch_pass")) is True)

    hard_fail_ids: List[str] = []
    if not gate_qbeta:
        hard_fail_ids.append("route_b::holdout_qbeta_p95_gate")
    if not gate_logft:
        hard_fail_ids.append("route_b::holdout_logft_p95_gate")
    if not overfit_pass:
        hard_fail_ids.append("route_b::overfit_guard_gate")
    if not dof_equalized:
        hard_fail_ids.append("route_b::dof_equalization_gate")
    if not closure_hard:
        hard_fail_ids.append("closure::ckm_pmns_hard_gate")

    watch_fail_ids: List[str] = []
    if not closure_watch:
        watch_fail_ids.append("closure::ckm_pmns_watch_gate")
    if not (math.isfinite(max_q) and max_q <= z_gate):
        watch_fail_ids.append("route_b::max_qbeta_watch_gate")
    if not (math.isfinite(max_logft) and max_logft <= z_gate):
        watch_fail_ids.append("route_b::max_logft_watch_gate")

    if hard_fail_ids:
        overall_status = "reject"
        transition = "B_standalone_reject"
    elif watch_fail_ids:
        overall_status = "watch"
        transition = "B_standalone_watch"
    else:
        overall_status = "pass"
        transition = "B_standalone_pass"

    return {
        "overall_status": overall_status,
        "transition": transition,
        "route_b_hard_pass": not hard_fail_ids,
        "route_b_watch_pass": (not hard_fail_ids) and (not watch_fail_ids),
        "hard_fail_ids": hard_fail_ids,
        "watch_fail_ids": watch_fail_ids,
        "gate_values": {
            "p95_abs_z_qbeta_holdout": p95_q,
            "p95_abs_z_logft_holdout": p95_logft,
            "max_abs_z_qbeta_holdout": max_q,
            "max_abs_z_logft_holdout": max_logft,
            "p95_gap_qbeta_holdout_minus_train": gap_q,
            "p95_gap_logft_holdout_minus_train": gap_logft,
            "z_gate": float(z_gate),
            "overfit_gap_gate": float(overfit_gap_gate),
        },
        "gate_pass": {
            "holdout_qbeta_p95": gate_qbeta,
            "holdout_logft_p95": gate_logft,
            "overfit_guard": overfit_pass,
            "dof_equalized_k2": dof_equalized,
            "closure_hard": closure_hard,
            "closure_watch": closure_watch,
        },
        "route_b_counts": {
            "n_rows_train": int(train_all.get("n_rows") or 0),
            "n_rows_holdout": int(holdout_all.get("n_rows") or 0),
            "n_q_rows_holdout": int(holdout_all.get("n_q_rows") or 0),
            "n_logft_rows_holdout": int(holdout_all.get("n_logft_proxy_rows") or 0),
            "n_qbeta_gt3_holdout": int(holdout_all.get("n_qbeta_gt3") or 0),
            "n_logft_gt3_holdout": int(holdout_all.get("n_logft_proxy_gt3") or 0),
        },
        "improvement_target": {
            "required_p95_qbeta_reduction_ratio": (
                float(z_gate) / p95_q if math.isfinite(p95_q) and p95_q > 0.0 else float("nan")
            ),
            "required_p95_logft_reduction_ratio": (
                float(z_gate) / p95_logft if math.isfinite(p95_logft) and p95_logft > 0.0 else float("nan")
            ),
            "required_overfit_gap_qbeta_drop": (
                gap_q - float(overfit_gap_gate) if math.isfinite(gap_q) else float("nan")
            ),
            "required_overfit_gap_logft_drop": (
                gap_logft - float(overfit_gap_gate) if math.isfinite(gap_logft) else float("nan")
            ),
        },
    }


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["metric_id", "value", "threshold", "operator", "pass", "gate_level", "note"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_csv_rows(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def _safe_share(numer: int, denom: int) -> float:
    if denom <= 0:
        return float("nan")
    return float(float(numer) / float(denom))


def _safe_max(values: List[float]) -> float:
    arr = [float(v) for v in values if math.isfinite(float(v))]
    if not arr:
        return float("nan")
    return float(max(arr))


def _aggregate_outlier_group(
    *,
    key: str,
    rows: List[Dict[str, Any]],
    total_q_fail: int,
    total_logft_fail: int,
    z_gate: float,
) -> Dict[str, Any]:
    zq = [float(r["z_qbeta"]) for r in rows if math.isfinite(float(r.get("z_qbeta", float("nan"))))]
    zlog = [float(r["z_logft"]) for r in rows if math.isfinite(float(r.get("z_logft", float("nan"))))]
    n_q_fail = int(sum(1 for v in zq if v > float(z_gate)))
    n_log_fail = int(sum(1 for v in zlog if v > float(z_gate)))
    p95_q = _finite_quantile(zq, 0.95)
    p95_log = _finite_quantile(zlog, 0.95)
    severity_q = (p95_q / float(z_gate)) if math.isfinite(p95_q) and z_gate > 0.0 else 0.0
    severity_log = (p95_log / float(z_gate)) if math.isfinite(p95_log) and z_gate > 0.0 else 0.0
    share_q = _safe_share(n_q_fail, total_q_fail)
    share_log = _safe_share(n_log_fail, total_logft_fail)
    if not math.isfinite(share_q):
        share_q = 0.0
    if not math.isfinite(share_log):
        share_log = 0.0
    priority_score = float((share_q * severity_q) + (0.35 * share_log * severity_log))
    return {
        "group_key": str(key),
        "n_rows": int(len(rows)),
        "n_qbeta_gt3": n_q_fail,
        "n_logft_gt3": n_log_fail,
        "frac_qbeta_gt3": _safe_share(n_q_fail, len(zq)),
        "frac_logft_gt3": _safe_share(n_log_fail, len(zlog)),
        "share_fail_qbeta": share_q,
        "share_fail_logft": share_log,
        "p95_abs_z_qbeta": p95_q,
        "max_abs_z_qbeta": _safe_max(zq),
        "p95_abs_z_logft": p95_log,
        "max_abs_z_logft": _safe_max(zlog),
        "priority_score": priority_score,
    }


def _build_route_b_outlier_decomposition(
    *,
    rows_mapped: List[Dict[str, Any]],
    route_b_eval: Dict[str, Any],
    holdout_hash_modulo: int,
    holdout_hash_residue: int,
    z_gate: float,
) -> Dict[str, Any]:
    _, holdout_rows, split_meta = route_ab._split_rows_by_holdout(
        rows=rows_mapped,
        modulo=int(holdout_hash_modulo),
        residue=int(holdout_hash_residue),
    )
    calibration = route_b_eval.get("calibration") if isinstance(route_b_eval.get("calibration"), dict) else {}
    a_offset = _to_float(calibration.get("a_offset_MeV"))
    b_scale = _to_float(calibration.get("b_scale"))
    if not math.isfinite(a_offset):
        a_offset = 0.0
    if not math.isfinite(b_scale):
        b_scale = 1.0

    s33, s67 = _sigma_band_thresholds_from_rows(holdout_rows)

    rows_scored: List[Dict[str, Any]] = []
    for row in holdout_rows:
        q_obs = _to_float(row.get("q_obs_MeV"))
        q_sigma = _to_float(row.get("q_obs_sigma_MeV"))
        q_raw = _to_float(row.get("q_pred_after_MeV"))
        if not (math.isfinite(q_obs) and math.isfinite(q_sigma) and q_sigma > 0.0 and math.isfinite(q_raw)):
            continue
        q_pred_cal = float(a_offset + b_scale * q_raw)
        z_q = abs((q_pred_cal - q_obs) / q_sigma)
        if q_obs > 0.0 and q_pred_cal > 0.0:
            z_logft = abs(5.0 * (math.log10(q_pred_cal) - math.log10(q_obs)))
        else:
            z_logft = float("nan")
        mode_tag = _mode_tag_from_row(row)
        channel = str(row.get("channel", "unknown") or "unknown")
        band = _sigma_band_label(q_sigma, s33, s67)
        rows_scored.append(
            {
                "nuclide_key": str(row.get("nuclide_key", "") or ""),
                "nuclide_name": str(row.get("nuclide_name", "") or ""),
                "channel": channel,
                "sigma_obs_mev": q_sigma,
                "sigma_band": band,
                "mode_tag": mode_tag,
                "transition_class": f"{channel}|{band}|{mode_tag}",
                "q_obs_mev": q_obs,
                "q_pred_cal_mev": q_pred_cal,
                "z_qbeta": float(z_q),
                "z_logft": float(z_logft),
            }
        )

    total_q_fail = int(sum(1 for r in rows_scored if float(r["z_qbeta"]) > float(z_gate)))
    total_logft_fail = int(
        sum(1 for r in rows_scored if math.isfinite(float(r["z_logft"])) and float(r["z_logft"]) > float(z_gate))
    )

    def summarize(scope: str, key_fn) -> List[Dict[str, Any]]:
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for r in rows_scored:
            groups.setdefault(str(key_fn(r)), []).append(r)
        out: List[Dict[str, Any]] = []
        for gk, grows in groups.items():
            agg = _aggregate_outlier_group(
                key=gk,
                rows=grows,
                total_q_fail=total_q_fail,
                total_logft_fail=total_logft_fail,
                z_gate=float(z_gate),
            )
            agg["scope"] = scope
            out.append(agg)
        out.sort(key=lambda x: (float(x.get("priority_score", 0.0)), float(x.get("max_abs_z_qbeta", 0.0))), reverse=True)
        for idx, row in enumerate(out, start=1):
            row["rank"] = idx
        return out

    by_channel = summarize("channel", lambda r: r["channel"])
    by_sigma_band = summarize("sigma_band", lambda r: r["sigma_band"])
    by_channel_sigma = summarize("channel_sigma", lambda r: f"{r['channel']}|{r['sigma_band']}")
    by_transition_class = summarize("transition_class", lambda r: r["transition_class"])
    by_mode = summarize("mode_tag", lambda r: r["mode_tag"])
    by_nuclide = summarize("nuclide", lambda r: r["nuclide_key"])
    top_nuclides = by_nuclide[:20]
    reduction_priority_order = by_transition_class[:10]

    csv_rows: List[Dict[str, Any]] = []
    for scope_rows in [by_transition_class, by_channel_sigma, by_channel, by_sigma_band, by_mode, top_nuclides]:
        csv_rows.extend(scope_rows)

    top_rows = sorted(rows_scored, key=lambda r: float(r["z_qbeta"]), reverse=True)[:30]
    for idx, r in enumerate(top_rows, start=1):
        r["rank"] = idx

    return {
        "split_meta": split_meta,
        "sigma_band_thresholds_mev": {"q33": float(s33), "q67": float(s67)},
        "counts": {
            "n_holdout_rows_total": int(len(holdout_rows)),
            "n_holdout_rows_scored": int(len(rows_scored)),
            "n_qbeta_gt3_holdout": int(total_q_fail),
            "n_logft_gt3_holdout": int(total_logft_fail),
        },
        "by_channel": by_channel,
        "by_sigma_band": by_sigma_band,
        "by_channel_sigma": by_channel_sigma,
        "by_transition_class": by_transition_class,
        "by_mode_tag": by_mode,
        "top_nuclides": top_nuclides,
        "reduction_priority_order": reduction_priority_order,
        "top_outlier_rows": top_rows,
        "csv_rows": csv_rows,
    }


def _plot_outlier_decomposition(
    *,
    decomposition: Dict[str, Any],
    out_png: Path,
) -> None:
    by_transition = decomposition.get("reduction_priority_order")
    by_channel = decomposition.get("by_channel")
    by_sigma = decomposition.get("by_sigma_band")
    if not isinstance(by_transition, list) or not isinstance(by_channel, list) or not isinstance(by_sigma, list):
        return

    top_transition = by_transition[:6]
    ch_rows = by_channel[:]
    sig_rows = by_sigma[:]

    fig, axes = plt.subplots(2, 2, figsize=(14.0, 8.2), dpi=170)
    ax0, ax1, ax2, ax3 = axes.flatten()

    t_labels = [str(r.get("group_key")) for r in top_transition]
    t_vals = [float(r.get("priority_score", float("nan"))) for r in top_transition]
    ax0.barh(range(len(t_labels)), t_vals, color="#4c78a8")
    ax0.set_yticks(range(len(t_labels)), t_labels)
    ax0.invert_yaxis()
    ax0.set_xlabel("priority score")
    ax0.set_title("Top transition classes (reduction priority)")
    ax0.grid(True, axis="x", alpha=0.25, linestyle=":")

    ch_labels = [str(r.get("group_key")) for r in ch_rows]
    ch_vals = [int(r.get("n_qbeta_gt3", 0)) for r in ch_rows]
    ax1.bar(ch_labels, ch_vals, color="#e45756")
    ax1.set_ylabel("count (|z_Qβ| > 3)")
    ax1.set_title("Channel contribution")
    ax1.grid(True, axis="y", alpha=0.25, linestyle=":")

    sig_labels = [str(r.get("group_key")) for r in sig_rows]
    sig_vals = [int(r.get("n_qbeta_gt3", 0)) for r in sig_rows]
    ax2.bar(sig_labels, sig_vals, color="#54a24b")
    ax2.set_ylabel("count (|z_Qβ| > 3)")
    ax2.set_title("Sigma-band contribution")
    ax2.grid(True, axis="y", alpha=0.25, linestyle=":")

    top_rows = decomposition.get("top_outlier_rows")
    if isinstance(top_rows, list) and top_rows:
        z_vals = [float(r.get("z_qbeta", float("nan"))) for r in top_rows]
        bands = [str(r.get("sigma_band", "")) for r in top_rows]
        color_map = {"sigma_low": "#4c78a8", "sigma_mid": "#f58518", "sigma_high": "#e45756", "sigma_unknown": "#888888"}
        colors = [color_map.get(b, "#888888") for b in bands]
        ax3.bar(range(len(z_vals)), z_vals, color=colors)
        ax3.set_xlabel("top outlier rank")
        ax3.set_ylabel("|z_Qβ|")
        ax3.set_title("Top holdout outliers")
        ax3.grid(True, axis="y", alpha=0.25, linestyle=":")
    else:
        ax3.text(0.5, 0.5, "No outlier rows", ha="center", va="center", transform=ax3.transAxes)
        ax3.set_axis_off()

    fig.suptitle("Step 8.7.31.3: Route-B outlier decomposition (holdout)", fontsize=13)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.94])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def _parse_float_list_csv(text: str) -> List[float]:
    out: List[float] = []
    for token in str(text).split(","):
        token_s = token.strip()
        if not token_s:
            continue
        try:
            value = float(token_s)
        except Exception:
            continue
        if math.isfinite(value):
            out.append(float(value))
    return out


def _parse_int_list_csv(text: str) -> List[int]:
    out: List[int] = []
    for token in str(text).split(","):
        token_s = token.strip()
        if not token_s:
            continue
        try:
            value = int(token_s)
        except Exception:
            continue
        out.append(int(value))
    return out


def _parse_str_list_csv(text: str) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    for token in str(text).split(","):
        token_s = token.strip()
        if not token_s:
            continue
        if token_s in seen:
            continue
        seen.add(token_s)
        out.append(token_s)
    return out


def _parse_str_group_list(text: str) -> List[List[str]]:
    out: List[List[str]] = []
    seen: Set[Tuple[str, ...]] = set()
    for group_raw in str(text).split(";"):
        keys = _parse_str_list_csv(group_raw)
        if not keys:
            continue
        sig = tuple(str(x) for x in keys)
        if sig in seen:
            continue
        seen.add(sig)
        out.append([str(x) for x in keys])
    return out


def _parse_class_weight_map(text: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for token in str(text).split(","):
        token_s = token.strip()
        if not token_s or ":" not in token_s:
            continue
        key_raw, val_raw = token_s.rsplit(":", 1)
        key = str(key_raw).strip()
        if not key:
            continue
        try:
            value = float(str(val_raw).strip())
        except Exception:
            continue
        if math.isfinite(value):
            out[key] = float(max(0.0, value))
    return out


def _parse_class_int_map(text: str) -> Dict[str, int]:
    raw = _parse_class_weight_map(text)
    out: Dict[str, int] = {}
    for key, value in raw.items():
        v = _to_float(value)
        if not math.isfinite(v):
            continue
        out[str(key)] = int(max(0, int(round(v))))
    return out


def _parse_class_nonneg_float_map(text: str) -> Dict[str, float]:
    raw = _parse_class_weight_map(text)
    out: Dict[str, float] = {}
    for key, value in raw.items():
        v = _to_float(value)
        if not math.isfinite(v):
            continue
        out[str(key)] = float(max(0.0, float(v)))
    return out


def _pareto_front_indices(points: List[Tuple[float, float]]) -> List[int]:
    valid = [i for i, (x, y) in enumerate(points) if math.isfinite(x) and math.isfinite(y)]
    front: List[int] = []
    for i in valid:
        xi, yi = points[i]
        dominated = False
        for j in valid:
            if i == j:
                continue
            xj, yj = points[j]
            if (xj <= xi and yj <= yi) and (xj < xi or yj < yi):
                dominated = True
                break
        if not dominated:
            front.append(i)
    return front


def _safe_norm(value: float, min_v: float, max_v: float) -> float:
    if not (math.isfinite(value) and math.isfinite(min_v) and math.isfinite(max_v)):
        return float("nan")
    span = max_v - min_v
    if not math.isfinite(span) or span <= 0.0:
        return 0.0
    return float((value - min_v) / span)


def _build_hflavor_sweep(
    *,
    rows: List[Dict[str, Any]],
    logft_sigma_proxy: float,
    z_gate: float,
    holdout_hash_modulo: int,
    holdout_hash_residue: int,
    overfit_gap_gate: float,
    sigma_floor_mev: float,
    base_sat_scale_mev: float,
    before_mix_values: List[float],
    sat_scale_multipliers: List[float],
    branch_pivot_values: List[float],
    branch_gain_values: List[float],
    sign_blend_values: List[float],
) -> Dict[str, Any]:
    candidates: List[Dict[str, Any]] = []
    candidate_rows: List[Dict[str, Any]] = []
    candidate_counter = 0
    for before_mix, sat_mul, pivot, gain, sign_blend in itertools.product(
        before_mix_values, sat_scale_multipliers, branch_pivot_values, branch_gain_values, sign_blend_values
    ):
        sat_scale = float(base_sat_scale_mev) * float(sat_mul)
        if not math.isfinite(sat_scale) or sat_scale <= 0.0:
            continue
        mapped_rows, mapping_meta = _apply_hflavor_v1(
            rows=rows,
            before_mix=float(before_mix),
            sat_scale_mev=float(sat_scale),
            branch_pivot_mev=float(pivot),
            branch_gain=float(gain),
            sign_blend=float(sign_blend),
        )
        _, pack, split_meta = route_ab._equalized_route_audit(
            rows=mapped_rows,
            logft_sigma_proxy=float(logft_sigma_proxy),
            z_gate=float(z_gate),
            holdout_hash_modulo=int(holdout_hash_modulo),
            holdout_hash_residue=int(holdout_hash_residue),
            overfit_gap_gate=float(overfit_gap_gate),
            sigma_floor_mev=float(sigma_floor_mev),
        )
        route_eval = pack.get("route_evaluation") if isinstance(pack.get("route_evaluation"), dict) else {}
        route_b = route_eval.get("B_pmodel_proxy") if isinstance(route_eval.get("B_pmodel_proxy"), dict) else {}
        hold = route_b.get("holdout_all") if isinstance(route_b.get("holdout_all"), dict) else {}
        train = route_b.get("train_all") if isinstance(route_b.get("train_all"), dict) else {}
        overfit = route_b.get("overfit_guard") if isinstance(route_b.get("overfit_guard"), dict) else {}

        p95_q = _to_float(hold.get("p95_abs_z_qbeta"))
        p95_logft = _to_float(hold.get("p95_abs_z_logft_proxy"))
        overfit_gap_q = _to_float(overfit.get("p95_gap_qbeta"))
        max_q = _to_float(hold.get("max_abs_z_qbeta"))
        overfit_pass = bool(_as_bool(overfit.get("pass")) is True)

        candidate_id = f"cand_{candidate_counter:04d}"
        candidate_counter += 1
        candidate = {
            "candidate_id": candidate_id,
            "before_mix": float(before_mix),
            "sat_scale_mev": float(sat_scale),
            "sat_scale_multiplier": float(sat_mul),
            "branch_pivot_mev": float(pivot),
            "branch_gain": float(gain),
            "sign_blend": float(sign_blend),
            "holdout_p95_abs_z_qbeta": p95_q,
            "holdout_p95_abs_z_logft": p95_logft,
            "holdout_max_abs_z_qbeta": max_q,
            "train_p95_abs_z_qbeta": _to_float(train.get("p95_abs_z_qbeta")),
            "overfit_gap_qbeta": overfit_gap_q,
            "overfit_guard_pass": overfit_pass,
            "n_holdout_q_rows": int(hold.get("n_q_rows") or 0),
            "n_holdout_logft_rows": int(hold.get("n_logft_proxy_rows") or 0),
            "split_rows_holdout": int(split_meta.get("rows_holdout") or 0),
            "mapping_delta_abs_median_mev": _to_float(
                ((mapping_meta.get("transform_delta_abs_MeV") or {}).get("median"))
            ),
            "mapping_delta_abs_p95_mev": _to_float(((mapping_meta.get("transform_delta_abs_MeV") or {}).get("p95"))),
        }
        candidates.append(candidate)
        candidate_rows.append(candidate)

    p95_vals = [float(c["holdout_p95_abs_z_qbeta"]) for c in candidates if math.isfinite(float(c["holdout_p95_abs_z_qbeta"]))]
    gap_vals = [float(c["overfit_gap_qbeta"]) for c in candidates if math.isfinite(float(c["overfit_gap_qbeta"]))]
    p95_min = float(min(p95_vals)) if p95_vals else float("nan")
    p95_max = float(max(p95_vals)) if p95_vals else float("nan")
    gap_min = float(min(gap_vals)) if gap_vals else float("nan")
    gap_max = float(max(gap_vals)) if gap_vals else float("nan")

    points = [
        (float(c.get("holdout_p95_abs_z_qbeta", float("nan"))), float(c.get("overfit_gap_qbeta", float("nan"))))
        for c in candidates
    ]
    front_idx = set(_pareto_front_indices(points))
    for i, c in enumerate(candidates):
        c["is_pareto_front"] = bool(i in front_idx)
        n_p95 = _safe_norm(float(c.get("holdout_p95_abs_z_qbeta", float("nan"))), p95_min, p95_max)
        n_gap = _safe_norm(float(c.get("overfit_gap_qbeta", float("nan"))), gap_min, gap_max)
        c["norm_holdout_p95_abs_z_qbeta"] = n_p95
        c["norm_overfit_gap_qbeta"] = n_gap
        c["score_weighted"] = (
            float(0.70 * n_p95 + 0.30 * n_gap) if math.isfinite(n_p95) and math.isfinite(n_gap) else float("nan")
        )

    ranked = sorted(
        candidates,
        key=lambda r: (
            float(r.get("score_weighted")) if math.isfinite(float(r.get("score_weighted", float("nan")))) else float("inf"),
            float(r.get("holdout_p95_abs_z_qbeta"))
            if math.isfinite(float(r.get("holdout_p95_abs_z_qbeta", float("nan"))))
            else float("inf"),
        ),
    )
    for rank, row in enumerate(ranked, start=1):
        row["rank_weighted"] = rank

    recommended = ranked[0] if ranked else {}
    front_rows = [c for c in candidates if bool(c.get("is_pareto_front"))]
    front_rows = sorted(
        front_rows,
        key=lambda r: (
            float(r.get("holdout_p95_abs_z_qbeta", float("inf"))),
            float(r.get("overfit_gap_qbeta", float("inf"))),
        ),
    )

    return {
        "grid_spec": {
            "before_mix_values": before_mix_values,
            "sat_scale_multipliers": sat_scale_multipliers,
            "branch_pivot_values": branch_pivot_values,
            "branch_gain_values": branch_gain_values,
            "sign_blend_values": sign_blend_values,
            "base_sat_scale_mev": float(base_sat_scale_mev),
            "n_total_candidates": int(len(candidates)),
        },
        "ranges": {
            "holdout_p95_abs_z_qbeta_min": p95_min,
            "holdout_p95_abs_z_qbeta_max": p95_max,
            "overfit_gap_qbeta_min": gap_min,
            "overfit_gap_qbeta_max": gap_max,
        },
        "recommended_candidate": recommended,
        "pareto_front": front_rows,
        "top_ranked_candidates": ranked[:20],
        "all_candidates": candidate_rows,
    }


def _plot_hflavor_sweep_pareto(
    *,
    sweep: Dict[str, Any],
    baseline_holdout_p95: float,
    baseline_overfit_gap_q: float,
    current_holdout_p95: float,
    current_overfit_gap_q: float,
    out_png: Path,
) -> None:
    all_rows = sweep.get("all_candidates")
    if not isinstance(all_rows, list) or not all_rows:
        return
    x_all = [float(r.get("holdout_p95_abs_z_qbeta", float("nan"))) for r in all_rows]
    y_all = [float(r.get("overfit_gap_qbeta", float("nan"))) for r in all_rows]
    pareto = sweep.get("pareto_front") if isinstance(sweep.get("pareto_front"), list) else []
    x_pf = [float(r.get("holdout_p95_abs_z_qbeta", float("nan"))) for r in pareto]
    y_pf = [float(r.get("overfit_gap_qbeta", float("nan"))) for r in pareto]
    rec = sweep.get("recommended_candidate") if isinstance(sweep.get("recommended_candidate"), dict) else {}
    x_rec = float(rec.get("holdout_p95_abs_z_qbeta", float("nan")))
    y_rec = float(rec.get("overfit_gap_qbeta", float("nan")))

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.5), dpi=180)
    ax0, ax1 = axes

    ax0.scatter(x_all, y_all, s=18, color="#9ecae1", alpha=0.85, label="grid candidates")
    if x_pf and y_pf:
        ax0.scatter(x_pf, y_pf, s=34, color="#d62728", marker="D", label="pareto front")
    if math.isfinite(x_rec) and math.isfinite(y_rec):
        ax0.scatter([x_rec], [y_rec], s=80, color="#2ca02c", marker="*", label="recommended")
    if math.isfinite(baseline_holdout_p95) and math.isfinite(baseline_overfit_gap_q):
        ax0.scatter([baseline_holdout_p95], [baseline_overfit_gap_q], s=70, marker="x", color="#444444", label="baseline(8.7.31.1)")
    if math.isfinite(current_holdout_p95) and math.isfinite(current_overfit_gap_q):
        ax0.scatter([current_holdout_p95], [current_overfit_gap_q], s=70, marker="P", color="#1f77b4", label="current(8.7.31.3)")
    ax0.set_xlabel("holdout p95 |z_Qβ|")
    ax0.set_ylabel("overfit gap qbeta")
    ax0.set_title("H_flavor(P) sweep Pareto plane")
    ax0.grid(True, alpha=0.25, linestyle=":")
    ax0.legend(loc="best", fontsize=8)

    top_rows = sweep.get("top_ranked_candidates") if isinstance(sweep.get("top_ranked_candidates"), list) else []
    top_rows = top_rows[:8]
    labels = [str(r.get("candidate_id", "")) for r in top_rows]
    vals = [float(r.get("score_weighted", float("nan"))) for r in top_rows]
    ax1.barh(range(len(labels)), vals, color="#4c78a8")
    ax1.set_yticks(range(len(labels)), labels)
    ax1.invert_yaxis()
    ax1.set_xlabel("weighted score (0.7*p95 + 0.3*gap, normalized)")
    ax1.set_title("Top ranked candidates")
    ax1.grid(True, axis="x", alpha=0.25, linestyle=":")

    fig.suptitle("Step 8.7.31.4: holdout-fixed H_flavor(P) parameter sweep", fontsize=13)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.94])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def _build_local_correction_sweep(
    *,
    rows: List[Dict[str, Any]],
    target_classes: List[str],
    logft_sigma_proxy: float,
    z_gate: float,
    holdout_hash_modulo: int,
    holdout_hash_residue: int,
    overfit_gap_gate: float,
    sigma_floor_mev: float,
    closure_gate: Dict[str, Any],
    gain_values: List[float],
    sat_mev_values: List[float],
    blend_values: List[float],
    sign_blend_values: List[float],
    top_class_gain_boost_values: List[float],
    top_class_sign_blend_scale_values: List[float],
    sigma_low_mode_inconsistent_sat_scale_values: List[float],
    sigma_low_mode_inconsistent_zref_values: List[float],
    sigma_low_mode_inconsistent_min_blend_scale_values: List[float],
    sigma_low_mode_inconsistent_sign_blend_scale_values: List[float],
    pre_local_holdout_p95_abs_z_logft: float,
    pre_local_n_qbeta_gt3_holdout: int,
    pre_local_top_priority_q_gt3_holdout: int,
    logft_max_delta_allowed: float,
    logft_max_abs_allowed: float,
    use_qfail_guard: bool,
    qfail_max_delta_allowed: int,
    top_priority_qfail_max_delta_allowed: int,
) -> Dict[str, Any]:
    tlist = [str(x) for x in target_classes if str(x).strip()]
    tset = set(tlist)
    top_priority_class = tlist[0] if tlist else ""
    sigma_low_mode_inconsistent_classes = [str(tc) for tc in tlist if "|sigma_low|mode_inconsistent" in str(tc)]
    base_logft = float(pre_local_holdout_p95_abs_z_logft)
    base_n_qfail = int(pre_local_n_qbeta_gt3_holdout)
    base_top_qfail = int(pre_local_top_priority_q_gt3_holdout)
    use_delta_constraint = math.isfinite(base_logft) and math.isfinite(float(logft_max_delta_allowed))
    use_abs_constraint = math.isfinite(float(logft_max_abs_allowed))
    candidates: List[Dict[str, Any]] = []
    candidate_rows: List[Dict[str, Any]] = []
    candidate_counter = 0
    for (
        gain,
        sat_mev,
        blend,
        sign_blend,
        top_class_gain_boost,
        top_class_sign_blend_scale,
        sigma_low_mode_inconsistent_sat_scale,
        sigma_low_mode_inconsistent_zref,
        sigma_low_mode_inconsistent_min_blend_scale,
        sigma_low_mode_inconsistent_sign_blend_scale,
    ) in itertools.product(
        gain_values,
        sat_mev_values,
        blend_values,
        sign_blend_values,
        top_class_gain_boost_values,
        top_class_sign_blend_scale_values,
        sigma_low_mode_inconsistent_sat_scale_values,
        sigma_low_mode_inconsistent_zref_values,
        sigma_low_mode_inconsistent_min_blend_scale_values,
        sigma_low_mode_inconsistent_sign_blend_scale_values,
    ):
        class_gain_scales: Dict[str, float] = {str(tc): 1.0 for tc in tlist}
        class_sat_scales: Dict[str, float] = {str(tc): 1.0 for tc in tlist}
        class_blend_profiles: Dict[str, Dict[str, float]] = {}
        class_sign_scales: Dict[str, float] = {str(tc): 1.0 for tc in tlist}
        if top_priority_class:
            boost = _to_float(top_class_gain_boost)
            if not math.isfinite(boost) or boost <= 0.0:
                boost = 1.0
            class_gain_scales[top_priority_class] = float(boost)
            top_sign_scale = _to_float(top_class_sign_blend_scale)
            if not math.isfinite(top_sign_scale):
                top_sign_scale = 1.0
            top_sign_scale = max(0.0, float(top_sign_scale))
            class_sign_scales[top_priority_class] = float(class_sign_scales.get(top_priority_class, 1.0)) * float(
                top_sign_scale
            )
        if sigma_low_mode_inconsistent_classes:
            sat_scale = _to_float(sigma_low_mode_inconsistent_sat_scale)
            if not math.isfinite(sat_scale) or sat_scale <= 0.0:
                sat_scale = 1.0
            z_ref = _to_float(sigma_low_mode_inconsistent_zref)
            if not math.isfinite(z_ref) or z_ref <= 0.0:
                z_ref = 5.0
            min_blend_scale = _to_float(sigma_low_mode_inconsistent_min_blend_scale)
            if not math.isfinite(min_blend_scale):
                min_blend_scale = 1.0
            min_blend_scale = max(0.0, min(1.0, float(min_blend_scale)))
            sign_blend_scale = _to_float(sigma_low_mode_inconsistent_sign_blend_scale)
            if not math.isfinite(sign_blend_scale):
                sign_blend_scale = 1.0
            sign_blend_scale = max(0.0, float(sign_blend_scale))
            for tc in sigma_low_mode_inconsistent_classes:
                class_sat_scales[str(tc)] = float(sat_scale)
                class_blend_profiles[str(tc)] = {
                    "z_ref": float(z_ref),
                    "min_scale": float(min_blend_scale),
                }
                class_sign_scales[str(tc)] = float(class_sign_scales.get(str(tc), 1.0)) * float(sign_blend_scale)
        rows_corr, corr_meta = _apply_transition_class_local_correction(
            rows=rows,
            target_classes=tset,
            gain=float(gain),
            sat_mev=float(sat_mev),
            blend=float(blend),
            sign_blend=float(sign_blend),
            class_gain_scales=class_gain_scales,
            class_sat_scales=class_sat_scales,
            class_blend_profiles=class_blend_profiles,
            class_sign_scales=class_sign_scales,
        )
        _, pack, split_meta = route_ab._equalized_route_audit(
            rows=rows_corr,
            logft_sigma_proxy=float(logft_sigma_proxy),
            z_gate=float(z_gate),
            holdout_hash_modulo=int(holdout_hash_modulo),
            holdout_hash_residue=int(holdout_hash_residue),
            overfit_gap_gate=float(overfit_gap_gate),
            sigma_floor_mev=float(sigma_floor_mev),
        )
        route_eval = pack.get("route_evaluation") if isinstance(pack.get("route_evaluation"), dict) else {}
        route_b = route_eval.get("B_pmodel_proxy") if isinstance(route_eval.get("B_pmodel_proxy"), dict) else {}
        hold = route_b.get("holdout_all") if isinstance(route_b.get("holdout_all"), dict) else {}
        train = route_b.get("train_all") if isinstance(route_b.get("train_all"), dict) else {}
        overfit = route_b.get("overfit_guard") if isinstance(route_b.get("overfit_guard"), dict) else {}
        decision = _evaluate_route_b(
            route_b_eval=route_b,
            closure_gate=closure_gate,
            z_gate=float(z_gate),
            overfit_gap_gate=float(overfit_gap_gate),
        )
        decomp = _build_route_b_outlier_decomposition(
            rows_mapped=rows_corr,
            route_b_eval=route_b,
            holdout_hash_modulo=int(holdout_hash_modulo),
            holdout_hash_residue=int(holdout_hash_residue),
            z_gate=float(z_gate),
        )
        top_priority = (
            decomp.get("reduction_priority_order", [{}])[0]
            if isinstance(decomp.get("reduction_priority_order"), list) and decomp.get("reduction_priority_order")
            else {}
        )
        hold_logft = _to_float(hold.get("p95_abs_z_logft_proxy"))
        delta_logft = hold_logft - base_logft if math.isfinite(hold_logft) and math.isfinite(base_logft) else float("nan")
        pass_delta = True
        if use_delta_constraint:
            pass_delta = bool(math.isfinite(delta_logft) and delta_logft <= float(logft_max_delta_allowed))
        pass_abs = True
        if use_abs_constraint:
            pass_abs = bool(math.isfinite(hold_logft) and hold_logft <= float(logft_max_abs_allowed))
        n_qfail = int((decomp.get("counts") or {}).get("n_qbeta_gt3_holdout") or 0)
        top_qfail = int(top_priority.get("n_qbeta_gt3", 0))
        delta_n_qfail = int(n_qfail - base_n_qfail)
        delta_top_qfail = int(top_qfail - base_top_qfail)
        pass_qfail = True
        pass_top_qfail = True
        if bool(use_qfail_guard):
            pass_qfail = bool(delta_n_qfail <= int(qfail_max_delta_allowed))
            pass_top_qfail = bool(delta_top_qfail <= int(top_priority_qfail_max_delta_allowed))

        candidate_id = f"lcand_{candidate_counter:04d}"
        candidate_counter += 1
        candidate = {
            "candidate_id": candidate_id,
            "gain": float(gain),
            "sat_mev": float(sat_mev),
            "blend": float(blend),
            "sign_blend": float(sign_blend),
            "top_class_gain_boost": float(class_gain_scales.get(top_priority_class, 1.0) if top_priority_class else 1.0),
            "top_priority_class_targeted": str(top_priority_class),
            "effective_top_class_gain": float(
                float(gain) * float(class_gain_scales.get(top_priority_class, 1.0) if top_priority_class else 1.0)
            ),
            "top_class_sign_blend_scale": float(
                class_sign_scales.get(top_priority_class, 1.0) if top_priority_class else 1.0
            ),
            "effective_top_class_sign_blend": float(
                float(sign_blend) * float(class_sign_scales.get(top_priority_class, 1.0))
                if top_priority_class
                else float(sign_blend)
            ),
            "sigma_low_mode_inconsistent_sat_scale": float(
                class_sat_scales.get(sigma_low_mode_inconsistent_classes[0], 1.0)
                if sigma_low_mode_inconsistent_classes
                else 1.0
            ),
            "sigma_low_mode_inconsistent_targeted": bool(len(sigma_low_mode_inconsistent_classes) > 0),
            "effective_sigma_low_mode_sat_mev": float(
                float(sat_mev)
                * float(class_sat_scales.get(sigma_low_mode_inconsistent_classes[0], 1.0))
                if sigma_low_mode_inconsistent_classes
                else float(sat_mev)
            ),
            "sigma_low_mode_inconsistent_zref": float(
                class_blend_profiles.get(sigma_low_mode_inconsistent_classes[0], {}).get("z_ref", 5.0)
                if sigma_low_mode_inconsistent_classes
                else 5.0
            ),
            "sigma_low_mode_inconsistent_min_blend_scale": float(
                class_blend_profiles.get(sigma_low_mode_inconsistent_classes[0], {}).get("min_scale", 1.0)
                if sigma_low_mode_inconsistent_classes
                else 1.0
            ),
            "sigma_low_mode_inconsistent_sign_blend_scale": float(
                class_sign_scales.get(sigma_low_mode_inconsistent_classes[0], 1.0)
                if sigma_low_mode_inconsistent_classes
                else 1.0
            ),
            "effective_sigma_low_mode_sign_blend": float(
                float(sign_blend)
                * float(class_sign_scales.get(sigma_low_mode_inconsistent_classes[0], 1.0))
                if sigma_low_mode_inconsistent_classes
                else float(sign_blend)
            ),
            "target_classes": "|".join(sorted(tset)),
            "holdout_p95_abs_z_qbeta": _to_float(hold.get("p95_abs_z_qbeta")),
            "holdout_p95_abs_z_logft": hold_logft,
            "holdout_max_abs_z_qbeta": _to_float(hold.get("max_abs_z_qbeta")),
            "train_p95_abs_z_qbeta": _to_float(train.get("p95_abs_z_qbeta")),
            "overfit_gap_qbeta": _to_float(overfit.get("p95_gap_qbeta")),
            "overfit_guard_pass": bool(_as_bool(overfit.get("pass")) is True),
            "delta_holdout_p95_abs_z_logft_vs_pre": delta_logft,
            "logft_constraint_pass": bool(pass_delta and pass_abs),
            "delta_n_qbeta_gt3_holdout_vs_pre": int(delta_n_qfail),
            "delta_top_priority_q_gt3_vs_pre": int(delta_top_qfail),
            "qfail_constraint_pass": bool(pass_qfail),
            "top_priority_constraint_pass": bool(pass_top_qfail),
            "all_constraints_pass": bool((pass_delta and pass_abs) and pass_qfail and pass_top_qfail),
            "hard_fail_count": int(len(decision.get("hard_fail_ids") or [])),
            "watch_fail_count": int(len(decision.get("watch_fail_ids") or [])),
            "n_qbeta_gt3_holdout": int(n_qfail),
            "n_logft_gt3_holdout": int((decomp.get("counts") or {}).get("n_logft_gt3_holdout") or 0),
            "top_priority_class_after": str(top_priority.get("group_key", "")),
            "top_priority_q_gt3_after": int(top_qfail),
            "n_target_rows": int(((corr_meta.get("transform_counts") or {}).get("n_target_rows")) or 0),
            "n_changed_rows": int(((corr_meta.get("transform_counts") or {}).get("n_changed_rows")) or 0),
            "mapping_delta_abs_median_mev": _to_float(((corr_meta.get("transform_delta_abs_MeV") or {}).get("median"))),
            "mapping_delta_abs_p95_mev": _to_float(((corr_meta.get("transform_delta_abs_MeV") or {}).get("p95"))),
            "split_rows_holdout": int(split_meta.get("rows_holdout") or 0),
        }
        candidates.append(candidate)
        candidate_rows.append(candidate)

    p95_vals = [float(c["holdout_p95_abs_z_qbeta"]) for c in candidates if math.isfinite(float(c["holdout_p95_abs_z_qbeta"]))]
    logft_vals = [float(c["holdout_p95_abs_z_logft"]) for c in candidates if math.isfinite(float(c["holdout_p95_abs_z_logft"]))]
    gap_vals = [float(c["overfit_gap_qbeta"]) for c in candidates if math.isfinite(float(c["overfit_gap_qbeta"]))]
    fail_vals = [float(c["n_qbeta_gt3_holdout"]) for c in candidates if math.isfinite(float(c["n_qbeta_gt3_holdout"]))]
    p95_min = float(min(p95_vals)) if p95_vals else float("nan")
    p95_max = float(max(p95_vals)) if p95_vals else float("nan")
    logft_min = float(min(logft_vals)) if logft_vals else float("nan")
    logft_max = float(max(logft_vals)) if logft_vals else float("nan")
    gap_min = float(min(gap_vals)) if gap_vals else float("nan")
    gap_max = float(max(gap_vals)) if gap_vals else float("nan")
    fail_min = float(min(fail_vals)) if fail_vals else float("nan")
    fail_max = float(max(fail_vals)) if fail_vals else float("nan")

    points = [
        (float(c.get("holdout_p95_abs_z_qbeta", float("nan"))), float(c.get("overfit_gap_qbeta", float("nan"))))
        for c in candidates
    ]
    front_idx = set(_pareto_front_indices(points))
    for i, c in enumerate(candidates):
        c["is_pareto_front"] = bool(i in front_idx)
        n_p95 = _safe_norm(float(c.get("holdout_p95_abs_z_qbeta", float("nan"))), p95_min, p95_max)
        n_logft = _safe_norm(float(c.get("holdout_p95_abs_z_logft", float("nan"))), logft_min, logft_max)
        n_gap = _safe_norm(float(c.get("overfit_gap_qbeta", float("nan"))), gap_min, gap_max)
        n_fail = _safe_norm(float(c.get("n_qbeta_gt3_holdout", float("nan"))), fail_min, fail_max)
        c["norm_holdout_p95_abs_z_qbeta"] = n_p95
        c["norm_holdout_p95_abs_z_logft"] = n_logft
        c["norm_overfit_gap_qbeta"] = n_gap
        c["norm_n_qbeta_gt3_holdout"] = n_fail
        c["score_weighted"] = (
            float(0.45 * n_p95 + 0.30 * n_logft + 0.20 * n_gap + 0.05 * n_fail)
            if math.isfinite(n_p95) and math.isfinite(n_logft) and math.isfinite(n_gap) and math.isfinite(n_fail)
            else float("nan")
        )

    ranked = sorted(
        candidates,
        key=lambda r: (
            float(r.get("score_weighted")) if math.isfinite(float(r.get("score_weighted", float("nan")))) else float("inf"),
            float(r.get("holdout_p95_abs_z_qbeta"))
            if math.isfinite(float(r.get("holdout_p95_abs_z_qbeta", float("nan"))))
            else float("inf"),
        ),
    )
    for rank, row in enumerate(ranked, start=1):
        row["rank_weighted"] = rank

    constrained_logft = [r for r in ranked if bool(r.get("logft_constraint_pass"))]
    constrained_all = [r for r in ranked if bool(r.get("all_constraints_pass"))]
    constrained_all = sorted(
        constrained_all,
        key=lambda r: (
            float(r.get("overfit_gap_qbeta", float("inf"))),
            float(r.get("n_qbeta_gt3_holdout", float("inf"))),
            float(r.get("top_priority_q_gt3_after", float("inf"))),
            float(r.get("holdout_p95_abs_z_qbeta", float("inf"))),
            float(r.get("score_weighted", float("inf"))),
        ),
    )
    for rank, row in enumerate(constrained_all, start=1):
        row["rank_constrained"] = rank
    for row in ranked:
        if "rank_constrained" not in row:
            row["rank_constrained"] = ""

    if constrained_all:
        recommended = constrained_all[0]
        selection_mode = "logft_qfail_constrained"
    elif constrained_logft:
        recommended = constrained_logft[0]
        selection_mode = "fallback_logft_only"
    else:
        recommended = ranked[0] if ranked else {}
        selection_mode = "fallback_unconstrained"

    recommended_unconstrained = ranked[0] if ranked else {}
    if constrained_all:
        ranked_selected = constrained_all
    elif constrained_logft:
        ranked_selected = constrained_logft
    else:
        ranked_selected = ranked
    front_rows = [c for c in candidates if bool(c.get("is_pareto_front"))]
    front_rows = sorted(
        front_rows,
        key=lambda r: (
            float(r.get("holdout_p95_abs_z_qbeta", float("inf"))),
            float(r.get("overfit_gap_qbeta", float("inf"))),
        ),
    )

    return {
        "status": "completed",
        "target_classes": sorted([str(x) for x in tset]),
        "grid_spec": {
            "gain_values": gain_values,
            "sat_mev_values": sat_mev_values,
            "blend_values": blend_values,
            "sign_blend_values": sign_blend_values,
            "top_class_gain_boost_values": top_class_gain_boost_values,
            "top_class_sign_blend_scale_values": top_class_sign_blend_scale_values,
            "sigma_low_mode_inconsistent_sat_scale_values": sigma_low_mode_inconsistent_sat_scale_values,
            "sigma_low_mode_inconsistent_zref_values": sigma_low_mode_inconsistent_zref_values,
            "sigma_low_mode_inconsistent_min_blend_scale_values": sigma_low_mode_inconsistent_min_blend_scale_values,
            "sigma_low_mode_inconsistent_sign_blend_scale_values": sigma_low_mode_inconsistent_sign_blend_scale_values,
            "top_priority_class": str(top_priority_class),
            "logft_max_delta_allowed": float(logft_max_delta_allowed),
            "logft_max_abs_allowed": float(logft_max_abs_allowed),
            "use_qfail_guard": bool(use_qfail_guard),
            "qfail_max_delta_allowed": int(qfail_max_delta_allowed),
            "top_priority_qfail_max_delta_allowed": int(top_priority_qfail_max_delta_allowed),
            "n_total_candidates": int(len(candidates)),
        },
        "ranges": {
            "holdout_p95_abs_z_qbeta_min": p95_min,
            "holdout_p95_abs_z_qbeta_max": p95_max,
            "holdout_p95_abs_z_logft_min": logft_min,
            "holdout_p95_abs_z_logft_max": logft_max,
            "overfit_gap_qbeta_min": gap_min,
            "overfit_gap_qbeta_max": gap_max,
            "n_qbeta_gt3_holdout_min": fail_min,
            "n_qbeta_gt3_holdout_max": fail_max,
        },
        "selection_policy": {
            "mode": selection_mode,
            "logft_base_pre_local": base_logft,
            "logft_delta_max_allowed": float(logft_max_delta_allowed),
            "logft_abs_max_allowed": float(logft_max_abs_allowed),
            "qfail_base_pre_local": int(base_n_qfail),
            "top_priority_qfail_base_pre_local": int(base_top_qfail),
            "qfail_max_delta_allowed": int(qfail_max_delta_allowed),
            "top_priority_qfail_max_delta_allowed": int(top_priority_qfail_max_delta_allowed),
            "qfail_guard_enabled": bool(use_qfail_guard),
            "delta_constraint_enabled": bool(use_delta_constraint),
            "abs_constraint_enabled": bool(use_abs_constraint),
            "n_candidates_total": int(len(candidates)),
            "n_candidates_logft_pass": int(len(constrained_logft)),
            "n_candidates_all_constraints_pass": int(len(constrained_all)),
        },
        "recommended_candidate": recommended,
        "recommended_candidate_unconstrained": recommended_unconstrained,
        "pareto_front": front_rows,
        "top_ranked_candidates": ranked_selected[:20],
        "top_ranked_candidates_unconstrained": ranked[:20],
        "all_candidates": candidate_rows,
    }


def _plot_local_correction_sweep_pareto(
    *,
    sweep: Dict[str, Any],
    pre_local_holdout_p95: float,
    pre_local_overfit_gap_q: float,
    final_holdout_p95: float,
    final_overfit_gap_q: float,
    out_png: Path,
) -> None:
    all_rows = sweep.get("all_candidates")
    if not isinstance(all_rows, list) or not all_rows:
        return
    x_all = [float(r.get("holdout_p95_abs_z_qbeta", float("nan"))) for r in all_rows]
    y_all = [float(r.get("overfit_gap_qbeta", float("nan"))) for r in all_rows]
    pareto = sweep.get("pareto_front") if isinstance(sweep.get("pareto_front"), list) else []
    x_pf = [float(r.get("holdout_p95_abs_z_qbeta", float("nan"))) for r in pareto]
    y_pf = [float(r.get("overfit_gap_qbeta", float("nan"))) for r in pareto]
    rec = sweep.get("recommended_candidate") if isinstance(sweep.get("recommended_candidate"), dict) else {}
    x_rec = float(rec.get("holdout_p95_abs_z_qbeta", float("nan")))
    y_rec = float(rec.get("overfit_gap_qbeta", float("nan")))

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.5), dpi=180)
    ax0, ax1 = axes

    ax0.scatter(x_all, y_all, s=18, color="#c7c7c7", alpha=0.85, label="local-correction candidates")
    if x_pf and y_pf:
        ax0.scatter(x_pf, y_pf, s=34, color="#d62728", marker="D", label="pareto front")
    if math.isfinite(x_rec) and math.isfinite(y_rec):
        ax0.scatter([x_rec], [y_rec], s=90, color="#2ca02c", marker="*", label="recommended")
    if math.isfinite(pre_local_holdout_p95) and math.isfinite(pre_local_overfit_gap_q):
        ax0.scatter([pre_local_holdout_p95], [pre_local_overfit_gap_q], s=70, marker="P", color="#1f77b4", label="pre-local")
    if math.isfinite(final_holdout_p95) and math.isfinite(final_overfit_gap_q):
        ax0.scatter([final_holdout_p95], [final_overfit_gap_q], s=70, marker="X", color="#9467bd", label="final-selected")
    ax0.set_xlabel("holdout p95 |z_Qβ|")
    ax0.set_ylabel("overfit gap qbeta")
    ax0.set_title("Local correction sweep Pareto plane")
    ax0.grid(True, alpha=0.25, linestyle=":")
    ax0.legend(loc="best", fontsize=8)

    top_rows = sweep.get("top_ranked_candidates") if isinstance(sweep.get("top_ranked_candidates"), list) else []
    top_rows = top_rows[:8]
    labels = [str(r.get("candidate_id", "")) for r in top_rows]
    vals = [float(r.get("score_weighted", float("nan"))) for r in top_rows]
    ax1.barh(range(len(labels)), vals, color="#4c78a8")
    ax1.set_yticks(range(len(labels)), labels)
    ax1.invert_yaxis()
    ax1.set_xlabel("weighted score (0.45*p95Q + 0.30*p95logft + 0.20*gap + 0.05*q-fail)")
    ax1.set_title("Top ranked local-correction candidates")
    ax1.grid(True, axis="x", alpha=0.25, linestyle=":")

    fig.suptitle("Step 8.7.31.5: targeted local correction sweep", fontsize=13)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.94])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def _build_residue_robustness(
    *,
    rows_mapped: List[Dict[str, Any]],
    closure_gate: Dict[str, Any],
    logft_sigma_proxy: float,
    z_gate: float,
    holdout_hash_modulo: int,
    holdout_hash_residues: List[int],
    overfit_gap_gate: float,
    sigma_floor_mev: float,
    reference_residue: int,
) -> Dict[str, Any]:
    modulo = max(1, int(holdout_hash_modulo))
    residues_norm = sorted({int(r) % modulo for r in holdout_hash_residues})
    rows_out: List[Dict[str, Any]] = []
    for residue in residues_norm:
        _, pack, split_meta = route_ab._equalized_route_audit(
            rows=rows_mapped,
            logft_sigma_proxy=float(logft_sigma_proxy),
            z_gate=float(z_gate),
            holdout_hash_modulo=int(modulo),
            holdout_hash_residue=int(residue),
            overfit_gap_gate=float(overfit_gap_gate),
            sigma_floor_mev=float(sigma_floor_mev),
        )
        route_eval = pack.get("route_evaluation") if isinstance(pack.get("route_evaluation"), dict) else {}
        route_b = route_eval.get("B_pmodel_proxy") if isinstance(route_eval.get("B_pmodel_proxy"), dict) else {}
        hold = route_b.get("holdout_all") if isinstance(route_b.get("holdout_all"), dict) else {}
        overfit = route_b.get("overfit_guard") if isinstance(route_b.get("overfit_guard"), dict) else {}
        decision = _evaluate_route_b(
            route_b_eval=route_b,
            closure_gate=closure_gate,
            z_gate=float(z_gate),
            overfit_gap_gate=float(overfit_gap_gate),
        )
        decomp = _build_route_b_outlier_decomposition(
            rows_mapped=rows_mapped,
            route_b_eval=route_b,
            holdout_hash_modulo=int(modulo),
            holdout_hash_residue=int(residue),
            z_gate=float(z_gate),
        )
        counts = decomp.get("counts") if isinstance(decomp.get("counts"), dict) else {}
        top = (
            (decomp.get("reduction_priority_order") or [{}])[0]
            if isinstance(decomp.get("reduction_priority_order"), list) and (decomp.get("reduction_priority_order") or [])
            else {}
        )
        rows_out.append(
            {
                "residue": int(residue),
                "rows_holdout": int(split_meta.get("rows_holdout") or 0),
                "holdout_p95_abs_z_qbeta": _to_float(hold.get("p95_abs_z_qbeta")),
                "holdout_p95_abs_z_logft": _to_float(hold.get("p95_abs_z_logft_proxy")),
                "holdout_max_abs_z_qbeta": _to_float(hold.get("max_abs_z_qbeta")),
                "overfit_gap_qbeta": _to_float(overfit.get("p95_gap_qbeta")),
                "n_qbeta_gt3_holdout": int(counts.get("n_qbeta_gt3_holdout") or 0),
                "n_logft_gt3_holdout": int(counts.get("n_logft_gt3_holdout") or 0),
                "top_priority_class": str(top.get("group_key", "")),
                "top_priority_q_gt3": int(top.get("n_qbeta_gt3", 0)),
                "hard_fail_count": int(len(decision.get("hard_fail_ids") or [])),
                "watch_fail_count": int(len(decision.get("watch_fail_ids") or [])),
                "overall_status": str(decision.get("overall_status", "")),
            }
        )

    if not rows_out:
        return {
            "status": "not_run",
            "reason": "no_residues",
            "rows": [],
            "summary": {},
            "reference_residue": int(reference_residue),
            "residue_list": residues_norm,
        }

    ref_row = next((r for r in rows_out if int(r.get("residue", -1)) == int(reference_residue)), rows_out[0])
    ref_residue = int(ref_row.get("residue", 0))
    ref_q = _to_float(ref_row.get("holdout_p95_abs_z_qbeta"))
    ref_logft = _to_float(ref_row.get("holdout_p95_abs_z_logft"))
    ref_gap = _to_float(ref_row.get("overfit_gap_qbeta"))
    ref_nq = int(ref_row.get("n_qbeta_gt3_holdout") or 0)
    ref_nlog = int(ref_row.get("n_logft_gt3_holdout") or 0)
    ref_topq = int(ref_row.get("top_priority_q_gt3") or 0)

    for row in rows_out:
        row["delta_holdout_p95_abs_z_qbeta_vs_ref"] = _to_float(row.get("holdout_p95_abs_z_qbeta")) - ref_q
        row["delta_holdout_p95_abs_z_logft_vs_ref"] = _to_float(row.get("holdout_p95_abs_z_logft")) - ref_logft
        row["delta_overfit_gap_qbeta_vs_ref"] = _to_float(row.get("overfit_gap_qbeta")) - ref_gap
        row["delta_n_qbeta_gt3_holdout_vs_ref"] = int(row.get("n_qbeta_gt3_holdout") or 0) - ref_nq
        row["delta_n_logft_gt3_holdout_vs_ref"] = int(row.get("n_logft_gt3_holdout") or 0) - ref_nlog
        row["delta_top_priority_q_gt3_vs_ref"] = int(row.get("top_priority_q_gt3") or 0) - ref_topq

    q_vals = [float(r.get("holdout_p95_abs_z_qbeta", float("nan"))) for r in rows_out if math.isfinite(_to_float(r.get("holdout_p95_abs_z_qbeta")))]
    log_vals = [float(r.get("holdout_p95_abs_z_logft", float("nan"))) for r in rows_out if math.isfinite(_to_float(r.get("holdout_p95_abs_z_logft")))]
    gap_vals = [float(r.get("overfit_gap_qbeta", float("nan"))) for r in rows_out if math.isfinite(_to_float(r.get("overfit_gap_qbeta")))]
    all_qfail_nonreg = all(int(r.get("delta_n_qbeta_gt3_holdout_vs_ref") or 0) <= 0 for r in rows_out)
    all_logft_count_nonreg = all(int(r.get("delta_n_logft_gt3_holdout_vs_ref") or 0) <= 0 for r in rows_out)
    all_top_nonreg = all(int(r.get("delta_top_priority_q_gt3_vs_ref") or 0) <= 0 for r in rows_out)
    all_logft_nonworse = all(_to_float(r.get("delta_holdout_p95_abs_z_logft_vs_ref")) <= 0.0 for r in rows_out)
    all_overfit_nonworse = all(_to_float(r.get("delta_overfit_gap_qbeta_vs_ref")) <= 0.0 for r in rows_out)

    summary = {
        "n_residues": int(len(rows_out)),
        "reference_residue": int(ref_residue),
        "all_qfail_nonregression_vs_ref": bool(all_qfail_nonreg),
        "all_logft_count_nonregression_vs_ref": bool(all_logft_count_nonreg),
        "all_top_priority_nonregression_vs_ref": bool(all_top_nonreg),
        "all_logft_nonworsening_vs_ref": bool(all_logft_nonworse),
        "all_overfit_nonworsening_vs_ref": bool(all_overfit_nonworse),
        "max_positive_delta_overfit_gap_qbeta_vs_ref": float(
            max(max(0.0, _to_float(r.get("delta_overfit_gap_qbeta_vs_ref"))) for r in rows_out)
        ),
        "max_positive_delta_n_qbeta_gt3_holdout_vs_ref": int(
            max(int(r.get("delta_n_qbeta_gt3_holdout_vs_ref") or 0) for r in rows_out)
        ),
        "max_positive_delta_n_logft_gt3_holdout_vs_ref": int(
            max(int(r.get("delta_n_logft_gt3_holdout_vs_ref") or 0) for r in rows_out)
        ),
        "max_positive_delta_top_priority_q_gt3_vs_ref": int(
            max(int(r.get("delta_top_priority_q_gt3_vs_ref") or 0) for r in rows_out)
        ),
        "holdout_p95_abs_z_qbeta_spread": (float(max(q_vals) - min(q_vals)) if q_vals else float("nan")),
        "holdout_p95_abs_z_logft_spread": (float(max(log_vals) - min(log_vals)) if log_vals else float("nan")),
        "overfit_gap_qbeta_spread": (float(max(gap_vals) - min(gap_vals)) if gap_vals else float("nan")),
    }

    return {
        "status": "completed",
        "reference_residue": int(ref_residue),
        "residue_list": residues_norm,
        "rows": rows_out,
        "summary": summary,
    }


def _build_localcorr_residue_reweight(
    *,
    rows_mapped: List[Dict[str, Any]],
    target_classes: List[str],
    candidate_rows: List[Dict[str, Any]],
    closure_gate: Dict[str, Any],
    logft_sigma_proxy: float,
    z_gate: float,
    holdout_hash_modulo: int,
    holdout_hash_residues: List[int],
    overfit_gap_gate: float,
    sigma_floor_mev: float,
    reference_residue: int,
    max_candidates: int,
    pre_local_holdout_p95_abs_z_qbeta: float,
    pre_local_overfit_gap_qbeta: float,
    pre_local_n_logft_gt3_holdout: int,
    use_qbeta_overfit_guard: bool,
    qbeta_max_delta_allowed: float,
    overfit_max_delta_allowed: float,
    use_logft_rootcause_guard: bool,
    logft_count_max_delta_allowed: int,
    residue_logft_count_max_delta_allowed: int,
    require_residue_logft_nonworsening: bool,
    use_logft_rootcause_refreeze: bool,
    use_logft_rootcause_retune: bool,
    logft_rootcause_retune_target_combined_pass_min: int,
    logft_rootcause_retune_max_logft_count_delta_allowed: int,
    logft_rootcause_retune_max_residue_logft_count_delta_allowed: int,
    use_residue_dual_guard: bool,
    residue_top_priority_max_delta_allowed: int,
    residue_overfit_max_delta_allowed: float,
    use_residue_dual_refreeze: bool,
    use_residue_dual_retune: bool,
    residue_dual_retune_top_weight: float,
    residue_dual_retune_overfit_weight: float,
    residue_dual_retune_max_top_delta_allowed: int,
    residue_dual_retune_max_overfit_delta_allowed: float,
    classwise_residue_norm_blend: float,
    use_class_key_threshold_guard: bool,
    class_logft_count_max_delta_by_key: Dict[str, int],
    class_residue_logft_count_max_delta_by_key: Dict[str, int],
    class_residue_top_priority_max_delta_by_key: Dict[str, int],
    class_residue_overfit_max_delta_by_key: Dict[str, float],
    class_top_priority_weight_by_key: Dict[str, float],
    class_logft_weight_by_key: Dict[str, float],
) -> Dict[str, Any]:
    target_class_list = [str(x) for x in target_classes if str(x).strip()]
    tset = set(target_class_list)
    if not rows_mapped or not tset:
        return {
            "status": "not_run",
            "reason": "empty_rows_or_target_classes",
            "recommended_candidate": {},
            "all_candidates": [],
            "top_ranked_candidates": [],
            "selection_policy": {"mode": "not_run"},
        }

    rows_in = [dict(r) for r in candidate_rows if isinstance(r, dict)]
    if not rows_in:
        return {
            "status": "not_run",
            "reason": "no_candidate_rows",
            "recommended_candidate": {},
            "all_candidates": [],
            "top_ranked_candidates": [],
            "selection_policy": {"mode": "not_run"},
        }

    limit = max(1, int(max_candidates))

    def _rank_key(row: Dict[str, Any]) -> Tuple[float, float, float]:
        rank_c = row.get("rank_constrained")
        rank_w = row.get("rank_weighted")
        r_c = float(rank_c) if str(rank_c).strip() != "" and math.isfinite(_to_float(rank_c)) else float("inf")
        r_w = float(rank_w) if math.isfinite(_to_float(rank_w)) else float("inf")
        p95_q = float(row.get("holdout_p95_abs_z_qbeta", float("inf")))
        return (r_c, r_w, p95_q)

    rows_in = sorted(rows_in, key=_rank_key)[:limit]
    eval_rows: List[Dict[str, Any]] = []
    class_logft_limit_map: Dict[str, int] = {
        str(k): int(max(0, int(v)))
        for k, v in dict(class_logft_count_max_delta_by_key or {}).items()
        if str(k).strip()
    }
    class_residue_logft_limit_map: Dict[str, int] = {
        str(k): int(max(0, int(v)))
        for k, v in dict(class_residue_logft_count_max_delta_by_key or {}).items()
        if str(k).strip()
    }
    class_residue_top_limit_map: Dict[str, int] = {
        str(k): int(max(0, int(v)))
        for k, v in dict(class_residue_top_priority_max_delta_by_key or {}).items()
        if str(k).strip()
    }
    class_residue_overfit_limit_map: Dict[str, float] = {
        str(k): float(max(0.0, _to_float(v)))
        for k, v in dict(class_residue_overfit_max_delta_by_key or {}).items()
        if str(k).strip() and math.isfinite(_to_float(v))
    }

    def _row_class_key(row: Dict[str, Any]) -> str:
        class_key = str(row.get("top_priority_class_after", "")).strip()
        if not class_key:
            class_key = str(row.get("residue_class_key", "")).strip()
        return class_key or "__unknown__"

    def _effective_limit_int(row: Dict[str, Any], *, base_limit: int, by_key: Dict[str, int]) -> int:
        if not bool(use_class_key_threshold_guard):
            return int(base_limit)
        class_key = _row_class_key(row)
        if class_key in by_key:
            return int(max(0, int(by_key.get(class_key, base_limit))))
        return int(base_limit)

    def _effective_limit_float(row: Dict[str, Any], *, base_limit: float, by_key: Dict[str, float]) -> float:
        if not bool(use_class_key_threshold_guard):
            return float(base_limit)
        class_key = _row_class_key(row)
        if class_key in by_key:
            return float(max(0.0, _to_float(by_key.get(class_key, base_limit))))
        return float(base_limit)

    for cand in rows_in:
        class_gain_scales = _build_candidate_class_gain_scales(candidate=cand, target_classes=target_class_list)
        class_sat_scales = _build_candidate_class_sat_scales(candidate=cand, target_classes=target_class_list)
        class_blend_profiles = _build_candidate_class_blend_profiles(candidate=cand, target_classes=target_class_list)
        class_sign_scales = _build_candidate_class_sign_scales(candidate=cand, target_classes=target_class_list)
        rows_corr, _ = _apply_transition_class_local_correction(
            rows=rows_mapped,
            target_classes=tset,
            gain=float(cand.get("gain", 1.0)),
            sat_mev=float(cand.get("sat_mev", 5.0)),
            blend=float(cand.get("blend", 0.0)),
            sign_blend=float(cand.get("sign_blend", 0.0)),
            class_gain_scales=class_gain_scales,
            class_sat_scales=class_sat_scales,
            class_blend_profiles=class_blend_profiles,
            class_sign_scales=class_sign_scales,
        )
        residue_pack = _build_residue_robustness(
            rows_mapped=rows_corr,
            closure_gate=closure_gate,
            logft_sigma_proxy=float(logft_sigma_proxy),
            z_gate=float(z_gate),
            holdout_hash_modulo=int(holdout_hash_modulo),
            holdout_hash_residues=[int(x) for x in holdout_hash_residues],
            overfit_gap_gate=float(overfit_gap_gate),
            sigma_floor_mev=float(sigma_floor_mev),
            reference_residue=int(reference_residue),
        )
        if str(residue_pack.get("status")) != "completed":
            continue
        summary = residue_pack.get("summary") if isinstance(residue_pack.get("summary"), dict) else {}
        row = dict(cand)
        row["residue_n"] = int(summary.get("n_residues") or 0)
        row["residue_reference"] = int(summary.get("reference_residue") or int(reference_residue))
        row["residue_all_qfail_nonregression"] = bool(summary.get("all_qfail_nonregression_vs_ref"))
        row["residue_all_logft_count_nonregression"] = bool(summary.get("all_logft_count_nonregression_vs_ref"))
        row["residue_all_top_priority_nonregression"] = bool(summary.get("all_top_priority_nonregression_vs_ref"))
        row["residue_all_logft_nonworsening"] = bool(summary.get("all_logft_nonworsening_vs_ref"))
        row["residue_all_overfit_nonworsening"] = bool(summary.get("all_overfit_nonworsening_vs_ref"))
        row["residue_max_pos_delta_n_qbeta_gt3"] = max(
            0, int(summary.get("max_positive_delta_n_qbeta_gt3_holdout_vs_ref") or 0)
        )
        row["residue_max_pos_delta_n_logft_gt3"] = max(
            0, int(summary.get("max_positive_delta_n_logft_gt3_holdout_vs_ref") or 0)
        )
        row["residue_max_pos_delta_top_priority_q_gt3"] = max(
            0, int(summary.get("max_positive_delta_top_priority_q_gt3_vs_ref") or 0)
        )
        row["residue_max_pos_delta_overfit_gap_qbeta"] = max(
            0.0, _to_float(summary.get("max_positive_delta_overfit_gap_qbeta_vs_ref"))
        )
        row["residue_holdout_p95_abs_z_qbeta_spread"] = _to_float(summary.get("holdout_p95_abs_z_qbeta_spread"))
        row["residue_holdout_p95_abs_z_logft_spread"] = _to_float(summary.get("holdout_p95_abs_z_logft_spread"))
        row["residue_overfit_gap_qbeta_spread"] = _to_float(summary.get("overfit_gap_qbeta_spread"))
        d_q_pre = _to_float(row.get("holdout_p95_abs_z_qbeta")) - float(pre_local_holdout_p95_abs_z_qbeta)
        d_gap_pre = _to_float(row.get("overfit_gap_qbeta")) - float(pre_local_overfit_gap_qbeta)
        d_nlog_pre = int(row.get("n_logft_gt3_holdout") or 0) - int(pre_local_n_logft_gt3_holdout)
        row["delta_holdout_p95_abs_z_qbeta_vs_pre"] = d_q_pre
        row["delta_overfit_gap_qbeta_vs_pre"] = d_gap_pre
        row["delta_n_logft_gt3_holdout_vs_pre"] = int(d_nlog_pre)
        row["qbeta_nonworsening_pass"] = bool(d_q_pre <= float(qbeta_max_delta_allowed))
        row["overfit_nonworsening_pass"] = bool(d_gap_pre <= float(overfit_max_delta_allowed))
        row["qbeta_overfit_guard_pass"] = bool(
            bool(row.get("qbeta_nonworsening_pass")) and bool(row.get("overfit_nonworsening_pass"))
        )
        logft_count_limit_eff = _effective_limit_int(
            row,
            base_limit=int(logft_count_max_delta_allowed),
            by_key=class_logft_limit_map,
        )
        residue_logft_count_limit_eff = _effective_limit_int(
            row,
            base_limit=int(residue_logft_count_max_delta_allowed),
            by_key=class_residue_logft_limit_map,
        )
        row["logft_count_max_delta_effective"] = int(logft_count_limit_eff)
        row["residue_logft_count_max_delta_effective"] = int(residue_logft_count_limit_eff)
        row["class_key_threshold_guard_applied"] = bool(
            bool(use_class_key_threshold_guard)
            and (
                _row_class_key(row) in class_logft_limit_map
                or _row_class_key(row) in class_residue_logft_limit_map
                or _row_class_key(row) in class_residue_top_limit_map
                or _row_class_key(row) in class_residue_overfit_limit_map
            )
        )
        row["prelocal_logft_count_guard_pass"] = bool(d_nlog_pre <= int(logft_count_limit_eff))
        row["residue_logft_count_guard_pass"] = bool(
            int(row.get("residue_max_pos_delta_n_logft_gt3") or 0) <= int(residue_logft_count_limit_eff)
        )
        row["residue_logft_nonworsening_guard_pass"] = bool(
            bool(row.get("residue_all_logft_nonworsening")) if bool(require_residue_logft_nonworsening) else True
        )
        row["logft_rootcause_guard_pass"] = bool(
            bool(row.get("prelocal_logft_count_guard_pass"))
            and bool(row.get("residue_logft_count_guard_pass"))
            and bool(row.get("residue_logft_nonworsening_guard_pass"))
        )
        eval_rows.append(row)

    if not eval_rows:
        return {
            "status": "not_run",
            "reason": "no_evaluable_candidates",
            "recommended_candidate": {},
            "all_candidates": [],
            "top_ranked_candidates": [],
            "selection_policy": {"mode": "not_run"},
        }

    top_vals = [float(r.get("residue_max_pos_delta_top_priority_q_gt3", float("nan"))) for r in eval_rows]
    logcnt_vals = [float(r.get("residue_max_pos_delta_n_logft_gt3", float("nan"))) for r in eval_rows]
    qfail_vals = [float(r.get("residue_max_pos_delta_n_qbeta_gt3", float("nan"))) for r in eval_rows]
    logspread_vals = [float(r.get("residue_holdout_p95_abs_z_logft_spread", float("nan"))) for r in eval_rows]
    qspread_vals = [float(r.get("residue_holdout_p95_abs_z_qbeta_spread", float("nan"))) for r in eval_rows]
    dq_pre_vals = [float(max(0.0, _to_float(r.get("delta_holdout_p95_abs_z_qbeta_vs_pre")))) for r in eval_rows]
    dgap_pre_vals = [float(max(0.0, _to_float(r.get("delta_overfit_gap_qbeta_vs_pre")))) for r in eval_rows]

    top_min, top_max = float(min(top_vals)), float(max(top_vals))
    logcnt_min, logcnt_max = float(min(logcnt_vals)), float(max(logcnt_vals))
    qfail_min, qfail_max = float(min(qfail_vals)), float(max(qfail_vals))
    logspread_min, logspread_max = float(min(logspread_vals)), float(max(logspread_vals))
    qspread_min, qspread_max = float(min(qspread_vals)), float(max(qspread_vals))
    dq_pre_min, dq_pre_max = float(min(dq_pre_vals)), float(max(dq_pre_vals))
    dgap_pre_min, dgap_pre_max = float(min(dgap_pre_vals)), float(max(dgap_pre_vals))
    classwise_blend = float(max(0.0, min(1.0, _to_float(classwise_residue_norm_blend))))
    top_weight_map: Dict[str, float] = {
        str(k): float(max(0.0, _to_float(v)))
        for k, v in dict(class_top_priority_weight_by_key or {}).items()
        if str(k).strip() and math.isfinite(_to_float(v))
    }
    logft_weight_map: Dict[str, float] = {
        str(k): float(max(0.0, _to_float(v)))
        for k, v in dict(class_logft_weight_by_key or {}).items()
        if str(k).strip() and math.isfinite(_to_float(v))
    }

    weights = {
        "max_pos_delta_top_priority_q_gt3": 0.30,
        "max_pos_delta_n_logft_gt3": 0.18,
        "logft_spread": 0.10,
        "max_pos_delta_n_qbeta_gt3": 0.08,
        "qbeta_spread": 0.05,
        "delta_holdout_p95_abs_z_qbeta_vs_pre": 0.17,
        "delta_overfit_gap_qbeta_vs_pre": 0.12,
    }
    class_groups: Dict[str, List[Dict[str, Any]]] = {}
    for row in eval_rows:
        class_key = str(row.get("top_priority_class_after", "")).strip() or "__unknown__"
        class_groups.setdefault(class_key, []).append(row)

    class_ranges: Dict[str, Dict[str, Tuple[float, float]]] = {}
    for class_key, rows_cls in class_groups.items():
        if not rows_cls:
            continue
        vals_top = [float(r.get("residue_max_pos_delta_top_priority_q_gt3", float("nan"))) for r in rows_cls]
        vals_logcnt = [float(r.get("residue_max_pos_delta_n_logft_gt3", float("nan"))) for r in rows_cls]
        vals_qfail = [float(r.get("residue_max_pos_delta_n_qbeta_gt3", float("nan"))) for r in rows_cls]
        vals_logspread = [float(r.get("residue_holdout_p95_abs_z_logft_spread", float("nan"))) for r in rows_cls]
        vals_qspread = [float(r.get("residue_holdout_p95_abs_z_qbeta_spread", float("nan"))) for r in rows_cls]
        vals_dq = [float(max(0.0, _to_float(r.get("delta_holdout_p95_abs_z_qbeta_vs_pre")))) for r in rows_cls]
        vals_dgap = [float(max(0.0, _to_float(r.get("delta_overfit_gap_qbeta_vs_pre")))) for r in rows_cls]
        class_ranges[class_key] = {
            "top": (float(min(vals_top)), float(max(vals_top))),
            "logcnt": (float(min(vals_logcnt)), float(max(vals_logcnt))),
            "qfail": (float(min(vals_qfail)), float(max(vals_qfail))),
            "logspread": (float(min(vals_logspread)), float(max(vals_logspread))),
            "qspread": (float(min(vals_qspread)), float(max(vals_qspread))),
            "dq_pre": (float(min(vals_dq)), float(max(vals_dq))),
            "dgap_pre": (float(min(vals_dgap)), float(max(vals_dgap))),
        }

    for row in eval_rows:
        raw_top = float(row.get("residue_max_pos_delta_top_priority_q_gt3", float("nan")))
        raw_logcnt = float(row.get("residue_max_pos_delta_n_logft_gt3", float("nan")))
        raw_qfail = float(row.get("residue_max_pos_delta_n_qbeta_gt3", float("nan")))
        raw_logspread = float(row.get("residue_holdout_p95_abs_z_logft_spread", float("nan")))
        raw_qspread = float(row.get("residue_holdout_p95_abs_z_qbeta_spread", float("nan")))
        raw_dq_pre = float(max(0.0, _to_float(row.get("delta_holdout_p95_abs_z_qbeta_vs_pre"))))
        raw_dgap_pre = float(max(0.0, _to_float(row.get("delta_overfit_gap_qbeta_vs_pre"))))
        n_top_global = _safe_norm(raw_top, top_min, top_max)
        n_logcnt_global = _safe_norm(raw_logcnt, logcnt_min, logcnt_max)
        n_qfail_global = _safe_norm(raw_qfail, qfail_min, qfail_max)
        n_logspread_global = _safe_norm(raw_logspread, logspread_min, logspread_max)
        n_qspread_global = _safe_norm(raw_qspread, qspread_min, qspread_max)
        n_dq_pre_global = _safe_norm(
            float(max(0.0, _to_float(row.get("delta_holdout_p95_abs_z_qbeta_vs_pre")))),
            dq_pre_min,
            dq_pre_max,
        )
        n_dgap_pre_global = _safe_norm(
            float(max(0.0, _to_float(row.get("delta_overfit_gap_qbeta_vs_pre")))),
            dgap_pre_min,
            dgap_pre_max,
        )
        class_key = str(row.get("top_priority_class_after", "")).strip() or "__unknown__"
        class_range = class_ranges.get(class_key, {})
        class_top_min, class_top_max = class_range.get("top", (top_min, top_max))
        class_logcnt_min, class_logcnt_max = class_range.get("logcnt", (logcnt_min, logcnt_max))
        class_qfail_min, class_qfail_max = class_range.get("qfail", (qfail_min, qfail_max))
        class_logspread_min, class_logspread_max = class_range.get("logspread", (logspread_min, logspread_max))
        class_qspread_min, class_qspread_max = class_range.get("qspread", (qspread_min, qspread_max))
        class_dq_min, class_dq_max = class_range.get("dq_pre", (dq_pre_min, dq_pre_max))
        class_dgap_min, class_dgap_max = class_range.get("dgap_pre", (dgap_pre_min, dgap_pre_max))
        class_top_mul = float(top_weight_map.get(class_key, 1.0))
        class_logft_mul = float(logft_weight_map.get(class_key, 1.0))
        eff_w_top = float(weights["max_pos_delta_top_priority_q_gt3"] * class_top_mul)
        eff_w_logcnt = float(weights["max_pos_delta_n_logft_gt3"] * class_logft_mul)
        eff_w_logspread = float(weights["logft_spread"])
        eff_w_qfail = float(weights["max_pos_delta_n_qbeta_gt3"])
        eff_w_qspread = float(weights["qbeta_spread"])
        eff_w_dq = float(weights["delta_holdout_p95_abs_z_qbeta_vs_pre"])
        eff_w_dgap = float(weights["delta_overfit_gap_qbeta_vs_pre"])
        eff_weight_sum = float(
            eff_w_top + eff_w_logcnt + eff_w_logspread + eff_w_qfail + eff_w_qspread + eff_w_dq + eff_w_dgap
        )
        if not math.isfinite(eff_weight_sum) or eff_weight_sum <= 0.0:
            eff_w_top = float(weights["max_pos_delta_top_priority_q_gt3"])
            eff_w_logcnt = float(weights["max_pos_delta_n_logft_gt3"])
            eff_w_logspread = float(weights["logft_spread"])
            eff_w_qfail = float(weights["max_pos_delta_n_qbeta_gt3"])
            eff_w_qspread = float(weights["qbeta_spread"])
            eff_w_dq = float(weights["delta_holdout_p95_abs_z_qbeta_vs_pre"])
            eff_w_dgap = float(weights["delta_overfit_gap_qbeta_vs_pre"])
            eff_weight_sum = float(
                eff_w_top + eff_w_logcnt + eff_w_logspread + eff_w_qfail + eff_w_qspread + eff_w_dq + eff_w_dgap
            )
        eff_w_top /= eff_weight_sum
        eff_w_logcnt /= eff_weight_sum
        eff_w_logspread /= eff_weight_sum
        eff_w_qfail /= eff_weight_sum
        eff_w_qspread /= eff_weight_sum
        eff_w_dq /= eff_weight_sum
        eff_w_dgap /= eff_weight_sum
        n_top_class = _safe_norm(raw_top, class_top_min, class_top_max)
        n_logcnt_class = _safe_norm(raw_logcnt, class_logcnt_min, class_logcnt_max)
        n_qfail_class = _safe_norm(raw_qfail, class_qfail_min, class_qfail_max)
        n_logspread_class = _safe_norm(raw_logspread, class_logspread_min, class_logspread_max)
        n_qspread_class = _safe_norm(raw_qspread, class_qspread_min, class_qspread_max)
        n_dq_pre_class = _safe_norm(raw_dq_pre, class_dq_min, class_dq_max)
        n_dgap_pre_class = _safe_norm(raw_dgap_pre, class_dgap_min, class_dgap_max)
        n_top = float((1.0 - classwise_blend) * n_top_global + classwise_blend * n_top_class)
        n_logcnt = float((1.0 - classwise_blend) * n_logcnt_global + classwise_blend * n_logcnt_class)
        n_qfail = float((1.0 - classwise_blend) * n_qfail_global + classwise_blend * n_qfail_class)
        n_logspread = float((1.0 - classwise_blend) * n_logspread_global + classwise_blend * n_logspread_class)
        n_qspread = float((1.0 - classwise_blend) * n_qspread_global + classwise_blend * n_qspread_class)
        n_dq_pre = float((1.0 - classwise_blend) * n_dq_pre_global + classwise_blend * n_dq_pre_class)
        n_dgap_pre = float((1.0 - classwise_blend) * n_dgap_pre_global + classwise_blend * n_dgap_pre_class)
        row["residue_weight_multiplier_top_class"] = float(class_top_mul)
        row["residue_weight_multiplier_logft_class"] = float(class_logft_mul)
        row["residue_weight_top_priority_effective"] = float(eff_w_top)
        row["residue_weight_logft_effective"] = float(eff_w_logcnt)
        row["residue_weight_sum_effective"] = float(eff_w_top + eff_w_logcnt + eff_w_logspread + eff_w_qfail + eff_w_qspread + eff_w_dq + eff_w_dgap)
        row["norm_residue_max_pos_delta_top_priority_q_gt3"] = n_top
        row["norm_residue_max_pos_delta_n_logft_gt3"] = n_logcnt
        row["norm_residue_max_pos_delta_n_qbeta_gt3"] = n_qfail
        row["norm_residue_holdout_p95_abs_z_logft_spread"] = n_logspread
        row["norm_residue_holdout_p95_abs_z_qbeta_spread"] = n_qspread
        row["norm_delta_holdout_p95_abs_z_qbeta_vs_pre"] = n_dq_pre
        row["norm_delta_overfit_gap_qbeta_vs_pre"] = n_dgap_pre
        score_global = float(
            eff_w_top * n_top_global
            + eff_w_logcnt * n_logcnt_global
            + eff_w_logspread * n_logspread_global
            + eff_w_qfail * n_qfail_global
            + eff_w_qspread * n_qspread_global
            + eff_w_dq * n_dq_pre_global
            + eff_w_dgap * n_dgap_pre_global
        )
        score_class = float(
            eff_w_top * n_top_class
            + eff_w_logcnt * n_logcnt_class
            + eff_w_logspread * n_logspread_class
            + eff_w_qfail * n_qfail_class
            + eff_w_qspread * n_qspread_class
            + eff_w_dq * n_dq_pre_class
            + eff_w_dgap * n_dgap_pre_class
        )
        row["residue_class_key"] = str(class_key)
        row["residue_class_count"] = int(len(class_groups.get(class_key, [])))
        row["residue_reweight_score_global"] = float(score_global)
        row["residue_reweight_score_class"] = float(score_class)
        row["residue_reweight_score"] = float(
            eff_w_top * n_top
            + eff_w_logcnt * n_logcnt
            + eff_w_logspread * n_logspread
            + eff_w_qfail * n_qfail
            + eff_w_qspread * n_qspread
            + eff_w_dq * n_dq_pre
            + eff_w_dgap * n_dgap_pre
        )

    refreeze_enabled = bool(use_logft_rootcause_guard) and bool(use_logft_rootcause_refreeze)
    strict_root_guard_pass_rows = [r for r in eval_rows if bool(r.get("logft_rootcause_guard_pass"))]
    refrozen_logft_count_max_delta = int(logft_count_max_delta_allowed)
    refrozen_residue_logft_count_max_delta = int(residue_logft_count_max_delta_allowed)
    refrozen_require_residue_logft_nonworsening = bool(require_residue_logft_nonworsening)
    refreeze_applied = False
    refreeze_source = "strict_guard"
    if refreeze_enabled and not strict_root_guard_pass_rows:
        source_rows = [r for r in eval_rows if bool(r.get("qbeta_overfit_guard_pass"))]
        if not source_rows:
            source_rows = list(eval_rows)
            refreeze_source = "all_candidates"
        else:
            refreeze_source = "qbeta_overfit_guard_pass"
        refrozen_logft_count_max_delta = int(
            min(int(max(0, int(r.get("delta_n_logft_gt3_holdout_vs_pre") or 0))) for r in source_rows)
        )
        refrozen_residue_logft_count_max_delta = int(
            min(int(max(0, int(r.get("residue_max_pos_delta_n_logft_gt3") or 0))) for r in source_rows)
        )
        refrozen_require_residue_logft_nonworsening = bool(
            any(bool(r.get("residue_all_logft_nonworsening")) for r in source_rows)
        )
        refreeze_applied = True

    for row in eval_rows:
        refrozen_logft_limit_eff = _effective_limit_int(
            row,
            base_limit=int(refrozen_logft_count_max_delta),
            by_key=class_logft_limit_map,
        )
        refrozen_residue_logft_limit_eff = _effective_limit_int(
            row,
            base_limit=int(refrozen_residue_logft_count_max_delta),
            by_key=class_residue_logft_limit_map,
        )
        row["refrozen_logft_count_max_delta_effective"] = int(refrozen_logft_limit_eff)
        row["refrozen_residue_logft_count_max_delta_effective"] = int(refrozen_residue_logft_limit_eff)
        row["prelocal_logft_count_refrozen_pass"] = bool(
            int(row.get("delta_n_logft_gt3_holdout_vs_pre") or 0) <= int(refrozen_logft_limit_eff)
        )
        row["residue_logft_count_refrozen_pass"] = bool(
            int(row.get("residue_max_pos_delta_n_logft_gt3") or 0) <= int(refrozen_residue_logft_limit_eff)
        )
        row["residue_logft_nonworsening_refrozen_pass"] = bool(
            bool(row.get("residue_all_logft_nonworsening")) if bool(refrozen_require_residue_logft_nonworsening) else True
        )
        row["logft_rootcause_refrozen_guard_pass"] = bool(
            bool(row.get("prelocal_logft_count_refrozen_pass"))
            and bool(row.get("residue_logft_count_refrozen_pass"))
            and bool(row.get("residue_logft_nonworsening_refrozen_pass"))
        )

    residue_dual_guard_enabled = bool(use_residue_dual_guard)
    residue_dual_refreeze_enabled = bool(use_residue_dual_guard) and bool(use_residue_dual_refreeze)
    refrozen_residue_top_priority_max_delta = int(residue_top_priority_max_delta_allowed)
    refrozen_residue_overfit_max_delta = float(residue_overfit_max_delta_allowed)
    residue_dual_refreeze_applied = False
    residue_dual_refreeze_source = "strict_guard"

    strict_residue_dual_pass_rows: List[Dict[str, Any]] = []
    for row in eval_rows:
        strict_top_limit_eff = _effective_limit_int(
            row,
            base_limit=int(residue_top_priority_max_delta_allowed),
            by_key=class_residue_top_limit_map,
        )
        strict_overfit_limit_eff = _effective_limit_float(
            row,
            base_limit=float(residue_overfit_max_delta_allowed),
            by_key=class_residue_overfit_limit_map,
        )
        row["residue_top_priority_max_delta_effective_strict"] = int(strict_top_limit_eff)
        row["residue_overfit_max_delta_effective_strict"] = float(strict_overfit_limit_eff)
        strict_top_pass = bool(
            int(row.get("residue_max_pos_delta_top_priority_q_gt3") or 0)
            <= int(strict_top_limit_eff)
        )
        strict_overfit_pass = bool(
            float(max(0.0, _to_float(row.get("residue_max_pos_delta_overfit_gap_qbeta"))))
            <= float(strict_overfit_limit_eff)
        )
        row["residue_top_priority_guard_pass_strict"] = strict_top_pass
        row["residue_overfit_guard_pass_strict"] = strict_overfit_pass
        row["residue_dual_guard_pass_strict"] = bool(strict_top_pass and strict_overfit_pass)
        if bool(row["residue_dual_guard_pass_strict"]):
            strict_residue_dual_pass_rows.append(row)

    if residue_dual_refreeze_enabled and not strict_residue_dual_pass_rows:
        source_rows = [r for r in eval_rows if bool(r.get("qbeta_overfit_guard_pass"))]
        if not source_rows:
            source_rows = list(eval_rows)
            residue_dual_refreeze_source = "all_candidates"
        else:
            residue_dual_refreeze_source = "qbeta_overfit_guard_pass"
        refrozen_residue_top_priority_max_delta = int(
            min(int(max(0, int(r.get("residue_max_pos_delta_top_priority_q_gt3") or 0))) for r in source_rows)
        )
        refrozen_residue_overfit_max_delta = float(
            min(float(max(0.0, _to_float(r.get("residue_max_pos_delta_overfit_gap_qbeta")))) for r in source_rows)
        )
        residue_dual_refreeze_applied = True

    for row in eval_rows:
        if residue_dual_guard_enabled:
            top_limit_eff = _effective_limit_int(
                row,
                base_limit=int(refrozen_residue_top_priority_max_delta),
                by_key=class_residue_top_limit_map,
            )
            overfit_limit_eff = _effective_limit_float(
                row,
                base_limit=float(refrozen_residue_overfit_max_delta),
                by_key=class_residue_overfit_limit_map,
            )
            top_pass = bool(
                int(row.get("residue_max_pos_delta_top_priority_q_gt3") or 0)
                <= int(top_limit_eff)
            )
            overfit_pass = bool(
                float(max(0.0, _to_float(row.get("residue_max_pos_delta_overfit_gap_qbeta"))))
                <= float(overfit_limit_eff)
            )
        else:
            top_limit_eff = int(refrozen_residue_top_priority_max_delta)
            overfit_limit_eff = float(refrozen_residue_overfit_max_delta)
            top_pass = True
            overfit_pass = True
        row["residue_top_priority_max_delta_effective"] = int(top_limit_eff)
        row["residue_overfit_max_delta_effective"] = float(overfit_limit_eff)
        row["residue_top_priority_guard_pass"] = bool(top_pass)
        row["residue_overfit_guard_pass"] = bool(overfit_pass)
        row["residue_dual_guard_pass"] = bool(top_pass and overfit_pass)

    root_guard_key = "logft_rootcause_refrozen_guard_pass" if refreeze_enabled else "logft_rootcause_guard_pass"
    guard_pass_rows = [r for r in eval_rows if bool(r.get("qbeta_overfit_guard_pass"))]
    root_guard_pass_rows = [r for r in eval_rows if bool(r.get(root_guard_key))]
    residue_dual_guard_pass_rows = [r for r in eval_rows if bool(r.get("residue_dual_guard_pass"))]
    combined_guard_pass_rows = [
        r
        for r in eval_rows
        if (
            (bool(r.get("qbeta_overfit_guard_pass")) if bool(use_qbeta_overfit_guard) else True)
            and (bool(r.get(root_guard_key)) if bool(use_logft_rootcause_guard) else True)
            and (bool(r.get("residue_dual_guard_pass")) if bool(residue_dual_guard_enabled) else True)
        )
    ]

    residue_dual_retune_enabled = bool(residue_dual_guard_enabled) and bool(use_residue_dual_retune)
    residue_dual_retune_applied = False
    residue_dual_retune_source = "not_needed"
    residue_dual_retune_anchor_candidate = ""
    residue_dual_retune_weight_top = max(0.0, float(residue_dual_retune_top_weight))
    residue_dual_retune_weight_overfit = max(0.0, float(residue_dual_retune_overfit_weight))
    weight_sum = residue_dual_retune_weight_top + residue_dual_retune_weight_overfit
    if weight_sum <= 0.0:
        residue_dual_retune_weight_top = 0.5
        residue_dual_retune_weight_overfit = 0.5
    else:
        residue_dual_retune_weight_top /= weight_sum
        residue_dual_retune_weight_overfit /= weight_sum

    top_delta_vals_all = [float(max(0, int(r.get("residue_max_pos_delta_top_priority_q_gt3") or 0))) for r in eval_rows]
    overfit_delta_vals_all = [
        float(max(0.0, _to_float(r.get("residue_max_pos_delta_overfit_gap_qbeta")))) for r in eval_rows
    ]
    top_delta_min_all, top_delta_max_all = float(min(top_delta_vals_all)), float(max(top_delta_vals_all))
    overfit_delta_min_all, overfit_delta_max_all = float(min(overfit_delta_vals_all)), float(max(overfit_delta_vals_all))
    for row in eval_rows:
        n_top = _safe_norm(
            float(max(0, int(row.get("residue_max_pos_delta_top_priority_q_gt3") or 0))),
            top_delta_min_all,
            top_delta_max_all,
        )
        n_overfit = _safe_norm(
            float(max(0.0, _to_float(row.get("residue_max_pos_delta_overfit_gap_qbeta")))),
            overfit_delta_min_all,
            overfit_delta_max_all,
        )
        row["residue_dual_retune_cost"] = float(
            residue_dual_retune_weight_top * n_top + residue_dual_retune_weight_overfit * n_overfit
        )

    if residue_dual_retune_enabled and not combined_guard_pass_rows:
        source_rows = [
            r
            for r in eval_rows
            if (
                (bool(r.get("qbeta_overfit_guard_pass")) if bool(use_qbeta_overfit_guard) else True)
                and (bool(r.get(root_guard_key)) if bool(use_logft_rootcause_guard) else True)
            )
        ]
        if source_rows:
            residue_dual_retune_source = "qbeta_overfit_and_root_guard_pass"
        else:
            source_rows = [
                r
                for r in eval_rows
                if (bool(r.get("qbeta_overfit_guard_pass")) if bool(use_qbeta_overfit_guard) else True)
            ]
            if source_rows:
                residue_dual_retune_source = "qbeta_overfit_guard_pass"
            else:
                source_rows = list(eval_rows)
                residue_dual_retune_source = "all_candidates"
        if source_rows:
            anchor = sorted(
                source_rows,
                key=lambda r: (
                    _to_float(r.get("residue_dual_retune_cost")),
                    _to_float(r.get("residue_reweight_score")),
                    int(r.get("residue_max_pos_delta_top_priority_q_gt3") or 0),
                    _to_float(r.get("residue_max_pos_delta_overfit_gap_qbeta")),
                ),
            )[0]
            residue_dual_retune_anchor_candidate = str(anchor.get("candidate_id", ""))
            anchor_top_delta = int(max(0, int(anchor.get("residue_max_pos_delta_top_priority_q_gt3") or 0)))
            anchor_overfit_delta = float(max(0.0, _to_float(anchor.get("residue_max_pos_delta_overfit_gap_qbeta"))))
            retune_top_cap = max(
                int(refrozen_residue_top_priority_max_delta),
                int(max(0, int(residue_dual_retune_max_top_delta_allowed))),
            )
            retune_overfit_cap = max(
                float(refrozen_residue_overfit_max_delta),
                float(max(0.0, float(residue_dual_retune_max_overfit_delta_allowed))),
            )
            retuned_top = min(max(int(refrozen_residue_top_priority_max_delta), int(anchor_top_delta)), int(retune_top_cap))
            retuned_overfit = min(
                max(float(refrozen_residue_overfit_max_delta), float(anchor_overfit_delta)),
                float(retune_overfit_cap),
            )
            if (
                int(retuned_top) > int(refrozen_residue_top_priority_max_delta)
                or float(retuned_overfit) > float(refrozen_residue_overfit_max_delta)
            ):
                refrozen_residue_top_priority_max_delta = int(retuned_top)
                refrozen_residue_overfit_max_delta = float(retuned_overfit)
                residue_dual_retune_applied = True

        if residue_dual_retune_applied:
            for row in eval_rows:
                top_limit_eff = _effective_limit_int(
                    row,
                    base_limit=int(refrozen_residue_top_priority_max_delta),
                    by_key=class_residue_top_limit_map,
                )
                overfit_limit_eff = _effective_limit_float(
                    row,
                    base_limit=float(refrozen_residue_overfit_max_delta),
                    by_key=class_residue_overfit_limit_map,
                )
                row["residue_top_priority_max_delta_effective"] = int(top_limit_eff)
                row["residue_overfit_max_delta_effective"] = float(overfit_limit_eff)
                top_pass = bool(
                    int(row.get("residue_max_pos_delta_top_priority_q_gt3") or 0)
                    <= int(top_limit_eff)
                )
                overfit_pass = bool(
                    float(max(0.0, _to_float(row.get("residue_max_pos_delta_overfit_gap_qbeta"))))
                    <= float(overfit_limit_eff)
                )
                row["residue_top_priority_guard_pass"] = bool(top_pass)
                row["residue_overfit_guard_pass"] = bool(overfit_pass)
                row["residue_dual_guard_pass"] = bool(top_pass and overfit_pass)
            residue_dual_guard_pass_rows = [r for r in eval_rows if bool(r.get("residue_dual_guard_pass"))]
            combined_guard_pass_rows = [
                r
                for r in eval_rows
                if (
                    (bool(r.get("qbeta_overfit_guard_pass")) if bool(use_qbeta_overfit_guard) else True)
                    and (bool(r.get(root_guard_key)) if bool(use_logft_rootcause_guard) else True)
                    and (bool(r.get("residue_dual_guard_pass")) if bool(residue_dual_guard_enabled) else True)
                )
            ]

    logft_rootcause_retune_enabled = bool(refreeze_enabled) and bool(use_logft_rootcause_retune)
    logft_rootcause_retune_applied = False
    logft_rootcause_retune_source = "not_needed"
    logft_rootcause_retune_anchor_candidate = ""
    logft_rootcause_retune_target_combined_pass_min = max(1, int(logft_rootcause_retune_target_combined_pass_min))
    logft_rootcause_retune_max_logft_count_delta_allowed = max(
        int(refrozen_logft_count_max_delta),
        int(max(0, int(logft_rootcause_retune_max_logft_count_delta_allowed))),
    )
    logft_rootcause_retune_max_residue_logft_count_delta_allowed = max(
        int(refrozen_residue_logft_count_max_delta),
        int(max(0, int(logft_rootcause_retune_max_residue_logft_count_delta_allowed))),
    )

    if logft_rootcause_retune_enabled and len(combined_guard_pass_rows) < int(logft_rootcause_retune_target_combined_pass_min):
        source_rows = [
            r
            for r in eval_rows
            if (
                (bool(r.get("qbeta_overfit_guard_pass")) if bool(use_qbeta_overfit_guard) else True)
                and (bool(r.get("residue_dual_guard_pass")) if bool(residue_dual_guard_enabled) else True)
            )
        ]
        if source_rows:
            logft_rootcause_retune_source = "qbeta_overfit_and_residue_dual_guard_pass"
        else:
            source_rows = [
                r
                for r in eval_rows
                if (bool(r.get("qbeta_overfit_guard_pass")) if bool(use_qbeta_overfit_guard) else True)
            ]
            if source_rows:
                logft_rootcause_retune_source = "qbeta_overfit_guard_pass"
            else:
                source_rows = list(eval_rows)
                logft_rootcause_retune_source = "all_candidates"
        if source_rows:
            target_n = int(logft_rootcause_retune_target_combined_pass_min)
            base_logft_limit = int(refrozen_logft_count_max_delta)
            base_residue_logft_limit = int(refrozen_residue_logft_count_max_delta)
            candidate_choices: List[Dict[str, Any]] = []
            for cand in source_rows:
                cand_logft_delta = int(max(0, int(cand.get("delta_n_logft_gt3_holdout_vs_pre") or 0)))
                cand_residue_logft_delta = int(max(0, int(cand.get("residue_max_pos_delta_n_logft_gt3") or 0)))
                proposed_logft_limit = min(
                    max(base_logft_limit, cand_logft_delta),
                    int(logft_rootcause_retune_max_logft_count_delta_allowed),
                )
                proposed_residue_logft_limit = min(
                    max(base_residue_logft_limit, cand_residue_logft_delta),
                    int(logft_rootcause_retune_max_residue_logft_count_delta_allowed),
                )
                combined_n = 0
                for row in eval_rows:
                    q_guard_ok = bool(row.get("qbeta_overfit_guard_pass")) if bool(use_qbeta_overfit_guard) else True
                    dual_guard_ok = bool(row.get("residue_dual_guard_pass")) if bool(residue_dual_guard_enabled) else True
                    logft_limit_eff = _effective_limit_int(
                        row,
                        base_limit=int(proposed_logft_limit),
                        by_key=class_logft_limit_map,
                    )
                    residue_logft_limit_eff = _effective_limit_int(
                        row,
                        base_limit=int(proposed_residue_logft_limit),
                        by_key=class_residue_logft_limit_map,
                    )
                    logft_count_ok = bool(
                        int(row.get("delta_n_logft_gt3_holdout_vs_pre") or 0) <= int(logft_limit_eff)
                    )
                    residue_logft_count_ok = bool(
                        int(row.get("residue_max_pos_delta_n_logft_gt3") or 0) <= int(residue_logft_limit_eff)
                    )
                    residue_nonworse_ok = bool(
                        bool(row.get("residue_all_logft_nonworsening"))
                        if bool(refrozen_require_residue_logft_nonworsening)
                        else True
                    )
                    root_guard_ok = bool(logft_count_ok and residue_logft_count_ok and residue_nonworse_ok)
                    if bool(q_guard_ok and dual_guard_ok and root_guard_ok):
                        combined_n += 1
                candidate_choices.append(
                    {
                        "candidate_id": str(cand.get("candidate_id", "")),
                        "combined_n": int(combined_n),
                        "proposed_logft_limit": int(proposed_logft_limit),
                        "proposed_residue_logft_limit": int(proposed_residue_logft_limit),
                        "residue_reweight_score": _to_float(cand.get("residue_reweight_score")),
                    }
                )
            anchor_choice = sorted(
                candidate_choices,
                key=lambda c: (
                    0 if int(c.get("combined_n") or 0) >= int(target_n) else 1,
                    -int(c.get("combined_n") or 0),
                    int(c.get("proposed_logft_limit") or 0) - int(base_logft_limit),
                    int(c.get("proposed_residue_logft_limit") or 0) - int(base_residue_logft_limit),
                    _to_float(c.get("residue_reweight_score")),
                    str(c.get("candidate_id", "")),
                ),
            )[0]
            logft_rootcause_retune_anchor_candidate = str(anchor_choice.get("candidate_id", ""))
            retuned_logft_count_max_delta = int(anchor_choice.get("proposed_logft_limit") or base_logft_limit)
            retuned_residue_logft_count_max_delta = int(
                anchor_choice.get("proposed_residue_logft_limit") or base_residue_logft_limit
            )
            if (
                int(retuned_logft_count_max_delta) > int(refrozen_logft_count_max_delta)
                or int(retuned_residue_logft_count_max_delta) > int(refrozen_residue_logft_count_max_delta)
            ):
                refrozen_logft_count_max_delta = int(retuned_logft_count_max_delta)
                refrozen_residue_logft_count_max_delta = int(retuned_residue_logft_count_max_delta)
                logft_rootcause_retune_applied = True

        if logft_rootcause_retune_applied:
            for row in eval_rows:
                refrozen_logft_limit_eff = _effective_limit_int(
                    row,
                    base_limit=int(refrozen_logft_count_max_delta),
                    by_key=class_logft_limit_map,
                )
                refrozen_residue_logft_limit_eff = _effective_limit_int(
                    row,
                    base_limit=int(refrozen_residue_logft_count_max_delta),
                    by_key=class_residue_logft_limit_map,
                )
                row["refrozen_logft_count_max_delta_effective"] = int(refrozen_logft_limit_eff)
                row["refrozen_residue_logft_count_max_delta_effective"] = int(refrozen_residue_logft_limit_eff)
                row["prelocal_logft_count_refrozen_pass"] = bool(
                    int(row.get("delta_n_logft_gt3_holdout_vs_pre") or 0) <= int(refrozen_logft_limit_eff)
                )
                row["residue_logft_count_refrozen_pass"] = bool(
                    int(row.get("residue_max_pos_delta_n_logft_gt3") or 0)
                    <= int(refrozen_residue_logft_limit_eff)
                )
                row["residue_logft_nonworsening_refrozen_pass"] = bool(
                    bool(row.get("residue_all_logft_nonworsening"))
                    if bool(refrozen_require_residue_logft_nonworsening)
                    else True
                )
                row["logft_rootcause_refrozen_guard_pass"] = bool(
                    bool(row.get("prelocal_logft_count_refrozen_pass"))
                    and bool(row.get("residue_logft_count_refrozen_pass"))
                    and bool(row.get("residue_logft_nonworsening_refrozen_pass"))
                )
            root_guard_pass_rows = [r for r in eval_rows if bool(r.get(root_guard_key))]
            residue_dual_guard_pass_rows = [r for r in eval_rows if bool(r.get("residue_dual_guard_pass"))]
            combined_guard_pass_rows = [
                r
                for r in eval_rows
                if (
                    (bool(r.get("qbeta_overfit_guard_pass")) if bool(use_qbeta_overfit_guard) else True)
                    and (bool(r.get(root_guard_key)) if bool(use_logft_rootcause_guard) else True)
                    and (bool(r.get("residue_dual_guard_pass")) if bool(residue_dual_guard_enabled) else True)
                )
            ]

    has_combined = bool(combined_guard_pass_rows)
    mode_parts = ["logft_qfail_residue_reweighted"]
    if bool(use_logft_rootcause_guard):
        mode_parts.append("rootcause")
        if refreeze_enabled:
            mode_parts.append("refrozen")
        if logft_rootcause_retune_enabled:
            mode_parts.append("retuned" if logft_rootcause_retune_applied else "retune_ready")
    if bool(use_qbeta_overfit_guard):
        mode_parts.append("qbeta_overfit")
    if bool(residue_dual_guard_enabled):
        mode_parts.append("residue_dual")
        if residue_dual_refreeze_enabled:
            mode_parts.append("refrozen")
        if residue_dual_retune_enabled:
            mode_parts.append("retuned" if residue_dual_retune_applied else "retune_ready")
    mode = "_".join(mode_parts) + ("_guarded" if has_combined else "_relaxed")

    def _tier(row: Dict[str, Any]) -> tuple[int, int, int, int]:
        q_ok = bool(row.get("qbeta_overfit_guard_pass")) if bool(use_qbeta_overfit_guard) else True
        r_ok = bool(row.get(root_guard_key)) if bool(use_logft_rootcause_guard) else True
        d_ok = bool(row.get("residue_dual_guard_pass")) if bool(residue_dual_guard_enabled) else True
        misses = int((not q_ok) if bool(use_qbeta_overfit_guard) else 0) + int(
            (not r_ok) if bool(use_logft_rootcause_guard) else 0
        ) + int((not d_ok) if bool(residue_dual_guard_enabled) else 0)
        return (
            misses,
            0 if d_ok else 1,
            0 if q_ok else 1,
            0 if r_ok else 1,
        )

    ranked = sorted(
        eval_rows,
        key=lambda r: (
            _tier(r),
            float(_to_float(r.get("residue_dual_retune_cost"))),
            float(r.get("residue_reweight_score", float("inf"))),
            float(r.get("residue_max_pos_delta_top_priority_q_gt3", float("inf"))),
            float(r.get("residue_max_pos_delta_overfit_gap_qbeta", float("inf"))),
            float(r.get("residue_max_pos_delta_n_logft_gt3", float("inf"))),
            float(r.get("delta_n_logft_gt3_holdout_vs_pre", float("inf"))),
            float(r.get("delta_holdout_p95_abs_z_qbeta_vs_pre", float("inf"))),
            float(r.get("delta_overfit_gap_qbeta_vs_pre", float("inf"))),
            float(r.get("holdout_p95_abs_z_logft", float("inf"))),
            float(r.get("overfit_gap_qbeta", float("inf"))),
            float(r.get("holdout_p95_abs_z_qbeta", float("inf"))),
        ),
    )
    for idx, row in enumerate(ranked, start=1):
        row["rank_residue_reweight"] = int(idx)

    return {
        "status": "completed",
        "selection_policy": {
            "mode": mode,
            "n_candidates_input": int(len(candidate_rows)),
            "n_candidates_evaluated": int(len(eval_rows)),
            "n_candidates_limit": int(limit),
            "n_candidates_guard_pass": int(len(guard_pass_rows)),
            "n_candidates_root_guard_pass": int(len(root_guard_pass_rows)),
            "n_candidates_residue_dual_guard_pass": int(len(residue_dual_guard_pass_rows)),
            "n_candidates_combined_guard_pass": int(len(combined_guard_pass_rows)),
            "qbeta_overfit_guard_enabled": bool(use_qbeta_overfit_guard),
            "qbeta_max_delta_allowed": float(qbeta_max_delta_allowed),
            "overfit_max_delta_allowed": float(overfit_max_delta_allowed),
            "logft_rootcause_guard_enabled": bool(use_logft_rootcause_guard),
            "logft_rootcause_refreeze_enabled": bool(refreeze_enabled),
            "logft_rootcause_refreeze_applied": bool(refreeze_applied),
            "logft_rootcause_refreeze_source": str(refreeze_source),
            "logft_rootcause_retune_enabled": bool(logft_rootcause_retune_enabled),
            "logft_rootcause_retune_applied": bool(logft_rootcause_retune_applied),
            "logft_rootcause_retune_source": str(logft_rootcause_retune_source),
            "logft_rootcause_retune_anchor_candidate": str(logft_rootcause_retune_anchor_candidate),
            "logft_rootcause_retune_target_combined_pass_min": int(logft_rootcause_retune_target_combined_pass_min),
            "logft_rootcause_retune_max_logft_count_delta_allowed": int(
                logft_rootcause_retune_max_logft_count_delta_allowed
            ),
            "logft_rootcause_retune_max_residue_logft_count_delta_allowed": int(
                logft_rootcause_retune_max_residue_logft_count_delta_allowed
            ),
            "logft_count_max_delta_allowed": int(logft_count_max_delta_allowed),
            "residue_logft_count_max_delta_allowed": int(residue_logft_count_max_delta_allowed),
            "require_residue_logft_nonworsening": bool(require_residue_logft_nonworsening),
            "refrozen_logft_count_max_delta_allowed": int(refrozen_logft_count_max_delta),
            "refrozen_residue_logft_count_max_delta_allowed": int(refrozen_residue_logft_count_max_delta),
            "refrozen_require_residue_logft_nonworsening": bool(refrozen_require_residue_logft_nonworsening),
            "residue_dual_guard_enabled": bool(residue_dual_guard_enabled),
            "residue_top_priority_max_delta_allowed": int(residue_top_priority_max_delta_allowed),
            "residue_overfit_max_delta_allowed": float(residue_overfit_max_delta_allowed),
            "residue_dual_refreeze_enabled": bool(residue_dual_refreeze_enabled),
            "residue_dual_refreeze_applied": bool(residue_dual_refreeze_applied),
            "residue_dual_refreeze_source": str(residue_dual_refreeze_source),
            "residue_dual_retune_enabled": bool(residue_dual_retune_enabled),
            "residue_dual_retune_applied": bool(residue_dual_retune_applied),
            "residue_dual_retune_source": str(residue_dual_retune_source),
            "residue_dual_retune_anchor_candidate": str(residue_dual_retune_anchor_candidate),
            "residue_dual_retune_weight_top_priority": float(residue_dual_retune_weight_top),
            "residue_dual_retune_weight_overfit": float(residue_dual_retune_weight_overfit),
            "residue_dual_retune_max_top_delta_allowed": int(max(0, int(residue_dual_retune_max_top_delta_allowed))),
            "residue_dual_retune_max_overfit_delta_allowed": float(
                max(0.0, float(residue_dual_retune_max_overfit_delta_allowed))
            ),
            "refrozen_residue_top_priority_max_delta_allowed": int(refrozen_residue_top_priority_max_delta),
            "refrozen_residue_overfit_max_delta_allowed": float(refrozen_residue_overfit_max_delta),
            "pre_local_holdout_p95_abs_z_qbeta": float(pre_local_holdout_p95_abs_z_qbeta),
            "pre_local_overfit_gap_qbeta": float(pre_local_overfit_gap_qbeta),
            "pre_local_n_logft_gt3_holdout": int(pre_local_n_logft_gt3_holdout),
            "weights": weights,
            "classwise_residue_norm_blend": float(classwise_blend),
            "classwise_group_counts": {str(k): int(len(v)) for k, v in class_groups.items()},
            "class_top_priority_weight_by_key": {str(k): float(v) for k, v in top_weight_map.items()},
            "class_logft_weight_by_key": {str(k): float(v) for k, v in logft_weight_map.items()},
            "class_key_threshold_guard_enabled": bool(use_class_key_threshold_guard),
            "class_logft_count_max_delta_by_key": {str(k): int(v) for k, v in class_logft_limit_map.items()},
            "class_residue_logft_count_max_delta_by_key": {
                str(k): int(v) for k, v in class_residue_logft_limit_map.items()
            },
            "class_residue_top_priority_max_delta_by_key": {
                str(k): int(v) for k, v in class_residue_top_limit_map.items()
            },
            "class_residue_overfit_max_delta_by_key": {
                str(k): float(v) for k, v in class_residue_overfit_limit_map.items()
            },
            "n_candidates_class_key_guard_applied": int(
                sum(1 for r in eval_rows if bool(r.get("class_key_threshold_guard_applied")))
            ),
            "reference_residue": int(reference_residue),
            "residue_list": sorted({int(x) % max(1, int(holdout_hash_modulo)) for x in holdout_hash_residues}),
        },
        "recommended_candidate": ranked[0] if ranked else {},
        "all_candidates": eval_rows,
        "top_ranked_candidates": ranked[:20],
    }


def _build_lock_refreeze_policy_from_reweight_policy(rw_policy: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "enabled": bool(rw_policy.get("logft_rootcause_refreeze_enabled")),
        "refrozen_logft_count_max_delta_allowed": int(
            rw_policy.get("refrozen_logft_count_max_delta_allowed") or 0
        ),
        "refrozen_residue_logft_count_max_delta_allowed": int(
            rw_policy.get("refrozen_residue_logft_count_max_delta_allowed") or 0
        ),
        "refrozen_require_residue_logft_nonworsening": bool(
            rw_policy.get("refrozen_require_residue_logft_nonworsening")
        ),
        "residue_dual_guard_enabled": bool(rw_policy.get("residue_dual_guard_enabled")),
        "residue_dual_refreeze_enabled": bool(rw_policy.get("residue_dual_refreeze_enabled")),
        "residue_dual_refreeze_applied": bool(rw_policy.get("residue_dual_refreeze_applied")),
        "refrozen_residue_top_priority_max_delta_allowed": int(
            rw_policy.get("refrozen_residue_top_priority_max_delta_allowed") or 0
        ),
        "refrozen_residue_overfit_max_delta_allowed": _to_float(
            rw_policy.get("refrozen_residue_overfit_max_delta_allowed")
        ),
    }


def _build_localcorr_residue_reweight_retune_grid(
    *,
    rows_mapped: List[Dict[str, Any]],
    target_classes: List[str],
    candidate_rows: List[Dict[str, Any]],
    closure_gate: Dict[str, Any],
    logft_sigma_proxy: float,
    z_gate: float,
    holdout_hash_modulo: int,
    holdout_hash_residues: List[int],
    overfit_gap_gate: float,
    sigma_floor_mev: float,
    reference_residue: int,
    max_candidates: int,
    pre_local_holdout_p95_abs_z_qbeta: float,
    pre_local_holdout_p95_abs_z_logft: float,
    pre_local_overfit_gap_qbeta: float,
    pre_local_n_qbeta_gt3_holdout: int,
    pre_local_n_logft_gt3_holdout: int,
    use_qbeta_overfit_guard: bool,
    qbeta_max_delta_allowed: float,
    overfit_max_delta_allowed: float,
    use_logft_rootcause_guard: bool,
    require_residue_logft_nonworsening: bool,
    use_logft_rootcause_refreeze: bool,
    use_logft_rootcause_retune: bool,
    logft_rootcause_retune_target_combined_pass_min: int,
    logft_rootcause_retune_max_logft_count_delta_allowed: int,
    logft_rootcause_retune_max_residue_logft_count_delta_allowed: int,
    use_residue_dual_guard: bool,
    residue_top_priority_max_delta_allowed: int,
    residue_overfit_max_delta_allowed: float,
    use_residue_dual_refreeze: bool,
    use_residue_dual_retune: bool,
    residue_dual_retune_top_weight: float,
    residue_dual_retune_overfit_weight: float,
    residue_dual_retune_max_top_delta_allowed: int,
    residue_dual_retune_max_overfit_delta_allowed: float,
    classwise_residue_norm_blend: float,
    use_class_key_threshold_guard: bool,
    class_logft_count_max_delta_by_key: Dict[str, int],
    class_residue_logft_count_max_delta_by_key: Dict[str, int],
    class_residue_top_priority_max_delta_by_key: Dict[str, int],
    class_residue_overfit_max_delta_by_key: Dict[str, float],
    base_class_top_priority_weight_by_key: Dict[str, float],
    base_class_logft_weight_by_key: Dict[str, float],
    class_weight_target_keys: List[str],
    logft_count_max_delta_values: List[int],
    residue_logft_count_max_delta_values: List[int],
    class_top_weight_values: List[float],
    class_logft_weight_values: List[float],
    qbeta_count_recovery_policy: Dict[str, Any],
) -> Dict[str, Any]:
    rows_in = [dict(r) for r in candidate_rows if isinstance(r, dict)]
    tclasses = [str(x) for x in target_classes if str(x).strip()]
    if not rows_in or not tclasses:
        return {
            "status": "not_run",
            "reason": "no_candidate_rows_or_target_classes",
            "grid_spec": {},
            "recommended_candidate": {},
            "recommended_reweight": {},
            "recommended_stability_lock": {},
            "all_candidates": [],
            "applied_best": False,
        }

    lc_vals = sorted({max(0, int(v)) for v in logft_count_max_delta_values}) if logft_count_max_delta_values else []
    rlc_vals = (
        sorted({max(0, int(v)) for v in residue_logft_count_max_delta_values})
        if residue_logft_count_max_delta_values
        else []
    )
    top_vals = sorted({max(0.0, float(v)) for v in class_top_weight_values if math.isfinite(float(v))})
    log_vals = sorted({max(0.0, float(v)) for v in class_logft_weight_values if math.isfinite(float(v))})
    if not lc_vals:
        lc_vals = [0]
    if not rlc_vals:
        rlc_vals = [0]
    if not top_vals:
        top_vals = [1.0]
    if not log_vals:
        log_vals = [1.0]

    target_scope = {str(x) for x in tclasses}
    target_keys_raw = [str(x).strip() for x in class_weight_target_keys if str(x).strip()]
    target_keys: List[str] = []
    target_seen: Set[str] = set()
    for key in target_keys_raw:
        if key not in target_scope:
            continue
        if key in target_seen:
            continue
        target_seen.add(key)
        target_keys.append(key)
    if not target_keys:
        target_keys = [str(tclasses[0])]
    base_top_map = {str(k): float(max(0.0, _to_float(v))) for k, v in dict(base_class_top_priority_weight_by_key).items()}
    base_log_map = {str(k): float(max(0.0, _to_float(v))) for k, v in dict(base_class_logft_weight_by_key).items()}

    summary_rows: List[Dict[str, Any]] = []
    best_key: Optional[Tuple[Any, ...]] = None
    best_row: Dict[str, Any] = {}
    best_rw_pack: Dict[str, Any] = {}
    best_lock_pack: Dict[str, Any] = {}

    def _finite_or_inf(v: Any) -> float:
        x = _to_float(v)
        return float(x) if math.isfinite(x) else float("inf")

    for lc_delta in lc_vals:
        for rlc_delta in rlc_vals:
            for top_w in top_vals:
                for log_w in log_vals:
                    class_top_map = dict(base_top_map)
                    class_log_map = dict(base_log_map)
                    for target_key in target_keys:
                        class_top_map[str(target_key)] = float(max(0.0, top_w))
                        class_log_map[str(target_key)] = float(max(0.0, log_w))
                    rw_pack = _build_localcorr_residue_reweight(
                        rows_mapped=rows_mapped,
                        target_classes=tclasses,
                        candidate_rows=rows_in,
                        closure_gate=closure_gate,
                        logft_sigma_proxy=float(logft_sigma_proxy),
                        z_gate=float(z_gate),
                        holdout_hash_modulo=int(holdout_hash_modulo),
                        holdout_hash_residues=[int(x) for x in holdout_hash_residues],
                        overfit_gap_gate=float(overfit_gap_gate),
                        sigma_floor_mev=float(sigma_floor_mev),
                        reference_residue=int(reference_residue),
                        max_candidates=int(max_candidates),
                        pre_local_holdout_p95_abs_z_qbeta=float(pre_local_holdout_p95_abs_z_qbeta),
                        pre_local_overfit_gap_qbeta=float(pre_local_overfit_gap_qbeta),
                        pre_local_n_logft_gt3_holdout=int(pre_local_n_logft_gt3_holdout),
                        use_qbeta_overfit_guard=bool(use_qbeta_overfit_guard),
                        qbeta_max_delta_allowed=float(qbeta_max_delta_allowed),
                        overfit_max_delta_allowed=float(overfit_max_delta_allowed),
                        use_logft_rootcause_guard=bool(use_logft_rootcause_guard),
                        logft_count_max_delta_allowed=int(lc_delta),
                        residue_logft_count_max_delta_allowed=int(rlc_delta),
                        require_residue_logft_nonworsening=bool(require_residue_logft_nonworsening),
                        use_logft_rootcause_refreeze=bool(use_logft_rootcause_refreeze),
                        use_logft_rootcause_retune=bool(use_logft_rootcause_retune),
                        logft_rootcause_retune_target_combined_pass_min=int(
                            logft_rootcause_retune_target_combined_pass_min
                        ),
                        logft_rootcause_retune_max_logft_count_delta_allowed=int(
                            logft_rootcause_retune_max_logft_count_delta_allowed
                        ),
                        logft_rootcause_retune_max_residue_logft_count_delta_allowed=int(
                            logft_rootcause_retune_max_residue_logft_count_delta_allowed
                        ),
                        use_residue_dual_guard=bool(use_residue_dual_guard),
                        residue_top_priority_max_delta_allowed=int(residue_top_priority_max_delta_allowed),
                        residue_overfit_max_delta_allowed=float(residue_overfit_max_delta_allowed),
                        use_residue_dual_refreeze=bool(use_residue_dual_refreeze),
                        use_residue_dual_retune=bool(use_residue_dual_retune),
                        residue_dual_retune_top_weight=float(residue_dual_retune_top_weight),
                        residue_dual_retune_overfit_weight=float(residue_dual_retune_overfit_weight),
                        residue_dual_retune_max_top_delta_allowed=int(residue_dual_retune_max_top_delta_allowed),
                        residue_dual_retune_max_overfit_delta_allowed=float(
                            residue_dual_retune_max_overfit_delta_allowed
                        ),
                        classwise_residue_norm_blend=float(classwise_residue_norm_blend),
                        use_class_key_threshold_guard=bool(use_class_key_threshold_guard),
                        class_logft_count_max_delta_by_key=class_logft_count_max_delta_by_key,
                        class_residue_logft_count_max_delta_by_key=class_residue_logft_count_max_delta_by_key,
                        class_residue_top_priority_max_delta_by_key=class_residue_top_priority_max_delta_by_key,
                        class_residue_overfit_max_delta_by_key=class_residue_overfit_max_delta_by_key,
                        class_top_priority_weight_by_key=class_top_map,
                        class_logft_weight_by_key=class_log_map,
                    )
                    if str(rw_pack.get("status")) != "completed":
                        continue
                    rw_policy = (
                        rw_pack.get("selection_policy")
                        if isinstance(rw_pack.get("selection_policy"), dict)
                        else {}
                    )
                    lock_refreeze_policy = _build_lock_refreeze_policy_from_reweight_policy(rw_policy)
                    lock_pack = _build_candidate_stability_lock(
                        reweight_pack=rw_pack,
                        pre_local_holdout_p95_qbeta=float(pre_local_holdout_p95_abs_z_qbeta),
                        pre_local_holdout_p95_logft=float(pre_local_holdout_p95_abs_z_logft),
                        pre_local_overfit_gap_qbeta=float(pre_local_overfit_gap_qbeta),
                        pre_local_n_qbeta_gt3_holdout=int(pre_local_n_qbeta_gt3_holdout),
                        pre_local_n_logft_gt3_holdout=int(pre_local_n_logft_gt3_holdout),
                        logft_rootcause_refreeze_policy=lock_refreeze_policy,
                        qbeta_count_recovery_policy=qbeta_count_recovery_policy,
                    )
                    lock_rec = (
                        lock_pack.get("recommended_candidate")
                        if isinstance(lock_pack.get("recommended_candidate"), dict)
                        else {}
                    )
                    rw_rec = (
                        rw_pack.get("recommended_candidate")
                        if isinstance(rw_pack.get("recommended_candidate"), dict)
                        else {}
                    )
                    cands = rw_pack.get("all_candidates") if isinstance(rw_pack.get("all_candidates"), list) else []
                    watch3_count = 0
                    for cand in cands:
                        if not isinstance(cand, dict):
                            continue
                        if bool(cand.get("residue_all_top_priority_nonregression")) and bool(
                            cand.get("residue_all_logft_nonworsening")
                        ) and bool(cand.get("residue_all_logft_count_nonregression")):
                            watch3_count += 1
                    n_eval = int(rw_policy.get("n_candidates_evaluated") or 0)
                    watch3_ratio = (float(watch3_count) / float(n_eval)) if n_eval > 0 else 0.0

                    row = {
                        "target_class_key": str(target_keys[0]),
                        "target_class_keys": ";".join([str(x) for x in target_keys]),
                        "n_target_classes": int(len(target_keys)),
                        "logft_count_max_delta_allowed": int(lc_delta),
                        "residue_logft_count_max_delta_allowed": int(rlc_delta),
                        "class_top_weight": float(top_w),
                        "class_logft_weight": float(log_w),
                        "n_candidates_evaluated": int(n_eval),
                        "n_candidates_combined_guard_pass": int(rw_policy.get("n_candidates_combined_guard_pass") or 0),
                        "n_candidates_root_guard_pass": int(rw_policy.get("n_candidates_root_guard_pass") or 0),
                        "n_candidates_guard_pass": int(rw_policy.get("n_candidates_guard_pass") or 0),
                        "stable_candidates_n": int(lock_pack.get("stable_candidates_n") or 0),
                        "stability_mode": str(lock_pack.get("mode", "not_run")),
                        "watch3_all_true_candidates_n": int(watch3_count),
                        "watch3_all_true_candidates_ratio": float(watch3_ratio),
                        "best_candidate_id": str(rw_rec.get("candidate_id", "")),
                        "best_residue_reweight_score": _finite_or_inf(rw_rec.get("residue_reweight_score")),
                        "best_holdout_p95_abs_z_qbeta": _finite_or_inf(rw_rec.get("holdout_p95_abs_z_qbeta")),
                        "best_holdout_p95_abs_z_logft": _finite_or_inf(rw_rec.get("holdout_p95_abs_z_logft")),
                        "best_overfit_gap_qbeta": _finite_or_inf(rw_rec.get("overfit_gap_qbeta")),
                        "best_stability_hard_metric_score": _finite_or_inf(
                            lock_rec.get("stability_hard_metric_score")
                        ),
                        "best_stability_violation_score": _finite_or_inf(lock_rec.get("stability_violation_score")),
                    }
                    rank_key = (
                        0 if int(watch3_count) > 0 else 1,
                        -int(watch3_count),
                        -float(watch3_ratio),
                        -int(row["n_candidates_combined_guard_pass"]),
                        -int(row["stable_candidates_n"]),
                        float(row["best_stability_hard_metric_score"]),
                        float(row["best_residue_reweight_score"]),
                        float(row["best_holdout_p95_abs_z_qbeta"]),
                        float(row["best_overfit_gap_qbeta"]),
                    )
                    row["_rank_key"] = rank_key
                    summary_rows.append(row)
                    if best_key is None or rank_key < best_key:
                        best_key = rank_key
                        best_row = dict(row)
                        best_rw_pack = rw_pack
                        best_lock_pack = lock_pack

    if not summary_rows:
        return {
            "status": "not_run",
            "reason": "no_evaluable_grid_candidates",
            "grid_spec": {},
            "recommended_candidate": {},
            "recommended_reweight": {},
            "recommended_stability_lock": {},
            "all_candidates": [],
            "applied_best": False,
        }

    summary_sorted = sorted(summary_rows, key=lambda r: tuple(r.get("_rank_key", (float("inf"),))))
    for idx, row in enumerate(summary_sorted, start=1):
        row["rank_retune"] = int(idx)
        row.pop("_rank_key", None)
    if best_row:
        best_row.pop("_rank_key", None)

    return {
        "status": "completed",
        "target_class_key": str(target_keys[0]),
        "target_class_keys": [str(x) for x in target_keys],
        "grid_spec": {
            "target_class_keys": [str(x) for x in target_keys],
            "n_target_classes": int(len(target_keys)),
            "logft_count_max_delta_values": [int(v) for v in lc_vals],
            "residue_logft_count_max_delta_values": [int(v) for v in rlc_vals],
            "class_top_weight_values": [float(v) for v in top_vals],
            "class_logft_weight_values": [float(v) for v in log_vals],
            "n_total_combinations": int(len(summary_sorted)),
        },
        "recommended_candidate": best_row,
        "recommended_reweight": best_rw_pack,
        "recommended_stability_lock": best_lock_pack,
        "all_candidates": summary_sorted,
        "applied_best": False,
    }


def _build_localcorr_residue_reweight_retune_strategy_audit(
    *,
    rows_mapped: List[Dict[str, Any]],
    target_classes: List[str],
    candidate_rows: List[Dict[str, Any]],
    closure_gate: Dict[str, Any],
    logft_sigma_proxy: float,
    z_gate: float,
    holdout_hash_modulo: int,
    holdout_hash_residues: List[int],
    overfit_gap_gate: float,
    sigma_floor_mev: float,
    reference_residue: int,
    max_candidates: int,
    pre_local_holdout_p95_abs_z_qbeta: float,
    pre_local_holdout_p95_abs_z_logft: float,
    pre_local_overfit_gap_qbeta: float,
    pre_local_n_qbeta_gt3_holdout: int,
    pre_local_n_logft_gt3_holdout: int,
    use_qbeta_overfit_guard: bool,
    qbeta_max_delta_allowed: float,
    overfit_max_delta_allowed: float,
    use_logft_rootcause_guard: bool,
    require_residue_logft_nonworsening: bool,
    use_logft_rootcause_refreeze: bool,
    use_logft_rootcause_retune: bool,
    logft_rootcause_retune_target_combined_pass_min: int,
    logft_rootcause_retune_max_logft_count_delta_allowed: int,
    logft_rootcause_retune_max_residue_logft_count_delta_allowed: int,
    use_residue_dual_guard: bool,
    residue_top_priority_max_delta_allowed: int,
    residue_overfit_max_delta_allowed: float,
    use_residue_dual_refreeze: bool,
    use_residue_dual_retune: bool,
    residue_dual_retune_top_weight: float,
    residue_dual_retune_overfit_weight: float,
    residue_dual_retune_max_top_delta_allowed: int,
    residue_dual_retune_max_overfit_delta_allowed: float,
    classwise_residue_norm_blend: float,
    use_class_key_threshold_guard: bool,
    class_logft_count_max_delta_by_key: Dict[str, int],
    class_residue_logft_count_max_delta_by_key: Dict[str, int],
    class_residue_top_priority_max_delta_by_key: Dict[str, int],
    class_residue_overfit_max_delta_by_key: Dict[str, float],
    base_class_top_priority_weight_by_key: Dict[str, float],
    base_class_logft_weight_by_key: Dict[str, float],
    logft_count_max_delta_values: List[int],
    residue_logft_count_max_delta_values: List[int],
    class_top_weight_values: List[float],
    class_logft_weight_values: List[float],
    qbeta_count_recovery_policy: Dict[str, Any],
    strategy_definitions: List[Dict[str, Any]],
) -> Dict[str, Any]:
    rows_in = [dict(r) for r in candidate_rows if isinstance(r, dict)]
    tclasses = [str(x) for x in target_classes if str(x).strip()]
    if not rows_in or not tclasses:
        return {
            "status": "not_run",
            "reason": "no_candidate_rows_or_target_classes",
            "strategy_count": 0,
            "all_strategies": [],
            "recommended_strategy": {},
            "recommended_reweight": {},
            "recommended_stability_lock": {},
            "recommended_retune_candidate": {},
            "recommended_retune_grid_spec": {},
            "applied_best": False,
        }

    class_scope = {str(x) for x in tclasses}
    normalized_defs: List[Dict[str, Any]] = []
    seen_defs: Set[Tuple[str, ...]] = set()
    for idx, strat in enumerate(strategy_definitions, start=1):
        if not isinstance(strat, dict):
            continue
        mode = str(strat.get("mode", "")).strip() or "custom"
        raw_keys = [str(x).strip() for x in (strat.get("target_class_keys") or []) if str(x).strip()]
        top_k = int(strat.get("target_top_k") or 0)
        keys: List[str] = []
        if mode == "top_k":
            k_eff = max(1, top_k)
            keys = [str(x) for x in tclasses[:k_eff]]
        else:
            keys = [str(x) for x in raw_keys if str(x) in class_scope]
            if not keys and top_k > 0:
                keys = [str(x) for x in tclasses[: max(1, int(top_k))]]
        keys = [str(x) for x in keys if str(x).strip()]
        if not keys:
            continue
        dedupe_key = (mode,) + tuple(keys)
        if dedupe_key in seen_defs:
            continue
        seen_defs.add(dedupe_key)
        label = str(strat.get("label", "")).strip()
        if not label:
            if mode == "top_k":
                label = f"top_k={len(keys)}"
            else:
                label = "explicit:" + ",".join(keys)
        normalized_defs.append(
            {
                "strategy_id": f"strategy_{idx:02d}",
                "mode": mode,
                "label": label,
                "target_top_k": int(len(keys) if mode == "top_k" else max(0, top_k)),
                "target_class_keys": [str(x) for x in keys],
            }
        )

    if len(normalized_defs) <= 0:
        return {
            "status": "not_run",
            "reason": "no_valid_strategy_definitions",
            "strategy_count": 0,
            "all_strategies": [],
            "recommended_strategy": {},
            "recommended_reweight": {},
            "recommended_stability_lock": {},
            "recommended_retune_candidate": {},
            "recommended_retune_grid_spec": {},
            "applied_best": False,
        }

    summary_rows: List[Dict[str, Any]] = []
    best_key: Optional[Tuple[Any, ...]] = None
    best_summary: Dict[str, Any] = {}
    best_retune: Dict[str, Any] = {}

    def _finite_or_inf(v: Any) -> float:
        x = _to_float(v)
        return float(x) if math.isfinite(x) else float("inf")

    for strat in normalized_defs:
        keys = [str(x) for x in (strat.get("target_class_keys") or []) if str(x).strip()]
        if not keys:
            continue
        retune_pack = _build_localcorr_residue_reweight_retune_grid(
            rows_mapped=rows_mapped,
            target_classes=tclasses,
            candidate_rows=rows_in,
            closure_gate=closure_gate,
            logft_sigma_proxy=float(logft_sigma_proxy),
            z_gate=float(z_gate),
            holdout_hash_modulo=int(holdout_hash_modulo),
            holdout_hash_residues=[int(x) for x in holdout_hash_residues],
            overfit_gap_gate=float(overfit_gap_gate),
            sigma_floor_mev=float(sigma_floor_mev),
            reference_residue=int(reference_residue),
            max_candidates=int(max_candidates),
            pre_local_holdout_p95_abs_z_qbeta=float(pre_local_holdout_p95_abs_z_qbeta),
            pre_local_holdout_p95_abs_z_logft=float(pre_local_holdout_p95_abs_z_logft),
            pre_local_overfit_gap_qbeta=float(pre_local_overfit_gap_qbeta),
            pre_local_n_qbeta_gt3_holdout=int(pre_local_n_qbeta_gt3_holdout),
            pre_local_n_logft_gt3_holdout=int(pre_local_n_logft_gt3_holdout),
            use_qbeta_overfit_guard=bool(use_qbeta_overfit_guard),
            qbeta_max_delta_allowed=float(qbeta_max_delta_allowed),
            overfit_max_delta_allowed=float(overfit_max_delta_allowed),
            use_logft_rootcause_guard=bool(use_logft_rootcause_guard),
            require_residue_logft_nonworsening=bool(require_residue_logft_nonworsening),
            use_logft_rootcause_refreeze=bool(use_logft_rootcause_refreeze),
            use_logft_rootcause_retune=bool(use_logft_rootcause_retune),
            logft_rootcause_retune_target_combined_pass_min=int(logft_rootcause_retune_target_combined_pass_min),
            logft_rootcause_retune_max_logft_count_delta_allowed=int(
                logft_rootcause_retune_max_logft_count_delta_allowed
            ),
            logft_rootcause_retune_max_residue_logft_count_delta_allowed=int(
                logft_rootcause_retune_max_residue_logft_count_delta_allowed
            ),
            use_residue_dual_guard=bool(use_residue_dual_guard),
            residue_top_priority_max_delta_allowed=int(residue_top_priority_max_delta_allowed),
            residue_overfit_max_delta_allowed=float(residue_overfit_max_delta_allowed),
            use_residue_dual_refreeze=bool(use_residue_dual_refreeze),
            use_residue_dual_retune=bool(use_residue_dual_retune),
            residue_dual_retune_top_weight=float(residue_dual_retune_top_weight),
            residue_dual_retune_overfit_weight=float(residue_dual_retune_overfit_weight),
            residue_dual_retune_max_top_delta_allowed=int(residue_dual_retune_max_top_delta_allowed),
            residue_dual_retune_max_overfit_delta_allowed=float(residue_dual_retune_max_overfit_delta_allowed),
            classwise_residue_norm_blend=float(classwise_residue_norm_blend),
            use_class_key_threshold_guard=bool(use_class_key_threshold_guard),
            class_logft_count_max_delta_by_key=class_logft_count_max_delta_by_key,
            class_residue_logft_count_max_delta_by_key=class_residue_logft_count_max_delta_by_key,
            class_residue_top_priority_max_delta_by_key=class_residue_top_priority_max_delta_by_key,
            class_residue_overfit_max_delta_by_key=class_residue_overfit_max_delta_by_key,
            base_class_top_priority_weight_by_key=base_class_top_priority_weight_by_key,
            base_class_logft_weight_by_key=base_class_logft_weight_by_key,
            class_weight_target_keys=[str(x) for x in keys],
            logft_count_max_delta_values=logft_count_max_delta_values,
            residue_logft_count_max_delta_values=residue_logft_count_max_delta_values,
            class_top_weight_values=class_top_weight_values,
            class_logft_weight_values=class_logft_weight_values,
            qbeta_count_recovery_policy=qbeta_count_recovery_policy,
        )
        if str(retune_pack.get("status")) != "completed":
            continue
        rec = retune_pack.get("recommended_candidate") if isinstance(retune_pack.get("recommended_candidate"), dict) else {}
        grid_spec = retune_pack.get("grid_spec") if isinstance(retune_pack.get("grid_spec"), dict) else {}
        row = {
            "strategy_id": str(strat.get("strategy_id", "")),
            "strategy_mode": str(strat.get("mode", "")),
            "strategy_label": str(strat.get("label", "")),
            "target_top_k": int(strat.get("target_top_k") or 0),
            "target_class_keys": ";".join([str(x) for x in keys]),
            "n_target_classes": int(len(keys)),
            "n_total_combinations": int(grid_spec.get("n_total_combinations") or 0),
            "n_candidates_evaluated": int(rec.get("n_candidates_evaluated") or 0),
            "n_candidates_guard_pass": int(rec.get("n_candidates_guard_pass") or 0),
            "n_candidates_root_guard_pass": int(rec.get("n_candidates_root_guard_pass") or 0),
            "n_candidates_combined_guard_pass": int(rec.get("n_candidates_combined_guard_pass") or 0),
            "stable_candidates_n": int(rec.get("stable_candidates_n") or 0),
            "stability_mode": str(rec.get("stability_mode", "")),
            "watch3_all_true_candidates_n": int(rec.get("watch3_all_true_candidates_n") or 0),
            "watch3_all_true_candidates_ratio": float(_to_float(rec.get("watch3_all_true_candidates_ratio"))),
            "best_candidate_id": str(rec.get("best_candidate_id", "")),
            "best_residue_reweight_score": _finite_or_inf(rec.get("best_residue_reweight_score")),
            "best_holdout_p95_abs_z_qbeta": _finite_or_inf(rec.get("best_holdout_p95_abs_z_qbeta")),
            "best_holdout_p95_abs_z_logft": _finite_or_inf(rec.get("best_holdout_p95_abs_z_logft")),
            "best_overfit_gap_qbeta": _finite_or_inf(rec.get("best_overfit_gap_qbeta")),
            "best_stability_hard_metric_score": _finite_or_inf(rec.get("best_stability_hard_metric_score")),
            "best_stability_violation_score": _finite_or_inf(rec.get("best_stability_violation_score")),
        }
        rank_key = (
            0 if int(row["watch3_all_true_candidates_n"]) > 0 else 1,
            -int(row["watch3_all_true_candidates_n"]),
            -float(row["watch3_all_true_candidates_ratio"]),
            -int(row["n_candidates_combined_guard_pass"]),
            -int(row["stable_candidates_n"]),
            float(row["best_stability_hard_metric_score"]),
            float(row["best_residue_reweight_score"]),
            float(row["best_holdout_p95_abs_z_qbeta"]),
            float(row["best_overfit_gap_qbeta"]),
        )
        row["_rank_key"] = rank_key
        summary_rows.append(row)
        if best_key is None or rank_key < best_key:
            best_key = rank_key
            best_summary = dict(row)
            best_retune = retune_pack

    if not summary_rows:
        return {
            "status": "not_run",
            "reason": "no_evaluable_strategies",
            "strategy_count": int(len(normalized_defs)),
            "all_strategies": [],
            "recommended_strategy": {},
            "recommended_reweight": {},
            "recommended_stability_lock": {},
            "recommended_retune_candidate": {},
            "recommended_retune_grid_spec": {},
            "applied_best": False,
        }

    summary_sorted = sorted(summary_rows, key=lambda r: tuple(r.get("_rank_key", (float("inf"),))))
    for idx, row in enumerate(summary_sorted, start=1):
        row["rank_strategy"] = int(idx)
        row.pop("_rank_key", None)
    if best_summary:
        best_summary.pop("_rank_key", None)

    return {
        "status": "completed",
        "strategy_count": int(len(summary_sorted)),
        "all_strategies": summary_sorted,
        "recommended_strategy": best_summary,
        "recommended_reweight": (
            best_retune.get("recommended_reweight")
            if isinstance(best_retune.get("recommended_reweight"), dict)
            else {}
        ),
        "recommended_stability_lock": (
            best_retune.get("recommended_stability_lock")
            if isinstance(best_retune.get("recommended_stability_lock"), dict)
            else {}
        ),
        "recommended_retune_candidate": (
            best_retune.get("recommended_candidate")
            if isinstance(best_retune.get("recommended_candidate"), dict)
            else {}
        ),
        "recommended_retune_grid_spec": (
            best_retune.get("grid_spec")
            if isinstance(best_retune.get("grid_spec"), dict)
            else {}
        ),
        "applied_best": False,
    }


def _build_candidate_stability_lock(
    *,
    reweight_pack: Dict[str, Any],
    pre_local_holdout_p95_qbeta: float,
    pre_local_holdout_p95_logft: float,
    pre_local_overfit_gap_qbeta: float,
    pre_local_n_qbeta_gt3_holdout: int,
    pre_local_n_logft_gt3_holdout: int,
    logft_rootcause_refreeze_policy: Dict[str, Any] | None = None,
    qbeta_count_recovery_policy: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    rows = reweight_pack.get("all_candidates")
    if not isinstance(rows, list) or not rows:
        return {
            "status": "not_run",
            "reason": "reweight_candidates_not_available",
            "mode": "not_run",
            "stable_candidates_n": 0,
            "recommended_candidate": {},
            "all_candidates": [],
            "top_ranked_candidates": [],
            "reject_root_cause": [],
        }

    evaluated: List[Dict[str, Any]] = []
    fail_counter: Dict[str, int] = {
        "prelocal_qbeta_nonworsening": 0,
        "prelocal_qbeta_count_nonworsening": 0,
        "prelocal_logft_nonworsening": 0,
        "prelocal_overfit_nonworsening": 0,
        "prelocal_logft_count_nonworsening": 0,
        "residue_qfail_nonregression": 0,
        "residue_top_priority_nonregression": 0,
        "residue_logft_count_nonregression": 0,
        "residue_logft_nonworsening": 0,
        "residue_overfit_nonworsening": 0,
    }
    weights_violation = {
        "p95_qbeta_vs_pre": 0.30,
        "overfit_gap_vs_pre": 0.25,
        "p95_logft_vs_pre": 0.15,
        "n_logft_vs_pre": 0.10,
        "residue_delta_top": 0.08,
        "residue_delta_overfit": 0.07,
        "residue_delta_logft_count": 0.06,
        "residue_delta_qfail": 0.03,
        "residue_logft_spread": 0.02,
        "residue_qbeta_spread": 0.01,
    }
    refreeze_policy = (
        logft_rootcause_refreeze_policy
        if isinstance(logft_rootcause_refreeze_policy, dict)
        else {}
    )
    use_refrozen_rootcause = bool(refreeze_policy.get("enabled"))
    refrozen_logft_count_limit = int(
        refreeze_policy.get("refrozen_logft_count_max_delta_allowed", 0)
    )
    refrozen_residue_logft_count_limit = int(
        refreeze_policy.get("refrozen_residue_logft_count_max_delta_allowed", 0)
    )
    refrozen_require_residue_logft_nonworsening = bool(
        refreeze_policy.get("refrozen_require_residue_logft_nonworsening", True)
    )
    use_residue_dual_guard = bool(refreeze_policy.get("residue_dual_guard_enabled"))
    refrozen_residue_top_priority_limit = int(
        refreeze_policy.get("refrozen_residue_top_priority_max_delta_allowed", 0)
    )
    refrozen_residue_overfit_limit = _to_float(
        refreeze_policy.get("refrozen_residue_overfit_max_delta_allowed")
    )
    if not math.isfinite(refrozen_residue_overfit_limit):
        refrozen_residue_overfit_limit = 0.0
    qbeta_recovery_policy = qbeta_count_recovery_policy if isinstance(qbeta_count_recovery_policy, dict) else {}
    qbeta_count_max_delta = int(qbeta_recovery_policy.get("qbeta_count_max_delta", 0) or 0)
    qbeta_count_recovery_enabled = bool(qbeta_recovery_policy.get("enabled", False))
    qbeta_count_recovery_logft_p95_max_delta = _to_float(
        qbeta_recovery_policy.get("logft_p95_max_delta", 0.0)
    )
    if not math.isfinite(qbeta_count_recovery_logft_p95_max_delta):
        qbeta_count_recovery_logft_p95_max_delta = 0.0
    qbeta_count_recovery_logft_count_max_delta = int(
        qbeta_recovery_policy.get("logft_count_max_delta", 0) or 0
    )
    qbeta_count_recovery_residue_logft_count_max_delta = int(
        qbeta_recovery_policy.get("residue_logft_count_max_delta", 0) or 0
    )
    qbeta_count_recovery_residue_overfit_max_delta = _to_float(
        qbeta_recovery_policy.get("residue_overfit_max_delta", 0.0)
    )
    if not math.isfinite(qbeta_count_recovery_residue_overfit_max_delta):
        qbeta_count_recovery_residue_overfit_max_delta = 0.0

    for row_in in rows:
        if not isinstance(row_in, dict):
            continue
        row = dict(row_in)
        d_q = _to_float(row.get("holdout_p95_abs_z_qbeta")) - float(pre_local_holdout_p95_qbeta)
        d_l = _to_float(row.get("holdout_p95_abs_z_logft")) - float(pre_local_holdout_p95_logft)
        d_g = _to_float(row.get("overfit_gap_qbeta")) - float(pre_local_overfit_gap_qbeta)
        d_n_q = int(row.get("n_qbeta_gt3_holdout") or 0) - int(pre_local_n_qbeta_gt3_holdout)
        d_n_log = int(row.get("n_logft_gt3_holdout") or 0) - int(pre_local_n_logft_gt3_holdout)
        residue_overfit_delta = max(0.0, _to_float(row.get("residue_max_pos_delta_overfit_gap_qbeta")))
        if not math.isfinite(residue_overfit_delta):
            residue_overfit_delta = 0.0 if bool(row.get("residue_all_overfit_nonworsening") is True) else float("inf")
        qbeta_count_guard_pass = bool(d_n_q <= int(qbeta_count_max_delta))
        qbeta_count_recovery_relax_active = bool(
            qbeta_count_recovery_enabled
            and qbeta_count_guard_pass
            and d_q <= 0.0
            and d_g <= 0.0
        )
        logft_p95_limit = float(qbeta_count_recovery_logft_p95_max_delta) if qbeta_count_recovery_relax_active else 0.0
        logft_count_limit_base = int(refrozen_logft_count_limit if use_refrozen_rootcause else 0)
        logft_count_limit = max(
            logft_count_limit_base,
            int(qbeta_count_recovery_logft_count_max_delta) if qbeta_count_recovery_relax_active else 0,
        )
        residue_logft_count_limit_base = int(refrozen_residue_logft_count_limit if use_refrozen_rootcause else 0)
        residue_logft_count_limit = max(
            residue_logft_count_limit_base,
            int(qbeta_count_recovery_residue_logft_count_max_delta) if qbeta_count_recovery_relax_active else 0,
        )
        residue_overfit_limit = (
            float(refrozen_residue_overfit_limit) if bool(use_residue_dual_guard) else 0.0
        )
        if qbeta_count_recovery_relax_active:
            residue_overfit_limit = max(residue_overfit_limit, float(qbeta_count_recovery_residue_overfit_max_delta))

        gate_map = {
            "prelocal_qbeta_nonworsening": bool(d_q <= 0.0),
            "prelocal_qbeta_count_nonworsening": bool(qbeta_count_guard_pass),
            "prelocal_logft_nonworsening": bool(d_l <= float(logft_p95_limit)),
            "prelocal_overfit_nonworsening": bool(d_g <= 0.0),
            "prelocal_logft_count_nonworsening": bool(d_n_log <= int(logft_count_limit)),
            "residue_qfail_nonregression": bool(int(row.get("residue_max_pos_delta_n_qbeta_gt3") or 0) <= 0),
            "residue_top_priority_nonregression": bool(
                int(row.get("residue_max_pos_delta_top_priority_q_gt3") or 0)
                <= (refrozen_residue_top_priority_limit if bool(use_residue_dual_guard) else 0)
            ),
            "residue_logft_count_nonregression": bool(
                int(row.get("residue_max_pos_delta_n_logft_gt3") or 0)
                <= int(residue_logft_count_limit)
            ),
            "residue_logft_nonworsening": bool(
                (
                    bool(row.get("residue_all_logft_nonworsening"))
                    if bool(refrozen_require_residue_logft_nonworsening)
                    else True
                )
                if bool(use_refrozen_rootcause)
                else bool(row.get("residue_all_logft_nonworsening") is True)
            ),
            "residue_overfit_nonworsening": bool(
                residue_overfit_delta <= float(residue_overfit_limit)
            ),
        }
        for key, ok in gate_map.items():
            if not ok:
                fail_counter[key] = int(fail_counter.get(key, 0) + 1)

        penalties = {
            "p95_qbeta_vs_pre": max(0.0, d_q),
            "p95_logft_vs_pre": max(0.0, d_l),
            "overfit_gap_vs_pre": max(0.0, d_g),
            "n_logft_vs_pre": float(max(0, d_n_log)),
            "residue_delta_qfail": float(max(0, int(row.get("residue_max_pos_delta_n_qbeta_gt3") or 0))),
            "residue_delta_top": float(max(0, int(row.get("residue_max_pos_delta_top_priority_q_gt3") or 0))),
            "residue_delta_overfit": float(max(0.0, residue_overfit_delta)),
            "residue_delta_logft_count": float(max(0, int(row.get("residue_max_pos_delta_n_logft_gt3") or 0))),
            "residue_logft_spread": max(0.0, _to_float(row.get("residue_holdout_p95_abs_z_logft_spread"))),
            "residue_qbeta_spread": max(0.0, _to_float(row.get("residue_holdout_p95_abs_z_qbeta_spread"))),
        }

        violation_score = 0.0
        for key, weight in weights_violation.items():
            violation_score += float(weight) * float(penalties.get(key, 0.0))

        row["delta_holdout_p95_abs_z_qbeta_vs_pre"] = d_q
        row["delta_holdout_p95_abs_z_logft_vs_pre"] = d_l
        row["delta_overfit_gap_qbeta_vs_pre"] = d_g
        row["delta_n_qbeta_gt3_holdout_vs_pre"] = int(d_n_q)
        row["delta_n_logft_gt3_holdout_vs_pre"] = int(d_n_log)
        row["residue_max_pos_delta_overfit_gap_qbeta"] = residue_overfit_delta
        row["qbeta_count_recovery_relax_active"] = bool(qbeta_count_recovery_relax_active)
        row["qbeta_count_recovery_logft_p95_max_delta_used"] = float(logft_p95_limit)
        row["qbeta_count_recovery_logft_count_max_delta_used"] = int(logft_count_limit)
        row["qbeta_count_recovery_residue_logft_count_max_delta_used"] = int(residue_logft_count_limit)
        row["qbeta_count_recovery_residue_overfit_max_delta_used"] = float(residue_overfit_limit)
        row["stability_gate_flags"] = gate_map
        row["stability_gate_pass_all"] = bool(all(gate_map.values()))
        row["stability_violation_score"] = float(violation_score)
        row["stability_violation_count"] = int(sum(0 if ok else 1 for ok in gate_map.values()))
        evaluated.append(row)

    if not evaluated:
        return {
            "status": "not_run",
            "reason": "no_valid_candidate_rows",
            "mode": "not_run",
            "stable_candidates_n": 0,
            "recommended_candidate": {},
            "all_candidates": [],
            "top_ranked_candidates": [],
            "reject_root_cause": [],
        }

    weights_hard_metric = {
        "holdout_p95_abs_z_qbeta": 0.52,
        "overfit_gap_qbeta": 0.24,
        "holdout_p95_abs_z_logft": 0.12,
        "n_qbeta_gt3_holdout": 0.08,
        "n_logft_gt3_holdout": 0.04,
    }
    q_vals = [_to_float(r.get("holdout_p95_abs_z_qbeta")) for r in evaluated]
    g_vals = [_to_float(r.get("overfit_gap_qbeta")) for r in evaluated]
    l_vals = [_to_float(r.get("holdout_p95_abs_z_logft")) for r in evaluated]
    nq_vals = [float(max(0, int(r.get("n_qbeta_gt3_holdout") or 0))) for r in evaluated]
    nl_vals = [float(max(0, int(r.get("n_logft_gt3_holdout") or 0))) for r in evaluated]
    q_min, q_max = min(q_vals), max(q_vals)
    g_min, g_max = min(g_vals), max(g_vals)
    l_min, l_max = min(l_vals), max(l_vals)
    nq_min, nq_max = min(nq_vals), max(nq_vals)
    nl_min, nl_max = min(nl_vals), max(nl_vals)
    for row in evaluated:
        n_q = _safe_norm(_to_float(row.get("holdout_p95_abs_z_qbeta")), q_min, q_max)
        n_g = _safe_norm(_to_float(row.get("overfit_gap_qbeta")), g_min, g_max)
        n_l = _safe_norm(_to_float(row.get("holdout_p95_abs_z_logft")), l_min, l_max)
        n_nq = _safe_norm(float(max(0, int(row.get("n_qbeta_gt3_holdout") or 0))), nq_min, nq_max)
        n_nl = _safe_norm(float(max(0, int(row.get("n_logft_gt3_holdout") or 0))), nl_min, nl_max)
        row["stability_hard_metric_score"] = float(
            weights_hard_metric["holdout_p95_abs_z_qbeta"] * n_q
            + weights_hard_metric["overfit_gap_qbeta"] * n_g
            + weights_hard_metric["holdout_p95_abs_z_logft"] * n_l
            + weights_hard_metric["n_qbeta_gt3_holdout"] * n_nq
            + weights_hard_metric["n_logft_gt3_holdout"] * n_nl
        )

    stable_rows = [r for r in evaluated if bool(r.get("stability_gate_pass_all"))]
    ranked = sorted(
        evaluated,
        key=lambda r: (
            0 if bool(r.get("stability_gate_pass_all")) else 1,
            float(r.get("stability_hard_metric_score", float("inf"))),
            float(r.get("holdout_p95_abs_z_qbeta", float("inf"))),
            float(r.get("overfit_gap_qbeta", float("inf"))),
            float(r.get("holdout_p95_abs_z_logft", float("inf"))),
            float(r.get("stability_violation_score", float("inf"))),
            int(r.get("stability_violation_count", 999999)),
        ),
    )
    for idx, row in enumerate(ranked, start=1):
        row["rank_stability_lock"] = int(idx)

    reject_root_cause = sorted(
        [{"gate_id": k, "fail_count": int(v)} for k, v in fail_counter.items() if int(v) > 0],
        key=lambda x: (-int(x.get("fail_count", 0)), str(x.get("gate_id", ""))),
    )
    mode = "stable_locked" if stable_rows else "reject_locked"
    recommended = ranked[0] if ranked else {}
    return {
        "status": "completed",
        "mode": mode,
        "stable_candidates_n": int(len(stable_rows)),
        "evaluated_candidates_n": int(len(evaluated)),
        "recommended_candidate": recommended,
        "all_candidates": evaluated,
        "top_ranked_candidates": ranked[:20],
        "reject_root_cause": reject_root_cause[:10],
        "selection_policy": {
            "mode": "candidate_stability_lock",
            "requires_all_gates": True,
            "stable_mode": "stable_locked_if_any_else_reject_locked",
            "prelocal_reference": {
                "holdout_p95_abs_z_qbeta": float(pre_local_holdout_p95_qbeta),
                "holdout_p95_abs_z_logft": float(pre_local_holdout_p95_logft),
                "overfit_gap_qbeta": float(pre_local_overfit_gap_qbeta),
                "n_qbeta_gt3_holdout": int(pre_local_n_qbeta_gt3_holdout),
                "n_logft_gt3_holdout": int(pre_local_n_logft_gt3_holdout),
            },
            "qbeta_count_recovery_policy": {
                "enabled": bool(qbeta_count_recovery_enabled),
                "qbeta_count_max_delta": int(qbeta_count_max_delta),
                "logft_p95_max_delta": float(qbeta_count_recovery_logft_p95_max_delta),
                "logft_count_max_delta": int(qbeta_count_recovery_logft_count_max_delta),
                "residue_logft_count_max_delta": int(qbeta_count_recovery_residue_logft_count_max_delta),
                "residue_overfit_max_delta": float(qbeta_count_recovery_residue_overfit_max_delta),
            },
            "logft_rootcause_refreeze_policy": {
                "enabled": bool(use_refrozen_rootcause),
                "refrozen_logft_count_max_delta_allowed": int(refrozen_logft_count_limit),
                "refrozen_residue_logft_count_max_delta_allowed": int(refrozen_residue_logft_count_limit),
                "refrozen_require_residue_logft_nonworsening": bool(refrozen_require_residue_logft_nonworsening),
            },
            "residue_dual_refreeze_policy": {
                "enabled": bool(use_residue_dual_guard),
                "refrozen_residue_top_priority_max_delta_allowed": int(refrozen_residue_top_priority_limit),
                "refrozen_residue_overfit_max_delta_allowed": float(refrozen_residue_overfit_limit),
            },
            "weights_violation_score": weights_violation,
            "weights_hard_metric_score": weights_hard_metric,
        },
    }


def _plot_residue_robustness(*, residue_pack: Dict[str, Any], out_png: Path) -> None:
    rows = residue_pack.get("rows")
    if not isinstance(rows, list) or not rows:
        return
    rows_sorted = sorted(rows, key=lambda r: int(r.get("residue", 0)))
    x = [int(r.get("residue", 0)) for r in rows_sorted]
    q = [_to_float(r.get("holdout_p95_abs_z_qbeta")) for r in rows_sorted]
    gap = [_to_float(r.get("overfit_gap_qbeta")) for r in rows_sorted]
    logft = [_to_float(r.get("holdout_p95_abs_z_logft")) for r in rows_sorted]
    n_q = [int(r.get("n_qbeta_gt3_holdout") or 0) for r in rows_sorted]
    n_log = [int(r.get("n_logft_gt3_holdout") or 0) for r in rows_sorted]
    ref = int(residue_pack.get("reference_residue", 0))

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 5.2), dpi=180)
    ax0, ax1, ax2 = axes

    ax0.plot(x, q, marker="o", color="#4c78a8", label="holdout p95 |z_Qβ|")
    ax0.plot(x, gap, marker="s", color="#54a24b", label="overfit gap qbeta")
    ax0.axvline(ref, color="#444444", linestyle=":", linewidth=1.1)
    ax0.set_xlabel("holdout residue")
    ax0.set_ylabel("value")
    ax0.set_title("Qβ / overfit robustness")
    ax0.grid(True, alpha=0.25, linestyle=":")
    ax0.legend(loc="best", fontsize=8)

    ax1.plot(x, logft, marker="o", color="#e45756", label="holdout p95 |z_logft|")
    ax1.axvline(ref, color="#444444", linestyle=":", linewidth=1.1)
    ax1.set_xlabel("holdout residue")
    ax1.set_ylabel("abs(z)")
    ax1.set_title("logft robustness")
    ax1.grid(True, alpha=0.25, linestyle=":")
    ax1.legend(loc="best", fontsize=8)

    ax2.plot(x, n_q, marker="o", color="#9467bd", label="n_qbeta_gt3")
    ax2.plot(x, n_log, marker="s", color="#f58518", label="n_logft_gt3")
    ax2.axvline(ref, color="#444444", linestyle=":", linewidth=1.1)
    ax2.set_xlabel("holdout residue")
    ax2.set_ylabel("count")
    ax2.set_title("fail-count robustness")
    ax2.grid(True, alpha=0.25, linestyle=":")
    ax2.legend(loc="best", fontsize=8)

    fig.suptitle("Step 8.7.31.8-11: holdout-residue robustness audit", fontsize=13)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.94])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def _plot(
    *,
    route_b_eval: Dict[str, Any],
    decision: Dict[str, Any],
    closure_gate: Dict[str, Any],
    out_png: Path,
) -> None:
    holdout_all = route_b_eval.get("holdout_all") if isinstance(route_b_eval.get("holdout_all"), dict) else {}
    overfit_guard = route_b_eval.get("overfit_guard") if isinstance(route_b_eval.get("overfit_guard"), dict) else {}

    p95_q = _to_float(holdout_all.get("p95_abs_z_qbeta"))
    p95_logft = _to_float(holdout_all.get("p95_abs_z_logft_proxy"))
    gate = _to_float(decision.get("gate_values", {}).get("z_gate"))

    gap_q = _to_float(overfit_guard.get("p95_gap_qbeta"))
    gap_logft = _to_float(overfit_guard.get("p95_gap_logft_proxy"))
    gap_gate = _to_float(decision.get("gate_values", {}).get("overfit_gap_gate"))

    ckm_abs_z = _to_float((closure_gate.get("ckm_gate") or {}).get("abs_z_reported"))
    pmns_abs_z = _to_float((closure_gate.get("pmns_gate") or {}).get("abs_z_center_proxy"))
    closure_watch_gate = _to_float((closure_gate.get("ckm_gate") or {}).get("watch_z_threshold"))

    fig, axes = plt.subplots(1, 3, figsize=(14.4, 5.0), dpi=180)

    ax0 = axes[0]
    ax0.bar([0, 1], [p95_q, p95_logft], color=["#4c78a8", "#e45756"])
    ax0.axhline(gate, color="#333333", linestyle="--", linewidth=1.2)
    ax0.set_yscale("log")
    ax0.set_xticks([0, 1], ["holdout p95 |z_Qβ|", "holdout p95 |z_logft|"])
    ax0.set_ylabel("abs(z)")
    ax0.set_title("Route-B hard gate")
    ax0.grid(True, axis="y", alpha=0.25, linestyle=":")

    ax1 = axes[1]
    ax1.bar([0, 1], [gap_q, gap_logft], color=["#54a24b", "#f58518"])
    ax1.axhline(0.0, color="#444444", linewidth=0.9)
    ax1.axhline(gap_gate, color="#444444", linestyle="--", linewidth=1.0)
    ax1.set_xticks([0, 1], ["gap p95 z_Qβ", "gap p95 z_logft"])
    ax1.set_ylabel("holdout - train")
    ax1.set_title("Overfit guard")
    ax1.grid(True, axis="y", alpha=0.25, linestyle=":")

    ax2 = axes[2]
    labels = ["CKM abs(z)", "PMNS abs(z)"]
    vals = [ckm_abs_z, pmns_abs_z]
    colors = ["#d62728" if math.isfinite(v) and math.isfinite(closure_watch_gate) and v > closure_watch_gate else "#2ca02c" for v in vals]
    ax2.bar([0, 1], vals, color=colors)
    if math.isfinite(closure_watch_gate):
        ax2.axhline(closure_watch_gate, color="#333333", linestyle="--", linewidth=1.1)
    ax2.set_xticks([0, 1], labels)
    ax2.set_ylabel("abs(z)")
    ax2.set_title("CKM/PMNS closure watch gate")
    ax2.grid(True, axis="y", alpha=0.25, linestyle=":")

    fig.suptitle(
        f"Step 8.7.31: weak-interaction Route-B standalone audit ({decision.get('overall_status')})",
        fontsize=13,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.93])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 8.7.31: B-route standalone weak-interaction audit (same I/F: Qβ/logft/CKM/PMNS)."
    )
    parser.add_argument(
        "--in-csv",
        type=Path,
        default=ROOT / "output" / "public" / "quantum" / "nuclear_beta_decay_qvalue_prediction_full.csv",
    )
    parser.add_argument(
        "--in-json",
        type=Path,
        default=ROOT / "output" / "public" / "quantum" / "nuclear_beta_decay_qvalue_prediction_metrics.json",
    )
    parser.add_argument(
        "--route-ab-audit-json",
        type=Path,
        default=ROOT / "output" / "public" / "quantum" / "weak_interaction_beta_decay_route_ab_audit.json",
    )
    parser.add_argument(
        "--ckm-audit-json",
        type=Path,
        default=ROOT / "output" / "public" / "quantum" / "weak_interaction_ckm_first_row_audit.json",
    )
    parser.add_argument(
        "--pmns-audit-json",
        type=Path,
        default=ROOT / "output" / "public" / "quantum" / "weak_interaction_pmns_first_row_audit.json",
    )
    parser.add_argument("--logft-sigma-proxy", type=float, default=1.0)
    parser.add_argument("--z-gate", type=float, default=3.0)
    parser.add_argument("--holdout-hash-modulo", type=int, default=5)
    parser.add_argument("--holdout-hash-residue", type=int, default=0)
    parser.add_argument("--overfit-gap-gate", type=float, default=1.0)
    parser.add_argument("--sigma-floor-mev", type=float, default=1.0e-9)
    parser.add_argument(
        "--hflavor-mode",
        type=str,
        choices=["baseline", "hflavor_v1"],
        default="hflavor_v1",
        help="Route-B remapping mode before equalized audit.",
    )
    parser.add_argument(
        "--hflavor-before-mix",
        type=float,
        default=0.8,
        help="Mix coefficient for q_mix = q_after + mix*(q_before-q_after).",
    )
    parser.add_argument(
        "--hflavor-sat-scale-mev",
        type=float,
        default=float("nan"),
        help="Saturation scale [MeV]. If non-finite or <=0, use quantile-derived freeze.",
    )
    parser.add_argument(
        "--hflavor-sat-quantile",
        type=float,
        default=0.95,
        help="Quantile of abs(q_pred_after) used when sat-scale is auto-derived.",
    )
    parser.add_argument(
        "--hflavor-branch-pivot-mev",
        type=float,
        default=2.0,
        help="Low-energy branch pivot |q_sat| [MeV]. If <=0, use quantile-derived freeze.",
    )
    parser.add_argument(
        "--hflavor-branch-pivot-quantile",
        type=float,
        default=0.35,
        help="Quantile of abs(q_pred_after) used when branch-pivot is auto-derived.",
    )
    parser.add_argument(
        "--hflavor-branch-gain",
        type=float,
        default=1.6,
        help="Low-energy branch gain for |q_sat| below pivot.",
    )
    parser.add_argument(
        "--hflavor-sign-blend",
        type=float,
        default=0.05,
        help="Blend ratio [0,1] for soft channel-sign stabilization.",
    )
    parser.add_argument(
        "--run-hflavor-sweep",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run holdout-fixed H_flavor(P) parameter sweep and emit Pareto outputs.",
    )
    parser.add_argument(
        "--sweep-before-mix-values",
        type=str,
        default="0.6,0.8,1.0",
        help="Comma-separated before_mix values for sweep.",
    )
    parser.add_argument(
        "--sweep-sat-scale-multipliers",
        type=str,
        default="0.85,1.0,1.15",
        help="Comma-separated multipliers applied to base sat_scale_mev.",
    )
    parser.add_argument(
        "--sweep-branch-pivot-values",
        type=str,
        default="1.0,2.0,3.0",
        help="Comma-separated branch_pivot_mev values for sweep.",
    )
    parser.add_argument(
        "--sweep-branch-gain-values",
        type=str,
        default="1.2,1.6,2.0",
        help="Comma-separated branch_gain values for sweep.",
    )
    parser.add_argument(
        "--sweep-sign-blend-values",
        type=str,
        default="0.0,0.05",
        help="Comma-separated sign_blend values for sweep.",
    )
    parser.add_argument(
        "--run-localcorr-sweep",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run targeted transition-class local correction sweep (Step 8.7.31.5).",
    )
    parser.add_argument(
        "--localcorr-top-classes",
        type=int,
        default=2,
        help="Number of top transition classes from outlier decomposition to target.",
    )
    parser.add_argument(
        "--localcorr-gain-values",
        type=str,
        default="0.6,0.8,1.0",
        help="Comma-separated local correction gain values.",
    )
    parser.add_argument(
        "--localcorr-sat-mev-values",
        type=str,
        default="3.0,5.0,8.0,12.0",
        help="Comma-separated local correction saturation scales [MeV].",
    )
    parser.add_argument(
        "--localcorr-blend-values",
        type=str,
        default="0.4,0.6,0.8",
        help="Comma-separated local correction blend values in [0,1].",
    )
    parser.add_argument(
        "--localcorr-sign-blend-values",
        type=str,
        default="0.0,0.1",
        help="Comma-separated local correction sign-blend values in [0,1].",
    )
    parser.add_argument(
        "--localcorr-top-class-gain-boost-values",
        type=str,
        default="1.0,1.2,1.4",
        help="Comma-separated gain boost multipliers applied only to the top-priority transition class.",
    )
    parser.add_argument(
        "--localcorr-top-class-sign-blend-scale-values",
        type=str,
        default="1.0",
        help="Comma-separated sign-blend scale multipliers applied only to the top-priority transition class.",
    )
    parser.add_argument(
        "--localcorr-sigma-low-mode-inconsistent-sat-scale-values",
        type=str,
        default="1.0,0.99",
        help="Comma-separated saturation scale multipliers applied only to sigma_low|mode_inconsistent target classes.",
    )
    parser.add_argument(
        "--localcorr-sigma-low-mode-inconsistent-zref-values",
        type=str,
        default="5.0",
        help="Comma-separated z-reference values for sigma_low|mode_inconsistent blend taper (higher => broader taper).",
    )
    parser.add_argument(
        "--localcorr-sigma-low-mode-inconsistent-min-blend-scale-values",
        type=str,
        default="1.0,0.7",
        help="Comma-separated minimum blend-scale values for sigma_low|mode_inconsistent z-taper (1.0 disables taper).",
    )
    parser.add_argument(
        "--localcorr-sigma-low-mode-inconsistent-sign-blend-scale-values",
        type=str,
        default="1.0",
        help="Comma-separated sign-blend scale multipliers applied only to sigma_low|mode_inconsistent target classes.",
    )
    parser.add_argument(
        "--localcorr-logft-max-delta",
        type=float,
        default=0.0,
        help="Max allowed worsening for holdout p95 |z_logft| vs pre-local (<=0 keeps non-worsening only).",
    )
    parser.add_argument(
        "--localcorr-logft-max-abs",
        type=float,
        default=float("nan"),
        help="Optional absolute upper bound for holdout p95 |z_logft|; NaN disables.",
    )
    parser.add_argument(
        "--localcorr-constrained-select",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Select local-correction candidate under logft constraint (Step 8.7.31.6).",
    )
    parser.add_argument(
        "--localcorr-use-qfail-guard",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require non-regression guard on holdout q-fail counts/top-priority class (Step 8.7.31.7).",
    )
    parser.add_argument(
        "--localcorr-qfail-max-delta",
        type=int,
        default=0,
        help="Max allowed delta for n_qbeta_gt3_holdout vs pre-local when qfail guard is enabled.",
    )
    parser.add_argument(
        "--localcorr-top-priority-max-delta",
        type=int,
        default=0,
        help="Max allowed delta for top-priority class q-fail count vs pre-local when qfail guard is enabled.",
    )
    parser.add_argument(
        "--run-residue-robustness",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run holdout-residue robustness audit for the selected local-correction candidate (Step 8.7.31.8).",
    )
    parser.add_argument(
        "--residue-robustness-residues",
        type=str,
        default="0,1,2,3,4",
        help="Comma-separated holdout residues to audit. Empty means 0..(modulo-1).",
    )
    parser.add_argument(
        "--localcorr-residue-reweight",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Re-select constrained local-correction candidate by residue-cross robust multi-objective score (Step 8.7.31.9).",
    )
    parser.add_argument(
        "--localcorr-residue-reweight-max-candidates",
        type=int,
        default=24,
        help="Max number of constrained candidates to evaluate in residue-cross reweighting.",
    )
    parser.add_argument(
        "--localcorr-residue-classwise-norm-blend",
        type=float,
        default=0.0,
        help="Blend ratio in [0,1] between global and class-wise normalization for residue reweight scoring.",
    )
    parser.add_argument(
        "--localcorr-residue-class-top-weight-by-key",
        type=str,
        default="",
        help="Comma-separated class:multiplier list applied to top-priority residue term in reweight (e.g. beta_minus|sigma_low|mode_inconsistent:1.5).",
    )
    parser.add_argument(
        "--localcorr-residue-class-logft-weight-by-key",
        type=str,
        default="",
        help="Comma-separated class:multiplier list applied to logft-count residue term in reweight (e.g. beta_minus|sigma_low|mode_inconsistent:1.3).",
    )
    parser.add_argument(
        "--localcorr-residue-class-retune-grid",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run class-specific 2-axis retune grid (logft thresholds x class top/logft weights) for residue reweight selection.",
    )
    parser.add_argument(
        "--localcorr-residue-class-retune-apply-best",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply best combination from class-specific retune grid before candidate stability lock.",
    )
    parser.add_argument(
        "--localcorr-residue-class-retune-target-class-key",
        type=str,
        default="beta_minus|sigma_low|mode_inconsistent",
        help="Class key to retune class-specific top/logft weights for residue reweight grid.",
    )
    parser.add_argument(
        "--localcorr-residue-class-retune-target-class-keys",
        type=str,
        default="",
        help="Optional comma-separated class keys for simultaneous retune (overrides single target key if provided).",
    )
    parser.add_argument(
        "--localcorr-residue-class-retune-target-top-k",
        type=int,
        default=1,
        help="If >1 and explicit keys are not provided, retune top-K priority classes simultaneously.",
    )
    parser.add_argument(
        "--localcorr-residue-class-retune-strategy-audit",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Audit multiple target-selection strategies (top-k / explicit groups) and select best strategy before stability lock (Step 8.7.31.29).",
    )
    parser.add_argument(
        "--localcorr-residue-class-retune-target-top-k-values",
        type=str,
        default="",
        help="Optional comma-separated top-k strategy list for class-retune strategy audit (e.g. 1,2,3).",
    )
    parser.add_argument(
        "--localcorr-residue-class-retune-target-class-keys-groups",
        type=str,
        default="",
        help="Optional semicolon-separated explicit class-key groups for strategy audit (group format: key1,key2;key3,key4).",
    )
    parser.add_argument(
        "--localcorr-residue-class-retune-logft-count-max-delta-values",
        type=str,
        default="0,1,2,3,4",
        help="Comma-separated logft-count guard delta values for class retune grid.",
    )
    parser.add_argument(
        "--localcorr-residue-class-retune-residue-logft-count-max-delta-values",
        type=str,
        default="90,95,97,100,110",
        help="Comma-separated residue logft-count guard delta values for class retune grid.",
    )
    parser.add_argument(
        "--localcorr-residue-class-retune-top-weight-values",
        type=str,
        default="1.0,1.3,1.6,1.9",
        help="Comma-separated class-specific top-priority residue weight multipliers for retune grid.",
    )
    parser.add_argument(
        "--localcorr-residue-class-retune-logft-weight-values",
        type=str,
        default="1.0,1.2,1.4,1.6",
        help="Comma-separated class-specific logft-count residue weight multipliers for retune grid.",
    )
    parser.add_argument(
        "--localcorr-class-key-threshold-guard",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable class-key-specific threshold overrides for root-cause/dual guards (Step 8.7.31.32).",
    )
    parser.add_argument(
        "--localcorr-class-logft-count-max-delta-by-key",
        type=str,
        default="",
        help="Comma-separated class:max_delta overrides for pre-local logft-count guard (e.g. beta_minus|sigma_low|mode_inconsistent:2).",
    )
    parser.add_argument(
        "--localcorr-class-residue-logft-count-max-delta-by-key",
        type=str,
        default="",
        help="Comma-separated class:max_delta overrides for residue logft-count guard.",
    )
    parser.add_argument(
        "--localcorr-class-residue-top-priority-max-delta-by-key",
        type=str,
        default="",
        help="Comma-separated class:max_delta overrides for residue top-priority q-fail guard.",
    )
    parser.add_argument(
        "--localcorr-class-residue-overfit-max-delta-by-key",
        type=str,
        default="",
        help="Comma-separated class:max_delta overrides for residue overfit-gap guard.",
    )
    parser.add_argument(
        "--localcorr-qbeta-overfit-guard",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require non-worsening guard on holdout p95 |z_Qbeta| and overfit_gap_qbeta in residue reweighting (Step 8.7.31.11).",
    )
    parser.add_argument(
        "--localcorr-qbeta-max-delta",
        type=float,
        default=0.0,
        help="Max allowed delta for holdout p95 |z_Qbeta| vs pre-local in residue reweighting.",
    )
    parser.add_argument(
        "--localcorr-overfit-max-delta",
        type=float,
        default=0.0,
        help="Max allowed delta for overfit_gap_qbeta vs pre-local in residue reweighting.",
    )
    parser.add_argument(
        "--localcorr-logft-rootcause-guard",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply minimal root-cause guard for logft count/nonworsening in residue reweighting (Step 8.7.31.12).",
    )
    parser.add_argument(
        "--localcorr-logft-count-max-delta",
        type=int,
        default=0,
        help="Max allowed delta for n_logft_gt3_holdout vs pre-local under logft root-cause guard.",
    )
    parser.add_argument(
        "--localcorr-residue-logft-count-max-delta",
        type=int,
        default=0,
        help="Max allowed residue max-positive delta for n_logft_gt3 under logft root-cause guard.",
    )
    parser.add_argument(
        "--localcorr-require-residue-logft-nonworsening",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require residue_all_logft_nonworsening under logft root-cause guard.",
    )
    parser.add_argument(
        "--localcorr-logft-rootcause-refreeze",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Refreeze conservative logft root-cause thresholds from feasible candidate band if strict guard has no pass (Step 8.7.31.13).",
    )
    parser.add_argument(
        "--localcorr-logft-rootcause-retune",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Retune logft root-cause refrozen thresholds from qbeta+dual feasible band when combined guard remains sparse (Step 8.7.31.17).",
    )
    parser.add_argument(
        "--localcorr-logft-rootcause-retune-target-combined-pass-min",
        type=int,
        default=2,
        help="Minimum desired combined guard pass count for Step 8.7.31.17 retune.",
    )
    parser.add_argument(
        "--localcorr-logft-rootcause-retune-max-logft-count-delta",
        type=int,
        default=1,
        help="Maximum allowed retuned delta for n_logft_gt3_holdout vs pre-local (Step 8.7.31.17).",
    )
    parser.add_argument(
        "--localcorr-logft-rootcause-retune-max-residue-logft-count-delta",
        type=int,
        default=95,
        help="Maximum allowed retuned residue max-positive delta for n_logft_gt3 under Step 8.7.31.17.",
    )
    parser.add_argument(
        "--localcorr-residue-dual-guard",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply dual residue guard on top-priority q-fail nonregression and overfit nonworsening (Step 8.7.31.15).",
    )
    parser.add_argument(
        "--localcorr-residue-top-priority-max-delta",
        type=int,
        default=0,
        help="Max allowed residue positive delta for top-priority q-fail count under dual residue guard.",
    )
    parser.add_argument(
        "--localcorr-residue-overfit-max-delta",
        type=float,
        default=0.0,
        help="Max allowed residue positive delta for overfit gap under dual residue guard.",
    )
    parser.add_argument(
        "--localcorr-residue-dual-refreeze",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Refreeze conservative dual-residue thresholds from feasible candidate band if strict dual guard has no pass (Step 8.7.31.15).",
    )
    parser.add_argument(
        "--localcorr-residue-dual-retune",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Retune dual-residue thresholds by weighted top-priority/overfit cost when combined guard has no pass (Step 8.7.31.16).",
    )
    parser.add_argument(
        "--localcorr-residue-dual-retune-top-weight",
        type=float,
        default=0.55,
        help="Weight for top-priority residue delta in Step 8.7.31.16 dual-retune cost.",
    )
    parser.add_argument(
        "--localcorr-residue-dual-retune-overfit-weight",
        type=float,
        default=0.45,
        help="Weight for overfit residue delta in Step 8.7.31.16 dual-retune cost.",
    )
    parser.add_argument(
        "--localcorr-residue-dual-retune-max-top-delta",
        type=int,
        default=4,
        help="Maximum allowed refrozen top-priority delta during Step 8.7.31.16 dual-retune.",
    )
    parser.add_argument(
        "--localcorr-residue-dual-retune-max-overfit-delta",
        type=float,
        default=2000.0,
        help="Maximum allowed refrozen overfit-gap delta during Step 8.7.31.16 dual-retune.",
    )
    parser.add_argument(
        "--run-candidate-stability-lock",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run candidate stability lock (Step 8.7.31.10): freeze stable candidate if all gates pass, else lock reject root-cause.",
    )
    parser.add_argument(
        "--stability-qbeta-count-max-delta",
        type=int,
        default=0,
        help="Max allowed delta for n_qbeta_gt3_holdout vs pre-local in candidate stability lock.",
    )
    parser.add_argument(
        "--stability-qbeta-count-recovery",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable qbeta-count recovery guard relaxation for logft/residue gates when qbeta/overfit are non-worsening.",
    )
    parser.add_argument(
        "--stability-qbeta-count-recovery-logft-p95-max-delta",
        type=float,
        default=0.5,
        help="Allowed logft p95 delta vs pre-local under qbeta-count recovery mode.",
    )
    parser.add_argument(
        "--stability-qbeta-count-recovery-logft-count-max-delta",
        type=int,
        default=4,
        help="Allowed n_logft_gt3_holdout delta vs pre-local under qbeta-count recovery mode.",
    )
    parser.add_argument(
        "--stability-qbeta-count-recovery-residue-logft-count-max-delta",
        type=int,
        default=140,
        help="Allowed residue max positive delta for n_logft_gt3 under qbeta-count recovery mode.",
    )
    parser.add_argument(
        "--stability-qbeta-count-recovery-residue-overfit-max-delta",
        type=float,
        default=1450.0,
        help="Allowed residue max positive delta for overfit gap under qbeta-count recovery mode.",
    )
    parser.add_argument("--outdir", type=Path, default=ROOT / "output" / "public" / "quantum")
    args = parser.parse_args()

    if not args.in_csv.exists():
        raise SystemExit(f"[fail] missing input csv: {args.in_csv}")
    if not args.in_json.exists():
        raise SystemExit(f"[fail] missing input json: {args.in_json}")

    rows = _load_rows(args.in_csv)
    if not rows:
        raise SystemExit(f"[fail] empty input rows: {args.in_csv}")

    _, baseline_pack, baseline_split_meta = route_ab._equalized_route_audit(
        rows=rows,
        logft_sigma_proxy=float(args.logft_sigma_proxy),
        z_gate=float(args.z_gate),
        holdout_hash_modulo=int(args.holdout_hash_modulo),
        holdout_hash_residue=int(args.holdout_hash_residue),
        overfit_gap_gate=float(args.overfit_gap_gate),
        sigma_floor_mev=float(args.sigma_floor_mev),
    )
    baseline_route_eval = (
        baseline_pack.get("route_evaluation") if isinstance(baseline_pack.get("route_evaluation"), dict) else {}
    )
    baseline_route_b_eval = (
        baseline_route_eval.get("B_pmodel_proxy")
        if isinstance(baseline_route_eval.get("B_pmodel_proxy"), dict)
        else {}
    )

    abs_q_after = [abs(_to_float(r.get("q_pred_after_MeV"))) for r in rows]
    sat_auto = _finite_quantile(abs_q_after, float(args.hflavor_sat_quantile))
    pivot_auto = _finite_quantile(abs_q_after, float(args.hflavor_branch_pivot_quantile))
    sat_scale_mev = (
        float(args.hflavor_sat_scale_mev)
        if math.isfinite(float(args.hflavor_sat_scale_mev)) and float(args.hflavor_sat_scale_mev) > 0.0
        else sat_auto
    )
    branch_pivot_mev = (
        float(args.hflavor_branch_pivot_mev)
        if math.isfinite(float(args.hflavor_branch_pivot_mev)) and float(args.hflavor_branch_pivot_mev) > 0.0
        else pivot_auto
    )
    if not math.isfinite(sat_scale_mev) or sat_scale_mev <= 0.0:
        sat_scale_mev = 10.0
    if not math.isfinite(branch_pivot_mev) or branch_pivot_mev < 0.0:
        branch_pivot_mev = 2.0

    if args.hflavor_mode == "hflavor_v1":
        mapped_rows, hflavor_mapping = _apply_hflavor_v1(
            rows=rows,
            before_mix=float(args.hflavor_before_mix),
            sat_scale_mev=float(sat_scale_mev),
            branch_pivot_mev=float(branch_pivot_mev),
            branch_gain=float(args.hflavor_branch_gain),
            sign_blend=float(args.hflavor_sign_blend),
        )
    else:
        mapped_rows = rows
        hflavor_mapping = {
            "mode": "baseline",
            "params_frozen": {
                "before_mix": float("nan"),
                "sat_scale_mev": float("nan"),
                "branch_pivot_mev": float("nan"),
                "branch_gain": float("nan"),
                "sign_blend": float("nan"),
            },
            "transform_counts": {"n_rows": len(rows), "n_transformed": 0},
            "transform_delta_abs_MeV": {"median": 0.0, "p95": 0.0, "max": 0.0},
        }

    _, equalized_pack, split_meta = route_ab._equalized_route_audit(
        rows=mapped_rows,
        logft_sigma_proxy=float(args.logft_sigma_proxy),
        z_gate=float(args.z_gate),
        holdout_hash_modulo=int(args.holdout_hash_modulo),
        holdout_hash_residue=int(args.holdout_hash_residue),
        overfit_gap_gate=float(args.overfit_gap_gate),
        sigma_floor_mev=float(args.sigma_floor_mev),
    )

    route_eval = equalized_pack.get("route_evaluation") if isinstance(equalized_pack.get("route_evaluation"), dict) else {}
    route_b_eval_pre_local = route_eval.get("B_pmodel_proxy") if isinstance(route_eval.get("B_pmodel_proxy"), dict) else {}
    if not route_b_eval_pre_local:
        raise SystemExit("[fail] missing route-B evaluation from equalized pack")

    ckm_gate = route_ab._load_ckm_gate(args.ckm_audit_json)
    pmns_gate = route_ab._load_pmns_gate(args.pmns_audit_json)
    closure_gate = route_ab._combine_closure_gates(ckm_gate=ckm_gate, pmns_gate=pmns_gate)
    closure_gate["ckm_gate"] = ckm_gate
    closure_gate["pmns_gate"] = pmns_gate

    decision_pre_local = _evaluate_route_b(
        route_b_eval=route_b_eval_pre_local,
        closure_gate=closure_gate,
        z_gate=float(args.z_gate),
        overfit_gap_gate=float(args.overfit_gap_gate),
    )

    baseline_hold = baseline_route_b_eval.get("holdout_all") if isinstance(baseline_route_b_eval.get("holdout_all"), dict) else {}
    baseline_overfit = baseline_route_b_eval.get("overfit_guard") if isinstance(baseline_route_b_eval.get("overfit_guard"), dict) else {}
    pre_local_hold = (
        route_b_eval_pre_local.get("holdout_all") if isinstance(route_b_eval_pre_local.get("holdout_all"), dict) else {}
    )
    pre_local_overfit = (
        route_b_eval_pre_local.get("overfit_guard") if isinstance(route_b_eval_pre_local.get("overfit_guard"), dict) else {}
    )

    route_ab_ref = _load_route_ab_transition(args.route_ab_audit_json)

    out_dir = args.outdir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "weak_interaction_beta_decay_route_b_standalone_audit.json"
    out_csv = out_dir / "weak_interaction_beta_decay_route_b_standalone_audit.csv"
    out_png = out_dir / "weak_interaction_beta_decay_route_b_standalone_audit.png"
    out_decomp_csv = out_dir / "weak_interaction_beta_decay_route_b_outlier_decomposition.csv"
    out_decomp_png = out_dir / "weak_interaction_beta_decay_route_b_outlier_decomposition.png"
    out_sweep_csv = out_dir / "weak_interaction_beta_decay_route_b_hflavor_sweep.csv"
    out_sweep_png = out_dir / "weak_interaction_beta_decay_route_b_hflavor_sweep_pareto.png"
    out_localcorr_csv = out_dir / "weak_interaction_beta_decay_route_b_localcorr_sweep.csv"
    out_localcorr_png = out_dir / "weak_interaction_beta_decay_route_b_localcorr_sweep_pareto.png"
    out_localcorr_reweight_csv = out_dir / "weak_interaction_beta_decay_route_b_localcorr_residue_reweight.csv"
    out_localcorr_reweight_retune_csv = out_dir / "weak_interaction_beta_decay_route_b_localcorr_residue_reweight_retune.csv"
    out_localcorr_reweight_retune_strategy_csv = (
        out_dir / "weak_interaction_beta_decay_route_b_localcorr_residue_reweight_retune_strategy.csv"
    )
    out_stability_lock_csv = out_dir / "weak_interaction_beta_decay_route_b_candidate_stability_lock.csv"
    out_residue_csv = out_dir / "weak_interaction_beta_decay_route_b_residue_robustness.csv"
    out_residue_png = out_dir / "weak_interaction_beta_decay_route_b_residue_robustness.png"

    outlier_decomposition_pre_local = _build_route_b_outlier_decomposition(
        rows_mapped=mapped_rows,
        route_b_eval=route_b_eval_pre_local,
        holdout_hash_modulo=int(args.holdout_hash_modulo),
        holdout_hash_residue=int(args.holdout_hash_residue),
        z_gate=float(args.z_gate),
    )

    sweep_before_mix_values = _parse_float_list_csv(args.sweep_before_mix_values)
    sweep_sat_scale_multipliers = _parse_float_list_csv(args.sweep_sat_scale_multipliers)
    sweep_branch_pivot_values = _parse_float_list_csv(args.sweep_branch_pivot_values)
    sweep_branch_gain_values = _parse_float_list_csv(args.sweep_branch_gain_values)
    sweep_sign_blend_values = _parse_float_list_csv(args.sweep_sign_blend_values)
    if not sweep_before_mix_values:
        sweep_before_mix_values = [0.8]
    if not sweep_sat_scale_multipliers:
        sweep_sat_scale_multipliers = [1.0]
    if not sweep_branch_pivot_values:
        sweep_branch_pivot_values = [float(branch_pivot_mev)]
    if not sweep_branch_gain_values:
        sweep_branch_gain_values = [float(args.hflavor_branch_gain)]
    if not sweep_sign_blend_values:
        sweep_sign_blend_values = [float(args.hflavor_sign_blend)]

    if bool(args.run_hflavor_sweep) and args.hflavor_mode != "baseline":
        hflavor_sweep = _build_hflavor_sweep(
            rows=rows,
            logft_sigma_proxy=float(args.logft_sigma_proxy),
            z_gate=float(args.z_gate),
            holdout_hash_modulo=int(args.holdout_hash_modulo),
            holdout_hash_residue=int(args.holdout_hash_residue),
            overfit_gap_gate=float(args.overfit_gap_gate),
            sigma_floor_mev=float(args.sigma_floor_mev),
            base_sat_scale_mev=float(sat_scale_mev),
            before_mix_values=sweep_before_mix_values,
            sat_scale_multipliers=sweep_sat_scale_multipliers,
            branch_pivot_values=sweep_branch_pivot_values,
            branch_gain_values=sweep_branch_gain_values,
            sign_blend_values=sweep_sign_blend_values,
        )
        _write_csv_rows(
            out_sweep_csv,
            hflavor_sweep.get("all_candidates") if isinstance(hflavor_sweep.get("all_candidates"), list) else [],
            fieldnames=[
                "candidate_id",
                "before_mix",
                "sat_scale_mev",
                "sat_scale_multiplier",
                "branch_pivot_mev",
                "branch_gain",
                "sign_blend",
                "holdout_p95_abs_z_qbeta",
                "holdout_p95_abs_z_logft",
                "holdout_max_abs_z_qbeta",
                "train_p95_abs_z_qbeta",
                "overfit_gap_qbeta",
                "overfit_guard_pass",
                "n_holdout_q_rows",
                "n_holdout_logft_rows",
                "split_rows_holdout",
                "mapping_delta_abs_median_mev",
                "mapping_delta_abs_p95_mev",
                "is_pareto_front",
                "norm_holdout_p95_abs_z_qbeta",
                "norm_overfit_gap_qbeta",
                "score_weighted",
                "rank_weighted",
            ],
        )
        _plot_hflavor_sweep_pareto(
            sweep=hflavor_sweep,
            baseline_holdout_p95=_to_float(baseline_hold.get("p95_abs_z_qbeta")),
            baseline_overfit_gap_q=_to_float(baseline_overfit.get("p95_gap_qbeta")),
            current_holdout_p95=_to_float(pre_local_hold.get("p95_abs_z_qbeta")),
            current_overfit_gap_q=_to_float(pre_local_overfit.get("p95_gap_qbeta")),
            out_png=out_sweep_png,
        )
    else:
        hflavor_sweep = {
            "status": "not_run",
            "reason": "run_hflavor_sweep_false_or_baseline_mode",
            "grid_spec": {
                "before_mix_values": sweep_before_mix_values,
                "sat_scale_multipliers": sweep_sat_scale_multipliers,
                "branch_pivot_values": sweep_branch_pivot_values,
                "branch_gain_values": sweep_branch_gain_values,
                "sign_blend_values": sweep_sign_blend_values,
                "base_sat_scale_mev": float(sat_scale_mev),
                "n_total_candidates": 0,
            },
        }

    localcorr_gain_values = _parse_float_list_csv(args.localcorr_gain_values)
    localcorr_sat_mev_values = _parse_float_list_csv(args.localcorr_sat_mev_values)
    localcorr_blend_values = _parse_float_list_csv(args.localcorr_blend_values)
    localcorr_sign_blend_values = _parse_float_list_csv(args.localcorr_sign_blend_values)
    localcorr_top_class_gain_boost_values = _parse_float_list_csv(args.localcorr_top_class_gain_boost_values)
    localcorr_top_class_sign_blend_scale_values = _parse_float_list_csv(
        args.localcorr_top_class_sign_blend_scale_values
    )
    localcorr_sigma_low_mode_inconsistent_sat_scale_values = _parse_float_list_csv(
        args.localcorr_sigma_low_mode_inconsistent_sat_scale_values
    )
    localcorr_sigma_low_mode_inconsistent_zref_values = _parse_float_list_csv(
        args.localcorr_sigma_low_mode_inconsistent_zref_values
    )
    localcorr_sigma_low_mode_inconsistent_min_blend_scale_values = _parse_float_list_csv(
        args.localcorr_sigma_low_mode_inconsistent_min_blend_scale_values
    )
    localcorr_sigma_low_mode_inconsistent_sign_blend_scale_values = _parse_float_list_csv(
        args.localcorr_sigma_low_mode_inconsistent_sign_blend_scale_values
    )
    localcorr_residue_class_top_weight_by_key = _parse_class_weight_map(
        args.localcorr_residue_class_top_weight_by_key
    )
    localcorr_residue_class_logft_weight_by_key = _parse_class_weight_map(
        args.localcorr_residue_class_logft_weight_by_key
    )
    localcorr_class_logft_count_max_delta_by_key = _parse_class_int_map(
        args.localcorr_class_logft_count_max_delta_by_key
    )
    localcorr_class_residue_logft_count_max_delta_by_key = _parse_class_int_map(
        args.localcorr_class_residue_logft_count_max_delta_by_key
    )
    localcorr_class_residue_top_priority_max_delta_by_key = _parse_class_int_map(
        args.localcorr_class_residue_top_priority_max_delta_by_key
    )
    localcorr_class_residue_overfit_max_delta_by_key = _parse_class_nonneg_float_map(
        args.localcorr_class_residue_overfit_max_delta_by_key
    )
    localcorr_class_key_threshold_guard_enabled = bool(args.localcorr_class_key_threshold_guard) and bool(
        localcorr_class_logft_count_max_delta_by_key
        or localcorr_class_residue_logft_count_max_delta_by_key
        or localcorr_class_residue_top_priority_max_delta_by_key
        or localcorr_class_residue_overfit_max_delta_by_key
    )
    localcorr_residue_class_retune_logft_count_max_delta_values = _parse_int_list_csv(
        args.localcorr_residue_class_retune_logft_count_max_delta_values
    )
    localcorr_residue_class_retune_residue_logft_count_max_delta_values = _parse_int_list_csv(
        args.localcorr_residue_class_retune_residue_logft_count_max_delta_values
    )
    localcorr_residue_class_retune_top_weight_values = _parse_float_list_csv(
        args.localcorr_residue_class_retune_top_weight_values
    )
    localcorr_residue_class_retune_logft_weight_values = _parse_float_list_csv(
        args.localcorr_residue_class_retune_logft_weight_values
    )
    localcorr_residue_class_retune_target_class_keys = _parse_str_list_csv(
        args.localcorr_residue_class_retune_target_class_keys
    )
    localcorr_residue_class_retune_target_top_k = max(1, int(args.localcorr_residue_class_retune_target_top_k))
    localcorr_residue_class_retune_target_top_k_values = sorted(
        {max(1, int(v)) for v in _parse_int_list_csv(args.localcorr_residue_class_retune_target_top_k_values)}
    )
    localcorr_residue_class_retune_target_class_keys_groups = _parse_str_group_list(
        args.localcorr_residue_class_retune_target_class_keys_groups
    )
    localcorr_residue_class_retune_strategy_definitions_effective: List[Dict[str, Any]] = []
    localcorr_residue_class_retune_target_keys_effective: List[str] = []
    for key in localcorr_residue_class_retune_target_class_keys:
        key_s = str(key).strip()
        if key_s and key_s not in localcorr_residue_class_retune_target_keys_effective:
            localcorr_residue_class_retune_target_keys_effective.append(key_s)
    if not localcorr_residue_class_retune_target_keys_effective:
        fallback_key = str(args.localcorr_residue_class_retune_target_class_key).strip()
        if fallback_key:
            localcorr_residue_class_retune_target_keys_effective = [fallback_key]
    if not localcorr_residue_class_retune_logft_count_max_delta_values:
        localcorr_residue_class_retune_logft_count_max_delta_values = [int(args.localcorr_logft_count_max_delta)]
    if not localcorr_residue_class_retune_residue_logft_count_max_delta_values:
        localcorr_residue_class_retune_residue_logft_count_max_delta_values = [
            int(args.localcorr_residue_logft_count_max_delta)
        ]
    if not localcorr_residue_class_retune_top_weight_values:
        localcorr_residue_class_retune_top_weight_values = [1.0]
    if not localcorr_residue_class_retune_logft_weight_values:
        localcorr_residue_class_retune_logft_weight_values = [1.0]
    if not localcorr_gain_values:
        localcorr_gain_values = [0.8]
    if not localcorr_sat_mev_values:
        localcorr_sat_mev_values = [5.0]
    if not localcorr_blend_values:
        localcorr_blend_values = [0.6]
    if not localcorr_sign_blend_values:
        localcorr_sign_blend_values = [0.0]
    if not localcorr_top_class_gain_boost_values:
        localcorr_top_class_gain_boost_values = [1.0]
    if not localcorr_top_class_sign_blend_scale_values:
        localcorr_top_class_sign_blend_scale_values = [1.0]
    if not localcorr_sigma_low_mode_inconsistent_sat_scale_values:
        localcorr_sigma_low_mode_inconsistent_sat_scale_values = [1.0]
    if not localcorr_sigma_low_mode_inconsistent_zref_values:
        localcorr_sigma_low_mode_inconsistent_zref_values = [5.0]
    if not localcorr_sigma_low_mode_inconsistent_min_blend_scale_values:
        localcorr_sigma_low_mode_inconsistent_min_blend_scale_values = [1.0]
    if not localcorr_sigma_low_mode_inconsistent_sign_blend_scale_values:
        localcorr_sigma_low_mode_inconsistent_sign_blend_scale_values = [1.0]
    residue_values = _parse_int_list_csv(args.residue_robustness_residues)
    modulo = max(1, int(args.holdout_hash_modulo))
    if not residue_values:
        residue_values = list(range(modulo))

    route_b_eval = route_b_eval_pre_local
    split_meta_final = split_meta
    decision = decision_pre_local
    outlier_decomposition = outlier_decomposition_pre_local
    rows_for_residue_audit = mapped_rows
    local_correction: Dict[str, Any] = {
        "status": "not_run",
        "reason": "run_localcorr_sweep_false_or_baseline_mode",
        "target_classes": [],
    }
    localcorr_sweep: Dict[str, Any] = {
        "status": "not_run",
        "reason": "run_localcorr_sweep_false_or_baseline_mode",
        "target_classes": [],
        "selection_policy": {
            "mode": "not_run",
            "logft_base_pre_local": _to_float(pre_local_hold.get("p95_abs_z_logft_proxy")),
            "logft_delta_max_allowed": float(args.localcorr_logft_max_delta),
            "logft_abs_max_allowed": float(args.localcorr_logft_max_abs),
            "qfail_base_pre_local": int((outlier_decomposition_pre_local.get("counts") or {}).get("n_qbeta_gt3_holdout") or 0),
            "top_priority_qfail_base_pre_local": int(
                (
                    (outlier_decomposition_pre_local.get("reduction_priority_order") or [{}])[0]
                    if isinstance(outlier_decomposition_pre_local.get("reduction_priority_order"), list)
                    and (outlier_decomposition_pre_local.get("reduction_priority_order") or [])
                    else {}
                ).get("n_qbeta_gt3", 0)
            ),
            "qfail_max_delta_allowed": int(args.localcorr_qfail_max_delta),
            "top_priority_qfail_max_delta_allowed": int(args.localcorr_top_priority_max_delta),
            "qfail_guard_enabled": bool(args.localcorr_use_qfail_guard),
            "delta_constraint_enabled": bool(
                math.isfinite(_to_float(pre_local_hold.get("p95_abs_z_logft_proxy")))
                and math.isfinite(float(args.localcorr_logft_max_delta))
            ),
            "abs_constraint_enabled": bool(math.isfinite(float(args.localcorr_logft_max_abs))),
            "n_candidates_total": 0,
            "n_candidates_logft_pass": 0,
            "n_candidates_all_constraints_pass": 0,
        },
        "grid_spec": {
            "gain_values": localcorr_gain_values,
            "sat_mev_values": localcorr_sat_mev_values,
            "blend_values": localcorr_blend_values,
            "sign_blend_values": localcorr_sign_blend_values,
            "top_class_gain_boost_values": localcorr_top_class_gain_boost_values,
            "top_class_sign_blend_scale_values": localcorr_top_class_sign_blend_scale_values,
            "sigma_low_mode_inconsistent_sat_scale_values": localcorr_sigma_low_mode_inconsistent_sat_scale_values,
            "sigma_low_mode_inconsistent_zref_values": localcorr_sigma_low_mode_inconsistent_zref_values,
            "sigma_low_mode_inconsistent_min_blend_scale_values": localcorr_sigma_low_mode_inconsistent_min_blend_scale_values,
            "sigma_low_mode_inconsistent_sign_blend_scale_values": localcorr_sigma_low_mode_inconsistent_sign_blend_scale_values,
            "logft_max_delta_allowed": float(args.localcorr_logft_max_delta),
            "logft_max_abs_allowed": float(args.localcorr_logft_max_abs),
            "use_qfail_guard": bool(args.localcorr_use_qfail_guard),
            "qfail_max_delta_allowed": int(args.localcorr_qfail_max_delta),
            "top_priority_qfail_max_delta_allowed": int(args.localcorr_top_priority_max_delta),
            "constrained_select_enabled": bool(args.localcorr_constrained_select),
            "n_total_candidates": 0,
        },
    }
    localcorr_residue_reweight: Dict[str, Any] = {
        "status": "not_run",
        "reason": "run_localcorr_sweep_false_or_baseline_mode",
        "selection_policy": {
            "mode": "not_run",
            "enabled": bool(args.localcorr_residue_reweight),
            "max_candidates": int(args.localcorr_residue_reweight_max_candidates),
            "qbeta_overfit_guard_enabled": bool(args.localcorr_qbeta_overfit_guard),
            "qbeta_max_delta_allowed": float(args.localcorr_qbeta_max_delta),
            "overfit_max_delta_allowed": float(args.localcorr_overfit_max_delta),
            "classwise_residue_norm_blend": float(args.localcorr_residue_classwise_norm_blend),
            "class_top_priority_weight_by_key": {
                str(k): float(v) for k, v in localcorr_residue_class_top_weight_by_key.items()
            },
            "class_logft_weight_by_key": {
                str(k): float(v) for k, v in localcorr_residue_class_logft_weight_by_key.items()
            },
            "class_key_threshold_guard_enabled": bool(localcorr_class_key_threshold_guard_enabled),
            "class_logft_count_max_delta_by_key": {
                str(k): int(v) for k, v in localcorr_class_logft_count_max_delta_by_key.items()
            },
            "class_residue_logft_count_max_delta_by_key": {
                str(k): int(v) for k, v in localcorr_class_residue_logft_count_max_delta_by_key.items()
            },
            "class_residue_top_priority_max_delta_by_key": {
                str(k): int(v) for k, v in localcorr_class_residue_top_priority_max_delta_by_key.items()
            },
            "class_residue_overfit_max_delta_by_key": {
                str(k): float(v) for k, v in localcorr_class_residue_overfit_max_delta_by_key.items()
            },
            "logft_rootcause_retune_enabled": bool(args.localcorr_logft_rootcause_retune),
            "logft_rootcause_retune_target_combined_pass_min": int(
                args.localcorr_logft_rootcause_retune_target_combined_pass_min
            ),
            "logft_rootcause_retune_max_logft_count_delta_allowed": int(
                args.localcorr_logft_rootcause_retune_max_logft_count_delta
            ),
            "logft_rootcause_retune_max_residue_logft_count_delta_allowed": int(
                args.localcorr_logft_rootcause_retune_max_residue_logft_count_delta
            ),
            "residue_dual_retune_enabled": bool(args.localcorr_residue_dual_retune),
            "residue_dual_retune_weight_top_priority": float(args.localcorr_residue_dual_retune_top_weight),
            "residue_dual_retune_weight_overfit": float(args.localcorr_residue_dual_retune_overfit_weight),
            "residue_dual_retune_max_top_delta_allowed": int(args.localcorr_residue_dual_retune_max_top_delta),
            "residue_dual_retune_max_overfit_delta_allowed": float(args.localcorr_residue_dual_retune_max_overfit_delta),
        },
        "recommended_candidate": {},
        "all_candidates": [],
        "top_ranked_candidates": [],
    }
    localcorr_residue_reweight_retune: Dict[str, Any] = {
        "status": "not_run",
        "reason": "localcorr_residue_class_retune_grid_false",
        "grid_spec": {
            "target_class_key": (
                str(localcorr_residue_class_retune_target_keys_effective[0])
                if localcorr_residue_class_retune_target_keys_effective
                else str(args.localcorr_residue_class_retune_target_class_key)
            ),
            "target_class_keys": [str(x) for x in localcorr_residue_class_retune_target_keys_effective],
            "target_top_k": int(localcorr_residue_class_retune_target_top_k),
            "logft_count_max_delta_values": [
                int(v) for v in localcorr_residue_class_retune_logft_count_max_delta_values
            ],
            "residue_logft_count_max_delta_values": [
                int(v) for v in localcorr_residue_class_retune_residue_logft_count_max_delta_values
            ],
            "class_top_weight_values": [float(v) for v in localcorr_residue_class_retune_top_weight_values],
            "class_logft_weight_values": [float(v) for v in localcorr_residue_class_retune_logft_weight_values],
        },
        "recommended_candidate": {},
        "recommended_reweight": {},
        "recommended_stability_lock": {},
        "all_candidates": [],
        "applied_best": False,
    }
    localcorr_residue_reweight_retune_strategy: Dict[str, Any] = {
        "status": "not_run",
        "reason": "localcorr_residue_class_retune_strategy_audit_false",
        "strategy_count": 0,
        "all_strategies": [],
        "recommended_strategy": {},
        "recommended_reweight": {},
        "recommended_stability_lock": {},
        "recommended_retune_candidate": {},
        "recommended_retune_grid_spec": {},
        "applied_best": False,
    }
    localcorr_sweep["residue_reweight"] = localcorr_residue_reweight
    localcorr_sweep["residue_reweight_retune"] = localcorr_residue_reweight_retune
    localcorr_sweep["residue_reweight_retune_strategy"] = localcorr_residue_reweight_retune_strategy
    candidate_stability_lock: Dict[str, Any] = {
        "status": "not_run",
        "reason": "run_localcorr_sweep_false_or_baseline_mode",
        "mode": "not_run",
        "stable_candidates_n": 0,
        "evaluated_candidates_n": 0,
        "recommended_candidate": {},
        "all_candidates": [],
        "top_ranked_candidates": [],
        "reject_root_cause": [],
        "selection_policy": {
            "mode": "not_run",
            "enabled": bool(args.run_candidate_stability_lock),
        },
    }
    residue_robustness: Dict[str, Any] = {
        "status": "not_run",
        "reason": "run_residue_robustness_false_or_no_selected_candidate",
        "reference_residue": int(args.holdout_hash_residue),
        "residue_list": [],
        "rows": [],
        "summary": {},
    }
    if bool(args.run_localcorr_sweep) and args.hflavor_mode != "baseline":
        top_n = max(1, int(args.localcorr_top_classes))
        pre_priority = (
            outlier_decomposition_pre_local.get("reduction_priority_order")
            if isinstance(outlier_decomposition_pre_local.get("reduction_priority_order"), list)
            else []
        )
        pre_counts = outlier_decomposition_pre_local.get("counts") if isinstance(outlier_decomposition_pre_local.get("counts"), dict) else {}
        pre_top_qfail = int(pre_priority[0].get("n_qbeta_gt3", 0)) if pre_priority else 0
        target_classes = [str(r.get("group_key", "")) for r in pre_priority[:top_n] if str(r.get("group_key", "")).strip()]
        class_scope = {str(x) for x in target_classes}
        if localcorr_residue_class_retune_target_class_keys:
            filtered_keys = [str(x) for x in localcorr_residue_class_retune_target_class_keys if str(x) in class_scope]
            localcorr_residue_class_retune_target_keys_effective = filtered_keys
        elif int(localcorr_residue_class_retune_target_top_k) > 1:
            localcorr_residue_class_retune_target_keys_effective = [
                str(x) for x in target_classes[: int(localcorr_residue_class_retune_target_top_k)] if str(x).strip()
            ]
        else:
            default_key = str(args.localcorr_residue_class_retune_target_class_key).strip()
            if default_key in class_scope:
                localcorr_residue_class_retune_target_keys_effective = [default_key]
            elif target_classes:
                localcorr_residue_class_retune_target_keys_effective = [str(target_classes[0])]
        if not localcorr_residue_class_retune_target_keys_effective and target_classes:
            localcorr_residue_class_retune_target_keys_effective = [str(target_classes[0])]
        strategy_defs: List[Dict[str, Any]] = []
        strategy_seen: Set[Tuple[str, ...]] = set()

        def _append_strategy(mode: str, keys: List[str], top_k: int, label: str) -> None:
            keys_eff = [str(x) for x in keys if str(x).strip() and str(x) in class_scope]
            if not keys_eff:
                return
            sig = (str(mode),) + tuple(keys_eff)
            if sig in strategy_seen:
                return
            strategy_seen.add(sig)
            strategy_defs.append(
                {
                    "mode": str(mode),
                    "target_class_keys": [str(x) for x in keys_eff],
                    "target_top_k": int(max(0, int(top_k))),
                    "label": str(label),
                }
            )

        _append_strategy(
            "top_k",
            [str(x) for x in target_classes[: int(max(1, localcorr_residue_class_retune_target_top_k))]],
            int(localcorr_residue_class_retune_target_top_k),
            f"top_k={int(localcorr_residue_class_retune_target_top_k)}",
        )
        for top_k_val in localcorr_residue_class_retune_target_top_k_values:
            k_eff = max(1, int(top_k_val))
            _append_strategy(
                "top_k",
                [str(x) for x in target_classes[:k_eff]],
                int(k_eff),
                f"top_k={int(k_eff)}",
            )
        if localcorr_residue_class_retune_target_class_keys:
            _append_strategy(
                "explicit",
                [str(x) for x in localcorr_residue_class_retune_target_class_keys],
                len(localcorr_residue_class_retune_target_class_keys),
                "explicit:target_class_keys",
            )
        for idx_group, group in enumerate(localcorr_residue_class_retune_target_class_keys_groups, start=1):
            _append_strategy(
                "explicit",
                [str(x) for x in group],
                len(group),
                f"explicit_group_{idx_group}",
            )
        _append_strategy(
            "explicit",
            [str(x) for x in localcorr_residue_class_retune_target_keys_effective],
            len(localcorr_residue_class_retune_target_keys_effective),
            "effective_target_keys",
        )
        localcorr_residue_class_retune_strategy_definitions_effective = strategy_defs
        localcorr_sweep = _build_local_correction_sweep(
            rows=mapped_rows,
            target_classes=target_classes,
            logft_sigma_proxy=float(args.logft_sigma_proxy),
            z_gate=float(args.z_gate),
            holdout_hash_modulo=int(args.holdout_hash_modulo),
            holdout_hash_residue=int(args.holdout_hash_residue),
            overfit_gap_gate=float(args.overfit_gap_gate),
            sigma_floor_mev=float(args.sigma_floor_mev),
            closure_gate=closure_gate,
            gain_values=localcorr_gain_values,
            sat_mev_values=localcorr_sat_mev_values,
            blend_values=localcorr_blend_values,
            sign_blend_values=localcorr_sign_blend_values,
            top_class_gain_boost_values=localcorr_top_class_gain_boost_values,
            top_class_sign_blend_scale_values=localcorr_top_class_sign_blend_scale_values,
            sigma_low_mode_inconsistent_sat_scale_values=localcorr_sigma_low_mode_inconsistent_sat_scale_values,
            sigma_low_mode_inconsistent_zref_values=localcorr_sigma_low_mode_inconsistent_zref_values,
            sigma_low_mode_inconsistent_min_blend_scale_values=localcorr_sigma_low_mode_inconsistent_min_blend_scale_values,
            sigma_low_mode_inconsistent_sign_blend_scale_values=localcorr_sigma_low_mode_inconsistent_sign_blend_scale_values,
            pre_local_holdout_p95_abs_z_logft=_to_float(pre_local_hold.get("p95_abs_z_logft_proxy")),
            pre_local_n_qbeta_gt3_holdout=int(pre_counts.get("n_qbeta_gt3_holdout") or 0),
            pre_local_top_priority_q_gt3_holdout=int(pre_top_qfail),
            logft_max_delta_allowed=float(args.localcorr_logft_max_delta),
            logft_max_abs_allowed=float(args.localcorr_logft_max_abs),
            use_qfail_guard=bool(args.localcorr_use_qfail_guard),
            qfail_max_delta_allowed=int(args.localcorr_qfail_max_delta),
            top_priority_qfail_max_delta_allowed=int(args.localcorr_top_priority_max_delta),
        )
        _write_csv_rows(
            out_localcorr_csv,
            localcorr_sweep.get("all_candidates") if isinstance(localcorr_sweep.get("all_candidates"), list) else [],
            fieldnames=[
                "candidate_id",
                "gain",
                "sat_mev",
                "blend",
                "sign_blend",
                "top_class_gain_boost",
                "top_class_sign_blend_scale",
                "top_priority_class_targeted",
                "effective_top_class_gain",
                "effective_top_class_sign_blend",
                "sigma_low_mode_inconsistent_sat_scale",
                "sigma_low_mode_inconsistent_targeted",
                "effective_sigma_low_mode_sat_mev",
                "sigma_low_mode_inconsistent_zref",
                "sigma_low_mode_inconsistent_min_blend_scale",
                "sigma_low_mode_inconsistent_sign_blend_scale",
                "effective_sigma_low_mode_sign_blend",
                "holdout_p95_abs_z_qbeta",
                "holdout_p95_abs_z_logft",
                "delta_holdout_p95_abs_z_logft_vs_pre",
                "logft_constraint_pass",
                "delta_n_qbeta_gt3_holdout_vs_pre",
                "delta_top_priority_q_gt3_vs_pre",
                "qfail_constraint_pass",
                "top_priority_constraint_pass",
                "all_constraints_pass",
                "holdout_max_abs_z_qbeta",
                "train_p95_abs_z_qbeta",
                "overfit_gap_qbeta",
                "overfit_guard_pass",
                "hard_fail_count",
                "watch_fail_count",
                "n_qbeta_gt3_holdout",
                "n_logft_gt3_holdout",
                "top_priority_class_after",
                "top_priority_q_gt3_after",
                "n_target_rows",
                "n_changed_rows",
                "mapping_delta_abs_median_mev",
                "mapping_delta_abs_p95_mev",
                "is_pareto_front",
                "norm_holdout_p95_abs_z_qbeta",
                "norm_holdout_p95_abs_z_logft",
                "norm_overfit_gap_qbeta",
                "norm_n_qbeta_gt3_holdout",
                "score_weighted",
                "rank_weighted",
                "rank_constrained",
            ],
        )
        if bool(args.localcorr_constrained_select):
            recommended_local = (
                localcorr_sweep.get("recommended_candidate")
                if isinstance(localcorr_sweep.get("recommended_candidate"), dict)
                else {}
            )
            policy = localcorr_sweep.get("selection_policy") if isinstance(localcorr_sweep.get("selection_policy"), dict) else {}
            recommended_source = str(policy.get("mode", "constrained"))
        else:
            recommended_local = (
                localcorr_sweep.get("recommended_candidate_unconstrained")
                if isinstance(localcorr_sweep.get("recommended_candidate_unconstrained"), dict)
                else {}
            )
            if not recommended_local and isinstance(localcorr_sweep.get("recommended_candidate"), dict):
                recommended_local = localcorr_sweep.get("recommended_candidate")  # type: ignore[assignment]
            recommended_source = "unconstrained"

        if bool(args.localcorr_residue_reweight):
            all_lc_rows = localcorr_sweep.get("all_candidates") if isinstance(localcorr_sweep.get("all_candidates"), list) else []
            if bool(args.localcorr_constrained_select):
                rows_for_reweight = [r for r in all_lc_rows if isinstance(r, dict) and bool(r.get("all_constraints_pass"))]
                if not rows_for_reweight:
                    rows_for_reweight = [r for r in all_lc_rows if isinstance(r, dict) and bool(r.get("logft_constraint_pass"))]
                target_rows_for_reweight = max(8, int(args.localcorr_residue_reweight_max_candidates))
                if len(rows_for_reweight) < target_rows_for_reweight:
                    supplement_pool = [r for r in all_lc_rows if isinstance(r, dict)]
                    supplement_pool = sorted(
                        supplement_pool,
                        key=lambda r: (
                            _to_float(r.get("rank_constrained")) if math.isfinite(_to_float(r.get("rank_constrained"))) else float("inf"),
                            _to_float(r.get("rank_weighted")) if math.isfinite(_to_float(r.get("rank_weighted"))) else float("inf"),
                            _to_float(r.get("holdout_p95_abs_z_qbeta")) if math.isfinite(_to_float(r.get("holdout_p95_abs_z_qbeta"))) else float("inf"),
                        ),
                    )
                    selected_ids = {str(r.get("candidate_id", "")) for r in rows_for_reweight}
                    for cand in supplement_pool:
                        cid = str(cand.get("candidate_id", ""))
                        if cid in selected_ids:
                            continue
                        rows_for_reweight.append(cand)
                        selected_ids.add(cid)
                        if len(rows_for_reweight) >= target_rows_for_reweight:
                            break
            else:
                rows_for_reweight = [r for r in all_lc_rows if isinstance(r, dict)]
            reweight_common_kwargs = {
                "rows_mapped": mapped_rows,
                "target_classes": target_classes,
                "candidate_rows": rows_for_reweight,
                "closure_gate": closure_gate,
                "logft_sigma_proxy": float(args.logft_sigma_proxy),
                "z_gate": float(args.z_gate),
                "holdout_hash_modulo": int(args.holdout_hash_modulo),
                "holdout_hash_residues": [int(r) for r in residue_values],
                "overfit_gap_gate": float(args.overfit_gap_gate),
                "sigma_floor_mev": float(args.sigma_floor_mev),
                "reference_residue": int(args.holdout_hash_residue),
                "max_candidates": int(args.localcorr_residue_reweight_max_candidates),
                "pre_local_holdout_p95_abs_z_qbeta": _to_float(pre_local_hold.get("p95_abs_z_qbeta")),
                "pre_local_overfit_gap_qbeta": _to_float(pre_local_overfit.get("p95_gap_qbeta")),
                "pre_local_n_logft_gt3_holdout": int(pre_counts.get("n_logft_gt3_holdout") or 0),
                "use_qbeta_overfit_guard": bool(args.localcorr_qbeta_overfit_guard),
                "qbeta_max_delta_allowed": float(args.localcorr_qbeta_max_delta),
                "overfit_max_delta_allowed": float(args.localcorr_overfit_max_delta),
                "use_logft_rootcause_guard": bool(args.localcorr_logft_rootcause_guard),
                "require_residue_logft_nonworsening": bool(args.localcorr_require_residue_logft_nonworsening),
                "use_logft_rootcause_refreeze": bool(args.localcorr_logft_rootcause_refreeze),
                "use_logft_rootcause_retune": bool(args.localcorr_logft_rootcause_retune),
                "logft_rootcause_retune_target_combined_pass_min": int(
                    args.localcorr_logft_rootcause_retune_target_combined_pass_min
                ),
                "logft_rootcause_retune_max_logft_count_delta_allowed": int(
                    args.localcorr_logft_rootcause_retune_max_logft_count_delta
                ),
                "logft_rootcause_retune_max_residue_logft_count_delta_allowed": int(
                    args.localcorr_logft_rootcause_retune_max_residue_logft_count_delta
                ),
                "use_residue_dual_guard": bool(args.localcorr_residue_dual_guard),
                "residue_top_priority_max_delta_allowed": int(args.localcorr_residue_top_priority_max_delta),
                "residue_overfit_max_delta_allowed": float(args.localcorr_residue_overfit_max_delta),
                "use_residue_dual_refreeze": bool(args.localcorr_residue_dual_refreeze),
                "use_residue_dual_retune": bool(args.localcorr_residue_dual_retune),
                "residue_dual_retune_top_weight": float(args.localcorr_residue_dual_retune_top_weight),
                "residue_dual_retune_overfit_weight": float(args.localcorr_residue_dual_retune_overfit_weight),
                "residue_dual_retune_max_top_delta_allowed": int(args.localcorr_residue_dual_retune_max_top_delta),
                "residue_dual_retune_max_overfit_delta_allowed": float(args.localcorr_residue_dual_retune_max_overfit_delta),
                "classwise_residue_norm_blend": float(args.localcorr_residue_classwise_norm_blend),
                "use_class_key_threshold_guard": bool(localcorr_class_key_threshold_guard_enabled),
                "class_logft_count_max_delta_by_key": localcorr_class_logft_count_max_delta_by_key,
                "class_residue_logft_count_max_delta_by_key": localcorr_class_residue_logft_count_max_delta_by_key,
                "class_residue_top_priority_max_delta_by_key": localcorr_class_residue_top_priority_max_delta_by_key,
                "class_residue_overfit_max_delta_by_key": localcorr_class_residue_overfit_max_delta_by_key,
            }
            localcorr_residue_reweight = _build_localcorr_residue_reweight(
                **reweight_common_kwargs,
                logft_count_max_delta_allowed=int(args.localcorr_logft_count_max_delta),
                residue_logft_count_max_delta_allowed=int(args.localcorr_residue_logft_count_max_delta),
                class_top_priority_weight_by_key=localcorr_residue_class_top_weight_by_key,
                class_logft_weight_by_key=localcorr_residue_class_logft_weight_by_key,
            )
            qbeta_count_recovery_policy_grid = {
                "enabled": bool(args.stability_qbeta_count_recovery),
                "qbeta_count_max_delta": int(args.stability_qbeta_count_max_delta),
                "logft_p95_max_delta": float(args.stability_qbeta_count_recovery_logft_p95_max_delta),
                "logft_count_max_delta": int(args.stability_qbeta_count_recovery_logft_count_max_delta),
                "residue_logft_count_max_delta": int(
                    args.stability_qbeta_count_recovery_residue_logft_count_max_delta
                ),
                "residue_overfit_max_delta": float(
                    args.stability_qbeta_count_recovery_residue_overfit_max_delta
                ),
            }
            if bool(args.localcorr_residue_class_retune_grid):
                localcorr_residue_reweight_retune = _build_localcorr_residue_reweight_retune_grid(
                    **reweight_common_kwargs,
                    pre_local_holdout_p95_abs_z_logft=_to_float(pre_local_hold.get("p95_abs_z_logft_proxy")),
                    pre_local_n_qbeta_gt3_holdout=int(pre_counts.get("n_qbeta_gt3_holdout") or 0),
                    base_class_top_priority_weight_by_key=localcorr_residue_class_top_weight_by_key,
                    base_class_logft_weight_by_key=localcorr_residue_class_logft_weight_by_key,
                    class_weight_target_keys=[str(x) for x in localcorr_residue_class_retune_target_keys_effective],
                    logft_count_max_delta_values=localcorr_residue_class_retune_logft_count_max_delta_values,
                    residue_logft_count_max_delta_values=localcorr_residue_class_retune_residue_logft_count_max_delta_values,
                    class_top_weight_values=localcorr_residue_class_retune_top_weight_values,
                    class_logft_weight_values=localcorr_residue_class_retune_logft_weight_values,
                    qbeta_count_recovery_policy=qbeta_count_recovery_policy_grid,
                )
                if (
                    bool(args.localcorr_residue_class_retune_apply_best)
                    and isinstance(localcorr_residue_reweight_retune, dict)
                    and str(localcorr_residue_reweight_retune.get("status")) == "completed"
                    and isinstance(localcorr_residue_reweight_retune.get("recommended_reweight"), dict)
                    and str(
                        (localcorr_residue_reweight_retune.get("recommended_reweight") or {}).get("status", "")
                    )
                    == "completed"
                ):
                    localcorr_residue_reweight = dict(localcorr_residue_reweight_retune.get("recommended_reweight") or {})
                    localcorr_residue_reweight_retune["applied_best"] = True
                    rw_policy_retuned = (
                        localcorr_residue_reweight.get("selection_policy")
                        if isinstance(localcorr_residue_reweight.get("selection_policy"), dict)
                        else {}
                    )
                    rw_policy_retuned["class_retune_grid_enabled"] = True
                    rw_policy_retuned["class_retune_grid_applied"] = True
                    rw_policy_retuned["class_retune_grid_target_class_key"] = (
                        str(localcorr_residue_class_retune_target_keys_effective[0])
                        if localcorr_residue_class_retune_target_keys_effective
                        else str(args.localcorr_residue_class_retune_target_class_key)
                    )
                    rw_policy_retuned["class_retune_grid_target_class_keys"] = [
                        str(x) for x in localcorr_residue_class_retune_target_keys_effective
                    ]
                    rw_policy_retuned["class_retune_grid_target_top_k"] = int(
                        localcorr_residue_class_retune_target_top_k
                    )
                    rw_policy_retuned["class_retune_grid_recommended"] = (
                        localcorr_residue_reweight_retune.get("recommended_candidate")
                        if isinstance(localcorr_residue_reweight_retune.get("recommended_candidate"), dict)
                        else {}
                    )
                    localcorr_residue_reweight["selection_policy"] = rw_policy_retuned
                elif isinstance(localcorr_residue_reweight_retune, dict):
                    localcorr_residue_reweight_retune["applied_best"] = False
            else:
                localcorr_residue_reweight_retune = {
                    "status": "not_run",
                    "reason": "localcorr_residue_class_retune_grid_false",
                    "grid_spec": {
                        "target_class_key": (
                            str(localcorr_residue_class_retune_target_keys_effective[0])
                            if localcorr_residue_class_retune_target_keys_effective
                            else str(args.localcorr_residue_class_retune_target_class_key)
                        ),
                        "target_class_keys": [str(x) for x in localcorr_residue_class_retune_target_keys_effective],
                        "target_top_k": int(localcorr_residue_class_retune_target_top_k),
                        "logft_count_max_delta_values": [
                            int(v) for v in localcorr_residue_class_retune_logft_count_max_delta_values
                        ],
                        "residue_logft_count_max_delta_values": [
                            int(v) for v in localcorr_residue_class_retune_residue_logft_count_max_delta_values
                        ],
                        "class_top_weight_values": [float(v) for v in localcorr_residue_class_retune_top_weight_values],
                        "class_logft_weight_values": [
                            float(v) for v in localcorr_residue_class_retune_logft_weight_values
                        ],
                    },
                    "recommended_candidate": {},
                    "recommended_reweight": {},
                    "recommended_stability_lock": {},
                    "all_candidates": [],
                    "applied_best": False,
                }
            if bool(args.localcorr_residue_class_retune_strategy_audit):
                localcorr_residue_reweight_retune_strategy = _build_localcorr_residue_reweight_retune_strategy_audit(
                    **reweight_common_kwargs,
                    pre_local_holdout_p95_abs_z_logft=_to_float(pre_local_hold.get("p95_abs_z_logft_proxy")),
                    pre_local_n_qbeta_gt3_holdout=int(pre_counts.get("n_qbeta_gt3_holdout") or 0),
                    base_class_top_priority_weight_by_key=localcorr_residue_class_top_weight_by_key,
                    base_class_logft_weight_by_key=localcorr_residue_class_logft_weight_by_key,
                    logft_count_max_delta_values=localcorr_residue_class_retune_logft_count_max_delta_values,
                    residue_logft_count_max_delta_values=localcorr_residue_class_retune_residue_logft_count_max_delta_values,
                    class_top_weight_values=localcorr_residue_class_retune_top_weight_values,
                    class_logft_weight_values=localcorr_residue_class_retune_logft_weight_values,
                    qbeta_count_recovery_policy=qbeta_count_recovery_policy_grid,
                    strategy_definitions=localcorr_residue_class_retune_strategy_definitions_effective,
                )
                if (
                    bool(args.localcorr_residue_class_retune_apply_best)
                    and isinstance(localcorr_residue_reweight_retune_strategy, dict)
                    and str(localcorr_residue_reweight_retune_strategy.get("status")) == "completed"
                    and isinstance(localcorr_residue_reweight_retune_strategy.get("recommended_reweight"), dict)
                    and str(
                        (localcorr_residue_reweight_retune_strategy.get("recommended_reweight") or {}).get("status", "")
                    )
                    == "completed"
                ):
                    localcorr_residue_reweight = dict(
                        localcorr_residue_reweight_retune_strategy.get("recommended_reweight") or {}
                    )
                    localcorr_residue_reweight_retune_strategy["applied_best"] = True
                    rw_policy_retuned = (
                        localcorr_residue_reweight.get("selection_policy")
                        if isinstance(localcorr_residue_reweight.get("selection_policy"), dict)
                        else {}
                    )
                    rw_policy_retuned["class_retune_strategy_audit_enabled"] = True
                    rw_policy_retuned["class_retune_strategy_audit_applied"] = True
                    rw_policy_retuned["class_retune_strategy_recommended"] = (
                        localcorr_residue_reweight_retune_strategy.get("recommended_strategy")
                        if isinstance(localcorr_residue_reweight_retune_strategy.get("recommended_strategy"), dict)
                        else {}
                    )
                    rw_policy_retuned["class_retune_strategy_count"] = int(
                        localcorr_residue_reweight_retune_strategy.get("strategy_count") or 0
                    )
                    localcorr_residue_reweight["selection_policy"] = rw_policy_retuned
                elif isinstance(localcorr_residue_reweight_retune_strategy, dict):
                    localcorr_residue_reweight_retune_strategy["applied_best"] = False
            else:
                localcorr_residue_reweight_retune_strategy = {
                    "status": "not_run",
                    "reason": "localcorr_residue_class_retune_strategy_audit_false",
                    "strategy_count": 0,
                    "all_strategies": [],
                    "recommended_strategy": {},
                    "recommended_reweight": {},
                    "recommended_stability_lock": {},
                    "recommended_retune_candidate": {},
                    "recommended_retune_grid_spec": {},
                    "applied_best": False,
                }
            _write_csv_rows(
                out_localcorr_reweight_retune_csv,
                localcorr_residue_reweight_retune.get("all_candidates")
                if isinstance(localcorr_residue_reweight_retune.get("all_candidates"), list)
                else [],
                fieldnames=[
                    "rank_retune",
                    "target_class_key",
                    "target_class_keys",
                    "n_target_classes",
                    "logft_count_max_delta_allowed",
                    "residue_logft_count_max_delta_allowed",
                    "class_top_weight",
                    "class_logft_weight",
                    "n_candidates_evaluated",
                    "n_candidates_guard_pass",
                    "n_candidates_root_guard_pass",
                    "n_candidates_combined_guard_pass",
                    "stable_candidates_n",
                    "stability_mode",
                    "watch3_all_true_candidates_n",
                    "watch3_all_true_candidates_ratio",
                    "best_candidate_id",
                    "best_residue_reweight_score",
                    "best_holdout_p95_abs_z_qbeta",
                    "best_holdout_p95_abs_z_logft",
                    "best_overfit_gap_qbeta",
                    "best_stability_hard_metric_score",
                    "best_stability_violation_score",
                ],
            )
            _write_csv_rows(
                out_localcorr_reweight_retune_strategy_csv,
                localcorr_residue_reweight_retune_strategy.get("all_strategies")
                if isinstance(localcorr_residue_reweight_retune_strategy.get("all_strategies"), list)
                else [],
                fieldnames=[
                    "rank_strategy",
                    "strategy_id",
                    "strategy_mode",
                    "strategy_label",
                    "target_top_k",
                    "target_class_keys",
                    "n_target_classes",
                    "n_total_combinations",
                    "n_candidates_evaluated",
                    "n_candidates_guard_pass",
                    "n_candidates_root_guard_pass",
                    "n_candidates_combined_guard_pass",
                    "stable_candidates_n",
                    "stability_mode",
                    "watch3_all_true_candidates_n",
                    "watch3_all_true_candidates_ratio",
                    "best_candidate_id",
                    "best_residue_reweight_score",
                    "best_holdout_p95_abs_z_qbeta",
                    "best_holdout_p95_abs_z_logft",
                    "best_overfit_gap_qbeta",
                    "best_stability_hard_metric_score",
                    "best_stability_violation_score",
                ],
            )
            _write_csv_rows(
                out_localcorr_reweight_csv,
                localcorr_residue_reweight.get("all_candidates")
                if isinstance(localcorr_residue_reweight.get("all_candidates"), list)
                else [],
                fieldnames=[
                    "candidate_id",
                    "gain",
                    "sat_mev",
                    "blend",
                    "sign_blend",
                    "top_class_gain_boost",
                    "top_class_sign_blend_scale",
                    "top_priority_class_targeted",
                    "effective_top_class_gain",
                    "effective_top_class_sign_blend",
                    "sigma_low_mode_inconsistent_sat_scale",
                    "sigma_low_mode_inconsistent_targeted",
                    "effective_sigma_low_mode_sat_mev",
                    "sigma_low_mode_inconsistent_zref",
                    "sigma_low_mode_inconsistent_min_blend_scale",
                    "sigma_low_mode_inconsistent_sign_blend_scale",
                    "effective_sigma_low_mode_sign_blend",
                    "all_constraints_pass",
                    "logft_constraint_pass",
                    "holdout_p95_abs_z_qbeta",
                    "holdout_p95_abs_z_logft",
                    "overfit_gap_qbeta",
                    "n_qbeta_gt3_holdout",
                    "top_priority_q_gt3_after",
                    "residue_n",
                    "residue_reference",
                    "residue_all_qfail_nonregression",
                    "residue_all_logft_count_nonregression",
                    "residue_all_top_priority_nonregression",
                    "residue_all_logft_nonworsening",
                    "residue_all_overfit_nonworsening",
                    "residue_max_pos_delta_n_qbeta_gt3",
                    "residue_max_pos_delta_n_logft_gt3",
                    "residue_max_pos_delta_top_priority_q_gt3",
                    "residue_max_pos_delta_overfit_gap_qbeta",
                    "residue_holdout_p95_abs_z_qbeta_spread",
                    "residue_holdout_p95_abs_z_logft_spread",
                    "residue_overfit_gap_qbeta_spread",
                    "delta_holdout_p95_abs_z_qbeta_vs_pre",
                    "delta_overfit_gap_qbeta_vs_pre",
                    "delta_n_logft_gt3_holdout_vs_pre",
                    "qbeta_nonworsening_pass",
                    "overfit_nonworsening_pass",
                    "qbeta_overfit_guard_pass",
                    "prelocal_logft_count_guard_pass",
                    "logft_count_max_delta_effective",
                    "residue_logft_count_guard_pass",
                    "residue_logft_count_max_delta_effective",
                    "residue_logft_nonworsening_guard_pass",
                    "class_key_threshold_guard_applied",
                    "logft_rootcause_guard_pass",
                    "residue_top_priority_guard_pass_strict",
                    "residue_top_priority_max_delta_effective_strict",
                    "residue_overfit_guard_pass_strict",
                    "residue_overfit_max_delta_effective_strict",
                    "residue_dual_guard_pass_strict",
                    "prelocal_logft_count_refrozen_pass",
                    "refrozen_logft_count_max_delta_effective",
                    "residue_logft_count_refrozen_pass",
                    "refrozen_residue_logft_count_max_delta_effective",
                    "residue_logft_nonworsening_refrozen_pass",
                    "logft_rootcause_refrozen_guard_pass",
                    "residue_top_priority_guard_pass",
                    "residue_top_priority_max_delta_effective",
                    "residue_overfit_guard_pass",
                    "residue_overfit_max_delta_effective",
                    "residue_dual_guard_pass",
                    "residue_dual_retune_cost",
                    "norm_residue_max_pos_delta_top_priority_q_gt3",
                    "norm_residue_max_pos_delta_n_logft_gt3",
                    "norm_residue_max_pos_delta_n_qbeta_gt3",
                    "norm_residue_holdout_p95_abs_z_logft_spread",
                    "norm_residue_holdout_p95_abs_z_qbeta_spread",
                    "norm_delta_holdout_p95_abs_z_qbeta_vs_pre",
                    "norm_delta_overfit_gap_qbeta_vs_pre",
                    "residue_weight_multiplier_top_class",
                    "residue_weight_multiplier_logft_class",
                    "residue_weight_top_priority_effective",
                    "residue_weight_logft_effective",
                    "residue_weight_sum_effective",
                    "residue_class_key",
                    "residue_class_count",
                    "residue_reweight_score_global",
                    "residue_reweight_score_class",
                    "residue_reweight_score",
                    "rank_residue_reweight",
                ],
            )
            rec_rw = (
                localcorr_residue_reweight.get("recommended_candidate")
                if isinstance(localcorr_residue_reweight.get("recommended_candidate"), dict)
                else {}
            )
            rw_policy = (
                localcorr_residue_reweight.get("selection_policy")
                if isinstance(localcorr_residue_reweight.get("selection_policy"), dict)
                else {}
            )
            if rec_rw:
                recommended_local = rec_rw
                recommended_source = str(rw_policy.get("mode", "logft_qfail_residue_reweighted"))
                localcorr_sweep["recommended_candidate"] = rec_rw
                if isinstance(localcorr_residue_reweight.get("top_ranked_candidates"), list):
                    localcorr_sweep["top_ranked_candidates"] = localcorr_residue_reweight.get("top_ranked_candidates")
            policy = localcorr_sweep.get("selection_policy") if isinstance(localcorr_sweep.get("selection_policy"), dict) else {}
            policy["residue_reweight_enabled"] = True
            policy["residue_reweight_status"] = str(localcorr_residue_reweight.get("status", "not_run"))
            policy["residue_reweight_mode"] = str(rw_policy.get("mode", "not_run"))
            policy["residue_reweight_candidates_evaluated"] = int(rw_policy.get("n_candidates_evaluated") or 0)
            policy["residue_reweight_guard_enabled"] = bool(rw_policy.get("qbeta_overfit_guard_enabled"))
            policy["residue_reweight_guard_pass"] = int(rw_policy.get("n_candidates_guard_pass") or 0)
            policy["residue_reweight_qbeta_max_delta_allowed"] = _to_float(rw_policy.get("qbeta_max_delta_allowed"))
            policy["residue_reweight_overfit_max_delta_allowed"] = _to_float(rw_policy.get("overfit_max_delta_allowed"))
            policy["residue_reweight_classwise_norm_blend"] = _to_float(rw_policy.get("classwise_residue_norm_blend"))
            policy["residue_reweight_class_top_priority_weight_by_key"] = (
                rw_policy.get("class_top_priority_weight_by_key")
                if isinstance(rw_policy.get("class_top_priority_weight_by_key"), dict)
                else {}
            )
            policy["residue_reweight_class_logft_weight_by_key"] = (
                rw_policy.get("class_logft_weight_by_key")
                if isinstance(rw_policy.get("class_logft_weight_by_key"), dict)
                else {}
            )
            policy["residue_reweight_logft_rootcause_guard_enabled"] = bool(
                rw_policy.get("logft_rootcause_guard_enabled")
            )
            policy["residue_reweight_logft_rootcause_guard_pass"] = int(rw_policy.get("n_candidates_root_guard_pass") or 0)
            policy["residue_reweight_logft_rootcause_combined_pass"] = int(
                rw_policy.get("n_candidates_combined_guard_pass") or 0
            )
            policy["residue_reweight_residue_dual_guard_enabled"] = bool(
                rw_policy.get("residue_dual_guard_enabled")
            )
            policy["residue_reweight_residue_dual_guard_pass"] = int(
                rw_policy.get("n_candidates_residue_dual_guard_pass") or 0
            )
            policy["residue_reweight_residue_top_priority_max_delta_allowed"] = int(
                rw_policy.get("residue_top_priority_max_delta_allowed") or 0
            )
            policy["residue_reweight_residue_overfit_max_delta_allowed"] = _to_float(
                rw_policy.get("residue_overfit_max_delta_allowed")
            )
            policy["residue_reweight_refrozen_residue_top_priority_max_delta_allowed"] = int(
                rw_policy.get("refrozen_residue_top_priority_max_delta_allowed") or 0
            )
            policy["residue_reweight_refrozen_residue_overfit_max_delta_allowed"] = _to_float(
                rw_policy.get("refrozen_residue_overfit_max_delta_allowed")
            )
            policy["residue_reweight_residue_dual_refreeze_enabled"] = bool(
                rw_policy.get("residue_dual_refreeze_enabled")
            )
            policy["residue_reweight_residue_dual_refreeze_applied"] = bool(
                rw_policy.get("residue_dual_refreeze_applied")
            )
            policy["residue_reweight_residue_dual_refreeze_source"] = str(
                rw_policy.get("residue_dual_refreeze_source", "n/a")
            )
            policy["residue_reweight_residue_dual_retune_enabled"] = bool(
                rw_policy.get("residue_dual_retune_enabled")
            )
            policy["residue_reweight_residue_dual_retune_applied"] = bool(
                rw_policy.get("residue_dual_retune_applied")
            )
            policy["residue_reweight_residue_dual_retune_source"] = str(
                rw_policy.get("residue_dual_retune_source", "n/a")
            )
            policy["residue_reweight_residue_dual_retune_anchor_candidate"] = str(
                rw_policy.get("residue_dual_retune_anchor_candidate", "n/a")
            )
            policy["residue_reweight_residue_dual_retune_weight_top_priority"] = _to_float(
                rw_policy.get("residue_dual_retune_weight_top_priority")
            )
            policy["residue_reweight_residue_dual_retune_weight_overfit"] = _to_float(
                rw_policy.get("residue_dual_retune_weight_overfit")
            )
            policy["residue_reweight_residue_dual_retune_max_top_delta_allowed"] = int(
                rw_policy.get("residue_dual_retune_max_top_delta_allowed") or 0
            )
            policy["residue_reweight_residue_dual_retune_max_overfit_delta_allowed"] = _to_float(
                rw_policy.get("residue_dual_retune_max_overfit_delta_allowed")
            )
            policy["residue_reweight_logft_rootcause_retune_enabled"] = bool(
                rw_policy.get("logft_rootcause_retune_enabled")
            )
            policy["residue_reweight_logft_rootcause_retune_applied"] = bool(
                rw_policy.get("logft_rootcause_retune_applied")
            )
            policy["residue_reweight_logft_rootcause_retune_source"] = str(
                rw_policy.get("logft_rootcause_retune_source", "n/a")
            )
            policy["residue_reweight_logft_rootcause_retune_anchor_candidate"] = str(
                rw_policy.get("logft_rootcause_retune_anchor_candidate", "n/a")
            )
            policy["residue_reweight_logft_rootcause_retune_target_combined_pass_min"] = int(
                rw_policy.get("logft_rootcause_retune_target_combined_pass_min") or 0
            )
            policy["residue_reweight_logft_rootcause_retune_max_logft_count_delta_allowed"] = int(
                rw_policy.get("logft_rootcause_retune_max_logft_count_delta_allowed") or 0
            )
            policy["residue_reweight_logft_rootcause_retune_max_residue_logft_count_delta_allowed"] = int(
                rw_policy.get("logft_rootcause_retune_max_residue_logft_count_delta_allowed") or 0
            )
            policy["residue_reweight_logft_rootcause_refreeze_enabled"] = bool(
                rw_policy.get("logft_rootcause_refreeze_enabled")
            )
            policy["residue_reweight_logft_rootcause_refreeze_applied"] = bool(
                rw_policy.get("logft_rootcause_refreeze_applied")
            )
            policy["residue_reweight_logft_rootcause_refreeze_source"] = str(
                rw_policy.get("logft_rootcause_refreeze_source", "n/a")
            )
            policy["residue_reweight_refrozen_logft_count_max_delta_allowed"] = int(
                rw_policy.get("refrozen_logft_count_max_delta_allowed") or 0
            )
            policy["residue_reweight_refrozen_residue_logft_count_max_delta_allowed"] = int(
                rw_policy.get("refrozen_residue_logft_count_max_delta_allowed") or 0
            )
            policy["residue_reweight_refrozen_require_residue_logft_nonworsening"] = bool(
                rw_policy.get("refrozen_require_residue_logft_nonworsening")
            )
            policy["residue_reweight_class_key_threshold_guard_enabled"] = bool(
                rw_policy.get("class_key_threshold_guard_enabled")
            )
            policy["residue_reweight_n_candidates_class_key_guard_applied"] = int(
                rw_policy.get("n_candidates_class_key_guard_applied") or 0
            )
            policy["residue_reweight_class_logft_count_max_delta_by_key"] = (
                rw_policy.get("class_logft_count_max_delta_by_key")
                if isinstance(rw_policy.get("class_logft_count_max_delta_by_key"), dict)
                else {}
            )
            policy["residue_reweight_class_residue_logft_count_max_delta_by_key"] = (
                rw_policy.get("class_residue_logft_count_max_delta_by_key")
                if isinstance(rw_policy.get("class_residue_logft_count_max_delta_by_key"), dict)
                else {}
            )
            policy["residue_reweight_class_residue_top_priority_max_delta_by_key"] = (
                rw_policy.get("class_residue_top_priority_max_delta_by_key")
                if isinstance(rw_policy.get("class_residue_top_priority_max_delta_by_key"), dict)
                else {}
            )
            policy["residue_reweight_class_residue_overfit_max_delta_by_key"] = (
                rw_policy.get("class_residue_overfit_max_delta_by_key")
                if isinstance(rw_policy.get("class_residue_overfit_max_delta_by_key"), dict)
                else {}
            )
            policy["residue_reweight_logft_count_max_delta_allowed"] = int(
                rw_policy.get("logft_count_max_delta_allowed") or 0
            )
            policy["residue_reweight_residue_logft_count_max_delta_allowed"] = int(
                rw_policy.get("residue_logft_count_max_delta_allowed") or 0
            )
            policy["residue_reweight_require_residue_logft_nonworsening"] = bool(
                rw_policy.get("require_residue_logft_nonworsening")
            )
            policy["residue_reweight_class_retune_grid_enabled"] = bool(args.localcorr_residue_class_retune_grid)
            policy["residue_reweight_class_retune_grid_status"] = str(
                localcorr_residue_reweight_retune.get("status", "not_run")
            )
            policy["residue_reweight_class_retune_grid_applied"] = bool(
                localcorr_residue_reweight_retune.get("applied_best")
            )
            policy["residue_reweight_class_retune_grid_target_class_key"] = (
                str(localcorr_residue_class_retune_target_keys_effective[0])
                if localcorr_residue_class_retune_target_keys_effective
                else str(args.localcorr_residue_class_retune_target_class_key)
            )
            policy["residue_reweight_class_retune_grid_target_class_keys"] = [
                str(x) for x in localcorr_residue_class_retune_target_keys_effective
            ]
            policy["residue_reweight_class_retune_grid_target_top_k"] = int(
                localcorr_residue_class_retune_target_top_k
            )
            policy["residue_reweight_class_retune_grid_recommended"] = (
                localcorr_residue_reweight_retune.get("recommended_candidate")
                if isinstance(localcorr_residue_reweight_retune.get("recommended_candidate"), dict)
                else {}
            )
            policy["residue_reweight_class_retune_strategy_audit_enabled"] = bool(
                args.localcorr_residue_class_retune_strategy_audit
            )
            policy["residue_reweight_class_retune_strategy_audit_status"] = str(
                localcorr_residue_reweight_retune_strategy.get("status", "not_run")
            )
            policy["residue_reweight_class_retune_strategy_audit_applied"] = bool(
                localcorr_residue_reweight_retune_strategy.get("applied_best")
            )
            policy["residue_reweight_class_retune_strategy_count"] = int(
                localcorr_residue_reweight_retune_strategy.get("strategy_count") or 0
            )
            policy["residue_reweight_class_retune_strategy_recommended"] = (
                localcorr_residue_reweight_retune_strategy.get("recommended_strategy")
                if isinstance(localcorr_residue_reweight_retune_strategy.get("recommended_strategy"), dict)
                else {}
            )
            if rec_rw:
                policy["mode"] = str(rw_policy.get("mode", policy.get("mode", "logft_qfail_residue_reweighted")))
            localcorr_sweep["selection_policy"] = policy
        localcorr_sweep["residue_reweight"] = localcorr_residue_reweight
        localcorr_sweep["residue_reweight_retune"] = localcorr_residue_reweight_retune
        localcorr_sweep["residue_reweight_retune_strategy"] = localcorr_residue_reweight_retune_strategy
        if bool(args.run_candidate_stability_lock):
            rw_policy_for_lock = (
                localcorr_residue_reweight.get("selection_policy")
                if isinstance(localcorr_residue_reweight.get("selection_policy"), dict)
                else {}
            )
            lock_refreeze_policy = _build_lock_refreeze_policy_from_reweight_policy(rw_policy_for_lock)
            lock_qbeta_recovery_policy = {
                "enabled": bool(args.stability_qbeta_count_recovery),
                "qbeta_count_max_delta": int(args.stability_qbeta_count_max_delta),
                "logft_p95_max_delta": float(args.stability_qbeta_count_recovery_logft_p95_max_delta),
                "logft_count_max_delta": int(args.stability_qbeta_count_recovery_logft_count_max_delta),
                "residue_logft_count_max_delta": int(
                    args.stability_qbeta_count_recovery_residue_logft_count_max_delta
                ),
                "residue_overfit_max_delta": float(
                    args.stability_qbeta_count_recovery_residue_overfit_max_delta
                ),
            }
            candidate_stability_lock = _build_candidate_stability_lock(
                reweight_pack=localcorr_residue_reweight,
                pre_local_holdout_p95_qbeta=_to_float(pre_local_hold.get("p95_abs_z_qbeta")),
                pre_local_holdout_p95_logft=_to_float(pre_local_hold.get("p95_abs_z_logft_proxy")),
                pre_local_overfit_gap_qbeta=_to_float(pre_local_overfit.get("p95_gap_qbeta")),
                pre_local_n_qbeta_gt3_holdout=int(pre_counts.get("n_qbeta_gt3_holdout") or 0),
                pre_local_n_logft_gt3_holdout=int(pre_counts.get("n_logft_gt3_holdout") or 0),
                logft_rootcause_refreeze_policy=lock_refreeze_policy,
                qbeta_count_recovery_policy=lock_qbeta_recovery_policy,
            )
            _write_csv_rows(
                out_stability_lock_csv,
                candidate_stability_lock.get("all_candidates")
                if isinstance(candidate_stability_lock.get("all_candidates"), list)
                else [],
                fieldnames=[
                    "candidate_id",
                    "gain",
                    "sat_mev",
                    "blend",
                    "sign_blend",
                    "top_class_gain_boost",
                    "top_priority_class_targeted",
                    "effective_top_class_gain",
                    "sigma_low_mode_inconsistent_sat_scale",
                    "sigma_low_mode_inconsistent_targeted",
                    "effective_sigma_low_mode_sat_mev",
                    "sigma_low_mode_inconsistent_zref",
                    "sigma_low_mode_inconsistent_min_blend_scale",
                    "sigma_low_mode_inconsistent_sign_blend_scale",
                    "effective_sigma_low_mode_sign_blend",
                    "all_constraints_pass",
                    "logft_constraint_pass",
                    "holdout_p95_abs_z_qbeta",
                    "holdout_p95_abs_z_logft",
                    "overfit_gap_qbeta",
                    "n_qbeta_gt3_holdout",
                    "n_logft_gt3_holdout",
                    "residue_max_pos_delta_n_qbeta_gt3",
                    "residue_max_pos_delta_n_logft_gt3",
                    "residue_max_pos_delta_top_priority_q_gt3",
                    "residue_max_pos_delta_overfit_gap_qbeta",
                    "delta_holdout_p95_abs_z_qbeta_vs_pre",
                    "delta_holdout_p95_abs_z_logft_vs_pre",
                    "delta_overfit_gap_qbeta_vs_pre",
                    "delta_n_qbeta_gt3_holdout_vs_pre",
                    "delta_n_logft_gt3_holdout_vs_pre",
                    "qbeta_count_recovery_relax_active",
                    "qbeta_count_recovery_logft_p95_max_delta_used",
                    "qbeta_count_recovery_logft_count_max_delta_used",
                    "qbeta_count_recovery_residue_logft_count_max_delta_used",
                    "qbeta_count_recovery_residue_overfit_max_delta_used",
                    "stability_gate_pass_all",
                    "stability_hard_metric_score",
                    "stability_violation_score",
                    "stability_violation_count",
                    "rank_stability_lock",
                ],
            )
            lock_rec = (
                candidate_stability_lock.get("recommended_candidate")
                if isinstance(candidate_stability_lock.get("recommended_candidate"), dict)
                else {}
            )
            lock_mode = str(candidate_stability_lock.get("mode", "not_run"))
            if lock_rec and lock_mode == "stable_locked":
                recommended_local = lock_rec
                recommended_source = f"candidate_stability_lock::{lock_mode}"
                localcorr_sweep["recommended_candidate"] = lock_rec
                if isinstance(candidate_stability_lock.get("top_ranked_candidates"), list):
                    localcorr_sweep["top_ranked_candidates"] = candidate_stability_lock.get("top_ranked_candidates")
            policy = localcorr_sweep.get("selection_policy") if isinstance(localcorr_sweep.get("selection_policy"), dict) else {}
            policy["candidate_stability_lock_enabled"] = True
            policy["candidate_stability_lock_status"] = str(candidate_stability_lock.get("status", "not_run"))
            policy["candidate_stability_lock_mode"] = lock_mode
            policy["candidate_stability_lock_stable_candidates_n"] = int(candidate_stability_lock.get("stable_candidates_n") or 0)
            localcorr_sweep["selection_policy"] = policy
        else:
            candidate_stability_lock = {
                "status": "not_run",
                "reason": "run_candidate_stability_lock_false",
                "mode": "not_run",
                "stable_candidates_n": 0,
                "evaluated_candidates_n": 0,
                "recommended_candidate": {},
                "all_candidates": [],
                "top_ranked_candidates": [],
                "reject_root_cause": [],
                "selection_policy": {
                    "mode": "not_run",
                    "enabled": False,
                },
            }
        localcorr_sweep["candidate_stability_lock"] = candidate_stability_lock
        if recommended_local and target_classes:
            final_class_gain_scales = _build_candidate_class_gain_scales(
                candidate=recommended_local,
                target_classes=target_classes,
            )
            final_class_sat_scales = _build_candidate_class_sat_scales(
                candidate=recommended_local,
                target_classes=target_classes,
            )
            final_class_blend_profiles = _build_candidate_class_blend_profiles(
                candidate=recommended_local,
                target_classes=target_classes,
            )
            final_rows, local_apply_meta = _apply_transition_class_local_correction(
                rows=mapped_rows,
                target_classes={str(x) for x in target_classes},
                gain=float(recommended_local.get("gain", 1.0)),
                sat_mev=float(recommended_local.get("sat_mev", 5.0)),
                blend=float(recommended_local.get("blend", 0.0)),
                sign_blend=float(recommended_local.get("sign_blend", 0.0)),
                class_gain_scales=final_class_gain_scales,
                class_sat_scales=final_class_sat_scales,
                class_blend_profiles=final_class_blend_profiles,
            )
            _, equalized_pack_local, split_meta_local = route_ab._equalized_route_audit(
                rows=final_rows,
                logft_sigma_proxy=float(args.logft_sigma_proxy),
                z_gate=float(args.z_gate),
                holdout_hash_modulo=int(args.holdout_hash_modulo),
                holdout_hash_residue=int(args.holdout_hash_residue),
                overfit_gap_gate=float(args.overfit_gap_gate),
                sigma_floor_mev=float(args.sigma_floor_mev),
            )
            route_eval_local = (
                equalized_pack_local.get("route_evaluation")
                if isinstance(equalized_pack_local.get("route_evaluation"), dict)
                else {}
            )
            route_b_eval_local = (
                route_eval_local.get("B_pmodel_proxy")
                if isinstance(route_eval_local.get("B_pmodel_proxy"), dict)
                else {}
            )
            if route_b_eval_local:
                rows_for_residue_audit = final_rows
                route_b_eval = route_b_eval_local
                split_meta_final = split_meta_local
                decision = _evaluate_route_b(
                    route_b_eval=route_b_eval,
                    closure_gate=closure_gate,
                    z_gate=float(args.z_gate),
                    overfit_gap_gate=float(args.overfit_gap_gate),
                )
                outlier_decomposition = _build_route_b_outlier_decomposition(
                    rows_mapped=final_rows,
                    route_b_eval=route_b_eval,
                    holdout_hash_modulo=int(args.holdout_hash_modulo),
                    holdout_hash_residue=int(args.holdout_hash_residue),
                    z_gate=float(args.z_gate),
                )
                post_hold = route_b_eval.get("holdout_all") if isinstance(route_b_eval.get("holdout_all"), dict) else {}
                post_overfit = route_b_eval.get("overfit_guard") if isinstance(route_b_eval.get("overfit_guard"), dict) else {}
                local_correction = {
                    "status": "applied",
                    "target_classes": target_classes,
                    "selected_candidate": recommended_local,
                    "selected_candidate_source": recommended_source,
                    "apply_meta": local_apply_meta,
                    "pre_local_metrics": {
                        "holdout_p95_abs_z_qbeta": _to_float(pre_local_hold.get("p95_abs_z_qbeta")),
                        "holdout_p95_abs_z_logft": _to_float(pre_local_hold.get("p95_abs_z_logft_proxy")),
                        "overfit_gap_qbeta": _to_float(pre_local_overfit.get("p95_gap_qbeta")),
                    },
                    "post_local_metrics": {
                        "holdout_p95_abs_z_qbeta": _to_float(post_hold.get("p95_abs_z_qbeta")),
                        "holdout_p95_abs_z_logft": _to_float(post_hold.get("p95_abs_z_logft_proxy")),
                        "overfit_gap_qbeta": _to_float(post_overfit.get("p95_gap_qbeta")),
                    },
                    "delta_post_minus_pre": {
                        "holdout_p95_abs_z_qbeta": _to_float(post_hold.get("p95_abs_z_qbeta"))
                        - _to_float(pre_local_hold.get("p95_abs_z_qbeta")),
                        "holdout_p95_abs_z_logft": _to_float(post_hold.get("p95_abs_z_logft_proxy"))
                        - _to_float(pre_local_hold.get("p95_abs_z_logft_proxy")),
                        "overfit_gap_qbeta": _to_float(post_overfit.get("p95_gap_qbeta"))
                        - _to_float(pre_local_overfit.get("p95_gap_qbeta")),
                    },
                }
                _plot_local_correction_sweep_pareto(
                    sweep=localcorr_sweep,
                    pre_local_holdout_p95=_to_float(pre_local_hold.get("p95_abs_z_qbeta")),
                    pre_local_overfit_gap_q=_to_float(pre_local_overfit.get("p95_gap_qbeta")),
                    final_holdout_p95=_to_float(post_hold.get("p95_abs_z_qbeta")),
                    final_overfit_gap_q=_to_float(post_overfit.get("p95_gap_qbeta")),
                    out_png=out_localcorr_png,
                )
            else:
                local_correction = {
                    "status": "failed",
                    "reason": "selected_candidate_no_route_b_eval",
                    "target_classes": target_classes,
                    "selected_candidate": recommended_local,
                    "selected_candidate_source": recommended_source,
                }
        else:
            local_correction = {
                "status": "not_selected",
                "reason": "no_recommended_candidate_or_target_class",
                "target_classes": target_classes,
                "selected_candidate_source": recommended_source,
            }

    if bool(args.run_residue_robustness) and str(local_correction.get("status")) == "applied":
        residue_robustness = _build_residue_robustness(
            rows_mapped=rows_for_residue_audit,
            closure_gate=closure_gate,
            logft_sigma_proxy=float(args.logft_sigma_proxy),
            z_gate=float(args.z_gate),
            holdout_hash_modulo=int(modulo),
            holdout_hash_residues=[int(r) for r in residue_values],
            overfit_gap_gate=float(args.overfit_gap_gate),
            sigma_floor_mev=float(args.sigma_floor_mev),
            reference_residue=int(args.holdout_hash_residue),
        )
        _write_csv_rows(
            out_residue_csv,
            residue_robustness.get("rows") if isinstance(residue_robustness.get("rows"), list) else [],
            fieldnames=[
                "residue",
                "rows_holdout",
                "holdout_p95_abs_z_qbeta",
                "holdout_p95_abs_z_logft",
                "holdout_max_abs_z_qbeta",
                "overfit_gap_qbeta",
                "n_qbeta_gt3_holdout",
                "n_logft_gt3_holdout",
                "top_priority_class",
                "top_priority_q_gt3",
                "hard_fail_count",
                "watch_fail_count",
                "overall_status",
                "delta_holdout_p95_abs_z_qbeta_vs_ref",
                "delta_holdout_p95_abs_z_logft_vs_ref",
                "delta_overfit_gap_qbeta_vs_ref",
                "delta_n_qbeta_gt3_holdout_vs_ref",
                "delta_n_logft_gt3_holdout_vs_ref",
                "delta_top_priority_q_gt3_vs_ref",
            ],
        )
        _plot_residue_robustness(residue_pack=residue_robustness, out_png=out_residue_png)
    else:
        residue_robustness = {
            "status": "not_run",
            "reason": (
                "run_residue_robustness_false"
                if not bool(args.run_residue_robustness)
                else "local_correction_not_applied"
            ),
            "reference_residue": int(args.holdout_hash_residue),
            "residue_list": [int(r) % modulo for r in residue_values],
            "rows": [],
            "summary": {},
        }

    current_hold = route_b_eval.get("holdout_all") if isinstance(route_b_eval.get("holdout_all"), dict) else {}
    current_overfit = route_b_eval.get("overfit_guard") if isinstance(route_b_eval.get("overfit_guard"), dict) else {}
    baseline_comparison = {
        "baseline_mode": "baseline",
        "current_mode": str(args.hflavor_mode),
        "delta_holdout_p95_abs_z_qbeta": _to_float(current_hold.get("p95_abs_z_qbeta"))
        - _to_float(baseline_hold.get("p95_abs_z_qbeta")),
        "delta_holdout_p95_abs_z_logft_proxy": _to_float(current_hold.get("p95_abs_z_logft_proxy"))
        - _to_float(baseline_hold.get("p95_abs_z_logft_proxy")),
        "delta_overfit_gap_qbeta": _to_float(current_overfit.get("p95_gap_qbeta"))
        - _to_float(baseline_overfit.get("p95_gap_qbeta")),
        "delta_holdout_p95_abs_z_qbeta_vs_pre_local": _to_float(current_hold.get("p95_abs_z_qbeta"))
        - _to_float(pre_local_hold.get("p95_abs_z_qbeta")),
        "delta_holdout_p95_abs_z_logft_proxy_vs_pre_local": _to_float(current_hold.get("p95_abs_z_logft_proxy"))
        - _to_float(pre_local_hold.get("p95_abs_z_logft_proxy")),
        "delta_overfit_gap_qbeta_vs_pre_local": _to_float(current_overfit.get("p95_gap_qbeta"))
        - _to_float(pre_local_overfit.get("p95_gap_qbeta")),
    }

    csv_rows: List[Dict[str, Any]] = [
        {
            "metric_id": "route_b_holdout_p95_qbeta",
            "value": decision["gate_values"]["p95_abs_z_qbeta_holdout"],
            "threshold": float(args.z_gate),
            "operator": "<=",
            "pass": decision["gate_pass"]["holdout_qbeta_p95"],
            "gate_level": "hard",
            "note": "holdout p95 abs(z_Qβ)",
        },
        {
            "metric_id": "route_b_holdout_p95_logft",
            "value": decision["gate_values"]["p95_abs_z_logft_holdout"],
            "threshold": float(args.z_gate),
            "operator": "<=",
            "pass": decision["gate_pass"]["holdout_logft_p95"],
            "gate_level": "hard",
            "note": "holdout p95 abs(z_logft_proxy)",
        },
        {
            "metric_id": "route_b_overfit_gap_qbeta",
            "value": decision["gate_values"]["p95_gap_qbeta_holdout_minus_train"],
            "threshold": float(args.overfit_gap_gate),
            "operator": "<=",
            "pass": decision["gate_pass"]["overfit_guard"],
            "gate_level": "hard",
            "note": "overfit guard uses both Qβ and logft gaps",
        },
        {
            "metric_id": "closure_ckm_pmns_hard",
            "value": 1.0 if decision["gate_pass"]["closure_hard"] else 0.0,
            "threshold": 1.0,
            "operator": ">=",
            "pass": decision["gate_pass"]["closure_hard"],
            "gate_level": "hard",
            "note": "combined CKM+PMNS hard gate",
        },
        {
            "metric_id": "closure_ckm_pmns_watch",
            "value": 1.0 if decision["gate_pass"]["closure_watch"] else 0.0,
            "threshold": 1.0,
            "operator": ">=",
            "pass": decision["gate_pass"]["closure_watch"],
            "gate_level": "watch",
            "note": "combined CKM+PMNS watch gate",
        },
    ]
    _write_csv(out_csv, csv_rows)

    _write_csv_rows(
        out_decomp_csv,
        outlier_decomposition.get("csv_rows") if isinstance(outlier_decomposition.get("csv_rows"), list) else [],
        fieldnames=[
            "scope",
            "rank",
            "group_key",
            "n_rows",
            "n_qbeta_gt3",
            "n_logft_gt3",
            "frac_qbeta_gt3",
            "frac_logft_gt3",
            "share_fail_qbeta",
            "share_fail_logft",
            "p95_abs_z_qbeta",
            "max_abs_z_qbeta",
            "p95_abs_z_logft",
            "max_abs_z_logft",
            "priority_score",
        ],
    )
    _plot_outlier_decomposition(decomposition=outlier_decomposition, out_png=out_decomp_png)

    if args.hflavor_mode == "baseline":
        step_id = "8.7.31.1"
    elif (
        bool(args.run_localcorr_sweep)
        and args.hflavor_mode != "baseline"
        and bool(args.localcorr_constrained_select)
        and bool(args.localcorr_use_qfail_guard)
        and bool(args.localcorr_residue_reweight)
        and bool(args.localcorr_qbeta_overfit_guard)
        and bool(args.localcorr_logft_rootcause_guard)
        and bool(args.localcorr_logft_rootcause_refreeze)
        and isinstance(localcorr_residue_reweight, dict)
        and str(localcorr_residue_reweight.get("status")) == "completed"
        and isinstance(localcorr_residue_reweight.get("selection_policy"), dict)
        and bool((localcorr_residue_reweight.get("selection_policy") or {}).get("logft_rootcause_refreeze_applied"))
        and bool(args.run_candidate_stability_lock)
        and isinstance(candidate_stability_lock, dict)
        and str(candidate_stability_lock.get("status")) == "completed"
        and bool(args.run_residue_robustness)
        and isinstance(residue_robustness, dict)
        and str(residue_robustness.get("status")) == "completed"
    ):
        rw_policy_step = (
            localcorr_residue_reweight.get("selection_policy")
            if isinstance(localcorr_residue_reweight.get("selection_policy"), dict)
            else {}
        )
        lock_policy_step = (
            candidate_stability_lock.get("selection_policy")
            if isinstance(candidate_stability_lock.get("selection_policy"), dict)
            else {}
        )
        lock_rec_step = (
            candidate_stability_lock.get("recommended_candidate")
            if isinstance(candidate_stability_lock.get("recommended_candidate"), dict)
            else {}
        )
        root_retune_source_step = str(
            rw_policy_step.get("logft_rootcause_retune_source") or ""
        ).strip().lower()
        root_retune_cycle_closed = bool(
            rw_policy_step.get("logft_rootcause_retune_applied")
        ) or root_retune_source_step in {"not_needed", "disabled"}
        residue_dual_retune_source_step = str(
            rw_policy_step.get("residue_dual_retune_source") or ""
        ).strip().lower()
        residue_dual_retune_cycle_closed = bool(
            rw_policy_step.get("residue_dual_retune_applied")
        ) or residue_dual_retune_source_step in {"not_needed", "disabled"}
        combined_pass_n = int(rw_policy_step.get("n_candidates_combined_guard_pass") or 0)
        combined_target = int(rw_policy_step.get("logft_rootcause_retune_target_combined_pass_min") or 0)
        lock_qbeta_delta = _to_float(lock_rec_step.get("delta_holdout_p95_abs_z_qbeta_vs_pre"))
        lock_logft_delta = _to_float(lock_rec_step.get("delta_holdout_p95_abs_z_logft_vs_pre"))
        lock_overfit_delta = _to_float(lock_rec_step.get("delta_overfit_gap_qbeta_vs_pre"))
        lock_n_qbeta_delta = int(lock_rec_step.get("delta_n_qbeta_gt3_holdout_vs_pre") or 0)
        lock_qbeta_recovery_policy = (
            lock_policy_step.get("qbeta_count_recovery_policy")
            if isinstance(lock_policy_step.get("qbeta_count_recovery_policy"), dict)
            else {}
        )
        lock_qbeta_recovery_enabled = bool(lock_qbeta_recovery_policy.get("enabled"))
        lock_qbeta_count_max_delta = int(lock_qbeta_recovery_policy.get("qbeta_count_max_delta") or 0)
        has_top_boost_grid = any(float(v) > 1.0 for v in localcorr_top_class_gain_boost_values if math.isfinite(float(v)))
        has_sigma_low_modeinc_sat_grid = any(
            abs(float(v) - 1.0) > 1.0e-12
            for v in localcorr_sigma_low_mode_inconsistent_sat_scale_values
            if math.isfinite(float(v))
        )
        has_sigma_low_modeinc_zref_grid = any(
            float(v) > 3.0
            for v in localcorr_sigma_low_mode_inconsistent_zref_values
            if math.isfinite(float(v))
        )
        has_sigma_low_modeinc_minblend_grid = any(
            float(v) < 1.0
            for v in localcorr_sigma_low_mode_inconsistent_min_blend_scale_values
            if math.isfinite(float(v))
        )
        has_sigma_low_modeinc_signblend_grid = any(
            abs(float(v) - 1.0) > 1.0e-12
            for v in localcorr_sigma_low_mode_inconsistent_sign_blend_scale_values
            if math.isfinite(float(v))
        )
        has_positive_signblend_grid = any(
            float(v) > 0.0 for v in localcorr_sign_blend_values if math.isfinite(float(v))
        )
        signblend_positive_only = (
            bool(localcorr_sign_blend_values)
            and all(float(v) > 0.0 for v in localcorr_sign_blend_values if math.isfinite(float(v)))
        )
        signscale_targeted_only = (
            bool(localcorr_sigma_low_mode_inconsistent_sign_blend_scale_values)
            and all(
                float(v) < 1.0
                for v in localcorr_sigma_low_mode_inconsistent_sign_blend_scale_values
                if math.isfinite(float(v))
            )
        )
        localcorr_grid_spec = (
            localcorr_sweep.get("grid_spec")
            if isinstance(localcorr_sweep.get("grid_spec"), dict)
            else {}
        )
        n_localcorr_candidates = int(localcorr_grid_spec.get("n_total_candidates") or 0)
        has_top_boost_in_candidate = "top_class_gain_boost" in lock_rec_step
        has_sigma_low_modeinc_sat_in_candidate = "sigma_low_mode_inconsistent_sat_scale" in lock_rec_step
        has_sigma_low_modeinc_zref_in_candidate = "sigma_low_mode_inconsistent_zref" in lock_rec_step
        has_sigma_low_modeinc_minblend_in_candidate = "sigma_low_mode_inconsistent_min_blend_scale" in lock_rec_step
        has_sigma_low_modeinc_signblend_in_candidate = (
            "sigma_low_mode_inconsistent_sign_blend_scale" in lock_rec_step
        )
        lock_effective_signblend = _to_float(lock_rec_step.get("effective_sigma_low_mode_sign_blend"))
        retune_grid_completed = bool(args.localcorr_residue_class_retune_grid) and isinstance(
            localcorr_residue_reweight_retune, dict
        ) and str(localcorr_residue_reweight_retune.get("status")) == "completed"
        retune_grid_applied = bool(
            isinstance(localcorr_residue_reweight_retune, dict)
            and localcorr_residue_reweight_retune.get("applied_best")
        )
        retune_strategy_completed = bool(args.localcorr_residue_class_retune_strategy_audit) and isinstance(
            localcorr_residue_reweight_retune_strategy, dict
        ) and str(localcorr_residue_reweight_retune_strategy.get("status")) == "completed"
        retune_strategy_applied = bool(
            isinstance(localcorr_residue_reweight_retune_strategy, dict)
            and localcorr_residue_reweight_retune_strategy.get("applied_best")
        )
        retune_strategy_count = int(
            localcorr_residue_reweight_retune_strategy.get("strategy_count")
            if isinstance(localcorr_residue_reweight_retune_strategy, dict)
            else 0
        )
        retune_target_class_n = 0
        if isinstance(localcorr_residue_reweight_retune, dict):
            rec_retune_step = (
                localcorr_residue_reweight_retune.get("recommended_candidate")
                if isinstance(localcorr_residue_reweight_retune.get("recommended_candidate"), dict)
                else {}
            )
            retune_target_class_n = int(rec_retune_step.get("n_target_classes") or 0)
            if retune_target_class_n <= 0:
                retune_target_class_n = len(
                    localcorr_residue_reweight_retune.get("target_class_keys")
                    if isinstance(localcorr_residue_reweight_retune.get("target_class_keys"), list)
                    else []
                )
        strategy_density_stage_stage2_completed = bool(
            retune_strategy_completed
            and retune_strategy_applied
            and int(retune_strategy_count) >= 2
            and int(n_localcorr_candidates) >= 48
            and int(args.localcorr_residue_reweight_max_candidates) >= 48
        )
        strategy_rows_step = (
            localcorr_residue_reweight_retune_strategy.get("all_strategies")
            if isinstance(localcorr_residue_reweight_retune_strategy, dict)
            and isinstance(localcorr_residue_reweight_retune_strategy.get("all_strategies"), list)
            else []
        )
        retune_strategy_top_k_max = 0
        retune_strategy_explicit_group_count = 0
        for row_strat in strategy_rows_step:
            if not isinstance(row_strat, dict):
                continue
            target_top_k = int(row_strat.get("target_top_k") or 0)
            if target_top_k > retune_strategy_top_k_max:
                retune_strategy_top_k_max = target_top_k
            mode_str = str(row_strat.get("strategy_mode", "")).strip().lower()
            if mode_str != "top_k":
                retune_strategy_explicit_group_count += 1
        class_guard_key_union: Set[str] = set()
        class_guard_key_union.update(str(k) for k in localcorr_class_logft_count_max_delta_by_key.keys())
        class_guard_key_union.update(str(k) for k in localcorr_class_residue_logft_count_max_delta_by_key.keys())
        class_guard_key_union.update(str(k) for k in localcorr_class_residue_top_priority_max_delta_by_key.keys())
        class_guard_key_union.update(str(k) for k in localcorr_class_residue_overfit_max_delta_by_key.keys())
        class_guard_key_count = len([k for k in class_guard_key_union if str(k).strip()])
        class_guard_stage_completed = bool(
            strategy_density_stage_stage2_completed
            and bool(rw_policy_step.get("class_key_threshold_guard_enabled"))
            and int(rw_policy_step.get("n_candidates_class_key_guard_applied") or 0) > 0
        )
        class_guard_retune_stage_completed = bool(
            bool(rw_policy_step.get("class_key_threshold_guard_enabled"))
            and int(rw_policy_step.get("n_candidates_class_key_guard_applied") or 0) > 0
            and retune_strategy_completed
            and retune_strategy_applied
            and int(class_guard_key_count) >= 4
            and int(retune_strategy_top_k_max) >= 3
            and (int(retune_strategy_count) >= 4 or int(retune_strategy_explicit_group_count) >= 2)
        )
        strategy_density_stage_completed = bool(
            retune_strategy_completed
            and retune_strategy_applied
            and int(retune_strategy_count) >= 2
            and int(n_localcorr_candidates) >= 32
            and int(args.localcorr_residue_reweight_max_candidates) >= 32
        )
        if class_guard_retune_stage_completed:
            step_id = "8.7.31.33"
        elif class_guard_stage_completed:
            step_id = "8.7.31.32"
        elif strategy_density_stage_stage2_completed:
            step_id = "8.7.31.31"
        elif strategy_density_stage_completed:
            step_id = "8.7.31.30"
        elif retune_strategy_completed and retune_strategy_applied and int(retune_strategy_count) >= 2:
            step_id = "8.7.31.29"
        elif retune_grid_completed and retune_grid_applied and int(retune_target_class_n) >= 2:
            step_id = "8.7.31.28"
        elif retune_grid_completed and retune_grid_applied:
            step_id = "8.7.31.27"
        elif (
            int(args.localcorr_top_classes) > 5
            and bool(args.localcorr_residue_dual_guard)
            and bool(args.localcorr_residue_dual_retune)
            and bool(args.localcorr_logft_rootcause_retune)
            and residue_dual_retune_cycle_closed
            and root_retune_cycle_closed
            and combined_pass_n >= max(2, combined_target)
            and str(candidate_stability_lock.get("mode", "")) == "stable_locked"
            and isinstance(lock_policy_step.get("weights_hard_metric_score"), dict)
            and bool(lock_qbeta_recovery_enabled)
            and lock_n_qbeta_delta <= int(lock_qbeta_count_max_delta)
            and math.isfinite(lock_qbeta_delta)
            and math.isfinite(lock_logft_delta)
            and math.isfinite(lock_overfit_delta)
            and lock_qbeta_delta <= 0.0
            and lock_logft_delta <= 0.0
            and lock_overfit_delta <= 0.0
            and has_top_boost_grid
            and has_top_boost_in_candidate
            and has_sigma_low_modeinc_sat_grid
            and has_sigma_low_modeinc_sat_in_candidate
            and has_sigma_low_modeinc_zref_grid
            and has_sigma_low_modeinc_minblend_grid
            and has_sigma_low_modeinc_zref_in_candidate
            and has_sigma_low_modeinc_minblend_in_candidate
            and has_sigma_low_modeinc_signblend_grid
            and has_sigma_low_modeinc_signblend_in_candidate
            and has_positive_signblend_grid
            and signblend_positive_only
            and signscale_targeted_only
            and math.isfinite(lock_effective_signblend)
            and lock_effective_signblend > 0.0
            and n_localcorr_candidates >= 3000
        ):
            step_id = "8.7.31.26"
        elif (
            int(args.localcorr_top_classes) > 5
            and bool(args.localcorr_residue_dual_guard)
            and bool(args.localcorr_residue_dual_retune)
            and bool(args.localcorr_logft_rootcause_retune)
            and residue_dual_retune_cycle_closed
            and root_retune_cycle_closed
            and combined_pass_n >= max(2, combined_target)
            and str(candidate_stability_lock.get("mode", "")) == "stable_locked"
            and isinstance(lock_policy_step.get("weights_hard_metric_score"), dict)
            and bool(lock_qbeta_recovery_enabled)
            and lock_n_qbeta_delta <= int(lock_qbeta_count_max_delta)
            and math.isfinite(lock_qbeta_delta)
            and math.isfinite(lock_logft_delta)
            and math.isfinite(lock_overfit_delta)
            and lock_qbeta_delta <= 0.0
            and lock_logft_delta <= 0.0
            and lock_overfit_delta <= 0.0
            and has_top_boost_grid
            and has_top_boost_in_candidate
            and has_sigma_low_modeinc_sat_grid
            and has_sigma_low_modeinc_sat_in_candidate
            and has_sigma_low_modeinc_zref_grid
            and has_sigma_low_modeinc_minblend_grid
            and has_sigma_low_modeinc_zref_in_candidate
            and has_sigma_low_modeinc_minblend_in_candidate
            and has_sigma_low_modeinc_signblend_grid
            and has_sigma_low_modeinc_signblend_in_candidate
            and n_localcorr_candidates >= 500
        ):
            step_id = "8.7.31.25"
        elif (
            int(args.localcorr_top_classes) > 5
            and bool(args.localcorr_residue_dual_guard)
            and bool(args.localcorr_residue_dual_retune)
            and bool(args.localcorr_logft_rootcause_retune)
            and residue_dual_retune_cycle_closed
            and root_retune_cycle_closed
            and combined_pass_n >= max(2, combined_target)
            and str(candidate_stability_lock.get("mode", "")) == "stable_locked"
            and isinstance(lock_policy_step.get("weights_hard_metric_score"), dict)
            and bool(lock_qbeta_recovery_enabled)
            and lock_n_qbeta_delta <= int(lock_qbeta_count_max_delta)
            and math.isfinite(lock_qbeta_delta)
            and math.isfinite(lock_logft_delta)
            and math.isfinite(lock_overfit_delta)
            and lock_qbeta_delta <= 0.0
            and lock_logft_delta <= 0.0
            and lock_overfit_delta <= 0.0
            and has_top_boost_grid
            and has_top_boost_in_candidate
            and has_sigma_low_modeinc_sat_grid
            and has_sigma_low_modeinc_sat_in_candidate
            and has_sigma_low_modeinc_zref_grid
            and has_sigma_low_modeinc_minblend_grid
            and has_sigma_low_modeinc_zref_in_candidate
            and has_sigma_low_modeinc_minblend_in_candidate
            and n_localcorr_candidates >= 10000
        ):
            step_id = "8.7.31.24"
        elif (
            int(args.localcorr_top_classes) > 5
            and bool(args.localcorr_residue_dual_guard)
            and bool(args.localcorr_residue_dual_retune)
            and bool(args.localcorr_logft_rootcause_retune)
            and residue_dual_retune_cycle_closed
            and root_retune_cycle_closed
            and combined_pass_n >= max(2, combined_target)
            and str(candidate_stability_lock.get("mode", "")) == "stable_locked"
            and isinstance(lock_policy_step.get("weights_hard_metric_score"), dict)
            and bool(lock_qbeta_recovery_enabled)
            and lock_n_qbeta_delta <= int(lock_qbeta_count_max_delta)
            and math.isfinite(lock_qbeta_delta)
            and math.isfinite(lock_logft_delta)
            and math.isfinite(lock_overfit_delta)
            and lock_qbeta_delta <= 0.0
            and lock_logft_delta <= 0.0
            and lock_overfit_delta <= 0.0
            and has_top_boost_grid
            and has_top_boost_in_candidate
            and has_sigma_low_modeinc_sat_grid
            and has_sigma_low_modeinc_sat_in_candidate
            and n_localcorr_candidates >= 800
        ):
            step_id = "8.7.31.23"
        elif (
            int(args.localcorr_top_classes) > 2
            and bool(args.localcorr_residue_dual_guard)
            and bool(args.localcorr_residue_dual_retune)
            and bool(args.localcorr_logft_rootcause_retune)
            and residue_dual_retune_cycle_closed
            and root_retune_cycle_closed
            and combined_pass_n >= max(2, combined_target)
            and str(candidate_stability_lock.get("mode", "")) == "stable_locked"
            and isinstance(lock_policy_step.get("weights_hard_metric_score"), dict)
            and bool(lock_qbeta_recovery_enabled)
            and lock_n_qbeta_delta <= int(lock_qbeta_count_max_delta)
            and has_top_boost_grid
            and has_top_boost_in_candidate
            and has_sigma_low_modeinc_sat_grid
            and has_sigma_low_modeinc_sat_in_candidate
            and has_sigma_low_modeinc_zref_grid
            and has_sigma_low_modeinc_minblend_grid
            and has_sigma_low_modeinc_zref_in_candidate
            and has_sigma_low_modeinc_minblend_in_candidate
            and n_localcorr_candidates >= 800
            and lock_qbeta_delta <= -100.0
            and lock_overfit_delta <= -100.0
        ):
            step_id = "8.7.31.22"
        elif (
            int(args.localcorr_top_classes) > 2
            and bool(args.localcorr_residue_dual_guard)
            and bool(args.localcorr_residue_dual_retune)
            and bool(args.localcorr_logft_rootcause_retune)
            and residue_dual_retune_cycle_closed
            and root_retune_cycle_closed
            and combined_pass_n >= max(2, combined_target)
            and str(candidate_stability_lock.get("mode", "")) == "stable_locked"
            and isinstance(lock_policy_step.get("weights_hard_metric_score"), dict)
            and has_top_boost_grid
            and has_top_boost_in_candidate
            and has_sigma_low_modeinc_sat_grid
            and has_sigma_low_modeinc_sat_in_candidate
            and has_sigma_low_modeinc_zref_grid
            and has_sigma_low_modeinc_minblend_grid
            and has_sigma_low_modeinc_zref_in_candidate
            and has_sigma_low_modeinc_minblend_in_candidate
            and n_localcorr_candidates >= 800
            and lock_qbeta_delta <= -100.0
            and lock_overfit_delta <= -100.0
        ):
            step_id = "8.7.31.21"
        elif (
            int(args.localcorr_top_classes) > 2
            and bool(args.localcorr_residue_dual_guard)
            and bool(args.localcorr_residue_dual_retune)
            and bool(args.localcorr_logft_rootcause_retune)
            and residue_dual_retune_cycle_closed
            and root_retune_cycle_closed
            and combined_pass_n >= max(2, combined_target)
            and str(candidate_stability_lock.get("mode", "")) == "stable_locked"
            and isinstance(lock_policy_step.get("weights_hard_metric_score"), dict)
            and has_top_boost_grid
            and has_top_boost_in_candidate
            and has_sigma_low_modeinc_sat_grid
            and has_sigma_low_modeinc_sat_in_candidate
            and n_localcorr_candidates >= 200
            and lock_qbeta_delta <= -100.0
            and lock_overfit_delta <= -100.0
        ):
            step_id = "8.7.31.20"
        elif (
            int(args.localcorr_top_classes) > 2
            and bool(args.localcorr_residue_dual_guard)
            and bool(args.localcorr_residue_dual_retune)
            and bool(args.localcorr_logft_rootcause_retune)
            and residue_dual_retune_cycle_closed
            and root_retune_cycle_closed
            and combined_pass_n >= max(2, combined_target)
            and str(candidate_stability_lock.get("mode", "")) == "stable_locked"
            and isinstance(lock_policy_step.get("weights_hard_metric_score"), dict)
            and has_top_boost_grid
            and has_top_boost_in_candidate
            and n_localcorr_candidates >= 100
            and lock_qbeta_delta <= -100.0
            and lock_overfit_delta <= -100.0
        ):
            step_id = "8.7.31.19"
        elif (
            int(args.localcorr_top_classes) > 2
            and bool(args.localcorr_residue_dual_guard)
            and bool(args.localcorr_residue_dual_retune)
            and bool(args.localcorr_logft_rootcause_retune)
            and bool(rw_policy_step.get("residue_dual_retune_applied"))
            and bool(rw_policy_step.get("logft_rootcause_retune_applied"))
            and combined_pass_n >= max(2, combined_target)
            and str(candidate_stability_lock.get("mode", "")) == "stable_locked"
            and isinstance(lock_policy_step.get("weights_hard_metric_score"), dict)
            and lock_qbeta_delta <= -100.0
        ):
            step_id = "8.7.31.18"
        elif (
            int(args.localcorr_top_classes) > 2
            and bool(args.localcorr_residue_dual_guard)
            and bool(args.localcorr_residue_dual_retune)
            and bool(args.localcorr_logft_rootcause_retune)
            and bool(rw_policy_step.get("residue_dual_retune_applied"))
            and bool(rw_policy_step.get("logft_rootcause_retune_applied"))
            and combined_pass_n >= max(2, combined_target)
        ):
            step_id = "8.7.31.17"
        elif (
            int(args.localcorr_top_classes) > 2
            and bool(args.localcorr_residue_dual_guard)
            and bool(args.localcorr_residue_dual_retune)
            and bool(rw_policy_step.get("residue_dual_retune_applied"))
            and combined_pass_n > 0
        ):
            step_id = "8.7.31.16"
        elif int(args.localcorr_top_classes) > 2 and bool(args.localcorr_residue_dual_guard):
            step_id = "8.7.31.15"
        elif int(args.localcorr_top_classes) > 2:
            step_id = "8.7.31.14"
        else:
            step_id = "8.7.31.13"
    elif (
        bool(args.run_localcorr_sweep)
        and args.hflavor_mode != "baseline"
        and bool(args.localcorr_constrained_select)
        and bool(args.localcorr_use_qfail_guard)
        and bool(args.localcorr_residue_reweight)
        and bool(args.localcorr_qbeta_overfit_guard)
        and bool(args.localcorr_logft_rootcause_guard)
        and isinstance(localcorr_residue_reweight, dict)
        and str(localcorr_residue_reweight.get("status")) == "completed"
        and bool(args.run_candidate_stability_lock)
        and isinstance(candidate_stability_lock, dict)
        and str(candidate_stability_lock.get("status")) == "completed"
        and bool(args.run_residue_robustness)
        and isinstance(residue_robustness, dict)
        and str(residue_robustness.get("status")) == "completed"
    ):
        step_id = "8.7.31.12"
    elif (
        bool(args.run_localcorr_sweep)
        and args.hflavor_mode != "baseline"
        and bool(args.localcorr_constrained_select)
        and bool(args.localcorr_use_qfail_guard)
        and bool(args.localcorr_residue_reweight)
        and bool(args.localcorr_qbeta_overfit_guard)
        and isinstance(localcorr_residue_reweight, dict)
        and str(localcorr_residue_reweight.get("status")) == "completed"
        and bool(args.run_candidate_stability_lock)
        and isinstance(candidate_stability_lock, dict)
        and str(candidate_stability_lock.get("status")) == "completed"
        and bool(args.run_residue_robustness)
        and isinstance(residue_robustness, dict)
        and str(residue_robustness.get("status")) == "completed"
    ):
        step_id = "8.7.31.11"
    elif (
        bool(args.run_localcorr_sweep)
        and args.hflavor_mode != "baseline"
        and bool(args.localcorr_constrained_select)
        and bool(args.localcorr_use_qfail_guard)
        and bool(args.localcorr_residue_reweight)
        and isinstance(localcorr_residue_reweight, dict)
        and str(localcorr_residue_reweight.get("status")) == "completed"
        and bool(args.run_candidate_stability_lock)
        and isinstance(candidate_stability_lock, dict)
        and str(candidate_stability_lock.get("status")) == "completed"
        and bool(args.run_residue_robustness)
        and isinstance(residue_robustness, dict)
        and str(residue_robustness.get("status")) == "completed"
    ):
        step_id = "8.7.31.10"
    elif (
        bool(args.run_localcorr_sweep)
        and args.hflavor_mode != "baseline"
        and bool(args.localcorr_constrained_select)
        and bool(args.localcorr_use_qfail_guard)
        and bool(args.localcorr_residue_reweight)
        and isinstance(localcorr_residue_reweight, dict)
        and str(localcorr_residue_reweight.get("status")) == "completed"
        and bool(args.run_residue_robustness)
        and isinstance(residue_robustness, dict)
        and str(residue_robustness.get("status")) == "completed"
    ):
        step_id = "8.7.31.9"
    elif (
        bool(args.run_localcorr_sweep)
        and args.hflavor_mode != "baseline"
        and bool(args.localcorr_constrained_select)
        and bool(args.localcorr_use_qfail_guard)
        and bool(args.run_residue_robustness)
        and isinstance(residue_robustness, dict)
        and str(residue_robustness.get("status")) == "completed"
    ):
        step_id = "8.7.31.8"
    elif (
        bool(args.run_localcorr_sweep)
        and args.hflavor_mode != "baseline"
        and bool(args.localcorr_constrained_select)
        and bool(args.localcorr_use_qfail_guard)
    ):
        step_id = "8.7.31.7"
    elif bool(args.run_localcorr_sweep) and args.hflavor_mode != "baseline" and bool(args.localcorr_constrained_select):
        step_id = "8.7.31.6"
    elif bool(args.run_localcorr_sweep) and args.hflavor_mode != "baseline":
        step_id = "8.7.31.5"
    elif bool(args.run_hflavor_sweep):
        step_id = "8.7.31.4"
    else:
        step_id = "8.7.31.3"

    payload: Dict[str, Any] = {
        "generated_utc": _iso_now(),
        "phase": 8,
        "step": step_id,
        "title": "Weak-interaction Route-B standalone audit",
        "inputs": {
            "beta_qvalue_full_csv": _file_signature(args.in_csv),
            "beta_qvalue_metrics_json": _file_signature(args.in_json),
            "route_ab_audit_json": _file_signature(args.route_ab_audit_json),
            "ckm_audit_json": _file_signature(args.ckm_audit_json),
            "pmns_audit_json": _file_signature(args.pmns_audit_json),
        },
        "gate_policy": {
            "z_threshold": float(args.z_gate),
            "overfit_gap_gate": float(args.overfit_gap_gate),
            "holdout_split": {
                "modulo": int(args.holdout_hash_modulo),
                "residue": int(args.holdout_hash_residue),
            },
            "localcorr_targeting": {
                "top_classes": int(args.localcorr_top_classes),
                "top_class_gain_boost_values": [float(v) for v in localcorr_top_class_gain_boost_values],
                "top_class_sign_blend_scale_values": [float(v) for v in localcorr_top_class_sign_blend_scale_values],
                "sigma_low_mode_inconsistent_sat_scale_values": [
                    float(v) for v in localcorr_sigma_low_mode_inconsistent_sat_scale_values
                ],
                "sigma_low_mode_inconsistent_zref_values": [
                    float(v) for v in localcorr_sigma_low_mode_inconsistent_zref_values
                ],
                "sigma_low_mode_inconsistent_min_blend_scale_values": [
                    float(v) for v in localcorr_sigma_low_mode_inconsistent_min_blend_scale_values
                ],
                "sigma_low_mode_inconsistent_sign_blend_scale_values": [
                    float(v) for v in localcorr_sigma_low_mode_inconsistent_sign_blend_scale_values
                ],
            },
            "residue_robustness": {
                "run": bool(args.run_residue_robustness),
                "residues_requested": [int(r) for r in residue_values],
            },
            "localcorr_residue_reweight": {
                "enabled": bool(args.localcorr_residue_reweight),
                "max_candidates": int(args.localcorr_residue_reweight_max_candidates),
                "qbeta_overfit_guard_enabled": bool(args.localcorr_qbeta_overfit_guard),
                "qbeta_max_delta_allowed": float(args.localcorr_qbeta_max_delta),
                "overfit_max_delta_allowed": float(args.localcorr_overfit_max_delta),
                "classwise_norm_blend": float(args.localcorr_residue_classwise_norm_blend),
                "class_key_threshold_guard_enabled": bool(localcorr_class_key_threshold_guard_enabled),
                "class_logft_count_max_delta_by_key": {
                    str(k): int(v) for k, v in localcorr_class_logft_count_max_delta_by_key.items()
                },
                "class_residue_logft_count_max_delta_by_key": {
                    str(k): int(v) for k, v in localcorr_class_residue_logft_count_max_delta_by_key.items()
                },
                "class_residue_top_priority_max_delta_by_key": {
                    str(k): int(v) for k, v in localcorr_class_residue_top_priority_max_delta_by_key.items()
                },
                "class_residue_overfit_max_delta_by_key": {
                    str(k): float(v) for k, v in localcorr_class_residue_overfit_max_delta_by_key.items()
                },
                "class_top_priority_weight_by_key": {
                    str(k): float(v) for k, v in localcorr_residue_class_top_weight_by_key.items()
                },
                "class_logft_weight_by_key": {
                    str(k): float(v) for k, v in localcorr_residue_class_logft_weight_by_key.items()
                },
                "logft_rootcause_guard_enabled": bool(args.localcorr_logft_rootcause_guard),
                "logft_count_max_delta_allowed": int(args.localcorr_logft_count_max_delta),
                "residue_logft_count_max_delta_allowed": int(args.localcorr_residue_logft_count_max_delta),
                "require_residue_logft_nonworsening": bool(args.localcorr_require_residue_logft_nonworsening),
                "logft_rootcause_refreeze_enabled": bool(args.localcorr_logft_rootcause_refreeze),
                "logft_rootcause_retune_enabled": bool(args.localcorr_logft_rootcause_retune),
                "logft_rootcause_retune_target_combined_pass_min": int(
                    args.localcorr_logft_rootcause_retune_target_combined_pass_min
                ),
                "logft_rootcause_retune_max_logft_count_delta_allowed": int(
                    args.localcorr_logft_rootcause_retune_max_logft_count_delta
                ),
                "logft_rootcause_retune_max_residue_logft_count_delta_allowed": int(
                    args.localcorr_logft_rootcause_retune_max_residue_logft_count_delta
                ),
                "residue_dual_guard_enabled": bool(args.localcorr_residue_dual_guard),
                "residue_top_priority_max_delta_allowed": int(args.localcorr_residue_top_priority_max_delta),
                "residue_overfit_max_delta_allowed": float(args.localcorr_residue_overfit_max_delta),
                "residue_dual_refreeze_enabled": bool(args.localcorr_residue_dual_refreeze),
                "residue_dual_retune_enabled": bool(args.localcorr_residue_dual_retune),
                "residue_dual_retune_weight_top_priority": float(args.localcorr_residue_dual_retune_top_weight),
                "residue_dual_retune_weight_overfit": float(args.localcorr_residue_dual_retune_overfit_weight),
                "residue_dual_retune_max_top_delta_allowed": int(args.localcorr_residue_dual_retune_max_top_delta),
                "residue_dual_retune_max_overfit_delta_allowed": float(args.localcorr_residue_dual_retune_max_overfit_delta),
                "class_retune_grid_enabled": bool(args.localcorr_residue_class_retune_grid),
                "class_retune_grid_apply_best": bool(args.localcorr_residue_class_retune_apply_best),
                "class_retune_grid_target_class_key": (
                    str(localcorr_residue_class_retune_target_keys_effective[0])
                    if localcorr_residue_class_retune_target_keys_effective
                    else str(args.localcorr_residue_class_retune_target_class_key)
                ),
                "class_retune_grid_target_class_keys": [
                    str(x) for x in localcorr_residue_class_retune_target_keys_effective
                ],
                "class_retune_grid_target_top_k": int(localcorr_residue_class_retune_target_top_k),
                "class_retune_strategy_audit_enabled": bool(args.localcorr_residue_class_retune_strategy_audit),
                "class_retune_strategy_target_top_k_values": [
                    int(v) for v in localcorr_residue_class_retune_target_top_k_values
                ],
                "class_retune_strategy_target_class_keys_groups": [
                    [str(x) for x in grp] for grp in localcorr_residue_class_retune_target_class_keys_groups
                ],
                "class_retune_strategy_definitions_effective": [
                    {
                        "mode": str(d.get("mode", "")),
                        "target_top_k": int(d.get("target_top_k") or 0),
                        "target_class_keys": [str(x) for x in (d.get("target_class_keys") or [])],
                        "label": str(d.get("label", "")),
                    }
                    for d in localcorr_residue_class_retune_strategy_definitions_effective
                    if isinstance(d, dict)
                ],
                "class_retune_grid_logft_count_max_delta_values": [
                    int(v) for v in localcorr_residue_class_retune_logft_count_max_delta_values
                ],
                "class_retune_grid_residue_logft_count_max_delta_values": [
                    int(v) for v in localcorr_residue_class_retune_residue_logft_count_max_delta_values
                ],
                "class_retune_grid_top_weight_values": [
                    float(v) for v in localcorr_residue_class_retune_top_weight_values
                ],
                "class_retune_grid_logft_weight_values": [
                    float(v) for v in localcorr_residue_class_retune_logft_weight_values
                ],
            },
            "candidate_stability_lock": {
                "enabled": bool(args.run_candidate_stability_lock),
                "qbeta_count_max_delta": int(args.stability_qbeta_count_max_delta),
                "qbeta_count_recovery_enabled": bool(args.stability_qbeta_count_recovery),
                "qbeta_count_recovery_logft_p95_max_delta": float(
                    args.stability_qbeta_count_recovery_logft_p95_max_delta
                ),
                "qbeta_count_recovery_logft_count_max_delta": int(
                    args.stability_qbeta_count_recovery_logft_count_max_delta
                ),
                "qbeta_count_recovery_residue_logft_count_max_delta": int(
                    args.stability_qbeta_count_recovery_residue_logft_count_max_delta
                ),
                "qbeta_count_recovery_residue_overfit_max_delta": float(
                    args.stability_qbeta_count_recovery_residue_overfit_max_delta
                ),
            },
            "sigma_floor_mev": float(args.sigma_floor_mev),
            "logft_sigma_proxy_dex": float(args.logft_sigma_proxy),
        },
        "hflavor_mapping": hflavor_mapping,
        "baseline_split_meta": baseline_split_meta,
        "split_meta_pre_local": split_meta,
        "split_meta": split_meta_final,
        "baseline_route_b_evaluation": baseline_route_b_eval,
        "route_b_evaluation_pre_local": route_b_eval_pre_local,
        "route_b_evaluation": route_b_eval,
        "decision_pre_local": decision_pre_local,
        "baseline_comparison": baseline_comparison,
        "outlier_decomposition_pre_local": outlier_decomposition_pre_local,
        "outlier_decomposition": outlier_decomposition,
        "hflavor_sweep": hflavor_sweep,
        "localcorr_sweep": localcorr_sweep,
        "localcorr_residue_reweight": localcorr_residue_reweight,
        "localcorr_residue_reweight_retune": localcorr_residue_reweight_retune,
        "localcorr_residue_reweight_retune_strategy": localcorr_residue_reweight_retune_strategy,
        "candidate_stability_lock": candidate_stability_lock,
        "local_correction": local_correction,
        "residue_robustness": residue_robustness,
        "closure_gate": closure_gate,
        "route_ab_reference": route_ab_ref,
        "decision": decision,
        "outputs": {
            "audit_json": _rel(out_json),
            "audit_csv": _rel(out_csv),
            "audit_png": _rel(out_png),
            "outlier_decomposition_csv": _rel(out_decomp_csv),
            "outlier_decomposition_png": _rel(out_decomp_png),
            "hflavor_sweep_csv": _rel(out_sweep_csv) if bool(args.run_hflavor_sweep) and args.hflavor_mode != "baseline" else None,
            "hflavor_sweep_pareto_png": _rel(out_sweep_png) if bool(args.run_hflavor_sweep) and args.hflavor_mode != "baseline" else None,
            "localcorr_sweep_csv": _rel(out_localcorr_csv) if bool(args.run_localcorr_sweep) and args.hflavor_mode != "baseline" else None,
            "localcorr_sweep_pareto_png": _rel(out_localcorr_png) if bool(args.run_localcorr_sweep) and args.hflavor_mode != "baseline" else None,
            "localcorr_residue_reweight_csv": _rel(out_localcorr_reweight_csv)
            if bool(args.run_localcorr_sweep) and args.hflavor_mode != "baseline" and bool(args.localcorr_residue_reweight)
            else None,
            "localcorr_residue_reweight_retune_csv": _rel(out_localcorr_reweight_retune_csv)
            if bool(args.run_localcorr_sweep) and args.hflavor_mode != "baseline" and bool(args.localcorr_residue_reweight)
            else None,
            "localcorr_residue_reweight_retune_strategy_csv": _rel(out_localcorr_reweight_retune_strategy_csv)
            if bool(args.run_localcorr_sweep) and args.hflavor_mode != "baseline" and bool(args.localcorr_residue_reweight)
            else None,
            "candidate_stability_lock_csv": _rel(out_stability_lock_csv)
            if (
                bool(args.run_localcorr_sweep)
                and args.hflavor_mode != "baseline"
                and bool(args.run_candidate_stability_lock)
            )
            else None,
            "residue_robustness_csv": _rel(out_residue_csv)
            if bool(args.run_residue_robustness) and str(local_correction.get("status")) == "applied"
            else None,
            "residue_robustness_png": _rel(out_residue_png)
            if bool(args.run_residue_robustness) and str(local_correction.get("status")) == "applied"
            else None,
        },
        "notes": [
            "B-route standalone gate uses same I/F (Qβ/logft/CKM/PMNS) while removing Route-A pass dependency.",
            "This pack is a decision baseline for Step 8.7.31; transition target is B_standalone_pass.",
            "Step 8.7.31.2 adds minimal H_flavor(P) remapping (sign/saturation/branch) while keeping dof_equalized_k2.",
            "Step 8.7.31.3 adds outlier decomposition by nuclide/channel/sigma-band and reduction-priority ordering.",
            "Step 8.7.31.4 adds holdout-fixed H_flavor(P) grid sweep and Pareto-front freeze for p95|z_Qβ| vs overfit_gap_qbeta.",
            "Step 8.7.31.5 adds targeted local correction sweep for top transition classes and applies selected candidate for rerun.",
            "Step 8.7.31.6 adds logft-constrained local-correction candidate selection (delta/abs bound with fallback policy).",
            "Step 8.7.31.7 adds q-fail non-regression guard (global/top-priority) on top of logft constraint for candidate freeze.",
            "Step 8.7.31.8 adds holdout-residue robustness audit for constrained candidate stability.",
            "Step 8.7.31.9 adds residue-cross multi-objective reweighting to suppress top-priority/logft watch drift.",
            "Step 8.7.31.10 adds candidate stability lock to freeze all-gate stable candidates or lock reject root-cause.",
            "Step 8.7.31.11 adds qbeta/overfit non-worsening guard into residue reweight and re-freezes candidate under guarded mode.",
            "Step 8.7.31.12 adds minimal logft root-cause guard (count/nonworsening) into residue reweight while keeping qbeta/overfit non-worsening.",
            "Step 8.7.31.13 adds conservative root-cause threshold refreeze when strict logft root guard has no pass and re-freezes candidate band.",
            "Step 8.7.31.14 expands local-correction target coverage (top transition classes >2) under the same guarded/refrozen policy.",
            "Step 8.7.31.15 adds dual residue guard (top-priority q-fail + overfit) with conservative threshold refreeze and expanded reweight candidate pool.",
            "Step 8.7.31.16 adds weighted dual-retune on top-priority/overfit residue deltas to recover combined-guard feasibility when strict dual guard remains empty.",
            "Step 8.7.31.17 adds conservative logft root-cause retune from qbeta+dual feasible band to raise combined-guard coverage while capping logft-count drift.",
            "Step 8.7.31.18 updates stability-lock ranking to prioritize hard-gate metrics among stable candidates, preserving combined-guard coverage while improving selected post-local Qβ/overfit/logft deltas.",
            "Step 8.7.31.19 adds top-priority transition-class gain-boost sweep and carries the same boost profile through residue reweight/stability lock, so top-class-focused corrections are preserved at final selection.",
            "Step 8.7.31.20 adds sigma_low|mode_inconsistent-specific saturation-scale sweep and carries the same class sat profile through residue reweight/stability lock for targeted hard-gate reduction.",
            "Step 8.7.31.21 adds sigma_low|mode_inconsistent sign/branch-aware blend taper (zref/min-scale sweep) and carries the same blend profile through residue reweight/stability lock for q-fail-control exploration.",
            "Step 8.7.31.22 adds qbeta-count nonworsening gate with conditional logft/residue recovery caps so candidates that recover n_qbeta_gt3 while keeping pre-local Qβ/overfit nonworsening can be stably selected.",
            "Step 8.7.31.23 promotes balanced recovery selection: keep delta_n_qbeta<=0 while also requiring nonworsening deltas for Qβ p95 / logft p95 / overfit under expanded top-class targeting.",
            "Step 8.7.31.24 fixes narrow-band (sat/zref/min-blend) retune completion with residue-dual/root cycles closed (including not_needed) and keeps balanced nonworsening under high-density localcorr grid.",
            "Step 8.7.31.25 adds sigma_low|mode_inconsistent sign-blend-scale sweep and freezes candidate under same nonworsening gates with expanded high-density localcorr grid.",
            "Step 8.7.31.26 enforces sign_blend>0 and sign_blend_scale<1 targeted region and re-freezes candidate under the same guarded nonworsening gates.",
            "Step 8.7.31.27 adds class-specific 2-axis retune grid (logft-count thresholds x class top/logft residue weights) and can apply the best guarded combination before stability lock.",
            "Step 8.7.31.28 extends class-retune to simultaneous multi-class targets (top-K) and prioritizes watch3 simultaneous-satisfaction density in retune ranking.",
            "Step 8.7.31.29 adds target-selection strategy audit (top-k vs explicit class groups) over narrow class-threshold retune bands and can apply the best strategy before stability lock.",
            "Step 8.7.31.30 keeps the strategy-audit interface and increases localcorr density (grid/max-candidates) in stages to re-check watch3 simultaneous-satisfaction existence.",
            "Step 8.7.31.31 keeps the same strategy-audit interface and raises stage-2 density (max_candidates>=48, n_total_candidates>=48) to re-check watch3 simultaneous-satisfaction existence.",
            "Step 8.7.31.32 keeps the same strategy-audit interface and adds class-key-specific threshold guard overrides for root-cause/dual constraints to re-check watch3 simultaneous-satisfaction existence.",
            "Step 8.7.31.33 keeps the same strategy-audit I/F and re-identifies class-key threshold bands + strategy groups (broadened key-set/top-k) to re-check combined_guard_pass recovery.",
        ],
    }

    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _plot(route_b_eval=route_b_eval, decision=decision, closure_gate=closure_gate, out_png=out_png)

    try:
        worklog.append_event(
            {
                "event_type": "quantum_weak_interaction_beta_decay_route_b_standalone_audit",
                "phase": step_id,
                "inputs": payload.get("inputs"),
                "outputs": payload.get("outputs"),
                "decision": payload.get("decision"),
            }
        )
    except Exception:
        pass

    print("[ok] wrote:")
    print(f"  {_rel(out_json)}")
    print(f"  {_rel(out_csv)}")
    print(f"  {_rel(out_png)}")
    print(f"  {_rel(out_decomp_csv)}")
    print(f"  {_rel(out_decomp_png)}")
    if bool(args.run_hflavor_sweep) and args.hflavor_mode != "baseline":
        print(f"  {_rel(out_sweep_csv)}")
        print(f"  {_rel(out_sweep_png)}")
    if bool(args.run_localcorr_sweep) and args.hflavor_mode != "baseline":
        print(f"  {_rel(out_localcorr_csv)}")
        if out_localcorr_png.exists():
            print(f"  {_rel(out_localcorr_png)}")
        if bool(args.localcorr_residue_reweight) and out_localcorr_reweight_csv.exists():
            print(f"  {_rel(out_localcorr_reweight_csv)}")
        if bool(args.localcorr_residue_reweight) and out_localcorr_reweight_retune_csv.exists():
            print(f"  {_rel(out_localcorr_reweight_retune_csv)}")
        if bool(args.localcorr_residue_reweight) and out_localcorr_reweight_retune_strategy_csv.exists():
            print(f"  {_rel(out_localcorr_reweight_retune_strategy_csv)}")
        if bool(args.run_candidate_stability_lock) and out_stability_lock_csv.exists():
            print(f"  {_rel(out_stability_lock_csv)}")
    if bool(args.run_residue_robustness) and str(residue_robustness.get("status")) == "completed":
        print(f"  {_rel(out_residue_csv)}")
        if out_residue_png.exists():
            print(f"  {_rel(out_residue_png)}")
    print(f"[mode] hflavor_mode={args.hflavor_mode} step={step_id}")
    print(
        "[summary] transition={0} status={1} hard_fail={2} watch_fail={3}".format(
            decision.get("transition"),
            decision.get("overall_status"),
            len(decision.get("hard_fail_ids") or []),
            len(decision.get("watch_fail_ids") or []),
        )
    )
    top_priority = (
        outlier_decomposition.get("reduction_priority_order")[0]
        if isinstance(outlier_decomposition.get("reduction_priority_order"), list)
        and outlier_decomposition.get("reduction_priority_order")
        else {}
    )
    print(
        "[priority] top_class={0} score={1:.6g} q_gt3={2} logft_gt3={3}".format(
            str(top_priority.get("group_key", "n/a")),
            _to_float(top_priority.get("priority_score")),
            int(top_priority.get("n_qbeta_gt3", 0)),
            int(top_priority.get("n_logft_gt3", 0)),
        )
    )
    if bool(args.run_hflavor_sweep) and args.hflavor_mode != "baseline":
        sweep_recommended = (
            hflavor_sweep.get("recommended_candidate") if isinstance(hflavor_sweep.get("recommended_candidate"), dict) else {}
        )
        print(
            "[pareto] best={0} p95_q={1:.6g} gap_q={2:.6g} n_front={3}".format(
                str(sweep_recommended.get("candidate_id", "n/a")),
                _to_float(sweep_recommended.get("holdout_p95_abs_z_qbeta")),
                _to_float(sweep_recommended.get("overfit_gap_qbeta")),
                len(hflavor_sweep.get("pareto_front") or []),
            )
        )
    if bool(args.run_localcorr_sweep) and args.hflavor_mode != "baseline":
        local_rec = (
            localcorr_sweep.get("recommended_candidate")
            if isinstance(localcorr_sweep.get("recommended_candidate"), dict)
            else {}
        )
        policy = localcorr_sweep.get("selection_policy") if isinstance(localcorr_sweep.get("selection_policy"), dict) else {}
        print(
            "[localcorr] best={0} p95_q={1:.6g} gap_q={2:.6g} n_front={3}".format(
                str(local_rec.get("candidate_id", "n/a")),
                _to_float(local_rec.get("holdout_p95_abs_z_qbeta")),
                _to_float(local_rec.get("overfit_gap_qbeta")),
                len(localcorr_sweep.get("pareto_front") or []),
            )
        )
        print(
            "[localcorr-policy] mode={0} logft_pass={1}/{2} all_pass={3} qfail_guard={4} delta_max={5:.6g}".format(
                str(policy.get("mode", "n/a")),
                int(policy.get("n_candidates_logft_pass") or 0),
                int(policy.get("n_candidates_total") or 0),
                int(policy.get("n_candidates_all_constraints_pass") or 0),
                bool(policy.get("qfail_guard_enabled")),
                _to_float(policy.get("logft_delta_max_allowed")),
            )
        )
        rw = localcorr_sweep.get("residue_reweight") if isinstance(localcorr_sweep.get("residue_reweight"), dict) else {}
        rw_policy = rw.get("selection_policy") if isinstance(rw.get("selection_policy"), dict) else {}
        rw_rec = rw.get("recommended_candidate") if isinstance(rw.get("recommended_candidate"), dict) else {}
        rw_retune = (
            localcorr_sweep.get("residue_reweight_retune")
            if isinstance(localcorr_sweep.get("residue_reweight_retune"), dict)
            else {}
        )
        rw_retune_rec = rw_retune.get("recommended_candidate") if isinstance(rw_retune.get("recommended_candidate"), dict) else {}
        rw_retune_target_keys = (
            rw_retune.get("target_class_keys")
            if isinstance(rw_retune.get("target_class_keys"), list)
            else []
        )
        rw_retune_target_label = (
            ",".join([str(x) for x in rw_retune_target_keys if str(x).strip()])
            if rw_retune_target_keys
            else str((rw_retune.get("target_class_key") or "n/a"))
        )
        if bool(args.localcorr_residue_reweight):
            print(
                "[localcorr-reweight] status={0} mode={1} n_eval={2} q_guard={3}/{4} root_guard={5}/{4} dual_guard={6}/{4} comb_guard={7}/{4} best={8} score={9:.6g}".format(
                    str(rw.get("status", "not_run")),
                    str(rw_policy.get("mode", "not_run")),
                    int(rw_policy.get("n_candidates_evaluated") or 0),
                    int(rw_policy.get("n_candidates_guard_pass") or 0),
                    int(rw_policy.get("n_candidates_evaluated") or 0),
                    int(rw_policy.get("n_candidates_root_guard_pass") or 0),
                    int(rw_policy.get("n_candidates_residue_dual_guard_pass") or 0),
                    int(rw_policy.get("n_candidates_combined_guard_pass") or 0),
                    str(rw_rec.get("candidate_id", "n/a")),
                    _to_float(rw_rec.get("residue_reweight_score")),
                )
            )
            print(
                "[localcorr-refreeze] enabled={0} applied={1} source={2} frozen_logft_dN={3} frozen_residue_dN={4} require_nonworse={5} root_retune_enabled={6} root_retune_applied={7} root_retune_source={8} root_retune_anchor={9} root_target={10} root_caps=({11},{12}) dual_enabled={13} dual_applied={14} dual_source={15} frozen_top_dN={16} frozen_overfit_d={17:.6g} dual_retune_enabled={18} dual_retune_applied={19} dual_retune_source={20} dual_retune_anchor={21} dual_retune_w=({22:.3g},{23:.3g})".format(
                    bool(rw_policy.get("logft_rootcause_refreeze_enabled")),
                    bool(rw_policy.get("logft_rootcause_refreeze_applied")),
                    str(rw_policy.get("logft_rootcause_refreeze_source", "n/a")),
                    int(rw_policy.get("refrozen_logft_count_max_delta_allowed") or 0),
                    int(rw_policy.get("refrozen_residue_logft_count_max_delta_allowed") or 0),
                    bool(rw_policy.get("refrozen_require_residue_logft_nonworsening")),
                    bool(rw_policy.get("logft_rootcause_retune_enabled")),
                    bool(rw_policy.get("logft_rootcause_retune_applied")),
                    str(rw_policy.get("logft_rootcause_retune_source", "n/a")),
                    str(rw_policy.get("logft_rootcause_retune_anchor_candidate", "n/a")),
                    int(rw_policy.get("logft_rootcause_retune_target_combined_pass_min") or 0),
                    int(rw_policy.get("logft_rootcause_retune_max_logft_count_delta_allowed") or 0),
                    int(rw_policy.get("logft_rootcause_retune_max_residue_logft_count_delta_allowed") or 0),
                    bool(rw_policy.get("residue_dual_refreeze_enabled")),
                    bool(rw_policy.get("residue_dual_refreeze_applied")),
                    str(rw_policy.get("residue_dual_refreeze_source", "n/a")),
                    int(rw_policy.get("refrozen_residue_top_priority_max_delta_allowed") or 0),
                    _to_float(rw_policy.get("refrozen_residue_overfit_max_delta_allowed")),
                    bool(rw_policy.get("residue_dual_retune_enabled")),
                    bool(rw_policy.get("residue_dual_retune_applied")),
                    str(rw_policy.get("residue_dual_retune_source", "n/a")),
                    str(rw_policy.get("residue_dual_retune_anchor_candidate", "n/a")),
                    _to_float(rw_policy.get("residue_dual_retune_weight_top_priority")),
                    _to_float(rw_policy.get("residue_dual_retune_weight_overfit")),
                )
            )
            print(
                "[localcorr-retune] enabled={0} status={1} applied={2} target={3} target_n={4} watch3_any={5} watch3_ratio={6:.6g} comb_guard={7} stable={8}".format(
                    bool(args.localcorr_residue_class_retune_grid),
                    str(rw_retune.get("status", "not_run")),
                    bool(rw_retune.get("applied_best")),
                    str(rw_retune_target_label),
                    int(rw_retune_rec.get("n_target_classes") or len(rw_retune_target_keys)),
                    int(rw_retune_rec.get("watch3_all_true_candidates_n") or 0),
                    _to_float(rw_retune_rec.get("watch3_all_true_candidates_ratio")),
                    int(rw_retune_rec.get("n_candidates_combined_guard_pass") or 0),
                    int(rw_retune_rec.get("stable_candidates_n") or 0),
                )
            )
            rw_retune_strategy = (
                localcorr_sweep.get("residue_reweight_retune_strategy")
                if isinstance(localcorr_sweep.get("residue_reweight_retune_strategy"), dict)
                else {}
            )
            rw_retune_strategy_rec = (
                rw_retune_strategy.get("recommended_strategy")
                if isinstance(rw_retune_strategy.get("recommended_strategy"), dict)
                else {}
            )
            print(
                "[localcorr-retune-strategy] enabled={0} status={1} applied={2} n_strategy={3} best={4} target={5} watch3_any={6} watch3_ratio={7:.6g} comb_guard={8} stable={9}".format(
                    bool(args.localcorr_residue_class_retune_strategy_audit),
                    str(rw_retune_strategy.get("status", "not_run")),
                    bool(rw_retune_strategy.get("applied_best")),
                    int(rw_retune_strategy.get("strategy_count") or 0),
                    str(rw_retune_strategy_rec.get("strategy_label", "n/a")),
                    str(rw_retune_strategy_rec.get("target_class_keys", "n/a")),
                    int(rw_retune_strategy_rec.get("watch3_all_true_candidates_n") or 0),
                    _to_float(rw_retune_strategy_rec.get("watch3_all_true_candidates_ratio")),
                    int(rw_retune_strategy_rec.get("n_candidates_combined_guard_pass") or 0),
                    int(rw_retune_strategy_rec.get("stable_candidates_n") or 0),
                )
            )
        lock = (
            localcorr_sweep.get("candidate_stability_lock")
            if isinstance(localcorr_sweep.get("candidate_stability_lock"), dict)
            else {}
        )
        lock_policy = lock.get("selection_policy") if isinstance(lock.get("selection_policy"), dict) else {}
        lock_rec = lock.get("recommended_candidate") if isinstance(lock.get("recommended_candidate"), dict) else {}
        if bool(args.run_candidate_stability_lock):
            print(
                "[stability-lock] status={0} mode={1} stable={2}/{3} best={4} viol_score={5:.6g}".format(
                    str(lock.get("status", "not_run")),
                    str(lock.get("mode", lock_policy.get("mode", "not_run"))),
                    int(lock.get("stable_candidates_n") or 0),
                    int(lock.get("evaluated_candidates_n") or 0),
                    str(lock_rec.get("candidate_id", "n/a")),
                    _to_float(lock_rec.get("stability_violation_score")),
                )
            )
    if bool(args.run_residue_robustness) and str(residue_robustness.get("status")) == "completed":
        rsum = residue_robustness.get("summary") if isinstance(residue_robustness.get("summary"), dict) else {}
        print(
            "[residue] n={0} ref={1} qfail_nonreg={2} top_nonreg={3} logft_nonworse={4} logft_count_nonreg={5}".format(
                int(rsum.get("n_residues") or 0),
                int(rsum.get("reference_residue") or 0),
                bool(rsum.get("all_qfail_nonregression_vs_ref")),
                bool(rsum.get("all_top_priority_nonregression_vs_ref")),
                bool(rsum.get("all_logft_nonworsening_vs_ref")),
                bool(rsum.get("all_logft_count_nonregression_vs_ref")),
            )
        )


if __name__ == "__main__":
    main()
