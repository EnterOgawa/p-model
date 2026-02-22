from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.summary import worklog


@dataclass(frozen=True)
class RouteDef:
    route_id: str
    label: str
    q_column: str
    description: str


ROUTES = (
    RouteDef(
        route_id="A_transfer_surrogate",
        label="A: EW transfer surrogate",
        q_column="q_obs_from_binding_MeV",
        description="Observed-binding transfer baseline (operational surrogate for Route A).",
    ),
    RouteDef(
        route_id="B_pmodel_proxy",
        label="B: P-model proxy",
        q_column="q_pred_after_MeV",
        description="P-model residual channel from frozen nuclear mapping (first quantitative proxy).",
    ),
)

CHANNELS = ("all", "beta_minus", "beta_plus")


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _parse_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


def _parse_mode_channel_flags(observed_modes: list[dict[str, Any]]) -> tuple[bool, bool, float, float]:
    has_beta_minus = False
    has_beta_plus = False
    branch_beta_minus = 0.0
    branch_beta_plus = 0.0
    for mode_entry in observed_modes:
        if not isinstance(mode_entry, dict):
            continue
        mode_text = str(mode_entry.get("mode", "")).strip().upper()
        value = _parse_float(mode_entry.get("value"))
        if "B-" in mode_text:
            has_beta_minus = True
            if math.isfinite(value):
                branch_beta_minus += float(value)
        if ("EC" in mode_text) or ("B+" in mode_text):
            has_beta_plus = True
            if math.isfinite(value):
                branch_beta_plus += float(value)
    return has_beta_minus, has_beta_plus, float(branch_beta_minus), float(branch_beta_plus)


def _load_nudat_primary_modes(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for key, item in payload.items():
        if not isinstance(item, dict):
            continue
        levels = item.get("levels")
        level0 = levels[0] if isinstance(levels, list) and levels and isinstance(levels[0], dict) else {}
        decay_modes = level0.get("decayModes") if isinstance(level0, dict) else None
        observed = decay_modes.get("observed") if isinstance(decay_modes, dict) else []
        observed = observed if isinstance(observed, list) else []
        has_bm, has_bp, branch_bm, branch_bp = _parse_mode_channel_flags(observed)
        mode_texts = [str(v.get("mode", "")).strip() for v in observed if isinstance(v, dict) and str(v.get("mode", "")).strip()]
        out[str(key)] = {
            "observed_modes": mode_texts,
            "has_beta_minus_mode": has_bm,
            "has_beta_plus_mode": has_bp,
            "branch_beta_minus_percent": branch_bm if branch_bm > 0.0 else float("nan"),
            "branch_beta_plus_percent": branch_bp if branch_bp > 0.0 else float("nan"),
        }
    return out


def _sha256(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_bytes)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _file_signature(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"exists": False, "path": None}
    payload: dict[str, Any] = {"exists": bool(path.exists()), "path": _rel(path)}
    if not path.exists():
        return payload
    try:
        stat = path.stat()
        payload["size_bytes"] = int(stat.st_size)
        payload["mtime_utc"] = datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat()
    except Exception:
        pass
    try:
        payload["sha256"] = str(_sha256(path)).strip().upper()
    except Exception:
        payload["sha256"] = None
    return payload


def _load_previous_ckm_watchpack(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    diagnostics = payload.get("diagnostics")
    if not isinstance(diagnostics, dict):
        return {}
    watchpack = diagnostics.get("ckm_primary_update_watchpack")
    if not isinstance(watchpack, dict):
        return {}
    return watchpack


def _derive_ckm_primary_update_watchpack(
    *,
    current_input_signature: dict[str, Any],
    previous_watchpack: dict[str, Any],
    closure_gate: dict[str, Any],
) -> dict[str, Any]:
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
    curr_size = _parse_float(current_input_signature.get("size_bytes"))
    prev_size = _parse_float(previous_signature.get("size_bytes"))
    metadata_changed_without_hash_change = False
    if curr_exists and prev_exists and (not hash_changed):
        if (curr_mtime and prev_mtime and curr_mtime != prev_mtime) or (
            math.isfinite(curr_size) and math.isfinite(prev_size) and curr_size != prev_size
        ):
            metadata_changed_without_hash_change = True

    baseline_initialized_now = curr_exists and (not prev_exists)
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

    closure_status = str(closure_gate.get("status") or "not_evaluated")
    watch_locked = bool(closure_gate.get("watch_locked_by_current_primary_source"))
    if update_event_detected:
        next_action = "run_ckm_pmns_closure_recheck_now"
    elif closure_status == "watch" and watch_locked:
        next_action = "wait_for_ckm_primary_input_hash_change_then_rerun_step_8_7_22"
    elif closure_status == "watch":
        next_action = "keep_watch_and_rerun_on_input_update"
    else:
        next_action = "none"

    return {
        "closure_status": closure_status,
        "watch_locked_by_current_primary_source": watch_locked,
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
            "CKM+PMNS closure rerun trigger is CKM primary-input hash change. "
            "Metadata-only changes are logged and do not increment the event counter."
        ),
    }


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    s = sorted(float(v) for v in values)
    idx = int(round((len(s) - 1) * p))
    idx = max(0, min(len(s) - 1, idx))
    return float(s[idx])


def _median(values: list[float]) -> float:
    if not values:
        return float("nan")
    s = sorted(float(v) for v in values)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return float(s[mid])
    return float((s[mid - 1] + s[mid]) / 2.0)


def _safe_max(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(max(values))


def _safe_frac_le(values: list[float], threshold: float) -> float:
    if not values:
        return float("nan")
    return float(sum(1 for v in values if float(v) <= threshold) / float(len(values)))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        if not rows:
            f.write("")
            return
        headers = list(rows[0].keys())
        writer = csv.writer(f)
        writer.writerow(headers)
        for row in rows:
            writer.writerow([row.get(h) for h in headers])


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


def _route_channel_summary(
    *,
    rows: list[dict[str, Any]],
    route: RouteDef,
    channel: str,
    logft_sigma_proxy: float,
    z_gate: float,
) -> dict[str, Any]:
    use_rows = rows if channel == "all" else [r for r in rows if str(r.get("channel", "")) == channel]

    abs_z_q: list[float] = []
    abs_z_logft: list[float] = []
    n_q_rows = 0
    n_logft_rows = 0
    n_gate_q_fail = 0
    n_gate_logft_fail = 0

    for row in use_rows:
        q_obs = _parse_float(row.get("q_obs_MeV"))
        q_sigma = _parse_float(row.get("q_obs_sigma_MeV"))
        q_pred = _parse_float(row.get(route.q_column))
        if math.isfinite(q_obs) and math.isfinite(q_sigma) and q_sigma > 0.0 and math.isfinite(q_pred):
            z_q = abs((q_pred - q_obs) / q_sigma)
            abs_z_q.append(float(z_q))
            n_q_rows += 1
            if z_q > z_gate:
                n_gate_q_fail += 1

        if math.isfinite(q_obs) and math.isfinite(q_pred) and q_obs > 0.0 and q_pred > 0.0 and logft_sigma_proxy > 0.0:
            delta_logft_proxy = 5.0 * (math.log10(q_pred) - math.log10(q_obs))
            z_logft = abs(delta_logft_proxy / logft_sigma_proxy)
            abs_z_logft.append(float(z_logft))
            n_logft_rows += 1
            if z_logft > z_gate:
                n_gate_logft_fail += 1

    p95_q = _percentile(abs_z_q, 0.95)
    p95_logft = _percentile(abs_z_logft, 0.95)
    max_q = _safe_max(abs_z_q)
    max_logft = _safe_max(abs_z_logft)
    hard_pass_q = bool(math.isfinite(p95_q) and p95_q <= z_gate)
    hard_pass_logft = bool(math.isfinite(p95_logft) and p95_logft <= z_gate)
    watch_pass_q = bool(math.isfinite(max_q) and max_q <= z_gate)
    watch_pass_logft = bool(math.isfinite(max_logft) and max_logft <= z_gate)

    return {
        "route_id": route.route_id,
        "route_label": route.label,
        "channel": channel,
        "n_rows": len(use_rows),
        "n_q_rows": n_q_rows,
        "n_logft_proxy_rows": n_logft_rows,
        "max_abs_z_qbeta": max_q,
        "p95_abs_z_qbeta": p95_q,
        "median_abs_z_qbeta": _median(abs_z_q),
        "frac_abs_z_qbeta_le3": _safe_frac_le(abs_z_q, z_gate),
        "n_qbeta_gt3": n_gate_q_fail,
        "max_abs_z_logft_proxy": max_logft,
        "p95_abs_z_logft_proxy": p95_logft,
        "median_abs_z_logft_proxy": _median(abs_z_logft),
        "frac_abs_z_logft_proxy_le3": _safe_frac_le(abs_z_logft, z_gate),
        "n_logft_proxy_gt3": n_gate_logft_fail,
        "hard_pass_qbeta": hard_pass_q,
        "hard_pass_logft_proxy": hard_pass_logft,
        "watch_pass_qbeta": watch_pass_q,
        "watch_pass_logft_proxy": watch_pass_logft,
        "hard_pass": bool(hard_pass_q and hard_pass_logft),
        "watch_pass": bool(watch_pass_q and watch_pass_logft),
    }


def _decision(all_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_route = {str(r["route_id"]): r for r in all_rows if str(r.get("channel")) == "all"}
    route_a = by_route.get("A_transfer_surrogate")
    route_b = by_route.get("B_pmodel_proxy")

    route_a_hard = bool(route_a and bool(route_a.get("hard_pass")))
    route_b_hard = bool(route_b and bool(route_b.get("hard_pass")))

    if route_a_hard and (not route_b_hard):
        transition = "A_stay_B_reject"
    elif route_a_hard and route_b_hard:
        transition = "A_and_B_supported"
    elif (not route_a_hard) and route_b_hard:
        transition = "A_reject_B_continue"
    else:
        transition = "A_and_B_reject"

    return {
        "route_a_hard_pass": route_a_hard,
        "route_b_hard_pass": route_b_hard,
        "route_a_watch_pass": bool(route_a and bool(route_a.get("watch_pass"))),
        "route_b_watch_pass": bool(route_b and bool(route_b.get("watch_pass"))),
        "transition": transition,
    }


def _deterministic_holdout_flag(*, nuclide_key: str, modulo: int, residue: int) -> bool:
    if modulo <= 1:
        return False
    import hashlib

    digest = hashlib.sha256(str(nuclide_key).encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % int(modulo)
    return bool(bucket == int(residue))


def _split_rows_by_holdout(
    *,
    rows: list[dict[str, Any]],
    modulo: int,
    residue: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    train_rows: list[dict[str, Any]] = []
    holdout_rows: list[dict[str, Any]] = []
    nuclide_bucket: dict[str, bool] = {}
    for row in rows:
        nuclide_key = str(row.get("nuclide_key", "")).strip().lower()
        if not nuclide_key:
            nuclide_key = f"fallback::{str(row.get('nuclide_name', '')).strip().lower()}"
        if nuclide_key not in nuclide_bucket:
            nuclide_bucket[nuclide_key] = _deterministic_holdout_flag(
                nuclide_key=nuclide_key,
                modulo=modulo,
                residue=residue,
            )
        if nuclide_bucket[nuclide_key]:
            holdout_rows.append(row)
        else:
            train_rows.append(row)

    unique_holdout = {str(r.get("nuclide_key", "")).strip().lower() for r in holdout_rows}
    unique_train = {str(r.get("nuclide_key", "")).strip().lower() for r in train_rows}
    split_meta = {
        "method": "deterministic_hash_modulo_on_nuclide_key",
        "modulo": int(modulo),
        "residue": int(residue),
        "rows_total": len(rows),
        "rows_train": len(train_rows),
        "rows_holdout": len(holdout_rows),
        "nuclides_train": len(unique_train),
        "nuclides_holdout": len(unique_holdout),
        "holdout_fraction_rows": (float(len(holdout_rows)) / float(len(rows))) if rows else float("nan"),
    }
    return train_rows, holdout_rows, split_meta


def _fit_weighted_linear_q(
    *,
    rows: list[dict[str, Any]],
    route: RouteDef,
    sigma_floor: float,
) -> dict[str, Any]:
    sum_w = 0.0
    sum_wx = 0.0
    sum_wy = 0.0
    sum_wxx = 0.0
    sum_wxy = 0.0
    n_fit = 0
    for row in rows:
        q_obs = _parse_float(row.get("q_obs_MeV"))
        q_sigma = _parse_float(row.get("q_obs_sigma_MeV"))
        q_raw = _parse_float(row.get(route.q_column))
        if not (math.isfinite(q_obs) and math.isfinite(q_sigma) and q_sigma > 0.0 and math.isfinite(q_raw)):
            continue
        sigma_eff = max(float(q_sigma), float(sigma_floor))
        w = 1.0 / (sigma_eff * sigma_eff)
        sum_w += w
        sum_wx += w * q_raw
        sum_wy += w * q_obs
        sum_wxx += w * q_raw * q_raw
        sum_wxy += w * q_raw * q_obs
        n_fit += 1
    denom = (sum_w * sum_wxx) - (sum_wx * sum_wx)
    if n_fit < 3 or (not math.isfinite(denom)) or abs(denom) < 1.0e-24:
        return {
            "status": "identity_fallback",
            "a_offset_MeV": 0.0,
            "b_scale": 1.0,
            "n_fit_rows": n_fit,
            "k_params": 2,
            "sigma_floor_MeV": float(sigma_floor),
            "reason": "insufficient_rows_or_singular_fit",
        }
    b_scale = ((sum_w * sum_wxy) - (sum_wx * sum_wy)) / denom
    a_offset = (sum_wy - b_scale * sum_wx) / sum_w
    return {
        "status": "ok",
        "a_offset_MeV": float(a_offset),
        "b_scale": float(b_scale),
        "n_fit_rows": n_fit,
        "k_params": 2,
        "sigma_floor_MeV": float(sigma_floor),
    }


def _route_metrics_calibrated(
    *,
    rows: list[dict[str, Any]],
    route: RouteDef,
    channel: str,
    logft_sigma_proxy: float,
    z_gate: float,
    calibration: dict[str, Any],
) -> dict[str, Any]:
    use_rows = rows if channel == "all" else [r for r in rows if str(r.get("channel", "")) == channel]
    a_offset = _parse_float(calibration.get("a_offset_MeV"))
    b_scale = _parse_float(calibration.get("b_scale"))
    if not math.isfinite(a_offset):
        a_offset = 0.0
    if not math.isfinite(b_scale):
        b_scale = 1.0

    abs_z_q: list[float] = []
    abs_z_logft: list[float] = []
    chi2_q = 0.0
    chi2_logft = 0.0
    n_q_rows = 0
    n_logft_rows = 0
    n_gate_q_fail = 0
    n_gate_logft_fail = 0
    for row in use_rows:
        q_obs = _parse_float(row.get("q_obs_MeV"))
        q_sigma = _parse_float(row.get("q_obs_sigma_MeV"))
        q_raw = _parse_float(row.get(route.q_column))
        if math.isfinite(q_obs) and math.isfinite(q_sigma) and q_sigma > 0.0 and math.isfinite(q_raw):
            q_pred = float(a_offset + b_scale * q_raw)
            z_q = abs((q_pred - q_obs) / q_sigma)
            abs_z_q.append(float(z_q))
            chi2_q += float(z_q * z_q)
            n_q_rows += 1
            if z_q > z_gate:
                n_gate_q_fail += 1
            if math.isfinite(q_obs) and q_obs > 0.0 and q_pred > 0.0 and logft_sigma_proxy > 0.0:
                delta_logft_proxy = 5.0 * (math.log10(q_pred) - math.log10(q_obs))
                z_logft = abs(delta_logft_proxy / logft_sigma_proxy)
                abs_z_logft.append(float(z_logft))
                chi2_logft += float(z_logft * z_logft)
                n_logft_rows += 1
                if z_logft > z_gate:
                    n_gate_logft_fail += 1

    p95_q = _percentile(abs_z_q, 0.95)
    p95_logft = _percentile(abs_z_logft, 0.95)
    max_q = _safe_max(abs_z_q)
    max_logft = _safe_max(abs_z_logft)
    hard_pass_q = bool(math.isfinite(p95_q) and p95_q <= z_gate)
    hard_pass_logft = bool(math.isfinite(p95_logft) and p95_logft <= z_gate)
    watch_pass_q = bool(math.isfinite(max_q) and max_q <= z_gate)
    watch_pass_logft = bool(math.isfinite(max_logft) and max_logft <= z_gate)
    return {
        "route_id": route.route_id,
        "route_label": route.label,
        "channel": channel,
        "n_rows": len(use_rows),
        "n_q_rows": n_q_rows,
        "n_logft_proxy_rows": n_logft_rows,
        "max_abs_z_qbeta": max_q,
        "p95_abs_z_qbeta": p95_q,
        "median_abs_z_qbeta": _median(abs_z_q),
        "frac_abs_z_qbeta_le3": _safe_frac_le(abs_z_q, z_gate),
        "n_qbeta_gt3": n_gate_q_fail,
        "chi2_qbeta": float(chi2_q) if n_q_rows > 0 else float("nan"),
        "max_abs_z_logft_proxy": max_logft,
        "p95_abs_z_logft_proxy": p95_logft,
        "median_abs_z_logft_proxy": _median(abs_z_logft),
        "frac_abs_z_logft_proxy_le3": _safe_frac_le(abs_z_logft, z_gate),
        "n_logft_proxy_gt3": n_gate_logft_fail,
        "chi2_logft_proxy": float(chi2_logft) if n_logft_rows > 0 else float("nan"),
        "hard_pass_qbeta": hard_pass_q,
        "hard_pass_logft_proxy": hard_pass_logft,
        "watch_pass_qbeta": watch_pass_q,
        "watch_pass_logft_proxy": watch_pass_logft,
        "hard_pass": bool(hard_pass_q and hard_pass_logft),
        "watch_pass": bool(watch_pass_q and watch_pass_logft),
        "calibration_a_offset_MeV": float(a_offset),
        "calibration_b_scale": float(b_scale),
    }


def _aic_like(
    *,
    chi2: float,
    n_obs: int,
    k_params: int,
) -> dict[str, Any]:
    if not (math.isfinite(chi2) and n_obs > 0 and k_params >= 0):
        return {"aic_chi2": float("nan"), "aicc_chi2": float("nan"), "bic_chi2": float("nan")}
    aic = float(chi2 + (2.0 * k_params))
    if n_obs > (k_params + 1):
        aicc = float(aic + (2.0 * k_params * (k_params + 1)) / float(n_obs - k_params - 1))
    else:
        aicc = float("inf")
    bic = float(chi2 + k_params * math.log(float(n_obs)))
    return {"aic_chi2": aic, "aicc_chi2": aicc, "bic_chi2": bic}


def _equalized_route_audit(
    *,
    rows: list[dict[str, Any]],
    logft_sigma_proxy: float,
    z_gate: float,
    holdout_hash_modulo: int,
    holdout_hash_residue: int,
    overfit_gap_gate: float,
    sigma_floor_mev: float,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    train_rows, holdout_rows, split_meta = _split_rows_by_holdout(
        rows=rows,
        modulo=holdout_hash_modulo,
        residue=holdout_hash_residue,
    )
    if not holdout_rows:
        holdout_rows = train_rows[-max(1, len(train_rows) // 10) :]
        train_rows = train_rows[: max(1, len(train_rows) - len(holdout_rows))]
        split_meta["fallback"] = "holdout_empty_fallback_last10pct_rows"
        split_meta["rows_train"] = len(train_rows)
        split_meta["rows_holdout"] = len(holdout_rows)
        split_meta["holdout_fraction_rows"] = (float(len(holdout_rows)) / float(len(rows))) if rows else float("nan")

    summary_rows: list[dict[str, Any]] = []
    route_eval: dict[str, Any] = {}
    for route in ROUTES:
        calibration = _fit_weighted_linear_q(
            rows=train_rows,
            route=route,
            sigma_floor=sigma_floor_mev,
        )
        split_channel_metrics: dict[str, dict[str, Any]] = {"train": {}, "holdout": {}}
        for split_name, split_rows in (("train", train_rows), ("holdout", holdout_rows)):
            for channel in CHANNELS:
                row_metrics = _route_metrics_calibrated(
                    rows=split_rows,
                    route=route,
                    channel=channel,
                    logft_sigma_proxy=logft_sigma_proxy,
                    z_gate=z_gate,
                    calibration=calibration,
                )
                split_channel_metrics[split_name][channel] = row_metrics
                summary_rows.append(
                    {
                        "route_id": route.route_id,
                        "route_label": route.label,
                        "split": split_name,
                        **row_metrics,
                    }
                )

        train_all = split_channel_metrics["train"]["all"]
        holdout_all = split_channel_metrics["holdout"]["all"]
        aic_train = _aic_like(
            chi2=_parse_float(train_all.get("chi2_qbeta")),
            n_obs=int(train_all.get("n_q_rows") or 0),
            k_params=int(calibration.get("k_params") or 0),
        )
        overfit_gap_q = _parse_float(holdout_all.get("p95_abs_z_qbeta")) - _parse_float(train_all.get("p95_abs_z_qbeta"))
        overfit_gap_logft = _parse_float(holdout_all.get("p95_abs_z_logft_proxy")) - _parse_float(train_all.get("p95_abs_z_logft_proxy"))
        overfit_guard_pass = bool(
            math.isfinite(overfit_gap_q)
            and math.isfinite(overfit_gap_logft)
            and overfit_gap_q <= overfit_gap_gate
            and overfit_gap_logft <= overfit_gap_gate
        )
        hard_pass = bool(holdout_all.get("hard_pass")) and overfit_guard_pass
        watch_pass = bool(holdout_all.get("watch_pass")) and overfit_guard_pass
        dof_equalized = bool(int(calibration.get("k_params") or 0) == 2)
        route_eval[route.route_id] = {
            "route_id": route.route_id,
            "route_label": route.label,
            "calibration": calibration,
            "dof_equalized": dof_equalized,
            "dof_penalty": {
                "k_params": int(calibration.get("k_params") or 0),
                "n_train_q_rows": int(train_all.get("n_q_rows") or 0),
                "k_over_n_train": (
                    float(calibration.get("k_params") or 0) / float(max(1, int(train_all.get("n_q_rows") or 0)))
                ),
                **aic_train,
            },
            "train_all": train_all,
            "holdout_all": holdout_all,
            "overfit_guard": {
                "p95_gap_qbeta": overfit_gap_q,
                "p95_gap_logft_proxy": overfit_gap_logft,
                "gap_threshold": float(overfit_gap_gate),
                "pass": overfit_guard_pass,
            },
            "hard_pass": hard_pass,
            "watch_pass": watch_pass,
        }

    route_a = route_eval.get("A_transfer_surrogate", {})
    route_b = route_eval.get("B_pmodel_proxy", {})
    route_a_hard = bool(route_a.get("hard_pass"))
    route_b_hard = bool(route_b.get("hard_pass"))
    if route_a_hard and (not route_b_hard):
        transition = "A_stay_B_reject"
    elif route_a_hard and route_b_hard:
        transition = "A_and_B_supported"
    elif (not route_a_hard) and route_b_hard:
        transition = "A_reject_B_continue"
    else:
        transition = "A_and_B_reject"

    equalized_decision = {
        "route_a_hard_pass": route_a_hard,
        "route_b_hard_pass": route_b_hard,
        "route_a_watch_pass": bool(route_a.get("watch_pass")),
        "route_b_watch_pass": bool(route_b.get("watch_pass")),
        "transition": transition,
        "overfit_gate": {
            "threshold_p95_gap": float(overfit_gap_gate),
            "route_a_pass": bool((route_a.get("overfit_guard") or {}).get("pass")),
            "route_b_pass": bool((route_b.get("overfit_guard") or {}).get("pass")),
        },
        "dof_equalization": {
            "applied": True,
            "common_observables": ["Q_beta", "logft_proxy", "Delta_CKM", "Delta_PMNS"],
            "route_a_k_params": int(((route_a.get("dof_penalty") or {}).get("k_params") or 0)),
            "route_b_k_params": int(((route_b.get("dof_penalty") or {}).get("k_params") or 0)),
            "equal_k": bool(
                int(((route_a.get("dof_penalty") or {}).get("k_params") or 0))
                == int(((route_b.get("dof_penalty") or {}).get("k_params") or 0))
            ),
        },
    }
    equalized_pack = {
        "split_policy": split_meta,
        "route_evaluation": route_eval,
        "decision": equalized_decision,
    }
    return summary_rows, equalized_pack, split_meta


def _build_equalized_holdout_figure(
    *,
    equalized_pack: dict[str, Any],
    out_png: Path,
    z_gate: float,
) -> None:
    route_eval = equalized_pack.get("route_evaluation") if isinstance(equalized_pack.get("route_evaluation"), dict) else {}
    route_a = route_eval.get("A_transfer_surrogate") if isinstance(route_eval.get("A_transfer_surrogate"), dict) else {}
    route_b = route_eval.get("B_pmodel_proxy") if isinstance(route_eval.get("B_pmodel_proxy"), dict) else {}
    rows = [route_a, route_b]
    labels = [str(r.get("route_label") or "") for r in rows]
    holdout_q = [_parse_float(((r.get("holdout_all") or {}).get("p95_abs_z_qbeta"))) for r in rows]
    holdout_logft = [_parse_float(((r.get("holdout_all") or {}).get("p95_abs_z_logft_proxy"))) for r in rows]
    overfit_gap_q = [_parse_float(((r.get("overfit_guard") or {}).get("p95_gap_qbeta"))) for r in rows]
    x = [0, 1]
    w = 0.36

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.4), dpi=170)
    ax0, ax1 = axes
    ax0.bar([v - w / 2 for v in x], holdout_q, width=w, color="#4c78a8", label="holdout p95 abs(z_Qβ)")
    ax0.bar([v + w / 2 for v in x], holdout_logft, width=w, color="#e45756", label="holdout p95 abs(z_logft)")
    ax0.axhline(z_gate, color="#444444", ls="--", lw=1.0, label=f"hard gate z={z_gate:g}")
    ax0.set_yscale("log")
    ax0.set_xticks(x)
    ax0.set_xticklabels(labels)
    ax0.set_ylabel("holdout p95 abs(z)")
    ax0.set_title("DoF-equalized holdout gate")
    ax0.grid(True, ls=":", lw=0.6, alpha=0.6)
    ax0.legend(loc="upper right", fontsize=8)

    ax1.bar(x, overfit_gap_q, color=["#54a24b", "#f58518"])
    ax1.axhline(0.0, color="#444444", lw=0.8)
    ax1.axhline(float(equalized_pack.get("decision", {}).get("overfit_gate", {}).get("threshold_p95_gap", 1.0)), color="#444444", ls="--", lw=1.0)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("p95 gap (holdout - train) for z_Qβ")
    ax1.set_title("Overfit guard (gap test)")
    ax1.grid(True, ls=":", lw=0.6, alpha=0.6)

    fig.suptitle("Weak-interaction Route A/B dof-equalized + holdout audit (Step 8.7.22.8)")
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def _build_figure(
    *,
    all_rows: list[dict[str, Any]],
    out_png: Path,
    z_gate: float,
) -> None:
    by_route = {str(r["route_id"]): r for r in all_rows if str(r.get("channel")) == "all"}
    route_ids = ["A_transfer_surrogate", "B_pmodel_proxy"]
    labels = [by_route[rid]["route_label"] for rid in route_ids]

    max_q = [float(by_route[rid]["max_abs_z_qbeta"]) for rid in route_ids]
    p95_q = [float(by_route[rid]["p95_abs_z_qbeta"]) for rid in route_ids]
    max_logft = [float(by_route[rid]["max_abs_z_logft_proxy"]) for rid in route_ids]
    p95_logft = [float(by_route[rid]["p95_abs_z_logft_proxy"]) for rid in route_ids]

    x = [0, 1]
    width = 0.36

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.4), dpi=170)
    ax0, ax1 = axes

    ax0.bar([v - width / 2 for v in x], max_q, width=width, label="max abs(z_Qβ)", color="#4c78a8")
    ax0.bar([v + width / 2 for v in x], p95_q, width=width, label="p95 abs(z_Qβ)", color="#f58518")
    ax0.axhline(z_gate, color="#444444", ls="--", lw=1.0, label="gate z=3")
    ax0.set_yscale("log")
    ax0.set_xticks(x)
    ax0.set_xticklabels(labels, rotation=0)
    ax0.set_ylabel("abs(z_Qβ)")
    ax0.set_title("Qβ gate audit")
    ax0.grid(True, ls=":", lw=0.6, alpha=0.6)
    ax0.legend(loc="upper right", fontsize=8)

    ax1.bar([v - width / 2 for v in x], max_logft, width=width, label="max abs(z_logft proxy)", color="#54a24b")
    ax1.bar([v + width / 2 for v in x], p95_logft, width=width, label="p95 abs(z_logft proxy)", color="#e45756")
    ax1.axhline(z_gate, color="#444444", ls="--", lw=1.0, label="gate z=3")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=0)
    ax1.set_ylabel("abs(z_logft proxy)")
    ax1.set_title("logft-proxy gate audit")
    ax1.grid(True, ls=":", lw=0.6, alpha=0.6)
    ax1.legend(loc="upper right", fontsize=8)

    fig.suptitle("Weak-interaction beta-decay Route A/B quantitative audit (Step 8.7.22)")
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def _load_ckm_gate(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "status": "not_evaluated",
            "reason": "CKM audit json is missing.",
            "source": {"path": _rel(path)},
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    gate = payload.get("gate") if isinstance(payload.get("gate"), dict) else {}
    status = str(gate.get("status") or "not_evaluated")
    out: dict[str, Any] = {
        "status": status,
        "source": {
            "path": _rel(path),
            "sha256": _sha256(path),
        },
        "hard_pass": bool(gate.get("hard_pass")) if "hard_pass" in gate else None,
        "watch_pass": bool(gate.get("watch_pass")) if "watch_pass" in gate else None,
        "hard_z_threshold": float(gate.get("hard_z_threshold")) if gate.get("hard_z_threshold") is not None else None,
        "watch_z_threshold": float(gate.get("watch_z_threshold")) if gate.get("watch_z_threshold") is not None else None,
        "rule": str(gate.get("rule") or ""),
        "abs_z_reported": float(payload.get("derived", {}).get("abs_z_reported"))
        if isinstance(payload.get("derived"), dict) and payload.get("derived", {}).get("abs_z_reported") is not None
        else None,
        "delta_ckm_reported": float(payload.get("derived", {}).get("delta_ckm_reported"))
        if isinstance(payload.get("derived"), dict) and payload.get("derived", {}).get("delta_ckm_reported") is not None
        else None,
    }
    pmns_gate = payload.get("pmns_gate") if isinstance(payload.get("pmns_gate"), dict) else {}
    if pmns_gate:
        out["pmns_gate"] = pmns_gate
    correlation_reassessment = (
        payload.get("correlation_reassessment")
        if isinstance(payload.get("correlation_reassessment"), dict)
        else {}
    )
    if correlation_reassessment:
        out["correlation_reassessment"] = correlation_reassessment
        out["watch_resolution_status"] = str(correlation_reassessment.get("watch_resolution_status") or "")
        out["watch_lock_reason"] = str(correlation_reassessment.get("watch_lock_reason") or "")
    return out


def _load_pmns_gate(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "status": "not_evaluated",
            "reason": "PMNS audit json is missing.",
            "source": {"path": _rel(path)},
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    gate = payload.get("gate") if isinstance(payload.get("gate"), dict) else {}
    derived = payload.get("derived") if isinstance(payload.get("derived"), dict) else {}
    selected_dataset = str(gate.get("dataset") or "with_sk_atm")
    selected_derived = derived.get(selected_dataset) if isinstance(derived.get(selected_dataset), dict) else {}
    out: dict[str, Any] = {
        "status": str(gate.get("status") or "not_evaluated"),
        "source": {
            "path": _rel(path),
            "sha256": _sha256(path),
        },
        "dataset": selected_dataset,
        "hard_pass": bool(gate.get("hard_pass")) if "hard_pass" in gate else None,
        "watch_pass": bool(gate.get("watch_pass")) if "watch_pass" in gate else None,
        "hard_z_threshold": float(gate.get("hard_z_threshold")) if gate.get("hard_z_threshold") is not None else None,
        "watch_z_threshold": float(gate.get("watch_z_threshold")) if gate.get("watch_z_threshold") is not None else None,
        "rule": str(gate.get("rule") or ""),
        "abs_z_center_proxy": float(selected_derived.get("abs_z_center_proxy"))
        if selected_derived.get("abs_z_center_proxy") is not None
        else None,
        "delta_pmns_center_proxy": float(selected_derived.get("delta_pmns_center_proxy"))
        if selected_derived.get("delta_pmns_center_proxy") is not None
        else None,
    }
    return out


def _combine_closure_gates(*, ckm_gate: dict[str, Any], pmns_gate: dict[str, Any]) -> dict[str, Any]:
    ckm_hard = ckm_gate.get("hard_pass")
    pmns_hard = pmns_gate.get("hard_pass")
    ckm_watch = ckm_gate.get("watch_pass")
    pmns_watch = pmns_gate.get("watch_pass")
    ckm_ready = isinstance(ckm_hard, bool)
    pmns_ready = isinstance(pmns_hard, bool)
    if not (ckm_ready and pmns_ready):
        return {
            "status": "not_evaluated",
            "hard_pass": False,
            "watch_pass": False,
            "rule": "combined closure requires both CKM and PMNS hard/watch gates.",
        }
    hard_pass = bool(ckm_hard and pmns_hard)
    watch_pass = bool(isinstance(ckm_watch, bool) and isinstance(pmns_watch, bool) and ckm_watch and pmns_watch)
    if hard_pass and watch_pass:
        status = "pass"
    elif hard_pass:
        status = "watch"
    else:
        status = "reject"
    watch_gap_reasons: list[str] = []
    if hard_pass and not watch_pass:
        if isinstance(ckm_watch, bool) and (not ckm_watch):
            watch_gap_reasons.append(str(ckm_gate.get("watch_resolution_status") or "ckm_watch_fail"))
        if isinstance(pmns_watch, bool) and (not pmns_watch):
            watch_gap_reasons.append("pmns_watch_fail")
    watch_locked_by_current_primary_source = bool(
        "watch_locked_by_current_primary_source_precision" in watch_gap_reasons
    )
    return {
        "status": status,
        "hard_pass": hard_pass,
        "watch_pass": watch_pass,
        "rule": "combined hard/watch pass requires both CKM and PMNS gates.",
        "watch_gap_reasons": watch_gap_reasons,
        "watch_locked_by_current_primary_source": watch_locked_by_current_primary_source,
        "next_action": (
            "keep watch until CKM first-row primary inputs are updated"
            if watch_locked_by_current_primary_source
            else "none"
        ),
    }


def _route_a_watch_outliers(
    *,
    rows: list[dict[str, Any]],
    z_gate: float,
    primary_modes: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        q_obs = _parse_float(row.get("q_obs_MeV"))
        q_sigma = _parse_float(row.get("q_obs_sigma_MeV"))
        q_pred_a = _parse_float(row.get("q_obs_from_binding_MeV"))
        if not (math.isfinite(q_obs) and math.isfinite(q_sigma) and q_sigma > 0.0 and math.isfinite(q_pred_a)):
            continue
        delta_q = float(q_pred_a - q_obs)
        abs_z = abs(delta_q / q_sigma)
        if abs_z <= z_gate:
            continue
        observed_decay_has_channel = _parse_bool(row.get("observed_decay_has_channel"))
        legacy_predicted_q_positive = _parse_bool(row.get("predicted_q_positive"))
        legacy_mode_consistent = _parse_bool(row.get("mode_consistent"))
        route_a_predicted_q_positive = bool(q_pred_a > 0.0)
        route_a_mode_consistent = (
            bool(observed_decay_has_channel == route_a_predicted_q_positive)
            if observed_decay_has_channel is not None
            else None
        )
        if observed_decay_has_channel is False or route_a_mode_consistent is False:
            cause_tag = "mode_or_definition_watch"
        elif abs(delta_q) <= 0.2 and q_sigma <= 0.05:
            cause_tag = "high_precision_small_offset_watch"
        else:
            cause_tag = "residual_watch"
        if legacy_mode_consistent is False and route_a_mode_consistent is True:
            review_action = "relabel_route_a_mode_consistent_keep_watch"
            review_reason = "legacy mode_consistent used q_pred_after sign; Route-A audit now uses q_route_a sign."
        elif legacy_mode_consistent is False and route_a_mode_consistent is False:
            review_action = "keep_mode_definition_watch"
            review_reason = "Route-A sign and observed branch both indicate a transition-definition mismatch."
        else:
            review_action = "keep"
            review_reason = "No relabel required."
        nuclide_key = str(row.get("nuclide_key", ""))
        pmode = primary_modes.get(nuclide_key, {})
        observed_modes = pmode.get("observed_modes")
        observed_modes_text = ", ".join(str(v) for v in observed_modes) if isinstance(observed_modes, list) else ""
        branch_beta_minus = _parse_float(pmode.get("branch_beta_minus_percent"))
        branch_beta_plus = _parse_float(pmode.get("branch_beta_plus_percent"))
        out.append(
            {
                "nuclide_key": nuclide_key,
                "nuclide_name": str(row.get("nuclide_name", "")),
                "A": int(float(row.get("A", "nan"))) if str(row.get("A", "")).strip() else -1,
                "Z": int(float(row.get("Z", "nan"))) if str(row.get("Z", "")).strip() else -1,
                "N": int(float(row.get("N", "nan"))) if str(row.get("N", "")).strip() else -1,
                "channel": str(row.get("channel", "")),
                "daughter_Z": int(float(row.get("daughter_Z", "nan"))) if str(row.get("daughter_Z", "")).strip() else -1,
                "daughter_N": int(float(row.get("daughter_N", "nan"))) if str(row.get("daughter_N", "")).strip() else -1,
                "q_obs_MeV": q_obs,
                "q_obs_sigma_MeV": q_sigma,
                "q_route_a_MeV": q_pred_a,
                "delta_q_MeV": delta_q,
                "abs_z_qbeta": abs_z,
                "observed_decay_has_channel": observed_decay_has_channel,
                "legacy_predicted_q_positive": legacy_predicted_q_positive,
                "legacy_mode_consistent": legacy_mode_consistent,
                "route_a_predicted_q_positive": route_a_predicted_q_positive,
                "route_a_mode_consistent": route_a_mode_consistent,
                "observed_modes_primary": observed_modes_text,
                "has_beta_minus_mode_primary": bool(pmode.get("has_beta_minus_mode")) if "has_beta_minus_mode" in pmode else None,
                "has_beta_plus_mode_primary": bool(pmode.get("has_beta_plus_mode")) if "has_beta_plus_mode" in pmode else None,
                "branch_beta_minus_percent_primary": branch_beta_minus if math.isfinite(branch_beta_minus) else None,
                "branch_beta_plus_percent_primary": branch_beta_plus if math.isfinite(branch_beta_plus) else None,
                "cause_tag": cause_tag,
                "review_action": review_action,
                "review_reason": review_reason,
            }
        )
    return sorted(out, key=lambda row: float(row.get("abs_z_qbeta", 0.0)), reverse=True)


def _route_a_watch_outlier_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_channel = Counter(str(row.get("channel", "")) for row in rows)
    by_a = Counter(int(row.get("A", -1)) for row in rows)
    by_cause = Counter(str(row.get("cause_tag", "")) for row in rows)
    by_action = Counter(str(row.get("review_action", "")) for row in rows)
    abs_delta_vals = [abs(float(row.get("delta_q_MeV", float("nan")))) for row in rows if math.isfinite(float(row.get("delta_q_MeV", float("nan"))))]
    abs_z_vals = [float(row.get("abs_z_qbeta", float("nan"))) for row in rows if math.isfinite(float(row.get("abs_z_qbeta", float("nan"))))]
    return {
        "n_rows": len(rows),
        "by_channel": dict(sorted(by_channel.items())),
        "by_mass_number_A": {str(k): int(v) for k, v in sorted(by_a.items())},
        "by_cause_tag": dict(sorted(by_cause.items())),
        "by_review_action": dict(sorted(by_action.items())),
        "max_abs_z_qbeta": _safe_max(abs_z_vals),
        "median_abs_delta_q_MeV": _median(abs_delta_vals),
        "max_abs_delta_q_MeV": _safe_max(abs_delta_vals),
    }


def _build_route_a_watch_outlier_figure(
    *,
    rows: list[dict[str, Any]],
    out_png: Path,
    z_gate: float,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.4), dpi=170)
    ax0, ax1 = axes

    if not rows:
        for ax in (ax0, ax1):
            ax.axis("off")
            ax.text(0.5, 0.5, "No Route-A watch outliers", ha="center", va="center", fontsize=11)
        fig.suptitle("Weak-interaction beta-decay Route-A watch outliers")
        fig.tight_layout()
        fig.savefig(out_png, bbox_inches="tight")
        plt.close(fig)
        return

    labels = [f"{str(r['nuclide_name'])} ({str(r['channel'])})" for r in rows]
    z_vals = [float(r["abs_z_qbeta"]) for r in rows]
    deltas = [abs(float(r["delta_q_MeV"])) for r in rows]
    sigmas = [float(r["q_obs_sigma_MeV"]) for r in rows]
    channel_colors = {"beta_minus": "#4c78a8", "beta_plus": "#f58518"}
    colors = [channel_colors.get(str(r.get("channel", "")), "#888888") for r in rows]
    x = list(range(len(rows)))

    ax0.bar(x, z_vals, color=colors, alpha=0.9)
    ax0.axhline(z_gate, color="#444444", ls="--", lw=1.0, label=f"gate z={z_gate:g}")
    ax0.set_xticks(x)
    ax0.set_xticklabels(labels, rotation=20, ha="right")
    ax0.set_ylabel("abs(z_Qβ)")
    ax0.set_title("Route-A watch outliers (abs(z_Qβ)>3)")
    ax0.grid(True, axis="y", ls=":", lw=0.6, alpha=0.6)
    ax0.legend(loc="upper right", fontsize=8)

    for idx, row in enumerate(rows):
        marker = "o" if str(row.get("channel", "")) == "beta_minus" else "s"
        ax1.scatter(sigmas[idx], deltas[idx], s=70.0, marker=marker, color=colors[idx], alpha=0.9)
        ax1.annotate(str(row.get("nuclide_name", "")), (sigmas[idx], deltas[idx]), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax1.set_xlabel("Q_obs sigma [MeV]")
    ax1.set_ylabel("abs(delta Q) [MeV]")
    ax1.set_title("Outlier scale: sigma vs abs(delta Q)")
    ax1.grid(True, ls=":", lw=0.6, alpha=0.6)
    ax1.set_xlim(left=0.0)
    ax1.set_ylim(bottom=0.0)

    fig.suptitle("Weak-interaction beta-decay Route-A watch outlier diagnostics")
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 8.7.22: weak-interaction beta-decay Route A/B quantitative audit"
    )
    parser.add_argument(
        "--in-csv",
        type=Path,
        default=ROOT / "output" / "public" / "quantum" / "nuclear_beta_decay_qvalue_prediction_full.csv",
        help="Input CSV from Step 7.16.16.",
    )
    parser.add_argument(
        "--in-json",
        type=Path,
        default=ROOT / "output" / "public" / "quantum" / "nuclear_beta_decay_qvalue_prediction_metrics.json",
        help="Input metrics JSON from Step 7.16.16.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=ROOT / "output" / "public" / "quantum",
        help="Output directory.",
    )
    parser.add_argument(
        "--z-gate",
        type=float,
        default=3.0,
        help="Absolute z threshold for gate checks.",
    )
    parser.add_argument(
        "--logft-sigma-proxy",
        type=float,
        default=1.0,
        help="Operational sigma proxy for z_logft proxy scaling (dex).",
    )
    parser.add_argument(
        "--ckm-audit-json",
        type=Path,
        default=ROOT / "output" / "public" / "quantum" / "weak_interaction_ckm_first_row_audit.json",
        help="Optional CKM first-row audit JSON. If missing, ckm_gate remains not_evaluated.",
    )
    parser.add_argument(
        "--pmns-audit-json",
        type=Path,
        default=ROOT / "output" / "public" / "quantum" / "weak_interaction_pmns_first_row_audit.json",
        help="Optional PMNS first-row audit JSON. If missing, pmns_gate remains not_evaluated.",
    )
    parser.add_argument(
        "--nudat-primary-json",
        type=Path,
        default=ROOT / "data" / "quantum" / "sources" / "nndc_nudat3_primary_secondary" / "primary.json",
        help="NuDat primary.json used for transition-definition re-audit of Route-A watch outliers.",
    )
    parser.add_argument(
        "--holdout-hash-modulo",
        type=int,
        default=5,
        help="Deterministic holdout split modulus on nuclide_key hash.",
    )
    parser.add_argument(
        "--holdout-hash-residue",
        type=int,
        default=0,
        help="Deterministic holdout split residue on nuclide_key hash.",
    )
    parser.add_argument(
        "--overfit-gap-gate",
        type=float,
        default=1.0,
        help="Holdout overfit guard threshold for p95 gap (holdout-train).",
    )
    parser.add_argument(
        "--sigma-floor-mev",
        type=float,
        default=1.0e-9,
        help="Sigma floor [MeV] for weighted linear calibration stability.",
    )
    args = parser.parse_args()

    in_csv = args.in_csv
    in_json = args.in_json
    out_dir = args.outdir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "weak_interaction_beta_decay_route_ab_audit.json"
    previous_ckm_watchpack = _load_previous_ckm_watchpack(out_json)

    if not in_csv.exists():
        raise SystemExit(f"[fail] missing input csv: {in_csv}")
    if not in_json.exists():
        raise SystemExit(f"[fail] missing input json: {in_json}")

    with in_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"[fail] input csv has no rows: {in_csv}")

    upstream = json.loads(in_json.read_text(encoding="utf-8"))

    summary_rows_legacy: list[dict[str, Any]] = []
    for route in ROUTES:
        for channel in CHANNELS:
            summary_rows_legacy.append(
                _route_channel_summary(
                    rows=rows,
                    route=route,
                    channel=channel,
                    logft_sigma_proxy=float(args.logft_sigma_proxy),
                    z_gate=float(args.z_gate),
                )
            )

    decision_legacy = _decision(summary_rows_legacy)
    equalized_rows, equalized_pack, split_meta = _equalized_route_audit(
        rows=rows,
        logft_sigma_proxy=float(args.logft_sigma_proxy),
        z_gate=float(args.z_gate),
        holdout_hash_modulo=int(args.holdout_hash_modulo),
        holdout_hash_residue=int(args.holdout_hash_residue),
        overfit_gap_gate=float(args.overfit_gap_gate),
        sigma_floor_mev=float(args.sigma_floor_mev),
    )
    decision = dict(equalized_pack.get("decision") or {})
    decision["legacy"] = decision_legacy

    ckm_gate = _load_ckm_gate(args.ckm_audit_json)
    pmns_gate = _load_pmns_gate(args.pmns_audit_json)
    decision["ckm_gate"] = ckm_gate
    decision["pmns_gate"] = pmns_gate
    decision["ckm_pmns_closure"] = _combine_closure_gates(ckm_gate=ckm_gate, pmns_gate=pmns_gate)
    ckm_primary_update_watchpack = _derive_ckm_primary_update_watchpack(
        current_input_signature=_file_signature(args.ckm_audit_json),
        previous_watchpack=previous_ckm_watchpack,
        closure_gate=decision["ckm_pmns_closure"],
    )

    out_csv = out_dir / "weak_interaction_beta_decay_route_ab_audit_summary.csv"
    out_png = out_dir / "weak_interaction_beta_decay_route_ab_audit.png"
    out_equalized_csv = out_dir / "weak_interaction_beta_decay_route_ab_equalized_summary.csv"
    out_equalized_png = out_dir / "weak_interaction_beta_decay_route_ab_equalized_holdout.png"

    _write_csv(out_csv, summary_rows_legacy)
    _write_csv(out_equalized_csv, equalized_rows)
    _build_figure(all_rows=summary_rows_legacy, out_png=out_png, z_gate=float(args.z_gate))
    _build_equalized_holdout_figure(
        equalized_pack=equalized_pack,
        out_png=out_equalized_png,
        z_gate=float(args.z_gate),
    )

    primary_modes = _load_nudat_primary_modes(args.nudat_primary_json)
    route_a_watch_rows = _route_a_watch_outliers(
        rows=rows,
        z_gate=float(args.z_gate),
        primary_modes=primary_modes,
    )
    route_a_watch_summary = _route_a_watch_outlier_summary(route_a_watch_rows)
    out_watch_csv = out_dir / "weak_interaction_beta_decay_route_a_watch_outliers.csv"
    out_watch_json = out_dir / "weak_interaction_beta_decay_route_a_watch_outliers.json"
    out_watch_png = out_dir / "weak_interaction_beta_decay_route_a_watch_outliers.png"
    _write_csv(out_watch_csv, route_a_watch_rows)
    _build_route_a_watch_outlier_figure(
        rows=route_a_watch_rows,
        out_png=out_watch_png,
        z_gate=float(args.z_gate),
    )
    out_watch_json.write_text(
        json.dumps(
            {
                "generated_utc": _iso_now(),
                "phase": 8,
                "step": "8.7.22.8",
                "title": "Weak-interaction beta-decay Route-A watch outlier audit",
                "gate_z_threshold": float(args.z_gate),
                "summary": route_a_watch_summary,
                "rows": route_a_watch_rows,
                "outputs": {
                    "watch_outliers_csv": _rel(out_watch_csv),
                    "watch_outliers_png": _rel(out_watch_png),
                },
                "inputs": {
                    "nudat_primary_json": {
                        "path": _rel(args.nudat_primary_json),
                        "exists": bool(args.nudat_primary_json.exists()),
                        "sha256": _sha256(args.nudat_primary_json) if args.nudat_primary_json.exists() else None,
                    }
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    payload = {
        "generated_utc": _iso_now(),
        "phase": 8,
        "step": "8.7.22.8",
        "title": "Weak-interaction beta-decay Route A/B quantitative audit (DoF-equalized + holdout + CKM update watchpack)",
        "inputs": {
            "beta_qvalue_full_csv": {"path": _rel(in_csv), "sha256": _sha256(in_csv)},
            "beta_qvalue_metrics_json": {"path": _rel(in_json), "sha256": _sha256(in_json)},
            "ckm_audit_json": {"path": _rel(args.ckm_audit_json), "exists": bool(args.ckm_audit_json.exists())},
            "pmns_audit_json": {"path": _rel(args.pmns_audit_json), "exists": bool(args.pmns_audit_json.exists())},
            "nudat_primary_json": {
                "path": _rel(args.nudat_primary_json),
                "exists": bool(args.nudat_primary_json.exists()),
                "sha256": _sha256(args.nudat_primary_json) if args.nudat_primary_json.exists() else None,
            },
        },
        "gate_policy": {
            "z_threshold": float(args.z_gate),
            "hard_gate": {
                "qbeta": "p95 abs(z_Qβ) <= z_threshold",
                "logft_proxy": "p95 abs(z_logft_proxy) <= z_threshold",
            },
            "watch_gate": {
                "qbeta": "max abs(z_Qβ) <= z_threshold",
                "logft_proxy": "max abs(z_logft_proxy) <= z_threshold",
            },
            "logft_proxy_definition": "z_logft_proxy = 5*(log10(Q_pred)-log10(Q_obs))/sigma_logft_proxy",
            "sigma_logft_proxy_dex": float(args.logft_sigma_proxy),
        },
        "dof_holdout_policy": {
            "holdout_hash_modulo": int(args.holdout_hash_modulo),
            "holdout_hash_residue": int(args.holdout_hash_residue),
            "overfit_gap_gate": float(args.overfit_gap_gate),
            "sigma_floor_mev": float(args.sigma_floor_mev),
            "split": split_meta,
        },
        "routes": [r for r in summary_rows_legacy if str(r.get("channel")) == "all"],
        "per_channel": summary_rows_legacy,
        "routes_equalized_holdout": equalized_pack.get("route_evaluation"),
        "equalized_per_split_channel": equalized_rows,
        "decision": decision,
        "diagnostics": {
            "ckm_primary_update_watchpack": ckm_primary_update_watchpack,
        },
        "route_a_watch_outliers": {
            "summary": route_a_watch_summary,
            "outputs": {
                "watch_outliers_csv": _rel(out_watch_csv),
                "watch_outliers_json": _rel(out_watch_json),
                "watch_outliers_png": _rel(out_watch_png),
            },
        },
        "upstream_counts": upstream.get("counts"),
        "outputs": {
            "summary_csv": _rel(out_csv),
            "equalized_summary_csv": _rel(out_equalized_csv),
            "audit_json": _rel(out_json),
            "audit_png": _rel(out_png),
            "equalized_holdout_png": _rel(out_equalized_png),
        },
        "notes": [
            "Legacy rows keep pre-existing all-channel/per-channel gate metrics for continuity.",
            "Step 8.7.22.8 keeps DoF-equalized calibration (same 2-parameter affine map) for Route A/B.",
            "Holdout is nuclide-level deterministic split and is used as the primary hard/watch gate.",
            "Overfit guard uses p95 gap (holdout-train) for Qβ and logft-proxy.",
            "CKM gate and PMNS gate are loaded when their audit JSONs are available.",
            "Combined CKM+PMNS closure is reported as decision.ckm_pmns_closure.",
            "CKM primary-input hash watchpack is emitted as diagnostics.ckm_primary_update_watchpack.",
            "Route-A outlier relabel review compares legacy mode-consistency proxy with transition-definition consistency from NuDat primary modes.",
            "Route-A watch outliers are emitted as a dedicated nuclide-level audit bundle.",
        ],
    }

    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    worklog.append_event(
        {
            "type": "step_update",
            "step": "8.7.22.8",
            "script": "scripts/quantum/weak_interaction_beta_decay_route_ab_audit.py",
            "inputs": [_rel(in_csv), _rel(in_json)],
            "outputs": [
                _rel(out_csv),
                _rel(out_equalized_csv),
                _rel(out_json),
                _rel(out_png),
                _rel(out_equalized_png),
                _rel(out_watch_csv),
                _rel(out_watch_json),
                _rel(out_watch_png),
            ],
            "decision": decision,
            "diagnostics": {"ckm_primary_update_watchpack": ckm_primary_update_watchpack},
        }
    )

    print("[ok] wrote:")
    print(f"  {out_csv}")
    print(f"  {out_equalized_csv}")
    print(f"  {out_json}")
    print(f"  {out_png}")
    print(f"  {out_equalized_png}")
    print(f"  {out_watch_csv}")
    print(f"  {out_watch_json}")
    print(f"  {out_watch_png}")


if __name__ == "__main__":
    main()
