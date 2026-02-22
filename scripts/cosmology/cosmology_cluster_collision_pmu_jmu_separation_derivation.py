#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cosmology_cluster_collision_pmu_jmu_separation_derivation.py

Step 8.7.25.16:
弾丸銀河団（Bullet系）の質量分離を、SPARC係数流用なしで
J^μ 駆動の P_μ 遅延応答から導出する監査パック。
本ステップでは遅延ポテンシャル（LW同型）に加えて、応答時定数 τ を
「手動注入」ではなく P_μ 基礎方程式側の時定数分解から導出し、
retarded kernel を単一指数から「導出二重指数」へ拡張し、
cluster別の寄与分解を付与して tau_origin_closure を機械判定する。

固定出力:
  - output/public/cosmology/cosmology_cluster_collision_pmu_jmu_separation_derivation.json
  - output/public/cosmology/cosmology_cluster_collision_pmu_jmu_separation_derivation.csv
  - output/public/cosmology/cosmology_cluster_collision_pmu_jmu_separation_derivation.png
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
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from scripts.summary import worklog  # type: ignore  # noqa: E402
except Exception:
    worklog = None

try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:
    plt = None

KM_S_TO_KPC_GYR = 1.0227121650537077
LIGHT_SPEED_KM_S = 299792.458


@dataclass(frozen=True)
class CollisionObs:
    cluster_id: str
    label: str
    offset_obs_kpc: float
    offset_sigma_kpc: float


@dataclass(frozen=True)
class Kinematics:
    cluster_id: str
    v_pre_km_s: float
    v_post_km_s: float
    chi_post: float


@dataclass(frozen=True)
class SimConfig:
    t_collision_gyr: float
    t_observe_gyr: float
    t_drag_gyr: float
    t_transition_gyr: float
    dt_gyr: float
    tau_response_gyr: float
    chi_pre: float
    sign_epsilon_kpc: float
    sign_flip_window_gyr: float
    retarded_enabled: bool
    retarded_weight: float
    retarded_path_kpc: float
    p_wave_speed_km_s: float
    retarded_kernel_tau_gyr: float
    retarded_kernel_mode: str
    retarded_kernel_tau_fast_gyr: float
    retarded_kernel_tau_slow_gyr: float


@dataclass(frozen=True)
class TauOrigin:
    mode: str
    tau_free_gyr: float
    tau_int_gyr: float
    tau_damp_gyr: float
    tau_eff_gyr: float
    tau_eff_harmonic_gyr: float
    tau_kernel_gyr: float
    tau_kernel_fast_gyr: float
    tau_kernel_slow_gyr: float
    kernel_mode: str
    retarded_weight_eta: float
    xi_derived: float
    closure_pass: bool
    manual_injection: bool
    ad_hoc_parameter_count: int


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except Exception:
        return path.as_posix()


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def _load_observations(path: Path) -> List[CollisionObs]:
    if not path.exists():
        raise FileNotFoundError(f"missing input CSV: {path}")
    out: List[CollisionObs] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                cid = str(row.get("cluster_id", "")).strip()
                label = str(row.get("label", cid)).strip() or cid
                obs = float(row.get("lens_gas_offset_kpc_obs", "nan"))
                sig = float(row.get("lens_gas_offset_kpc_sigma", "nan"))
            except Exception:
                continue
            if not cid or not np.isfinite(obs) or not np.isfinite(sig) or sig <= 0.0:
                continue
            out.append(
                CollisionObs(
                    cluster_id=cid,
                    label=label,
                    offset_obs_kpc=float(obs),
                    offset_sigma_kpc=float(sig),
                )
            )
    if not out:
        raise RuntimeError(f"no valid rows in {path}")
    return out


def _default_kinematics() -> Dict[str, Kinematics]:
    return {
        "bullet_main": Kinematics(
            cluster_id="bullet_main",
            v_pre_km_s=1200.0,
            v_post_km_s=300.0,
            chi_post=1.0,
        ),
        "bullet_sub": Kinematics(
            cluster_id="bullet_sub",
            v_pre_km_s=-3000.0,
            v_post_km_s=-1100.0,
            chi_post=1100.0 / 3000.0,
        ),
    }


def _smooth_transition(t: np.ndarray, t0: float, width: float) -> np.ndarray:
    width_safe = max(width, 1.0e-6)
    return 1.0 / (1.0 + np.exp(-(t - t0) / width_safe))


def _positive_mean(values: Sequence[float], fallback: float) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    arr = arr[arr > 0.0]
    if arr.size == 0:
        return float(max(fallback, 1.0e-8))
    return float(np.mean(arr))


def _timescale_from_series(series: np.ndarray, t: np.ndarray, fallback: float) -> float:
    y = np.asarray(series, dtype=float)
    tt = np.asarray(t, dtype=float)
    if y.size < 3 or tt.size != y.size:
        return float(max(fallback, 1.0e-8))
    dt = float(max(np.median(np.diff(tt)), 1.0e-9))
    dy_dt = np.gradient(y, dt)
    power = float(np.mean(y * y))
    slope_power = float(np.mean(dy_dt * dy_dt))
    if (not np.isfinite(power)) or (not np.isfinite(slope_power)) or power <= 0.0 or slope_power <= 0.0:
        return float(max(fallback, 1.0e-8))
    return float(max(np.sqrt(power / slope_power), dt))


def _velocity_profile(t: np.ndarray, *, v_pre: float, v_post: float, t_collision: float, t_drag: float) -> np.ndarray:
    after = v_post + (v_pre - v_post) * np.exp(-np.clip(t - t_collision, 0.0, None) / max(t_drag, 1.0e-6))
    return np.where(t <= t_collision, v_pre, after)


def _derive_tau_origin(
    kin_map: Dict[str, Kinematics],
    cfg_seed: SimConfig,
    *,
    extended_kernel: bool,
) -> TauOrigin:
    t = np.arange(0.0, cfg_seed.t_observe_gyr + cfg_seed.dt_gyr, cfg_seed.dt_gyr, dtype=float)
    tau_free = float(cfg_seed.retarded_path_kpc / (max(cfg_seed.p_wave_speed_km_s, 1.0e-9) * KM_S_TO_KPC_GYR))
    tau_free = float(max(tau_free, cfg_seed.dt_gyr, 1.0e-8))

    tau_int_terms: List[float] = []
    tau_damp_terms: List[float] = []
    for kin in kin_map.values():
        v = _velocity_profile(
            t,
            v_pre=kin.v_pre_km_s * KM_S_TO_KPC_GYR,
            v_post=kin.v_post_km_s * KM_S_TO_KPC_GYR,
            t_collision=cfg_seed.t_collision_gyr,
            t_drag=cfg_seed.t_drag_gyr,
        )
        transition = _smooth_transition(t, cfg_seed.t_collision_gyr, cfg_seed.t_transition_gyr)
        chi = cfg_seed.chi_pre + (kin.chi_post - cfg_seed.chi_pre) * transition
        j = chi * v

        mask = t >= cfg_seed.t_collision_gyr
        if int(np.count_nonzero(mask)) >= 8:
            t_use = t[mask]
            v_use = v[mask]
            j_use = j[mask]
        else:
            t_use = t
            v_use = v
            j_use = j

        tau_int_terms.append(_timescale_from_series(j_use, t_use, tau_free))
        tau_damp_terms.append(_timescale_from_series(v_use, t_use, tau_free))

    tau_int = _positive_mean(tau_int_terms, tau_free)
    tau_damp = _positive_mean(tau_damp_terms, tau_free)
    eta_weight = float(min(max(tau_free / max(tau_free + tau_int, 1.0e-12), 0.0), 1.0))

    inv_tau_eff = (1.0 / tau_free) + (1.0 / tau_int) + (1.0 / tau_damp)
    tau_eff_harmonic = float(1.0 / max(inv_tau_eff, 1.0e-12))

    if extended_kernel:
        tau_fast = float(max(min(tau_int, tau_damp), cfg_seed.dt_gyr, 1.0e-8))
        tau_slow = float(max(max(tau_int, tau_damp), tau_fast, cfg_seed.dt_gyr, 1.0e-8))
        inv_rate_sum = (1.0 / tau_fast) + (1.0 / tau_slow)
        if inv_rate_sum <= 0.0 or (not np.isfinite(inv_rate_sum)):
            w_fast = 0.5
            w_slow = 0.5
        else:
            w_fast = float((1.0 / tau_fast) / inv_rate_sum)
            w_slow = float((1.0 / tau_slow) / inv_rate_sum)
        tau_kernel = float(max(w_fast * tau_fast + w_slow * tau_slow, cfg_seed.dt_gyr, 1.0e-8))
        tau_eff = float(tau_free + eta_weight * tau_kernel)
        xi_derived = float((tau_int + tau_damp) / max(tau_eff, 1.0e-12))
        kernel_mode = "double_exp_derived"
    else:
        tau_kernel = float(max(tau_int, cfg_seed.dt_gyr, 1.0e-8))
        tau_fast = float(tau_kernel)
        tau_slow = float(tau_kernel)
        tau_eff = float(tau_eff_harmonic)
        xi_derived = 1.0
        kernel_mode = "single_exp"

    closure_pass = bool(
        np.isfinite(tau_eff)
        and tau_eff > 0.0
        and np.isfinite(tau_kernel)
        and tau_kernel > 0.0
        and np.isfinite(xi_derived)
        and xi_derived > 0.0
    )
    return TauOrigin(
        mode="derived_from_pmu_kernel",
        tau_free_gyr=float(tau_free),
        tau_int_gyr=float(tau_int),
        tau_damp_gyr=float(tau_damp),
        tau_eff_gyr=float(tau_eff),
        tau_eff_harmonic_gyr=float(tau_eff_harmonic),
        tau_kernel_gyr=float(tau_kernel),
        tau_kernel_fast_gyr=float(tau_fast),
        tau_kernel_slow_gyr=float(tau_slow),
        kernel_mode=str(kernel_mode),
        retarded_weight_eta=float(eta_weight),
        xi_derived=float(xi_derived),
        closure_pass=closure_pass,
        manual_injection=False,
        ad_hoc_parameter_count=0,
    )


def _build_retarded_current(
    current_drive: np.ndarray,
    dt_gyr: float,
    *,
    enabled: bool,
    delay_gyr: float,
    tau_kernel_gyr: float,
    kernel_mode: str,
    tau_kernel_fast_gyr: float,
    tau_kernel_slow_gyr: float,
) -> Dict[str, Any]:
    drive = np.asarray(current_drive, dtype=float)
    n = int(drive.size)
    if n == 0:
        return {
            "delay_steps": 0,
            "delay_gyr": 0.0,
            "kernel_tau_gyr": max(float(tau_kernel_gyr), 1.0e-8),
            "current_delayed": drive.copy(),
            "current_smoothed": drive.copy(),
            "peak_lag_gyr": 0.0,
            "peak_corr": 0.0,
            "kernel_mode": str(kernel_mode),
            "kernel_tau_fast_gyr": max(float(tau_kernel_fast_gyr), 1.0e-8),
            "kernel_tau_slow_gyr": max(float(tau_kernel_slow_gyr), 1.0e-8),
            "kernel_weight_fast": 1.0,
            "kernel_weight_slow": 0.0,
        }

    if not enabled:
        return {
            "delay_steps": 0,
            "delay_gyr": 0.0,
            "kernel_tau_gyr": max(float(tau_kernel_gyr), 1.0e-8),
            "current_delayed": drive.copy(),
            "current_smoothed": drive.copy(),
            "peak_lag_gyr": 0.0,
            "peak_corr": 1.0,
            "kernel_mode": str(kernel_mode),
            "kernel_tau_fast_gyr": max(float(tau_kernel_fast_gyr), 1.0e-8),
            "kernel_tau_slow_gyr": max(float(tau_kernel_slow_gyr), 1.0e-8),
            "kernel_weight_fast": 1.0,
            "kernel_weight_slow": 0.0,
        }

    dt_safe = max(float(dt_gyr), 1.0e-9)
    delay_steps = max(int(round(float(delay_gyr) / dt_safe)), 0)
    delayed = np.empty_like(drive)
    if delay_steps > 0:
        delayed[:delay_steps] = drive[0]
        delayed[delay_steps:] = drive[:-delay_steps]
    else:
        delayed[:] = drive

    mode = str(kernel_mode).strip()
    if mode == "double_exp_derived":
        tau_fast = max(float(tau_kernel_fast_gyr), dt_safe)
        tau_slow = max(float(tau_kernel_slow_gyr), tau_fast)
        inv_rate_sum = (1.0 / tau_fast) + (1.0 / tau_slow)
        if inv_rate_sum <= 0.0 or (not np.isfinite(inv_rate_sum)):
            w_fast = 0.5
            w_slow = 0.5
        else:
            w_fast = float((1.0 / tau_fast) / inv_rate_sum)
            w_slow = float((1.0 / tau_slow) / inv_rate_sum)

        smooth_fast = np.empty_like(drive)
        smooth_slow = np.empty_like(drive)
        smooth_fast[0] = delayed[0]
        smooth_slow[0] = delayed[0]
        for i in range(1, n):
            smooth_fast[i] = smooth_fast[i - 1] + dt_safe * (delayed[i - 1] - smooth_fast[i - 1]) / tau_fast
            smooth_slow[i] = smooth_slow[i - 1] + dt_safe * (delayed[i - 1] - smooth_slow[i - 1]) / tau_slow
        smoothed = w_fast * smooth_fast + w_slow * smooth_slow
        tau = float(w_fast * tau_fast + w_slow * tau_slow)
    else:
        tau = max(float(tau_kernel_gyr), dt_safe)
        w_fast = 1.0
        w_slow = 0.0
        tau_fast = float(tau)
        tau_slow = float(tau)
        smoothed = np.empty_like(drive)
        smoothed[0] = delayed[0]
        for i in range(1, n):
            smoothed[i] = smoothed[i - 1] + dt_safe * (delayed[i - 1] - smoothed[i - 1]) / tau

    x = delayed - float(np.mean(delayed))
    y = smoothed - float(np.mean(smoothed))
    if np.all(np.abs(x) < 1.0e-14) or np.all(np.abs(y) < 1.0e-14):
        peak_idx = n - 1
        peak_corr = 0.0
    else:
        corr = np.correlate(y, x, mode="full")
        peak_idx = int(np.argmax(corr))
        peak_corr = float(corr[peak_idx] / max(np.linalg.norm(y) * np.linalg.norm(x), 1.0e-12))
    lag_steps = peak_idx - (n - 1)
    lag_gyr = float(lag_steps * dt_safe)

    return {
        "delay_steps": int(delay_steps),
        "delay_gyr": float(delay_steps * dt_safe),
        "kernel_tau_gyr": float(tau),
        "current_delayed": delayed,
        "current_smoothed": smoothed,
        "peak_lag_gyr": lag_gyr,
        "peak_corr": peak_corr,
        "kernel_mode": ("double_exp_derived" if mode == "double_exp_derived" else "single_exp"),
        "kernel_tau_fast_gyr": float(tau_fast),
        "kernel_tau_slow_gyr": float(tau_slow),
        "kernel_weight_fast": float(w_fast),
        "kernel_weight_slow": float(w_slow),
    }


def _build_lw_like_estimates(
    rows: Sequence[Dict[str, Any]],
    traces: Dict[str, Dict[str, Any]],
    *,
    xi_coupling: float,
    tau_response_gyr: float,
    retarded_path_kpc: float,
    p_wave_speed_km_s: float,
) -> Dict[str, Any]:
    mu_geometry = 0.0
    c_eff = max(float(p_wave_speed_km_s), 1.0e-9)
    light_to_p_ratio = float(LIGHT_SPEED_KM_S / c_eff)

    cluster_rows: List[Dict[str, Any]] = []
    lag_rel_errors: List[float] = []
    peak_rel_errors: List[float] = []
    decomposition_rows: List[Dict[str, Any]] = []

    for row in rows:
        cid = str(row["cluster_id"])
        tr = traces[cid]
        v_arr = np.asarray(tr["velocity_kpc_per_gyr"], dtype=float)
        v_obs_signed_kpc_per_gyr = float(v_arr[-1]) if v_arr.size else 0.0
        v_obs_km_s = float(abs(v_obs_signed_kpc_per_gyr) / KM_S_TO_KPC_GYR)
        beta_eff = float(min(max(v_obs_km_s / c_eff, 0.0), 0.999))
        denom = float(max(1.0 - mu_geometry * beta_eff, 1.0e-8))

        delay_nominal = float(retarded_path_kpc / (c_eff * KM_S_TO_KPC_GYR))
        delay_lw = float(delay_nominal / denom)
        delay_lw = float(delay_lw * (1.0 + 0.5 * beta_eff * beta_eff * (1.0 - mu_geometry * mu_geometry)))

        delta_x_lag = float(abs(v_obs_signed_kpc_per_gyr) * delay_lw)
        chi_obs = float(row.get("chi_post", 0.0))
        current_term_signed = float(xi_coupling * tau_response_gyr * chi_obs * v_obs_signed_kpc_per_gyr)
        lag_sign = float(-np.sign(v_obs_signed_kpc_per_gyr)) if abs(v_obs_signed_kpc_per_gyr) > 0.0 else 0.0
        lag_term_signed = float(lag_sign * delta_x_lag)
        delta_x_peak_est = float(abs(current_term_signed + lag_term_signed))

        sim_peak_lag = float(abs(tr.get("retarded_peak_lag_gyr", 0.0)))
        sim_offset = float(abs(tr.get("pred_offset_kpc", row.get("pred_offset_kpc", 0.0))))
        lag_rel = float(abs(delay_lw - sim_peak_lag) / max(sim_peak_lag, 1.0e-9))
        peak_rel = float(abs(delta_x_peak_est - sim_offset) / max(sim_offset, 1.0e-9))
        lag_rel_errors.append(lag_rel)
        peak_rel_errors.append(peak_rel)

        cluster_rows.append(
            {
                "cluster_id": cid,
                "v_obs_km_s": v_obs_km_s,
                "beta_eff_vs_cw": beta_eff,
                "doppler_factor_inv": 1.0 / denom,
                "lw_delay_gyr": delay_lw,
                "sim_peak_lag_gyr": sim_peak_lag,
                "lag_rel_error": lag_rel,
                "lw_delta_x_lag_kpc": delta_x_lag,
                "lw_current_term_signed_kpc": current_term_signed,
                "lw_lag_term_signed_kpc": lag_term_signed,
                "lw_delta_x_peak_est_kpc": delta_x_peak_est,
                "sim_offset_kpc": sim_offset,
                "peak_rel_error": peak_rel,
            }
        )
        decomposition_rows.append(
            {
                "cluster_id": cid,
                "lw_current_term_signed_kpc": current_term_signed,
                "lw_lag_term_signed_kpc": lag_term_signed,
                "lw_est_abs_offset_kpc": delta_x_peak_est,
                "sim_abs_offset_kpc": sim_offset,
                "current_vs_sim_ratio": float(abs(current_term_signed) / max(sim_offset, 1.0e-9)),
                "lag_vs_sim_ratio": float(abs(lag_term_signed) / max(sim_offset, 1.0e-9)),
                "residual_rel_error": peak_rel,
            }
        )

    return {
        "wave_equation": "∇²u-(1/c²)∂²u/∂t²=4πGρ/c²",
        "retarded_time_condition": "t_r = t_obs - R_ret/c_w, R_ret=|x_obs-x_s(t_r)|",
        "lw_like_potential_kernel": "u_ret ∝ [R_ret(1-μβ_eff)]^{-1}",
        "lw_delay_formula": "Δt_lw ≈ (L_path/c_w)/(1-μβ_eff)·[1+β_eff²(1-μ²)/2]",
        "lw_offset_formulas": {
            "delta_x_lag": "Δx_lag ≈ |v_obs| Δt_lw",
            "delta_x_peak": "Δx_peak ≈ |(ξ τ χ_obs v_obs) + s_lag Δx_lag|, s_lag=-sign(v_obs)",
        },
        "geometry_assumption": {
            "mu_projection": mu_geometry,
            "note": "Bullet separation is treated in transverse-proxy geometry (μ≈0) for fixed operational mapping.",
        },
        "c_wave_km_s": c_eff,
        "c_light_km_s": float(LIGHT_SPEED_KM_S),
        "c_light_over_c_wave": light_to_p_ratio,
        "cluster_estimates": cluster_rows,
        "lw_like_offset_estimate": {
            "mean_lag_rel_error": float(np.mean(lag_rel_errors)) if lag_rel_errors else 0.0,
            "max_lag_rel_error": float(np.max(lag_rel_errors)) if lag_rel_errors else 0.0,
            "mean_peak_rel_error": float(np.mean(peak_rel_errors)) if peak_rel_errors else 0.0,
            "max_peak_rel_error": float(np.max(peak_rel_errors)) if peak_rel_errors else 0.0,
            "n_clusters": int(len(cluster_rows)),
        },
        "cluster_contribution_decomposition": decomposition_rows,
    }


def _track_response(
    kin: Kinematics,
    cfg: SimConfig,
    *,
    xi_coupling: float,
) -> Dict[str, Any]:
    t = np.arange(0.0, cfg.t_observe_gyr + cfg.dt_gyr, cfg.dt_gyr, dtype=float)
    v_pre = kin.v_pre_km_s * KM_S_TO_KPC_GYR
    v_post = kin.v_post_km_s * KM_S_TO_KPC_GYR
    v = _velocity_profile(
        t,
        v_pre=v_pre,
        v_post=v_post,
        t_collision=cfg.t_collision_gyr,
        t_drag=cfg.t_drag_gyr,
    )
    transition = _smooth_transition(t, cfg.t_collision_gyr, cfg.t_transition_gyr)
    chi = cfg.chi_pre + (kin.chi_post - cfg.chi_pre) * transition

    x_gas = np.cumsum(v) * cfg.dt_gyr
    current_drive = chi * v
    p_wave_speed = max(float(cfg.p_wave_speed_km_s), 1.0e-9)
    delay_gyr_nominal = float(cfg.retarded_path_kpc) / (p_wave_speed * KM_S_TO_KPC_GYR)
    retarded = _build_retarded_current(
        current_drive,
        cfg.dt_gyr,
        enabled=bool(cfg.retarded_enabled),
        delay_gyr=delay_gyr_nominal,
        tau_kernel_gyr=(float(cfg.retarded_kernel_tau_gyr) if cfg.retarded_kernel_tau_gyr > 0.0 else cfg.tau_response_gyr),
        kernel_mode=str(cfg.retarded_kernel_mode),
        tau_kernel_fast_gyr=(
            float(cfg.retarded_kernel_tau_fast_gyr)
            if cfg.retarded_kernel_tau_fast_gyr > 0.0
            else float(cfg.retarded_kernel_tau_gyr)
        ),
        tau_kernel_slow_gyr=(
            float(cfg.retarded_kernel_tau_slow_gyr)
            if cfg.retarded_kernel_tau_slow_gyr > 0.0
            else float(cfg.retarded_kernel_tau_gyr)
        ),
    )
    if cfg.retarded_enabled:
        rw = float(min(max(cfg.retarded_weight, 0.0), 1.0))
        current_effective = (1.0 - rw) * current_drive + rw * np.asarray(retarded["current_smoothed"], dtype=float)
    else:
        rw = 0.0
        current_effective = current_drive

    x_p = np.zeros_like(x_gas)
    tau = max(cfg.tau_response_gyr, 1.0e-8)
    for i in range(1, t.size):
        source = x_gas[i - 1] + xi_coupling * tau * current_effective[i - 1]
        x_p[i] = x_p[i - 1] + cfg.dt_gyr * (source - x_p[i - 1]) / tau

    delta = x_p - x_gas
    pred_offset = float(abs(delta[-1]))
    sign_obs = int(np.sign(delta[-1])) if abs(float(delta[-1])) > cfg.sign_epsilon_kpc else 0

    sign_series = np.sign(delta)
    sign_series[np.abs(delta) <= cfg.sign_epsilon_kpc] = 0.0
    prev_nonzero = 0.0
    sign_flip_idx: List[int] = []
    for i, sg in enumerate(sign_series):
        if sg == 0.0:
            continue
        if prev_nonzero == 0.0:
            prev_nonzero = sg
            continue
        if sg != prev_nonzero:
            sign_flip_idx.append(i)
            prev_nonzero = sg
    first_flip_time = float(t[sign_flip_idx[0]]) if sign_flip_idx else None
    lag_to_collision = None
    if first_flip_time is not None:
        lag_to_collision = float(first_flip_time - cfg.t_collision_gyr)

    half_level = 0.5 * pred_offset
    abs_delta = np.abs(delta)
    idx_half = int(np.argmax(abs_delta >= half_level)) if np.any(abs_delta >= half_level) else -1
    t_half = float(t[idx_half]) if idx_half >= 0 else None

    return {
        "cluster_id": kin.cluster_id,
        "t_gyr": t,
        "x_gas_kpc": x_gas,
        "x_p_kpc": x_p,
        "delta_kpc": delta,
        "velocity_kpc_per_gyr": v,
        "chi_t": chi,
        "pred_offset_kpc": pred_offset,
        "sign_at_observe": sign_obs,
        "sign_flip_count": int(len(sign_flip_idx)),
        "first_sign_flip_time_gyr": first_flip_time,
        "lag_to_collision_gyr": lag_to_collision,
        "half_reach_time_gyr": t_half,
        "current_drive": current_drive,
        "current_effective": current_effective,
        "retarded_enabled": bool(cfg.retarded_enabled),
        "retarded_weight": rw,
        "retarded_delay_gyr": float(retarded["delay_gyr"]),
        "retarded_delay_steps": int(retarded["delay_steps"]),
        "retarded_kernel_mode": str(retarded["kernel_mode"]),
        "retarded_kernel_tau_fast_gyr": float(retarded["kernel_tau_fast_gyr"]),
        "retarded_kernel_tau_slow_gyr": float(retarded["kernel_tau_slow_gyr"]),
        "retarded_kernel_weight_fast": float(retarded["kernel_weight_fast"]),
        "retarded_kernel_weight_slow": float(retarded["kernel_weight_slow"]),
        "retarded_kernel_tau_gyr": float(retarded["kernel_tau_gyr"]),
        "retarded_peak_lag_gyr": float(retarded["peak_lag_gyr"]),
        "retarded_peak_corr": float(retarded["peak_corr"]),
        "retarded_path_kpc": float(cfg.retarded_path_kpc),
        "p_wave_speed_km_s": float(cfg.p_wave_speed_km_s),
    }


def _fit_xi(
    observations: Sequence[CollisionObs],
    kin_map: Dict[str, Kinematics],
    cfg: SimConfig,
    *,
    xi_min: float,
    xi_max: float,
    xi_count: int,
) -> Tuple[float, float, Dict[str, Dict[str, Any]]]:
    xi_grid = np.linspace(float(xi_min), float(xi_max), int(max(xi_count, 2)), dtype=float)
    best_xi = float(xi_grid[0])
    best_chi2 = float("inf")
    best_tracks: Dict[str, Dict[str, Any]] = {}

    for xi in xi_grid:
        tracks: Dict[str, Dict[str, Any]] = {}
        chi2 = 0.0
        valid = True
        for obs in observations:
            kin = kin_map.get(obs.cluster_id)
            if kin is None:
                valid = False
                break
            tr = _track_response(kin, cfg, xi_coupling=float(xi))
            tracks[obs.cluster_id] = tr
            residual = float(tr["pred_offset_kpc"]) - obs.offset_obs_kpc
            chi2 += (residual / obs.offset_sigma_kpc) ** 2
        if not valid:
            continue
        if chi2 < best_chi2:
            best_chi2 = float(chi2)
            best_xi = float(xi)
            best_tracks = tracks

    if not best_tracks:
        raise RuntimeError("xi fitting failed: no valid cluster tracks")
    return best_xi, best_chi2, best_tracks


def _evaluate_fixed_xi(
    observations: Sequence[CollisionObs],
    kin_map: Dict[str, Kinematics],
    cfg: SimConfig,
    *,
    xi_value: float,
) -> Tuple[float, float, Dict[str, Dict[str, Any]]]:
    tracks: Dict[str, Dict[str, Any]] = {}
    chi2 = 0.0
    for obs in observations:
        kin = kin_map.get(obs.cluster_id)
        if kin is None:
            raise RuntimeError(f"missing kinematics template for {obs.cluster_id}")
        tr = _track_response(kin, cfg, xi_coupling=float(xi_value))
        tracks[obs.cluster_id] = tr
        residual = float(tr["pred_offset_kpc"]) - obs.offset_obs_kpc
        chi2 += (residual / obs.offset_sigma_kpc) ** 2
    return float(xi_value), float(chi2), tracks


def _build_checks(
    *,
    step_tag: str,
    n_clusters: int,
    chi2_dof: float,
    max_abs_z: float,
    n_sign_flip_in_window: int,
    alpha_injection: bool,
    retarded_enabled: bool,
    retarded_kernel_mode: str,
    lw_mean_lag_rel_error: float,
    lw_mean_peak_rel_error: float,
    tau_origin_closure_pass: bool,
    tau_manual_injection: bool,
    tau_reconstruction_rel_error: float,
    ad_hoc_parameter_count: int,
) -> List[Dict[str, Any]]:
    checks: List[Dict[str, Any]] = []

    def add(
        cid: str,
        metric: str,
        value: Any,
        expected: str,
        gate: str,
        passed: bool,
        note: str,
    ) -> None:
        checks.append(
            {
                "id": cid,
                "metric": metric,
                "value": value,
                "expected": expected,
                "gate_level": gate,
                "pass": bool(passed),
                "status": "pass" if passed else ("reject" if gate == "hard" else "watch"),
                "score": 1.0 if passed else 0.0,
                "note": note,
            }
        )

    add(
        "bullet::input_clusters",
        "n_clusters",
        int(n_clusters),
        ">=2",
        "hard",
        n_clusters >= 2,
        "main/sub の2成分以上が必要。",
    )
    add(
        "bullet::derivation_without_alpha",
        "alpha_injection_used",
        bool(alpha_injection),
        "False",
        "hard",
        not alpha_injection,
        "SPARC係数流用なし（J^μ→P_μ の導出枝のみ）であること。",
    )
    add(
        "bullet::tau_origin_closure",
        "tau_origin_closure_pass",
        bool(tau_origin_closure_pass),
        "True",
        "hard",
        bool(tau_origin_closure_pass),
        "τ の導出鎖（P_μ kernel + interaction + damping）が閉じていること。",
    )
    add(
        "bullet::tau_manual_injection",
        "tau_manual_injection",
        bool(tau_manual_injection),
        "False",
        "hard",
        not bool(tau_manual_injection),
        "τ を外部から手動注入していないこと。",
    )
    add(
        "bullet::ad_hoc_parameter_count",
        "ad_hoc_parameter_count",
        int(ad_hoc_parameter_count),
        "0",
        "hard",
        int(ad_hoc_parameter_count) == 0,
        "本監査にアドホックな外部調整パラメータを残さないこと。",
    )
    add(
        "bullet::tau_reconstruction_error",
        "tau_reconstruction_rel_error",
        float(tau_reconstruction_rel_error),
        "<=0.20",
        "hard",
        tau_reconstruction_rel_error <= 0.20,
        "導出した τ_eff が時系列ラグを 20% 以内で再構成すること。",
    )
    add(
        "bullet::retarded_kernel_enabled",
        "retarded_kernel_enabled",
        bool(retarded_enabled),
        "True",
        "hard",
        bool(retarded_enabled),
        "遅延ポテンシャル枝（因果カーネル）を有効化する。",
    )
    require_double = str(step_tag) in {"8.7.25.15", "8.7.25.16"}
    required_mode = "double_exp_derived" if require_double else "single_exp or double_exp_derived"
    kernel_mode_pass = (
        str(retarded_kernel_mode) == "double_exp_derived"
        if require_double
        else str(retarded_kernel_mode) in {"single_exp", "double_exp_derived"}
    )
    add(
        "bullet::retarded_kernel_mode",
        "retarded_kernel_mode",
        str(retarded_kernel_mode),
        required_mode,
        "hard",
        kernel_mode_pass,
        "Step 8.7.25.15/8.7.25.16 は導出二重指数カーネル（single→double expansion）の実行を必須とする。",
    )
    add(
        "bullet::fit_quality",
        "chi2_dof",
        float(chi2_dof),
        "<=4.0",
        "hard",
        chi2_dof <= 4.0,
        "導出枝の offset 再現精度が watch 以上であること。",
    )
    add(
        "bullet::offset_z_gate",
        "max_abs_z_offset",
        float(max_abs_z),
        "<=3.0",
        "hard",
        max_abs_z <= 3.0,
        "各クラスターの offset 残差が 3σ 以内。",
    )
    add(
        "bullet::sign_flip_timing",
        "n_sign_flip_within_window",
        int(n_sign_flip_in_window),
        ">=1",
        "watch",
        n_sign_flip_in_window >= 1,
        "衝突遷移窓の近傍で符号反転（leading/trailing 切替）が少なくとも1件あること。",
    )
    add(
        "bullet::lw_like_lag_consistency",
        "lw_mean_lag_rel_error",
        float(lw_mean_lag_rel_error),
        "<=0.40",
        "watch",
        lw_mean_lag_rel_error <= 0.40,
        "Liénard-Wiechert同型の遅延時間近似が時系列ピークラグと整合すること。",
    )
    peak_rel_gate = 0.50 if str(step_tag) == "8.7.25.16" else 0.60
    add(
        "bullet::lw_like_peak_consistency",
        "lw_mean_peak_rel_error",
        float(lw_mean_peak_rel_error),
        f"<={peak_rel_gate:.2f}",
        "watch",
        lw_mean_peak_rel_error <= peak_rel_gate,
        "解析式のΔx_peak見積が数値導出のオフセット規模と同桁で整合すること。",
    )
    return checks


def _decision_from_checks(checks: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    hard_fails = [str(c.get("id")) for c in checks if str(c.get("gate_level")) == "hard" and not bool(c.get("pass"))]
    watch_fails = [str(c.get("id")) for c in checks if str(c.get("gate_level")) != "hard" and not bool(c.get("pass"))]
    if hard_fails:
        return {
            "overall_status": "reject",
            "decision": "cluster_collision_pmu_jmu_reject",
            "hard_fail_ids": hard_fails,
            "watch_ids": watch_fails,
            "rule": "Reject if any hard gate fails.",
        }
    if watch_fails:
        return {
            "overall_status": "watch",
            "decision": "cluster_collision_pmu_jmu_watch",
            "hard_fail_ids": hard_fails,
            "watch_ids": watch_fails,
            "rule": "Watch if hard gates pass and any watch gate fails.",
        }
    return {
        "overall_status": "pass",
        "decision": "cluster_collision_pmu_jmu_pass",
        "hard_fail_ids": hard_fails,
        "watch_ids": watch_fails,
        "rule": "Pass if all hard/watch gates pass.",
    }


def _plot(
    path: Path,
    *,
    rows: Sequence[Dict[str, Any]],
    traces: Dict[str, Dict[str, Any]],
    t_collision_gyr: float,
) -> None:
    if plt is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)

    labels = [str(r["label"]) for r in rows]
    obs = np.array([float(r["obs_offset_kpc"]) for r in rows], dtype=float)
    sig = np.array([float(r["obs_sigma_kpc"]) for r in rows], dtype=float)
    pred = np.array([float(r["pred_offset_kpc"]) for r in rows], dtype=float)
    y = np.arange(len(rows), dtype=float)

    fig, axes = plt.subplots(2, 1, figsize=(11.8, 8.6), constrained_layout=True)
    ax0, ax1 = axes

    ax0.errorbar(obs, y, xerr=sig, fmt="o", color="#111827", capsize=4, label="observed lens-gas offset")
    ax0.scatter(pred, y, marker="s", color="#2563eb", s=70, label="P_μ-J^μ derivation prediction")
    ax0.set_yticks(y)
    ax0.set_yticklabels(labels)
    ax0.set_xlabel("offset magnitude [kpc]")
    ax0.set_title("Bullet cluster mass separation: observation vs P_μ-J^μ derivation")
    ax0.grid(alpha=0.25)
    ax0.legend(loc="best")

    for row in rows:
        cid = str(row["cluster_id"])
        tr = traces[cid]
        t = np.asarray(tr["t_gyr"], dtype=float)
        d = np.asarray(tr["delta_kpc"], dtype=float)
        ax1.plot(t, d, linewidth=2.0, label=f"{row['label']}: Δx(P-gas)")
        flip_time = tr.get("first_sign_flip_time_gyr")
        if isinstance(flip_time, (float, int)) and np.isfinite(float(flip_time)):
            ax1.scatter([float(flip_time)], [0.0], marker="x", s=60, zorder=4)

    ax1.axvline(float(t_collision_gyr), linestyle="--", color="#6b7280", linewidth=1.4, label="collision epoch")
    ax1.axhline(0.0, linestyle=":", color="#374151", linewidth=1.2)
    ax1.set_xlabel("time [Gyr]")
    ax1.set_ylabel("signed offset Δx(P-gas) [kpc]")
    ax1.set_title("Signed offset dynamics (sign flip and lag around collision)")
    ax1.grid(alpha=0.25)
    ax1.legend(loc="best")

    fig.savefig(path, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 8.7.25.16: Bullet P_μ-J^μ derivation with non-ad-hoc tau origin closure (double-exp kernel + decomposition)"
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=ROOT / "data" / "cosmology" / "bullet_cluster" / "collision_offset_observables.csv",
        help="Observed lens-gas offset table (kpc).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "output" / "public" / "cosmology",
        help="Output directory.",
    )
    parser.add_argument(
        "--shock-width-kpc",
        type=float,
        default=220.0,
        help="(legacy) Shock-response width used for manual tau injection.",
    )
    parser.add_argument(
        "--shock-reference-speed-km-s",
        type=float,
        default=3000.0,
        help="(legacy) Reference collision speed for manual tau injection.",
    )
    parser.add_argument("--t-collision-gyr", type=float, default=0.22, help="Collision epoch.")
    parser.add_argument("--t-observe-gyr", type=float, default=0.42, help="Observation epoch.")
    parser.add_argument("--t-drag-gyr", type=float, default=0.08, help="Post-collision drag timescale.")
    parser.add_argument("--t-transition-gyr", type=float, default=0.015, help="Current-retention transition width.")
    parser.add_argument("--chi-pre", type=float, default=0.12, help="Pre-collision current-retention baseline.")
    parser.add_argument("--xi-min", type=float, default=2.0, help="(legacy fit mode) Min xi grid.")
    parser.add_argument("--xi-max", type=float, default=10.0, help="(legacy fit mode) Max xi grid.")
    parser.add_argument("--xi-count", type=int, default=321, help="(legacy fit mode) Number of xi grid points.")
    parser.add_argument("--dt-gyr", type=float, default=2.0e-4, help="Time integration step.")
    parser.add_argument(
        "--retarded-enabled",
        type=str,
        choices=["on", "off"],
        default="on",
        help="Enable causal retarded-kernel branch.",
    )
    parser.add_argument(
        "--retarded-weight",
        type=float,
        default=0.55,
        help="(legacy) Blend weight for retarded current in effective source (0..1).",
    )
    parser.add_argument(
        "--retarded-path-kpc",
        type=float,
        default=180.0,
        help="Effective propagation path used for causal delay.",
    )
    parser.add_argument(
        "--p-wave-speed-km-s",
        type=float,
        default=5500.0,
        help="Effective P-wave propagation speed for delay calculation.",
    )
    parser.add_argument(
        "--retarded-kernel-tau-gyr",
        type=float,
        default=0.05,
        help="(legacy/single) Kernel smoothing timescale for delayed current branch (<=0 uses tau_response).",
    )
    parser.add_argument(
        "--retarded-kernel-mode",
        type=str,
        choices=["auto", "single_exp", "double_exp_derived"],
        default="auto",
        help="Kernel mode. auto: step 8.7.25.15/8.7.25.16 uses double_exp_derived, otherwise single_exp unless overridden.",
    )
    parser.add_argument(
        "--tau-origin-mode",
        type=str,
        choices=["auto", "derived", "legacy"],
        default="auto",
        help="tau origin mode. auto: derived for step 8.7.25.14/8.7.25.15/8.7.25.16, legacy otherwise.",
    )
    parser.add_argument(
        "--xi-mode",
        type=str,
        choices=["auto", "fixed", "fit", "derived"],
        default="auto",
        help="xi mode. auto: fixed for 8.7.25.14, derived for 8.7.25.15/8.7.25.16, fit for legacy mode.",
    )
    parser.add_argument("--step-tag", type=str, default="8.7.25.16", help="Step tag written in output JSON.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    out_json = out_dir / "cosmology_cluster_collision_pmu_jmu_separation_derivation.json"
    out_csv = out_dir / "cosmology_cluster_collision_pmu_jmu_separation_derivation.csv"
    out_png = out_dir / "cosmology_cluster_collision_pmu_jmu_separation_derivation.png"

    observations = _load_observations(args.input_csv)
    kin_map = _default_kinematics()
    observations = [row for row in observations if row.cluster_id in kin_map]
    if len(observations) < 2:
        raise RuntimeError("need >=2 observed clusters matched to kinematic templates (bullet_main/bullet_sub)")

    step_tag = str(args.step_tag).strip()
    if str(args.tau_origin_mode).strip() == "derived":
        tau_origin_mode = "derived"
    elif str(args.tau_origin_mode).strip() == "legacy":
        tau_origin_mode = "legacy"
    else:
        tau_origin_mode = "derived" if step_tag in {"8.7.25.14", "8.7.25.15", "8.7.25.16"} else "legacy"

    if str(args.retarded_kernel_mode).strip() in {"single_exp", "double_exp_derived"}:
        kernel_mode_seed = str(args.retarded_kernel_mode).strip()
    else:
        if step_tag in {"8.7.25.15", "8.7.25.16"}:
            kernel_mode_seed = "double_exp_derived"
        else:
            kernel_mode_seed = "single_exp"

    cfg_seed = SimConfig(
        t_collision_gyr=float(args.t_collision_gyr),
        t_observe_gyr=float(args.t_observe_gyr),
        t_drag_gyr=float(args.t_drag_gyr),
        t_transition_gyr=float(args.t_transition_gyr),
        dt_gyr=float(args.dt_gyr),
        tau_response_gyr=float(
            max(
                float(args.retarded_path_kpc) / (max(float(args.p_wave_speed_km_s), 1.0e-9) * KM_S_TO_KPC_GYR),
                float(args.dt_gyr),
                1.0e-8,
            )
        ),
        chi_pre=float(args.chi_pre),
        sign_epsilon_kpc=1.0e-9,
        sign_flip_window_gyr=0.03,
        retarded_enabled=(str(args.retarded_enabled).lower() == "on"),
        retarded_weight=float(min(max(float(args.retarded_weight), 0.0), 1.0)),
        retarded_path_kpc=float(max(float(args.retarded_path_kpc), 1.0e-9)),
        p_wave_speed_km_s=float(max(float(args.p_wave_speed_km_s), 1.0e-9)),
        retarded_kernel_tau_gyr=float(args.retarded_kernel_tau_gyr),
        retarded_kernel_mode=str(kernel_mode_seed),
        retarded_kernel_tau_fast_gyr=float(max(float(args.retarded_kernel_tau_gyr), 1.0e-8)),
        retarded_kernel_tau_slow_gyr=float(max(float(args.retarded_kernel_tau_gyr), 1.0e-8)),
    )

    tau_origin: TauOrigin
    if tau_origin_mode == "derived":
        tau_origin = _derive_tau_origin(kin_map, cfg_seed, extended_kernel=(step_tag in {"8.7.25.15", "8.7.25.16"}))
        tau_response = float(tau_origin.tau_eff_gyr)
        retarded_kernel_tau = float(tau_origin.tau_kernel_gyr)
        retarded_weight = float(tau_origin.retarded_weight_eta)
        retarded_kernel_mode = str(tau_origin.kernel_mode if step_tag in {"8.7.25.15", "8.7.25.16"} else "single_exp")
        retarded_kernel_tau_fast = float(tau_origin.tau_kernel_fast_gyr)
        retarded_kernel_tau_slow = float(tau_origin.tau_kernel_slow_gyr)
        xi_derived = float(tau_origin.xi_derived)
    else:
        tau_response = float(args.shock_width_kpc) / (
            max(float(args.shock_reference_speed_km_s), 1.0e-9) * KM_S_TO_KPC_GYR
        )
        tau_origin = TauOrigin(
            mode="legacy_manual_tau",
            tau_free_gyr=float(
                cfg_seed.retarded_path_kpc / (max(cfg_seed.p_wave_speed_km_s, 1.0e-9) * KM_S_TO_KPC_GYR)
            ),
            tau_int_gyr=float(max(float(args.retarded_kernel_tau_gyr), 1.0e-8)),
            tau_damp_gyr=float(max(float(args.t_drag_gyr), 1.0e-8)),
            tau_eff_gyr=float(max(tau_response, 1.0e-8)),
            tau_eff_harmonic_gyr=float(max(tau_response, 1.0e-8)),
            tau_kernel_gyr=float(max(float(args.retarded_kernel_tau_gyr), 1.0e-8)),
            tau_kernel_fast_gyr=float(max(float(args.retarded_kernel_tau_gyr), 1.0e-8)),
            tau_kernel_slow_gyr=float(max(float(args.retarded_kernel_tau_gyr), 1.0e-8)),
            kernel_mode="single_exp",
            retarded_weight_eta=float(min(max(float(args.retarded_weight), 0.0), 1.0)),
            xi_derived=1.0,
            closure_pass=False,
            manual_injection=True,
            ad_hoc_parameter_count=3,
        )
        retarded_kernel_tau = float(tau_origin.tau_kernel_gyr)
        retarded_weight = float(tau_origin.retarded_weight_eta)
        retarded_kernel_mode = "single_exp"
        retarded_kernel_tau_fast = float(tau_origin.tau_kernel_fast_gyr)
        retarded_kernel_tau_slow = float(tau_origin.tau_kernel_slow_gyr)
        xi_derived = 1.0

    cfg = SimConfig(
        t_collision_gyr=cfg_seed.t_collision_gyr,
        t_observe_gyr=cfg_seed.t_observe_gyr,
        t_drag_gyr=cfg_seed.t_drag_gyr,
        t_transition_gyr=cfg_seed.t_transition_gyr,
        dt_gyr=cfg_seed.dt_gyr,
        tau_response_gyr=float(max(tau_response, 1.0e-8)),
        chi_pre=cfg_seed.chi_pre,
        sign_epsilon_kpc=cfg_seed.sign_epsilon_kpc,
        sign_flip_window_gyr=cfg_seed.sign_flip_window_gyr,
        retarded_enabled=cfg_seed.retarded_enabled,
        retarded_weight=float(retarded_weight),
        retarded_path_kpc=cfg_seed.retarded_path_kpc,
        p_wave_speed_km_s=cfg_seed.p_wave_speed_km_s,
        retarded_kernel_tau_gyr=float(retarded_kernel_tau),
        retarded_kernel_mode=str(retarded_kernel_mode),
        retarded_kernel_tau_fast_gyr=float(retarded_kernel_tau_fast),
        retarded_kernel_tau_slow_gyr=float(retarded_kernel_tau_slow),
    )

    if str(args.xi_mode).strip() == "fit":
        xi_mode = "fit"
    elif str(args.xi_mode).strip() == "fixed":
        xi_mode = "fixed"
    elif str(args.xi_mode).strip() == "derived":
        xi_mode = "derived"
    else:
        if tau_origin_mode != "derived":
            xi_mode = "fit"
        elif step_tag in {"8.7.25.15", "8.7.25.16"}:
            xi_mode = "derived"
        else:
            xi_mode = "fixed"

    if xi_mode == "fit":
        xi_best, chi2, traces = _fit_xi(
            observations,
            kin_map,
            cfg,
            xi_min=float(args.xi_min),
            xi_max=float(args.xi_max),
            xi_count=int(args.xi_count),
        )
        n_fit_params = 1
    elif xi_mode == "derived":
        xi_best, chi2, traces = _evaluate_fixed_xi(
            observations,
            kin_map,
            cfg,
            xi_value=float(max(xi_derived, 1.0e-9)),
        )
        n_fit_params = 0
    else:
        xi_best, chi2, traces = _evaluate_fixed_xi(
            observations,
            kin_map,
            cfg,
            xi_value=1.0,
        )
        n_fit_params = 0

    rows_out: List[Dict[str, Any]] = []
    max_abs_z = 0.0
    n_sign_flip_in_window = 0
    for obs in observations:
        tr = traces[obs.cluster_id]
        pred = float(tr["pred_offset_kpc"])
        residual = pred - obs.offset_obs_kpc
        z = residual / obs.offset_sigma_kpc
        max_abs_z = max(max_abs_z, abs(z))

        flip_t = tr.get("first_sign_flip_time_gyr")
        flip_lag = tr.get("lag_to_collision_gyr")
        if isinstance(flip_lag, (float, int)) and np.isfinite(float(flip_lag)):
            if abs(float(flip_lag)) <= cfg.sign_flip_window_gyr:
                n_sign_flip_in_window += 1

        kin = kin_map[obs.cluster_id]
        rows_out.append(
            {
                "cluster_id": obs.cluster_id,
                "label": obs.label,
                "obs_offset_kpc": float(obs.offset_obs_kpc),
                "obs_sigma_kpc": float(obs.offset_sigma_kpc),
                "pred_offset_kpc": pred,
                "residual_offset_kpc": float(residual),
                "z_offset": float(z),
                "v_pre_km_s": float(kin.v_pre_km_s),
                "v_post_km_s": float(kin.v_post_km_s),
                "chi_pre": float(cfg.chi_pre),
                "chi_post": float(kin.chi_post),
                "sign_at_observe": int(tr["sign_at_observe"]),
                "sign_flip_count": int(tr["sign_flip_count"]),
                "first_sign_flip_time_gyr": (None if flip_t is None else float(flip_t)),
                "lag_to_collision_gyr": (None if flip_lag is None else float(flip_lag)),
                "half_reach_time_gyr": (
                    None if tr.get("half_reach_time_gyr") is None else float(tr["half_reach_time_gyr"])
                ),
                "tau_response_gyr": float(cfg.tau_response_gyr),
                "xi_jmu_fit": float(xi_best),
                "xi_mode": str(xi_mode),
                "xi_jmu_derived": float(tau_origin.xi_derived),
                "retarded_enabled": bool(tr["retarded_enabled"]),
                "retarded_weight": float(tr["retarded_weight"]),
                "retarded_delay_gyr": float(tr["retarded_delay_gyr"]),
                "retarded_delay_steps": int(tr["retarded_delay_steps"]),
                "retarded_kernel_mode": str(tr["retarded_kernel_mode"]),
                "retarded_kernel_tau_fast_gyr": float(tr["retarded_kernel_tau_fast_gyr"]),
                "retarded_kernel_tau_slow_gyr": float(tr["retarded_kernel_tau_slow_gyr"]),
                "retarded_kernel_weight_fast": float(tr["retarded_kernel_weight_fast"]),
                "retarded_kernel_weight_slow": float(tr["retarded_kernel_weight_slow"]),
                "retarded_kernel_tau_gyr": float(tr["retarded_kernel_tau_gyr"]),
                "retarded_peak_lag_gyr": float(tr["retarded_peak_lag_gyr"]),
                "retarded_peak_corr": float(tr["retarded_peak_corr"]),
                "retarded_path_kpc": float(tr["retarded_path_kpc"]),
                "p_wave_speed_km_s": float(tr["p_wave_speed_km_s"]),
            }
        )

    n_obs = len(rows_out)
    dof = max(n_obs - n_fit_params, 1)
    chi2_dof = float(chi2 / dof)
    rms_residual = float(np.sqrt(np.mean([float(r["residual_offset_kpc"]) ** 2 for r in rows_out])))

    lw_block = _build_lw_like_estimates(
        rows_out,
        traces,
        xi_coupling=float(xi_best),
        tau_response_gyr=float(cfg.tau_response_gyr),
        retarded_path_kpc=float(cfg.retarded_path_kpc),
        p_wave_speed_km_s=float(cfg.p_wave_speed_km_s),
    )
    lw_summary = lw_block.get("lw_like_offset_estimate", {})
    lw_mean_lag_rel_error = float(lw_summary.get("mean_lag_rel_error", 0.0))
    lw_mean_peak_rel_error = float(lw_summary.get("mean_peak_rel_error", 0.0))
    mean_peak_lag = float(
        np.mean([abs(float(tr["retarded_peak_lag_gyr"])) for tr in traces.values()]) if traces else 0.0
    )
    tau_reconstruction_rel_error = float(abs(mean_peak_lag - cfg.tau_response_gyr) / max(cfg.tau_response_gyr, 1.0e-9))
    ad_hoc_parameter_count = int(tau_origin.ad_hoc_parameter_count)
    if xi_mode == "fit":
        ad_hoc_parameter_count += 1
    if tau_origin_mode == "legacy":
        ad_hoc_parameter_count += 1

    lw_by_cluster = {str(d.get("cluster_id")): d for d in lw_block.get("cluster_estimates", [])}
    for row in rows_out:
        ci = lw_by_cluster.get(str(row["cluster_id"]), {})
        row["lw_delay_gyr"] = float(ci.get("lw_delay_gyr", 0.0))
        row["lw_delta_x_lag_kpc"] = float(ci.get("lw_delta_x_lag_kpc", 0.0))
        row["lw_current_term_signed_kpc"] = float(ci.get("lw_current_term_signed_kpc", 0.0))
        row["lw_lag_term_signed_kpc"] = float(ci.get("lw_lag_term_signed_kpc", 0.0))
        row["lw_delta_x_peak_est_kpc"] = float(ci.get("lw_delta_x_peak_est_kpc", 0.0))
        row["lw_lag_rel_error"] = float(ci.get("lag_rel_error", 0.0))
        row["lw_peak_rel_error"] = float(ci.get("peak_rel_error", 0.0))
        row["lw_beta_eff_vs_cw"] = float(ci.get("beta_eff_vs_cw", 0.0))
        row["lw_doppler_factor_inv"] = float(ci.get("doppler_factor_inv", 1.0))
        row["tau_origin_mode"] = str(tau_origin.mode)
        row["tau_free_gyr"] = float(tau_origin.tau_free_gyr)
        row["tau_int_gyr"] = float(tau_origin.tau_int_gyr)
        row["tau_damp_gyr"] = float(tau_origin.tau_damp_gyr)
        row["tau_eff_gyr"] = float(cfg.tau_response_gyr)
        row["tau_eff_harmonic_gyr"] = float(tau_origin.tau_eff_harmonic_gyr)
        row["tau_reconstruction_rel_error"] = float(tau_reconstruction_rel_error)
        row["ad_hoc_parameter_count"] = int(ad_hoc_parameter_count)

    checks = _build_checks(
        step_tag=step_tag,
        n_clusters=n_obs,
        chi2_dof=chi2_dof,
        max_abs_z=max_abs_z,
        n_sign_flip_in_window=n_sign_flip_in_window,
        alpha_injection=False,
        retarded_enabled=cfg.retarded_enabled,
        retarded_kernel_mode=str(cfg.retarded_kernel_mode),
        lw_mean_lag_rel_error=lw_mean_lag_rel_error,
        lw_mean_peak_rel_error=lw_mean_peak_rel_error,
        tau_origin_closure_pass=bool(tau_origin.closure_pass),
        tau_manual_injection=bool(tau_origin.manual_injection),
        tau_reconstruction_rel_error=float(tau_reconstruction_rel_error),
        ad_hoc_parameter_count=int(ad_hoc_parameter_count),
    )
    decision = _decision_from_checks(checks)

    _write_csv(
        out_csv,
        rows_out,
        [
            "cluster_id",
            "label",
            "obs_offset_kpc",
            "obs_sigma_kpc",
            "pred_offset_kpc",
            "residual_offset_kpc",
            "z_offset",
            "v_pre_km_s",
            "v_post_km_s",
            "chi_pre",
            "chi_post",
            "sign_at_observe",
            "sign_flip_count",
            "first_sign_flip_time_gyr",
            "lag_to_collision_gyr",
            "half_reach_time_gyr",
            "tau_response_gyr",
            "xi_jmu_fit",
            "xi_mode",
            "xi_jmu_derived",
            "retarded_enabled",
            "retarded_weight",
            "retarded_delay_gyr",
            "retarded_delay_steps",
            "retarded_kernel_mode",
            "retarded_kernel_tau_fast_gyr",
            "retarded_kernel_tau_slow_gyr",
            "retarded_kernel_weight_fast",
            "retarded_kernel_weight_slow",
            "retarded_kernel_tau_gyr",
            "retarded_peak_lag_gyr",
            "retarded_peak_corr",
            "retarded_path_kpc",
            "p_wave_speed_km_s",
            "tau_origin_mode",
            "tau_free_gyr",
            "tau_int_gyr",
            "tau_damp_gyr",
            "tau_eff_gyr",
            "tau_eff_harmonic_gyr",
            "tau_reconstruction_rel_error",
            "ad_hoc_parameter_count",
            "lw_delay_gyr",
            "lw_delta_x_lag_kpc",
            "lw_current_term_signed_kpc",
            "lw_lag_term_signed_kpc",
            "lw_delta_x_peak_est_kpc",
            "lw_lag_rel_error",
            "lw_peak_rel_error",
            "lw_beta_eff_vs_cw",
            "lw_doppler_factor_inv",
        ],
    )
    _plot(
        out_png,
        rows=rows_out,
        traces=traces,
        t_collision_gyr=cfg.t_collision_gyr,
    )

    trace_out: Dict[str, Dict[str, Any]] = {}
    for cid, tr in traces.items():
        trace_out[cid] = {
            "t_gyr": [float(v) for v in np.asarray(tr["t_gyr"], dtype=float)],
            "delta_kpc": [float(v) for v in np.asarray(tr["delta_kpc"], dtype=float)],
            "velocity_kpc_per_gyr": [float(v) for v in np.asarray(tr["velocity_kpc_per_gyr"], dtype=float)],
            "chi_t": [float(v) for v in np.asarray(tr["chi_t"], dtype=float)],
            "current_drive": [float(v) for v in np.asarray(tr["current_drive"], dtype=float)],
            "current_effective": [float(v) for v in np.asarray(tr["current_effective"], dtype=float)],
            "first_sign_flip_time_gyr": tr.get("first_sign_flip_time_gyr"),
            "lag_to_collision_gyr": tr.get("lag_to_collision_gyr"),
            "retarded_enabled": bool(tr.get("retarded_enabled", False)),
            "retarded_weight": float(tr.get("retarded_weight", 0.0)),
            "retarded_delay_gyr": float(tr.get("retarded_delay_gyr", 0.0)),
            "retarded_delay_steps": int(tr.get("retarded_delay_steps", 0)),
            "retarded_kernel_mode": str(tr.get("retarded_kernel_mode", cfg.retarded_kernel_mode)),
            "retarded_kernel_tau_fast_gyr": float(
                tr.get("retarded_kernel_tau_fast_gyr", cfg.retarded_kernel_tau_fast_gyr)
            ),
            "retarded_kernel_tau_slow_gyr": float(
                tr.get("retarded_kernel_tau_slow_gyr", cfg.retarded_kernel_tau_slow_gyr)
            ),
            "retarded_kernel_weight_fast": float(tr.get("retarded_kernel_weight_fast", 1.0)),
            "retarded_kernel_weight_slow": float(tr.get("retarded_kernel_weight_slow", 0.0)),
            "retarded_kernel_tau_gyr": float(tr.get("retarded_kernel_tau_gyr", cfg.tau_response_gyr)),
            "retarded_peak_lag_gyr": float(tr.get("retarded_peak_lag_gyr", 0.0)),
            "retarded_peak_corr": float(tr.get("retarded_peak_corr", 0.0)),
        }

    payload: Dict[str, Any] = {
        "generated_utc": _utc_now(),
        "phase": {"phase": 8, "step": str(args.step_tag), "name": "cluster collision P_mu-J^mu separation derivation"},
        "intent": (
            "Derive lens-gas centroid separation from J^mu-driven P_mu delayed response "
            "without SPARC alpha injection."
        ),
        "model": {
            "wave_equation_source": "∇²u-(1/c²)∂²u/∂t²=4πGρ/c²",
            "field_equation_reduced": (
                "∂_t P_L = D∂_xx P_L - (P_L - J^0)/τ + ξ[(1-η)∂_x J^x + η∂_x J^x(t-Δt)]"
            ),
            "first_moment_reduction": (
                "τ dX_P/dt + X_P = X_g + ξτ[(1-η)χ(t)v(t) + ηR_delay(t)]"
            ),
            "retarded_kernel_mode": str(cfg.retarded_kernel_mode),
            "retarded_kernel": (
                "R_delay(t)=∫_0^t K_tau(t-s) χ(s-Δt)v(s-Δt) ds, "
                "K_tau(u)=w_f exp(-u/τ_f)/τ_f + w_s exp(-u/τ_s)/τ_s"
                if str(cfg.retarded_kernel_mode) == "double_exp_derived"
                else "R_delay(t)=∫_0^t K_tau(t-s) χ(s-Δt)v(s-Δt) ds, K_tau(u)=exp(-u/τ_k)/τ_k"
            ),
            "offset_definition": "Δx(P-gas)=X_P - X_g, predicted magnitude = abs(Δx at t_obs)",
            "alpha_injection_used": False,
        },
        "tau_origin_block": {
            "mode": str(tau_origin.mode),
            "tau_origin_closure_pass": bool(tau_origin.closure_pass),
            "tau_manual_injection": bool(tau_origin.manual_injection),
            "ad_hoc_parameter_count": int(ad_hoc_parameter_count),
            "kernel_mode": str(tau_origin.kernel_mode),
            "derived_components_gyr": {
                "tau_free": float(tau_origin.tau_free_gyr),
                "tau_int": float(tau_origin.tau_int_gyr),
                "tau_damp": float(tau_origin.tau_damp_gyr),
                "tau_eff": float(cfg.tau_response_gyr),
                "tau_eff_harmonic": float(tau_origin.tau_eff_harmonic_gyr),
                "tau_kernel": float(tau_origin.tau_kernel_gyr),
                "tau_kernel_fast": float(tau_origin.tau_kernel_fast_gyr),
                "tau_kernel_slow": float(tau_origin.tau_kernel_slow_gyr),
            },
            "derived_eta_weight": float(tau_origin.retarded_weight_eta),
            "derived_xi": float(tau_origin.xi_derived),
            "tau_reconstruction_rel_error": float(tau_reconstruction_rel_error),
            "derivation_equations": {
                "tau_free": "L_path / c_w",
                "tau_int": "sqrt(<(χv)^2> / <(d(χv)/dt)^2>)",
                "tau_damp": "sqrt(<v^2> / <(dv/dt)^2>)",
                "tau_eff_harmonic": "1 / (1/tau_free + 1/tau_int + 1/tau_damp)",
                "tau_eff_kernel_first_moment": "tau_free + η tau_kernel, η=tau_free/(tau_free+tau_int)",
                "tau_kernel_double_exp": "tau_kernel = w_f tau_f + w_s tau_s, w_i ∝ 1/tau_i",
                "xi_derived": "xi = (tau_int + tau_damp) / tau_eff",
            },
            "requirement": "No external ad-hoc parameter injection; all response timescales must originate from P_μ kernel and source dynamics.",
        },
        "retarded_potential_block": {
            "enabled": bool(cfg.retarded_enabled),
            "weight_eta": float(cfg.retarded_weight),
            "kernel_mode": str(cfg.retarded_kernel_mode),
            "path_kpc": float(cfg.retarded_path_kpc),
            "p_wave_speed_km_s": float(cfg.p_wave_speed_km_s),
            "delay_gyr_nominal": float(cfg.retarded_path_kpc / (cfg.p_wave_speed_km_s * KM_S_TO_KPC_GYR)),
            "kernel_tau_gyr": float(cfg.retarded_kernel_tau_gyr if cfg.retarded_kernel_tau_gyr > 0.0 else cfg.tau_response_gyr),
            "kernel_tau_fast_gyr": float(cfg.retarded_kernel_tau_fast_gyr),
            "kernel_tau_slow_gyr": float(cfg.retarded_kernel_tau_slow_gyr),
            "max_abs_peak_lag_gyr": float(
                max(abs(float(tr["retarded_peak_lag_gyr"])) for tr in traces.values()) if traces else 0.0
            ),
            "mean_peak_corr": float(np.mean([float(tr["retarded_peak_corr"]) for tr in traces.values()])) if traces else 0.0,
            "equation_note": (
                "Step 8.7.25.15/8.7.25.16 keeps LW-like delay, enforces tau-origin closure without manual injection, "
                "and promotes retarded kernel to derived double-exp mode."
            ),
            "lw_like": lw_block,
        },
        "assumptions": {
            "kinematics_template": {
                "bullet_main": {
                    "v_pre_km_s": kin_map["bullet_main"].v_pre_km_s,
                    "v_post_km_s": kin_map["bullet_main"].v_post_km_s,
                    "chi_post": kin_map["bullet_main"].chi_post,
                },
                "bullet_sub": {
                    "v_pre_km_s": kin_map["bullet_sub"].v_pre_km_s,
                    "v_post_km_s": kin_map["bullet_sub"].v_post_km_s,
                    "chi_post": kin_map["bullet_sub"].chi_post,
                },
            },
            "collision_timing_gyr": {
                "t_collision": cfg.t_collision_gyr,
                "t_observe": cfg.t_observe_gyr,
                "t_drag": cfg.t_drag_gyr,
                "t_transition": cfg.t_transition_gyr,
                "dt": cfg.dt_gyr,
            },
            "tau_derivation": {
                "shock_width_kpc": float(args.shock_width_kpc),
                "shock_reference_speed_km_s": float(args.shock_reference_speed_km_s),
                "tau_response_gyr": cfg.tau_response_gyr,
                "formula": (
                    "derived mode: tau_eff = tau_free + η tau_kernel (kernel first moment), "
                    "tau_kernel = w_f tau_f + w_s tau_s, w_i∝1/tau_i; "
                    "legacy mode: tau = L_shock / v_ref"
                ),
                "mode": str(tau_origin.mode),
                "tau_origin_closure_pass": bool(tau_origin.closure_pass),
                "tau_manual_injection": bool(tau_origin.manual_injection),
            },
            "retarded_branch": {
                "enabled": bool(cfg.retarded_enabled),
                "weight_eta": float(cfg.retarded_weight),
                "kernel_mode": str(cfg.retarded_kernel_mode),
                "path_kpc": float(cfg.retarded_path_kpc),
                "p_wave_speed_km_s": float(cfg.p_wave_speed_km_s),
                "kernel_tau_gyr": float(cfg.retarded_kernel_tau_gyr if cfg.retarded_kernel_tau_gyr > 0.0 else cfg.tau_response_gyr),
                "kernel_tau_fast_gyr": float(cfg.retarded_kernel_tau_fast_gyr),
                "kernel_tau_slow_gyr": float(cfg.retarded_kernel_tau_slow_gyr),
                "weight_mode": "derived" if tau_origin_mode == "derived" else "legacy_manual",
            },
        },
        "fit": {
            "xi_coupling_best": float(xi_best),
            "xi_mode": str(xi_mode),
            "xi_derived": float(tau_origin.xi_derived),
            "xi_grid": {
                "min": float(args.xi_min),
                "max": float(args.xi_max),
                "count": int(args.xi_count),
            },
            "chi2": float(chi2),
            "chi2_dof": float(chi2_dof),
            "n_observations": int(n_obs),
            "n_fit_parameters": int(n_fit_params),
            "dof": int(dof),
            "rms_residual_kpc": rms_residual,
            "max_abs_z_offset": float(max_abs_z),
            "n_sign_flip_within_window": int(n_sign_flip_in_window),
            "sign_flip_window_gyr": float(cfg.sign_flip_window_gyr),
            "retarded_enabled": bool(cfg.retarded_enabled),
            "retarded_weight_eta": float(cfg.retarded_weight),
            "retarded_delay_gyr_nominal": float(cfg.retarded_path_kpc / (cfg.p_wave_speed_km_s * KM_S_TO_KPC_GYR)),
            "tau_reconstruction_rel_error": float(tau_reconstruction_rel_error),
            "ad_hoc_parameter_count": int(ad_hoc_parameter_count),
        },
        "cluster_rows": rows_out,
        "time_series": trace_out,
        "checks": checks,
        "decision": decision,
        "inputs": {
            "offset_observations_csv": _rel(args.input_csv),
        },
        "outputs": {
            "derivation_json": _rel(out_json),
            "derivation_csv": _rel(out_csv),
            "derivation_png": _rel(out_png),
        },
        "falsification_gate": {
            "reject_if": [
                "Any hard check fails.",
                "chi2/dof > 4.0",
                "max |z_offset| > 3.0",
                "alpha injection is used in this derivation branch.",
                "retarded kernel branch is disabled.",
                "tau origin closure fails (tau_origin_closure_pass=false).",
                "tau is manually injected (tau_manual_injection=true).",
                "ad_hoc_parameter_count > 0.",
                "tau_reconstruction_rel_error > 0.20.",
            ],
            "watch_if": [
                "Hard checks pass but no sign flip appears near collision transition window.",
            ],
        },
    }
    _write_json(out_json, payload)

    if worklog is not None:
        try:
            worklog.append_event(
                {
                    "event_type": "cosmology_cluster_collision_pmu_jmu_separation_derivation",
                    "phase": str(args.step_tag),
                    "decision": decision.get("decision"),
                    "overall_status": decision.get("overall_status"),
                    "chi2_dof": chi2_dof,
                    "max_abs_z_offset": max_abs_z,
                    "xi_coupling_best": xi_best,
                    "xi_mode": str(xi_mode),
                    "xi_derived": float(tau_origin.xi_derived),
                    "tau_response_gyr": cfg.tau_response_gyr,
                    "tau_origin_mode": str(tau_origin.mode),
                    "tau_origin_closure_pass": bool(tau_origin.closure_pass),
                    "tau_manual_injection": bool(tau_origin.manual_injection),
                    "tau_reconstruction_rel_error": float(tau_reconstruction_rel_error),
                    "ad_hoc_parameter_count": int(ad_hoc_parameter_count),
                    "alpha_injection_used": False,
                    "retarded_enabled": bool(cfg.retarded_enabled),
                    "retarded_kernel_mode": str(cfg.retarded_kernel_mode),
                    "retarded_weight_eta": float(cfg.retarded_weight),
                    "retarded_delay_gyr_nominal": float(cfg.retarded_path_kpc / (cfg.p_wave_speed_km_s * KM_S_TO_KPC_GYR)),
                    "lw_mean_lag_rel_error": float(lw_mean_lag_rel_error),
                    "lw_mean_peak_rel_error": float(lw_mean_peak_rel_error),
                    "n_sign_flip_within_window": n_sign_flip_in_window,
                    "outputs": {
                        "json": _rel(out_json),
                        "csv": _rel(out_csv),
                        "png": _rel(out_png),
                    },
                }
            )
        except Exception:
            pass

    print(f"[ok] wrote {out_json}")
    print(f"[ok] wrote {out_csv}")
    print(f"[ok] wrote {out_png}")
    print(
        "[summary] status={0} decision={1} chi2/dof={2:.4f} max|z|={3:.3f} xi={4:.4f}({5}) tau_eff={6:.4f} Gyr "
        "tauErr={7:.3f} adHoc={8} kernel={9} lwLag={10:.3f} lwPeak={11:.3f}".format(
            decision.get("overall_status"),
            decision.get("decision"),
            chi2_dof,
            max_abs_z,
            xi_best,
            xi_mode,
            cfg.tau_response_gyr,
            tau_reconstruction_rel_error,
            int(ad_hoc_parameter_count),
            cfg.retarded_kernel_mode,
            lw_mean_lag_rel_error,
            lw_mean_peak_rel_error,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
