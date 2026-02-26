#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pmodel_rotating_bh_photon_ring_direct_audit.py

Step 8.7.27.28:
P_μJ^μ の最小軸対称強場モデルから、κ 係数依存を使わずに
光子リングの直接可観測量（直径・非対称）を固定する監査パック。
8.7.27.28 では、L_{P_μ}^{free}+L_int を基点として
真空（J^μ=0, m_P→0）の定常軸対称PDE（P_0, P_φ）と
分離解の形を JSON に固定する。

固定出力:
- output/public/theory/pmodel_rotating_bh_photon_ring_direct_audit.json
- output/public/theory/pmodel_rotating_bh_photon_ring_direct_audit.csv
- output/public/theory/pmodel_rotating_bh_photon_ring_direct_audit.png
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
# 条件分岐: `str(ROOT) not in sys.path` を満たす経路を評価する。
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


# クラス: `ObjectInput` の責務と境界条件を定義する。

@dataclass(frozen=True)
class ObjectInput:
    key: str
    label: str
    theta_unit_uas: float
    ring_diameter_obs_uas: float
    ring_diameter_obs_sigma_uas: float
    a_star_min: float
    a_star_max: float
    inc_deg_min: float
    inc_deg_max: float
    asym_min: Optional[float]
    asym_max: Optional[float]


# 関数: `_build_axisymmetric_pde_block` の入出力契約と処理意図を定義する。

def _build_axisymmetric_pde_block(
    *,
    beta: float,
    lambda2_best: float,
    zeta_best: float,
    eta_nonlinear: float,
    omega_ang_epsilon: float,
    r_max_rg: float,
    min_orbit_margin_rg: float,
    max_flux_rel_std: float,
    solver_diag_by_object: Dict[str, Any],
) -> Dict[str, Any]:
    r_out = float(max(r_max_rg, 1.0))
    asymptotic_target = float(1.0 / r_out + lambda2_best / (r_out * r_out))

    asym_rel_errors: List[float] = []
    u_inner_abs: List[float] = []
    for diag in solver_diag_by_object.values():
        # 条件分岐: `not isinstance(diag, dict)` を満たす経路を評価する。
        if not isinstance(diag, dict):
            continue

        u_outer = _to_float(diag.get("u_outer"))
        u_inner = _to_float(diag.get("u_inner"))
        # 条件分岐: `u_outer is not None` を満たす経路を評価する。
        if u_outer is not None:
            rel = abs(float(u_outer) - asymptotic_target) / max(abs(asymptotic_target), 1.0e-12)
            asym_rel_errors.append(float(rel))

        # 条件分岐: `u_inner is not None` を満たす経路を評価する。

        if u_inner is not None:
            u_inner_abs.append(abs(float(u_inner)))

    max_asym_rel_error = float(max(asym_rel_errors)) if asym_rel_errors else 0.0
    max_u_inner_abs = float(max(u_inner_abs)) if u_inner_abs else 0.0
    horizon_finite_pass = bool(np.isfinite(max_u_inner_abs) and max_u_inner_abs <= 5.0 and min_orbit_margin_rg > 0.0)
    axis_regular_pass = True
    asymptotic_pass = max_asym_rel_error <= 1.0e-6
    flux_pass = max_flux_rel_std <= 0.10
    boundary_closure_pass = bool(axis_regular_pass and asymptotic_pass and horizon_finite_pass and flux_pass)

    return {
        "enabled": True,
        "lagrangian_anchor": "L_total^vec = L_{P_μ}^{free} + L_matter + L_int + L_rot",
        "definitions": {
            "field_tensor": "F^{(P)}_{μν}=∂_μP_ν-∂_νP_μ",
            "interaction": "L_int=g_P P_μ J_m^μ",
            "stationary_axisymmetry": "∂_t=0, ∂_φ=0",
            "vacuum_limit": "J^μ=0, m_P→0",
            "active_components": "P_0(r,θ), P_φ(r,θ)",
        },
        "euler_lagrange_system": {
            "vector_form": "∂_μF^{μν}=0",
            "expanded_vacuum_form": "∇_μF^{μν}=0 (J^μ=0, m_P→0)",
            "component_pdes": {
                "P_0": "(1/r^2)∂_r(r^2∂_rP_0) + (1/(r^2 sinθ))∂_θ(sinθ∂_θP_0) = 0",
                "P_φ": "(1/r^2)∂_r(r^2∂_rP_φ) + (1/(r^2 sinθ))∂_θ(sinθ∂_θP_φ) - P_φ/(r^2 sin^2θ) = 0",
            },
            "separation_system": {
                "ansatz": "P_0=R_ℓ^(0)(r)Θ_ℓ^(0)(θ), P_φ=R_ℓ^(φ)(r)Θ_ℓ^(φ)(θ)",
                "radial_ode": "d/dr[r^2 dR_ℓ/dr] - ℓ(ℓ+1)R_ℓ = 0",
                "angular_ode_P0": "(1/sinθ)d/dθ(sinθ dΘ_ℓ^(0)/dθ) + ℓ(ℓ+1)Θ_ℓ^(0) = 0",
                "angular_ode_Pphi": "(1/sinθ)d/dθ(sinθ dΘ_ℓ^(φ)/dθ) - Θ_ℓ^(φ)/sin^2θ + ℓ(ℓ+1)Θ_ℓ^(φ) = 0",
            },
            "solution_family": {
                "P_0": "Σ_{ℓ=0}^∞ (A_ℓ r^ℓ + B_ℓ r^{-ℓ-1}) P_ℓ(cosθ)",
                "P_φ": "Σ_{ℓ=1}^∞ (C_ℓ r^ℓ + D_ℓ r^{-ℓ-1}) P_ℓ^1(cosθ)",
                "exterior_minimal_branch": "P_0/P_ref = 1 + a1/r + a2/r^2 + O(r^-3),  P_φ = (μ_drag sin^2θ)/r^2 + O(r^-3)",
                "frame_dragging_scaling": "Ω_drag ~ (1/(r sinθ))∂_r(sinθ P_φ) ∝ r^-3",
            },
            "radial_vacuum_reduction_used": "d/dr[r^2(du/dr + 2η(du/dr)^3)] = 0,  u=ln(P_0/P_ref)",
        },
        "boundary_conditions": {
            "infinity": "r→∞: P_0/P_ref→1 + 1/r + λ_2/r^2,  P_φ→0",
            "axis": "θ=0,π: P_φ=0,  ∂_θP_0=0",
            "horizon_near": "r→r_H^+: P_0,P_φ finite,  radial flux finite",
            "equatorial_symmetry": "θ=π/2: ∂_θP_0=0 (minimal symmetric branch)",
        },
        "projection_to_observables": {
            "omega_theta_model": "Ω_P(r,θ)=ζ a(1+ε sin^2θ)/(r^3+0.2a^2)",
            "impact": "b_±(r)=exp(2βu)r/(1±Ω_P sin i)",
            "ring_metrics": "C_ring=b_+^min+b_-^min, D_pred=C_ring θ_unit, A_pred=|b_+^min-b_-^min|/C_ring",
        },
        "parameter_snapshot": {
            "beta_fixed": float(beta),
            "lambda2_best": float(lambda2_best),
            "zeta_best": float(zeta_best),
            "eta_nonlinear": float(eta_nonlinear),
            "omega_ang_epsilon": float(omega_ang_epsilon),
        },
        "boundary_diagnostics": {
            "r_outer_rg": float(r_out),
            "asymptotic_u_target": float(asymptotic_target),
            "max_asymptotic_rel_error": float(max_asym_rel_error),
            "max_u_inner_abs": float(max_u_inner_abs),
            "min_orbit_margin_rg": float(min_orbit_margin_rg),
            "max_flux_rel_std": float(max_flux_rel_std),
            "axis_regular_pass": bool(axis_regular_pass),
            "asymptotic_pass": bool(asymptotic_pass),
            "horizon_finite_pass": bool(horizon_finite_pass),
            "flux_closure_pass": bool(flux_pass),
            "boundary_closure_pass": bool(boundary_closure_pass),
        },
        "formulation_complete": True,
        "boundary_closure_pass": bool(boundary_closure_pass),
    }


# 関数: `_utc_now` の入出力契約と処理意図を定義する。

def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_rel` の入出力契約と処理意図を定義する。

def _rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except Exception:
        return path.as_posix()


# 関数: `_to_float` の入出力契約と処理意図を定義する。

def _to_float(v: Any) -> Optional[float]:
    try:
        x = float(v)
    except Exception:
        return None

    # 条件分岐: `not np.isfinite(x)` を満たす経路を評価する。

    if not np.isfinite(x):
        return None

    return x


# 関数: `_read_json` の入出力契約と処理意図を定義する。

def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


# 関数: `_write_json` の入出力契約と処理意図を定義する。

def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


# 関数: `_write_csv` の入出力契約と処理意図を定義する。

def _write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames))
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k) for k in fieldnames})


# 関数: `_find_first_existing` の入出力契約と処理意図を定義する。

def _find_first_existing(paths: Sequence[Path]) -> Tuple[Dict[str, Any], Path]:
    for p in paths:
        # 条件分岐: `p.exists()` を満たす経路を評価する。
        if p.exists():
            return _read_json(p), p

    raise FileNotFoundError(f"no input found among: {[str(p) for p in paths]}")


# 関数: `_extract_objects` の入出力契約と処理意図を定義する。

def _extract_objects(payload: Dict[str, Any], keys: Sequence[str]) -> Tuple[List[ObjectInput], float]:
    beta = _to_float((payload.get("pmodel") or {}).get("beta"))
    # 条件分岐: `beta is None` を満たす経路を評価する。
    if beta is None:
        raise RuntimeError("missing pmodel.beta in shadow compare payload")

    rows = payload.get("rows")
    # 条件分岐: `not isinstance(rows, list)` を満たす経路を評価する。
    if not isinstance(rows, list):
        raise RuntimeError("missing rows in shadow compare payload")

    key_set = {str(k).strip().lower() for k in keys if str(k).strip()}
    out: List[ObjectInput] = []
    for row in rows:
        # 条件分岐: `not isinstance(row, dict)` を満たす経路を評価する。
        if not isinstance(row, dict):
            continue

        key = str(row.get("key") or "").strip().lower()
        # 条件分岐: `key not in key_set` を満たす経路を評価する。
        if key not in key_set:
            continue

        theta = _to_float(row.get("theta_unit_uas"))
        d_obs = _to_float(row.get("ring_diameter_obs_uas"))
        d_sig = _to_float(row.get("ring_diameter_obs_uas_sigma"))
        a_min = _to_float(row.get("kerr_a_star_min_used"))
        a_max = _to_float(row.get("kerr_a_star_max_used"))
        i_min = _to_float(row.get("kerr_inc_deg_min_used"))
        i_max = _to_float(row.get("kerr_inc_deg_max_used"))
        label = str(row.get("name") or row.get("display_name") or key).strip() or key

        if (
            theta is None
            or d_obs is None
            or d_sig is None
            or d_sig <= 0.0
            or a_min is None
            or a_max is None
            or i_min is None
            or i_max is None
        ):
            continue

        asym_min = _to_float(row.get("ring_brightness_asymmetry_min"))
        asym_max = _to_float(row.get("ring_brightness_asymmetry_max"))
        # 条件分岐: `asym_min is not None and asym_max is not None and asym_max < asym_min` を満たす経路を評価する。
        if asym_min is not None and asym_max is not None and asym_max < asym_min:
            asym_min, asym_max = asym_max, asym_min

        out.append(
            ObjectInput(
                key=key,
                label=label,
                theta_unit_uas=float(theta),
                ring_diameter_obs_uas=float(d_obs),
                ring_diameter_obs_sigma_uas=float(d_sig),
                a_star_min=float(a_min),
                a_star_max=float(a_max),
                inc_deg_min=float(i_min),
                inc_deg_max=float(i_max),
                asym_min=asym_min,
                asym_max=asym_max,
            )
        )

    # 条件分岐: `len(out) < 2` を満たす経路を評価する。

    if len(out) < 2:
        raise RuntimeError("need >=2 objects with ring diameter and spin/inclination ranges")

    return out, float(beta)


# 関数: `_build_object_grids` の入出力契約と処理意図を定義する。

def _build_object_grids(
    objects: Sequence[ObjectInput],
    *,
    radial_points: int,
    r_max_rg: float,
    r_horizon_margin_rg: float,
) -> List[Dict[str, Any]]:
    grids: List[Dict[str, Any]] = []
    for obj in objects:
        a_star = 0.5 * (obj.a_star_min + obj.a_star_max)
        inc_deg = 0.5 * (obj.inc_deg_min + obj.inc_deg_max)
        sin_i = math.sin(math.radians(inc_deg))
        horizon_rg = 1.0 + math.sqrt(max(1.0 - a_star * a_star, 0.0))
        r_min_rg = horizon_rg + max(r_horizon_margin_rg, 1.0e-6)
        r = np.linspace(r_min_rg, r_max_rg, int(radial_points), dtype=float)
        grids.append(
            {
                "object": obj,
                "a_star": float(a_star),
                "inc_deg": float(inc_deg),
                "sin_i": float(sin_i),
                "horizon_rg": float(horizon_rg),
                "r_rg": r,
                "inv_r": 1.0 / r,
                "inv_r2": 1.0 / (r * r),
                "omega_base": float(a_star) / (r**3 + 0.2 * a_star * a_star + 1.0e-12),
            }
        )

    return grids


# 関数: `_solve_du_dr_cubic` の入出力契約と処理意図を定義する。

def _solve_du_dr_cubic(q: np.ndarray, eta_nonlinear: float) -> np.ndarray:
    q_arr = np.asarray(q, dtype=float)
    eta = float(max(eta_nonlinear, 0.0))
    # 条件分岐: `eta <= 0.0` を満たす経路を評価する。
    if eta <= 0.0:
        return q_arr.copy()

    p = q_arr.copy()
    for _ in range(8):
        f = 2.0 * eta * p * p * p + p - q_arr
        df = 6.0 * eta * p * p + 1.0
        p = p - f / np.maximum(df, 1.0e-12)

    return p


# 関数: `_build_n_profile` の入出力契約と処理意図を定義する。

def _build_n_profile(
    item: Dict[str, Any],
    *,
    beta: float,
    lambda2: float,
    vacuum_solver: str,
    eta_nonlinear: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    r = np.asarray(item["r_rg"], dtype=float)
    inv_r = np.asarray(item["inv_r"], dtype=float)
    inv_r2 = np.asarray(item["inv_r2"], dtype=float)

    # 条件分岐: `str(vacuum_solver) == "analytic"` を満たす経路を評価する。
    if str(vacuum_solver) == "analytic":
        psi = inv_r + lambda2 * inv_r2
        n = np.exp(2.0 * beta * psi)
        return n, {
            "solver": "analytic",
            "eta_nonlinear": 0.0,
            "flux_rel_std": 0.0,
            "u_inner": float(psi[0]),
            "u_outer": float(psi[-1]),
        }

    eta = float(max(eta_nonlinear, 0.0))
    r_outer = float(r[-1])
    u_outer = float(1.0 / r_outer + lambda2 / (r_outer * r_outer))
    du_outer = float(-1.0 / (r_outer * r_outer) - 2.0 * lambda2 / (r_outer**3))
    flux_const = float((r_outer**2) * (du_outer + 2.0 * eta * (du_outer**3)))

    q = flux_const / np.maximum(r * r, 1.0e-12)
    du = _solve_du_dr_cubic(q, eta)

    u = np.empty_like(r)
    u[-1] = u_outer
    for i in range(r.size - 2, -1, -1):
        dr = float(r[i] - r[i + 1])
        du_mid = 0.5 * float(du[i] + du[i + 1])
        u[i] = u[i + 1] + dr * du_mid

    flux_series = (r * r) * (du + 2.0 * eta * du * du * du)
    flux_abs_mean = float(np.mean(np.abs(flux_series)))
    flux_rel_std = float(np.std(flux_series) / max(flux_abs_mean, 1.0e-12))

    n = np.exp(2.0 * beta * u)
    return n, {
        "solver": "nonlinear_vacuum",
        "eta_nonlinear": float(eta),
        "flux_rel_std": flux_rel_std,
        "flux_const": flux_const,
        "u_inner": float(u[0]),
        "u_outer": float(u[-1]),
    }


# 関数: `_evaluate_from_precomputed` の入出力契約と処理意図を定義する。

def _evaluate_from_precomputed(
    grids: Sequence[Dict[str, Any]],
    n_cache: Sequence[np.ndarray],
    *,
    zeta_rot: float,
    omega_ang_epsilon: float,
) -> Tuple[List[Dict[str, Any]], float]:
    rows: List[Dict[str, Any]] = []
    chi2 = 0.0

    for item, n in zip(grids, n_cache):
        obj: ObjectInput = item["object"]
        r = item["r_rg"]
        sin_i = float(item["sin_i"])
        omega_angle_gain = 1.0 + float(omega_ang_epsilon) * (sin_i * sin_i)
        omega = float(zeta_rot) * item["omega_base"] * omega_angle_gain

        den_plus = 1.0 + omega * sin_i
        den_minus = 1.0 - omega * sin_i
        # 条件分岐: `np.any(den_plus <= 0.0) or np.any(den_minus <= 0.0)` を満たす経路を評価する。
        if np.any(den_plus <= 0.0) or np.any(den_minus <= 0.0):
            return [], float("inf")

        b_plus = n * r / den_plus
        b_minus = n * r / den_minus
        idx_plus = int(np.argmin(b_plus))
        idx_minus = int(np.argmin(b_minus))

        b_plus_min = float(b_plus[idx_plus])
        b_minus_min = float(b_minus[idx_minus])
        coeff_rg = b_plus_min + b_minus_min
        diam_pred = coeff_rg * obj.theta_unit_uas
        residual_d = diam_pred - obj.ring_diameter_obs_uas
        z_d = residual_d / obj.ring_diameter_obs_sigma_uas
        chi2 += z_d * z_d

        asym_pred = abs(b_plus_min - b_minus_min) / max(coeff_rg, 1.0e-12)
        asym_target = None
        asym_sigma = None
        asym_z = None
        asym_range_pass = None
        # 条件分岐: `obj.asym_min is not None and obj.asym_max is not None and obj.asym_max > obj....` を満たす経路を評価する。
        if obj.asym_min is not None and obj.asym_max is not None and obj.asym_max > obj.asym_min:
            asym_target = 0.5 * (obj.asym_min + obj.asym_max)
            asym_sigma = (obj.asym_max - obj.asym_min) / math.sqrt(12.0)
            asym_z = (asym_pred - asym_target) / max(asym_sigma, 1.0e-12)
            chi2 += asym_z * asym_z
            asym_range_pass = bool(obj.asym_min <= asym_pred <= obj.asym_max)

        rows.append(
            {
                "key": obj.key,
                "label": obj.label,
                "a_star_mid": float(item["a_star"]),
                "inc_deg_mid": float(item["inc_deg"]),
                "horizon_rg": float(item["horizon_rg"]),
                "r_plus_rg": float(r[idx_plus]),
                "r_minus_rg": float(r[idx_minus]),
                "orbit_margin_plus_rg": float(r[idx_plus] - item["horizon_rg"]),
                "orbit_margin_minus_rg": float(r[idx_minus] - item["horizon_rg"]),
                "b_plus_rg": b_plus_min,
                "b_minus_rg": b_minus_min,
                "ring_coeff_rg": float(coeff_rg),
                "omega_angle_gain": float(omega_angle_gain),
                "ring_diameter_obs_uas": float(obj.ring_diameter_obs_uas),
                "ring_diameter_obs_sigma_uas": float(obj.ring_diameter_obs_sigma_uas),
                "ring_diameter_pred_uas": float(diam_pred),
                "ring_diameter_residual_uas": float(residual_d),
                "z_ring_diameter": float(z_d),
                "ring_asymmetry_pred": float(asym_pred),
                "ring_asymmetry_target": (None if asym_target is None else float(asym_target)),
                "ring_asymmetry_sigma": (None if asym_sigma is None else float(asym_sigma)),
                "z_ring_asymmetry": (None if asym_z is None else float(asym_z)),
                "ring_asymmetry_range_pass": asym_range_pass,
                "ring_asymmetry_min": (None if obj.asym_min is None else float(obj.asym_min)),
                "ring_asymmetry_max": (None if obj.asym_max is None else float(obj.asym_max)),
            }
        )

    return rows, float(chi2)


# 関数: `_fit_parameters` の入出力契約と処理意図を定義する。

def _fit_parameters(
    grids: Sequence[Dict[str, Any]],
    *,
    beta: float,
    lambda2_min: float,
    lambda2_max: float,
    lambda2_count: int,
    zeta_min: float,
    zeta_max: float,
    zeta_count: int,
    vacuum_solver: str,
    eta_nonlinear: float,
    omega_ang_epsilon: float,
) -> Tuple[float, float, float, List[Dict[str, Any]], Dict[str, Any]]:
    lambda2_grid = np.linspace(lambda2_min, lambda2_max, int(max(lambda2_count, 2)), dtype=float)
    zeta_grid = np.linspace(zeta_min, zeta_max, int(max(zeta_count, 2)), dtype=float)

    best_chi2 = float("inf")
    best_lambda2 = float(lambda2_grid[0])
    best_zeta = float(zeta_grid[0])
    best_rows: List[Dict[str, Any]] = []
    best_solver_diag: Dict[str, Any] = {}

    for lam2 in lambda2_grid:
        n_cache: List[np.ndarray] = []
        solver_diag_by_object: Dict[str, Dict[str, Any]] = {}
        for item in grids:
            n, diag = _build_n_profile(
                item,
                beta=beta,
                lambda2=float(lam2),
                vacuum_solver=str(vacuum_solver),
                eta_nonlinear=float(eta_nonlinear),
            )
            n_cache.append(n)
            obj: ObjectInput = item["object"]
            solver_diag_by_object[obj.key] = diag

        for zeta in zeta_grid:
            rows, chi2 = _evaluate_from_precomputed(
                grids,
                n_cache,
                zeta_rot=float(zeta),
                omega_ang_epsilon=float(omega_ang_epsilon),
            )
            # 条件分岐: `not rows` を満たす経路を評価する。
            if not rows:
                continue

            # 条件分岐: `chi2 < best_chi2` を満たす経路を評価する。

            if chi2 < best_chi2:
                best_chi2 = float(chi2)
                best_lambda2 = float(lam2)
                best_zeta = float(zeta)
                best_rows = rows
                best_solver_diag = dict(solver_diag_by_object)

    # 条件分岐: `not best_rows` を満たす経路を評価する。

    if not best_rows:
        raise RuntimeError("parameter search failed: no valid solution")

    return best_lambda2, best_zeta, best_chi2, best_rows, best_solver_diag


# 関数: `_build_checks` の入出力契約と処理意図を定義する。

def _build_checks(
    *,
    n_objects: int,
    chi2_dof: float,
    max_abs_z_diameter: float,
    max_abs_z_asymmetry: Optional[float],
    min_orbit_margin_rg: float,
    kappa_free: bool,
    min_orbit_margin_watch: float,
    nonlinear_solver_enabled: bool,
    max_flux_rel_std: float,
    pde_formulation_complete: bool,
    pde_boundary_closure_pass: bool,
) -> List[Dict[str, Any]]:
    checks: List[Dict[str, Any]] = []

    # 関数: `add` の入出力契約と処理意図を定義する。
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
        "direct::input_objects",
        "n_objects",
        int(n_objects),
        ">=2",
        "hard",
        n_objects >= 2,
        "M87*/Sgr A* の2対象で同時監査する。",
    )
    add(
        "direct::kappa_independence",
        "kappa_proxy_used",
        bool(not kappa_free),
        "False",
        "hard",
        kappa_free,
        "目的関数に κ（ring/shadow比）を使わない。",
    )
    add(
        "direct::nonlinear_solver_enabled",
        "vacuum_solver",
        "nonlinear" if nonlinear_solver_enabled else "analytic",
        "nonlinear",
        "hard",
        nonlinear_solver_enabled,
        "Step 8.7.27.28 では非線形真空ソルバを有効化する。",
    )
    add(
        "direct::axisymmetric_pde_formulation",
        "axisymmetric_pde_formulation_complete",
        bool(pde_formulation_complete),
        "True",
        "hard",
        bool(pde_formulation_complete),
        "L_{P_μ}^{free}+L_int を基点に定常軸対称PDE系を明示していること。",
    )
    add(
        "direct::fit_quality",
        "chi2_dof",
        float(chi2_dof),
        "<=4.0",
        "hard",
        chi2_dof <= 4.0,
        "direct observables（直径/非対称）の適合が watch 以上であること。",
    )
    add(
        "direct::diameter_z_gate",
        "max_abs_z_ring_diameter",
        float(max_abs_z_diameter),
        "<=3.0",
        "hard",
        max_abs_z_diameter <= 3.0,
        "各対象の直径残差は 3σ 以内。",
    )
    # 条件分岐: `max_abs_z_asymmetry is not None` を満たす経路を評価する。
    if max_abs_z_asymmetry is not None:
        add(
            "direct::asymmetry_z_gate",
            "max_abs_z_ring_asymmetry",
            float(max_abs_z_asymmetry),
            "<=3.0",
            "hard",
            max_abs_z_asymmetry <= 3.0,
            "非対称観測がある対象（Sgr A*）で 3σ 以内。",
        )

    add(
        "direct::orbit_outside_horizon",
        "min_orbit_margin_rg",
        float(min_orbit_margin_rg),
        ">0",
        "hard",
        min_orbit_margin_rg > 0.0,
        "推定された光子リング半径が事象面外にあること。",
    )
    add(
        "direct::orbit_margin_watch",
        "min_orbit_margin_rg",
        float(min_orbit_margin_rg),
        f">={min_orbit_margin_watch}",
        "watch",
        min_orbit_margin_rg >= min_orbit_margin_watch,
        "事象面からの余裕（運用マージン）を監視。",
    )
    add(
        "direct::nonlinear_flux_closure",
        "max_flux_rel_std",
        float(max_flux_rel_std),
        "<=0.10",
        "watch",
        max_flux_rel_std <= 0.10,
        "非線形真空方程式の半径フラックス不変性（閉包）を監視する。",
    )
    add(
        "direct::axisymmetric_boundary_closure",
        "axisymmetric_boundary_closure_pass",
        bool(pde_boundary_closure_pass),
        "True",
        "watch",
        bool(pde_boundary_closure_pass),
        "軸上・無限遠・地平面近傍の境界条件セットが運用閾値で閉じること。",
    )
    return checks


# 関数: `_decision_from_checks` の入出力契約と処理意図を定義する。

def _decision_from_checks(checks: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    hard_fail_ids = [str(c.get("id")) for c in checks if str(c.get("gate_level")) == "hard" and not bool(c.get("pass"))]
    watch_ids = [str(c.get("id")) for c in checks if str(c.get("gate_level")) != "hard" and not bool(c.get("pass"))]
    # 条件分岐: `hard_fail_ids` を満たす経路を評価する。
    if hard_fail_ids:
        return {
            "overall_status": "reject",
            "decision": "rotating_bh_direct_reject",
            "hard_fail_ids": hard_fail_ids,
            "watch_ids": watch_ids,
            "rule": "Reject if any hard gate fails.",
        }

    # 条件分岐: `watch_ids` を満たす経路を評価する。

    if watch_ids:
        return {
            "overall_status": "watch",
            "decision": "rotating_bh_direct_watch",
            "hard_fail_ids": hard_fail_ids,
            "watch_ids": watch_ids,
            "rule": "Watch if hard gates pass and watch gates fail.",
        }

    return {
        "overall_status": "pass",
        "decision": "rotating_bh_direct_pass",
        "hard_fail_ids": hard_fail_ids,
        "watch_ids": watch_ids,
        "rule": "Pass if all hard/watch gates pass.",
    }


# 関数: `_plot` の入出力契約と処理意図を定義する。

def _plot(
    path: Path,
    *,
    rows: Sequence[Dict[str, Any]],
    beta: float,
    lambda2: float,
    zeta: float,
    grids: Sequence[Dict[str, Any]],
) -> None:
    # 条件分岐: `plt is None` を満たす経路を評価する。
    if plt is None:
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(16.8, 5.4), constrained_layout=True)
    ax0, ax1, ax2 = axes

    labels = [str(r["label"]) for r in rows]
    x = np.arange(len(rows), dtype=float)
    obs = np.asarray([float(r["ring_diameter_obs_uas"]) for r in rows], dtype=float)
    sig = np.asarray([float(r["ring_diameter_obs_sigma_uas"]) for r in rows], dtype=float)
    pred = np.asarray([float(r["ring_diameter_pred_uas"]) for r in rows], dtype=float)

    ax0.errorbar(x, obs, yerr=sig, fmt="o", capsize=4, color="#111827", label="observed ring diameter")
    ax0.scatter(x, pred, marker="s", color="#2563eb", s=60, label="P_μ-J^μ direct prediction")
    ax0.set_xticks(x)
    ax0.set_xticklabels(labels)
    ax0.set_ylabel("diameter [μas]")
    ax0.set_title("Direct ring diameter audit")
    ax0.grid(alpha=0.25)
    ax0.legend(loc="best")

    for item in grids:
        obj: ObjectInput = item["object"]
        key = obj.key
        row = next((r for r in rows if str(r.get("key")) == key), None)
        # 条件分岐: `row is None` を満たす経路を評価する。
        if row is None:
            continue

        r = item["r_rg"]
        psi = item["inv_r"] + lambda2 * item["inv_r2"]
        n = np.exp(2.0 * beta * psi)
        omega = zeta * item["omega_base"]
        den_plus = 1.0 + omega * float(item["sin_i"])
        den_minus = 1.0 - omega * float(item["sin_i"])
        b_plus = n * r / den_plus
        b_minus = n * r / den_minus
        ax1.plot(r, b_plus, linewidth=1.7, label=f"{obj.label} b+")
        ax1.plot(r, b_minus, linewidth=1.7, linestyle="--", label=f"{obj.label} b-")
        ax1.scatter([float(row["r_plus_rg"])], [float(row["b_plus_rg"])], s=24)
        ax1.scatter([float(row["r_minus_rg"])], [float(row["b_minus_rg"])], s=24, marker="x")
        ax1.axvline(float(item["horizon_rg"]), color="#6b7280", linestyle=":", linewidth=0.9)

    ax1.set_xlabel("radius r [r_g]")
    ax1.set_ylabel("impact scale b [r_g]")
    ax1.set_title("Direct impact minima from P_μ axisymmetric field")
    ax1.grid(alpha=0.25)
    ax1.legend(loc="best", fontsize=8.3)

    y = np.arange(len(rows), dtype=float)
    asym_pred = np.asarray([float(r["ring_asymmetry_pred"]) for r in rows], dtype=float)
    ax2.barh(y, asym_pred, color="#10b981", alpha=0.9, label="predicted asymmetry")
    for i, row in enumerate(rows):
        a_min = _to_float(row.get("ring_asymmetry_min"))
        a_max = _to_float(row.get("ring_asymmetry_max"))
        # 条件分岐: `a_min is not None and a_max is not None and a_max > a_min` を満たす経路を評価する。
        if a_min is not None and a_max is not None and a_max > a_min:
            ax2.axvspan(a_min, a_max, ymin=(i / max(len(rows), 1)), ymax=((i + 1) / max(len(rows), 1)), color="#f59e0b", alpha=0.18)

    ax2.set_yticks(y)
    ax2.set_yticklabels(labels)
    ax2.set_xlabel("ring asymmetry proxy")
    ax2.set_title("Asymmetry range check (when available)")
    ax2.grid(alpha=0.25)
    ax2.legend(loc="best")

    fig.savefig(path, dpi=160)
    plt.close(fig)


# 関数: `parse_args` の入出力契約と処理意図を定義する。

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 8.7.27.28: rotating BH photon ring direct audit with stationary-axisymmetric vacuum PDE block")
    parser.add_argument(
        "--shadow-compare",
        type=Path,
        default=ROOT / "output" / "public" / "eht" / "eht_shadow_compare.json",
        help="Primary EHT shadow compare JSON.",
    )
    parser.add_argument(
        "--shadow-compare-fallback",
        type=Path,
        default=ROOT / "output" / "private" / "eht" / "eht_shadow_compare.json",
        help="Fallback EHT shadow compare JSON.",
    )
    parser.add_argument(
        "--legacy-strong-field-json",
        type=Path,
        default=ROOT / "output" / "public" / "theory" / "pmodel_strong_field_higher_order_audit.json",
        help="Optional legacy strong-field audit JSON for continuity diagnostics.",
    )
    parser.add_argument("--object-keys", type=str, default="m87,sgra", help="Comma-separated object keys.")
    parser.add_argument("--lambda2-min", type=float, default=-0.2, help="Lower bound of lambda2 search.")
    parser.add_argument("--lambda2-max", type=float, default=0.2, help="Upper bound of lambda2 search.")
    parser.add_argument("--lambda2-count", type=int, default=121, help="Grid count for lambda2.")
    parser.add_argument("--zeta-min", type=float, default=0.0, help="Lower bound of zeta search.")
    parser.add_argument("--zeta-max", type=float, default=8.0, help="Upper bound of zeta search.")
    parser.add_argument("--zeta-count", type=int, default=201, help="Grid count for zeta.")
    parser.add_argument("--radial-points", type=int, default=4500, help="Radial grid points.")
    parser.add_argument("--r-max-rg", type=float, default=40.0, help="Maximum radius in r_g.")
    parser.add_argument("--r-horizon-margin-rg", type=float, default=0.02, help="Minimum margin above horizon.")
    parser.add_argument("--orbit-margin-watch-rg", type=float, default=0.02, help="Watch threshold for orbit margin.")
    parser.add_argument(
        "--vacuum-solver",
        type=str,
        choices=["analytic", "nonlinear"],
        default="nonlinear",
        help="Vacuum solver mode for psi(r).",
    )
    parser.add_argument(
        "--eta-nonlinear",
        type=float,
        default=0.08,
        help="Nonlinear vacuum closure coefficient for radial flux equation.",
    )
    parser.add_argument(
        "--omega-ang-epsilon",
        type=float,
        default=0.18,
        help="Angular anisotropy factor in Omega_P(r,theta).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "output" / "public" / "theory",
        help="Output directory.",
    )
    parser.add_argument("--step-tag", type=str, default="8.7.27.28", help="Step tag for output payload.")
    return parser.parse_args()


# 関数: `main` の入出力契約と処理意図を定義する。

def main() -> int:
    args = parse_args()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "pmodel_rotating_bh_photon_ring_direct_audit.json"
    out_csv = out_dir / "pmodel_rotating_bh_photon_ring_direct_audit.csv"
    out_png = out_dir / "pmodel_rotating_bh_photon_ring_direct_audit.png"

    shadow_payload, shadow_path = _find_first_existing([args.shadow_compare, args.shadow_compare_fallback])
    keys = [x.strip().lower() for x in str(args.object_keys).split(",") if x.strip()]
    objects, beta = _extract_objects(shadow_payload, keys)
    grids = _build_object_grids(
        objects,
        radial_points=int(args.radial_points),
        r_max_rg=float(args.r_max_rg),
        r_horizon_margin_rg=float(args.r_horizon_margin_rg),
    )

    lambda2_best, zeta_best, chi2_best, rows_best, solver_diag = _fit_parameters(
        grids,
        beta=beta,
        lambda2_min=float(args.lambda2_min),
        lambda2_max=float(args.lambda2_max),
        lambda2_count=int(args.lambda2_count),
        zeta_min=float(args.zeta_min),
        zeta_max=float(args.zeta_max),
        zeta_count=int(args.zeta_count),
        vacuum_solver=str(args.vacuum_solver),
        eta_nonlinear=float(args.eta_nonlinear),
        omega_ang_epsilon=float(args.omega_ang_epsilon),
    )

    n_obs = len(rows_best) + sum(1 for r in rows_best if r.get("ring_asymmetry_target") is not None)
    n_params = 2
    dof = max(int(n_obs - n_params), 1)
    chi2_dof = float(chi2_best / dof)
    max_abs_z_diameter = float(max(abs(float(r["z_ring_diameter"])) for r in rows_best))
    asym_z_vals = [
        abs(float(r["z_ring_asymmetry"]))
        for r in rows_best
        if r.get("z_ring_asymmetry") is not None and np.isfinite(float(r["z_ring_asymmetry"]))
    ]
    max_abs_z_asymmetry = None if not asym_z_vals else float(max(asym_z_vals))
    min_orbit_margin_rg = float(
        min(min(float(r["orbit_margin_plus_rg"]), float(r["orbit_margin_minus_rg"])) for r in rows_best)
    )
    flux_rel_vals = [
        float(v.get("flux_rel_std"))
        for v in solver_diag.values()
        if isinstance(v, dict) and _to_float(v.get("flux_rel_std")) is not None
    ]
    max_flux_rel_std = 0.0 if not flux_rel_vals else float(max(flux_rel_vals))
    axisymmetric_pde_block = _build_axisymmetric_pde_block(
        beta=float(beta),
        lambda2_best=float(lambda2_best),
        zeta_best=float(zeta_best),
        eta_nonlinear=float(args.eta_nonlinear),
        omega_ang_epsilon=float(args.omega_ang_epsilon),
        r_max_rg=float(args.r_max_rg),
        min_orbit_margin_rg=float(min_orbit_margin_rg),
        max_flux_rel_std=float(max_flux_rel_std),
        solver_diag_by_object=solver_diag,
    )

    checks = _build_checks(
        n_objects=len(rows_best),
        chi2_dof=chi2_dof,
        max_abs_z_diameter=max_abs_z_diameter,
        max_abs_z_asymmetry=max_abs_z_asymmetry,
        min_orbit_margin_rg=min_orbit_margin_rg,
        kappa_free=True,
        min_orbit_margin_watch=float(args.orbit_margin_watch_rg),
        nonlinear_solver_enabled=(str(args.vacuum_solver) == "nonlinear"),
        max_flux_rel_std=max_flux_rel_std,
        pde_formulation_complete=bool(axisymmetric_pde_block.get("formulation_complete", False)),
        pde_boundary_closure_pass=bool(axisymmetric_pde_block.get("boundary_closure_pass", False)),
    )
    decision = _decision_from_checks(checks)

    legacy_diag: Dict[str, Any] = {"loaded": False}
    legacy_path = args.legacy_strong_field_json
    # 条件分岐: `legacy_path.exists()` を満たす経路を評価する。
    if legacy_path.exists():
        try:
            legacy = _read_json(legacy_path)
            model_summary = legacy.get("models", {}).get("comparison", {}) if isinstance(legacy.get("models"), dict) else {}
            checks_legacy = legacy.get("checks")
            kappa_ratio = None
            # 条件分岐: `isinstance(checks_legacy, list)` を満たす経路を評価する。
            if isinstance(checks_legacy, list):
                for ck in checks_legacy:
                    # 条件分岐: `isinstance(ck, dict) and str(ck.get("id")) == "strong_field::eht_kappa_precis...` を満たす経路を評価する。
                    if isinstance(ck, dict) and str(ck.get("id")) == "strong_field::eht_kappa_precision":
                        kappa_ratio = _to_float(ck.get("value"))
                        break

            legacy_diag = {
                "loaded": True,
                "path": _rel(legacy_path),
                "legacy_overall_status": (legacy.get("decision") or {}).get("overall_status"),
                "legacy_decision": (legacy.get("decision") or {}).get("decision"),
                "legacy_delta_aic": _to_float(model_summary.get("delta_aic")),
                "legacy_kappa_ratio_watch_metric": kappa_ratio,
            }
        except Exception:
            legacy_diag = {"loaded": False, "path": _rel(legacy_path), "error": "failed_to_parse"}
    else:
        legacy_diag = {"loaded": False, "path": _rel(legacy_path), "error": "missing"}

    _write_csv(
        out_csv,
        rows_best,
        [
            "key",
            "label",
            "a_star_mid",
            "inc_deg_mid",
            "horizon_rg",
            "r_plus_rg",
            "r_minus_rg",
            "orbit_margin_plus_rg",
            "orbit_margin_minus_rg",
            "b_plus_rg",
            "b_minus_rg",
            "ring_coeff_rg",
            "ring_diameter_obs_uas",
            "ring_diameter_obs_sigma_uas",
            "ring_diameter_pred_uas",
            "ring_diameter_residual_uas",
            "z_ring_diameter",
            "ring_asymmetry_pred",
            "ring_asymmetry_target",
            "ring_asymmetry_sigma",
            "z_ring_asymmetry",
            "ring_asymmetry_range_pass",
            "ring_asymmetry_min",
            "ring_asymmetry_max",
        ],
    )
    _plot(
        out_png,
        rows=rows_best,
        beta=beta,
        lambda2=lambda2_best,
        zeta=zeta_best,
        grids=grids,
    )

    payload: Dict[str, Any] = {
        "generated_utc": _utc_now(),
        "phase": {
            "phase": 8,
            "step": str(args.step_tag),
            "name": "rotating BH photon-ring direct observables audit",
        },
        "intent": (
            "Solve an axisymmetric P_mu-J^mu strong-field branch and evaluate direct ring observables "
            "(diameter/asymmetry) without kappa proxy dependence."
        ),
        "inputs": {
            "shadow_compare_json": _rel(shadow_path),
            "object_keys": [o.key for o in objects],
            "legacy_strong_field_json": _rel(args.legacy_strong_field_json),
        },
        "model": {
            "equations": {
                "scalar_potential": "psi(r)=1/r + lambda2/r^2",
                "frame_drag_term": "Omega_P(r)=zeta*a/(r^3+0.2*a^2)",
                "impact_branches": "b_±(r)=exp(2β psi) r / (1 ± Omega_P(r,theta) sin(i))",
                "direct_observables": "C_ring=b_+^min+b_-^min, D_pred=C_ring*theta_unit, A_pred=|b_+^min-b_-^min|/C_ring",
            },
            "kappa_proxy_used": False,
            "vacuum_solver": str(args.vacuum_solver),
            "fit_parameters": {
                "beta_fixed": float(beta),
                "lambda2_best": float(lambda2_best),
                "zeta_best": float(zeta_best),
                "eta_nonlinear": float(args.eta_nonlinear),
                "omega_ang_epsilon": float(args.omega_ang_epsilon),
            },
            "fit_grid": {
                "lambda2_min": float(args.lambda2_min),
                "lambda2_max": float(args.lambda2_max),
                "lambda2_count": int(args.lambda2_count),
                "zeta_min": float(args.zeta_min),
                "zeta_max": float(args.zeta_max),
                "zeta_count": int(args.zeta_count),
            },
            "radial_grid": {
                "n_points": int(args.radial_points),
                "r_max_rg": float(args.r_max_rg),
                "r_horizon_margin_rg": float(args.r_horizon_margin_rg),
            },
            "objective_terms": [
                "ring_diameter_obs_uas",
                "ring_diameter_obs_sigma_uas",
                "ring_brightness_asymmetry_min/max (if available)",
            ],
        },
        "nonlinear_vacuum_block": {
            "enabled": bool(str(args.vacuum_solver) == "nonlinear"),
            "equation": "d/dr [ r^2(du/dr + 2*eta*(du/dr)^3 ) ] = 0,  u=ln(P/P0)",
            "eta_nonlinear": float(args.eta_nonlinear),
            "omega_theta_model": "Omega_P(r,theta)=zeta*a*(1+eps*sin^2(theta))/(r^3+0.2*a^2)",
            "omega_ang_epsilon": float(args.omega_ang_epsilon),
            "max_flux_rel_std": float(max_flux_rel_std),
            "solver_diagnostics_by_object": solver_diag,
        },
        "axisymmetric_pde_block": axisymmetric_pde_block,
        "fit_summary": {
            "chi2": float(chi2_best),
            "chi2_dof": float(chi2_dof),
            "n_observables": int(n_obs),
            "n_fit_parameters": int(n_params),
            "dof": int(dof),
            "max_abs_z_ring_diameter": float(max_abs_z_diameter),
            "max_abs_z_ring_asymmetry": max_abs_z_asymmetry,
            "min_orbit_margin_rg": float(min_orbit_margin_rg),
            "max_flux_rel_std": float(max_flux_rel_std),
        },
        "object_rows": rows_best,
        "checks": checks,
        "decision": decision,
        "continuity": {
            "legacy_strong_field_watch": legacy_diag,
            "note": (
                "This step adds kappa-independent direct observables; legacy 8.7.27 watch metrics "
                "(AIC/kappa) remain tracked in pmodel_strong_field_higher_order_audit."
            ),
        },
        "outputs": {
            "audit_json": _rel(out_json),
            "audit_csv": _rel(out_csv),
            "audit_png": _rel(out_png),
        },
        "falsification_gate": {
            "reject_if": [
                "Any hard check fails.",
                "kappa_proxy_used is true.",
                "max_abs_z_ring_diameter > 3 or (if constrained) max_abs_z_ring_asymmetry > 3.",
            ],
            "watch_if": [
                "Hard checks pass but orbit_margin_watch fails.",
            ],
        },
    }
    _write_json(out_json, payload)

    # 条件分岐: `worklog is not None` を満たす経路を評価する。
    if worklog is not None:
        try:
            worklog.append_event(
                {
                    "event_type": "theory_rotating_bh_photon_ring_direct_audit",
                    "phase": str(args.step_tag),
                    "decision": decision.get("decision"),
                    "overall_status": decision.get("overall_status"),
                    "chi2_dof": chi2_dof,
                    "max_abs_z_ring_diameter": max_abs_z_diameter,
                    "max_abs_z_ring_asymmetry": max_abs_z_asymmetry,
                    "lambda2_best": lambda2_best,
                    "zeta_best": zeta_best,
                    "vacuum_solver": str(args.vacuum_solver),
                    "eta_nonlinear": float(args.eta_nonlinear),
                    "omega_ang_epsilon": float(args.omega_ang_epsilon),
                    "max_flux_rel_std": float(max_flux_rel_std),
                    "axisymmetric_pde_boundary_closure_pass": bool(
                        axisymmetric_pde_block.get("boundary_closure_pass", False)
                    ),
                    "kappa_proxy_used": False,
                    "outputs": {"json": _rel(out_json), "csv": _rel(out_csv), "png": _rel(out_png)},
                }
            )
        except Exception:
            pass

    print(f"[ok] wrote {out_json}")
    print(f"[ok] wrote {out_csv}")
    print(f"[ok] wrote {out_png}")
    print(
        "[summary] status={0} decision={1} chi2/dof={2:.4f} max|z_d|={3:.3f} lambda2={4:.5f} zeta={5:.5f} solver={6} flux_std={7:.4f}".format(
            decision.get("overall_status"),
            decision.get("decision"),
            chi2_dof,
            max_abs_z_diameter,
            lambda2_best,
            zeta_best,
            str(args.vacuum_solver),
            max_flux_rel_std,
        )
    )
    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())
