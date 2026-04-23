#!/usr/bin/env python3
"""Audit the beta-sensitivity equation behind the Trial-2 common-root selector.

Purpose:
    Start one genuinely new strict-theorem route after the practical target-free
    common-root closeout and the negative strict-theorem followup. The route
    promotes the beta-sensitivity equation

        u_beta(x) = partial_beta y_beta(x)

    into one exact object and checks whether the localized ground-state branch
    already supplies strong sign support for a future monotonicity / uniqueness
    theorem of

        Delta_common(beta) = alpha_qstar(beta) - alpha_R8(beta).

Inputs:
    - scripts/quantum/mass_origin_qball_charge_mapping_branch.py
    - scripts/quantum/trial2_alpha_beta_curve_backend.py
    - scripts/quantum/trial2_interaction_total_over_harmonic_sq_exact_relation_backend.py

Outputs:
    - One in-memory audit pack consumed by `.5639-.5646` wrappers

Assumptions:
    - The common-root selector remains fixed at beta_common_root
    - No new parameter is introduced
    - Finite-difference beta scans are used only as local numerical support,
      not as a replacement for the desired strict theorem
"""

from __future__ import annotations

import math
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.quantum.mass_origin_qball_charge_mapping_branch import load_qball_module
from scripts.quantum.mass_origin_qball_charge_mapping_branch import solve_full_profile
from scripts.quantum.scalar_proxy_alpha_q_curve_backend import ALPHA_TARGET
from scripts.quantum.trial2_alpha_beta_curve_backend import build_beta_family_row
from scripts.quantum.trial2_interaction_total_over_harmonic_sq_exact_relation_backend import (
    build_exact_relation_row,
)


BETA_COMMON_ROOT = 0.9983014161324819
H_VALUES = (1.0e-4, 5.0e-5, 1.0e-5, 5.0e-6, 1.0e-6)
GRID_POINT_COUNT = 4000
WINDOW_X_MIN = 0.05
WINDOW_X_MAX = 20.0
LINEARIZED_REL_RMS_TOL = 5.0e-3


# 関数: retained pivot module を cached で返す。
@lru_cache(maxsize=1)
def get_qball_pivot_module():
    """Return the retained scalar Q-ball pivot solver module."""
    return load_qball_module()


# 関数: one beta で full localized profile row を返す。

@lru_cache(maxsize=None)
def build_profile_row(beta: float) -> dict:
    """Return one localized scalar profile row for the requested beta."""
    beta = float(beta)
    qball_pivot = get_qball_pivot_module()
    amplitude = qball_pivot.find_amp(beta)
    if amplitude is None:
        raise SystemExit(f"[fail] localized scalar profile is unavailable for beta={beta}")

    radius, profile, profile_prime = solve_full_profile(beta, float(amplitude))
    return {
        "beta": beta,
        "central_amplitude": float(amplitude),
        "radius": np.asarray(radius, dtype=float),
        "profile": np.asarray(profile, dtype=float),
        "profile_prime": np.asarray(profile_prime, dtype=float),
    }


# 関数: 3本の profile に共通な interpolation grid を返す。

def build_common_grid(center_row: dict, plus_row: dict, minus_row: dict) -> np.ndarray:
    """Return one overlap grid shared by the central / plus / minus profiles."""
    lower = max(
        float(center_row["radius"][0]),
        float(plus_row["radius"][0]),
        float(minus_row["radius"][0]),
        WINDOW_X_MIN,
    )
    upper = min(
        float(center_row["radius"][-1]),
        float(plus_row["radius"][-1]),
        float(minus_row["radius"][-1]),
    )
    if upper <= lower:
        raise SystemExit("[fail] beta-sensitivity overlap grid is empty")

    return np.linspace(lower, upper, GRID_POINT_COUNT, dtype=float)


# 関数: beta-sensitivity finite-difference row を構築する。

def build_beta_sensitivity_row(beta_common_root: float, h_value: float) -> dict:
    """Return one finite-difference beta-sensitivity support row."""
    beta_common_root = float(beta_common_root)
    h_value = float(h_value)
    center_row = build_profile_row(beta_common_root)
    plus_row = build_profile_row(beta_common_root + h_value)
    minus_row = build_profile_row(beta_common_root - h_value)
    grid = build_common_grid(center_row, plus_row, minus_row)

    # 関数: `interp` の入出力契約と処理意図を定義する。
    def interp(source_row: dict, key: str) -> np.ndarray:
        return np.interp(grid, source_row["radius"], source_row[key])

    profile = interp(center_row, "profile")
    profile_plus = interp(plus_row, "profile")
    profile_minus = interp(minus_row, "profile")
    profile_prime_plus = interp(plus_row, "profile_prime")
    profile_prime_minus = interp(minus_row, "profile_prime")

    u_beta = (profile_plus - profile_minus) / (2.0 * h_value)
    u_beta_prime = (profile_prime_plus - profile_prime_minus) / (2.0 * h_value)
    u_beta_second = np.gradient(u_beta_prime, grid)

    linearized_potential = (
        beta_common_root * beta_common_root
        - 1.0
        + 6.0 * profile
        + 3.0 * np.square(profile)
    )
    source = -2.0 * beta_common_root * profile
    linearized_residual = (
        u_beta_second
        + 2.0 * u_beta_prime / grid
        + linearized_potential * u_beta
        - source
    )

    mask = np.logical_and(grid >= WINDOW_X_MIN, grid <= WINDOW_X_MAX)
    if not np.any(mask):
        raise SystemExit("[fail] beta-sensitivity support mask is empty")

    masked_u = u_beta[mask]
    masked_residual = linearized_residual[mask]
    masked_source = source[mask]
    source_rms = float(np.sqrt(np.mean(np.square(masked_source))))
    residual_rms = float(np.sqrt(np.mean(np.square(masked_residual))))

    return {
        "h": h_value,
        "grid_x_min": float(grid[0]),
        "grid_x_max": float(grid[-1]),
        "u_beta_min": float(np.min(masked_u)),
        "u_beta_max": float(np.max(masked_u)),
        "u_beta_negative_fraction": float(np.mean(masked_u < 0.0)),
        "u_beta_positive_fraction": float(np.mean(masked_u > 0.0)),
        "linearized_residual_abs_max": float(np.max(np.abs(masked_residual))),
        "linearized_residual_rms": residual_rms,
        "linearized_source_abs_max": float(np.max(np.abs(masked_source))),
        "linearized_source_rms": source_rms,
        "linearized_residual_rel_rms": float(
            residual_rms / max(source_rms, 1.0e-30)
        ),
    }


# 関数: exact row family から beta-derivative sign pack を返す。

def build_integral_derivative_pack(beta_common_root: float, h_value: float) -> dict:
    """Return one finite-difference derivative pack for exact beta objects."""
    beta_common_root = float(beta_common_root)
    h_value = float(h_value)
    alpha_plus = build_beta_family_row(beta_common_root + h_value)
    alpha_minus = build_beta_family_row(beta_common_root - h_value)
    if alpha_plus is None or alpha_minus is None:
        raise SystemExit("[fail] alpha_qstar row is unavailable near beta_common_root")

    exact_plus = build_exact_relation_row(beta_common_root + h_value)
    exact_minus = build_exact_relation_row(beta_common_root - h_value)
    scale = 2.0 * h_value
    return {
        "h": h_value,
        "d_alpha_qstar_dbeta": float(
            (float(alpha_plus["alpha_at_q_star"]) - float(alpha_minus["alpha_at_q_star"]))
            / scale
        ),
        "d_alpha_r8_dbeta": float(
            (
                float(exact_plus["exact_relation_from_integrals"])
                - float(exact_minus["exact_relation_from_integrals"])
            )
            / scale
        ),
        "d_i2_dbeta": float((float(exact_plus["i2"]) - float(exact_minus["i2"])) / scale),
        "d_ig_dbeta": float((float(exact_plus["ig"]) - float(exact_minus["ig"])) / scale),
        "d_i4_dbeta": float((float(exact_plus["i4"]) - float(exact_minus["i4"])) / scale),
        "d_boundary_dbeta": float(
            (float(exact_plus["boundary_weighted_eom"]) - float(exact_minus["boundary_weighted_eom"]))
            / scale
        ),
    }


# 関数: beta-sensitivity equation route の監査 pack 全体を返す。

def build_trial2_beta_sensitivity_equation_pack() -> dict:
    """Return one audit pack for the beta-sensitivity theorem route."""
    beta_common_root = float(BETA_COMMON_ROOT)
    support_rows = [
        build_beta_sensitivity_row(beta_common_root, h_value)
        for h_value in H_VALUES
    ]
    derivative_rows = [
        build_integral_derivative_pack(beta_common_root, h_value)
        for h_value in H_VALUES
    ]
    exact_relation_row = build_exact_relation_row(beta_common_root)
    alpha_row = build_beta_family_row(beta_common_root)
    if alpha_row is None:
        raise SystemExit("[fail] alpha_qstar row is unavailable at beta_common_root")

    local_beta_sensitivity_support_available_now = bool(
        all(
            row["u_beta_negative_fraction"] == 1.0
            and row["linearized_residual_rel_rms"] <= LINEARIZED_REL_RMS_TOL
            for row in support_rows
        )
    )
    u_beta_negative_support_available_now = bool(
        all(row["u_beta_negative_fraction"] == 1.0 for row in support_rows)
    )
    alpha_qstar_derivative_positive_now = bool(
        all(row["d_alpha_qstar_dbeta"] > 0.0 for row in derivative_rows)
    )
    alpha_r8_derivative_negative_now = bool(
        all(row["d_alpha_r8_dbeta"] < 0.0 for row in derivative_rows)
    )
    i2_derivative_negative_now = bool(
        all(row["d_i2_dbeta"] < 0.0 for row in derivative_rows)
    )
    ig_derivative_negative_now = bool(
        all(row["d_ig_dbeta"] < 0.0 for row in derivative_rows)
    )
    i4_derivative_negative_now = bool(
        all(row["d_i4_dbeta"] < 0.0 for row in derivative_rows)
    )
    boundary_derivative_negative_now = bool(
        all(row["d_boundary_dbeta"] < 0.0 for row in derivative_rows)
    )

    exact_beta_sensitivity_equation_available_now = True
    exact_common_root_monotonicity_theorem_available_now = False
    beta_sensitivity_route_available_now = bool(
        exact_beta_sensitivity_equation_available_now
        and local_beta_sensitivity_support_available_now
    )
    beta_sensitivity_monotonicity_followup_required_now = bool(
        beta_sensitivity_route_available_now
        and not exact_common_root_monotonicity_theorem_available_now
    )

    return {
        "beta_common_root": beta_common_root,
        "alpha_target": float(ALPHA_TARGET),
        "alpha_common_value": float(alpha_row["alpha_at_q_star"]),
        "alpha_common_rel_error_vs_target": float(
            (float(alpha_row["alpha_at_q_star"]) - float(ALPHA_TARGET)) / float(ALPHA_TARGET)
        ),
        "r8_common_value": float(exact_relation_row["exact_relation_from_integrals"]),
        "q_star_common_over_m0": float(alpha_row["q_star_over_m0"]),
        "support_rows": support_rows,
        "derivative_rows": derivative_rows,
        "exact_beta_sensitivity_equation_available_now": (
            exact_beta_sensitivity_equation_available_now
        ),
        "local_beta_sensitivity_support_available_now": (
            local_beta_sensitivity_support_available_now
        ),
        "u_beta_negative_support_available_now": u_beta_negative_support_available_now,
        "alpha_qstar_derivative_positive_now": alpha_qstar_derivative_positive_now,
        "alpha_r8_derivative_negative_now": alpha_r8_derivative_negative_now,
        "i2_derivative_negative_now": i2_derivative_negative_now,
        "ig_derivative_negative_now": ig_derivative_negative_now,
        "i4_derivative_negative_now": i4_derivative_negative_now,
        "boundary_derivative_negative_now": boundary_derivative_negative_now,
        "u_beta_min_global": float(min(row["u_beta_min"] for row in support_rows)),
        "u_beta_max_global": float(max(row["u_beta_max"] for row in support_rows)),
        "linearized_residual_rel_rms_max": float(
            max(row["linearized_residual_rel_rms"] for row in support_rows)
        ),
        "linearized_residual_rel_rms_min": float(
            min(row["linearized_residual_rel_rms"] for row in support_rows)
        ),
        "d_alpha_qstar_dbeta_min": float(
            min(row["d_alpha_qstar_dbeta"] for row in derivative_rows)
        ),
        "d_alpha_qstar_dbeta_max": float(
            max(row["d_alpha_qstar_dbeta"] for row in derivative_rows)
        ),
        "d_alpha_r8_dbeta_min": float(
            min(row["d_alpha_r8_dbeta"] for row in derivative_rows)
        ),
        "d_alpha_r8_dbeta_max": float(
            max(row["d_alpha_r8_dbeta"] for row in derivative_rows)
        ),
        "d_i2_dbeta_min": float(min(row["d_i2_dbeta"] for row in derivative_rows)),
        "d_i2_dbeta_max": float(max(row["d_i2_dbeta"] for row in derivative_rows)),
        "d_ig_dbeta_min": float(min(row["d_ig_dbeta"] for row in derivative_rows)),
        "d_ig_dbeta_max": float(max(row["d_ig_dbeta"] for row in derivative_rows)),
        "d_i4_dbeta_min": float(min(row["d_i4_dbeta"] for row in derivative_rows)),
        "d_i4_dbeta_max": float(max(row["d_i4_dbeta"] for row in derivative_rows)),
        "d_boundary_dbeta_min": float(
            min(row["d_boundary_dbeta"] for row in derivative_rows)
        ),
        "d_boundary_dbeta_max": float(
            max(row["d_boundary_dbeta"] for row in derivative_rows)
        ),
        "exact_common_root_monotonicity_theorem_available_now": (
            exact_common_root_monotonicity_theorem_available_now
        ),
        "beta_sensitivity_route_available_now": beta_sensitivity_route_available_now,
        "beta_sensitivity_monotonicity_followup_required_now": (
            beta_sensitivity_monotonicity_followup_required_now
        ),
    }


# 関数: backend 単体実行時に compact summary を表示する。

def main() -> None:
    """Run the beta-sensitivity equation audit directly."""
    pack = build_trial2_beta_sensitivity_equation_pack()
    print("[trial2_beta_sensitivity_equation_backend]")
    print(f"  beta_common_root = {pack['beta_common_root']:.15f}")
    print(f"  u_beta_min_global = {pack['u_beta_min_global']:.15f}")
    print(f"  linearized_residual_rel_rms_max = {pack['linearized_residual_rel_rms_max']:.15f}")
    print(
        "  beta_sensitivity_monotonicity_followup_required = "
        f"{pack['beta_sensitivity_monotonicity_followup_required_now']}"
    )


if __name__ == "__main__":
    main()
