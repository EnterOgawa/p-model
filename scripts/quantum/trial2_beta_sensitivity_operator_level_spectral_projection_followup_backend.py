#!/usr/bin/env python3
"""Audit weighted-integral sign support after continuum open-interval stability.

Purpose:
    Continue the strict-theorem hardening route after `.5671-.5678`, where the
    discrete pointwise-dominance theorem and the continuum open-interval
    support were already fixed. The remaining operator-level gap is narrower:

        can the retained open-interval support already force the signs of the
        weighted beta-derivative integrals

            dI_n / d beta = n ∫ y_beta^(n-1) u_beta x^2 dx

        strongly enough to promote one derivative-chain followup?

    This backend does not overclaim the final pure analytic continuum theorem.
    Instead it checks whether:

    1. the smallest retained continuum-supported interior window already keeps
       the weighted integrands one-sign,
    2. the boundary complement stays too small to reverse the total integral
       sign, and
    3. the resulting sign support is consistent with the already retained
       positive local support for d Delta_common / d beta.

Inputs:
    - scripts/quantum/trial2_beta_sensitivity_equation_backend.py
    - scripts/quantum/trial2_beta_sensitivity_continuum_spectral_projection_followup_backend.py

Outputs:
    - One in-memory audit pack consumed by `.5679-.5686` wrappers

Assumptions:
    - The canonical control window is the smallest fixed open interval already
      promoted in `.5671-.5678`, namely [0.10, 19.90]
    - No new parameter is introduced
    - The route targets weighted-integral sign support only; it does not claim
      the final pure analytic operator-level theorem
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.quantum.trial2_beta_sensitivity_continuum_spectral_projection_followup_backend import (
    build_trial2_beta_sensitivity_continuum_spectral_projection_followup_pack,
)
from scripts.quantum.trial2_beta_sensitivity_equation_backend import BETA_COMMON_ROOT
from scripts.quantum.trial2_beta_sensitivity_equation_backend import H_VALUES
from scripts.quantum.trial2_beta_sensitivity_equation_backend import build_common_grid
from scripts.quantum.trial2_beta_sensitivity_equation_backend import (
    build_integral_derivative_pack,
)
from scripts.quantum.trial2_beta_sensitivity_equation_backend import build_profile_row


CONTROL_WINDOW_X_MIN = 0.10
CONTROL_WINDOW_X_MAX = 19.90
WEIGHTED_INTEGRAL_ORDERS = (2, 3, 4)
BOUNDARY_COMPLEMENT_FRACTION_TOL = 1.0e-1
DELTA_COMMON_DERIVATIVE_REL_SPREAD_TOL = 5.0e-3


# 関数: 1つの `h` に対する weighted-integral sign row を返す。
def build_weighted_integral_sign_row(h_value: float) -> dict:
    """Return one weighted-integral sign row on the retained control window."""
    beta_common_root = float(BETA_COMMON_ROOT)
    h_value = float(h_value)
    center_row = build_profile_row(beta_common_root)
    plus_row = build_profile_row(beta_common_root + h_value)
    minus_row = build_profile_row(beta_common_root - h_value)
    grid = build_common_grid(center_row, plus_row, minus_row)

    profile = np.interp(grid, center_row["radius"], center_row["profile"])
    profile_plus = np.interp(grid, plus_row["radius"], plus_row["profile"])
    profile_minus = np.interp(grid, minus_row["radius"], minus_row["profile"])
    u_beta = (profile_plus - profile_minus) / (2.0 * h_value)
    control_mask = (grid >= CONTROL_WINDOW_X_MIN) & (grid <= CONTROL_WINDOW_X_MAX)
    if not np.any(control_mask):
        raise SystemExit("[fail] operator-level control window mask is empty")

    integral_rows = []
    for order in WEIGHTED_INTEGRAL_ORDERS:
        integrand = np.asarray(
            np.power(profile, order - 1) * u_beta * np.square(grid),
            dtype=float,
        )
        total_integral = float(np.trapezoid(integrand, grid))
        interior_integral = float(
            np.trapezoid(integrand[control_mask], grid[control_mask])
        )
        boundary_integral = float(total_integral - interior_integral)
        boundary_abs_fraction = float(
            abs(boundary_integral) / max(abs(total_integral), 1.0e-30)
        )
        integral_rows.append(
            {
                "order_n": int(order),
                "weighted_total_integral": total_integral,
                "weighted_interior_integral": interior_integral,
                "weighted_boundary_integral": boundary_integral,
                "boundary_complement_abs_fraction": boundary_abs_fraction,
                "control_window_negative_fraction": float(
                    np.mean(integrand[control_mask] < 0.0)
                ),
                "weighted_total_negative_now": bool(total_integral < 0.0),
                "weighted_interior_negative_now": bool(interior_integral < 0.0),
                "weighted_boundary_nonreversing_now": bool(
                    boundary_abs_fraction <= BOUNDARY_COMPLEMENT_FRACTION_TOL
                ),
                "d_integral_order_dbeta": float(order * total_integral),
            }
        )

    return {
        "h": h_value,
        "control_window_x_min": float(CONTROL_WINDOW_X_MIN),
        "control_window_x_max": float(CONTROL_WINDOW_X_MAX),
        "u_beta_control_negative_fraction": float(np.mean(u_beta[control_mask] < 0.0)),
        "integral_rows": integral_rows,
    }


# 関数: operator-level spectral-projection followup 監査 pack を返す。

def build_trial2_beta_sensitivity_operator_level_spectral_projection_followup_pack() -> dict:
    """Return one operator-level weighted-integral sign audit pack."""
    continuum_pack = (
        build_trial2_beta_sensitivity_continuum_spectral_projection_followup_pack()
    )
    weighted_rows = [build_weighted_integral_sign_row(h_value) for h_value in H_VALUES]
    derivative_rows = [
        build_integral_derivative_pack(float(BETA_COMMON_ROOT), h_value)
        for h_value in H_VALUES
    ]
    smallest_window_summary = next(
        summary
        for summary in continuum_pack["interior_window_summaries"]
        if abs(float(summary["x_min"]) - CONTROL_WINDOW_X_MIN) < 1.0e-12
        and abs(float(summary["x_max"]) - CONTROL_WINDOW_X_MAX) < 1.0e-12
    )

    delta_common_derivative_values = [
        float(row["d_alpha_qstar_dbeta"] - row["d_alpha_r8_dbeta"])
        for row in derivative_rows
    ]
    delta_common_derivative_rel_spread = float(
        (max(delta_common_derivative_values) - min(delta_common_derivative_values))
        / max(abs(np.mean(delta_common_derivative_values)), 1.0e-30)
    )

    control_window_continuum_support_available_now = bool(
        continuum_pack[
            "exact_trial2_beta_sensitivity_continuum_open_interval_support_available_now"
        ]
        and smallest_window_summary["continuum_margin_positive_now"]
        and smallest_window_summary["refinement_stable_now"]
    )
    weighted_integral_sign_support_available_now = bool(
        control_window_continuum_support_available_now
        and all(row["u_beta_control_negative_fraction"] == 1.0 for row in weighted_rows)
        and all(
            integral_row["control_window_negative_fraction"] == 1.0
            and integral_row["weighted_total_negative_now"]
            and integral_row["weighted_interior_negative_now"]
            and integral_row["weighted_boundary_nonreversing_now"]
            for row in weighted_rows
            for integral_row in row["integral_rows"]
        )
    )
    delta_common_derivative_positive_local_support_now = bool(
        all(value > 0.0 for value in delta_common_derivative_values)
        and delta_common_derivative_rel_spread <= DELTA_COMMON_DERIVATIVE_REL_SPREAD_TOL
    )
    exact_trial2_beta_sensitivity_operator_level_spectral_projection_theorem_available_now = (
        False
    )
    updated_pack_trial2_beta_sensitivity_derivative_chain_followup_required_now = bool(
        weighted_integral_sign_support_available_now
        and delta_common_derivative_positive_local_support_now
        and not exact_trial2_beta_sensitivity_operator_level_spectral_projection_theorem_available_now
    )

    by_order = {}
    for order in WEIGHTED_INTEGRAL_ORDERS:
        order_rows = [
            next(item for item in row["integral_rows"] if item["order_n"] == order)
            for row in weighted_rows
        ]
        by_order[str(order)] = {
            "d_integral_order_dbeta_min": float(
                min(item["d_integral_order_dbeta"] for item in order_rows)
            ),
            "d_integral_order_dbeta_max": float(
                max(item["d_integral_order_dbeta"] for item in order_rows)
            ),
            "boundary_complement_abs_fraction_max": float(
                max(item["boundary_complement_abs_fraction"] for item in order_rows)
            ),
            "boundary_complement_abs_fraction_min": float(
                min(item["boundary_complement_abs_fraction"] for item in order_rows)
            ),
            "weighted_total_integral_min": float(
                min(item["weighted_total_integral"] for item in order_rows)
            ),
            "weighted_total_integral_max": float(
                max(item["weighted_total_integral"] for item in order_rows)
            ),
        }

    return {
        "beta_common_root": float(BETA_COMMON_ROOT),
        "control_window_x_min": float(CONTROL_WINDOW_X_MIN),
        "control_window_x_max": float(CONTROL_WINDOW_X_MAX),
        "h_values": [float(value) for value in H_VALUES],
        "continuum_pack": continuum_pack,
        "smallest_window_summary": smallest_window_summary,
        "weighted_rows": weighted_rows,
        "derivative_rows": derivative_rows,
        "weighted_integral_by_order": by_order,
        "delta_common_derivative_values": delta_common_derivative_values,
        "delta_common_derivative_min": float(min(delta_common_derivative_values)),
        "delta_common_derivative_max": float(max(delta_common_derivative_values)),
        "delta_common_derivative_rel_spread": delta_common_derivative_rel_spread,
        "control_window_continuum_support_available_now": (
            control_window_continuum_support_available_now
        ),
        "exact_trial2_beta_sensitivity_weighted_integral_sign_support_available_now": (
            weighted_integral_sign_support_available_now
        ),
        "delta_common_derivative_positive_local_support_now": (
            delta_common_derivative_positive_local_support_now
        ),
        "exact_trial2_beta_sensitivity_operator_level_spectral_projection_theorem_available_now": (
            exact_trial2_beta_sensitivity_operator_level_spectral_projection_theorem_available_now
        ),
        "updated_pack_trial2_beta_sensitivity_derivative_chain_followup_required_now": (
            updated_pack_trial2_beta_sensitivity_derivative_chain_followup_required_now
        ),
    }


# 関数: backend 単体実行時に retained metrics を表示する。

def main() -> None:
    """Run the operator-level followup backend directly and print key metrics."""
    pack = build_trial2_beta_sensitivity_operator_level_spectral_projection_followup_pack()
    print("[trial2-beta-operator-level-spectral-projection-followup]")
    print(f"beta_common_root = {pack['beta_common_root']:.16f}")
    print(
        "control_window = "
        f"[{pack['control_window_x_min']:.2f}, {pack['control_window_x_max']:.2f}]"
    )
    for order, order_row in pack["weighted_integral_by_order"].items():
        print(
            f"n={order} "
            f"dI/dbeta range="
            f"[{order_row['d_integral_order_dbeta_min']:.16f}, "
            f"{order_row['d_integral_order_dbeta_max']:.16f}] "
            f"boundary_abs_frac_max={order_row['boundary_complement_abs_fraction_max']:.16f}"
        )

    print(
        "delta_common_derivative range = "
        f"[{pack['delta_common_derivative_min']:.16f}, "
        f"{pack['delta_common_derivative_max']:.16f}]"
    )
    print(
        "weighted_integral_sign_support_available_now = "
        f"{pack['exact_trial2_beta_sensitivity_weighted_integral_sign_support_available_now']}"
    )
    print(
        "derivative_chain_followup_required_now = "
        f"{pack['updated_pack_trial2_beta_sensitivity_derivative_chain_followup_required_now']}"
    )


if __name__ == "__main__":
    main()
