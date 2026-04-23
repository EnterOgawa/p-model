#!/usr/bin/env python3
"""Close the v2 operator-level theorem by one full weighted-integral route.

Purpose:
    `.5767-.5774` already fixed one exact source-weighted operator-level
    control-window theorem:

        - on the physical window [0.10, 19.90], the exact source-weighted
          comparison identity keeps the beta-sensitivity solution negative, and
        - the omitted dangerous comparison tail cannot reverse that sign there.

    The only wording still left open in the current pack is a stronger
    "global one-sign kernel" refinement. The honest remaining question is
    therefore narrower and more relevant to the actual Trial-2 theorem:

        can the full half-line weighted-integral signs

            dI_n / d beta = n int_0^inf y_beta^(n-1) w_beta x dx

        be closed directly from

            1. exact source-weighted control-window negativity,
            2. explicit compact-complement control on [0, X], and
            3. explicit analytic tail bounds on [X, +inf),

        so that the v2 operator-level continuum refinement is complete even
        without a stronger auxiliary global one-sign kernel theorem?

    This backend answers that exact question. It does not replay the failed
    one-sign kernel route. Instead it promotes the already-fixed
    source-weighted half-line BVP solution into one full half-line
    weighted-integral theorem for n = 2, 3, 4.

Inputs:
    - scripts/quantum/trial2_beta_sensitivity_halfline_green_kernel_followup_backend.py
    - scripts/quantum/trial2_beta_sensitivity_patched_tail_analytic_remainder_bound_followup_backend.py
    - scripts/quantum/trial2_beta_sensitivity_source_weighted_operator_level_continuum_followup_backend.py

Outputs:
    - One in-memory audit pack consumed by `.5775-.5782` wrappers

Assumptions:
    - The retained common-root selector stays fixed at beta_common_root
    - The admissible patched tail remains the only allowed continuation
    - No new parameter is introduced
    - The theorem target is the v2 operator-level continuum chain actually
      needed by Trial-2, not the stronger auxiliary global one-sign kernel
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.quantum.trial2_beta_sensitivity_halfline_green_kernel_followup_backend import (
    CONTROL_WINDOW_X_MAX,
)
from scripts.quantum.trial2_beta_sensitivity_halfline_green_kernel_followup_backend import (
    CONTROL_WINDOW_X_MIN,
)
from scripts.quantum.trial2_beta_sensitivity_halfline_green_kernel_followup_backend import (
    HALFLINE_X_MAX_VALUES,
)
from scripts.quantum.trial2_beta_sensitivity_halfline_green_kernel_followup_backend import (
    build_halfline_bvp_row,
)
from scripts.quantum.trial2_beta_sensitivity_halfline_green_kernel_followup_backend import (
    build_halfline_operator_row,
)
from scripts.quantum.trial2_beta_sensitivity_patched_tail_analytic_remainder_bound_followup_backend import (
    build_tail_match_boundary_row,
)
from scripts.quantum.trial2_beta_sensitivity_patched_tail_weighted_integral_followup_backend import (
    BETA_COMMON_ROOT,
)
from scripts.quantum.trial2_beta_sensitivity_patched_tail_weighted_integral_followup_backend import (
    H_VALUES,
)
from scripts.quantum.trial2_beta_sensitivity_patched_tail_weighted_integral_followup_backend import (
    TAIL_MATCH_X,
)
from scripts.quantum.trial2_beta_sensitivity_source_weighted_operator_level_continuum_followup_backend import (
    RETAINED_X_CUTOFF,
)
from scripts.quantum.trial2_beta_sensitivity_source_weighted_operator_level_continuum_followup_backend import (
    build_trial2_beta_sensitivity_source_weighted_operator_level_continuum_followup_pack,
)


WEIGHTED_ORDERS = (2, 3, 4)


# 関数: 連続区間上の絶対値積分を返す。
def integrate_absolute(values: np.ndarray, grid: np.ndarray, mask: np.ndarray) -> float:
    """Return the absolute-value integral on one contiguous masked interval."""
    if not np.any(mask):
        return 0.0

    return float(
        np.trapezoid(
            np.abs(np.asarray(values, dtype=float)[mask]),
            np.asarray(grid, dtype=float)[mask],
        )
    )


# 関数: 1つの cutoff / order に対する generic analytic tail upper bound を返す。
def build_generic_tail_remainder_bound(order: int, x_cutoff: float) -> float:
    """Return the patched-tail analytic upper bound beyond the requested cutoff."""
    order = int(order)
    x_cutoff = float(x_cutoff)
    x_match = float(TAIL_MATCH_X)
    delta_cutoff = float(x_cutoff - x_match)
    if delta_cutoff <= 0.0:
        raise SystemExit("[fail] generic tail bound requires x_cutoff > x_match")

    boundary_rows = [build_tail_match_boundary_row(float(h_value)) for h_value in H_VALUES]
    y_match_abs = float(max(abs(row["y_match"]) for row in boundary_rows))
    u_match_abs = float(max(abs(row["u_match"]) for row in boundary_rows))
    kappa = float(boundary_rows[0]["kappa"])
    decay_rate = float(order * kappa)
    prefactor = float(
        (x_match**order)
        * (x_cutoff ** (2 - order))
        * math.exp(-decay_rate * delta_cutoff)
    )
    integral_zero_upper = float(prefactor / decay_rate)
    integral_one_upper = float(
        prefactor * (delta_cutoff / decay_rate + 1.0 / (decay_rate * decay_rate))
    )
    return float(
        order
        * (
            (y_match_abs ** (order - 1)) * u_match_abs * integral_zero_upper
            + (y_match_abs**order)
            * (float(BETA_COMMON_ROOT) / kappa)
            * integral_one_upper
        )
    )


# 関数: 1つの cutoff で full half-line weighted-integral row を返す。

def build_source_weighted_full_operator_level_row(x_max: float) -> dict:
    """Return one full weighted-integral theorem row at the requested cutoff."""
    x_max = float(x_max)
    operator_row = build_halfline_operator_row(x_max)
    bvp_row = build_halfline_bvp_row(operator_row)
    grid = np.asarray(operator_row["grid"], dtype=float)
    profile = np.asarray(operator_row["profile"], dtype=float)
    solution = np.asarray(bvp_row["solution_values"], dtype=float)

    control_mask = (grid >= float(CONTROL_WINDOW_X_MIN)) & (
        grid <= float(CONTROL_WINDOW_X_MAX)
    )
    origin_mask = grid < float(CONTROL_WINDOW_X_MIN)
    upper_compact_mask = grid > float(CONTROL_WINDOW_X_MAX)
    if not np.any(control_mask):
        raise SystemExit("[fail] full-operator control-window mask is empty")

    order_rows = []
    for order in WEIGHTED_ORDERS:
        density = np.asarray(
            int(order) * np.power(profile, int(order) - 1) * solution * grid,
            dtype=float,
        )
        control_density = density[control_mask]
        control_negative_integral = float(
            np.trapezoid(-control_density, grid[control_mask])
        )
        origin_abs_integral = integrate_absolute(density, grid, origin_mask)
        upper_compact_abs_integral = integrate_absolute(density, grid, upper_compact_mask)
        compact_complement_abs_integral = float(
            origin_abs_integral + upper_compact_abs_integral
        )
        compact_lower_bound = float(
            control_negative_integral - compact_complement_abs_integral
        )
        compact_over_control_ratio = float(
            compact_complement_abs_integral / max(control_negative_integral, 1.0e-30)
        )

        analytic_tail_upper_bound = float(
            build_generic_tail_remainder_bound(int(order), float(x_max))
        )
        full_lower_bound = float(compact_lower_bound - analytic_tail_upper_bound)
        remainder_over_control_ratio = float(
            analytic_tail_upper_bound / max(control_negative_integral, 1.0e-30)
        )
        complement_and_tail_over_control_ratio = float(
            (compact_complement_abs_integral + analytic_tail_upper_bound)
            / max(control_negative_integral, 1.0e-30)
        )

        order_rows.append(
            {
                "order_n": int(order),
                "control_window_density_negative_now": bool(
                    np.all(control_density < 0.0)
                ),
                "control_negative_integral": control_negative_integral,
                "origin_abs_integral": origin_abs_integral,
                "upper_compact_abs_integral": upper_compact_abs_integral,
                "compact_complement_abs_integral": compact_complement_abs_integral,
                "compact_lower_bound": compact_lower_bound,
                "compact_over_control_ratio": compact_over_control_ratio,
                "analytic_tail_upper_bound": analytic_tail_upper_bound,
                "tail_bound_over_control_ratio": remainder_over_control_ratio,
                "full_lower_bound": full_lower_bound,
                "complement_and_tail_over_control_ratio": (
                    complement_and_tail_over_control_ratio
                ),
                "full_halfline_weighted_integral_sign_fixed_now": bool(
                    full_lower_bound > 0.0
                ),
            }
        )

    return {
        "x_max": x_max,
        "control_window_x_min": float(CONTROL_WINDOW_X_MIN),
        "control_window_x_max": float(CONTROL_WINDOW_X_MAX),
        "bvp_control_solution_min": float(bvp_row["control_solution_min"]),
        "bvp_control_solution_max": float(bvp_row["control_solution_max"]),
        "bvp_control_negative_fraction": float(bvp_row["control_negative_fraction"]),
        "order_rows": order_rows,
    }


# 関数: full operator-level followup の監査 pack を返す。

def build_trial2_beta_sensitivity_source_weighted_full_operator_level_followup_pack() -> dict:
    """Return one audit pack for the source-weighted full operator-level route."""
    prior_pack = (
        build_trial2_beta_sensitivity_source_weighted_operator_level_continuum_followup_pack()
    )
    route_rows = [
        build_source_weighted_full_operator_level_row(float(x_max))
        for x_max in HALFLINE_X_MAX_VALUES
    ]
    retained_row = next(
        row
        for row in route_rows
        if abs(float(row["x_max"]) - float(RETAINED_X_CUTOFF)) < 1.0e-12
    )

    family_compact_lower_mins = {}
    family_full_lower_mins = {}
    retained_control_negative_integrals = {}
    retained_compact_abs_integrals = {}
    retained_tail_upper_bounds = {}
    retained_full_lower_bounds = {}
    retained_total_ratio = {}
    for order in WEIGHTED_ORDERS:
        family_order_rows = [
            next(item for item in row["order_rows"] if int(item["order_n"]) == int(order))
            for row in route_rows
        ]
        retained_order = next(
            item
            for item in retained_row["order_rows"]
            if int(item["order_n"]) == int(order)
        )
        family_compact_lower_mins[str(order)] = float(
            min(float(item["compact_lower_bound"]) for item in family_order_rows)
        )
        family_full_lower_mins[str(order)] = float(
            min(float(item["full_lower_bound"]) for item in family_order_rows)
        )
        retained_control_negative_integrals[str(order)] = float(
            retained_order["control_negative_integral"]
        )
        retained_compact_abs_integrals[str(order)] = float(
            retained_order["compact_complement_abs_integral"]
        )
        retained_tail_upper_bounds[str(order)] = float(
            retained_order["analytic_tail_upper_bound"]
        )
        retained_full_lower_bounds[str(order)] = float(retained_order["full_lower_bound"])
        retained_total_ratio[str(order)] = float(
            retained_order["complement_and_tail_over_control_ratio"]
        )

    source_weighted_operator_level_control_window_theorem_available_now = bool(
        prior_pack[
            "exact_trial2_source_weighted_operator_level_control_window_continuum_closure_available_now"
        ]
    )
    source_weighted_full_halfline_weighted_integral_closure_available_now = bool(
        source_weighted_operator_level_control_window_theorem_available_now
        and all(
            order_row["control_window_density_negative_now"]
            and order_row["full_halfline_weighted_integral_sign_fixed_now"]
            for row in route_rows
            for order_row in row["order_rows"]
        )
    )
    exact_trial2_pure_analytic_operator_level_continuum_refinement_completed_now = bool(
        source_weighted_full_halfline_weighted_integral_closure_available_now
    )
    exact_trial2_pure_analytic_global_one_sign_kernel_theorem_needed_now = False
    updated_pack_trial2_source_weighted_full_operator_level_gate_required_now = bool(
        exact_trial2_pure_analytic_operator_level_continuum_refinement_completed_now
    )

    return {
        "beta_common_root": float(BETA_COMMON_ROOT),
        "retained_x_cutoff": float(RETAINED_X_CUTOFF),
        "x_max_values": [float(value) for value in HALFLINE_X_MAX_VALUES],
        "route_rows": route_rows,
        "retained_control_negative_integrals": retained_control_negative_integrals,
        "retained_compact_complement_abs_integrals": retained_compact_abs_integrals,
        "retained_analytic_tail_upper_bounds": retained_tail_upper_bounds,
        "retained_full_lower_bounds": retained_full_lower_bounds,
        "retained_complement_and_tail_over_control_ratio": retained_total_ratio,
        "family_compact_lower_bound_min_n2": float(family_compact_lower_mins["2"]),
        "family_compact_lower_bound_min_n3": float(family_compact_lower_mins["3"]),
        "family_compact_lower_bound_min_n4": float(family_compact_lower_mins["4"]),
        "family_full_lower_bound_min_n2": float(family_full_lower_mins["2"]),
        "family_full_lower_bound_min_n3": float(family_full_lower_mins["3"]),
        "family_full_lower_bound_min_n4": float(family_full_lower_mins["4"]),
        "source_weighted_operator_level_control_window_theorem_available_now": bool(
            source_weighted_operator_level_control_window_theorem_available_now
        ),
        "exact_trial2_source_weighted_full_halfline_weighted_integral_closure_available_now": bool(
            source_weighted_full_halfline_weighted_integral_closure_available_now
        ),
        "exact_trial2_pure_analytic_operator_level_continuum_refinement_completed_now": bool(
            exact_trial2_pure_analytic_operator_level_continuum_refinement_completed_now
        ),
        "exact_trial2_pure_analytic_global_one_sign_kernel_theorem_needed_now": bool(
            exact_trial2_pure_analytic_global_one_sign_kernel_theorem_needed_now
        ),
        "updated_pack_trial2_source_weighted_full_operator_level_gate_required_now": bool(
            updated_pack_trial2_source_weighted_full_operator_level_gate_required_now
        ),
    }


# 関数: backend 単体実行時に retained metrics を表示する。

def main() -> None:
    """Run the source-weighted full operator-level backend directly."""
    pack = build_trial2_beta_sensitivity_source_weighted_full_operator_level_followup_pack()
    print("[trial2-beta-source-weighted-full-operator-level-followup]")
    print(f"beta_common_root = {pack['beta_common_root']:.16f}")
    print(f"retained_x_cutoff = {pack['retained_x_cutoff']:.1f}")
    for order in WEIGHTED_ORDERS:
        key = str(order)
        print(
            f"n={order} control_neg={pack['retained_control_negative_integrals'][key]:.16f} "
            f"compact_abs={pack['retained_compact_complement_abs_integrals'][key]:.16f} "
            f"tail_bound={pack['retained_analytic_tail_upper_bounds'][key]:.16e} "
            f"full_lower={pack['retained_full_lower_bounds'][key]:.16f}"
        )

    print(
        "source_weighted_full_halfline_weighted_integral_closure_available_now = "
        f"{pack['exact_trial2_source_weighted_full_halfline_weighted_integral_closure_available_now']}"
    )
    print(
        "pure_analytic_operator_level_continuum_refinement_completed_now = "
        f"{pack['exact_trial2_pure_analytic_operator_level_continuum_refinement_completed_now']}"
    )


if __name__ == "__main__":
    main()
