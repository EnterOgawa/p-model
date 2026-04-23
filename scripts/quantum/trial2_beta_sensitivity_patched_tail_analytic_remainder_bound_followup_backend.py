#!/usr/bin/env python3
"""Audit one explicit analytic tail-remainder bound for the patched-tail route.

Purpose:
    Continue the pure-analytic refinement after `.5719-.5726`, where the
    admissible positive-decay patched tail already fixed full-domain weighted
    sign support numerically up to large cutoffs `x_max in {60, 80, 100, 140}`.

    The remaining honest question is narrower:

        once the weighted beta-derivative integrals are known to stay negative
        up to the largest tested cutoff `X = 140`, can one explicit closed-form
        upper bound control the omitted patched tail on `[X, +∞)` so that the
        sign survives the limit `X -> +∞` without another replay?

    This backend does not claim the final operator-level pure analytic theorem.
    It promotes one support layer only:

    1. write the patched tail remainder in an explicit closed-form bound,
    2. compare that bound against the already-fixed negative sign margin at
       the largest tested cutoff, and
    3. decide whether the patched weighted-integral route now admits one
       honest pure-continuum promotion.

Inputs:
    - scripts/quantum/trial2_beta_sensitivity_patched_tail_weighted_integral_followup_backend.py

Outputs:
    - One in-memory audit pack consumed by `.5727-.5734` wrappers

Assumptions:
    - The retained common-root selector remains fixed at beta_common_root
    - The admissible value-matched patch remains matched at x_match = 21.0
    - No new parameter is introduced
    - This route targets the patched-tail remainder only; it does not yet claim
      the final pure analytic operator-level continuum theorem
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.quantum.trial2_beta_sensitivity_patched_tail_weighted_integral_followup_backend import (
    BETA_COMMON_ROOT,
)
from scripts.quantum.trial2_beta_sensitivity_patched_tail_weighted_integral_followup_backend import (
    H_VALUES,
)
from scripts.quantum.trial2_beta_sensitivity_patched_tail_weighted_integral_followup_backend import (
    PATCHED_TAIL_ORDERS,
)
from scripts.quantum.trial2_beta_sensitivity_patched_tail_weighted_integral_followup_backend import (
    PATCHED_TAIL_X_MAX_VALUES,
)
from scripts.quantum.trial2_beta_sensitivity_patched_tail_weighted_integral_followup_backend import (
    TAIL_MATCH_X,
)
from scripts.quantum.trial2_beta_sensitivity_patched_tail_weighted_integral_followup_backend import (
    build_patched_profile_row,
)
from scripts.quantum.trial2_beta_sensitivity_patched_tail_weighted_integral_followup_backend import (
    build_trial2_beta_sensitivity_patched_tail_weighted_integral_followup_pack,
)


PATCHED_REMAINDER_X_CUTOFF = float(PATCHED_TAIL_X_MAX_VALUES[-1])


# 関数: largest retained cutoff で tail-match boundary data を返す。
def build_tail_match_boundary_row(h_value: float) -> dict:
    """Return the patched-tail boundary data needed by the analytic remainder bound."""
    beta_common_root = float(BETA_COMMON_ROOT)
    h_value = float(h_value)
    cutoff = float(PATCHED_REMAINDER_X_CUTOFF)
    center_row = build_patched_profile_row(beta_common_root, cutoff)
    plus_row = build_patched_profile_row(beta_common_root + h_value, cutoff)
    minus_row = build_patched_profile_row(beta_common_root - h_value, cutoff)
    y_match = float(np.interp(float(TAIL_MATCH_X), center_row["radius"], center_row["profile"]))
    y_match_plus = float(
        np.interp(float(TAIL_MATCH_X), plus_row["radius"], plus_row["profile"])
    )
    y_match_minus = float(
        np.interp(float(TAIL_MATCH_X), minus_row["radius"], minus_row["profile"])
    )
    u_match = float((y_match_plus - y_match_minus) / (2.0 * h_value))
    return {
        "h": h_value,
        "x_cutoff": cutoff,
        "y_match": y_match,
        "u_match": u_match,
        "kappa": float(center_row["kappa"]),
    }


# 関数: order ごとの analytic remainder upper bound row を返す。

def build_remainder_bound_row(order: int, boundary_rows: list[dict], weighted_pack: dict) -> dict:
    """Return one closed-form patched-tail remainder bound for the selected order."""
    order = int(order)
    x_match = float(TAIL_MATCH_X)
    x_cutoff = float(PATCHED_REMAINDER_X_CUTOFF)
    delta_cutoff = float(x_cutoff - x_match)
    beta_common_root = float(BETA_COMMON_ROOT)
    y_match_abs = float(max(abs(row["y_match"]) for row in boundary_rows))
    u_match_abs = float(max(abs(row["u_match"]) for row in boundary_rows))
    kappa = float(boundary_rows[0]["kappa"])
    decay_rate = float(order * kappa)
    prefactor = float(
        (x_match**order) * (x_cutoff ** (2 - order)) * math.exp(-decay_rate * delta_cutoff)
    )
    integral_zero_upper = float(prefactor / decay_rate)
    integral_one_upper = float(
        prefactor * (delta_cutoff / decay_rate + 1.0 / (decay_rate * decay_rate))
    )
    remainder_abs_bound = float(
        order
        * (
            (y_match_abs ** (order - 1)) * u_match_abs * integral_zero_upper
            + (y_match_abs**order)
            * (beta_common_root / kappa)
            * integral_one_upper
        )
    )

    cutoff_rows = [
        row
        for row in weighted_pack["weighted_rows"]
        if abs(float(row["x_max"]) - x_cutoff) < 1.0e-12
    ]
    total_rows = [
        next(item for item in row["order_rows"] if int(item["order_n"]) == order)
        for row in cutoff_rows
    ]
    total_abs_min = float(
        min(abs(float(row["weighted_total_integral"])) for row in total_rows)
    )
    total_abs_max = float(
        max(abs(float(row["weighted_total_integral"])) for row in total_rows)
    )
    bound_over_total_abs_min = float(remainder_abs_bound / max(total_abs_min, 1.0e-30))
    nonreversing_now = bool(remainder_abs_bound < total_abs_min)

    return {
        "order_n": order,
        "decay_rate": decay_rate,
        "prefactor": prefactor,
        "integral_zero_upper": integral_zero_upper,
        "integral_one_upper": integral_one_upper,
        "remainder_abs_bound": remainder_abs_bound,
        "total_abs_min_at_cutoff": total_abs_min,
        "total_abs_max_at_cutoff": total_abs_max,
        "bound_over_total_abs_min": bound_over_total_abs_min,
        "analytic_remainder_nonreversing_now": nonreversing_now,
    }


# 関数: patched-tail analytic remainder-bound followup の監査 pack を返す。

def build_trial2_beta_sensitivity_patched_tail_analytic_remainder_bound_followup_pack() -> dict:
    """Return one audit pack for the patched-tail analytic remainder-bound route."""
    weighted_pack = build_trial2_beta_sensitivity_patched_tail_weighted_integral_followup_pack()
    boundary_rows = [build_tail_match_boundary_row(float(h_value)) for h_value in H_VALUES]
    order_rows = [
        build_remainder_bound_row(order, boundary_rows, weighted_pack)
        for order in PATCHED_TAIL_ORDERS
    ]

    analytic_remainder_bound_available_now = bool(
        weighted_pack[
            "exact_trial2_beta_sensitivity_patched_tail_weighted_integral_sign_support_available_now"
        ]
        and all(row["analytic_remainder_nonreversing_now"] for row in order_rows)
    )
    patched_tail_pure_continuum_promotion_available_now = bool(
        analytic_remainder_bound_available_now
        and weighted_pack[
            "exact_trial2_beta_sensitivity_patched_tail_remainder_nonreversing_now"
        ]
        and weighted_pack["exact_trial2_beta_sensitivity_patched_tail_cutoff_stable_now"]
    )
    exact_trial2_pure_analytic_operator_level_continuum_refinement_available_now = False
    updated_pack_trial2_patched_tail_pure_continuum_closure_refresh_required_now = bool(
        patched_tail_pure_continuum_promotion_available_now
        and not exact_trial2_pure_analytic_operator_level_continuum_refinement_available_now
    )

    return {
        "beta_common_root": float(BETA_COMMON_ROOT),
        "tail_match_x": float(TAIL_MATCH_X),
        "x_cutoff": float(PATCHED_REMAINDER_X_CUTOFF),
        "boundary_rows": boundary_rows,
        "order_rows": order_rows,
        "y_match_abs_max": float(max(abs(row["y_match"]) for row in boundary_rows)),
        "u_match_abs_max": float(max(abs(row["u_match"]) for row in boundary_rows)),
        "u_match_min": float(min(float(row["u_match"]) for row in boundary_rows)),
        "u_match_max": float(max(float(row["u_match"]) for row in boundary_rows)),
        "u_match_rel_spread": float(
            (max(float(row["u_match"]) for row in boundary_rows) - min(float(row["u_match"]) for row in boundary_rows))
            / max(abs(np.mean([float(row["u_match"]) for row in boundary_rows])), 1.0e-30)
        ),
        "weighted_integral_sign_support_available_now": bool(
            weighted_pack[
                "exact_trial2_beta_sensitivity_patched_tail_weighted_integral_sign_support_available_now"
            ]
        ),
        "exact_trial2_beta_sensitivity_patched_tail_analytic_remainder_bound_available_now": (
            analytic_remainder_bound_available_now
        ),
        "exact_trial2_beta_sensitivity_patched_tail_pure_continuum_promotion_available_now": (
            patched_tail_pure_continuum_promotion_available_now
        ),
        "exact_trial2_pure_analytic_operator_level_continuum_refinement_available_now": (
            exact_trial2_pure_analytic_operator_level_continuum_refinement_available_now
        ),
        "updated_pack_trial2_patched_tail_pure_continuum_closure_refresh_required_now": (
            updated_pack_trial2_patched_tail_pure_continuum_closure_refresh_required_now
        ),
    }


# 関数: backend 単体実行時に retained metrics を表示する。

def main() -> None:
    """Run the patched-tail analytic remainder-bound backend directly."""
    pack = build_trial2_beta_sensitivity_patched_tail_analytic_remainder_bound_followup_pack()
    print("[trial2-beta-patched-tail-analytic-remainder-bound-followup]")
    print(f"beta_common_root = {pack['beta_common_root']:.16f}")
    print(f"tail_match_x = {pack['tail_match_x']:.1f}")
    print(f"x_cutoff = {pack['x_cutoff']:.1f}")
    print(f"y_match_abs_max = {pack['y_match_abs_max']:.16e}")
    print(f"u_match_abs_max = {pack['u_match_abs_max']:.16e}")
    for row in pack["order_rows"]:
        print(
            f"order={row['order_n']} "
            f"remainder_abs_bound={row['remainder_abs_bound']:.16e} "
            f"total_abs_min_at_cutoff={row['total_abs_min_at_cutoff']:.16e} "
            f"bound_over_total_abs_min={row['bound_over_total_abs_min']:.16e}"
        )

    print(
        "patched_tail_analytic_remainder_bound_available = "
        f"{pack['exact_trial2_beta_sensitivity_patched_tail_analytic_remainder_bound_available_now']}"
    )
    print(
        "patched_tail_pure_continuum_promotion_available = "
        f"{pack['exact_trial2_beta_sensitivity_patched_tail_pure_continuum_promotion_available_now']}"
    )


if __name__ == "__main__":
    main()
