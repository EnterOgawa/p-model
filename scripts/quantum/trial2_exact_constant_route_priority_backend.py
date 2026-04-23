#!/usr/bin/env python3
"""Audit route priority for zero-residual exact-constant extraction.

Purpose:
    The current pack already provides one exact finite-invariant alpha formula
    at the symbolic selector root, but that value still misses the exact goal
    alpha = 1 / 137 by a small positive residual. The honest next question is
    no longer whether the selector exists, but which exact-constant route
    should be promoted first.

    This helper classifies the remaining routes using the current symbolic-row
    formula itself. It computes:

        alpha = f(beta)^2 / (4 pi)
        f = J / I2
        g = Ig / I2
        q = I4 / I2
        b = B / I2

    together with local sensitivities and the compensating shifts required to
    remove the exact-goal residual.

Inputs:
    - scripts/quantum/trial2_invariant_reduction_backend.py
    - scripts/quantum/trial2_exact_alpha_closed_form_extraction_backend.py

Outputs:
    - One in-memory audit pack consumed by `.5803-.5810` wrappers

Assumptions:
    - No new parameter is introduced
    - 1/137 is the exact-goal comparator
    - Route ordering is chosen from the current exact formula, not by fitting
"""

from __future__ import annotations

import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.quantum.trial2_exact_alpha_closed_form_extraction_backend import (
    build_trial2_exact_alpha_closed_form_extraction_pack,
)
from scripts.quantum.trial2_invariant_reduction_backend import (
    build_trial2_invariant_reduction_pack,
)


LARGE_REL_SHIFT = 0.1


# 関数: 1つの symbolic-row route-priority row を返す。
def build_exact_constant_route_priority_row() -> dict:
    """Return one symbolic-row residual-priority row."""
    invariant_pack = build_trial2_invariant_reduction_pack()
    extraction_pack = build_trial2_exact_alpha_closed_form_extraction_pack()
    row = invariant_pack["symbolic_row"]

    beta = float(row["beta"])
    epsilon_beta = float(row["epsilon_beta"])
    f_beta = float(row["f_beta"])
    g_beta = float(row["g_beta"])
    q_beta = float(row["q_beta"])
    b_beta = float(row["b_beta"])
    alpha_exact_symbolic = float(extraction_pack["alpha_exact_symbolic"])
    alpha_goal = float(extraction_pack["alpha_goal_exact_one_over_137"])
    alpha_goal_gap = float(alpha_exact_symbolic - alpha_goal)

    one_plus_beta_sq = float(1.0 + beta * beta)
    reduced_cubic_factor = float(4.0 * (g_beta + epsilon_beta - b_beta) - q_beta)
    reduced_total_factor = float(
        2.0 * (5.0 + beta * beta) + 10.0 * g_beta - q_beta - 4.0 * b_beta
    )
    denominator = float(36.0 * one_plus_beta_sq * one_plus_beta_sq)

    d_alpha_df = float(f_beta / (2.0 * math.pi))
    d_alpha_dg = float(
        (4.0 * reduced_total_factor + 10.0 * reduced_cubic_factor) / denominator
    )
    d_alpha_dq = float(
        (-reduced_total_factor - reduced_cubic_factor) / denominator
    )
    d_alpha_db = float(
        (-4.0 * reduced_total_factor - 4.0 * reduced_cubic_factor) / denominator
    )

    delta_f_needed = float(-alpha_goal_gap / d_alpha_df)
    delta_g_needed = float(-alpha_goal_gap / d_alpha_dg)
    delta_q_needed = float(-alpha_goal_gap / d_alpha_dq)
    delta_b_needed = float(-alpha_goal_gap / d_alpha_db)

    return {
        "beta_symbolic_root": beta,
        "epsilon_beta": epsilon_beta,
        "f_beta": f_beta,
        "g_beta": g_beta,
        "q_beta": q_beta,
        "b_beta": b_beta,
        "alpha_exact_symbolic": alpha_exact_symbolic,
        "alpha_goal_exact_one_over_137": alpha_goal,
        "alpha_goal_gap": alpha_goal_gap,
        "alpha_goal_gap_rel": float(alpha_goal_gap / alpha_goal),
        "d_alpha_df": d_alpha_df,
        "d_alpha_dg": d_alpha_dg,
        "d_alpha_dq": d_alpha_dq,
        "d_alpha_db": d_alpha_db,
        "delta_f_needed": delta_f_needed,
        "delta_g_needed": delta_g_needed,
        "delta_q_needed": delta_q_needed,
        "delta_b_needed": delta_b_needed,
        "delta_f_needed_rel": float(delta_f_needed / f_beta),
        "delta_g_needed_rel": float(delta_g_needed / g_beta),
        "delta_q_needed_rel": float(delta_q_needed / q_beta),
        "delta_b_needed_rel": float(delta_b_needed / b_beta),
    }


# 関数: exact-constant route inventory 全体を束ねる。
def build_trial2_exact_constant_route_priority_pack() -> dict:
    """Return one audit pack for exact-constant route prioritization."""
    extraction_pack = build_trial2_exact_alpha_closed_form_extraction_pack()
    row = build_exact_constant_route_priority_row()

    exact_constant_unavailable = not bool(
        extraction_pack["exact_trial2_constant_extraction_one_over_137_available_now"]
    )
    j_over_i2_primary_now = bool(
        exact_constant_unavailable
        and abs(row["delta_f_needed_rel"]) < abs(row["delta_b_needed_rel"])
        and abs(row["delta_f_needed_rel"]) < abs(row["delta_q_needed_rel"])
    )
    b_elimination_secondary_now = bool(
        exact_constant_unavailable
        and abs(row["delta_b_needed_rel"]) < abs(row["delta_q_needed_rel"])
    )
    q_elimination_reserve_now = bool(
        exact_constant_unavailable
        and abs(row["delta_q_needed_rel"]) >= LARGE_REL_SHIFT
    )
    fourd_augmentation_hold_now = True
    j_over_i2_normalization_audit_required_now = bool(j_over_i2_primary_now)

    return {
        "symbolic_row": row,
        "exact_trial2_same_3d_invariant_algebra_zero_residual_available_now": False,
        "exact_trial2_j_over_i2_normalization_primary_now": j_over_i2_primary_now,
        "exact_trial2_b_elimination_secondary_now": b_elimination_secondary_now,
        "exact_trial2_q_elimination_reserve_now": q_elimination_reserve_now,
        "exact_trial2_fourd_time_component_augmentation_hold_now": (
            fourd_augmentation_hold_now
        ),
        "updated_pack_trial2_j_over_i2_normalization_audit_required_now": (
            j_over_i2_normalization_audit_required_now
        ),
    }


# 関数: backend 単体実行時に compact summary を表示する。
def main() -> None:
    """Run the exact-constant route-priority audit directly."""
    pack = build_trial2_exact_constant_route_priority_pack()
    row = pack["symbolic_row"]
    print("[trial2_exact_constant_route_priority_backend]")
    print(f"  beta_symbolic_root = {row['beta_symbolic_root']:.15f}")
    print(f"  alpha_goal_gap = {row['alpha_goal_gap']:.15e}")
    print(f"  delta_f_needed_rel = {row['delta_f_needed_rel']:.15e}")
    print(f"  delta_b_needed_rel = {row['delta_b_needed_rel']:.15e}")
    print(f"  delta_q_needed_rel = {row['delta_q_needed_rel']:.15e}")
    print(
        "  j_over_i2_primary = "
        f"{pack['exact_trial2_j_over_i2_normalization_primary_now']}"
    )


if __name__ == "__main__":
    main()
