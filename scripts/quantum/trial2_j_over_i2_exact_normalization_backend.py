#!/usr/bin/env python3
"""Audit the J/I2 exact-normalization route for the Trial-2 exact goal.

Purpose:
    After `.5803-.5810`, the current exact-goal pack already fixes one honest
    route ordering:

        1. J / I2 exact normalization
        2. b = B / I2 elimination
        3. q = I4 / I2 elimination
        4. 4D time-component augmentation

    The first question is therefore no longer route discovery. It is whether
    the current 3D exact algebra already contains one additional theorem object
    that collapses

        alpha = f(beta)^2 / (4 pi),  f = J / I2

    all the way to the exact-goal constant alpha = 1/137.

    This helper keeps the current exact formula fixed, rewrites the exact-goal
    condition as one exact normalization target for f(beta), and tests whether
    the current pack already supplies a genuinely new theorem surface for that
    normalization. If not, the honest next route is b-elimination rather than
    replaying the same 3D algebra once more.

Inputs:
    - scripts/quantum/trial2_exact_constant_route_priority_backend.py
    - scripts/quantum/trial2_invariant_reduction_backend.py
    - scripts/quantum/trial2_exact_alpha_closed_form_extraction_backend.py

Outputs:
    - One in-memory audit pack consumed by `.5811-.5814` wrappers

Assumptions:
    - No new parameter is introduced
    - 1/137 is the exact-goal comparator
    - The audit decides theorem availability, not numerical fit quality
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
from scripts.quantum.trial2_exact_constant_route_priority_backend import (
    build_trial2_exact_constant_route_priority_pack,
)
from scripts.quantum.trial2_invariant_reduction_backend import (
    build_trial2_invariant_reduction_pack,
)


# 関数: J/I2 exact-normalization route の audit pack を返す。
def build_trial2_j_over_i2_exact_normalization_pack() -> dict:
    """Return one audit pack for the J/I2 exact-normalization route."""
    priority_pack = build_trial2_exact_constant_route_priority_pack()
    invariant_pack = build_trial2_invariant_reduction_pack()
    extraction_pack = build_trial2_exact_alpha_closed_form_extraction_pack()
    row = priority_pack["symbolic_row"]

    beta_symbolic_root = float(row["beta_symbolic_root"])
    f_beta = float(row["f_beta"])
    f_goal_exact_one_over_137 = float(
        math.sqrt(4.0 * math.pi * extraction_pack["alpha_goal_exact_one_over_137"])
    )
    f_goal_gap = float(f_beta - f_goal_exact_one_over_137)
    f_goal_gap_rel = float(f_goal_gap / f_goal_exact_one_over_137)

    exact_alpha_formula_available_now = bool(
        invariant_pack["exact_trial2_finite_invariant_alpha_form_available_now"]
    )
    same_3d_algebra_replay_only_now = bool(
        exact_alpha_formula_available_now
        and extraction_pack["updated_pack_trial2_zero_residual_final_theorem_gate_required_now"]
    )
    exact_trial2_j_over_i2_target_constant_identity_available_now = False
    exact_trial2_j_over_i2_exact_normalization_theorem_available_now = False
    exact_trial2_b_elimination_followup_required_now = bool(
        priority_pack["exact_trial2_b_elimination_secondary_now"]
        and not exact_trial2_j_over_i2_exact_normalization_theorem_available_now
    )
    exact_trial2_q_elimination_reserve_retained_now = bool(
        priority_pack["exact_trial2_q_elimination_reserve_now"]
    )
    exact_trial2_fourd_time_component_hold_retained_now = bool(
        priority_pack["exact_trial2_fourd_time_component_augmentation_hold_now"]
    )

    return {
        "beta_symbolic_root": beta_symbolic_root,
        "f_beta": f_beta,
        "f_goal_exact_one_over_137": f_goal_exact_one_over_137,
        "f_goal_gap": f_goal_gap,
        "f_goal_gap_rel": f_goal_gap_rel,
        "alpha_exact_symbolic": float(extraction_pack["alpha_exact_symbolic"]),
        "alpha_goal_exact_one_over_137": float(
            extraction_pack["alpha_goal_exact_one_over_137"]
        ),
        "alpha_exact_symbolic_rel_error_vs_exact_goal": float(
            extraction_pack["alpha_exact_symbolic_rel_error_vs_exact_goal"]
        ),
        "delta_f_needed_rel_linearized": float(row["delta_f_needed_rel"]),
        "same_3d_algebra_replay_only_now": same_3d_algebra_replay_only_now,
        "exact_trial2_j_over_i2_target_constant_identity_available_now": (
            exact_trial2_j_over_i2_target_constant_identity_available_now
        ),
        "exact_trial2_j_over_i2_exact_normalization_theorem_available_now": (
            exact_trial2_j_over_i2_exact_normalization_theorem_available_now
        ),
        "exact_trial2_b_elimination_followup_required_now": (
            exact_trial2_b_elimination_followup_required_now
        ),
        "exact_trial2_q_elimination_reserve_retained_now": (
            exact_trial2_q_elimination_reserve_retained_now
        ),
        "exact_trial2_fourd_time_component_hold_retained_now": (
            exact_trial2_fourd_time_component_hold_retained_now
        ),
    }


# 関数: backend 単体実行時に compact summary を表示する。

def main() -> None:
    """Run the J/I2 exact-normalization audit directly."""
    pack = build_trial2_j_over_i2_exact_normalization_pack()
    print("[trial2_j_over_i2_exact_normalization_backend]")
    print(f"  beta_symbolic_root = {pack['beta_symbolic_root']:.15f}")
    print(f"  f_beta = {pack['f_beta']:.15f}")
    print(f"  f_goal = {pack['f_goal_exact_one_over_137']:.15f}")
    print(f"  f_goal_gap_rel = {pack['f_goal_gap_rel']:.15e}")
    print(
        "  exact_j_over_i2_normalization_theorem = "
        f"{pack['exact_trial2_j_over_i2_exact_normalization_theorem_available_now']}"
    )


if __name__ == "__main__":
    main()
