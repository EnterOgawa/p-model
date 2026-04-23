#!/usr/bin/env python3
"""Audit b = B / I2 elimination for the Trial-2 exact goal.

Purpose:
    After the J/I2 exact-normalization route closes negatively, the next honest
    3D internal route is whether the reduced boundary-weighted invariant

        b(beta) = B(beta) / I2(beta)

    can be eliminated from the current exact-goal algebra in a way that
    actually selects the exact constant alpha = 1/137.

    The current exact formula is

        alpha(beta)
          = [4(g + eps - b) - q][2(5 + beta^2) + 10 g - q - 4 b]
            / [36 (1 + beta^2)^2].

    This helper rewrites the exact-goal condition as a quadratic equation in b,
    checks whether the current pack already supplies a selector for its roots,
    and classifies the route honestly.

Inputs:
    - scripts/quantum/trial2_exact_constant_route_priority_backend.py
    - scripts/quantum/trial2_exact_alpha_closed_form_extraction_backend.py

Outputs:
    - One in-memory audit pack consumed by `.5815-.5818` wrappers

Assumptions:
    - No new parameter is introduced
    - 1/137 is used only as an audit comparator
    - The task is route classification, not theorem search outside the pack
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


# 関数: b elimination route の audit pack を返す。
def build_trial2_b_elimination_pack() -> dict:
    """Return one audit pack for the b-elimination route."""
    priority_pack = build_trial2_exact_constant_route_priority_pack()
    extraction_pack = build_trial2_exact_alpha_closed_form_extraction_pack()
    row = priority_pack["symbolic_row"]

    beta = float(row["beta_symbolic_root"])
    epsilon_beta = float(row["epsilon_beta"])
    g_beta = float(row["g_beta"])
    q_beta = float(row["q_beta"])
    b_beta = float(row["b_beta"])
    alpha_goal = float(extraction_pack["alpha_goal_exact_one_over_137"])
    denominator = float(36.0 * (1.0 + beta * beta) ** 2)

    a0 = float(4.0 * (g_beta + epsilon_beta) - q_beta)
    t0 = float(2.0 * (5.0 + beta * beta) + 10.0 * g_beta - q_beta)
    c2 = float(16.0)
    c1 = float(-4.0 * (a0 + t0))
    c0 = float(a0 * t0 - alpha_goal * denominator)
    discriminant = float(c1 * c1 - 4.0 * c2 * c0)
    sqrt_discriminant = float(math.sqrt(discriminant))
    b_root_near = float((-c1 - sqrt_discriminant) / (2.0 * c2))
    b_root_far = float((-c1 + sqrt_discriminant) / (2.0 * c2))

    b_root_near_rel_shift = float((b_root_near - b_beta) / b_beta)
    b_root_far_rel_shift = float((b_root_far - b_beta) / b_beta)

    alpha_if_b_zero = float((a0 * t0) / denominator)
    alpha_if_b_zero_rel_error_vs_goal = float((alpha_if_b_zero - alpha_goal) / alpha_goal)

    exact_trial2_b_root_selector_available_now = False
    exact_trial2_b_elimination_theorem_available_now = False
    exact_trial2_q_elimination_followup_required_now = bool(
        priority_pack["exact_trial2_q_elimination_reserve_now"]
        and not exact_trial2_b_elimination_theorem_available_now
    )
    exact_trial2_fourd_time_component_hold_retained_now = bool(
        priority_pack["exact_trial2_fourd_time_component_augmentation_hold_now"]
    )

    return {
        "beta_symbolic_root": beta,
        "epsilon_beta": epsilon_beta,
        "g_beta": g_beta,
        "q_beta": q_beta,
        "b_beta": b_beta,
        "alpha_goal_exact_one_over_137": alpha_goal,
        "b_quadratic_c2": c2,
        "b_quadratic_c1": c1,
        "b_quadratic_c0": c0,
        "b_quadratic_discriminant": discriminant,
        "b_root_near": b_root_near,
        "b_root_far": b_root_far,
        "b_root_near_rel_shift": b_root_near_rel_shift,
        "b_root_far_rel_shift": b_root_far_rel_shift,
        "alpha_if_b_zero": alpha_if_b_zero,
        "alpha_if_b_zero_rel_error_vs_goal": alpha_if_b_zero_rel_error_vs_goal,
        "exact_trial2_b_root_selector_available_now": (
            exact_trial2_b_root_selector_available_now
        ),
        "exact_trial2_b_elimination_theorem_available_now": (
            exact_trial2_b_elimination_theorem_available_now
        ),
        "exact_trial2_q_elimination_followup_required_now": (
            exact_trial2_q_elimination_followup_required_now
        ),
        "exact_trial2_fourd_time_component_hold_retained_now": (
            exact_trial2_fourd_time_component_hold_retained_now
        ),
    }


# 関数: backend 単体実行時に compact summary を表示する。

def main() -> None:
    """Run the b-elimination audit directly."""
    pack = build_trial2_b_elimination_pack()
    print("[trial2_b_elimination_backend]")
    print(f"  b_beta = {pack['b_beta']:.15f}")
    print(f"  b_root_near = {pack['b_root_near']:.15f}")
    print(f"  b_root_near_rel_shift = {pack['b_root_near_rel_shift']:.15e}")
    print(f"  b_root_far = {pack['b_root_far']:.15f}")
    print(
        "  exact_b_elimination_theorem = "
        f"{pack['exact_trial2_b_elimination_theorem_available_now']}"
    )


if __name__ == "__main__":
    main()
