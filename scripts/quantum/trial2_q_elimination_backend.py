#!/usr/bin/env python3
"""Audit q = I4 / I2 elimination for the Trial-2 exact goal.

Purpose:
    After both J/I2 exact normalization and b elimination close negatively,
    the last honest 3D internal route is the quartic reduced invariant

        q(beta) = I4(beta) / I2(beta).

    This helper rewrites the exact-goal condition as a quadratic equation in q,
    checks whether the current pack supplies a selector for its roots, and
    measures how large the near-root shift would have to be.

Inputs:
    - scripts/quantum/trial2_exact_constant_route_priority_backend.py
    - scripts/quantum/trial2_exact_alpha_closed_form_extraction_backend.py

Outputs:
    - One in-memory audit pack consumed by `.5819-.5822` wrappers

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
    LARGE_REL_SHIFT,
    build_trial2_exact_constant_route_priority_pack,
)


# 関数: q elimination route の audit pack を返す。
def build_trial2_q_elimination_pack() -> dict:
    """Return one audit pack for the q-elimination route."""
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

    a0 = float(4.0 * (g_beta + epsilon_beta - b_beta))
    t0 = float(2.0 * (5.0 + beta * beta) + 10.0 * g_beta - 4.0 * b_beta)
    c2 = float(1.0)
    c1 = float(-(a0 + t0))
    c0 = float(a0 * t0 - alpha_goal * denominator)
    discriminant = float(c1 * c1 - 4.0 * c2 * c0)
    sqrt_discriminant = float(math.sqrt(discriminant))
    q_root_near = float((-c1 - sqrt_discriminant) / (2.0 * c2))
    q_root_far = float((-c1 + sqrt_discriminant) / (2.0 * c2))

    q_root_near_rel_shift = float((q_root_near - q_beta) / q_beta)
    q_root_far_rel_shift = float((q_root_far - q_beta) / q_beta)

    exact_trial2_q_root_selector_available_now = False
    exact_trial2_q_elimination_theorem_available_now = False
    exact_trial2_q_order_one_shift_required_now = bool(
        abs(q_root_near_rel_shift) >= LARGE_REL_SHIFT
    )
    exact_trial2_fourd_time_component_augmentation_required_now = bool(
        not exact_trial2_q_elimination_theorem_available_now
    )

    return {
        "beta_symbolic_root": beta,
        "epsilon_beta": epsilon_beta,
        "g_beta": g_beta,
        "q_beta": q_beta,
        "b_beta": b_beta,
        "alpha_goal_exact_one_over_137": alpha_goal,
        "q_quadratic_c2": c2,
        "q_quadratic_c1": c1,
        "q_quadratic_c0": c0,
        "q_quadratic_discriminant": discriminant,
        "q_root_near": q_root_near,
        "q_root_far": q_root_far,
        "q_root_near_rel_shift": q_root_near_rel_shift,
        "q_root_far_rel_shift": q_root_far_rel_shift,
        "exact_trial2_q_root_selector_available_now": (
            exact_trial2_q_root_selector_available_now
        ),
        "exact_trial2_q_elimination_theorem_available_now": (
            exact_trial2_q_elimination_theorem_available_now
        ),
        "exact_trial2_q_order_one_shift_required_now": (
            exact_trial2_q_order_one_shift_required_now
        ),
        "exact_trial2_fourd_time_component_augmentation_required_now": (
            exact_trial2_fourd_time_component_augmentation_required_now
        ),
    }


# 関数: backend 単体実行時に compact summary を表示する。

def main() -> None:
    """Run the q-elimination audit directly."""
    pack = build_trial2_q_elimination_pack()
    print("[trial2_q_elimination_backend]")
    print(f"  q_beta = {pack['q_beta']:.15f}")
    print(f"  q_root_near = {pack['q_root_near']:.15f}")
    print(f"  q_root_near_rel_shift = {pack['q_root_near_rel_shift']:.15e}")
    print(f"  q_root_far = {pack['q_root_far']:.15f}")
    print(
        "  exact_q_elimination_theorem = "
        f"{pack['exact_trial2_q_elimination_theorem_available_now']}"
    )


if __name__ == "__main__":
    main()
