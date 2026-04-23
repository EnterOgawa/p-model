#!/usr/bin/env python3
"""Audit exact-alpha closed-form extraction after invariant reduction.

Purpose:
    Once the common-root selector has been reduced to a finite invariant
    algebra, the next exact-goal question is no longer whether alpha admits an
    exact formula, but whether that exact formula collapses all the way to the
    zero-residual constant target

        alpha = 1 / 137.

    This helper keeps the newly obtained finite-invariant exact alpha formula
    fixed and classifies whether the current pack actually extracts the target
    constant, or only one exact finite-invariant expression evaluated at the
    retained symbolic root.

Inputs:
    - scripts/quantum/trial2_invariant_reduction_backend.py

Outputs:
    - One in-memory audit pack consumed by `.5795-.5798` wrappers

Assumptions:
    - No new parameter is introduced
    - 1/137 is the exact-goal audit comparator
    - The observed alpha_target remains a secondary external comparator only
"""

from __future__ import annotations

import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.quantum.scalar_proxy_alpha_q_curve_backend import ALPHA_TARGET
from scripts.quantum.trial2_invariant_reduction_backend import (
    build_trial2_invariant_reduction_pack,
)


EXACT_ALPHA_GOAL = 1.0 / 137.0
ZERO_TOL = 1.0e-14


# 関数: exact-alpha closed-form extraction の監査 pack を返す。
def build_trial2_exact_alpha_closed_form_extraction_pack() -> dict:
    """Return one audit pack for exact-alpha closed-form extraction."""
    invariant_pack = build_trial2_invariant_reduction_pack()
    retained_row = invariant_pack["retained_row"]
    symbolic_row = invariant_pack["symbolic_row"]

    alpha_exact_retained = float(retained_row["alpha_from_reduced_invariants"])
    alpha_exact_symbolic = float(symbolic_row["alpha_from_reduced_invariants"])
    one_over_alpha_symbolic = float(1.0 / alpha_exact_symbolic)

    relative_error_vs_exact_goal = float(
        (alpha_exact_symbolic - EXACT_ALPHA_GOAL) / EXACT_ALPHA_GOAL
    )
    relative_error_vs_observed_target = float(
        (alpha_exact_symbolic - ALPHA_TARGET) / ALPHA_TARGET
    )
    exact_constant_extraction_available_now = bool(
        abs(alpha_exact_symbolic - EXACT_ALPHA_GOAL) <= ZERO_TOL
    )
    observed_target_zero_residual_available_now = bool(
        abs(alpha_exact_symbolic - ALPHA_TARGET) <= ZERO_TOL
    )
    exact_alpha_finite_invariant_form_retained_now = bool(
        invariant_pack["exact_trial2_finite_invariant_alpha_form_available_now"]
    )
    zero_residual_final_theorem_gate_required_now = bool(
        exact_alpha_finite_invariant_form_retained_now
        and not exact_constant_extraction_available_now
    )

    return {
        "alpha_target_observed": float(ALPHA_TARGET),
        "alpha_goal_exact_one_over_137": float(EXACT_ALPHA_GOAL),
        "beta_common_root": float(invariant_pack["beta_common_root"]),
        "beta_symbolic_root": float(invariant_pack["beta_symbolic_root"]),
        "alpha_exact_retained": alpha_exact_retained,
        "alpha_exact_symbolic": alpha_exact_symbolic,
        "one_over_alpha_exact_symbolic": one_over_alpha_symbolic,
        "alpha_exact_symbolic_minus_exact_goal": float(
            alpha_exact_symbolic - EXACT_ALPHA_GOAL
        ),
        "alpha_exact_symbolic_rel_error_vs_exact_goal": relative_error_vs_exact_goal,
        "alpha_exact_symbolic_minus_observed_target": float(
            alpha_exact_symbolic - ALPHA_TARGET
        ),
        "alpha_exact_symbolic_rel_error_vs_observed_target": (
            relative_error_vs_observed_target
        ),
        "exact_trial2_finite_invariant_alpha_form_retained_now": (
            exact_alpha_finite_invariant_form_retained_now
        ),
        "exact_trial2_constant_extraction_one_over_137_available_now": (
            exact_constant_extraction_available_now
        ),
        "exact_trial2_observed_target_zero_residual_available_now": (
            observed_target_zero_residual_available_now
        ),
        "updated_pack_trial2_zero_residual_final_theorem_gate_required_now": (
            zero_residual_final_theorem_gate_required_now
        ),
    }


# 関数: backend 単体実行時に compact summary を表示する。

def main() -> None:
    """Run the exact-alpha extraction audit directly."""
    pack = build_trial2_exact_alpha_closed_form_extraction_pack()
    print("[trial2_exact_alpha_closed_form_extraction_backend]")
    print(f"  beta_symbolic_root = {pack['beta_symbolic_root']:.15f}")
    print(f"  alpha_exact_symbolic = {pack['alpha_exact_symbolic']:.15f}")
    print(f"  1/alpha_exact_symbolic = {pack['one_over_alpha_exact_symbolic']:.12f}")
    print(
        "  exact_one_over_137_available = "
        f"{pack['exact_trial2_constant_extraction_one_over_137_available_now']}"
    )


if __name__ == "__main__":
    main()
