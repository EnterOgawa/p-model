#!/usr/bin/env python3
"""Audit selector-level 4D mixed-normalization exactification.

Purpose:
    After full mode accumulation closes negatively, the next honest question is
    whether the missing exact-goal law lives inside the same canonical selector
    family as a mixed charge-mass normalization rule.

    This helper keeps the leading selector `(ell, s) = (1, ±1)` fixed and
    studies the one-parameter family

        alpha_4D,mix(eta) = alpha_3D / (C_4D^eta M_4D^(2-eta))

    which interpolates continuously between

        eta = 0   -> mass_sq_inv       = alpha_3D / M_4D^2
        eta = 1   -> charge_mass_inv   = alpha_3D / (C_4D M_4D)

    Because the two endpoints bracket the exact goal `1/137`, the family has
    one unique exact-goal interpolant `eta_*`. The route then asks whether the
    most natural deterministic candidate available inside the current selector
    family, namely the normalized leading-selector weight in the nonzero-time
    family, nearly saturates that interpolant.

Inputs:
    - scripts/quantum/trial2_4d_exact_constant_selector_theorem_backend.py
    - scripts/quantum/trial2_4d_full_mode_summation_directional_check_backend.py

Outputs:
    - One in-memory audit pack consumed by `.5851-.5858` wrappers

Assumptions:
    - The canonical 4D selector theorem is already fixed
    - The exact goal `1/137` is used only as a comparator
    - No new parameter is introduced
"""

from __future__ import annotations

import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.quantum.trial2_4d_full_mode_summation_directional_check_backend import (
    build_trial2_4d_full_mode_summation_directional_check_pack,
)
from scripts.quantum.trial2_4d_time_component_augmentation_backend import EXACT_GOAL


HALF_POWER_ETA = 0.5


# 関数: mixed-normalization corrected alpha を返す。
def mixed_alpha(alpha_3d: float, charge_factor: float, mass_factor: float, eta: float) -> float:
    """Return alpha_3D divided by the mixed selector-level denominator."""
    return float(alpha_3d / (charge_factor**eta * mass_factor ** (2.0 - eta)))


# 関数: exact-goal interpolant eta_* を返す。

def solve_eta_star(alpha_3d: float, charge_factor: float, mass_factor: float) -> float:
    """Return the unique eta where alpha_4D,mix(eta) = 1/137."""
    numerator = math.log(alpha_3d / EXACT_GOAL) - 2.0 * math.log(mass_factor)
    denominator = math.log(charge_factor / mass_factor)
    return float(numerator / denominator)


# 関数: mixed-normalization exactification pack を返す。

def build_trial2_selector_4d_mixed_normalization_exactification_pack() -> dict:
    """Return the selector-level 4D mixed-normalization audit pack."""
    directional_pack = build_trial2_4d_full_mode_summation_directional_check_pack()
    canonical_row = dict(directional_pack["canonical_row"])
    alpha_3d = float(directional_pack["alpha_exact_symbolic"])
    charge_factor = float(
        directional_pack["best_row"]["weighted_rows"][0]["charge_factor"]
    )
    mass_factor = float(
        directional_pack["best_row"]["weighted_rows"][0]["mass_factor"]
    )
    eta_exact_goal = solve_eta_star(alpha_3d, charge_factor, mass_factor)
    eta_half_power = float(HALF_POWER_ETA)
    eta_weighted_candidate = float(
        directional_pack["best_row"]["weighted_rows"][0]["normalized_weight"]
    )

    alpha_half_power = mixed_alpha(alpha_3d, charge_factor, mass_factor, eta_half_power)
    alpha_weighted_candidate = mixed_alpha(
        alpha_3d, charge_factor, mass_factor, eta_weighted_candidate
    )
    rel_error_half_power = float((alpha_half_power - EXACT_GOAL) / EXACT_GOAL)
    rel_error_weighted_candidate = float(
        (alpha_weighted_candidate - EXACT_GOAL) / EXACT_GOAL
    )
    rel_error_canonical = float(canonical_row["corrected_alpha_rel_error_vs_exact_goal"])

    unique_eta_star_now = bool(0.0 < eta_exact_goal < 1.0)
    weighted_candidate_improves_canonical = bool(
        abs(rel_error_weighted_candidate) < abs(rel_error_canonical)
    )
    weighted_candidate_near_exact_goal = bool(abs(rel_error_weighted_candidate) < 1.0e-5)
    selector_level_positive_partial_now = bool(
        unique_eta_star_now
        and weighted_candidate_improves_canonical
        and weighted_candidate_near_exact_goal
    )
    zero_residual_theorem_available_now = bool(abs(rel_error_weighted_candidate) <= 1.0e-14)

    return {
        "alpha_3d_exact": alpha_3d,
        "canonical_row": canonical_row,
        "canonical_charge_factor": charge_factor,
        "canonical_mass_factor": mass_factor,
        "eta_exact_goal_interpolant": eta_exact_goal,
        "eta_half_power_candidate": eta_half_power,
        "eta_weighted_candidate": eta_weighted_candidate,
        "eta_weighted_minus_exact_goal_interpolant": float(
            eta_weighted_candidate - eta_exact_goal
        ),
        "eta_weighted_rel_gap_vs_exact_goal_interpolant": float(
            (eta_weighted_candidate - eta_exact_goal) / max(abs(eta_exact_goal), 1.0e-30)
        ),
        "alpha_half_power_candidate": alpha_half_power,
        "alpha_half_power_rel_error_vs_exact_goal": rel_error_half_power,
        "alpha_weighted_candidate": alpha_weighted_candidate,
        "alpha_weighted_candidate_rel_error_vs_exact_goal": rel_error_weighted_candidate,
        "canonical_rel_error_vs_exact_goal": rel_error_canonical,
        "weighted_candidate_improvement_factor_vs_canonical": float(
            abs(rel_error_canonical) / max(abs(rel_error_weighted_candidate), 1.0e-30)
        ),
        "weighted_candidate_improvement_factor_vs_3d": float(
            abs(float(directional_pack["alpha_exact_symbolic_rel_error_vs_exact_goal"]))
            / max(abs(rel_error_weighted_candidate), 1.0e-30)
        ),
        "exact_trial2_selector_4d_unique_exact_goal_interpolant_now": unique_eta_star_now,
        "exact_trial2_selector_4d_weighted_candidate_improves_canonical_now": (
            weighted_candidate_improves_canonical
        ),
        "exact_trial2_selector_4d_weighted_candidate_near_exact_goal_now": (
            weighted_candidate_near_exact_goal
        ),
        "exact_trial2_selector_4d_mixed_normalization_positive_partial_now": (
            selector_level_positive_partial_now
        ),
        "exact_trial2_selector_4d_zero_residual_theorem_available_now": (
            zero_residual_theorem_available_now
        ),
    }


# 関数: backend 単体実行時に compact summary を表示する。

def main() -> None:
    """Run the selector-level mixed-normalization audit directly."""
    pack = build_trial2_selector_4d_mixed_normalization_exactification_pack()
    print("[trial2_selector_4d_mixed_normalization_exactification_backend]")
    print(f"  eta_exact_goal_interpolant = {pack['eta_exact_goal_interpolant']:.15f}")
    print(f"  eta_weighted_candidate = {pack['eta_weighted_candidate']:.15f}")
    print(f"  alpha_weighted_candidate = {pack['alpha_weighted_candidate']:.15f}")
    print(
        "  rel_error_vs_exact_goal = "
        f"{pack['alpha_weighted_candidate_rel_error_vs_exact_goal']:+.12e}"
    )
    print(
        "  improvement_vs_canonical = "
        f"{pack['weighted_candidate_improvement_factor_vs_canonical']:.12f}"
    )


if __name__ == "__main__":
    main()
