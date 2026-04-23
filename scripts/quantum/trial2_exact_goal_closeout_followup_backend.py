#!/usr/bin/env python3
"""Audit whether the deterministic 4D vertex candidate can close the exact goal.

Purpose:
    The current best deterministic candidate is the 4D external-probe /
    current-vertex readout

        alpha_4D,vertex = alpha_3D / (C_4D^eta_vertex M_4D^(2-eta_vertex)),

    with relative exact-goal residual `+1.3216610545872462e-06`. This helper
    asks one computation-only question:

        is that remaining gap small enough to treat as numerical noise, or does
        the current pack localize it as a genuine selector-weight gap?

    The route keeps the mixed-normalization family fixed and quantifies the
    exact-goal residual by comparing the deterministic weight `eta_vertex`
    against the unique exact-goal interpolant `eta_*`.

Inputs:
    - scripts/quantum/trial2_selector_4d_mixed_normalization_exactification_backend.py
    - scripts/quantum/trial2_4d_full_integral_external_probe_current_vertex_exactification_backend.py

Outputs:
    - One in-memory audit pack consumed by `.5867-.5870` wrappers

Assumptions:
    - The exact goal `1/137` is used only as a comparator
    - No new free parameter is introduced
    - The current pack already fixes `eta_*` and `eta_vertex`
"""

from __future__ import annotations

import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.quantum.trial2_4d_full_integral_external_probe_current_vertex_exactification_backend import (
    build_trial2_4d_full_integral_external_probe_current_vertex_exactification_pack,
)
from scripts.quantum.trial2_selector_4d_mixed_normalization_exactification_backend import (
    build_trial2_selector_4d_mixed_normalization_exactification_pack,
)


ZERO_RESIDUAL_TOL = 1.0e-14
LINEARIZATION_TOL = 1.0e-5
NUMERICAL_NOISE_FLOOR = 1.0e-12


# 関数: exact-goal closeout followup pack を返す。
def build_trial2_exact_goal_closeout_followup_pack() -> dict:
    """Return the computation-first exact-goal closeout followup pack."""
    mixed_pack = build_trial2_selector_4d_mixed_normalization_exactification_pack()
    vertex_pack = (
        build_trial2_4d_full_integral_external_probe_current_vertex_exactification_pack()
    )

    exact_goal_alpha = float(vertex_pack["alpha_goal_exact_one_over_137"])
    alpha_vertex = float(vertex_pack["alpha_vertex_candidate"])
    eta_star = float(mixed_pack["eta_exact_goal_interpolant"])
    eta_vertex = float(vertex_pack["eta_vertex_weight_candidate"])
    charge_factor = float(mixed_pack["canonical_charge_factor"])
    mass_factor = float(mixed_pack["canonical_mass_factor"])

    log_mass_over_charge = float(math.log(mass_factor / charge_factor))
    delta_eta = float(eta_vertex - eta_star)
    alpha_gap_abs = float(alpha_vertex - exact_goal_alpha)
    relative_gap = float(alpha_gap_abs / exact_goal_alpha)
    dalpha_deta_at_goal = float(exact_goal_alpha * log_mass_over_charge)
    linearized_gap_abs = float(dalpha_deta_at_goal * delta_eta)
    linearization_ratio = float(
        alpha_gap_abs / max(linearized_gap_abs, 1.0e-30)
    )
    required_denominator_multiplier = float(alpha_vertex / exact_goal_alpha)
    required_eta_correction = float(eta_star - eta_vertex)
    required_eta_rel_correction_vs_vertex = float(
        required_eta_correction / max(abs(eta_vertex), 1.0e-30)
    )

    residual_localized_as_eta_gap = bool(
        abs(linearization_ratio - 1.0) <= LINEARIZATION_TOL
    )
    not_numerical_noise = bool(abs(alpha_gap_abs) > NUMERICAL_NOISE_FLOOR)
    zero_residual_available = bool(abs(alpha_gap_abs) <= ZERO_RESIDUAL_TOL)
    exact_goal_closeout_unavailable = bool(
        residual_localized_as_eta_gap and not_numerical_noise and not zero_residual_available
    )

    return {
        "alpha_goal_exact_one_over_137": exact_goal_alpha,
        "alpha_vertex_candidate": alpha_vertex,
        "alpha_vertex_candidate_abs_gap_vs_exact_goal": alpha_gap_abs,
        "alpha_vertex_candidate_rel_gap_vs_exact_goal": relative_gap,
        "eta_exact_goal_interpolant": eta_star,
        "eta_vertex_weight_candidate": eta_vertex,
        "delta_eta_vertex_minus_exact_goal_interpolant": delta_eta,
        "canonical_charge_factor": charge_factor,
        "canonical_mass_factor": mass_factor,
        "log_mass_over_charge": log_mass_over_charge,
        "dalpha_deta_at_exact_goal": dalpha_deta_at_goal,
        "linearized_exact_goal_gap_abs": linearized_gap_abs,
        "linearization_ratio_actual_over_linearized": linearization_ratio,
        "required_denominator_multiplier_for_exact_goal": (
            required_denominator_multiplier
        ),
        "required_eta_correction_to_exact_goal": required_eta_correction,
        "required_eta_rel_correction_vs_vertex": required_eta_rel_correction_vs_vertex,
        "exact_trial2_4d_vertex_gap_localized_as_eta_gap_now": (
            residual_localized_as_eta_gap
        ),
        "exact_trial2_4d_vertex_gap_not_numerical_noise_now": not_numerical_noise,
        "exact_trial2_4d_zero_residual_exact_goal_available_now": (
            zero_residual_available
        ),
        "exact_trial2_4d_exact_goal_closeout_unavailable_current_pack_now": (
            exact_goal_closeout_unavailable
        ),
    }


# 関数: backend 単体実行時に compact summary を表示する。

def main() -> None:
    """Run the exact-goal closeout followup directly."""
    pack = build_trial2_exact_goal_closeout_followup_pack()
    print("[trial2_exact_goal_closeout_followup_backend]")
    print(
        "  alpha_gap_abs = "
        f"{pack['alpha_vertex_candidate_abs_gap_vs_exact_goal']:+.12e}"
    )
    print(
        "  delta_eta = "
        f"{pack['delta_eta_vertex_minus_exact_goal_interpolant']:+.12e}"
    )
    print(
        "  linearization_ratio = "
        f"{pack['linearization_ratio_actual_over_linearized']:.12f}"
    )
    print(
        "  required_denominator_multiplier = "
        f"{pack['required_denominator_multiplier_for_exact_goal']:.12f}"
    )


if __name__ == "__main__":
    main()
