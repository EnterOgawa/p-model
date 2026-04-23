#!/usr/bin/env python3
"""Audit 4D full-integral / external-probe current-vertex exactification.

Purpose:
    After selector-level mixed-normalization exactification, the next honest
    computation route is to ask whether the same 4D selector family already
    carries one deterministic external-probe / full-integral weighting rule.

    The current pack keeps the canonical selector family fixed and reuses the
    old 4D current-vertex lesson only as a localization cue:

        J_ext^mu[Q](x) := delta S_frozen[Q;A] / delta A_mu(x) |_(A=0).

    Instead of replaying the old theorem inventory, this helper evaluates the
    current mixed family with the normalized leading nonzero-time selector
    weight interpreted as the external-probe participation fraction of the
    retained full-integral family.

Inputs:
    - scripts/quantum/trial2_selector_4d_mixed_normalization_exactification_backend.py
    - scripts/quantum/trial2_4d_full_mode_summation_directional_check_backend.py

Outputs:
    - One in-memory audit pack consumed by `.5859-.5866` wrappers

Assumptions:
    - The exact goal `1/137` is used only as a comparator
    - No new free parameter is introduced
    - The route remains computation-only and does not claim a selector theorem
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.quantum.trial2_4d_full_mode_summation_directional_check_backend import (
    build_trial2_4d_full_mode_summation_directional_check_pack,
)
from scripts.quantum.trial2_selector_4d_mixed_normalization_exactification_backend import (
    build_trial2_selector_4d_mixed_normalization_exactification_pack,
)
from scripts.quantum.trial2_selector_4d_mixed_normalization_exactification_backend import (
    mixed_alpha,
)


NEAR_EXACT_GOAL_THRESHOLD = 1.0e-5
EXACT_GOAL_ALPHA = 1.0 / 137.0
OBSERVED_TARGET_ALPHA = 1.0 / 137.035999084


# 関数: leading nonzero-time selector row を返す。
def get_leading_weighted_row(weighted_rows: list[dict]) -> dict:
    """Return the deterministic leading weighted row from the current family."""
    for row in weighted_rows:
        if str(row["label"]) == "leading_nontrivial_time_component":
            return dict(row)

    raise ValueError("leading_nontrivial_time_component row is missing")


# 関数: 4D full-integral / external-probe exactification pack を返す。

def build_trial2_4d_full_integral_external_probe_current_vertex_exactification_pack() -> dict:
    """Return the computation-first pack for the 4D full-integral route."""
    mixed_pack = build_trial2_selector_4d_mixed_normalization_exactification_pack()
    directional_pack = build_trial2_4d_full_mode_summation_directional_check_pack()
    canonical_row = dict(directional_pack["canonical_row"])
    best_full_mode_row = dict(directional_pack["best_row"])
    leading_weighted_row = get_leading_weighted_row(best_full_mode_row["weighted_rows"])

    alpha_3d = float(mixed_pack["alpha_3d_exact"])
    charge_factor = float(mixed_pack["canonical_charge_factor"])
    mass_factor = float(mixed_pack["canonical_mass_factor"])
    eta_exact_goal = float(mixed_pack["eta_exact_goal_interpolant"])
    eta_vertex = float(leading_weighted_row["normalized_weight"])
    alpha_vertex = mixed_alpha(alpha_3d, charge_factor, mass_factor, eta_vertex)
    rel_error_vertex = float(
        (alpha_vertex - EXACT_GOAL_ALPHA)
        / EXACT_GOAL_ALPHA
    )
    rel_error_canonical = float(mixed_pack["canonical_rel_error_vs_exact_goal"])
    rel_error_3d = float((alpha_3d - EXACT_GOAL_ALPHA) / EXACT_GOAL_ALPHA)
    rel_error_full_mode = float(best_full_mode_row["corrected_alpha_rel_error_vs_exact_goal"])

    external_probe_surface_explicit = True
    weight_candidate_available = True
    positive_partial = bool(abs(rel_error_vertex) < abs(rel_error_canonical))
    near_exact_goal = bool(abs(rel_error_vertex) < NEAR_EXACT_GOAL_THRESHOLD)
    zero_residual_available = bool(abs(rel_error_vertex) <= 1.0e-14)
    exact_goal_followup_required = bool(positive_partial and not zero_residual_available)
    sign_flip_bracket_retained = bool(rel_error_canonical < 0.0 < rel_error_full_mode)

    return {
        "alpha_goal_exact_one_over_137": EXACT_GOAL_ALPHA,
        "alpha_target_observed": OBSERVED_TARGET_ALPHA,
        "alpha_3d_exact": alpha_3d,
        "canonical_row": canonical_row,
        "best_full_mode_row": best_full_mode_row,
        "leading_weighted_row": leading_weighted_row,
        "eta_exact_goal_interpolant": eta_exact_goal,
        "eta_vertex_weight_candidate": eta_vertex,
        "eta_vertex_minus_exact_goal_interpolant": float(eta_vertex - eta_exact_goal),
        "eta_vertex_rel_gap_vs_exact_goal_interpolant": float(
            (eta_vertex - eta_exact_goal) / max(abs(eta_exact_goal), 1.0e-30)
        ),
        "alpha_vertex_candidate": alpha_vertex,
        "alpha_vertex_candidate_rel_error_vs_exact_goal": rel_error_vertex,
        "alpha_vertex_candidate_rel_error_vs_canonical_row": float(
            (alpha_vertex - float(canonical_row["corrected_alpha"]))
            / max(abs(float(canonical_row["corrected_alpha"])), 1.0e-30)
        ),
        "alpha_vertex_candidate_rel_error_vs_best_full_mode_row": float(
            (alpha_vertex - float(best_full_mode_row["corrected_alpha"]))
            / max(abs(float(best_full_mode_row["corrected_alpha"])), 1.0e-30)
        ),
        "vertex_candidate_improvement_factor_vs_canonical": float(
            abs(rel_error_canonical) / max(abs(rel_error_vertex), 1.0e-30)
        ),
        "vertex_candidate_improvement_factor_vs_3d": float(
            abs(rel_error_3d) / max(abs(rel_error_vertex), 1.0e-30)
        ),
        "vertex_candidate_improvement_factor_vs_best_full_mode": float(
            abs(rel_error_full_mode) / max(abs(rel_error_vertex), 1.0e-30)
        ),
        "exact_trial2_4d_external_probe_current_vertex_target_surface_explicit_now": (
            external_probe_surface_explicit
        ),
        "exact_trial2_4d_external_probe_weight_candidate_available_now": (
            weight_candidate_available
        ),
        "exact_trial2_4d_external_probe_current_vertex_positive_partial_now": (
            positive_partial
        ),
        "exact_trial2_4d_external_probe_current_vertex_near_exact_goal_now": (
            near_exact_goal
        ),
        "exact_trial2_4d_zero_residual_exact_goal_available_now": (
            zero_residual_available
        ),
        "exact_trial2_4d_exact_goal_closeout_followup_required_now": (
            exact_goal_followup_required
        ),
        "exact_trial2_4d_sign_flip_bracket_retained_now": sign_flip_bracket_retained,
    }


# 関数: backend 単体実行時に compact summary を表示する。

def main() -> None:
    """Run the 4D full-integral / external-probe exactification directly."""
    pack = build_trial2_4d_full_integral_external_probe_current_vertex_exactification_pack()
    print("[trial2_4d_full_integral_external_probe_current_vertex_exactification_backend]")
    print(f"  eta_vertex = {pack['eta_vertex_weight_candidate']:.15f}")
    print(f"  alpha_vertex = {pack['alpha_vertex_candidate']:.15f}")
    print(
        "  rel_error_vs_exact_goal = "
        f"{pack['alpha_vertex_candidate_rel_error_vs_exact_goal']:+.12e}"
    )
    print(
        "  improvement_vs_canonical = "
        f"{pack['vertex_candidate_improvement_factor_vs_canonical']:.12f}"
    )


if __name__ == "__main__":
    main()
