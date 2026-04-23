#!/usr/bin/env python3
"""Audit the sign-flip interpolation geometry around the 4D mixed selector family.

Purpose:
    After the deterministic external-probe candidate is fixed, the remaining
    structural diagnostic is the sign-flip geometry inside the same one-parameter
    mixed-normalization family

        alpha_4D,mix(eta) = alpha_3D / (C_4D^eta M_4D^(2-eta)).

    This helper does not claim a selector theorem. It quantifies the local
    bracket

        alpha_4D,mix(0) < 1/137 < alpha_4D,mix(eta_vertex),

    verifies monotonicity, and records how close the deterministic weight sits
    to the unique interpolant `eta_*`.

Inputs:
    - scripts/quantum/trial2_selector_4d_mixed_normalization_exactification_backend.py
    - scripts/quantum/trial2_4d_full_integral_external_probe_current_vertex_exactification_backend.py

Outputs:
    - One in-memory audit pack consumed by `.5871-.5874` wrappers

Assumptions:
    - The exact goal `1/137` is used only as a comparator
    - The route remains diagnostic and theorem-neutral
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
from scripts.quantum.trial2_selector_4d_mixed_normalization_exactification_backend import (
    mixed_alpha,
)


# 関数: sign-flip interpolation diagnostic pack を返す。
def build_trial2_sign_flip_interpolation_diagnostic_pack() -> dict:
    """Return the sign-flip interpolation diagnostic pack."""
    mixed_pack = build_trial2_selector_4d_mixed_normalization_exactification_pack()
    vertex_pack = (
        build_trial2_4d_full_integral_external_probe_current_vertex_exactification_pack()
    )

    alpha_3d = float(mixed_pack["alpha_3d_exact"])
    charge_factor = float(mixed_pack["canonical_charge_factor"])
    mass_factor = float(mixed_pack["canonical_mass_factor"])
    eta_star = float(mixed_pack["eta_exact_goal_interpolant"])
    eta_vertex = float(vertex_pack["eta_vertex_weight_candidate"])
    exact_goal_alpha = float(vertex_pack["alpha_goal_exact_one_over_137"])

    alpha_eta_zero = mixed_alpha(alpha_3d, charge_factor, mass_factor, 0.0)
    alpha_eta_vertex = float(vertex_pack["alpha_vertex_candidate"])
    alpha_eta_one = mixed_alpha(alpha_3d, charge_factor, mass_factor, 1.0)
    slope_factor = float(math.log(mass_factor / charge_factor))
    derivative_eta_zero = float(alpha_eta_zero * slope_factor)
    derivative_eta_vertex = float(alpha_eta_vertex * slope_factor)
    derivative_eta_one = float(alpha_eta_one * slope_factor)
    eta_star_inside_local_bracket = bool(0.0 < eta_star < eta_vertex)
    monotone_positive = bool(
        derivative_eta_zero > 0.0 and derivative_eta_vertex > 0.0 and derivative_eta_one > 0.0
    )
    local_sign_flip_bracket = bool(alpha_eta_zero < exact_goal_alpha < alpha_eta_vertex)
    global_sign_flip_bracket = bool(alpha_eta_zero < exact_goal_alpha < alpha_eta_one)
    eta_star_position_inside_local_bracket = float(
        eta_star / max(eta_vertex, 1.0e-30)
    )

    return {
        "alpha_goal_exact_one_over_137": exact_goal_alpha,
        "eta_exact_goal_interpolant": eta_star,
        "eta_vertex_weight_candidate": eta_vertex,
        "eta_star_position_inside_local_bracket": eta_star_position_inside_local_bracket,
        "alpha_eta_zero_canonical": alpha_eta_zero,
        "alpha_eta_vertex": alpha_eta_vertex,
        "alpha_eta_one_charge_mass": alpha_eta_one,
        "derivative_eta_zero": derivative_eta_zero,
        "derivative_eta_vertex": derivative_eta_vertex,
        "derivative_eta_one": derivative_eta_one,
        "exact_trial2_sign_flip_local_bracket_available_now": local_sign_flip_bracket,
        "exact_trial2_sign_flip_global_bracket_available_now": global_sign_flip_bracket,
        "exact_trial2_eta_star_inside_local_bracket_now": eta_star_inside_local_bracket,
        "exact_trial2_mixed_family_monotone_positive_now": monotone_positive,
    }


# 関数: backend 単体実行時に compact summary を表示する。

def main() -> None:
    """Run the sign-flip interpolation diagnostic directly."""
    pack = build_trial2_sign_flip_interpolation_diagnostic_pack()
    print("[trial2_sign_flip_interpolation_diagnostic_backend]")
    print(
        "  local_bracket = "
        f"{pack['exact_trial2_sign_flip_local_bracket_available_now']}"
    )
    print(
        "  eta_star_position_inside_local_bracket = "
        f"{pack['eta_star_position_inside_local_bracket']:.12f}"
    )
    print(
        "  derivative_eta_vertex = "
        f"{pack['derivative_eta_vertex']:.12e}"
    )


if __name__ == "__main__":
    main()
