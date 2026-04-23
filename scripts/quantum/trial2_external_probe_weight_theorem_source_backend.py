#!/usr/bin/env python3
"""Audit whether the deterministic external-probe weight has a theorem source.

Purpose:
    The current exact-goal pack already has one strong deterministic candidate
    weight

        eta_vertex = normalized leading nonzero-time selector weight.

    The remaining theorem question is narrower:

        does the current pack provide a frozen-action theorem selecting that
        exact normalization, or is `eta_vertex` still only a computation-level
        choice among nearby deterministic normalizations?

    This helper compares the retained nonzero-time normalization against the
    full-selector normalization. If the exact-goal readout changes materially
    under that normalization-set change, the theorem source is still missing.

Inputs:
    - scripts/quantum/trial2_4d_time_component_augmentation_backend.py
    - scripts/quantum/trial2_selector_4d_mixed_normalization_exactification_backend.py
    - scripts/quantum/trial2_4d_full_integral_external_probe_current_vertex_exactification_backend.py

Outputs:
    - One in-memory audit pack consumed by `.5875-.5878` wrappers

Assumptions:
    - No new free parameter is introduced
    - The exact goal `1/137` is used only as a comparator
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.quantum.trial2_4d_full_integral_external_probe_current_vertex_exactification_backend import (
    build_trial2_4d_full_integral_external_probe_current_vertex_exactification_pack,
)
from scripts.quantum.trial2_4d_time_component_augmentation_backend import (
    build_trial2_4d_time_component_augmentation_pack,
)
from scripts.quantum.trial2_selector_4d_mixed_normalization_exactification_backend import (
    build_trial2_selector_4d_mixed_normalization_exactification_pack,
)
from scripts.quantum.trial2_selector_4d_mixed_normalization_exactification_backend import (
    mixed_alpha,
)


ZERO_RESIDUAL_TOL = 1.0e-14


# 関数: full selector family での leading weight を返す。
def build_full_selector_leading_weight(selector_rows: list[dict]) -> float:
    """Return the leading selector weight normalized over the full family."""
    weight_sum = sum(float(row["polarization_weight"]) for row in selector_rows)
    leading_row = next(
        row
        for row in selector_rows
        if str(row["label"]) == "leading_nontrivial_time_component"
    )
    return float(float(leading_row["polarization_weight"]) / max(weight_sum, 1.0e-30))


# 関数: deterministic weight theorem-source pack を返す。

def build_trial2_external_probe_weight_theorem_source_pack() -> dict:
    """Return the theorem-source audit pack for the deterministic 4D weight."""
    mixed_pack = build_trial2_selector_4d_mixed_normalization_exactification_pack()
    vertex_pack = (
        build_trial2_4d_full_integral_external_probe_current_vertex_exactification_pack()
    )
    augmentation_pack = build_trial2_4d_time_component_augmentation_pack()

    alpha_3d = float(mixed_pack["alpha_3d_exact"])
    charge_factor = float(mixed_pack["canonical_charge_factor"])
    mass_factor = float(mixed_pack["canonical_mass_factor"])
    eta_star = float(mixed_pack["eta_exact_goal_interpolant"])
    eta_nonzero_time = float(vertex_pack["eta_vertex_weight_candidate"])
    eta_all_selectors = build_full_selector_leading_weight(augmentation_pack["selector_rows"])
    alpha_nonzero_time = float(vertex_pack["alpha_vertex_candidate"])
    alpha_all_selectors = float(
        mixed_alpha(alpha_3d, charge_factor, mass_factor, eta_all_selectors)
    )
    exact_goal_alpha = float(vertex_pack["alpha_goal_exact_one_over_137"])

    theorem_source_available = bool(
        abs(alpha_nonzero_time - exact_goal_alpha) <= ZERO_RESIDUAL_TOL
        and abs(eta_nonzero_time - eta_star) <= ZERO_RESIDUAL_TOL
    )
    normalization_set_sensitive = bool(abs(eta_nonzero_time - eta_all_selectors) > 1.0e-6)
    exact_goal_side_flips = bool(
        (alpha_nonzero_time - exact_goal_alpha) * (alpha_all_selectors - exact_goal_alpha) < 0.0
    )
    computation_source_available = True

    return {
        "alpha_goal_exact_one_over_137": exact_goal_alpha,
        "eta_exact_goal_interpolant": eta_star,
        "eta_nonzero_time_weight_candidate": eta_nonzero_time,
        "eta_all_selector_weight_candidate": eta_all_selectors,
        "eta_nonzero_time_minus_all_selector": float(
            eta_nonzero_time - eta_all_selectors
        ),
        "eta_nonzero_time_rel_gap_vs_all_selector": float(
            (eta_nonzero_time - eta_all_selectors)
            / max(abs(eta_nonzero_time), 1.0e-30)
        ),
        "alpha_nonzero_time_weight_candidate": alpha_nonzero_time,
        "alpha_nonzero_time_rel_error_vs_exact_goal": float(
            (alpha_nonzero_time - exact_goal_alpha) / exact_goal_alpha
        ),
        "alpha_all_selector_weight_candidate": alpha_all_selectors,
        "alpha_all_selector_rel_error_vs_exact_goal": float(
            (alpha_all_selectors - exact_goal_alpha) / exact_goal_alpha
        ),
        "exact_trial2_external_probe_weight_computation_source_available_now": (
            computation_source_available
        ),
        "exact_trial2_external_probe_weight_normalization_set_sensitive_now": (
            normalization_set_sensitive
        ),
        "exact_trial2_external_probe_weight_exact_goal_side_flips_now": (
            exact_goal_side_flips
        ),
        "exact_trial2_external_probe_weight_theorem_source_available_now": (
            theorem_source_available
        ),
    }


# 関数: backend 単体実行時に compact summary を表示する。

def main() -> None:
    """Run the deterministic external-probe weight theorem-source audit directly."""
    pack = build_trial2_external_probe_weight_theorem_source_pack()
    print("[trial2_external_probe_weight_theorem_source_backend]")
    print(
        "  eta_nonzero_time = "
        f"{pack['eta_nonzero_time_weight_candidate']:.15f}"
    )
    print(
        "  eta_all_selectors = "
        f"{pack['eta_all_selector_weight_candidate']:.15f}"
    )
    print(
        "  normalization_set_sensitive = "
        f"{pack['exact_trial2_external_probe_weight_normalization_set_sensitive_now']}"
    )
    print(
        "  theorem_source_available = "
        f"{pack['exact_trial2_external_probe_weight_theorem_source_available_now']}"
    )


if __name__ == "__main__":
    main()
