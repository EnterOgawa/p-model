#!/usr/bin/env python3
"""Audit source-weighted comparison after the half-line kernel split verdict.

Purpose:
    Continue the pure-analytic operator-level reopen route after `.5743-.5750`,
    where the patched half-line kernel route closed negatively at the kernel
    level while the actual source-weighted control-window solution remained
    negative.

    The honest narrower question is now:

        can the control-window negativity be rewritten as one exact
        source-weighted comparison identity, where the positive source-weighted
        part of the mixed-sign half-line kernel dominates the negative
        source-weighted part pointwise across the physical window?

    This backend therefore does not replay one-sign kernel attempts. It checks:

    1. the exact positive/negative source-weighted kernel decomposition,
    2. the pointwise dominance of the positive weighted contribution on the
       retained control window,
    3. the exact identity against the retained Green-convolution solution, and
    4. the stability of the resulting comparison margins across the retained
       truncation family `X in {80, 100, 140}`.

Inputs:
    - scripts/quantum/trial2_beta_sensitivity_halfline_green_kernel_followup_backend.py

Outputs:
    - One in-memory audit pack consumed by `.5751-.5758` wrappers

Assumptions:
    - The admissible patched half-line route remains the only allowed reopen
      route inside the current pack
    - No new parameter is introduced
    - This branch targets source-weighted comparison only; it does not yet
      claim full pure-continuum operator-level closure on `[0, +inf)`
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.quantum.trial2_beta_sensitivity_halfline_green_kernel_followup_backend import (
    BETA_COMMON_ROOT,
)
from scripts.quantum.trial2_beta_sensitivity_halfline_green_kernel_followup_backend import (
    CONTROL_WINDOW_X_MAX,
)
from scripts.quantum.trial2_beta_sensitivity_halfline_green_kernel_followup_backend import (
    CONTROL_WINDOW_X_MIN,
)
from scripts.quantum.trial2_beta_sensitivity_halfline_green_kernel_followup_backend import (
    HALFLINE_X_MAX_VALUES,
)
from scripts.quantum.trial2_beta_sensitivity_halfline_green_kernel_followup_backend import (
    build_green_convolution_row,
)
from scripts.quantum.trial2_beta_sensitivity_halfline_green_kernel_followup_backend import (
    build_halfline_bvp_row,
)
from scripts.quantum.trial2_beta_sensitivity_halfline_green_kernel_followup_backend import (
    build_halfline_operator_row,
)
from scripts.quantum.trial2_beta_sensitivity_halfline_green_kernel_followup_backend import (
    build_homogeneous_pair_row,
)


COMPARISON_IDENTITY_REL_LINF_TOL = 1.0e-6
COMPARISON_RATIO_REL_SPREAD_TOL = 5.0e-3
COMPARISON_RELGAP_REL_SPREAD_TOL = 1.0e-2
GREEN_BVP_REL_LINF_TOL = 5.0e-2


# 関数: one control-window source-weighted comparison row を返す。
def build_source_weighted_comparison_row(x_max: float) -> dict:
    """Return one exact source-weighted comparison row on the retained window."""
    operator_row = build_halfline_operator_row(float(x_max))
    homogeneous_row = build_homogeneous_pair_row(operator_row)
    green_row = build_green_convolution_row(operator_row, homogeneous_row)
    bvp_row = build_halfline_bvp_row(operator_row)

    grid = np.asarray(operator_row["grid"], dtype=float)
    source = np.asarray(operator_row["source"], dtype=float)
    control_mask = np.asarray(operator_row["control_mask"], dtype=bool)
    control_indices = np.flatnonzero(control_mask)
    left_w = np.asarray(homogeneous_row["left_w"], dtype=float)
    right_w = np.asarray(homogeneous_row["right_w"], dtype=float)
    wronskian = np.asarray(homogeneous_row["wronskian_values"], dtype=float)
    green_values = -np.asarray(green_row["solution_values"], dtype=float)[control_mask]
    bvp_values = -np.asarray(bvp_row["solution_values"], dtype=float)[control_mask]

    positive_contributions = []
    negative_contributions = []
    comparison_values = []
    comparison_ratios = []
    comparison_relative_gaps = []
    control_grid = grid[control_mask]

    for control_index in control_indices:
        x_value = float(grid[control_index])
        kernel = np.empty_like(grid)
        left_mask = grid <= x_value
        denominator = float(wronskian[control_index])
        kernel[left_mask] = (
            left_w[left_mask] * right_w[control_index] / denominator
        )
        kernel[~left_mask] = (
            left_w[control_index] * right_w[~left_mask] / denominator
        )
        positive_weight = np.asarray(np.clip(kernel, 0.0, None) * source, dtype=float)
        negative_weight = np.asarray(np.clip(-kernel, 0.0, None) * source, dtype=float)
        positive_contribution = float(np.trapezoid(positive_weight, grid))
        negative_contribution = float(np.trapezoid(negative_weight, grid))
        comparison_value = float(positive_contribution - negative_contribution)
        comparison_ratio = float(
            positive_contribution / max(negative_contribution, 1.0e-30)
        )
        comparison_relative_gap = float(
            comparison_value
            / max(positive_contribution + negative_contribution, 1.0e-30)
        )
        positive_contributions.append(positive_contribution)
        negative_contributions.append(negative_contribution)
        comparison_values.append(comparison_value)
        comparison_ratios.append(comparison_ratio)
        comparison_relative_gaps.append(comparison_relative_gap)

    positive_contributions_array = np.asarray(positive_contributions, dtype=float)
    negative_contributions_array = np.asarray(negative_contributions, dtype=float)
    comparison_values_array = np.asarray(comparison_values, dtype=float)
    comparison_ratios_array = np.asarray(comparison_ratios, dtype=float)
    comparison_relative_gaps_array = np.asarray(comparison_relative_gaps, dtype=float)

    identity_abs_error = np.abs(comparison_values_array - green_values)
    identity_rel_error = identity_abs_error / np.maximum(np.abs(green_values), 1.0e-30)
    bvp_rel_error = np.abs(comparison_values_array - bvp_values) / np.maximum(
        np.abs(bvp_values),
        1.0e-30,
    )

    min_ratio_index = int(np.argmin(comparison_ratios_array))
    min_relgap_index = int(np.argmin(comparison_relative_gaps_array))
    min_margin_index = int(np.argmin(comparison_values_array))
    identity_rel_index = int(np.argmax(identity_rel_error))
    bvp_rel_index = int(np.argmax(bvp_rel_error))

    return {
        "x_max": float(x_max),
        "control_point_count": int(control_grid.size),
        "min_comparison_ratio": float(comparison_ratios_array[min_ratio_index]),
        "min_comparison_ratio_x": float(control_grid[min_ratio_index]),
        "min_comparison_relative_gap": float(
            comparison_relative_gaps_array[min_relgap_index]
        ),
        "min_comparison_relative_gap_x": float(control_grid[min_relgap_index]),
        "min_comparison_margin": float(comparison_values_array[min_margin_index]),
        "min_comparison_margin_x": float(control_grid[min_margin_index]),
        "max_identity_abs_error": float(identity_abs_error[identity_rel_index]),
        "max_identity_rel_error": float(identity_rel_error[identity_rel_index]),
        "max_identity_rel_error_x": float(control_grid[identity_rel_index]),
        "max_bvp_rel_error": float(bvp_rel_error[bvp_rel_index]),
        "max_bvp_rel_error_x": float(control_grid[bvp_rel_index]),
        "green_bvp_control_rel_linf": float(
            np.max(np.abs(green_values - bvp_values))
            / max(float(np.max(np.abs(bvp_values))), 1.0e-30)
        ),
        "source_weighted_positive_dominance_now": bool(
            np.all(positive_contributions_array > negative_contributions_array)
            and np.all(comparison_values_array > 0.0)
            and np.all(comparison_ratios_array > 1.0)
        ),
        "source_weighted_comparison_identity_available_now": bool(
            float(np.max(identity_rel_error)) <= COMPARISON_IDENTITY_REL_LINF_TOL
        ),
        "green_bvp_comparison_coherent_now": bool(
            float(
                np.max(np.abs(green_values - bvp_values))
                / max(float(np.max(np.abs(bvp_values))), 1.0e-30)
            )
            <= GREEN_BVP_REL_LINF_TOL
        ),
    }


# 関数: source-weighted comparison followup の監査 pack を返す。
def build_trial2_beta_sensitivity_source_weighted_comparison_followup_pack() -> dict:
    """Return one audit pack for the source-weighted comparison route."""
    route_rows = [
        build_source_weighted_comparison_row(x_max)
        for x_max in HALFLINE_X_MAX_VALUES
    ]
    retained_row = route_rows[-1]

    min_ratios = np.asarray(
        [float(row["min_comparison_ratio"]) for row in route_rows],
        dtype=float,
    )
    min_relative_gaps = np.asarray(
        [float(row["min_comparison_relative_gap"]) for row in route_rows],
        dtype=float,
    )
    ratio_rel_spread = float(
        (float(np.max(min_ratios)) - float(np.min(min_ratios)))
        / max(abs(float(np.mean(min_ratios))), 1.0e-30)
    )
    relative_gap_rel_spread = float(
        (float(np.max(min_relative_gaps)) - float(np.min(min_relative_gaps)))
        / max(abs(float(np.mean(min_relative_gaps))), 1.0e-30)
    )

    source_weighted_comparison_surface_available_now = True
    source_weighted_positive_dominance_control_window_now = bool(
        all(row["source_weighted_positive_dominance_now"] for row in route_rows)
    )
    source_weighted_comparison_identity_available_now = bool(
        all(row["source_weighted_comparison_identity_available_now"] for row in route_rows)
    )
    source_weighted_comparison_green_bvp_coherent_now = bool(
        all(row["green_bvp_comparison_coherent_now"] for row in route_rows)
    )
    source_weighted_comparison_stable_now = bool(
        ratio_rel_spread <= COMPARISON_RATIO_REL_SPREAD_TOL
        and relative_gap_rel_spread <= COMPARISON_RELGAP_REL_SPREAD_TOL
    )
    exact_trial2_beta_sensitivity_source_weighted_comparison_support_available_now = bool(
        source_weighted_comparison_surface_available_now
        and source_weighted_positive_dominance_control_window_now
        and source_weighted_comparison_identity_available_now
        and source_weighted_comparison_green_bvp_coherent_now
        and source_weighted_comparison_stable_now
    )
    exact_trial2_pure_analytic_operator_level_continuum_refinement_available_now = False
    updated_pack_trial2_source_weighted_comparison_pure_continuum_followup_required_now = bool(
        exact_trial2_beta_sensitivity_source_weighted_comparison_support_available_now
        and not exact_trial2_pure_analytic_operator_level_continuum_refinement_available_now
    )

    return {
        "beta_common_root": float(BETA_COMMON_ROOT),
        "control_window_x_min": float(CONTROL_WINDOW_X_MIN),
        "control_window_x_max": float(CONTROL_WINDOW_X_MAX),
        "route_rows": route_rows,
        "retained_x_max": float(retained_row["x_max"]),
        "retained_min_comparison_ratio": float(retained_row["min_comparison_ratio"]),
        "retained_min_comparison_ratio_x": float(
            retained_row["min_comparison_ratio_x"]
        ),
        "retained_min_comparison_relative_gap": float(
            retained_row["min_comparison_relative_gap"]
        ),
        "retained_min_comparison_relative_gap_x": float(
            retained_row["min_comparison_relative_gap_x"]
        ),
        "retained_min_comparison_margin": float(retained_row["min_comparison_margin"]),
        "retained_min_comparison_margin_x": float(
            retained_row["min_comparison_margin_x"]
        ),
        "retained_max_identity_rel_error": float(
            retained_row["max_identity_rel_error"]
        ),
        "retained_max_identity_rel_error_x": float(
            retained_row["max_identity_rel_error_x"]
        ),
        "retained_green_bvp_control_rel_linf": float(
            retained_row["green_bvp_control_rel_linf"]
        ),
        "comparison_ratio_rel_spread": ratio_rel_spread,
        "comparison_relative_gap_rel_spread": relative_gap_rel_spread,
        "source_weighted_comparison_surface_available_now": (
            source_weighted_comparison_surface_available_now
        ),
        "exact_trial2_beta_sensitivity_source_weighted_positive_dominance_control_window_now": (
            source_weighted_positive_dominance_control_window_now
        ),
        "exact_trial2_beta_sensitivity_source_weighted_comparison_identity_available_now": (
            source_weighted_comparison_identity_available_now
        ),
        "exact_trial2_beta_sensitivity_source_weighted_comparison_green_bvp_coherent_now": (
            source_weighted_comparison_green_bvp_coherent_now
        ),
        "exact_trial2_beta_sensitivity_source_weighted_comparison_stable_now": (
            source_weighted_comparison_stable_now
        ),
        "exact_trial2_beta_sensitivity_source_weighted_comparison_support_available_now": (
            exact_trial2_beta_sensitivity_source_weighted_comparison_support_available_now
        ),
        "exact_trial2_pure_analytic_operator_level_continuum_refinement_available_now": (
            exact_trial2_pure_analytic_operator_level_continuum_refinement_available_now
        ),
        "updated_pack_trial2_source_weighted_comparison_pure_continuum_followup_required_now": (
            updated_pack_trial2_source_weighted_comparison_pure_continuum_followup_required_now
        ),
    }


# 関数: backend 単体実行時に retained metrics を表示する。
def main() -> None:
    """Run the source-weighted comparison backend directly."""
    pack = build_trial2_beta_sensitivity_source_weighted_comparison_followup_pack()
    print("[trial2-beta-source-weighted-comparison-followup]")
    print(f"beta_common_root = {pack['beta_common_root']:.16f}")
    print(f"retained_x_max = {pack['retained_x_max']:.1f}")
    print(
        "retained_min_ratio = "
        f"{pack['retained_min_comparison_ratio']:.16f} "
        f"at x = {pack['retained_min_comparison_ratio_x']:.16f}"
    )
    print(
        "retained_min_relative_gap = "
        f"{pack['retained_min_comparison_relative_gap']:.16f} "
        f"at x = {pack['retained_min_comparison_relative_gap_x']:.16f}"
    )
    print(
        "retained_min_margin = "
        f"{pack['retained_min_comparison_margin']:.16f} "
        f"at x = {pack['retained_min_comparison_margin_x']:.16f}"
    )
    print(
        "retained_max_identity_rel_error = "
        f"{pack['retained_max_identity_rel_error']:.16e} "
        f"at x = {pack['retained_max_identity_rel_error_x']:.16f}"
    )
    print(
        "comparison_ratio_rel_spread = "
        f"{pack['comparison_ratio_rel_spread']:.16f}"
    )
    print(
        "comparison_relative_gap_rel_spread = "
        f"{pack['comparison_relative_gap_rel_spread']:.16f}"
    )
    print(
        "source_weighted_comparison_support_available_now = "
        f"{pack['exact_trial2_beta_sensitivity_source_weighted_comparison_support_available_now']}"
    )
    print(
        "source_weighted_comparison_pure_continuum_followup_required_now = "
        f"{pack['updated_pack_trial2_source_weighted_comparison_pure_continuum_followup_required_now']}"
    )


if __name__ == "__main__":
    main()
