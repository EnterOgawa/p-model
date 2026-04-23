#!/usr/bin/env python3
"""Promote source-weighted pure-continuum support into one control-window theorem.

Purpose:
    Continue `.5759-.5766`, where the current pack already fixed

        1. the exact source-weighted comparison identity on the physical
           control window,
        2. one explicit omitted-tail nonreversal bound beyond a retained
           cutoff, and
        3. one honest pure-continuum support layer.

    The remaining narrower question is:

        can those two ingredients be bundled into one operator-level
        control-window continuum closure, without overclaiming a full global
        one-sign kernel theorem?

    This backend therefore does not replay the old global-kernel route.
    Instead it checks one stronger but still honest statement:

        for x in the physical control window, the full half-line
        source-weighted comparison margin remains strictly positive because
        the omitted dangerous tail is uniformly bounded below the retained
        finite-cutoff comparison margin.

Inputs:
    - scripts/quantum/trial2_beta_sensitivity_source_weighted_comparison_followup_backend.py
    - scripts/quantum/trial2_beta_sensitivity_halfline_green_kernel_followup_backend.py
    - scripts/quantum/trial2_beta_sensitivity_patched_tail_weighted_integral_followup_backend.py

Outputs:
    - One in-memory audit pack consumed by `.5767-.5774` wrappers

Assumptions:
    - The retained common-root selector remains fixed at beta_common_root
    - The admissible positive-decay patched tail remains the only allowed
      continuation beyond the retained half-line cutoff
    - No new parameter is introduced
    - This route closes one operator-level control-window continuum layer
      only; it does not yet claim a full global one-sign kernel theorem
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.quantum.trial2_beta_sensitivity_equation_backend import BETA_COMMON_ROOT
from scripts.quantum.trial2_beta_sensitivity_halfline_green_kernel_followup_backend import (
    CONTROL_WINDOW_X_MAX,
)
from scripts.quantum.trial2_beta_sensitivity_halfline_green_kernel_followup_backend import (
    CONTROL_WINDOW_X_MIN,
)
from scripts.quantum.trial2_beta_sensitivity_halfline_green_kernel_followup_backend import (
    build_halfline_operator_row,
)
from scripts.quantum.trial2_beta_sensitivity_halfline_green_kernel_followup_backend import (
    build_homogeneous_pair_row,
)
from scripts.quantum.trial2_beta_sensitivity_patched_tail_weighted_integral_followup_backend import (
    TAIL_MATCH_X,
)
from scripts.quantum.trial2_beta_sensitivity_patched_tail_weighted_integral_followup_backend import (
    build_patched_profile_row,
)
from scripts.quantum.trial2_beta_sensitivity_source_weighted_comparison_followup_backend import (
    build_source_weighted_comparison_row,
)


RETAINED_X_CUTOFF = 140.0
OPERATOR_LEVEL_X_MAX_VALUES = (80.0, 100.0, 140.0, 180.0, 220.0)


# 関数: one control-window coefficient row を返す。
def build_control_coefficient_row(x_max: float) -> dict:
    """Return the dangerous comparison coefficients at the requested cutoff."""
    operator_row = build_halfline_operator_row(float(x_max))
    homogeneous_row = build_homogeneous_pair_row(operator_row)
    grid = np.asarray(operator_row["grid"], dtype=float)
    control_mask = np.asarray(operator_row["control_mask"], dtype=bool)
    control_grid = grid[control_mask]
    control_coefficients = (
        np.asarray(homogeneous_row["left_w"], dtype=float)
        / np.asarray(homogeneous_row["wronskian_values"], dtype=float)
    )[control_mask]
    negative_coefficients = np.maximum(-control_coefficients, 0.0)
    positive_coefficients = np.maximum(control_coefficients, 0.0)
    dangerous_negative_index = int(np.argmax(negative_coefficients))
    positive_index = int(np.argmax(positive_coefficients))
    return {
        "retained_negative_control_coeff_max": float(
            negative_coefficients[dangerous_negative_index]
        ),
        "retained_negative_control_coeff_max_x": float(
            control_grid[dangerous_negative_index]
        ),
        "retained_positive_control_coeff_max": float(
            positive_coefficients[positive_index]
        ),
        "retained_positive_control_coeff_max_x": float(control_grid[positive_index]),
        "retained_negative_control_fraction": float(
            np.mean(control_coefficients < 0.0)
        ),
    }


# 関数: one patched-tail majorant row を返す。

def build_patched_tail_majorant_row(x_max: float) -> dict:
    """Return one explicit patched-tail majorant row at the requested cutoff."""
    beta_common_root = float(BETA_COMMON_ROOT)
    x_cutoff = float(x_max)
    x_match = float(TAIL_MATCH_X)
    patched_row = build_patched_profile_row(beta_common_root, x_cutoff)
    kappa = float(patched_row["kappa"])
    y_match = float(
        np.interp(
            x_match,
            np.asarray(patched_row["radius"], dtype=float),
            np.asarray(patched_row["profile"], dtype=float),
        )
    )
    amplitude = float(y_match * x_match)
    delta_cutoff = float(x_cutoff - x_match)
    tail_contraction_upper_bound = float(
        3.0 * amplitude / (kappa * kappa * x_cutoff) * math.exp(-kappa * delta_cutoff)
        + 3.0
        * amplitude
        * amplitude
        / (4.0 * kappa * kappa * x_cutoff * x_cutoff)
        * math.exp(-2.0 * kappa * delta_cutoff)
    )
    tail_resolvent_multiplier_upper_bound = float(
        1.0 / max(1.0 - tail_contraction_upper_bound, 1.0e-30)
    )
    source_tail_integral_upper_bound = float(
        beta_common_root * amplitude / kappa * math.exp(-kappa * delta_cutoff)
    )
    return {
        "beta_common_root": beta_common_root,
        "tail_match_x": x_match,
        "retained_x_cutoff": x_cutoff,
        "kappa": kappa,
        "tail_match_value": y_match,
        "tail_amplitude": amplitude,
        "tail_contraction_upper_bound": tail_contraction_upper_bound,
        "tail_resolvent_multiplier_upper_bound": (
            tail_resolvent_multiplier_upper_bound
        ),
        "source_tail_integral_upper_bound": source_tail_integral_upper_bound,
    }


# 関数: one operator-level control-window continuum row を返す。

def build_operator_level_continuum_row(x_max: float) -> dict:
    """Return one operator-level control-window continuum row."""
    retained_row = build_source_weighted_comparison_row(float(x_max))
    coefficient_row = build_control_coefficient_row(float(x_max))
    majorant_row = build_patched_tail_majorant_row(float(x_max))

    omitted_negative_tail_upper_bound = float(
        coefficient_row["retained_negative_control_coeff_max"]
        * majorant_row["tail_resolvent_multiplier_upper_bound"]
        * majorant_row["source_tail_integral_upper_bound"]
    )
    comparison_margin_lower_bound = float(
        retained_row["min_comparison_margin"] - omitted_negative_tail_upper_bound
    )
    omitted_negative_tail_over_retained_margin = float(
        omitted_negative_tail_upper_bound
        / max(retained_row["min_comparison_margin"], 1.0e-30)
    )

    return {
        "x_max": float(x_max),
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
        "retained_negative_control_coeff_max": float(
            coefficient_row["retained_negative_control_coeff_max"]
        ),
        "retained_negative_control_coeff_max_x": float(
            coefficient_row["retained_negative_control_coeff_max_x"]
        ),
        "retained_positive_control_coeff_max": float(
            coefficient_row["retained_positive_control_coeff_max"]
        ),
        "retained_positive_control_coeff_max_x": float(
            coefficient_row["retained_positive_control_coeff_max_x"]
        ),
        "retained_negative_control_fraction": float(
            coefficient_row["retained_negative_control_fraction"]
        ),
        "tail_match_x": float(majorant_row["tail_match_x"]),
        "tail_match_value": float(majorant_row["tail_match_value"]),
        "kappa": float(majorant_row["kappa"]),
        "tail_amplitude": float(majorant_row["tail_amplitude"]),
        "tail_contraction_upper_bound": float(
            majorant_row["tail_contraction_upper_bound"]
        ),
        "tail_resolvent_multiplier_upper_bound": float(
            majorant_row["tail_resolvent_multiplier_upper_bound"]
        ),
        "source_tail_integral_upper_bound": float(
            majorant_row["source_tail_integral_upper_bound"]
        ),
        "omitted_negative_tail_upper_bound": omitted_negative_tail_upper_bound,
        "comparison_margin_lower_bound": comparison_margin_lower_bound,
        "omitted_negative_tail_over_retained_margin": (
            omitted_negative_tail_over_retained_margin
        ),
        "operator_level_control_window_nonreversal_now": bool(
            comparison_margin_lower_bound > 0.0
        ),
    }


# 関数: source-weighted operator-level continuum followup の監査 pack を返す。

def build_trial2_beta_sensitivity_source_weighted_operator_level_continuum_followup_pack() -> dict:
    """Return one audit pack for the operator-level control-window route."""
    route_rows = [
        build_operator_level_continuum_row(x_max)
        for x_max in OPERATOR_LEVEL_X_MAX_VALUES
    ]
    retained_row = next(
        row
        for row in route_rows
        if abs(float(row["x_max"]) - float(RETAINED_X_CUTOFF)) < 1.0e-12
    )

    lower_bounds = np.asarray(
        [float(row["comparison_margin_lower_bound"]) for row in route_rows],
        dtype=float,
    )
    omitted_ratios = np.asarray(
        [float(row["omitted_negative_tail_over_retained_margin"]) for row in route_rows],
        dtype=float,
    )
    contractions = np.asarray(
        [float(row["tail_contraction_upper_bound"]) for row in route_rows],
        dtype=float,
    )

    source_weighted_operator_level_control_window_continuum_support_available_now = (
        bool(
            np.all(lower_bounds > 0.0)
            and np.all(omitted_ratios < 1.0)
            and np.all(contractions < 1.0)
        )
    )
    exact_trial2_source_weighted_operator_level_control_window_continuum_closure_available_now = bool(
        source_weighted_operator_level_control_window_continuum_support_available_now
    )
    exact_trial2_pure_analytic_global_one_sign_kernel_theorem_available_now = False
    updated_pack_trial2_source_weighted_operator_level_continuum_gate_required_now = bool(
        exact_trial2_source_weighted_operator_level_control_window_continuum_closure_available_now
        and not exact_trial2_pure_analytic_global_one_sign_kernel_theorem_available_now
    )

    return {
        "beta_common_root": float(BETA_COMMON_ROOT),
        "control_window_x_min": float(CONTROL_WINDOW_X_MIN),
        "control_window_x_max": float(CONTROL_WINDOW_X_MAX),
        "retained_x_cutoff": float(RETAINED_X_CUTOFF),
        "x_max_values": [float(value) for value in OPERATOR_LEVEL_X_MAX_VALUES],
        "route_rows": route_rows,
        "retained_min_comparison_margin": float(retained_row["retained_min_comparison_margin"]),
        "retained_min_comparison_margin_x": float(
            retained_row["retained_min_comparison_margin_x"]
        ),
        "retained_negative_control_coeff_max": float(
            retained_row["retained_negative_control_coeff_max"]
        ),
        "retained_negative_control_coeff_max_x": float(
            retained_row["retained_negative_control_coeff_max_x"]
        ),
        "retained_positive_control_coeff_max": float(
            retained_row["retained_positive_control_coeff_max"]
        ),
        "retained_positive_control_coeff_max_x": float(
            retained_row["retained_positive_control_coeff_max_x"]
        ),
        "retained_tail_contraction_upper_bound": float(
            retained_row["tail_contraction_upper_bound"]
        ),
        "retained_source_tail_integral_upper_bound": float(
            retained_row["source_tail_integral_upper_bound"]
        ),
        "retained_omitted_negative_tail_upper_bound": float(
            retained_row["omitted_negative_tail_upper_bound"]
        ),
        "retained_comparison_margin_lower_bound": float(
            retained_row["comparison_margin_lower_bound"]
        ),
        "retained_omitted_negative_tail_over_retained_margin": float(
            retained_row["omitted_negative_tail_over_retained_margin"]
        ),
        "family_comparison_margin_lower_bound_min": float(np.min(lower_bounds)),
        "family_comparison_margin_lower_bound_max": float(np.max(lower_bounds)),
        "family_omitted_negative_tail_ratio_min": float(np.min(omitted_ratios)),
        "family_omitted_negative_tail_ratio_max": float(np.max(omitted_ratios)),
        "family_tail_contraction_upper_bound_min": float(np.min(contractions)),
        "family_tail_contraction_upper_bound_max": float(np.max(contractions)),
        "source_weighted_operator_level_control_window_continuum_support_available_now": bool(
            source_weighted_operator_level_control_window_continuum_support_available_now
        ),
        "exact_trial2_source_weighted_operator_level_control_window_continuum_closure_available_now": bool(
            exact_trial2_source_weighted_operator_level_control_window_continuum_closure_available_now
        ),
        "exact_trial2_pure_analytic_global_one_sign_kernel_theorem_available_now": bool(
            exact_trial2_pure_analytic_global_one_sign_kernel_theorem_available_now
        ),
        "updated_pack_trial2_source_weighted_operator_level_continuum_gate_required_now": bool(
            updated_pack_trial2_source_weighted_operator_level_continuum_gate_required_now
        ),
    }


# 関数: backend 単体実行時に retained metrics を表示する。

def main() -> None:
    """Run the operator-level control-window continuum backend directly."""
    pack = (
        build_trial2_beta_sensitivity_source_weighted_operator_level_continuum_followup_pack()
    )
    print("[trial2-beta-source-weighted-operator-level-continuum-followup]")
    print(f"beta_common_root = {pack['beta_common_root']:.16f}")
    print(f"retained_x_cutoff = {pack['retained_x_cutoff']:.1f}")
    print(
        "retained_min_margin = "
        f"{pack['retained_min_comparison_margin']:.16f} "
        f"at x = {pack['retained_min_comparison_margin_x']:.16f}"
    )
    print(
        "retained_omitted_negative_tail_upper_bound = "
        f"{pack['retained_omitted_negative_tail_upper_bound']:.16f}"
    )
    print(
        "retained_comparison_margin_lower_bound = "
        f"{pack['retained_comparison_margin_lower_bound']:.16f}"
    )
    print(
        "family_comparison_margin_lower_bound_min = "
        f"{pack['family_comparison_margin_lower_bound_min']:.16f}"
    )
    print(
        "family_omitted_negative_tail_ratio_max = "
        f"{pack['family_omitted_negative_tail_ratio_max']:.16f}"
    )
    print(
        "source_weighted_operator_level_control_window_continuum_support_available_now = "
        f"{pack['source_weighted_operator_level_control_window_continuum_support_available_now']}"
    )
    print(
        "source_weighted_operator_level_continuum_gate_required_now = "
        f"{pack['updated_pack_trial2_source_weighted_operator_level_continuum_gate_required_now']}"
    )


if __name__ == "__main__":
    main()
