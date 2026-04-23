#!/usr/bin/env python3
"""Promote source-weighted comparison sign support toward pure-continuum form.

Purpose:
    Continue `.5751-.5758`, where the retained control-window negativity was
    rewritten exactly as one source-weighted comparison identity

        w_beta(x) = N_beta^(S)(x) - P_beta^(S)(x)

    with stable positive dominance on the physical window `[0.10, 19.90]`.

    The next honest question is narrower:

        can the retained finite-cutoff comparison support at `X = 140` be
        promoted one layer further by proving that the omitted tail on
        `[X, +∞)` cannot reverse the comparison margin?

    This backend does not claim the final full operator-level continuum
    theorem. It closes one support layer only:

    1. extract the dangerous negative comparison coefficient on the retained
       control window,
    2. write one explicit patched-tail contraction bound on `[X, +∞)`,
    3. bound the omitted negative comparison tail in closed form, and
    4. compare that bound against the already-fixed retained comparison margin.

Inputs:
    - scripts/quantum/trial2_beta_sensitivity_source_weighted_comparison_followup_backend.py
    - scripts/quantum/trial2_beta_sensitivity_halfline_green_kernel_followup_backend.py
    - scripts/quantum/trial2_beta_sensitivity_patched_tail_weighted_integral_followup_backend.py

Outputs:
    - One in-memory audit pack consumed by `.5759-.5766` wrappers

Assumptions:
    - The retained common-root selector remains fixed at beta_common_root
    - The admissible positive-decay patched tail remains the only allowed
      continuation beyond the retained half-line cutoff
    - No new parameter is introduced
    - This branch targets pure-continuum support for the source-weighted
      comparison route only; it does not yet claim the final full
      operator-level continuum theorem
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


# 関数: retained control-window coefficient row を返す。
def build_control_coefficient_row() -> dict:
    """Return the dangerous comparison coefficients on the retained window."""
    operator_row = build_halfline_operator_row(float(RETAINED_X_CUTOFF))
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


# 関数: patched-tail analytic tail-majorant row を返す。

def build_patched_tail_majorant_row() -> dict:
    """Return one explicit tail-majorant row at the retained cutoff."""
    beta_common_root = float(BETA_COMMON_ROOT)
    x_cutoff = float(RETAINED_X_CUTOFF)
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
        "tail_contraction_admissible_now": bool(tail_contraction_upper_bound < 1.0),
    }


# 関数: source-weighted comparison pure-continuum followup の監査 pack を返す。

def build_trial2_beta_sensitivity_source_weighted_comparison_pure_continuum_followup_pack() -> dict:
    """Return one audit pack for the source-weighted pure-continuum route."""
    retained_row = build_source_weighted_comparison_row(float(RETAINED_X_CUTOFF))
    coefficient_row = build_control_coefficient_row()
    majorant_row = build_patched_tail_majorant_row()

    omitted_negative_tail_upper_bound = float(
        coefficient_row["retained_negative_control_coeff_max"]
        * majorant_row["tail_resolvent_multiplier_upper_bound"]
        * majorant_row["source_tail_integral_upper_bound"]
    )
    omitted_positive_tail_upper_bound = float(
        coefficient_row["retained_positive_control_coeff_max"]
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
    source_weighted_comparison_pure_continuum_support_available_now = bool(
        retained_row["source_weighted_positive_dominance_now"]
        and retained_row["source_weighted_comparison_identity_available_now"]
        and retained_row["green_bvp_comparison_coherent_now"]
        and majorant_row["tail_contraction_admissible_now"]
        and comparison_margin_lower_bound > 0.0
    )
    exact_trial2_pure_analytic_operator_level_continuum_refinement_available_now = (
        False
    )
    updated_pack_trial2_source_weighted_comparison_pure_continuum_gate_required_now = bool(
        source_weighted_comparison_pure_continuum_support_available_now
        and not exact_trial2_pure_analytic_operator_level_continuum_refinement_available_now
    )

    return {
        "beta_common_root": float(BETA_COMMON_ROOT),
        "control_window_x_min": float(CONTROL_WINDOW_X_MIN),
        "control_window_x_max": float(CONTROL_WINDOW_X_MAX),
        "retained_x_cutoff": float(RETAINED_X_CUTOFF),
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
        "omitted_positive_tail_upper_bound": omitted_positive_tail_upper_bound,
        "comparison_margin_lower_bound": comparison_margin_lower_bound,
        "omitted_negative_tail_over_retained_margin": (
            omitted_negative_tail_over_retained_margin
        ),
        "source_weighted_comparison_pure_continuum_support_available_now": bool(
            source_weighted_comparison_pure_continuum_support_available_now
        ),
        "exact_trial2_pure_analytic_operator_level_continuum_refinement_available_now": (
            exact_trial2_pure_analytic_operator_level_continuum_refinement_available_now
        ),
        "updated_pack_trial2_source_weighted_comparison_pure_continuum_gate_required_now": (
            updated_pack_trial2_source_weighted_comparison_pure_continuum_gate_required_now
        ),
    }


# 関数: backend 単体実行時に retained metrics を表示する。

def main() -> None:
    """Run the source-weighted comparison pure-continuum backend directly."""
    pack = (
        build_trial2_beta_sensitivity_source_weighted_comparison_pure_continuum_followup_pack()
    )
    print("[trial2-beta-source-weighted-comparison-pure-continuum-followup]")
    print(f"beta_common_root = {pack['beta_common_root']:.16f}")
    print(f"retained_x_cutoff = {pack['retained_x_cutoff']:.1f}")
    print(
        "retained_negative_control_coeff_max = "
        f"{pack['retained_negative_control_coeff_max']:.16f} "
        f"at x = {pack['retained_negative_control_coeff_max_x']:.16f}"
    )
    print(
        "tail_contraction_upper_bound = "
        f"{pack['tail_contraction_upper_bound']:.16e}"
    )
    print(
        "source_tail_integral_upper_bound = "
        f"{pack['source_tail_integral_upper_bound']:.16e}"
    )
    print(
        "omitted_negative_tail_upper_bound = "
        f"{pack['omitted_negative_tail_upper_bound']:.16e}"
    )
    print(
        "comparison_margin_lower_bound = "
        f"{pack['comparison_margin_lower_bound']:.16f}"
    )
    print(
        "omitted_negative_tail_over_retained_margin = "
        f"{pack['omitted_negative_tail_over_retained_margin']:.16f}"
    )
    print(
        "source_weighted_comparison_pure_continuum_support_available_now = "
        f"{pack['source_weighted_comparison_pure_continuum_support_available_now']}"
    )
    print(
        "source_weighted_comparison_pure_continuum_gate_required_now = "
        f"{pack['updated_pack_trial2_source_weighted_comparison_pure_continuum_gate_required_now']}"
    )


if __name__ == "__main__":
    main()
