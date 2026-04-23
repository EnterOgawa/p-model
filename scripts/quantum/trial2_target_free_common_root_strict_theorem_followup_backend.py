#!/usr/bin/env python3
"""Audit whether the target-free common-root selector can be elevated to a theorem.

Purpose:
    The prior branch fixed one practical target-free selector on the retained
    scalar family by solving

        alpha_qstar(beta) = alpha_R8(beta)

    where

        alpha_qstar(beta) = F_beta(q_star(beta))^2 / (4 pi)
        alpha_R8(beta) = R8_exact(beta)

    This helper asks the next narrower question only:

        can the selector be upgraded from sampled numerical uniqueness to
        one strict analytic theorem?

    The audit therefore separates two layers cleanly:

    1. numerical transversality / monotonicity support near the common root,
    2. exact theorem surfaces that would be required for strict uniqueness.

Inputs:
    - scripts/quantum/trial2_interaction_total_over_harmonic_sq_beta_root_followup_backend.py
    - scripts/quantum/trial2_alpha_beta_curve_backend.py
    - scripts/quantum/trial2_interaction_total_over_harmonic_sq_exact_relation_backend.py

Outputs:
    - One in-memory audit pack consumed by `.5631-.5638` wrappers

Assumptions:
    - No new parameter is introduced
    - alpha_target remains an external comparator only
    - The common-root selector is already fixed numerically before this audit
"""

from __future__ import annotations

import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.quantum.trial2_alpha_beta_curve_backend import build_beta_family_row
from scripts.quantum.trial2_interaction_total_over_harmonic_sq_beta_root_followup_backend import (
    build_trial2_interaction_total_over_harmonic_sq_beta_root_followup_pack,
)
from scripts.quantum.trial2_interaction_total_over_harmonic_sq_exact_relation_backend import (
    build_exact_relation_row,
)


DERIVATIVE_H_VALUES = (1.0e-4, 5.0e-5, 1.0e-5, 5.0e-6, 1.0e-6)
DERIVATIVE_REL_SPREAD_TOL = 5.0e-3


# 関数: one beta で alpha_qstar(beta) を返す。
def alpha_qstar_value(beta: float) -> float:
    """Return the q_star-based direct-alpha readout on one localized beta row."""
    row = build_beta_family_row(float(beta))
    if row is None:
        raise SystemExit(f"[fail] alpha_qstar row is unavailable for beta={beta}")

    return float(row["alpha_at_q_star"])


# 関数: one beta で alpha_R8(beta) を返す。

def alpha_r8_value(beta: float) -> float:
    """Return the exact R8 direct-alpha readout on one localized beta row."""
    row = build_exact_relation_row(float(beta))
    return float(row["exact_relation_from_integrals"])


# 関数: common-root 近傍の local derivative pack を返す。

def build_local_derivative_pack(beta_common_root: float) -> dict:
    """Return local derivative evidence for the common-root selector."""
    beta_common_root = float(beta_common_root)
    derivative_rows: list[dict] = []
    alpha_qstar_derivatives: list[float] = []
    alpha_r8_derivatives: list[float] = []
    difference_derivatives: list[float] = []

    for h_value in DERIVATIVE_H_VALUES:
        h_value = float(h_value)
        alpha_qstar_plus = alpha_qstar_value(beta_common_root + h_value)
        alpha_qstar_minus = alpha_qstar_value(beta_common_root - h_value)
        alpha_r8_plus = alpha_r8_value(beta_common_root + h_value)
        alpha_r8_minus = alpha_r8_value(beta_common_root - h_value)
        alpha_qstar_derivative = float(
            (alpha_qstar_plus - alpha_qstar_minus) / (2.0 * h_value)
        )
        alpha_r8_derivative = float((alpha_r8_plus - alpha_r8_minus) / (2.0 * h_value))
        difference_derivative = float(alpha_qstar_derivative - alpha_r8_derivative)

        derivative_rows.append(
            {
                "h_value": h_value,
                "alpha_qstar_derivative": alpha_qstar_derivative,
                "alpha_r8_derivative": alpha_r8_derivative,
                "difference_derivative": difference_derivative,
            }
        )
        alpha_qstar_derivatives.append(alpha_qstar_derivative)
        alpha_r8_derivatives.append(alpha_r8_derivative)
        difference_derivatives.append(difference_derivative)

    difference_derivative_min = float(min(difference_derivatives))
    difference_derivative_max = float(max(difference_derivatives))
    difference_derivative_reference = float(abs(difference_derivatives[-1]))
    difference_derivative_rel_spread = float(
        (difference_derivative_max - difference_derivative_min)
        / max(difference_derivative_reference, 1.0e-30)
    )

    return {
        "derivative_rows": derivative_rows,
        "alpha_qstar_derivative_positive_now": bool(
            all(value > 0.0 for value in alpha_qstar_derivatives)
        ),
        "alpha_r8_derivative_negative_now": bool(
            all(value < 0.0 for value in alpha_r8_derivatives)
        ),
        "difference_derivative_positive_now": bool(
            all(value > 0.0 for value in difference_derivatives)
        ),
        "difference_derivative_min": difference_derivative_min,
        "difference_derivative_max": difference_derivative_max,
        "difference_derivative_rel_spread": difference_derivative_rel_spread,
        "difference_derivative_stable_now": bool(
            difference_derivative_rel_spread <= DERIVATIVE_REL_SPREAD_TOL
        ),
        "alpha_qstar_derivative_min": float(min(alpha_qstar_derivatives)),
        "alpha_qstar_derivative_max": float(max(alpha_qstar_derivatives)),
        "alpha_r8_derivative_min": float(min(alpha_r8_derivatives)),
        "alpha_r8_derivative_max": float(max(alpha_r8_derivatives)),
    }


# 関数: strict-theorem followup 全体を official pack に束ねる。

def build_trial2_target_free_common_root_strict_theorem_followup_pack() -> dict:
    """Return one strict-theorem followup pack for the common-root selector."""
    prior_pack = build_trial2_interaction_total_over_harmonic_sq_beta_root_followup_pack()
    beta_common_root = float(prior_pack["beta_common_root"])
    derivative_pack = build_local_derivative_pack(beta_common_root)

    target_free_beta_selector_available_now = bool(
        prior_pack["target_free_beta_selector_available_now"]
    )
    practical_direct_alpha_closeout_available_now = bool(
        prior_pack["practical_direct_alpha_closeout_available_now"]
    )
    local_transversality_support_available_now = bool(
        derivative_pack["alpha_qstar_derivative_positive_now"]
        and derivative_pack["alpha_r8_derivative_negative_now"]
        and derivative_pack["difference_derivative_positive_now"]
        and derivative_pack["difference_derivative_stable_now"]
    )

    exact_alpha_qstar_monotone_theorem_available_now = False
    exact_alpha_r8_monotone_theorem_available_now = False
    exact_common_root_uniqueness_theorem_available_now = False
    strict_target_free_theorem_closeout_available_now = False

    strict_theorem_negative_closeout_available_now = bool(
        target_free_beta_selector_available_now
        and practical_direct_alpha_closeout_available_now
        and local_transversality_support_available_now
        and not exact_common_root_uniqueness_theorem_available_now
    )
    conditional_hold_restored_primary_now = bool(
        strict_theorem_negative_closeout_available_now
    )
    no_unconditional_next_official_branch_now = bool(
        strict_theorem_negative_closeout_available_now
    )

    return {
        **prior_pack,
        **derivative_pack,
        "target_free_beta_selector_available_now": target_free_beta_selector_available_now,
        "practical_direct_alpha_closeout_available_now": (
            practical_direct_alpha_closeout_available_now
        ),
        "local_transversality_support_available_now": (
            local_transversality_support_available_now
        ),
        "exact_alpha_qstar_monotone_theorem_available_now": (
            exact_alpha_qstar_monotone_theorem_available_now
        ),
        "exact_alpha_r8_monotone_theorem_available_now": (
            exact_alpha_r8_monotone_theorem_available_now
        ),
        "exact_common_root_uniqueness_theorem_available_now": (
            exact_common_root_uniqueness_theorem_available_now
        ),
        "strict_target_free_theorem_closeout_available_now": (
            strict_target_free_theorem_closeout_available_now
        ),
        "strict_theorem_negative_closeout_available_now": (
            strict_theorem_negative_closeout_available_now
        ),
        "conditional_hold_restored_primary_now": (
            conditional_hold_restored_primary_now
        ),
        "no_unconditional_next_official_branch_now": (
            no_unconditional_next_official_branch_now
        ),
    }


# 関数: helper 単体実行時に compact summary を表示する。

def main() -> None:
    """Run the strict-theorem followup helper directly."""
    pack = build_trial2_target_free_common_root_strict_theorem_followup_pack()
    print("[trial2_target_free_common_root_strict_theorem_followup_backend]")
    print(f"  beta_common_root = {pack['beta_common_root']:.15f}")
    print(
        "  difference_derivative_range = "
        f"[{pack['difference_derivative_min']:.12f}, {pack['difference_derivative_max']:.12f}]"
    )
    print(
        f"  strict_theorem_negative_closeout = {pack['strict_theorem_negative_closeout_available_now']}"
    )


if __name__ == "__main__":
    main()
