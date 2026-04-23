#!/usr/bin/env python3
"""Audit the exact symbolic selector behind the Trial-2 common root.

Purpose:
    The retained direct-alpha infrastructure already fixes one target-free
    common-root selector numerically,

        alpha_qstar(beta) = alpha_R8(beta),

    with

        alpha_qstar(beta) = F_beta(q_star(beta))^2 / (4 pi)
        alpha_R8(beta) = R8_exact(beta).

    The next honest blocker for the exact closed-form roadmap is not selector
    existence but selector exactification. This helper rewrites the equality
    into one exact symbolic equation by eliminating the common I2(beta)^2
    denominator:

        9 (1 + beta^2)^2 J(beta)^2 = pi N(beta),

    where J(beta) = I2(beta) F_beta(q_star(beta)) and
    N(beta) = exact_cubic_numerator(beta) * exact_total_numerator(beta).

Inputs:
    - scripts/quantum/trial2_alpha_beta_curve_backend.py
    - scripts/quantum/trial2_interaction_total_over_harmonic_sq_exact_relation_backend.py
    - scripts/quantum/trial2_beta_sensitivity_equation_backend.py

Outputs:
    - One in-memory audit pack consumed by `.5783-.5790` wrappers

Assumptions:
    - No new parameter is introduced
    - alpha_target is used only as an external audit comparator
    - The goal is selector exactification, not yet exact value extraction
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.quantum.scalar_proxy_alpha_q_curve_backend import ALPHA_TARGET
from scripts.quantum.trial2_alpha_beta_curve_backend import build_beta_family_row
from scripts.quantum.trial2_beta_sensitivity_equation_backend import BETA_COMMON_ROOT
from scripts.quantum.trial2_interaction_total_over_harmonic_sq_exact_relation_backend import (
    build_exact_relation_row,
)


PRIOR_RETAINED_BETA = 0.9982557379261291
PRIOR_ALPHA_BETA_ROOT = 0.9982996989044647
PRIOR_R8_BETA_ROOT = 0.9983085664490956
LOCAL_BETA_OFFSETS = (-1.0e-4, 0.0, 1.0e-4)
SELECTOR_REL_TOL = 1.0e-12


# 関数: one beta に対する symbolic-selector row を返す。
def build_symbolic_selector_row(beta: float) -> dict:
    """Return one exactified symbolic-selector row for one localized beta."""
    beta = float(beta)
    alpha_row = build_beta_family_row(beta)
    if alpha_row is None:
        raise SystemExit(f"[fail] localized alpha(beta) row is unavailable for beta={beta}")

    exact_row = build_exact_relation_row(beta)
    one_plus_beta_sq = float(1.0 + beta * beta)
    i2 = float(exact_row["i2"])
    form_factor_at_q_star = float(alpha_row["F_at_q_star"])
    j_beta = float(i2 * form_factor_at_q_star)
    exact_numerator = float(
        float(exact_row["exact_cubic_numerator"]) * float(exact_row["exact_total_numerator"])
    )
    selector_lhs = float(9.0 * one_plus_beta_sq * one_plus_beta_sq * j_beta * j_beta)
    selector_rhs = float(math.pi * exact_numerator)
    selector_residual = float(selector_lhs - selector_rhs)
    positive_factor = float(
        36.0 * math.pi * one_plus_beta_sq * one_plus_beta_sq * i2 * i2
    )
    weighted_eom_difference = float(
        float(alpha_row["alpha_at_q_star"]) - float(exact_row["exact_relation_from_weighted_eom"])
    )
    integral_difference = float(
        float(alpha_row["alpha_at_q_star"]) - float(exact_row["exact_relation_from_integrals"])
    )
    selector_from_weighted_eom_difference = float(positive_factor * weighted_eom_difference)
    selector_from_integral_difference = float(positive_factor * integral_difference)

    return {
        "beta": beta,
        "q_star_over_m0": float(alpha_row["q_star_over_m0"]),
        "alpha_qstar": float(alpha_row["alpha_at_q_star"]),
        "alpha_r8_exact": float(exact_row["exact_relation_from_integrals"]),
        "i2": i2,
        "j_beta": j_beta,
        "one_plus_beta_sq": one_plus_beta_sq,
        "exact_cubic_numerator": float(exact_row["exact_cubic_numerator"]),
        "exact_total_numerator": float(exact_row["exact_total_numerator"]),
        "exact_numerator": exact_numerator,
        "selector_lhs": selector_lhs,
        "selector_rhs": selector_rhs,
        "selector_residual": selector_residual,
        "selector_residual_abs": float(abs(selector_residual)),
        "selector_residual_rel": float(
            selector_residual / max(abs(selector_rhs), 1.0e-30)
        ),
        "positive_factor": positive_factor,
        "weighted_eom_difference": weighted_eom_difference,
        "integral_difference": integral_difference,
        "selector_from_weighted_eom_difference": selector_from_weighted_eom_difference,
        "selector_from_integral_difference": selector_from_integral_difference,
        "selector_weighted_eom_consistency_residual": float(
            selector_residual - selector_from_weighted_eom_difference
        ),
        "selector_integral_consistency_residual": float(
            selector_residual - selector_from_integral_difference
        ),
    }


# 関数: common-root 近傍の local symbolic scan を返す。

def build_local_symbolic_scan(beta_center: float) -> list[dict]:
    """Return one compact local symbolic scan around the retained common root."""
    rows: list[dict] = []
    for offset in LOCAL_BETA_OFFSETS:
        rows.append(build_symbolic_selector_row(float(beta_center + offset)))

    return rows


# 関数: symbolic selector residual の local root を返す。

def find_symbolic_selector_root(beta_center: float) -> dict:
    """Return one exact symbolic root pack around the retained common root."""
    local_rows = build_local_symbolic_scan(beta_center)
    left_row = local_rows[0]
    right_row = local_rows[-1]
    left_beta = float(left_row["beta"])
    right_beta = float(right_row["beta"])
    left_value = float(left_row["selector_residual"])
    right_value = float(right_row["selector_residual"])
    root_available_now = bool(left_value < 0.0 and right_value > 0.0)
    if not root_available_now:
        return {
            "local_rows": local_rows,
            "symbolic_root_available_now": False,
            "beta_symbolic_root": math.nan,
            "beta_symbolic_root_rel_shift_vs_prior_common_root": math.nan,
            "symbolic_root_row": None,
        }

    beta_symbolic_root = float(
        brentq(
            lambda beta: float(build_symbolic_selector_row(float(beta))["selector_residual"]),
            left_beta,
            right_beta,
        )
    )
    symbolic_root_row = build_symbolic_selector_row(beta_symbolic_root)
    return {
        "local_rows": local_rows,
        "symbolic_root_available_now": True,
        "beta_symbolic_root": beta_symbolic_root,
        "beta_symbolic_root_rel_shift_vs_prior_common_root": float(
            (beta_symbolic_root - beta_center) / beta_center
        ),
        "symbolic_root_row": symbolic_root_row,
    }


# 関数: symbolic common-root exactification の監査 pack 全体を返す。

def build_trial2_symbolic_common_root_exactification_pack() -> dict:
    """Return one audit pack for the exact symbolic common-root selector."""
    beta_common_root = float(BETA_COMMON_ROOT)
    retained_row = build_symbolic_selector_row(PRIOR_RETAINED_BETA)
    prior_alpha_row = build_symbolic_selector_row(PRIOR_ALPHA_BETA_ROOT)
    prior_r8_row = build_symbolic_selector_row(PRIOR_R8_BETA_ROOT)
    common_row = build_symbolic_selector_row(beta_common_root)
    root_pack = find_symbolic_selector_root(beta_common_root)
    local_rows = root_pack["local_rows"]

    low_row = local_rows[0]
    high_row = local_rows[-1]
    low_negative = bool(float(low_row["selector_residual"]) < 0.0)
    high_positive = bool(float(high_row["selector_residual"]) > 0.0)
    denominator_cancellation_available_now = bool(
        float(common_row["positive_factor"]) > 0.0
        and abs(float(common_row["selector_weighted_eom_consistency_residual"]))
        <= SELECTOR_REL_TOL * max(abs(float(common_row["selector_lhs"])), 1.0)
    )
    exact_symbolic_common_root_selector_available_now = bool(
        denominator_cancellation_available_now
        and low_negative
        and high_positive
        and bool(root_pack["symbolic_root_available_now"])
    )
    exact_symbolic_common_root_closed_form_value_available_now = False
    invariant_reduction_refresh_required_now = bool(
        exact_symbolic_common_root_selector_available_now
        and not exact_symbolic_common_root_closed_form_value_available_now
    )

    return {
        "alpha_target": float(ALPHA_TARGET),
        "prior_retained_beta": float(PRIOR_RETAINED_BETA),
        "prior_alpha_beta_root": float(PRIOR_ALPHA_BETA_ROOT),
        "prior_r8_beta_root": float(PRIOR_R8_BETA_ROOT),
        "beta_common_root": beta_common_root,
        "retained_row": retained_row,
        "prior_alpha_row": prior_alpha_row,
        "prior_r8_row": prior_r8_row,
        "common_row": common_row,
        **root_pack,
        "local_rows": local_rows,
        "local_low_residual_negative_now": low_negative,
        "local_high_residual_positive_now": high_positive,
        "exact_symbolic_common_root_denominator_cancellation_available_now": (
            denominator_cancellation_available_now
        ),
        "exact_symbolic_common_root_selector_available_now": (
            exact_symbolic_common_root_selector_available_now
        ),
        "exact_symbolic_common_root_closed_form_value_available_now": (
            exact_symbolic_common_root_closed_form_value_available_now
        ),
        "updated_pack_trial2_invariant_reduction_refresh_required_now": (
            invariant_reduction_refresh_required_now
        ),
    }


# 関数: backend 単体実行時に compact summary を表示する。

def main() -> None:
    """Run the symbolic common-root exactification audit directly."""
    pack = build_trial2_symbolic_common_root_exactification_pack()
    common_row = pack["common_row"]
    print("[trial2_symbolic_common_root_exactification_backend]")
    print(
        "  exact_symbolic_common_root_selector = "
        f"{pack['exact_symbolic_common_root_selector_available_now']}"
    )
    print(f"  beta_common_root = {pack['beta_common_root']:.15f}")
    print(f"  selector_residual_abs = {common_row['selector_residual_abs']:.6e}")


if __name__ == "__main__":
    main()
