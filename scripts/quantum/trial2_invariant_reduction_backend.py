#!/usr/bin/env python3
"""Audit finite invariant reduction for the Trial-2 exact selector.

Purpose:
    After `.5783-.5790`, the common-root selector is already exactified into
    one symbolic equation. The next honest blocker is whether the remaining
    profile functionals can be reduced to a finite beta-native invariant
    algebra rather than left as raw integrals.

    This helper performs the minimal exact reduction:

        f(beta) = J(beta) / I2(beta)
        g(beta) = Ig(beta) / I2(beta)
        q(beta) = I4(beta) / I2(beta)
        b(beta) = B(beta) / I2(beta)

    so that the selector becomes

        9 (1 + beta^2)^2 f(beta)^2
            = pi [4(g + eps - b) - q] [2(5 + beta^2) + 10 g - q - 4 b].

Inputs:
    - scripts/quantum/trial2_symbolic_common_root_exactification_backend.py
    - scripts/quantum/trial2_interaction_total_over_harmonic_sq_exact_relation_backend.py

Outputs:
    - One in-memory audit pack consumed by `.5791-.5794` wrappers

Assumptions:
    - No new parameter is introduced
    - alpha_target is used only as an audit comparator
    - The task is finite invariant reduction, not yet zero-residual extraction
"""

from __future__ import annotations

import math
import sys
from functools import lru_cache
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.quantum.scalar_proxy_alpha_q_curve_backend import ALPHA_TARGET
from scripts.quantum.trial2_interaction_total_over_harmonic_sq_exact_relation_backend import (
    build_exact_relation_row,
)
from scripts.quantum.trial2_symbolic_common_root_exactification_backend import (
    build_symbolic_selector_row,
)
from scripts.quantum.trial2_symbolic_common_root_exactification_backend import (
    build_trial2_symbolic_common_root_exactification_pack,
)


REL_TOL = 1.0e-10
RETAINED_REL_CONTINUITY_TOL = 2.0e-6


# 関数: 1つの beta に対する reduced-invariant row を返す。
@lru_cache(maxsize=None)
def build_invariant_reduction_row(beta: float) -> dict:
    """Return one reduced-invariant row for one selector beta."""
    beta = float(beta)
    symbolic_row = build_symbolic_selector_row(beta)
    exact_row = build_exact_relation_row(beta)
    i2 = float(exact_row["i2"])
    if not i2 > 0.0:
        raise SystemExit(f"[fail] invariant reduction requires I2(beta) > 0, got {i2}")

    epsilon_beta = float(exact_row["epsilon_beta"])
    f_beta = float(symbolic_row["j_beta"] / i2)
    g_beta = float(exact_row["ig"] / i2)
    q_beta = float(exact_row["i4"] / i2)
    b_beta = float(exact_row["boundary_weighted_eom"] / i2)

    reduced_cubic_factor = float(4.0 * (g_beta + epsilon_beta - b_beta) - q_beta)
    reduced_total_factor = float(
        2.0 * (5.0 + beta * beta) + 10.0 * g_beta - q_beta - 4.0 * b_beta
    )
    one_plus_beta_sq = float(1.0 + beta * beta)
    selector_reduced_lhs = float(9.0 * one_plus_beta_sq * one_plus_beta_sq * f_beta * f_beta)
    selector_reduced_rhs = float(math.pi * reduced_cubic_factor * reduced_total_factor)
    selector_reduced_residual = float(selector_reduced_lhs - selector_reduced_rhs)

    alpha_from_form_factor = float(symbolic_row["alpha_qstar"])
    alpha_from_exact_r8 = float(exact_row["exact_relation_from_integrals"])
    alpha_from_reduced_invariants = float(
        (reduced_cubic_factor * reduced_total_factor)
        / (36.0 * one_plus_beta_sq * one_plus_beta_sq)
    )

    return {
        "beta": beta,
        "epsilon_beta": epsilon_beta,
        "i2": i2,
        "f_beta": f_beta,
        "g_beta": g_beta,
        "q_beta": q_beta,
        "b_beta": b_beta,
        "reduced_cubic_factor": reduced_cubic_factor,
        "reduced_total_factor": reduced_total_factor,
        "selector_reduced_lhs": selector_reduced_lhs,
        "selector_reduced_rhs": selector_reduced_rhs,
        "selector_reduced_residual": selector_reduced_residual,
        "selector_reduced_residual_abs": float(abs(selector_reduced_residual)),
        "selector_reduced_residual_rel": float(
            selector_reduced_residual / max(abs(selector_reduced_rhs), 1.0e-30)
        ),
        "alpha_from_form_factor": alpha_from_form_factor,
        "alpha_from_exact_r8": alpha_from_exact_r8,
        "alpha_from_reduced_invariants": alpha_from_reduced_invariants,
        "alpha_reduced_minus_form_factor": float(
            alpha_from_reduced_invariants - alpha_from_form_factor
        ),
        "alpha_reduced_minus_exact_r8": float(
            alpha_from_reduced_invariants - alpha_from_exact_r8
        ),
        "alpha_reduced_rel_error_vs_target": float(
            (alpha_from_reduced_invariants - ALPHA_TARGET) / ALPHA_TARGET
        ),
    }


# 関数: invariant-reduction audit 全体を束ねる。

def build_trial2_invariant_reduction_pack() -> dict:
    """Return one audit pack for the finite invariant reduction step."""
    symbolic_pack = build_trial2_symbolic_common_root_exactification_pack()
    beta_common_root = float(symbolic_pack["beta_common_root"])
    beta_symbolic_root = float(symbolic_pack["beta_symbolic_root"])
    retained_row = build_invariant_reduction_row(beta_common_root)
    symbolic_row = build_invariant_reduction_row(beta_symbolic_root)

    symbolic_exact_selector_now = bool(
        symbolic_row["selector_reduced_residual_abs"]
        <= REL_TOL * max(abs(symbolic_row["selector_reduced_rhs"]), 1.0)
    )
    retained_continuity_support_now = bool(
        abs(retained_row["alpha_reduced_minus_form_factor"])
        <= RETAINED_REL_CONTINUITY_TOL
        * max(abs(retained_row["alpha_from_form_factor"]), 1.0e-30)
    )
    finite_invariant_algebra_available_now = bool(
        symbolic_exact_selector_now
        and retained_continuity_support_now
        and all(
            math.isfinite(retained_row[key])
            for key in ("f_beta", "g_beta", "q_beta", "b_beta")
        )
        and all(
            math.isfinite(symbolic_row[key])
            for key in ("f_beta", "g_beta", "q_beta", "b_beta")
        )
    )
    exact_alpha_finite_invariant_form_available_now = bool(
        finite_invariant_algebra_available_now
        and abs(symbolic_row["alpha_reduced_minus_form_factor"])
        <= REL_TOL * max(abs(symbolic_row["alpha_from_form_factor"]), 1.0)
    )
    zero_residual_closed_form_available_now = False
    exact_alpha_extraction_required_now = bool(
        exact_alpha_finite_invariant_form_available_now
        and not zero_residual_closed_form_available_now
    )

    return {
        "alpha_target": float(ALPHA_TARGET),
        "beta_common_root": beta_common_root,
        "beta_symbolic_root": beta_symbolic_root,
        "retained_row": retained_row,
        "symbolic_row": symbolic_row,
        "exact_trial2_selector_invariant_reduction_available_now": (
            symbolic_exact_selector_now
        ),
        "exact_trial2_retained_continuity_support_available_now": (
            retained_continuity_support_now
        ),
        "exact_trial2_finite_invariant_algebra_available_now": (
            finite_invariant_algebra_available_now
        ),
        "exact_trial2_finite_invariant_alpha_form_available_now": (
            exact_alpha_finite_invariant_form_available_now
        ),
        "exact_trial2_zero_residual_closed_form_available_now": (
            zero_residual_closed_form_available_now
        ),
        "updated_pack_trial2_exact_alpha_extraction_required_now": (
            exact_alpha_extraction_required_now
        ),
    }


# 関数: backend 単体実行時に compact summary を表示する。

def main() -> None:
    """Run the invariant-reduction audit directly."""
    pack = build_trial2_invariant_reduction_pack()
    retained_row = pack["retained_row"]
    print("[trial2_invariant_reduction_backend]")
    print(
        "  finite_invariant_algebra = "
        f"{pack['exact_trial2_finite_invariant_algebra_available_now']}"
    )
    print(f"  beta_common_root = {pack['beta_common_root']:.15f}")
    print(
        f"  alpha_from_reduced = {retained_row['alpha_from_reduced_invariants']:.15f}"
    )
    print(
        f"  selector_reduced_residual_abs = "
        f"{retained_row['selector_reduced_residual_abs']:.6e}"
    )


if __name__ == "__main__":
    main()
