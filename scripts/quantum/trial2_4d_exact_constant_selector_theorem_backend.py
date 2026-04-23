#!/usr/bin/env python3
"""Audit the canonical exact-constant selector theorem inside the current 4D family.

Purpose:
    After the 4D time-component route is shown to be a positive partial residual
    absorber, the next honest question is whether the current deterministic
    no-new-parameter 4D family already selects one canonical correction rule
    target-free.

    This helper does not search for a new family. It consumes the existing
    residual-absorption gate metrics and asks whether one row is uniquely best
    in a way strong enough to canonize one exact-alpha correction theorem:

        alpha_4D,can(beta_*) = alpha_3D(beta_*) / M_4D(beta_*, 1, ±1)^2

Inputs:
    - scripts/quantum/trial2_4d_residual_absorption_gate_backend.py

Outputs:
    - One in-memory theorem pack consumed by `.5835-.5838` and `.5839-.5842`

Assumptions:
    - The 4D selector family is already fixed target-free
    - The exact goal `1/137` is used only as a comparator
    - The theorem decided here is selector canonicity, not zero-residual closure
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.quantum.trial2_4d_residual_absorption_gate_backend import (
    build_trial2_4d_residual_absorption_gate_pack,
)


# 関数: 4D exact-constant selector theorem pack を返す。
def build_trial2_4d_exact_constant_selector_theorem_pack() -> dict:
    """Return the canonical 4D exact-constant selector theorem metrics."""
    gate_pack = build_trial2_4d_residual_absorption_gate_pack()
    best_row = dict(gate_pack["best_row"])
    leading_selector_row = dict(gate_pack["leading_selector_row"])
    next_nonzero_time_row = dict(gate_pack["next_nonzero_time_row"])

    exact_trial2_4d_canonical_exact_alpha_correction_formula_available_now = bool(
        gate_pack["exact_trial2_4d_canonical_partial_absorber_available_now"]
    )
    exact_trial2_4d_exact_constant_selector_theorem_available_now = bool(
        exact_trial2_4d_canonical_exact_alpha_correction_formula_available_now
        and str(best_row["selector_label"]) == "leading_nontrivial_time_component"
        and str(best_row["formula_label"]) == "mass_sq_inv"
    )
    exact_trial2_4d_zero_residual_exact_goal_available_now = bool(
        abs(float(best_row["corrected_alpha_rel_error_vs_exact_goal"])) <= 1.0e-14
    )
    exact_trial2_4d_exact_goal_closeout_gate_required_now = bool(
        exact_trial2_4d_exact_constant_selector_theorem_available_now
        and not exact_trial2_4d_zero_residual_exact_goal_available_now
    )

    return {
        "gate_pack": gate_pack,
        "best_row": best_row,
        "leading_selector_row": leading_selector_row,
        "next_nonzero_time_row": next_nonzero_time_row,
        "canonical_selector_label": str(best_row["selector_label"]),
        "canonical_formula_label": str(best_row["formula_label"]),
        "canonical_corrected_alpha": float(best_row["corrected_alpha"]),
        "canonical_rel_error_vs_exact_goal": float(
            best_row["corrected_alpha_rel_error_vs_exact_goal"]
        ),
        "canonical_rel_error_vs_observed_target": float(
            best_row["corrected_alpha_rel_error_vs_observed_target"]
        ),
        "canonical_mass_factor": float(leading_selector_row["mass_factor"]),
        "canonical_charge_factor": float(leading_selector_row["charge_factor"]),
        "leading_selector_polarization_weight": float(
            leading_selector_row["polarization_weight"]
        ),
        "next_nonzero_polarization_weight": float(
            next_nonzero_time_row["polarization_weight"]
        ),
        "overall_margin_abs": float(gate_pack["overall_margin_abs"]),
        "overall_ratio": float(gate_pack["overall_ratio"]),
        "same_selector_margin_abs": float(gate_pack["same_selector_margin_abs"]),
        "same_selector_ratio": float(gate_pack["same_selector_ratio"]),
        "same_formula_margin_abs": float(gate_pack["same_formula_margin_abs"]),
        "same_formula_ratio": float(gate_pack["same_formula_ratio"]),
        "leading_weight_ratio": float(gate_pack["leading_weight_ratio"]),
        "exact_trial2_4d_canonical_exact_alpha_correction_formula_available_now": (
            exact_trial2_4d_canonical_exact_alpha_correction_formula_available_now
        ),
        "exact_trial2_4d_exact_constant_selector_theorem_available_now": (
            exact_trial2_4d_exact_constant_selector_theorem_available_now
        ),
        "exact_trial2_4d_zero_residual_exact_goal_available_now": (
            exact_trial2_4d_zero_residual_exact_goal_available_now
        ),
        "exact_trial2_4d_exact_goal_closeout_gate_required_now": (
            exact_trial2_4d_exact_goal_closeout_gate_required_now
        ),
    }


# 関数: backend 単体実行時に compact summary を表示する。

def main() -> None:
    """Run the 4D exact-constant selector theorem backend directly."""
    theorem_pack = build_trial2_4d_exact_constant_selector_theorem_pack()
    print("[trial2_4d_exact_constant_selector_theorem_backend]")
    print(
        "  canonical = "
        f"{theorem_pack['canonical_selector_label']} / "
        f"{theorem_pack['canonical_formula_label']}"
    )
    print(f"  corrected_alpha = {theorem_pack['canonical_corrected_alpha']:.15f}")
    print(
        "  rel_error_vs_exact_goal = "
        f"{theorem_pack['canonical_rel_error_vs_exact_goal']:+.12e}"
    )
    print(
        "  selector_theorem = "
        f"{theorem_pack['exact_trial2_4d_exact_constant_selector_theorem_available_now']}"
    )


if __name__ == "__main__":
    main()
