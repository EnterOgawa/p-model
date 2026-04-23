#!/usr/bin/env python3
"""Audit 4D time-component augmentation after 3D exactification exhaustion.

Purpose:
    Once the current 3D reduced invariant algebra is exhausted, the next honest
    exact-goal route is to augment the exact Trial-2 alpha formula with one
    time-component-sensitive 4D correction inherited from the retained vector
    Q-ball multicomponent reconstruction.

    This helper does not replay the old theorem-side 4D note inventory. It
    instead keeps the current exact Trial-2 alpha formula fixed and asks one
    computation-only question:

        can one deterministic 4D correction, evaluated at the retained
        symbolic root beta_symbolic, materially reduce the residual against
        the exact goal alpha = 1 / 137 ?

Inputs:
    - scripts/quantum/trial2_exact_alpha_closed_form_extraction_backend.py
    - scripts/quantum/mass_origin_vector_qball_full_coupled_solver_branch.py
    - output/public/quantum/mass_origin_vector_qball_exact_mass_table_handoff_retry_metrics.json

Outputs:
    - One in-memory audit pack consumed by `.5823-.5830` wrappers

Assumptions:
    - No new parameter is introduced
    - 1/137 is used only as an exact-goal audit comparator
    - The 4D selector family is frozen target-free before comparison
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.quantum.mass_origin_vector_qball_full_coupled_solver_branch import (
    coupled_charge_factor,
)
from scripts.quantum.mass_origin_vector_qball_full_coupled_solver_branch import (
    coupled_mass_factor,
)
from scripts.quantum.mass_origin_vector_qball_full_coupled_solver_branch import (
    polarization_weight,
)
from scripts.quantum.scalar_proxy_alpha_q_curve_backend import ALPHA_TARGET
from scripts.quantum.trial2_exact_alpha_closed_form_extraction_backend import (
    build_trial2_exact_alpha_closed_form_extraction_pack,
)


EXACT_GOAL = 1.0 / 137.0
EXACT_HANDOFF = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "mass_origin_vector_qball_exact_mass_table_handoff_retry_metrics.json"
)
LEADING_SELECTOR = {"label": "leading_nontrivial_time_component", "ell": 1, "s": 1}
REFERENCE_SELECTORS = (
    {"label": "helicity_zero_reference", "ell": 1, "s": 0},
    {"label": "retained_exact_ladder_anchor", "ell": 2, "s": -1},
    {"label": "higher_time_component_reference", "ell": 3, "s": 1},
)
FORMULA_LABELS = (
    "charge_inv",
    "charge_sq_inv",
    "mass_inv",
    "mass_sq_inv",
    "charge_mass_inv",
)
RESIDUAL_REDUCTION_THRESHOLD = 5.0


# 関数: target-free selector family を返す。
def build_selector_rows(beta_symbolic_root: float) -> list[dict]:
    """Return the frozen target-free selector family at the retained beta."""
    selector_rows: list[dict] = []
    for item in (LEADING_SELECTOR, *REFERENCE_SELECTORS):
        ell = int(item["ell"])
        s = int(item["s"])
        selector_rows.append(
            {
                "label": str(item["label"]),
                "ell": ell,
                "s": s,
                "beta": float(beta_symbolic_root),
                "polarization_weight": float(polarization_weight(beta_symbolic_root, ell, s)),
                "charge_factor": float(coupled_charge_factor(beta_symbolic_root, ell, s)),
                "mass_factor": float(coupled_mass_factor(beta_symbolic_root, ell, s)),
            }
        )

    return selector_rows


# 関数: 1つの selector row に対する 4D correction family を返す。

def build_formula_rows(alpha_exact_symbolic: float, selector_row: dict) -> list[dict]:
    """Return the deterministic 4D correction family for one selector row."""
    charge_factor = float(selector_row["charge_factor"])
    mass_factor = float(selector_row["mass_factor"])
    formula_rows: list[dict] = []

    formula_values = {
        "charge_inv": float(alpha_exact_symbolic / charge_factor),
        "charge_sq_inv": float(alpha_exact_symbolic / (charge_factor * charge_factor)),
        "mass_inv": float(alpha_exact_symbolic / mass_factor),
        "mass_sq_inv": float(alpha_exact_symbolic / (mass_factor * mass_factor)),
        "charge_mass_inv": float(alpha_exact_symbolic / (charge_factor * mass_factor)),
    }
    for label in FORMULA_LABELS:
        corrected_alpha = float(formula_values[label])
        formula_rows.append(
            {
                "selector_label": str(selector_row["label"]),
                "ell": int(selector_row["ell"]),
                "s": int(selector_row["s"]),
                "formula_label": label,
                "corrected_alpha": corrected_alpha,
                "corrected_alpha_rel_error_vs_exact_goal": float(
                    (corrected_alpha - EXACT_GOAL) / EXACT_GOAL
                ),
                "corrected_alpha_rel_error_vs_observed_target": float(
                    (corrected_alpha - ALPHA_TARGET) / ALPHA_TARGET
                ),
            }
        )

    return formula_rows


# 関数: retained exact-ladder anchor row を読む。

def read_exact_ladder_anchor() -> dict:
    """Return the retained exact-ladder anchor row from the public artifact."""
    payload = json.loads(EXACT_HANDOFF.read_text(encoding="utf-8"))
    return dict(payload["summary"]["best_exact_match_or_none"])


# 関数: 4D time-component augmentation 監査 pack を返す。

def build_trial2_4d_time_component_augmentation_pack() -> dict:
    """Return one audit pack for the 4D time-component augmentation route."""
    exact_pack = build_trial2_exact_alpha_closed_form_extraction_pack()
    alpha_exact_symbolic = float(exact_pack["alpha_exact_symbolic"])
    beta_symbolic_root = float(exact_pack["beta_symbolic_root"])
    baseline_rel_error_vs_exact_goal = float(
        exact_pack["alpha_exact_symbolic_rel_error_vs_exact_goal"]
    )
    selector_rows = build_selector_rows(beta_symbolic_root)
    formula_rows: list[dict] = []
    for selector_row in selector_rows:
        formula_rows.extend(build_formula_rows(alpha_exact_symbolic, selector_row))

    primary_formula_row = next(
        row
        for row in formula_rows
        if row["selector_label"] == LEADING_SELECTOR["label"]
        and row["formula_label"] == "mass_sq_inv"
    )
    best_formula_row = min(
        formula_rows,
        key=lambda row: abs(float(row["corrected_alpha_rel_error_vs_exact_goal"])),
    )

    primary_rel_error_vs_exact_goal = float(
        primary_formula_row["corrected_alpha_rel_error_vs_exact_goal"]
    )
    primary_rel_error_vs_observed_target = float(
        primary_formula_row["corrected_alpha_rel_error_vs_observed_target"]
    )
    residual_reduction_factor = float(
        abs(baseline_rel_error_vs_exact_goal)
        / max(abs(primary_rel_error_vs_exact_goal), 1.0e-30)
    )
    sign_crossing_now = bool(
        baseline_rel_error_vs_exact_goal > 0.0 and primary_rel_error_vs_exact_goal < 0.0
    )
    exact_goal_residual_reduced_now = bool(
        abs(primary_rel_error_vs_exact_goal) < abs(baseline_rel_error_vs_exact_goal)
    )
    positive_partial_absorption_now = bool(
        exact_goal_residual_reduced_now
        and residual_reduction_factor >= RESIDUAL_REDUCTION_THRESHOLD
    )
    zero_residual_available_now = bool(
        abs(primary_rel_error_vs_exact_goal) <= 1.0e-14
    )
    exact_alpha_correction_required_now = bool(
        positive_partial_absorption_now and not zero_residual_available_now
    )

    return {
        "alpha_target_observed": float(ALPHA_TARGET),
        "alpha_goal_exact_one_over_137": float(EXACT_GOAL),
        "beta_symbolic_root": beta_symbolic_root,
        "alpha_exact_symbolic": alpha_exact_symbolic,
        "alpha_exact_symbolic_rel_error_vs_exact_goal": baseline_rel_error_vs_exact_goal,
        "alpha_exact_symbolic_rel_error_vs_observed_target": float(
            exact_pack["alpha_exact_symbolic_rel_error_vs_observed_target"]
        ),
        "retained_exact_ladder_anchor_row": read_exact_ladder_anchor(),
        "selector_rows": selector_rows,
        "formula_rows": formula_rows,
        "leading_selector_label": str(LEADING_SELECTOR["label"]),
        "leading_primary_formula_label": "mass_sq_inv",
        "leading_primary_formula_row": primary_formula_row,
        "best_formula_row": best_formula_row,
        "exact_trial2_4d_selector_family_machine_readable_now": True,
        "exact_trial2_4d_leading_time_component_selector_available_now": True,
        "exact_trial2_4d_positive_partial_residual_absorption_now": (
            positive_partial_absorption_now
        ),
        "exact_trial2_4d_zero_residual_exact_constant_available_now": (
            zero_residual_available_now
        ),
        "exact_trial2_4d_exact_alpha_correction_required_now": (
            exact_alpha_correction_required_now
        ),
        "exact_trial2_4d_sign_crossing_now": sign_crossing_now,
        "exact_trial2_4d_exact_goal_residual_reduction_factor": (
            residual_reduction_factor
        ),
        "leading_primary_rel_error_vs_exact_goal": primary_rel_error_vs_exact_goal,
        "leading_primary_rel_error_vs_observed_target": (
            primary_rel_error_vs_observed_target
        ),
    }


# 関数: backend 単体実行時に compact summary を表示する。

def main() -> None:
    """Run the 4D time-component augmentation audit directly."""
    pack = build_trial2_4d_time_component_augmentation_pack()
    primary = pack["leading_primary_formula_row"]
    print("[trial2_4d_time_component_augmentation_backend]")
    print(f"  beta_symbolic_root = {pack['beta_symbolic_root']:.15f}")
    print(f"  alpha_exact_symbolic = {pack['alpha_exact_symbolic']:.15f}")
    print(
        "  leading_primary_formula = "
        f"{primary['selector_label']} / {primary['formula_label']}"
    )
    print(f"  corrected_alpha = {primary['corrected_alpha']:.15f}")
    print(
        "  rel_error_vs_exact_goal = "
        f"{primary['corrected_alpha_rel_error_vs_exact_goal']:+.12e}"
    )


if __name__ == "__main__":
    main()
