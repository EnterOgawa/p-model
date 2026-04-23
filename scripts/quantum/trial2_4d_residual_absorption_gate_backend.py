#!/usr/bin/env python3
"""Compute the canonical 4D residual-absorption gate metrics for Trial-2.

Purpose:
    Consume the already materialized deterministic 4D correction family and
    decide whether the leading 4D correction is not only helpful, but also the
    unique best residual absorber inside the current no-new-parameter family.

Inputs:
    - scripts/quantum/trial2_4d_time_component_augmentation_backend.py

Outputs:
    - One in-memory gate pack consumed by `.5831-.5834` wrappers

Assumptions:
    - The 3D internal exactification routes are already exhausted
    - The leading selector family is already fixed target-free
    - The exact goal `1/137` is used only as a comparator
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.quantum.trial2_4d_time_component_augmentation_backend import (
    LEADING_SELECTOR,
)
from scripts.quantum.trial2_4d_time_component_augmentation_backend import (
    build_trial2_4d_time_component_augmentation_pack,
)


UNIQUE_MARGIN_MIN = 1.0e-6


# 関数: exact-goal residual の絶対値で row を整列する。
def sort_rows_by_exact_goal_error(rows: list[dict]) -> list[dict]:
    """Return rows ordered by absolute exact-goal residual."""
    return sorted(rows, key=lambda row: abs(float(row["corrected_alpha_rel_error_vs_exact_goal"])))


# 関数: 4D residual-absorption gate pack を返す。

def build_trial2_4d_residual_absorption_gate_pack() -> dict:
    """Return the canonical 4D residual-absorption gate metrics."""
    augmentation_pack = build_trial2_4d_time_component_augmentation_pack()
    all_rows = sort_rows_by_exact_goal_error(list(augmentation_pack["formula_rows"]))
    best_row = dict(all_rows[0])
    second_row = dict(all_rows[1])

    same_selector_rows = sort_rows_by_exact_goal_error(
        [
            row
            for row in all_rows
            if str(row["selector_label"]) == str(best_row["selector_label"])
        ]
    )
    same_formula_rows = sort_rows_by_exact_goal_error(
        [
            row
            for row in all_rows
            if str(row["formula_label"]) == str(best_row["formula_label"])
        ]
    )
    selector_rows = list(augmentation_pack["selector_rows"])
    leading_selector_row = next(
        row
        for row in selector_rows
        if str(row["label"]) == str(LEADING_SELECTOR["label"])
    )
    nonzero_time_rows = [
        row for row in selector_rows if int(row["s"]) != 0
    ]
    ordered_nonzero_time_rows = sorted(
        nonzero_time_rows,
        key=lambda row: float(row["polarization_weight"]),
        reverse=True,
    )
    next_nonzero_time_row = dict(ordered_nonzero_time_rows[1])

    best_exact_gap_abs = abs(float(best_row["corrected_alpha_rel_error_vs_exact_goal"]))
    second_exact_gap_abs = abs(float(second_row["corrected_alpha_rel_error_vs_exact_goal"]))
    overall_margin_abs = float(second_exact_gap_abs - best_exact_gap_abs)
    overall_ratio = float(second_exact_gap_abs / max(best_exact_gap_abs, 1.0e-30))

    same_selector_best = dict(same_selector_rows[0])
    same_selector_second = dict(same_selector_rows[1])
    same_selector_margin_abs = float(
        abs(float(same_selector_second["corrected_alpha_rel_error_vs_exact_goal"]))
        - abs(float(same_selector_best["corrected_alpha_rel_error_vs_exact_goal"]))
    )
    same_selector_ratio = float(
        abs(float(same_selector_second["corrected_alpha_rel_error_vs_exact_goal"]))
        / max(abs(float(same_selector_best["corrected_alpha_rel_error_vs_exact_goal"])), 1.0e-30)
    )

    same_formula_best = dict(same_formula_rows[0])
    same_formula_second = dict(same_formula_rows[1])
    same_formula_margin_abs = float(
        abs(float(same_formula_second["corrected_alpha_rel_error_vs_exact_goal"]))
        - abs(float(same_formula_best["corrected_alpha_rel_error_vs_exact_goal"]))
    )
    same_formula_ratio = float(
        abs(float(same_formula_second["corrected_alpha_rel_error_vs_exact_goal"]))
        / max(abs(float(same_formula_best["corrected_alpha_rel_error_vs_exact_goal"])), 1.0e-30)
    )

    leading_polarization_weight = float(leading_selector_row["polarization_weight"])
    next_nonzero_polarization_weight = float(next_nonzero_time_row["polarization_weight"])
    leading_weight_ratio = float(
        leading_polarization_weight / max(next_nonzero_polarization_weight, 1.0e-30)
    )

    primary_is_overall_best = bool(
        str(best_row["selector_label"]) == str(LEADING_SELECTOR["label"])
        and str(best_row["formula_label"]) == "mass_sq_inv"
    )
    exact_trial2_4d_unique_best_overall_now = bool(
        primary_is_overall_best and overall_margin_abs > UNIQUE_MARGIN_MIN
    )
    exact_trial2_4d_unique_best_within_selector_now = bool(
        str(same_selector_best["formula_label"]) == "mass_sq_inv"
        and same_selector_margin_abs > UNIQUE_MARGIN_MIN
    )
    exact_trial2_4d_unique_best_within_formula_now = bool(
        str(same_formula_best["selector_label"]) == str(LEADING_SELECTOR["label"])
        and same_formula_margin_abs > UNIQUE_MARGIN_MIN
    )
    exact_trial2_4d_leading_selector_polarization_dominance_now = bool(
        leading_polarization_weight > next_nonzero_polarization_weight
    )
    exact_trial2_4d_canonical_partial_absorber_available_now = bool(
        augmentation_pack["exact_trial2_4d_positive_partial_residual_absorption_now"]
        and exact_trial2_4d_unique_best_overall_now
        and exact_trial2_4d_unique_best_within_selector_now
        and exact_trial2_4d_unique_best_within_formula_now
        and exact_trial2_4d_leading_selector_polarization_dominance_now
    )

    return {
        "augmentation_pack": augmentation_pack,
        "best_row": best_row,
        "second_row": second_row,
        "same_selector_best_row": same_selector_best,
        "same_selector_second_row": same_selector_second,
        "same_formula_best_row": same_formula_best,
        "same_formula_second_row": same_formula_second,
        "leading_selector_row": leading_selector_row,
        "next_nonzero_time_row": next_nonzero_time_row,
        "best_exact_gap_abs": float(best_exact_gap_abs),
        "second_exact_gap_abs": float(second_exact_gap_abs),
        "overall_margin_abs": overall_margin_abs,
        "overall_ratio": overall_ratio,
        "same_selector_margin_abs": same_selector_margin_abs,
        "same_selector_ratio": same_selector_ratio,
        "same_formula_margin_abs": same_formula_margin_abs,
        "same_formula_ratio": same_formula_ratio,
        "leading_weight_ratio": leading_weight_ratio,
        "exact_trial2_4d_unique_best_overall_now": exact_trial2_4d_unique_best_overall_now,
        "exact_trial2_4d_unique_best_within_selector_now": (
            exact_trial2_4d_unique_best_within_selector_now
        ),
        "exact_trial2_4d_unique_best_within_formula_now": (
            exact_trial2_4d_unique_best_within_formula_now
        ),
        "exact_trial2_4d_leading_selector_polarization_dominance_now": (
            exact_trial2_4d_leading_selector_polarization_dominance_now
        ),
        "exact_trial2_4d_canonical_partial_absorber_available_now": (
            exact_trial2_4d_canonical_partial_absorber_available_now
        ),
    }


# 関数: backend 単体実行時に compact summary を表示する。

def main() -> None:
    """Run the 4D residual-absorption gate backend directly."""
    gate_pack = build_trial2_4d_residual_absorption_gate_pack()
    best_row = gate_pack["best_row"]
    print("[trial2_4d_residual_absorption_gate_backend]")
    print(
        "  best = "
        f"{best_row['selector_label']} / {best_row['formula_label']}"
    )
    print(f"  corrected_alpha = {best_row['corrected_alpha']:.15f}")
    print(
        "  rel_error_vs_exact_goal = "
        f"{best_row['corrected_alpha_rel_error_vs_exact_goal']:+.12e}"
    )
    print(f"  overall_margin_abs = {gate_pack['overall_margin_abs']:.12e}")
    print(f"  overall_ratio = {gate_pack['overall_ratio']:.12f}")


if __name__ == "__main__":
    main()
