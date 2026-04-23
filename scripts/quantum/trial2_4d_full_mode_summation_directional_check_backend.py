#!/usr/bin/env python3
"""Evaluate whether full 4D mode summation can close the exact-goal residual.

Purpose:
    After canonical 4D correction is fixed as the unique best single-row
    residual absorber, this helper checks the cheapest remaining computation
    question from the expert route inventory:

        can full mode accumulation across the already materialized 4D selector
        family beat the canonical single-row correction, or does it merely
        provide directional evidence for a different missing normalization law?

    The branch is intentionally computation-only. It does not search for a new
    theorem. It evaluates deterministic weighted aggregates built from the
    current selector family and compares them against both the 3D baseline and
    the canonical 4D correction.

Inputs:
    - scripts/quantum/trial2_4d_time_component_augmentation_backend.py
    - scripts/quantum/trial2_4d_residual_absorption_gate_backend.py

Outputs:
    - One in-memory audit pack consumed by `.5843-.5850` wrappers

Assumptions:
    - The deterministic 4D selector family is already fixed target-free
    - Canonical 4D exact-alpha correction is already available
    - The exact goal `1/137` is used only as a comparator
"""

from __future__ import annotations

import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.quantum.trial2_4d_residual_absorption_gate_backend import (
    build_trial2_4d_residual_absorption_gate_pack,
)
from scripts.quantum.trial2_4d_time_component_augmentation_backend import EXACT_GOAL


AGGREGATION_SPECS = (
    {
        "selector_set": "nonzero_time",
        "formula_label": "mass_sq_weighted_sum",
        "kind": "arithmetic",
        "note": "alpha_3D divided by normalized sum of M_4D^2 over nonzero-time selectors.",
    },
    {
        "selector_set": "nonzero_time",
        "formula_label": "charge_mass_weighted_sum",
        "kind": "arithmetic",
        "note": "alpha_3D divided by normalized sum of C_4D M_4D over nonzero-time selectors.",
    },
    {
        "selector_set": "nonzero_time",
        "formula_label": "half_mixed_geometric_sum",
        "kind": "geometric",
        "note": "Diagnostic only: alpha_3D divided by exp(sum w_hat [0.5 ln C_4D + 1.5 ln M_4D]).",
    },
    {
        "selector_set": "all_selectors",
        "formula_label": "mass_sq_weighted_sum",
        "kind": "arithmetic",
        "note": "alpha_3D divided by normalized sum of M_4D^2 over the full selector family.",
    },
    {
        "selector_set": "all_selectors",
        "formula_label": "charge_mass_weighted_sum",
        "kind": "arithmetic",
        "note": "alpha_3D divided by normalized sum of C_4D M_4D over the full selector family.",
    },
)


# 関数: selector rows を指定 family で返す。
def select_selector_rows(selector_rows: list[dict], selector_set: str) -> list[dict]:
    """Return the requested deterministic selector subfamily."""
    if selector_set == "nonzero_time":
        return [row for row in selector_rows if int(row["s"]) != 0]

    if selector_set == "all_selectors":
        return list(selector_rows)

    raise ValueError(f"unsupported selector_set: {selector_set}")


# 関数: family 内の正規化重みを返す。

def build_weighted_rows(selector_rows: list[dict]) -> list[dict]:
    """Return selector rows augmented with normalized polarization weights."""
    weight_sum = sum(float(row["polarization_weight"]) for row in selector_rows)
    return [
        {
            **row,
            "normalized_weight": float(float(row["polarization_weight"]) / max(weight_sum, 1.0e-30)),
        }
        for row in selector_rows
    ]


# 関数: one summation denominator を返す。

def build_denominator(weighted_rows: list[dict], formula_label: str, kind: str) -> float:
    """Return one deterministic weighted denominator."""
    if kind == "arithmetic" and formula_label == "mass_sq_weighted_sum":
        return float(
            sum(
                float(row["normalized_weight"]) * float(row["mass_factor"]) ** 2
                for row in weighted_rows
            )
        )

    if kind == "arithmetic" and formula_label == "charge_mass_weighted_sum":
        return float(
            sum(
                float(row["normalized_weight"])
                * float(row["charge_factor"])
                * float(row["mass_factor"])
                for row in weighted_rows
            )
        )

    if kind == "geometric" and formula_label == "half_mixed_geometric_sum":
        exponent = sum(
            float(row["normalized_weight"])
            * (
                0.5 * math.log(float(row["charge_factor"]))
                + 1.5 * math.log(float(row["mass_factor"]))
            )
            for row in weighted_rows
        )
        return float(math.exp(exponent))

    raise ValueError(f"unsupported aggregation: {formula_label=} {kind=}")


# 関数: one summation candidate row を返す。

def build_aggregated_row(
    alpha_3d: float,
    baseline_rel_error: float,
    weighted_rows: list[dict],
    selector_set: str,
    formula_label: str,
    kind: str,
    note: str,
) -> dict:
    """Return one machine-readable summation candidate row."""
    denominator = build_denominator(weighted_rows, formula_label, kind)
    corrected_alpha = float(alpha_3d / denominator)
    rel_error_vs_exact_goal = float((corrected_alpha - EXACT_GOAL) / EXACT_GOAL)
    reduction_factor = float(
        abs(baseline_rel_error) / max(abs(rel_error_vs_exact_goal), 1.0e-30)
    )
    return {
        "selector_set": selector_set,
        "formula_label": formula_label,
        "aggregation_kind": kind,
        "denominator": denominator,
        "corrected_alpha": corrected_alpha,
        "corrected_alpha_rel_error_vs_exact_goal": rel_error_vs_exact_goal,
        "exact_goal_residual_reduction_factor": reduction_factor,
        "note": note,
        "weighted_rows": weighted_rows,
    }


# 関数: 4D full mode summation directional pack を返す。

def build_trial2_4d_full_mode_summation_directional_check_pack() -> dict:
    """Return the deterministic full-mode directional check pack."""
    gate_pack = build_trial2_4d_residual_absorption_gate_pack()
    augmentation_pack = gate_pack["augmentation_pack"]
    alpha_3d = float(augmentation_pack["alpha_exact_symbolic"])
    baseline_rel_error = float(augmentation_pack["alpha_exact_symbolic_rel_error_vs_exact_goal"])
    selector_rows = list(augmentation_pack["selector_rows"])
    canonical_row = dict(gate_pack["best_row"])
    canonical_rel_error = float(canonical_row["corrected_alpha_rel_error_vs_exact_goal"])

    aggregated_rows: list[dict] = []
    for spec in AGGREGATION_SPECS:
        family_rows = select_selector_rows(selector_rows, str(spec["selector_set"]))
        weighted_rows = build_weighted_rows(family_rows)
        aggregated_rows.append(
            build_aggregated_row(
                alpha_3d=alpha_3d,
                baseline_rel_error=baseline_rel_error,
                weighted_rows=weighted_rows,
                selector_set=str(spec["selector_set"]),
                formula_label=str(spec["formula_label"]),
                kind=str(spec["kind"]),
                note=str(spec["note"]),
            )
        )

    ordered_rows = sorted(
        aggregated_rows,
        key=lambda row: abs(float(row["corrected_alpha_rel_error_vs_exact_goal"])),
    )
    best_row = dict(ordered_rows[0])
    second_row = dict(ordered_rows[1])
    best_rel_error = float(best_row["corrected_alpha_rel_error_vs_exact_goal"])
    second_rel_error = float(second_row["corrected_alpha_rel_error_vs_exact_goal"])

    best_improves_baseline = bool(abs(best_rel_error) < abs(baseline_rel_error))
    best_beats_canonical = bool(abs(best_rel_error) < abs(canonical_rel_error))
    directional_negative_closeout = bool(best_improves_baseline and not best_beats_canonical)
    structural_support_for_mixed_normalization = bool(
        best_improves_baseline
        and not best_beats_canonical
        and best_rel_error > 0.0
        and canonical_rel_error < 0.0
    )

    return {
        "alpha_exact_symbolic": alpha_3d,
        "alpha_exact_symbolic_rel_error_vs_exact_goal": baseline_rel_error,
        "canonical_row": canonical_row,
        "canonical_rel_error_vs_exact_goal": canonical_rel_error,
        "aggregated_rows": aggregated_rows,
        "best_row": best_row,
        "second_row": second_row,
        "best_row_margin_abs": float(abs(second_rel_error) - abs(best_rel_error)),
        "best_row_ratio_vs_second": float(
            abs(second_rel_error) / max(abs(best_rel_error), 1.0e-30)
        ),
        "best_improves_baseline_now": best_improves_baseline,
        "best_beats_canonical_now": best_beats_canonical,
        "exact_trial2_4d_full_mode_directional_negative_closeout_now": (
            directional_negative_closeout
        ),
        "exact_trial2_4d_selector_mixed_normalization_route_supported_now": (
            structural_support_for_mixed_normalization
        ),
        "canonical_advantage_factor": float(
            abs(best_rel_error) / max(abs(canonical_rel_error), 1.0e-30)
        ),
    }


# 関数: backend 単体実行時に compact summary を表示する。

def main() -> None:
    """Run the 4D full mode directional check backend directly."""
    pack = build_trial2_4d_full_mode_summation_directional_check_pack()
    best_row = pack["best_row"]
    canonical_row = pack["canonical_row"]
    print("[trial2_4d_full_mode_summation_directional_check_backend]")
    print(
        "  best_full_mode = "
        f"{best_row['selector_set']} / {best_row['formula_label']}"
    )
    print(f"  corrected_alpha = {best_row['corrected_alpha']:.15f}")
    print(
        "  rel_error_vs_exact_goal = "
        f"{best_row['corrected_alpha_rel_error_vs_exact_goal']:+.12e}"
    )
    print(
        "  canonical_rel_error_vs_exact_goal = "
        f"{canonical_row['corrected_alpha_rel_error_vs_exact_goal']:+.12e}"
    )
    print(
        "  structural_support_for_mixed_normalization = "
        f"{pack['exact_trial2_4d_selector_mixed_normalization_route_supported_now']}"
    )


if __name__ == "__main__":
    main()
