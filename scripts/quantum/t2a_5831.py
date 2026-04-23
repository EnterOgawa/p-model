#!/usr/bin/env python3
"""Generate 8.7.56.5831-.5834 Trial-2 4D residual-absorption gate artifacts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.quantum.t2a_1843 as sign_base
from scripts.quantum.trial2_4d_residual_absorption_gate_backend import (
    build_trial2_4d_residual_absorption_gate_pack,
)
from scripts.utils.windows_length_policy import build_compact_artifact_stem
from scripts.utils.windows_length_policy import build_metrics_paths


PUBLIC_OUT = ROOT / "output" / "public" / "quantum"
PRIOR_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5827-5830",
        "updated_pack_trial2_4d_time_component_exact_alpha_correction_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5831-5834"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack Trial-2 "
    "4D residual-absorption gate"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_trial2_4d_residual_absorption_gate",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_4d_time_component_leading_mass_sq_correction_"
    "positive_partial_residual_absorption_gate_primary_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_4d_residual_absorption_gate_completed_"
    "exact_constant_selector_theorem_primary_next"
)


# 関数: JSON/CSV artifact を書き出す。
def write_artifact(kind: str, data: dict) -> dict[str, str]:
    """Write one JSON payload and one rows CSV."""
    PUBLIC_OUT.mkdir(parents=True, exist_ok=True)
    paths = build_metrics_paths(PUBLIC_OUT, STEM, kind)
    paths["json"].write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with paths["csv"].open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["row_id", "status", "metric", "value", "note"],
        )
        writer.writeheader()
        writer.writerows(data["rows"])

    return {"json": sign_base.display_path(paths["json"])}


# 関数: route で固定する式 bundle を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas fixed by the 4D residual-absorption gate."""
    return {
        "leading_correction_rule": (
            "alpha_4D,lead(beta_*) = alpha_3D(beta_*) / M_4D(beta_*, 1, ±1)^2"
        ),
        "gate_reading": (
            "The gate is passed only if the leading 4D correction is uniquely best "
            "overall, uniquely best within the leading selector, uniquely best within "
            "the mass-squared formula family, and polarization-dominant among the "
            "nonzero-time selectors."
        ),
        "residual_metric": (
            "rank rows by |(alpha_4D - 1/137) / (1/137)|"
        ),
    }


# 関数: `.5831-.5834` を実行する。

def main() -> None:
    """Execute the 4D residual-absorption gate."""
    sign_base.require(PRIOR_GATE)

    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    pack = build_trial2_4d_residual_absorption_gate_pack()
    best_row = pack["best_row"]

    route_selected = (
        str(prior_summary["trial2_numeric_alpha_problem_classification"]) == PRIOR_CLASS
    )
    unique_best_overall = bool(pack["exact_trial2_4d_unique_best_overall_now"])
    unique_best_within_selector = bool(
        pack["exact_trial2_4d_unique_best_within_selector_now"]
    )
    unique_best_within_formula = bool(
        pack["exact_trial2_4d_unique_best_within_formula_now"]
    )
    polarization_dominance = bool(
        pack["exact_trial2_4d_leading_selector_polarization_dominance_now"]
    )
    canonical_partial_absorber = bool(
        pack["exact_trial2_4d_canonical_partial_absorber_available_now"]
    )

    rows = [
        sign_base.row(
            "updated_pack_trial2_4d_exact_alpha_correction_selected_now",
            "pass" if route_selected else "reject",
            "updated-pack Trial-2 4D exact-alpha correction selected now",
            sign_base.truth(route_selected),
            "The residual-absorption gate starts only after the leading 4D correction has already been fixed as a positive partial residual absorber.",
        ),
        sign_base.row(
            "exact_trial2_4d_unique_best_overall_now",
            "pass" if unique_best_overall else "reject",
            "exact Trial-2 4D unique best overall now",
            sign_base.truth(unique_best_overall),
            "Across the full deterministic 4D family, the leading nontrivial time-component mass-squared correction has the smallest absolute exact-goal residual.",
        ),
        sign_base.row(
            "exact_trial2_4d_unique_best_within_selector_now",
            "pass" if unique_best_within_selector else "reject",
            "exact Trial-2 4D unique best within selector now",
            sign_base.truth(unique_best_within_selector),
            "Inside the leading selector itself, the inverse-squared mass rule is the unique best residual absorber.",
        ),
        sign_base.row(
            "exact_trial2_4d_unique_best_within_formula_now",
            "pass" if unique_best_within_formula else "reject",
            "exact Trial-2 4D unique best within formula now",
            sign_base.truth(unique_best_within_formula),
            "Inside the mass-squared correction family, the leading nontrivial selector is the unique best row.",
        ),
        sign_base.row(
            "exact_trial2_4d_leading_selector_polarization_dominance_now",
            "pass" if polarization_dominance else "reject",
            "exact Trial-2 4D leading selector polarization dominance now",
            sign_base.truth(polarization_dominance),
            "The retained leading selector also has the largest polarization weight among the nonzero-time selectors.",
        ),
        sign_base.row(
            "exact_trial2_4d_canonical_partial_absorber_available_now",
            "pass" if canonical_partial_absorber else "reject",
            "exact Trial-2 4D canonical partial absorber available now",
            sign_base.truth(canonical_partial_absorber),
            "The 4D correction is now canonized as one target-free best residual absorber inside the current deterministic family.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "canonical_selector_label": str(best_row["selector_label"]),
        "canonical_formula_label": str(best_row["formula_label"]),
        "canonical_corrected_alpha": float(best_row["corrected_alpha"]),
        "canonical_rel_error_vs_exact_goal": float(
            best_row["corrected_alpha_rel_error_vs_exact_goal"]
        ),
        "canonical_rel_error_vs_observed_target": float(
            best_row["corrected_alpha_rel_error_vs_observed_target"]
        ),
        "overall_margin_abs": float(pack["overall_margin_abs"]),
        "overall_ratio": float(pack["overall_ratio"]),
        "same_selector_margin_abs": float(pack["same_selector_margin_abs"]),
        "same_selector_ratio": float(pack["same_selector_ratio"]),
        "same_formula_margin_abs": float(pack["same_formula_margin_abs"]),
        "same_formula_ratio": float(pack["same_formula_ratio"]),
        "leading_weight_ratio": float(pack["leading_weight_ratio"]),
        "exact_trial2_4d_canonical_partial_absorber_available_now": bool(
            canonical_partial_absorber
        ),
        "selected_next_generation_route": "trial2_4d_exact_constant_selector_theorem_audit",
        "recommended_next_route_or_none": ".5835-.5838",
        "selected_followup_route": "trial2_4d_exact_goal_closeout_gate",
        "selected_followup_route_or_none": ".5839-.5842",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5833",
        STEP_NAME + " declaration gate",
        {
            "source_files": {"prior_gate": sign_base.display_path(PRIOR_GATE)},
            "formulae": build_formulae(),
        },
        rows,
        summary,
        {
            "overall_status": "trial2_4d_residual_absorption_gate_completed",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {
            "best_row": best_row,
            "second_row": pack["second_row"],
            "same_selector_second_row": pack["same_selector_second_row"],
            "same_formula_second_row": pack["same_formula_second_row"],
            "leading_selector_row": pack["leading_selector_row"],
            "next_nonzero_time_row": pack["next_nonzero_time_row"],
        },
    )
    outputs = write_artifact("declaration_gate", payload)
    print("[done] 8.7.56.5831-5834 Trial-2 4D residual-absorption gate completed")
    print(f"[done] declaration: {outputs['json']}")


if __name__ == "__main__":
    main()
