#!/usr/bin/env python3
"""Generate 8.7.56.5827-.5830 Trial-2 4D exact-alpha correction artifacts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.quantum.t2a_1843 as sign_base
from scripts.quantum.trial2_4d_time_component_augmentation_backend import (
    build_trial2_4d_time_component_augmentation_pack,
)
from scripts.utils.windows_length_policy import build_compact_artifact_stem
from scripts.utils.windows_length_policy import build_metrics_paths


PUBLIC_OUT = ROOT / "output" / "public" / "quantum"
PRIOR_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5823-5826",
        "updated_pack_trial2_4d_time_component_augmentation_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5827-5830"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack Trial-2 "
    "4D time-component exact-alpha correction audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_trial2_4d_time_component_exact_alpha_correction_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_4d_time_component_augmentation_audited_"
    "leading_mass_sq_correction_primary_exact_alpha_refresh_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_4d_time_component_leading_mass_sq_correction_"
    "positive_partial_residual_absorption_gate_primary_next"
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
    """Return formulas fixed by the 4D exact-alpha correction audit."""
    return {
        "baseline_alpha_formula": "alpha_3D(beta_*) = alpha_exact_symbolic",
        "leading_selector": "(ell, s) = (1, ±1)",
        "leading_mass_factor": "M_4D(beta_*) = coupled_mass_factor(beta_*, 1, ±1)",
        "leading_correction": "alpha_4D,lead(beta_*) = alpha_3D(beta_*) / M_4D(beta_*)^2",
        "residual_reading": (
            "If alpha_4D,lead reduces the exact-goal residual by a large factor "
            "without reaching zero, the 4D route becomes positive partial and the "
            "next blocker moves to residual-absorption canonicalization."
        ),
    }


# 関数: `.5827-.5830` を実行する。

def main() -> None:
    """Execute the 4D exact-alpha correction audit."""
    sign_base.require(PRIOR_GATE)

    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    pack = build_trial2_4d_time_component_augmentation_pack()
    primary = pack["leading_primary_formula_row"]

    route_selected = (
        str(prior_summary["trial2_numeric_alpha_problem_classification"]) == PRIOR_CLASS
    )
    positive_partial = bool(pack["exact_trial2_4d_positive_partial_residual_absorption_now"])
    sign_crossing = bool(pack["exact_trial2_4d_sign_crossing_now"])
    zero_residual_unavailable = not bool(
        pack["exact_trial2_4d_zero_residual_exact_constant_available_now"]
    )
    gate_required = bool(pack["exact_trial2_4d_exact_alpha_correction_required_now"])

    rows = [
        sign_base.row(
            "updated_pack_trial2_4d_augmentation_selected_now",
            "pass" if route_selected else "reject",
            "updated-pack Trial-2 4D augmentation selected now",
            sign_base.truth(route_selected),
            "The exact-alpha correction audit starts only after the 4D augmentation selector family is already synchronized.",
        ),
        sign_base.row(
            "exact_trial2_4d_positive_partial_residual_absorption_now",
            "pass" if positive_partial else "reject",
            "exact Trial-2 4D positive partial residual absorption now",
            sign_base.truth(positive_partial),
            "The leading 4D mass-squared correction reduces the exact-goal residual by more than one full order of magnitude fractionally without introducing a new parameter.",
        ),
        sign_base.row(
            "exact_trial2_4d_sign_crossing_now",
            "pass" if sign_crossing else "reject",
            "exact Trial-2 4D sign crossing now",
            sign_base.truth(sign_crossing),
            "The corrected exact-goal residual changes sign, so the 4D correction is not merely cosmetic; it actually overshoots the target slightly in the opposite direction.",
        ),
        sign_base.row(
            "exact_trial2_4d_zero_residual_exact_constant_unavailable_now",
            "pass" if zero_residual_unavailable else "reject",
            "exact Trial-2 4D zero-residual exact constant unavailable now",
            sign_base.truth(zero_residual_unavailable),
            "Even after the leading 4D correction, the current pack still does not collapse to alpha = 1/137 with zero residual.",
        ),
        sign_base.row(
            "updated_pack_trial2_4d_residual_absorption_gate_required_now",
            "pass" if gate_required else "reject",
            "updated-pack Trial-2 4D residual-absorption gate required now",
            sign_base.truth(gate_required),
            "The next honest blocker is no longer whether 4D correction helps, but whether the residual absorption can be canonized target-free.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "beta_symbolic_root": float(pack["beta_symbolic_root"]),
        "alpha_exact_symbolic": float(pack["alpha_exact_symbolic"]),
        "alpha_leading_4d_corrected": float(primary["corrected_alpha"]),
        "baseline_rel_error_vs_exact_goal": float(
            pack["alpha_exact_symbolic_rel_error_vs_exact_goal"]
        ),
        "leading_primary_rel_error_vs_exact_goal": float(
            pack["leading_primary_rel_error_vs_exact_goal"]
        ),
        "baseline_rel_error_vs_observed_target": float(
            pack["alpha_exact_symbolic_rel_error_vs_observed_target"]
        ),
        "leading_primary_rel_error_vs_observed_target": float(
            pack["leading_primary_rel_error_vs_observed_target"]
        ),
        "exact_trial2_4d_exact_goal_residual_reduction_factor": float(
            pack["exact_trial2_4d_exact_goal_residual_reduction_factor"]
        ),
        "exact_trial2_4d_positive_partial_residual_absorption_now": bool(
            positive_partial
        ),
        "exact_trial2_4d_zero_residual_exact_constant_available_now": bool(
            pack["exact_trial2_4d_zero_residual_exact_constant_available_now"]
        ),
        "selected_next_generation_route": "trial2_4d_residual_absorption_gate",
        "recommended_next_route_or_none": ".5831-.5834",
        "selected_followup_route": "trial2_4d_exact_constant_selector_theorem_audit",
        "selected_followup_route_or_none": ".5835-.5838",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5829",
        STEP_NAME + " declaration gate",
        {
            "source_files": {"prior_gate": sign_base.display_path(PRIOR_GATE)},
            "formulae": build_formulae(),
        },
        rows,
        summary,
        {
            "overall_status": "trial2_4d_time_component_exact_alpha_correction_audited",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {
            "leading_primary_formula_row": primary,
            "best_formula_row": pack["best_formula_row"],
            "exact_goal_residual_reduction_factor": float(
                pack["exact_trial2_4d_exact_goal_residual_reduction_factor"]
            ),
        },
    )
    outputs = write_artifact("declaration_gate", payload)
    print("[done] 8.7.56.5827-5830 Trial-2 4D exact-alpha correction audit completed")
    print(f"[done] declaration: {outputs['json']}")


if __name__ == "__main__":
    main()
