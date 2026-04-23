#!/usr/bin/env python3
"""Generate 8.7.56.5651-.5654 Trial-2 beta-sensitivity monotonicity gate artifacts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.quantum.t2a_1843 as sign_base
from scripts.utils.windows_length_policy import build_compact_artifact_stem
from scripts.utils.windows_length_policy import build_metrics_paths


PUBLIC_OUT = ROOT / "output" / "public" / "quantum"
PRIOR_AUDIT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5647-5650",
        "updated_pack_trial2_beta_sensitivity_monotonicity_followup_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5651-5654"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack Trial-2 "
    "beta-sensitivity monotonicity gate / conditional-hold refresh"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_trial2_beta_sensitivity_maximum_principle_gate",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_beta_sensitivity_maximum_principle_audited_green_kernel_gate_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_beta_sensitivity_maximum_principle_negative_closeout_completed_"
    "green_kernel_followup_primary_conditional_hold_secondary_next"
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


# 関数: gate で使う式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used by the maximum-principle gate."""
    return {
        "gate_a": "Gate A = transformed beta-sensitivity operator is available now",
        "gate_b": "Gate B = canonical-window maximum-principle path closes negatively now",
        "gate_c": "Gate C = green-kernel followup is promoted while conditional hold stays secondary",
    }


# 関数: `.5651-.5654` を実行する。

def main() -> None:
    """Execute the Trial-2 beta-sensitivity monotonicity gate / refresh."""
    sign_base.require(PRIOR_AUDIT)
    prior_summary = sign_base.read_json(PRIOR_AUDIT)["summary"]

    gate_a = bool(
        prior_summary["exact_trial2_beta_sensitivity_transformed_operator_available_now"]
        and prior_summary["exact_trial2_beta_sensitivity_source_positive_on_canonical_window_now"]
        and prior_summary["exact_trial2_beta_sensitivity_potential_positive_on_canonical_window_now"]
    )
    gate_b = bool(
        prior_summary["exact_trial2_beta_sensitivity_principal_eigenvalue_sign_flip_available_now"]
        and prior_summary["exact_trial2_beta_sensitivity_maximum_principle_negative_closeout_available_now"]
        and not prior_summary[
            "exact_trial2_beta_sensitivity_maximum_principle_available_on_canonical_window_now"
        ]
    )
    gate_c = bool(
        gate_a
        and gate_b
        and prior_summary["updated_pack_trial2_beta_sensitivity_green_kernel_followup_required_now"]
    )

    trial2_beta_sensitivity_monotonicity_followup_lane_completed_now = bool(gate_b)
    trial2_beta_sensitivity_green_kernel_followup_promoted_now = bool(gate_c)
    conditional_hold_secondary_now = bool(gate_c)

    rows = [
        sign_base.row(
            "gate_a_trial2_beta_sensitivity_transformed_operator_available_now",
            "pass" if gate_a else "reject",
            "gate A Trial-2 beta-sensitivity transformed operator available now",
            sign_base.truth(gate_a),
            "The maximum-principle branch only becomes official once the transformed operator and its source/potential sign data are machine-readable.",
        ),
        sign_base.row(
            "gate_b_trial2_beta_sensitivity_maximum_principle_negative_closeout_now",
            "pass" if gate_b else "reject",
            "gate B Trial-2 beta-sensitivity maximum-principle negative closeout now",
            sign_base.truth(gate_b),
            "The classical maximum-principle path closes negatively once lambda_1 turns negative on the canonical window despite favorable source and potential signs.",
        ),
        sign_base.row(
            "gate_c_trial2_beta_sensitivity_green_kernel_followup_promoted_now",
            "pass" if gate_c else "reject",
            "gate C Trial-2 beta-sensitivity green-kernel followup promoted now",
            sign_base.truth(gate_c),
            "Once the maximum-principle path closes negatively, the honest strict-theorem followup is Green-kernel / resolvent sign control.",
        ),
        sign_base.row(
            "trial2_beta_sensitivity_monotonicity_followup_lane_completed_now",
            "pass" if trial2_beta_sensitivity_monotonicity_followup_lane_completed_now else "reject",
            "Trial-2 beta-sensitivity monotonicity followup lane completed now",
            sign_base.truth(trial2_beta_sensitivity_monotonicity_followup_lane_completed_now),
            "This lane is complete once the naive maximum-principle route is either proven or honestly closed; here it closes negatively.",
        ),
        sign_base.row(
            "trial2_beta_sensitivity_green_kernel_followup_promoted_now",
            "pass" if trial2_beta_sensitivity_green_kernel_followup_promoted_now else "reject",
            "Trial-2 beta-sensitivity green-kernel followup promoted now",
            sign_base.truth(trial2_beta_sensitivity_green_kernel_followup_promoted_now),
            "The next strict-theorem blocker is now the sign of the Green kernel / resolvent, not the naive maximum principle.",
        ),
        sign_base.row(
            "trial2_conditional_hold_secondary_now",
            "pass" if conditional_hold_secondary_now else "reject",
            "Trial-2 conditional hold secondary now",
            sign_base.truth(conditional_hold_secondary_now),
            "Conditional hold remains only as the fallback if the Green-kernel route dead-ends honestly.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "interaction_total_over_harmonic_sq_beta_common_root": float(
            prior_summary["interaction_total_over_harmonic_sq_beta_common_root"]
        ),
        "interaction_total_over_harmonic_sq_alpha_common_value": float(
            prior_summary["interaction_total_over_harmonic_sq_alpha_common_value"]
        ),
        "interaction_total_over_harmonic_sq_alpha_common_rel_error_vs_target": float(
            prior_summary["interaction_total_over_harmonic_sq_alpha_common_rel_error_vs_target"]
        ),
        "exact_trial2_beta_sensitivity_transformed_operator_available_now": bool(
            prior_summary["exact_trial2_beta_sensitivity_transformed_operator_available_now"]
        ),
        "exact_trial2_beta_sensitivity_maximum_principle_available_on_inner_window_now": bool(
            prior_summary["exact_trial2_beta_sensitivity_maximum_principle_available_on_inner_window_now"]
        ),
        "exact_trial2_beta_sensitivity_maximum_principle_available_on_canonical_window_now": bool(
            prior_summary["exact_trial2_beta_sensitivity_maximum_principle_available_on_canonical_window_now"]
        ),
        "exact_trial2_beta_sensitivity_maximum_principle_negative_closeout_available_now": bool(
            prior_summary["exact_trial2_beta_sensitivity_maximum_principle_negative_closeout_available_now"]
        ),
        "updated_pack_trial2_beta_sensitivity_green_kernel_followup_required_now": bool(
            prior_summary["updated_pack_trial2_beta_sensitivity_green_kernel_followup_required_now"]
        ),
        "trial2_beta_sensitivity_monotonicity_followup_lane_completed_now": (
            trial2_beta_sensitivity_monotonicity_followup_lane_completed_now
        ),
        "trial2_beta_sensitivity_green_kernel_followup_promoted_now": (
            trial2_beta_sensitivity_green_kernel_followup_promoted_now
        ),
        "trial2_conditional_hold_secondary_now": conditional_hold_secondary_now,
        "selected_primary_completion_lane": "trial2_beta_sensitivity_green_kernel_followup",
        "selected_secondary_completion_lane": "conditional_hold_only",
        "selected_reserve_completion_lane": "conditional_hold_only",
        "selected_next_generation_route": "trial2_beta_sensitivity_green_kernel_followup",
        "recommended_next_route_or_none": "trial2_beta_sensitivity_green_kernel_followup",
        "selected_followup_route": "trial2_beta_sensitivity_green_kernel_followup",
        "selected_followup_route_or_none": "trial2_beta_sensitivity_green_kernel_followup",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5653",
        STEP_NAME + " declaration gate",
        {
            "source_files": {"prior_audit": sign_base.display_path(PRIOR_AUDIT)},
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "trial2_beta_sensitivity_green_kernel_followup",
                "followup_route": "conditional_hold_only",
            },
        },
        rows,
        summary,
        {
            "overall_status": "trial2_beta_sensitivity_maximum_principle_gate_completed",
            "branch_completed": True,
            "breakthrough_passed_now": gate_b,
            "physical_reject_required": False,
        },
        {"formulae": build_formulae()},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(f"[done] {STEP_TAG} Trial-2 beta-sensitivity monotonicity gate completed")
    print(f"[done] declaration: {declaration_paths['json']}")


if __name__ == "__main__":
    main()
