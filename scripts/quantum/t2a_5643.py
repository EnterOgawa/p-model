#!/usr/bin/env python3
"""Generate 8.7.56.5643-.5646 Trial-2 beta-sensitivity gate artifacts."""

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
        "8.7.56.5639-5642",
        "updated_pack_trial2_beta_sensitivity_equation_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5643-5646"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack Trial-2 "
    "beta-sensitivity gate / monotonicity refresh"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_trial2_beta_sensitivity_equation_gate",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_beta_sensitivity_equation_audited_sign_support_gate_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_beta_sensitivity_equation_audited_sign_support_monotonicity_"
    "followup_primary_conditional_hold_secondary_next"
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
    """Return formulas used by the beta-sensitivity gate."""
    return {
        "gate_a": "Gate A = beta-sensitivity sign-support surface is available now",
        "gate_b": "Gate B = practical common-root direct-alpha closeout remains valid now",
        "gate_c": "Gate C = monotonicity followup is promoted while conditional hold stays secondary",
    }


# 関数: `.5643-.5646` を実行する。

def main() -> None:
    """Execute the Trial-2 beta-sensitivity gate / monotonicity refresh."""
    sign_base.require(PRIOR_AUDIT)
    prior_summary = sign_base.read_json(PRIOR_AUDIT)["summary"]

    gate_a = bool(
        prior_summary["exact_trial2_beta_sensitivity_local_support_available_now"]
        and prior_summary["exact_trial2_beta_sensitivity_u_beta_negative_support_available_now"]
        and prior_summary["exact_trial2_beta_sensitivity_alpha_qstar_derivative_positive_now"]
        and prior_summary["exact_trial2_beta_sensitivity_alpha_r8_derivative_negative_now"]
    )
    gate_b = bool(
        abs(float(prior_summary["interaction_total_over_harmonic_sq_alpha_common_rel_error_vs_target"]))
        <= 1.0e-3
    )
    gate_c = bool(
        gate_a
        and gate_b
        and prior_summary["updated_pack_trial2_beta_sensitivity_monotonicity_followup_required_now"]
    )

    trial2_beta_sensitivity_equation_lane_completed_now = bool(gate_a)
    trial2_beta_sensitivity_monotonicity_followup_promoted_now = bool(gate_c)
    conditional_hold_secondary_now = bool(gate_c)

    rows = [
        sign_base.row(
            "gate_a_trial2_beta_sensitivity_sign_support_available_now",
            "pass" if gate_a else "reject",
            "gate A Trial-2 beta-sensitivity sign-support available now",
            sign_base.truth(gate_a),
            "The beta-sensitivity route becomes live only after u_beta sign support and local linearized-equation support are both available.",
        ),
        sign_base.row(
            "gate_b_trial2_common_root_practical_closeout_retained_now",
            "pass" if gate_b else "reject",
            "gate B Trial-2 common-root practical closeout retained now",
            sign_base.truth(gate_b),
            "The route keeps the practical direct-alpha closeout while trying to promote one strict theorem on top of it.",
        ),
        sign_base.row(
            "gate_c_trial2_beta_sensitivity_monotonicity_followup_promoted_now",
            "pass" if gate_c else "reject",
            "gate C Trial-2 beta-sensitivity monotonicity followup promoted now",
            sign_base.truth(gate_c),
            "Once the operator and sign-support surfaces exist, the honest blocker becomes the monotonicity / uniqueness theorem itself.",
        ),
        sign_base.row(
            "trial2_beta_sensitivity_equation_lane_completed_now",
            "pass" if trial2_beta_sensitivity_equation_lane_completed_now else "reject",
            "Trial-2 beta-sensitivity equation lane completed now",
            sign_base.truth(trial2_beta_sensitivity_equation_lane_completed_now),
            "This lane is complete once the exact equation and local sign-support surface are machine-readable.",
        ),
        sign_base.row(
            "trial2_beta_sensitivity_monotonicity_followup_promoted_now",
            "pass" if trial2_beta_sensitivity_monotonicity_followup_promoted_now else "reject",
            "Trial-2 beta-sensitivity monotonicity followup promoted now",
            sign_base.truth(trial2_beta_sensitivity_monotonicity_followup_promoted_now),
            "The next honest branch is now the strict monotonicity / uniqueness theorem built on the beta-sensitivity equation.",
        ),
        sign_base.row(
            "trial2_conditional_hold_secondary_now",
            "pass" if conditional_hold_secondary_now else "reject",
            "Trial-2 conditional hold secondary now",
            sign_base.truth(conditional_hold_secondary_now),
            "Conditional hold remains only as the fallback if the beta-sensitivity monotonicity followup dead-ends honestly.",
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
        "exact_trial2_beta_sensitivity_equation_available_now": bool(
            prior_summary["exact_trial2_beta_sensitivity_equation_available_now"]
        ),
        "exact_trial2_beta_sensitivity_local_support_available_now": bool(
            prior_summary["exact_trial2_beta_sensitivity_local_support_available_now"]
        ),
        "exact_trial2_beta_sensitivity_u_beta_negative_support_available_now": bool(
            prior_summary["exact_trial2_beta_sensitivity_u_beta_negative_support_available_now"]
        ),
        "exact_trial2_beta_sensitivity_alpha_qstar_derivative_positive_now": bool(
            prior_summary["exact_trial2_beta_sensitivity_alpha_qstar_derivative_positive_now"]
        ),
        "exact_trial2_beta_sensitivity_alpha_r8_derivative_negative_now": bool(
            prior_summary["exact_trial2_beta_sensitivity_alpha_r8_derivative_negative_now"]
        ),
        "exact_trial2_common_root_monotonicity_theorem_available_now": False,
        "trial2_beta_sensitivity_equation_lane_completed_now": (
            trial2_beta_sensitivity_equation_lane_completed_now
        ),
        "trial2_beta_sensitivity_monotonicity_followup_promoted_now": (
            trial2_beta_sensitivity_monotonicity_followup_promoted_now
        ),
        "trial2_conditional_hold_secondary_now": conditional_hold_secondary_now,
        "selected_primary_completion_lane": "trial2_beta_sensitivity_monotonicity_followup",
        "selected_secondary_completion_lane": "conditional_hold_only",
        "selected_reserve_completion_lane": "conditional_hold_only",
        "selected_next_generation_route": "trial2_beta_sensitivity_monotonicity_followup",
        "recommended_next_route_or_none": "trial2_beta_sensitivity_monotonicity_followup",
        "selected_followup_route": "trial2_beta_sensitivity_monotonicity_followup",
        "selected_followup_route_or_none": "trial2_beta_sensitivity_monotonicity_followup",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5645",
        STEP_NAME + " declaration gate",
        {
            "source_files": {"prior_audit": sign_base.display_path(PRIOR_AUDIT)},
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "trial2_beta_sensitivity_monotonicity_followup",
                "followup_route": "conditional_hold_only",
            },
        },
        rows,
        summary,
        {
            "overall_status": "trial2_beta_sensitivity_gate_completed",
            "branch_completed": True,
            "breakthrough_passed_now": gate_a,
            "physical_reject_required": False,
        },
        {"formulae": build_formulae()},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(f"[done] {STEP_TAG} Trial-2 beta-sensitivity gate completed")
    print(f"[done] declaration: {declaration_paths['json']}")


if __name__ == "__main__":
    main()
