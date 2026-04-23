#!/usr/bin/env python3
"""Generate 8.7.56.5635-.5638 Trial-2 strict-theorem gate artifacts."""

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
        "8.7.56.5631-5634",
        "updated_pack_trial2_target_free_common_root_strict_theorem_followup_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5635-5638"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack Trial-2 "
    "target-free common-root strict-theorem gate / conditional-hold refresh"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_trial2_target_free_common_root_strict_theorem_gate",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_target_free_common_root_strict_theorem_followup_audited_"
    "negative_closeout_gate_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_target_free_common_root_practical_closeout_completed_"
    "strict_theorem_negative_closeout_conditional_hold_only_next"
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
    """Return formulas used by the strict-theorem gate."""
    return {
        "gate_a": "Gate A = practical target-free common-root closeout remains valid now",
        "gate_b": "Gate B = strict theorem followup closes negatively now",
        "gate_c": "Gate C = conditional hold is restored as the only honest next state",
    }


# 関数: `.5635-.5638` を実行する。

def main() -> None:
    """Execute the Trial-2 strict-theorem gate / conditional-hold refresh."""
    sign_base.require(PRIOR_AUDIT)
    prior_summary = sign_base.read_json(PRIOR_AUDIT)["summary"]

    gate_a = bool(
        abs(float(prior_summary["interaction_total_over_harmonic_sq_alpha_common_rel_error_vs_target"]))
        <= 1.0e-3
    )
    gate_b = bool(
        prior_summary[
            "exact_trial2_target_free_common_root_strict_theorem_negative_closeout_available_now"
        ]
    )
    gate_c = bool(
        gate_b and prior_summary["updated_pack_trial2_conditional_hold_restored_primary_now"]
    )

    trial2_target_free_common_root_practical_closeout_completed_now = bool(gate_a and gate_b)
    trial2_target_free_common_root_strict_theorem_lane_completed_now = bool(gate_b)
    no_unconditional_next_official_branch_now = bool(
        gate_c and prior_summary["updated_pack_trial2_no_unconditional_next_official_branch_now"]
    )

    rows = [
        sign_base.row(
            "gate_a_trial2_target_free_common_root_practical_closeout_retained_now",
            "pass" if gate_a else "reject",
            "gate A Trial-2 target-free common-root practical closeout retained now",
            sign_base.truth(gate_a),
            "The common-root selector remains a practical closeout because the retained common alpha stays within one-per-mille of alpha_target.",
        ),
        sign_base.row(
            "gate_b_trial2_target_free_common_root_strict_theorem_negative_closeout_now",
            "pass" if gate_b else "reject",
            "gate B Trial-2 target-free common-root strict-theorem negative closeout now",
            sign_base.truth(gate_b),
            "The strict theorem lane closes negatively once only numerical transversality support is available while analytic uniqueness remains absent.",
        ),
        sign_base.row(
            "gate_c_trial2_conditional_hold_only_now",
            "pass" if gate_c else "reject",
            "gate C Trial-2 conditional hold only now",
            sign_base.truth(gate_c),
            "After the strict-theorem followup dead-ends honestly, no unconditional next official branch remains inside the current pack.",
        ),
        sign_base.row(
            "trial2_target_free_common_root_practical_closeout_completed_now",
            "pass" if trial2_target_free_common_root_practical_closeout_completed_now else "reject",
            "Trial-2 target-free common-root practical closeout completed now",
            sign_base.truth(trial2_target_free_common_root_practical_closeout_completed_now),
            "The pack now keeps one target-free practical direct-alpha closeout even though strict theorem closure is unavailable.",
        ),
        sign_base.row(
            "trial2_target_free_common_root_strict_theorem_lane_completed_now",
            "pass" if trial2_target_free_common_root_strict_theorem_lane_completed_now else "reject",
            "Trial-2 target-free common-root strict-theorem lane completed now",
            sign_base.truth(trial2_target_free_common_root_strict_theorem_lane_completed_now),
            "This lane is complete once the strict theorem followup is synchronized as a negative closeout.",
        ),
        sign_base.row(
            "trial2_no_unconditional_next_official_branch_now",
            "pass" if no_unconditional_next_official_branch_now else "reject",
            "Trial-2 no unconditional next official branch now",
            sign_base.truth(no_unconditional_next_official_branch_now),
            "Any further progress now requires a genuinely new theorem route or a genuinely new selected-extension-native source / computation branch.",
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
        "common_root_difference_derivative_min": float(
            prior_summary["common_root_difference_derivative_min"]
        ),
        "common_root_difference_derivative_max": float(
            prior_summary["common_root_difference_derivative_max"]
        ),
        "common_root_difference_derivative_rel_spread": float(
            prior_summary["common_root_difference_derivative_rel_spread"]
        ),
        "exact_trial2_target_free_common_root_local_transversality_support_available_now": bool(
            prior_summary[
                "exact_trial2_target_free_common_root_local_transversality_support_available_now"
            ]
        ),
        "exact_trial2_target_free_common_root_uniqueness_theorem_available_now": bool(
            prior_summary["exact_trial2_target_free_common_root_uniqueness_theorem_available_now"]
        ),
        "exact_trial2_target_free_common_root_strict_theorem_closeout_available_now": False,
        "exact_trial2_target_free_common_root_practical_closeout_completed_now": (
            trial2_target_free_common_root_practical_closeout_completed_now
        ),
        "exact_trial2_target_free_common_root_strict_theorem_lane_completed_now": (
            trial2_target_free_common_root_strict_theorem_lane_completed_now
        ),
        "trial2_no_unconditional_next_official_branch_now": (
            no_unconditional_next_official_branch_now
        ),
        "selected_primary_completion_lane": "conditional_hold_only",
        "selected_secondary_completion_lane": "conditional_hold_only",
        "selected_reserve_completion_lane": "conditional_hold_only",
        "selected_next_generation_route": "none",
        "recommended_next_route_or_none": None,
        "selected_followup_route": "conditional_hold_only",
        "selected_followup_route_or_none": None,
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5637",
        STEP_NAME + " declaration gate",
        {
            "source_files": {"prior_audit": sign_base.display_path(PRIOR_AUDIT)},
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": None,
                "followup_route": "conditional_hold_only",
            },
        },
        rows,
        summary,
        {
            "overall_status": "trial2_target_free_common_root_strict_theorem_gate_completed",
            "branch_completed": True,
            "breakthrough_passed_now": gate_a,
            "physical_reject_required": False,
        },
        {"formulae": build_formulae()},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(f"[done] {STEP_TAG} Trial-2 strict-theorem gate completed")
    print(f"[done] declaration: {declaration_paths['json']}")


if __name__ == "__main__":
    main()
