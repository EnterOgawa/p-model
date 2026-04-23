#!/usr/bin/env python3
"""Generate 8.7.56.5163-.5166 blind-vector residual-origin verdict gate artifacts."""

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
        "8.7.56.5159-5162",
        "updated_pack_blind_vector_residual_origin_verdict_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5163-5166"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack blind-vector "
    "residual-origin verdict gate / route refresh"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_blind_vector_residual_origin_verdict_gate",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "blind_vector_residual_origin_selector_ambiguity_cleared_solver_"
    "deformation_required_primary_pack_refresh_secondary_gate"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "blind_vector_residual_origin_verdict_audited_solver_deformation_"
    "inventory_primary_hybrid_reserve_secondary_next"
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
    """Return formulas used in the blind-vector residual-origin verdict gate."""
    return {
        "gate_a": "Gate A = residual origin is not selector choice",
        "gate_b": "Gate B = solver-side deformation inventory promoted next",
        "gate_c": "Gate C = farther hybrid continuation reopen required now",
    }


# 関数: `.5163-.5166` を実行する。

def main() -> None:
    """Execute the blind-vector residual-origin verdict gate / route refresh."""
    sign_base.require(PRIOR_AUDIT)
    prior_summary = sign_base.read_json(PRIOR_AUDIT)["summary"]

    gate_a_updated_pack_blind_vector_residual_origin_not_selector_choice_available_now = bool(
        prior_summary[
            "exact_blind_vector_residual_origin_not_selector_choice_theorem_available_now"
        ]
        and prior_summary[
            "exact_blind_vector_selected_extension_wrong_sign_no_improvement_theorem_available_now"
        ]
    )
    gate_b_updated_pack_blind_vector_solver_side_deformation_inventory_promoted_next = bool(
        prior_summary[
            "updated_pack_blind_vector_solver_side_deformation_followup_required"
        ]
    )
    gate_c_farther_hybrid_continuation_reopen_required_now = bool(
        prior_summary["farther_hybrid_continuation_reopen_required_now"]
    )
    retry_mode = bool(prior_summary["retry_gate_computation_mode_selected"])
    non_surrogate_guard = bool(prior_summary["failure_matrix_non_surrogate_guard_preserved"])
    selected_extension_negative_closeout_available_now = bool(
        prior_summary["exact_blind_vector_selected_extension_negative_closeout_available_now"]
    )
    pack_update_required_now = bool(
        gate_b_updated_pack_blind_vector_solver_side_deformation_inventory_promoted_next
    )

    rows = [
        sign_base.row(
            "gate_a_updated_pack_blind_vector_residual_origin_not_selector_choice_available_now",
            "pass"
            if gate_a_updated_pack_blind_vector_residual_origin_not_selector_choice_available_now
            else "reject",
            "gate A updated-pack blind-vector residual origin not selector choice available now",
            sign_base.truth(
                gate_a_updated_pack_blind_vector_residual_origin_not_selector_choice_available_now
            ),
            "The selected-extension first shot now rules out selector ambiguity as the live residual-origin explanation.",
        ),
        sign_base.row(
            "gate_b_updated_pack_blind_vector_solver_side_deformation_inventory_promoted_next",
            "pass"
            if gate_b_updated_pack_blind_vector_solver_side_deformation_inventory_promoted_next
            else "reject",
            "gate B updated-pack blind-vector solver-side deformation inventory promoted next",
            sign_base.truth(
                gate_b_updated_pack_blind_vector_solver_side_deformation_inventory_promoted_next
            ),
            "Because the selected-extension first shot is only a retained replay, the honest next blocker is solver-side deformation inventory rather than final negative closeout.",
        ),
        sign_base.row(
            "gate_c_farther_hybrid_continuation_reopen_required_now",
            "pass" if gate_c_farther_hybrid_continuation_reopen_required_now else "reject",
            "gate C farther hybrid continuation reopen required now",
            sign_base.truth(gate_c_farther_hybrid_continuation_reopen_required_now),
            "Farther hybrid continuation remains reserve-only because the current blocker is still solver-side deformation under the fixed selected extension.",
        ),
        sign_base.row(
            "retry_gate_computation_mode_selected",
            "pass" if retry_mode else "reject",
            "retry gate computation mode selected",
            sign_base.truth(retry_mode),
            "This route refresh follows a substantive verdict reduction rather than another theorem-family replay.",
        ),
        sign_base.row(
            "failure_matrix_non_surrogate_guard_preserved",
            "pass" if non_surrogate_guard else "reject",
            "failure-matrix non-surrogate guard preserved",
            sign_base.truth(non_surrogate_guard),
            "Promoting the residual-origin verdict does not reopen exhausted surrogate or selector-family lanes.",
        ),
        sign_base.row(
            "exact_blind_vector_selected_extension_negative_closeout_available_now",
            "pass" if selected_extension_negative_closeout_available_now else "reject",
            "exact blind-vector selected-extension negative closeout available now",
            sign_base.truth(selected_extension_negative_closeout_available_now),
            "Negative closeout on the selected extension itself remains unavailable because the current result is still inherited replay, not a post-deformation recomputation.",
        ),
        sign_base.row(
            "pack_update_required_now",
            "pass" if pack_update_required_now else "reject",
            "updated-pack substantive pack update required now",
            sign_base.truth(pack_update_required_now),
            "A substantive lane shift happened here: selector ambiguity is cleared and the route now moves to solver-side deformation inventory.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "blind_F_at_q_theory": float(prior_summary["blind_F_at_q_theory"]),
        "blind_alpha_at_q_theory": float(prior_summary["blind_alpha_at_q_theory"]),
        "delta_alpha_sel_exact": float(prior_summary["delta_alpha_sel_exact"]),
        "relative_exact_residual": float(prior_summary["relative_exact_residual"]),
        "gate_a_updated_pack_blind_vector_residual_origin_not_selector_choice_available_now": gate_a_updated_pack_blind_vector_residual_origin_not_selector_choice_available_now,
        "gate_b_updated_pack_blind_vector_solver_side_deformation_inventory_promoted_next": gate_b_updated_pack_blind_vector_solver_side_deformation_inventory_promoted_next,
        "gate_c_farther_hybrid_continuation_reopen_required_now": gate_c_farther_hybrid_continuation_reopen_required_now,
        "retry_gate_computation_mode_selected": retry_mode,
        "failure_matrix_non_surrogate_guard_preserved": non_surrogate_guard,
        "exact_blind_vector_selected_extension_negative_closeout_available_now": selected_extension_negative_closeout_available_now,
        "pack_update_required_now": pack_update_required_now,
        "selected_primary_completion_lane": "updated_pack_blind_vector_solver_side_deformation_inventory_audit",
        "selected_secondary_completion_lane": "farther_hybrid_extra_q_range_only",
        "selected_reserve_completion_lane": "selected_extension_negative_closeout_only_after_solver_deformation_check",
        "selected_next_generation_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_blind_vector_solver_side_deformation_inventory_audit",
        "recommended_next_route_or_none": "8.7.56.5167",
        "selected_followup_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_blind_vector_solver_side_deformation_inventory_gate",
        "selected_followup_route_or_none": "8.7.56.5171",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5165",
        STEP_NAME + " declaration gate",
        {
            "source_files": {"prior_audit": sign_base.display_path(PRIOR_AUDIT)},
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "8.7.56.5167",
                "followup_route": "8.7.56.5171",
            },
        },
        rows,
        summary,
        {
            "overall_status": "vector_qball_form_factor_blind_vector_residual_origin_verdict_gate_declared",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {"formulae": build_formulae()},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(f"[done] {STEP_TAG} blind-vector residual-origin verdict gate completed")
    print(f"[done] declaration: {declaration_paths['json']}")


if __name__ == "__main__":
    main()
