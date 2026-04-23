#!/usr/bin/env python3
"""Generate 8.7.56.5259-.5262 selected-extension solver-recompute residual-origin refresh gate artifacts."""

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
        "8.7.56.5255-5258",
        "updated_pack_selected_extension_solver_recompute_residual_origin_refresh_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5259-5262"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack selected-"
    "extension solver-recompute residual-origin refresh gate / route refresh"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_selected_extension_solver_recompute_residual_origin_refresh_gate",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "selected_extension_solver_recompute_retained_q_rerun_preserves_phase3_"
    "failure_theorem_derived_solver_deformation_required_primary_pack_refresh_"
    "secondary_gate"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "selected_extension_solver_recompute_retained_q_rerun_preserves_phase3_"
    "failure_closeout_completed_solver_deformation_inventory_primary_hybrid_"
    "reserve_secondary_next"
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
    """Return formulas used in the selected-extension solver-recompute residual-origin refresh gate."""
    return {
        "gate_a": "Gate A = selected-extension solver-recompute lane negative closeout available now",
        "gate_b": "Gate B = selected-extension solver-side deformation inventory promoted next",
        "gate_c": "Gate C = farther hybrid continuation reopen required now",
    }


# 関数: `.5259-.5262` を実行する。

def main() -> None:
    """Execute the selected-extension solver-recompute residual-origin refresh gate / route refresh."""
    sign_base.require(PRIOR_AUDIT)
    prior_summary = sign_base.read_json(PRIOR_AUDIT)["summary"]

    gate_a = bool(
        prior_summary[
            "exact_selected_extension_solver_recompute_rerun_preserves_phase3_failure_theorem_available_now"
        ]
        and prior_summary[
            "exact_selected_extension_solver_recompute_helper_not_residual_origin_theorem_available_now"
        ]
        and prior_summary[
            "exact_selected_extension_solver_side_deformation_lane_required_theorem_available_now"
        ]
        and prior_summary[
            "exact_selected_extension_solver_recompute_lane_negative_closeout_available_now"
        ]
    )
    gate_b = bool(
        prior_summary[
            "updated_pack_selected_extension_solver_side_deformation_followup_required"
        ]
    )
    gate_c = bool(prior_summary["farther_hybrid_continuation_reopen_required_now"])
    retry_mode = bool(prior_summary["retry_gate_computation_mode_selected"])
    non_surrogate_guard = bool(prior_summary["failure_matrix_non_surrogate_guard_preserved"])
    same_schema_replay_detected = bool(
        prior_summary[
            "updated_pack_same_schema_selected_extension_solver_recompute_residual_refresh_replay_detected_now"
        ]
    )
    pack_update_required_now = bool(gate_b)

    rows = [
        sign_base.row(
            "gate_a_updated_pack_exact_selected_extension_solver_recompute_lane_negative_closeout_available_now",
            "pass" if gate_a else "reject",
            "gate A updated-pack exact selected-extension solver-recompute lane negative closeout available now",
            sign_base.truth(gate_a),
            "The selected-extension solver-recompute lane now closes negatively: retained-q rerun preserves the old failed surface and does not resolve residual origin.",
        ),
        sign_base.row(
            "gate_b_updated_pack_selected_extension_solver_side_deformation_inventory_promoted_next",
            "pass" if gate_b else "reject",
            "gate B updated-pack selected-extension solver-side deformation inventory promoted next",
            sign_base.truth(gate_b),
            "Because the retained-q recompute surface preserves the old failure, the honest next blocker is solver-side deformation inventory rather than more helper-side replay.",
        ),
        sign_base.row(
            "gate_c_farther_hybrid_continuation_reopen_required_now",
            "pass" if gate_c else "reject",
            "gate C farther hybrid continuation reopen required now",
            sign_base.truth(gate_c),
            "Farther hybrid continuation remains reserve-only because the current blocker has moved to solver-side deformation inventory under the fixed selected extension.",
        ),
        sign_base.row(
            "retry_gate_computation_mode_selected",
            "pass" if retry_mode else "reject",
            "retry gate computation mode selected",
            sign_base.truth(retry_mode),
            "This route refresh follows a substantive negative closeout rather than another theorem-family replay.",
        ),
        sign_base.row(
            "failure_matrix_non_surrogate_guard_preserved",
            "pass" if non_surrogate_guard else "reject",
            "failure-matrix non-surrogate guard preserved",
            sign_base.truth(non_surrogate_guard),
            "Promoting the recompute negative closeout does not reopen exhausted surrogate or selector-family lanes.",
        ),
        sign_base.row(
            "updated_pack_same_schema_selected_extension_solver_recompute_residual_refresh_replay_detected_now",
            "pass" if same_schema_replay_detected else "reject",
            "updated-pack same-schema selected-extension solver-recompute residual refresh replay detected now",
            sign_base.truth(same_schema_replay_detected),
            "False means the live blocker has genuinely moved from retained-q rerun interpretation to a new solver-side deformation lane.",
        ),
        sign_base.row(
            "pack_update_required_now",
            "pass" if pack_update_required_now else "reject",
            "updated-pack substantive pack update required now",
            sign_base.truth(pack_update_required_now),
            "A substantive lane shift happened here: selected-extension solver-recompute is closed negatively and solver-side deformation inventory is next.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "selected_extension_label": prior_summary["selected_extension_label"],
        "q_theory_over_m0": float(prior_summary["q_theory_over_m0"]),
        "blind_F_recomp_at_q_theory": float(prior_summary["blind_F_recomp_at_q_theory"]),
        "blind_alpha_recomp_at_q_theory": float(
            prior_summary["blind_alpha_recomp_at_q_theory"]
        ),
        "delta_alpha_sel_recomp_exact": float(
            prior_summary["delta_alpha_sel_recomp_exact"]
        ),
        "relative_exact_residual_recomp": float(
            prior_summary["relative_exact_residual_recomp"]
        ),
        "gate_a_updated_pack_exact_selected_extension_solver_recompute_lane_negative_closeout_available_now": gate_a,
        "gate_b_updated_pack_selected_extension_solver_side_deformation_inventory_promoted_next": gate_b,
        "gate_c_farther_hybrid_continuation_reopen_required_now": gate_c,
        "retry_gate_computation_mode_selected": retry_mode,
        "failure_matrix_non_surrogate_guard_preserved": non_surrogate_guard,
        "updated_pack_same_schema_selected_extension_solver_recompute_residual_refresh_replay_detected_now": same_schema_replay_detected,
        "pack_update_required_now": pack_update_required_now,
        "selected_primary_completion_lane": "updated_pack_selected_extension_solver_side_deformation_inventory_audit",
        "selected_secondary_completion_lane": "farther_hybrid_extra_q_range_only",
        "selected_reserve_completion_lane": "selected_extension_solver_recompute_lane_closed_negative",
        "selected_next_generation_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_selected_extension_solver_side_deformation_inventory_audit",
        "recommended_next_route_or_none": "8.7.56.5263",
        "selected_followup_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_selected_extension_solver_side_deformation_inventory_gate",
        "selected_followup_route_or_none": "8.7.56.5267",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5261",
        STEP_NAME + " declaration gate",
        {
            "source_files": {"prior_audit": sign_base.display_path(PRIOR_AUDIT)},
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "8.7.56.5263",
                "followup_route": "8.7.56.5267",
            },
        },
        rows,
        summary,
        {
            "overall_status": "vector_qball_form_factor_selected_extension_solver_recompute_residual_origin_refresh_gate_declared",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {"formulae": build_formulae()},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(f"[done] {STEP_TAG} selected-extension solver-recompute residual-origin refresh gate completed")
    print(f"[done] declaration: {declaration_paths['json']}")


if __name__ == "__main__":
    main()
