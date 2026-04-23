#!/usr/bin/env python3
"""Generate 8.7.56.5267-.5270 selected-extension solver-side deformation inventory gate artifacts."""

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
        "8.7.56.5263-5266",
        "updated_pack_selected_extension_solver_side_deformation_inventory_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5267-5270"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack selected-"
    "extension solver-side deformation inventory gate / route refresh"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_selected_extension_solver_side_deformation_inventory_gate",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "selected_extension_solver_side_deformation_inventory_nonempty_theorem_"
    "derived_front_runner_primary_pack_refresh_secondary_gate"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "selected_extension_solver_side_deformation_inventory_audited_front_runner_"
    "primary_hybrid_reserve_secondary_next"
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


# 関数: solver-side deformation inventory gate の式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used in the selected-extension solver-side deformation inventory gate."""
    return {
        "gate_a": "Gate A = selected-extension solver-side deformation inventory nonempty theorem available now",
        "gate_b": "Gate B = selected-extension solver-side deformation front-runner promoted next",
        "gate_c": "Gate C = farther hybrid continuation reopen required now",
    }


# 関数: `.5267-.5270` を実行する。

def main() -> None:
    """Execute the selected-extension solver-side deformation inventory gate / route refresh."""
    sign_base.require(PRIOR_AUDIT)
    prior_summary = sign_base.read_json(PRIOR_AUDIT)["summary"]

    gate_a = bool(
        prior_summary[
            "exact_selected_extension_solver_side_deformation_inventory_nonempty_theorem_available_now"
        ]
        and prior_summary[
            "exact_selected_extension_solver_side_deformation_front_runner_compatibility_theorem_available_now"
        ]
    )
    gate_b = bool(
        prior_summary[
            "updated_pack_selected_extension_solver_side_deformation_front_runner_followup_required"
        ]
    )
    gate_c = bool(prior_summary["farther_hybrid_continuation_reopen_required_now"])
    retry_mode = bool(prior_summary["retry_gate_computation_mode_selected"])
    non_surrogate_guard = bool(
        prior_summary["failure_matrix_non_surrogate_guard_preserved"]
    )
    same_schema_replay_detected = bool(
        prior_summary[
            "updated_pack_same_schema_selected_extension_solver_side_deformation_inventory_replay_detected_now"
        ]
    )
    pack_update_required_now = bool(gate_b)

    rows = [
        sign_base.row(
            "gate_a_updated_pack_exact_selected_extension_solver_side_deformation_inventory_nonempty_available_now",
            "pass" if gate_a else "reject",
            "gate A updated-pack exact selected-extension solver-side deformation inventory nonempty available now",
            sign_base.truth(gate_a),
            "The active selected-extension lane now has a finite solver-side deformation inventory compatible with the already-closed recompute negative verdict.",
        ),
        sign_base.row(
            "gate_b_updated_pack_selected_extension_solver_side_deformation_front_runner_promoted_next",
            "pass" if gate_b else "reject",
            "gate B updated-pack selected-extension solver-side deformation front-runner promoted next",
            sign_base.truth(gate_b),
            "The honest next blocker is no longer generic deformation inventory, but candidate-specific audit of the retained-q solver-side deformation front-runner.",
        ),
        sign_base.row(
            "gate_c_farther_hybrid_continuation_reopen_required_now",
            "pass" if gate_c else "reject",
            "gate C farther hybrid continuation reopen required now",
            sign_base.truth(gate_c),
            "Farther hybrid continuation stays reserve-only because retained-q deformation must be exhausted before extra-q reopen can be honest.",
        ),
        sign_base.row(
            "retry_gate_computation_mode_selected",
            "pass" if retry_mode else "reject",
            "retry gate computation mode selected",
            sign_base.truth(retry_mode),
            "This route refresh follows a substantive computation-side blocker reduction rather than theorem-family or replay recursion.",
        ),
        sign_base.row(
            "failure_matrix_non_surrogate_guard_preserved",
            "pass" if non_surrogate_guard else "reject",
            "failure-matrix non-surrogate guard preserved",
            sign_base.truth(non_surrogate_guard),
            "Promoting the solver-side deformation front-runner does not reopen exhausted surrogate or selector-choice routes.",
        ),
        sign_base.row(
            "updated_pack_same_schema_selected_extension_solver_side_deformation_inventory_replay_detected_now",
            "pass" if same_schema_replay_detected else "reject",
            "updated-pack same-schema selected-extension solver-side deformation inventory replay detected now",
            sign_base.truth(same_schema_replay_detected),
            "False means the inventory layer genuinely compressed the blocker to one front-runner instead of replaying the already-closed recompute interpretation.",
        ),
        sign_base.row(
            "pack_update_required_now",
            "pass" if pack_update_required_now else "reject",
            "updated-pack substantive pack update required now",
            sign_base.truth(pack_update_required_now),
            "A substantive lane shift happened here: the next official blocker is front-runner deformation audit on the fixed selected extension.",
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
        "gate_a_updated_pack_exact_selected_extension_solver_side_deformation_inventory_nonempty_available_now": gate_a,
        "gate_b_updated_pack_selected_extension_solver_side_deformation_front_runner_promoted_next": gate_b,
        "gate_c_farther_hybrid_continuation_reopen_required_now": gate_c,
        "retry_gate_computation_mode_selected": retry_mode,
        "failure_matrix_non_surrogate_guard_preserved": non_surrogate_guard,
        "updated_pack_same_schema_selected_extension_solver_side_deformation_inventory_replay_detected_now": same_schema_replay_detected,
        "pack_update_required_now": pack_update_required_now,
        "selected_primary_completion_lane": "updated_pack_selected_extension_solver_side_deformation_front_runner_audit",
        "selected_secondary_completion_lane": "updated_pack_selected_extension_solver_side_deformation_front_runner_gate",
        "selected_reserve_completion_lane": "farther_hybrid_extra_q_range_only",
        "selected_next_generation_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_selected_extension_solver_side_deformation_front_runner_audit",
        "recommended_next_route_or_none": "8.7.56.5271",
        "selected_followup_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_selected_extension_solver_side_deformation_front_runner_gate",
        "selected_followup_route_or_none": "8.7.56.5275",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5269",
        STEP_NAME + " declaration gate",
        {
            "source_files": {"prior_audit": sign_base.display_path(PRIOR_AUDIT)},
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "8.7.56.5271",
                "followup_route": "8.7.56.5275",
            },
        },
        rows,
        summary,
        {
            "overall_status": "vector_qball_form_factor_selected_extension_solver_side_deformation_inventory_gate_declared",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {"formulae": build_formulae()},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(
        "[done] 8.7.56.5267-5270 selected-extension solver-side deformation inventory gate completed"
    )
    print(f"[done] declaration: {declaration_paths['json']}")


# 関数: CLI entrypoint から gate を実行する。

if __name__ == "__main__":
    main()
