#!/usr/bin/env python3
"""Generate 8.7.56.5187-.5190 blind-vector solver-side numeric rerun gate artifacts."""

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
        "8.7.56.5183-5186",
        "updated_pack_blind_vector_solver_side_numeric_rerun_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5187-5190"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack blind-vector "
    "solver-side numeric rerun gate / residual-origin refresh"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_blind_vector_solver_side_numeric_rerun_gate",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "blind_vector_solver_side_numeric_rerun_backend_gap_audited_backend_"
    "inventory_primary_hybrid_reserve_secondary_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "blind_vector_solver_side_numeric_rerun_backend_gap_audited_backend_"
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


# 関数: numeric-rerun gate の式を返す。
def build_formulae() -> dict[str, str]:
    """Return formulas used in the blind-vector solver-side numeric rerun gate."""
    return {
        "gate_a": "Gate A = numeric rerun backend gap theorem available now",
        "gate_b": "Gate B = solver-side backend inventory promoted next",
        "gate_c": "Gate C = farther hybrid continuation reopen required now",
    }


# 関数: `.5187-.5190` を実行する。
def main() -> None:
    """Execute the blind-vector solver-side numeric rerun gate / residual-origin refresh."""
    sign_base.require(PRIOR_AUDIT)
    prior_summary = sign_base.read_json(PRIOR_AUDIT)["summary"]

    gate_a = bool(
        prior_summary[
            "exact_blind_vector_solver_side_numeric_rerun_backend_gap_theorem_available_now"
        ]
        and prior_summary[
            "exact_blind_vector_solver_side_effective_kernel_backend_requirement_theorem_available_now"
        ]
        and prior_summary[
            "exact_blind_vector_solver_side_internal_resolvent_backend_requirement_theorem_available_now"
        ]
        and prior_summary[
            "exact_blind_vector_solver_side_retained_q_numeric_backend_requirement_theorem_available_now"
        ]
    )
    gate_b = bool(
        prior_summary["updated_pack_blind_vector_solver_side_backend_inventory_followup_required"]
    )
    gate_c = bool(prior_summary["farther_hybrid_continuation_reopen_required_now"])
    retry_mode = bool(prior_summary["retry_gate_computation_mode_selected"])
    non_surrogate_guard = bool(
        prior_summary["failure_matrix_non_surrogate_guard_preserved"]
    )
    same_schema_replay_detected = bool(
        prior_summary["updated_pack_same_schema_blind_vector_numeric_rerun_replay_detected_now"]
    )
    actual_rerun_available = bool(
        prior_summary["actual_blind_vector_solver_side_numeric_rerun_available_now"]
    )
    pack_update_required_now = bool(gate_b)

    rows = [
        sign_base.row(
            "gate_a_updated_pack_exact_blind_vector_solver_side_numeric_rerun_backend_gap_available_now",
            "pass" if gate_a else "reject",
            "gate A updated-pack exact blind-vector solver-side numeric rerun backend gap available now",
            sign_base.truth(gate_a),
            "The live blind-vector blocker is now explicit: selected extension and rerun contract are fixed, but backend integration is missing.",
        ),
        sign_base.row(
            "gate_b_updated_pack_blind_vector_solver_side_backend_inventory_promoted_next",
            "pass" if gate_b else "reject",
            "gate B updated-pack blind-vector solver-side backend inventory promoted next",
            sign_base.truth(gate_b),
            "The honest next blocker is backend inventory/integration, not another theorem-family replay or farther-hybrid reopen.",
        ),
        sign_base.row(
            "gate_c_farther_hybrid_continuation_reopen_required_now",
            "pass" if gate_c else "reject",
            "gate C farther hybrid continuation reopen required now",
            sign_base.truth(gate_c),
            "Farther hybrid continuation stays reserve-only while retained-q rerun is blocked upstream by missing backend integration.",
        ),
        sign_base.row(
            "retry_gate_computation_mode_selected",
            "pass" if retry_mode else "reject",
            "retry gate computation mode selected",
            sign_base.truth(retry_mode),
            "This route refresh follows computation-side blocker reduction rather than theorem-family recursion.",
        ),
        sign_base.row(
            "failure_matrix_non_surrogate_guard_preserved",
            "pass" if non_surrogate_guard else "reject",
            "failure-matrix non-surrogate guard preserved",
            sign_base.truth(non_surrogate_guard),
            "Promoting backend inventory does not reopen exhausted surrogate or selector-choice branches.",
        ),
        sign_base.row(
            "actual_blind_vector_solver_side_numeric_rerun_available_now",
            "pass" if actual_rerun_available else "reject",
            "actual blind-vector solver-side numeric rerun available now",
            sign_base.truth(actual_rerun_available),
            "False means the branch still lacks concrete recomputed kernel/resolvent outputs on the fixed selected extension.",
        ),
        sign_base.row(
            "updated_pack_same_schema_blind_vector_numeric_rerun_replay_detected_now",
            "pass" if same_schema_replay_detected else "reject",
            "updated-pack same-schema blind-vector numeric-rerun replay detected now",
            sign_base.truth(same_schema_replay_detected),
            "False means this branch compressed the live blocker to backend inventory instead of replaying the already fixed rerun contract schema.",
        ),
        sign_base.row(
            "pack_update_required_now",
            "pass" if pack_update_required_now else "reject",
            "updated-pack substantive pack update required now",
            sign_base.truth(pack_update_required_now),
            "A substantive lane shift happened here: the next official blocker is solver-side backend inventory on the fixed selected extension.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "q_theory_over_m0": float(prior_summary["q_theory_over_m0"]),
        "blind_F_at_q_theory": float(prior_summary["blind_F_at_q_theory"]),
        "blind_alpha_at_q_theory": float(prior_summary["blind_alpha_at_q_theory"]),
        "alpha_exact_at_q_theory": float(prior_summary["alpha_exact_at_q_theory"]),
        "gate_a_updated_pack_exact_blind_vector_solver_side_numeric_rerun_backend_gap_available_now": gate_a,
        "gate_b_updated_pack_blind_vector_solver_side_backend_inventory_promoted_next": gate_b,
        "gate_c_farther_hybrid_continuation_reopen_required_now": gate_c,
        "retry_gate_computation_mode_selected": retry_mode,
        "failure_matrix_non_surrogate_guard_preserved": non_surrogate_guard,
        "actual_blind_vector_solver_side_numeric_rerun_available_now": actual_rerun_available,
        "updated_pack_same_schema_blind_vector_numeric_rerun_replay_detected_now": same_schema_replay_detected,
        "pack_update_required_now": pack_update_required_now,
        "selected_primary_completion_lane": "updated_pack_blind_vector_solver_side_backend_inventory_audit",
        "selected_secondary_completion_lane": "updated_pack_blind_vector_residual_origin_refresh_after_backend_integration",
        "selected_reserve_completion_lane": "farther_hybrid_extra_q_range_only",
        "selected_next_generation_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_blind_vector_solver_side_backend_inventory_audit",
        "recommended_next_route_or_none": "8.7.56.5191",
        "selected_followup_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_blind_vector_solver_side_backend_inventory_gate",
        "selected_followup_route_or_none": "8.7.56.5195",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5189",
        STEP_NAME + " declaration gate",
        {
            "source_files": {"prior_audit": sign_base.display_path(PRIOR_AUDIT)},
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "8.7.56.5191",
                "followup_route": "8.7.56.5195",
            },
        },
        rows,
        summary,
        {
            "overall_status": "vector_qball_form_factor_blind_vector_solver_side_numeric_rerun_gate_declared",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {"formulae": build_formulae()},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(f"[done] {STEP_TAG} blind-vector solver-side numeric rerun gate completed")
    print(f"[done] declaration: {declaration_paths['json']}")


if __name__ == "__main__":
    main()
