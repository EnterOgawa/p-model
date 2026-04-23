#!/usr/bin/env python3
"""Generate 8.7.56.5203-.5206 blind-vector backend-adapter gate artifacts."""

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
        "8.7.56.5199-5202",
        "updated_pack_blind_vector_solver_side_backend_adapter_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5203-5206"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack blind-vector "
    "solver-side backend adapter gate / route refresh"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_blind_vector_solver_side_backend_adapter_gate",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "blind_vector_solver_side_backend_adapter_contract_audited_implementation_"
    "primary_hybrid_reserve_secondary_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "blind_vector_solver_side_backend_adapter_contract_audited_implementation_"
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


# 関数: backend-adapter gate の式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used in the backend-adapter gate."""
    return {
        "gate_a": "Gate A = backend adapter contract available now",
        "gate_b": "Gate B = blind-vector backend implementation promoted next",
        "gate_c": "Gate C = farther hybrid continuation reopen required now",
    }


# 関数: `.5203-.5206` を実行する。

def main() -> None:
    """Execute the blind-vector backend-adapter gate / route refresh."""
    sign_base.require(PRIOR_AUDIT)
    prior_summary = sign_base.read_json(PRIOR_AUDIT)["summary"]

    gate_a = bool(
        prior_summary[
            "exact_blind_vector_solver_side_backend_adapter_input_pack_formula_available_now"
        ]
        and prior_summary[
            "exact_blind_vector_solver_side_backend_adapter_output_pack_formula_available_now"
        ]
        and prior_summary[
            "exact_blind_vector_solver_side_backend_adapter_contract_formula_available_now"
        ]
        and prior_summary[
            "exact_blind_vector_solver_side_backend_adapter_front_runner_formula_available_now"
        ]
        and prior_summary[
            "exact_blind_vector_solver_side_backend_adapter_compatibility_theorem_available_now"
        ]
    )
    gate_b = bool(
        prior_summary[
            "updated_pack_blind_vector_solver_side_backend_implementation_followup_required"
        ]
    )
    gate_c = bool(prior_summary["farther_hybrid_continuation_reopen_required_now"])
    retry_mode = bool(prior_summary["retry_gate_computation_mode_selected"])
    non_surrogate_guard = bool(
        prior_summary["failure_matrix_non_surrogate_guard_preserved"]
    )
    same_schema_replay_detected = bool(
        prior_summary["updated_pack_same_schema_blind_vector_backend_adapter_replay_detected_now"]
    )
    pack_update_required_now = bool(gate_b)

    rows = [
        sign_base.row(
            "gate_a_updated_pack_exact_blind_vector_solver_side_backend_adapter_contract_available_now",
            "pass" if gate_a else "reject",
            "gate A updated-pack exact blind-vector solver-side backend adapter contract available now",
            sign_base.truth(gate_a),
            "The live blind-vector blocker is no longer generic backend wiring because one concrete selected-extension adapter contract is now official.",
        ),
        sign_base.row(
            "gate_b_updated_pack_blind_vector_solver_side_backend_implementation_promoted_next",
            "pass" if gate_b else "reject",
            "gate B updated-pack blind-vector solver-side backend implementation promoted next",
            sign_base.truth(gate_b),
            "The honest next blocker is actual implementation of the front-runner adapter contract on the fixed selected extension.",
        ),
        sign_base.row(
            "gate_c_farther_hybrid_continuation_reopen_required_now",
            "pass" if gate_c else "reject",
            "gate C farther hybrid continuation reopen required now",
            sign_base.truth(gate_c),
            "Farther hybrid continuation stays reserve-only while retained-q rerun still waits on concrete backend implementation.",
        ),
        sign_base.row(
            "retry_gate_computation_mode_selected",
            "pass" if retry_mode else "reject",
            "retry gate computation mode selected",
            sign_base.truth(retry_mode),
            "This route refresh stays on computation-side blocker reduction rather than theorem-family replay.",
        ),
        sign_base.row(
            "failure_matrix_non_surrogate_guard_preserved",
            "pass" if non_surrogate_guard else "reject",
            "failure-matrix non-surrogate guard preserved",
            sign_base.truth(non_surrogate_guard),
            "Promoting backend implementation does not reopen exhausted surrogate or selector-choice branches.",
        ),
        sign_base.row(
            "updated_pack_same_schema_blind_vector_backend_adapter_replay_detected_now",
            "pass" if same_schema_replay_detected else "reject",
            "updated-pack same-schema blind-vector backend adapter replay detected now",
            sign_base.truth(same_schema_replay_detected),
            "False means this branch compressed the live blocker to actual implementation instead of replaying adapter-family declarations.",
        ),
        sign_base.row(
            "pack_update_required_now",
            "pass" if pack_update_required_now else "reject",
            "updated-pack substantive pack update required now",
            sign_base.truth(pack_update_required_now),
            "A substantive lane shift happened here: the next official blocker is backend implementation on the fixed selected extension.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "q_theory_over_m0": float(prior_summary["q_theory_over_m0"]),
        "blind_F_at_q_theory": float(prior_summary["blind_F_at_q_theory"]),
        "blind_alpha_at_q_theory": float(prior_summary["blind_alpha_at_q_theory"]),
        "alpha_exact_at_q_theory": float(prior_summary["alpha_exact_at_q_theory"]),
        "gate_a_updated_pack_exact_blind_vector_solver_side_backend_adapter_contract_available_now": gate_a,
        "gate_b_updated_pack_blind_vector_solver_side_backend_implementation_promoted_next": gate_b,
        "gate_c_farther_hybrid_continuation_reopen_required_now": gate_c,
        "retry_gate_computation_mode_selected": retry_mode,
        "failure_matrix_non_surrogate_guard_preserved": non_surrogate_guard,
        "updated_pack_same_schema_blind_vector_backend_adapter_replay_detected_now": same_schema_replay_detected,
        "pack_update_required_now": pack_update_required_now,
        "selected_primary_completion_lane": "updated_pack_blind_vector_solver_side_backend_implementation_audit",
        "selected_secondary_completion_lane": "updated_pack_blind_vector_numeric_rerun_after_backend_implementation",
        "selected_reserve_completion_lane": "farther_hybrid_extra_q_range_only",
        "selected_next_generation_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_blind_vector_solver_side_backend_implementation_audit",
        "recommended_next_route_or_none": "8.7.56.5207",
        "selected_followup_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_blind_vector_solver_side_backend_implementation_gate",
        "selected_followup_route_or_none": "8.7.56.5211",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5205",
        STEP_NAME + " declaration gate",
        {
            "source_files": {"prior_audit": sign_base.display_path(PRIOR_AUDIT)},
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "8.7.56.5207",
                "followup_route": "8.7.56.5211",
            },
        },
        rows,
        summary,
        {
            "overall_status": "vector_qball_form_factor_blind_vector_solver_side_backend_adapter_gate_declared",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {"formulae": build_formulae()},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(f"[done] {STEP_TAG} blind-vector backend-adapter gate completed")
    print(f"[done] declaration: {declaration_paths['json']}")


if __name__ == "__main__":
    main()
