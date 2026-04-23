#!/usr/bin/env python3
"""Generate 8.7.56.5223-.5226 backend-integrated residual-origin refresh artifacts."""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.quantum.t2a_1843 as sign_base
from scripts.quantum.blind_vector_selected_extension_backend import (
    build_selected_extension_backend_pack,
)
from scripts.utils.windows_length_policy import build_compact_artifact_stem
from scripts.utils.windows_length_policy import build_metrics_paths


PUBLIC_OUT = ROOT / "output" / "public" / "quantum"
PRIOR_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5219-5222",
        "updated_pack_blind_vector_backend_integrated_retained_q_rerun_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
PHASE3_EVAL = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_unified_closure_"
    "blind_vector_observable_gate_numeric_evaluation_metrics.json"
)
SCALAR_TARGET = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_qball_projection_overlap_coupled_tail_"
    "reconciliation_review_numeric_evaluation_metrics.json"
)

STEP_TAG = "8.7.56.5223-5226"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack blind-vector "
    "backend-integrated residual-origin refresh audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_blind_vector_backend_integrated_residual_origin_refresh_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "blind_vector_backend_integrated_retained_q_rerun_audited_residual_origin_"
    "refresh_primary_hybrid_reserve_secondary_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "blind_vector_backend_integrated_retained_q_rerun_preserves_phase3_failure_"
    "closeout_completed_selected_extension_solver_recompute_primary_hybrid_"
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


# 関数: refresh closeout の式を返す。
def build_formulae() -> dict[str, str]:
    """Return formulas used in the backend-integrated residual-origin refresh audit."""
    return {
        "checkpoint_preservation": (
            "{F_blind^(pilot-HS,backend), alpha_blind^(pilot-HS,backend)}_Qret = "
            "{F_blind, alpha_blind}_Phase3"
        ),
        "backend_not_origin": (
            "backend_not_origin iff backend-integrated retained-q rerun preserves "
            "the wrong-sign / low-alpha Phase 3 failure surface"
        ),
        "remaining_blocker": (
            "D_solver^(selected-extension,recompute) := actual recomputation of "
            "K_eff^(pilot-HS), Z_eff^(pilot-HS,T), F_blind^(pilot-HS), "
            "alpha_blind^(pilot-HS) on the selected extension"
        ),
    }


# 関数: `.5223-.5226` を実行する。
def main() -> None:
    """Execute the backend-integrated residual-origin refresh audit."""
    for path in (PRIOR_GATE, PHASE3_EVAL, SCALAR_TARGET):
        sign_base.require(path)

    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    phase3_summary = sign_base.read_json(PHASE3_EVAL)["summary"]
    scalar_summary = sign_base.read_json(SCALAR_TARGET)["summary"]

    audit_selected = bool(
        prior_summary[
            "gate_b_updated_pack_blind_vector_backend_integrated_residual_origin_refresh_promoted_next"
        ]
        and prior_summary[
            "gate_a_updated_pack_exact_blind_vector_backend_integrated_retained_q_rerun_available_now"
        ]
    )
    retry_mode = bool(prior_summary["retry_gate_computation_mode_selected"])
    non_surrogate_guard = bool(
        prior_summary["failure_matrix_non_surrogate_guard_preserved"]
    )

    backend_pack = build_selected_extension_backend_pack()
    blind_F_at_q_theory = float(backend_pack["blind_target_keys"]["blind_F_at_q_theory"])
    blind_alpha_at_q_theory = float(
        backend_pack["blind_target_keys"]["blind_alpha_at_q_theory"]
    )
    alpha_exact_at_q_theory = float(scalar_summary["alpha_exact_at_q_theory"])
    delta_alpha_sel_exact = float(blind_alpha_at_q_theory - alpha_exact_at_q_theory)
    relative_exact_residual = float(abs(delta_alpha_sel_exact) / alpha_exact_at_q_theory)
    checkpoint_preserved_now = bool(
        math.isclose(
            blind_F_at_q_theory,
            float(phase3_summary["blind_F_at_q_theory"]),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        and math.isclose(
            blind_alpha_at_q_theory,
            float(phase3_summary["blind_alpha_at_q_theory"]),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
    )
    wrong_sign_persists_now = bool(blind_F_at_q_theory < 0.0)
    low_alpha_persists_now = bool(blind_alpha_at_q_theory < alpha_exact_at_q_theory)
    exact_blind_vector_backend_integrated_rerun_preserves_phase3_failure_theorem_available_now = bool(
        audit_selected
        and retry_mode
        and non_surrogate_guard
        and checkpoint_preserved_now
        and wrong_sign_persists_now
        and low_alpha_persists_now
    )
    exact_blind_vector_backend_adapter_not_residual_origin_theorem_available_now = bool(
        exact_blind_vector_backend_integrated_rerun_preserves_phase3_failure_theorem_available_now
    )
    exact_blind_vector_selected_extension_solver_recompute_lane_required_theorem_available_now = bool(
        exact_blind_vector_backend_integrated_rerun_preserves_phase3_failure_theorem_available_now
    )
    exact_blind_vector_direct_computation_lane_negative_closeout_available_now = bool(
        exact_blind_vector_backend_integrated_rerun_preserves_phase3_failure_theorem_available_now
    )
    updated_pack_selected_extension_solver_recompute_followup_required = bool(
        exact_blind_vector_direct_computation_lane_negative_closeout_available_now
    )
    updated_pack_same_schema_blind_vector_backend_integrated_residual_refresh_replay_detected_now = (
        False
    )
    farther_hybrid_continuation_reopen_required_now = False

    rows = [
        sign_base.row(
            "updated_pack_blind_vector_backend_integrated_residual_origin_refresh_audit_selected",
            "pass" if audit_selected else "reject",
            "updated-pack blind-vector backend-integrated residual-origin refresh audit selected",
            sign_base.truth(audit_selected),
            "This branch is worth running only after the integrated retained-q rerun itself is already official and residual-origin refresh is the live blocker.",
        ),
        sign_base.row(
            "retry_gate_computation_mode_selected",
            "pass" if retry_mode else "reject",
            "retry gate computation mode selected",
            sign_base.truth(retry_mode),
            "The blind-vector lane stays on computation-side verdict work instead of reopening selector or same-tag replay branches.",
        ),
        sign_base.row(
            "failure_matrix_non_surrogate_guard_preserved",
            "pass" if non_surrogate_guard else "reject",
            "failure-matrix non-surrogate guard preserved",
            sign_base.truth(non_surrogate_guard),
            "Residual-origin refresh remains honest only if exhausted surrogate and replay lanes stay closed.",
        ),
        sign_base.row(
            "checkpoint_preserved_now",
            "pass" if checkpoint_preserved_now else "reject",
            "backend-integrated retained-q checkpoint preserved now",
            sign_base.truth(checkpoint_preserved_now),
            "The backend-integrated rerun reproduces the retained Phase 3 blind checkpoint values exactly.",
        ),
        sign_base.row(
            "wrong_sign_persists_now",
            "pass" if wrong_sign_persists_now else "reject",
            "wrong sign persists now",
            sign_base.truth(wrong_sign_persists_now),
            "The backend-integrated rerun still lives in the wrong-sign sector at q_theory.",
        ),
        sign_base.row(
            "low_alpha_persists_now",
            "pass" if low_alpha_persists_now else "reject",
            "low alpha persists now",
            sign_base.truth(low_alpha_persists_now),
            "The backend-integrated rerun still sits below the retained exact scalar alpha at q_theory.",
        ),
        sign_base.row(
            "exact_blind_vector_backend_integrated_rerun_preserves_phase3_failure_theorem_available_now",
            "pass"
            if exact_blind_vector_backend_integrated_rerun_preserves_phase3_failure_theorem_available_now
            else "reject",
            "exact blind-vector backend-integrated rerun preserves Phase 3 failure theorem available now",
            sign_base.truth(
                exact_blind_vector_backend_integrated_rerun_preserves_phase3_failure_theorem_available_now
            ),
            "Backend integration does not move the wrong-sign / low-alpha surface; it preserves the retained Phase 3 failure after the adapter gap is removed.",
        ),
        sign_base.row(
            "exact_blind_vector_backend_adapter_not_residual_origin_theorem_available_now",
            "pass"
            if exact_blind_vector_backend_adapter_not_residual_origin_theorem_available_now
            else "reject",
            "exact blind-vector backend adapter not residual-origin theorem available now",
            sign_base.truth(
                exact_blind_vector_backend_adapter_not_residual_origin_theorem_available_now
            ),
            "The backend adapter is no longer an honest explanation of the blind-vector residual once the integrated rerun preserves the same failed surface.",
        ),
        sign_base.row(
            "exact_blind_vector_selected_extension_solver_recompute_lane_required_theorem_available_now",
            "pass"
            if exact_blind_vector_selected_extension_solver_recompute_lane_required_theorem_available_now
            else "reject",
            "exact blind-vector selected-extension solver-recompute lane required theorem available now",
            sign_base.truth(
                exact_blind_vector_selected_extension_solver_recompute_lane_required_theorem_available_now
            ),
            "The remaining blocker is actual selected-extension solver recomputation, not selector ambiguity, backend implementation, or adapter integration.",
        ),
        sign_base.row(
            "exact_blind_vector_direct_computation_lane_negative_closeout_available_now",
            "pass"
            if exact_blind_vector_direct_computation_lane_negative_closeout_available_now
            else "reject",
            "exact blind-vector direct-computation lane negative closeout available now",
            sign_base.truth(
                exact_blind_vector_direct_computation_lane_negative_closeout_available_now
            ),
            "The blind-vector direct-computation lane itself now closes negatively: integrated rerun does not fix the retained wrong-sign / low-alpha failure.",
        ),
        sign_base.row(
            "delta_alpha_sel_exact",
            "watch",
            "backend-integrated delta alpha vs exact scalar target",
            delta_alpha_sel_exact,
            "The selected extension still undershoots the retained exact scalar alpha after backend integration.",
        ),
        sign_base.row(
            "relative_exact_residual",
            "watch",
            "backend-integrated relative residual vs exact scalar target",
            relative_exact_residual,
            "The backend-integrated rerun still differs from the retained exact scalar alpha by about 91.6%.",
        ),
        sign_base.row(
            "updated_pack_selected_extension_solver_recompute_followup_required",
            "pass" if updated_pack_selected_extension_solver_recompute_followup_required else "reject",
            "updated-pack selected-extension solver recompute followup required",
            sign_base.truth(updated_pack_selected_extension_solver_recompute_followup_required),
            "The honest next lane is selected-extension solver recomputation on the retained-q surface.",
        ),
        sign_base.row(
            "updated_pack_same_schema_blind_vector_backend_integrated_residual_refresh_replay_detected_now",
            "pass"
            if updated_pack_same_schema_blind_vector_backend_integrated_residual_refresh_replay_detected_now
            else "reject",
            "updated-pack same-schema blind-vector backend-integrated residual refresh replay detected now",
            sign_base.truth(
                updated_pack_same_schema_blind_vector_backend_integrated_residual_refresh_replay_detected_now
            ),
            "False means this turn compressed the live blocker from backend integration to a new solver-recompute lane.",
        ),
        sign_base.row(
            "farther_hybrid_continuation_reopen_required_now",
            "pass" if farther_hybrid_continuation_reopen_required_now else "reject",
            "farther hybrid continuation reopen required now",
            sign_base.truth(farther_hybrid_continuation_reopen_required_now),
            "Extra q-range stays reserve-only because the next honest task is selected-extension solver recomputation at retained q.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "q_theory_over_m0": float(backend_pack["retained_q_window"]["q_theory_over_m0"]),
        "blind_F_at_q_theory": blind_F_at_q_theory,
        "blind_alpha_at_q_theory": blind_alpha_at_q_theory,
        "alpha_exact_at_q_theory": alpha_exact_at_q_theory,
        "delta_alpha_sel_exact": delta_alpha_sel_exact,
        "relative_exact_residual": relative_exact_residual,
        "exact_blind_vector_backend_integrated_rerun_preserves_phase3_failure_theorem_available_now": exact_blind_vector_backend_integrated_rerun_preserves_phase3_failure_theorem_available_now,
        "exact_blind_vector_backend_adapter_not_residual_origin_theorem_available_now": exact_blind_vector_backend_adapter_not_residual_origin_theorem_available_now,
        "exact_blind_vector_selected_extension_solver_recompute_lane_required_theorem_available_now": exact_blind_vector_selected_extension_solver_recompute_lane_required_theorem_available_now,
        "exact_blind_vector_direct_computation_lane_negative_closeout_available_now": exact_blind_vector_direct_computation_lane_negative_closeout_available_now,
        "updated_pack_selected_extension_solver_recompute_followup_required": updated_pack_selected_extension_solver_recompute_followup_required,
        "updated_pack_same_schema_blind_vector_backend_integrated_residual_refresh_replay_detected_now": updated_pack_same_schema_blind_vector_backend_integrated_residual_refresh_replay_detected_now,
        "retry_gate_computation_mode_selected": retry_mode,
        "failure_matrix_non_surrogate_guard_preserved": non_surrogate_guard,
        "farther_hybrid_continuation_reopen_required_now": farther_hybrid_continuation_reopen_required_now,
        "pack_update_required_now": updated_pack_selected_extension_solver_recompute_followup_required,
        "selected_primary_completion_lane": "updated_pack_selected_extension_solver_recompute_contract_audit",
        "selected_secondary_completion_lane": "farther_hybrid_extra_q_range_only",
        "selected_reserve_completion_lane": "blind_vector_lane_closed_negative",
        "selected_next_generation_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_blind_vector_backend_integrated_residual_origin_refresh_gate",
        "recommended_next_route_or_none": "8.7.56.5227",
        "selected_followup_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_selected_extension_solver_recompute_contract_audit",
        "selected_followup_route_or_none": "8.7.56.5231",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5225",
        STEP_NAME + " declaration gate",
        {
            "source_files": {
                "prior_gate": sign_base.display_path(PRIOR_GATE),
                "phase3_eval": sign_base.display_path(PHASE3_EVAL),
                "scalar_target_eval": sign_base.display_path(SCALAR_TARGET),
                "backend_helper": sign_base.display_path(
                    ROOT / "scripts" / "quantum" / "blind_vector_selected_extension_backend.py"
                ),
            },
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "8.7.56.5227",
                "followup_route": "8.7.56.5231",
            },
        },
        rows,
        summary,
        {
            "overall_status": "vector_qball_form_factor_blind_vector_backend_integrated_residual_origin_refresh_declared",
            "branch_completed": True,
            "blind_vector_direct_computation_lane_negative_closeout_available_now": exact_blind_vector_direct_computation_lane_negative_closeout_available_now,
            "physical_reject_required": False,
        },
        {"formulae": build_formulae()},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(f"[done] {STEP_TAG} blind-vector backend-integrated residual-origin refresh completed")
    print(f"[done] declaration: {declaration_paths['json']}")


if __name__ == "__main__":
    main()
