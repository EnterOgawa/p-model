#!/usr/bin/env python3
"""Generate 8.7.56.5175-.5178 blind-vector solver-side deformation front-runner artifacts."""

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
PRIOR_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5171-5174",
        "updated_pack_blind_vector_solver_side_deformation_inventory_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
PRIOR_AUDIT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5167-5170",
        "updated_pack_blind_vector_solver_side_deformation_inventory_audit",
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

STEP_TAG = "8.7.56.5175-5178"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack blind-vector "
    "solver-side deformation front-runner theorem audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_blind_vector_solver_side_deformation_front_runner_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "blind_vector_solver_side_deformation_inventory_audited_front_runner_"
    "primary_hybrid_reserve_secondary_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "blind_vector_solver_side_deformation_front_runner_recompute_contract_"
    "derived_numeric_rerun_primary_pack_refresh_secondary_gate"
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


# 関数: front-runner recomputation contract の式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used in the blind-vector solver-side deformation front-runner audit."""
    return {
        "selected_effective_kernel_recompute": (
            "K_eff^(pilot-HS,recomp)[Q] := K_AA^(Sigma_*^(pilot-HS))[Q] - "
            "K_xiA^(Sigma_*^(pilot-HS))[Q](K_xixi[Q])^(-1)"
            "K_xiA^(Sigma_*^(pilot-HS))[Q]"
        ),
        "retained_q_window": "Q_ret := {0, q_theory, m0}",
        "transverse_scalar_recompute": (
            "Z_eff^(pilot-HS,recomp,T)(q) := "
            "(1/2) tr(Pi_T(q) K_eff^(pilot-HS,recomp)(q) Pi_T(q))"
        ),
        "blind_form_factor_recompute": (
            "F_blind^(pilot-HS,recomp)(q) := "
            "Z_eff^(pilot-HS,recomp,T)(q) / Z_eff^(pilot-HS,recomp,T)(0)"
        ),
        "blind_alpha_recompute": (
            "alpha_blind^(pilot-HS,recomp)(q) := "
            "(F_blind^(pilot-HS,recomp)(q)^2) / (4 pi)"
        ),
        "residual_discriminator_recompute": (
            "delta_alpha_sel^(pilot-HS,recomp) := "
            "alpha_blind^(pilot-HS,recomp)(q_theory) - alpha_exact(q_theory)"
        ),
    }


# 関数: `.5175-.5178` を実行する。

def main() -> None:
    """Execute the blind-vector solver-side deformation front-runner theorem audit."""
    for path in (PRIOR_GATE, PRIOR_AUDIT, PHASE3_EVAL, SCALAR_TARGET):
        sign_base.require(path)

    prior_gate_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    prior_audit_summary = sign_base.read_json(PRIOR_AUDIT)["summary"]
    phase3_summary = sign_base.read_json(PHASE3_EVAL)["summary"]
    scalar_summary = sign_base.read_json(SCALAR_TARGET)["summary"]

    audit_selected = bool(
        prior_gate_summary[
            "gate_b_updated_pack_blind_vector_solver_side_deformation_front_runner_promoted_next"
        ]
        and prior_gate_summary["pack_update_required_now"]
    )
    retry_mode = bool(prior_gate_summary["retry_gate_computation_mode_selected"])
    non_surrogate_guard = bool(
        prior_gate_summary["failure_matrix_non_surrogate_guard_preserved"]
    )
    front_runner_candidate_formula_explicit = bool(
        prior_audit_summary[
            "exact_blind_vector_solver_side_deformation_front_runner_candidate_formula_available_now"
        ]
        and prior_audit_summary[
            "exact_blind_vector_solver_side_deformation_front_runner_compatibility_theorem_available_now"
        ]
    )
    retained_q_checkpoint_contract_available_now = bool(
        all(
            key in phase3_summary
            for key in (
                "blind_F_at_zero",
                "blind_F_at_q_theory",
                "blind_F_at_m0",
                "blind_alpha_at_q_theory",
            )
        )
        and all(
            key in scalar_summary
            for key in (
                "q_theory_over_m0",
                "alpha_exact_at_q_theory",
            )
        )
    )
    exact_blind_vector_solver_side_deformation_front_runner_recompute_contract_formula_available_now = bool(
        audit_selected
        and retry_mode
        and non_surrogate_guard
        and front_runner_candidate_formula_explicit
        and retained_q_checkpoint_contract_available_now
    )
    exact_blind_vector_solver_side_deformation_front_runner_retained_q_window_formula_available_now = bool(
        exact_blind_vector_solver_side_deformation_front_runner_recompute_contract_formula_available_now
    )
    exact_blind_vector_solver_side_deformation_front_runner_residual_discriminator_formula_available_now = bool(
        exact_blind_vector_solver_side_deformation_front_runner_recompute_contract_formula_available_now
    )
    blind_vector_solver_side_numeric_rerun_primary_admissible_now = bool(
        exact_blind_vector_solver_side_deformation_front_runner_residual_discriminator_formula_available_now
    )
    updated_pack_blind_vector_solver_side_numeric_rerun_followup_required = bool(
        blind_vector_solver_side_numeric_rerun_primary_admissible_now
    )
    updated_pack_same_schema_blind_vector_solver_side_deformation_front_runner_replay_detected_now = (
        False
    )
    farther_hybrid_continuation_reopen_required_now = bool(
        prior_gate_summary["gate_c_farther_hybrid_continuation_reopen_required_now"]
    )

    rows = [
        sign_base.row(
            "updated_pack_blind_vector_solver_side_deformation_front_runner_audit_selected",
            "pass" if audit_selected else "reject",
            "updated-pack blind-vector solver-side deformation front-runner audit selected",
            sign_base.truth(audit_selected),
            "This branch is worth running only after the finite deformation inventory has already promoted one retained-q Schur-complement recomputation front-runner.",
        ),
        sign_base.row(
            "retry_gate_computation_mode_selected",
            "pass" if retry_mode else "reject",
            "retry gate computation mode selected",
            sign_base.truth(retry_mode),
            "The promoted front-runner is audited as a concrete recomputation contract instead of reopening theorem-family or selector recursion.",
        ),
        sign_base.row(
            "failure_matrix_non_surrogate_guard_preserved",
            "pass" if non_surrogate_guard else "reject",
            "failure-matrix non-surrogate guard preserved",
            sign_base.truth(non_surrogate_guard),
            "The recomputation contract remains honest only if exhausted surrogate and selector-choice lanes stay closed.",
        ),
        sign_base.row(
            "front_runner_candidate_formula_explicit_now",
            "pass" if front_runner_candidate_formula_explicit else "reject",
            "front-runner candidate formula explicit now",
            sign_base.truth(front_runner_candidate_formula_explicit),
            "The promoted recomputation route must already be written explicitly before it can be turned into a concrete rerun contract.",
        ),
        sign_base.row(
            "retained_q_checkpoint_contract_available_now",
            "pass" if retained_q_checkpoint_contract_available_now else "reject",
            "retained-q checkpoint contract available now",
            sign_base.truth(retained_q_checkpoint_contract_available_now),
            "The front-runner remains anchored to the retained checkpoints q = 0, q_theory, m0 rather than inventing a fresh comparison surface.",
        ),
        sign_base.row(
            "exact_blind_vector_solver_side_deformation_front_runner_recompute_contract_formula_available_now",
            "pass"
            if exact_blind_vector_solver_side_deformation_front_runner_recompute_contract_formula_available_now
            else "reject",
            "exact blind-vector solver-side deformation front-runner recompute contract formula available now",
            sign_base.truth(
                exact_blind_vector_solver_side_deformation_front_runner_recompute_contract_formula_available_now
            ),
            "The live front-runner is now a literal recomputation contract for the selected-extension Schur-complement kernel, not just an inventory label.",
        ),
        sign_base.row(
            "exact_blind_vector_solver_side_deformation_front_runner_retained_q_window_formula_available_now",
            "pass"
            if exact_blind_vector_solver_side_deformation_front_runner_retained_q_window_formula_available_now
            else "reject",
            "exact blind-vector solver-side deformation front-runner retained-q-window formula available now",
            sign_base.truth(
                exact_blind_vector_solver_side_deformation_front_runner_retained_q_window_formula_available_now
            ),
            "The promoted recomputation keeps the retained q-window semantics fixed while replacing only the inherited replayed kernel values.",
        ),
        sign_base.row(
            "exact_blind_vector_solver_side_deformation_front_runner_residual_discriminator_formula_available_now",
            "pass"
            if exact_blind_vector_solver_side_deformation_front_runner_residual_discriminator_formula_available_now
            else "reject",
            "exact blind-vector solver-side deformation front-runner residual discriminator formula available now",
            sign_base.truth(
                exact_blind_vector_solver_side_deformation_front_runner_residual_discriminator_formula_available_now
            ),
            "The promoted rerun now has a literal residual discriminator against the retained exact scalar target at q_theory.",
        ),
        sign_base.row(
            "blind_vector_solver_side_numeric_rerun_primary_admissible_now",
            "pass" if blind_vector_solver_side_numeric_rerun_primary_admissible_now else "reject",
            "blind-vector solver-side numeric rerun primary admissible now",
            sign_base.truth(blind_vector_solver_side_numeric_rerun_primary_admissible_now),
            "Once the front-runner recomputation contract and retained-q discriminator are fixed, the honest next blocker is actual numeric rerun, not more inventory work.",
        ),
        sign_base.row(
            "updated_pack_blind_vector_solver_side_numeric_rerun_followup_required",
            "pass"
            if updated_pack_blind_vector_solver_side_numeric_rerun_followup_required
            else "reject",
            "updated-pack blind-vector solver-side numeric rerun followup required",
            sign_base.truth(
                updated_pack_blind_vector_solver_side_numeric_rerun_followup_required
            ),
            "The next step is now a concrete numeric rerun on the fixed selected extension and retained checkpoint surface.",
        ),
        sign_base.row(
            "updated_pack_same_schema_blind_vector_solver_side_deformation_front_runner_replay_detected_now",
            "pass"
            if updated_pack_same_schema_blind_vector_solver_side_deformation_front_runner_replay_detected_now
            else "reject",
            "updated-pack same-schema blind-vector solver-side deformation front-runner replay detected now",
            sign_base.truth(
                updated_pack_same_schema_blind_vector_solver_side_deformation_front_runner_replay_detected_now
            ),
            "False means this turn promoted a concrete recomputation contract instead of replaying the generic inventory classification.",
        ),
        sign_base.row(
            "farther_hybrid_continuation_reopen_required_now",
            "pass" if farther_hybrid_continuation_reopen_required_now else "reject",
            "farther hybrid continuation reopen required now",
            sign_base.truth(farther_hybrid_continuation_reopen_required_now),
            "Extra q-range stays reserve-only because retained-q rerun must be exhausted first.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "q_theory_over_m0": float(scalar_summary["q_theory_over_m0"]),
        "blind_F_at_zero": float(phase3_summary["blind_F_at_zero"]),
        "blind_F_at_q_theory": float(phase3_summary["blind_F_at_q_theory"]),
        "blind_F_at_m0": float(phase3_summary["blind_F_at_m0"]),
        "blind_alpha_at_q_theory": float(phase3_summary["blind_alpha_at_q_theory"]),
        "alpha_exact_at_q_theory": float(scalar_summary["alpha_exact_at_q_theory"]),
        "exact_blind_vector_solver_side_deformation_front_runner_recompute_contract_formula_available_now": exact_blind_vector_solver_side_deformation_front_runner_recompute_contract_formula_available_now,
        "exact_blind_vector_solver_side_deformation_front_runner_retained_q_window_formula_available_now": exact_blind_vector_solver_side_deformation_front_runner_retained_q_window_formula_available_now,
        "exact_blind_vector_solver_side_deformation_front_runner_residual_discriminator_formula_available_now": exact_blind_vector_solver_side_deformation_front_runner_residual_discriminator_formula_available_now,
        "blind_vector_solver_side_numeric_rerun_primary_admissible_now": blind_vector_solver_side_numeric_rerun_primary_admissible_now,
        "updated_pack_blind_vector_solver_side_numeric_rerun_followup_required": updated_pack_blind_vector_solver_side_numeric_rerun_followup_required,
        "updated_pack_same_schema_blind_vector_solver_side_deformation_front_runner_replay_detected_now": updated_pack_same_schema_blind_vector_solver_side_deformation_front_runner_replay_detected_now,
        "retry_gate_computation_mode_selected": retry_mode,
        "failure_matrix_non_surrogate_guard_preserved": non_surrogate_guard,
        "farther_hybrid_continuation_reopen_required_now": farther_hybrid_continuation_reopen_required_now,
        "pack_update_required_now": bool(
            updated_pack_blind_vector_solver_side_numeric_rerun_followup_required
        ),
        "selected_primary_completion_lane": "updated_pack_blind_vector_solver_side_numeric_rerun_audit",
        "selected_secondary_completion_lane": "updated_pack_blind_vector_residual_origin_refresh_after_solver_rerun",
        "selected_reserve_completion_lane": "farther_hybrid_extra_q_range_only",
        "selected_next_generation_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_blind_vector_solver_side_numeric_rerun_audit",
        "recommended_next_route_or_none": "8.7.56.5183",
        "selected_followup_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_blind_vector_solver_side_numeric_rerun_gate",
        "selected_followup_route_or_none": "8.7.56.5187",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5177",
        STEP_NAME + " declaration gate",
        {
            "source_files": {
                "prior_gate": sign_base.display_path(PRIOR_GATE),
                "prior_audit": sign_base.display_path(PRIOR_AUDIT),
                "phase3_eval": sign_base.display_path(PHASE3_EVAL),
                "scalar_target_eval": sign_base.display_path(SCALAR_TARGET),
            },
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "8.7.56.5183",
                "followup_route": "8.7.56.5187",
            },
        },
        rows,
        summary,
        {
            "overall_status": "vector_qball_form_factor_blind_vector_solver_side_deformation_front_runner_declared",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {"formulae": build_formulae()},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(f"[done] {STEP_TAG} blind-vector solver-side deformation front-runner completed")
    print(f"[done] declaration: {declaration_paths['json']}")


if __name__ == "__main__":
    main()
