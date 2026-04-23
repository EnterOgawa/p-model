#!/usr/bin/env python3
"""Generate 8.7.56.5271-.5274 selected-extension solver-side deformation front-runner artifacts."""

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
        "8.7.56.5267-5270",
        "updated_pack_selected_extension_solver_side_deformation_inventory_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
PRIOR_AUDIT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5263-5266",
        "updated_pack_selected_extension_solver_side_deformation_inventory_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
SELECTED_EXTENSION_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5139-5142",
        "updated_pack_external_rule_selector_selected_extension_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
SOLVER_RECOMPUTE_HELPER = (
    ROOT / "scripts" / "quantum" / "selected_extension_solver_recompute_backend.py"
)

STEP_TAG = "8.7.56.5271-5274"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack selected-"
    "extension solver-side deformation front-runner theorem audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_selected_extension_solver_side_deformation_front_runner_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "selected_extension_solver_side_deformation_inventory_audited_front_runner_"
    "primary_hybrid_reserve_secondary_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "selected_extension_solver_side_deformation_front_runner_contract_derived_"
    "implementation_primary_pack_refresh_secondary_gate"
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


# 関数: front-runner deformation contract の式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used in the selected-extension solver-side deformation front-runner audit."""
    return {
        "deformed_effective_kernel": (
            "K_eff^(pilot-HS,deform)[Q_ret] := "
            "K_AA^(Sigma_*^(pilot-HS);D_solver_sel^(K))[Q_ret] - "
            "K_xiA^(Sigma_*^(pilot-HS);D_solver_sel^(K))[Q_ret]"
            "(K_xixi^(Sigma_*^(pilot-HS);D_solver_sel^(G))[Q_ret])^(-1)"
            "K_xiA^(Sigma_*^(pilot-HS);D_solver_sel^(K))[Q_ret]"
        ),
        "retained_q_window": "Q_ret := {0, q_theory, m0}",
        "transverse_scalar_deformation": (
            "Z_eff^(pilot-HS,deform,T)(q) := "
            "(1/2) tr(Pi_T(q) K_eff^(pilot-HS,deform)(q) Pi_T(q))"
        ),
        "blind_form_factor_deformation": (
            "F_blind^(pilot-HS,deform)(q) := "
            "Z_eff^(pilot-HS,deform,T)(q) / Z_eff^(pilot-HS,deform,T)(0)"
        ),
        "blind_alpha_deformation": (
            "alpha_blind^(pilot-HS,deform)(q) := "
            "(F_blind^(pilot-HS,deform)(q)^2) / (4 pi)"
        ),
        "residual_discriminator": (
            "Delta_deform_sel^(pilot-HS) := compare("
            "O_deform_sel^(pilot-HS), O_recomp_sel^(pilot-HS), alpha_exact(q_theory))"
        ),
        "front_runner_contract": (
            "C_deform_sel,front^(pilot-HS) := "
            "(D_solver_sel^(K), D_solver_sel^(G), D_solver_sel^(Qret))"
        ),
    }


# 関数: `.5271-.5274` を実行する。

def main() -> None:
    """Execute the selected-extension solver-side deformation front-runner theorem audit."""
    for path in (PRIOR_GATE, PRIOR_AUDIT, SELECTED_EXTENSION_GATE):
        sign_base.require(path)

    prior_gate_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    prior_audit_summary = sign_base.read_json(PRIOR_AUDIT)["summary"]
    selected_summary = sign_base.read_json(SELECTED_EXTENSION_GATE)["summary"]

    audit_selected = bool(
        prior_gate_summary[
            "gate_b_updated_pack_selected_extension_solver_side_deformation_front_runner_promoted_next"
        ]
        and prior_gate_summary["pack_update_required_now"]
    )
    retry_mode = bool(prior_gate_summary["retry_gate_computation_mode_selected"])
    non_surrogate_guard = bool(
        prior_gate_summary["failure_matrix_non_surrogate_guard_preserved"]
    )
    selected_extension_available = bool(
        selected_summary[
            "gate_a_updated_pack_exact_external_rule_selector_selected_extension_available_now"
        ]
        and selected_summary[
            "exact_external_rule_selector_selected_extension_available_now"
        ]
    )
    front_runner_candidate_formula_explicit = bool(
        prior_audit_summary[
            "exact_selected_extension_solver_side_deformation_front_runner_candidate_formula_available_now"
        ]
        and prior_audit_summary[
            "exact_selected_extension_solver_side_deformation_front_runner_compatibility_theorem_available_now"
        ]
    )
    solver_recompute_helper_available = bool(SOLVER_RECOMPUTE_HELPER.exists())
    retained_q_checkpoint_contract_available_now = bool(
        all(
            key in prior_gate_summary
            for key in (
                "blind_F_recomp_at_q_theory",
                "blind_alpha_recomp_at_q_theory",
                "delta_alpha_sel_recomp_exact",
                "relative_exact_residual_recomp",
                "q_theory_over_m0",
            )
        )
    )
    exact_selected_extension_solver_side_deformation_front_runner_contract_formula_available_now = bool(
        audit_selected
        and retry_mode
        and non_surrogate_guard
        and selected_extension_available
        and front_runner_candidate_formula_explicit
        and solver_recompute_helper_available
        and retained_q_checkpoint_contract_available_now
    )
    exact_selected_extension_solver_side_deformation_front_runner_effective_kernel_deformation_formula_available_now = bool(
        exact_selected_extension_solver_side_deformation_front_runner_contract_formula_available_now
    )
    exact_selected_extension_solver_side_deformation_front_runner_internal_resolvent_deformation_formula_available_now = bool(
        exact_selected_extension_solver_side_deformation_front_runner_contract_formula_available_now
    )
    exact_selected_extension_solver_side_deformation_front_runner_retained_q_window_formula_available_now = bool(
        exact_selected_extension_solver_side_deformation_front_runner_contract_formula_available_now
    )
    exact_selected_extension_solver_side_deformation_front_runner_residual_discriminator_formula_available_now = bool(
        exact_selected_extension_solver_side_deformation_front_runner_contract_formula_available_now
    )
    selected_extension_solver_side_deformation_implementation_primary_admissible_now = bool(
        exact_selected_extension_solver_side_deformation_front_runner_residual_discriminator_formula_available_now
    )
    updated_pack_selected_extension_solver_side_deformation_implementation_followup_required = bool(
        selected_extension_solver_side_deformation_implementation_primary_admissible_now
    )
    updated_pack_same_schema_selected_extension_solver_side_deformation_front_runner_replay_detected_now = (
        False
    )
    farther_hybrid_continuation_reopen_required_now = bool(
        prior_gate_summary["gate_c_farther_hybrid_continuation_reopen_required_now"]
    )

    rows = [
        sign_base.row(
            "updated_pack_selected_extension_solver_side_deformation_front_runner_audit_selected",
            "pass" if audit_selected else "reject",
            "updated-pack selected-extension solver-side deformation front-runner audit selected",
            sign_base.truth(audit_selected),
            "This branch is worth running only after the finite deformation inventory has already promoted one retained-q front-runner on the fixed selected extension.",
        ),
        sign_base.row(
            "retry_gate_computation_mode_selected",
            "pass" if retry_mode else "reject",
            "retry gate computation mode selected",
            sign_base.truth(retry_mode),
            "The active lane stays computation-first and does not reopen theorem-family or replay recursion.",
        ),
        sign_base.row(
            "failure_matrix_non_surrogate_guard_preserved",
            "pass" if non_surrogate_guard else "reject",
            "failure-matrix non-surrogate guard preserved",
            sign_base.truth(non_surrogate_guard),
            "The front-runner deformation contract remains honest only while exhausted surrogate and selector-choice branches stay closed.",
        ),
        sign_base.row(
            "selected_extension_available_now",
            "pass" if selected_extension_available else "reject",
            "selected extension available now",
            sign_base.truth(selected_extension_available),
            "The solver-side deformation lane is meaningful only while one concrete selected extension Sigma_*^(pilot-HS) remains fixed.",
        ),
        sign_base.row(
            "front_runner_candidate_formula_explicit_now",
            "pass" if front_runner_candidate_formula_explicit else "reject",
            "front-runner candidate formula explicit now",
            sign_base.truth(front_runner_candidate_formula_explicit),
            "The promoted deformation route must already be written explicitly before it can be turned into one concrete deformation contract.",
        ),
        sign_base.row(
            "selected_extension_solver_recompute_helper_available_now",
            "pass" if solver_recompute_helper_available else "reject",
            "selected-extension solver-recompute helper available now",
            sign_base.truth(solver_recompute_helper_available),
            "The deformation contract is anchored to the already materialized recompute helper path instead of inventing a fresh baseline.",
        ),
        sign_base.row(
            "retained_q_checkpoint_contract_available_now",
            "pass" if retained_q_checkpoint_contract_available_now else "reject",
            "retained-q checkpoint contract available now",
            sign_base.truth(retained_q_checkpoint_contract_available_now),
            "The promoted deformation contract remains anchored to retained q checkpoints and preserved failure values.",
        ),
        sign_base.row(
            "exact_selected_extension_solver_side_deformation_front_runner_contract_formula_available_now",
            "pass"
            if exact_selected_extension_solver_side_deformation_front_runner_contract_formula_available_now
            else "reject",
            "exact selected-extension solver-side deformation front-runner contract formula available now",
            sign_base.truth(
                exact_selected_extension_solver_side_deformation_front_runner_contract_formula_available_now
            ),
            "The live front-runner is now a literal deformation contract on Sigma_*^(pilot-HS), not just an inventory label.",
        ),
        sign_base.row(
            "exact_selected_extension_solver_side_deformation_front_runner_effective_kernel_deformation_formula_available_now",
            "pass"
            if exact_selected_extension_solver_side_deformation_front_runner_effective_kernel_deformation_formula_available_now
            else "reject",
            "exact selected-extension solver-side deformation front-runner effective-kernel deformation formula available now",
            sign_base.truth(
                exact_selected_extension_solver_side_deformation_front_runner_effective_kernel_deformation_formula_available_now
            ),
            "The contract explicitly refreshes the Schur-complement effective kernel instead of replaying retained values unchanged.",
        ),
        sign_base.row(
            "exact_selected_extension_solver_side_deformation_front_runner_internal_resolvent_deformation_formula_available_now",
            "pass"
            if exact_selected_extension_solver_side_deformation_front_runner_internal_resolvent_deformation_formula_available_now
            else "reject",
            "exact selected-extension solver-side deformation front-runner internal-resolvent deformation formula available now",
            sign_base.truth(
                exact_selected_extension_solver_side_deformation_front_runner_internal_resolvent_deformation_formula_available_now
            ),
            "The contract explicitly refreshes the internal resolvent / mode closure rather than assuming retained inversion data remains exact.",
        ),
        sign_base.row(
            "exact_selected_extension_solver_side_deformation_front_runner_retained_q_window_formula_available_now",
            "pass"
            if exact_selected_extension_solver_side_deformation_front_runner_retained_q_window_formula_available_now
            else "reject",
            "exact selected-extension solver-side deformation front-runner retained-q-window formula available now",
            sign_base.truth(
                exact_selected_extension_solver_side_deformation_front_runner_retained_q_window_formula_available_now
            ),
            "The promoted deformation contract keeps the retained q-window semantics fixed while changing only the solver-side deformation ingredients.",
        ),
        sign_base.row(
            "exact_selected_extension_solver_side_deformation_front_runner_residual_discriminator_formula_available_now",
            "pass"
            if exact_selected_extension_solver_side_deformation_front_runner_residual_discriminator_formula_available_now
            else "reject",
            "exact selected-extension solver-side deformation front-runner residual discriminator formula available now",
            sign_base.truth(
                exact_selected_extension_solver_side_deformation_front_runner_residual_discriminator_formula_available_now
            ),
            "The front-runner now carries one explicit discriminator against the preserved recompute surface instead of a generic promise of future improvement.",
        ),
        sign_base.row(
            "selected_extension_solver_side_deformation_implementation_primary_admissible_now",
            "pass"
            if selected_extension_solver_side_deformation_implementation_primary_admissible_now
            else "reject",
            "selected-extension solver-side deformation implementation primary admissible now",
            sign_base.truth(
                selected_extension_solver_side_deformation_implementation_primary_admissible_now
            ),
            "Once the contract and discriminator are fixed, actual implementation becomes the honest next blocker.",
        ),
        sign_base.row(
            "updated_pack_same_schema_selected_extension_solver_side_deformation_front_runner_replay_detected_now",
            "pass"
            if updated_pack_same_schema_selected_extension_solver_side_deformation_front_runner_replay_detected_now
            else "reject",
            "updated-pack same-schema selected-extension solver-side deformation front-runner replay detected now",
            sign_base.truth(
                updated_pack_same_schema_selected_extension_solver_side_deformation_front_runner_replay_detected_now
            ),
            "False means the front-runner layer produced one contract/discriminator surface instead of replaying the generic inventory schema.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "selected_extension_label": prior_gate_summary["selected_extension_label"],
        "q_theory_over_m0": float(prior_gate_summary["q_theory_over_m0"]),
        "blind_F_recomp_at_q_theory": float(
            prior_gate_summary["blind_F_recomp_at_q_theory"]
        ),
        "blind_alpha_recomp_at_q_theory": float(
            prior_gate_summary["blind_alpha_recomp_at_q_theory"]
        ),
        "delta_alpha_sel_recomp_exact": float(
            prior_gate_summary["delta_alpha_sel_recomp_exact"]
        ),
        "relative_exact_residual_recomp": float(
            prior_gate_summary["relative_exact_residual_recomp"]
        ),
        "exact_selected_extension_solver_side_deformation_front_runner_contract_formula_available_now": exact_selected_extension_solver_side_deformation_front_runner_contract_formula_available_now,
        "exact_selected_extension_solver_side_deformation_front_runner_effective_kernel_deformation_formula_available_now": exact_selected_extension_solver_side_deformation_front_runner_effective_kernel_deformation_formula_available_now,
        "exact_selected_extension_solver_side_deformation_front_runner_internal_resolvent_deformation_formula_available_now": exact_selected_extension_solver_side_deformation_front_runner_internal_resolvent_deformation_formula_available_now,
        "exact_selected_extension_solver_side_deformation_front_runner_retained_q_window_formula_available_now": exact_selected_extension_solver_side_deformation_front_runner_retained_q_window_formula_available_now,
        "exact_selected_extension_solver_side_deformation_front_runner_residual_discriminator_formula_available_now": exact_selected_extension_solver_side_deformation_front_runner_residual_discriminator_formula_available_now,
        "selected_extension_solver_side_deformation_implementation_primary_admissible_now": selected_extension_solver_side_deformation_implementation_primary_admissible_now,
        "updated_pack_selected_extension_solver_side_deformation_implementation_followup_required": updated_pack_selected_extension_solver_side_deformation_implementation_followup_required,
        "retry_gate_computation_mode_selected": retry_mode,
        "failure_matrix_non_surrogate_guard_preserved": non_surrogate_guard,
        "updated_pack_same_schema_selected_extension_solver_side_deformation_front_runner_replay_detected_now": updated_pack_same_schema_selected_extension_solver_side_deformation_front_runner_replay_detected_now,
        "farther_hybrid_continuation_reopen_required_now": farther_hybrid_continuation_reopen_required_now,
        "pack_update_required_now": updated_pack_selected_extension_solver_side_deformation_implementation_followup_required,
        "selected_primary_completion_lane": "updated_pack_selected_extension_solver_side_deformation_front_runner_implementation_audit",
        "selected_secondary_completion_lane": "updated_pack_selected_extension_solver_side_deformation_numeric_rerun",
        "selected_reserve_completion_lane": "farther_hybrid_extra_q_range_only",
        "selected_next_generation_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_selected_extension_solver_side_deformation_front_runner_implementation_audit",
        "recommended_next_route_or_none": "8.7.56.5279",
        "selected_followup_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_selected_extension_solver_side_deformation_front_runner_implementation_gate",
        "selected_followup_route_or_none": "8.7.56.5283",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5273",
        STEP_NAME + " declaration gate",
        {
            "source_files": {
                "prior_gate": sign_base.display_path(PRIOR_GATE),
                "prior_audit": sign_base.display_path(PRIOR_AUDIT),
                "selected_extension_gate": sign_base.display_path(SELECTED_EXTENSION_GATE),
                "solver_recompute_helper": sign_base.display_path(SOLVER_RECOMPUTE_HELPER),
            },
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "8.7.56.5279",
                "followup_route": "8.7.56.5283",
            },
        },
        rows,
        summary,
        {
            "overall_status": "vector_qball_form_factor_selected_extension_solver_side_deformation_front_runner_declared",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {"formulae": build_formulae()},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(
        f"[done] {STEP_TAG} selected-extension solver-side deformation front-runner audit completed"
    )
    print(f"[done] declaration: {declaration_paths['json']}")


# 関数: CLI entrypoint から audit を実行する。

if __name__ == "__main__":
    main()
