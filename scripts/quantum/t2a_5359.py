#!/usr/bin/env python3
"""Generate 8.7.56.5359-.5362 source-materialization front-runner artifacts."""

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
        "8.7.56.5355-5358",
        "updated_pack_selected_extension_independent_extra_q_range_source_materialization_inventory_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
PRIOR_AUDIT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5351-5354",
        "updated_pack_selected_extension_independent_extra_q_range_source_materialization_inventory_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
DEFORMATION_HELPER = (
    ROOT / "scripts" / "quantum" / "selected_extension_solver_side_deformation_backend.py"
)
EXTRA_Q_RANGE_HELPER = (
    ROOT / "scripts" / "quantum" / "selected_extension_solver_side_extra_q_range_backend.py"
)

STEP_TAG = "8.7.56.5359-5362"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack selected-"
    "extension independent extra-q-range source-materialization front-runner audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_selected_extension_independent_extra_q_range_source_materialization_front_runner_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "selected_extension_independent_extra_q_range_source_materialization_"
    "inventory_audited_helper_implementation_primary_hybrid_reserve_"
    "secondary_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "selected_extension_independent_extra_q_range_source_materialization_"
    "helper_implementation_contract_derived_implementation_primary_pack_"
    "refresh_secondary_gate"
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


# 関数: front-runner helper-implementation contract の式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used in the source-materialization front-runner audit."""
    return {
        "augmented_q_window": "Q_aug^(pilot-HS) := Q_ret union Q_ext^(ind)",
        "helper_contract": (
            "build_selected_extension_solver_side_extra_q_range_pack(...) := "
            "extend build_selected_extension_solver_side_deformation_pack(...) "
            "from Q_ret to Q_aug^(pilot-HS) while preserving all retained-q "
            "checkpoint values exactly"
        ),
        "effective_kernel_pack": (
            "K_eff^(pilot-HS,qext)[Q_aug] := extend "
            "K_eff_deform_transverse_projector_pack from Q_ret to Q_aug^(pilot-HS)"
        ),
        "transverse_scalar_pack": (
            "Z_eff^(pilot-HS,qext,T)(q) := "
            "(1/2) tr(Pi_T(q) K_eff^(pilot-HS,qext)(q) Pi_T(q))"
        ),
        "blind_output_pack": (
            "O_qext_sel^(pilot-HS) := {K_eff^(pilot-HS,qext)[Q_aug], "
            "Z_eff^(pilot-HS,qext,T)[Q_aug], F_blind^(pilot-HS,qext)[Q_aug], "
            "alpha_blind^(pilot-HS,q_theory), Delta_qext_sel^(pilot-HS)}"
        ),
        "residual_discriminator": (
            "Delta_qext_sel^(pilot-HS) := compare("
            "O_qext_sel^(pilot-HS), O_deform_sel^(pilot-HS), alpha_exact(q_theory))"
        ),
    }


# 関数: `.5359-.5362` を実行する。

def main() -> None:
    """Execute the source-materialization front-runner audit."""
    for path in (PRIOR_GATE, PRIOR_AUDIT):
        sign_base.require(path)

    prior_gate_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    prior_audit_summary = sign_base.read_json(PRIOR_AUDIT)["summary"]

    audit_selected = bool(
        prior_gate_summary[
            "gate_b_updated_pack_selected_extension_independent_extra_q_range_source_materialization_front_runner_promoted_next"
        ]
        and prior_gate_summary["pack_update_required_now"]
    )
    retry_mode = bool(prior_gate_summary["retry_gate_computation_mode_selected"])
    non_surrogate_guard = bool(
        prior_gate_summary["failure_matrix_non_surrogate_guard_preserved"]
    )
    inventory_nonempty = bool(
        prior_gate_summary[
            "gate_a_updated_pack_exact_selected_extension_independent_extra_q_range_source_materialization_inventory_nonempty_available_now"
        ]
        and prior_audit_summary[
            "exact_selected_extension_independent_extra_q_range_source_materialization_inventory_nonempty_theorem_available_now"
        ]
    )
    front_runner_formula_explicit = bool(
        prior_audit_summary[
            "exact_selected_extension_independent_extra_q_range_source_materialization_front_runner_route_formula_available_now"
        ]
        and prior_audit_summary[
            "exact_selected_extension_independent_extra_q_range_source_materialization_front_runner_compatibility_theorem_available_now"
        ]
    )
    deformation_helper_available = bool(DEFORMATION_HELPER.exists())
    extra_q_helper_available_now = bool(EXTRA_Q_RANGE_HELPER.exists())

    contract_formula_explicit = bool(
        audit_selected
        and retry_mode
        and non_surrogate_guard
        and inventory_nonempty
        and front_runner_formula_explicit
        and deformation_helper_available
        and not extra_q_helper_available_now
    )
    exact_selected_extension_independent_extra_q_range_source_materialization_front_runner_helper_implementation_contract_formula_available_now = bool(
        contract_formula_explicit
    )
    exact_selected_extension_independent_extra_q_range_source_materialization_front_runner_augmented_q_window_formula_available_now = bool(
        contract_formula_explicit
    )
    exact_selected_extension_independent_extra_q_range_source_materialization_front_runner_output_pack_formula_available_now = bool(
        contract_formula_explicit
    )
    exact_selected_extension_independent_extra_q_range_source_materialization_front_runner_residual_discriminator_formula_available_now = bool(
        contract_formula_explicit
    )
    selected_extension_independent_extra_q_range_source_materialization_implementation_primary_admissible_now = bool(
        contract_formula_explicit
    )
    updated_pack_selected_extension_independent_extra_q_range_source_materialization_implementation_followup_required = bool(
        selected_extension_independent_extra_q_range_source_materialization_implementation_primary_admissible_now
    )
    updated_pack_same_schema_selected_extension_independent_extra_q_range_source_materialization_front_runner_replay_detected_now = (
        False
    )
    farther_hybrid_continuation_reopen_required_now = False

    rows = [
        sign_base.row(
            "updated_pack_selected_extension_independent_extra_q_range_source_materialization_front_runner_audit_selected",
            "pass" if audit_selected else "reject",
            "updated-pack selected-extension independent extra-q-range source-materialization front-runner audit selected",
            sign_base.truth(audit_selected),
            "This branch is worth running only after the source-materialization inventory has already promoted one helper-implementation front-runner route.",
        ),
        sign_base.row(
            "retry_gate_computation_mode_selected",
            "pass" if retry_mode else "reject",
            "retry gate computation mode selected",
            sign_base.truth(retry_mode),
            "The active lane stays computation-first and does not reopen closed evidence candidates or hybrid reserve bookkeeping.",
        ),
        sign_base.row(
            "failure_matrix_non_surrogate_guard_preserved",
            "pass" if non_surrogate_guard else "reject",
            "failure-matrix non-surrogate guard preserved",
            sign_base.truth(non_surrogate_guard),
            "The front-runner route remains honest only while helper/public/hybrid evidence candidates stay closed.",
        ),
        sign_base.row(
            "source_materialization_inventory_nonempty_now",
            "pass" if inventory_nonempty else "reject",
            "source-materialization inventory nonempty now",
            sign_base.truth(inventory_nonempty),
            "The front-runner route can be audited only after the source-materialization inventory itself is theorem-side nonempty.",
        ),
        sign_base.row(
            "front_runner_route_formula_explicit_now",
            "pass" if front_runner_formula_explicit else "reject",
            "front-runner route formula explicit now",
            sign_base.truth(front_runner_formula_explicit),
            "The promoted helper-implementation route must already be explicit before it can be turned into one concrete implementation contract.",
        ),
        sign_base.row(
            "selected_extension_solver_side_deformation_backend_available_now",
            "pass" if deformation_helper_available else "reject",
            "selected-extension solver-side deformation backend available now",
            sign_base.truth(deformation_helper_available),
            "The helper-implementation route is anchored to the already materialized selected-extension deformation backend.",
        ),
        sign_base.row(
            "selected_extension_solver_side_extra_q_range_helper_available_now",
            "pass" if extra_q_helper_available_now else "reject",
            "selected-extension solver-side extra-q-range helper available now",
            sign_base.truth(extra_q_helper_available_now),
            "Reject means the live blocker is still actual helper implementation, not theorem-side contract syntax.",
        ),
        sign_base.row(
            "exact_selected_extension_independent_extra_q_range_source_materialization_front_runner_helper_implementation_contract_formula_available_now",
            "pass"
            if exact_selected_extension_independent_extra_q_range_source_materialization_front_runner_helper_implementation_contract_formula_available_now
            else "reject",
            "exact selected-extension independent extra-q-range source-materialization front-runner helper-implementation contract formula available now",
            sign_base.truth(
                exact_selected_extension_independent_extra_q_range_source_materialization_front_runner_helper_implementation_contract_formula_available_now
            ),
            "The front-runner route is now written honestly as one concrete helper implementation contract on the fixed selected extension.",
        ),
        sign_base.row(
            "exact_selected_extension_independent_extra_q_range_source_materialization_front_runner_output_pack_formula_available_now",
            "pass"
            if exact_selected_extension_independent_extra_q_range_source_materialization_front_runner_output_pack_formula_available_now
            else "reject",
            "exact selected-extension independent extra-q-range source-materialization front-runner output-pack formula available now",
            sign_base.truth(
                exact_selected_extension_independent_extra_q_range_source_materialization_front_runner_output_pack_formula_available_now
            ),
            "The missing helper is now constrained to produce one explicit selected-extension extra-q output pack on the augmented q window.",
        ),
        sign_base.row(
            "selected_extension_independent_extra_q_range_source_materialization_implementation_primary_admissible_now",
            "pass"
            if selected_extension_independent_extra_q_range_source_materialization_implementation_primary_admissible_now
            else "reject",
            "selected-extension independent extra-q-range source-materialization implementation primary admissible now",
            sign_base.truth(
                selected_extension_independent_extra_q_range_source_materialization_implementation_primary_admissible_now
            ),
            "The helper-implementation route is now the honest primary way to try actual extra-q materialization.",
        ),
        sign_base.row(
            "updated_pack_selected_extension_independent_extra_q_range_source_materialization_implementation_followup_required",
            "pass"
            if updated_pack_selected_extension_independent_extra_q_range_source_materialization_implementation_followup_required
            else "reject",
            "updated-pack selected-extension independent extra-q-range source-materialization implementation followup required",
            sign_base.truth(
                updated_pack_selected_extension_independent_extra_q_range_source_materialization_implementation_followup_required
            ),
            "The honest next blocker is actual helper implementation, not another route-family replay.",
        ),
        sign_base.row(
            "updated_pack_same_schema_selected_extension_independent_extra_q_range_source_materialization_front_runner_replay_detected_now",
            "pass"
            if updated_pack_same_schema_selected_extension_independent_extra_q_range_source_materialization_front_runner_replay_detected_now
            else "reject",
            "updated-pack same-schema selected-extension independent extra-q-range source-materialization front-runner replay detected now",
            sign_base.truth(
                updated_pack_same_schema_selected_extension_independent_extra_q_range_source_materialization_front_runner_replay_detected_now
            ),
            "False means the blocker genuinely moved from inventory syntax to one concrete helper implementation contract.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "selected_extension_label": prior_audit_summary["selected_extension_label"],
        "solver_side_deformation_label": prior_audit_summary["solver_side_deformation_label"],
        "q_theory_over_m0": float(prior_audit_summary["q_theory_over_m0"]),
        "blind_F_deform_at_q_theory": float(
            prior_audit_summary["blind_F_deform_at_q_theory"]
        ),
        "blind_alpha_deform_at_q_theory": float(
            prior_audit_summary["blind_alpha_deform_at_q_theory"]
        ),
        "delta_alpha_sel_deform_exact": float(
            prior_audit_summary["delta_alpha_sel_deform_exact"]
        ),
        "relative_exact_residual_deform": float(
            prior_audit_summary["relative_exact_residual_deform"]
        ),
        "retry_gate_computation_mode_selected": retry_mode,
        "failure_matrix_non_surrogate_guard_preserved": non_surrogate_guard,
        "source_materialization_inventory_nonempty_now": inventory_nonempty,
        "front_runner_route_formula_explicit_now": front_runner_formula_explicit,
        "selected_extension_solver_side_deformation_backend_available_now": deformation_helper_available,
        "selected_extension_solver_side_extra_q_range_helper_available_now": extra_q_helper_available_now,
        "exact_selected_extension_independent_extra_q_range_source_materialization_front_runner_helper_implementation_contract_formula_available_now": exact_selected_extension_independent_extra_q_range_source_materialization_front_runner_helper_implementation_contract_formula_available_now,
        "exact_selected_extension_independent_extra_q_range_source_materialization_front_runner_augmented_q_window_formula_available_now": exact_selected_extension_independent_extra_q_range_source_materialization_front_runner_augmented_q_window_formula_available_now,
        "exact_selected_extension_independent_extra_q_range_source_materialization_front_runner_output_pack_formula_available_now": exact_selected_extension_independent_extra_q_range_source_materialization_front_runner_output_pack_formula_available_now,
        "exact_selected_extension_independent_extra_q_range_source_materialization_front_runner_residual_discriminator_formula_available_now": exact_selected_extension_independent_extra_q_range_source_materialization_front_runner_residual_discriminator_formula_available_now,
        "selected_extension_independent_extra_q_range_source_materialization_implementation_primary_admissible_now": selected_extension_independent_extra_q_range_source_materialization_implementation_primary_admissible_now,
        "updated_pack_selected_extension_independent_extra_q_range_source_materialization_implementation_followup_required": updated_pack_selected_extension_independent_extra_q_range_source_materialization_implementation_followup_required,
        "updated_pack_same_schema_selected_extension_independent_extra_q_range_source_materialization_front_runner_replay_detected_now": updated_pack_same_schema_selected_extension_independent_extra_q_range_source_materialization_front_runner_replay_detected_now,
        "farther_hybrid_continuation_reopen_required_now": farther_hybrid_continuation_reopen_required_now,
        "pack_update_required_now": updated_pack_selected_extension_independent_extra_q_range_source_materialization_implementation_followup_required,
        "selected_primary_completion_lane": "updated_pack_selected_extension_independent_extra_q_range_source_materialization_front_runner_gate",
        "selected_secondary_completion_lane": "updated_pack_selected_extension_independent_extra_q_range_source_materialization_implementation_audit",
        "selected_reserve_completion_lane": "farther_hybrid_reserve_only_until_source_route_materialized",
        "selected_next_generation_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_selected_extension_independent_extra_q_range_source_materialization_front_runner_gate",
        "recommended_next_route_or_none": "8.7.56.5363",
        "selected_followup_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_selected_extension_independent_extra_q_range_source_materialization_implementation_audit",
        "selected_followup_route_or_none": "8.7.56.5367",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5361",
        STEP_NAME + " declaration gate",
        {
            "source_files": {
                "prior_gate": sign_base.display_path(PRIOR_GATE),
                "prior_audit": sign_base.display_path(PRIOR_AUDIT),
                "deformation_helper": sign_base.display_path(DEFORMATION_HELPER),
                "extra_q_helper_candidate": sign_base.display_path(EXTRA_Q_RANGE_HELPER),
            },
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "8.7.56.5363",
                "followup_route": "8.7.56.5367",
            },
        },
        rows,
        summary,
        {
            "overall_status": "vector_qball_form_factor_selected_extension_independent_extra_q_range_source_materialization_front_runner_declared",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {"formulae": build_formulae()},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(
        "[done] 8.7.56.5359-5362 selected-extension independent extra-q-range source-materialization front-runner completed"
    )
    print(f"[done] declaration: {declaration_paths['json']}")


# 関数: CLI entrypoint から front-runner audit を実行する。

if __name__ == "__main__":
    main()
