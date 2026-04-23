#!/usr/bin/env python3
"""Generate 8.7.56.5367-.5370 source-materialization implementation artifacts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.quantum.t2a_1843 as sign_base
from scripts.quantum.selected_extension_solver_side_extra_q_range_backend import (
    build_selected_extension_solver_side_extra_q_range_pack,
)
from scripts.utils.windows_length_policy import build_compact_artifact_stem
from scripts.utils.windows_length_policy import build_metrics_paths


PUBLIC_OUT = ROOT / "output" / "public" / "quantum"
PRIOR_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5363-5366",
        "updated_pack_selected_extension_independent_extra_q_range_source_materialization_front_runner_gate",
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

STEP_TAG = "8.7.56.5367-5370"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack selected-"
    "extension independent extra-q-range source-materialization implementation audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_selected_extension_independent_extra_q_range_source_materialization_implementation_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "selected_extension_independent_extra_q_range_source_materialization_"
    "front_runner_audited_implementation_primary_hybrid_reserve_secondary_"
    "next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "selected_extension_independent_extra_q_range_source_materialization_"
    "implementation_derived_numeric_rerun_primary_pack_refresh_secondary_"
    "gate"
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


# 関数: source-materialization implementation audit で使う式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used in the selected-extension source-materialization implementation audit."""
    return {
        "implementation_call": (
            "C_qsrc_sel,impl^(pilot-HS) := "
            "build_selected_extension_solver_side_extra_q_range_pack(ell_values=(1,2,3))"
        ),
        "independent_window": "Q_ext^(ind) := {q : q in Q_aug^(pilot-HS) and q not in Q_ret}",
        "materialized_output_pack": (
            "O_qext_sel,impl^(pilot-HS) := {K_eff^(pilot-HS,qext)[Q_aug], "
            "Z_eff^(pilot-HS,qext,T)[Q_aug], F_blind^(pilot-HS,qext)[Q_aug], "
            "alpha_blind^(pilot-HS,q_theory), delta_alpha_sel^(pilot-HS,qext)}"
        ),
        "retained_surface_check": (
            "retained_surface_preserved iff O_qext_sel,impl matches "
            "O_deform_sel^(pilot-HS) on Q_ret"
        ),
    }


# 関数: `.5367-.5370` を実行する。

def main() -> None:
    """Execute the selected-extension source-materialization implementation audit."""
    for path in (PRIOR_GATE, SELECTED_EXTENSION_GATE):
        sign_base.require(path)

    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    selected_summary = sign_base.read_json(SELECTED_EXTENSION_GATE)["summary"]

    audit_selected = bool(
        prior_summary[
            "gate_b_updated_pack_selected_extension_independent_extra_q_range_source_materialization_implementation_promoted_next"
        ]
        and prior_summary[
            "gate_a_updated_pack_exact_selected_extension_independent_extra_q_range_source_materialization_front_runner_contract_available_now"
        ]
    )
    selected_extension_available = bool(
        selected_summary[
            "gate_a_updated_pack_exact_external_rule_selector_selected_extension_available_now"
        ]
        and selected_summary[
            "exact_external_rule_selector_selected_extension_available_now"
        ]
    )
    retry_mode = bool(prior_summary["retry_gate_computation_mode_selected"])
    non_surrogate_guard = bool(
        prior_summary["failure_matrix_non_surrogate_guard_preserved"]
    )

    qext_pack = build_selected_extension_solver_side_extra_q_range_pack()
    helper_module_available_now = True
    q_ext_ind_nonempty_now = bool(qext_pack["q_ext_ind_nonempty_now"])
    q_aug_materialized_now = bool(qext_pack["q_aug_materialized_now"])
    retained_surface_preserved_now = bool(qext_pack["retained_surface_preserved_now"])
    k_eff_qext_materialized_now = bool(
        set(qext_pack["q_aug_window"]) <= set(qext_pack["K_eff_qext_transverse_projector_pack"])
    )
    z_eff_qext_materialized_now = bool(
        set(qext_pack["q_aug_window"]) <= set(qext_pack["Z_eff_qext_transverse_scalar_pack"])
    )
    f_blind_qext_materialized_now = bool(
        set(qext_pack["q_aug_window"]) <= set(qext_pack["F_blind_qext_pack"])
    )
    alpha_blind_qext_available_now = bool("alpha_blind_qext_at_q_theory" in qext_pack)
    delta_alpha_sel_qext_available_now = bool(
        "delta_alpha_sel_qext_exact" in qext_pack
    )

    exact_selected_extension_independent_extra_q_range_source_materialization_implementation_formula_available_now = bool(
        audit_selected
        and selected_extension_available
        and retry_mode
        and non_surrogate_guard
    )
    exact_selected_extension_independent_extra_q_range_source_materialization_materialized_output_pack_available_now = bool(
        helper_module_available_now
        and q_aug_materialized_now
        and k_eff_qext_materialized_now
        and z_eff_qext_materialized_now
        and f_blind_qext_materialized_now
        and alpha_blind_qext_available_now
        and delta_alpha_sel_qext_available_now
    )
    exact_selected_extension_independent_extra_q_range_source_materialization_retained_surface_preservation_theorem_available_now = bool(
        exact_selected_extension_independent_extra_q_range_source_materialization_materialized_output_pack_available_now
        and retained_surface_preserved_now
    )
    exact_selected_extension_independent_extra_q_range_source_materialization_independent_extra_q_support_materialized_theorem_available_now = bool(
        exact_selected_extension_independent_extra_q_range_source_materialization_materialized_output_pack_available_now
        and q_ext_ind_nonempty_now
    )
    exact_selected_extension_independent_extra_q_range_source_materialization_implementation_available_now = bool(
        exact_selected_extension_independent_extra_q_range_source_materialization_implementation_formula_available_now
        and exact_selected_extension_independent_extra_q_range_source_materialization_materialized_output_pack_available_now
        and exact_selected_extension_independent_extra_q_range_source_materialization_retained_surface_preservation_theorem_available_now
        and exact_selected_extension_independent_extra_q_range_source_materialization_independent_extra_q_support_materialized_theorem_available_now
    )
    updated_pack_selected_extension_independent_extra_q_range_source_materialization_numeric_rerun_followup_required = bool(
        exact_selected_extension_independent_extra_q_range_source_materialization_implementation_available_now
    )
    updated_pack_same_schema_selected_extension_independent_extra_q_range_source_materialization_implementation_replay_detected_now = (
        False
    )
    farther_hybrid_continuation_reopen_required_now = False

    rows = [
        sign_base.row(
            "updated_pack_selected_extension_independent_extra_q_range_source_materialization_implementation_audit_selected",
            "pass" if audit_selected else "reject",
            "updated-pack selected-extension independent extra-q-range source-materialization implementation audit selected",
            sign_base.truth(audit_selected),
            "This branch is worth running only after one literal front-runner helper contract is already official.",
        ),
        sign_base.row(
            "selected_extension_available_now",
            "pass" if selected_extension_available else "reject",
            "selected extension available now",
            sign_base.truth(selected_extension_available),
            "Actual source-materialization implementation is meaningful only while Sigma_*^(pilot-HS) remains official.",
        ),
        sign_base.row(
            "helper_module_available_now",
            "pass" if helper_module_available_now else "reject",
            "selected-extension source-materialization helper module available now",
            sign_base.truth(helper_module_available_now),
            "The actual implementation is now realized as one reusable helper that extends the deformation pack to Q_aug.",
        ),
        sign_base.row(
            "q_ext_ind_nonempty_now",
            "pass" if q_ext_ind_nonempty_now else "reject",
            "independent extra-q support nonempty now",
            sign_base.truth(q_ext_ind_nonempty_now),
            "The helper now materializes explicit extra-q checkpoints beyond Q_ret instead of staying trapped in the retained window.",
        ),
        sign_base.row(
            "q_aug_materialized_now",
            "pass" if q_aug_materialized_now else "reject",
            "augmented q window materialized now",
            sign_base.truth(q_aug_materialized_now),
            "The helper now materializes one actual Q_aug^(pilot-HS) surface rather than only a theorem-side union symbol.",
        ),
        sign_base.row(
            "retained_surface_preserved_now",
            "pass" if retained_surface_preserved_now else "reject",
            "retained surface preserved now",
            sign_base.truth(retained_surface_preserved_now),
            "The first source-materialization helper must preserve every retained-q checkpoint exactly.",
        ),
        sign_base.row(
            "k_eff_qext_materialized_now",
            "pass" if k_eff_qext_materialized_now else "reject",
            "K_eff qext pack materialized now",
            sign_base.truth(k_eff_qext_materialized_now),
            "The helper now materializes an effective-kernel pack on the augmented q window.",
        ),
        sign_base.row(
            "z_eff_qext_materialized_now",
            "pass" if z_eff_qext_materialized_now else "reject",
            "Z_eff qext transverse-scalar pack materialized now",
            sign_base.truth(z_eff_qext_materialized_now),
            "The helper now materializes the transverse scalarization on every augmented q label.",
        ),
        sign_base.row(
            "f_blind_qext_materialized_now",
            "pass" if f_blind_qext_materialized_now else "reject",
            "F_blind qext pack materialized now",
            sign_base.truth(f_blind_qext_materialized_now),
            "The helper now exposes one actual blind form-factor pack on Q_aug.",
        ),
        sign_base.row(
            "alpha_blind_qext_available_now",
            "pass" if alpha_blind_qext_available_now else "reject",
            "alpha_blind qext available now",
            sign_base.truth(alpha_blind_qext_available_now),
            "The q_theory blind alpha remains explicit in the augmented output pack.",
        ),
        sign_base.row(
            "delta_alpha_sel_qext_available_now",
            "pass" if delta_alpha_sel_qext_available_now else "reject",
            "delta alpha qext available now",
            sign_base.truth(delta_alpha_sel_qext_available_now),
            "The augmented output pack keeps one explicit residual discriminator against the retained exact scalar target.",
        ),
        sign_base.row(
            "exact_selected_extension_independent_extra_q_range_source_materialization_implementation_formula_available_now",
            "pass"
            if exact_selected_extension_independent_extra_q_range_source_materialization_implementation_formula_available_now
            else "reject",
            "exact selected-extension independent extra-q-range source-materialization implementation formula available now",
            sign_base.truth(
                exact_selected_extension_independent_extra_q_range_source_materialization_implementation_formula_available_now
            ),
            "The implementation call is now explicit as one helper-backed selected-extension source-materialization route.",
        ),
        sign_base.row(
            "exact_selected_extension_independent_extra_q_range_source_materialization_materialized_output_pack_available_now",
            "pass"
            if exact_selected_extension_independent_extra_q_range_source_materialization_materialized_output_pack_available_now
            else "reject",
            "exact selected-extension independent extra-q-range source-materialization materialized output pack available now",
            sign_base.truth(
                exact_selected_extension_independent_extra_q_range_source_materialization_materialized_output_pack_available_now
            ),
            "The contract output pack is no longer abstract; K_eff, Z_eff, F_blind, alpha_blind, and delta alpha are materialized now on Q_aug.",
        ),
        sign_base.row(
            "exact_selected_extension_independent_extra_q_range_source_materialization_retained_surface_preservation_theorem_available_now",
            "pass"
            if exact_selected_extension_independent_extra_q_range_source_materialization_retained_surface_preservation_theorem_available_now
            else "reject",
            "exact selected-extension independent extra-q-range source-materialization retained-surface preservation theorem available now",
            sign_base.truth(
                exact_selected_extension_independent_extra_q_range_source_materialization_retained_surface_preservation_theorem_available_now
            ),
            "The helper-backed source route preserves the fixed retained-q selected-extension surface exactly.",
        ),
        sign_base.row(
            "exact_selected_extension_independent_extra_q_range_source_materialization_independent_extra_q_support_materialized_theorem_available_now",
            "pass"
            if exact_selected_extension_independent_extra_q_range_source_materialization_independent_extra_q_support_materialized_theorem_available_now
            else "reject",
            "exact selected-extension independent extra-q-range source-materialization independent extra-q support materialized theorem available now",
            sign_base.truth(
                exact_selected_extension_independent_extra_q_range_source_materialization_independent_extra_q_support_materialized_theorem_available_now
            ),
            "The helper now materializes one nonempty Q_ext^(ind) beyond the retained-q window.",
        ),
        sign_base.row(
            "exact_selected_extension_independent_extra_q_range_source_materialization_implementation_available_now",
            "pass"
            if exact_selected_extension_independent_extra_q_range_source_materialization_implementation_available_now
            else "reject",
            "exact selected-extension independent extra-q-range source-materialization implementation available now",
            sign_base.truth(
                exact_selected_extension_independent_extra_q_range_source_materialization_implementation_available_now
            ),
            "One actual helper-backed selected-extension extra-q source-materialization route is now available.",
        ),
        sign_base.row(
            "updated_pack_same_schema_selected_extension_independent_extra_q_range_source_materialization_implementation_replay_detected_now",
            "pass"
            if updated_pack_same_schema_selected_extension_independent_extra_q_range_source_materialization_implementation_replay_detected_now
            else "reject",
            "updated-pack same-schema selected-extension independent extra-q-range source-materialization implementation replay detected now",
            sign_base.truth(
                updated_pack_same_schema_selected_extension_independent_extra_q_range_source_materialization_implementation_replay_detected_now
            ),
            "False means the blocker genuinely moved from missing helper existence to actual helper-backed surface evaluation.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "selected_extension_label": qext_pack["selected_extension_label"],
        "solver_side_deformation_label": qext_pack["solver_side_deformation_label"],
        "source_materialization_label": qext_pack["source_materialization_label"],
        "q_theory_over_m0": float(qext_pack["retained_q_window"]["q_theory_over_m0"]),
        "q_ext_ind_window": qext_pack["q_ext_ind_window"],
        "q_aug_window": qext_pack["q_aug_window"],
        "blind_F_qext_pack": qext_pack["F_blind_qext_pack"],
        "blind_alpha_qext_at_q_theory": float(
            qext_pack["alpha_blind_qext_at_q_theory"]
        ),
        "delta_alpha_sel_qext_exact": float(qext_pack["delta_alpha_sel_qext_exact"]),
        "relative_exact_residual_qext": float(qext_pack["relative_exact_residual_qext"]),
        "exact_selected_extension_independent_extra_q_range_source_materialization_implementation_formula_available_now": exact_selected_extension_independent_extra_q_range_source_materialization_implementation_formula_available_now,
        "exact_selected_extension_independent_extra_q_range_source_materialization_materialized_output_pack_available_now": exact_selected_extension_independent_extra_q_range_source_materialization_materialized_output_pack_available_now,
        "exact_selected_extension_independent_extra_q_range_source_materialization_retained_surface_preservation_theorem_available_now": exact_selected_extension_independent_extra_q_range_source_materialization_retained_surface_preservation_theorem_available_now,
        "exact_selected_extension_independent_extra_q_range_source_materialization_independent_extra_q_support_materialized_theorem_available_now": exact_selected_extension_independent_extra_q_range_source_materialization_independent_extra_q_support_materialized_theorem_available_now,
        "exact_selected_extension_independent_extra_q_range_source_materialization_implementation_available_now": exact_selected_extension_independent_extra_q_range_source_materialization_implementation_available_now,
        "selected_extension_solver_side_extra_q_range_helper_available_now": bool(
            qext_pack[
                "selected_extension_solver_side_extra_q_range_helper_available_now"
            ]
        ),
        "updated_pack_selected_extension_independent_extra_q_range_source_materialization_numeric_rerun_followup_required": updated_pack_selected_extension_independent_extra_q_range_source_materialization_numeric_rerun_followup_required,
        "retry_gate_computation_mode_selected": retry_mode,
        "failure_matrix_non_surrogate_guard_preserved": non_surrogate_guard,
        "updated_pack_same_schema_selected_extension_independent_extra_q_range_source_materialization_implementation_replay_detected_now": updated_pack_same_schema_selected_extension_independent_extra_q_range_source_materialization_implementation_replay_detected_now,
        "farther_hybrid_continuation_reopen_required_now": farther_hybrid_continuation_reopen_required_now,
        "pack_update_required_now": updated_pack_selected_extension_independent_extra_q_range_source_materialization_numeric_rerun_followup_required,
        "selected_primary_completion_lane": "updated_pack_selected_extension_independent_extra_q_range_source_materialization_implementation_gate",
        "selected_secondary_completion_lane": "updated_pack_selected_extension_independent_extra_q_range_source_materialization_numeric_rerun_audit",
        "selected_reserve_completion_lane": "farther_hybrid_reserve_only_until_qext_numeric_surface_evaluated",
        "selected_next_generation_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_selected_extension_independent_extra_q_range_source_materialization_implementation_gate",
        "recommended_next_route_or_none": "8.7.56.5371",
        "selected_followup_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_selected_extension_independent_extra_q_range_source_materialization_numeric_rerun_audit",
        "selected_followup_route_or_none": "8.7.56.5375",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5369",
        STEP_NAME + " declaration gate",
        {
            "source_files": {
                "prior_gate": sign_base.display_path(PRIOR_GATE),
                "selected_extension_gate": sign_base.display_path(
                    SELECTED_EXTENSION_GATE
                ),
            },
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "8.7.56.5371",
                "followup_route": "8.7.56.5375",
            },
        },
        rows,
        summary,
        {
            "overall_status": "vector_qball_form_factor_selected_extension_source_materialization_implementation_declared",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {"formulae": build_formulae()},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(
        f"[done] {STEP_TAG} selected-extension source-materialization implementation audit completed"
    )
    print(f"[done] declaration: {declaration_paths['json']}")


# 関数: CLI entrypoint から implementation audit を実行する。

if __name__ == "__main__":
    main()
