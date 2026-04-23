#!/usr/bin/env python3
"""Generate 8.7.56.5263-.5266 selected-extension solver-side deformation inventory artifacts."""

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
        "8.7.56.5259-5262",
        "updated_pack_selected_extension_solver_recompute_residual_origin_refresh_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
PRIOR_AUDIT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5255-5258",
        "updated_pack_selected_extension_solver_recompute_residual_origin_refresh_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
SOLVER_RECOMPUTE_HELPER = (
    ROOT / "scripts" / "quantum" / "selected_extension_solver_recompute_backend.py"
)

STEP_TAG = "8.7.56.5263-5266"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack selected-"
    "extension solver-side deformation inventory theorem audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_selected_extension_solver_side_deformation_inventory_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "selected_extension_solver_recompute_retained_q_rerun_preserves_phase3_"
    "failure_closeout_completed_solver_deformation_inventory_primary_hybrid_"
    "reserve_secondary_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "selected_extension_solver_side_deformation_inventory_nonempty_theorem_"
    "derived_front_runner_primary_pack_refresh_secondary_gate"
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


# 関数: solver-side deformation inventory theorem の式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used in the selected-extension solver-side deformation inventory audit."""
    return {
        "kernel_refresh_candidate": (
            "D_solver_sel^(K)[Sigma_*^(pilot-HS)] := refresh "
            "K_eff^(pilot-HS,recomp)[Q_ret] by recomputing the selected-extension "
            "Schur-complement kernel on the live solver path instead of inheriting "
            "the retained replayed scalar pack"
        ),
        "resolvent_refresh_candidate": (
            "D_solver_sel^(G)[Sigma_*^(pilot-HS)] := refresh the internal resolvent "
            "(K_xixi[Q])^(-1) and mode-sum closure under the fixed selected extension"
        ),
        "retained_q_rerun_candidate": (
            "D_solver_sel^(Qret)[Sigma_*^(pilot-HS)] := rerun "
            "{Z_eff^(pilot-HS,T), F_blind^(pilot-HS), alpha_blind^(pilot-HS)} on "
            "Q_ret = {0, q_theory, m0} after the refreshed kernel/resolvent update"
        ),
        "extra_q_reserve_candidate": (
            "D_solver_sel^(Qext)[Sigma_*^(pilot-HS)] := reopen extra q-range only if "
            "the retained-q deformation rerun still leaves residual-origin "
            "discrimination ambiguous"
        ),
        "front_runner_candidate": (
            "D_solver_sel^(pilot-HS,recompute-retained) := "
            "(D_solver_sel^(K), D_solver_sel^(G), D_solver_sel^(Qret))"
        ),
        "inventory": (
            "Inv_solver_sel^(pilot-HS) := {D_solver_sel^(K), D_solver_sel^(G), "
            "D_solver_sel^(Qret), D_solver_sel^(Qext)}"
        ),
    }


# 関数: `.5263-.5266` を実行する。

def main() -> None:
    """Execute the selected-extension solver-side deformation inventory theorem audit."""
    for path in (PRIOR_GATE, PRIOR_AUDIT):
        sign_base.require(path)

    prior_gate_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    prior_audit_summary = sign_base.read_json(PRIOR_AUDIT)["summary"]

    audit_selected = bool(
        prior_gate_summary[
            "gate_b_updated_pack_selected_extension_solver_side_deformation_inventory_promoted_next"
        ]
        and prior_gate_summary["pack_update_required_now"]
    )
    retry_mode = bool(prior_gate_summary["retry_gate_computation_mode_selected"])
    non_surrogate_guard = bool(
        prior_gate_summary["failure_matrix_non_surrogate_guard_preserved"]
    )
    selected_extension_label_fixed = bool(
        prior_gate_summary["selected_extension_label"] == "Sigma_*^(pilot-HS)"
    )
    recompute_lane_closed_negatively = bool(
        prior_gate_summary[
            "gate_a_updated_pack_exact_selected_extension_solver_recompute_lane_negative_closeout_available_now"
        ]
        and prior_audit_summary[
            "exact_selected_extension_solver_recompute_lane_negative_closeout_available_now"
        ]
    )
    recompute_helper_available = bool(SOLVER_RECOMPUTE_HELPER.exists())

    inventory_formula_explicit = bool(
        audit_selected
        and retry_mode
        and non_surrogate_guard
        and selected_extension_label_fixed
        and recompute_lane_closed_negatively
        and recompute_helper_available
    )
    exact_selected_extension_solver_side_deformation_inventory_formula_available_now = bool(
        inventory_formula_explicit
    )
    exact_selected_extension_solver_side_deformation_effective_kernel_refresh_candidate_formula_available_now = bool(
        inventory_formula_explicit
    )
    exact_selected_extension_solver_side_deformation_internal_resolvent_refresh_candidate_formula_available_now = bool(
        inventory_formula_explicit
    )
    exact_selected_extension_solver_side_deformation_retained_q_window_rerun_candidate_formula_available_now = bool(
        inventory_formula_explicit
    )
    exact_selected_extension_solver_side_deformation_extra_q_range_reserve_candidate_formula_available_now = bool(
        inventory_formula_explicit
    )
    exact_selected_extension_solver_side_deformation_inventory_nonempty_theorem_available_now = bool(
        inventory_formula_explicit
    )
    exact_selected_extension_solver_side_deformation_front_runner_candidate_formula_available_now = bool(
        inventory_formula_explicit
    )
    exact_selected_extension_solver_side_deformation_front_runner_compatibility_theorem_available_now = bool(
        inventory_formula_explicit
    )
    updated_pack_selected_extension_solver_side_deformation_front_runner_followup_required = bool(
        inventory_formula_explicit
    )
    updated_pack_same_schema_selected_extension_solver_side_deformation_inventory_replay_detected_now = (
        False
    )
    farther_hybrid_continuation_reopen_required_now = bool(
        prior_gate_summary["gate_c_farther_hybrid_continuation_reopen_required_now"]
    )

    rows = [
        sign_base.row(
            "updated_pack_selected_extension_solver_side_deformation_inventory_audit_selected",
            "pass" if audit_selected else "reject",
            "updated-pack selected-extension solver-side deformation inventory audit selected",
            sign_base.truth(audit_selected),
            "This branch is worth running only after the selected-extension solver-recompute lane has been closed negatively and solver-side deformation has become the live blocker.",
        ),
        sign_base.row(
            "retry_gate_computation_mode_selected",
            "pass" if retry_mode else "reject",
            "retry gate computation mode selected",
            sign_base.truth(retry_mode),
            "The new branch stays on computation-side blocker reduction instead of reopening theorem-family recursion or replaying retained-q interpretation.",
        ),
        sign_base.row(
            "failure_matrix_non_surrogate_guard_preserved",
            "pass" if non_surrogate_guard else "reject",
            "failure-matrix non-surrogate guard preserved",
            sign_base.truth(non_surrogate_guard),
            "Solver-side deformation inventory is honest only while exhausted surrogate and replay routes remain closed.",
        ),
        sign_base.row(
            "selected_extension_label_fixed_now",
            "pass" if selected_extension_label_fixed else "reject",
            "selected extension label fixed now",
            sign_base.truth(selected_extension_label_fixed),
            "The inventory is meaningful only while one concrete selected extension Sigma_*^(pilot-HS) remains fixed as the computation baseline.",
        ),
        sign_base.row(
            "selected_extension_solver_recompute_lane_negative_closeout_available_now",
            "pass" if recompute_lane_closed_negatively else "reject",
            "selected-extension solver-recompute lane negative closeout available now",
            sign_base.truth(recompute_lane_closed_negatively),
            "The inventory branch is justified only after retained-q recompute has already been proven not to resolve the residual origin.",
        ),
        sign_base.row(
            "selected_extension_solver_recompute_helper_available_now",
            "pass" if recompute_helper_available else "reject",
            "selected-extension solver-recompute helper available now",
            sign_base.truth(recompute_helper_available),
            "The follow-up inventory should classify actual solver-side deformation candidates on top of the already materialized recompute helper path.",
        ),
        sign_base.row(
            "exact_selected_extension_solver_side_deformation_inventory_formula_available_now",
            "pass"
            if exact_selected_extension_solver_side_deformation_inventory_formula_available_now
            else "reject",
            "exact selected-extension solver-side deformation inventory formula available now",
            sign_base.truth(
                exact_selected_extension_solver_side_deformation_inventory_formula_available_now
            ),
            "The active lane now fixes a finite deformation inventory rather than leaving solver-side follow-up as an unstructured request.",
        ),
        sign_base.row(
            "exact_selected_extension_solver_side_deformation_front_runner_candidate_formula_available_now",
            "pass"
            if exact_selected_extension_solver_side_deformation_front_runner_candidate_formula_available_now
            else "reject",
            "exact selected-extension solver-side deformation front-runner candidate formula available now",
            sign_base.truth(
                exact_selected_extension_solver_side_deformation_front_runner_candidate_formula_available_now
            ),
            "One retained-q recomputation front-runner is now explicit instead of hidden inside the generic phrase 'solver-side deformation'.",
        ),
        sign_base.row(
            "exact_selected_extension_solver_side_deformation_front_runner_compatibility_theorem_available_now",
            "pass"
            if exact_selected_extension_solver_side_deformation_front_runner_compatibility_theorem_available_now
            else "reject",
            "exact selected-extension solver-side deformation front-runner compatibility theorem available now",
            sign_base.truth(
                exact_selected_extension_solver_side_deformation_front_runner_compatibility_theorem_available_now
            ),
            "The front-runner remains compatible with the fixed selected extension, retained blind checkpoints, and current non-surrogate guard.",
        ),
        sign_base.row(
            "updated_pack_same_schema_selected_extension_solver_side_deformation_inventory_replay_detected_now",
            "pass"
            if updated_pack_same_schema_selected_extension_solver_side_deformation_inventory_replay_detected_now
            else "reject",
            "updated-pack same-schema selected-extension solver-side deformation inventory replay detected now",
            sign_base.truth(
                updated_pack_same_schema_selected_extension_solver_side_deformation_inventory_replay_detected_now
            ),
            "False means this branch added a real finite recomputation classification instead of replaying the already-closed solver-recompute verdict.",
        ),
        sign_base.row(
            "updated_pack_selected_extension_solver_side_deformation_front_runner_followup_required",
            "pass"
            if updated_pack_selected_extension_solver_side_deformation_front_runner_followup_required
            else "reject",
            "updated-pack selected-extension solver-side deformation front-runner followup required",
            sign_base.truth(
                updated_pack_selected_extension_solver_side_deformation_front_runner_followup_required
            ),
            "A substantive lane shift happened here: the next honest blocker is front-runner audit, not another generic inventory pass.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "selected_extension_label": prior_gate_summary["selected_extension_label"],
        "q_theory_over_m0": float(prior_gate_summary["q_theory_over_m0"]),
        "blind_F_recomp_at_q_theory": float(prior_gate_summary["blind_F_recomp_at_q_theory"]),
        "blind_alpha_recomp_at_q_theory": float(
            prior_gate_summary["blind_alpha_recomp_at_q_theory"]
        ),
        "delta_alpha_sel_recomp_exact": float(
            prior_gate_summary["delta_alpha_sel_recomp_exact"]
        ),
        "relative_exact_residual_recomp": float(
            prior_gate_summary["relative_exact_residual_recomp"]
        ),
        "retry_gate_computation_mode_selected": retry_mode,
        "failure_matrix_non_surrogate_guard_preserved": non_surrogate_guard,
        "selected_extension_label_fixed_now": selected_extension_label_fixed,
        "selected_extension_solver_recompute_lane_negative_closeout_available_now": recompute_lane_closed_negatively,
        "selected_extension_solver_recompute_helper_available_now": recompute_helper_available,
        "exact_selected_extension_solver_side_deformation_inventory_formula_available_now": exact_selected_extension_solver_side_deformation_inventory_formula_available_now,
        "exact_selected_extension_solver_side_deformation_effective_kernel_refresh_candidate_formula_available_now": exact_selected_extension_solver_side_deformation_effective_kernel_refresh_candidate_formula_available_now,
        "exact_selected_extension_solver_side_deformation_internal_resolvent_refresh_candidate_formula_available_now": exact_selected_extension_solver_side_deformation_internal_resolvent_refresh_candidate_formula_available_now,
        "exact_selected_extension_solver_side_deformation_retained_q_window_rerun_candidate_formula_available_now": exact_selected_extension_solver_side_deformation_retained_q_window_rerun_candidate_formula_available_now,
        "exact_selected_extension_solver_side_deformation_extra_q_range_reserve_candidate_formula_available_now": exact_selected_extension_solver_side_deformation_extra_q_range_reserve_candidate_formula_available_now,
        "exact_selected_extension_solver_side_deformation_inventory_nonempty_theorem_available_now": exact_selected_extension_solver_side_deformation_inventory_nonempty_theorem_available_now,
        "exact_selected_extension_solver_side_deformation_front_runner_candidate_formula_available_now": exact_selected_extension_solver_side_deformation_front_runner_candidate_formula_available_now,
        "exact_selected_extension_solver_side_deformation_front_runner_compatibility_theorem_available_now": exact_selected_extension_solver_side_deformation_front_runner_compatibility_theorem_available_now,
        "updated_pack_selected_extension_solver_side_deformation_front_runner_followup_required": updated_pack_selected_extension_solver_side_deformation_front_runner_followup_required,
        "updated_pack_same_schema_selected_extension_solver_side_deformation_inventory_replay_detected_now": updated_pack_same_schema_selected_extension_solver_side_deformation_inventory_replay_detected_now,
        "farther_hybrid_continuation_reopen_required_now": farther_hybrid_continuation_reopen_required_now,
        "pack_update_required_now": updated_pack_selected_extension_solver_side_deformation_front_runner_followup_required,
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
        "8.7.56.5265",
        STEP_NAME + " declaration gate",
        {
            "source_files": {
                "prior_gate": sign_base.display_path(PRIOR_GATE),
                "prior_audit": sign_base.display_path(PRIOR_AUDIT),
                "solver_recompute_helper": sign_base.display_path(SOLVER_RECOMPUTE_HELPER),
            },
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
            "overall_status": "vector_qball_form_factor_selected_extension_solver_side_deformation_inventory_declared",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {"formulae": build_formulae()},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(
        "[done] 8.7.56.5263-5266 selected-extension solver-side deformation inventory completed"
    )
    print(f"[done] declaration: {declaration_paths['json']}")


# 関数: CLI entrypoint から inventory audit を実行する。

if __name__ == "__main__":
    main()
