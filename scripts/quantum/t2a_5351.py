#!/usr/bin/env python3
"""Generate 8.7.56.5351-.5354 source-materialization inventory artifacts."""

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
        "8.7.56.5347-5350",
        "updated_pack_selected_extension_independent_extra_q_range_evidence_negative_closeout_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
PRIOR_AUDIT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5343-5346",
        "updated_pack_selected_extension_independent_extra_q_range_evidence_negative_closeout_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
DEFORMATION_HELPER = (
    ROOT / "scripts" / "quantum" / "selected_extension_solver_side_deformation_backend.py"
)
RECOMPUTE_HELPER = (
    ROOT / "scripts" / "quantum" / "selected_extension_solver_recompute_backend.py"
)
EXTRA_Q_RANGE_HELPER = (
    ROOT / "scripts" / "quantum" / "selected_extension_solver_side_extra_q_range_backend.py"
)

STEP_TAG = "8.7.56.5351-5354"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack selected-"
    "extension independent extra-q-range source-materialization inventory audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_selected_extension_independent_extra_q_range_source_materialization_inventory_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "selected_extension_independent_extra_q_range_evidence_negative_closeout_"
    "completed_source_materialization_inventory_primary_hybrid_reserve_"
    "secondary_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "selected_extension_independent_extra_q_range_source_materialization_"
    "inventory_nonempty_theorem_derived_helper_implementation_primary_pack_"
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


# 関数: source-materialization inventory theorem の式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used in the source-materialization inventory audit."""
    return {
        "independent_q_window": "Q_ext^(ind) := Q_probe^(cand) \\ Q_ret",
        "helper_implementation_route": (
            "R_qsrc^(helper_impl)[Sigma_*^(pilot-HS)] := materialize "
            "selected_extension_solver_side_extra_q_range_backend.py on top of "
            "selected_extension_solver_side_deformation_backend.py so that "
            "O_qext_sel^(pilot-HS)[Q_ret union Q_ext^(ind)] extends the fixed "
            "retained-q surface without changing Q_ret checkpoints"
        ),
        "public_checkpoint_synthesis_route": (
            "R_qsrc^(pub_sync)[Sigma_*^(pilot-HS)] := synthesize one canonical "
            "public checkpoint pack A_qext_sel^(pilot-HS)[Q_ext^(ind)] from the "
            "selected-extension helper stack and promote it into output/public/quantum"
        ),
        "hybrid_bridge_materialization_route": (
            "R_qsrc^(hyb_bridge)[Sigma_*^(pilot-HS)] := materialize one farther-"
            "hybrid bridge H_qext^(hyb)[Q_ext^(ind)] only after reserve-only "
            "hybrid continuation is independently promoted"
        ),
        "inventory": (
            "Inv_qsrc_sel^(pilot-HS) := {R_qsrc^(helper_impl), "
            "R_qsrc^(pub_sync), R_qsrc^(hyb_bridge)}"
        ),
        "front_runner": (
            "R_qsrc,front^(pilot-HS) := R_qsrc^(helper_impl)[Sigma_*^(pilot-HS)]"
        ),
    }


# 関数: `.5351-.5354` を実行する。

def main() -> None:
    """Execute the selected-extension source-materialization inventory audit."""
    for path in (PRIOR_GATE, PRIOR_AUDIT):
        sign_base.require(path)

    prior_gate_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    prior_audit_summary = sign_base.read_json(PRIOR_AUDIT)["summary"]

    audit_selected = bool(
        prior_gate_summary[
            "gate_b_updated_pack_selected_extension_independent_extra_q_range_source_materialization_inventory_promoted_next"
        ]
        and prior_gate_summary["pack_update_required_now"]
    )
    retry_mode = bool(prior_gate_summary["retry_gate_computation_mode_selected"])
    non_surrogate_guard = bool(
        prior_gate_summary["failure_matrix_non_surrogate_guard_preserved"]
    )
    selected_extension_label_fixed = bool(
        prior_audit_summary["selected_extension_label"] == "Sigma_*^(pilot-HS)"
    )
    evidence_lane_closed_negatively = bool(
        prior_gate_summary[
            "gate_a_updated_pack_exact_selected_extension_independent_extra_q_range_evidence_lane_negative_closeout_available_now"
        ]
        and prior_audit_summary[
            "exact_selected_extension_independent_extra_q_range_evidence_lane_negative_closeout_available_now"
        ]
    )
    deformation_helper_available = bool(DEFORMATION_HELPER.exists())
    recompute_helper_available = bool(RECOMPUTE_HELPER.exists())
    extra_q_helper_available_now = bool(EXTRA_Q_RANGE_HELPER.exists())

    inventory_formula_explicit = bool(
        audit_selected
        and retry_mode
        and non_surrogate_guard
        and selected_extension_label_fixed
        and evidence_lane_closed_negatively
        and deformation_helper_available
        and recompute_helper_available
    )
    exact_selected_extension_independent_extra_q_range_source_materialization_inventory_formula_available_now = bool(
        inventory_formula_explicit
    )
    exact_selected_extension_independent_extra_q_range_source_materialization_helper_implementation_route_formula_available_now = bool(
        inventory_formula_explicit
    )
    exact_selected_extension_independent_extra_q_range_source_materialization_public_checkpoint_synthesis_route_formula_available_now = bool(
        inventory_formula_explicit
    )
    exact_selected_extension_independent_extra_q_range_source_materialization_hybrid_bridge_route_formula_available_now = bool(
        inventory_formula_explicit
    )
    exact_selected_extension_independent_extra_q_range_source_materialization_inventory_nonempty_theorem_available_now = bool(
        inventory_formula_explicit
    )
    exact_selected_extension_independent_extra_q_range_source_materialization_front_runner_route_formula_available_now = bool(
        inventory_formula_explicit
    )
    exact_selected_extension_independent_extra_q_range_source_materialization_front_runner_compatibility_theorem_available_now = bool(
        inventory_formula_explicit
    )
    exact_selected_extension_independent_extra_q_range_source_materialization_available_now = False
    updated_pack_selected_extension_independent_extra_q_range_source_materialization_front_runner_followup_required = bool(
        exact_selected_extension_independent_extra_q_range_source_materialization_front_runner_compatibility_theorem_available_now
    )
    updated_pack_same_schema_selected_extension_independent_extra_q_range_source_materialization_inventory_replay_detected_now = (
        False
    )
    farther_hybrid_continuation_reopen_required_now = False

    rows = [
        sign_base.row(
            "updated_pack_selected_extension_independent_extra_q_range_source_materialization_inventory_audit_selected",
            "pass" if audit_selected else "reject",
            "updated-pack selected-extension independent extra-q-range source-materialization inventory audit selected",
            sign_base.truth(audit_selected),
            "This branch is worth running only after the whole independent extra-q evidence lane has already closed negatively.",
        ),
        sign_base.row(
            "retry_gate_computation_mode_selected",
            "pass" if retry_mode else "reject",
            "retry gate computation mode selected",
            sign_base.truth(retry_mode),
            "The new lane must inventory actual materialization routes instead of replaying helper/public/hybrid evidence candidates.",
        ),
        sign_base.row(
            "failure_matrix_non_surrogate_guard_preserved",
            "pass" if non_surrogate_guard else "reject",
            "failure-matrix non-surrogate guard preserved",
            sign_base.truth(non_surrogate_guard),
            "Source-materialization inventory is honest only while exhausted evidence candidates remain closed.",
        ),
        sign_base.row(
            "selected_extension_label_fixed_now",
            "pass" if selected_extension_label_fixed else "reject",
            "selected extension label fixed now",
            sign_base.truth(selected_extension_label_fixed),
            "The inventory is meaningful only while one concrete selected extension Sigma_*^(pilot-HS) remains fixed.",
        ),
        sign_base.row(
            "selected_extension_independent_extra_q_range_evidence_lane_negative_closeout_available_now",
            "pass" if evidence_lane_closed_negatively else "reject",
            "selected-extension independent extra-q-range evidence lane negative closeout available now",
            sign_base.truth(evidence_lane_closed_negatively),
            "The source-materialization lane begins only after helper/public/hybrid evidence candidates are all closed negatively.",
        ),
        sign_base.row(
            "selected_extension_solver_side_deformation_backend_available_now",
            "pass" if deformation_helper_available else "reject",
            "selected-extension solver-side deformation backend available now",
            sign_base.truth(deformation_helper_available),
            "The front-runner route should extend the already materialized selected-extension deformation helper rather than inventing a fresh backend stack.",
        ),
        sign_base.row(
            "selected_extension_solver_recompute_backend_available_now",
            "pass" if recompute_helper_available else "reject",
            "selected-extension solver-recompute backend available now",
            sign_base.truth(recompute_helper_available),
            "The source-materialization lane is anchored to the already materialized retained-q recompute helper path.",
        ),
        sign_base.row(
            "selected_extension_solver_side_extra_q_range_helper_available_now",
            "pass" if extra_q_helper_available_now else "reject",
            "selected-extension solver-side extra-q-range helper available now",
            sign_base.truth(extra_q_helper_available_now),
            "Reject means helper implementation itself is still a live route rather than a completed artifact.",
        ),
        sign_base.row(
            "exact_selected_extension_independent_extra_q_range_source_materialization_inventory_formula_available_now",
            "pass"
            if exact_selected_extension_independent_extra_q_range_source_materialization_inventory_formula_available_now
            else "reject",
            "exact selected-extension independent extra-q-range source-materialization inventory formula available now",
            sign_base.truth(
                exact_selected_extension_independent_extra_q_range_source_materialization_inventory_formula_available_now
            ),
            "The active lane now fixes a finite route inventory instead of leaving source materialization as an unstructured request.",
        ),
        sign_base.row(
            "exact_selected_extension_independent_extra_q_range_source_materialization_front_runner_route_formula_available_now",
            "pass"
            if exact_selected_extension_independent_extra_q_range_source_materialization_front_runner_route_formula_available_now
            else "reject",
            "exact selected-extension independent extra-q-range source-materialization front-runner route formula available now",
            sign_base.truth(
                exact_selected_extension_independent_extra_q_range_source_materialization_front_runner_route_formula_available_now
            ),
            "The least-arbitrary route is now fixed literally as helper implementation on top of the already materialized selected-extension helper stack.",
        ),
        sign_base.row(
            "exact_selected_extension_independent_extra_q_range_source_materialization_available_now",
            "pass" if exact_selected_extension_independent_extra_q_range_source_materialization_available_now else "reject",
            "exact selected-extension independent extra-q-range source-materialization available now",
            sign_base.truth(
                exact_selected_extension_independent_extra_q_range_source_materialization_available_now
            ),
            "Inventory and front-runner promotion do not yet materialize one actual extra-q source route.",
        ),
        sign_base.row(
            "updated_pack_selected_extension_independent_extra_q_range_source_materialization_front_runner_followup_required",
            "pass"
            if updated_pack_selected_extension_independent_extra_q_range_source_materialization_front_runner_followup_required
            else "reject",
            "updated-pack selected-extension independent extra-q-range source-materialization front-runner followup required",
            sign_base.truth(
                updated_pack_selected_extension_independent_extra_q_range_source_materialization_front_runner_followup_required
            ),
            "The honest next blocker is candidate-specific audit of the promoted source-materialization front-runner route.",
        ),
        sign_base.row(
            "updated_pack_same_schema_selected_extension_independent_extra_q_range_source_materialization_inventory_replay_detected_now",
            "pass"
            if updated_pack_same_schema_selected_extension_independent_extra_q_range_source_materialization_inventory_replay_detected_now
            else "reject",
            "updated-pack same-schema selected-extension independent extra-q-range source-materialization inventory replay detected now",
            sign_base.truth(
                updated_pack_same_schema_selected_extension_independent_extra_q_range_source_materialization_inventory_replay_detected_now
            ),
            "False means this branch added a real finite route inventory instead of replaying the already closed evidence candidate stack.",
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
        "selected_extension_label_fixed_now": selected_extension_label_fixed,
        "selected_extension_independent_extra_q_range_evidence_lane_negative_closeout_available_now": evidence_lane_closed_negatively,
        "selected_extension_solver_side_deformation_backend_available_now": deformation_helper_available,
        "selected_extension_solver_recompute_backend_available_now": recompute_helper_available,
        "selected_extension_solver_side_extra_q_range_helper_available_now": extra_q_helper_available_now,
        "exact_selected_extension_independent_extra_q_range_source_materialization_inventory_formula_available_now": exact_selected_extension_independent_extra_q_range_source_materialization_inventory_formula_available_now,
        "exact_selected_extension_independent_extra_q_range_source_materialization_helper_implementation_route_formula_available_now": exact_selected_extension_independent_extra_q_range_source_materialization_helper_implementation_route_formula_available_now,
        "exact_selected_extension_independent_extra_q_range_source_materialization_public_checkpoint_synthesis_route_formula_available_now": exact_selected_extension_independent_extra_q_range_source_materialization_public_checkpoint_synthesis_route_formula_available_now,
        "exact_selected_extension_independent_extra_q_range_source_materialization_hybrid_bridge_route_formula_available_now": exact_selected_extension_independent_extra_q_range_source_materialization_hybrid_bridge_route_formula_available_now,
        "exact_selected_extension_independent_extra_q_range_source_materialization_inventory_nonempty_theorem_available_now": exact_selected_extension_independent_extra_q_range_source_materialization_inventory_nonempty_theorem_available_now,
        "exact_selected_extension_independent_extra_q_range_source_materialization_front_runner_route_formula_available_now": exact_selected_extension_independent_extra_q_range_source_materialization_front_runner_route_formula_available_now,
        "exact_selected_extension_independent_extra_q_range_source_materialization_front_runner_compatibility_theorem_available_now": exact_selected_extension_independent_extra_q_range_source_materialization_front_runner_compatibility_theorem_available_now,
        "exact_selected_extension_independent_extra_q_range_source_materialization_available_now": exact_selected_extension_independent_extra_q_range_source_materialization_available_now,
        "updated_pack_selected_extension_independent_extra_q_range_source_materialization_front_runner_followup_required": updated_pack_selected_extension_independent_extra_q_range_source_materialization_front_runner_followup_required,
        "updated_pack_same_schema_selected_extension_independent_extra_q_range_source_materialization_inventory_replay_detected_now": updated_pack_same_schema_selected_extension_independent_extra_q_range_source_materialization_inventory_replay_detected_now,
        "farther_hybrid_continuation_reopen_required_now": farther_hybrid_continuation_reopen_required_now,
        "pack_update_required_now": updated_pack_selected_extension_independent_extra_q_range_source_materialization_front_runner_followup_required,
        "selected_primary_completion_lane": "updated_pack_selected_extension_independent_extra_q_range_source_materialization_inventory_gate",
        "selected_secondary_completion_lane": "updated_pack_selected_extension_independent_extra_q_range_source_materialization_front_runner_audit",
        "selected_reserve_completion_lane": "farther_hybrid_reserve_only_until_source_route_promoted",
        "selected_next_generation_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_selected_extension_independent_extra_q_range_source_materialization_inventory_gate",
        "recommended_next_route_or_none": "8.7.56.5355",
        "selected_followup_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_selected_extension_independent_extra_q_range_source_materialization_front_runner_audit",
        "selected_followup_route_or_none": "8.7.56.5359",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5353",
        STEP_NAME + " declaration gate",
        {
            "source_files": {
                "prior_gate": sign_base.display_path(PRIOR_GATE),
                "prior_audit": sign_base.display_path(PRIOR_AUDIT),
                "deformation_helper": sign_base.display_path(DEFORMATION_HELPER),
                "recompute_helper": sign_base.display_path(RECOMPUTE_HELPER),
                "extra_q_helper_candidate": sign_base.display_path(EXTRA_Q_RANGE_HELPER),
            },
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "8.7.56.5355",
                "followup_route": "8.7.56.5359",
            },
        },
        rows,
        summary,
        {
            "overall_status": "vector_qball_form_factor_selected_extension_independent_extra_q_range_source_materialization_inventory_declared",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {"formulae": build_formulae()},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(
        "[done] 8.7.56.5351-5354 selected-extension independent extra-q-range source-materialization inventory completed"
    )
    print(f"[done] declaration: {declaration_paths['json']}")


# 関数: CLI entrypoint から inventory audit を実行する。

if __name__ == "__main__":
    main()
