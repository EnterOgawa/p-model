#!/usr/bin/env python3
"""Generate 8.7.56.2615-.2618 updated-pack exact ell=0 operator-refresh artifacts."""

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

STATUS = ROOT / "doc" / "STATUS.md"
ROADMAP = ROOT / "doc" / "ROADMAP.md"
AI_CONTEXT = ROOT / "doc" / "AI_CONTEXT_MIN.json"
WORK_HISTORY_RECENT = ROOT / "doc" / "WORK_HISTORY_RECENT.md"
CURRENT_PROBLEM = ROOT / "doc" / "quantum" / "34_trial2_numeric_alpha_current_problem.md"
CURRENT_STATUS = ROOT / "doc" / "quantum" / "36_trial2_numeric_alpha_current_status.md"
UNIFIED_ROADMAP = ROOT / "doc" / "quantum" / "39_trial2_vector_qball_unified_closure_roadmap.md"
LONG_ROADMAP = ROOT / "doc" / "quantum" / "55_trial2_numeric_alpha_vector_qball_long_horizon_roadmap.md"
PART5 = ROOT / "doc" / "paper" / "14_part5_future_predictions.md"

PRIOR_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.2611-2614",
        "updated_pack_exact_source_theorem_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
OLD_OPERATOR_DERIVATION = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.1471-1474",
        "ell0_exact_operator_derivation",
        prefix="q",
    ),
    "audit",
)["json"]
OLD_OPERATOR_COMPLETION = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.2367-2370",
        "exact_operator_completion_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
SERIES_AUDIT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.2447-2450",
        "updated_pack_exact_ell0_series_operator_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
SOURCE_THEOREM_AUDIT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.2607-2610",
        "updated_pack_exact_source_theorem_closeout_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.2615-2618"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack exact "
    "ell=0 action-level operator refresh audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_exact_ell0_operator_refresh_audit",
    prefix="q",
)

PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_exact_source_"
    "theorem_no_go_derived_exact_ell0_operator_primary_blind_vector_reserve_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_exact_ell0_"
    "operator_refresh_audited_cross_term_primary_constraint_secondary_gate"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_exact_ell0_"
    "operator_gate_blind_vector_reserve_refresh"
)
NEXT_ROUTE = "8.7.56.2619"
FOLLOWUP_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_exact_action_level_"
    "cross_term_completion_audit"
)
FOLLOWUP_ROUTE = "8.7.56.2623"


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

    return {
        "json": sign_base.display_path(paths["json"]),
        "csv": sign_base.display_path(paths["csv"]),
    }


# 関数: operator refresh で使う式を返す。

def build_formulae(
    series_formulas: dict[str, str],
    completion_formulas: dict[str, str],
    source_formulas: dict[str, str],
) -> dict[str, str]:
    """Return formulas used in the updated-pack exact ell=0 operator refresh audit."""
    return {
        "exact_two_component_series": series_formulas["exact_two_component_series"],
        "b1_decision_rule": series_formulas["b1_decision_rule"],
        "longitudinal_operator_surface": series_formulas["longitudinal_placeholder_operator"],
        "operator_surface_requirements": series_formulas["exact_formulation_requirements"],
        "same_field_no_go_rule": source_formulas["exact_source_theorem"],
        "operator_refresh_ordering": completion_formulas["ordering_rule"],
        "operator_refresh_logic": (
            "exact ell=0 operator refresh = retained free/off-diagonal backbone + "
            "updated-pack series/operator surface + cross-term realization first + "
            "constraint elimination second + noncollapsed ell=0 closure reserve"
        ),
    }


# 関数: `main` の入出力契約と処理意図を定義する。
def main() -> None:
    """Execute the updated-pack exact ell=0 action-level operator refresh audit."""
    for path in (
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        WORK_HISTORY_RECENT,
        CURRENT_PROBLEM,
        CURRENT_STATUS,
        UNIFIED_ROADMAP,
        LONG_ROADMAP,
        PART5,
        PRIOR_GATE,
        OLD_OPERATOR_DERIVATION,
        OLD_OPERATOR_COMPLETION,
        SERIES_AUDIT,
        SOURCE_THEOREM_AUDIT,
    ):
        sign_base.require(path)

    status_text = sign_base.read_text(STATUS)
    roadmap_text = sign_base.read_text(ROADMAP)
    current_problem_text = sign_base.read_text(CURRENT_PROBLEM)
    current_status_text = sign_base.read_text(CURRENT_STATUS)
    unified_text = sign_base.read_text(UNIFIED_ROADMAP)
    long_text = sign_base.read_text(LONG_ROADMAP)
    part5_text = sign_base.read_text(PART5)

    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    old_operator_summary = sign_base.read_json(OLD_OPERATOR_DERIVATION)["summary"]
    old_completion_payload = sign_base.read_json(OLD_OPERATOR_COMPLETION)
    old_completion_summary = old_completion_payload["summary"]
    old_completion_formulas = old_completion_payload["evidence"]["formulas"]
    series_payload = sign_base.read_json(SERIES_AUDIT)
    series_summary = series_payload["summary"]
    series_formulas = series_payload["evidence"]["formulas"]
    source_payload = sign_base.read_json(SOURCE_THEOREM_AUDIT)
    source_summary = source_payload["summary"]
    source_formulas = source_payload["evidence"]["formulas"]

    updated_pack_exact_ell0_operator_refresh_audit_selected = bool(
        prior_summary["gate_b_updated_pack_exact_ell0_action_level_operator_primary_selected"]
        and prior_summary["exact_source_theorem_derived_now"]
        and prior_summary["exact_source_theorem_no_go_verdict_fixed"]
        and not prior_summary["gate_c_blind_vector_computation_primary_admissible_now"]
    )
    retained_linear_backbone_available = bool(
        old_operator_summary["exact_action_level_linear_backbone_available"]
    )
    retained_offdiag_backbone_available = bool(
        old_completion_summary["offdiag_backbone_available"]
    )
    updated_pack_exact_ell0_series_surface_explicit = bool(
        series_summary["updated_pack_exact_ell0_series_surface_explicit"]
    )
    updated_pack_b1_decision_surface_complete = bool(
        series_summary["updated_pack_b1_decision_surface_complete"]
    )
    updated_pack_exact_longitudinal_operator_surface_explicit = bool(
        series_summary["updated_pack_exact_longitudinal_operator_surface_explicit"]
    )
    updated_pack_green_function_shooting_requirements_explicit = bool(
        series_summary["updated_pack_green_function_shooting_requirements_explicit"]
    )
    updated_pack_source_theorem_no_go_fixed = bool(
        source_summary["updated_pack_exact_source_theorem_no_go_verdict_passed"]
        and prior_summary["exact_source_theorem_no_go_verdict_fixed"]
    )
    phase1_exact_solver_cross_term_present = bool(
        old_operator_summary["phase1_exact_solver_cross_term_present"]
    )
    phase1_exact_solver_constraint_elimination_present = bool(
        old_operator_summary["phase1_exact_solver_constraint_elimination_present"]
    )
    phase1_exact_solver_scalar_nonlinear_ansatz_only = bool(
        old_operator_summary["phase1_exact_solver_scalar_nonlinear_ansatz_only"]
    )
    old_family_map_on_current_pilot_admissible = bool(
        old_operator_summary["old_family_map_on_current_pilot_admissible"]
    )
    retained_cross_term_primary_completion_supported = bool(
        old_completion_summary["cross_term_primary_completion_supported"]
    )
    retained_constraint_elimination_secondary_completion_supported = bool(
        old_completion_summary["constraint_elimination_secondary_completion_supported"]
    )
    retained_noncollapsed_ell0_closure_reserve_supported = bool(
        old_completion_summary["noncollapsed_ell0_closure_reserve_supported"]
    )
    retained_operator_completion_order_stable = bool(
        retained_cross_term_primary_completion_supported
        and retained_constraint_elimination_secondary_completion_supported
        and retained_noncollapsed_ell0_closure_reserve_supported
    )
    updated_pack_exact_ell0_operator_refresh_machine_readable_now = bool(
        updated_pack_exact_ell0_operator_refresh_audit_selected
        and retained_linear_backbone_available
        and retained_offdiag_backbone_available
        and updated_pack_exact_ell0_series_surface_explicit
        and updated_pack_b1_decision_surface_complete
        and updated_pack_exact_longitudinal_operator_surface_explicit
        and updated_pack_green_function_shooting_requirements_explicit
        and updated_pack_source_theorem_no_go_fixed
        and retained_operator_completion_order_stable
    )
    updated_pack_cross_term_primary_refresh_required = bool(
        updated_pack_exact_ell0_operator_refresh_machine_readable_now
        and retained_cross_term_primary_completion_supported
        and not phase1_exact_solver_cross_term_present
    )
    updated_pack_constraint_elimination_secondary_refresh_required = bool(
        updated_pack_cross_term_primary_refresh_required
        and retained_constraint_elimination_secondary_completion_supported
        and not phase1_exact_solver_constraint_elimination_present
    )
    updated_pack_noncollapsed_ell0_closure_reserve_required = bool(
        updated_pack_constraint_elimination_secondary_refresh_required
        and retained_noncollapsed_ell0_closure_reserve_supported
        and phase1_exact_solver_scalar_nonlinear_ansatz_only
    )
    updated_pack_exact_ell0_action_level_operator_available_now = False
    updated_pack_exact_ell0_operator_refresh_closes_missing_action_blocker_now = False
    blind_vector_observable_gate_still_blocked = True
    farther_hybrid_continuation_reopen_required_now = False

    rows = [
        sign_base.row(
            "updated_pack_exact_ell0_operator_refresh_audit_selected",
            "pass" if updated_pack_exact_ell0_operator_refresh_audit_selected else "reject",
            "updated-pack exact ell=0 operator refresh audit selected",
            sign_base.truth(updated_pack_exact_ell0_operator_refresh_audit_selected),
            "Once the current-pack source theorem closes as no-go, the honest remaining mainline returns to the exact ell=0 operator lane.",
        ),
        sign_base.row(
            "retained_linear_backbone_available",
            "pass" if retained_linear_backbone_available else "reject",
            "retained exact action-level linear backbone available",
            sign_base.truth(retained_linear_backbone_available),
            "The old operator-derivation audit already fixed that the free post-photon backbone exists and does not need reopening.",
        ),
        sign_base.row(
            "retained_offdiag_backbone_available",
            "pass" if retained_offdiag_backbone_available else "reject",
            "retained off-diagonal backbone available",
            sign_base.truth(retained_offdiag_backbone_available),
            "The old completion audit already fixed that the off-diagonal mixing ingredient is frozen in the public backbone.",
        ),
        sign_base.row(
            "updated_pack_exact_ell0_series_surface_explicit",
            "pass" if updated_pack_exact_ell0_series_surface_explicit else "reject",
            "updated-pack exact ell=0 series surface explicit",
            sign_base.truth(updated_pack_exact_ell0_series_surface_explicit),
            "The updated-pack already exposes the exact two-component near-origin series rather than only a rho-shared heuristic surface.",
        ),
        sign_base.row(
            "updated_pack_b1_decision_surface_complete",
            "pass" if updated_pack_b1_decision_surface_complete else "reject",
            "updated-pack b1 decision surface complete",
            sign_base.truth(updated_pack_b1_decision_surface_complete),
            "The updated-pack already fixes the forced-zero / sourced-nonzero / free-shooting discriminator for the longitudinal branch.",
        ),
        sign_base.row(
            "updated_pack_exact_longitudinal_operator_surface_explicit",
            "pass" if updated_pack_exact_longitudinal_operator_surface_explicit else "reject",
            "updated-pack exact longitudinal operator surface explicit",
            sign_base.truth(updated_pack_exact_longitudinal_operator_surface_explicit),
            "The operator refresh is anchored to the explicit target L_L[f_L] = S[f_0], not to a blind numeric retry.",
        ),
        sign_base.row(
            "updated_pack_green_function_shooting_requirements_explicit",
            "pass" if updated_pack_green_function_shooting_requirements_explicit else "reject",
            "updated-pack Green-function / shooting requirements explicit",
            sign_base.truth(updated_pack_green_function_shooting_requirements_explicit),
            "Constraint, Stueckelberg, boundary, and decaying-tail requirements are already spelled out on the updated-pack theorem surface.",
        ),
        sign_base.row(
            "updated_pack_source_theorem_no_go_fixed",
            "pass" if updated_pack_source_theorem_no_go_fixed else "reject",
            "updated-pack source-theorem no-go fixed",
            sign_base.truth(updated_pack_source_theorem_no_go_fixed),
            "The same-field theorem now closes as zero-source / no-go, so operator refresh is the only honest route left to change the source picture.",
        ),
        sign_base.row(
            "phase1_exact_solver_cross_term_present",
            "pass" if phase1_exact_solver_cross_term_present else "reject",
            "Phase 1 exact solver cross term present",
            sign_base.truth(phase1_exact_solver_cross_term_present),
            "The current exact pilot would only close the operator if the mixed ell=0 cross term were already literal in the implementation.",
        ),
        sign_base.row(
            "phase1_exact_solver_constraint_elimination_present",
            "pass" if phase1_exact_solver_constraint_elimination_present else "reject",
            "Phase 1 exact solver constraint elimination present",
            sign_base.truth(phase1_exact_solver_constraint_elimination_present),
            "A closed exact ell=0 operator still needs an explicit constraint-elimination step or equivalent statement inside the implementation.",
        ),
        sign_base.row(
            "phase1_exact_solver_scalar_nonlinear_ansatz_only",
            "watch" if phase1_exact_solver_scalar_nonlinear_ansatz_only else "pass",
            "Phase 1 exact solver scalar-style nonlinear ansatz only",
            sign_base.truth(phase1_exact_solver_scalar_nonlinear_ansatz_only),
            "The current pilot still retains the scalar-style nonlinear closure, so noncollapsed ell=0 closure remains downstream reserve even after the theorem no-go is fixed.",
        ),
        sign_base.row(
            "old_family_map_on_current_pilot_admissible",
            "pass" if old_family_map_on_current_pilot_admissible else "reject",
            "old family map on current pilot admissible",
            sign_base.truth(old_family_map_on_current_pilot_admissible),
            "The old family map remains inadmissible on the unchanged pilot, so the refresh cannot skip directly to blind-vector or family reuse.",
        ),
        sign_base.row(
            "retained_cross_term_primary_completion_supported",
            "pass" if retained_cross_term_primary_completion_supported else "reject",
            "retained cross-term primary completion supported",
            sign_base.truth(retained_cross_term_primary_completion_supported),
            "The old operator-completion audit already fixed cross-term realization as the first exact completion layer.",
        ),
        sign_base.row(
            "retained_constraint_elimination_secondary_completion_supported",
            "pass" if retained_constraint_elimination_secondary_completion_supported else "reject",
            "retained constraint-elimination secondary completion supported",
            sign_base.truth(retained_constraint_elimination_secondary_completion_supported),
            "Constraint elimination remains the next exact layer after the mixed operator is realized.",
        ),
        sign_base.row(
            "retained_noncollapsed_ell0_closure_reserve_supported",
            "pass" if retained_noncollapsed_ell0_closure_reserve_supported else "reject",
            "retained noncollapsed ell=0 closure reserve supported",
            sign_base.truth(retained_noncollapsed_ell0_closure_reserve_supported),
            "The nonlinear ell=0 closure still stays reserve because it depends on the coupled linear operator and its elimination being completed first.",
        ),
    ]

    rows.extend(
        [
            sign_base.row(
                "retained_operator_completion_order_stable",
                "pass" if retained_operator_completion_order_stable else "reject",
                "retained operator-completion order stable",
                sign_base.truth(retained_operator_completion_order_stable),
                "The no-go theorem does not change the old completion ordering; it only removes the false hope that blind-vector direct computation can bypass it.",
            ),
            sign_base.row(
                "updated_pack_exact_ell0_operator_refresh_machine_readable_now",
                "pass" if updated_pack_exact_ell0_operator_refresh_machine_readable_now else "reject",
                "updated-pack exact ell=0 operator refresh machine-readable now",
                sign_base.truth(updated_pack_exact_ell0_operator_refresh_machine_readable_now),
                "The refresh now bundles the old backbone, updated-pack operator surface, and no-go theorem consequence into one explicit operator lane.",
            ),
            sign_base.row(
                "updated_pack_cross_term_primary_refresh_required",
                "pass" if updated_pack_cross_term_primary_refresh_required else "reject",
                "updated-pack cross-term primary refresh required",
                sign_base.truth(updated_pack_cross_term_primary_refresh_required),
                "Because the mixed operator term is still absent in the exact pilot, cross-term completion remains the first honest refresh target.",
            ),
            sign_base.row(
                "updated_pack_constraint_elimination_secondary_refresh_required",
                "pass" if updated_pack_constraint_elimination_secondary_refresh_required else "reject",
                "updated-pack constraint-elimination secondary refresh required",
                sign_base.truth(updated_pack_constraint_elimination_secondary_refresh_required),
                "Constraint elimination stays downstream of cross-term completion, just as in the old completion audit.",
            ),
            sign_base.row(
                "updated_pack_noncollapsed_ell0_closure_reserve_required",
                "pass" if updated_pack_noncollapsed_ell0_closure_reserve_required else "reject",
                "updated-pack noncollapsed ell=0 closure reserve required",
                sign_base.truth(updated_pack_noncollapsed_ell0_closure_reserve_required),
                "The nonlinear ell=0 closure remains reserve because the current pilot is still scalar-style and the linear coupled operator is not yet completed.",
            ),
            sign_base.row(
                "updated_pack_exact_ell0_action_level_operator_available_now",
                "pass" if updated_pack_exact_ell0_action_level_operator_available_now else "reject",
                "updated-pack exact ell=0 action-level operator available now",
                sign_base.truth(updated_pack_exact_ell0_action_level_operator_available_now),
                "The refresh lane is explicit, but the exact operator itself is still absent because mixed term, constraint elimination, and non-heuristic nonlinear closure are not all present.",
            ),
            sign_base.row(
                "updated_pack_exact_ell0_operator_refresh_closes_missing_action_blocker_now",
                "pass" if updated_pack_exact_ell0_operator_refresh_closes_missing_action_blocker_now else "reject",
                "updated-pack exact ell=0 operator refresh closes missing-action blocker now",
                sign_base.truth(updated_pack_exact_ell0_operator_refresh_closes_missing_action_blocker_now),
                "This branch fixes the honest refresh ordering but does not yet derive the missing action-level operator.",
            ),
            sign_base.row(
                "blind_vector_observable_gate_still_blocked",
                "pass" if blind_vector_observable_gate_still_blocked else "reject",
                "blind vector observable gate still blocked",
                sign_base.truth(blind_vector_observable_gate_still_blocked),
                "The same-field theorem is no-go under the current pack, so blind-vector direct computation remains reserve-only until the operator lane changes the source object.",
            ),
            sign_base.row(
                "farther_hybrid_continuation_reopen_required_now",
                "pass" if farther_hybrid_continuation_reopen_required_now else "reject",
                "farther hybrid continuation reopen required now",
                sign_base.truth(farther_hybrid_continuation_reopen_required_now),
                "The blocker is now localized to the operator lane, so extra q-range evidence remains unnecessary.",
            ),
        ]
    )

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "retained_scalar_residual_rel": float(prior_summary["retained_scalar_residual_rel"]),
        "updated_pack_exact_ell0_operator_refresh_audit_selected": updated_pack_exact_ell0_operator_refresh_audit_selected,
        "retained_linear_backbone_available": retained_linear_backbone_available,
        "retained_offdiag_backbone_available": retained_offdiag_backbone_available,
        "updated_pack_exact_ell0_series_surface_explicit": updated_pack_exact_ell0_series_surface_explicit,
        "updated_pack_b1_decision_surface_complete": updated_pack_b1_decision_surface_complete,
        "updated_pack_exact_longitudinal_operator_surface_explicit": updated_pack_exact_longitudinal_operator_surface_explicit,
        "updated_pack_green_function_shooting_requirements_explicit": updated_pack_green_function_shooting_requirements_explicit,
        "updated_pack_source_theorem_no_go_fixed": updated_pack_source_theorem_no_go_fixed,
        "phase1_exact_solver_cross_term_present": phase1_exact_solver_cross_term_present,
        "phase1_exact_solver_constraint_elimination_present": phase1_exact_solver_constraint_elimination_present,
        "phase1_exact_solver_scalar_nonlinear_ansatz_only": phase1_exact_solver_scalar_nonlinear_ansatz_only,
        "old_family_map_on_current_pilot_admissible": old_family_map_on_current_pilot_admissible,
        "retained_cross_term_primary_completion_supported": retained_cross_term_primary_completion_supported,
        "retained_constraint_elimination_secondary_completion_supported": retained_constraint_elimination_secondary_completion_supported,
        "retained_noncollapsed_ell0_closure_reserve_supported": retained_noncollapsed_ell0_closure_reserve_supported,
        "retained_operator_completion_order_stable": retained_operator_completion_order_stable,
        "updated_pack_exact_ell0_operator_refresh_machine_readable_now": updated_pack_exact_ell0_operator_refresh_machine_readable_now,
        "updated_pack_cross_term_primary_refresh_required": updated_pack_cross_term_primary_refresh_required,
        "updated_pack_constraint_elimination_secondary_refresh_required": updated_pack_constraint_elimination_secondary_refresh_required,
        "updated_pack_noncollapsed_ell0_closure_reserve_required": updated_pack_noncollapsed_ell0_closure_reserve_required,
        "updated_pack_exact_ell0_action_level_operator_available_now": updated_pack_exact_ell0_action_level_operator_available_now,
        "updated_pack_exact_ell0_operator_refresh_closes_missing_action_blocker_now": updated_pack_exact_ell0_operator_refresh_closes_missing_action_blocker_now,
        "blind_vector_observable_gate_still_blocked": blind_vector_observable_gate_still_blocked,
        "farther_hybrid_continuation_reopen_required_now": farther_hybrid_continuation_reopen_required_now,
        "selected_primary_pack_update_surface": "updated_pack_exact_action_level_cross_term_completion",
        "selected_secondary_pack_update_surface": "updated_pack_constraint_elimination",
        "selected_reserve_completion_lane": "updated_pack_noncollapsed_ell0_closure_then_blind_vector",
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": FOLLOWUP_ROUTE_NAME,
        "selected_followup_route_or_none": FOLLOWUP_ROUTE,
        "physical_reject_required": False,
    }

    declaration_payload = sign_base.payload(
        "8.7.56.2617",
        STEP_NAME + " declaration gate",
        {
            "source_files": {
                "status": sign_base.display_path(STATUS),
                "roadmap": sign_base.display_path(ROADMAP),
                "ai_context": sign_base.display_path(AI_CONTEXT),
                "work_history_recent": sign_base.display_path(WORK_HISTORY_RECENT),
                "current_problem": sign_base.display_path(CURRENT_PROBLEM),
                "current_status": sign_base.display_path(CURRENT_STATUS),
                "unified_roadmap": sign_base.display_path(UNIFIED_ROADMAP),
                "long_roadmap": sign_base.display_path(LONG_ROADMAP),
                "part5": sign_base.display_path(PART5),
                "prior_gate": sign_base.display_path(PRIOR_GATE),
                "old_operator_derivation": sign_base.display_path(OLD_OPERATOR_DERIVATION),
                "old_operator_completion": sign_base.display_path(OLD_OPERATOR_COMPLETION),
                "series_audit": sign_base.display_path(SERIES_AUDIT),
                "source_theorem_audit": sign_base.display_path(SOURCE_THEOREM_AUDIT),
            },
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route_name": NEXT_ROUTE_NAME,
                "next_route": NEXT_ROUTE,
                "followup_route_name": FOLLOWUP_ROUTE_NAME,
                "followup_route": FOLLOWUP_ROUTE,
            },
        },
        rows,
        summary,
        {
            "overall_status": "vector_qball_form_factor_updated_pack_exact_ell0_operator_refresh_declared",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "formulas": build_formulae(
                series_formulas,
                old_completion_formulas,
                source_formulas,
            ),
            "hits": {
                "status_branch_hit": sign_base.hit(status_text, "8.7.56.2611"),
                "roadmap_branch_hit": sign_base.hit(roadmap_text, ".2611-.2614"),
                "current_problem_hit": sign_base.hit(current_problem_text, "exact ell=0 action-level operator"),
                "current_status_hit": sign_base.hit(current_status_text, "exact ell=0 action-level operator"),
                "unified_roadmap_hit": sign_base.hit(unified_text, "exact action-level operator completion audit"),
                "long_roadmap_hit": sign_base.hit(long_text, ".2611-.2614"),
                "part5_hit": sign_base.hit(part5_text, "exact ell=0 action-level operator"),
            },
            "inference": {
                "operator_lane_is_only_honest_remaining_primary": True,
                "why": (
                    "The current updated-pack source theorem is already derived and "
                    "closes on the no-go branch, so blind-vector direct computation "
                    "cannot supply a new nonzero source object. The remaining honest "
                    "route is therefore to refresh the exact ell=0 operator itself, "
                    "retaining cross term first, constraint elimination second, and "
                    "noncollapsed ell=0 closure as reserve."
                ),
            },
        },
    )
    declaration_paths = write_artifact("declaration_gate", declaration_payload)

    route_payload = {
        "generated_utc": sign_base.now_iso(),
        "phase": {
            "phase": 8,
            "step": "8.7.56.2618",
            "name": STEP_NAME + " route sync",
        },
        "inputs": declaration_paths,
        "rows": rows,
        "summary": summary,
        "decision": {
            "overall_status": "vector_qball_form_factor_updated_pack_exact_ell0_operator_refresh_route_synced",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        "evidence": {
            "formulas": build_formulae(
                series_formulas,
                old_completion_formulas,
                source_formulas,
            ),
            "disposition": {
                "cross_term_primary_refresh_required": updated_pack_cross_term_primary_refresh_required,
                "constraint_elimination_secondary_refresh_required": updated_pack_constraint_elimination_secondary_refresh_required,
                "noncollapsed_ell0_closure_reserve_required": updated_pack_noncollapsed_ell0_closure_reserve_required,
                "direct_blind_vector_still_blocked": blind_vector_observable_gate_still_blocked,
            },
        },
    }
    route_paths = write_artifact("route_sync", route_payload)

    print("[ok] updated-pack exact ell=0 operator-refresh artifacts written")
    print(f"  declaration_gate: {declaration_paths['json']}")
    print(f"  route_sync: {route_paths['json']}")


if __name__ == "__main__":
    main()
