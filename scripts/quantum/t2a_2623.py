#!/usr/bin/env python3
"""Generate 8.7.56.2623-.2626 updated-pack cross-term completion audit artifacts."""

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
        "8.7.56.2619-2622",
        "updated_pack_exact_ell0_operator_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
PRIOR_AUDIT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.2615-2618",
        "updated_pack_exact_ell0_operator_refresh_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
OLD_CROSS_TERM_AUDIT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.2375-2378",
        "cross_term_completion_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
OLD_LITERAL_AUDIT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.2383-2386",
        "phase1_literal_cross_term_realization_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
ELL0_OPERATOR_AUDIT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.1471-1474",
        "ell0_exact_operator_derivation",
        prefix="q",
    ),
    "audit",
)["json"]

PHASE1_SOLVER = ROOT / "scripts" / "quantum" / "t2a_1419.py"

STEP_TAG = "8.7.56.2623-2626"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack exact "
    "action-level cross-term completion audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_cross_term_completion_audit",
    prefix="q",
)

PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_exact_ell0_"
    "operator_refresh_audited_cross_term_primary_constraint_secondary_blind_vector_"
    "reserve_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_cross_term_"
    "phase1_literal_target_constraint_secondary_gate"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_cross_term_completion_"
    "gate_constraint_elimination_refresh"
)
NEXT_ROUTE = "8.7.56.2627"
FOLLOWUP_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_phase1_exact_solver_"
    "literal_cross_term_realization_audit"
)
FOLLOWUP_ROUTE = "8.7.56.2631"


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


# 関数: hit text に token が含まれるかを返す。

def hit_has_token(hit_obj: dict | None, token: str) -> bool:
    """Return whether one hit payload contains one token."""
    return bool(hit_obj and token in str(hit_obj["text"]))


# 関数: cross-term audit で使う式を返す。

def build_formulae(
    prior_formulas: dict[str, str],
    old_cross_formulas: dict[str, str],
    old_literal_formulas: dict[str, str],
) -> dict[str, str]:
    """Return formulas used in the updated-pack cross-term completion audit."""
    return {
        "exact_two_component_series": prior_formulas["exact_two_component_series"],
        "longitudinal_operator_surface": prior_formulas["longitudinal_operator_surface"],
        "backbone": old_cross_formulas["backbone"],
        "kinetic_identity": old_cross_formulas["kinetic_identity"],
        "triangular_literal_realization": old_literal_formulas["triangular_literal_realization"],
        "same_field_no_go": prior_formulas["same_field_no_go_rule"],
        "ordering_rule": (
            "updated-pack cross term -> updated-pack constraint elimination -> "
            "updated-pack noncollapsed ell=0 closure reserve -> blind-vector reserve"
        ),
    }


# 関数: `main` の入出力契約と処理意図を定義する。

def main() -> None:
    """Execute the updated-pack exact action-level cross-term completion audit."""
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
        PRIOR_AUDIT,
        OLD_CROSS_TERM_AUDIT,
        OLD_LITERAL_AUDIT,
        ELL0_OPERATOR_AUDIT,
        PHASE1_SOLVER,
    ):
        sign_base.require(path)

    status_text = sign_base.read_text(STATUS)
    roadmap_text = sign_base.read_text(ROADMAP)
    current_problem_text = sign_base.read_text(CURRENT_PROBLEM)
    current_status_text = sign_base.read_text(CURRENT_STATUS)
    unified_text = sign_base.read_text(UNIFIED_ROADMAP)
    long_text = sign_base.read_text(LONG_ROADMAP)
    part5_text = sign_base.read_text(PART5)
    phase1_text = sign_base.read_text(PHASE1_SOLVER)

    prior_gate_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    prior_audit_payload = sign_base.read_json(PRIOR_AUDIT)
    prior_audit_summary = prior_audit_payload["summary"]
    prior_audit_formulas = prior_audit_payload["evidence"]["formulas"]
    old_cross_payload = sign_base.read_json(OLD_CROSS_TERM_AUDIT)
    old_cross_summary = old_cross_payload["summary"]
    old_cross_formulas = old_cross_payload["evidence"]["formulas"]
    old_literal_payload = sign_base.read_json(OLD_LITERAL_AUDIT)
    old_literal_summary = old_literal_payload["summary"]
    old_literal_formulas = old_literal_payload["evidence"]["formulas"]
    operator_summary = sign_base.read_json(ELL0_OPERATOR_AUDIT)["summary"]

    diagnostic_source_hit = sign_base.hit(
        phase1_text,
        "source = float(beta) * float(np.interp(safe_r, radius, field_prime))",
    )
    exact_f0_hit = sign_base.hit(
        phase1_text,
        "f0_double_prime = -(2.0 / safe_r) * f0_prime",
    )
    exact_fl_hit = sign_base.hit(
        phase1_text,
        "f_l_double_prime = -(2.0 / safe_r) * f_l_prime",
    )

    updated_pack_exact_action_level_cross_term_completion_audit_selected = bool(
        prior_gate_summary["gate_b_updated_pack_cross_term_primary_selected"]
        and prior_gate_summary["exact_source_theorem_no_go_verdict_fixed"]
        and not prior_gate_summary["gate_c_blind_vector_computation_primary_admissible_now"]
    )
    phase1_diagnostic_cross_term_template_present = bool(
        old_cross_summary["phase1_diagnostic_cross_term_template_present"]
        and diagnostic_source_hit
    )
    phase1_exact_solver_literal_cross_term_present = bool(
        hit_has_token(exact_f0_hit, "f_l")
        and hit_has_token(exact_fl_hit, "f0")
    )
    updated_pack_phase1_exact_solver_primary_target_supported = bool(
        updated_pack_exact_action_level_cross_term_completion_audit_selected
        and phase1_diagnostic_cross_term_template_present
        and not phase1_exact_solver_literal_cross_term_present
        and prior_audit_summary["updated_pack_cross_term_primary_refresh_required"]
        and old_cross_summary["phase1_exact_solver_primary_target_supported"]
    )
    updated_pack_trial3_family_ell0_collapse_secondary_only = bool(
        updated_pack_phase1_exact_solver_primary_target_supported
        and old_cross_summary["trial3_family_ell0_collapse_secondary_only"]
        and operator_summary["trial3_family_solver_ell0_coupling_collapses"]
    )
    updated_pack_source_theorem_no_go_preserved_during_cross_term_audit = bool(
        prior_gate_summary["exact_source_theorem_no_go_verdict_fixed"]
        and prior_audit_summary["updated_pack_source_theorem_no_go_fixed"]
    )
    updated_pack_literal_cross_term_realization_formula_available = bool(
        old_literal_summary["phase1_literal_cross_term_realization_formula_available"]
    )
    updated_pack_literal_cross_term_realization_is_triangular = bool(
        old_literal_summary["phase1_literal_cross_term_realization_is_triangular"]
    )
    updated_pack_exact_cross_term_completion_supported_under_current_pack = bool(
        updated_pack_phase1_exact_solver_primary_target_supported
        and updated_pack_source_theorem_no_go_preserved_during_cross_term_audit
        and operator_summary["exact_action_level_linear_backbone_available"]
        and prior_audit_summary["updated_pack_exact_longitudinal_operator_surface_explicit"]
        and prior_audit_summary["updated_pack_green_function_shooting_requirements_explicit"]
        and updated_pack_literal_cross_term_realization_formula_available
        and updated_pack_literal_cross_term_realization_is_triangular
    )
    updated_pack_constraint_elimination_downstream_of_cross_term = bool(
        updated_pack_exact_cross_term_completion_supported_under_current_pack
        and prior_audit_summary["updated_pack_constraint_elimination_secondary_refresh_required"]
        and old_cross_summary["constraint_elimination_downstream_of_cross_term"]
    )
    updated_pack_cross_term_completion_requires_pack_update_now = False
    updated_pack_exact_ell0_action_level_operator_available_now = False
    blind_vector_observable_gate_still_blocked = True
    farther_hybrid_continuation_reopen_required_now = bool(
        prior_gate_summary["farther_hybrid_continuation_reopen_required_now"]
    )

    rows = [
        sign_base.row(
            "updated_pack_exact_action_level_cross_term_completion_audit_selected",
            "pass" if updated_pack_exact_action_level_cross_term_completion_audit_selected else "reject",
            "updated-pack exact action-level cross-term completion audit selected",
            sign_base.truth(updated_pack_exact_action_level_cross_term_completion_audit_selected),
            "Once the exact source theorem closes on the no-go branch, the only honest primary operator move is cross-term completion.",
        ),
        sign_base.row(
            "phase1_diagnostic_cross_term_template_present",
            "pass" if phase1_diagnostic_cross_term_template_present else "reject",
            "phase-1 diagnostic cross-term template present",
            sign_base.truth(phase1_diagnostic_cross_term_template_present),
            "The retained diagnostic branch already injects beta * f_0' as the driven longitudinal source, so the literal cross-term structure is known.",
        ),
        sign_base.row(
            "phase1_exact_solver_literal_cross_term_present",
            "pass" if phase1_exact_solver_literal_cross_term_present else "reject",
            "phase-1 exact solver literal cross term present",
            sign_base.truth(phase1_exact_solver_literal_cross_term_present),
            "The current exact pilot still evolves diagonal f_0 and f_L equations, so the mixed ell=0 source is not yet literal there.",
        ),
        sign_base.row(
            "updated_pack_phase1_exact_solver_primary_target_supported",
            "pass" if updated_pack_phase1_exact_solver_primary_target_supported else "reject",
            "updated-pack phase-1 exact solver primary target supported",
            sign_base.truth(updated_pack_phase1_exact_solver_primary_target_supported),
            "The smallest honest completion target remains the phase-1 exact pilot itself rather than a reopened family map or blind-vector retry.",
        ),
        sign_base.row(
            "updated_pack_trial3_family_ell0_collapse_secondary_only",
            "pass" if updated_pack_trial3_family_ell0_collapse_secondary_only else "reject",
            "updated-pack Trial-3 ell=0 collapse secondary only",
            sign_base.truth(updated_pack_trial3_family_ell0_collapse_secondary_only),
            "The old Trial-3 family still collapses at ell=0, so it only explains why the family route cannot be the primary fix.",
        ),
        sign_base.row(
            "updated_pack_source_theorem_no_go_preserved_during_cross_term_audit",
            "pass" if updated_pack_source_theorem_no_go_preserved_during_cross_term_audit else "reject",
            "updated-pack source-theorem no-go preserved during cross-term audit",
            sign_base.truth(updated_pack_source_theorem_no_go_preserved_during_cross_term_audit),
            "The current same-field theorem remains zero-source throughout this branch, so cross-term completion must be evaluated without pretending that blind-vector support already exists.",
        ),
        sign_base.row(
            "updated_pack_literal_cross_term_realization_formula_available",
            "pass" if updated_pack_literal_cross_term_realization_formula_available else "reject",
            "updated-pack literal cross-term realization formula available",
            sign_base.truth(updated_pack_literal_cross_term_realization_formula_available),
            "Repo-local prior artifacts already preserve the field-strength identity and driven longitudinal equation needed for the first literal realization.",
        ),
        sign_base.row(
            "updated_pack_literal_cross_term_realization_is_triangular",
            "pass" if updated_pack_literal_cross_term_realization_is_triangular else "reject",
            "updated-pack literal cross-term realization is triangular",
            sign_base.truth(updated_pack_literal_cross_term_realization_is_triangular),
            "The first honest restoration remains scalar-first f_0 plus driven f_L, with reciprocal backreaction left downstream.",
        ),
        sign_base.row(
            "updated_pack_exact_cross_term_completion_supported_under_current_pack",
            "pass" if updated_pack_exact_cross_term_completion_supported_under_current_pack else "reject",
            "updated-pack exact cross-term completion supported under current pack",
            sign_base.truth(updated_pack_exact_cross_term_completion_supported_under_current_pack),
            "The updated-pack operator surface, source-theorem no-go, and retained literal formulas already support cross-term completion without a substantive new pack.",
        ),
        sign_base.row(
            "updated_pack_constraint_elimination_downstream_of_cross_term",
            "pass" if updated_pack_constraint_elimination_downstream_of_cross_term else "reject",
            "updated-pack constraint elimination downstream of cross term",
            sign_base.truth(updated_pack_constraint_elimination_downstream_of_cross_term),
            "Constraint elimination still acts on the coupled operator after the missing cross term is restored, so it remains the followup lane.",
        ),
        sign_base.row(
            "updated_pack_cross_term_completion_requires_pack_update_now",
            "pass" if updated_pack_cross_term_completion_requires_pack_update_now else "reject",
            "updated-pack cross-term completion requires substantive pack update now",
            sign_base.truth(updated_pack_cross_term_completion_requires_pack_update_now),
            "The current blocker is an internal literal omission inside the retained exact pilot, not a missing external theorem surface.",
        ),
        sign_base.row(
            "updated_pack_exact_ell0_action_level_operator_available_now",
            "pass" if updated_pack_exact_ell0_action_level_operator_available_now else "reject",
            "updated-pack exact ell=0 action-level operator available now",
            sign_base.truth(updated_pack_exact_ell0_action_level_operator_available_now),
            "Cross-term completion is only the first operator layer; the exact ell=0 operator is still not closed at this audit.",
        ),
        sign_base.row(
            "blind_vector_observable_gate_still_blocked",
            "pass" if blind_vector_observable_gate_still_blocked else "reject",
            "blind-vector observable gate still blocked",
            sign_base.truth(blind_vector_observable_gate_still_blocked),
            "Because the same-field theorem remains no-go and the coupled operator is still open, blind-vector direct computation stays reserve-only.",
        ),
        sign_base.row(
            "farther_hybrid_continuation_reopen_required_now",
            "pass" if farther_hybrid_continuation_reopen_required_now else "reject",
            "farther hybrid continuation reopen required now",
            sign_base.truth(farther_hybrid_continuation_reopen_required_now),
            "The blocker remains operator-side, so extra q-range evidence still does not need reopening.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "retained_scalar_residual_rel": float(prior_gate_summary["retained_scalar_residual_rel"]),
        "updated_pack_exact_action_level_cross_term_completion_audit_selected": updated_pack_exact_action_level_cross_term_completion_audit_selected,
        "phase1_diagnostic_cross_term_template_present": phase1_diagnostic_cross_term_template_present,
        "phase1_exact_solver_literal_cross_term_present": phase1_exact_solver_literal_cross_term_present,
        "updated_pack_phase1_exact_solver_primary_target_supported": updated_pack_phase1_exact_solver_primary_target_supported,
        "updated_pack_trial3_family_ell0_collapse_secondary_only": updated_pack_trial3_family_ell0_collapse_secondary_only,
        "updated_pack_source_theorem_no_go_preserved_during_cross_term_audit": updated_pack_source_theorem_no_go_preserved_during_cross_term_audit,
        "updated_pack_literal_cross_term_realization_formula_available": updated_pack_literal_cross_term_realization_formula_available,
        "updated_pack_literal_cross_term_realization_is_triangular": updated_pack_literal_cross_term_realization_is_triangular,
        "updated_pack_exact_cross_term_completion_supported_under_current_pack": updated_pack_exact_cross_term_completion_supported_under_current_pack,
        "updated_pack_constraint_elimination_downstream_of_cross_term": updated_pack_constraint_elimination_downstream_of_cross_term,
        "updated_pack_cross_term_completion_requires_pack_update_now": updated_pack_cross_term_completion_requires_pack_update_now,
        "updated_pack_exact_ell0_action_level_operator_available_now": updated_pack_exact_ell0_action_level_operator_available_now,
        "blind_vector_observable_gate_still_blocked": blind_vector_observable_gate_still_blocked,
        "farther_hybrid_continuation_reopen_required_now": farther_hybrid_continuation_reopen_required_now,
        "selected_primary_cross_term_target": "updated_pack_phase1_exact_solver_literal_completion",
        "selected_secondary_cross_term_issue": "updated_pack_constraint_elimination_followup",
        "selected_reserve_completion_lane": "updated_pack_noncollapsed_ell0_closure_then_blind_vector",
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": FOLLOWUP_ROUTE_NAME,
        "selected_followup_route_or_none": FOLLOWUP_ROUTE,
        "physical_reject_required": False,
    }

    declaration_payload = sign_base.payload(
        "8.7.56.2625",
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
                "prior_audit": sign_base.display_path(PRIOR_AUDIT),
                "old_cross_term_audit": sign_base.display_path(OLD_CROSS_TERM_AUDIT),
                "old_literal_audit": sign_base.display_path(OLD_LITERAL_AUDIT),
                "ell0_operator_audit": sign_base.display_path(ELL0_OPERATOR_AUDIT),
                "phase1_solver": sign_base.display_path(PHASE1_SOLVER),
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
            "overall_status": "vector_qball_form_factor_updated_pack_cross_term_completion_audited",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "formulas": build_formulae(
                prior_audit_formulas,
                old_cross_formulas,
                old_literal_formulas,
            ),
            "hits": {
                "status_branch_hit": sign_base.hit(status_text, "8.7.56.2619"),
                "roadmap_branch_hit": sign_base.hit(roadmap_text, ".2619-.2622"),
                "current_problem_hit": sign_base.hit(current_problem_text, "exact action-level cross-term completion"),
                "current_status_hit": sign_base.hit(current_status_text, "exact action-level cross-term completion"),
                "unified_roadmap_hit": sign_base.hit(unified_text, ".2619-.2622"),
                "long_roadmap_hit": sign_base.hit(long_text, ".2619-.2622"),
                "part5_hit": sign_base.hit(part5_text, "updated-pack exact action-level cross-term completion audit"),
                "phase1_diagnostic_source_hit": diagnostic_source_hit,
                "phase1_exact_f0_hit": exact_f0_hit,
                "phase1_exact_fl_hit": exact_fl_hit,
            },
            "inference": {
                "updated_pack_cross_term_is_primary_after_theorem_no_go": True,
                "why": (
                    "The exact source theorem is already derived on the current-pack "
                    "no-go branch, so cross-term completion is the first honest move "
                    "that can change the action-level operator without reopening a "
                    "new theorem family."
                ),
            },
        },
    )
    declaration_paths = write_artifact("declaration_gate", declaration_payload)

    route_payload = {
        "generated_utc": sign_base.now_iso(),
        "phase": {
            "phase": 8,
            "step": "8.7.56.2626",
            "name": STEP_NAME + " route sync",
        },
        "inputs": declaration_paths,
        "rows": rows,
        "summary": summary,
        "decision": {
            "overall_status": "vector_qball_form_factor_updated_pack_cross_term_completion_route_synced",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        "evidence": {
            "formulas": build_formulae(
                prior_audit_formulas,
                old_cross_formulas,
                old_literal_formulas,
            ),
            "selected_route": {
                "next_route_name": NEXT_ROUTE_NAME,
                "next_route": NEXT_ROUTE,
                "followup_route_name": FOLLOWUP_ROUTE_NAME,
                "followup_route": FOLLOWUP_ROUTE,
            },
        },
    }
    route_paths = write_artifact("route_sync", route_payload)

    print("[ok] updated-pack cross-term completion audit artifacts written")
    print(f"  declaration_gate: {declaration_paths['json']}")
    print(f"  route_sync: {route_paths['json']}")


if __name__ == "__main__":
    main()
