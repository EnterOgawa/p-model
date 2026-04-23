#!/usr/bin/env python3
"""Generate 8.7.56.2631-.2634 updated-pack literal cross-term realization artifacts."""

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
        "8.7.56.2627-2630",
        "updated_pack_cross_term_completion_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
PRIOR_AUDIT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.2623-2626",
        "updated_pack_cross_term_completion_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
PRIOR_OPERATOR_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.2619-2622",
        "updated_pack_exact_ell0_operator_gate",
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

PHASE1_SOLVER = ROOT / "scripts" / "quantum" / "t2a_1419.py"

STEP_TAG = "8.7.56.2631-2634"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack phase-1 "
    "literal cross-term realization audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_phase1_literal_cross_term_realization_audit",
    prefix="q",
)

PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_phase1_"
    "literal_cross_term_primary_constraint_elimination_secondary_blind_vector_"
    "reserve_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_phase1_"
    "literal_cross_term_triangulated_realization_constraint_followup_gate"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_phase1_literal_cross_"
    "term_gate_constraint_elimination_refresh"
)
NEXT_ROUTE = "8.7.56.2635"
FOLLOWUP_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_phase1_exact_solver_"
    "constraint_elimination_audit"
)
FOLLOWUP_ROUTE = "8.7.56.2639"


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


# 関数: literal realization audit で使う式を返す。

def build_formulae(old_literal_formulas: dict[str, str]) -> dict[str, str]:
    """Return formulas used in the updated-pack literal realization audit."""
    return {
        "kinetic_identity": old_literal_formulas["kinetic_identity"],
        "triangular_literal_realization": old_literal_formulas["triangular_literal_realization"],
        "backreaction_followup": old_literal_formulas["backreaction_followup"],
        "ordering_rule": (
            "updated-pack phase-1 triangular literal realization -> updated-pack "
            "constraint elimination -> updated-pack noncollapsed ell=0 closure reserve"
        ),
    }


# 関数: `main` の入出力契約と処理意図を定義する。

def main() -> None:
    """Execute the updated-pack phase-1 literal cross-term realization audit."""
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
        PRIOR_OPERATOR_GATE,
        OLD_LITERAL_AUDIT,
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
    prior_audit_summary = sign_base.read_json(PRIOR_AUDIT)["summary"]
    prior_operator_gate_summary = sign_base.read_json(PRIOR_OPERATOR_GATE)["summary"]
    old_literal_payload = sign_base.read_json(OLD_LITERAL_AUDIT)
    old_literal_summary = old_literal_payload["summary"]
    old_literal_formulas = old_literal_payload["evidence"]["formulas"]

    exact_f0_hit = sign_base.hit(
        phase1_text,
        "f0_double_prime = -(2.0 / safe_r) * f0_prime",
    )
    exact_fl_hit = sign_base.hit(
        phase1_text,
        "f_l_double_prime = -(2.0 / safe_r) * f_l_prime",
    )

    updated_pack_phase1_literal_cross_term_realization_audit_selected = bool(
        prior_gate_summary["gate_a_updated_pack_phase1_literal_cross_term_selected"]
        and prior_gate_summary["gate_b_updated_pack_constraint_elimination_followup_retained"]
        and prior_gate_summary["exact_source_theorem_no_go_verdict_fixed"]
        and not prior_gate_summary["gate_c_blind_vector_computation_primary_admissible_now"]
    )
    updated_pack_phase1_literal_cross_term_realization_formula_available = bool(
        old_literal_summary["phase1_literal_cross_term_realization_formula_available"]
        and prior_audit_summary["updated_pack_literal_cross_term_realization_formula_available"]
    )
    updated_pack_phase1_literal_cross_term_realization_is_triangular = bool(
        old_literal_summary["phase1_literal_cross_term_realization_is_triangular"]
        and not hit_has_token(exact_f0_hit, "f_l")
        and not hit_has_token(exact_fl_hit, "f0")
    )
    updated_pack_perturbative_case_gamma_blocks_direct_numeric_reuse = bool(
        old_literal_summary["perturbative_case_gamma_blocks_direct_numeric_reuse"]
    )
    updated_pack_phase1_literal_cross_term_realization_supported_under_current_pack = bool(
        updated_pack_phase1_literal_cross_term_realization_audit_selected
        and prior_audit_summary["updated_pack_phase1_exact_solver_primary_target_supported"]
        and updated_pack_phase1_literal_cross_term_realization_formula_available
        and updated_pack_phase1_literal_cross_term_realization_is_triangular
        and prior_operator_gate_summary["exact_source_theorem_no_go_verdict_fixed"]
    )
    updated_pack_phase1_backreaction_followup_still_open = bool(
        updated_pack_phase1_literal_cross_term_realization_supported_under_current_pack
        and old_literal_summary["phase1_backreaction_followup_still_open"]
        and not hit_has_token(exact_f0_hit, "f_l")
    )
    updated_pack_phase1_literal_cross_term_realization_closes_exact_coupled_operator_now = False
    updated_pack_constraint_elimination_followup_required = bool(
        prior_gate_summary["gate_b_updated_pack_constraint_elimination_followup_retained"]
        and updated_pack_phase1_backreaction_followup_still_open
    )
    updated_pack_noncollapsed_ell0_closure_reserve_retained = bool(
        prior_gate_summary["noncollapsed_ell0_closure_reserve_retained"]
    )
    trial3_family_primary_reuse_admissible_now = False
    blind_vector_observable_gate_still_blocked = True
    pack_update_required_now = False

    rows = [
        sign_base.row(
            "updated_pack_phase1_literal_cross_term_realization_audit_selected",
            "pass" if updated_pack_phase1_literal_cross_term_realization_audit_selected else "reject",
            "updated-pack phase-1 literal cross-term realization audit selected",
            sign_base.truth(updated_pack_phase1_literal_cross_term_realization_audit_selected),
            "This branch starts only after cross-term completion is promoted as the honest primary operator move and blind-vector remains blocked.",
        ),
        sign_base.row(
            "updated_pack_phase1_literal_cross_term_realization_formula_available",
            "pass" if updated_pack_phase1_literal_cross_term_realization_formula_available else "reject",
            "updated-pack phase-1 literal cross-term realization formula available",
            sign_base.truth(updated_pack_phase1_literal_cross_term_realization_formula_available),
            "Repo-local prior artifacts already preserve the kinetic identity and driven longitudinal source template needed for the first literal realization.",
        ),
        sign_base.row(
            "updated_pack_phase1_literal_cross_term_realization_is_triangular",
            "pass" if updated_pack_phase1_literal_cross_term_realization_is_triangular else "reject",
            "updated-pack phase-1 literal cross-term realization is triangular",
            sign_base.truth(updated_pack_phase1_literal_cross_term_realization_is_triangular),
            "The first honest restoration still solves f_0 first and then the driven f_L equation before reciprocal backreaction is restored.",
        ),
        sign_base.row(
            "updated_pack_perturbative_case_gamma_blocks_direct_numeric_reuse",
            "pass" if updated_pack_perturbative_case_gamma_blocks_direct_numeric_reuse else "reject",
            "updated-pack perturbative case-gamma blocks direct numeric reuse",
            sign_base.truth(updated_pack_perturbative_case_gamma_blocks_direct_numeric_reuse),
            "The old perturbative numeric branch remains structural evidence only and is not promoted as the final exact pilot.",
        ),
        sign_base.row(
            "updated_pack_phase1_literal_cross_term_realization_supported_under_current_pack",
            "pass" if updated_pack_phase1_literal_cross_term_realization_supported_under_current_pack else "reject",
            "updated-pack phase-1 literal cross-term realization supported under current pack",
            sign_base.truth(updated_pack_phase1_literal_cross_term_realization_supported_under_current_pack),
            "No new theorem or pack ingredient is required to promote the missing source term from diagnostic-only status to the updated-pack exact pilot first shot.",
        ),
        sign_base.row(
            "updated_pack_phase1_backreaction_followup_still_open",
            "pass" if updated_pack_phase1_backreaction_followup_still_open else "reject",
            "updated-pack phase-1 backreaction followup still open",
            sign_base.truth(updated_pack_phase1_backreaction_followup_still_open),
            "Restoring the driven f_L equation does not yet restore reciprocal f_L -> f_0 backreaction, so the exact coupled operator remains open.",
        ),
        sign_base.row(
            "updated_pack_phase1_literal_cross_term_realization_closes_exact_coupled_operator_now",
            "pass" if updated_pack_phase1_literal_cross_term_realization_closes_exact_coupled_operator_now else "reject",
            "updated-pack phase-1 literal cross-term realization closes exact coupled operator now",
            sign_base.truth(updated_pack_phase1_literal_cross_term_realization_closes_exact_coupled_operator_now),
            "Literal realization alone is not enough because constraint elimination and noncollapsed ell=0 closure still remain downstream.",
        ),
        sign_base.row(
            "updated_pack_constraint_elimination_followup_required",
            "pass" if updated_pack_constraint_elimination_followup_required else "reject",
            "updated-pack constraint-elimination followup required",
            sign_base.truth(updated_pack_constraint_elimination_followup_required),
            "Once the literal source term is restored, the next scientific blocker becomes constraint elimination rather than another same-level search.",
        ),
        sign_base.row(
            "updated_pack_noncollapsed_ell0_closure_reserve_retained",
            "pass" if updated_pack_noncollapsed_ell0_closure_reserve_retained else "reject",
            "updated-pack noncollapsed ell=0 closure reserve retained",
            sign_base.truth(updated_pack_noncollapsed_ell0_closure_reserve_retained),
            "The nonlinear ell=0 closure remains reserve because it still depends on the completed linear coupled operator and its elimination.",
        ),
        sign_base.row(
            "trial3_family_primary_reuse_admissible_now",
            "pass" if trial3_family_primary_reuse_admissible_now else "reject",
            "Trial-3 family primary reuse admissible now",
            sign_base.truth(trial3_family_primary_reuse_admissible_now),
            "The old family map still collapses at ell=0 and remains reserve-only while the updated-pack literal realization is active.",
        ),
        sign_base.row(
            "blind_vector_observable_gate_still_blocked",
            "pass" if blind_vector_observable_gate_still_blocked else "reject",
            "blind-vector observable gate still blocked",
            sign_base.truth(blind_vector_observable_gate_still_blocked),
            "Even after literal realization is selected, blind-vector direct computation stays blocked until the coupled operator is closed beyond the no-go theorem branch.",
        ),
        sign_base.row(
            "pack_update_required_now",
            "pass" if pack_update_required_now else "reject",
            "substantive pack update required now",
            sign_base.truth(pack_update_required_now),
            "The current task is still an internal realization under the retained updated pack, not a new external action surface.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "retained_scalar_residual_rel": float(prior_gate_summary["retained_scalar_residual_rel"]),
        "updated_pack_phase1_literal_cross_term_realization_audit_selected": updated_pack_phase1_literal_cross_term_realization_audit_selected,
        "updated_pack_phase1_literal_cross_term_realization_formula_available": updated_pack_phase1_literal_cross_term_realization_formula_available,
        "updated_pack_phase1_literal_cross_term_realization_is_triangular": updated_pack_phase1_literal_cross_term_realization_is_triangular,
        "updated_pack_perturbative_case_gamma_blocks_direct_numeric_reuse": updated_pack_perturbative_case_gamma_blocks_direct_numeric_reuse,
        "updated_pack_phase1_literal_cross_term_realization_supported_under_current_pack": updated_pack_phase1_literal_cross_term_realization_supported_under_current_pack,
        "updated_pack_phase1_backreaction_followup_still_open": updated_pack_phase1_backreaction_followup_still_open,
        "updated_pack_phase1_literal_cross_term_realization_closes_exact_coupled_operator_now": updated_pack_phase1_literal_cross_term_realization_closes_exact_coupled_operator_now,
        "updated_pack_constraint_elimination_followup_required": updated_pack_constraint_elimination_followup_required,
        "updated_pack_noncollapsed_ell0_closure_reserve_retained": updated_pack_noncollapsed_ell0_closure_reserve_retained,
        "trial3_family_primary_reuse_admissible_now": trial3_family_primary_reuse_admissible_now,
        "blind_vector_observable_gate_still_blocked": blind_vector_observable_gate_still_blocked,
        "pack_update_required_now": pack_update_required_now,
        "legacy_diagnostic_max_abs_ratio": float(old_literal_summary["legacy_diagnostic_max_abs_ratio"]),
        "legacy_diagnostic_alpha_at_q_theory": float(old_literal_summary["legacy_diagnostic_alpha_at_q_theory"]),
        "selected_primary_realization_form": "updated_pack_phase1_scalar_first_then_driven_fL_triangular_realization",
        "selected_secondary_completion_lane": "updated_pack_constraint_elimination",
        "selected_reserve_completion_lane": "updated_pack_noncollapsed_ell0_closure",
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": FOLLOWUP_ROUTE_NAME,
        "selected_followup_route_or_none": FOLLOWUP_ROUTE,
        "physical_reject_required": False,
    }

    declaration_payload = sign_base.payload(
        "8.7.56.2633",
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
                "prior_operator_gate": sign_base.display_path(PRIOR_OPERATOR_GATE),
                "old_literal_audit": sign_base.display_path(OLD_LITERAL_AUDIT),
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
            "overall_status": "vector_qball_form_factor_updated_pack_phase1_literal_cross_term_realization_audited",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "formulas": build_formulae(old_literal_formulas),
            "hits": {
                "status_branch_hit": sign_base.hit(status_text, "8.7.56.2627"),
                "roadmap_branch_hit": sign_base.hit(roadmap_text, ".2627-.2630"),
                "current_problem_hit": sign_base.hit(current_problem_text, "phase-1 literal cross-term realization"),
                "current_status_hit": sign_base.hit(current_status_text, "phase-1 literal cross-term realization"),
                "unified_roadmap_hit": sign_base.hit(unified_text, ".2627-.2630"),
                "long_roadmap_hit": sign_base.hit(long_text, ".2627-.2630"),
                "part5_hit": sign_base.hit(part5_text, "updated-pack phase-1 literal cross-term realization audit"),
                "phase1_exact_f0_hit": exact_f0_hit,
                "phase1_exact_fl_hit": exact_fl_hit,
            },
        },
    )
    declaration_paths = write_artifact("declaration_gate", declaration_payload)

    route_payload = {
        "generated_utc": sign_base.now_iso(),
        "phase": {
            "phase": 8,
            "step": "8.7.56.2634",
            "name": STEP_NAME + " route sync",
        },
        "inputs": declaration_paths,
        "rows": rows,
        "summary": summary,
        "decision": {
            "overall_status": "vector_qball_form_factor_updated_pack_phase1_literal_cross_term_realization_route_synced",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        "evidence": {
            "formulas": build_formulae(old_literal_formulas),
            "selected_route": {
                "next_route_name": NEXT_ROUTE_NAME,
                "next_route": NEXT_ROUTE,
                "followup_route_name": FOLLOWUP_ROUTE_NAME,
                "followup_route": FOLLOWUP_ROUTE,
            },
        },
    }
    route_paths = write_artifact("route_sync", route_payload)

    print("[ok] updated-pack literal cross-term realization audit artifacts written")
    print(f"  declaration_gate: {declaration_paths['json']}")
    print(f"  route_sync: {route_paths['json']}")


if __name__ == "__main__":
    main()
