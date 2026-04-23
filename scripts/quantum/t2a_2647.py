#!/usr/bin/env python3
"""Generate 8.7.56.2647-.2650 updated-pack literal constraint-elimination artifacts."""

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
        "8.7.56.2643-2646",
        "updated_pack_phase1_constraint_elimination_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
PRIOR_AUDIT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.2639-2642",
        "updated_pack_phase1_constraint_elimination_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
PRIOR_LITERAL_AUDIT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.2631-2634",
        "updated_pack_phase1_literal_cross_term_realization_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

POST_PHOTON_QFORM = PUBLIC_OUT / "mass_origin_v2_post_photon_nontransverse_two_by_two_quadratic_form_metrics.json"
POST_PHOTON_DIAG = PUBLIC_OUT / "mass_origin_v2_post_photon_nontransverse_diagonalization_basis_statement_metrics.json"
PHASE1_SOLVER = ROOT / "scripts" / "quantum" / "t2a_1419.py"
SOLVER_FIX_CANDIDATES = (
    Path(r"C:\Users\ogawa\Downloads\pmodel_v2_trial2_solver_fix_final.md"),
    ROOT
    / "output"
    / "private"
    / "quantum"
    / "expert_review_bundle_20260327_103258"
    / "pmodel_v2_trial2_solver_fix_final.md",
    ROOT
    / "output"
    / "private"
    / "quantum"
    / "expert_review_bundle_20260327_103144"
    / "pmodel_v2_trial2_solver_fix_final.md",
)
SOLVER_FIX = next((path for path in SOLVER_FIX_CANDIDATES if path.exists()), SOLVER_FIX_CANDIDATES[0])

STEP_TAG = "8.7.56.2647-2650"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack phase-1 "
    "literal constraint-elimination realization audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_phase1_literal_constraint_elimination_realization_audit",
    prefix="q",
)

PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_phase1_"
    "constraint_elimination_selected_literal_realization_primary_ell0_closure_"
    "reserve_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_phase1_"
    "literal_constraint_elimination_realization_supported_ell0_closure_followup_gate"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_phase1_constraint_"
    "elimination_realization_gate_ell0_refresh"
)
NEXT_ROUTE = "8.7.56.2651"
FOLLOWUP_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_noncollapsed_ell0_"
    "closure_audit"
)
FOLLOWUP_ROUTE = "8.7.56.2655"


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


# 関数: 2つの marker 間の text slice を返す。

def slice_between(text: str, start: str, end: str) -> str:
    """Return the text slice between two markers."""
    start_index = text.find(start)
    if start_index < 0:
        return ""

    end_index = text.find(end, start_index)
    if end_index < 0:
        return text[start_index:]

    return text[start_index:end_index]


# 関数: literal constraint-elimination audit で使う式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used in the literal constraint-elimination audit."""
    return {
        "quadratic_constraint_row": "-omega k * delta P_0 + omega^2 * delta P_L = 0",
        "field_strength_identity": "F_{0r}^{(P)} = i omega f_L - f_0'",
        "reduced_state_target": "y_red = [f_0, f_0', r f_L]",
        "ordering_rule": (
            "updated-pack phase-1 literal constraint elimination -> updated-pack "
            "noncollapsed ell=0 closure -> only then any pack-update reconsideration"
        ),
    }


# 関数: `.2647-.2650` を実行する。

def main() -> None:
    """Execute the updated-pack phase-1 literal constraint-elimination realization audit."""
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
        PRIOR_LITERAL_AUDIT,
        POST_PHOTON_QFORM,
        POST_PHOTON_DIAG,
        PHASE1_SOLVER,
        SOLVER_FIX,
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
    solver_fix_text = sign_base.read_text(SOLVER_FIX)

    prior_gate_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    prior_audit_summary = sign_base.read_json(PRIOR_AUDIT)["summary"]
    prior_literal_summary = sign_base.read_json(PRIOR_LITERAL_AUDIT)["summary"]
    qform_payload = sign_base.read_json(POST_PHOTON_QFORM)
    diag_payload = sign_base.read_json(POST_PHOTON_DIAG)
    qform_summary = qform_payload["summary"]
    diag_summary = diag_payload["summary"]

    phase1_exact_slice = slice_between(
        phase1_text,
        "def solve_exact_profile(",
        "def run_exact_scan(",
    )

    phase1_current_state_dimension = 4.0
    phase1_exact_solver_keeps_full_four_component_state = (
        "y0 = [float(amp0), 0.0, float(amp_l) * r0, float(amp_l)]" in phase1_exact_slice
        and "return [f0_prime, f0_double_prime, f_l_prime, f_l_double_prime]" in phase1_exact_slice
    )
    public_constraint_row_formula_available = bool(
        "-omega k" in str(qform_payload["formulas"]["quadratic_form_matrix"])
        and int(diag_summary["post_photon_nontransverse_constraint_mode_count"]) == 1
    )
    field_strength_identity_available = (
        sign_base.hit(solver_fix_text, "F_{0r}^{(P)} = i\\omega f_L - f_0'") is not None
    )
    phase1_reduced_state_target_dimension = 3.0 if public_constraint_row_formula_available else 0.0
    phase1_exact_solver_literal_constraint_elimination_present = (
        "y_red" in phase1_exact_slice
        or "reduced_state" in phase1_exact_slice
        or "constraint-eliminated" in phase1_exact_slice
    )
    updated_pack_phase1_literal_constraint_elimination_formula_available = bool(
        public_constraint_row_formula_available
        and field_strength_identity_available
        and qform_summary["working_action_nontransverse_two_by_two_quadratic_form_available"]
        and diag_summary["working_action_nontransverse_quadratic_diagonalization_available"]
    )
    updated_pack_phase1_literal_constraint_elimination_realization_supported_under_current_pack = bool(
        prior_gate_summary["gate_a_updated_pack_phase1_constraint_elimination_selected"]
        and prior_gate_summary["gate_b_updated_pack_phase1_literal_constraint_elimination_realization_promoted_next"]
        and prior_audit_summary["updated_pack_phase1_constraint_elimination_supported_under_current_pack"]
        and phase1_exact_solver_keeps_full_four_component_state
        and updated_pack_phase1_literal_constraint_elimination_formula_available
        and not phase1_exact_solver_literal_constraint_elimination_present
    )
    updated_pack_phase1_literal_constraint_elimination_realization_closes_exact_coupled_operator_now = False
    updated_pack_noncollapsed_ell0_closure_followup_required = bool(
        updated_pack_phase1_literal_constraint_elimination_realization_supported_under_current_pack
    )
    updated_pack_phase1_backreaction_followup_still_open = bool(
        prior_literal_summary["updated_pack_phase1_backreaction_followup_still_open"]
    )
    trial3_family_primary_reuse_admissible_now = False
    blind_vector_observable_gate_still_blocked = True
    pack_update_required_now = False

    rows = [
        sign_base.row(
            "phase1_exact_solver_keeps_full_four_component_state",
            "watch" if phase1_exact_solver_keeps_full_four_component_state else "reject",
            "phase-1 exact solver keeps the full four-component state",
            sign_base.truth(phase1_exact_solver_keeps_full_four_component_state),
            "The current exact pilot still literalizes four first-order components and therefore has one removable state component if the public constraint branch is used literally.",
        ),
        sign_base.row(
            "public_constraint_row_formula_available",
            "pass" if public_constraint_row_formula_available else "reject",
            "public quadratic-form constraint row available",
            sign_base.truth(public_constraint_row_formula_available),
            "The post-photon 2x2 backbone already freezes the non-propagating constraint row needed to target a reduced-state realization.",
        ),
        sign_base.row(
            "field_strength_identity_available",
            "pass" if field_strength_identity_available else "reject",
            "field-strength identity available",
            sign_base.truth(field_strength_identity_available),
            "The retained solver-fix note already freezes the identity F_{0r}^{(P)} = i omega f_L - f_0', so the literal reduced-state target is not guesswork.",
        ),
        sign_base.row(
            "phase1_reduced_state_target_dimension",
            "pass" if phase1_reduced_state_target_dimension == 3.0 else "reject",
            "phase-1 reduced-state target dimension",
            phase1_reduced_state_target_dimension,
            "One non-propagating branch implies the full four-component pilot should reduce to a three-component literal target before noncollapsed ell=0 closure is attempted.",
        ),
        sign_base.row(
            "phase1_exact_solver_literal_constraint_elimination_present",
            "pass" if phase1_exact_solver_literal_constraint_elimination_present else "reject",
            "phase-1 exact solver literal constraint elimination present",
            sign_base.truth(phase1_exact_solver_literal_constraint_elimination_present),
            "The present exact pilot still lacks an explicit reduced-state literalization such as y_red or an equivalent constraint-eliminated implementation.",
        ),
        sign_base.row(
            "updated_pack_phase1_literal_constraint_elimination_formula_available",
            "pass" if updated_pack_phase1_literal_constraint_elimination_formula_available else "reject",
            "updated-pack phase-1 literal constraint-elimination formula available",
            sign_base.truth(updated_pack_phase1_literal_constraint_elimination_formula_available),
            "The retained pack already contains both the post-photon constraint row and the field-strength identity, which together define the missing literal target.",
        ),
        sign_base.row(
            "updated_pack_phase1_literal_constraint_elimination_realization_supported_under_current_pack",
            "pass" if updated_pack_phase1_literal_constraint_elimination_realization_supported_under_current_pack else "reject",
            "updated-pack phase-1 literal constraint-elimination realization supported under current pack",
            sign_base.truth(updated_pack_phase1_literal_constraint_elimination_realization_supported_under_current_pack),
            "The missing move is now a literal solver realization of an already-fixed reduced-state target, not a new theory search.",
        ),
        sign_base.row(
            "updated_pack_phase1_literal_constraint_elimination_realization_closes_exact_coupled_operator_now",
            "pass" if updated_pack_phase1_literal_constraint_elimination_realization_closes_exact_coupled_operator_now else "reject",
            "updated-pack phase-1 literal constraint-elimination realization closes exact coupled operator now",
            sign_base.truth(updated_pack_phase1_literal_constraint_elimination_realization_closes_exact_coupled_operator_now),
            "Even after literal reduced-state realization is selected, noncollapsed ell=0 closure still remains downstream.",
        ),
        sign_base.row(
            "updated_pack_noncollapsed_ell0_closure_followup_required",
            "pass" if updated_pack_noncollapsed_ell0_closure_followup_required else "reject",
            "updated-pack noncollapsed ell=0 closure followup required",
            sign_base.truth(updated_pack_noncollapsed_ell0_closure_followup_required),
            "The reserve theorem remains noncollapsed ell=0 closure rather than another retry of the same phase-1 pilot.",
        ),
        sign_base.row(
            "updated_pack_phase1_backreaction_followup_still_open",
            "pass" if updated_pack_phase1_backreaction_followup_still_open else "reject",
            "updated-pack phase-1 backreaction followup still open",
            sign_base.truth(updated_pack_phase1_backreaction_followup_still_open),
            "Constraint elimination does not remove the earlier finding that reciprocal backreaction still remains downstream of the triangular literal realization.",
        ),
        sign_base.row(
            "blind_vector_observable_gate_still_blocked",
            "pass" if blind_vector_observable_gate_still_blocked else "reject",
            "blind-vector observable gate still blocked",
            sign_base.truth(blind_vector_observable_gate_still_blocked),
            "Even after reduced-state realization is selected, blind-vector direct computation stays blocked until the coupled operator is closed beyond the no-go theorem branch.",
        ),
        sign_base.row(
            "pack_update_required_now",
            "pass" if pack_update_required_now else "reject",
            "substantive pack update required now",
            sign_base.truth(pack_update_required_now),
            "Current literal reduced-state realization still advances inside the retained updated pack.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "retained_scalar_residual_rel": float(prior_gate_summary["retained_scalar_residual_rel"]),
        "phase1_current_state_dimension": phase1_current_state_dimension,
        "phase1_reduced_state_target_dimension": phase1_reduced_state_target_dimension,
        "phase1_exact_solver_keeps_full_four_component_state": phase1_exact_solver_keeps_full_four_component_state,
        "public_constraint_row_formula_available": public_constraint_row_formula_available,
        "field_strength_identity_available": field_strength_identity_available,
        "phase1_exact_solver_literal_constraint_elimination_present": phase1_exact_solver_literal_constraint_elimination_present,
        "updated_pack_phase1_literal_constraint_elimination_formula_available": updated_pack_phase1_literal_constraint_elimination_formula_available,
        "updated_pack_phase1_literal_constraint_elimination_realization_supported_under_current_pack": updated_pack_phase1_literal_constraint_elimination_realization_supported_under_current_pack,
        "updated_pack_phase1_literal_constraint_elimination_realization_closes_exact_coupled_operator_now": updated_pack_phase1_literal_constraint_elimination_realization_closes_exact_coupled_operator_now,
        "updated_pack_noncollapsed_ell0_closure_followup_required": updated_pack_noncollapsed_ell0_closure_followup_required,
        "updated_pack_phase1_backreaction_followup_still_open": updated_pack_phase1_backreaction_followup_still_open,
        "trial3_family_primary_reuse_admissible_now": trial3_family_primary_reuse_admissible_now,
        "blind_vector_observable_gate_still_blocked": blind_vector_observable_gate_still_blocked,
        "pack_update_required_now": pack_update_required_now,
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": FOLLOWUP_ROUTE_NAME,
        "selected_followup_route_or_none": FOLLOWUP_ROUTE,
        "physical_reject_required": False,
    }

    declaration_payload = sign_base.payload(
        "8.7.56.2649",
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
                "prior_literal_audit": sign_base.display_path(PRIOR_LITERAL_AUDIT),
                "post_photon_qform": sign_base.display_path(POST_PHOTON_QFORM),
                "post_photon_diag": sign_base.display_path(POST_PHOTON_DIAG),
                "phase1_solver": sign_base.display_path(PHASE1_SOLVER),
                "solver_fix": sign_base.display_path(SOLVER_FIX),
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
            "overall_status": "vector_qball_form_factor_updated_pack_phase1_literal_constraint_elimination_realization_audited",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "formulas": build_formulae(),
            "hits": {
                "status_branch_hit": sign_base.hit(status_text, "8.7.56.2647"),
                "roadmap_branch_hit": sign_base.hit(roadmap_text, ".2647-.2650"),
                "current_problem_hit": sign_base.hit(current_problem_text, "phase-1 literal constraint-elimination realization audit"),
                "current_status_hit": sign_base.hit(current_status_text, "phase-1 literal constraint-elimination realization audit"),
                "unified_roadmap_hit": sign_base.hit(unified_text, ".2647-.2650"),
                "long_roadmap_hit": sign_base.hit(long_text, ".2647-.2650"),
                "part5_hit": sign_base.hit(part5_text, "updated-pack phase-1 literal constraint-elimination realization audit"),
                "qform_matrix_hit": qform_payload["formulas"]["quadratic_form_matrix"],
                "diag_constraint_rule_hit": diag_payload["formulas"]["constraint_branch_rule"],
                "phase1_state_hit": sign_base.hit(phase1_exact_slice, "y0 = [float(amp0), 0.0, float(amp_l) * r0, float(amp_l)]"),
                "phase1_return_hit": sign_base.hit(phase1_exact_slice, "return [f0_prime, f0_double_prime, f_l_prime, f_l_double_prime]"),
                "solver_fix_identity_hit": sign_base.hit(solver_fix_text, "F_{0r}^{(P)} = i\\omega f_L - f_0'"),
            },
        },
    )
    declaration_paths = write_artifact("declaration_gate", declaration_payload)

    route_payload = {
        "generated_utc": sign_base.now_iso(),
        "phase": {
            "phase": 8,
            "step": "8.7.56.2650",
            "name": STEP_NAME + " route sync",
        },
        "inputs": declaration_paths,
        "rows": rows,
        "summary": summary,
        "decision": {
            "overall_status": "vector_qball_form_factor_updated_pack_phase1_literal_constraint_elimination_realization_route_synced",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        "evidence": {
            "selected_route": {
                "next_route_name": NEXT_ROUTE_NAME,
                "next_route": NEXT_ROUTE,
                "followup_route_name": FOLLOWUP_ROUTE_NAME,
                "followup_route": FOLLOWUP_ROUTE,
            }
        },
    }
    route_paths = write_artifact("route_sync", route_payload)

    print("[ok] updated-pack phase-1 literal constraint-elimination realization artifacts written")
    print(f"  declaration_gate: {declaration_paths['json']}")
    print(f"  route_sync: {route_paths['json']}")


if __name__ == "__main__":
    main()
