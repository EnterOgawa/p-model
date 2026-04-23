#!/usr/bin/env python3
"""Generate 8.7.56.2447-.2450 updated-pack exact ell=0 series/operator audit artifacts."""

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
        "8.7.56.2443-2446",
        "substantive_pack_update_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
PRIOR_PACK_AUDIT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.2439-2442",
        "substantive_pack_update_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
RECIPROCAL_AUDIT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.2415-2418",
        "phase1_reciprocal_backreaction_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
NONLINEAR_AUDIT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.2423-2426",
        "phase1_nonheuristic_two_component_nonlinear_closure_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

SOLVER_FIX = Path(r"C:\Users\ogawa\Downloads\pmodel_v2_trial2_solver_fix_final.md")
NEXT_STEPS = Path(r"C:\Users\ogawa\Downloads\trial2_vector_qball_next_steps_20260327.md")

STEP_TAG = "8.7.56.2447-2450"
STEP_NAME = "Trial-2 numeric alpha vector Q-ball form-factor updated-pack exact ell=0 series/operator audit"
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_exact_ell0_series_operator_audit",
    prefix="q",
)

PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_substantive_pack_exact_ell0_"
    "series_operator_primary_effective_source_secondary_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_exact_ell0_"
    "series_operator_surface_explicit_effective_source_followup_gate"
)
NEXT_ROUTE_NAME = "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_exact_ell0_series_operator_gate_effective_source_refresh"
NEXT_ROUTE = "8.7.56.2451"
FOLLOWUP_ROUTE_NAME = "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_exact_effective_source_theorem_audit"
FOLLOWUP_ROUTE = "8.7.56.2455"


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


# 関数: updated-pack exact ell=0 series/operator audit で使う式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used in the updated-pack exact ell=0 series/operator audit."""
    return {
        "exact_two_component_series": "f_0(r)=a_0+a_2 r^2 + a_4 r^4 + ...,  f_L(r)=b_1 r + b_3 r^3 + b_5 r^5 + ...",
        "b1_decision_rule": "b_1=0 => ell=0 vector route no-go,  b_1 sourced nonzero => exact vector route reopen,  b_1 free => boundary-value / shooting redesign",
        "longitudinal_placeholder_operator": "f_L'' + 2 f_L'/r + (omega^2-m_0^2) f_L = S[f_0],  then L_L[f_L] = S[f_0]",
        "exact_formulation_requirements": "constraint equation + Stueckelberg sector + boundary conditions + decaying tail condition => Green function / shooting formulation",
        "solver_fix_placeholders": "f_0'' + 2 f_0'/r + (beta^2 - 1) f_0 + NL(f_0) = - coupling(f_L),  f_L'' + 2 f_L'/r - 2 f_L/r^2 - kappa^2 f_L = beta y_0'(x)",
    }


# 関数: `.2447-.2450` を実行する。

def main() -> None:
    """Execute the updated-pack exact ell=0 series/operator audit."""
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
        PRIOR_PACK_AUDIT,
        RECIPROCAL_AUDIT,
        NONLINEAR_AUDIT,
        SOLVER_FIX,
        NEXT_STEPS,
    ):
        sign_base.require(path)

    status_text = sign_base.read_text(STATUS)
    roadmap_text = sign_base.read_text(ROADMAP)
    current_problem_text = sign_base.read_text(CURRENT_PROBLEM)
    current_status_text = sign_base.read_text(CURRENT_STATUS)
    unified_text = sign_base.read_text(UNIFIED_ROADMAP)
    long_text = sign_base.read_text(LONG_ROADMAP)
    part5_text = sign_base.read_text(PART5)
    solver_fix_text = sign_base.read_text(SOLVER_FIX)
    next_steps_text = sign_base.read_text(NEXT_STEPS)

    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    pack_audit_summary = sign_base.read_json(PRIOR_PACK_AUDIT)["summary"]
    reciprocal_summary = sign_base.read_json(RECIPROCAL_AUDIT)["summary"]
    nonlinear_summary = sign_base.read_json(NONLINEAR_AUDIT)["summary"]

    updated_pack_exact_ell0_series_operator_audit_selected = bool(
        prior_summary["gate_a_updated_pack_exact_ell0_series_operator_selected"]
        and prior_summary["updated_pack_primary_route_is_theorem_surface_not_numeric_recompute"]
    )
    updated_pack_exact_ell0_series_surface_explicit = bool(
        sign_base.hit(next_steps_text, "f_0(r)=a_0+a_2 r^2 + a_4 r^4 + \\cdots") is not None
        and sign_base.hit(next_steps_text, "f_L(r)=b_1 r + b_3 r^3 + b_5 r^5 + \\cdots") is not None
    )
    updated_pack_b1_forced_zero_no_go_branch_explicit = bool(
        sign_base.hit(next_steps_text, "b_1` が **方程式から 0 に固定**されるなら") is not None
    )
    updated_pack_b1_nonzero_source_reopen_branch_explicit = bool(
        sign_base.hit(next_steps_text, "b_1` が **nonzero に source される**なら") is not None
    )
    updated_pack_b1_free_shooting_branch_explicit = bool(
        sign_base.hit(next_steps_text, "b_1` が **自由 shooting parameter** なら") is not None
    )
    updated_pack_b1_decision_surface_complete = bool(
        updated_pack_b1_forced_zero_no_go_branch_explicit
        and updated_pack_b1_nonzero_source_reopen_branch_explicit
        and updated_pack_b1_free_shooting_branch_explicit
    )
    updated_pack_longitudinal_placeholder_equation_available = bool(
        sign_base.hit(next_steps_text, "f_L'' + \\frac{2}{r} f_L' + (\\omega^2-m_0^2)f_L = S[f_0]") is not None
    )
    updated_pack_exact_longitudinal_operator_surface_explicit = bool(
        sign_base.hit(next_steps_text, "L_L[f_L] = S[f_0]") is not None
    )
    updated_pack_green_function_shooting_requirements_explicit = bool(
        sign_base.hit(next_steps_text, "constraint equation") is not None
        and sign_base.hit(next_steps_text, "Stückelberg sector") is not None
        and sign_base.hit(next_steps_text, "boundary conditions") is not None
        and sign_base.hit(next_steps_text, "decaying tail condition") is not None
        and sign_base.hit(next_steps_text, "Green function / shooting formulation") is not None
    )
    solver_fix_driven_fl_placeholder_available = bool(
        sign_base.hit(solver_fix_text, "f_L'' + \\frac{2}{r}f_L' - \\frac{2}{r^2}f_L - \\kappa^2 f_L = \\beta\\,y_0'(x)") is not None
    )
    solver_fix_backreaction_placeholder_available = bool(
        sign_base.hit(solver_fix_text, "\\text{coupling}(f_L)") is not None
    )
    current_pack_exact_series_operator_still_absent = bool(
        pack_audit_summary["current_pack_missing_action_formulae_remain_placeholder_only"]
        and not reciprocal_summary["phase1_literal_reciprocal_backreaction_formula_available"]
        and not nonlinear_summary["phase1_literal_two_component_nonlinear_formula_available"]
    )
    updated_pack_exact_ell0_series_operator_supported_now = bool(
        updated_pack_exact_ell0_series_operator_audit_selected
        and pack_audit_summary["substantive_pack_update_adoptable_now"]
        and updated_pack_exact_ell0_series_surface_explicit
        and updated_pack_b1_decision_surface_complete
        and updated_pack_exact_longitudinal_operator_surface_explicit
        and updated_pack_green_function_shooting_requirements_explicit
        and solver_fix_driven_fl_placeholder_available
        and solver_fix_backreaction_placeholder_available
    )
    updated_pack_exact_ell0_series_operator_closes_missing_action_blocker_now = False
    updated_pack_effective_source_theorem_followup_retained = bool(
        prior_summary["gate_b_effective_source_theorem_retained_as_followup"]
    )
    farther_hybrid_continuation_reopen_required_now = False

    rows = [
        sign_base.row(
            "updated_pack_exact_ell0_series_operator_audit_selected",
            "pass" if updated_pack_exact_ell0_series_operator_audit_selected else "reject",
            "updated-pack exact ell=0 series/operator audit selected",
            sign_base.truth(updated_pack_exact_ell0_series_operator_audit_selected),
            "The substantive pack-update gate already promoted this theorem surface as the next mainline.",
        ),
        sign_base.row(
            "updated_pack_exact_ell0_series_surface_explicit",
            "pass" if updated_pack_exact_ell0_series_surface_explicit else "reject",
            "updated-pack exact ell=0 series surface explicit",
            sign_base.truth(updated_pack_exact_ell0_series_surface_explicit),
            "Step A already spells out the exact two-component near-origin series rather than leaving the updated pack at a heuristic rho-only level.",
        ),
        sign_base.row(
            "updated_pack_b1_decision_surface_complete",
            "pass" if updated_pack_b1_decision_surface_complete else "reject",
            "updated-pack b1 decision surface complete",
            sign_base.truth(updated_pack_b1_decision_surface_complete),
            "The updated pack already fixes the exact yes/no discriminator: forced zero, sourced nonzero, or free shooting parameter.",
        ),
        sign_base.row(
            "updated_pack_longitudinal_placeholder_equation_available",
            "pass" if updated_pack_longitudinal_placeholder_equation_available else "reject",
            "updated-pack longitudinal placeholder equation available",
            sign_base.truth(updated_pack_longitudinal_placeholder_equation_available),
            "Step B already isolates the longitudinal equation as a source-driven exact operator problem rather than a blind numeric continuation problem.",
        ),
        sign_base.row(
            "updated_pack_exact_longitudinal_operator_surface_explicit",
            "pass" if updated_pack_exact_longitudinal_operator_surface_explicit else "reject",
            "updated-pack exact longitudinal operator surface explicit",
            sign_base.truth(updated_pack_exact_longitudinal_operator_surface_explicit),
            "The new primary theorem target is explicitly the exact operator equation `L_L[f_L] = S[f_0]`.",
        ),
        sign_base.row(
            "updated_pack_green_function_shooting_requirements_explicit",
            "pass" if updated_pack_green_function_shooting_requirements_explicit else "reject",
            "updated-pack Green function / shooting requirements explicit",
            sign_base.truth(updated_pack_green_function_shooting_requirements_explicit),
            "Constraint, Stückelberg, boundary, and decaying-tail requirements are already enumerated as part of the exact operator surface.",
        ),
        sign_base.row(
            "solver_fix_driven_fl_placeholder_available",
            "pass" if solver_fix_driven_fl_placeholder_available else "reject",
            "solver-fix driven fL placeholder available",
            sign_base.truth(solver_fix_driven_fl_placeholder_available),
            "The retained solver-fix memo already contains the driven longitudinal ODE placeholder that the updated-pack operator must replace by an exact theorem surface.",
        ),
        sign_base.row(
            "solver_fix_backreaction_placeholder_available",
            "pass" if solver_fix_backreaction_placeholder_available else "reject",
            "solver-fix backreaction placeholder available",
            sign_base.truth(solver_fix_backreaction_placeholder_available),
            "The updated pack is anchored to the same missing action-level placeholders identified in the retained solver-fix memo.",
        ),
        sign_base.row(
            "current_pack_exact_series_operator_still_absent",
            "watch" if current_pack_exact_series_operator_still_absent else "pass",
            "current pack exact series/operator still absent",
            sign_base.truth(current_pack_exact_series_operator_still_absent),
            "The retained pack still lacks literal reciprocal backreaction and literal two-component nonlinear closure, so the exact ell=0 operator is still open under the old surface.",
        ),
        sign_base.row(
            "updated_pack_exact_ell0_series_operator_supported_now",
            "pass" if updated_pack_exact_ell0_series_operator_supported_now else "reject",
            "updated-pack exact ell=0 series/operator supported now",
            sign_base.truth(updated_pack_exact_ell0_series_operator_supported_now),
            "The new theorem surface is explicit enough to be an honest mainline audit without reopening density/proxy/eigenvalue retries.",
        ),
        sign_base.row(
            "updated_pack_exact_ell0_series_operator_closes_missing_action_blocker_now",
            "pass" if updated_pack_exact_ell0_series_operator_closes_missing_action_blocker_now else "reject",
            "updated-pack exact ell=0 series/operator closes missing-action blocker now",
            sign_base.truth(updated_pack_exact_ell0_series_operator_closes_missing_action_blocker_now),
            "This audit fixes the theorem target but does not claim that the exact operator itself is already derived in the retained public canon.",
        ),
        sign_base.row(
            "updated_pack_effective_source_theorem_followup_retained",
            "pass" if updated_pack_effective_source_theorem_followup_retained else "reject",
            "updated-pack effective source theorem followup retained",
            sign_base.truth(updated_pack_effective_source_theorem_followup_retained),
            "After the operator surface is audited, the effective source theorem remains the next downstream canonical observable question.",
        ),
        sign_base.row(
            "farther_hybrid_continuation_reopen_required_now",
            "pass" if farther_hybrid_continuation_reopen_required_now else "reject",
            "farther hybrid continuation reopen required now",
            sign_base.truth(farther_hybrid_continuation_reopen_required_now),
            "Extra q-range evidence remains reserve-only because the blocker is still localized to the updated-pack theorem surface.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "retained_scalar_residual_rel": float(prior_summary["retained_scalar_residual_rel"]),
        "updated_pack_exact_ell0_series_operator_audit_selected": updated_pack_exact_ell0_series_operator_audit_selected,
        "updated_pack_exact_ell0_series_surface_explicit": updated_pack_exact_ell0_series_surface_explicit,
        "updated_pack_b1_forced_zero_no_go_branch_explicit": updated_pack_b1_forced_zero_no_go_branch_explicit,
        "updated_pack_b1_nonzero_source_reopen_branch_explicit": updated_pack_b1_nonzero_source_reopen_branch_explicit,
        "updated_pack_b1_free_shooting_branch_explicit": updated_pack_b1_free_shooting_branch_explicit,
        "updated_pack_b1_decision_surface_complete": updated_pack_b1_decision_surface_complete,
        "updated_pack_longitudinal_placeholder_equation_available": updated_pack_longitudinal_placeholder_equation_available,
        "updated_pack_exact_longitudinal_operator_surface_explicit": updated_pack_exact_longitudinal_operator_surface_explicit,
        "updated_pack_green_function_shooting_requirements_explicit": updated_pack_green_function_shooting_requirements_explicit,
        "solver_fix_driven_fl_placeholder_available": solver_fix_driven_fl_placeholder_available,
        "solver_fix_backreaction_placeholder_available": solver_fix_backreaction_placeholder_available,
        "current_pack_exact_series_operator_still_absent": current_pack_exact_series_operator_still_absent,
        "updated_pack_exact_ell0_series_operator_supported_now": updated_pack_exact_ell0_series_operator_supported_now,
        "updated_pack_exact_ell0_series_operator_closes_missing_action_blocker_now": updated_pack_exact_ell0_series_operator_closes_missing_action_blocker_now,
        "updated_pack_effective_source_theorem_followup_retained": updated_pack_effective_source_theorem_followup_retained,
        "farther_hybrid_continuation_reopen_required_now": farther_hybrid_continuation_reopen_required_now,
        "selected_primary_pack_update_surface": "exact_ell0_two_component_series_and_longitudinal_operator",
        "selected_secondary_pack_update_surface": "exact_effective_source_theorem",
        "selected_reserve_completion_lane": "farther_hybrid_extra_q_range_only",
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": FOLLOWUP_ROUTE_NAME,
        "selected_followup_route_or_none": FOLLOWUP_ROUTE,
        "physical_reject_required": False,
    }

    declaration_payload = sign_base.payload(
        "8.7.56.2449",
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
                "prior_pack_audit": sign_base.display_path(PRIOR_PACK_AUDIT),
                "reciprocal_audit": sign_base.display_path(RECIPROCAL_AUDIT),
                "nonlinear_audit": sign_base.display_path(NONLINEAR_AUDIT),
                "solver_fix": sign_base.display_path(SOLVER_FIX),
                "next_steps": sign_base.display_path(NEXT_STEPS),
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
            "overall_status": "vector_qball_form_factor_updated_pack_exact_ell0_series_operator_audit_declared",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "formulas": build_formulae(),
            "hits": {
                "status_branch_hit": sign_base.hit(status_text, "8.7.56.2447"),
                "roadmap_branch_hit": sign_base.hit(roadmap_text, ".2447-.2450"),
                "current_problem_hit": sign_base.hit(current_problem_text, "updated-pack exact ell=0 series/operator audit"),
                "current_status_hit": sign_base.hit(current_status_text, "updated-pack exact ell=0 series/operator audit"),
                "unified_roadmap_hit": sign_base.hit(unified_text, ".2439-.2442"),
                "long_roadmap_hit": sign_base.hit(long_text, ".2439-.2442"),
                "part5_hit": sign_base.hit(part5_text, "updated-pack exact ell=0 series/operator audit"),
                "next_steps_step_a_hit": sign_base.hit(next_steps_text, "### Step A."),
                "next_steps_step_b_hit": sign_base.hit(next_steps_text, "### Step B."),
                "next_steps_b1_zero_hit": sign_base.hit(next_steps_text, "b_1` が **方程式から 0 に固定**されるなら"),
                "next_steps_b1_nonzero_hit": sign_base.hit(next_steps_text, "b_1` が **nonzero に source される**なら"),
                "next_steps_b1_free_hit": sign_base.hit(next_steps_text, "b_1` が **自由 shooting parameter** なら"),
                "next_steps_operator_hit": sign_base.hit(next_steps_text, "L_L[f_L] = S[f_0]"),
                "next_steps_green_hit": sign_base.hit(next_steps_text, "Green function / shooting formulation"),
                "solver_fix_driven_fl_hit": sign_base.hit(solver_fix_text, "f_L'' + \\frac{2}{r}f_L' - \\frac{2}{r^2}f_L - \\kappa^2 f_L = \\beta\\,y_0'(x)"),
                "solver_fix_backreaction_hit": sign_base.hit(solver_fix_text, "\\text{coupling}(f_L)"),
            },
        },
    )
    declaration_paths = write_artifact("declaration_gate", declaration_payload)

    route_payload = {
        "generated_utc": sign_base.now_iso(),
        "phase": {
            "phase": 8,
            "step": "8.7.56.2450",
            "name": STEP_NAME + " route sync",
        },
        "inputs": declaration_paths,
        "rows": rows,
        "summary": summary,
        "decision": {
            "overall_status": "vector_qball_form_factor_updated_pack_exact_ell0_series_operator_route_synced",
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
    write_artifact("route_sync", route_payload)

    print(f"[done] {STEP_TAG} updated-pack exact ell=0 series/operator audit completed")


if __name__ == "__main__":
    main()
