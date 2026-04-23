#!/usr/bin/env python3
"""Generate 8.7.56.2423-.2426 phase-1 nonlinear-closure audit artifacts."""

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
        "8.7.56.2419-2422",
        "phase1_reciprocal_backreaction_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
PRIOR_AUDIT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.2415-2418",
        "phase1_reciprocal_backreaction_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
NONCOLLAPSED_AUDIT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.2407-2410",
        "noncollapsed_ell0_closure_audit",
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
SOLVER_FIX = Path(r"C:\Users\ogawa\Downloads\pmodel_v2_trial2_solver_fix_final.md")
NEXT_STEPS = Path(r"C:\Users\ogawa\Downloads\trial2_vector_qball_next_steps_20260327.md")

STEP_TAG = "8.7.56.2423-2426"
STEP_NAME = "Trial-2 numeric alpha vector Q-ball form-factor phase-1 non-heuristic two-component nonlinear closure audit"
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "phase1_nonheuristic_two_component_nonlinear_closure_audit",
    prefix="q",
)

PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_phase1_shared_rho_even_backreaction_only_"
    "nonlinear_closure_primary_trial3_ell0_reserve_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_phase1_nonheuristic_two_component_"
    "nonlinear_closure_not_literal_trial3_ell0_reserve_gate"
)
NEXT_ROUTE_NAME = "trial2_numeric_alpha_vector_qball_form_factor_phase1_nonlinear_closure_gate_trial3_ell0_reserve_refresh"
NEXT_ROUTE = "8.7.56.2427"
FOLLOWUP_ROUTE_NAME = "trial2_numeric_alpha_vector_qball_form_factor_trial3_ell0_closure_reserve_audit"
FOLLOWUP_ROUTE = "8.7.56.2431"


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


# 関数: nonlinear-closure audit で使う式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used in the nonlinear-closure audit."""
    return {
        "shared_rho_heuristic": "rho = sqrt(f_0^2 + f_L^2), nonlinear_coeff = 3 rho + rho^2",
        "current_exact_solver": "f_0'' + ... = - nonlinear_coeff f_0,  f_L'' + ... = - nonlinear_coeff f_L",
        "literal_target": "f_0'' + ... = - NL_0(f_0, f_L),  f_L'' + ... = - NL_L(f_0, f_L)",
        "closure_rule": "non-heuristic two-component closure requires component-specific nonlinear/backreaction structure, not one shared scalar coefficient",
    }


# 関数: `.2423-.2426` を実行する。

def main() -> None:
    """Execute the phase-1 non-heuristic two-component nonlinear-closure audit."""
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
        NONCOLLAPSED_AUDIT,
        ELL0_OPERATOR_AUDIT,
        PHASE1_SOLVER,
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
    phase1_text = sign_base.read_text(PHASE1_SOLVER)
    solver_fix_text = sign_base.read_text(SOLVER_FIX)
    next_steps_text = sign_base.read_text(NEXT_STEPS)

    prior_gate_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    prior_audit_summary = sign_base.read_json(PRIOR_AUDIT)["summary"]
    noncollapsed_summary = sign_base.read_json(NONCOLLAPSED_AUDIT)["summary"]
    ell0_summary = sign_base.read_json(ELL0_OPERATOR_AUDIT)["summary"]

    phase1_exact_slice = slice_between(
        phase1_text,
        "def solve_exact_profile(",
        "def run_exact_scan(",
    )

    nonlinear_closure_primary_selected = bool(
        prior_gate_summary["gate_b_nonheuristic_two_component_nonlinear_closure_promoted_next"]
        and prior_audit_summary["nonheuristic_two_component_nonlinear_closure_primary_followup_required"]
        and noncollapsed_summary["phase1_exact_solver_nonheuristic_two_component_nonlinear_closure_present"] is False
    )
    phase1_exact_solver_shared_single_nonlinear_coeff_only = bool(
        "rho = math.sqrt(max(f0 * f0 + f_l * f_l, 0.0))" in phase1_exact_slice
        and "nonlinear_coeff = 3.0 * rho + rho * rho" in phase1_exact_slice
        and "f0_double_prime = -(2.0 / safe_r) * f0_prime - (float(beta * beta) - float(pivot.RADIAL_MASS_SQUARED)) * f0 - nonlinear_coeff * f0" in phase1_exact_slice
        and "f_l_double_prime = -(2.0 / safe_r) * f_l_prime - (float(beta * beta) - float(pivot.LONGITUDINAL_DIRECT_MASS_SQUARED)) * f_l - nonlinear_coeff * f_l" in phase1_exact_slice
    )
    phase1_exact_solver_component_specific_nonlinear_feedback_present = bool(
        "nonlinear_coeff_f0" in phase1_exact_slice
        or "nonlinear_coeff_fl" in phase1_exact_slice
        or "nl_f0" in phase1_exact_slice.lower()
        or "nl_fl" in phase1_exact_slice.lower()
        or "coupling(" in phase1_exact_slice
    )
    phase1_exact_solver_nonheuristic_two_component_nonlinear_closure_present = bool(
        (not phase1_exact_solver_shared_single_nonlinear_coeff_only)
        and phase1_exact_solver_component_specific_nonlinear_feedback_present
    )
    solver_fix_nonlinear_placeholder_available = bool(
        sign_base.hit(solver_fix_text, "\\text{NL}(f_0)") is not None
        and sign_base.hit(solver_fix_text, "\\text{coupling}(f_L)") is not None
    )
    next_steps_two_component_series_target_available = bool(
        sign_base.hit(next_steps_text, "### Step A.") is not None
        and sign_base.hit(next_steps_text, "f_0(r)=a_0+a_2 r^2 + a_4 r^4 + \\cdots") is not None
        and sign_base.hit(next_steps_text, "f_L(r)=b_1 r + b_3 r^3 + b_5 r^5 + \\cdots") is not None
    )
    exact_action_level_closed_ell0_operator_available = bool(
        ell0_summary["exact_action_level_closed_ell0_operator_available"]
    )
    phase1_nonheuristic_two_component_nonlinear_closure_supported_under_current_pack = bool(
        nonlinear_closure_primary_selected
        and phase1_exact_solver_shared_single_nonlinear_coeff_only
        and solver_fix_nonlinear_placeholder_available
        and next_steps_two_component_series_target_available
        and not exact_action_level_closed_ell0_operator_available
    )
    phase1_literal_two_component_nonlinear_formula_available = False
    phase1_nonheuristic_two_component_nonlinear_closure_closes_exact_coupled_operator_now = False
    trial3_family_ell0_closure_reserve_retained = bool(
        prior_gate_summary["gate_c_trial3_ell0_closure_reserve_retained"]
    )
    pack_update_required_now = False

    rows = [
        sign_base.row(
            "nonlinear_closure_primary_selected",
            "pass" if nonlinear_closure_primary_selected else "reject",
            "non-heuristic two-component nonlinear closure primary selected",
            sign_base.truth(nonlinear_closure_primary_selected),
            "This audit starts only after `.2419-.2422` promoted the nonlinear-closure lane as the next exact completion move.",
        ),
        sign_base.row(
            "phase1_exact_solver_shared_single_nonlinear_coeff_only",
            "watch" if phase1_exact_solver_shared_single_nonlinear_coeff_only else "pass",
            "phase-1 exact solver uses one shared nonlinear coefficient only",
            sign_base.truth(phase1_exact_solver_shared_single_nonlinear_coeff_only),
            "The present pilot still closes both component equations through the same scalarized `3 rho + rho^2` coefficient.",
        ),
        sign_base.row(
            "phase1_exact_solver_component_specific_nonlinear_feedback_present",
            "pass" if phase1_exact_solver_component_specific_nonlinear_feedback_present else "reject",
            "phase-1 exact solver component-specific nonlinear feedback present",
            sign_base.truth(phase1_exact_solver_component_specific_nonlinear_feedback_present),
            "A literal two-component closure would need component-specific nonlinear/backreaction structure rather than one shared scalar coefficient.",
        ),
        sign_base.row(
            "phase1_exact_solver_nonheuristic_two_component_nonlinear_closure_present",
            "pass" if phase1_exact_solver_nonheuristic_two_component_nonlinear_closure_present else "reject",
            "phase-1 exact solver non-heuristic two-component nonlinear closure present",
            sign_base.truth(phase1_exact_solver_nonheuristic_two_component_nonlinear_closure_present),
            "Current code still does not literalize distinct nonlinear closure terms for `f_0` and `f_L`.",
        ),
        sign_base.row(
            "solver_fix_nonlinear_placeholder_available",
            "pass" if solver_fix_nonlinear_placeholder_available else "reject",
            "solver-fix nonlinear placeholder available",
            sign_base.truth(solver_fix_nonlinear_placeholder_available),
            "The retained solver-fix note already isolates `NL(f_0)` and `coupling(f_L)` as missing ingredients, so the lane is supported even though code is not literalized.",
        ),
        sign_base.row(
            "next_steps_two_component_series_target_available",
            "pass" if next_steps_two_component_series_target_available else "reject",
            "next-steps two-component series target available",
            sign_base.truth(next_steps_two_component_series_target_available),
            "The retained next-steps note still identifies the exact two-component near-origin series as the decisive theorem target.",
        ),
        sign_base.row(
            "exact_action_level_closed_ell0_operator_available",
            "pass" if exact_action_level_closed_ell0_operator_available else "reject",
            "exact action-level closed ell=0 operator available",
            sign_base.truth(exact_action_level_closed_ell0_operator_available),
            "A fully literal two-component nonlinear closure would close the operator only if the exact ell=0 action-level operator were already complete, which it is not.",
        ),
        sign_base.row(
            "phase1_nonheuristic_two_component_nonlinear_closure_supported_under_current_pack",
            "pass" if phase1_nonheuristic_two_component_nonlinear_closure_supported_under_current_pack else "reject",
            "phase-1 non-heuristic two-component nonlinear closure supported under current pack",
            sign_base.truth(phase1_nonheuristic_two_component_nonlinear_closure_supported_under_current_pack),
            "The retained pack still supports this lane as an internal theorem-completion target without reopening farther-q evidence or external-input routes.",
        ),
        sign_base.row(
            "phase1_literal_two_component_nonlinear_formula_available",
            "pass" if phase1_literal_two_component_nonlinear_formula_available else "reject",
            "phase-1 literal two-component nonlinear formula available",
            sign_base.truth(phase1_literal_two_component_nonlinear_formula_available),
            "No current public note or exact-solver implementation provides the literal component-specific nonlinear closure formula.",
        ),
        sign_base.row(
            "phase1_nonheuristic_two_component_nonlinear_closure_closes_exact_coupled_operator_now",
            "pass" if phase1_nonheuristic_two_component_nonlinear_closure_closes_exact_coupled_operator_now else "reject",
            "phase-1 non-heuristic two-component nonlinear closure closes exact coupled operator now",
            sign_base.truth(phase1_nonheuristic_two_component_nonlinear_closure_closes_exact_coupled_operator_now),
            "The coupled operator still stays open because the present pack never gets beyond the shared-rho heuristic closure.",
        ),
        sign_base.row(
            "trial3_family_ell0_closure_reserve_retained",
            "pass" if trial3_family_ell0_closure_reserve_retained else "reject",
            "trial-3 ell=0 closure reserve retained",
            sign_base.truth(trial3_family_ell0_closure_reserve_retained),
            "The old trial-3 family remains reserve-only while the phase-1 nonlinear-closure lane fails to become literal.",
        ),
        sign_base.row(
            "pack_update_required_now",
            "pass" if pack_update_required_now else "reject",
            "substantive pack update required now",
            sign_base.truth(pack_update_required_now),
            "The lane is still an internal theorem-completion issue inside the retained pack.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "retained_scalar_residual_rel": float(prior_gate_summary["retained_scalar_residual_rel"]),
        "nonlinear_closure_primary_selected": nonlinear_closure_primary_selected,
        "phase1_exact_solver_shared_single_nonlinear_coeff_only": phase1_exact_solver_shared_single_nonlinear_coeff_only,
        "phase1_exact_solver_component_specific_nonlinear_feedback_present": phase1_exact_solver_component_specific_nonlinear_feedback_present,
        "phase1_exact_solver_nonheuristic_two_component_nonlinear_closure_present": phase1_exact_solver_nonheuristic_two_component_nonlinear_closure_present,
        "solver_fix_nonlinear_placeholder_available": solver_fix_nonlinear_placeholder_available,
        "next_steps_two_component_series_target_available": next_steps_two_component_series_target_available,
        "exact_action_level_closed_ell0_operator_available": exact_action_level_closed_ell0_operator_available,
        "phase1_nonheuristic_two_component_nonlinear_closure_supported_under_current_pack": phase1_nonheuristic_two_component_nonlinear_closure_supported_under_current_pack,
        "phase1_literal_two_component_nonlinear_formula_available": phase1_literal_two_component_nonlinear_formula_available,
        "phase1_nonheuristic_two_component_nonlinear_closure_closes_exact_coupled_operator_now": phase1_nonheuristic_two_component_nonlinear_closure_closes_exact_coupled_operator_now,
        "trial3_family_ell0_closure_reserve_retained": trial3_family_ell0_closure_reserve_retained,
        "pack_update_required_now": pack_update_required_now,
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": FOLLOWUP_ROUTE_NAME,
        "selected_followup_route_or_none": FOLLOWUP_ROUTE,
        "physical_reject_required": False,
    }

    declaration_payload = sign_base.payload(
        "8.7.56.2425",
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
                "noncollapsed_audit": sign_base.display_path(NONCOLLAPSED_AUDIT),
                "ell0_operator_audit": sign_base.display_path(ELL0_OPERATOR_AUDIT),
                "phase1_solver": sign_base.display_path(PHASE1_SOLVER),
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
            "overall_status": "vector_qball_form_factor_phase1_nonheuristic_two_component_nonlinear_closure_audit_declared",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "formulas": build_formulae(),
            "hits": {
                "status_branch_hit": sign_base.hit(status_text, "8.7.56.2423"),
                "roadmap_branch_hit": sign_base.hit(roadmap_text, ".2423-.2426"),
                "current_problem_hit": sign_base.hit(current_problem_text, "phase-1 non-heuristic two-component nonlinear closure audit"),
                "current_status_hit": sign_base.hit(current_status_text, "phase-1 non-heuristic two-component nonlinear closure audit"),
                "unified_roadmap_hit": sign_base.hit(unified_text, ".2419-.2422"),
                "long_roadmap_hit": sign_base.hit(long_text, ".2419-.2422"),
                "part5_hit": sign_base.hit(part5_text, "phase-1 non-heuristic two-component nonlinear closure audit"),
                "phase1_rho_hit": sign_base.hit(phase1_exact_slice, "rho = math.sqrt(max(f0 * f0 + f_l * f_l, 0.0))"),
                "phase1_nonlinear_coeff_hit": sign_base.hit(phase1_exact_slice, "nonlinear_coeff = 3.0 * rho + rho * rho"),
                "phase1_f0_line_hit": sign_base.hit(phase1_exact_slice, "f0_double_prime = -(2.0 / safe_r) * f0_prime - (float(beta * beta) - float(pivot.RADIAL_MASS_SQUARED)) * f0 - nonlinear_coeff * f0"),
                "phase1_fl_line_hit": sign_base.hit(phase1_exact_slice, "f_l_double_prime = -(2.0 / safe_r) * f_l_prime - (float(beta * beta) - float(pivot.LONGITUDINAL_DIRECT_MASS_SQUARED)) * f_l - nonlinear_coeff * f_l"),
                "solver_fix_nl_hit": sign_base.hit(solver_fix_text, "\\text{NL}(f_0)"),
                "solver_fix_coupling_hit": sign_base.hit(solver_fix_text, "\\text{coupling}(f_L)"),
                "next_steps_step_a_hit": sign_base.hit(next_steps_text, "### Step A."),
                "next_steps_f0_series_hit": sign_base.hit(next_steps_text, "f_0(r)=a_0+a_2 r^2 + a_4 r^4 + \\cdots"),
                "next_steps_fl_series_hit": sign_base.hit(next_steps_text, "f_L(r)=b_1 r + b_3 r^3 + b_5 r^5 + \\cdots"),
            },
        },
    )
    declaration_paths = write_artifact("declaration_gate", declaration_payload)

    route_payload = {
        "generated_utc": sign_base.now_iso(),
        "phase": {
            "phase": 8,
            "step": "8.7.56.2426",
            "name": STEP_NAME + " route sync",
        },
        "inputs": declaration_paths,
        "rows": rows,
        "summary": summary,
        "decision": {
            "overall_status": "vector_qball_form_factor_phase1_nonheuristic_two_component_nonlinear_closure_route_synced",
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

    print(f"[done] {STEP_TAG} phase-1 non-heuristic two-component nonlinear closure audit completed")


if __name__ == "__main__":
    main()
