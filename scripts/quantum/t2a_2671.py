#!/usr/bin/env python3
"""Generate 8.7.56.2671-.2674 updated-pack nonlinear-closure audit artifacts."""

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
        "8.7.56.2667-2670",
        "updated_pack_phase1_reciprocal_backreaction_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
PRIOR_AUDIT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.2663-2666",
        "updated_pack_phase1_reciprocal_backreaction_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
NONCOLLAPSED_AUDIT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.2655-2658",
        "updated_pack_noncollapsed_ell0_closure_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

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
NEXT_STEPS_CANDIDATES = (
    Path(r"C:\Users\ogawa\Downloads\trial2_vector_qball_next_steps_20260327.md"),
    ROOT
    / "output"
    / "private"
    / "quantum"
    / "expert_review_bundle_20260327_103144"
    / "trial2_vector_qball_next_steps_20260327.md",
)
NEXT_STEPS = next((path for path in NEXT_STEPS_CANDIDATES if path.exists()), NEXT_STEPS_CANDIDATES[0])

STEP_TAG = "8.7.56.2671-2674"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack phase-1 "
    "non-heuristic two-component nonlinear closure audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_phase1_nonheuristic_two_component_nonlinear_closure_audit",
    prefix="q",
)

PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_phase1_"
    "shared_rho_even_backreaction_only_nonlinear_closure_primary_trial3_ell0_"
    "reserve_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_phase1_"
    "nonheuristic_two_component_nonlinear_closure_not_literal_trial3_ell0_"
    "reserve_gate"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_phase1_nonlinear_"
    "closure_gate_trial3_ell0_reserve_refresh"
)
NEXT_ROUTE = "8.7.56.2675"
FOLLOWUP_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_trial3_ell0_"
    "closure_reserve_audit"
)
FOLLOWUP_ROUTE = "8.7.56.2679"


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
    """Return formulas used in the updated-pack nonlinear-closure audit."""
    return {
        "shared_rho_heuristic": "rho = sqrt(f_0^2 + f_L^2), nonlinear_coeff = 3 rho + rho^2",
        "current_exact_solver": "f_0'' + ... = - nonlinear_coeff f_0,  f_L'' + ... = - nonlinear_coeff f_L",
        "literal_target": "f_0'' + ... = - NL_0(f_0, f_L),  f_L'' + ... = - NL_L(f_0, f_L)",
        "closure_rule": "non-heuristic two-component closure requires component-specific nonlinear/backreaction structure, not one shared scalar coefficient",
    }


# 関数: `.2671-.2674` を実行する。

def main() -> None:
    """Execute the updated-pack phase-1 nonlinear-closure audit."""
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

    phase1_exact_slice = slice_between(
        phase1_text,
        "def solve_exact_profile(",
        "def run_exact_scan(",
    )

    updated_pack_phase1_nonlinear_closure_primary_selected = bool(
        prior_gate_summary["gate_b_updated_pack_nonheuristic_two_component_nonlinear_closure_promoted_next"]
        and prior_audit_summary["updated_pack_nonheuristic_two_component_nonlinear_closure_primary_followup_required"]
    )
    phase1_exact_solver_shared_single_nonlinear_coeff_only = bool(
        "rho = math.sqrt(max(f0 * f0 + f_l * f_l, 0.0))" in phase1_exact_slice
        and "nonlinear_coeff = 3.0 * rho + rho * rho" in phase1_exact_slice
        and "f0_double_prime = -(2.0 / safe_r) * f0_prime" in phase1_exact_slice
        and "f_l_double_prime = -(2.0 / safe_r) * f_l_prime" in phase1_exact_slice
    )
    phase1_exact_solver_component_specific_nonlinear_feedback_present = bool(
        "nonlinear_coeff_f0" in phase1_exact_slice
        or "nonlinear_coeff_fl" in phase1_exact_slice
        or "nl_f0" in phase1_exact_slice.lower()
        or "nl_fl" in phase1_exact_slice.lower()
        or "component_specific" in phase1_exact_slice.lower()
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
    exact_action_level_closed_ell0_operator_available = False
    updated_pack_phase1_nonheuristic_two_component_nonlinear_closure_supported_under_current_pack = bool(
        updated_pack_phase1_nonlinear_closure_primary_selected
        and phase1_exact_solver_shared_single_nonlinear_coeff_only
        and solver_fix_nonlinear_placeholder_available
        and next_steps_two_component_series_target_available
        and not exact_action_level_closed_ell0_operator_available
        and noncollapsed_summary["updated_pack_noncollapsed_ell0_closure_supported_under_current_pack"]
    )
    updated_pack_phase1_literal_two_component_nonlinear_formula_available = False
    updated_pack_phase1_nonheuristic_two_component_nonlinear_closure_closes_exact_coupled_operator_now = False
    updated_pack_trial3_family_ell0_closure_reserve_retained = bool(
        prior_gate_summary["gate_c_updated_pack_trial3_ell0_closure_reserve_retained"]
    )
    blind_vector_observable_gate_still_blocked = bool(
        prior_gate_summary["blind_vector_observable_gate_still_blocked"]
    )
    pack_update_required_now = False

    rows = [
        sign_base.row(
            "updated_pack_phase1_nonlinear_closure_primary_selected",
            "pass" if updated_pack_phase1_nonlinear_closure_primary_selected else "reject",
            "updated-pack phase-1 nonlinear closure primary selected",
            sign_base.truth(updated_pack_phase1_nonlinear_closure_primary_selected),
            "This audit starts only after `.2667-.2670` promoted the nonlinear-closure lane as the next exact completion move.",
        ),
        sign_base.row(
            "phase1_exact_solver_shared_single_nonlinear_coeff_only",
            "watch" if phase1_exact_solver_shared_single_nonlinear_coeff_only else "pass",
            "phase-1 exact solver uses one shared nonlinear coefficient only",
            sign_base.truth(phase1_exact_solver_shared_single_nonlinear_coeff_only),
            "The present solver still closes both component equations through the same scalarized `3 rho + rho^2` coefficient.",
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
            "Current updated-pack solver code still lacks the component-specific closure required for a literal exact operator.",
        ),
        sign_base.row(
            "solver_fix_nonlinear_placeholder_available",
            "pass" if solver_fix_nonlinear_placeholder_available else "reject",
            "solver-fix nonlinear placeholder available",
            sign_base.truth(solver_fix_nonlinear_placeholder_available),
            "The solver-fix note already marks the nonlinear two-component target surface through NL(f_0) and coupling(f_L) placeholders.",
        ),
        sign_base.row(
            "next_steps_two_component_series_target_available",
            "pass" if next_steps_two_component_series_target_available else "reject",
            "next-steps two-component series target available",
            sign_base.truth(next_steps_two_component_series_target_available),
            "The updated-pack next-steps note still fixes the same two-component near-origin series target for the literal closure followup.",
        ),
        sign_base.row(
            "exact_action_level_closed_ell0_operator_available",
            "pass" if exact_action_level_closed_ell0_operator_available else "reject",
            "exact action-level closed ell=0 operator available",
            sign_base.truth(exact_action_level_closed_ell0_operator_available),
            "The exact ell=0 operator remains open, so nonlinear closure cannot close the operator lane here.",
        ),
        sign_base.row(
            "updated_pack_phase1_nonheuristic_two_component_nonlinear_closure_supported_under_current_pack",
            "pass" if updated_pack_phase1_nonheuristic_two_component_nonlinear_closure_supported_under_current_pack else "reject",
            "updated-pack phase-1 non-heuristic two-component nonlinear closure supported under current pack",
            sign_base.truth(updated_pack_phase1_nonheuristic_two_component_nonlinear_closure_supported_under_current_pack),
            "The retained updated pack still supports this lane as the honest next closure target even though the literal formula is absent.",
        ),
        sign_base.row(
            "updated_pack_phase1_literal_two_component_nonlinear_formula_available",
            "pass" if updated_pack_phase1_literal_two_component_nonlinear_formula_available else "reject",
            "updated-pack phase-1 literal two-component nonlinear formula available",
            sign_base.truth(updated_pack_phase1_literal_two_component_nonlinear_formula_available),
            "No literal component-specific nonlinear closure formula is available yet in the exact solver.",
        ),
        sign_base.row(
            "updated_pack_phase1_nonheuristic_two_component_nonlinear_closure_closes_exact_coupled_operator_now",
            "pass" if updated_pack_phase1_nonheuristic_two_component_nonlinear_closure_closes_exact_coupled_operator_now else "reject",
            "updated-pack phase-1 non-heuristic two-component nonlinear closure closes exact coupled operator now",
            sign_base.truth(updated_pack_phase1_nonheuristic_two_component_nonlinear_closure_closes_exact_coupled_operator_now),
            "Because the component-specific nonlinear closure is still not literal, the exact coupled operator does not close here.",
        ),
        sign_base.row(
            "updated_pack_trial3_family_ell0_closure_reserve_retained",
            "pass" if updated_pack_trial3_family_ell0_closure_reserve_retained else "reject",
            "updated-pack trial-3 ell=0 closure reserve retained",
            sign_base.truth(updated_pack_trial3_family_ell0_closure_reserve_retained),
            "The old trial-3 family remains reserve-only while the phase-1 nonlinear-closure lane is still open.",
        ),
        sign_base.row(
            "blind_vector_observable_gate_still_blocked",
            "pass" if blind_vector_observable_gate_still_blocked else "reject",
            "blind-vector observable gate still blocked",
            sign_base.truth(blind_vector_observable_gate_still_blocked),
            "The blind-vector lane stays reserve-only until the exact operator lane changes the source picture.",
        ),
        sign_base.row(
            "pack_update_required_now",
            "pass" if pack_update_required_now else "reject",
            "substantive pack update required now",
            sign_base.truth(pack_update_required_now),
            "The route still advances inside the retained updated pack and does not yet require a new pack update.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "retained_scalar_residual_rel": float(prior_gate_summary["retained_scalar_residual_rel"]),
        "updated_pack_phase1_nonlinear_closure_primary_selected": updated_pack_phase1_nonlinear_closure_primary_selected,
        "phase1_exact_solver_shared_single_nonlinear_coeff_only": phase1_exact_solver_shared_single_nonlinear_coeff_only,
        "phase1_exact_solver_component_specific_nonlinear_feedback_present": phase1_exact_solver_component_specific_nonlinear_feedback_present,
        "phase1_exact_solver_nonheuristic_two_component_nonlinear_closure_present": phase1_exact_solver_nonheuristic_two_component_nonlinear_closure_present,
        "solver_fix_nonlinear_placeholder_available": solver_fix_nonlinear_placeholder_available,
        "next_steps_two_component_series_target_available": next_steps_two_component_series_target_available,
        "exact_action_level_closed_ell0_operator_available": exact_action_level_closed_ell0_operator_available,
        "updated_pack_phase1_nonheuristic_two_component_nonlinear_closure_supported_under_current_pack": updated_pack_phase1_nonheuristic_two_component_nonlinear_closure_supported_under_current_pack,
        "updated_pack_phase1_literal_two_component_nonlinear_formula_available": updated_pack_phase1_literal_two_component_nonlinear_formula_available,
        "updated_pack_phase1_nonheuristic_two_component_nonlinear_closure_closes_exact_coupled_operator_now": updated_pack_phase1_nonheuristic_two_component_nonlinear_closure_closes_exact_coupled_operator_now,
        "updated_pack_trial3_family_ell0_closure_reserve_retained": updated_pack_trial3_family_ell0_closure_reserve_retained,
        "blind_vector_observable_gate_still_blocked": blind_vector_observable_gate_still_blocked,
        "pack_update_required_now": pack_update_required_now,
        "selected_primary_completion_lane": "updated_pack_phase1_nonlinear_closure_gate",
        "selected_secondary_completion_lane": "updated_pack_trial3_ell0_closure_reserve_audit",
        "selected_reserve_completion_lane": "blind_vector_reserve",
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": FOLLOWUP_ROUTE_NAME,
        "selected_followup_route_or_none": FOLLOWUP_ROUTE,
        "physical_reject_required": False,
    }

    declaration_payload = sign_base.payload(
        "8.7.56.2673",
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
            "overall_status": "vector_qball_form_factor_updated_pack_phase1_nonlinear_closure_declared",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "formulas": build_formulae(),
            "hits": {
                "status_branch_hit": sign_base.hit(status_text, "8.7.56.2671"),
                "roadmap_branch_hit": sign_base.hit(roadmap_text, ".2671-.2674"),
                "current_problem_hit": sign_base.hit(current_problem_text, "updated-pack phase-1 non-heuristic two-component nonlinear closure audit"),
                "current_status_hit": sign_base.hit(current_status_text, "updated-pack phase-1 non-heuristic two-component nonlinear closure audit"),
                "unified_roadmap_hit": sign_base.hit(unified_text, ".2667-.2670"),
                "long_roadmap_hit": sign_base.hit(long_text, ".2667-.2670"),
                "part5_hit": sign_base.hit(part5_text, "updated-pack phase-1 non-heuristic two-component nonlinear closure audit"),
            },
        },
    )
    declaration_paths = write_artifact("declaration_gate", declaration_payload)

    route_payload = {
        "generated_utc": sign_base.now_iso(),
        "phase": {
            "phase": 8,
            "step": "8.7.56.2674",
            "name": STEP_NAME + " route sync",
        },
        "inputs": declaration_paths,
        "rows": rows,
        "summary": summary,
        "decision": {
            "overall_status": "vector_qball_form_factor_updated_pack_phase1_nonlinear_closure_route_synced",
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

    print("[ok] updated-pack phase-1 non-heuristic two-component nonlinear closure audit artifacts written")
    print(f"  declaration_gate: {declaration_paths['json']}")
    print(f"  route_sync: {route_paths['json']}")


if __name__ == "__main__":
    main()
