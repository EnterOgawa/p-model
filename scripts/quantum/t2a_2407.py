#!/usr/bin/env python3
"""Generate 8.7.56.2407-.2410 noncollapsed ell=0 closure artifacts."""

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
        "8.7.56.2403-2406",
        "phase1_constraint_elimination_realization_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
PRIOR_AUDIT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.2399-2402",
        "phase1_literal_constraint_elimination_realization_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
CROSS_TERM_AUDIT = build_metrics_paths(
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
SOLVER_FIX = Path(r"C:\Users\ogawa\Downloads\pmodel_v2_trial2_solver_fix_final.md")
NEXT_STEPS = Path(r"C:\Users\ogawa\Downloads\trial2_vector_qball_next_steps_20260327.md")

STEP_TAG = "8.7.56.2407-2410"
STEP_NAME = "Trial-2 numeric alpha vector Q-ball form-factor noncollapsed ell=0 closure audit"
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "noncollapsed_ell0_closure_audit",
    prefix="q",
)

PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_phase1_literal_constraint_"
    "elimination_realization_selected_ell0_closure_reserve_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_noncollapsed_ell0_closure_not_literal_"
    "phase1_backreaction_primary_nonlinear_closure_reserve_gate"
)
NEXT_ROUTE_NAME = "trial2_numeric_alpha_vector_qball_form_factor_noncollapsed_ell0_closure_gate_missing_action_refresh"
NEXT_ROUTE = "8.7.56.2411"
FOLLOWUP_ROUTE_NAME = "trial2_numeric_alpha_vector_qball_form_factor_phase1_reciprocal_backreaction_audit"
FOLLOWUP_ROUTE = "8.7.56.2415"


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


# 関数: noncollapsed ell=0 closure audit で使う式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used in the noncollapsed ell=0 closure audit."""
    return {
        "heuristic_phase1_closure": "rho = sqrt(f_0^2 + f_L^2), nonlinear_coeff = 3 rho + rho^2",
        "solver_fix_backreaction_placeholder": "f_0'' + 2 f_0'/r + (beta^2 - 1) f_0 + NL(f_0) = - coupling(f_L)",
        "closure_rule": "noncollapsed ell=0 closure requires reciprocal backreaction plus a non-heuristic two-component nonlinear closure",
        "ordering_rule": "literal reduced-state realization -> noncollapsed ell=0 closure audit -> reciprocal backreaction first shot if closure is still not literal",
    }


# 関数: `.2407-.2410` を実行する。

def main() -> None:
    """Execute the noncollapsed ell=0 closure audit."""
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
        CROSS_TERM_AUDIT,
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
    cross_term_summary = sign_base.read_json(CROSS_TERM_AUDIT)["summary"]
    ell0_summary = sign_base.read_json(ELL0_OPERATOR_AUDIT)["summary"]

    phase1_exact_slice = slice_between(
        phase1_text,
        "def solve_exact_profile(",
        "def run_exact_scan(",
    )

    phase1_literal_constraint_elimination_realization_selected = bool(
        prior_gate_summary["gate_a_phase1_literal_constraint_elimination_realization_selected"]
        and prior_audit_summary["phase1_literal_constraint_elimination_realization_supported_under_current_pack"]
    )
    phase1_backreaction_followup_still_open = bool(
        cross_term_summary["phase1_backreaction_followup_still_open"]
    )
    phase1_exact_solver_scalar_nonlinear_ansatz_only = bool(
        ell0_summary["phase1_exact_solver_scalar_nonlinear_ansatz_only"]
        and "rho = math.sqrt(max(f0 * f0 + f_l * f_l, 0.0))" in phase1_exact_slice
        and "nonlinear_coeff = 3.0 * rho + rho * rho" in phase1_exact_slice
    )
    phase1_exact_solver_shared_rho_nonlinear_closure_only = bool(
        phase1_exact_solver_scalar_nonlinear_ansatz_only
        and "f0_double_prime = -(2.0 / safe_r) * f0_prime" in phase1_exact_slice
        and "f_l_double_prime = -(2.0 / safe_r) * f_l_prime" in phase1_exact_slice
    )
    phase1_exact_solver_nonheuristic_two_component_nonlinear_closure_present = bool(
        "coupling(" in phase1_exact_slice
        or "NL(" in phase1_exact_slice
        or "noncollapsed" in phase1_exact_slice.lower()
    )
    solver_fix_backreaction_placeholder_available = bool(
        sign_base.hit(solver_fix_text, "\\text{coupling}(f_L)") is not None
        and sign_base.hit(solver_fix_text, "\\text{NL}(f_0)") is not None
    )
    next_steps_noncollapsed_ell0_target_available = bool(
        sign_base.hit(next_steps_text, "### Step A.") is not None
        and sign_base.hit(next_steps_text, "f_L(r)=b_1 r + b_3 r^3 + b_5 r^5 + \\cdots") is not None
    )
    trial3_family_solver_ell0_coupling_collapses = bool(
        ell0_summary["trial3_family_solver_ell0_coupling_collapses"]
    )
    noncollapsed_ell0_closure_supported_under_current_pack = bool(
        phase1_literal_constraint_elimination_realization_selected
        and phase1_backreaction_followup_still_open
        and phase1_exact_solver_shared_rho_nonlinear_closure_only
        and solver_fix_backreaction_placeholder_available
        and next_steps_noncollapsed_ell0_target_available
        and trial3_family_solver_ell0_coupling_collapses
    )
    noncollapsed_ell0_closure_literal_available_now = bool(
        phase1_exact_solver_nonheuristic_two_component_nonlinear_closure_present
    )
    noncollapsed_ell0_closure_closes_exact_coupled_operator_now = False
    phase1_reciprocal_backreaction_primary_followup_required = bool(
        noncollapsed_ell0_closure_supported_under_current_pack
        and not noncollapsed_ell0_closure_literal_available_now
    )
    nonheuristic_two_component_nonlinear_closure_reserve_required = bool(
        phase1_reciprocal_backreaction_primary_followup_required
    )
    trial3_family_primary_reuse_admissible_now = False
    pack_update_required_now = False

    rows = [
        sign_base.row(
            "phase1_literal_constraint_elimination_realization_selected",
            "pass" if phase1_literal_constraint_elimination_realization_selected else "reject",
            "phase-1 literal constraint-elimination realization selected",
            sign_base.truth(phase1_literal_constraint_elimination_realization_selected),
            "This audit starts only after the reduced-state target has been selected as the current exact-solver completion move.",
        ),
        sign_base.row(
            "phase1_backreaction_followup_still_open",
            "pass" if phase1_backreaction_followup_still_open else "reject",
            "phase-1 backreaction followup still open",
            sign_base.truth(phase1_backreaction_followup_still_open),
            "The triangular cross-term realization already fixed that reciprocal f_L -> f_0 backreaction remains downstream.",
        ),
        sign_base.row(
            "phase1_exact_solver_scalar_nonlinear_ansatz_only",
            "watch" if phase1_exact_solver_scalar_nonlinear_ansatz_only else "pass",
            "phase-1 exact solver uses scalar-style nonlinear ansatz only",
            sign_base.truth(phase1_exact_solver_scalar_nonlinear_ansatz_only),
            "The retained phase-1 pilot still closes both components through the shared rho = sqrt(f_0^2 + f_L^2) ansatz rather than an action-derived noncollapsed ell=0 closure.",
        ),
        sign_base.row(
            "phase1_exact_solver_shared_rho_nonlinear_closure_only",
            "watch" if phase1_exact_solver_shared_rho_nonlinear_closure_only else "pass",
            "phase-1 exact solver shared-rho nonlinear closure only",
            sign_base.truth(phase1_exact_solver_shared_rho_nonlinear_closure_only),
            "The present pilot carries f_L back into f_0 only through the shared rho heuristic, not through a literal reciprocal backreaction term.",
        ),
        sign_base.row(
            "phase1_exact_solver_nonheuristic_two_component_nonlinear_closure_present",
            "pass" if phase1_exact_solver_nonheuristic_two_component_nonlinear_closure_present else "reject",
            "phase-1 exact solver non-heuristic two-component nonlinear closure present",
            sign_base.truth(phase1_exact_solver_nonheuristic_two_component_nonlinear_closure_present),
            "A literal noncollapsed ell=0 closure would require an explicit two-component nonlinear/backreaction term in the exact solver.",
        ),
        sign_base.row(
            "solver_fix_backreaction_placeholder_available",
            "pass" if solver_fix_backreaction_placeholder_available else "reject",
            "solver-fix backreaction placeholder available",
            sign_base.truth(solver_fix_backreaction_placeholder_available),
            "The retained fix note already says the missing branch should feed back into the f_0 equation, but only as a placeholder coupling(f_L) statement.",
        ),
        sign_base.row(
            "next_steps_noncollapsed_ell0_target_available",
            "pass" if next_steps_noncollapsed_ell0_target_available else "reject",
            "next-steps noncollapsed ell=0 target available",
            sign_base.truth(next_steps_noncollapsed_ell0_target_available),
            "The retained theorem note still identifies the noncollapsed ell=0 near-origin branch as the next exact target.",
        ),
        sign_base.row(
            "trial3_family_solver_ell0_coupling_collapses",
            "watch" if trial3_family_solver_ell0_coupling_collapses else "pass",
            "trial-3 family solver ell=0 coupling collapses",
            sign_base.truth(trial3_family_solver_ell0_coupling_collapses),
            "The old trial-3 family still cannot stand in for the noncollapsed ell=0 operator because its ell-dependent coupling vanishes at ell=0.",
        ),
        sign_base.row(
            "noncollapsed_ell0_closure_supported_under_current_pack",
            "pass" if noncollapsed_ell0_closure_supported_under_current_pack else "reject",
            "noncollapsed ell=0 closure supported under current pack",
            sign_base.truth(noncollapsed_ell0_closure_supported_under_current_pack),
            "The retained pack still supports continuation of the missing-action lane without reopening pack-update or farther-q evidence routes.",
        ),
        sign_base.row(
            "noncollapsed_ell0_closure_literal_available_now",
            "pass" if noncollapsed_ell0_closure_literal_available_now else "reject",
            "noncollapsed ell=0 closure literal available now",
            sign_base.truth(noncollapsed_ell0_closure_literal_available_now),
            "Current code and notes still do not literalize a noncollapsed two-component ell=0 closure inside the phase-1 exact solver.",
        ),
        sign_base.row(
            "phase1_reciprocal_backreaction_primary_followup_required",
            "pass" if phase1_reciprocal_backreaction_primary_followup_required else "reject",
            "phase-1 reciprocal backreaction primary followup required",
            sign_base.truth(phase1_reciprocal_backreaction_primary_followup_required),
            "Because closure is not yet literal, the next honest move is to realize the reciprocal f_L -> f_0 backreaction directly inside the exact solver.",
        ),
        sign_base.row(
            "nonheuristic_two_component_nonlinear_closure_reserve_required",
            "pass" if nonheuristic_two_component_nonlinear_closure_reserve_required else "reject",
            "non-heuristic two-component nonlinear closure reserve required",
            sign_base.truth(nonheuristic_two_component_nonlinear_closure_reserve_required),
            "After reciprocal backreaction is localized as primary, a fully non-heuristic two-component nonlinear closure remains the next reserve theorem target.",
        ),
        sign_base.row(
            "pack_update_required_now",
            "pass" if pack_update_required_now else "reject",
            "substantive pack update required now",
            sign_base.truth(pack_update_required_now),
            "The closure gap remains an implementation/theorem-completion issue inside the retained pack, not an external-input problem.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "retained_scalar_residual_rel": float(prior_gate_summary["retained_scalar_residual_rel"]),
        "phase1_literal_constraint_elimination_realization_selected": phase1_literal_constraint_elimination_realization_selected,
        "phase1_backreaction_followup_still_open": phase1_backreaction_followup_still_open,
        "phase1_exact_solver_scalar_nonlinear_ansatz_only": phase1_exact_solver_scalar_nonlinear_ansatz_only,
        "phase1_exact_solver_shared_rho_nonlinear_closure_only": phase1_exact_solver_shared_rho_nonlinear_closure_only,
        "phase1_exact_solver_nonheuristic_two_component_nonlinear_closure_present": phase1_exact_solver_nonheuristic_two_component_nonlinear_closure_present,
        "solver_fix_backreaction_placeholder_available": solver_fix_backreaction_placeholder_available,
        "next_steps_noncollapsed_ell0_target_available": next_steps_noncollapsed_ell0_target_available,
        "trial3_family_solver_ell0_coupling_collapses": trial3_family_solver_ell0_coupling_collapses,
        "noncollapsed_ell0_closure_supported_under_current_pack": noncollapsed_ell0_closure_supported_under_current_pack,
        "noncollapsed_ell0_closure_literal_available_now": noncollapsed_ell0_closure_literal_available_now,
        "noncollapsed_ell0_closure_closes_exact_coupled_operator_now": noncollapsed_ell0_closure_closes_exact_coupled_operator_now,
        "phase1_reciprocal_backreaction_primary_followup_required": phase1_reciprocal_backreaction_primary_followup_required,
        "nonheuristic_two_component_nonlinear_closure_reserve_required": nonheuristic_two_component_nonlinear_closure_reserve_required,
        "trial3_family_primary_reuse_admissible_now": trial3_family_primary_reuse_admissible_now,
        "pack_update_required_now": pack_update_required_now,
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": FOLLOWUP_ROUTE_NAME,
        "selected_followup_route_or_none": FOLLOWUP_ROUTE,
        "physical_reject_required": False,
    }

    declaration_payload = sign_base.payload(
        "8.7.56.2409",
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
                "cross_term_audit": sign_base.display_path(CROSS_TERM_AUDIT),
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
            "overall_status": "vector_qball_form_factor_noncollapsed_ell0_closure_audit_declared",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "formulas": build_formulae(),
            "hits": {
                "status_branch_hit": sign_base.hit(status_text, "8.7.56.2407"),
                "roadmap_branch_hit": sign_base.hit(roadmap_text, ".2407-.2410"),
                "current_problem_hit": sign_base.hit(current_problem_text, "noncollapsed ell=0 closure audit"),
                "current_status_hit": sign_base.hit(current_status_text, "noncollapsed ell=0 closure audit"),
                "unified_roadmap_hit": sign_base.hit(unified_text, ".2407-.2410"),
                "long_roadmap_hit": sign_base.hit(long_text, ".2407-.2410"),
                "part5_hit": sign_base.hit(part5_text, "noncollapsed ell=0 closure audit"),
                "phase1_rho_hit": sign_base.hit(phase1_exact_slice, "rho = math.sqrt(max(f0 * f0 + f_l * f_l, 0.0))"),
                "phase1_nonlinear_coeff_hit": sign_base.hit(phase1_exact_slice, "nonlinear_coeff = 3.0 * rho + rho * rho"),
                "solver_fix_backreaction_hit": sign_base.hit(solver_fix_text, "\\text{coupling}(f_L)"),
                "next_steps_step_a_hit": sign_base.hit(next_steps_text, "### Step A."),
                "next_steps_odd_branch_hit": sign_base.hit(next_steps_text, "f_L(r)=b_1 r + b_3 r^3 + b_5 r^5 + \\cdots"),
            },
        },
    )
    declaration_paths = write_artifact("declaration_gate", declaration_payload)

    route_payload = {
        "generated_utc": sign_base.now_iso(),
        "phase": {
            "phase": 8,
            "step": "8.7.56.2410",
            "name": STEP_NAME + " route sync",
        },
        "inputs": declaration_paths,
        "rows": rows,
        "summary": summary,
        "decision": {
            "overall_status": "vector_qball_form_factor_noncollapsed_ell0_closure_route_synced",
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

    print(f"[done] {STEP_TAG} noncollapsed ell=0 closure audit completed")


if __name__ == "__main__":
    main()
