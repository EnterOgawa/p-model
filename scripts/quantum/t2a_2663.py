#!/usr/bin/env python3
"""Generate 8.7.56.2663-.2666 updated-pack phase-1 reciprocal backreaction artifacts."""

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
        "8.7.56.2659-2662",
        "updated_pack_noncollapsed_ell0_closure_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
PRIOR_AUDIT = build_metrics_paths(
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
PERTURBATIVE_NOTE_CANDIDATES = (
    Path(r"C:\Users\ogawa\Downloads\pmodel_v2_trial2_perturbative_fL_correction.md"),
    ROOT
    / "output"
    / "private"
    / "quantum"
    / "expert_review_bundle_20260327_103144"
    / "pmodel_v2_trial2_perturbative_fL_correction.md",
)
PERTURBATIVE_NOTE = next((path for path in PERTURBATIVE_NOTE_CANDIDATES if path.exists()), PERTURBATIVE_NOTE_CANDIDATES[0])

STEP_TAG = "8.7.56.2663-2666"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack phase-1 "
    "reciprocal backreaction audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_phase1_reciprocal_backreaction_audit",
    prefix="q",
)

PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_phase1_"
    "reciprocal_backreaction_primary_nonlinear_closure_reserve_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_phase1_"
    "shared_rho_even_backreaction_only_nonlinear_closure_promotion_gate"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_phase1_reciprocal_"
    "backreaction_gate_nonlinear_closure_refresh"
)
NEXT_ROUTE = "8.7.56.2667"
FOLLOWUP_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_phase1_nonheuristic_"
    "two_component_nonlinear_closure_audit"
)
FOLLOWUP_ROUTE = "8.7.56.2671"


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


# 関数: marker 間の text slice を返す。

def slice_between(text: str, start: str, end: str) -> str:
    """Return the text slice between two markers."""
    start_index = text.find(start)
    if start_index < 0:
        return ""

    end_index = text.find(end, start_index)
    if end_index < 0:
        return text[start_index:]

    return text[start_index:end_index]


# 関数: reciprocal-backreaction audit で使う式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used in the updated-pack reciprocal-backreaction audit."""
    return {
        "shared_rho_closure": "rho = sqrt(f_0^2 + f_L^2), nonlinear_force_on_f0 = (3 rho + rho^2) f_0",
        "placeholder_backreaction": "f_0'' + 2 f_0'/r + (beta^2 - 1) f_0 + NL(f_0) = - coupling(f_L)",
        "even_parity_obstruction": "rho(f_L) = rho(-f_L) => d[(3 rho + rho^2) f_0]/d f_L |_(f_L=0) = 0",
        "ordering_rule": "updated-pack shared-rho even backreaction only -> non-heuristic two-component nonlinear closure next",
    }


# 関数: `.2663-.2666` を実行する。

def main() -> None:
    """Execute the updated-pack phase-1 reciprocal backreaction audit."""
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
        PHASE1_SOLVER,
        SOLVER_FIX,
        PERTURBATIVE_NOTE,
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
    perturbative_text = sign_base.read_text(PERTURBATIVE_NOTE)

    prior_gate_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    prior_audit_summary = sign_base.read_json(PRIOR_AUDIT)["summary"]

    phase1_exact_slice = slice_between(
        phase1_text,
        "def solve_exact_profile(",
        "def run_exact_scan(",
    )

    updated_pack_phase1_reciprocal_backreaction_primary_selected = bool(
        prior_gate_summary["gate_b_updated_pack_phase1_reciprocal_backreaction_promoted_next"]
        and prior_audit_summary["updated_pack_phase1_reciprocal_backreaction_primary_followup_required"]
    )
    phase1_shared_rho_present = bool(
        "rho = math.sqrt(max(f0 * f0 + f_l * f_l, 0.0))" in phase1_exact_slice
        and "nonlinear_coeff = 3.0 * rho + rho * rho" in phase1_exact_slice
    )
    f0_line = str(sign_base.hit(phase1_exact_slice, "f0_double_prime ="))
    phase1_exact_solver_explicit_fl_source_in_f0_equation_present = "f_l" in f0_line
    phase1_shared_rho_even_backreaction_only = bool(
        updated_pack_phase1_reciprocal_backreaction_primary_selected
        and phase1_shared_rho_present
        and not phase1_exact_solver_explicit_fl_source_in_f0_equation_present
    )
    phase1_shared_rho_first_nonzero_fl_order = 2.0
    solver_fix_backreaction_placeholder_available = bool(
        sign_base.hit(solver_fix_text, "\\text{coupling}(f_L)") is not None
    )
    perturbative_identity_available = bool(
        sign_base.hit(perturbative_text, "F_{0r}^{(P)} = i\\omega f_L - f_0'") is not None
    )
    updated_pack_phase1_literal_reciprocal_backreaction_formula_available = False
    updated_pack_phase1_reciprocal_backreaction_supported_under_current_pack = bool(
        phase1_shared_rho_even_backreaction_only and solver_fix_backreaction_placeholder_available
    )
    updated_pack_phase1_reciprocal_backreaction_closes_exact_coupled_operator_now = False
    updated_pack_nonheuristic_two_component_nonlinear_closure_primary_followup_required = bool(
        updated_pack_phase1_reciprocal_backreaction_supported_under_current_pack
        and not updated_pack_phase1_literal_reciprocal_backreaction_formula_available
    )
    updated_pack_trial3_family_ell0_closure_reserve_retained = True
    blind_vector_observable_gate_still_blocked = True
    pack_update_required_now = False

    rows = [
        sign_base.row(
            "updated_pack_phase1_reciprocal_backreaction_primary_selected",
            "pass" if updated_pack_phase1_reciprocal_backreaction_primary_selected else "reject",
            "updated-pack phase-1 reciprocal backreaction primary selected",
            sign_base.truth(updated_pack_phase1_reciprocal_backreaction_primary_selected),
            "This audit starts only after the noncollapsed ell=0 gate promoted reciprocal f_L -> f_0 backreaction as the next exact completion move.",
        ),
        sign_base.row(
            "phase1_shared_rho_even_backreaction_only",
            "watch" if phase1_shared_rho_even_backreaction_only else "reject",
            "phase-1 solver carries shared-rho even backreaction only",
            sign_base.truth(phase1_shared_rho_even_backreaction_only),
            "Current f_0 dynamics depends on f_L only through rho = sqrt(f_0^2 + f_L^2), so the first nonzero f_L contribution is even and quadratic.",
        ),
        sign_base.row(
            "phase1_shared_rho_first_nonzero_fl_order",
            "watch",
            "phase-1 shared-rho first nonzero f_L order",
            phase1_shared_rho_first_nonzero_fl_order,
            "A literal reciprocal backreaction would need a linear f_L source in the f_0 equation, but the current ansatz starts at O(f_L^2).",
        ),
        sign_base.row(
            "solver_fix_backreaction_placeholder_available",
            "pass" if solver_fix_backreaction_placeholder_available else "reject",
            "solver-fix backreaction placeholder available",
            sign_base.truth(solver_fix_backreaction_placeholder_available),
            "The retained solver-fix note already names a coupling(f_L) placeholder, so the lane is physically motivated even though it is not literalized in code.",
        ),
        sign_base.row(
            "perturbative_identity_available",
            "pass" if perturbative_identity_available else "reject",
            "perturbative identity available",
            sign_base.truth(perturbative_identity_available),
            "The retained perturbative note still freezes the field-strength identity used to interpret the missing backreaction lane.",
        ),
        sign_base.row(
            "updated_pack_phase1_literal_reciprocal_backreaction_formula_available",
            "pass" if updated_pack_phase1_literal_reciprocal_backreaction_formula_available else "reject",
            "updated-pack phase-1 literal reciprocal backreaction formula available",
            sign_base.truth(updated_pack_phase1_literal_reciprocal_backreaction_formula_available),
            "No current public source provides the literal f_L -> f_0 source term beyond the shared-rho even closure.",
        ),
        sign_base.row(
            "updated_pack_phase1_reciprocal_backreaction_supported_under_current_pack",
            "pass" if updated_pack_phase1_reciprocal_backreaction_supported_under_current_pack else "reject",
            "updated-pack phase-1 reciprocal backreaction lane supported under current pack",
            sign_base.truth(updated_pack_phase1_reciprocal_backreaction_supported_under_current_pack),
            "The retained pack supports this lane as a missing-action theorem-completion target, even though it still lacks a literal formula.",
        ),
        sign_base.row(
            "updated_pack_phase1_reciprocal_backreaction_closes_exact_coupled_operator_now",
            "pass" if updated_pack_phase1_reciprocal_backreaction_closes_exact_coupled_operator_now else "reject",
            "updated-pack phase-1 reciprocal backreaction closes exact coupled operator now",
            sign_base.truth(updated_pack_phase1_reciprocal_backreaction_closes_exact_coupled_operator_now),
            "The present branch does not yet close the coupled operator because the backreaction remains heuristic and even in f_L.",
        ),
        sign_base.row(
            "updated_pack_nonheuristic_two_component_nonlinear_closure_primary_followup_required",
            "pass" if updated_pack_nonheuristic_two_component_nonlinear_closure_primary_followup_required else "reject",
            "updated-pack non-heuristic two-component nonlinear closure primary followup required",
            sign_base.truth(updated_pack_nonheuristic_two_component_nonlinear_closure_primary_followup_required),
            "Once reciprocal backreaction is found to be even-only under shared-rho, the honest next move is the nonlinear two-component closure audit.",
        ),
        sign_base.row(
            "updated_pack_trial3_family_ell0_closure_reserve_retained",
            "pass" if updated_pack_trial3_family_ell0_closure_reserve_retained else "reject",
            "updated-pack trial-3 ell=0 closure reserve retained",
            sign_base.truth(updated_pack_trial3_family_ell0_closure_reserve_retained),
            "The old trial-3 family remains reserve-only while the phase-1 nonlinear closure is still heuristic.",
        ),
        sign_base.row(
            "blind_vector_observable_gate_still_blocked",
            "pass" if blind_vector_observable_gate_still_blocked else "reject",
            "blind-vector observable gate still blocked",
            sign_base.truth(blind_vector_observable_gate_still_blocked),
            "The blind-vector lane remains reserve-only while the operator route is still open.",
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
        "updated_pack_phase1_reciprocal_backreaction_primary_selected": updated_pack_phase1_reciprocal_backreaction_primary_selected,
        "phase1_shared_rho_even_backreaction_only": phase1_shared_rho_even_backreaction_only,
        "phase1_exact_solver_explicit_fl_source_in_f0_equation_present": phase1_exact_solver_explicit_fl_source_in_f0_equation_present,
        "phase1_shared_rho_first_nonzero_fl_order": phase1_shared_rho_first_nonzero_fl_order,
        "solver_fix_backreaction_placeholder_available": solver_fix_backreaction_placeholder_available,
        "perturbative_identity_available": perturbative_identity_available,
        "updated_pack_phase1_literal_reciprocal_backreaction_formula_available": updated_pack_phase1_literal_reciprocal_backreaction_formula_available,
        "updated_pack_phase1_reciprocal_backreaction_supported_under_current_pack": updated_pack_phase1_reciprocal_backreaction_supported_under_current_pack,
        "updated_pack_phase1_reciprocal_backreaction_closes_exact_coupled_operator_now": updated_pack_phase1_reciprocal_backreaction_closes_exact_coupled_operator_now,
        "updated_pack_nonheuristic_two_component_nonlinear_closure_primary_followup_required": updated_pack_nonheuristic_two_component_nonlinear_closure_primary_followup_required,
        "updated_pack_trial3_family_ell0_closure_reserve_retained": updated_pack_trial3_family_ell0_closure_reserve_retained,
        "blind_vector_observable_gate_still_blocked": blind_vector_observable_gate_still_blocked,
        "pack_update_required_now": pack_update_required_now,
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": FOLLOWUP_ROUTE_NAME,
        "selected_followup_route_or_none": FOLLOWUP_ROUTE,
        "physical_reject_required": False,
    }

    declaration_payload = sign_base.payload(
        "8.7.56.2665",
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
                "phase1_solver": sign_base.display_path(PHASE1_SOLVER),
                "solver_fix": sign_base.display_path(SOLVER_FIX),
                "perturbative_note": sign_base.display_path(PERTURBATIVE_NOTE),
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
            "overall_status": "vector_qball_form_factor_updated_pack_phase1_reciprocal_backreaction_audited",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "formulas": build_formulae(),
            "hits": {
                "status_branch_hit": sign_base.hit(status_text, "8.7.56.2663"),
                "roadmap_branch_hit": sign_base.hit(roadmap_text, ".2663-.2666"),
                "current_problem_hit": sign_base.hit(current_problem_text, "updated-pack phase-1 reciprocal backreaction audit"),
                "current_status_hit": sign_base.hit(current_status_text, "updated-pack phase-1 reciprocal backreaction audit"),
                "unified_roadmap_hit": sign_base.hit(unified_text, ".2663-.2666"),
                "long_roadmap_hit": sign_base.hit(long_text, ".2663-.2666"),
                "part5_hit": sign_base.hit(part5_text, "updated-pack phase-1 reciprocal backreaction audit"),
                "phase1_rho_hit": sign_base.hit(phase1_exact_slice, "rho = math.sqrt(max(f0 * f0 + f_l * f_l, 0.0))"),
                "phase1_nonlin_hit": sign_base.hit(phase1_exact_slice, "nonlinear_coeff = 3.0 * rho + rho * rho"),
                "phase1_f0_hit": sign_base.hit(phase1_exact_slice, "f0_double_prime ="),
                "solver_fix_placeholder_hit": sign_base.hit(solver_fix_text, "\\text{coupling}(f_L)"),
                "perturbative_identity_hit": sign_base.hit(perturbative_text, "F_{0r}^{(P)} = i\\omega f_L - f_0'"),
            },
        },
    )
    declaration_paths = write_artifact("declaration_gate", declaration_payload)

    route_payload = {
        "generated_utc": sign_base.now_iso(),
        "phase": {
            "phase": 8,
            "step": "8.7.56.2666",
            "name": STEP_NAME + " route sync",
        },
        "inputs": declaration_paths,
        "rows": rows,
        "summary": summary,
        "decision": {
            "overall_status": "vector_qball_form_factor_updated_pack_phase1_reciprocal_backreaction_route_synced",
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

    print("[ok] updated-pack phase-1 reciprocal backreaction audit artifacts written")
    print(f"  declaration_gate: {declaration_paths['json']}")
    print(f"  route_sync: {route_paths['json']}")


if __name__ == "__main__":
    main()
