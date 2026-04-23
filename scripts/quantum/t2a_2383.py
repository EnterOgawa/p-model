#!/usr/bin/env python3
"""Generate 8.7.56.2383-.2386 phase-1 literal cross-term realization artifacts."""

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
        "8.7.56.2379-2382",
        "cross_term_completion_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
PRIOR_AUDIT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.2375-2378",
        "cross_term_completion_audit",
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
OLD_DIAGNOSTIC_DECL = (
    PUBLIC_OUT
    / "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_unified_closure_perturbative_fl_driven_ode_diagnostic_reopen_review_declaration_gate_metrics.json"
)
OLD_DIAGNOSTIC_EVAL = (
    PUBLIC_OUT
    / "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_unified_closure_perturbative_fl_driven_ode_diagnostic_reopen_review_numeric_evaluation_metrics.json"
)

PHASE1_SOLVER = ROOT / "scripts" / "quantum" / "t2a_1419.py"
SOLVER_FIX = Path(r"C:\Users\ogawa\Downloads\pmodel_v2_trial2_solver_fix_final.md")
PERTURBATIVE_NOTE = Path(r"C:\Users\ogawa\Downloads\pmodel_v2_trial2_perturbative_fL_correction.md")

STEP_TAG = "8.7.56.2383-2386"
STEP_NAME = "Trial-2 numeric alpha vector Q-ball form-factor phase-1 exact-solver literal cross-term realization audit"
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "phase1_literal_cross_term_realization_audit",
    prefix="q",
)

PRIOR_CLASS = "vector_qball_form_factor_residual_origin_missing_action_phase1_literal_cross_term_primary_constraint_elimination_secondary_ell0_closure_reserve_next"
BRANCH_CLASS = "vector_qball_form_factor_residual_origin_missing_action_phase1_literal_cross_term_triangulated_realization_constraint_followup_gate"
NEXT_ROUTE_NAME = "trial2_numeric_alpha_vector_qball_form_factor_phase1_literal_cross_term_gate_constraint_elimination_refresh"
NEXT_ROUTE = "8.7.56.2387"
FOLLOWUP_ROUTE_NAME = "trial2_numeric_alpha_vector_qball_form_factor_phase1_exact_solver_constraint_elimination_audit"
FOLLOWUP_ROUTE = "8.7.56.2391"


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

def build_formulae() -> dict[str, str]:
    """Return formulas used in the literal realization audit."""
    return {
        "kinetic_identity": "F_{0r}^{(P)} = i omega f_L - f_0'",
        "triangular_literal_realization": "f_0 solved first, then f_L'' + 2 f_L'/r - 2 f_L/r^2 - kappa^2 f_L = beta f_0'",
        "backreaction_followup": "f_0'' + 2 f_0'/r + (beta^2 - 1) f_0 + NL(f_0) = - coupling(f_L)",
        "ordering_rule": "phase-1 triangular literal realization -> constraint elimination -> noncollapsed ell=0 closure",
    }


# 関数: `.2383-.2386` を実行する。

def main() -> None:
    """Execute the phase-1 literal cross-term realization audit."""
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
        ELL0_OPERATOR_AUDIT,
        OLD_DIAGNOSTIC_DECL,
        OLD_DIAGNOSTIC_EVAL,
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
    ell0_summary = sign_base.read_json(ELL0_OPERATOR_AUDIT)["summary"]
    old_decl_summary = sign_base.read_json(OLD_DIAGNOSTIC_DECL)["summary"]
    old_eval_summary = sign_base.read_json(OLD_DIAGNOSTIC_EVAL)["summary"]

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
    solver_fix_identity_hit = sign_base.hit(
        solver_fix_text,
        "F_{0r}^{(P)} = i\\omega f_L - f_0'",
    )
    solver_fix_driven_hit = sign_base.hit(
        solver_fix_text,
        "source = beta * np.interp(r, x, dy0)",
    )
    solver_fix_backreaction_hit = sign_base.hit(
        solver_fix_text,
        "\\text{coupling}(f_L)",
    )
    perturb_identity_hit = sign_base.hit(
        perturbative_text,
        "F_{0r}^{(P)} = \\partial_0 P_r - \\partial_r P_0 = i\\omega f_L - f_0'",
    )
    perturb_green_hit = sign_base.hit(
        perturbative_text,
        "f_L(r) = \\int G_L(r, r')\\,S(f_0; r')\\,dr'",
    )

    phase1_literal_cross_term_realization_formula_available = bool(
        diagnostic_source_hit
        and solver_fix_identity_hit
        and solver_fix_driven_hit
        and perturb_identity_hit
    )
    phase1_literal_cross_term_realization_is_triangular = bool(
        phase1_literal_cross_term_realization_formula_available
        and not hit_has_token(exact_f0_hit, "f_l")
        and not hit_has_token(exact_fl_hit, "f0")
        and bool(solver_fix_backreaction_hit)
    )
    perturbative_case_gamma_blocks_direct_numeric_reuse = bool(
        old_decl_summary["case_gamma_selected"]
        and old_eval_summary["perturbative_breakdown_detected"]
    )
    phase1_literal_cross_term_realization_supported_under_current_pack = bool(
        prior_gate_summary["gate_a_phase1_literal_cross_term_selected"]
        and prior_audit_summary["phase1_diagnostic_cross_term_template_present"]
        and phase1_literal_cross_term_realization_formula_available
        and phase1_literal_cross_term_realization_is_triangular
    )
    phase1_backreaction_followup_still_open = bool(
        phase1_literal_cross_term_realization_supported_under_current_pack
        and bool(solver_fix_backreaction_hit)
        and not hit_has_token(exact_f0_hit, "f_l")
    )
    phase1_literal_cross_term_realization_closes_exact_coupled_operator_now = False
    constraint_elimination_followup_required = bool(
        prior_gate_summary["gate_b_constraint_elimination_followup_retained"]
        and phase1_literal_cross_term_realization_supported_under_current_pack
        and phase1_backreaction_followup_still_open
    )
    trial3_family_primary_reuse_admissible_now = False
    pack_update_required_now = False

    rows = [
        sign_base.row(
            "inventory_ready",
            "pass",
            "phase-1 literal cross-term realization inventory ready",
            1.0,
            "This branch starts only after the literal gap has been localized to the phase-1 exact pilot and constraint elimination has been kept as the official followup.",
        ),
        sign_base.row(
            "phase1_literal_cross_term_realization_formula_available",
            "pass" if phase1_literal_cross_term_realization_formula_available else "reject",
            "phase-1 literal cross-term realization formula available",
            sign_base.truth(phase1_literal_cross_term_realization_formula_available),
            "The current pack already contains the field-strength identity and the driven f_L source template needed to write the missing phase-1 mixing literally.",
        ),
        sign_base.row(
            "phase1_literal_cross_term_realization_is_triangular",
            "pass" if phase1_literal_cross_term_realization_is_triangular else "reject",
            "phase-1 literal cross-term realization is triangular",
            sign_base.truth(phase1_literal_cross_term_realization_is_triangular),
            "The first honest realization keeps scalar-first exact f_0 and restores the driven f_L equation before backreaction and constraint elimination are closed.",
        ),
        sign_base.row(
            "perturbative_case_gamma_blocks_direct_numeric_reuse",
            "pass" if perturbative_case_gamma_blocks_direct_numeric_reuse else "reject",
            "legacy perturbative case-gamma outcome blocks direct numeric reuse",
            sign_base.truth(perturbative_case_gamma_blocks_direct_numeric_reuse),
            "The old perturbative numeric branch is retained only as structural evidence for the source term; its own case-gamma breakdown means it cannot be promoted as the final exact pilot.",
        ),
        sign_base.row(
            "phase1_literal_cross_term_realization_supported_under_current_pack",
            "pass" if phase1_literal_cross_term_realization_supported_under_current_pack else "reject",
            "phase-1 literal cross-term realization supported under current pack",
            sign_base.truth(phase1_literal_cross_term_realization_supported_under_current_pack),
            "No new pack ingredient is required to promote the missing source term from diagnostic-only status to the official phase-1 pilot first shot.",
        ),
        sign_base.row(
            "phase1_backreaction_followup_still_open",
            "pass" if phase1_backreaction_followup_still_open else "reject",
            "phase-1 backreaction followup still open",
            sign_base.truth(phase1_backreaction_followup_still_open),
            "Restoring the driven f_L equation does not yet restore the reciprocal f_L -> f_0 backreaction, so the operator is still not closed at this branch.",
        ),
        sign_base.row(
            "phase1_literal_cross_term_realization_closes_exact_coupled_operator_now",
            "pass" if phase1_literal_cross_term_realization_closes_exact_coupled_operator_now else "reject",
            "phase-1 literal cross-term realization closes the exact coupled operator now",
            sign_base.truth(phase1_literal_cross_term_realization_closes_exact_coupled_operator_now),
            "Literal realization alone is not enough because constraint elimination and noncollapsed ell=0 closure still remain downstream.",
        ),
        sign_base.row(
            "constraint_elimination_followup_required",
            "pass" if constraint_elimination_followup_required else "reject",
            "constraint-elimination followup required after literal realization",
            sign_base.truth(constraint_elimination_followup_required),
            "Once the literal source term is restored, the next scientific blocker becomes constraint elimination rather than another same-level search.",
        ),
        sign_base.row(
            "trial3_family_primary_reuse_admissible_now",
            "pass" if trial3_family_primary_reuse_admissible_now else "reject",
            "trial-3 family primary reuse admissible now",
            sign_base.truth(trial3_family_primary_reuse_admissible_now),
            "The old family map still collapses at ell=0 and remains reserve-only while the phase-1 literal realization is active.",
        ),
        sign_base.row(
            "pack_update_required_now",
            "pass" if pack_update_required_now else "reject",
            "substantive pack update required now",
            sign_base.truth(pack_update_required_now),
            "The current task is still an internal realization under the retained pack, not a new external action surface.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "retained_scalar_residual_rel": float(prior_gate_summary["retained_scalar_residual_rel"]),
        "delta_beta2_exact_profile_fixed": float(prior_audit_summary["delta_beta2_exact_profile_fixed"]),
        "operator_coefficient_proxy_from_max_ratio_sq": float(
            prior_audit_summary["operator_coefficient_proxy_from_max_ratio_sq"]
        ),
        "phase1_literal_cross_term_realization_formula_available": phase1_literal_cross_term_realization_formula_available,
        "phase1_literal_cross_term_realization_is_triangular": phase1_literal_cross_term_realization_is_triangular,
        "perturbative_case_gamma_blocks_direct_numeric_reuse": perturbative_case_gamma_blocks_direct_numeric_reuse,
        "phase1_literal_cross_term_realization_supported_under_current_pack": phase1_literal_cross_term_realization_supported_under_current_pack,
        "phase1_backreaction_followup_still_open": phase1_backreaction_followup_still_open,
        "phase1_literal_cross_term_realization_closes_exact_coupled_operator_now": phase1_literal_cross_term_realization_closes_exact_coupled_operator_now,
        "constraint_elimination_followup_required": constraint_elimination_followup_required,
        "trial3_family_primary_reuse_admissible_now": trial3_family_primary_reuse_admissible_now,
        "pack_update_required_now": pack_update_required_now,
        "legacy_diagnostic_max_abs_ratio": float(old_eval_summary["diagnostic_max_abs_ratio"]),
        "legacy_diagnostic_alpha_at_q_theory": float(old_eval_summary["diagnostic_alpha_at_q_theory"]),
        "selected_primary_realization_form": "phase1_scalar_first_then_driven_fL_triangular_realization",
        "selected_secondary_completion_lane": "constraint_elimination",
        "selected_reserve_completion_lane": "noncollapsed_ell0_closure",
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": FOLLOWUP_ROUTE_NAME,
        "selected_followup_route_or_none": FOLLOWUP_ROUTE,
        "physical_reject_required": False,
    }

    declaration_payload = sign_base.payload(
        "8.7.56.2385",
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
                "ell0_operator_audit": sign_base.display_path(ELL0_OPERATOR_AUDIT),
                "old_diagnostic_decl": sign_base.display_path(OLD_DIAGNOSTIC_DECL),
                "old_diagnostic_eval": sign_base.display_path(OLD_DIAGNOSTIC_EVAL),
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
            "overall_status": "vector_qball_form_factor_phase1_literal_cross_term_realization_audited",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "formulas": build_formulae(),
            "hits": {
                "status_branch_hit": sign_base.hit(status_text, "8.7.56.2383"),
                "roadmap_branch_hit": sign_base.hit(roadmap_text, ".2383-.2386"),
                "current_problem_hit": sign_base.hit(current_problem_text, "phase-1 exact-solver literal cross-term realization audit"),
                "current_status_hit": sign_base.hit(current_status_text, "phase-1 exact-solver literal cross-term realization audit"),
                "unified_roadmap_hit": sign_base.hit(unified_text, ".2383-.2386"),
                "long_roadmap_hit": sign_base.hit(long_text, ".2383-.2386"),
                "part5_hit": sign_base.hit(part5_text, "phase-1 exact-solver literal cross-term realization audit"),
                "phase1_diagnostic_source_hit": diagnostic_source_hit,
                "phase1_exact_f0_hit": exact_f0_hit,
                "phase1_exact_fl_hit": exact_fl_hit,
                "solver_fix_identity_hit": solver_fix_identity_hit,
                "solver_fix_driven_hit": solver_fix_driven_hit,
                "solver_fix_backreaction_hit": solver_fix_backreaction_hit,
                "perturb_identity_hit": perturb_identity_hit,
                "perturb_green_hit": perturb_green_hit,
                "legacy_case_gamma_summary": {
                    "case_gamma_selected": old_decl_summary["case_gamma_selected"],
                    "perturbative_breakdown_detected": old_eval_summary["perturbative_breakdown_detected"],
                    "diagnostic_max_abs_ratio": old_eval_summary["diagnostic_max_abs_ratio"],
                    "diagnostic_alpha_at_q_theory": old_eval_summary["diagnostic_alpha_at_q_theory"],
                },
                "ell0_operator_summary": ell0_summary,
            },
        },
    )
    declaration_paths = write_artifact("declaration_gate", declaration_payload)

    route_payload = {
        "generated_utc": sign_base.now_iso(),
        "phase": {
            "phase": 8,
            "step": "8.7.56.2386",
            "name": STEP_NAME + " route sync",
        },
        "inputs": declaration_paths,
        "rows": rows,
        "summary": summary,
        "decision": {
            "overall_status": "vector_qball_form_factor_phase1_literal_cross_term_realization_route_synced",
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

    print(f"[done] {STEP_TAG} phase-1 literal cross-term realization audit completed")
    print(f"[info] declaration_gate_json={declaration_paths['json']}")
    print(f"[info] declaration_gate_csv={declaration_paths['csv']}")


if __name__ == "__main__":
    main()
