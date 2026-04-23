#!/usr/bin/env python3
"""Generate 8.7.56.2439-.2442 substantive pack-update audit artifacts."""

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
THEORY_LESSONS = ROOT / "doc" / "quantum" / "56_trial2_numeric_alpha_vector_qball_theory_lessons_after_interval_extension.md"
PART5 = ROOT / "doc" / "paper" / "14_part5_future_predictions.md"

PRIOR_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.2435-2438",
        "trial3_ell0_reserve_gate",
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
TRIAL3_ELL0_AUDIT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.2431-2434",
        "trial3_ell0_closure_reserve_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

SOLVER_FIX = Path(r"C:\Users\ogawa\Downloads\pmodel_v2_trial2_solver_fix_final.md")
NEXT_STEPS = Path(r"C:\Users\ogawa\Downloads\trial2_vector_qball_next_steps_20260327.md")

STEP_TAG = "8.7.56.2439-2442"
STEP_NAME = "Trial-2 numeric alpha vector Q-ball form-factor substantive pack update audit"
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "substantive_pack_update_audit",
    prefix="q",
)

PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_trial3_ell0_reserve_exhausted_"
    "pack_update_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_substantive_pack_exact_ell0_"
    "series_operator_primary_effective_source_followup_gate"
)
NEXT_ROUTE_NAME = "trial2_numeric_alpha_vector_qball_form_factor_substantive_pack_update_gate_hybrid_reserve_refresh"
NEXT_ROUTE = "8.7.56.2443"
FOLLOWUP_ROUTE_NAME = "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_exact_ell0_series_operator_audit"
FOLLOWUP_ROUTE = "8.7.56.2447"


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


# 関数: substantive pack-update audit で使う式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used in the substantive pack-update audit."""
    return {
        "current_pack_heuristic": "rho = sqrt(f_0^2 + f_L^2), nonlinear_coeff = 3 rho + rho^2",
        "solver_fix_placeholder": "f_0'' + 2 f_0'/r + (beta^2 - 1) f_0 + NL(f_0) = - coupling(f_L)",
        "primary_pack_surface": "f_0(r)=a_0+a_2 r^2 + a_4 r^4 + ...,  f_L(r)=b_1 r + b_3 r^3 + b_5 r^5 + ... together with L_L[f_L] = S[f_0]",
        "secondary_pack_surface": "L \\supset a_mu J_eff^mu[P^Qball]",
        "pack_update_rule": "substantive pack update = a new public-canonical surface that changes the exact ell=0 operator/current theorem, not a retry of density/proxy/eigenvalue or farther-q evidence lanes",
    }


# 関数: `.2439-.2442` を実行する。

def main() -> None:
    """Execute the substantive pack-update audit."""
    for path in (
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        WORK_HISTORY_RECENT,
        CURRENT_PROBLEM,
        CURRENT_STATUS,
        UNIFIED_ROADMAP,
        LONG_ROADMAP,
        THEORY_LESSONS,
        PART5,
        PRIOR_GATE,
        RECIPROCAL_AUDIT,
        NONLINEAR_AUDIT,
        TRIAL3_ELL0_AUDIT,
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
    lessons_text = sign_base.read_text(THEORY_LESSONS)
    part5_text = sign_base.read_text(PART5)
    solver_fix_text = sign_base.read_text(SOLVER_FIX)
    next_steps_text = sign_base.read_text(NEXT_STEPS)

    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    reciprocal_summary = sign_base.read_json(RECIPROCAL_AUDIT)["summary"]
    nonlinear_summary = sign_base.read_json(NONLINEAR_AUDIT)["summary"]
    trial3_summary = sign_base.read_json(TRIAL3_ELL0_AUDIT)["summary"]

    current_pack_exhausted_honestly = bool(
        prior_summary["pack_update_required_now"]
        and not trial3_summary["trial3_ell0_reserve_closes_current_missing_action_blocker_now"]
        and not reciprocal_summary["phase1_reciprocal_backreaction_closes_exact_coupled_operator_now"]
        and not nonlinear_summary["phase1_nonheuristic_two_component_nonlinear_closure_closes_exact_coupled_operator_now"]
    )
    current_pack_missing_action_formulae_remain_placeholder_only = bool(
        not reciprocal_summary["phase1_literal_reciprocal_backreaction_formula_available"]
        and not nonlinear_summary["phase1_literal_two_component_nonlinear_formula_available"]
        and not trial3_summary["exact_action_level_closed_ell0_operator_available"]
    )
    solver_fix_placeholder_surface_available = bool(
        sign_base.hit(solver_fix_text, "\\text{NL}(f_0)") is not None
        and sign_base.hit(solver_fix_text, "\\text{coupling}(f_L)") is not None
        and sign_base.hit(solver_fix_text, "f_L'' + \\frac{2}{r}f_L'") is not None
    )
    next_steps_exact_ell0_series_surface_available = bool(
        sign_base.hit(next_steps_text, "### Step A.") is not None
        and sign_base.hit(next_steps_text, "f_0(r)=a_0+a_2 r^2 + a_4 r^4 + \\cdots") is not None
        and sign_base.hit(next_steps_text, "f_L(r)=b_1 r + b_3 r^3 + b_5 r^5 + \\cdots") is not None
    )
    next_steps_exact_longitudinal_operator_surface_available = bool(
        sign_base.hit(next_steps_text, "### Step B.") is not None
        and sign_base.hit(next_steps_text, "L_L[f_L] = S[f_0]") is not None
    )
    next_steps_effective_source_theorem_surface_available = bool(
        sign_base.hit(next_steps_text, "### Step C.") is not None
        and sign_base.hit(next_steps_text, "\\mathcal L \\supset a_\\mu\\,J^{\\mu}_{\\rm eff}[P^{\\rm Qball}]") is not None
        and sign_base.hit(next_steps_text, "J_eff^0") is not None
    )
    substantive_pack_update_requires_new_public_canonical_surface = bool(
        current_pack_exhausted_honestly
        and current_pack_missing_action_formulae_remain_placeholder_only
    )
    pack_update_primary_surface_changes_internal_operator_surface = bool(
        next_steps_exact_ell0_series_surface_available
        and next_steps_exact_longitudinal_operator_surface_available
    )
    pack_update_secondary_surface_targets_canonical_source_rule = bool(
        next_steps_effective_source_theorem_surface_available
    )
    old_density_proxy_eigenvalue_retry_reopened_by_pack_update = False
    farther_hybrid_continuation_reopen_required_now = False
    substantive_pack_update_surface_explicit_now = bool(
        substantive_pack_update_requires_new_public_canonical_surface
        and solver_fix_placeholder_surface_available
        and pack_update_primary_surface_changes_internal_operator_surface
        and pack_update_secondary_surface_targets_canonical_source_rule
    )
    substantive_pack_update_adoptable_now = bool(
        substantive_pack_update_surface_explicit_now
        and not old_density_proxy_eigenvalue_retry_reopened_by_pack_update
        and not farther_hybrid_continuation_reopen_required_now
    )

    rows = [
        sign_base.row(
            "current_pack_exhausted_honestly",
            "pass" if current_pack_exhausted_honestly else "reject",
            "current pack exhausted honestly",
            sign_base.truth(current_pack_exhausted_honestly),
            "The retained pack has now failed the literal reciprocal/nonlinear/ell=0 closure shots without leaving a current-pack primary fix.",
        ),
        sign_base.row(
            "current_pack_missing_action_formulae_remain_placeholder_only",
            "watch" if current_pack_missing_action_formulae_remain_placeholder_only else "pass",
            "current pack missing-action formulae remain placeholder only",
            sign_base.truth(current_pack_missing_action_formulae_remain_placeholder_only),
            "The retained pack still has placeholders and theorem notes, but no literal reciprocal backreaction, no literal two-component nonlinear closure, and no closed exact ell=0 operator.",
        ),
        sign_base.row(
            "solver_fix_placeholder_surface_available",
            "pass" if solver_fix_placeholder_surface_available else "reject",
            "solver-fix placeholder surface available",
            sign_base.truth(solver_fix_placeholder_surface_available),
            "The retained solver-fix note already isolates the missing operator as `NL(f_0)` plus `coupling(f_L)` rather than as a farther-q or density retry problem.",
        ),
        sign_base.row(
            "next_steps_exact_ell0_series_surface_available",
            "pass" if next_steps_exact_ell0_series_surface_available else "reject",
            "next-steps exact ell=0 series surface available",
            sign_base.truth(next_steps_exact_ell0_series_surface_available),
            "Step A already spells out the exact two-component near-origin series, so the pack update can target a new ell=0 operator surface directly.",
        ),
        sign_base.row(
            "next_steps_exact_longitudinal_operator_surface_available",
            "pass" if next_steps_exact_longitudinal_operator_surface_available else "reject",
            "next-steps exact longitudinal operator surface available",
            sign_base.truth(next_steps_exact_longitudinal_operator_surface_available),
            "Step B already identifies the exact longitudinal operator theorem target `L_L[f_L] = S[f_0]`.",
        ),
        sign_base.row(
            "next_steps_effective_source_theorem_surface_available",
            "pass" if next_steps_effective_source_theorem_surface_available else "reject",
            "next-steps effective source theorem surface available",
            sign_base.truth(next_steps_effective_source_theorem_surface_available),
            "Step C already isolates the exact source/current theorem as the downstream canonical observable surface.",
        ),
        sign_base.row(
            "substantive_pack_update_requires_new_public_canonical_surface",
            "pass" if substantive_pack_update_requires_new_public_canonical_surface else "reject",
            "substantive pack update requires new public-canonical surface",
            sign_base.truth(substantive_pack_update_requires_new_public_canonical_surface),
            "Because the blocker is now a missing action-level term rather than numeric evidence, the next honest move is a new canonical surface rather than another same-pack retry.",
        ),
        sign_base.row(
            "pack_update_primary_surface_changes_internal_operator_surface",
            "pass" if pack_update_primary_surface_changes_internal_operator_surface else "reject",
            "pack-update primary surface changes internal operator surface",
            sign_base.truth(pack_update_primary_surface_changes_internal_operator_surface),
            "The primary updated-pack surface is the exact ell=0 two-component series/operator theorem, which changes the internal action-level operator rather than the observable readout only.",
        ),
        sign_base.row(
            "pack_update_secondary_surface_targets_canonical_source_rule",
            "pass" if pack_update_secondary_surface_targets_canonical_source_rule else "reject",
            "pack-update secondary surface targets canonical source rule",
            sign_base.truth(pack_update_secondary_surface_targets_canonical_source_rule),
            "The secondary updated-pack surface is the exact effective source theorem for photon coupling after the operator is fixed.",
        ),
        sign_base.row(
            "old_density_proxy_eigenvalue_retry_reopened_by_pack_update",
            "reject" if not old_density_proxy_eigenvalue_retry_reopened_by_pack_update else "pass",
            "old density/proxy/eigenvalue retry reopened by pack update",
            sign_base.truth(old_density_proxy_eigenvalue_retry_reopened_by_pack_update),
            "The pack update must not be used as a pretext to reopen old density, proxy, or eigenvalue retry families that were already exhausted honestly.",
        ),
        sign_base.row(
            "farther_hybrid_continuation_reopen_required_now",
            "pass" if farther_hybrid_continuation_reopen_required_now else "reject",
            "farther hybrid continuation reopen required now",
            sign_base.truth(farther_hybrid_continuation_reopen_required_now),
            "Extra q-range remains reserve-only because the current blocker is an action-level theorem surface, not an origin-discrimination evidence shortage.",
        ),
        sign_base.row(
            "substantive_pack_update_surface_explicit_now",
            "pass" if substantive_pack_update_surface_explicit_now else "reject",
            "substantive pack-update surface explicit now",
            sign_base.truth(substantive_pack_update_surface_explicit_now),
            "The retained notes now point to a concrete updated-pack surface: exact ell=0 series/operator first, exact effective source theorem second.",
        ),
        sign_base.row(
            "substantive_pack_update_adoptable_now",
            "pass" if substantive_pack_update_adoptable_now else "reject",
            "substantive pack update adoptable now",
            sign_base.truth(substantive_pack_update_adoptable_now),
            "The update is honest now because it changes the missing action-level surface directly while keeping old retries and farther hybrid continuation closed.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "retained_scalar_residual_rel": float(prior_summary["retained_scalar_residual_rel"]),
        "trial3_family_ell0_best_component_ratio": float(
            trial3_summary["trial3_family_ell0_best_component_ratio"]
        ),
        "current_pack_exhausted_honestly": current_pack_exhausted_honestly,
        "current_pack_missing_action_formulae_remain_placeholder_only": current_pack_missing_action_formulae_remain_placeholder_only,
        "solver_fix_placeholder_surface_available": solver_fix_placeholder_surface_available,
        "next_steps_exact_ell0_series_surface_available": next_steps_exact_ell0_series_surface_available,
        "next_steps_exact_longitudinal_operator_surface_available": next_steps_exact_longitudinal_operator_surface_available,
        "next_steps_effective_source_theorem_surface_available": next_steps_effective_source_theorem_surface_available,
        "substantive_pack_update_requires_new_public_canonical_surface": substantive_pack_update_requires_new_public_canonical_surface,
        "pack_update_primary_surface_changes_internal_operator_surface": pack_update_primary_surface_changes_internal_operator_surface,
        "pack_update_secondary_surface_targets_canonical_source_rule": pack_update_secondary_surface_targets_canonical_source_rule,
        "old_density_proxy_eigenvalue_retry_reopened_by_pack_update": old_density_proxy_eigenvalue_retry_reopened_by_pack_update,
        "farther_hybrid_continuation_reopen_required_now": farther_hybrid_continuation_reopen_required_now,
        "substantive_pack_update_surface_explicit_now": substantive_pack_update_surface_explicit_now,
        "substantive_pack_update_adoptable_now": substantive_pack_update_adoptable_now,
        "selected_primary_pack_update_surface": "exact_ell0_two_component_series_and_longitudinal_operator",
        "selected_secondary_pack_update_surface": "exact_effective_source_theorem",
        "selected_reserve_pack_update_surface": "farther_hybrid_extra_q_range_only",
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": FOLLOWUP_ROUTE_NAME,
        "selected_followup_route_or_none": FOLLOWUP_ROUTE,
        "physical_reject_required": False,
    }

    declaration_payload = sign_base.payload(
        "8.7.56.2441",
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
                "theory_lessons": sign_base.display_path(THEORY_LESSONS),
                "part5": sign_base.display_path(PART5),
                "prior_gate": sign_base.display_path(PRIOR_GATE),
                "reciprocal_audit": sign_base.display_path(RECIPROCAL_AUDIT),
                "nonlinear_audit": sign_base.display_path(NONLINEAR_AUDIT),
                "trial3_ell0_audit": sign_base.display_path(TRIAL3_ELL0_AUDIT),
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
            "overall_status": "vector_qball_form_factor_substantive_pack_update_audit_declared",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "formulas": build_formulae(),
            "hits": {
                "status_branch_hit": sign_base.hit(status_text, "8.7.56.2439"),
                "roadmap_branch_hit": sign_base.hit(roadmap_text, ".2439-.2442"),
                "current_problem_hit": sign_base.hit(current_problem_text, "substantive pack update audit"),
                "current_status_hit": sign_base.hit(current_status_text, "substantive pack update audit"),
                "unified_roadmap_hit": sign_base.hit(unified_text, ".2435-.2438"),
                "long_roadmap_branch_hit": sign_base.hit(long_text, ".2439-.2442"),
                "long_roadmap_no_old_retry_hit": sign_base.hit(long_text, "old density / proxy / eigenvalue retry"),
                "lessons_pack_update_hit": sign_base.hit(lessons_text, "retained real-branch sign-parity theorem を超える substantive pack update"),
                "part5_hit": sign_base.hit(part5_text, "substantive pack update audit"),
                "solver_fix_nl_hit": sign_base.hit(solver_fix_text, "\\text{NL}(f_0)"),
                "solver_fix_coupling_hit": sign_base.hit(solver_fix_text, "\\text{coupling}(f_L)"),
                "next_steps_step_a_hit": sign_base.hit(next_steps_text, "### Step A."),
                "next_steps_step_b_hit": sign_base.hit(next_steps_text, "### Step B."),
                "next_steps_step_c_hit": sign_base.hit(next_steps_text, "### Step C."),
                "next_steps_series_hit": sign_base.hit(next_steps_text, "f_L(r)=b_1 r + b_3 r^3 + b_5 r^5 + \\cdots"),
                "next_steps_operator_hit": sign_base.hit(next_steps_text, "L_L[f_L] = S[f_0]"),
                "next_steps_source_hit": sign_base.hit(next_steps_text, "\\mathcal L \\supset a_\\mu\\,J^{\\mu}_{\\rm eff}[P^{\\rm Qball}]"),
            },
        },
    )
    declaration_paths = write_artifact("declaration_gate", declaration_payload)

    route_payload = {
        "generated_utc": sign_base.now_iso(),
        "phase": {
            "phase": 8,
            "step": "8.7.56.2442",
            "name": STEP_NAME + " route sync",
        },
        "inputs": declaration_paths,
        "rows": rows,
        "summary": summary,
        "decision": {
            "overall_status": "vector_qball_form_factor_substantive_pack_update_route_synced",
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

    print(f"[done] {STEP_TAG} substantive pack update audit completed")


if __name__ == "__main__":
    main()
