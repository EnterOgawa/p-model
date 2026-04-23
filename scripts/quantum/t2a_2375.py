#!/usr/bin/env python3
"""Generate 8.7.56.2375-.2378 exact cross-term completion audit artifacts."""

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
        "8.7.56.2371-2374",
        "exact_operator_completion_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
PRIOR_AUDIT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.2367-2370",
        "exact_operator_completion_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
EIGSHIFT_AUDIT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.2359-2362",
        "exact_coupled_eigshift_theorem",
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
TRIAL3_FAMILY = ROOT / "scripts" / "quantum" / "mass_origin_v2_trial3_two_component_spectrum_branch.py"
SOLVER_FIX = Path(r"C:\Users\ogawa\Downloads\pmodel_v2_trial2_solver_fix_final.md")
PERTURBATIVE_NOTE = Path(r"C:\Users\ogawa\Downloads\pmodel_v2_trial2_perturbative_fL_correction.md")

STEP_TAG = "8.7.56.2375-2378"
STEP_NAME = "Trial-2 numeric alpha vector Q-ball form-factor exact action-level cross-term completion audit"
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "cross_term_completion_audit",
    prefix="q",
)

PRIOR_CLASS = "vector_qball_form_factor_residual_origin_missing_action_cross_term_primary_constraint_secondary_ell0_closure_reserve_next"
BRANCH_CLASS = "vector_qball_form_factor_residual_origin_missing_action_cross_term_phase1_literal_target_trial3_collapse_secondary_gate"
NEXT_ROUTE_NAME = "trial2_numeric_alpha_vector_qball_form_factor_cross_term_completion_decision_gate_constraint_elimination_refresh"
NEXT_ROUTE = "8.7.56.2379"
FOLLOWUP_ROUTE_NAME = "trial2_numeric_alpha_vector_qball_form_factor_phase1_exact_solver_literal_cross_term_realization_audit"
FOLLOWUP_ROUTE = "8.7.56.2383"

MODEST_PROXY_LIMIT = 0.01


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

def build_formulae() -> dict[str, str]:
    """Return formulas used in the cross-term audit."""
    return {
        "backbone": "M(omega,k) = [[k^2 + m_eff^2, -omega k], [-omega k, omega^2]]",
        "kinetic_identity": "F_{0r}^{(P)} = i omega f_L - f_0'",
        "ordering_rule": "phase-1 literal cross term -> constraint elimination -> noncollapsed ell=0 closure",
    }


# 関数: `.2375-.2378` を実行する。

def main() -> None:
    """Execute the exact action-level cross-term completion audit."""
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
        EIGSHIFT_AUDIT,
        ELL0_OPERATOR_AUDIT,
        PHASE1_SOLVER,
        TRIAL3_FAMILY,
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
    trial3_text = sign_base.read_text(TRIAL3_FAMILY)
    solver_fix_text = sign_base.read_text(SOLVER_FIX)
    perturbative_text = sign_base.read_text(PERTURBATIVE_NOTE)

    prior_gate_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    prior_audit_summary = sign_base.read_json(PRIOR_AUDIT)["summary"]
    eigshift_summary = sign_base.read_json(EIGSHIFT_AUDIT)["summary"]
    operator_audit = sign_base.read_json(ELL0_OPERATOR_AUDIT)
    operator_summary = operator_audit["summary"]
    operator_evidence = operator_audit["evidence"]

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
    trial3_kproxy_hit = sign_base.hit(
        trial3_text,
        "k_proxy = math.sqrt(max(float(ell * (ell + 1)), 0.0)) / rr",
    )
    solver_fix_identity_hit = sign_base.hit(
        solver_fix_text,
        "F_{0r}^{(P)} = i\\omega f_L - f_0'",
    )
    perturbative_identity_hit = sign_base.hit(
        perturbative_text,
        "F_{0r}^{(P)} = i\\omega f_L - f_0'",
    )

    phase1_diagnostic_cross_term_template_present = bool(diagnostic_source_hit)
    phase1_exact_solver_literal_cross_term_present = bool(
        hit_has_token(exact_f0_hit, "f_l")
        and hit_has_token(exact_fl_hit, "f0")
    )
    phase1_exact_solver_primary_target_supported = bool(
        phase1_diagnostic_cross_term_template_present
        and not phase1_exact_solver_literal_cross_term_present
        and prior_audit_summary["cross_term_primary_completion_supported"]
    )
    trial3_family_ell0_collapse_secondary_only = bool(
        operator_summary["trial3_family_solver_ell0_coupling_collapses"]
        and bool(trial3_kproxy_hit)
        and phase1_exact_solver_primary_target_supported
    )
    profile_fixed_candidate_consistent_with_cross_term_lane = bool(
        eigshift_summary["profile_fixed_candidate_retained_pending_operator_completion"]
        and float(eigshift_summary["operator_coefficient_proxy_from_max_ratio_sq"]) < MODEST_PROXY_LIMIT
    )
    exact_cross_term_completion_supported_under_current_pack = bool(
        prior_gate_summary["gate_a_cross_term_completion_selected"]
        and operator_summary["exact_action_level_linear_backbone_available"]
        and bool(operator_evidence.get("qform_matrix"))
        and bool(operator_evidence.get("solver_fix_offdiag_hit"))
        and phase1_exact_solver_primary_target_supported
    )
    constraint_elimination_downstream_of_cross_term = bool(
        prior_gate_summary["gate_b_constraint_elimination_retained"]
        and exact_cross_term_completion_supported_under_current_pack
    )
    pack_update_required_now = False

    rows = [
        sign_base.row(
            "inventory_ready",
            "pass",
            "exact cross-term completion audit inventory ready",
            1.0,
            "This branch starts only after the completion ordering has been frozen to cross term first, constraint elimination second, noncollapsed ell=0 closure reserve.",
        ),
        sign_base.row(
            "phase1_diagnostic_cross_term_template_present",
            "pass" if phase1_diagnostic_cross_term_template_present else "reject",
            "phase-1 diagnostic f_L source already carries the cross-term template",
            sign_base.truth(phase1_diagnostic_cross_term_template_present),
            "The diagnostic-only driven f_L solve already injects beta * f_0' as an explicit source, so the pack already knows the literal cross-term structure.",
        ),
        sign_base.row(
            "phase1_exact_solver_literal_cross_term_present",
            "pass" if phase1_exact_solver_literal_cross_term_present else "reject",
            "phase-1 exact pilot already carries the literal cross term",
            sign_base.truth(phase1_exact_solver_literal_cross_term_present),
            "The current exact pilot remains diagonal: the f_0 equation does not reference f_L and the f_L equation does not reference f_0.",
        ),
        sign_base.row(
            "phase1_exact_solver_primary_target_supported",
            "pass" if phase1_exact_solver_primary_target_supported else "reject",
            "phase-1 exact solver supported as the primary literal cross-term target",
            sign_base.truth(phase1_exact_solver_primary_target_supported),
            "Because the diagnostic template and the frozen backbone already expose the missing mixing, the smallest literal completion target is the phase-1 exact pilot rather than a new family map.",
        ),
        sign_base.row(
            "trial3_family_ell0_collapse_secondary_only",
            "pass" if trial3_family_ell0_collapse_secondary_only else "reject",
            "trial-3 family ell=0 collapse retained as secondary-only evidence",
            sign_base.truth(trial3_family_ell0_collapse_secondary_only),
            "The sqrt(ell(ell+1))/r proxy explains why the old family map cannot be the primary fix at ell=0, but that collapse is downstream of the literal phase-1 cross-term omission.",
        ),
        sign_base.row(
            "profile_fixed_candidate_consistent_with_cross_term_lane",
            "pass" if profile_fixed_candidate_consistent_with_cross_term_lane else "reject",
            "profile-fixed eigenvalue-shift candidate consistent with the cross-term lane",
            sign_base.truth(profile_fixed_candidate_consistent_with_cross_term_lane),
            "The required operator coefficient proxy stays modest, so the retained delta-beta1 candidate remains scientifically compatible with a literal cross-term realization.",
        ),
        sign_base.row(
            "exact_cross_term_completion_supported_under_current_pack",
            "pass" if exact_cross_term_completion_supported_under_current_pack else "reject",
            "exact cross-term completion supported under the current pack",
            sign_base.truth(exact_cross_term_completion_supported_under_current_pack),
            "The post-photon backbone, solver-fix kinetic identity, and retained profile-fixed candidate already supply the ingredients needed for a literal phase-1 cross-term realization without a pack update.",
        ),
        sign_base.row(
            "constraint_elimination_downstream_of_cross_term",
            "pass" if constraint_elimination_downstream_of_cross_term else "reject",
            "constraint elimination remains downstream of cross-term realization",
            sign_base.truth(constraint_elimination_downstream_of_cross_term),
            "Constraint elimination still acts on the coupled operator produced by the cross-term completion, so it is not the primary blocker of the current branch.",
        ),
        sign_base.row(
            "pack_update_required_now",
            "pass" if pack_update_required_now else "reject",
            "substantive pack update required now",
            sign_base.truth(pack_update_required_now),
            "The current blocker is a literal implementation omission inside the retained pack, not missing external physics.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "retained_scalar_residual_rel": float(prior_gate_summary["retained_scalar_residual_rel"]),
        "delta_beta2_exact_profile_fixed": float(eigshift_summary["delta_beta2_exact_profile_fixed"]),
        "operator_coefficient_proxy_from_max_ratio_sq": float(
            eigshift_summary["operator_coefficient_proxy_from_max_ratio_sq"]
        ),
        "phase1_diagnostic_cross_term_template_present": phase1_diagnostic_cross_term_template_present,
        "phase1_exact_solver_literal_cross_term_present": phase1_exact_solver_literal_cross_term_present,
        "phase1_exact_solver_primary_target_supported": phase1_exact_solver_primary_target_supported,
        "trial3_family_ell0_collapse_secondary_only": trial3_family_ell0_collapse_secondary_only,
        "profile_fixed_candidate_consistent_with_cross_term_lane": profile_fixed_candidate_consistent_with_cross_term_lane,
        "exact_cross_term_completion_supported_under_current_pack": exact_cross_term_completion_supported_under_current_pack,
        "constraint_elimination_downstream_of_cross_term": constraint_elimination_downstream_of_cross_term,
        "pack_update_required_now": pack_update_required_now,
        "selected_primary_cross_term_target": "phase1_exact_solver_literal_completion",
        "selected_secondary_cross_term_issue": "trial3_ell0_collapse_secondary_only",
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": FOLLOWUP_ROUTE_NAME,
        "selected_followup_route_or_none": FOLLOWUP_ROUTE,
        "physical_reject_required": False,
    }

    declaration_payload = sign_base.payload(
        "8.7.56.2377",
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
                "eigshift_audit": sign_base.display_path(EIGSHIFT_AUDIT),
                "ell0_operator_audit": sign_base.display_path(ELL0_OPERATOR_AUDIT),
                "phase1_solver": sign_base.display_path(PHASE1_SOLVER),
                "trial3_family": sign_base.display_path(TRIAL3_FAMILY),
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
            "overall_status": "vector_qball_form_factor_cross_term_completion_audited",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "formulas": build_formulae(),
            "hits": {
                "status_branch_hit": sign_base.hit(status_text, "8.7.56.2375"),
                "roadmap_branch_hit": sign_base.hit(roadmap_text, ".2375-.2378"),
                "current_problem_hit": sign_base.hit(current_problem_text, "cross-term completion audit"),
                "current_status_hit": sign_base.hit(current_status_text, "cross-term completion audit"),
                "unified_roadmap_hit": sign_base.hit(unified_text, ".2375-.2378"),
                "long_roadmap_hit": sign_base.hit(long_text, ".2375-.2378"),
                "part5_hit": sign_base.hit(part5_text, "exact action-level cross-term completion audit"),
                "phase1_diagnostic_source_hit": diagnostic_source_hit,
                "phase1_exact_f0_hit": exact_f0_hit,
                "phase1_exact_fl_hit": exact_fl_hit,
                "trial3_kproxy_hit": trial3_kproxy_hit,
                "solver_fix_identity_hit": solver_fix_identity_hit,
                "perturbative_identity_hit": perturbative_identity_hit,
                "operator_qform_matrix": operator_evidence.get("qform_matrix"),
            },
        },
    )
    declaration_paths = write_artifact("declaration_gate", declaration_payload)

    route_payload = {
        "generated_utc": sign_base.now_iso(),
        "phase": {
            "phase": 8,
            "step": "8.7.56.2378",
            "name": STEP_NAME + " route sync",
        },
        "inputs": declaration_paths,
        "rows": rows,
        "summary": summary,
        "decision": {
            "overall_status": "vector_qball_form_factor_cross_term_completion_route_synced",
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

    print(f"[done] {STEP_TAG} exact action-level cross-term completion audit completed")
    print(f"[info] declaration_gate_json={declaration_paths['json']}")
    print(f"[info] declaration_gate_csv={declaration_paths['csv']}")


if __name__ == "__main__":
    main()
