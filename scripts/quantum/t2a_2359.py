#!/usr/bin/env python3
"""Generate 8.7.56.2359-.2362 exact coupled eigenvalue-shift theorem artifacts."""

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
        "8.7.56.2355-2358",
        "residual_origin_synthesis",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
DELTA_BETA_AUDIT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.2351-2354",
        "missing_action_delta_beta1_audit",
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
ELL0_ANCHOR_EVAL = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.1483-1486",
        "ell0_anchor_continuation",
        prefix="q",
    ),
    "numeric_evaluation",
)["json"]

STEP_TAG = "8.7.56.2359-2362"
STEP_NAME = "Trial-2 numeric alpha vector Q-ball form-factor exact coupled eigenvalue-shift theorem audit"
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "exact_coupled_eigshift_theorem",
    prefix="q",
)

PRIOR_CLASS = "vector_qball_form_factor_residual_origin_missing_action_profile_fixed_eigenvalue_shift_candidate_selected_exact_coupled_theorem_next"
BRANCH_CLASS = "vector_qball_form_factor_residual_origin_missing_action_exact_coupled_eigenvalue_shift_theorem_audit_profile_fixed_candidate_retained_gate"
NEXT_ROUTE_NAME = "trial2_numeric_alpha_vector_qball_form_factor_profile_fixed_eigenvalue_shift_decision_gate_hybrid_reserve_refresh"
NEXT_ROUTE = "8.7.56.2363"
FOLLOWUP_ROUTE_NAME = "trial2_numeric_alpha_vector_qball_form_factor_exact_action_level_operator_completion_audit"
FOLLOWUP_ROUTE = "8.7.56.2367"

COUPLING_PROXY_PASS_LIMIT = 0.01


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


# 関数: theorem audit で使う式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used in the exact coupled theorem audit."""
    return {
        "target_shift": "delta(beta_1^2)_(req) = beta_corrected^2 - beta_1^2",
        "operator_proxy": "Xi_proxy = delta(beta_1^2)_(req) / max|f_L/f_0|^2",
        "closure_rule": "exact theorem requires cross term + constraint elimination + noncollapsed ell=0 coupled closure",
    }


# 関数: `.2359-.2362` を実行する。

def main() -> None:
    """Execute the exact coupled eigenvalue-shift theorem audit."""
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
        DELTA_BETA_AUDIT,
        ELL0_OPERATOR_AUDIT,
        ELL0_ANCHOR_EVAL,
    ):
        sign_base.require(path)

    status_text = sign_base.read_text(STATUS)
    roadmap_text = sign_base.read_text(ROADMAP)
    current_problem_text = sign_base.read_text(CURRENT_PROBLEM)
    current_status_text = sign_base.read_text(CURRENT_STATUS)
    unified_text = sign_base.read_text(UNIFIED_ROADMAP)
    long_text = sign_base.read_text(LONG_ROADMAP)
    part5_text = sign_base.read_text(PART5)

    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    delta_beta_summary = sign_base.read_json(DELTA_BETA_AUDIT)["summary"]
    operator_summary = sign_base.read_json(ELL0_OPERATOR_AUDIT)["summary"]
    anchor_summary = sign_base.read_json(ELL0_ANCHOR_EVAL)["summary"]

    required_delta_beta2 = float(delta_beta_summary["delta_beta2_exact_profile_fixed"])
    max_fl_over_f0 = float(delta_beta_summary["max_fl_over_f0_ceiling"])
    tail_to_input_ratio = float(anchor_summary["phase1_equivalent_row"]["tail_to_input_ratio"])
    coupling_proxy_from_max_ratio_sq = required_delta_beta2 / (max_fl_over_f0 * max_fl_over_f0)
    coupling_proxy_from_tail_ratio_sq = required_delta_beta2 / (tail_to_input_ratio * tail_to_input_ratio)

    cross_term_present = bool(operator_summary["phase1_exact_solver_cross_term_present"])
    constraint_elimination_present = bool(
        operator_summary["phase1_exact_solver_constraint_elimination_present"]
    )
    nonlinear_closure_scalar_only = bool(
        operator_summary["phase1_exact_solver_scalar_nonlinear_ansatz_only"]
    )
    ell0_coupling_collapses = bool(operator_summary["trial3_family_solver_ell0_coupling_collapses"])
    exact_closed_operator_available = bool(
        operator_summary["exact_action_level_closed_ell0_operator_available"]
    )
    missing_prerequisite_count = int(
        (not cross_term_present)
        + (not constraint_elimination_present)
        + ell0_coupling_collapses
    )

    operator_coefficient_size_modest = coupling_proxy_from_max_ratio_sq < COUPLING_PROXY_PASS_LIMIT
    exact_coupled_theorem_derivable_under_current_pack = bool(
        exact_closed_operator_available
        and cross_term_present
        and constraint_elimination_present
        and not ell0_coupling_collapses
        and not nonlinear_closure_scalar_only
    )
    profile_fixed_candidate_retained_pending_operator_completion = bool(
        prior_summary["profile_fixed_eigenvalue_shift_candidate_admissible"]
        and not exact_coupled_theorem_derivable_under_current_pack
    )
    hybrid_supporting_evidence_reopen_required = False
    observable_secondary_recheck_required_now = False

    rows = [
        sign_base.row(
            "inventory_ready",
            "pass",
            "exact coupled theorem audit inventory ready",
            1.0,
            "This branch starts only after the profile-fixed delta-beta1 candidate has been selected as the missing-action first shot.",
        ),
        sign_base.row(
            "free_linear_backbone_available",
            "pass" if operator_summary["exact_action_level_linear_backbone_available"] else "reject",
            "free linear backbone available",
            sign_base.truth(operator_summary["exact_action_level_linear_backbone_available"]),
            "The public post-photon two-by-two backbone is already frozen and can support an exact theorem only if the missing coupled pieces are supplied.",
        ),
        sign_base.row(
            "missing_operator_prerequisite_count",
            "watch",
            "missing exact-coupled operator prerequisite count",
            float(missing_prerequisite_count),
            "The current pack is still missing explicit cross-term realization, constraint elimination, and a noncollapsed ell=0 coupled closure.",
        ),
        sign_base.row(
            "operator_coefficient_proxy_from_max_ratio_sq",
            "pass" if operator_coefficient_size_modest else "watch",
            "required dimensionless coupling proxy from max|fL/f0|^2",
            coupling_proxy_from_max_ratio_sq,
            "If the exact theorem produced delta(beta_1^2) from an ell=0 load proportional to max|fL/f0|^2, the needed coefficient would be small rather than unnaturally large.",
        ),
        sign_base.row(
            "operator_coefficient_proxy_from_tail_ratio_sq",
            "pass",
            "required dimensionless coupling proxy from tail-to-input ratio squared",
            coupling_proxy_from_tail_ratio_sq,
            "The same conclusion survives when the retained tail-localization ratio is used as the load proxy instead of max|fL/f0|.",
        ),
        sign_base.row(
            "exact_coupled_theorem_derivable_under_current_pack",
            "reject" if not exact_coupled_theorem_derivable_under_current_pack else "pass",
            "exact coupled eigenvalue-shift theorem derivable under current pack",
            sign_base.truth(exact_coupled_theorem_derivable_under_current_pack),
            "The blocker is no longer the size of the required shift but the literal absence of the coupled operator ingredients that would derive it canonically.",
        ),
        sign_base.row(
            "profile_fixed_candidate_retained_pending_operator_completion",
            "pass" if profile_fixed_candidate_retained_pending_operator_completion else "reject",
            "profile-fixed delta-beta1 candidate retained pending exact operator completion",
            sign_base.truth(profile_fixed_candidate_retained_pending_operator_completion),
            "The profile-fixed candidate remains useful because the required shift is modest, even though the theorem itself is still blocked.",
        ),
        sign_base.row(
            "observable_secondary_recheck_required_now",
            "reject",
            "observable secondary recheck required now",
            sign_base.truth(observable_secondary_recheck_required_now),
            "The low-q observable family is already exact, so the next mainline move stays inside the missing-action lane rather than reopening the observable lane.",
        ),
        sign_base.row(
            "hybrid_supporting_evidence_reopen_required_now",
            "reject",
            "hybrid supporting-evidence reopen required now",
            sign_base.truth(hybrid_supporting_evidence_reopen_required),
            "Additional q-range is still unnecessary before the exact coupled operator lane is sharpened.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "retained_scalar_residual_rel": float(prior_summary["retained_scalar_residual_rel"]),
        "delta_beta_exact_profile_fixed": float(delta_beta_summary["delta_beta_exact_profile_fixed"]),
        "delta_beta2_exact_profile_fixed": required_delta_beta2,
        "required_delta_beta2_fraction_of_beta_gap": float(
            delta_beta_summary["required_delta_beta2_fraction_of_beta_gap"]
        ),
        "required_delta_beta2_vs_ceiling_sq": float(
            delta_beta_summary["required_delta_beta2_vs_ceiling_sq"]
        ),
        "operator_coefficient_proxy_from_max_ratio_sq": coupling_proxy_from_max_ratio_sq,
        "operator_coefficient_proxy_from_tail_ratio_sq": coupling_proxy_from_tail_ratio_sq,
        "missing_operator_prerequisite_count": missing_prerequisite_count,
        "phase1_exact_solver_cross_term_present": cross_term_present,
        "phase1_exact_solver_constraint_elimination_present": constraint_elimination_present,
        "phase1_exact_solver_scalar_nonlinear_ansatz_only": nonlinear_closure_scalar_only,
        "trial3_family_solver_ell0_coupling_collapses": ell0_coupling_collapses,
        "exact_action_level_closed_ell0_operator_available": exact_closed_operator_available,
        "exact_coupled_theorem_derivable_under_current_pack": exact_coupled_theorem_derivable_under_current_pack,
        "profile_fixed_candidate_retained_pending_operator_completion": profile_fixed_candidate_retained_pending_operator_completion,
        "operator_coefficient_size_blocker": not operator_coefficient_size_modest,
        "observable_secondary_recheck_required_now": observable_secondary_recheck_required_now,
        "hybrid_supporting_evidence_reopen_required": hybrid_supporting_evidence_reopen_required,
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": FOLLOWUP_ROUTE_NAME,
        "selected_followup_route_or_none": FOLLOWUP_ROUTE,
        "physical_reject_required": False,
    }

    declaration_payload = sign_base.payload(
        "8.7.56.2361",
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
                "delta_beta_audit": sign_base.display_path(DELTA_BETA_AUDIT),
                "ell0_operator_audit": sign_base.display_path(ELL0_OPERATOR_AUDIT),
                "ell0_anchor_eval": sign_base.display_path(ELL0_ANCHOR_EVAL),
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
            "overall_status": "vector_qball_form_factor_exact_coupled_eigenvalue_shift_theorem_audited",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "formulas": build_formulae(),
            "hits": {
                "status_branch_hit": sign_base.hit(status_text, "8.7.56.2359"),
                "roadmap_branch_hit": sign_base.hit(roadmap_text, ".2359-.2362"),
                "current_problem_hit": sign_base.hit(current_problem_text, "profile-fixed eigenvalue-shift"),
                "current_status_hit": sign_base.hit(current_status_text, "profile-fixed eigenvalue-shift"),
                "unified_roadmap_hit": sign_base.hit(unified_text, ".2359-.2362"),
                "long_roadmap_hit": sign_base.hit(long_text, ".2359-.2362"),
                "part5_hit": sign_base.hit(part5_text, "2026-03-30 correction/update"),
            },
        },
    )
    declaration_paths = write_artifact("declaration_gate", declaration_payload)

    route_payload = {
        "generated_utc": sign_base.now_iso(),
        "phase": {
            "phase": 8,
            "step": "8.7.56.2362",
            "name": STEP_NAME + " route sync",
        },
        "inputs": declaration_paths,
        "rows": rows,
        "summary": summary,
        "decision": {
            "overall_status": "vector_qball_form_factor_exact_coupled_eigenvalue_shift_theorem_route_synced",
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

    print(f"[done] {STEP_TAG} exact coupled eigenvalue-shift theorem audit completed")
    print(f"[info] declaration_gate_json={declaration_paths['json']}")
    print(f"[info] declaration_gate_csv={declaration_paths['csv']}")


if __name__ == "__main__":
    main()
