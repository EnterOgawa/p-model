#!/usr/bin/env python3
"""Generate 8.7.56.2367-.2370 exact operator-completion audit artifacts."""

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
        "8.7.56.2363-2366",
        "profile_fixed_eigshift_gate",
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

STEP_TAG = "8.7.56.2367-2370"
STEP_NAME = "Trial-2 numeric alpha vector Q-ball form-factor exact action-level operator completion audit"
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "exact_operator_completion_audit",
    prefix="q",
)

PRIOR_CLASS = "vector_qball_form_factor_residual_origin_missing_action_profile_fixed_eigenvalue_shift_candidate_retained_exact_operator_completion_next"
BRANCH_CLASS = "vector_qball_form_factor_residual_origin_missing_action_operator_completion_cross_term_primary_constraint_secondary_gate"
NEXT_ROUTE_NAME = "trial2_numeric_alpha_vector_qball_form_factor_exact_operator_completion_decision_gate_hybrid_reserve_refresh"
NEXT_ROUTE = "8.7.56.2371"
FOLLOWUP_ROUTE_NAME = "trial2_numeric_alpha_vector_qball_form_factor_exact_action_level_cross_term_completion_audit"
FOLLOWUP_ROUTE = "8.7.56.2375"


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


# 関数: operator completion audit で使う式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used in the operator-completion audit."""
    return {
        "ordering_rule": "cross term -> constraint elimination -> noncollapsed ell=0 closure",
        "primary_logic": "primary if the ingredient is already frozen in the public backbone but missing in the current exact implementation",
        "secondary_logic": "secondary if it acts on the coupled operator produced by the primary completion",
        "reserve_logic": "reserve if it depends on the completed linear coupled operator and its elimination before the nonlinear theorem can be closed",
    }


# 関数: rows を row_id で引く辞書を作る。

def row_map(payload: dict) -> dict[str, dict]:
    """Return rows indexed by row_id."""
    return {row["row_id"]: row for row in payload["rows"]}


# 関数: `.2367-.2370` を実行する。

def main() -> None:
    """Execute the exact action-level operator-completion audit."""
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
        EIGSHIFT_AUDIT,
        ELL0_OPERATOR_AUDIT,
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
    eigshift_summary = sign_base.read_json(EIGSHIFT_AUDIT)["summary"]
    operator_audit = sign_base.read_json(ELL0_OPERATOR_AUDIT)
    operator_summary = operator_audit["summary"]
    operator_rows = row_map(operator_audit)

    offdiag_backbone_available = bool(
        operator_rows["offdiag_omega_k_available"]["value"]
    )
    cross_term_present = bool(operator_summary["phase1_exact_solver_cross_term_present"])
    constraint_elimination_present = bool(
        operator_summary["phase1_exact_solver_constraint_elimination_present"]
    )
    scalar_nonlinear_ansatz_only = bool(
        operator_summary["phase1_exact_solver_scalar_nonlinear_ansatz_only"]
    )
    ell0_coupling_collapses = bool(operator_summary["trial3_family_solver_ell0_coupling_collapses"])
    solver_fix_offdiag_explicit = operator_audit["evidence"].get("solver_fix_offdiag_hit") is not None

    cross_term_primary_completion_supported = bool(
        offdiag_backbone_available and solver_fix_offdiag_explicit and not cross_term_present
    )
    constraint_elimination_secondary_completion_supported = bool(
        cross_term_primary_completion_supported and not constraint_elimination_present
    )
    noncollapsed_ell0_closure_reserve_supported = bool(
        constraint_elimination_secondary_completion_supported
        and ell0_coupling_collapses
        and scalar_nonlinear_ansatz_only
    )
    exact_operator_completion_requires_pack_update = False
    completion_order_stable = bool(
        cross_term_primary_completion_supported
        and constraint_elimination_secondary_completion_supported
        and noncollapsed_ell0_closure_reserve_supported
    )

    rows = [
        sign_base.row(
            "inventory_ready",
            "pass",
            "exact operator completion inventory ready",
            1.0,
            "This branch starts only after the profile-fixed eigenvalue-shift candidate has been retained and the exact coupled theorem has been shown to fail because of operator incompleteness.",
        ),
        sign_base.row(
            "offdiag_backbone_available",
            "pass" if offdiag_backbone_available else "reject",
            "free-backbone off-diagonal omega-k mixing available",
            sign_base.truth(offdiag_backbone_available),
            "The public post-photon nontransverse backbone already freezes the off-diagonal mixing term, so completing that ingredient does not require a new physical surface.",
        ),
        sign_base.row(
            "solver_fix_offdiag_explicit",
            "pass" if solver_fix_offdiag_explicit else "reject",
            "solver-fix note explicitly points to the missing off-diagonal coupling",
            sign_base.truth(solver_fix_offdiag_explicit),
            "The local note already states that f_L = 0 is inconsistent once the mixed temporal/longitudinal field strength is kept, which makes cross-term realization the minimal exact-completion target.",
        ),
        sign_base.row(
            "cross_term_primary_completion_supported",
            "pass" if cross_term_primary_completion_supported else "reject",
            "cross-term realization supported as the primary operator-completion lane",
            sign_base.truth(cross_term_primary_completion_supported),
            "Because the backbone already contains the mixing while the current exact pilot omits it, cross-term realization is the smallest exact-completion move.",
        ),
        sign_base.row(
            "constraint_elimination_secondary_completion_supported",
            "pass" if constraint_elimination_secondary_completion_supported else "reject",
            "constraint elimination supported as the secondary operator-completion lane",
            sign_base.truth(constraint_elimination_secondary_completion_supported),
            "Constraint elimination acts on the coupled nontransverse operator produced by the primary completion, so it is downstream of cross-term realization.",
        ),
        sign_base.row(
            "noncollapsed_ell0_closure_reserve_supported",
            "pass" if noncollapsed_ell0_closure_reserve_supported else "reject",
            "noncollapsed ell=0 closure supported as reserve completion lane",
            sign_base.truth(noncollapsed_ell0_closure_reserve_supported),
            "The nonlinear ell=0 closure remains reserve because it depends on the linear coupled operator and its elimination being completed first.",
        ),
        sign_base.row(
            "exact_operator_completion_requires_pack_update",
            "reject",
            "substantive pack update required for exact operator completion now",
            sign_base.truth(exact_operator_completion_requires_pack_update),
            "The missing ingredients are already implied by the current backbone and notes, so the next route is still a completion audit rather than a pack update.",
        ),
        sign_base.row(
            "completion_order_stable",
            "pass" if completion_order_stable else "reject",
            "operator-completion ordering stable",
            sign_base.truth(completion_order_stable),
            "The current pack supports a stable completion ordering: cross term first, then constraint elimination, then nonlinear ell=0 closure.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "retained_scalar_residual_rel": float(prior_summary["retained_scalar_residual_rel"]),
        "delta_beta2_exact_profile_fixed": float(
            eigshift_summary["delta_beta2_exact_profile_fixed"]
        ),
        "operator_coefficient_proxy_from_max_ratio_sq": float(
            eigshift_summary["operator_coefficient_proxy_from_max_ratio_sq"]
        ),
        "operator_coefficient_proxy_from_tail_ratio_sq": float(
            eigshift_summary["operator_coefficient_proxy_from_tail_ratio_sq"]
        ),
        "offdiag_backbone_available": offdiag_backbone_available,
        "solver_fix_offdiag_explicit": solver_fix_offdiag_explicit,
        "cross_term_primary_completion_supported": cross_term_primary_completion_supported,
        "constraint_elimination_secondary_completion_supported": constraint_elimination_secondary_completion_supported,
        "noncollapsed_ell0_closure_reserve_supported": noncollapsed_ell0_closure_reserve_supported,
        "completion_order_stable": completion_order_stable,
        "selected_primary_completion_lane": "cross_term_realization",
        "selected_secondary_completion_lane": "constraint_elimination",
        "selected_reserve_completion_lane": "noncollapsed_ell0_closure",
        "exact_operator_completion_requires_pack_update": exact_operator_completion_requires_pack_update,
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": FOLLOWUP_ROUTE_NAME,
        "selected_followup_route_or_none": FOLLOWUP_ROUTE,
        "physical_reject_required": False,
    }

    declaration_payload = sign_base.payload(
        "8.7.56.2369",
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
                "eigshift_audit": sign_base.display_path(EIGSHIFT_AUDIT),
                "ell0_operator_audit": sign_base.display_path(ELL0_OPERATOR_AUDIT),
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
            "overall_status": "vector_qball_form_factor_exact_operator_completion_audited",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "formulas": build_formulae(),
            "hits": {
                "status_branch_hit": sign_base.hit(status_text, "8.7.56.2367"),
                "roadmap_branch_hit": sign_base.hit(roadmap_text, ".2367-.2370"),
                "current_problem_hit": sign_base.hit(current_problem_text, "operator prerequisites"),
                "current_status_hit": sign_base.hit(current_status_text, "operator prerequisites"),
                "unified_roadmap_hit": sign_base.hit(unified_text, ".2367-.2370"),
                "long_roadmap_hit": sign_base.hit(long_text, ".2367-.2370"),
                "part5_hit": sign_base.hit(part5_text, "exact action-level operator completion audit"),
            },
        },
    )
    declaration_paths = write_artifact("declaration_gate", declaration_payload)

    route_payload = {
        "generated_utc": sign_base.now_iso(),
        "phase": {
            "phase": 8,
            "step": "8.7.56.2370",
            "name": STEP_NAME + " route sync",
        },
        "inputs": declaration_paths,
        "rows": rows,
        "summary": summary,
        "decision": {
            "overall_status": "vector_qball_form_factor_exact_operator_completion_route_synced",
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

    print(f"[done] {STEP_TAG} exact operator completion audit completed")
    print(f"[info] declaration_gate_json={declaration_paths['json']}")
    print(f"[info] declaration_gate_csv={declaration_paths['csv']}")


if __name__ == "__main__":
    main()
