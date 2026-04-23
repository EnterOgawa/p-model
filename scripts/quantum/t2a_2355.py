#!/usr/bin/env python3
"""Generate 8.7.56.2355-.2358 residual-origin synthesis artifacts."""

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
        "8.7.56.2351-2354",
        "missing_action_delta_beta1_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.2355-2358"
STEP_NAME = "Trial-2 numeric alpha vector Q-ball form-factor residual-origin synthesis"
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "residual_origin_synthesis",
    prefix="q",
)

PRIOR_CLASS = "vector_qball_form_factor_residual_origin_missing_action_profile_fixed_delta_beta1_candidate_audit_gate"
BRANCH_CLASS = "vector_qball_form_factor_residual_origin_missing_action_profile_fixed_eigenvalue_shift_candidate_selected_exact_coupled_theorem_next"
NEXT_ROUTE_NAME = "trial2_numeric_alpha_vector_qball_form_factor_exact_coupled_eigenvalue_shift_theorem_audit"
NEXT_ROUTE = "8.7.56.2359"
FOLLOWUP_ROUTE_NAME = "trial2_numeric_alpha_vector_qball_form_factor_profile_fixed_eigenvalue_shift_decision_gate_hybrid_reserve_refresh"
FOLLOWUP_ROUTE = "8.7.56.2363"


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


# 関数: synthesis で使う式を返す。
def build_formulae() -> dict[str, str]:
    """Return formulas used in the residual-origin synthesis."""
    return {
        "lane_order": "primary = missing_action_level_term, secondary = observable_definition_mismatch, reserve = boundary_artifact",
        "first_shot": "first_shot = profile_fixed_eigenvalue_shift_delta_beta1_candidate",
        "reserve_rule": "hybrid continuation reopens only if the residual-origin synthesis still needs extra q-range after the coupled-eigenvalue theorem audit",
    }


# 関数: `.2355-.2358` を実行する。
def main() -> None:
    """Execute the residual-origin synthesis branch."""
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

    inventory_ready = bool(prior_summary["profile_fixed_eigenvalue_shift_candidate_admissible"])
    boundary_primary_falsified = True
    observable_secondary_carryover = True
    missing_action_primary_retained = True
    first_shot_selected = True
    exact_coupled_theorem_available = bool(
        prior_summary["exact_coupled_eigenvalue_shift_theorem_available"]
    )
    hybrid_supporting_evidence_reopen_required = False

    rows = [
        sign_base.row(
            "inventory_ready",
            "pass" if inventory_ready else "reject",
            "residual-origin synthesis inventory ready",
            sign_base.truth(inventory_ready),
            "Synthesis starts only after the missing-action first shot has been reduced to a machine-readable admissible candidate.",
        ),
        sign_base.row(
            "boundary_primary_falsified",
            "pass",
            "boundary artifact falsified as primary lane",
            sign_base.truth(boundary_primary_falsified),
            "Boundary scales remain far outside the retained low-q residual point and stay closed as the primary explanation.",
        ),
        sign_base.row(
            "observable_secondary_carryover",
            "pass",
            "observable-definition mismatch retained only as a secondary carry-over",
            sign_base.truth(observable_secondary_carryover),
            "The low-q observable family is internally exact, so observable-definition mismatch remains only a secondary carry-over.",
        ),
        sign_base.row(
            "missing_action_primary_retained",
            "pass",
            "missing action-level term retained as primary lane",
            sign_base.truth(missing_action_primary_retained),
            "After cutting the other primary lanes, the residual-origin mainline remains anchored in the missing-action lane.",
        ),
        sign_base.row(
            "profile_fixed_eigenvalue_shift_first_shot_selected",
            "pass" if first_shot_selected else "reject",
            "profile-fixed eigenvalue-shift delta-beta1 first shot selected",
            sign_base.truth(first_shot_selected),
            "The first concrete next computation is no longer a generic omission audit but the coupled-eigenvalue theorem audit for the profile-fixed delta-beta1 candidate.",
        ),
        sign_base.row(
            "exact_coupled_eigenvalue_shift_theorem_available",
            "reject" if not exact_coupled_theorem_available else "pass",
            "exact coupled eigenvalue-shift theorem already available",
            sign_base.truth(exact_coupled_theorem_available),
            "The theorem is still open, so the next branch must test whether the coupled operator can actually derive the required delta-beta1 instead of only showing that it would be sufficient.",
        ),
        sign_base.row(
            "hybrid_supporting_evidence_reopen_required",
            "reject",
            "hybrid supporting-evidence continuation reopen required now",
            sign_base.truth(hybrid_supporting_evidence_reopen_required),
            "Extra q-range is not required before the exact coupled eigenvalue-shift theorem is tested, so hybrid continuation stays in reserve.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "retained_scalar_residual_rel": float(prior_summary["retained_scalar_residual_rel"]),
        "boundary_primary_falsified": boundary_primary_falsified,
        "observable_secondary_carryover": observable_secondary_carryover,
        "missing_action_primary_retained": missing_action_primary_retained,
        "profile_fixed_eigenvalue_shift_candidate_admissible": bool(
            prior_summary["profile_fixed_eigenvalue_shift_candidate_admissible"]
        ),
        "delta_beta_exact_profile_fixed": float(prior_summary["delta_beta_exact_profile_fixed"]),
        "delta_beta2_exact_profile_fixed": float(prior_summary["delta_beta2_exact_profile_fixed"]),
        "required_delta_beta2_fraction_of_beta_gap": float(
            prior_summary["required_delta_beta2_fraction_of_beta_gap"]
        ),
        "required_delta_beta2_vs_ceiling_sq": float(
            prior_summary["required_delta_beta2_vs_ceiling_sq"]
        ),
        "profile_fixed_eigenvalue_shift_first_shot_selected": first_shot_selected,
        "exact_coupled_eigenvalue_shift_theorem_available": exact_coupled_theorem_available,
        "hybrid_supporting_evidence_reopen_required": hybrid_supporting_evidence_reopen_required,
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": FOLLOWUP_ROUTE_NAME,
        "selected_followup_route_or_none": FOLLOWUP_ROUTE,
        "physical_reject_required": False,
    }

    declaration_payload = sign_base.payload(
        "8.7.56.2357",
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
            "overall_status": "vector_qball_form_factor_residual_origin_synthesis_declared",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "formulas": build_formulae(),
            "hits": {
                "status_branch_hit": sign_base.hit(status_text, "8.7.56.2355"),
                "roadmap_branch_hit": sign_base.hit(roadmap_text, ".2355-.2358"),
                "current_problem_hit": sign_base.hit(current_problem_text, "missing action-level term"),
                "current_status_hit": sign_base.hit(current_status_text, "missing action-level term"),
                "unified_roadmap_hit": sign_base.hit(unified_text, ".2355-.2358"),
                "long_roadmap_hit": sign_base.hit(long_text, ".2355-.2358"),
                "part5_hit": sign_base.hit(part5_text, "2026-03-30 update"),
            },
        },
    )
    declaration_paths = write_artifact("declaration_gate", declaration_payload)

    route_payload = {
        "generated_utc": sign_base.now_iso(),
        "phase": {
            "phase": 8,
            "step": "8.7.56.2358",
            "name": STEP_NAME + " route sync",
        },
        "inputs": {
            "source_files": {
                "status": sign_base.display_path(STATUS),
                "roadmap": sign_base.display_path(ROADMAP),
                "current_problem": sign_base.display_path(CURRENT_PROBLEM),
                "current_status": sign_base.display_path(CURRENT_STATUS),
                "unified_roadmap": sign_base.display_path(UNIFIED_ROADMAP),
                "long_roadmap": sign_base.display_path(LONG_ROADMAP),
                "part5": sign_base.display_path(PART5),
                "declaration_gate": declaration_paths["json"],
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
        "rows": [
            sign_base.row(
                "residual_origin_synthesis_synced",
                "pass",
                "residual-origin synthesis synced",
                1.0,
                "The residual-origin route reset is only complete when the missing-action first shot, the secondary/reserve carry-overs, and the hybrid reserve policy are written into public machine-readable artifacts.",
            ),
        ],
        "summary": summary,
        "decision": {
            "overall_status": "vector_qball_form_factor_residual_origin_synthesis_route_synced",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        "evidence": declaration_payload["evidence"],
    }
    route_paths = write_artifact("route_sync", route_payload)
    print("[write] declaration:", declaration_paths["json"])
    print("[write] route:", route_paths["json"])


if __name__ == "__main__":
    main()
