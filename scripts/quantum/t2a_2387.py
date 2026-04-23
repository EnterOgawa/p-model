#!/usr/bin/env python3
"""Generate 8.7.56.2387-.2390 phase-1 literal cross-term gate artifacts."""

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
        "8.7.56.2383-2386",
        "phase1_literal_cross_term_realization_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.2387-2390"
STEP_NAME = "Trial-2 numeric alpha vector Q-ball form-factor phase-1 literal cross-term gate"
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "phase1_literal_cross_term_gate",
    prefix="q",
)

PRIOR_CLASS = "vector_qball_form_factor_residual_origin_missing_action_phase1_literal_cross_term_triangulated_realization_constraint_followup_gate"
BRANCH_CLASS = "vector_qball_form_factor_residual_origin_missing_action_phase1_literal_cross_term_realization_selected_constraint_elimination_secondary_ell0_closure_reserve_next"
NEXT_ROUTE_NAME = "trial2_numeric_alpha_vector_qball_form_factor_phase1_exact_solver_constraint_elimination_audit"
NEXT_ROUTE = "8.7.56.2391"
FOLLOWUP_ROUTE_NAME = "trial2_numeric_alpha_vector_qball_form_factor_phase1_constraint_elimination_gate_ell0_reserve_refresh"
FOLLOWUP_ROUTE = "8.7.56.2395"


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


# 関数: decision gate で使う式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used in the decision gate."""
    return {
        "gate_a": "Gate A = phase-1 literal cross-term realization selected",
        "gate_b": "Gate B = phase-1 constraint-elimination audit promoted next",
        "gate_c": "Gate C = noncollapsed ell=0 closure reserve retained",
    }


# 関数: `.2387-.2390` を実行する。

def main() -> None:
    """Execute the phase-1 literal cross-term decision gate."""
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

    gate_a = bool(prior_summary["phase1_literal_cross_term_realization_supported_under_current_pack"])
    gate_b = bool(prior_summary["constraint_elimination_followup_required"])
    gate_c = True
    perturbative_numeric_reuse_as_primary_admissible_now = False
    pack_update_required_now = bool(prior_summary["pack_update_required_now"])

    rows = [
        sign_base.row(
            "gate_a_phase1_literal_cross_term_realization_selected",
            "pass" if gate_a else "reject",
            "Gate A phase-1 literal cross-term realization selected",
            sign_base.truth(gate_a),
            "The first exact-completion move is now the phase-1 literal source-term realization, not another generic completion search.",
        ),
        sign_base.row(
            "gate_b_phase1_constraint_elimination_promoted_next",
            "pass" if gate_b else "reject",
            "Gate B phase-1 constraint-elimination audit promoted next",
            sign_base.truth(gate_b),
            "Because the literal realization does not yet close the coupled operator, constraint elimination becomes the next official mainline move.",
        ),
        sign_base.row(
            "gate_c_noncollapsed_ell0_closure_reserve_retained",
            "pass" if gate_c else "reject",
            "Gate C noncollapsed ell=0 closure reserve retained",
            sign_base.truth(gate_c),
            "The nonlinear ell=0 closure remains reserve because it still depends on the completed linear coupled operator and its elimination.",
        ),
        sign_base.row(
            "perturbative_numeric_reuse_as_primary_admissible_now",
            "pass" if perturbative_numeric_reuse_as_primary_admissible_now else "reject",
            "legacy perturbative numeric reuse admissible now",
            sign_base.truth(perturbative_numeric_reuse_as_primary_admissible_now),
            "The old case-gamma driven solution stays evidence-only and is not reopened as the primary exact realization.",
        ),
        sign_base.row(
            "pack_update_required_now",
            "pass" if pack_update_required_now else "reject",
            "substantive pack update required now",
            sign_base.truth(pack_update_required_now),
            "The route still advances inside the retained pack through operator completion rather than a new external physics surface.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "retained_scalar_residual_rel": float(prior_summary["retained_scalar_residual_rel"]),
        "gate_a_phase1_literal_cross_term_realization_selected": gate_a,
        "gate_b_phase1_constraint_elimination_promoted_next": gate_b,
        "gate_c_noncollapsed_ell0_closure_reserve_retained": gate_c,
        "perturbative_numeric_reuse_as_primary_admissible_now": perturbative_numeric_reuse_as_primary_admissible_now,
        "pack_update_required_now": pack_update_required_now,
        "selected_primary_completion_lane": "phase1_literal_cross_term_realization",
        "selected_secondary_completion_lane": "phase1_constraint_elimination",
        "selected_reserve_completion_lane": "noncollapsed_ell0_closure",
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": FOLLOWUP_ROUTE_NAME,
        "selected_followup_route_or_none": FOLLOWUP_ROUTE,
        "physical_reject_required": False,
    }

    declaration_payload = sign_base.payload(
        "8.7.56.2389",
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
            "overall_status": "vector_qball_form_factor_phase1_literal_cross_term_gate_declared",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "formulas": build_formulae(),
            "hits": {
                "status_branch_hit": sign_base.hit(status_text, "8.7.56.2387"),
                "roadmap_branch_hit": sign_base.hit(roadmap_text, ".2387-.2390"),
                "current_problem_hit": sign_base.hit(current_problem_text, "phase-1 literal cross-term gate"),
                "current_status_hit": sign_base.hit(current_status_text, "constraint elimination"),
                "unified_roadmap_hit": sign_base.hit(unified_text, ".2387-.2390"),
                "long_roadmap_hit": sign_base.hit(long_text, ".2387-.2390"),
                "part5_hit": sign_base.hit(part5_text, "phase-1 literal cross-term gate / constraint-elimination reserve refresh"),
            },
        },
    )
    declaration_paths = write_artifact("declaration_gate", declaration_payload)

    route_payload = {
        "generated_utc": sign_base.now_iso(),
        "phase": {
            "phase": 8,
            "step": "8.7.56.2390",
            "name": STEP_NAME + " route sync",
        },
        "inputs": declaration_paths,
        "rows": rows,
        "summary": summary,
        "decision": {
            "overall_status": "vector_qball_form_factor_phase1_literal_cross_term_gate_route_synced",
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

    print(f"[done] {STEP_TAG} phase-1 literal cross-term gate completed")
    print(f"[info] declaration_gate_json={declaration_paths['json']}")
    print(f"[info] declaration_gate_csv={declaration_paths['csv']}")


if __name__ == "__main__":
    main()
