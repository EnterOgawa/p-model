#!/usr/bin/env python3
"""Generate 8.7.56.2371-.2374 exact operator-completion gate artifacts."""

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
        "8.7.56.2367-2370",
        "exact_operator_completion_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.2371-2374"
STEP_NAME = "Trial-2 numeric alpha vector Q-ball form-factor exact operator completion decision gate"
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "exact_operator_completion_gate",
    prefix="q",
)

PRIOR_CLASS = "vector_qball_form_factor_residual_origin_missing_action_operator_completion_cross_term_primary_constraint_secondary_gate"
BRANCH_CLASS = "vector_qball_form_factor_residual_origin_missing_action_cross_term_primary_constraint_secondary_ell0_closure_reserve_next"
NEXT_ROUTE_NAME = "trial2_numeric_alpha_vector_qball_form_factor_exact_action_level_cross_term_completion_audit"
NEXT_ROUTE = "8.7.56.2375"
FOLLOWUP_ROUTE_NAME = "trial2_numeric_alpha_vector_qball_form_factor_cross_term_completion_decision_gate_constraint_elimination_refresh"
FOLLOWUP_ROUTE = "8.7.56.2379"


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
        "gate_a": "Gate A = cross-term completion first shot selected",
        "gate_b": "Gate B = constraint-elimination followup retained",
        "gate_c": "Gate C = noncollapsed ell=0 closure reserve retained",
    }


# 関数: `.2371-.2374` を実行する。

def main() -> None:
    """Execute the exact operator completion decision gate."""
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

    gate_a = bool(prior_summary["cross_term_primary_completion_supported"])
    gate_b = bool(prior_summary["constraint_elimination_secondary_completion_supported"])
    gate_c = bool(prior_summary["noncollapsed_ell0_closure_reserve_supported"])
    pack_update_required = bool(prior_summary["exact_operator_completion_requires_pack_update"])
    hybrid_supporting_evidence_reopen_required = False

    rows = [
        sign_base.row(
            "gate_a_cross_term_completion_selected",
            "pass" if gate_a else "reject",
            "Gate A cross-term completion first shot selected",
            sign_base.truth(gate_a),
            "Cross-term realization is selected because it is already frozen in the public backbone yet still absent in the current exact implementation.",
        ),
        sign_base.row(
            "gate_b_constraint_elimination_retained",
            "pass" if gate_b else "reject",
            "Gate B constraint-elimination followup retained",
            sign_base.truth(gate_b),
            "Constraint elimination remains the next completion layer after cross-term realization has been attempted.",
        ),
        sign_base.row(
            "gate_c_ell0_closure_reserve_retained",
            "pass" if gate_c else "reject",
            "Gate C noncollapsed ell=0 closure reserve retained",
            sign_base.truth(gate_c),
            "The nonlinear ell=0 closure remains reserve because it depends on the completed linear coupled operator and its elimination.",
        ),
        sign_base.row(
            "pack_update_required_now",
            "pass" if pack_update_required else "reject",
            "substantive pack update required now",
            sign_base.truth(pack_update_required),
            "The current completion ordering still lives inside the existing pack, so no pack update is required before the cross-term first shot.",
        ),
        sign_base.row(
            "hybrid_supporting_evidence_reopen_required_now",
            "pass" if hybrid_supporting_evidence_reopen_required else "reject",
            "hybrid supporting-evidence reopen required now",
            sign_base.truth(hybrid_supporting_evidence_reopen_required),
            "The mainline stays inside operator completion and does not need farther-q support before the cross-term lane is tested.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "retained_scalar_residual_rel": 0.019262702271264597,
        "gate_a_cross_term_completion_selected": gate_a,
        "gate_b_constraint_elimination_retained": gate_b,
        "gate_c_ell0_closure_reserve_retained": gate_c,
        "pack_update_required_now": pack_update_required,
        "hybrid_supporting_evidence_reopen_required": hybrid_supporting_evidence_reopen_required,
        "selected_primary_completion_lane": "cross_term_realization",
        "selected_secondary_completion_lane": "constraint_elimination",
        "selected_reserve_completion_lane": "noncollapsed_ell0_closure",
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": FOLLOWUP_ROUTE_NAME,
        "selected_followup_route_or_none": FOLLOWUP_ROUTE,
        "physical_reject_required": False,
    }

    declaration_payload = sign_base.payload(
        "8.7.56.2373",
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
            "overall_status": "vector_qball_form_factor_exact_operator_completion_gate_declared",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "formulas": build_formulae(),
            "hits": {
                "status_branch_hit": sign_base.hit(status_text, "8.7.56.2371"),
                "roadmap_branch_hit": sign_base.hit(roadmap_text, ".2371-.2374"),
                "current_problem_hit": sign_base.hit(current_problem_text, "exact action-level operator completion audit"),
                "current_status_hit": sign_base.hit(current_status_text, "exact action-level operator completion audit"),
                "unified_roadmap_hit": sign_base.hit(unified_text, ".2371-.2374"),
                "long_roadmap_hit": sign_base.hit(long_text, ".2371-.2374"),
                "part5_hit": sign_base.hit(part5_text, "exact action-level operator completion audit"),
            },
        },
    )
    declaration_paths = write_artifact("declaration_gate", declaration_payload)

    route_payload = {
        "generated_utc": sign_base.now_iso(),
        "phase": {
            "phase": 8,
            "step": "8.7.56.2374",
            "name": STEP_NAME + " route sync",
        },
        "inputs": declaration_paths,
        "rows": rows,
        "summary": summary,
        "decision": {
            "overall_status": "vector_qball_form_factor_exact_operator_completion_gate_route_synced",
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

    print(f"[done] {STEP_TAG} exact operator completion decision gate completed")
    print(f"[info] declaration_gate_json={declaration_paths['json']}")
    print(f"[info] declaration_gate_csv={declaration_paths['csv']}")


if __name__ == "__main__":
    main()
