#!/usr/bin/env python3
"""Generate 8.7.56.2339-.2342 residual-origin decision-gate artifacts."""

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
        "8.7.56.2335-2338",
        "residual_origin_decomposition",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.2339-2342"
STEP_NAME = "Trial-2 numeric alpha vector Q-ball form-factor residual-origin decision gate"
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "residual_origin_decision_gate",
    prefix="q",
)

PRIOR_CLASS = "vector_qball_form_factor_residual_origin_missing_action_primary_observable_secondary_boundary_reserve_gate"
BRANCH_CLASS = "vector_qball_form_factor_residual_origin_missing_action_primary_observable_secondary_boundary_reserve_next"
NEXT_ROUTE_NAME = "trial2_numeric_alpha_vector_qball_form_factor_boundary_origin_falsification_audit"
NEXT_ROUTE = "8.7.56.2343"
FOLLOWUP_ROUTE_NAME = "trial2_numeric_alpha_vector_qball_form_factor_observable_definition_mismatch_audit"
FOLLOWUP_ROUTE = "8.7.56.2347"


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
    """Return formulas used in the residual-origin decision gate."""
    return {
        "primary_rule": "Primary lane = missing_action_level_term",
        "secondary_rule": "Secondary lane = observable_definition_mismatch",
        "reserve_rule": "Reserve lane = boundary_artifact",
        "support_rule": "hybrid_continuation = supporting evidence only",
    }


# 関数: `.2339-.2342` を実行する。

def main() -> None:
    """Execute the residual-origin decision gate."""
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

    inventory_ready = bool(prior_summary["missing_action_level_term_primary_supported"])
    primary_missing_action_level_term_selected = (
        prior_summary["primary_residual_lane"] == "missing_action_level_term"
    )
    secondary_observable_definition_selected = (
        prior_summary["secondary_residual_lane"] == "observable_definition_mismatch"
    )
    reserve_boundary_artifact_selected = (
        prior_summary["reserve_residual_lane"] == "boundary_artifact"
    )
    hybrid_supporting_evidence_retained = bool(prior_summary["hybrid_supporting_evidence_retained"])

    rows = [
        sign_base.row(
            "inventory_ready",
            "pass" if inventory_ready else "reject",
            "residual-origin decision inventory ready",
            sign_base.truth(inventory_ready),
            "The decision gate starts only after the decomposition audit has already fixed a machine-readable lane ordering.",
        ),
        sign_base.row(
            "primary_missing_action_level_term_selected",
            "pass" if primary_missing_action_level_term_selected else "reject",
            "primary residual lane selected as missing action-level term",
            sign_base.truth(primary_missing_action_level_term_selected),
            "The primary lane is the surviving explanation candidate after low-q observable self-consistency and boundary scale separation are accounted for.",
        ),
        sign_base.row(
            "secondary_observable_definition_selected",
            "pass" if secondary_observable_definition_selected else "reject",
            "secondary residual lane selected as observable-definition mismatch",
            sign_base.truth(secondary_observable_definition_selected),
            "Observable-definition mismatch remains open only as a secondary carry-over after exact low-q observable reproduction.",
        ),
        sign_base.row(
            "reserve_boundary_artifact_selected",
            "pass" if reserve_boundary_artifact_selected else "reject",
            "reserve residual lane selected as boundary artifact",
            sign_base.truth(reserve_boundary_artifact_selected),
            "Boundary structure is retained only as a reserve explanation because its scales are far separated from q_theory.",
        ),
        sign_base.row(
            "hybrid_supporting_evidence_retained",
            "pass" if hybrid_supporting_evidence_retained else "reject",
            "hybrid continuation retained as supporting evidence",
            sign_base.truth(hybrid_supporting_evidence_retained),
            "Farther hybrid continuation remains available only as a supporting lane if future origin discrimination needs extra q-range.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "retained_scalar_residual_rel": prior_summary["retained_scalar_residual_rel"],
        "primary_residual_lane": "missing_action_level_term",
        "secondary_residual_lane": "observable_definition_mismatch",
        "reserve_residual_lane": "boundary_artifact",
        "primary_missing_action_level_term_selected": primary_missing_action_level_term_selected,
        "secondary_observable_definition_selected": secondary_observable_definition_selected,
        "reserve_boundary_artifact_selected": reserve_boundary_artifact_selected,
        "hybrid_supporting_evidence_retained": hybrid_supporting_evidence_retained,
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": FOLLOWUP_ROUTE_NAME,
        "selected_followup_route_or_none": FOLLOWUP_ROUTE,
        "physical_reject_required": False,
    }

    declaration_payload = sign_base.payload(
        "8.7.56.2341",
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
            "overall_status": "vector_qball_form_factor_residual_origin_decision_gate_declared",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "formulas": build_formulae(),
            "hits": {
                "status_branch_hit": sign_base.hit(status_text, "8.7.56.2339"),
                "roadmap_branch_hit": sign_base.hit(roadmap_text, ".2339-.2342"),
                "current_problem_hit": sign_base.hit(current_problem_text, "missing action-level term"),
                "current_status_hit": sign_base.hit(current_status_text, "missing action-level term"),
                "unified_roadmap_hit": sign_base.hit(unified_text, ".2339-.2342"),
                "long_roadmap_hit": sign_base.hit(long_text, ".2339-.2342"),
                "part5_hit": sign_base.hit(part5_text, "2026-03-30 residual-origin update"),
            },
        },
    )
    declaration_paths = write_artifact("declaration_gate", declaration_payload)

    route_payload = {
        "generated_utc": sign_base.now_iso(),
        "phase": {
            "phase": 8,
            "step": "8.7.56.2342",
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
                "decision_gate_synced",
                "pass",
                "residual-origin decision gate synced",
                1.0,
                "The residual-origin mainline reset is only honest if the official lane ordering is already synchronized into the public machine-readable gate.",
            ),
        ],
        "summary": summary,
        "decision": {
            "overall_status": "vector_qball_form_factor_residual_origin_decision_gate_route_synced",
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
