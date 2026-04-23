#!/usr/bin/env python3
"""Generate 8.7.56.2643-.2646 updated-pack phase-1 constraint-elimination gate artifacts."""

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
        "8.7.56.2639-2642",
        "updated_pack_phase1_constraint_elimination_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
PRIOR_LITERAL_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.2635-2638",
        "updated_pack_phase1_literal_cross_term_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.2643-2646"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack phase-1 "
    "constraint-elimination gate"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_phase1_constraint_elimination_gate",
    prefix="q",
)

PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_phase1_"
    "constraint_elimination_supported_literal_realization_followup_gate"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_phase1_"
    "constraint_elimination_selected_literal_realization_primary_ell0_closure_"
    "reserve_next"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_phase1_literal_"
    "constraint_elimination_realization_audit"
)
NEXT_ROUTE = "8.7.56.2647"
FOLLOWUP_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_phase1_constraint_"
    "elimination_realization_gate_ell0_refresh"
)
FOLLOWUP_ROUTE = "8.7.56.2651"


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


# 関数: gate で使う式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used in the decision gate."""
    return {
        "gate_a": "Gate A = updated-pack phase-1 exact-solver constraint elimination selected",
        "gate_b": "Gate B = updated-pack phase-1 literal constraint-elimination realization promoted next",
        "gate_c": "Gate C = updated-pack noncollapsed ell=0 closure reserve retained",
    }


# 関数: `.2643-.2646` を実行する。

def main() -> None:
    """Execute the updated-pack phase-1 constraint-elimination decision gate."""
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
        PRIOR_LITERAL_GATE,
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
    prior_literal_summary = sign_base.read_json(PRIOR_LITERAL_GATE)["summary"]

    gate_a_updated_pack_phase1_constraint_elimination_selected = bool(
        prior_summary["updated_pack_phase1_constraint_elimination_supported_under_current_pack"]
    )
    gate_b_updated_pack_phase1_literal_constraint_elimination_realization_promoted_next = bool(
        prior_summary["updated_pack_phase1_constraint_elimination_requires_literal_realization_in_exact_solver"]
    )
    gate_c_updated_pack_noncollapsed_ell0_closure_reserve_retained = bool(
        prior_summary["updated_pack_noncollapsed_ell0_closure_followup_required"]
    )
    exact_source_theorem_no_go_verdict_fixed = bool(
        prior_literal_summary["exact_source_theorem_no_go_verdict_fixed"]
    )
    exact_ell0_action_level_operator_available_now = False
    blind_vector_direct_computation_primary_admissible_now = False
    trial3_family_primary_reuse_admissible_now = bool(
        prior_summary["trial3_family_primary_reuse_admissible_now"]
    )
    pack_update_required_now = bool(prior_summary["pack_update_required_now"])
    farther_hybrid_continuation_reopen_required_now = False

    rows = [
        sign_base.row(
            "gate_a_updated_pack_phase1_constraint_elimination_selected",
            "pass" if gate_a_updated_pack_phase1_constraint_elimination_selected else "reject",
            "Gate A updated-pack phase-1 exact-solver constraint elimination selected",
            sign_base.truth(gate_a_updated_pack_phase1_constraint_elimination_selected),
            "The primary operator lane now advances through exact-solver constraint elimination rather than another generic operator search.",
        ),
        sign_base.row(
            "gate_b_updated_pack_phase1_literal_constraint_elimination_realization_promoted_next",
            "pass" if gate_b_updated_pack_phase1_literal_constraint_elimination_realization_promoted_next else "reject",
            "Gate B updated-pack phase-1 literal constraint-elimination realization promoted next",
            sign_base.truth(gate_b_updated_pack_phase1_literal_constraint_elimination_realization_promoted_next),
            "Because the pack supports elimination but the exact solver still keeps the unreduced state, the next honest move is a literal reduced-state realization audit.",
        ),
        sign_base.row(
            "gate_c_updated_pack_noncollapsed_ell0_closure_reserve_retained",
            "pass" if gate_c_updated_pack_noncollapsed_ell0_closure_reserve_retained else "reject",
            "Gate C updated-pack noncollapsed ell=0 closure reserve retained",
            sign_base.truth(gate_c_updated_pack_noncollapsed_ell0_closure_reserve_retained),
            "The nonlinear noncollapsed ell=0 closure remains reserve until the reduced-state linear operator is explicitly realized.",
        ),
        sign_base.row(
            "exact_source_theorem_no_go_verdict_fixed",
            "pass" if exact_source_theorem_no_go_verdict_fixed else "reject",
            "exact source-theorem no-go verdict fixed",
            sign_base.truth(exact_source_theorem_no_go_verdict_fixed),
            "The current-pack no-go theorem remains fixed while the exact ell=0 operator lane advances.",
        ),
        sign_base.row(
            "exact_ell0_action_level_operator_available_now",
            "pass" if exact_ell0_action_level_operator_available_now else "reject",
            "exact ell=0 action-level operator available now",
            sign_base.truth(exact_ell0_action_level_operator_available_now),
            "Constraint elimination promotion still does not close the exact ell=0 operator by itself.",
        ),
        sign_base.row(
            "blind_vector_direct_computation_primary_admissible_now",
            "pass" if blind_vector_direct_computation_primary_admissible_now else "reject",
            "blind-vector direct computation primary admissible now",
            sign_base.truth(blind_vector_direct_computation_primary_admissible_now),
            "Blind-vector direct computation remains reserve-only because the exact coupled operator is still not closed.",
        ),
        sign_base.row(
            "trial3_family_primary_reuse_admissible_now",
            "pass" if trial3_family_primary_reuse_admissible_now else "reject",
            "trial-3 family primary reuse admissible now",
            sign_base.truth(trial3_family_primary_reuse_admissible_now),
            "The trial-3 family stays reserve-only while the exact phase-1 reduced-state operator is still being completed.",
        ),
        sign_base.row(
            "pack_update_required_now",
            "pass" if pack_update_required_now else "reject",
            "substantive pack update required now",
            sign_base.truth(pack_update_required_now),
            "Current operator completion still advances inside the retained updated pack rather than via a new external input.",
        ),
        sign_base.row(
            "farther_hybrid_continuation_reopen_required_now",
            "pass" if farther_hybrid_continuation_reopen_required_now else "reject",
            "farther hybrid continuation reopen required now",
            sign_base.truth(farther_hybrid_continuation_reopen_required_now),
            "The blocker remains operator-side, so extra q-range evidence stays closed reserve-only.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "retained_scalar_residual_rel": float(prior_summary["retained_scalar_residual_rel"]),
        "gate_a_updated_pack_phase1_constraint_elimination_selected": gate_a_updated_pack_phase1_constraint_elimination_selected,
        "gate_b_updated_pack_phase1_literal_constraint_elimination_realization_promoted_next": gate_b_updated_pack_phase1_literal_constraint_elimination_realization_promoted_next,
        "gate_c_updated_pack_noncollapsed_ell0_closure_reserve_retained": gate_c_updated_pack_noncollapsed_ell0_closure_reserve_retained,
        "exact_source_theorem_no_go_verdict_fixed": exact_source_theorem_no_go_verdict_fixed,
        "exact_ell0_action_level_operator_available_now": exact_ell0_action_level_operator_available_now,
        "blind_vector_direct_computation_primary_admissible_now": blind_vector_direct_computation_primary_admissible_now,
        "trial3_family_primary_reuse_admissible_now": trial3_family_primary_reuse_admissible_now,
        "pack_update_required_now": pack_update_required_now,
        "farther_hybrid_continuation_reopen_required_now": farther_hybrid_continuation_reopen_required_now,
        "selected_primary_completion_lane": "updated_pack_phase1_constraint_elimination",
        "selected_secondary_completion_lane": "updated_pack_phase1_literal_constraint_elimination_realization",
        "selected_reserve_completion_lane": "updated_pack_noncollapsed_ell0_closure",
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": FOLLOWUP_ROUTE_NAME,
        "selected_followup_route_or_none": FOLLOWUP_ROUTE,
        "physical_reject_required": False,
    }

    declaration_payload = sign_base.payload(
        "8.7.56.2645",
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
                "prior_audit": sign_base.display_path(PRIOR_GATE),
                "prior_literal_gate": sign_base.display_path(PRIOR_LITERAL_GATE),
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
            "overall_status": "vector_qball_form_factor_updated_pack_phase1_constraint_elimination_gate_declared",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "formulas": build_formulae(),
            "hits": {
                "status_branch_hit": sign_base.hit(status_text, "8.7.56.2643"),
                "roadmap_branch_hit": sign_base.hit(roadmap_text, ".2643-.2646"),
                "current_problem_hit": sign_base.hit(current_problem_text, "phase-1 constraint-elimination gate / ell0 reserve refresh"),
                "current_status_hit": sign_base.hit(current_status_text, "phase-1 constraint-elimination gate / ell0 reserve refresh"),
                "unified_roadmap_hit": sign_base.hit(unified_text, ".2643-.2646"),
                "long_roadmap_hit": sign_base.hit(long_text, ".2643-.2646"),
                "part5_hit": sign_base.hit(part5_text, "updated-pack phase-1 constraint-elimination gate"),
            },
        },
    )
    declaration_paths = write_artifact("declaration_gate", declaration_payload)

    route_payload = {
        "generated_utc": sign_base.now_iso(),
        "phase": {
            "phase": 8,
            "step": "8.7.56.2646",
            "name": STEP_NAME + " route sync",
        },
        "inputs": declaration_paths,
        "rows": rows,
        "summary": summary,
        "decision": {
            "overall_status": "vector_qball_form_factor_updated_pack_phase1_constraint_elimination_gate_route_synced",
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

    print("[ok] updated-pack phase-1 constraint-elimination gate artifacts written")
    print(f"  declaration_gate: {declaration_paths['json']}")
    print(f"  route_sync: {route_paths['json']}")


if __name__ == "__main__":
    main()
