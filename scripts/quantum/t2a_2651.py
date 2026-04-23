#!/usr/bin/env python3
"""Generate 8.7.56.2651-.2654 updated-pack constraint-elimination realization gate artifacts."""

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
        "8.7.56.2647-2650",
        "updated_pack_phase1_literal_constraint_elimination_realization_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.2651-2654"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack phase-1 "
    "constraint-elimination realization gate"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_phase1_constraint_elimination_realization_gate",
    prefix="q",
)

PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_phase1_"
    "literal_constraint_elimination_realization_supported_ell0_closure_followup_gate"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_phase1_"
    "literal_constraint_elimination_realization_selected_ell0_closure_reserve_next"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_noncollapsed_ell0_"
    "closure_audit"
)
NEXT_ROUTE = "8.7.56.2655"
FOLLOWUP_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_noncollapsed_ell0_"
    "closure_gate_missing_action_refresh"
)
FOLLOWUP_ROUTE = "8.7.56.2659"


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
        "gate_a": "Gate A = updated-pack phase-1 literal constraint-elimination realization selected",
        "gate_b": "Gate B = updated-pack noncollapsed ell=0 closure promoted next",
        "gate_c": "Gate C = updated-pack substantive pack update still not required",
    }


# 関数: `.2651-.2654` を実行する。

def main() -> None:
    """Execute the updated-pack phase-1 constraint-elimination realization decision gate."""
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

    gate_a_updated_pack_phase1_literal_constraint_elimination_realization_selected = bool(
        prior_summary["updated_pack_phase1_literal_constraint_elimination_realization_supported_under_current_pack"]
    )
    gate_b_updated_pack_noncollapsed_ell0_closure_promoted_next = bool(
        prior_summary["updated_pack_noncollapsed_ell0_closure_followup_required"]
    )
    gate_c_updated_pack_substantive_pack_update_still_not_required = not bool(
        prior_summary["pack_update_required_now"]
    )
    updated_pack_phase1_backreaction_followup_still_open = bool(
        prior_summary["updated_pack_phase1_backreaction_followup_still_open"]
    )
    blind_vector_observable_gate_still_blocked = bool(
        prior_summary["blind_vector_observable_gate_still_blocked"]
    )
    trial3_family_primary_reuse_admissible_now = bool(
        prior_summary["trial3_family_primary_reuse_admissible_now"]
    )
    pack_update_required_now = bool(prior_summary["pack_update_required_now"])
    farther_hybrid_continuation_reopen_required_now = False

    rows = [
        sign_base.row(
            "gate_a_updated_pack_phase1_literal_constraint_elimination_realization_selected",
            "pass" if gate_a_updated_pack_phase1_literal_constraint_elimination_realization_selected else "reject",
            "Gate A updated-pack phase-1 literal constraint-elimination realization selected",
            sign_base.truth(gate_a_updated_pack_phase1_literal_constraint_elimination_realization_selected),
            "The reduced-state literalization target is now the official next completion move inside the retained updated pack.",
        ),
        sign_base.row(
            "gate_b_updated_pack_noncollapsed_ell0_closure_promoted_next",
            "pass" if gate_b_updated_pack_noncollapsed_ell0_closure_promoted_next else "reject",
            "Gate B updated-pack noncollapsed ell=0 closure promoted next",
            sign_base.truth(gate_b_updated_pack_noncollapsed_ell0_closure_promoted_next),
            "Once the literal reduced-state target is fixed, the next unresolved theorem is the noncollapsed ell=0 closure itself.",
        ),
        sign_base.row(
            "gate_c_updated_pack_substantive_pack_update_still_not_required",
            "pass" if gate_c_updated_pack_substantive_pack_update_still_not_required else "reject",
            "Gate C updated-pack substantive pack update still not required",
            sign_base.truth(gate_c_updated_pack_substantive_pack_update_still_not_required),
            "Current missing-action completion still advances within the retained updated pack and does not require reopening the external-input lane.",
        ),
        sign_base.row(
            "updated_pack_phase1_backreaction_followup_still_open",
            "pass" if updated_pack_phase1_backreaction_followup_still_open else "reject",
            "updated-pack phase-1 backreaction followup still open",
            sign_base.truth(updated_pack_phase1_backreaction_followup_still_open),
            "The future noncollapsed ell=0 closure audit still inherits the earlier finding that reciprocal backreaction remains unresolved.",
        ),
        sign_base.row(
            "blind_vector_observable_gate_still_blocked",
            "pass" if blind_vector_observable_gate_still_blocked else "reject",
            "blind-vector observable gate still blocked",
            sign_base.truth(blind_vector_observable_gate_still_blocked),
            "The blind-vector lane remains reserve-only while the operator route is still open.",
        ),
        sign_base.row(
            "trial3_family_primary_reuse_admissible_now",
            "pass" if trial3_family_primary_reuse_admissible_now else "reject",
            "trial-3 family primary reuse admissible now",
            sign_base.truth(trial3_family_primary_reuse_admissible_now),
            "The old trial-3 family stays reserve-only while noncollapsed ell=0 closure is still unresolved.",
        ),
        sign_base.row(
            "pack_update_required_now",
            "pass" if pack_update_required_now else "reject",
            "substantive pack update required now",
            sign_base.truth(pack_update_required_now),
            "Current route selection does not force a new pack update.",
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
        "gate_a_updated_pack_phase1_literal_constraint_elimination_realization_selected": gate_a_updated_pack_phase1_literal_constraint_elimination_realization_selected,
        "gate_b_updated_pack_noncollapsed_ell0_closure_promoted_next": gate_b_updated_pack_noncollapsed_ell0_closure_promoted_next,
        "gate_c_updated_pack_substantive_pack_update_still_not_required": gate_c_updated_pack_substantive_pack_update_still_not_required,
        "updated_pack_phase1_backreaction_followup_still_open": updated_pack_phase1_backreaction_followup_still_open,
        "blind_vector_observable_gate_still_blocked": blind_vector_observable_gate_still_blocked,
        "trial3_family_primary_reuse_admissible_now": trial3_family_primary_reuse_admissible_now,
        "pack_update_required_now": pack_update_required_now,
        "farther_hybrid_continuation_reopen_required_now": farther_hybrid_continuation_reopen_required_now,
        "selected_primary_completion_lane": "updated_pack_phase1_literal_constraint_elimination_realization",
        "selected_secondary_completion_lane": "updated_pack_noncollapsed_ell0_closure",
        "selected_reserve_completion_lane": "blind_vector_reserve",
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": FOLLOWUP_ROUTE_NAME,
        "selected_followup_route_or_none": FOLLOWUP_ROUTE,
        "physical_reject_required": False,
    }

    declaration_payload = sign_base.payload(
        "8.7.56.2653",
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
            "overall_status": "vector_qball_form_factor_updated_pack_phase1_constraint_elimination_realization_gate_declared",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "formulas": build_formulae(),
            "hits": {
                "status_branch_hit": sign_base.hit(status_text, "8.7.56.2651"),
                "roadmap_branch_hit": sign_base.hit(roadmap_text, ".2651-.2654"),
                "current_problem_hit": sign_base.hit(current_problem_text, "phase-1 constraint-elimination realization gate / ell0 reserve refresh"),
                "current_status_hit": sign_base.hit(current_status_text, "phase-1 constraint-elimination realization gate / ell0 reserve refresh"),
                "unified_roadmap_hit": sign_base.hit(unified_text, ".2651-.2654"),
                "long_roadmap_hit": sign_base.hit(long_text, ".2651-.2654"),
                "part5_hit": sign_base.hit(part5_text, "updated-pack phase-1 constraint-elimination realization gate"),
            },
        },
    )
    declaration_paths = write_artifact("declaration_gate", declaration_payload)

    route_payload = {
        "generated_utc": sign_base.now_iso(),
        "phase": {
            "phase": 8,
            "step": "8.7.56.2654",
            "name": STEP_NAME + " route sync",
        },
        "inputs": declaration_paths,
        "rows": rows,
        "summary": summary,
        "decision": {
            "overall_status": "vector_qball_form_factor_updated_pack_phase1_constraint_elimination_realization_gate_route_synced",
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

    print("[ok] updated-pack phase-1 constraint-elimination realization gate artifacts written")
    print(f"  declaration_gate: {declaration_paths['json']}")
    print(f"  route_sync: {route_paths['json']}")


if __name__ == "__main__":
    main()
