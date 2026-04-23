#!/usr/bin/env python3
"""Generate 8.7.56.2683-.2686 updated-pack trial3 ell=0 reserve-gate artifacts."""

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
        "8.7.56.2679-2682",
        "updated_pack_trial3_ell0_closure_reserve_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.2683-2686"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack trial3 "
    "ell=0 reserve gate"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_trial3_ell0_reserve_gate",
    prefix="q",
)

PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_trial3_"
    "ell0_reserve_scalarlike_inventory_only_pack_update_gate"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_trial3_"
    "ell0_reserve_exhausted_pack_update_next"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_substantive_pack_"
    "update_audit"
)
NEXT_ROUTE = "8.7.56.2687"
FOLLOWUP_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_substantive_pack_"
    "update_gate_hybrid_reserve_refresh"
)
FOLLOWUP_ROUTE = "8.7.56.2691"


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
    """Return formulas used in the updated-pack decision gate."""
    return {
        "gate_a": "Gate A = updated-pack trial3 ell=0 reserve closes the missing-action blocker now",
        "gate_b": "Gate B = updated-pack substantive pack update promoted next",
        "gate_c": "Gate C = farther hybrid continuation reopen required now",
    }


# 関数: `.2683-.2686` を実行する。

def main() -> None:
    """Execute the updated-pack trial3 ell=0 reserve decision gate."""
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

    gate_a_updated_pack_trial3_ell0_reserve_closes_missing_action_blocker_now = bool(
        prior_summary["updated_pack_trial3_ell0_reserve_closes_current_missing_action_blocker_now"]
    )
    gate_b_updated_pack_substantive_pack_update_promoted_next = bool(
        (not gate_a_updated_pack_trial3_ell0_reserve_closes_missing_action_blocker_now)
        and prior_summary["updated_pack_substantive_pack_update_followup_supported"]
    )
    gate_c_farther_hybrid_continuation_reopen_required_now = False
    trial3_family_primary_reuse_admissible_now = bool(
        prior_summary["trial3_family_ell0_primary_fix_available"]
    )
    blind_vector_observable_gate_still_blocked = bool(
        prior_summary["blind_vector_observable_gate_still_blocked"]
    )
    pack_update_required_now = bool(gate_b_updated_pack_substantive_pack_update_promoted_next)
    hybrid_supporting_evidence_reopen_required = gate_c_farther_hybrid_continuation_reopen_required_now

    rows = [
        sign_base.row(
            "gate_a_updated_pack_trial3_ell0_reserve_closes_missing_action_blocker_now",
            "pass" if gate_a_updated_pack_trial3_ell0_reserve_closes_missing_action_blocker_now else "reject",
            "Gate A updated-pack trial3 ell=0 reserve closes missing-action blocker now",
            sign_base.truth(gate_a_updated_pack_trial3_ell0_reserve_closes_missing_action_blocker_now),
            "The archived trial3 ell=0 reserve would close here only if it supplied a literal coupled closure fix, which it does not.",
        ),
        sign_base.row(
            "gate_b_updated_pack_substantive_pack_update_promoted_next",
            "pass" if gate_b_updated_pack_substantive_pack_update_promoted_next else "reject",
            "Gate B updated-pack substantive pack update promoted next",
            sign_base.truth(gate_b_updated_pack_substantive_pack_update_promoted_next),
            "Once the updated-pack trial3 ell=0 reserve is fixed as support-only, the next honest move is an updated-pack substantive pack-update audit.",
        ),
        sign_base.row(
            "gate_c_farther_hybrid_continuation_reopen_required_now",
            "pass" if gate_c_farther_hybrid_continuation_reopen_required_now else "reject",
            "Gate C farther hybrid continuation reopen required now",
            sign_base.truth(gate_c_farther_hybrid_continuation_reopen_required_now),
            "Extra q-range evidence is still unnecessary, so the farther hybrid continuation remains closed as reserve support.",
        ),
        sign_base.row(
            "trial3_family_primary_reuse_admissible_now",
            "pass" if trial3_family_primary_reuse_admissible_now else "reject",
            "trial3 family primary reuse admissible now",
            sign_base.truth(trial3_family_primary_reuse_admissible_now),
            "The old trial3 family stays reserve-only and is not restored as the primary missing-action fix.",
        ),
        sign_base.row(
            "blind_vector_observable_gate_still_blocked",
            "pass" if blind_vector_observable_gate_still_blocked else "reject",
            "blind-vector observable gate still blocked",
            sign_base.truth(blind_vector_observable_gate_still_blocked),
            "The blind-vector lane remains reserve-only while the action-level operator route stays open.",
        ),
        sign_base.row(
            "pack_update_required_now",
            "pass" if pack_update_required_now else "reject",
            "updated-pack substantive pack update required now",
            sign_base.truth(pack_update_required_now),
            "The updated-pack trial3 reserve audit is now exhausted, so a substantive pack update becomes the next official mainline.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "retained_scalar_residual_rel": float(prior_summary["retained_scalar_residual_rel"]),
        "gate_a_updated_pack_trial3_ell0_reserve_closes_missing_action_blocker_now": gate_a_updated_pack_trial3_ell0_reserve_closes_missing_action_blocker_now,
        "gate_b_updated_pack_substantive_pack_update_promoted_next": gate_b_updated_pack_substantive_pack_update_promoted_next,
        "gate_c_farther_hybrid_continuation_reopen_required_now": gate_c_farther_hybrid_continuation_reopen_required_now,
        "trial3_family_primary_reuse_admissible_now": trial3_family_primary_reuse_admissible_now,
        "blind_vector_observable_gate_still_blocked": blind_vector_observable_gate_still_blocked,
        "pack_update_required_now": pack_update_required_now,
        "hybrid_supporting_evidence_reopen_required": hybrid_supporting_evidence_reopen_required,
        "selected_primary_completion_lane": "updated_pack_substantive_pack_update",
        "selected_secondary_completion_lane": "updated_pack_trial3_ell0_reserve_supporting_inventory",
        "selected_reserve_completion_lane": "farther_hybrid_continuation_extra_q_range_only",
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": FOLLOWUP_ROUTE_NAME,
        "selected_followup_route_or_none": FOLLOWUP_ROUTE,
        "physical_reject_required": False,
    }

    declaration_payload = sign_base.payload(
        "8.7.56.2685",
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
            "overall_status": "vector_qball_form_factor_updated_pack_trial3_ell0_reserve_gate_declared",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "formulas": build_formulae(),
            "hits": {
                "status_branch_hit": sign_base.hit(status_text, "8.7.56.2683"),
                "roadmap_branch_hit": sign_base.hit(roadmap_text, ".2683-.2686"),
                "current_problem_hit": sign_base.hit(current_problem_text, "updated-pack trial3 ell0 reserve gate / pack-update refresh"),
                "current_status_hit": sign_base.hit(current_status_text, "updated-pack trial3 ell0 reserve gate / pack-update refresh"),
                "unified_roadmap_hit": sign_base.hit(unified_text, ".2679-.2682"),
                "long_roadmap_hit": sign_base.hit(long_text, ".2679-.2682"),
                "part5_hit": sign_base.hit(part5_text, "updated-pack trial3 ell0 reserve gate / pack-update refresh"),
            },
        },
    )
    declaration_paths = write_artifact("declaration_gate", declaration_payload)

    route_payload = {
        "generated_utc": sign_base.now_iso(),
        "phase": {
            "phase": 8,
            "step": "8.7.56.2686",
            "name": STEP_NAME + " route sync",
        },
        "inputs": declaration_paths,
        "rows": rows,
        "summary": summary,
        "decision": {
            "overall_status": "vector_qball_form_factor_updated_pack_trial3_ell0_reserve_gate_route_synced",
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

    print(f"[done] {STEP_TAG} updated-pack trial3 ell=0 reserve gate completed")


if __name__ == "__main__":
    main()
