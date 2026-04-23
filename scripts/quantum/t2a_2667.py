#!/usr/bin/env python3
"""Generate 8.7.56.2667-.2670 updated-pack phase-1 reciprocal backreaction gate artifacts."""

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
        "8.7.56.2663-2666",
        "updated_pack_phase1_reciprocal_backreaction_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.2667-2670"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack phase-1 "
    "reciprocal backreaction gate"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_phase1_reciprocal_backreaction_gate",
    prefix="q",
)

PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_phase1_"
    "shared_rho_even_backreaction_only_nonlinear_closure_promotion_gate"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_phase1_"
    "shared_rho_even_backreaction_only_nonlinear_closure_primary_trial3_ell0_"
    "reserve_next"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_phase1_nonheuristic_"
    "two_component_nonlinear_closure_audit"
)
NEXT_ROUTE = "8.7.56.2671"
FOLLOWUP_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_phase1_nonlinear_"
    "closure_gate_trial3_ell0_reserve_refresh"
)
FOLLOWUP_ROUTE = "8.7.56.2675"


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
        "gate_a": "Gate A = updated-pack literal reciprocal backreaction available now",
        "gate_b": "Gate B = updated-pack non-heuristic two-component nonlinear closure promoted next",
        "gate_c": "Gate C = updated-pack trial-3 ell=0 closure reserve retained",
    }


# 関数: `.2667-.2670` を実行する。

def main() -> None:
    """Execute the updated-pack reciprocal-backreaction decision gate."""
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

    gate_a_updated_pack_phase1_literal_reciprocal_backreaction_available_now = bool(
        prior_summary["updated_pack_phase1_literal_reciprocal_backreaction_formula_available"]
    )
    gate_b_updated_pack_nonheuristic_two_component_nonlinear_closure_promoted_next = bool(
        (not gate_a_updated_pack_phase1_literal_reciprocal_backreaction_available_now)
        and prior_summary["updated_pack_phase1_reciprocal_backreaction_supported_under_current_pack"]
    )
    gate_c_updated_pack_trial3_ell0_closure_reserve_retained = bool(
        prior_summary["updated_pack_trial3_family_ell0_closure_reserve_retained"]
    )
    blind_vector_observable_gate_still_blocked = bool(
        prior_summary["blind_vector_observable_gate_still_blocked"]
    )
    trial3_family_primary_reuse_admissible_now = False
    pack_update_required_now = bool(prior_summary["pack_update_required_now"])
    farther_hybrid_continuation_reopen_required_now = False

    rows = [
        sign_base.row(
            "gate_a_updated_pack_phase1_literal_reciprocal_backreaction_available_now",
            "pass" if gate_a_updated_pack_phase1_literal_reciprocal_backreaction_available_now else "reject",
            "Gate A updated-pack phase-1 literal reciprocal backreaction available now",
            sign_base.truth(gate_a_updated_pack_phase1_literal_reciprocal_backreaction_available_now),
            "The current pack still lacks a literal linear f_L -> f_0 source term, so reciprocal backreaction does not close here.",
        ),
        sign_base.row(
            "gate_b_updated_pack_nonheuristic_two_component_nonlinear_closure_promoted_next",
            "pass" if gate_b_updated_pack_nonheuristic_two_component_nonlinear_closure_promoted_next else "reject",
            "Gate B updated-pack non-heuristic two-component nonlinear closure promoted next",
            sign_base.truth(gate_b_updated_pack_nonheuristic_two_component_nonlinear_closure_promoted_next),
            "Because the current solver only carries shared-rho even backreaction, the honest next move is the nonlinear two-component closure audit.",
        ),
        sign_base.row(
            "gate_c_updated_pack_trial3_ell0_closure_reserve_retained",
            "pass" if gate_c_updated_pack_trial3_ell0_closure_reserve_retained else "reject",
            "Gate C updated-pack trial-3 ell=0 closure reserve retained",
            sign_base.truth(gate_c_updated_pack_trial3_ell0_closure_reserve_retained),
            "The trial-3 family remains reserve-only while the phase-1 nonlinear closure is still heuristic.",
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
            "Same-level trial-3 reuse remains blocked as a primary route.",
        ),
        sign_base.row(
            "pack_update_required_now",
            "pass" if pack_update_required_now else "reject",
            "substantive pack update required now",
            sign_base.truth(pack_update_required_now),
            "The route still advances inside the retained pack and does not require new external input.",
        ),
        sign_base.row(
            "farther_hybrid_continuation_reopen_required_now",
            "pass" if farther_hybrid_continuation_reopen_required_now else "reject",
            "farther hybrid continuation reopen required now",
            sign_base.truth(farther_hybrid_continuation_reopen_required_now),
            "Extra q-range evidence is not required at this point, so farther hybrid continuation remains closed reserve-only.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "retained_scalar_residual_rel": float(prior_summary["retained_scalar_residual_rel"]),
        "gate_a_updated_pack_phase1_literal_reciprocal_backreaction_available_now": gate_a_updated_pack_phase1_literal_reciprocal_backreaction_available_now,
        "gate_b_updated_pack_nonheuristic_two_component_nonlinear_closure_promoted_next": gate_b_updated_pack_nonheuristic_two_component_nonlinear_closure_promoted_next,
        "gate_c_updated_pack_trial3_ell0_closure_reserve_retained": gate_c_updated_pack_trial3_ell0_closure_reserve_retained,
        "blind_vector_observable_gate_still_blocked": blind_vector_observable_gate_still_blocked,
        "trial3_family_primary_reuse_admissible_now": trial3_family_primary_reuse_admissible_now,
        "pack_update_required_now": pack_update_required_now,
        "farther_hybrid_continuation_reopen_required_now": farther_hybrid_continuation_reopen_required_now,
        "selected_primary_completion_lane": "updated_pack_nonheuristic_two_component_nonlinear_closure",
        "selected_secondary_completion_lane": "updated_pack_trial3_ell0_closure_reserve",
        "selected_reserve_completion_lane": "blind_vector_reserve",
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": FOLLOWUP_ROUTE_NAME,
        "selected_followup_route_or_none": FOLLOWUP_ROUTE,
        "physical_reject_required": False,
    }

    declaration_payload = sign_base.payload(
        "8.7.56.2669",
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
            "overall_status": "vector_qball_form_factor_updated_pack_phase1_reciprocal_backreaction_gate_declared",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "formulas": build_formulae(),
            "hits": {
                "status_branch_hit": sign_base.hit(status_text, "8.7.56.2667"),
                "roadmap_branch_hit": sign_base.hit(roadmap_text, ".2667-.2670"),
                "current_problem_hit": sign_base.hit(current_problem_text, "updated-pack phase-1 reciprocal backreaction gate / nonlinear-closure refresh"),
                "current_status_hit": sign_base.hit(current_status_text, "updated-pack phase-1 reciprocal backreaction gate / nonlinear-closure refresh"),
                "unified_roadmap_hit": sign_base.hit(unified_text, ".2667-.2670"),
                "long_roadmap_hit": sign_base.hit(long_text, ".2667-.2670"),
                "part5_hit": sign_base.hit(part5_text, "updated-pack phase-1 reciprocal backreaction gate / nonlinear-closure refresh"),
            },
        },
    )
    declaration_paths = write_artifact("declaration_gate", declaration_payload)

    route_payload = {
        "generated_utc": sign_base.now_iso(),
        "phase": {
            "phase": 8,
            "step": "8.7.56.2670",
            "name": STEP_NAME + " route sync",
        },
        "inputs": declaration_paths,
        "rows": rows,
        "summary": summary,
        "decision": {
            "overall_status": "vector_qball_form_factor_updated_pack_phase1_reciprocal_backreaction_gate_route_synced",
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

    print("[ok] updated-pack phase-1 reciprocal backreaction gate artifacts written")
    print(f"  declaration_gate: {declaration_paths['json']}")
    print(f"  route_sync: {route_paths['json']}")


if __name__ == "__main__":
    main()
