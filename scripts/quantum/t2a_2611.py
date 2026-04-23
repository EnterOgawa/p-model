#!/usr/bin/env python3
"""Generate 8.7.56.2611-.2614 updated-pack exact source-theorem gate artifacts."""

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
        "8.7.56.2607-2610",
        "updated_pack_exact_source_theorem_closeout_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.2611-2614"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack exact "
    "source-theorem gate"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_exact_source_theorem_gate",
    prefix="q",
)

PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_exact_source_"
    "theorem_no_go_derived_exact_ell0_operator_gate"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_exact_source_"
    "theorem_no_go_derived_exact_ell0_operator_primary_blind_vector_reserve_next"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_exact_ell0_"
    "action_level_operator_refresh_audit"
)
NEXT_ROUTE = "8.7.56.2615"
FOLLOWUP_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_exact_ell0_"
    "operator_gate_blind_vector_reserve_refresh"
)
FOLLOWUP_ROUTE = "8.7.56.2619"


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
    """Return formulas used in the updated-pack exact source-theorem gate."""
    return {
        "gate_a": "Gate A = updated-pack exact source theorem derived now",
        "gate_b": "Gate B = exact ell=0 action-level operator refresh selected as the next primary lane",
        "gate_c": "Gate C = blind-vector computation primary admissible now",
        "ordered_refresh": (
            "exact source-theorem closeout -> exact ell=0 action-level operator "
            "refresh -> blind-vector reserve"
        ),
    }


# 関数: `.2611-.2614` を実行する。

def main() -> None:
    """Execute the updated-pack exact source-theorem decision gate."""
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

    gate_a_updated_pack_exact_source_theorem_derived = bool(
        prior_summary["updated_pack_exact_source_theorem_derived_now"]
        and prior_summary["updated_pack_exact_source_theorem_no_go_verdict_passed"]
    )
    gate_b_updated_pack_exact_ell0_action_level_operator_primary_selected = bool(
        prior_summary["updated_pack_exact_ell0_action_level_operator_refresh_required"]
        and gate_a_updated_pack_exact_source_theorem_derived
    )
    gate_c_blind_vector_computation_primary_admissible_now = False
    exact_source_theorem_derived_now = bool(
        prior_summary["updated_pack_exact_source_theorem_derived_now"]
    )
    exact_source_theorem_no_go_verdict_fixed = bool(
        prior_summary["updated_pack_exact_source_theorem_no_go_verdict_passed"]
    )
    exact_ell0_action_level_operator_available_now = False
    residual_origin_theorem_explained_now = bool(
        prior_summary["residual_origin_theorem_explained_now"]
    )
    farther_hybrid_continuation_reopen_required_now = bool(
        prior_summary["farther_hybrid_continuation_reopen_required_now"]
    )
    old_density_proxy_eigenvalue_retry_admissible_now = False

    rows = [
        sign_base.row(
            "gate_a_updated_pack_exact_source_theorem_derived",
            "pass" if gate_a_updated_pack_exact_source_theorem_derived else "reject",
            "Gate A updated-pack exact source theorem derived",
            sign_base.truth(gate_a_updated_pack_exact_source_theorem_derived),
            "The current updated-pack theorem now closes on the no-go branch rather than remaining an unresolved placeholder.",
        ),
        sign_base.row(
            "gate_b_updated_pack_exact_ell0_action_level_operator_primary_selected",
            "pass" if gate_b_updated_pack_exact_ell0_action_level_operator_primary_selected else "reject",
            "Gate B updated-pack exact ell=0 action-level operator primary selected",
            sign_base.truth(gate_b_updated_pack_exact_ell0_action_level_operator_primary_selected),
            "Once the source theorem is synchronized, the next honest remaining mainline returns to the exact ell=0 action-level operator gap.",
        ),
        sign_base.row(
            "gate_c_blind_vector_computation_primary_admissible_now",
            "pass" if gate_c_blind_vector_computation_primary_admissible_now else "reject",
            "Gate C blind-vector computation primary admissible now",
            sign_base.truth(gate_c_blind_vector_computation_primary_admissible_now),
            "Blind-vector direct computation remains reserve-only because the current-pack theorem closes as no vector correction.",
        ),
        sign_base.row(
            "exact_source_theorem_derived_now",
            "pass" if exact_source_theorem_derived_now else "reject",
            "exact source theorem derived now",
            sign_base.truth(exact_source_theorem_derived_now),
            "The gate synchronizes that the current updated-pack same-field photon-side source theorem is no longer missing.",
        ),
        sign_base.row(
            "exact_source_theorem_no_go_verdict_fixed",
            "pass" if exact_source_theorem_no_go_verdict_fixed else "reject",
            "exact source-theorem no-go verdict fixed",
            sign_base.truth(exact_source_theorem_no_go_verdict_fixed),
            "The theorem now fixes no vector correction under the current same-field updated-pack route.",
        ),
        sign_base.row(
            "exact_ell0_action_level_operator_available_now",
            "pass" if exact_ell0_action_level_operator_available_now else "reject",
            "exact ell=0 action-level operator available now",
            sign_base.truth(exact_ell0_action_level_operator_available_now),
            "The remaining action-level operator gap is still open even after the source theorem is closed.",
        ),
        sign_base.row(
            "residual_origin_theorem_explained_now",
            "pass" if residual_origin_theorem_explained_now else "reject",
            "residual origin theorem explained now",
            sign_base.truth(residual_origin_theorem_explained_now),
            "The theorem closes as current-pack no-go, so the retained 1.9% scalar residual is still not explained theorem-level.",
        ),
        sign_base.row(
            "farther_hybrid_continuation_reopen_required_now",
            "pass" if farther_hybrid_continuation_reopen_required_now else "reject",
            "farther hybrid continuation reopen required now",
            sign_base.truth(farther_hybrid_continuation_reopen_required_now),
            "Extra q-range evidence remains reserve-only because the blocker has shifted back to the operator lane.",
        ),
        sign_base.row(
            "old_density_proxy_eigenvalue_retry_admissible_now",
            "pass" if old_density_proxy_eigenvalue_retry_admissible_now else "reject",
            "old density/proxy/eigenvalue retry admissible now",
            sign_base.truth(old_density_proxy_eigenvalue_retry_admissible_now),
            "Closing the source theorem as no-go does not reopen exhausted pre-update retry families.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "retained_scalar_residual_rel": float(prior_summary["retained_scalar_residual_rel"]),
        "gate_a_updated_pack_exact_source_theorem_derived": gate_a_updated_pack_exact_source_theorem_derived,
        "gate_b_updated_pack_exact_ell0_action_level_operator_primary_selected": gate_b_updated_pack_exact_ell0_action_level_operator_primary_selected,
        "gate_c_blind_vector_computation_primary_admissible_now": gate_c_blind_vector_computation_primary_admissible_now,
        "exact_source_theorem_derived_now": exact_source_theorem_derived_now,
        "exact_source_theorem_no_go_verdict_fixed": exact_source_theorem_no_go_verdict_fixed,
        "exact_ell0_action_level_operator_available_now": exact_ell0_action_level_operator_available_now,
        "residual_origin_theorem_explained_now": residual_origin_theorem_explained_now,
        "farther_hybrid_continuation_reopen_required_now": farther_hybrid_continuation_reopen_required_now,
        "old_density_proxy_eigenvalue_retry_admissible_now": old_density_proxy_eigenvalue_retry_admissible_now,
        "hybrid_supporting_evidence_reopen_required": farther_hybrid_continuation_reopen_required_now,
        "selected_primary_pack_update_surface": "updated_pack_exact_ell0_action_level_operator_refresh",
        "selected_secondary_pack_update_surface": "blind_vector_reserve_after_operator_refresh",
        "selected_reserve_completion_lane": "farther_hybrid_extra_q_range_only",
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": FOLLOWUP_ROUTE_NAME,
        "selected_followup_route_or_none": FOLLOWUP_ROUTE,
        "physical_reject_required": False,
    }

    declaration_payload = sign_base.payload(
        "8.7.56.2613",
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
            "overall_status": "vector_qball_form_factor_updated_pack_exact_source_theorem_gate_declared",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "formulas": build_formulae(),
            "hits": {
                "status_branch_hit": sign_base.hit(status_text, "8.7.56.2607"),
                "roadmap_branch_hit": sign_base.hit(roadmap_text, ".2587-.2590"),
                "current_problem_hit": sign_base.hit(current_problem_text, "exact source-theorem closeout"),
                "current_status_hit": sign_base.hit(current_status_text, "exact source-theorem closeout"),
                "unified_roadmap_hit": sign_base.hit(unified_text, ".2583-.2590"),
                "long_roadmap_hit": sign_base.hit(long_text, ".2587-.2590"),
                "part5_same_field_zero_hit": sign_base.hit(part5_text, "same-field on-shell zero"),
            },
        },
    )
    declaration_paths = write_artifact("declaration_gate", declaration_payload)

    route_payload = {
        "generated_utc": sign_base.now_iso(),
        "phase": {
            "phase": 8,
            "step": "8.7.56.2614",
            "name": STEP_NAME + " route sync",
        },
        "inputs": declaration_paths,
        "rows": rows,
        "summary": summary,
        "decision": {
            "overall_status": "vector_qball_form_factor_updated_pack_exact_source_theorem_gate_route_synced",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        "evidence": {
            "formulas": build_formulae(),
            "disposition": {
                "exact_source_theorem_derived_now": exact_source_theorem_derived_now,
                "exact_source_theorem_no_go_verdict_fixed": exact_source_theorem_no_go_verdict_fixed,
                "exact_ell0_operator_primary_selected": gate_b_updated_pack_exact_ell0_action_level_operator_primary_selected,
                "direct_blind_vector_still_blocked": not gate_c_blind_vector_computation_primary_admissible_now,
            },
        },
    }
    route_paths = write_artifact("route_sync", route_payload)

    print("[ok] updated-pack exact source-theorem gate artifacts written")
    print(f"  declaration_gate: {declaration_paths['json']}")
    print(f"  route_sync: {route_paths['json']}")


if __name__ == "__main__":
    main()
