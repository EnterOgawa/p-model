#!/usr/bin/env python3
"""Generate 8.7.56.2443-.2446 substantive pack-update gate artifacts."""

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
        "8.7.56.2439-2442",
        "substantive_pack_update_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.2443-2446"
STEP_NAME = "Trial-2 numeric alpha vector Q-ball form-factor substantive pack update gate"
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "substantive_pack_update_gate",
    prefix="q",
)

PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_substantive_pack_exact_ell0_"
    "series_operator_primary_effective_source_followup_gate"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_substantive_pack_exact_ell0_"
    "series_operator_primary_effective_source_secondary_next"
)
NEXT_ROUTE_NAME = "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_exact_ell0_series_operator_audit"
NEXT_ROUTE = "8.7.56.2447"
FOLLOWUP_ROUTE_NAME = "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_exact_ell0_series_operator_gate_effective_source_refresh"
FOLLOWUP_ROUTE = "8.7.56.2451"


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
    """Return formulas used in the substantive pack-update gate."""
    return {
        "gate_a": "Gate A = updated-pack exact ell=0 series/operator audit selected",
        "gate_b": "Gate B = effective source theorem retained as updated-pack followup",
        "gate_c": "Gate C = farther hybrid continuation reopen required now",
    }


# 関数: `.2443-.2446` を実行する。

def main() -> None:
    """Execute the substantive pack-update decision gate."""
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

    gate_a_updated_pack_exact_ell0_series_operator_selected = bool(
        prior_summary["substantive_pack_update_adoptable_now"]
        and prior_summary["pack_update_primary_surface_changes_internal_operator_surface"]
    )
    gate_b_effective_source_theorem_retained_as_followup = bool(
        gate_a_updated_pack_exact_ell0_series_operator_selected
        and prior_summary["pack_update_secondary_surface_targets_canonical_source_rule"]
    )
    gate_c_farther_hybrid_continuation_reopen_required_now = False
    old_density_proxy_eigenvalue_retry_admissible_now = False
    hybrid_supporting_evidence_reopen_required = gate_c_farther_hybrid_continuation_reopen_required_now
    updated_pack_primary_route_is_theorem_surface_not_numeric_recompute = bool(
        gate_a_updated_pack_exact_ell0_series_operator_selected
    )

    rows = [
        sign_base.row(
            "gate_a_updated_pack_exact_ell0_series_operator_selected",
            "pass" if gate_a_updated_pack_exact_ell0_series_operator_selected else "reject",
            "Gate A updated-pack exact ell=0 series/operator audit selected",
            sign_base.truth(gate_a_updated_pack_exact_ell0_series_operator_selected),
            "The next official move is to audit the exact ell=0 two-component series/operator surface under the newly adopted pack.",
        ),
        sign_base.row(
            "gate_b_effective_source_theorem_retained_as_followup",
            "pass" if gate_b_effective_source_theorem_retained_as_followup else "reject",
            "Gate B effective source theorem retained as followup",
            sign_base.truth(gate_b_effective_source_theorem_retained_as_followup),
            "Once the updated-pack operator surface is audited, the exact effective source theorem remains the next downstream canonical observable question.",
        ),
        sign_base.row(
            "gate_c_farther_hybrid_continuation_reopen_required_now",
            "pass" if gate_c_farther_hybrid_continuation_reopen_required_now else "reject",
            "Gate C farther hybrid continuation reopen required now",
            sign_base.truth(gate_c_farther_hybrid_continuation_reopen_required_now),
            "Extra q-range evidence is still unnecessary because the blocker is now fully localized to the updated-pack theorem surface.",
        ),
        sign_base.row(
            "updated_pack_primary_route_is_theorem_surface_not_numeric_recompute",
            "pass" if updated_pack_primary_route_is_theorem_surface_not_numeric_recompute else "reject",
            "updated-pack primary route is theorem surface, not numeric recompute",
            sign_base.truth(updated_pack_primary_route_is_theorem_surface_not_numeric_recompute),
            "The first updated-pack branch is a theorem-level operator audit rather than an immediate blind numeric recomputation.",
        ),
        sign_base.row(
            "old_density_proxy_eigenvalue_retry_admissible_now",
            "pass" if old_density_proxy_eigenvalue_retry_admissible_now else "reject",
            "old density/proxy/eigenvalue retry admissible now",
            sign_base.truth(old_density_proxy_eigenvalue_retry_admissible_now),
            "The updated pack still does not reopen exhausted density, proxy, or eigenvalue retry lanes.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "retained_scalar_residual_rel": float(prior_summary["retained_scalar_residual_rel"]),
        "gate_a_updated_pack_exact_ell0_series_operator_selected": gate_a_updated_pack_exact_ell0_series_operator_selected,
        "gate_b_effective_source_theorem_retained_as_followup": gate_b_effective_source_theorem_retained_as_followup,
        "gate_c_farther_hybrid_continuation_reopen_required_now": gate_c_farther_hybrid_continuation_reopen_required_now,
        "updated_pack_primary_route_is_theorem_surface_not_numeric_recompute": updated_pack_primary_route_is_theorem_surface_not_numeric_recompute,
        "old_density_proxy_eigenvalue_retry_admissible_now": old_density_proxy_eigenvalue_retry_admissible_now,
        "hybrid_supporting_evidence_reopen_required": hybrid_supporting_evidence_reopen_required,
        "selected_primary_pack_update_surface": "exact_ell0_two_component_series_and_longitudinal_operator",
        "selected_secondary_pack_update_surface": "exact_effective_source_theorem",
        "selected_reserve_completion_lane": "farther_hybrid_extra_q_range_only",
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": FOLLOWUP_ROUTE_NAME,
        "selected_followup_route_or_none": FOLLOWUP_ROUTE,
        "physical_reject_required": False,
    }

    declaration_payload = sign_base.payload(
        "8.7.56.2445",
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
            "overall_status": "vector_qball_form_factor_substantive_pack_update_gate_declared",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "formulas": build_formulae(),
            "hits": {
                "status_branch_hit": sign_base.hit(status_text, "8.7.56.2443"),
                "roadmap_branch_hit": sign_base.hit(roadmap_text, ".2443-.2446"),
                "current_problem_hit": sign_base.hit(current_problem_text, "substantive pack update gate / hybrid-reserve refresh"),
                "current_status_hit": sign_base.hit(current_status_text, "substantive pack update gate / hybrid-reserve refresh"),
                "unified_roadmap_hit": sign_base.hit(unified_text, ".2435-.2438"),
                "long_roadmap_hit": sign_base.hit(long_text, ".2443-.2446"),
                "part5_hit": sign_base.hit(part5_text, "substantive pack update gate / hybrid-reserve refresh"),
            },
        },
    )
    declaration_paths = write_artifact("declaration_gate", declaration_payload)

    route_payload = {
        "generated_utc": sign_base.now_iso(),
        "phase": {
            "phase": 8,
            "step": "8.7.56.2446",
            "name": STEP_NAME + " route sync",
        },
        "inputs": declaration_paths,
        "rows": rows,
        "summary": summary,
        "decision": {
            "overall_status": "vector_qball_form_factor_substantive_pack_update_gate_route_synced",
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

    print(f"[done] {STEP_TAG} substantive pack update gate completed")


if __name__ == "__main__":
    main()
