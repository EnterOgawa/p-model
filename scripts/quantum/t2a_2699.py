#!/usr/bin/env python3
"""Generate 8.7.56.2699-.2702 updated-pack 4D form-factor gate artifacts."""

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
        "8.7.56.2695-2698",
        "updated_pack_4d_formfactor_hypothesis_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.2699-2702"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack 4D "
    "form-factor gate"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_4d_formfactor_gate",
    prefix="q",
)

PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_4d_"
    "formfactor_static_q0_operator_gap_gate"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_4d_"
    "formfactor_static_q0_operator_gap_primary_hybrid_reserve_next"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_exact_4d_"
    "formfactor_operator_completion_audit"
)
NEXT_ROUTE = "8.7.56.2703"
FOLLOWUP_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_4d_operator_"
    "gate_pack_refresh"
)
FOLLOWUP_ROUTE = "8.7.56.2707"


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
    """Return formulas used in the updated-pack 4D form-factor gate."""
    return {
        "gate_a": "Gate A = updated-pack 4D form-factor breakthrough passed now",
        "gate_b": "Gate B = updated-pack exact 4D operator completion promoted next",
        "gate_c": "Gate C = farther hybrid continuation reopen required now",
    }


# 関数: `.2699-.2702` を実行する。

def main() -> None:
    """Execute the updated-pack 4D form-factor gate."""
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

    gate_a_updated_pack_4d_formfactor_breakthrough_passed_now = bool(
        prior_summary["fourd_formfactor_hypothesis_closes_missing_action_blocker_now"]
    )
    gate_b_updated_pack_exact_4d_operator_completion_promoted_next = bool(
        prior_summary["fourd_formfactor_hypothesis_refined_to_operator_completion_lane"]
        and (not gate_a_updated_pack_4d_formfactor_breakthrough_passed_now)
    )
    gate_c_farther_hybrid_continuation_reopen_required_now = False
    failure_matrix_non_surrogate_guard_preserved = bool(
        prior_summary["failure_matrix_non_surrogate_guard_preserved"]
    )
    blind_vector_observable_gate_still_blocked = bool(
        prior_summary["blind_vector_observable_gate_still_blocked"]
    )
    old_density_proxy_eigenvalue_retry_admissible_now = False
    pack_update_required_now = bool(
        gate_b_updated_pack_exact_4d_operator_completion_promoted_next
    )

    rows = [
        sign_base.row(
            "gate_a_updated_pack_4d_formfactor_breakthrough_passed_now",
            "pass" if gate_a_updated_pack_4d_formfactor_breakthrough_passed_now else "reject",
            "Gate A updated-pack 4D form-factor breakthrough passed now",
            sign_base.truth(gate_a_updated_pack_4d_formfactor_breakthrough_passed_now),
            "The current audit does not yet close the missing-action blocker, so the note is not promoted as a solved theorem at this gate.",
        ),
        sign_base.row(
            "gate_b_updated_pack_exact_4d_operator_completion_promoted_next",
            "pass" if gate_b_updated_pack_exact_4d_operator_completion_promoted_next else "reject",
            "Gate B updated-pack exact 4D operator completion promoted next",
            sign_base.truth(gate_b_updated_pack_exact_4d_operator_completion_promoted_next),
            "Because the note survives only as an operator-completion lane, the next official move is to derive the exact 4D observable operator itself.",
        ),
        sign_base.row(
            "gate_c_farther_hybrid_continuation_reopen_required_now",
            "pass" if gate_c_farther_hybrid_continuation_reopen_required_now else "reject",
            "Gate C farther hybrid continuation reopen required now",
            sign_base.truth(gate_c_farther_hybrid_continuation_reopen_required_now),
            "Extra q-range evidence is still reserve-only because the current blocker is the exact 4D operator, not the residual-origin sampling window.",
        ),
        sign_base.row(
            "failure_matrix_non_surrogate_guard_preserved",
            "pass" if failure_matrix_non_surrogate_guard_preserved else "reject",
            "failure-matrix non-surrogate guard preserved",
            sign_base.truth(failure_matrix_non_surrogate_guard_preserved),
            "Promoting the 4D lane does not reopen the already-falsified density/proxy/eigenvalue family.",
        ),
        sign_base.row(
            "blind_vector_observable_gate_still_blocked",
            "pass" if blind_vector_observable_gate_still_blocked else "reject",
            "blind-vector observable gate still blocked",
            sign_base.truth(blind_vector_observable_gate_still_blocked),
            "Blind-vector direct computation remains blocked until the 4D operator lane closes honestly.",
        ),
        sign_base.row(
            "old_density_proxy_eigenvalue_retry_admissible_now",
            "pass" if old_density_proxy_eigenvalue_retry_admissible_now else "reject",
            "old density/proxy/eigenvalue retry admissible now",
            sign_base.truth(old_density_proxy_eigenvalue_retry_admissible_now),
            "The gate keeps the exhausted same-level retry family closed.",
        ),
        sign_base.row(
            "pack_update_required_now",
            "pass" if pack_update_required_now else "reject",
            "updated-pack substantive pack update required now",
            sign_base.truth(pack_update_required_now),
            "The current mainline remains a substantive pack update because the 4D hypothesis is still only partially specified.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "retained_scalar_residual_rel": float(prior_summary["retained_scalar_residual_rel"]),
        "gate_a_updated_pack_4d_formfactor_breakthrough_passed_now": gate_a_updated_pack_4d_formfactor_breakthrough_passed_now,
        "gate_b_updated_pack_exact_4d_operator_completion_promoted_next": gate_b_updated_pack_exact_4d_operator_completion_promoted_next,
        "gate_c_farther_hybrid_continuation_reopen_required_now": gate_c_farther_hybrid_continuation_reopen_required_now,
        "failure_matrix_non_surrogate_guard_preserved": failure_matrix_non_surrogate_guard_preserved,
        "blind_vector_observable_gate_still_blocked": blind_vector_observable_gate_still_blocked,
        "old_density_proxy_eigenvalue_retry_admissible_now": old_density_proxy_eigenvalue_retry_admissible_now,
        "pack_update_required_now": pack_update_required_now,
        "selected_primary_completion_lane": "updated_pack_exact_4d_formfactor_operator_completion_audit",
        "selected_secondary_completion_lane": "updated_pack_4d_static_q0_discriminator_retained",
        "selected_reserve_completion_lane": "farther_hybrid_extra_q_range_only",
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": FOLLOWUP_ROUTE_NAME,
        "selected_followup_route_or_none": FOLLOWUP_ROUTE,
        "physical_reject_required": False,
    }

    declaration_payload = sign_base.payload(
        "8.7.56.2701",
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
            "overall_status": "vector_qball_form_factor_updated_pack_4d_formfactor_gate_declared",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "formulas": build_formulae(),
            "hits": {
                "status_branch_hit": sign_base.hit(status_text, "8.7.56.2699"),
                "roadmap_branch_hit": sign_base.hit(roadmap_text, ".2699-.2702"),
                "current_problem_hit": sign_base.hit(current_problem_text, "updated-pack 4D form-factor gate / failure-matrix refresh"),
                "current_status_hit": sign_base.hit(current_status_text, "updated-pack 4D form-factor gate / failure-matrix refresh"),
                "unified_roadmap_hit": sign_base.hit(unified_text, ".2695-.2698"),
                "long_roadmap_hit": sign_base.hit(long_text, ".2695-.2698"),
                "part5_hit": sign_base.hit(part5_text, "updated-pack 4D form-factor gate / failure-matrix refresh"),
            },
        },
    )
    declaration_paths = write_artifact("declaration_gate", declaration_payload)

    route_payload = {
        "generated_utc": sign_base.now_iso(),
        "phase": {
            "phase": 8,
            "step": "8.7.56.2702",
            "name": STEP_NAME + " route sync",
        },
        "inputs": declaration_paths,
        "rows": rows,
        "summary": summary,
        "decision": {
            "overall_status": "vector_qball_form_factor_updated_pack_4d_formfactor_gate_route_synced",
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

    print(f"[done] {STEP_TAG} updated-pack 4D form-factor gate completed")


if __name__ == "__main__":
    main()
