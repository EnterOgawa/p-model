#!/usr/bin/env python3
"""Generate 8.7.56.2743-.2746 updated-pack hybrid reserve refresh artifacts."""

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
        "8.7.56.2739-2742",
        "updated_pack_4d_pack_refresh_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.2743-2746"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack hybrid "
    "reserve refresh audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_hybrid_reserve_refresh_audit",
    prefix="q",
)

PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "4d_theorem_normalization_pack_refresh_audited_hybrid_reserve_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "hybrid_reserve_refresh_audited_reserve_registry_secondary_gate"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_hybrid_reserve_"
    "gate_reserve_registry_refresh"
)
NEXT_ROUTE = "8.7.56.2747"
FOLLOWUP_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_hybrid_reserve_"
    "registry_audit"
)
FOLLOWUP_ROUTE = "8.7.56.2751"


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


# 関数: hybrid reserve refresh で使う式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used in the hybrid reserve refresh audit."""
    return {
        "hybrid_reserve_order": (
            "theorem-normalization pack refresh -> hybrid reserve judgement -> "
            "reserve registry refresh -> only then extra q-range reopen"
        ),
        "extra_q_range_reopen_rule": (
            "Reopen farther hybrid continuation only if theorem-level residual-origin "
            "discrimination still requires extra q-range evidence after canonical "
            "surviving-observable and q_0 = 0 normalization decisions."
        ),
        "reserve_policy": (
            "Keep blind-vector direct computation blocked and keep farther hybrid "
            "continuation reserve-only while theorem/normalization canonicalization "
            "remains unresolved."
        ),
    }


# 関数: `.2743-.2746` を実行する。

def main() -> None:
    """Execute the updated-pack hybrid reserve refresh audit."""
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

    updated_pack_hybrid_reserve_refresh_audit_selected = bool(
        prior_summary["gate_b_updated_pack_hybrid_reserve_refresh_promoted_next"]
        and prior_summary["pack_update_required_now"]
    )
    updated_pack_4d_theorem_normalization_pack_refresh_surface_retained = bool(
        (not prior_summary["gate_a_exact_4d_theorem_normalization_pack_refresh_available_now"])
        and prior_summary["pack_update_required_now"]
    )
    updated_pack_farther_hybrid_extra_q_range_hold_explicit = bool(
        prior_summary["selected_reserve_completion_lane"] == "farther_hybrid_extra_q_range_only"
    )
    updated_pack_extra_q_range_reopen_condition_explicit = bool(
        updated_pack_farther_hybrid_extra_q_range_hold_explicit
        and (not prior_summary["gate_c_farther_hybrid_continuation_reopen_required_now"])
    )
    updated_pack_hybrid_reserve_registry_followup_explicit = bool(
        prior_summary["selected_secondary_completion_lane"]
        == "updated_pack_hybrid_reserve_gate_reserve_registry_refresh"
    )
    updated_pack_hybrid_reserve_refresh_target_surface_explicit = bool(
        updated_pack_hybrid_reserve_refresh_audit_selected
        and updated_pack_4d_theorem_normalization_pack_refresh_surface_retained
        and updated_pack_farther_hybrid_extra_q_range_hold_explicit
        and updated_pack_extra_q_range_reopen_condition_explicit
        and updated_pack_hybrid_reserve_registry_followup_explicit
    )
    updated_pack_hybrid_reserve_refresh_machine_readable_now = bool(
        updated_pack_hybrid_reserve_refresh_target_surface_explicit
    )
    exact_surviving_observable_selection_available_now = False
    exact_corrected_4d_normalization_available_now = False
    hybrid_reserve_judgement_available_now = False
    extra_q_range_evidence_required_now = False
    updated_pack_hybrid_reserve_registry_primary_followup_required = bool(
        updated_pack_hybrid_reserve_refresh_machine_readable_now
        and (not hybrid_reserve_judgement_available_now)
    )
    failure_matrix_non_surrogate_guard_preserved = bool(
        prior_summary["failure_matrix_non_surrogate_guard_preserved"]
    )
    blind_vector_observable_gate_still_blocked = bool(
        prior_summary["blind_vector_observable_gate_still_blocked"]
    )
    farther_hybrid_continuation_reopen_required_now = False

    rows = [
        sign_base.row(
            "updated_pack_hybrid_reserve_refresh_audit_selected",
            "pass" if updated_pack_hybrid_reserve_refresh_audit_selected else "reject",
            "updated-pack hybrid reserve refresh audit selected",
            sign_base.truth(updated_pack_hybrid_reserve_refresh_audit_selected),
            "Once theorem/normalization pack refresh is fixed but unresolved, the next honest downstream lane is reserve refresh rather than blind computation.",
        ),
        sign_base.row(
            "updated_pack_4d_theorem_normalization_pack_refresh_surface_retained",
            "pass" if updated_pack_4d_theorem_normalization_pack_refresh_surface_retained else "reject",
            "updated-pack 4D theorem-normalization pack-refresh surface retained",
            sign_base.truth(updated_pack_4d_theorem_normalization_pack_refresh_surface_retained),
            "The reserve refresh inherits the unresolved corrected 4D theorem/normalization surface instead of reopening old surrogate routes.",
        ),
        sign_base.row(
            "updated_pack_farther_hybrid_extra_q_range_hold_explicit",
            "pass" if updated_pack_farther_hybrid_extra_q_range_hold_explicit else "reject",
            "updated-pack farther hybrid extra q-range hold explicit",
            sign_base.truth(updated_pack_farther_hybrid_extra_q_range_hold_explicit),
            "The farther hybrid lane remains reserve-only and is tracked explicitly as an extra q-range item rather than an active mainline.",
        ),
        sign_base.row(
            "updated_pack_extra_q_range_reopen_condition_explicit",
            "pass" if updated_pack_extra_q_range_reopen_condition_explicit else "reject",
            "updated-pack extra q-range reopen condition explicit",
            sign_base.truth(updated_pack_extra_q_range_reopen_condition_explicit),
            "The route now states explicitly that extra q-range may reopen only after theorem-level residual-origin discrimination still needs it.",
        ),
        sign_base.row(
            "updated_pack_hybrid_reserve_registry_followup_explicit",
            "pass" if updated_pack_hybrid_reserve_registry_followup_explicit else "reject",
            "updated-pack hybrid reserve registry followup explicit",
            sign_base.truth(updated_pack_hybrid_reserve_registry_followup_explicit),
            "The branch already knows that reserve bookkeeping must be refreshed in one dedicated registry lane after this audit.",
        ),
        sign_base.row(
            "updated_pack_hybrid_reserve_refresh_target_surface_explicit",
            "pass" if updated_pack_hybrid_reserve_refresh_target_surface_explicit else "reject",
            "updated-pack hybrid reserve refresh target surface explicit",
            sign_base.truth(updated_pack_hybrid_reserve_refresh_target_surface_explicit),
            "The reserve refresh target is explicit: unresolved theorem/normalization stack plus reserve-only hybrid evidence and a controlled reopen condition.",
        ),
        sign_base.row(
            "updated_pack_hybrid_reserve_refresh_machine_readable_now",
            "pass" if updated_pack_hybrid_reserve_refresh_machine_readable_now else "reject",
            "updated-pack hybrid reserve refresh machine-readable now",
            sign_base.truth(updated_pack_hybrid_reserve_refresh_machine_readable_now),
            "The reserve-policy lane now lives on one explicit machine-readable surface.",
        ),
        sign_base.row(
            "exact_surviving_observable_selection_available_now",
            "pass" if exact_surviving_observable_selection_available_now else "reject",
            "exact surviving-observable selection available now",
            sign_base.truth(exact_surviving_observable_selection_available_now),
            "The reserve refresh does not yet derive which corrected 4D observable survives canonically.",
        ),
        sign_base.row(
            "exact_corrected_4d_normalization_available_now",
            "pass" if exact_corrected_4d_normalization_available_now else "reject",
            "exact corrected 4D normalization available now",
            sign_base.truth(exact_corrected_4d_normalization_available_now),
            "Normalization remains unavailable because the theorem/observable selection pair is still unresolved.",
        ),
        sign_base.row(
            "hybrid_reserve_judgement_available_now",
            "pass" if hybrid_reserve_judgement_available_now else "reject",
            "hybrid reserve judgement available now",
            sign_base.truth(hybrid_reserve_judgement_available_now),
            "The branch still cannot make a final scientific judgement about whether the retained hybrid evidence must reopen.",
        ),
        sign_base.row(
            "extra_q_range_evidence_required_now",
            "pass" if extra_q_range_evidence_required_now else "reject",
            "extra q-range evidence required now",
            sign_base.truth(extra_q_range_evidence_required_now),
            "Current evidence still says that extra q-range is not scientifically required yet.",
        ),
        sign_base.row(
            "updated_pack_hybrid_reserve_registry_primary_followup_required",
            "pass" if updated_pack_hybrid_reserve_registry_primary_followup_required else "reject",
            "updated-pack hybrid reserve registry primary followup required",
            sign_base.truth(updated_pack_hybrid_reserve_registry_primary_followup_required),
            "Once reserve policy is localized but unresolved, the next honest move is a registry refresh that keeps reserve bookkeeping explicit.",
        ),
        sign_base.row(
            "failure_matrix_non_surrogate_guard_preserved",
            "pass" if failure_matrix_non_surrogate_guard_preserved else "reject",
            "failure-matrix non-surrogate guard preserved",
            sign_base.truth(failure_matrix_non_surrogate_guard_preserved),
            "Reserve refresh keeps the exhausted density/proxy/eigenvalue family closed.",
        ),
        sign_base.row(
            "blind_vector_observable_gate_still_blocked",
            "pass" if blind_vector_observable_gate_still_blocked else "reject",
            "blind-vector observable gate still blocked",
            sign_base.truth(blind_vector_observable_gate_still_blocked),
            "Blind-vector direct computation remains blocked until theorem, observable selection, and normalization close honestly.",
        ),
        sign_base.row(
            "farther_hybrid_continuation_reopen_required_now",
            "pass" if farther_hybrid_continuation_reopen_required_now else "reject",
            "farther hybrid continuation reopen required now",
            sign_base.truth(farther_hybrid_continuation_reopen_required_now),
            "The current reserve refresh explicitly keeps farther hybrid continuation closed.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "retained_scalar_residual_rel": float(prior_summary["retained_scalar_residual_rel"]),
        "updated_pack_hybrid_reserve_refresh_audit_selected": updated_pack_hybrid_reserve_refresh_audit_selected,
        "updated_pack_4d_theorem_normalization_pack_refresh_surface_retained": updated_pack_4d_theorem_normalization_pack_refresh_surface_retained,
        "updated_pack_farther_hybrid_extra_q_range_hold_explicit": updated_pack_farther_hybrid_extra_q_range_hold_explicit,
        "updated_pack_extra_q_range_reopen_condition_explicit": updated_pack_extra_q_range_reopen_condition_explicit,
        "updated_pack_hybrid_reserve_registry_followup_explicit": updated_pack_hybrid_reserve_registry_followup_explicit,
        "updated_pack_hybrid_reserve_refresh_target_surface_explicit": updated_pack_hybrid_reserve_refresh_target_surface_explicit,
        "updated_pack_hybrid_reserve_refresh_machine_readable_now": updated_pack_hybrid_reserve_refresh_machine_readable_now,
        "exact_surviving_observable_selection_available_now": exact_surviving_observable_selection_available_now,
        "exact_corrected_4d_normalization_available_now": exact_corrected_4d_normalization_available_now,
        "hybrid_reserve_judgement_available_now": hybrid_reserve_judgement_available_now,
        "extra_q_range_evidence_required_now": extra_q_range_evidence_required_now,
        "updated_pack_hybrid_reserve_registry_primary_followup_required": updated_pack_hybrid_reserve_registry_primary_followup_required,
        "failure_matrix_non_surrogate_guard_preserved": failure_matrix_non_surrogate_guard_preserved,
        "blind_vector_observable_gate_still_blocked": blind_vector_observable_gate_still_blocked,
        "farther_hybrid_continuation_reopen_required_now": farther_hybrid_continuation_reopen_required_now,
        "selected_primary_pack_update_surface": "updated_pack_hybrid_reserve_refresh",
        "selected_secondary_pack_update_surface": "updated_pack_hybrid_reserve_registry_refresh",
        "selected_reserve_completion_lane": "farther_hybrid_extra_q_range_only",
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": FOLLOWUP_ROUTE_NAME,
        "selected_followup_route_or_none": FOLLOWUP_ROUTE,
        "physical_reject_required": False,
    }

    declaration_payload = sign_base.payload(
        "8.7.56.2745",
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
            "overall_status": "vector_qball_form_factor_updated_pack_hybrid_reserve_refresh_declared",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "formulas": build_formulae(),
            "hits": {
                "status_branch_hit": sign_base.hit(status_text, "8.7.56.2735"),
                "roadmap_branch_hit": sign_base.hit(roadmap_text, ".2731-.2734"),
                "current_problem_hit": sign_base.hit(
                    current_problem_text,
                    "corrected 4D normalization update",
                ),
                "current_status_hit": sign_base.hit(
                    current_status_text,
                    "corrected 4D normalization update",
                ),
                "unified_roadmap_hit": sign_base.hit(unified_text, ".2731-.2734"),
                "long_roadmap_hit": sign_base.hit(
                    long_text,
                    "corrected 4D normalization update",
                ),
                "part5_hit": sign_base.hit(
                    part5_text,
                    "corrected 4D normalization update",
                ),
            },
            "inference": {
                "reserve_policy_is_now_one_surface": True,
                "why": (
                    "The theorem/normalization bottleneck and the farther-hybrid "
                    "hold rule now live on one explicit reserve-policy surface, so "
                    "the unresolved point is scientific judgement rather than raw "
                    "inventory."
                ),
            },
        },
    )
    declaration_paths = write_artifact("declaration_gate", declaration_payload)

    route_payload = {
        "generated_utc": sign_base.now_iso(),
        "phase": {
            "phase": 8,
            "step": "8.7.56.2746",
            "name": STEP_NAME + " route sync",
        },
        "inputs": declaration_paths,
        "rows": rows,
        "summary": summary,
        "decision": {
            "overall_status": "vector_qball_form_factor_updated_pack_hybrid_reserve_refresh_route_synced",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        "evidence": {
            "formulas": build_formulae(),
            "disposition": {
                "reserve_registry_followup_required": updated_pack_hybrid_reserve_registry_primary_followup_required,
                "farther_hybrid_still_reserve": (not farther_hybrid_continuation_reopen_required_now),
            },
        },
    }
    write_artifact("route_sync", route_payload)

    print(f"[done] {STEP_TAG} updated-pack hybrid reserve refresh audit completed")


if __name__ == "__main__":
    main()
