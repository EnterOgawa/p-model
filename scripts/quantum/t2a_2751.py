#!/usr/bin/env python3
"""Generate 8.7.56.2751-.2754 updated-pack hybrid reserve registry artifacts."""

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
        "8.7.56.2747-2750",
        "updated_pack_hybrid_reserve_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
PRIOR_AUDIT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.2743-2746",
        "updated_pack_hybrid_reserve_refresh_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.2751-2754"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack hybrid "
    "reserve registry audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_hybrid_reserve_registry_audit",
    prefix="q",
)

PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "hybrid_reserve_refresh_audited_reserve_registry_refresh_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "hybrid_reserve_registry_audited_pack_refresh_secondary_gate"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_reserve_registry_"
    "gate_pack_refresh_sync"
)
NEXT_ROUTE = "8.7.56.2755"
FOLLOWUP_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_reserve_registry_"
    "pack_refresh_audit"
)
FOLLOWUP_ROUTE = "8.7.56.2759"


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


# 関数: reserve registry audit で使う式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used in the reserve registry audit."""
    return {
        "reserve_registry_role": (
            "Reserve registry := keep unresolved theorem / observable / normalization "
            "/ hybrid-judgement objects explicit without pretending that any one of "
            "them is already canonically closed."
        ),
        "registry_order": (
            "theorem-normalization pack refresh -> hybrid reserve judgement -> "
            "reserve registry -> pack-refresh sync -> only then extra q-range reopen"
        ),
        "reopen_rule": (
            "Farther hybrid continuation reopens only if exact residual-origin "
            "discrimination still requires extra q-range after the registry closes "
            "the unresolved state honestly."
        ),
    }


# 関数: `.2751-.2754` を実行する。

def main() -> None:
    """Execute the updated-pack hybrid reserve registry audit."""
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
        PRIOR_AUDIT,
    ):
        sign_base.require(path)

    status_text = sign_base.read_text(STATUS)
    roadmap_text = sign_base.read_text(ROADMAP)
    current_problem_text = sign_base.read_text(CURRENT_PROBLEM)
    current_status_text = sign_base.read_text(CURRENT_STATUS)
    unified_text = sign_base.read_text(UNIFIED_ROADMAP)
    long_text = sign_base.read_text(LONG_ROADMAP)
    part5_text = sign_base.read_text(PART5)

    prior_gate_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    prior_audit_summary = sign_base.read_json(PRIOR_AUDIT)["summary"]

    updated_pack_hybrid_reserve_registry_audit_selected = bool(
        prior_gate_summary["gate_b_updated_pack_hybrid_reserve_registry_refresh_promoted_next"]
        and prior_gate_summary["pack_update_required_now"]
    )
    updated_pack_hybrid_reserve_refresh_surface_retained = bool(
        prior_audit_summary["updated_pack_hybrid_reserve_refresh_machine_readable_now"]
    )
    updated_pack_farther_hybrid_hold_registry_explicit = bool(
        prior_audit_summary["updated_pack_farther_hybrid_extra_q_range_hold_explicit"]
        and prior_audit_summary["updated_pack_extra_q_range_reopen_condition_explicit"]
        and (not prior_gate_summary["gate_c_farther_hybrid_continuation_reopen_required_now"])
    )
    updated_pack_unresolved_exact_state_registry_explicit = bool(
        (not prior_audit_summary["exact_surviving_observable_selection_available_now"])
        and (not prior_audit_summary["exact_corrected_4d_normalization_available_now"])
        and (not prior_audit_summary["hybrid_reserve_judgement_available_now"])
        and (not prior_audit_summary["extra_q_range_evidence_required_now"])
    )
    updated_pack_reserve_registry_pack_refresh_followup_explicit = bool(
        prior_gate_summary["selected_secondary_completion_lane"]
        == "updated_pack_reserve_registry_gate_pack_refresh_sync"
    )
    updated_pack_hybrid_reserve_registry_target_surface_explicit = bool(
        updated_pack_hybrid_reserve_registry_audit_selected
        and updated_pack_hybrid_reserve_refresh_surface_retained
        and updated_pack_farther_hybrid_hold_registry_explicit
        and updated_pack_unresolved_exact_state_registry_explicit
        and updated_pack_reserve_registry_pack_refresh_followup_explicit
    )
    updated_pack_hybrid_reserve_registry_machine_readable_now = bool(
        updated_pack_hybrid_reserve_registry_target_surface_explicit
    )
    exact_surviving_observable_selection_available_now = False
    exact_corrected_4d_normalization_available_now = False
    hybrid_reserve_judgement_available_now = False
    extra_q_range_evidence_required_now = False
    updated_pack_reserve_registry_pack_refresh_primary_followup_required = bool(
        updated_pack_hybrid_reserve_registry_machine_readable_now
        and (not hybrid_reserve_judgement_available_now)
    )
    failure_matrix_non_surrogate_guard_preserved = bool(
        prior_gate_summary["failure_matrix_non_surrogate_guard_preserved"]
    )
    blind_vector_observable_gate_still_blocked = bool(
        prior_gate_summary["blind_vector_observable_gate_still_blocked"]
    )
    farther_hybrid_continuation_reopen_required_now = False

    rows = [
        sign_base.row(
            "updated_pack_hybrid_reserve_registry_audit_selected",
            "pass" if updated_pack_hybrid_reserve_registry_audit_selected else "reject",
            "updated-pack hybrid reserve registry audit selected",
            sign_base.truth(updated_pack_hybrid_reserve_registry_audit_selected),
            "Once reserve refresh is localized but unresolved, the next honest move is to register that unresolved state explicitly rather than reopen computation.",
        ),
        sign_base.row(
            "updated_pack_hybrid_reserve_refresh_surface_retained",
            "pass" if updated_pack_hybrid_reserve_refresh_surface_retained else "reject",
            "updated-pack hybrid reserve refresh surface retained",
            sign_base.truth(updated_pack_hybrid_reserve_refresh_surface_retained),
            "The registry inherits the already localized reserve-policy surface instead of replacing it with a new surrogate bookkeeping rule.",
        ),
        sign_base.row(
            "updated_pack_farther_hybrid_hold_registry_explicit",
            "pass" if updated_pack_farther_hybrid_hold_registry_explicit else "reject",
            "updated-pack farther hybrid hold registry explicit",
            sign_base.truth(updated_pack_farther_hybrid_hold_registry_explicit),
            "The registry keeps the farther hybrid lane in reserve-only status together with its controlled reopen condition.",
        ),
        sign_base.row(
            "updated_pack_unresolved_exact_state_registry_explicit",
            "pass" if updated_pack_unresolved_exact_state_registry_explicit else "reject",
            "updated-pack unresolved exact-state registry explicit",
            sign_base.truth(updated_pack_unresolved_exact_state_registry_explicit),
            "The registry records that surviving-observable selection, corrected normalization, reserve judgement, and extra q-range need all remain unresolved or false.",
        ),
        sign_base.row(
            "updated_pack_reserve_registry_pack_refresh_followup_explicit",
            "pass" if updated_pack_reserve_registry_pack_refresh_followup_explicit else "reject",
            "updated-pack reserve-registry pack-refresh followup explicit",
            sign_base.truth(updated_pack_reserve_registry_pack_refresh_followup_explicit),
            "The branch already knows that registry bookkeeping must feed back into one dedicated pack-refresh sync lane after this audit.",
        ),
        sign_base.row(
            "updated_pack_hybrid_reserve_registry_target_surface_explicit",
            "pass" if updated_pack_hybrid_reserve_registry_target_surface_explicit else "reject",
            "updated-pack hybrid reserve registry target surface explicit",
            sign_base.truth(updated_pack_hybrid_reserve_registry_target_surface_explicit),
            "The target surface is explicit: keep the unresolved theorem/normalization stack and the farther-hybrid hold rule on one registry object.",
        ),
        sign_base.row(
            "updated_pack_hybrid_reserve_registry_machine_readable_now",
            "pass" if updated_pack_hybrid_reserve_registry_machine_readable_now else "reject",
            "updated-pack hybrid reserve registry machine-readable now",
            sign_base.truth(updated_pack_hybrid_reserve_registry_machine_readable_now),
            "The reserve registry now lives on one explicit machine-readable surface.",
        ),
        sign_base.row(
            "exact_surviving_observable_selection_available_now",
            "pass" if exact_surviving_observable_selection_available_now else "reject",
            "exact surviving-observable selection available now",
            sign_base.truth(exact_surviving_observable_selection_available_now),
            "The registry does not solve the missing canonical observable selection by itself.",
        ),
        sign_base.row(
            "exact_corrected_4d_normalization_available_now",
            "pass" if exact_corrected_4d_normalization_available_now else "reject",
            "exact corrected 4D normalization available now",
            sign_base.truth(exact_corrected_4d_normalization_available_now),
            "Normalization remains unavailable because the zero-mode theorem and surviving observable are still unresolved.",
        ),
        sign_base.row(
            "hybrid_reserve_judgement_available_now",
            "pass" if hybrid_reserve_judgement_available_now else "reject",
            "hybrid reserve judgement available now",
            sign_base.truth(hybrid_reserve_judgement_available_now),
            "The registry keeps the scientific judgement pending instead of pretending that the retained hybrid evidence is already adjudicated.",
        ),
        sign_base.row(
            "extra_q_range_evidence_required_now",
            "pass" if extra_q_range_evidence_required_now else "reject",
            "extra q-range evidence required now",
            sign_base.truth(extra_q_range_evidence_required_now),
            "Current evidence still says that extra q-range is not scientifically required yet.",
        ),
        sign_base.row(
            "updated_pack_reserve_registry_pack_refresh_primary_followup_required",
            "pass"
            if updated_pack_reserve_registry_pack_refresh_primary_followup_required
            else "reject",
            "updated-pack reserve-registry pack-refresh primary followup required",
            sign_base.truth(updated_pack_reserve_registry_pack_refresh_primary_followup_required),
            "Once the unresolved state is registered honestly, the next honest move is to sync that registry back into the pack-refresh surface.",
        ),
        sign_base.row(
            "failure_matrix_non_surrogate_guard_preserved",
            "pass" if failure_matrix_non_surrogate_guard_preserved else "reject",
            "failure-matrix non-surrogate guard preserved",
            sign_base.truth(failure_matrix_non_surrogate_guard_preserved),
            "Reserve registry bookkeeping keeps the exhausted density/proxy/eigenvalue family closed.",
        ),
        sign_base.row(
            "blind_vector_observable_gate_still_blocked",
            "pass" if blind_vector_observable_gate_still_blocked else "reject",
            "blind-vector observable gate still blocked",
            sign_base.truth(blind_vector_observable_gate_still_blocked),
            "Blind-vector direct computation remains blocked until theorem, observable selection, normalization, and reserve judgement all close honestly.",
        ),
        sign_base.row(
            "farther_hybrid_continuation_reopen_required_now",
            "pass" if farther_hybrid_continuation_reopen_required_now else "reject",
            "farther hybrid continuation reopen required now",
            sign_base.truth(farther_hybrid_continuation_reopen_required_now),
            "The registry explicitly keeps farther hybrid continuation closed.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "retained_scalar_residual_rel": float(prior_gate_summary["retained_scalar_residual_rel"]),
        "updated_pack_hybrid_reserve_registry_audit_selected": updated_pack_hybrid_reserve_registry_audit_selected,
        "updated_pack_hybrid_reserve_refresh_surface_retained": updated_pack_hybrid_reserve_refresh_surface_retained,
        "updated_pack_farther_hybrid_hold_registry_explicit": updated_pack_farther_hybrid_hold_registry_explicit,
        "updated_pack_unresolved_exact_state_registry_explicit": updated_pack_unresolved_exact_state_registry_explicit,
        "updated_pack_reserve_registry_pack_refresh_followup_explicit": updated_pack_reserve_registry_pack_refresh_followup_explicit,
        "updated_pack_hybrid_reserve_registry_target_surface_explicit": updated_pack_hybrid_reserve_registry_target_surface_explicit,
        "updated_pack_hybrid_reserve_registry_machine_readable_now": updated_pack_hybrid_reserve_registry_machine_readable_now,
        "exact_surviving_observable_selection_available_now": exact_surviving_observable_selection_available_now,
        "exact_corrected_4d_normalization_available_now": exact_corrected_4d_normalization_available_now,
        "hybrid_reserve_judgement_available_now": hybrid_reserve_judgement_available_now,
        "extra_q_range_evidence_required_now": extra_q_range_evidence_required_now,
        "updated_pack_reserve_registry_pack_refresh_primary_followup_required": updated_pack_reserve_registry_pack_refresh_primary_followup_required,
        "failure_matrix_non_surrogate_guard_preserved": failure_matrix_non_surrogate_guard_preserved,
        "blind_vector_observable_gate_still_blocked": blind_vector_observable_gate_still_blocked,
        "farther_hybrid_continuation_reopen_required_now": farther_hybrid_continuation_reopen_required_now,
        "selected_primary_pack_update_surface": "updated_pack_hybrid_reserve_registry",
        "selected_secondary_pack_update_surface": "updated_pack_reserve_registry_pack_refresh_sync",
        "selected_reserve_completion_lane": "farther_hybrid_extra_q_range_only",
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": FOLLOWUP_ROUTE_NAME,
        "selected_followup_route_or_none": FOLLOWUP_ROUTE,
        "physical_reject_required": False,
    }

    declaration_payload = sign_base.payload(
        "8.7.56.2753",
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
                "prior_audit": sign_base.display_path(PRIOR_AUDIT),
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
            "overall_status": "vector_qball_form_factor_updated_pack_hybrid_reserve_registry_declared",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "formulas": build_formulae(),
            "hits": {
                "status_branch_hit": sign_base.hit(status_text, "8.7.56.2747"),
                "roadmap_branch_hit": sign_base.hit(roadmap_text, ".2747-.2750"),
                "current_problem_hit": sign_base.hit(
                    current_problem_text,
                    "4D theorem-normalization / hybrid reserve update",
                ),
                "current_status_hit": sign_base.hit(
                    current_status_text,
                    "4D theorem-normalization / hybrid reserve update",
                ),
                "unified_roadmap_hit": sign_base.hit(unified_text, ".2747-.2750"),
                "long_roadmap_hit": sign_base.hit(
                    long_text,
                    "hybrid reserve refresh update",
                ),
                "part5_hit": sign_base.hit(
                    part5_text,
                    "hybrid reserve refresh update",
                ),
            },
            "inference": {
                "reserve_registry_is_now_one_surface": True,
                "why": (
                    "The unresolved theorem/normalization stack and the farther-"
                    "hybrid hold rule can now be kept on one explicit registry "
                    "surface, so the next issue is synchronization rather than "
                    "new inventory."
                ),
            },
        },
    )
    declaration_paths = write_artifact("declaration_gate", declaration_payload)

    route_payload = {
        "generated_utc": sign_base.now_iso(),
        "phase": {
            "phase": 8,
            "step": "8.7.56.2754",
            "name": STEP_NAME + " route sync",
        },
        "inputs": declaration_paths,
        "rows": rows,
        "summary": summary,
        "decision": {
            "overall_status": "vector_qball_form_factor_updated_pack_hybrid_reserve_registry_route_synced",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        "evidence": {
            "formulas": build_formulae(),
            "disposition": {
                "reserve_registry_pack_refresh_followup_required": updated_pack_reserve_registry_pack_refresh_primary_followup_required,
                "farther_hybrid_still_reserve": (not farther_hybrid_continuation_reopen_required_now),
            },
        },
    }
    write_artifact("route_sync", route_payload)

    print(f"[done] {STEP_TAG} updated-pack hybrid reserve registry audit completed")


if __name__ == "__main__":
    main()
