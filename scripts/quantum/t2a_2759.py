#!/usr/bin/env python3
"""Generate 8.7.56.2759-.2762 updated-pack reserve-registry pack-refresh artifacts."""

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
        "8.7.56.2755-2758",
        "updated_pack_reserve_registry_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
PRIOR_AUDIT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.2751-2754",
        "updated_pack_hybrid_reserve_registry_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.2759-2762"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack reserve-registry "
    "pack-refresh audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_reserve_registry_pack_refresh_audit",
    prefix="q",
)

PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "reserve_registry_audited_pack_refresh_sync_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "reserve_registry_pack_refresh_audited_hybrid_reserve_registry_sync_"
    "secondary_gate"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_pack_refresh_"
    "gate_hybrid_reserve_registry_sync"
)
NEXT_ROUTE = "8.7.56.2763"
FOLLOWUP_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_hybrid_reserve_"
    "registry_sync_audit"
)
FOLLOWUP_ROUTE = "8.7.56.2767"


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


# 関数: reserve-registry pack-refresh audit で使う式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used in the reserve-registry pack-refresh audit."""
    return {
        "pack_refresh_role": (
            "Pack-refresh sync := feed the unresolved theorem / observable / "
            "normalization registry back into one canonical updated-pack verdict "
            "surface without reopening exhausted surrogate families."
        ),
        "pack_refresh_order": (
            "reserve registry -> pack-refresh sync -> hybrid reserve registry sync "
            "-> reserve-policy gate -> only then extra q-range reopen"
        ),
        "hold_rule": (
            "Keep farther hybrid continuation closed unless the synchronized pack "
            "refresh later shows that residual-origin discrimination still needs "
            "extra q-range evidence."
        ),
    }


# 関数: `.2759-.2762` を実行する。

def main() -> None:
    """Execute the updated-pack reserve-registry pack-refresh audit."""
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

    updated_pack_reserve_registry_pack_refresh_audit_selected = bool(
        prior_gate_summary["gate_b_updated_pack_reserve_registry_pack_refresh_sync_promoted_next"]
        and prior_gate_summary["pack_update_required_now"]
    )
    updated_pack_hybrid_reserve_registry_surface_retained = bool(
        prior_audit_summary["updated_pack_hybrid_reserve_registry_machine_readable_now"]
    )
    updated_pack_unresolved_exact_state_registry_retained = bool(
        prior_audit_summary["updated_pack_unresolved_exact_state_registry_explicit"]
    )
    updated_pack_farther_hybrid_hold_surface_retained = bool(
        prior_audit_summary["updated_pack_farther_hybrid_hold_registry_explicit"]
        and (not prior_gate_summary["gate_c_farther_hybrid_continuation_reopen_required_now"])
    )
    updated_pack_pack_refresh_sync_surface_explicit = bool(
        updated_pack_reserve_registry_pack_refresh_audit_selected
        and updated_pack_hybrid_reserve_registry_surface_retained
        and updated_pack_unresolved_exact_state_registry_retained
        and updated_pack_farther_hybrid_hold_surface_retained
    )
    updated_pack_reserve_registry_pack_refresh_target_surface_explicit = bool(
        updated_pack_pack_refresh_sync_surface_explicit
        and prior_gate_summary["failure_matrix_non_surrogate_guard_preserved"]
    )
    updated_pack_reserve_registry_pack_refresh_machine_readable_now = bool(
        updated_pack_reserve_registry_pack_refresh_target_surface_explicit
    )
    exact_reserve_registry_closeout_available_now = False
    exact_pack_refresh_sync_verdict_available_now = False
    exact_hybrid_reserve_registry_sync_available_now = False
    extra_q_range_evidence_required_now = False
    updated_pack_hybrid_reserve_registry_sync_primary_followup_required = bool(
        updated_pack_reserve_registry_pack_refresh_machine_readable_now
        and (not exact_pack_refresh_sync_verdict_available_now)
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
            "updated_pack_reserve_registry_pack_refresh_audit_selected",
            "pass" if updated_pack_reserve_registry_pack_refresh_audit_selected else "reject",
            "updated-pack reserve-registry pack-refresh audit selected",
            sign_base.truth(updated_pack_reserve_registry_pack_refresh_audit_selected),
            "Once reserve-registry bookkeeping is explicit, the next honest move is to sync that unresolved state back into one pack-refresh verdict surface.",
        ),
        sign_base.row(
            "updated_pack_hybrid_reserve_registry_surface_retained",
            "pass" if updated_pack_hybrid_reserve_registry_surface_retained else "reject",
            "updated-pack hybrid reserve registry surface retained",
            sign_base.truth(updated_pack_hybrid_reserve_registry_surface_retained),
            "The pack-refresh audit inherits the machine-readable registry surface instead of rebuilding a new surrogate inventory.",
        ),
        sign_base.row(
            "updated_pack_unresolved_exact_state_registry_retained",
            "pass" if updated_pack_unresolved_exact_state_registry_retained else "reject",
            "updated-pack unresolved exact-state registry retained",
            sign_base.truth(updated_pack_unresolved_exact_state_registry_retained),
            "The audit keeps the unresolved theorem / observable / normalization / reserve-judgement stack explicit while syncing it into the updated-pack verdict surface.",
        ),
        sign_base.row(
            "updated_pack_farther_hybrid_hold_surface_retained",
            "pass" if updated_pack_farther_hybrid_hold_surface_retained else "reject",
            "updated-pack farther hybrid hold surface retained",
            sign_base.truth(updated_pack_farther_hybrid_hold_surface_retained),
            "The pack-refresh audit preserves the hold rule that farther hybrid continuation remains reserve-only.",
        ),
        sign_base.row(
            "updated_pack_pack_refresh_sync_surface_explicit",
            "pass" if updated_pack_pack_refresh_sync_surface_explicit else "reject",
            "updated-pack pack-refresh sync surface explicit",
            sign_base.truth(updated_pack_pack_refresh_sync_surface_explicit),
            "Registry bookkeeping and the farther-hybrid hold rule now sit on one explicit pack-refresh sync surface.",
        ),
        sign_base.row(
            "updated_pack_reserve_registry_pack_refresh_target_surface_explicit",
            "pass"
            if updated_pack_reserve_registry_pack_refresh_target_surface_explicit
            else "reject",
            "updated-pack reserve-registry pack-refresh target surface explicit",
            sign_base.truth(
                updated_pack_reserve_registry_pack_refresh_target_surface_explicit
            ),
            "The target surface is explicit: sync the unresolved registry back into a canonical updated-pack verdict while keeping the non-surrogate guard active.",
        ),
        sign_base.row(
            "updated_pack_reserve_registry_pack_refresh_machine_readable_now",
            "pass"
            if updated_pack_reserve_registry_pack_refresh_machine_readable_now
            else "reject",
            "updated-pack reserve-registry pack-refresh machine-readable now",
            sign_base.truth(
                updated_pack_reserve_registry_pack_refresh_machine_readable_now
            ),
            "The reserve-registry to pack-refresh synchronization now lives on one explicit machine-readable surface.",
        ),
        sign_base.row(
            "exact_reserve_registry_closeout_available_now",
            "pass" if exact_reserve_registry_closeout_available_now else "reject",
            "exact reserve-registry closeout available now",
            sign_base.truth(exact_reserve_registry_closeout_available_now),
            "The pack-refresh audit does not magically close the unresolved theorem and normalization stack.",
        ),
        sign_base.row(
            "exact_pack_refresh_sync_verdict_available_now",
            "pass" if exact_pack_refresh_sync_verdict_available_now else "reject",
            "exact pack-refresh sync verdict available now",
            sign_base.truth(exact_pack_refresh_sync_verdict_available_now),
            "The synchronized pack-refresh surface is explicit, but the canonical verdict itself is still unavailable.",
        ),
        sign_base.row(
            "exact_hybrid_reserve_registry_sync_available_now",
            "pass" if exact_hybrid_reserve_registry_sync_available_now else "reject",
            "exact hybrid-reserve registry sync available now",
            sign_base.truth(exact_hybrid_reserve_registry_sync_available_now),
            "The branch still lacks the exact hybrid-reserve registry sync that would justify reopening reserve-policy adjudication.",
        ),
        sign_base.row(
            "extra_q_range_evidence_required_now",
            "pass" if extra_q_range_evidence_required_now else "reject",
            "extra q-range evidence required now",
            sign_base.truth(extra_q_range_evidence_required_now),
            "Current evidence still says that extra q-range is not scientifically required yet.",
        ),
        sign_base.row(
            "updated_pack_hybrid_reserve_registry_sync_primary_followup_required",
            "pass"
            if updated_pack_hybrid_reserve_registry_sync_primary_followup_required
            else "reject",
            "updated-pack hybrid-reserve registry sync primary followup required",
            sign_base.truth(
                updated_pack_hybrid_reserve_registry_sync_primary_followup_required
            ),
            "Once the pack-refresh sync surface is explicit but unresolved, the next honest move is to sync the hybrid reserve registry back into the canonical reserve-policy lane.",
        ),
        sign_base.row(
            "failure_matrix_non_surrogate_guard_preserved",
            "pass" if failure_matrix_non_surrogate_guard_preserved else "reject",
            "failure-matrix non-surrogate guard preserved",
            sign_base.truth(failure_matrix_non_surrogate_guard_preserved),
            "Pack-refresh synchronization keeps the exhausted density/proxy/eigenvalue family closed.",
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
            "The pack-refresh audit explicitly keeps farther hybrid continuation closed.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "retained_scalar_residual_rel": float(prior_gate_summary["retained_scalar_residual_rel"]),
        "updated_pack_reserve_registry_pack_refresh_audit_selected": updated_pack_reserve_registry_pack_refresh_audit_selected,
        "updated_pack_hybrid_reserve_registry_surface_retained": updated_pack_hybrid_reserve_registry_surface_retained,
        "updated_pack_unresolved_exact_state_registry_retained": updated_pack_unresolved_exact_state_registry_retained,
        "updated_pack_farther_hybrid_hold_surface_retained": updated_pack_farther_hybrid_hold_surface_retained,
        "updated_pack_pack_refresh_sync_surface_explicit": updated_pack_pack_refresh_sync_surface_explicit,
        "updated_pack_reserve_registry_pack_refresh_target_surface_explicit": updated_pack_reserve_registry_pack_refresh_target_surface_explicit,
        "updated_pack_reserve_registry_pack_refresh_machine_readable_now": updated_pack_reserve_registry_pack_refresh_machine_readable_now,
        "exact_reserve_registry_closeout_available_now": exact_reserve_registry_closeout_available_now,
        "exact_pack_refresh_sync_verdict_available_now": exact_pack_refresh_sync_verdict_available_now,
        "exact_hybrid_reserve_registry_sync_available_now": exact_hybrid_reserve_registry_sync_available_now,
        "extra_q_range_evidence_required_now": extra_q_range_evidence_required_now,
        "updated_pack_hybrid_reserve_registry_sync_primary_followup_required": updated_pack_hybrid_reserve_registry_sync_primary_followup_required,
        "failure_matrix_non_surrogate_guard_preserved": failure_matrix_non_surrogate_guard_preserved,
        "blind_vector_observable_gate_still_blocked": blind_vector_observable_gate_still_blocked,
        "farther_hybrid_continuation_reopen_required_now": farther_hybrid_continuation_reopen_required_now,
        "selected_primary_pack_update_surface": "updated_pack_reserve_registry_pack_refresh",
        "selected_secondary_pack_update_surface": "updated_pack_hybrid_reserve_registry_sync",
        "selected_reserve_completion_lane": "farther_hybrid_extra_q_range_only",
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": FOLLOWUP_ROUTE_NAME,
        "selected_followup_route_or_none": FOLLOWUP_ROUTE,
        "physical_reject_required": False,
    }

    declaration_payload = sign_base.payload(
        "8.7.56.2761",
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
            "overall_status": "vector_qball_form_factor_updated_pack_reserve_registry_pack_refresh_declared",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "formulas": build_formulae(),
            "hits": {
                "status_branch_hit": sign_base.hit(status_text, "8.7.56.2755"),
                "roadmap_branch_hit": sign_base.hit(roadmap_text, ".2751-.2754"),
                "current_problem_hit": sign_base.hit(
                    current_problem_text,
                    "reserve-registry pack-refresh update",
                ),
                "current_status_hit": sign_base.hit(
                    current_status_text,
                    "reserve-registry pack-refresh update",
                ),
                "unified_roadmap_hit": sign_base.hit(unified_text, ".2755-.2758"),
                "long_roadmap_hit": sign_base.hit(
                    long_text,
                    "reserve-registry pack-refresh update",
                ),
                "part5_hit": sign_base.hit(
                    part5_text,
                    "reserve-registry pack-refresh update",
                ),
            },
            "inference": {
                "reserve_registry_now_feeds_pack_refresh": True,
                "why": (
                    "The unresolved registry and the farther-hybrid hold rule now "
                    "sit on one explicit pack-refresh sync surface, so the next "
                    "unresolved point is canonical reserve-policy synchronization "
                    "rather than more bookkeeping."
                ),
            },
        },
    )
    declaration_paths = write_artifact("declaration_gate", declaration_payload)

    route_payload = {
        "generated_utc": sign_base.now_iso(),
        "phase": {
            "phase": 8,
            "step": "8.7.56.2762",
            "name": STEP_NAME + " route sync",
        },
        "inputs": declaration_paths,
        "rows": rows,
        "summary": summary,
        "decision": {
            "overall_status": "vector_qball_form_factor_updated_pack_reserve_registry_pack_refresh_route_synced",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        "evidence": {
            "formulas": build_formulae(),
            "disposition": {
                "hybrid_reserve_registry_sync_followup_required": updated_pack_hybrid_reserve_registry_sync_primary_followup_required,
                "farther_hybrid_still_reserve": (not farther_hybrid_continuation_reopen_required_now),
            },
        },
    }
    write_artifact("route_sync", route_payload)

    print(f"[done] {STEP_TAG} updated-pack reserve-registry pack-refresh audit completed")


if __name__ == "__main__":
    main()
