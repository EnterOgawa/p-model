#!/usr/bin/env python3
"""Generate 8.7.56.2735-.2738 updated-pack 4D theorem-normalization pack-refresh artifacts."""

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
        "8.7.56.2731-2734",
        "updated_pack_corrected_4d_normalization_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
PRIOR_AUDIT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.2727-2730",
        "updated_pack_corrected_4d_normalization_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.2735-2738"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack 4D "
    "theorem-normalization pack-refresh audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_4d_theorem_normalization_pack_refresh_audit",
    prefix="q",
)

PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "corrected_4d_normalization_audited_pack_refresh_primary_hybrid_reserve_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "4d_theorem_normalization_pack_refresh_audited_hybrid_reserve_secondary_gate"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_4d_pack_refresh_"
    "gate_hybrid_reserve_refresh"
)
NEXT_ROUTE = "8.7.56.2739"
FOLLOWUP_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_hybrid_reserve_"
    "refresh_audit"
)
FOLLOWUP_ROUTE = "8.7.56.2743"


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


# 関数: theorem-normalization pack-refresh で使う式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used in the theorem-normalization pack-refresh audit."""
    return {
        "theorem_normalization_order": (
            "exact static q0 theorem -> surviving observable selection -> "
            "q_0 = 0 normalization -> alpha mapping -> hybrid reserve judgement"
        ),
        "current_ratio_candidate": (
            "F_4(|q|) := tilde J_(0)^0(q_0=0, |q|) / tilde J_(0)^0(0, 0)"
        ),
        "kernel_fallback_candidate": (
            "If J_(0)^0 = 0, shift to Pi^{mu nu}(q_0=0, q) normalization rather "
            "than canonizing a one-point form factor."
        ),
        "hybrid_reserve_rule": (
            "Keep farther hybrid continuation reserve-only until theorem and "
            "normalization jointly decide whether extra q-range evidence is still "
            "scientifically required."
        ),
    }


# 関数: `.2735-.2738` を実行する。

def main() -> None:
    """Execute the updated-pack 4D theorem-normalization pack-refresh audit."""
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

    updated_pack_4d_theorem_normalization_pack_refresh_audit_selected = bool(
        prior_gate_summary["gate_b_updated_pack_theorem_normalization_pack_refresh_promoted_next"]
        and prior_gate_summary["pack_update_required_now"]
    )
    updated_pack_static_q0_current_theorem_surface_retained = bool(
        prior_audit_summary["updated_pack_static_q0_current_theorem_machine_readable_now"]
    )
    updated_pack_corrected_4d_normalization_surface_retained = bool(
        prior_audit_summary["updated_pack_corrected_4d_normalization_machine_readable_now"]
    )
    updated_pack_surviving_observable_selection_surface_explicit = bool(
        updated_pack_static_q0_current_theorem_surface_retained
        and prior_audit_summary["updated_pack_current_ratio_normalization_candidate_explicit"]
        and prior_audit_summary["updated_pack_kernel_fallback_normalization_candidate_explicit"]
    )
    updated_pack_current_ratio_candidate_retained = bool(
        prior_audit_summary["updated_pack_current_ratio_normalization_candidate_explicit"]
    )
    updated_pack_kernel_fallback_candidate_retained = bool(
        prior_audit_summary["updated_pack_kernel_fallback_normalization_candidate_explicit"]
    )
    updated_pack_alpha_mapping_hold_surface_retained = bool(
        prior_audit_summary["updated_pack_alpha_mapping_hold_rule_explicit"]
    )
    updated_pack_4d_theorem_normalization_pack_refresh_target_surface_explicit = bool(
        updated_pack_4d_theorem_normalization_pack_refresh_audit_selected
        and updated_pack_static_q0_current_theorem_surface_retained
        and updated_pack_corrected_4d_normalization_surface_retained
        and updated_pack_surviving_observable_selection_surface_explicit
        and updated_pack_alpha_mapping_hold_surface_retained
    )
    updated_pack_4d_theorem_normalization_pack_refresh_machine_readable_now = bool(
        updated_pack_4d_theorem_normalization_pack_refresh_target_surface_explicit
    )
    exact_static_q0_current_theorem_available_now = False
    exact_surviving_observable_selection_available_now = False
    exact_corrected_4d_normalization_available_now = False
    exact_corrected_4d_alpha_mapping_available_now = False
    updated_pack_hybrid_reserve_refresh_primary_followup_required = bool(
        updated_pack_4d_theorem_normalization_pack_refresh_machine_readable_now
        and (not exact_surviving_observable_selection_available_now)
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
            "updated_pack_4d_theorem_normalization_pack_refresh_audit_selected",
            "pass"
            if updated_pack_4d_theorem_normalization_pack_refresh_audit_selected
            else "reject",
            "updated-pack 4D theorem-normalization pack-refresh audit selected",
            sign_base.truth(
                updated_pack_4d_theorem_normalization_pack_refresh_audit_selected
            ),
            "The corrected normalization gate already promoted a theorem/normalization pack refresh as the next honest downstream lane.",
        ),
        sign_base.row(
            "updated_pack_static_q0_current_theorem_surface_retained",
            "pass" if updated_pack_static_q0_current_theorem_surface_retained else "reject",
            "updated-pack static q0 current theorem surface retained",
            sign_base.truth(updated_pack_static_q0_current_theorem_surface_retained),
            "The refresh keeps the zero-mode theorem surface as the upstream decision object rather than replacing it with a new surrogate.",
        ),
        sign_base.row(
            "updated_pack_corrected_4d_normalization_surface_retained",
            "pass" if updated_pack_corrected_4d_normalization_surface_retained else "reject",
            "updated-pack corrected 4D normalization surface retained",
            sign_base.truth(updated_pack_corrected_4d_normalization_surface_retained),
            "The refresh keeps the provisional normalization stack alive without yet canonizing it.",
        ),
        sign_base.row(
            "updated_pack_surviving_observable_selection_surface_explicit",
            "pass" if updated_pack_surviving_observable_selection_surface_explicit else "reject",
            "updated-pack surviving-observable selection surface explicit",
            sign_base.truth(updated_pack_surviving_observable_selection_surface_explicit),
            "The current-ratio candidate and the kernel fallback now form one explicit observable-selection decision surface.",
        ),
        sign_base.row(
            "updated_pack_current_ratio_candidate_retained",
            "pass" if updated_pack_current_ratio_candidate_retained else "reject",
            "updated-pack current-ratio candidate retained",
            sign_base.truth(updated_pack_current_ratio_candidate_retained),
            "The one-point zero-mode current ratio remains the first candidate normalized observable.",
        ),
        sign_base.row(
            "updated_pack_kernel_fallback_candidate_retained",
            "pass" if updated_pack_kernel_fallback_candidate_retained else "reject",
            "updated-pack kernel fallback candidate retained",
            sign_base.truth(updated_pack_kernel_fallback_candidate_retained),
            "If the one-point zero mode vanishes, the q0 = 0 response kernel remains the only non-surrogate fallback.",
        ),
        sign_base.row(
            "updated_pack_alpha_mapping_hold_surface_retained",
            "pass" if updated_pack_alpha_mapping_hold_surface_retained else "reject",
            "updated-pack alpha mapping hold surface retained",
            sign_base.truth(updated_pack_alpha_mapping_hold_surface_retained),
            "The branch retains the rule that alpha mapping stays downstream of theorem and normalization canonicalization.",
        ),
        sign_base.row(
            "updated_pack_4d_theorem_normalization_pack_refresh_target_surface_explicit",
            "pass"
            if updated_pack_4d_theorem_normalization_pack_refresh_target_surface_explicit
            else "reject",
            "updated-pack 4D theorem-normalization pack-refresh target surface explicit",
            sign_base.truth(
                updated_pack_4d_theorem_normalization_pack_refresh_target_surface_explicit
            ),
            "The refresh target is now explicit: theorem, observable selection, normalization, and only then alpha mapping or reserve judgement.",
        ),
        sign_base.row(
            "updated_pack_4d_theorem_normalization_pack_refresh_machine_readable_now",
            "pass"
            if updated_pack_4d_theorem_normalization_pack_refresh_machine_readable_now
            else "reject",
            "updated-pack 4D theorem-normalization pack-refresh machine-readable now",
            sign_base.truth(
                updated_pack_4d_theorem_normalization_pack_refresh_machine_readable_now
            ),
            "The theorem and normalization stacks now live on one explicit pack-refresh surface.",
        ),
        sign_base.row(
            "exact_static_q0_current_theorem_available_now",
            "pass" if exact_static_q0_current_theorem_available_now else "reject",
            "exact static q0 current theorem available now",
            sign_base.truth(exact_static_q0_current_theorem_available_now),
            "The refresh still cannot derive the exact zero-mode theorem itself.",
        ),
        sign_base.row(
            "exact_surviving_observable_selection_available_now",
            "pass" if exact_surviving_observable_selection_available_now else "reject",
            "exact surviving-observable selection available now",
            sign_base.truth(exact_surviving_observable_selection_available_now),
            "The branch still cannot decide theorem-level whether the current-ratio candidate survives or the kernel fallback must take over.",
        ),
        sign_base.row(
            "exact_corrected_4d_normalization_available_now",
            "pass" if exact_corrected_4d_normalization_available_now else "reject",
            "exact corrected 4D normalization available now",
            sign_base.truth(exact_corrected_4d_normalization_available_now),
            "Normalization remains unavailable because the surviving observable is still unresolved.",
        ),
        sign_base.row(
            "exact_corrected_4d_alpha_mapping_available_now",
            "pass" if exact_corrected_4d_alpha_mapping_available_now else "reject",
            "exact corrected 4D alpha mapping available now",
            sign_base.truth(exact_corrected_4d_alpha_mapping_available_now),
            "The alpha mapping stays secondary because the normalized observable has not yet been canonized.",
        ),
        sign_base.row(
            "updated_pack_hybrid_reserve_refresh_primary_followup_required",
            "pass" if updated_pack_hybrid_reserve_refresh_primary_followup_required else "reject",
            "updated-pack hybrid reserve refresh primary followup required",
            sign_base.truth(updated_pack_hybrid_reserve_refresh_primary_followup_required),
            "Once theorem/normalization refresh is fully localized but still unresolved, the next honest move is to refresh the remaining reserve bookkeeping rather than to reopen blind computation.",
        ),
        sign_base.row(
            "failure_matrix_non_surrogate_guard_preserved",
            "pass" if failure_matrix_non_surrogate_guard_preserved else "reject",
            "failure-matrix non-surrogate guard preserved",
            sign_base.truth(failure_matrix_non_surrogate_guard_preserved),
            "The pack-refresh lane still keeps the exhausted density/proxy/eigenvalue family closed.",
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
            "Extra q-range evidence remains reserve-only because the blocker is still theorem/normalization canonicalization rather than missing high-q support.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "retained_scalar_residual_rel": float(prior_gate_summary["retained_scalar_residual_rel"]),
        "updated_pack_4d_theorem_normalization_pack_refresh_audit_selected": updated_pack_4d_theorem_normalization_pack_refresh_audit_selected,
        "updated_pack_static_q0_current_theorem_surface_retained": updated_pack_static_q0_current_theorem_surface_retained,
        "updated_pack_corrected_4d_normalization_surface_retained": updated_pack_corrected_4d_normalization_surface_retained,
        "updated_pack_surviving_observable_selection_surface_explicit": updated_pack_surviving_observable_selection_surface_explicit,
        "updated_pack_current_ratio_candidate_retained": updated_pack_current_ratio_candidate_retained,
        "updated_pack_kernel_fallback_candidate_retained": updated_pack_kernel_fallback_candidate_retained,
        "updated_pack_alpha_mapping_hold_surface_retained": updated_pack_alpha_mapping_hold_surface_retained,
        "updated_pack_4d_theorem_normalization_pack_refresh_target_surface_explicit": updated_pack_4d_theorem_normalization_pack_refresh_target_surface_explicit,
        "updated_pack_4d_theorem_normalization_pack_refresh_machine_readable_now": updated_pack_4d_theorem_normalization_pack_refresh_machine_readable_now,
        "exact_static_q0_current_theorem_available_now": exact_static_q0_current_theorem_available_now,
        "exact_surviving_observable_selection_available_now": exact_surviving_observable_selection_available_now,
        "exact_corrected_4d_normalization_available_now": exact_corrected_4d_normalization_available_now,
        "exact_corrected_4d_alpha_mapping_available_now": exact_corrected_4d_alpha_mapping_available_now,
        "updated_pack_hybrid_reserve_refresh_primary_followup_required": updated_pack_hybrid_reserve_refresh_primary_followup_required,
        "failure_matrix_non_surrogate_guard_preserved": failure_matrix_non_surrogate_guard_preserved,
        "blind_vector_observable_gate_still_blocked": blind_vector_observable_gate_still_blocked,
        "farther_hybrid_continuation_reopen_required_now": farther_hybrid_continuation_reopen_required_now,
        "selected_primary_pack_update_surface": "updated_pack_4d_theorem_normalization_pack_refresh",
        "selected_secondary_pack_update_surface": "updated_pack_hybrid_reserve_refresh",
        "selected_reserve_completion_lane": "farther_hybrid_extra_q_range_only",
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": FOLLOWUP_ROUTE_NAME,
        "selected_followup_route_or_none": FOLLOWUP_ROUTE,
        "physical_reject_required": False,
    }

    declaration_payload = sign_base.payload(
        "8.7.56.2737",
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
            "overall_status": "vector_qball_form_factor_updated_pack_4d_theorem_normalization_pack_refresh_declared",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "formulas": build_formulae(),
            "hits": {
                "status_branch_hit": sign_base.hit(status_text, "8.7.56.2735"),
                "roadmap_branch_hit": sign_base.hit(roadmap_text, ".2727-.2730"),
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
                "theorem_normalization_refresh_is_now_one_surface": True,
                "why": (
                    "The zero-mode theorem stack, surviving-observable selection, "
                    "and corrected normalization stack now sit on one explicit "
                    "refresh surface, so the unresolved point is no longer raw "
                    "formula inventory but canonical selection and reserve judgement."
                ),
            },
        },
    )
    declaration_paths = write_artifact("declaration_gate", declaration_payload)

    route_payload = {
        "generated_utc": sign_base.now_iso(),
        "phase": {
            "phase": 8,
            "step": "8.7.56.2738",
            "name": STEP_NAME + " route sync",
        },
        "inputs": declaration_paths,
        "rows": rows,
        "summary": summary,
        "decision": {
            "overall_status": "vector_qball_form_factor_updated_pack_4d_theorem_normalization_pack_refresh_route_synced",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        "evidence": {
            "formulas": build_formulae(),
            "disposition": {
                "hybrid_reserve_followup_required": updated_pack_hybrid_reserve_refresh_primary_followup_required,
                "farther_hybrid_still_reserve": (not farther_hybrid_continuation_reopen_required_now),
            },
        },
    }
    write_artifact("route_sync", route_payload)

    print(f"[done] {STEP_TAG} updated-pack 4D theorem-normalization pack-refresh audit completed")


if __name__ == "__main__":
    main()
