#!/usr/bin/env python3
"""Generate 8.7.56.2719-.2722 updated-pack exact static q0 current-theorem artifacts."""

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
        "8.7.56.2715-2718",
        "updated_pack_4d_current_vertex_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
PRIOR_AUDIT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.2711-2714",
        "updated_pack_exact_4d_current_vertex_completion_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
FOURD_NOTE = Path(r"C:\Users\ogawa\Downloads\trial2_vector_qball_4d_formfactor_20260330.md")

STEP_TAG = "8.7.56.2719-2722"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack exact static "
    "q0 current-theorem audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_exact_static_q0_current_theorem_audit",
    prefix="q",
)

PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_4d_"
    "current_vertex_audited_static_q0_theorem_primary_normalization_secondary_"
    "hybrid_reserve_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_static_"
    "q0_current_theorem_audited_corrected_4d_norm_secondary_gate"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_static_q0_"
    "theorem_gate_corrected_4d_normalization_refresh"
)
NEXT_ROUTE = "8.7.56.2723"
FOLLOWUP_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_corrected_4d_"
    "normalization_audit"
)
FOLLOWUP_ROUTE = "8.7.56.2727"


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


# 関数: static q0 theorem audit で使う式を返す。

def build_formulae(prior_formulas: dict[str, str]) -> dict[str, str]:
    """Return formulas used in the updated-pack exact static q0 current-theorem audit."""
    return {
        "corrected_operator_definition": prior_formulas["corrected_operator_definition"],
        "static_projection_target": prior_formulas["static_projection_target"],
        "static_zero_mode_definition": (
            "J_(0)^mu(r) := (omega / 2pi) int_0^(2pi/omega) dt J_eff^mu[Q](t,r)"
        ),
        "static_q0_fourier_projection": (
            "tilde J_eff^mu(q_0=0, q) = int d^3x exp(-i q · x) J_(0)^mu(x)"
        ),
        "exact_static_q0_theorem_target": (
            "derive the exact support/no-go verdict for J_(0)^mu from the exact "
            "frozen-action current vertex"
        ),
        "normalization_hold_rule": (
            "Do not canonize corrected 4D normalization until the exact static q0 "
            "current theorem is explicit."
        ),
    }


# 関数: `.2719-.2722` を実行する。

def main() -> None:
    """Execute the updated-pack exact static q0 current-theorem audit."""
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
        FOURD_NOTE,
    ):
        sign_base.require(path)

    status_text = sign_base.read_text(STATUS)
    roadmap_text = sign_base.read_text(ROADMAP)
    current_problem_text = sign_base.read_text(CURRENT_PROBLEM)
    current_status_text = sign_base.read_text(CURRENT_STATUS)
    unified_text = sign_base.read_text(UNIFIED_ROADMAP)
    long_text = sign_base.read_text(LONG_ROADMAP)
    part5_text = sign_base.read_text(PART5)
    note_text = sign_base.read_text(FOURD_NOTE)

    prior_gate_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    prior_audit_payload = sign_base.read_json(PRIOR_AUDIT)
    prior_audit_summary = prior_audit_payload["summary"]
    prior_formulas = prior_audit_payload["evidence"]["formulas"]

    updated_pack_exact_static_q0_current_theorem_audit_selected = bool(
        prior_gate_summary["gate_b_updated_pack_static_q0_current_theorem_promoted_next"]
        and prior_gate_summary["pack_update_required_now"]
    )
    current_pack_same_field_no_go_fixed = bool(
        prior_audit_summary["current_pack_same_field_no_go_fixed"]
    )
    failure_matrix_non_surrogate_guard_preserved = bool(
        prior_gate_summary["failure_matrix_non_surrogate_guard_preserved"]
    )
    updated_pack_exact_4d_current_vertex_machine_readable_now = bool(
        prior_audit_summary["updated_pack_exact_4d_current_vertex_machine_readable_now"]
    )
    corrected_note_static_part_requirement_explicit = bool(
        sign_base.hit(note_text, "static part") is not None
    )
    corrected_note_time_average_bypass_explicit = bool(
        sign_base.hit(note_text, "time average") is not None
    )
    corrected_note_delta_k0_static_projection_explicit = bool(
        sign_base.hit(note_text, "delta(k_0)") is not None
    )
    corrected_note_elastic_channel_q0_zero_explicit = bool(
        sign_base.hit(note_text, "elastic channel") is not None
    )
    updated_pack_static_q0_projection_target_surface_explicit = bool(
        updated_pack_exact_4d_current_vertex_machine_readable_now
        and corrected_note_static_part_requirement_explicit
        and corrected_note_time_average_bypass_explicit
        and corrected_note_delta_k0_static_projection_explicit
        and corrected_note_elastic_channel_q0_zero_explicit
    )
    updated_pack_static_q0_current_theorem_target_surface_explicit = bool(
        updated_pack_exact_static_q0_current_theorem_audit_selected
        and current_pack_same_field_no_go_fixed
        and failure_matrix_non_surrogate_guard_preserved
        and updated_pack_static_q0_projection_target_surface_explicit
    )
    updated_pack_static_q0_current_theorem_machine_readable_now = bool(
        updated_pack_static_q0_current_theorem_target_surface_explicit
    )
    exact_static_q0_current_theorem_available_now = False
    exact_static_q0_current_formula_available_now = False
    exact_static_q0_current_support_or_no_go_verdict_available_now = False
    updated_pack_static_q0_current_theorem_fully_localized_now = bool(
        updated_pack_static_q0_current_theorem_machine_readable_now
    )
    updated_pack_corrected_4d_normalization_primary_followup_required = bool(
        updated_pack_static_q0_current_theorem_fully_localized_now
        and (not exact_static_q0_current_theorem_available_now)
    )
    blind_vector_observable_gate_still_blocked = bool(
        prior_gate_summary["blind_vector_observable_gate_still_blocked"]
    )
    farther_hybrid_continuation_reopen_required_now = False

    rows = [
        sign_base.row(
            "updated_pack_exact_static_q0_current_theorem_audit_selected",
            "pass" if updated_pack_exact_static_q0_current_theorem_audit_selected else "reject",
            "updated-pack exact static q0 current-theorem audit selected",
            sign_base.truth(updated_pack_exact_static_q0_current_theorem_audit_selected),
            "The current-vertex gate already promoted the static q0 theorem lane as the next honest followup.",
        ),
        sign_base.row(
            "current_pack_same_field_no_go_fixed",
            "pass" if current_pack_same_field_no_go_fixed else "reject",
            "current-pack same-field no-go fixed before static q0 theorem audit",
            sign_base.truth(current_pack_same_field_no_go_fixed),
            "The static theorem lane starts only after the same-field source theorem is already closed on the exact zero / no-go branch.",
        ),
        sign_base.row(
            "failure_matrix_non_surrogate_guard_preserved",
            "pass" if failure_matrix_non_surrogate_guard_preserved else "reject",
            "failure-matrix non-surrogate guard preserved during static q0 theorem audit",
            sign_base.truth(failure_matrix_non_surrogate_guard_preserved),
            "The static q0 theorem lane remains admissible only while the exhausted scalar surrogate family stays closed.",
        ),
        sign_base.row(
            "updated_pack_exact_4d_current_vertex_machine_readable_now",
            "pass" if updated_pack_exact_4d_current_vertex_machine_readable_now else "reject",
            "updated-pack exact 4D current-vertex stack machine-readable now",
            sign_base.truth(updated_pack_exact_4d_current_vertex_machine_readable_now),
            "The static theorem lane can only start after the corrected frozen-action current/source target is already explicit.",
        ),
        sign_base.row(
            "corrected_note_static_part_requirement_explicit",
            "pass" if corrected_note_static_part_requirement_explicit else "reject",
            "corrected note static-part requirement explicit",
            sign_base.truth(corrected_note_static_part_requirement_explicit),
            "The note already isolates the static part as the object that must survive the time integral in the elastic channel.",
        ),
        sign_base.row(
            "corrected_note_time_average_bypass_explicit",
            "pass" if corrected_note_time_average_bypass_explicit else "reject",
            "corrected note time-average bypass explicit",
            sign_base.truth(corrected_note_time_average_bypass_explicit),
            "Premature time averaging remains explicitly forbidden because it would collapse the theorem target before the static projection is tested.",
        ),
        sign_base.row(
            "corrected_note_delta_k0_static_projection_explicit",
            "pass" if corrected_note_delta_k0_static_projection_explicit else "reject",
            "corrected note delta(k0) static projection explicit",
            sign_base.truth(corrected_note_delta_k0_static_projection_explicit),
            "The note explicitly writes the delta(k0) × static-part split that the theorem lane must audit.",
        ),
        sign_base.row(
            "corrected_note_elastic_channel_q0_zero_explicit",
            "pass" if corrected_note_elastic_channel_q0_zero_explicit else "reject",
            "corrected note elastic channel q0=0 explicit",
            sign_base.truth(corrected_note_elastic_channel_q0_zero_explicit),
            "The theorem lane is anchored to the elastic q0 = 0 channel rather than to a generic off-shell time structure.",
        ),
        sign_base.row(
            "updated_pack_static_q0_projection_target_surface_explicit",
            "pass" if updated_pack_static_q0_projection_target_surface_explicit else "reject",
            "updated-pack static q0 projection target surface explicit",
            sign_base.truth(updated_pack_static_q0_projection_target_surface_explicit),
            "The static projection target is now explicit: keep the full time structure, decompose harmonics, and isolate the zero-mode current.",
        ),
        sign_base.row(
            "updated_pack_static_q0_current_theorem_target_surface_explicit",
            "pass" if updated_pack_static_q0_current_theorem_target_surface_explicit else "reject",
            "updated-pack static q0 current-theorem target surface explicit",
            sign_base.truth(updated_pack_static_q0_current_theorem_target_surface_explicit),
            "The theorem target is now explicit: derive the support/no-go verdict for the elastic zero-mode current from the exact frozen-action current object.",
        ),
        sign_base.row(
            "updated_pack_static_q0_current_theorem_machine_readable_now",
            "pass" if updated_pack_static_q0_current_theorem_machine_readable_now else "reject",
            "updated-pack static q0 current-theorem stack machine-readable now",
            sign_base.truth(updated_pack_static_q0_current_theorem_machine_readable_now),
            "The zero-mode definition, elastic q0 = 0 projection, and no-premature-time-average rule now form one explicit theorem stack.",
        ),
        sign_base.row(
            "exact_static_q0_current_theorem_available_now",
            "pass" if exact_static_q0_current_theorem_available_now else "reject",
            "exact static q0 current theorem available now",
            sign_base.truth(exact_static_q0_current_theorem_available_now),
            "The theorem target is explicit, but the exact zero-mode current theorem itself is still not derived at this branch.",
        ),
        sign_base.row(
            "exact_static_q0_current_formula_available_now",
            "pass" if exact_static_q0_current_formula_available_now else "reject",
            "exact static q0 current formula available now",
            sign_base.truth(exact_static_q0_current_formula_available_now),
            "The branch still lacks the exact J_(0)^mu formula implied by the frozen-action current vertex.",
        ),
        sign_base.row(
            "exact_static_q0_current_support_or_no_go_verdict_available_now",
            "pass" if exact_static_q0_current_support_or_no_go_verdict_available_now else "reject",
            "exact static q0 current support/no-go verdict available now",
            sign_base.truth(exact_static_q0_current_support_or_no_go_verdict_available_now),
            "Because the exact zero-mode current is still absent, neither the support verdict nor the no-go verdict can yet be fixed theorem-level here.",
        ),
        sign_base.row(
            "updated_pack_static_q0_current_theorem_fully_localized_now",
            "pass" if updated_pack_static_q0_current_theorem_fully_localized_now else "reject",
            "updated-pack static q0 current theorem fully localized now",
            sign_base.truth(updated_pack_static_q0_current_theorem_fully_localized_now),
            "The theorem blocker is now localized to the exact zero-mode current formula/verdict rather than to the old scalar surrogate family.",
        ),
        sign_base.row(
            "updated_pack_corrected_4d_normalization_primary_followup_required",
            "pass" if updated_pack_corrected_4d_normalization_primary_followup_required else "reject",
            "updated-pack corrected 4D normalization primary followup required",
            sign_base.truth(updated_pack_corrected_4d_normalization_primary_followup_required),
            "Once the static theorem lane is localized, the next honest followup is the corrected 4D normalization audit kept downstream of that theorem.",
        ),
        sign_base.row(
            "blind_vector_observable_gate_still_blocked",
            "pass" if blind_vector_observable_gate_still_blocked else "reject",
            "blind-vector observable gate still blocked",
            sign_base.truth(blind_vector_observable_gate_still_blocked),
            "Blind-vector direct computation remains blocked because the exact elastic zero-mode current theorem is still absent.",
        ),
        sign_base.row(
            "farther_hybrid_continuation_reopen_required_now",
            "pass" if farther_hybrid_continuation_reopen_required_now else "reject",
            "farther hybrid continuation reopen required now",
            sign_base.truth(farther_hybrid_continuation_reopen_required_now),
            "Extra q-range evidence remains reserve-only because the blocker is still the 4D theorem stack, not residual-origin sampling.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "retained_scalar_residual_rel": float(prior_gate_summary["retained_scalar_residual_rel"]),
        "updated_pack_exact_static_q0_current_theorem_audit_selected": updated_pack_exact_static_q0_current_theorem_audit_selected,
        "current_pack_same_field_no_go_fixed": current_pack_same_field_no_go_fixed,
        "failure_matrix_non_surrogate_guard_preserved": failure_matrix_non_surrogate_guard_preserved,
        "updated_pack_exact_4d_current_vertex_machine_readable_now": updated_pack_exact_4d_current_vertex_machine_readable_now,
        "corrected_note_static_part_requirement_explicit": corrected_note_static_part_requirement_explicit,
        "corrected_note_time_average_bypass_explicit": corrected_note_time_average_bypass_explicit,
        "corrected_note_delta_k0_static_projection_explicit": corrected_note_delta_k0_static_projection_explicit,
        "corrected_note_elastic_channel_q0_zero_explicit": corrected_note_elastic_channel_q0_zero_explicit,
        "updated_pack_static_q0_projection_target_surface_explicit": updated_pack_static_q0_projection_target_surface_explicit,
        "updated_pack_static_q0_current_theorem_target_surface_explicit": updated_pack_static_q0_current_theorem_target_surface_explicit,
        "updated_pack_static_q0_current_theorem_machine_readable_now": updated_pack_static_q0_current_theorem_machine_readable_now,
        "exact_static_q0_current_theorem_available_now": exact_static_q0_current_theorem_available_now,
        "exact_static_q0_current_formula_available_now": exact_static_q0_current_formula_available_now,
        "exact_static_q0_current_support_or_no_go_verdict_available_now": exact_static_q0_current_support_or_no_go_verdict_available_now,
        "updated_pack_static_q0_current_theorem_fully_localized_now": updated_pack_static_q0_current_theorem_fully_localized_now,
        "updated_pack_corrected_4d_normalization_primary_followup_required": updated_pack_corrected_4d_normalization_primary_followup_required,
        "blind_vector_observable_gate_still_blocked": blind_vector_observable_gate_still_blocked,
        "farther_hybrid_continuation_reopen_required_now": farther_hybrid_continuation_reopen_required_now,
        "selected_primary_pack_update_surface": "updated_pack_exact_static_q0_current_theorem",
        "selected_secondary_pack_update_surface": "updated_pack_corrected_4d_normalization",
        "selected_reserve_completion_lane": "farther_hybrid_extra_q_range_only",
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": FOLLOWUP_ROUTE_NAME,
        "selected_followup_route_or_none": FOLLOWUP_ROUTE,
        "physical_reject_required": False,
    }

    declaration_payload = sign_base.payload(
        "8.7.56.2721",
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
                "expert_note": sign_base.display_path(FOURD_NOTE),
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
            "overall_status": "vector_qball_form_factor_updated_pack_exact_static_q0_current_theorem_declared",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "formulas": build_formulae(prior_formulas),
            "hits": {
                "status_branch_hit": sign_base.hit(status_text, "8.7.56.2719"),
                "roadmap_branch_hit": sign_base.hit(roadmap_text, ".2715-.2718"),
                "current_problem_hit": sign_base.hit(
                    current_problem_text,
                    "exact 4D current-vertex completion update",
                ),
                "current_status_hit": sign_base.hit(
                    current_status_text,
                    "exact 4D current-vertex completion update",
                ),
                "unified_roadmap_hit": sign_base.hit(unified_text, ".2715-.2718"),
                "long_roadmap_hit": sign_base.hit(long_text, ".2711-.2718"),
                "part5_hit": sign_base.hit(
                    part5_text,
                    "exact 4D current-vertex completion",
                ),
                "note_static_part_hit": sign_base.hit(note_text, "static part"),
                "note_time_average_hit": sign_base.hit(note_text, "time average"),
                "note_delta_k0_hit": sign_base.hit(note_text, "delta(k_0)"),
                "note_elastic_channel_hit": sign_base.hit(note_text, "elastic channel"),
            },
            "inference": {
                "static_q0_theorem_blocker_fully_localized_after_current_vertex_audit": True,
                "why": (
                    "The corrected 4D lane now makes the elastic q0 = 0 theorem target "
                    "explicit: isolate the zero-mode current from the exact frozen-action "
                    "current object and then decide support or no-go. What remains absent "
                    "is the exact zero-mode current formula/verdict itself."
                ),
            },
        },
    )
    declaration_paths = write_artifact("declaration_gate", declaration_payload)

    route_payload = {
        "generated_utc": sign_base.now_iso(),
        "phase": {
            "phase": 8,
            "step": "8.7.56.2722",
            "name": STEP_NAME + " route sync",
        },
        "inputs": declaration_paths,
        "rows": rows,
        "summary": summary,
        "decision": {
            "overall_status": "vector_qball_form_factor_updated_pack_exact_static_q0_current_theorem_route_synced",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        "evidence": {
            "formulas": build_formulae(prior_formulas),
            "disposition": {
                "exact_static_q0_current_theorem_available_now": exact_static_q0_current_theorem_available_now,
                "updated_pack_corrected_4d_normalization_primary_followup_required": updated_pack_corrected_4d_normalization_primary_followup_required,
                "farther_hybrid_still_reserve": (not farther_hybrid_continuation_reopen_required_now),
            },
        },
    }
    write_artifact("route_sync", route_payload)

    print(f"[done] {STEP_TAG} updated-pack exact static q0 current-theorem audit completed")


if __name__ == "__main__":
    main()
