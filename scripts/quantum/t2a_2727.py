#!/usr/bin/env python3
"""Generate 8.7.56.2727-.2730 updated-pack corrected 4D normalization artifacts."""

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
        "8.7.56.2723-2726",
        "updated_pack_static_q0_theorem_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
PRIOR_AUDIT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.2719-2722",
        "updated_pack_exact_static_q0_current_theorem_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
FOURD_NOTE = Path(r"C:\Users\ogawa\Downloads\trial2_vector_qball_4d_formfactor_20260330.md")

STEP_TAG = "8.7.56.2727-2730"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack corrected "
    "4D normalization audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_corrected_4d_normalization_audit",
    prefix="q",
)

PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_static_"
    "q0_theorem_audited_corrected_4d_normalization_primary_hybrid_reserve_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "corrected_4d_normalization_audited_pack_refresh_secondary_gate"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_corrected_4d_"
    "normalization_gate_pack_refresh"
)
NEXT_ROUTE = "8.7.56.2731"
FOLLOWUP_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_4d_theorem_"
    "normalization_pack_refresh_audit"
)
FOLLOWUP_ROUTE = "8.7.56.2735"


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


# 関数: corrected 4D normalization audit で使う式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used in the corrected 4D normalization audit."""
    return {
        "current_ratio_candidate": (
            "F_4(|q|) := tilde J_(0)^0(q_0=0, |q|) / tilde J_(0)^0(0, 0)"
        ),
        "kernel_fallback_candidate": (
            "If J_(0)^0 = 0, shift to Pi^{mu nu}(q_0=0, q) normalization rather "
            "than canonizing a one-point form factor."
        ),
        "provisional_alpha_mapping": "alpha_4D := F_4(q_*)^2 / (4 pi)",
        "normalization_hold_rule": (
            "Do not canonize corrected 4D normalization or alpha_4D until the "
            "exact static q0 theorem decides which normalized observable survives."
        ),
        "normalization_order": (
            "exact static q0 theorem -> choose surviving observable -> normalize "
            "at q_0 = 0 -> map to alpha"
        ),
    }


# 関数: `.2727-.2730` を実行する。

def main() -> None:
    """Execute the updated-pack corrected 4D normalization audit."""
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
    prior_audit_summary = sign_base.read_json(PRIOR_AUDIT)["summary"]

    updated_pack_corrected_4d_normalization_audit_selected = bool(
        prior_gate_summary["gate_b_updated_pack_corrected_4d_normalization_promoted_next"]
        and prior_gate_summary["pack_update_required_now"]
    )
    updated_pack_static_q0_current_theorem_machine_readable_now = bool(
        prior_audit_summary["updated_pack_static_q0_current_theorem_machine_readable_now"]
    )
    corrected_note_original_f4_formula_explicit = bool(
        sign_base.hit(note_text, r"F_4(q_\mu)") is not None
    )
    corrected_note_original_alpha_mapping_explicit = bool(
        sign_base.hit(note_text, r"\alpha_{\rm 4D}") is not None
    )
    updated_pack_current_ratio_normalization_candidate_explicit = bool(
        updated_pack_static_q0_current_theorem_machine_readable_now
    )
    updated_pack_kernel_fallback_normalization_candidate_explicit = bool(
        updated_pack_static_q0_current_theorem_machine_readable_now
    )
    updated_pack_alpha_mapping_hold_rule_explicit = bool(
        updated_pack_current_ratio_normalization_candidate_explicit
        and updated_pack_kernel_fallback_normalization_candidate_explicit
    )
    updated_pack_corrected_4d_normalization_target_surface_explicit = bool(
        updated_pack_corrected_4d_normalization_audit_selected
        and updated_pack_static_q0_current_theorem_machine_readable_now
        and corrected_note_original_f4_formula_explicit
        and corrected_note_original_alpha_mapping_explicit
        and updated_pack_alpha_mapping_hold_rule_explicit
    )
    updated_pack_corrected_4d_normalization_machine_readable_now = bool(
        updated_pack_corrected_4d_normalization_target_surface_explicit
    )
    exact_corrected_4d_normalization_available_now = False
    exact_corrected_4d_alpha_mapping_available_now = False
    corrected_4d_normalization_canonical_verdict_available_now = False
    updated_pack_corrected_4d_normalization_fully_localized_now = bool(
        updated_pack_corrected_4d_normalization_machine_readable_now
    )
    updated_pack_pack_refresh_primary_followup_required = bool(
        updated_pack_corrected_4d_normalization_fully_localized_now
        and (not exact_corrected_4d_normalization_available_now)
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
            "updated_pack_corrected_4d_normalization_audit_selected",
            "pass" if updated_pack_corrected_4d_normalization_audit_selected else "reject",
            "updated-pack corrected 4D normalization audit selected",
            sign_base.truth(updated_pack_corrected_4d_normalization_audit_selected),
            "The static q0 theorem gate already promoted corrected 4D normalization as the next honest downstream lane.",
        ),
        sign_base.row(
            "updated_pack_static_q0_current_theorem_machine_readable_now",
            "pass" if updated_pack_static_q0_current_theorem_machine_readable_now else "reject",
            "updated-pack static q0 theorem stack machine-readable now",
            sign_base.truth(updated_pack_static_q0_current_theorem_machine_readable_now),
            "Normalization is only admissible after the zero-mode theorem surface is already explicit.",
        ),
        sign_base.row(
            "corrected_note_original_f4_formula_explicit",
            "pass" if corrected_note_original_f4_formula_explicit else "reject",
            "corrected note original F_4 formula explicit",
            sign_base.truth(corrected_note_original_f4_formula_explicit),
            "The expert note still carries the provisional F_4(q_mu) definition that must be re-audited rather than accepted as canonical.",
        ),
        sign_base.row(
            "corrected_note_original_alpha_mapping_explicit",
            "pass" if corrected_note_original_alpha_mapping_explicit else "reject",
            "corrected note original alpha_4D mapping explicit",
            sign_base.truth(corrected_note_original_alpha_mapping_explicit),
            "The expert note already writes a provisional alpha_4D mapping, so the audit can now test whether that mapping is canonical or still conditional.",
        ),
        sign_base.row(
            "updated_pack_current_ratio_normalization_candidate_explicit",
            "pass" if updated_pack_current_ratio_normalization_candidate_explicit else "reject",
            "updated-pack current-ratio normalization candidate explicit",
            sign_base.truth(updated_pack_current_ratio_normalization_candidate_explicit),
            "The corrected lane now has a rank-matched first candidate: normalize the elastic zero-mode current against its q = 0 value.",
        ),
        sign_base.row(
            "updated_pack_kernel_fallback_normalization_candidate_explicit",
            "pass" if updated_pack_kernel_fallback_normalization_candidate_explicit else "reject",
            "updated-pack kernel fallback normalization candidate explicit",
            sign_base.truth(updated_pack_kernel_fallback_normalization_candidate_explicit),
            "If the zero-mode one-point current vanishes, normalization must shift to the q0 = 0 response kernel rather than back to a scalar surrogate.",
        ),
        sign_base.row(
            "updated_pack_alpha_mapping_hold_rule_explicit",
            "pass" if updated_pack_alpha_mapping_hold_rule_explicit else "reject",
            "updated-pack alpha mapping hold rule explicit",
            sign_base.truth(updated_pack_alpha_mapping_hold_rule_explicit),
            "The provisional alpha_4D mapping stays conditional until the exact static theorem chooses the surviving normalized observable.",
        ),
        sign_base.row(
            "updated_pack_corrected_4d_normalization_target_surface_explicit",
            "pass" if updated_pack_corrected_4d_normalization_target_surface_explicit else "reject",
            "updated-pack corrected 4D normalization target surface explicit",
            sign_base.truth(updated_pack_corrected_4d_normalization_target_surface_explicit),
            "The audit target is now explicit: theorem first, normalization second, alpha mapping third.",
        ),
        sign_base.row(
            "updated_pack_corrected_4d_normalization_machine_readable_now",
            "pass" if updated_pack_corrected_4d_normalization_machine_readable_now else "reject",
            "updated-pack corrected 4D normalization stack machine-readable now",
            sign_base.truth(updated_pack_corrected_4d_normalization_machine_readable_now),
            "The current-ratio candidate, kernel fallback, and alpha hold rule now form one explicit normalization stack.",
        ),
        sign_base.row(
            "exact_corrected_4d_normalization_available_now",
            "pass" if exact_corrected_4d_normalization_available_now else "reject",
            "exact corrected 4D normalization available now",
            sign_base.truth(exact_corrected_4d_normalization_available_now),
            "Because the exact static theorem is still absent, no corrected 4D normalization can yet be canonized here.",
        ),
        sign_base.row(
            "exact_corrected_4d_alpha_mapping_available_now",
            "pass" if exact_corrected_4d_alpha_mapping_available_now else "reject",
            "exact corrected 4D alpha mapping available now",
            sign_base.truth(exact_corrected_4d_alpha_mapping_available_now),
            "The alpha mapping remains conditional because the normalized observable itself is not yet fixed theorem-level.",
        ),
        sign_base.row(
            "corrected_4d_normalization_canonical_verdict_available_now",
            "pass" if corrected_4d_normalization_canonical_verdict_available_now else "reject",
            "corrected 4D normalization canonical verdict available now",
            sign_base.truth(corrected_4d_normalization_canonical_verdict_available_now),
            "The branch still cannot decide whether the current-ratio candidate survives or whether the kernel fallback must replace it.",
        ),
        sign_base.row(
            "updated_pack_corrected_4d_normalization_fully_localized_now",
            "pass" if updated_pack_corrected_4d_normalization_fully_localized_now else "reject",
            "updated-pack corrected 4D normalization fully localized now",
            sign_base.truth(updated_pack_corrected_4d_normalization_fully_localized_now),
            "The blocker is now localized to canonicalization of the normalized observable rather than to the old scalar F_4 ansatz itself.",
        ),
        sign_base.row(
            "updated_pack_pack_refresh_primary_followup_required",
            "pass" if updated_pack_pack_refresh_primary_followup_required else "reject",
            "updated-pack pack-refresh primary followup required",
            sign_base.truth(updated_pack_pack_refresh_primary_followup_required),
            "Because normalization is localized but still unavailable, the next honest followup is a theorem/normalization pack refresh rather than blind computation.",
        ),
        sign_base.row(
            "failure_matrix_non_surrogate_guard_preserved",
            "pass" if failure_matrix_non_surrogate_guard_preserved else "reject",
            "failure-matrix non-surrogate guard preserved",
            sign_base.truth(failure_matrix_non_surrogate_guard_preserved),
            "The corrected normalization lane still keeps the exhausted density/proxy/eigenvalue family closed.",
        ),
        sign_base.row(
            "blind_vector_observable_gate_still_blocked",
            "pass" if blind_vector_observable_gate_still_blocked else "reject",
            "blind-vector observable gate still blocked",
            sign_base.truth(blind_vector_observable_gate_still_blocked),
            "Blind-vector direct computation remains blocked until both theorem and normalization close honestly.",
        ),
        sign_base.row(
            "farther_hybrid_continuation_reopen_required_now",
            "pass" if farther_hybrid_continuation_reopen_required_now else "reject",
            "farther hybrid continuation reopen required now",
            sign_base.truth(farther_hybrid_continuation_reopen_required_now),
            "Extra q-range evidence remains reserve-only because the blocker is still theorem/normalization canonicalization.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "retained_scalar_residual_rel": float(prior_gate_summary["retained_scalar_residual_rel"]),
        "updated_pack_corrected_4d_normalization_audit_selected": updated_pack_corrected_4d_normalization_audit_selected,
        "updated_pack_static_q0_current_theorem_machine_readable_now": updated_pack_static_q0_current_theorem_machine_readable_now,
        "corrected_note_original_f4_formula_explicit": corrected_note_original_f4_formula_explicit,
        "corrected_note_original_alpha_mapping_explicit": corrected_note_original_alpha_mapping_explicit,
        "updated_pack_current_ratio_normalization_candidate_explicit": updated_pack_current_ratio_normalization_candidate_explicit,
        "updated_pack_kernel_fallback_normalization_candidate_explicit": updated_pack_kernel_fallback_normalization_candidate_explicit,
        "updated_pack_alpha_mapping_hold_rule_explicit": updated_pack_alpha_mapping_hold_rule_explicit,
        "updated_pack_corrected_4d_normalization_target_surface_explicit": updated_pack_corrected_4d_normalization_target_surface_explicit,
        "updated_pack_corrected_4d_normalization_machine_readable_now": updated_pack_corrected_4d_normalization_machine_readable_now,
        "exact_corrected_4d_normalization_available_now": exact_corrected_4d_normalization_available_now,
        "exact_corrected_4d_alpha_mapping_available_now": exact_corrected_4d_alpha_mapping_available_now,
        "corrected_4d_normalization_canonical_verdict_available_now": corrected_4d_normalization_canonical_verdict_available_now,
        "updated_pack_corrected_4d_normalization_fully_localized_now": updated_pack_corrected_4d_normalization_fully_localized_now,
        "updated_pack_pack_refresh_primary_followup_required": updated_pack_pack_refresh_primary_followup_required,
        "failure_matrix_non_surrogate_guard_preserved": failure_matrix_non_surrogate_guard_preserved,
        "blind_vector_observable_gate_still_blocked": blind_vector_observable_gate_still_blocked,
        "farther_hybrid_continuation_reopen_required_now": farther_hybrid_continuation_reopen_required_now,
        "selected_primary_pack_update_surface": "updated_pack_corrected_4d_normalization",
        "selected_secondary_pack_update_surface": "updated_pack_4d_theorem_normalization_pack_refresh",
        "selected_reserve_completion_lane": "farther_hybrid_extra_q_range_only",
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": FOLLOWUP_ROUTE_NAME,
        "selected_followup_route_or_none": FOLLOWUP_ROUTE,
        "physical_reject_required": False,
    }

    declaration_payload = sign_base.payload(
        "8.7.56.2729",
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
            "overall_status": "vector_qball_form_factor_updated_pack_corrected_4d_normalization_declared",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "formulas": build_formulae(),
            "hits": {
                "status_branch_hit": sign_base.hit(status_text, "8.7.56.2727"),
                "roadmap_branch_hit": sign_base.hit(roadmap_text, ".2723-.2726"),
                "current_problem_hit": sign_base.hit(
                    current_problem_text,
                    "exact static q0 current-theorem update",
                ),
                "current_status_hit": sign_base.hit(
                    current_status_text,
                    "exact static q0 current-theorem update",
                ),
                "unified_roadmap_hit": sign_base.hit(unified_text, ".2723-.2726"),
                "long_roadmap_hit": sign_base.hit(
                    long_text,
                    "exact static q0 current-theorem update",
                ),
                "part5_hit": sign_base.hit(
                    part5_text,
                    "exact static q0 current-theorem update",
                ),
                "note_f4_hit": sign_base.hit(note_text, r"F_4(q_\mu)"),
                "note_alpha_hit": sign_base.hit(note_text, r"\alpha_{\rm 4D}"),
            },
            "inference": {
                "corrected_4d_normalization_blocker_fully_localized_after_static_theorem_audit": True,
                "why": (
                    "The lane now makes normalization explicit without canonizing it. "
                    "What remains absent is the theorem-level decision about which "
                    "observable survives normalization at q_0 = 0, so the honest "
                    "followup is a theorem/normalization pack refresh."
                ),
            },
        },
    )
    declaration_paths = write_artifact("declaration_gate", declaration_payload)

    route_payload = {
        "generated_utc": sign_base.now_iso(),
        "phase": {
            "phase": 8,
            "step": "8.7.56.2730",
            "name": STEP_NAME + " route sync",
        },
        "inputs": declaration_paths,
        "rows": rows,
        "summary": summary,
        "decision": {
            "overall_status": "vector_qball_form_factor_updated_pack_corrected_4d_normalization_route_synced",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        "evidence": {
            "formulas": build_formulae(),
            "disposition": {
                "exact_corrected_4d_normalization_available_now": exact_corrected_4d_normalization_available_now,
                "updated_pack_pack_refresh_primary_followup_required": updated_pack_pack_refresh_primary_followup_required,
                "farther_hybrid_still_reserve": (not farther_hybrid_continuation_reopen_required_now),
            },
        },
    }
    write_artifact("route_sync", route_payload)

    print(f"[done] {STEP_TAG} updated-pack corrected 4D normalization audit completed")


if __name__ == "__main__":
    main()
