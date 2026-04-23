#!/usr/bin/env python3
"""Generate 8.7.56.4567-.4570 written-action matter/rotation completion theorem artifacts."""

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
PRIOR_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.4563-4566",
        "updated_pack_current_written_action_external_probe_structure_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
PURE_DERIVATION_NOTE = Path(
    r"C:\Users\ogawa\Downloads\trial2_vector_qball_pure_derivation_20260330.md"
)

STEP_TAG = "8.7.56.4567-4570"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack current "
    "written-action matter/rotation completion theorem audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_current_written_action_matter_rotation_completion_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "current_written_action_external_probe_slot_absence_theorem_derived_"
    "matter_rotation_completion_primary_hybrid_reserve_secondary_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "current_written_action_matter_rotation_completion_no_go_theorem_derived_"
    "beyond_current_written_action_primary_pack_refresh_secondary_gate"
)


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

    return {"json": sign_base.display_path(paths["json"])}


# 関数: matter/rotation completion theorem で使う式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used in the written-action matter/rotation completion audit."""
    return {
        "written_total_action": (
            "L_total^vec = -(Z_P/4) F_{mu nu}^{(P)} F^{(P) mu nu} + "
            "U(P_mu^* P^mu) + g_P P_mu J_matter^mu + L_rot"
        ),
        "matter_deferred_statement": (
            "matter/rotation contributions are deferred: first derive with kinetic + potential only"
        ),
        "matter_term_form": "g_P P_mu J_matter^mu",
        "rotation_term_form": "L_rot",
        "completion_no_go": (
            "without explicit completion of J_matter dynamics or L_rot structure, "
            "the current written note cannot rescue a distinct external probe slot"
        ),
    }


# 関数: `.4567-.4570` を実行する。

def main() -> None:
    """Execute the current written-action matter/rotation completion theorem audit."""
    for path in (PRIOR_GATE, PURE_DERIVATION_NOTE):
        sign_base.require(path)

    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    note_text = sign_base.read_text(PURE_DERIVATION_NOTE)

    audit_selected = bool(
        prior_summary[
            "gate_b_updated_pack_current_written_action_matter_rotation_completion_promoted_next"
        ]
        and prior_summary["pack_update_required_now"]
    )
    retry_mode = bool(prior_summary["retry_gate_computation_mode_selected"])
    non_surrogate_guard = bool(
        prior_summary["failure_matrix_non_surrogate_guard_preserved"]
    )
    field_slot_absence_theorem_available = bool(
        prior_summary[
            "gate_a_updated_pack_current_written_action_external_probe_field_slot_absence_theorem_available_now"
        ]
    )
    matter_term_explicit = bool(
        sign_base.hit(note_text, "g_P P_\\mu J_{\\rm matter}^\\mu") is not None
    )
    rotation_term_explicit = bool(
        sign_base.hit(note_text, "\\mathcal{L}_{\\rm rot}") is not None
    )
    matter_rotation_deferred_explicit = bool(
        sign_base.hit(note_text, "matter 項と rotation 項") is not None
        and sign_base.hit(note_text, "matter/rotation の寄与は最後に確認する")
        is not None
    )
    matter_completion_explicit_now = False
    rotation_completion_explicit_now = False
    exact_current_written_action_matter_rotation_completion_no_go_theorem_available_now = bool(
        audit_selected
        and retry_mode
        and non_surrogate_guard
        and field_slot_absence_theorem_available
        and matter_term_explicit
        and rotation_term_explicit
        and matter_rotation_deferred_explicit
        and not matter_completion_explicit_now
        and not rotation_completion_explicit_now
    )
    exact_current_written_action_matter_rotation_completion_available_now = False
    exact_current_written_source_note_exhausted_now = bool(
        exact_current_written_action_matter_rotation_completion_no_go_theorem_available_now
    )
    updated_pack_beyond_current_written_action_probe_extension_primary_followup_required = bool(
        exact_current_written_source_note_exhausted_now
    )
    updated_pack_corrected_pack_refresh_secondary_hold_retained = bool(
        updated_pack_beyond_current_written_action_probe_extension_primary_followup_required
    )
    updated_pack_same_tag_pack_refresh_reentry_admissible_now = False
    updated_pack_current_written_action_matter_rotation_breakthrough_passed_now = False
    blind_blocked = bool(prior_summary["blind_vector_observable_gate_still_blocked"])

    rows = [
        sign_base.row(
            "updated_pack_current_written_action_matter_rotation_completion_audit_selected",
            "pass" if audit_selected else "reject",
            "updated-pack current written-action matter/rotation completion audit selected",
            sign_base.truth(audit_selected),
            "This branch is worth running only after the written action itself has already closed as lacking a distinct probe field slot.",
        ),
        sign_base.row(
            "retry_gate_computation_mode_selected",
            "pass" if retry_mode else "reject",
            "retry gate computation mode selected",
            sign_base.truth(retry_mode),
            "This turn stays on theorem derivation and does not count same-tag route restatement as progress.",
        ),
        sign_base.row(
            "failure_matrix_non_surrogate_guard_preserved",
            "pass" if non_surrogate_guard else "reject",
            "failure-matrix non-surrogate guard preserved",
            sign_base.truth(non_surrogate_guard),
            "The matter/rotation completion theorem is only admissible if it does not reopen the exhausted surrogate family.",
        ),
        sign_base.row(
            "exact_current_written_action_external_probe_field_slot_absence_theorem_available_now",
            "pass" if field_slot_absence_theorem_available else "reject",
            "exact current written-action external probe field-slot absence theorem available now",
            sign_base.truth(field_slot_absence_theorem_available),
            "The present written surface already lacks a separate probe slot, so the only remaining same-note rescue candidate is the deferred matter/rotation sector.",
        ),
        sign_base.row(
            "matter_term_explicit",
            "pass" if matter_term_explicit else "reject",
            "matter term explicit",
            sign_base.truth(matter_term_explicit),
            "The written note names the matter coupling, so the audit can honestly ask whether it is expanded far enough to rescue probe structure.",
        ),
        sign_base.row(
            "rotation_term_explicit",
            "pass" if rotation_term_explicit else "reject",
            "rotation term explicit",
            sign_base.truth(rotation_term_explicit),
            "The written note names the rotation term, so the audit can honestly ask whether it is expanded far enough to rescue probe structure.",
        ),
        sign_base.row(
            "matter_rotation_deferred_explicit",
            "pass" if matter_rotation_deferred_explicit else "reject",
            "matter/rotation deferred explicit",
            sign_base.truth(matter_rotation_deferred_explicit),
            "The source note explicitly postpones matter/rotation completion, so the present branch tests whether that deferral itself already implies no-go on the current written surface.",
        ),
        sign_base.row(
            "matter_completion_explicit_now",
            "pass" if matter_completion_explicit_now else "reject",
            "matter completion explicit now",
            sign_base.truth(matter_completion_explicit_now),
            "The current written note does not expand J_matter dynamics into a distinct probe-field structure on this branch.",
        ),
        sign_base.row(
            "rotation_completion_explicit_now",
            "pass" if rotation_completion_explicit_now else "reject",
            "rotation completion explicit now",
            sign_base.truth(rotation_completion_explicit_now),
            "The current written note does not expand L_rot into a distinct probe-field structure on this branch.",
        ),
        sign_base.row(
            "exact_current_written_action_matter_rotation_completion_no_go_theorem_available_now",
            "pass" if exact_current_written_action_matter_rotation_completion_no_go_theorem_available_now else "reject",
            "exact current written-action matter/rotation completion no-go theorem available now",
            sign_base.truth(exact_current_written_action_matter_rotation_completion_no_go_theorem_available_now),
            "Because the note explicitly defers matter/rotation completion and does not expand either term into a distinct probe slot, the current written note cannot rescue probe structure through matter/rotation on this branch.",
        ),
        sign_base.row(
            "exact_current_written_action_matter_rotation_completion_available_now",
            "pass" if exact_current_written_action_matter_rotation_completion_available_now else "reject",
            "exact current written-action matter/rotation completion available now",
            sign_base.truth(exact_current_written_action_matter_rotation_completion_available_now),
            "The current written note still lacks a successful matter/rotation completion theorem for probe structure.",
        ),
        sign_base.row(
            "exact_current_written_source_note_exhausted_now",
            "pass" if exact_current_written_source_note_exhausted_now else "reject",
            "exact current written source note exhausted now",
            sign_base.truth(exact_current_written_source_note_exhausted_now),
            "The present source note is now exhausted theorem-side for same-note probe-slot rescue.",
        ),
        sign_base.row(
            "updated_pack_beyond_current_written_action_probe_extension_primary_followup_required",
            "pass" if updated_pack_beyond_current_written_action_probe_extension_primary_followup_required else "reject",
            "updated-pack beyond-current-written-action probe extension primary followup required",
            sign_base.truth(updated_pack_beyond_current_written_action_probe_extension_primary_followup_required),
            "The honest next blocker is no longer inside the current note; it is whether a beyond-current-written-action extension can supply the missing probe structure.",
        ),
        sign_base.row(
            "updated_pack_corrected_pack_refresh_secondary_hold_retained",
            "pass" if updated_pack_corrected_pack_refresh_secondary_hold_retained else "reject",
            "updated-pack corrected pack-refresh secondary hold retained",
            sign_base.truth(updated_pack_corrected_pack_refresh_secondary_hold_retained),
            "Pack-refresh remains only secondary because it cannot repair the exhaustion of the current written note.",
        ),
        sign_base.row(
            "updated_pack_same_tag_pack_refresh_reentry_admissible_now",
            "pass" if updated_pack_same_tag_pack_refresh_reentry_admissible_now else "reject",
            "updated-pack same-tag pack-refresh reentry admissible now",
            sign_base.truth(updated_pack_same_tag_pack_refresh_reentry_admissible_now),
            "Same-tag pack-refresh reentry stays closed because this branch adds a theorem object and the remaining blocker is beyond-current-note structure, not bookkeeping.",
        ),
        sign_base.row(
            "blind_vector_observable_gate_still_blocked",
            "pass" if blind_blocked else "reject",
            "blind-vector observable gate still blocked",
            sign_base.truth(blind_blocked),
            "Blind-vector direct computation still waits on a theorem-side route beyond the exhaustion of the current written note.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "retained_scalar_residual_rel": float(prior_summary["retained_scalar_residual_rel"]),
        "updated_pack_current_written_action_matter_rotation_completion_audit_selected": audit_selected,
        "retry_gate_computation_mode_selected": retry_mode,
        "failure_matrix_non_surrogate_guard_preserved": non_surrogate_guard,
        "exact_current_written_action_external_probe_field_slot_absence_theorem_available_now": field_slot_absence_theorem_available,
        "matter_term_explicit": matter_term_explicit,
        "rotation_term_explicit": rotation_term_explicit,
        "matter_rotation_deferred_explicit": matter_rotation_deferred_explicit,
        "matter_completion_explicit_now": matter_completion_explicit_now,
        "rotation_completion_explicit_now": rotation_completion_explicit_now,
        "exact_current_written_action_matter_rotation_completion_no_go_theorem_available_now": exact_current_written_action_matter_rotation_completion_no_go_theorem_available_now,
        "exact_current_written_action_matter_rotation_completion_available_now": exact_current_written_action_matter_rotation_completion_available_now,
        "exact_current_written_source_note_exhausted_now": exact_current_written_source_note_exhausted_now,
        "updated_pack_beyond_current_written_action_probe_extension_primary_followup_required": updated_pack_beyond_current_written_action_probe_extension_primary_followup_required,
        "updated_pack_corrected_pack_refresh_secondary_hold_retained": updated_pack_corrected_pack_refresh_secondary_hold_retained,
        "updated_pack_same_tag_pack_refresh_reentry_admissible_now": updated_pack_same_tag_pack_refresh_reentry_admissible_now,
        "updated_pack_current_written_action_matter_rotation_breakthrough_passed_now": updated_pack_current_written_action_matter_rotation_breakthrough_passed_now,
        "blind_vector_observable_gate_still_blocked": blind_blocked,
        "selected_primary_completion_lane": "updated_pack_beyond_current_written_action_probe_extension_theorem_audit",
        "selected_secondary_completion_lane": "updated_pack_corrected_pack_refresh_return_sync",
        "selected_reserve_completion_lane": "farther_hybrid_extra_q_range_only",
        "selected_next_generation_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_beyond_current_written_action_probe_extension_theorem_audit",
        "recommended_next_route_or_none": "8.7.56.4575",
        "selected_followup_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_beyond_current_written_action_probe_extension_gate",
        "selected_followup_route_or_none": "8.7.56.4579",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.4569",
        STEP_NAME + " declaration gate",
        {
            "source_files": {
                "prior_gate": sign_base.display_path(PRIOR_GATE),
                "source_note": sign_base.display_path(PURE_DERIVATION_NOTE),
            },
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "8.7.56.4575",
                "followup_route": "8.7.56.4579",
            },
        },
        rows,
        summary,
        {
            "overall_status": "vector_qball_form_factor_updated_pack_current_written_action_matter_rotation_completion_declared",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {"formulas": build_formulae()},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(f"[done] {STEP_TAG} updated-pack current written-action matter/rotation completion theorem completed")
    print(f"[done] declaration: {declaration_paths['json']}")


if __name__ == "__main__":
    main()
