#!/usr/bin/env python3
"""Generate 8.7.56.4559-.4562 written-action external-probe structure theorem artifacts."""

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
        "8.7.56.4555-4558",
        "updated_pack_distinct_external_probe_separation_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
PURE_DERIVATION_NOTE = Path(
    r"C:\Users\ogawa\Downloads\trial2_vector_qball_pure_derivation_20260330.md"
)

STEP_TAG = "8.7.56.4559-4562"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack current "
    "written-action external-probe structure theorem audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_current_written_action_external_probe_structure_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "corrected_distinct_probe_separation_no_go_theorem_derived_"
    "external_probe_structure_primary_hybrid_reserve_secondary_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "current_written_action_external_probe_slot_absence_theorem_derived_"
    "matter_rotation_completion_primary_pack_refresh_secondary_gate"
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


# 関数: written-action external-probe structure theorem で使う式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used in the written-action external-probe structure audit."""
    return {
        "written_total_action": (
            "L_total^vec = -(Z_P/4) F_{mu nu}^{(P)} F^{(P) mu nu} + "
            "U(P_mu^* P^mu) + g_P P_mu J_matter^mu + L_rot"
        ),
        "single_field_slot": "all explicit dynamical field dependence is written through P_mu only",
        "matter_source_term": "g_P P_mu J_matter^mu contains an external source current, not a second probe field slot",
        "rotation_placeholder": "L_rot is named but not expanded into a distinct probe-field structure in the current written note",
        "slot_absence_theorem": (
            "the current written action contains no explicit external probe "
            "field slot A_mu distinct from P_mu"
        ),
    }


# 関数: `.4559-.4562` を実行する。

def main() -> None:
    """Execute the current written-action external-probe structure theorem audit."""
    for path in (PRIOR_GATE, PURE_DERIVATION_NOTE):
        sign_base.require(path)

    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    note_text = sign_base.read_text(PURE_DERIVATION_NOTE)

    audit_selected = bool(
        prior_summary[
            "gate_b_updated_pack_current_written_action_external_probe_structure_promoted_next"
        ]
        and prior_summary["pack_update_required_now"]
    )
    retry_mode = bool(prior_summary["retry_gate_computation_mode_selected"])
    non_surrogate_guard = bool(
        prior_summary["failure_matrix_non_surrogate_guard_preserved"]
    )
    distinct_probe_no_go_available = bool(
        prior_summary[
            "gate_a_updated_pack_exact_corrected_distinct_external_probe_separation_no_go_available_now"
        ]
    )
    written_total_action_formula_explicit = bool(
        sign_base.hit(note_text, "\\mathcal{L}_{\\rm total}^{\\rm vec} =")
        is not None
    )
    matter_coupling_single_field_explicit = bool(
        sign_base.hit(note_text, "g_P P_\\mu J_{\\rm matter}^\\mu") is not None
    )
    rotation_placeholder_explicit = bool(
        sign_base.hit(note_text, "\\mathcal{L}_{\\rm rot}") is not None
    )
    matter_rotation_deferred_explicit = bool(
        sign_base.hit(note_text, "matter 項と rotation 項") is not None
        and sign_base.hit(note_text, "matter/rotation の寄与は最後に確認する")
        is not None
    )
    current_written_action_external_probe_field_slot_available_now = False
    exact_current_written_action_external_probe_field_slot_absence_theorem_available_now = bool(
        audit_selected
        and retry_mode
        and non_surrogate_guard
        and distinct_probe_no_go_available
        and written_total_action_formula_explicit
        and matter_coupling_single_field_explicit
        and rotation_placeholder_explicit
        and matter_rotation_deferred_explicit
        and not current_written_action_external_probe_field_slot_available_now
    )
    exact_current_written_action_external_probe_structure_available_now = False
    updated_pack_current_written_action_matter_rotation_completion_primary_followup_required = bool(
        exact_current_written_action_external_probe_field_slot_absence_theorem_available_now
    )
    updated_pack_corrected_pack_refresh_secondary_hold_retained = bool(
        updated_pack_current_written_action_matter_rotation_completion_primary_followup_required
    )
    updated_pack_same_tag_pack_refresh_reentry_admissible_now = False
    updated_pack_current_written_action_external_probe_breakthrough_passed_now = False
    blind_blocked = bool(prior_summary["blind_vector_observable_gate_still_blocked"])

    rows = [
        sign_base.row(
            "updated_pack_current_written_action_external_probe_structure_audit_selected",
            "pass" if audit_selected else "reject",
            "updated-pack current written-action external-probe structure audit selected",
            sign_base.truth(audit_selected),
            "This branch is worth running only after same-action distinct-probe separation has already closed as a no-go theorem.",
        ),
        sign_base.row(
            "retry_gate_computation_mode_selected",
            "pass" if retry_mode else "reject",
            "retry gate computation mode selected",
            sign_base.truth(retry_mode),
            "This turn stays on theorem derivation rather than counting another same-tag route sync as progress.",
        ),
        sign_base.row(
            "failure_matrix_non_surrogate_guard_preserved",
            "pass" if non_surrogate_guard else "reject",
            "failure-matrix non-surrogate guard preserved",
            sign_base.truth(non_surrogate_guard),
            "The written-action field-slot theorem is only admissible if it does not reopen the exhausted surrogate family.",
        ),
        sign_base.row(
            "exact_corrected_distinct_external_probe_separation_no_go_theorem_available_now",
            "pass" if distinct_probe_no_go_available else "reject",
            "exact corrected distinct external-probe separation no-go theorem available now",
            sign_base.truth(distinct_probe_no_go_available),
            "The same-action algebra is already exhausted, so the honest next question is the literal field-slot structure of the written action itself.",
        ),
        sign_base.row(
            "written_total_action_formula_explicit",
            "pass" if written_total_action_formula_explicit else "reject",
            "written total action formula explicit",
            sign_base.truth(written_total_action_formula_explicit),
            "The source note still writes the total vector action explicitly and exposes every currently available field slot on that written surface.",
        ),
        sign_base.row(
            "matter_coupling_single_field_explicit",
            "pass" if matter_coupling_single_field_explicit else "reject",
            "matter coupling single-field explicit",
            sign_base.truth(matter_coupling_single_field_explicit),
            "The written matter term couples J_matter directly to P_mu and therefore does not by itself create a second probe field slot.",
        ),
        sign_base.row(
            "rotation_placeholder_explicit",
            "pass" if rotation_placeholder_explicit else "reject",
            "rotation placeholder explicit",
            sign_base.truth(rotation_placeholder_explicit),
            "The written action names L_rot, but the note does not expand it into a distinct probe-field structure on this branch.",
        ),
        sign_base.row(
            "matter_rotation_deferred_explicit",
            "pass" if matter_rotation_deferred_explicit else "reject",
            "matter/rotation deferred explicit",
            sign_base.truth(matter_rotation_deferred_explicit),
            "The source note explicitly postpones matter/rotation completion, so the current written surface cannot yet claim a rescued probe slot from those terms.",
        ),
        sign_base.row(
            "current_written_action_external_probe_field_slot_available_now",
            "pass" if current_written_action_external_probe_field_slot_available_now else "reject",
            "current written-action external probe field slot available now",
            sign_base.truth(current_written_action_external_probe_field_slot_available_now),
            "No separate action-level probe field A_mu is written on the present source surface.",
        ),
        sign_base.row(
            "exact_current_written_action_external_probe_field_slot_absence_theorem_available_now",
            "pass" if exact_current_written_action_external_probe_field_slot_absence_theorem_available_now else "reject",
            "exact current written-action external probe field-slot absence theorem available now",
            sign_base.truth(exact_current_written_action_external_probe_field_slot_absence_theorem_available_now),
            "The current written action depends explicitly only on P_mu and source placeholders, so no distinct external probe field slot is available theorem-side.",
        ),
        sign_base.row(
            "exact_current_written_action_external_probe_structure_available_now",
            "pass" if exact_current_written_action_external_probe_structure_available_now else "reject",
            "exact current written-action external probe structure available now",
            sign_base.truth(exact_current_written_action_external_probe_structure_available_now),
            "The present written note still lacks a distinct action-level probe structure beyond the same single-field slot.",
        ),
        sign_base.row(
            "updated_pack_current_written_action_matter_rotation_completion_primary_followup_required",
            "pass" if updated_pack_current_written_action_matter_rotation_completion_primary_followup_required else "reject",
            "updated-pack current written-action matter/rotation completion primary followup required",
            sign_base.truth(updated_pack_current_written_action_matter_rotation_completion_primary_followup_required),
            "The honest next blocker is whether the deferred matter/rotation sector can introduce a distinct probe structure without leaving the written action surface.",
        ),
        sign_base.row(
            "updated_pack_corrected_pack_refresh_secondary_hold_retained",
            "pass" if updated_pack_corrected_pack_refresh_secondary_hold_retained else "reject",
            "updated-pack corrected pack-refresh secondary hold retained",
            sign_base.truth(updated_pack_corrected_pack_refresh_secondary_hold_retained),
            "Pack-refresh remains only secondary because it cannot resolve the absence of a written probe field slot.",
        ),
        sign_base.row(
            "updated_pack_same_tag_pack_refresh_reentry_admissible_now",
            "pass" if updated_pack_same_tag_pack_refresh_reentry_admissible_now else "reject",
            "updated-pack same-tag pack-refresh reentry admissible now",
            sign_base.truth(updated_pack_same_tag_pack_refresh_reentry_admissible_now),
            "Same-tag pack-refresh reentry stays closed because this branch adds a theorem object and the remaining blocker is structural rather than bookkeeping.",
        ),
        sign_base.row(
            "blind_vector_observable_gate_still_blocked",
            "pass" if blind_blocked else "reject",
            "blind-vector observable gate still blocked",
            sign_base.truth(blind_blocked),
            "Blind-vector direct computation still waits on a theorem-side route beyond the current written-action field-slot absence theorem.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "retained_scalar_residual_rel": float(prior_summary["retained_scalar_residual_rel"]),
        "updated_pack_current_written_action_external_probe_structure_audit_selected": audit_selected,
        "retry_gate_computation_mode_selected": retry_mode,
        "failure_matrix_non_surrogate_guard_preserved": non_surrogate_guard,
        "exact_corrected_distinct_external_probe_separation_no_go_theorem_available_now": distinct_probe_no_go_available,
        "written_total_action_formula_explicit": written_total_action_formula_explicit,
        "matter_coupling_single_field_explicit": matter_coupling_single_field_explicit,
        "rotation_placeholder_explicit": rotation_placeholder_explicit,
        "matter_rotation_deferred_explicit": matter_rotation_deferred_explicit,
        "current_written_action_external_probe_field_slot_available_now": current_written_action_external_probe_field_slot_available_now,
        "exact_current_written_action_external_probe_field_slot_absence_theorem_available_now": exact_current_written_action_external_probe_field_slot_absence_theorem_available_now,
        "exact_current_written_action_external_probe_structure_available_now": exact_current_written_action_external_probe_structure_available_now,
        "updated_pack_current_written_action_matter_rotation_completion_primary_followup_required": updated_pack_current_written_action_matter_rotation_completion_primary_followup_required,
        "updated_pack_corrected_pack_refresh_secondary_hold_retained": updated_pack_corrected_pack_refresh_secondary_hold_retained,
        "updated_pack_same_tag_pack_refresh_reentry_admissible_now": updated_pack_same_tag_pack_refresh_reentry_admissible_now,
        "updated_pack_current_written_action_external_probe_breakthrough_passed_now": updated_pack_current_written_action_external_probe_breakthrough_passed_now,
        "blind_vector_observable_gate_still_blocked": blind_blocked,
        "selected_primary_completion_lane": "updated_pack_current_written_action_matter_rotation_completion_theorem_audit",
        "selected_secondary_completion_lane": "updated_pack_corrected_pack_refresh_return_sync",
        "selected_reserve_completion_lane": "farther_hybrid_extra_q_range_only",
        "selected_next_generation_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_current_written_action_matter_rotation_completion_theorem_audit",
        "recommended_next_route_or_none": "8.7.56.4567",
        "selected_followup_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_current_written_action_matter_rotation_completion_gate",
        "selected_followup_route_or_none": "8.7.56.4571",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.4561",
        STEP_NAME + " declaration gate",
        {
            "source_files": {
                "prior_gate": sign_base.display_path(PRIOR_GATE),
                "source_note": sign_base.display_path(PURE_DERIVATION_NOTE),
            },
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "8.7.56.4567",
                "followup_route": "8.7.56.4571",
            },
        },
        rows,
        summary,
        {
            "overall_status": "vector_qball_form_factor_updated_pack_current_written_action_external_probe_structure_declared",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {"formulas": build_formulae()},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(f"[done] {STEP_TAG} updated-pack current written-action external-probe structure theorem completed")
    print(f"[done] declaration: {declaration_paths['json']}")


if __name__ == "__main__":
    main()
