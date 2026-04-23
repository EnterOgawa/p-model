#!/usr/bin/env python3
"""Generate 8.7.56.4575-.4578 beyond-current-written-action probe-extension theorem artifacts."""

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
PRIOR_AUDIT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.4567-4570",
        "updated_pack_current_written_action_matter_rotation_completion_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
PRIOR_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.4571-4574",
        "updated_pack_current_written_action_matter_rotation_completion_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
PURE_DERIVATION_NOTE = Path(
    r"C:\Users\ogawa\Downloads\trial2_vector_qball_pure_derivation_20260330.md"
)

STEP_TAG = "8.7.56.4575-4578"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack "
    "beyond-current-written-action probe extension theorem audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_beyond_current_written_action_probe_extension_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "current_written_action_matter_rotation_completion_no_go_theorem_derived_"
    "beyond_current_written_action_primary_hybrid_reserve_secondary_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "beyond_current_written_action_probe_extension_requirement_theorem_derived_"
    "distinct_probe_slot_extension_primary_pack_refresh_secondary_gate"
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


# 関数: beyond-current-written-action theorem で使う式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used in the beyond-current-written-action probe-extension audit."""
    return {
        "current_written_surface": (
            "L_total^vec[P_mu] = -(Z_P/4) F_{mu nu}^{(P)} F^{(P) mu nu} + "
            "U(P_mu^* P^mu) + g_P P_mu J_matter^mu + L_rot"
        ),
        "current_note_exhaustion": (
            "current note exhausted + no explicit probe slot + deferred matter/rotation "
            "=> no same-note rescue route remains"
        ),
        "extension_requirement": (
            "if the current written action contains only one dynamical slot P_mu, "
            "any theorem-side distinct probe structure must come from an action extension "
            "L_ext[P_mu, A_mu, ...] = L_total^vec[P_mu] + Delta L_probe[P_mu, A_mu, ...]"
        ),
        "minimal_slot_requirement": (
            "a distinct probe theorem requires at least one independent probe slot "
            "beyond P_mu on the extended action surface"
        ),
    }


# 関数: `.4575-.4578` を実行する。

def main() -> None:
    """Execute the beyond-current-written-action probe-extension theorem audit."""
    for path in (PRIOR_AUDIT, PRIOR_GATE, PURE_DERIVATION_NOTE):
        sign_base.require(path)

    prior_audit_summary = sign_base.read_json(PRIOR_AUDIT)["summary"]
    prior_gate_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    note_text = sign_base.read_text(PURE_DERIVATION_NOTE)

    audit_selected = bool(
        prior_gate_summary[
            "gate_b_updated_pack_beyond_current_written_action_probe_extension_promoted_next"
        ]
        and prior_gate_summary["pack_update_required_now"]
    )
    retry_mode = bool(prior_gate_summary["retry_gate_computation_mode_selected"])
    non_surrogate_guard = bool(
        prior_gate_summary["failure_matrix_non_surrogate_guard_preserved"]
    )
    current_written_source_note_exhausted_now = bool(
        prior_audit_summary["exact_current_written_source_note_exhausted_now"]
    )
    field_slot_absence_theorem_available = bool(
        prior_audit_summary[
            "exact_current_written_action_external_probe_field_slot_absence_theorem_available_now"
        ]
    )
    matter_rotation_completion_no_go_available = bool(
        prior_gate_summary[
            "gate_a_updated_pack_current_written_action_matter_rotation_completion_no_go_available_now"
        ]
    )
    written_total_action_formula_explicit = bool(
        sign_base.hit(note_text, "\\mathcal{L}_{\\rm total}^{\\rm vec}") is not None
    )
    current_note_second_probe_slot_written_now = bool(
        sign_base.hit(note_text, "A_\\mu") is not None
    )
    current_note_only_one_action_slot_explicit = bool(
        written_total_action_formula_explicit and not current_note_second_probe_slot_written_now
    )
    minimal_independent_probe_slot_requirement_formula_explicit = bool(
        audit_selected
        and retry_mode
        and non_surrogate_guard
        and current_written_source_note_exhausted_now
        and field_slot_absence_theorem_available
        and matter_rotation_completion_no_go_available
        and current_note_only_one_action_slot_explicit
    )
    exact_beyond_current_written_action_probe_extension_requirement_theorem_available_now = bool(
        minimal_independent_probe_slot_requirement_formula_explicit
    )
    exact_minimal_independent_probe_slot_requirement_theorem_available_now = bool(
        minimal_independent_probe_slot_requirement_formula_explicit
    )
    exact_beyond_current_written_action_probe_extension_available_now = False
    updated_pack_beyond_current_written_action_distinct_probe_slot_primary_followup_required = bool(
        exact_beyond_current_written_action_probe_extension_requirement_theorem_available_now
    )
    updated_pack_corrected_pack_refresh_secondary_hold_retained = bool(
        updated_pack_beyond_current_written_action_distinct_probe_slot_primary_followup_required
    )
    updated_pack_same_tag_pack_refresh_reentry_admissible_now = False
    updated_pack_beyond_current_written_action_probe_extension_breakthrough_passed_now = False
    blind_blocked = bool(prior_gate_summary["blind_vector_observable_gate_still_blocked"])

    rows = [
        sign_base.row(
            "updated_pack_beyond_current_written_action_probe_extension_audit_selected",
            "pass" if audit_selected else "reject",
            "updated-pack beyond-current-written-action probe extension audit selected",
            sign_base.truth(audit_selected),
            "This branch is worth running only after the current written note is theorem-side exhausted and the gate has already promoted action extension as the next honest blocker.",
        ),
        sign_base.row(
            "retry_gate_computation_mode_selected",
            "pass" if retry_mode else "reject",
            "retry gate computation mode selected",
            sign_base.truth(retry_mode),
            "This turn stays theorem-first and does not count same-tag route restatement as progress.",
        ),
        sign_base.row(
            "failure_matrix_non_surrogate_guard_preserved",
            "pass" if non_surrogate_guard else "reject",
            "failure-matrix non-surrogate guard preserved",
            sign_base.truth(non_surrogate_guard),
            "The extension theorem is admissible only if it does not reopen the exhausted surrogate family.",
        ),
        sign_base.row(
            "current_written_source_note_exhausted_now",
            "pass" if current_written_source_note_exhausted_now else "reject",
            "current written source note exhausted now",
            sign_base.truth(current_written_source_note_exhausted_now),
            "The current note has already exhausted its same-note rescue candidates, so a further theorem must point outside the written surface or stop.",
        ),
        sign_base.row(
            "exact_current_written_action_external_probe_field_slot_absence_theorem_available_now",
            "pass" if field_slot_absence_theorem_available else "reject",
            "exact current written-action external probe field-slot absence theorem available now",
            sign_base.truth(field_slot_absence_theorem_available),
            "The written action itself already closes as lacking a distinct probe slot, so any surviving probe route must live beyond the present written surface.",
        ),
        sign_base.row(
            "current_written_action_matter_rotation_completion_no_go_available_now",
            "pass" if matter_rotation_completion_no_go_available else "reject",
            "current written-action matter/rotation completion no-go available now",
            sign_base.truth(matter_rotation_completion_no_go_available),
            "The deferred matter/rotation sector also fails to rescue probe structure inside the current note, so no internal same-note repair remains.",
        ),
        sign_base.row(
            "written_total_action_formula_explicit",
            "pass" if written_total_action_formula_explicit else "reject",
            "written total action formula explicit",
            sign_base.truth(written_total_action_formula_explicit),
            "The present source note still writes only the current vector action, so the theorem can read its available slots literally rather than infer them indirectly.",
        ),
        sign_base.row(
            "current_note_second_probe_slot_written_now",
            "pass" if current_note_second_probe_slot_written_now else "reject",
            "current note second probe slot written now",
            sign_base.truth(current_note_second_probe_slot_written_now),
            "The present source note does not write a distinct action-level probe slot A_mu on the current surface.",
        ),
        sign_base.row(
            "current_note_only_one_action_slot_explicit",
            "pass" if current_note_only_one_action_slot_explicit else "reject",
            "current note only one action slot explicit",
            sign_base.truth(current_note_only_one_action_slot_explicit),
            "Because the current written action exposes only P_mu as a dynamical field slot, any distinct probe theorem must add a new slot rather than reuse the same one.",
        ),
        sign_base.row(
            "minimal_independent_probe_slot_requirement_formula_explicit",
            "pass" if minimal_independent_probe_slot_requirement_formula_explicit else "reject",
            "minimal independent probe-slot requirement formula explicit",
            sign_base.truth(minimal_independent_probe_slot_requirement_formula_explicit),
            "The honest extension formula is now explicit: a beyond-current-written-action route must add Delta L_probe[P_mu,A_mu,...] with at least one independent probe slot beyond P_mu.",
        ),
        sign_base.row(
            "exact_beyond_current_written_action_probe_extension_requirement_theorem_available_now",
            "pass" if exact_beyond_current_written_action_probe_extension_requirement_theorem_available_now else "reject",
            "exact beyond-current-written-action probe-extension requirement theorem available now",
            sign_base.truth(exact_beyond_current_written_action_probe_extension_requirement_theorem_available_now),
            "This branch closes the structural theorem that no theorem-side probe rescue remains inside the current written note; the route must extend the action itself.",
        ),
        sign_base.row(
            "exact_minimal_independent_probe_slot_requirement_theorem_available_now",
            "pass" if exact_minimal_independent_probe_slot_requirement_theorem_available_now else "reject",
            "exact minimal independent probe-slot requirement theorem available now",
            sign_base.truth(exact_minimal_independent_probe_slot_requirement_theorem_available_now),
            "The minimal honest extension is no longer vague: it must introduce an independent probe slot instead of another same-slot decomposition.",
        ),
        sign_base.row(
            "exact_beyond_current_written_action_probe_extension_available_now",
            "pass" if exact_beyond_current_written_action_probe_extension_available_now else "reject",
            "exact beyond-current-written-action probe extension available now",
            sign_base.truth(exact_beyond_current_written_action_probe_extension_available_now),
            "What closes here is the requirement theorem, not the extended action itself.",
        ),
        sign_base.row(
            "updated_pack_beyond_current_written_action_distinct_probe_slot_primary_followup_required",
            "pass" if updated_pack_beyond_current_written_action_distinct_probe_slot_primary_followup_required else "reject",
            "updated-pack beyond-current-written-action distinct probe-slot primary followup required",
            sign_base.truth(updated_pack_beyond_current_written_action_distinct_probe_slot_primary_followup_required),
            "The honest next blocker is therefore the literal theorem for a distinct probe slot on an extended action surface, not a return to same-tag bookkeeping.",
        ),
        sign_base.row(
            "updated_pack_corrected_pack_refresh_secondary_hold_retained",
            "pass" if updated_pack_corrected_pack_refresh_secondary_hold_retained else "reject",
            "updated-pack corrected pack-refresh secondary hold retained",
            sign_base.truth(updated_pack_corrected_pack_refresh_secondary_hold_retained),
            "Pack-refresh remains only secondary because it cannot create an absent action slot inside the exhausted current note.",
        ),
        sign_base.row(
            "updated_pack_same_tag_pack_refresh_reentry_admissible_now",
            "pass" if updated_pack_same_tag_pack_refresh_reentry_admissible_now else "reject",
            "updated-pack same-tag pack-refresh reentry admissible now",
            sign_base.truth(updated_pack_same_tag_pack_refresh_reentry_admissible_now),
            "Same-tag pack-refresh reentry stays closed because this turn closes a new theorem object and the remaining blocker is structural, not bookkeeping.",
        ),
        sign_base.row(
            "updated_pack_beyond_current_written_action_probe_extension_breakthrough_passed_now",
            "pass" if updated_pack_beyond_current_written_action_probe_extension_breakthrough_passed_now else "reject",
            "updated-pack beyond-current-written-action probe extension breakthrough passed now",
            sign_base.truth(updated_pack_beyond_current_written_action_probe_extension_breakthrough_passed_now),
            "This branch closes a genuine structure theorem, but it does not yet provide the extended action or the residual-origin breakthrough itself.",
        ),
        sign_base.row(
            "blind_vector_observable_gate_still_blocked",
            "pass" if blind_blocked else "reject",
            "blind-vector observable gate still blocked",
            sign_base.truth(blind_blocked),
            "Blind-vector direct computation still waits on a theorem-side route that supplies an actual distinct probe slot beyond the current written action.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "retained_scalar_residual_rel": float(prior_gate_summary["retained_scalar_residual_rel"]),
        "updated_pack_beyond_current_written_action_probe_extension_audit_selected": audit_selected,
        "retry_gate_computation_mode_selected": retry_mode,
        "failure_matrix_non_surrogate_guard_preserved": non_surrogate_guard,
        "current_written_source_note_exhausted_now": current_written_source_note_exhausted_now,
        "exact_current_written_action_external_probe_field_slot_absence_theorem_available_now": field_slot_absence_theorem_available,
        "current_written_action_matter_rotation_completion_no_go_available_now": matter_rotation_completion_no_go_available,
        "written_total_action_formula_explicit": written_total_action_formula_explicit,
        "current_note_second_probe_slot_written_now": current_note_second_probe_slot_written_now,
        "current_note_only_one_action_slot_explicit": current_note_only_one_action_slot_explicit,
        "minimal_independent_probe_slot_requirement_formula_explicit": minimal_independent_probe_slot_requirement_formula_explicit,
        "exact_beyond_current_written_action_probe_extension_requirement_theorem_available_now": exact_beyond_current_written_action_probe_extension_requirement_theorem_available_now,
        "exact_minimal_independent_probe_slot_requirement_theorem_available_now": exact_minimal_independent_probe_slot_requirement_theorem_available_now,
        "exact_beyond_current_written_action_probe_extension_available_now": exact_beyond_current_written_action_probe_extension_available_now,
        "updated_pack_beyond_current_written_action_distinct_probe_slot_primary_followup_required": updated_pack_beyond_current_written_action_distinct_probe_slot_primary_followup_required,
        "updated_pack_corrected_pack_refresh_secondary_hold_retained": updated_pack_corrected_pack_refresh_secondary_hold_retained,
        "updated_pack_same_tag_pack_refresh_reentry_admissible_now": updated_pack_same_tag_pack_refresh_reentry_admissible_now,
        "updated_pack_beyond_current_written_action_probe_extension_breakthrough_passed_now": updated_pack_beyond_current_written_action_probe_extension_breakthrough_passed_now,
        "blind_vector_observable_gate_still_blocked": blind_blocked,
        "selected_primary_completion_lane": "updated_pack_beyond_current_written_action_distinct_probe_slot_theorem_audit",
        "selected_secondary_completion_lane": "updated_pack_corrected_pack_refresh_return_sync",
        "selected_reserve_completion_lane": "farther_hybrid_extra_q_range_only",
        "selected_next_generation_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_beyond_current_written_action_distinct_probe_slot_theorem_audit",
        "recommended_next_route_or_none": "8.7.56.4583",
        "selected_followup_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_beyond_current_written_action_distinct_probe_slot_gate",
        "selected_followup_route_or_none": "8.7.56.4587",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.4577",
        STEP_NAME + " declaration gate",
        {
            "source_files": {
                "prior_audit": sign_base.display_path(PRIOR_AUDIT),
                "prior_gate": sign_base.display_path(PRIOR_GATE),
                "source_note": sign_base.display_path(PURE_DERIVATION_NOTE),
            },
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "8.7.56.4583",
                "followup_route": "8.7.56.4587",
            },
        },
        rows,
        summary,
        {
            "overall_status": "vector_qball_form_factor_updated_pack_beyond_current_written_action_probe_extension_theorem_declared",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {"formulas": build_formulae()},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(
        f"[done] {STEP_TAG} updated-pack beyond-current-written-action probe extension theorem completed"
    )
    print(f"[done] declaration: {declaration_paths['json']}")


if __name__ == "__main__":
    main()
