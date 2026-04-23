#!/usr/bin/env python3
"""Generate 8.7.56.4591-.4594 explicit nonadditive probe-slot extension theorem artifacts."""

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
        "8.7.56.4587-4590",
        "updated_pack_beyond_current_written_action_distinct_probe_slot_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
PRIOR_AUDIT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.4583-4586",
        "updated_pack_beyond_current_written_action_distinct_probe_slot_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
PURE_DERIVATION_NOTE = Path(
    r"C:\Users\ogawa\Downloads\trial2_vector_qball_pure_derivation_20260330.md"
)

STEP_TAG = "8.7.56.4591-4594"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack beyond-current-"
    "written-action explicit nonadditive probe-slot extension audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_beyond_current_written_action_explicit_nonadditive_extension_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "beyond_current_written_action_distinct_nonadditive_probe_slot_theorem_derived_"
    "explicit_extension_primary_hybrid_reserve_secondary_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "beyond_current_written_action_explicit_nonadditive_extension_decomposition_"
    "theorem_derived_extension_selector_primary_pack_refresh_secondary_gate"
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


# 関数: explicit extension theorem で使う式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used in the explicit nonadditive probe-slot extension audit."""
    return {
        "reduction_condition": "L_ext[P_mu, A_mu] |_(A_mu = 0) = L_total^vec[P_mu]",
        "probe_sector_definition": (
            "L_probe[A_mu] := L_ext[0, A_mu] - L_ext[0, 0]"
        ),
        "mixed_sector_definition": (
            "L_mix[P_mu, A_mu] := "
            "L_ext[P_mu, A_mu] - L_ext[P_mu, 0] - L_ext[0, A_mu] + L_ext[0, 0]"
        ),
        "explicit_decomposition": (
            "L_ext[P_mu, A_mu] = L_total^vec[P_mu] + L_probe[A_mu] + L_mix[P_mu, A_mu]"
        ),
        "mixed_sector_boundary": "L_mix[P_mu, 0] = 0 and L_mix[0, A_mu] = 0",
        "decoupled_no_go": (
            "L_mix = 0 => delta^2 S_ext / (delta P_mu delta A_nu) = 0, so the "
            "independent probe sector decouples from the Q-ball sector"
        ),
    }


# 関数: `.4591-.4594` を実行する。

def main() -> None:
    """Execute the explicit nonadditive probe-slot extension audit."""
    for path in (PRIOR_GATE, PRIOR_AUDIT, PURE_DERIVATION_NOTE):
        sign_base.require(path)

    prior_gate_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    prior_audit_summary = sign_base.read_json(PRIOR_AUDIT)["summary"]
    note_text = sign_base.read_text(PURE_DERIVATION_NOTE)

    audit_selected = bool(
        prior_gate_summary[
            "gate_b_updated_pack_explicit_nonadditive_probe_slot_extension_promoted_next"
        ]
        and prior_gate_summary["pack_update_required_now"]
    )
    retry_mode = bool(prior_gate_summary["retry_gate_computation_mode_selected"])
    non_surrogate_guard = bool(
        prior_gate_summary["failure_matrix_non_surrogate_guard_preserved"]
    )
    distinct_probe_slot_theorem_available = bool(
        prior_gate_summary[
            "gate_a_updated_pack_exact_beyond_current_written_action_distinct_probe_slot_theorem_available_now"
        ]
    )
    same_combination_no_go_available = bool(
        prior_audit_summary[
            "exact_beyond_current_written_action_same_combination_probe_slot_no_go_theorem_available_now"
        ]
    )
    nonadditive_slot_requirement_available = bool(
        prior_audit_summary[
            "exact_beyond_current_written_action_nonadditive_probe_slot_requirement_theorem_available_now"
        ]
    )
    base_action_formula_explicit = bool(
        sign_base.hit(note_text, "\\mathcal{L}_{\\rm total}^{\\rm vec} =")
        is not None
    )
    written_field_strength_block_explicit = bool(
        sign_base.hit(note_text, "F_{\\mu\\nu}^{(P)} =") is not None
        and sign_base.hit(note_text, "f_{\\mu\\nu} = \\partial_\\mu a_\\nu - \\partial_\\nu a_\\mu")
        is not None
    )
    written_scalar_block_explicit = bool(
        sign_base.hit(note_text, "U(\\Phi) = \\lambda(\\Phi - v^2)^2") is not None
        and sign_base.hit(note_text, "\\delta\\Phi_1 = Q_\\mu^* a^\\mu + a_\\mu^* Q^\\mu")
        is not None
    )
    honest_extension_reduction_condition_required = bool(
        audit_selected
        and retry_mode
        and non_surrogate_guard
        and distinct_probe_slot_theorem_available
        and same_combination_no_go_available
        and nonadditive_slot_requirement_available
    )
    exact_beyond_current_written_action_explicit_extension_decomposition_formula_available_now = bool(
        honest_extension_reduction_condition_required
        and base_action_formula_explicit
    )
    exact_beyond_current_written_action_explicit_probe_sector_formula_available_now = bool(
        exact_beyond_current_written_action_explicit_extension_decomposition_formula_available_now
        and written_field_strength_block_explicit
        and written_scalar_block_explicit
    )
    exact_beyond_current_written_action_explicit_nonadditive_mixed_sector_boundary_formula_available_now = bool(
        exact_beyond_current_written_action_explicit_extension_decomposition_formula_available_now
    )
    exact_beyond_current_written_action_decoupled_probe_sector_no_go_theorem_available_now = bool(
        exact_beyond_current_written_action_explicit_nonadditive_mixed_sector_boundary_formula_available_now
        and same_combination_no_go_available
    )
    exact_beyond_current_written_action_explicit_nonadditive_probe_extension_template_available_now = bool(
        exact_beyond_current_written_action_explicit_probe_sector_formula_available_now
        and exact_beyond_current_written_action_explicit_nonadditive_mixed_sector_boundary_formula_available_now
        and exact_beyond_current_written_action_decoupled_probe_sector_no_go_theorem_available_now
    )
    exact_beyond_current_written_action_probe_extension_available_now = False
    updated_pack_beyond_current_written_action_extension_selector_primary_followup_required = bool(
        exact_beyond_current_written_action_explicit_nonadditive_probe_extension_template_available_now
    )
    updated_pack_corrected_pack_refresh_secondary_hold_retained = bool(
        updated_pack_beyond_current_written_action_extension_selector_primary_followup_required
    )
    updated_pack_same_tag_pack_refresh_reentry_admissible_now = False
    updated_pack_beyond_current_written_action_explicit_extension_breakthrough_passed_now = False
    blind_blocked = bool(prior_gate_summary["blind_vector_observable_gate_still_blocked"])

    rows = [
        sign_base.row(
            "updated_pack_beyond_current_written_action_explicit_extension_audit_selected",
            "pass" if audit_selected else "reject",
            "updated-pack beyond-current-written-action explicit extension audit selected",
            sign_base.truth(audit_selected),
            "This branch is worth running only after the independent nonadditive probe-slot theorem has already closed and same-tag pack-refresh reentry remains closed.",
        ),
        sign_base.row(
            "retry_gate_computation_mode_selected",
            "pass" if retry_mode else "reject",
            "retry gate computation mode selected",
            sign_base.truth(retry_mode),
            "This turn must close a new theorem object, not merely re-state the exhausted reserve loop.",
        ),
        sign_base.row(
            "failure_matrix_non_surrogate_guard_preserved",
            "pass" if non_surrogate_guard else "reject",
            "failure-matrix non-surrogate guard preserved",
            sign_base.truth(non_surrogate_guard),
            "The explicit extension theorem is admissible only if it does not reopen the exhausted surrogate family.",
        ),
        sign_base.row(
            "exact_beyond_current_written_action_distinct_probe_slot_theorem_available_now",
            "pass" if distinct_probe_slot_theorem_available else "reject",
            "exact beyond-current-written-action distinct probe-slot theorem available now",
            sign_base.truth(distinct_probe_slot_theorem_available),
            "The previous branch already fixed that an honest distinct slot exists only on an extended nonadditive surface.",
        ),
        sign_base.row(
            "exact_beyond_current_written_action_same_combination_probe_slot_no_go_theorem_available_now",
            "pass" if same_combination_no_go_available else "reject",
            "exact beyond-current-written-action same-combination probe-slot no-go theorem available now",
            sign_base.truth(same_combination_no_go_available),
            "The extension cannot be written honestly as the same combination P_mu + A_mu, so the explicit extension must decompose into A-only and mixed sectors.",
        ),
        sign_base.row(
            "exact_beyond_current_written_action_nonadditive_probe_slot_requirement_theorem_available_now",
            "pass" if nonadditive_slot_requirement_available else "reject",
            "exact beyond-current-written-action nonadditive probe-slot requirement theorem available now",
            sign_base.truth(nonadditive_slot_requirement_available),
            "The explicit extension must keep A_mu variationally independent and therefore nonadditive with respect to P_mu.",
        ),
        sign_base.row(
            "base_action_formula_explicit",
            "pass" if base_action_formula_explicit else "reject",
            "base action formula explicit",
            sign_base.truth(base_action_formula_explicit),
            "The current note still writes the base vector action explicitly, so the extended action may be required to reduce back to that literal surface when A_mu is turned off.",
        ),
        sign_base.row(
            "written_field_strength_block_explicit",
            "pass" if written_field_strength_block_explicit else "reject",
            "written field-strength block explicit",
            sign_base.truth(written_field_strength_block_explicit),
            "The source note already supplies a field-strength kinetic building block, so an honest independent probe sector may be defined with the same structural type instead of a vague placeholder.",
        ),
        sign_base.row(
            "written_scalar_block_explicit",
            "pass" if written_scalar_block_explicit else "reject",
            "written scalar block explicit",
            sign_base.truth(written_scalar_block_explicit),
            "The source note already supplies a scalar-potential block, so an honest probe sector may also be stated at the action level without collapsing back to the same combination lane.",
        ),
        sign_base.row(
            "honest_extension_reduction_condition_required",
            "pass" if honest_extension_reduction_condition_required else "reject",
            "honest extension reduction condition required",
            sign_base.truth(honest_extension_reduction_condition_required),
            "Any beyond-current-written-action candidate must reduce to the written action when the new probe slot is turned off.",
        ),
        sign_base.row(
            "exact_beyond_current_written_action_explicit_extension_decomposition_formula_available_now",
            "pass" if exact_beyond_current_written_action_explicit_extension_decomposition_formula_available_now else "reject",
            "exact beyond-current-written-action explicit extension decomposition formula available now",
            sign_base.truth(
                exact_beyond_current_written_action_explicit_extension_decomposition_formula_available_now
            ),
            "The honest two-field extension now closes exactly as L_ext[P,A] = L_total[P] + L_probe[A] + L_mix[P,A], with the decomposition fixed by reduction to the written P-only surface.",
        ),
        sign_base.row(
            "exact_beyond_current_written_action_explicit_probe_sector_formula_available_now",
            "pass" if exact_beyond_current_written_action_explicit_probe_sector_formula_available_now else "reject",
            "exact beyond-current-written-action explicit probe-sector formula available now",
            sign_base.truth(
                exact_beyond_current_written_action_explicit_probe_sector_formula_available_now
            ),
            "The independent probe sector now has an explicit theorem-side definition L_probe[A] := L_ext[0,A] - L_ext[0,0], and the note already supplies kinetic and scalar building-block types that can carry it.",
        ),
        sign_base.row(
            "exact_beyond_current_written_action_explicit_nonadditive_mixed_sector_boundary_formula_available_now",
            "pass" if exact_beyond_current_written_action_explicit_nonadditive_mixed_sector_boundary_formula_available_now else "reject",
            "exact beyond-current-written-action explicit nonadditive mixed-sector boundary formula available now",
            sign_base.truth(
                exact_beyond_current_written_action_explicit_nonadditive_mixed_sector_boundary_formula_available_now
            ),
            "The nonadditive mixed sector is now fixed exactly as the remainder L_mix[P,A] := L_ext[P,A] - L_ext[P,0] - L_ext[0,A] + L_ext[0,0], with L_mix[P,0] = L_mix[0,A] = 0.",
        ),
        sign_base.row(
            "exact_beyond_current_written_action_decoupled_probe_sector_no_go_theorem_available_now",
            "pass" if exact_beyond_current_written_action_decoupled_probe_sector_no_go_theorem_available_now else "reject",
            "exact beyond-current-written-action decoupled probe-sector no-go theorem available now",
            sign_base.truth(
                exact_beyond_current_written_action_decoupled_probe_sector_no_go_theorem_available_now
            ),
            "If L_mix vanishes, all P-A mixed functional derivatives vanish and the new probe sector decouples from the Q-ball sector, so it cannot serve as the missing form-factor interaction lane.",
        ),
        sign_base.row(
            "exact_beyond_current_written_action_explicit_nonadditive_probe_extension_template_available_now",
            "pass" if exact_beyond_current_written_action_explicit_nonadditive_probe_extension_template_available_now else "reject",
            "exact beyond-current-written-action explicit nonadditive probe-extension template available now",
            sign_base.truth(
                exact_beyond_current_written_action_explicit_nonadditive_probe_extension_template_available_now
            ),
            "The minimal honest explicit extension now exists theorem-side as a two-field decomposition with an independent A-only sector and a nonzero mixed sector.",
        ),
        sign_base.row(
            "exact_beyond_current_written_action_probe_extension_available_now",
            "pass" if exact_beyond_current_written_action_probe_extension_available_now else "reject",
            "exact beyond-current-written-action probe extension available now",
            sign_base.truth(exact_beyond_current_written_action_probe_extension_available_now),
            "This branch closes the exact template theorem, not the fully selected extended action with canonical coefficients or selector.",
        ),
        sign_base.row(
            "updated_pack_beyond_current_written_action_extension_selector_primary_followup_required",
            "pass" if updated_pack_beyond_current_written_action_extension_selector_primary_followup_required else "reject",
            "updated-pack beyond-current-written-action extension selector primary followup required",
            sign_base.truth(
                updated_pack_beyond_current_written_action_extension_selector_primary_followup_required
            ),
            "The honest next blocker is to decide whether the explicit two-field extension can be canonically selected from the written theory, not to return to same-tag bookkeeping.",
        ),
        sign_base.row(
            "updated_pack_corrected_pack_refresh_secondary_hold_retained",
            "pass" if updated_pack_corrected_pack_refresh_secondary_hold_retained else "reject",
            "updated-pack corrected pack-refresh secondary hold retained",
            sign_base.truth(updated_pack_corrected_pack_refresh_secondary_hold_retained),
            "Pack-refresh remains only secondary because the live blocker is now extension selection rather than route syntax.",
        ),
        sign_base.row(
            "updated_pack_same_tag_pack_refresh_reentry_admissible_now",
            "pass" if updated_pack_same_tag_pack_refresh_reentry_admissible_now else "reject",
            "updated-pack same-tag pack-refresh reentry admissible now",
            sign_base.truth(updated_pack_same_tag_pack_refresh_reentry_admissible_now),
            "Same-tag pack-refresh reentry stays closed because this branch added a new theorem object and the remaining blocker is extension selection rather than bookkeeping.",
        ),
        sign_base.row(
            "blind_vector_observable_gate_still_blocked",
            "pass" if blind_blocked else "reject",
            "blind-vector observable gate still blocked",
            sign_base.truth(blind_blocked),
            "Blind-vector direct computation still waits on a concrete selected extension rather than the current theorem-side template alone.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "retained_scalar_residual_rel": float(prior_gate_summary["retained_scalar_residual_rel"]),
        "updated_pack_beyond_current_written_action_explicit_extension_audit_selected": audit_selected,
        "retry_gate_computation_mode_selected": retry_mode,
        "failure_matrix_non_surrogate_guard_preserved": non_surrogate_guard,
        "exact_beyond_current_written_action_distinct_probe_slot_theorem_available_now": distinct_probe_slot_theorem_available,
        "exact_beyond_current_written_action_same_combination_probe_slot_no_go_theorem_available_now": same_combination_no_go_available,
        "exact_beyond_current_written_action_nonadditive_probe_slot_requirement_theorem_available_now": nonadditive_slot_requirement_available,
        "base_action_formula_explicit": base_action_formula_explicit,
        "written_field_strength_block_explicit": written_field_strength_block_explicit,
        "written_scalar_block_explicit": written_scalar_block_explicit,
        "honest_extension_reduction_condition_required": honest_extension_reduction_condition_required,
        "exact_beyond_current_written_action_explicit_extension_decomposition_formula_available_now": exact_beyond_current_written_action_explicit_extension_decomposition_formula_available_now,
        "exact_beyond_current_written_action_explicit_probe_sector_formula_available_now": exact_beyond_current_written_action_explicit_probe_sector_formula_available_now,
        "exact_beyond_current_written_action_explicit_nonadditive_mixed_sector_boundary_formula_available_now": exact_beyond_current_written_action_explicit_nonadditive_mixed_sector_boundary_formula_available_now,
        "exact_beyond_current_written_action_decoupled_probe_sector_no_go_theorem_available_now": exact_beyond_current_written_action_decoupled_probe_sector_no_go_theorem_available_now,
        "exact_beyond_current_written_action_explicit_nonadditive_probe_extension_template_available_now": exact_beyond_current_written_action_explicit_nonadditive_probe_extension_template_available_now,
        "exact_beyond_current_written_action_probe_extension_available_now": exact_beyond_current_written_action_probe_extension_available_now,
        "updated_pack_beyond_current_written_action_extension_selector_primary_followup_required": updated_pack_beyond_current_written_action_extension_selector_primary_followup_required,
        "updated_pack_corrected_pack_refresh_secondary_hold_retained": updated_pack_corrected_pack_refresh_secondary_hold_retained,
        "updated_pack_same_tag_pack_refresh_reentry_admissible_now": updated_pack_same_tag_pack_refresh_reentry_admissible_now,
        "updated_pack_beyond_current_written_action_explicit_extension_breakthrough_passed_now": updated_pack_beyond_current_written_action_explicit_extension_breakthrough_passed_now,
        "blind_vector_observable_gate_still_blocked": blind_blocked,
        "selected_primary_completion_lane": "updated_pack_beyond_current_written_action_extension_selector_theorem_audit",
        "selected_secondary_completion_lane": "updated_pack_corrected_pack_refresh_return_sync",
        "selected_reserve_completion_lane": "farther_hybrid_extra_q_range_only",
        "selected_next_generation_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_beyond_current_written_action_extension_selector_theorem_audit",
        "recommended_next_route_or_none": "8.7.56.4595",
        "selected_followup_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_beyond_current_written_action_extension_selector_gate",
        "selected_followup_route_or_none": "8.7.56.4599",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.4593",
        STEP_NAME + " declaration gate",
        {
            "source_files": {
                "prior_gate": sign_base.display_path(PRIOR_GATE),
                "prior_audit": sign_base.display_path(PRIOR_AUDIT),
                "source_note": sign_base.display_path(PURE_DERIVATION_NOTE),
            },
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "8.7.56.4595",
                "followup_route": "8.7.56.4599",
            },
        },
        rows,
        summary,
        {
            "overall_status": "vector_qball_form_factor_updated_pack_beyond_current_written_action_explicit_extension_declared",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {"formulae": build_formulae()},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(
        f"[done] {STEP_TAG} updated-pack beyond-current-written-action explicit extension completed"
    )
    print(f"[done] declaration: {declaration_paths['json']}")


if __name__ == "__main__":
    main()
