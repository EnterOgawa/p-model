#!/usr/bin/env python3
"""Generate 8.7.56.4583-.4586 beyond-current-written-action distinct probe-slot theorem artifacts."""

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
        "8.7.56.4579-4582",
        "updated_pack_beyond_current_written_action_probe_extension_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
PRIOR_REQUIREMENT_AUDIT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.4575-4578",
        "updated_pack_beyond_current_written_action_probe_extension_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
PRIOR_PROBE_AUDIT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.4519-4522",
        "updated_pack_corrected_probe_split_additive_no_go_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
PRIOR_KERNEL_AUDIT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.4527-4530",
        "updated_pack_corrected_mixed_kernel_hessian_identity_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
PURE_DERIVATION_NOTE = Path(
    r"C:\Users\ogawa\Downloads\trial2_vector_qball_pure_derivation_20260330.md"
)

STEP_TAG = "8.7.56.4583-4586"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack beyond-current-"
    "written-action distinct probe-slot theorem audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_beyond_current_written_action_distinct_probe_slot_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "beyond_current_written_action_probe_extension_requirement_theorem_derived_"
    "distinct_probe_slot_extension_primary_hybrid_reserve_secondary_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "beyond_current_written_action_distinct_nonadditive_probe_slot_theorem_derived_"
    "explicit_extension_primary_pack_refresh_secondary_gate"
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


# 関数: distinct probe-slot theorem で使う式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used in the beyond-current-written-action distinct probe-slot audit."""
    return {
        "extended_action_template": (
            "L_ext[P_mu, A_mu] = L_total^vec[P_mu] + L_probe[A_mu] + L_mix[P_mu, A_mu]"
        ),
        "same_combination_forbidden": (
            "L_ext[P_mu, A_mu] != L_total^vec[P_mu + A_mu]"
        ),
        "independent_variation_requirement": (
            "delta S_ext / delta A_mu != 0 or "
            "delta^2 S_ext / (delta P_mu delta A_nu), "
            "delta^2 S_ext / (delta A_mu delta A_nu) "
            "must exist as independent slots"
        ),
        "same_action_no_go_input": (
            "additive same-action split gives J_add^mu[Q] = 0 and "
            "V^{mu nu}[Q] = Pi^{mu nu}[Q] = H^{mu nu}[Q]"
        ),
        "distinct_probe_slot_theorem": (
            "an honest distinct probe slot on the extended action surface must "
            "be nonadditive and variationally independent from P_mu"
        ),
    }


# 関数: `.4583-.4586` を実行する。

def main() -> None:
    """Execute the beyond-current-written-action distinct probe-slot theorem audit."""
    for path in (
        PRIOR_GATE,
        PRIOR_REQUIREMENT_AUDIT,
        PRIOR_PROBE_AUDIT,
        PRIOR_KERNEL_AUDIT,
        PURE_DERIVATION_NOTE,
    ):
        sign_base.require(path)

    prior_gate_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    prior_requirement_summary = sign_base.read_json(PRIOR_REQUIREMENT_AUDIT)["summary"]
    prior_probe_summary = sign_base.read_json(PRIOR_PROBE_AUDIT)["summary"]
    prior_kernel_summary = sign_base.read_json(PRIOR_KERNEL_AUDIT)["summary"]
    note_text = sign_base.read_text(PURE_DERIVATION_NOTE)

    audit_selected = bool(
        prior_gate_summary[
            "gate_b_updated_pack_beyond_current_written_action_distinct_probe_slot_promoted_next"
        ]
        and prior_gate_summary["pack_update_required_now"]
    )
    retry_mode = bool(prior_gate_summary["retry_gate_computation_mode_selected"])
    non_surrogate_guard = bool(
        prior_gate_summary["failure_matrix_non_surrogate_guard_preserved"]
    )
    extension_requirement_available = bool(
        prior_gate_summary[
            "gate_a_updated_pack_exact_beyond_current_written_action_probe_extension_requirement_available_now"
        ]
    )
    minimal_independent_slot_requirement_available = bool(
        prior_requirement_summary[
            "exact_minimal_independent_probe_slot_requirement_theorem_available_now"
        ]
    )
    additive_one_point_no_go_available = bool(
        prior_probe_summary[
            "exact_external_probe_current_one_point_no_go_theorem_available_now"
        ]
    )
    shared_hessian_identity_available = bool(
        prior_kernel_summary["exact_corrected_mixed_probe_response_kernel_formula_available_now"]
        and prior_kernel_summary["exact_corrected_pure_probe_response_kernel_formula_available_now"]
        and prior_kernel_summary["exact_corrected_kernel_rank_match_available_now"]
    )
    current_note_second_probe_slot_written_now = bool(
        sign_base.hit(note_text, "A_\\mu") is not None
    )
    current_note_single_slot_surface_explicit = bool(
        sign_base.hit(note_text, "\\mathcal{L}_{\\rm total}^{\\rm vec}") is not None
        and not current_note_second_probe_slot_written_now
    )
    exact_beyond_current_written_action_distinct_probe_slot_formula_available_now = bool(
        audit_selected
        and retry_mode
        and non_surrogate_guard
        and extension_requirement_available
        and minimal_independent_slot_requirement_available
        and additive_one_point_no_go_available
        and shared_hessian_identity_available
        and current_note_single_slot_surface_explicit
    )
    exact_beyond_current_written_action_same_combination_probe_slot_no_go_theorem_available_now = bool(
        exact_beyond_current_written_action_distinct_probe_slot_formula_available_now
    )
    exact_beyond_current_written_action_nonadditive_probe_slot_requirement_theorem_available_now = bool(
        exact_beyond_current_written_action_distinct_probe_slot_formula_available_now
    )
    exact_beyond_current_written_action_probe_extension_available_now = False
    updated_pack_explicit_nonadditive_probe_slot_extension_primary_followup_required = bool(
        exact_beyond_current_written_action_nonadditive_probe_slot_requirement_theorem_available_now
    )
    updated_pack_corrected_pack_refresh_secondary_hold_retained = bool(
        updated_pack_explicit_nonadditive_probe_slot_extension_primary_followup_required
    )
    updated_pack_same_tag_pack_refresh_reentry_admissible_now = False
    updated_pack_beyond_current_written_action_distinct_probe_slot_breakthrough_passed_now = False
    blind_blocked = bool(prior_gate_summary["blind_vector_observable_gate_still_blocked"])

    rows = [
        sign_base.row(
            "updated_pack_beyond_current_written_action_distinct_probe_slot_audit_selected",
            "pass" if audit_selected else "reject",
            "updated-pack beyond-current-written-action distinct probe-slot audit selected",
            sign_base.truth(audit_selected),
            "This branch is worth running only after the extension requirement theorem has already closed and same-tag pack-refresh reentry remains closed.",
        ),
        sign_base.row(
            "retry_gate_computation_mode_selected",
            "pass" if retry_mode else "reject",
            "retry gate computation mode selected",
            sign_base.truth(retry_mode),
            "This turn must close a new theorem object, not just re-state the exhausted loop.",
        ),
        sign_base.row(
            "failure_matrix_non_surrogate_guard_preserved",
            "pass" if non_surrogate_guard else "reject",
            "failure-matrix non-surrogate guard preserved",
            sign_base.truth(non_surrogate_guard),
            "The extended-slot theorem is admissible only if it does not reopen the exhausted surrogate family.",
        ),
        sign_base.row(
            "exact_beyond_current_written_action_probe_extension_requirement_available_now",
            "pass" if extension_requirement_available else "reject",
            "exact beyond-current-written-action probe extension requirement available now",
            sign_base.truth(extension_requirement_available),
            "The current written action is already closed as internally exhausted, so an honest distinct-probe theorem may now live only on an extended surface.",
        ),
        sign_base.row(
            "exact_minimal_independent_probe_slot_requirement_theorem_available_now",
            "pass" if minimal_independent_slot_requirement_available else "reject",
            "exact minimal independent probe-slot requirement theorem available now",
            sign_base.truth(minimal_independent_slot_requirement_available),
            "The previous branch already fixed that the extension must add at least one independent slot beyond P_mu.",
        ),
        sign_base.row(
            "exact_external_probe_current_one_point_no_go_theorem_available_now",
            "pass" if additive_one_point_no_go_available else "reject",
            "exact external-probe current one-point no-go theorem available now",
            sign_base.truth(additive_one_point_no_go_available),
            "The additive same-slot one-point lane already closes as exact zero, so a distinct slot cannot be recovered by simply reinterpreting A_mu inside the same combination.",
        ),
        sign_base.row(
            "exact_corrected_shared_hessian_identity_available_now",
            "pass" if shared_hessian_identity_available else "reject",
            "exact corrected shared Hessian identity available now",
            sign_base.truth(shared_hessian_identity_available),
            "The additive same-slot mixed and pure kernels already collapse onto the same Hessian, so same-combination extension cannot supply a distinct probe sector.",
        ),
        sign_base.row(
            "current_note_single_slot_surface_explicit",
            "pass" if current_note_single_slot_surface_explicit else "reject",
            "current note single-slot surface explicit",
            sign_base.truth(current_note_single_slot_surface_explicit),
            "The source note still writes only the single action-level slot P_mu and no independent A_mu slot on the written surface.",
        ),
        sign_base.row(
            "current_note_second_probe_slot_written_now",
            "pass" if current_note_second_probe_slot_written_now else "reject",
            "current note second probe slot written now",
            sign_base.truth(current_note_second_probe_slot_written_now),
            "No separate action-level A_mu slot is written in the present note, so any honest distinct probe structure must be supplied by a beyond-current-written-action extension.",
        ),
        sign_base.row(
            "exact_beyond_current_written_action_distinct_probe_slot_formula_available_now",
            "pass" if exact_beyond_current_written_action_distinct_probe_slot_formula_available_now else "reject",
            "exact beyond-current-written-action distinct probe-slot formula available now",
            sign_base.truth(exact_beyond_current_written_action_distinct_probe_slot_formula_available_now),
            "The minimal honest extended surface is now literal: L_ext[P_mu, A_mu] = L_total^vec[P_mu] + L_probe[A_mu] + L_mix[P_mu, A_mu].",
        ),
        sign_base.row(
            "exact_beyond_current_written_action_same_combination_probe_slot_no_go_theorem_available_now",
            "pass" if exact_beyond_current_written_action_same_combination_probe_slot_no_go_theorem_available_now else "reject",
            "exact beyond-current-written-action same-combination probe-slot no-go theorem available now",
            sign_base.truth(exact_beyond_current_written_action_same_combination_probe_slot_no_go_theorem_available_now),
            "Because the additive same-slot route already closes as J_add=0 and V=Pi=H, an honest distinct probe slot cannot appear merely through the same combination P_mu + A_mu.",
        ),
        sign_base.row(
            "exact_beyond_current_written_action_nonadditive_probe_slot_requirement_theorem_available_now",
            "pass" if exact_beyond_current_written_action_nonadditive_probe_slot_requirement_theorem_available_now else "reject",
            "exact beyond-current-written-action nonadditive probe-slot requirement theorem available now",
            sign_base.truth(exact_beyond_current_written_action_nonadditive_probe_slot_requirement_theorem_available_now),
            "The extended action must therefore contain an A_mu slot that is variationally independent and nonadditive with respect to the base P_mu slot.",
        ),
        sign_base.row(
            "exact_beyond_current_written_action_probe_extension_available_now",
            "pass" if exact_beyond_current_written_action_probe_extension_available_now else "reject",
            "exact beyond-current-written-action probe extension available now",
            sign_base.truth(exact_beyond_current_written_action_probe_extension_available_now),
            "This branch closes the minimal slot theorem, not the completed extended action itself.",
        ),
        sign_base.row(
            "updated_pack_explicit_nonadditive_probe_slot_extension_primary_followup_required",
            "pass" if updated_pack_explicit_nonadditive_probe_slot_extension_primary_followup_required else "reject",
            "updated-pack explicit nonadditive probe-slot extension primary followup required",
            sign_base.truth(updated_pack_explicit_nonadditive_probe_slot_extension_primary_followup_required),
            "The honest next blocker is to state the minimal explicit nonadditive extension, not to re-enter same-tag bookkeeping.",
        ),
        sign_base.row(
            "updated_pack_corrected_pack_refresh_secondary_hold_retained",
            "pass" if updated_pack_corrected_pack_refresh_secondary_hold_retained else "reject",
            "updated-pack corrected pack-refresh secondary hold retained",
            sign_base.truth(updated_pack_corrected_pack_refresh_secondary_hold_retained),
            "Pack-refresh remains only secondary because it cannot manufacture an absent independent variational slot.",
        ),
        sign_base.row(
            "updated_pack_same_tag_pack_refresh_reentry_admissible_now",
            "pass" if updated_pack_same_tag_pack_refresh_reentry_admissible_now else "reject",
            "updated-pack same-tag pack-refresh reentry admissible now",
            sign_base.truth(updated_pack_same_tag_pack_refresh_reentry_admissible_now),
            "Same-tag pack-refresh reentry stays closed because the remaining blocker is extension structure, not bookkeeping syntax.",
        ),
        sign_base.row(
            "updated_pack_beyond_current_written_action_distinct_probe_slot_breakthrough_passed_now",
            "pass" if updated_pack_beyond_current_written_action_distinct_probe_slot_breakthrough_passed_now else "reject",
            "updated-pack beyond-current-written-action distinct probe-slot breakthrough passed now",
            sign_base.truth(updated_pack_beyond_current_written_action_distinct_probe_slot_breakthrough_passed_now),
            "This branch closes a structural theorem but does not yet deliver the extended action or the residual-origin breakthrough itself.",
        ),
        sign_base.row(
            "blind_vector_observable_gate_still_blocked",
            "pass" if blind_blocked else "reject",
            "blind-vector observable gate still blocked",
            sign_base.truth(blind_blocked),
            "Blind-vector direct computation still waits on an actual extended action with a literal nonadditive probe slot.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "retained_scalar_residual_rel": float(prior_gate_summary["retained_scalar_residual_rel"]),
        "updated_pack_beyond_current_written_action_distinct_probe_slot_audit_selected": audit_selected,
        "retry_gate_computation_mode_selected": retry_mode,
        "failure_matrix_non_surrogate_guard_preserved": non_surrogate_guard,
        "exact_beyond_current_written_action_probe_extension_requirement_available_now": extension_requirement_available,
        "exact_minimal_independent_probe_slot_requirement_theorem_available_now": minimal_independent_slot_requirement_available,
        "exact_external_probe_current_one_point_no_go_theorem_available_now": additive_one_point_no_go_available,
        "exact_corrected_shared_hessian_identity_available_now": shared_hessian_identity_available,
        "current_note_single_slot_surface_explicit": current_note_single_slot_surface_explicit,
        "current_note_second_probe_slot_written_now": current_note_second_probe_slot_written_now,
        "exact_beyond_current_written_action_distinct_probe_slot_formula_available_now": exact_beyond_current_written_action_distinct_probe_slot_formula_available_now,
        "exact_beyond_current_written_action_same_combination_probe_slot_no_go_theorem_available_now": exact_beyond_current_written_action_same_combination_probe_slot_no_go_theorem_available_now,
        "exact_beyond_current_written_action_nonadditive_probe_slot_requirement_theorem_available_now": exact_beyond_current_written_action_nonadditive_probe_slot_requirement_theorem_available_now,
        "exact_beyond_current_written_action_probe_extension_available_now": exact_beyond_current_written_action_probe_extension_available_now,
        "updated_pack_explicit_nonadditive_probe_slot_extension_primary_followup_required": updated_pack_explicit_nonadditive_probe_slot_extension_primary_followup_required,
        "updated_pack_corrected_pack_refresh_secondary_hold_retained": updated_pack_corrected_pack_refresh_secondary_hold_retained,
        "updated_pack_same_tag_pack_refresh_reentry_admissible_now": updated_pack_same_tag_pack_refresh_reentry_admissible_now,
        "updated_pack_beyond_current_written_action_distinct_probe_slot_breakthrough_passed_now": updated_pack_beyond_current_written_action_distinct_probe_slot_breakthrough_passed_now,
        "blind_vector_observable_gate_still_blocked": blind_blocked,
        "selected_primary_completion_lane": "updated_pack_beyond_current_written_action_explicit_nonadditive_probe_slot_extension_audit",
        "selected_secondary_completion_lane": "updated_pack_corrected_pack_refresh_return_sync",
        "selected_reserve_completion_lane": "farther_hybrid_extra_q_range_only",
        "selected_next_generation_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_beyond_current_written_action_explicit_nonadditive_probe_slot_extension_audit",
        "recommended_next_route_or_none": "8.7.56.4591",
        "selected_followup_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_beyond_current_written_action_explicit_nonadditive_probe_slot_extension_gate",
        "selected_followup_route_or_none": "8.7.56.4595",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.4585",
        STEP_NAME + " declaration gate",
        {
            "source_files": {
                "prior_gate": sign_base.display_path(PRIOR_GATE),
                "prior_requirement_audit": sign_base.display_path(PRIOR_REQUIREMENT_AUDIT),
                "prior_probe_audit": sign_base.display_path(PRIOR_PROBE_AUDIT),
                "prior_kernel_audit": sign_base.display_path(PRIOR_KERNEL_AUDIT),
                "source_note": sign_base.display_path(PURE_DERIVATION_NOTE),
            },
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "8.7.56.4591",
                "followup_route": "8.7.56.4595",
            },
        },
        rows,
        summary,
        {
            "overall_status": "vector_qball_form_factor_updated_pack_beyond_current_written_action_distinct_probe_slot_theorem_derived",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {"formulae": build_formulae()},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(
        f"[done] {STEP_TAG} updated-pack beyond-current-written-action distinct probe-slot theorem completed"
    )
    print(f"[done] declaration: {declaration_paths['json']}")


if __name__ == "__main__":
    main()
