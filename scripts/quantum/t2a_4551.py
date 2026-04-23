#!/usr/bin/env python3
"""Generate 8.7.56.4551-.4554 distinct external-probe separation theorem artifacts."""

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
        "8.7.56.4547-4550",
        "updated_pack_corrected_vacuum_selector_gate",
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

STEP_TAG = "8.7.56.4551-4554"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack distinct "
    "external-probe separation theorem audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_distinct_external_probe_separation_theorem_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "corrected_vacuum_selector_no_go_theorem_derived_distinct_probe_primary_"
    "hybrid_reserve_secondary_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "corrected_distinct_probe_separation_no_go_theorem_derived_"
    "external_probe_structure_primary_pack_refresh_secondary_gate"
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


# 関数: distinct external-probe separation theorem で使う式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used in the distinct external-probe separation theorem audit."""
    return {
        "single_field_action": "S = int d^4x L_total^vec[P_mu] with no separate probe field slot",
        "single_split_note": "P_mu(x) = P_mu^Q(x) + a_mu(x)",
        "additive_same_action_identity": (
            "S^(2)[Q;xi+A] = (1/2) int d^4x d^4y (xi_mu + A_mu) "
            "H^{mu nu}[Q](x,y) (xi_nu + A_nu)"
        ),
        "kernel_collapse": "V^{mu nu}[Q](x,y) = Pi^{mu nu}[Q](x,y) = H^{mu nu}[Q](x,y)",
        "distinct_probe_no_go": (
            "without a separate action slot, xi_mu and A_mu cannot be "
            "distinguished theorem-side inside the current same-action surface"
        ),
    }


# 関数: `.4551-.4554` を実行する。

def main() -> None:
    """Execute the distinct external-probe separation theorem audit."""
    for path in (PRIOR_GATE, PRIOR_KERNEL_AUDIT, PURE_DERIVATION_NOTE):
        sign_base.require(path)

    prior_gate_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    prior_kernel_summary = sign_base.read_json(PRIOR_KERNEL_AUDIT)["summary"]
    note_text = sign_base.read_text(PURE_DERIVATION_NOTE)

    audit_selected = bool(
        prior_gate_summary[
            "gate_b_updated_pack_distinct_external_probe_separation_promoted_next"
        ]
        and prior_gate_summary["pack_update_required_now"]
    )
    retry_mode = bool(prior_gate_summary["retry_gate_computation_mode_selected"])
    non_surrogate_guard = bool(
        prior_gate_summary["failure_matrix_non_surrogate_guard_preserved"]
    )
    selector_no_go_available = bool(
        prior_gate_summary[
            "gate_a_updated_pack_exact_corrected_vacuum_state_selector_no_go_available_now"
        ]
    )
    exact_corrected_mixed_probe_response_kernel_formula_available_now = bool(
        prior_kernel_summary["exact_corrected_mixed_probe_response_kernel_formula_available_now"]
    )
    exact_corrected_pure_probe_response_kernel_formula_available_now = bool(
        prior_kernel_summary["exact_corrected_pure_probe_response_kernel_formula_available_now"]
    )
    exact_corrected_kernel_rank_match_available_now = bool(
        prior_kernel_summary["exact_corrected_kernel_rank_match_available_now"]
    )
    single_field_action_surface_explicit = bool(
        sign_base.hit(note_text, "\\mathcal{L}_{\\rm total}^{\\rm vec}") is not None
        and sign_base.hit(note_text, "U(P_\\mu^* P^\\mu)") is not None
    )
    single_split_only_explicit = bool(
        sign_base.hit(note_text, "P_\\mu(x) = P_\\mu^{\\rm Q}(x) + a_\\mu(x)")
        is not None
    )
    matter_rotation_deferred_explicit = bool(
        sign_base.hit(note_text, "matter 項と rotation 項") is not None
        and sign_base.hit(note_text, "matter/rotation の寄与は最後に確認する")
        is not None
    )
    current_written_action_external_probe_field_available_now = False
    exact_corrected_distinct_external_probe_separation_no_go_theorem_available_now = bool(
        audit_selected
        and retry_mode
        and non_surrogate_guard
        and selector_no_go_available
        and exact_corrected_mixed_probe_response_kernel_formula_available_now
        and exact_corrected_pure_probe_response_kernel_formula_available_now
        and exact_corrected_kernel_rank_match_available_now
        and single_field_action_surface_explicit
        and single_split_only_explicit
        and matter_rotation_deferred_explicit
        and not current_written_action_external_probe_field_available_now
    )
    exact_corrected_distinct_external_probe_separation_available_now = False
    exact_corrected_same_action_probe_slot_exhausted_now = bool(
        exact_corrected_distinct_external_probe_separation_no_go_theorem_available_now
    )
    updated_pack_current_written_action_external_probe_structure_primary_followup_required = bool(
        exact_corrected_same_action_probe_slot_exhausted_now
    )
    updated_pack_corrected_pack_refresh_secondary_hold_retained = bool(
        updated_pack_current_written_action_external_probe_structure_primary_followup_required
    )
    updated_pack_same_tag_pack_refresh_reentry_admissible_now = False
    updated_pack_distinct_probe_separation_breakthrough_passed_now = False
    blind_blocked = bool(prior_gate_summary["blind_vector_observable_gate_still_blocked"])
    farther_hybrid = False

    rows = [
        sign_base.row(
            "updated_pack_distinct_external_probe_separation_audit_selected",
            "pass" if audit_selected else "reject",
            "updated-pack distinct external-probe separation audit selected",
            sign_base.truth(audit_selected),
            "This branch is worth running only after the same-action vacuum selector has already closed as a no-go theorem.",
        ),
        sign_base.row(
            "retry_gate_computation_mode_selected",
            "pass" if retry_mode else "reject",
            "retry gate computation mode selected",
            sign_base.truth(retry_mode),
            "This turn stays on exact theorem derivation and does not count same-tag re-sync as progress.",
        ),
        sign_base.row(
            "failure_matrix_non_surrogate_guard_preserved",
            "pass" if non_surrogate_guard else "reject",
            "failure-matrix non-surrogate guard preserved",
            sign_base.truth(non_surrogate_guard),
            "The distinct-probe theorem is only admissible if it does not reopen the exhausted density/proxy/eigenvalue family.",
        ),
        sign_base.row(
            "exact_corrected_vacuum_state_selector_no_go_theorem_available_now",
            "pass" if selector_no_go_available else "reject",
            "exact corrected vacuum-state selector no-go theorem available now",
            sign_base.truth(selector_no_go_available),
            "The same-action subtraction selector is already closed as impossible on the written vacuum manifold, so the honest next question is probe-slot separation.",
        ),
        sign_base.row(
            "exact_corrected_mixed_probe_response_kernel_formula_available_now",
            "pass" if exact_corrected_mixed_probe_response_kernel_formula_available_now else "reject",
            "exact corrected mixed probe-response kernel formula available now",
            sign_base.truth(exact_corrected_mixed_probe_response_kernel_formula_available_now),
            "The shared Hessian identity is already available and therefore exposes what the current same-action slot can and cannot distinguish.",
        ),
        sign_base.row(
            "exact_corrected_pure_probe_response_kernel_formula_available_now",
            "pass" if exact_corrected_pure_probe_response_kernel_formula_available_now else "reject",
            "exact corrected pure probe-response kernel formula available now",
            sign_base.truth(exact_corrected_pure_probe_response_kernel_formula_available_now),
            "The pure probe kernel already collapses to the same Hessian, so any distinct probe theorem must come from a separate action slot rather than more same-action expansion.",
        ),
        sign_base.row(
            "exact_corrected_kernel_rank_match_available_now",
            "pass" if exact_corrected_kernel_rank_match_available_now else "reject",
            "exact corrected kernel rank match available now",
            sign_base.truth(exact_corrected_kernel_rank_match_available_now),
            "Rank matching is already closed at the Hessian level, so the remaining question is slot separation rather than observable rank.",
        ),
        sign_base.row(
            "single_field_action_surface_explicit",
            "pass" if single_field_action_surface_explicit else "reject",
            "single-field action surface explicit",
            sign_base.truth(single_field_action_surface_explicit),
            "The written note still defines the full action with only one field slot P_mu and does not introduce a second probe field at the action level.",
        ),
        sign_base.row(
            "single_split_only_explicit",
            "pass" if single_split_only_explicit else "reject",
            "single split only explicit",
            sign_base.truth(single_split_only_explicit),
            "The source note still writes only P_mu = P_mu^Q + a_mu, so the fluctuation and the would-be external probe share the same slot.",
        ),
        sign_base.row(
            "matter_rotation_deferred_explicit",
            "pass" if matter_rotation_deferred_explicit else "reject",
            "matter/rotation deferred explicit",
            sign_base.truth(matter_rotation_deferred_explicit),
            "The note explicitly postpones matter and rotation, so no later term in the current written surface rescues a separate probe slot inside this branch.",
        ),
        sign_base.row(
            "current_written_action_external_probe_field_available_now",
            "pass" if current_written_action_external_probe_field_available_now else "reject",
            "current written-action external probe field available now",
            sign_base.truth(current_written_action_external_probe_field_available_now),
            "No distinct action-level probe field A_mu is written in the present same-action surface, so separation is not available theorem-side.",
        ),
        sign_base.row(
            "exact_corrected_distinct_external_probe_separation_no_go_theorem_available_now",
            "pass" if exact_corrected_distinct_external_probe_separation_no_go_theorem_available_now else "reject",
            "exact corrected distinct external-probe separation no-go theorem available now",
            sign_base.truth(exact_corrected_distinct_external_probe_separation_no_go_theorem_available_now),
            "Because the written action has only one field slot and the same-action Hessian identity already collapses V and Pi onto H, xi_mu and A_mu cannot be distinguished theorem-side on the current written surface.",
        ),
        sign_base.row(
            "exact_corrected_distinct_external_probe_separation_available_now",
            "pass" if exact_corrected_distinct_external_probe_separation_available_now else "reject",
            "exact corrected distinct external-probe separation available now",
            sign_base.truth(exact_corrected_distinct_external_probe_separation_available_now),
            "The current same-action surface still does not provide a literal external probe slot distinct from the internal fluctuation slot.",
        ),
        sign_base.row(
            "exact_corrected_same_action_probe_slot_exhausted_now",
            "pass" if exact_corrected_same_action_probe_slot_exhausted_now else "reject",
            "exact corrected same-action probe-slot exhausted now",
            sign_base.truth(exact_corrected_same_action_probe_slot_exhausted_now),
            "The present written same-action surface is now exhausted theorem-side for distinct external-probe separation.",
        ),
        sign_base.row(
            "updated_pack_current_written_action_external_probe_structure_primary_followup_required",
            "pass" if updated_pack_current_written_action_external_probe_structure_primary_followup_required else "reject",
            "updated-pack current written-action external-probe structure primary followup required",
            sign_base.truth(updated_pack_current_written_action_external_probe_structure_primary_followup_required),
            "The honest next blocker is now the literal current written-action probe-field structure theorem, not another same-tag pack-refresh restatement.",
        ),
        sign_base.row(
            "updated_pack_corrected_pack_refresh_secondary_hold_retained",
            "pass" if updated_pack_corrected_pack_refresh_secondary_hold_retained else "reject",
            "updated-pack corrected pack-refresh secondary hold retained",
            sign_base.truth(updated_pack_corrected_pack_refresh_secondary_hold_retained),
            "Pack-refresh remains only a secondary hold because it cannot solve the absence of a separate probe slot on the current written action surface.",
        ),
        sign_base.row(
            "updated_pack_same_tag_pack_refresh_reentry_admissible_now",
            "pass" if updated_pack_same_tag_pack_refresh_reentry_admissible_now else "reject",
            "updated-pack same-tag pack-refresh reentry admissible now",
            sign_base.truth(updated_pack_same_tag_pack_refresh_reentry_admissible_now),
            "Same-tag pack-refresh reentry remains closed because this branch added a theorem object and the remaining blocker is structural rather than bookkeeping.",
        ),
        sign_base.row(
            "blind_vector_observable_gate_still_blocked",
            "pass" if blind_blocked else "reject",
            "blind-vector observable gate still blocked",
            sign_base.truth(blind_blocked),
            "Blind-vector direct computation still waits on a theorem-side route beyond the current same-action probe-slot no-go.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "retained_scalar_residual_rel": float(prior_gate_summary["retained_scalar_residual_rel"]),
        "updated_pack_distinct_external_probe_separation_audit_selected": audit_selected,
        "retry_gate_computation_mode_selected": retry_mode,
        "failure_matrix_non_surrogate_guard_preserved": non_surrogate_guard,
        "exact_corrected_vacuum_state_selector_no_go_theorem_available_now": selector_no_go_available,
        "exact_corrected_mixed_probe_response_kernel_formula_available_now": exact_corrected_mixed_probe_response_kernel_formula_available_now,
        "exact_corrected_pure_probe_response_kernel_formula_available_now": exact_corrected_pure_probe_response_kernel_formula_available_now,
        "exact_corrected_kernel_rank_match_available_now": exact_corrected_kernel_rank_match_available_now,
        "single_field_action_surface_explicit": single_field_action_surface_explicit,
        "single_split_only_explicit": single_split_only_explicit,
        "matter_rotation_deferred_explicit": matter_rotation_deferred_explicit,
        "current_written_action_external_probe_field_available_now": current_written_action_external_probe_field_available_now,
        "exact_corrected_distinct_external_probe_separation_no_go_theorem_available_now": exact_corrected_distinct_external_probe_separation_no_go_theorem_available_now,
        "exact_corrected_distinct_external_probe_separation_available_now": exact_corrected_distinct_external_probe_separation_available_now,
        "exact_corrected_same_action_probe_slot_exhausted_now": exact_corrected_same_action_probe_slot_exhausted_now,
        "updated_pack_current_written_action_external_probe_structure_primary_followup_required": updated_pack_current_written_action_external_probe_structure_primary_followup_required,
        "updated_pack_corrected_pack_refresh_secondary_hold_retained": updated_pack_corrected_pack_refresh_secondary_hold_retained,
        "updated_pack_same_tag_pack_refresh_reentry_admissible_now": updated_pack_same_tag_pack_refresh_reentry_admissible_now,
        "updated_pack_distinct_probe_separation_breakthrough_passed_now": updated_pack_distinct_probe_separation_breakthrough_passed_now,
        "blind_vector_observable_gate_still_blocked": blind_blocked,
        "selected_primary_completion_lane": "updated_pack_current_written_action_external_probe_structure_theorem_audit",
        "selected_secondary_completion_lane": "updated_pack_corrected_pack_refresh_return_sync",
        "selected_reserve_completion_lane": "farther_hybrid_extra_q_range_only",
        "selected_next_generation_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_current_written_action_external_probe_structure_theorem_audit",
        "recommended_next_route_or_none": "8.7.56.4559",
        "selected_followup_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_current_written_action_external_probe_structure_gate",
        "selected_followup_route_or_none": "8.7.56.4563",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.4553",
        STEP_NAME + " declaration gate",
        {
            "source_files": {
                "prior_gate": sign_base.display_path(PRIOR_GATE),
                "prior_kernel_audit": sign_base.display_path(PRIOR_KERNEL_AUDIT),
                "source_note": sign_base.display_path(PURE_DERIVATION_NOTE),
            },
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "8.7.56.4559",
                "followup_route": "8.7.56.4563",
            },
        },
        rows,
        summary,
        {
            "overall_status": "vector_qball_form_factor_updated_pack_distinct_external_probe_separation_theorem_declared",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {"formulas": build_formulae()},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(f"[done] {STEP_TAG} updated-pack distinct external-probe separation theorem completed")
    print(f"[done] declaration: {declaration_paths['json']}")


if __name__ == "__main__":
    main()
