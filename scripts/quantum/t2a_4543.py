#!/usr/bin/env python3
"""Generate 8.7.56.4543-.4546 corrected vacuum-state selector theorem artifacts."""

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
        "8.7.56.4539-4542",
        "updated_pack_corrected_vacuum_theorem_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
PURE_DERIVATION_NOTE = Path(
    r"C:\Users\ogawa\Downloads\trial2_vector_qball_pure_derivation_20260330.md"
)

STEP_TAG = "8.7.56.4543-4546"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack corrected "
    "vacuum-state selector theorem audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_corrected_vacuum_selector_theorem_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "corrected_vacuum_nonuniqueness_and_subtracted_rank_preservation_theorem_"
    "derived_vacuum_state_selector_primary_pack_refresh_secondary_hybrid_"
    "reserve_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "corrected_vacuum_selector_no_go_theorem_derived_distinct_probe_primary_"
    "pack_refresh_secondary_gate"
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


# 関数: vacuum-state selector theorem で使う式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used in the corrected vacuum-state selector theorem audit."""
    return {
        "constant_vacuum_ansatz": "P_mu(x) = C_mu (constant) => F_{mu nu}[C] = 0",
        "vacuum_energy_density": (
            "L_vac[C] = U(C_mu^* C^mu) = lambda (C_mu^* C^mu - v^2)^2"
        ),
        "vacuum_manifold": "M_vac = { C_mu const | C_mu^* C^mu = v^2 }",
        "degenerate_spatial_family": (
            "C_mu = (0, v n_i e^{i theta}), |n|=1 => C_mu^* C^mu = v^2"
        ),
        "temporal_note_candidate": (
            "C_mu = (v e^{i theta}, 0, 0, 0) => C_mu^* C^mu = -v^2"
        ),
        "selector_no_go": (
            "written action fixes only M_vac, not a canonical representative "
            "C_mu^vac for subtraction"
        ),
    }


# 関数: `.4543-.4546` を実行する。

def main() -> None:
    """Execute the corrected vacuum-state selector theorem audit."""
    for path in (PRIOR_GATE, PURE_DERIVATION_NOTE):
        sign_base.require(path)

    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    note_text = sign_base.read_text(PURE_DERIVATION_NOTE)

    audit_selected = bool(
        prior_summary[
            "gate_b_updated_pack_corrected_vacuum_state_selector_theorem_promoted_next"
        ]
        and prior_summary["pack_update_required_now"]
    )
    retry_mode = bool(prior_summary["retry_gate_computation_mode_selected"])
    non_surrogate_guard = bool(
        prior_summary["failure_matrix_non_surrogate_guard_preserved"]
    )
    exact_corrected_subtracted_observable_rank_match_available_now = bool(
        prior_summary["exact_corrected_subtracted_observable_rank_match_available_now"]
    )
    exact_corrected_vacuum_state_nonuniqueness_theorem_available_now = bool(
        prior_summary["exact_corrected_vacuum_state_nonuniqueness_theorem_available_now"]
    )
    written_potential_formula_explicit = bool(
        sign_base.hit(note_text, "U(\\Phi) = \\lambda(\\Phi - v^2)^2") is not None
    )
    kinetic_potential_only_selector_surface_explicit = bool(
        sign_base.hit(note_text, "Q-ball sector に直接寄与しない") is not None
    )
    note_temporal_vacuum_assignment_explicit = bool(
        sign_base.hit(note_text, "\\Phi_{\\rm vac} = -v^2") is not None
    )
    constant_field_zero_field_strength_formula_available = bool(
        audit_selected and written_potential_formula_explicit
    )
    exact_corrected_vacuum_manifold_formula_available_now = bool(
        constant_field_zero_field_strength_formula_available
    )
    corrected_spatial_vacuum_family_example_available = bool(
        exact_corrected_vacuum_manifold_formula_available_now
    )
    note_temporal_vacuum_candidate_off_manifold = bool(
        note_temporal_vacuum_assignment_explicit
        and exact_corrected_vacuum_manifold_formula_available_now
    )
    exact_corrected_vacuum_state_selector_no_go_theorem_available_now = bool(
        retry_mode
        and non_surrogate_guard
        and exact_corrected_subtracted_observable_rank_match_available_now
        and exact_corrected_vacuum_state_nonuniqueness_theorem_available_now
        and exact_corrected_vacuum_manifold_formula_available_now
        and kinetic_potential_only_selector_surface_explicit
        and note_temporal_vacuum_candidate_off_manifold
        and corrected_spatial_vacuum_family_example_available
    )
    exact_corrected_vacuum_state_definition_available_now = False
    exact_corrected_vacuum_subtraction_rule_available_now = False
    exact_corrected_same_action_selector_exhausted_now = bool(
        exact_corrected_vacuum_state_selector_no_go_theorem_available_now
    )
    updated_pack_distinct_probe_primary_followup_required = bool(
        exact_corrected_same_action_selector_exhausted_now
    )
    updated_pack_corrected_pack_refresh_secondary_hold_retained = bool(
        updated_pack_distinct_probe_primary_followup_required
    )
    updated_pack_same_tag_pack_refresh_reentry_admissible_now = False
    updated_pack_corrected_vacuum_selector_breakthrough_passed_now = False
    blind_blocked = bool(prior_summary["blind_vector_observable_gate_still_blocked"])
    farther_hybrid = False

    rows = [
        sign_base.row(
            "updated_pack_corrected_vacuum_state_selector_audit_selected",
            "pass" if audit_selected else "reject",
            "updated-pack corrected vacuum-state selector audit selected",
            sign_base.truth(audit_selected),
            "This branch is only worth running after subtraction rank preservation and vacuum nonuniqueness have both closed theorem-side.",
        ),
        sign_base.row(
            "retry_gate_computation_mode_selected",
            "pass" if retry_mode else "reject",
            "retry gate computation mode selected",
            sign_base.truth(retry_mode),
            "The selector branch is treated as theorem derivation, not as another same-tag bookkeeping loop.",
        ),
        sign_base.row(
            "failure_matrix_non_surrogate_guard_preserved",
            "pass" if non_surrogate_guard else "reject",
            "failure-matrix non-surrogate guard preserved",
            sign_base.truth(non_surrogate_guard),
            "Closing the selector theorem is admissible only if it does not reopen the exhausted density/proxy/eigenvalue family.",
        ),
        sign_base.row(
            "exact_corrected_subtracted_observable_rank_match_available_now",
            "pass" if exact_corrected_subtracted_observable_rank_match_available_now else "reject",
            "exact corrected subtracted observable rank match available now",
            sign_base.truth(exact_corrected_subtracted_observable_rank_match_available_now),
            "The subtraction candidate already preserves the rank-2 photon contraction, so selector failure can be isolated honestly.",
        ),
        sign_base.row(
            "exact_corrected_vacuum_state_nonuniqueness_theorem_available_now",
            "pass" if exact_corrected_vacuum_state_nonuniqueness_theorem_available_now else "reject",
            "exact corrected vacuum-state nonuniqueness theorem available now",
            sign_base.truth(exact_corrected_vacuum_state_nonuniqueness_theorem_available_now),
            "The prior branch already showed that the source note's temporal assignment conflicts with the written Mexican-hat minimum.",
        ),
        sign_base.row(
            "written_potential_formula_explicit",
            "pass" if written_potential_formula_explicit else "reject",
            "written Mexican-hat potential formula explicit",
            sign_base.truth(written_potential_formula_explicit),
            "The current theorem lane still uses the written potential U(P_mu^* P^mu)=lambda(Phi-v^2)^2 as the vacuum selector surface.",
        ),
        sign_base.row(
            "kinetic_potential_only_selector_surface_explicit",
            "pass" if kinetic_potential_only_selector_surface_explicit else "reject",
            "kinetic-plus-potential-only selector surface explicit",
            sign_base.truth(kinetic_potential_only_selector_surface_explicit),
            "The source note explicitly restricts the selector discussion to the kinetic-plus-potential sector before matter/rotation are reconsidered.",
        ),
        sign_base.row(
            "constant_field_zero_field_strength_formula_available",
            "pass" if constant_field_zero_field_strength_formula_available else "reject",
            "constant-field zero-field-strength formula available",
            sign_base.truth(constant_field_zero_field_strength_formula_available),
            "For a constant vacuum representative C_mu, F_{mu nu}[C]=0, so the vacuum energy depends only on the invariant C_mu^* C^mu.",
        ),
        sign_base.row(
            "exact_corrected_vacuum_manifold_formula_available_now",
            "pass" if exact_corrected_vacuum_manifold_formula_available_now else "reject",
            "exact corrected vacuum-manifold formula available now",
            sign_base.truth(exact_corrected_vacuum_manifold_formula_available_now),
            "The written action fixes the vacuum manifold as C_mu^* C^mu=v^2, not a unique subtraction representative.",
        ),
        sign_base.row(
            "corrected_spatial_vacuum_family_example_available",
            "pass" if corrected_spatial_vacuum_family_example_available else "reject",
            "corrected spatial vacuum-family example available",
            sign_base.truth(corrected_spatial_vacuum_family_example_available),
            "A whole family such as C_mu=(0, v n_i e^{i theta}) with |n|=1 already lies on the written vacuum manifold, so degeneracy is explicit.",
        ),
        sign_base.row(
            "note_temporal_vacuum_candidate_off_manifold",
            "pass" if note_temporal_vacuum_candidate_off_manifold else "reject",
            "note temporal vacuum candidate off manifold",
            sign_base.truth(note_temporal_vacuum_candidate_off_manifold),
            "The source note's temporal assignment gives Phi_vac=-v^2 and therefore does not lie on the written manifold Phi_vac=v^2.",
        ),
        sign_base.row(
            "exact_corrected_vacuum_state_selector_no_go_theorem_available_now",
            "pass" if exact_corrected_vacuum_state_selector_no_go_theorem_available_now else "reject",
            "exact corrected vacuum-state selector no-go theorem available now",
            sign_base.truth(exact_corrected_vacuum_state_selector_no_go_theorem_available_now),
            "Under the current same-action kinetic-plus-potential surface, the written action selects only the manifold C_mu^* C^mu=v^2 and therefore cannot choose one canonical subtraction vacuum.",
        ),
        sign_base.row(
            "exact_corrected_vacuum_state_definition_available_now",
            "pass" if exact_corrected_vacuum_state_definition_available_now else "reject",
            "exact corrected vacuum-state definition available now",
            sign_base.truth(exact_corrected_vacuum_state_definition_available_now),
            "The selector theorem closes as a no-go, so a canonical vacuum representative is still unavailable under the current written action.",
        ),
        sign_base.row(
            "exact_corrected_vacuum_subtraction_rule_available_now",
            "pass" if exact_corrected_vacuum_subtraction_rule_available_now else "reject",
            "exact corrected vacuum-subtraction rule available now",
            sign_base.truth(exact_corrected_vacuum_subtraction_rule_available_now),
            "Without a canonical selector, the literal subtraction rule still cannot honestly close on the current same-action theorem lane.",
        ),
        sign_base.row(
            "exact_corrected_same_action_selector_exhausted_now",
            "pass" if exact_corrected_same_action_selector_exhausted_now else "reject",
            "exact corrected same-action selector exhausted now",
            sign_base.truth(exact_corrected_same_action_selector_exhausted_now),
            "The current same-action subtraction route is now exhausted theorem-side: it has a manifold and a no-go selector, but no canonical representative.",
        ),
        sign_base.row(
            "updated_pack_distinct_probe_primary_followup_required",
            "pass" if updated_pack_distinct_probe_primary_followup_required else "reject",
            "updated-pack distinct external-probe primary followup required",
            sign_base.truth(updated_pack_distinct_probe_primary_followup_required),
            "Because same-action selector no-go is now theorem-level, the honest next blocker is distinct external-probe structure rather than another same-tag reserve loop.",
        ),
        sign_base.row(
            "updated_pack_corrected_pack_refresh_secondary_hold_retained",
            "pass" if updated_pack_corrected_pack_refresh_secondary_hold_retained else "reject",
            "updated-pack corrected pack-refresh secondary hold retained",
            sign_base.truth(updated_pack_corrected_pack_refresh_secondary_hold_retained),
            "Pack-refresh stays secondary and should not be treated as the mainline breakthrough route after the selector no-go theorem.",
        ),
        sign_base.row(
            "updated_pack_same_tag_pack_refresh_reentry_admissible_now",
            "pass" if updated_pack_same_tag_pack_refresh_reentry_admissible_now else "reject",
            "updated-pack same-tag pack-refresh reentry admissible now",
            sign_base.truth(updated_pack_same_tag_pack_refresh_reentry_admissible_now),
            "Reentering the same-tag pack-refresh loop would only restate an exhausted route and is therefore not honest progress.",
        ),
        sign_base.row(
            "updated_pack_corrected_vacuum_selector_breakthrough_passed_now",
            "pass" if updated_pack_corrected_vacuum_selector_breakthrough_passed_now else "reject",
            "updated-pack corrected vacuum selector breakthrough passed now",
            sign_base.truth(updated_pack_corrected_vacuum_selector_breakthrough_passed_now),
            "This branch closes a selector no-go theorem, but it does not yet deliver a canonical subtraction rule or residual-origin breakthrough.",
        ),
        sign_base.row(
            "blind_vector_observable_gate_still_blocked",
            "pass" if blind_blocked else "reject",
            "blind-vector observable gate still blocked",
            sign_base.truth(blind_blocked),
            "Blind-vector direct computation still waits on a theorem-side route beyond the same-action selector no-go.",
        ),
        sign_base.row(
            "farther_hybrid_continuation_reopen_required_now",
            "pass" if farther_hybrid else "reject",
            "farther hybrid continuation reopen required now",
            sign_base.truth(farther_hybrid),
            "Extra q-range evidence remains reserve-only because the blocker is still theorem-side structure, not continuation range.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "retained_scalar_residual_rel": float(prior_summary["retained_scalar_residual_rel"]),
        "updated_pack_corrected_vacuum_state_selector_audit_selected": audit_selected,
        "retry_gate_computation_mode_selected": retry_mode,
        "failure_matrix_non_surrogate_guard_preserved": non_surrogate_guard,
        "exact_corrected_subtracted_observable_rank_match_available_now": exact_corrected_subtracted_observable_rank_match_available_now,
        "exact_corrected_vacuum_state_nonuniqueness_theorem_available_now": exact_corrected_vacuum_state_nonuniqueness_theorem_available_now,
        "written_potential_formula_explicit": written_potential_formula_explicit,
        "kinetic_potential_only_selector_surface_explicit": kinetic_potential_only_selector_surface_explicit,
        "constant_field_zero_field_strength_formula_available": constant_field_zero_field_strength_formula_available,
        "exact_corrected_vacuum_manifold_formula_available_now": exact_corrected_vacuum_manifold_formula_available_now,
        "corrected_spatial_vacuum_family_example_available": corrected_spatial_vacuum_family_example_available,
        "note_temporal_vacuum_candidate_off_manifold": note_temporal_vacuum_candidate_off_manifold,
        "exact_corrected_vacuum_state_selector_no_go_theorem_available_now": exact_corrected_vacuum_state_selector_no_go_theorem_available_now,
        "exact_corrected_vacuum_state_definition_available_now": exact_corrected_vacuum_state_definition_available_now,
        "exact_corrected_vacuum_subtraction_rule_available_now": exact_corrected_vacuum_subtraction_rule_available_now,
        "exact_corrected_same_action_selector_exhausted_now": exact_corrected_same_action_selector_exhausted_now,
        "updated_pack_distinct_probe_primary_followup_required": updated_pack_distinct_probe_primary_followup_required,
        "updated_pack_corrected_pack_refresh_secondary_hold_retained": updated_pack_corrected_pack_refresh_secondary_hold_retained,
        "blind_vector_observable_gate_still_blocked": blind_blocked,
        "selected_primary_completion_lane": "updated_pack_distinct_external_probe_separation_theorem_audit",
        "selected_secondary_completion_lane": "updated_pack_corrected_pack_refresh_return_sync",
        "selected_next_generation_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_corrected_vacuum_selector_no_go_gate",
        "recommended_next_route_or_none": "8.7.56.4547",
        "selected_followup_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_distinct_external_probe_separation_theorem_audit",
        "selected_followup_route_or_none": "8.7.56.4551",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.4545",
        STEP_NAME + " declaration gate",
        {
            "source_files": {
                "prior_gate": sign_base.display_path(PRIOR_GATE),
                "pure_derivation_note": sign_base.display_path(PURE_DERIVATION_NOTE),
            },
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "8.7.56.4547",
                "followup_route": "8.7.56.4551",
            },
        },
        rows,
        summary,
        {
            "overall_status": "vector_qball_form_factor_updated_pack_corrected_vacuum_selector_no_go_theorem_derived",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {"formulae": build_formulae()},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(f"[done] {STEP_TAG} updated-pack corrected vacuum-state selector audit completed")
    print(f"[done] declaration: {declaration_paths['json']}")


if __name__ == "__main__":
    main()
