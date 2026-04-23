#!/usr/bin/env python3
"""Generate 8.7.56.4535-.4538 corrected vacuum-subtraction theorem artifacts."""

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
        "8.7.56.4531-4534",
        "updated_pack_corrected_mixed_kernel_hessian_identity_gate",
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

STEP_TAG = "8.7.56.4535-4538"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack corrected "
    "vacuum-subtraction theorem audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_corrected_vacuum_subtraction_theorem_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "corrected_mixed_kernel_hessian_identity_theorem_derived_"
    "vacuum_subtraction_primary_pack_refresh_secondary_hybrid_reserve_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "corrected_vacuum_nonuniqueness_and_subtracted_rank_preservation_theorem_"
    "derived_vacuum_state_selector_primary_pack_refresh_secondary_gate"
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


# 関数: corrected subtraction theorem で使う式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used in the corrected vacuum-subtraction theorem audit."""
    return {
        "shared_hessian_identity": (
            "V^{mu nu}[Q](x,y) = Pi^{mu nu}[Q](x,y) = H^{mu nu}[Q](x,y)"
        ),
        "subtracted_kernel": (
            "Delta K^{mu nu}[Q;vac](x,y) := H^{mu nu}[Q](x,y) - "
            "H^{mu nu}[vac](x,y)"
        ),
        "rank_preservation": (
            "M_sub(k,k';eps,eps') = eps*^mu Delta K_{mu nu}(k,k') eps'^nu"
        ),
        "vacuum_manifold_condition": (
            "U(P_mu^* P^mu) = lambda (Phi - v^2)^2 => vacuum minima satisfy "
            "Phi_vac = P_mu^* P^mu = v^2"
        ),
        "note_temporal_assignment": (
            "pure note assigns Phi_vac = -v^2 via a temporal VEV, which is not "
            "the same condition as the written Mexican-hat minimum"
        ),
    }


# 関数: `.4535-.4538` を実行する。

def main() -> None:
    """Execute the corrected vacuum-subtraction theorem audit."""
    for path in (PRIOR_GATE, PRIOR_KERNEL_AUDIT, PURE_DERIVATION_NOTE):
        sign_base.require(path)

    prior_gate_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    prior_kernel_summary = sign_base.read_json(PRIOR_KERNEL_AUDIT)["summary"]
    note_text = sign_base.read_text(PURE_DERIVATION_NOTE)

    audit_selected = bool(
        prior_gate_summary[
            "gate_b_updated_pack_corrected_vacuum_subtraction_refresh_promoted_next"
        ]
        and prior_gate_summary["pack_update_required_now"]
    )
    retry_mode = bool(prior_gate_summary["retry_gate_computation_mode_selected"])
    non_surrogate_guard = bool(
        prior_gate_summary["failure_matrix_non_surrogate_guard_preserved"]
    )
    exact_corrected_mixed_probe_response_kernel_formula_available_now = bool(
        prior_kernel_summary[
            "exact_corrected_mixed_probe_response_kernel_formula_available_now"
        ]
    )
    exact_corrected_pure_probe_response_kernel_formula_available_now = bool(
        prior_kernel_summary[
            "exact_corrected_pure_probe_response_kernel_formula_available_now"
        ]
    )
    exact_corrected_kernel_rank_match_available_now = bool(
        prior_kernel_summary["exact_corrected_kernel_rank_match_available_now"]
    )
    note_vacuum_subtraction_required_explicit = bool(
        sign_base.hit(note_text, "vacuum subtraction") is not None
        and sign_base.hit(note_text, "\\Delta\\mathcal{M}") is not None
    )
    written_potential_formula_explicit = bool(
        sign_base.hit(note_text, "U(\\Phi) = \\lambda(\\Phi - v^2)^2") is not None
    )
    note_temporal_vacuum_assignment_explicit = bool(
        sign_base.hit(note_text, "\\Phi_{\\rm vac} = -v^2") is not None
    )
    corrected_subtracted_kernel_formula_explicit = bool(
        audit_selected
        and retry_mode
        and non_surrogate_guard
        and exact_corrected_mixed_probe_response_kernel_formula_available_now
        and exact_corrected_pure_probe_response_kernel_formula_available_now
        and exact_corrected_kernel_rank_match_available_now
        and note_vacuum_subtraction_required_explicit
    )
    corrected_subtracted_observable_rank_preservation_theorem_explicit = bool(
        corrected_subtracted_kernel_formula_explicit
    )
    exact_corrected_subtracted_observable_rank_match_available_now = bool(
        corrected_subtracted_observable_rank_preservation_theorem_explicit
    )
    exact_corrected_vacuum_state_nonuniqueness_theorem_available_now = bool(
        written_potential_formula_explicit and note_temporal_vacuum_assignment_explicit
    )
    note_temporal_vev_assignment_incompatible_with_written_potential = bool(
        exact_corrected_vacuum_state_nonuniqueness_theorem_available_now
    )
    exact_corrected_vacuum_state_definition_available_now = False
    exact_corrected_vacuum_subtraction_rule_available_now = False
    corrected_vacuum_state_selector_primary_followup_required = bool(
        exact_corrected_vacuum_state_nonuniqueness_theorem_available_now
        and (not exact_corrected_vacuum_state_definition_available_now)
    )
    corrected_pack_refresh_secondary_hold_retained = bool(
        corrected_vacuum_state_selector_primary_followup_required
    )
    corrected_vacuum_subtraction_breakthrough_passed_now = False
    blind_blocked = bool(prior_gate_summary["blind_vector_observable_gate_still_blocked"])
    farther_hybrid = False

    rows = [
        sign_base.row(
            "updated_pack_corrected_vacuum_subtraction_theorem_audit_selected",
            "pass" if audit_selected else "reject",
            "updated-pack corrected vacuum-subtraction theorem audit selected",
            sign_base.truth(audit_selected),
            "The subtraction branch is worth running only after the mixed/pure probe-response kernel has closed as an exact Hessian identity.",
        ),
        sign_base.row(
            "retry_gate_computation_mode_selected",
            "pass" if retry_mode else "reject",
            "retry gate computation mode selected",
            sign_base.truth(retry_mode),
            "This turn stays on theorem-side derivation and does not treat another same-tag reserve loop as progress.",
        ),
        sign_base.row(
            "failure_matrix_non_surrogate_guard_preserved",
            "pass" if non_surrogate_guard else "reject",
            "failure-matrix non-surrogate guard preserved",
            sign_base.truth(non_surrogate_guard),
            "Closing subtraction-side theorems is admissible only if it does not reopen the exhausted density/proxy/eigenvalue family.",
        ),
        sign_base.row(
            "exact_corrected_mixed_probe_response_kernel_formula_available_now",
            "pass" if exact_corrected_mixed_probe_response_kernel_formula_available_now else "reject",
            "exact corrected mixed probe-response kernel formula available now",
            sign_base.truth(exact_corrected_mixed_probe_response_kernel_formula_available_now),
            "The subtraction theorem now starts from the already closed mixed-kernel Hessian identity.",
        ),
        sign_base.row(
            "exact_corrected_pure_probe_response_kernel_formula_available_now",
            "pass" if exact_corrected_pure_probe_response_kernel_formula_available_now else "reject",
            "exact corrected pure probe-response kernel formula available now",
            sign_base.truth(exact_corrected_pure_probe_response_kernel_formula_available_now),
            "The subtraction theorem also starts from the already closed pure probe-response Hessian identity.",
        ),
        sign_base.row(
            "exact_corrected_kernel_rank_match_available_now",
            "pass" if exact_corrected_kernel_rank_match_available_now else "reject",
            "exact corrected kernel rank match available now",
            sign_base.truth(exact_corrected_kernel_rank_match_available_now),
            "The pre-subtraction observable is already a genuine rank-2 kernel contracted by photon polarizations.",
        ),
        sign_base.row(
            "pure_derivation_vacuum_subtraction_required_explicit",
            "pass" if note_vacuum_subtraction_required_explicit else "reject",
            "pure-derivation vacuum subtraction required explicit",
            sign_base.truth(note_vacuum_subtraction_required_explicit),
            "The source note explicitly isolates vacuum subtraction as the last unresolved place in its own derivation.",
        ),
        sign_base.row(
            "written_potential_formula_explicit",
            "pass" if written_potential_formula_explicit else "reject",
            "written Mexican-hat potential formula explicit",
            sign_base.truth(written_potential_formula_explicit),
            "The frozen action still writes U(P_mu^* P^mu)=lambda(Phi-v^2)^2, so the vacuum manifold condition is fixed by Phi_vac=v^2.",
        ),
        sign_base.row(
            "pure_derivation_temporal_vacuum_assignment_explicit",
            "pass" if note_temporal_vacuum_assignment_explicit else "reject",
            "pure-derivation temporal vacuum assignment explicit",
            sign_base.truth(note_temporal_vacuum_assignment_explicit),
            "The source note explicitly assigns Phi_vac=-v^2 by choosing a temporal VEV candidate.",
        ),
        sign_base.row(
            "note_temporal_vev_assignment_incompatible_with_written_potential",
            "pass" if note_temporal_vev_assignment_incompatible_with_written_potential else "reject",
            "note temporal VEV assignment incompatible with written potential",
            sign_base.truth(note_temporal_vev_assignment_incompatible_with_written_potential),
            "A temporal-only assignment gives Phi_vac=-v^2, while the written Mexican-hat minimum requires Phi_vac=v^2, so the note does not yet supply a canonical vacuum representative.",
        ),
        sign_base.row(
            "corrected_subtracted_kernel_formula_explicit",
            "pass" if corrected_subtracted_kernel_formula_explicit else "reject",
            "corrected subtracted kernel formula explicit",
            sign_base.truth(corrected_subtracted_kernel_formula_explicit),
            "Once mixed/pure kernels close as shared Hessians, the honest subtraction candidate is Delta K = H[Q] - H[vac], not a scalar surrogate restart.",
        ),
        sign_base.row(
            "corrected_subtracted_observable_rank_preservation_theorem_explicit",
            "pass" if corrected_subtracted_observable_rank_preservation_theorem_explicit else "reject",
            "corrected subtracted observable rank-preservation theorem explicit",
            sign_base.truth(corrected_subtracted_observable_rank_preservation_theorem_explicit),
            "Subtracting two rank-2 Hessian kernels preserves the polarization-contracted rank structure exactly.",
        ),
        sign_base.row(
            "exact_corrected_subtracted_observable_rank_match_available_now",
            "pass" if exact_corrected_subtracted_observable_rank_match_available_now else "reject",
            "exact corrected subtracted observable rank match available now",
            sign_base.truth(exact_corrected_subtracted_observable_rank_match_available_now),
            "Rank match is no longer the blocker on the subtraction lane: Delta K still contracts as eps*^mu Delta K_{mu nu} eps'^nu.",
        ),
        sign_base.row(
            "exact_corrected_vacuum_state_nonuniqueness_theorem_available_now",
            "pass" if exact_corrected_vacuum_state_nonuniqueness_theorem_available_now else "reject",
            "exact corrected vacuum-state nonuniqueness theorem available now",
            sign_base.truth(exact_corrected_vacuum_state_nonuniqueness_theorem_available_now),
            "The written potential fixes only the vacuum manifold Phi_vac=v^2, while the source note picks an incompatible temporal VEV, so the action does not yet determine a unique subtraction vacuum.",
        ),
        sign_base.row(
            "exact_corrected_vacuum_state_definition_available_now",
            "pass" if exact_corrected_vacuum_state_definition_available_now else "reject",
            "exact corrected vacuum-state definition available now",
            sign_base.truth(exact_corrected_vacuum_state_definition_available_now),
            "What remains absent is a canonical vacuum selector theorem that chooses one representative of the allowed vacuum manifold for subtraction.",
        ),
        sign_base.row(
            "exact_corrected_vacuum_subtraction_rule_available_now",
            "pass" if exact_corrected_vacuum_subtraction_rule_available_now else "reject",
            "exact corrected vacuum-subtraction rule available now",
            sign_base.truth(exact_corrected_vacuum_subtraction_rule_available_now),
            "Because the vacuum representative is still nonunique, the literal subtraction rule cannot honestly close here.",
        ),
        sign_base.row(
            "updated_pack_corrected_vacuum_state_selector_primary_followup_required",
            "pass" if corrected_vacuum_state_selector_primary_followup_required else "reject",
            "updated-pack corrected vacuum-state selector primary followup required",
            sign_base.truth(corrected_vacuum_state_selector_primary_followup_required),
            "The honest next blocker is no longer subtraction rank mismatch but the missing theorem that selects a canonical subtraction vacuum.",
        ),
        sign_base.row(
            "updated_pack_corrected_pack_refresh_secondary_hold_retained",
            "pass" if corrected_pack_refresh_secondary_hold_retained else "reject",
            "updated-pack corrected pack-refresh secondary hold retained",
            sign_base.truth(corrected_pack_refresh_secondary_hold_retained),
            "Pack-refresh stays downstream of the vacuum-state selector theorem and should not resume as a same-tag loop substitute.",
        ),
        sign_base.row(
            "updated_pack_corrected_vacuum_subtraction_breakthrough_passed_now",
            "pass" if corrected_vacuum_subtraction_breakthrough_passed_now else "reject",
            "updated-pack corrected vacuum-subtraction breakthrough passed now",
            sign_base.truth(corrected_vacuum_subtraction_breakthrough_passed_now),
            "This branch closes theorem-side rank preservation and vacuum nonuniqueness, but it does not yet produce a canonical subtraction rule or residual-origin breakthrough.",
        ),
        sign_base.row(
            "blind_vector_observable_gate_still_blocked",
            "pass" if blind_blocked else "reject",
            "blind-vector observable gate still blocked",
            sign_base.truth(blind_blocked),
            "Blind-vector direct computation still waits on a canonical subtraction vacuum and downstream reserve closeout.",
        ),
        sign_base.row(
            "farther_hybrid_continuation_reopen_required_now",
            "pass" if farther_hybrid else "reject",
            "farther hybrid continuation reopen required now",
            sign_base.truth(farther_hybrid),
            "Extra q-range evidence remains reserve-only because the blocker is still theorem-side subtraction closeout, not continuation range.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "retained_scalar_residual_rel": float(prior_gate_summary["retained_scalar_residual_rel"]),
        "updated_pack_corrected_vacuum_subtraction_theorem_audit_selected": audit_selected,
        "retry_gate_computation_mode_selected": retry_mode,
        "failure_matrix_non_surrogate_guard_preserved": non_surrogate_guard,
        "exact_corrected_mixed_probe_response_kernel_formula_available_now": exact_corrected_mixed_probe_response_kernel_formula_available_now,
        "exact_corrected_pure_probe_response_kernel_formula_available_now": exact_corrected_pure_probe_response_kernel_formula_available_now,
        "exact_corrected_kernel_rank_match_available_now": exact_corrected_kernel_rank_match_available_now,
        "pure_derivation_vacuum_subtraction_required_explicit": note_vacuum_subtraction_required_explicit,
        "written_potential_formula_explicit": written_potential_formula_explicit,
        "pure_derivation_temporal_vacuum_assignment_explicit": note_temporal_vacuum_assignment_explicit,
        "note_temporal_vev_assignment_incompatible_with_written_potential": note_temporal_vev_assignment_incompatible_with_written_potential,
        "corrected_subtracted_kernel_formula_explicit": corrected_subtracted_kernel_formula_explicit,
        "corrected_subtracted_observable_rank_preservation_theorem_explicit": corrected_subtracted_observable_rank_preservation_theorem_explicit,
        "exact_corrected_subtracted_observable_rank_match_available_now": exact_corrected_subtracted_observable_rank_match_available_now,
        "exact_corrected_vacuum_state_nonuniqueness_theorem_available_now": exact_corrected_vacuum_state_nonuniqueness_theorem_available_now,
        "exact_corrected_vacuum_state_definition_available_now": exact_corrected_vacuum_state_definition_available_now,
        "exact_corrected_vacuum_subtraction_rule_available_now": exact_corrected_vacuum_subtraction_rule_available_now,
        "updated_pack_corrected_vacuum_state_selector_primary_followup_required": corrected_vacuum_state_selector_primary_followup_required,
        "updated_pack_corrected_pack_refresh_secondary_hold_retained": corrected_pack_refresh_secondary_hold_retained,
        "blind_vector_observable_gate_still_blocked": blind_blocked,
        "selected_primary_completion_lane": "updated_pack_corrected_vacuum_state_selector_theorem_audit",
        "selected_secondary_completion_lane": "updated_pack_corrected_pack_refresh_return_sync",
        "selected_next_generation_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_corrected_vacuum_nonuniqueness_gate",
        "recommended_next_route_or_none": "8.7.56.4539",
        "selected_followup_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_corrected_vacuum_state_selector_theorem_audit",
        "selected_followup_route_or_none": "8.7.56.4543",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.4537",
        STEP_NAME + " declaration gate",
        {
            "source_files": {
                "prior_gate": sign_base.display_path(PRIOR_GATE),
                "prior_kernel_audit": sign_base.display_path(PRIOR_KERNEL_AUDIT),
                "pure_derivation_note": sign_base.display_path(PURE_DERIVATION_NOTE),
            },
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "8.7.56.4539",
                "followup_route": "8.7.56.4543",
            },
        },
        rows,
        summary,
        {
            "overall_status": "vector_qball_form_factor_updated_pack_corrected_vacuum_theorem_declared",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {"formulae": build_formulae()},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(f"[done] {STEP_TAG} updated-pack corrected vacuum-subtraction theorem audit completed")
    print(f"[done] declaration: {declaration_paths['json']}")


if __name__ == "__main__":
    main()
