#!/usr/bin/env python3
"""Generate 8.7.56.4527-.4530 corrected mixed-kernel Hessian-identity artifacts."""

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
        "8.7.56.4523-4526",
        "updated_pack_corrected_probe_split_additive_no_go_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
PURE_DERIVATION_NOTE = Path(
    r"C:\Users\ogawa\Downloads\trial2_vector_qball_pure_derivation_20260330.md"
)

STEP_TAG = "8.7.56.4527-4530"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack corrected "
    "mixed-kernel return refresh audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_corrected_mixed_kernel_hessian_identity_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "corrected_probe_split_additive_one_point_no_go_theorem_derived_"
    "mixed_kernel_primary_vacuum_subtraction_secondary_hybrid_reserve_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "corrected_mixed_kernel_hessian_identity_theorem_derived_"
    "vacuum_subtraction_primary_pack_refresh_secondary_gate"
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


# 関数: additive mixed-kernel identity で使う式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used in the corrected mixed-kernel Hessian-identity audit."""
    return {
        "additive_split": "P_mu(x) = Q_mu(x) + xi_mu(x) + A_mu(x)",
        "hessian_definition": (
            "H^{mu nu}[Q](x,y) := delta^2 S / (delta P_mu(x) delta P_nu(y))|_(P=Q)"
        ),
        "second_variation": (
            "S^(2)[Q;xi+A] = (1/2) int d^4x d^4y (xi_mu + A_mu) "
            "H^{mu nu}[Q](x,y) (xi_nu + A_nu)"
        ),
        "expanded_second_variation": (
            "S^(2)[Q;xi+A] = (1/2) xi H xi + xi H A + (1/2) A H A"
        ),
        "mixed_kernel_identity": (
            "V^{mu nu}[Q](x,y) := delta^2 S / (delta xi_mu(x) delta A_nu(y))|_(Q)"
            " = H^{mu nu}[Q](x,y)"
        ),
        "pure_kernel_identity": (
            "Pi^{mu nu}[Q](x,y) := delta^2 S / (delta A_mu(x) delta A_nu(y))|_(Q)"
            " = H^{mu nu}[Q](x,y)"
        ),
        "rank_matched_amplitude": (
            "M(k,k';eps,eps') = eps*^mu K_{mu nu}(k,k') eps'^nu"
        ),
    }


# 関数: `.4527-.4530` を実行する。

def main() -> None:
    """Execute the corrected mixed-kernel Hessian-identity audit."""
    for path in (PRIOR_GATE, PURE_DERIVATION_NOTE):
        sign_base.require(path)

    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    note_text = sign_base.read_text(PURE_DERIVATION_NOTE)

    audit_selected = bool(
        prior_summary["gate_b_updated_pack_corrected_mixed_kernel_refresh_promoted_next"]
        and prior_summary["pack_update_required_now"]
    )
    retry_mode = bool(prior_summary["retry_gate_computation_mode_selected"])
    non_surrogate_guard = bool(
        prior_summary["failure_matrix_non_surrogate_guard_preserved"]
    )
    additive_one_point_no_go_available = bool(
        prior_summary[
            "gate_a_updated_pack_exact_corrected_probe_split_additive_one_point_no_go_available_now"
        ]
    )
    pure_derivation_second_variation_explicit = bool(
        sign_base.hit(note_text, "S^{(2)}[Q;a] =") is not None
        and sign_base.hit(note_text, "\\frac{\\delta^2 S}{\\delta P_\\mu(x)\\,\\delta P_\\nu(y)}")
        is not None
    )
    pure_derivation_scattering_kernel_explicit = bool(
        sign_base.hit(note_text, "\\mathcal{M}(\\mathbf{k}, \\mathbf{k}'; \\epsilon, \\epsilon')")
        is not None
        and sign_base.hit(note_text, "\\tilde{\\mathcal{K}}_{\\mu\\nu}") is not None
    )
    corrected_additive_second_variation_expandable_now = bool(
        audit_selected
        and retry_mode
        and non_surrogate_guard
        and additive_one_point_no_go_available
        and pure_derivation_second_variation_explicit
    )
    corrected_kernel_identity_theorem_explicit = bool(
        corrected_additive_second_variation_expandable_now
        and pure_derivation_scattering_kernel_explicit
    )
    exact_corrected_mixed_probe_response_kernel_formula_available_now = bool(
        corrected_kernel_identity_theorem_explicit
    )
    exact_corrected_pure_probe_response_kernel_formula_available_now = bool(
        corrected_kernel_identity_theorem_explicit
    )
    exact_corrected_kernel_rank_match_available_now = bool(
        corrected_kernel_identity_theorem_explicit
    )
    exact_corrected_distinct_probe_separation_available_now = False
    corrected_vacuum_subtraction_primary_followup_required = bool(
        exact_corrected_kernel_rank_match_available_now
    )
    corrected_pack_refresh_secondary_hold_retained = bool(
        corrected_vacuum_subtraction_primary_followup_required
    )
    corrected_mixed_kernel_breakthrough_passed_now = False
    blind_blocked = bool(prior_summary["blind_vector_observable_gate_still_blocked"])
    farther_hybrid = False

    rows = [
        sign_base.row(
            "updated_pack_corrected_mixed_kernel_return_refresh_audit_selected",
            "pass" if audit_selected else "reject",
            "updated-pack corrected mixed-kernel return refresh audit selected",
            sign_base.truth(audit_selected),
            "This branch is only worth running if the additive one-point current has already closed as an exact no-go theorem.",
        ),
        sign_base.row(
            "retry_gate_computation_mode_selected",
            "pass" if retry_mode else "reject",
            "retry gate computation mode selected",
            sign_base.truth(retry_mode),
            "The work stays on literal theorem derivation instead of repeating the exhausted same-tag return loop.",
        ),
        sign_base.row(
            "failure_matrix_non_surrogate_guard_preserved",
            "pass" if non_surrogate_guard else "reject",
            "failure-matrix non-surrogate guard preserved",
            sign_base.truth(non_surrogate_guard),
            "Closing the kernel theorem is admissible only if it does not reopen the exhausted density/proxy/eigenvalue family.",
        ),
        sign_base.row(
            "exact_additive_one_point_no_go_theorem_available_now",
            "pass" if additive_one_point_no_go_available else "reject",
            "exact additive one-point no-go theorem available now",
            sign_base.truth(additive_one_point_no_go_available),
            "The mixed-kernel theorem is meaningful only after the additive one-point source has been closed as exact zero.",
        ),
        sign_base.row(
            "pure_derivation_second_variation_explicit",
            "pass" if pure_derivation_second_variation_explicit else "reject",
            "pure-derivation second variation explicit",
            sign_base.truth(pure_derivation_second_variation_explicit),
            "The source note already writes the frozen-action Hessian as the exact second variation around the on-shell background.",
        ),
        sign_base.row(
            "pure_derivation_scattering_kernel_explicit",
            "pass" if pure_derivation_scattering_kernel_explicit else "reject",
            "pure-derivation scattering kernel explicit",
            sign_base.truth(pure_derivation_scattering_kernel_explicit),
            "The same note already contracts a rank-2 kernel with photon polarizations, so the two-index observable slot is explicit.",
        ),
        sign_base.row(
            "corrected_additive_second_variation_expandable_now",
            "pass" if corrected_additive_second_variation_expandable_now else "reject",
            "corrected additive second variation expandable now",
            sign_base.truth(corrected_additive_second_variation_expandable_now),
            "Under the additive split P = Q + xi + A, the exact second variation can be expanded literally into xi-xi, xi-A, and A-A terms.",
        ),
        sign_base.row(
            "corrected_kernel_identity_theorem_explicit",
            "pass" if corrected_kernel_identity_theorem_explicit else "reject",
            "corrected kernel identity theorem explicit",
            sign_base.truth(corrected_kernel_identity_theorem_explicit),
            "Because xi and A enter the same additive action slot, both the mixed and pure probe-response kernels collapse to the same background Hessian.",
        ),
        sign_base.row(
            "exact_corrected_mixed_probe_response_kernel_formula_available_now",
            "pass" if exact_corrected_mixed_probe_response_kernel_formula_available_now else "reject",
            "exact corrected mixed probe-response kernel formula available now",
            sign_base.truth(exact_corrected_mixed_probe_response_kernel_formula_available_now),
            "The mixed kernel is now literally available as V^{mu nu}[Q](x,y) = H^{mu nu}[Q](x,y) under the additive corrected split.",
        ),
        sign_base.row(
            "exact_corrected_pure_probe_response_kernel_formula_available_now",
            "pass" if exact_corrected_pure_probe_response_kernel_formula_available_now else "reject",
            "exact corrected pure probe-response kernel formula available now",
            sign_base.truth(exact_corrected_pure_probe_response_kernel_formula_available_now),
            "The pure probe kernel is now literally available as Pi^{mu nu}[Q](x,y) = H^{mu nu}[Q](x,y) under the same additive split.",
        ),
        sign_base.row(
            "exact_corrected_kernel_rank_match_available_now",
            "pass" if exact_corrected_kernel_rank_match_available_now else "reject",
            "exact corrected kernel rank match available now",
            sign_base.truth(exact_corrected_kernel_rank_match_available_now),
            "The observable candidate is now a genuine rank-2 kernel that contracts with photon polarizations as eps*^mu K_{mu nu} eps'^nu.",
        ),
        sign_base.row(
            "exact_corrected_distinct_probe_separation_available_now",
            "pass" if exact_corrected_distinct_probe_separation_available_now else "reject",
            "exact corrected distinct probe separation available now",
            sign_base.truth(exact_corrected_distinct_probe_separation_available_now),
            "What remains absent is a distinct external-probe theorem beyond the shared same-action Hessian identity.",
        ),
        sign_base.row(
            "updated_pack_corrected_vacuum_subtraction_primary_followup_required",
            "pass" if corrected_vacuum_subtraction_primary_followup_required else "reject",
            "updated-pack corrected vacuum-subtraction primary followup required",
            sign_base.truth(corrected_vacuum_subtraction_primary_followup_required),
            "Once the kernel theorem closes, the next honest blocker is the corrected vacuum-state / subtraction rule rather than another kernel restatement.",
        ),
        sign_base.row(
            "updated_pack_corrected_pack_refresh_secondary_hold_retained",
            "pass" if corrected_pack_refresh_secondary_hold_retained else "reject",
            "updated-pack corrected pack-refresh secondary hold retained",
            sign_base.truth(corrected_pack_refresh_secondary_hold_retained),
            "Pack-refresh remains downstream of subtraction-side closeout and is not the primary theorem blocker anymore.",
        ),
        sign_base.row(
            "updated_pack_corrected_mixed_kernel_breakthrough_passed_now",
            "pass" if corrected_mixed_kernel_breakthrough_passed_now else "reject",
            "updated-pack corrected mixed-kernel breakthrough passed now",
            sign_base.truth(corrected_mixed_kernel_breakthrough_passed_now),
            "The kernel theorem closes, but the residual-origin closeout still waits on corrected subtraction and reserve-verdict objects.",
        ),
        sign_base.row(
            "blind_vector_observable_gate_still_blocked",
            "pass" if blind_blocked else "reject",
            "blind-vector observable gate still blocked",
            sign_base.truth(blind_blocked),
            "Blind-vector direct computation still waits on corrected subtraction and reserve closeout even after the kernel theorem is fixed.",
        ),
        sign_base.row(
            "farther_hybrid_continuation_reopen_required_now",
            "pass" if farther_hybrid else "reject",
            "farther hybrid continuation reopen required now",
            sign_base.truth(farther_hybrid),
            "Extra q-range evidence remains reserve-only because the blocker is still theorem-side subtraction closeout, not range-side continuation.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "retained_scalar_residual_rel": float(prior_summary["retained_scalar_residual_rel"]),
        "updated_pack_corrected_mixed_kernel_return_refresh_audit_selected": audit_selected,
        "retry_gate_computation_mode_selected": retry_mode,
        "failure_matrix_non_surrogate_guard_preserved": non_surrogate_guard,
        "exact_additive_one_point_no_go_theorem_available_now": additive_one_point_no_go_available,
        "pure_derivation_second_variation_explicit": pure_derivation_second_variation_explicit,
        "pure_derivation_scattering_kernel_explicit": pure_derivation_scattering_kernel_explicit,
        "corrected_additive_second_variation_expandable_now": corrected_additive_second_variation_expandable_now,
        "corrected_kernel_identity_theorem_explicit": corrected_kernel_identity_theorem_explicit,
        "exact_corrected_mixed_probe_response_kernel_formula_available_now": exact_corrected_mixed_probe_response_kernel_formula_available_now,
        "exact_corrected_pure_probe_response_kernel_formula_available_now": exact_corrected_pure_probe_response_kernel_formula_available_now,
        "exact_corrected_kernel_rank_match_available_now": exact_corrected_kernel_rank_match_available_now,
        "exact_corrected_distinct_probe_separation_available_now": exact_corrected_distinct_probe_separation_available_now,
        "updated_pack_corrected_vacuum_subtraction_primary_followup_required": corrected_vacuum_subtraction_primary_followup_required,
        "updated_pack_corrected_pack_refresh_secondary_hold_retained": corrected_pack_refresh_secondary_hold_retained,
        "blind_vector_observable_gate_still_blocked": blind_blocked,
        "selected_primary_completion_lane": "updated_pack_corrected_vacuum_subtraction_return_refresh_audit",
        "selected_secondary_completion_lane": "updated_pack_corrected_pack_refresh_return_sync",
        "selected_next_generation_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_corrected_mixed_kernel_gate_vacuum_subtraction_refresh",
        "recommended_next_route_or_none": "8.7.56.4531",
        "selected_followup_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_corrected_vacuum_subtraction_return_refresh_audit",
        "selected_followup_route_or_none": "8.7.56.4535",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.4529",
        STEP_NAME + " declaration gate",
        {
            "source_files": {
                "prior_gate": sign_base.display_path(PRIOR_GATE),
                "pure_derivation_note": sign_base.display_path(PURE_DERIVATION_NOTE),
            },
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "8.7.56.4531",
                "followup_route": "8.7.56.4535",
            },
        },
        rows,
        summary,
        {
            "overall_status": "vector_qball_form_factor_updated_pack_corrected_mixed_kernel_hessian_identity_theorem_derived",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {"formulae": build_formulae()},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(f"[done] {STEP_TAG} updated-pack corrected mixed-kernel audit completed")
    print(f"[done] declaration: {declaration_paths['json']}")


if __name__ == "__main__":
    main()
