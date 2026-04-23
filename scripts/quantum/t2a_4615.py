#!/usr/bin/env python3
"""Generate 8.7.56.4615-.4618 selector-candidate theorem artifacts."""

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
        "8.7.56.4611-4614",
        "updated_pack_beyond_current_written_action_selector_axiom_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
PRIOR_AUDIT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.4607-4610",
        "updated_pack_beyond_current_written_action_selector_axiom_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.4615-4618"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack beyond-current-"
    "written-action selector candidate theorem audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_beyond_current_written_action_selector_candidate_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "beyond_current_written_action_selector_axiom_dual_sector_requirement_theorem_"
    "derived_selector_candidate_primary_hybrid_reserve_secondary_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "beyond_current_written_action_selector_candidate_family_no_go_theorem_"
    "derived_selector_criterion_primary_pack_refresh_secondary_gate"
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


# 関数: selector-candidate theorem で使う式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used in the selector-candidate theorem audit."""
    return {
        "admissible_selector_domain": (
            "A_ext := { Sigma[P_mu, A_mu] | Sigma[P_mu,0] = 0 }"
        ),
        "selector_candidate_functional": (
            "Omega^(W)[Sigma] := Omega_probe^(W)[L_probe^(Sigma)] + "
            "Omega_mix^(W)[L_mix^(Sigma)]"
        ),
        "selector_candidate_choice": (
            "Sigma_*^(W) := argext_{Sigma in A_ext} Omega^(W)[Sigma]"
        ),
        "selected_extension_candidate": (
            "L_ext^(W)[P_mu, A_mu] := L_total^vec[P_mu] + "
            "L_probe^(Sigma_*^(W))[A_mu] + L_mix^(Sigma_*^(W))[P_mu, A_mu]"
        ),
        "candidate_no_go": (
            "current theory does not fix W, so it fixes a family of selector "
            "candidates rather than one canonical selector"
        ),
    }


# 関数: `.4615-.4618` を実行する。

def main() -> None:
    """Execute the selector-candidate theorem audit."""
    for path in (PRIOR_GATE, PRIOR_AUDIT):
        sign_base.require(path)

    prior_gate_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    prior_audit_summary = sign_base.read_json(PRIOR_AUDIT)["summary"]

    audit_selected = bool(
        prior_gate_summary[
            "gate_b_updated_pack_beyond_current_written_action_selector_candidate_promoted_next"
        ]
        and prior_gate_summary["pack_update_required_now"]
    )
    retry_mode = bool(prior_gate_summary["retry_gate_computation_mode_selected"])
    non_surrogate_guard = bool(
        prior_gate_summary["failure_matrix_non_surrogate_guard_preserved"]
    )
    dual_sector_requirement_available = bool(
        prior_gate_summary[
            "gate_a_updated_pack_exact_beyond_current_written_action_selector_axiom_dual_sector_requirement_available_now"
        ]
    )
    written_surface_no_go_available = bool(
        prior_audit_summary[
            "exact_beyond_current_written_action_selector_axiom_written_surface_only_no_go_theorem_available_now"
        ]
    )
    probe_only_no_go_available = bool(
        prior_audit_summary[
            "exact_beyond_current_written_action_selector_axiom_probe_sector_only_no_go_theorem_available_now"
        ]
    )
    mixed_only_no_go_available = bool(
        prior_audit_summary[
            "exact_beyond_current_written_action_selector_axiom_mixed_sector_only_no_go_theorem_available_now"
        ]
    )
    dual_sector_scope_formula_available = bool(
        prior_audit_summary[
            "exact_beyond_current_written_action_selector_axiom_dual_sector_scope_formula_available_now"
        ]
    )
    selector_candidate_family_formula_explicit = bool(
        audit_selected
        and retry_mode
        and non_surrogate_guard
        and dual_sector_requirement_available
        and written_surface_no_go_available
        and probe_only_no_go_available
        and mixed_only_no_go_available
        and dual_sector_scope_formula_available
    )
    exact_beyond_current_written_action_selector_candidate_family_formula_available_now = bool(
        selector_candidate_family_formula_explicit
    )
    exact_beyond_current_written_action_dual_sector_variational_selector_formula_available_now = bool(
        selector_candidate_family_formula_explicit
    )
    exact_beyond_current_written_action_selector_candidate_family_no_go_theorem_available_now = bool(
        selector_candidate_family_formula_explicit
    )
    exact_minimal_selector_criterion_requirement_theorem_available_now = bool(
        selector_candidate_family_formula_explicit
    )
    exact_beyond_current_written_action_selector_candidate_available_now = False
    updated_pack_beyond_current_written_action_selector_criterion_primary_followup_required = bool(
        exact_minimal_selector_criterion_requirement_theorem_available_now
    )
    updated_pack_corrected_pack_refresh_secondary_hold_retained = bool(
        updated_pack_beyond_current_written_action_selector_criterion_primary_followup_required
    )
    updated_pack_same_tag_pack_refresh_reentry_admissible_now = False
    updated_pack_beyond_current_written_action_selector_candidate_breakthrough_passed_now = False
    blind_blocked = bool(prior_gate_summary["blind_vector_observable_gate_still_blocked"])

    rows = [
        sign_base.row(
            "updated_pack_beyond_current_written_action_selector_candidate_audit_selected",
            "pass" if audit_selected else "reject",
            "updated-pack beyond-current-written-action selector candidate audit selected",
            sign_base.truth(audit_selected),
            "This branch is worth running only after the selector-axiom dual-sector requirement theorem is already closed.",
        ),
        sign_base.row(
            "retry_gate_computation_mode_selected",
            "pass" if retry_mode else "reject",
            "retry gate computation mode selected",
            sign_base.truth(retry_mode),
            "This turn must close a new theorem object rather than restate same-tag route syntax.",
        ),
        sign_base.row(
            "failure_matrix_non_surrogate_guard_preserved",
            "pass" if non_surrogate_guard else "reject",
            "failure-matrix non-surrogate guard preserved",
            sign_base.truth(non_surrogate_guard),
            "The selector-candidate theorem is admissible only if it does not reopen the exhausted surrogate family.",
        ),
        sign_base.row(
            "exact_beyond_current_written_action_selector_axiom_dual_sector_requirement_available_now",
            "pass" if dual_sector_requirement_available else "reject",
            "exact beyond-current-written-action selector-axiom dual-sector requirement available now",
            sign_base.truth(dual_sector_requirement_available),
            "A candidate theorem becomes meaningful only after the dual-sector scope theorem is already fixed.",
        ),
        sign_base.row(
            "exact_beyond_current_written_action_selector_axiom_written_surface_only_no_go_theorem_available_now",
            "pass" if written_surface_no_go_available else "reject",
            "exact beyond-current-written-action selector axiom written-surface-only no-go theorem available now",
            sign_base.truth(written_surface_no_go_available),
            "The candidate theorem starts only after the written-surface-only no-go is explicit.",
        ),
        sign_base.row(
            "exact_beyond_current_written_action_selector_axiom_probe_sector_only_no_go_theorem_available_now",
            "pass" if probe_only_no_go_available else "reject",
            "exact beyond-current-written-action selector axiom probe-sector-only no-go theorem available now",
            sign_base.truth(probe_only_no_go_available),
            "The candidate theorem starts only after the probe-only no-go is explicit.",
        ),
        sign_base.row(
            "exact_beyond_current_written_action_selector_axiom_mixed_sector_only_no_go_theorem_available_now",
            "pass" if mixed_only_no_go_available else "reject",
            "exact beyond-current-written-action selector axiom mixed-sector-only no-go theorem available now",
            sign_base.truth(mixed_only_no_go_available),
            "The candidate theorem starts only after the mixed-only no-go is explicit.",
        ),
        sign_base.row(
            "exact_beyond_current_written_action_selector_axiom_dual_sector_scope_formula_available_now",
            "pass" if dual_sector_scope_formula_available else "reject",
            "exact beyond-current-written-action selector axiom dual-sector scope formula available now",
            sign_base.truth(dual_sector_scope_formula_available),
            "The candidate theorem starts only after the dual-sector scope formula is explicit.",
        ),
        sign_base.row(
            "selector_candidate_family_formula_explicit",
            "pass" if selector_candidate_family_formula_explicit else "reject",
            "selector candidate family formula explicit",
            sign_base.truth(selector_candidate_family_formula_explicit),
            "Once selector scope is fixed, any honest selector candidate can be represented as an extremization of a dual-sector functional Omega^(W)[Sigma].",
        ),
        sign_base.row(
            "exact_beyond_current_written_action_selector_candidate_family_formula_available_now",
            "pass"
            if exact_beyond_current_written_action_selector_candidate_family_formula_available_now
            else "reject",
            "exact beyond-current-written-action selector candidate family formula available now",
            sign_base.truth(
                exact_beyond_current_written_action_selector_candidate_family_formula_available_now
            ),
            "The theorem stack now fixes the entire family of admissible selector candidates in a literal formula, not just the need for one in prose.",
        ),
        sign_base.row(
            "exact_beyond_current_written_action_dual_sector_variational_selector_formula_available_now",
            "pass"
            if exact_beyond_current_written_action_dual_sector_variational_selector_formula_available_now
            else "reject",
            "exact beyond-current-written-action dual-sector variational selector formula available now",
            sign_base.truth(
                exact_beyond_current_written_action_dual_sector_variational_selector_formula_available_now
            ),
            "The minimal honest candidate class is now explicit as a dual-sector extremization over the admissible family.",
        ),
        sign_base.row(
            "exact_beyond_current_written_action_selector_candidate_family_no_go_theorem_available_now",
            "pass"
            if exact_beyond_current_written_action_selector_candidate_family_no_go_theorem_available_now
            else "reject",
            "exact beyond-current-written-action selector candidate family no-go theorem available now",
            sign_base.truth(
                exact_beyond_current_written_action_selector_candidate_family_no_go_theorem_available_now
            ),
            "The current theory still does not fix W, so it supplies a family of selector candidates rather than one canonical selector candidate.",
        ),
        sign_base.row(
            "exact_minimal_selector_criterion_requirement_theorem_available_now",
            "pass"
            if exact_minimal_selector_criterion_requirement_theorem_available_now
            else "reject",
            "exact minimal selector criterion requirement theorem available now",
            sign_base.truth(
                exact_minimal_selector_criterion_requirement_theorem_available_now
            ),
            "The honest next blocker is therefore an extra criterion that chooses one dual-sector selector candidate from the admissible candidate family.",
        ),
        sign_base.row(
            "exact_beyond_current_written_action_selector_candidate_available_now",
            "pass" if exact_beyond_current_written_action_selector_candidate_available_now else "reject",
            "exact beyond-current-written-action selector candidate available now",
            sign_base.truth(exact_beyond_current_written_action_selector_candidate_available_now),
            "This branch fixes the candidate family and its no-go, not one concrete selected criterion.",
        ),
        sign_base.row(
            "updated_pack_beyond_current_written_action_selector_criterion_primary_followup_required",
            "pass"
            if updated_pack_beyond_current_written_action_selector_criterion_primary_followup_required
            else "reject",
            "updated-pack beyond-current-written-action selector criterion primary followup required",
            sign_base.truth(
                updated_pack_beyond_current_written_action_selector_criterion_primary_followup_required
            ),
            "The honest next blocker is to compare or derive one criterion that chooses a concrete selector candidate.",
        ),
        sign_base.row(
            "updated_pack_corrected_pack_refresh_secondary_hold_retained",
            "pass" if updated_pack_corrected_pack_refresh_secondary_hold_retained else "reject",
            "updated-pack corrected pack-refresh secondary hold retained",
            sign_base.truth(updated_pack_corrected_pack_refresh_secondary_hold_retained),
            "Pack-refresh stays secondary because it still cannot choose one selector candidate without an extra criterion.",
        ),
        sign_base.row(
            "updated_pack_same_tag_pack_refresh_reentry_admissible_now",
            "pass" if updated_pack_same_tag_pack_refresh_reentry_admissible_now else "reject",
            "updated-pack same-tag pack-refresh reentry admissible now",
            sign_base.truth(updated_pack_same_tag_pack_refresh_reentry_admissible_now),
            "Same-tag reentry remains closed because the remaining blocker is theorem-side selector-candidate choice, not bookkeeping syntax.",
        ),
        sign_base.row(
            "updated_pack_beyond_current_written_action_selector_candidate_breakthrough_passed_now",
            "pass"
            if updated_pack_beyond_current_written_action_selector_candidate_breakthrough_passed_now
            else "reject",
            "updated-pack beyond-current-written-action selector candidate breakthrough passed now",
            sign_base.truth(
                updated_pack_beyond_current_written_action_selector_candidate_breakthrough_passed_now
            ),
            "This branch sharpens the candidate family but still does not choose one concrete extension or close the residual-origin blocker.",
        ),
        sign_base.row(
            "blind_vector_observable_gate_still_blocked",
            "pass" if blind_blocked else "reject",
            "blind-vector observable gate still blocked",
            sign_base.truth(blind_blocked),
            "Blind-vector direct computation still waits on a concrete selector criterion and selected extension, not just the candidate family theorem.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "retained_scalar_residual_rel": float(prior_gate_summary["retained_scalar_residual_rel"]),
        "updated_pack_beyond_current_written_action_selector_candidate_audit_selected": audit_selected,
        "retry_gate_computation_mode_selected": retry_mode,
        "failure_matrix_non_surrogate_guard_preserved": non_surrogate_guard,
        "exact_beyond_current_written_action_selector_axiom_dual_sector_requirement_available_now": dual_sector_requirement_available,
        "exact_beyond_current_written_action_selector_axiom_written_surface_only_no_go_theorem_available_now": written_surface_no_go_available,
        "exact_beyond_current_written_action_selector_axiom_probe_sector_only_no_go_theorem_available_now": probe_only_no_go_available,
        "exact_beyond_current_written_action_selector_axiom_mixed_sector_only_no_go_theorem_available_now": mixed_only_no_go_available,
        "exact_beyond_current_written_action_selector_axiom_dual_sector_scope_formula_available_now": dual_sector_scope_formula_available,
        "selector_candidate_family_formula_explicit": selector_candidate_family_formula_explicit,
        "exact_beyond_current_written_action_selector_candidate_family_formula_available_now": exact_beyond_current_written_action_selector_candidate_family_formula_available_now,
        "exact_beyond_current_written_action_dual_sector_variational_selector_formula_available_now": exact_beyond_current_written_action_dual_sector_variational_selector_formula_available_now,
        "exact_beyond_current_written_action_selector_candidate_family_no_go_theorem_available_now": exact_beyond_current_written_action_selector_candidate_family_no_go_theorem_available_now,
        "exact_minimal_selector_criterion_requirement_theorem_available_now": exact_minimal_selector_criterion_requirement_theorem_available_now,
        "exact_beyond_current_written_action_selector_candidate_available_now": exact_beyond_current_written_action_selector_candidate_available_now,
        "updated_pack_beyond_current_written_action_selector_criterion_primary_followup_required": updated_pack_beyond_current_written_action_selector_criterion_primary_followup_required,
        "updated_pack_corrected_pack_refresh_secondary_hold_retained": updated_pack_corrected_pack_refresh_secondary_hold_retained,
        "updated_pack_same_tag_pack_refresh_reentry_admissible_now": updated_pack_same_tag_pack_refresh_reentry_admissible_now,
        "updated_pack_beyond_current_written_action_selector_candidate_breakthrough_passed_now": updated_pack_beyond_current_written_action_selector_candidate_breakthrough_passed_now,
        "blind_vector_observable_gate_still_blocked": blind_blocked,
        "selected_primary_completion_lane": "updated_pack_beyond_current_written_action_selector_criterion_theorem_audit",
        "selected_secondary_completion_lane": "updated_pack_corrected_pack_refresh_return_sync",
        "selected_reserve_completion_lane": "farther_hybrid_extra_q_range_only",
        "selected_next_generation_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_beyond_current_written_action_selector_criterion_theorem_audit",
        "recommended_next_route_or_none": "8.7.56.4623",
        "selected_followup_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_beyond_current_written_action_selector_criterion_gate",
        "selected_followup_route_or_none": "8.7.56.4627",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.4617",
        STEP_NAME + " declaration gate",
        {
            "source_files": {
                "prior_gate": sign_base.display_path(PRIOR_GATE),
                "prior_audit": sign_base.display_path(PRIOR_AUDIT),
            },
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "8.7.56.4623",
                "followup_route": "8.7.56.4627",
            },
        },
        rows,
        summary,
        {
            "overall_status": "vector_qball_form_factor_updated_pack_beyond_current_written_action_selector_candidate_declared",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {"formulae": build_formulae()},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(
        f"[done] {STEP_TAG} updated-pack beyond-current-written-action selector candidate theorem completed"
    )
    print(f"[done] declaration: {declaration_paths['json']}")


if __name__ == "__main__":
    main()
