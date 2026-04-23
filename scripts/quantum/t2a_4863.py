#!/usr/bin/env python3
"""Generate 8.7.56.4863-.4866 selected-extension-convention-selector selector-criterion theorem artifacts."""

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
        "8.7.56.4859-4862",
        "updated_pack_beyond_current_written_action_selected_extension_convention_selector_selector_candidate_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
PRIOR_AUDIT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.4855-4858",
        "updated_pack_beyond_current_written_action_selected_extension_convention_selector_selector_candidate_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.4863-4866"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack beyond-current-"
    "written-action selected extension convention selector selector criterion "
    "theorem audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_beyond_current_written_action_selected_extension_convention_selector_selector_criterion_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "beyond_current_written_action_selected_extension_convention_selector_"
    "selector_candidate_family_no_go_theorem_derived_selected_extension_"
    "convention_selector_selector_criterion_primary_hybrid_reserve_secondary_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "beyond_current_written_action_selected_extension_convention_selector_"
    "selector_criterion_monotone_equivalence_no_go_theorem_derived_selected_"
    "extension_convention_selector_selector_representative_primary_pack_refresh_"
    "secondary_gate"
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


# 関数: selector-criterion theorem で使う式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used in the selector-criterion theorem audit."""
    return {
        "selector_candidate_choice": (
            "A_conv_ext,*(W) := argext_{A_conv_ext in Rep_sel_conv_ext[B_sel_conv_ext;W,K]} "
            "Omega_sel_conv_ext^(W)[A_conv_ext]"
        ),
        "criterion_reparametrization": (
            "Omega_sel_conv_ext^(W')[A_conv_ext] := phi(Omega_sel_conv_ext^(W)[A_conv_ext]) "
            "with phi strictly monotone"
        ),
        "criterion_order_equivalence": (
            "W ~ W' iff argext_{A_conv_ext in Rep_sel_conv_ext[B_sel_conv_ext;W,K]} "
            "Omega_sel_conv_ext^(W')[A_conv_ext] = argext_{A_conv_ext in "
            "Rep_sel_conv_ext[B_sel_conv_ext;W,K]} Omega_sel_conv_ext^(W)[A_conv_ext]"
        ),
        "criterion_equivalence_class": "[W] := { W' | W' ~ W }",
        "criterion_no_go": (
            "current theory fixes only the order-class [W], not one canonical "
            "criterion representative W"
        ),
    }


# 関数: `.4863-.4866` を実行する。

def main() -> None:
    """Execute the selector-criterion theorem audit."""
    for path in (PRIOR_GATE, PRIOR_AUDIT):
        sign_base.require(path)

    prior_gate_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    prior_audit_summary = sign_base.read_json(PRIOR_AUDIT)["summary"]

    audit_selected = bool(
        prior_gate_summary[
            "gate_b_updated_pack_beyond_current_written_action_selected_extension_convention_selector_selector_criterion_promoted_next"
        ]
        and prior_gate_summary["pack_update_required_now"]
    )
    retry_mode = bool(prior_gate_summary["retry_gate_computation_mode_selected"])
    non_surrogate_guard = bool(
        prior_gate_summary["failure_matrix_non_surrogate_guard_preserved"]
    )
    selector_candidate_family_no_go_available = bool(
        prior_gate_summary[
            "gate_a_updated_pack_exact_beyond_current_written_action_selected_extension_convention_selector_selector_candidate_family_no_go_available_now"
        ]
    )
    selector_candidate_family_formula_available = bool(
        prior_audit_summary[
            "exact_beyond_current_written_action_selected_extension_convention_selector_selector_candidate_family_formula_available_now"
        ]
    )
    dual_sector_variational_selector_formula_available = bool(
        prior_audit_summary[
            "exact_beyond_current_written_action_selected_extension_convention_selector_dual_sector_variational_selector_formula_available_now"
        ]
    )
    selector_criterion_requirement_available = bool(
        prior_audit_summary[
            "exact_minimal_selected_extension_convention_selector_selector_criterion_requirement_theorem_available_now"
        ]
    )
    criterion_order_class_formula_explicit = bool(
        audit_selected
        and retry_mode
        and non_surrogate_guard
        and selector_candidate_family_no_go_available
        and selector_candidate_family_formula_available
        and dual_sector_variational_selector_formula_available
        and selector_criterion_requirement_available
    )
    exact_beyond_current_written_action_selected_extension_convention_selector_selector_criterion_equivalence_class_formula_available_now = bool(
        criterion_order_class_formula_explicit
    )
    exact_beyond_current_written_action_selected_extension_convention_selector_selector_criterion_monotone_equivalence_theorem_available_now = bool(
        criterion_order_class_formula_explicit
    )
    exact_beyond_current_written_action_selected_extension_convention_selector_selector_criterion_no_go_theorem_available_now = bool(
        criterion_order_class_formula_explicit
    )
    exact_minimal_selected_extension_convention_selector_selector_representative_requirement_theorem_available_now = bool(
        criterion_order_class_formula_explicit
    )
    exact_beyond_current_written_action_selected_extension_convention_selector_selector_criterion_available_now = False
    updated_pack_beyond_current_written_action_selected_extension_convention_selector_selector_representative_primary_followup_required = bool(
        exact_minimal_selected_extension_convention_selector_selector_representative_requirement_theorem_available_now
    )
    updated_pack_corrected_pack_refresh_secondary_hold_retained = bool(
        updated_pack_beyond_current_written_action_selected_extension_convention_selector_selector_representative_primary_followup_required
    )
    updated_pack_same_tag_pack_refresh_reentry_admissible_now = False
    updated_pack_beyond_current_written_action_selected_extension_convention_selector_selector_criterion_breakthrough_passed_now = False
    blind_blocked = bool(prior_gate_summary["blind_vector_observable_gate_still_blocked"])

    rows = [
        sign_base.row(
            "updated_pack_beyond_current_written_action_selected_extension_convention_selector_selector_criterion_audit_selected",
            "pass" if audit_selected else "reject",
            "updated-pack beyond-current-written-action selector criterion audit selected",
            sign_base.truth(audit_selected),
            "This branch is worth running only after the selector-candidate family is already explicit and same-tag route repetition remains closed.",
        ),
        sign_base.row(
            "retry_gate_computation_mode_selected",
            "pass" if retry_mode else "reject",
            "retry gate computation mode selected",
            sign_base.truth(retry_mode),
            "This turn must close a new theorem object rather than rephrase the candidate-family no-go.",
        ),
        sign_base.row(
            "failure_matrix_non_surrogate_guard_preserved",
            "pass" if non_surrogate_guard else "reject",
            "failure-matrix non-surrogate guard preserved",
            sign_base.truth(non_surrogate_guard),
            "The criterion theorem is admissible only if it does not reopen the exhausted surrogate family.",
        ),
        sign_base.row(
            "exact_beyond_current_written_action_selected_extension_convention_selector_selector_candidate_family_no_go_available_now",
            "pass" if selector_candidate_family_no_go_available else "reject",
            "exact beyond-current-written-action selector candidate family no-go available now",
            sign_base.truth(selector_candidate_family_no_go_available),
            "The criterion theorem starts only after the theory already closes that it fixes a family of selector candidates rather than one concrete candidate.",
        ),
        sign_base.row(
            "exact_beyond_current_written_action_selected_extension_convention_selector_selector_candidate_family_formula_available_now",
            "pass" if selector_candidate_family_formula_available else "reject",
            "exact beyond-current-written-action selector candidate family formula available now",
            sign_base.truth(selector_candidate_family_formula_available),
            "The criterion theorem starts only after the admissible selector-candidate family is already explicit.",
        ),
        sign_base.row(
            "exact_beyond_current_written_action_selected_extension_convention_selector_dual_sector_variational_selector_formula_available_now",
            "pass" if dual_sector_variational_selector_formula_available else "reject",
            "exact beyond-current-written-action dual-sector variational selector formula available now",
            sign_base.truth(dual_sector_variational_selector_formula_available),
            "The criterion theorem uses the already closed variational representation Sigma_*^(W)=argext Omega^(W).",
        ),
        sign_base.row(
            "exact_minimal_selected_extension_convention_selector_selector_criterion_requirement_theorem_available_now",
            "pass" if selector_criterion_requirement_available else "reject",
            "exact minimal selector criterion requirement theorem available now",
            sign_base.truth(selector_criterion_requirement_available),
            "The previous branch already fixed that some extra selector criterion is required.",
        ),
        sign_base.row(
            "criterion_order_class_formula_explicit",
            "pass" if criterion_order_class_formula_explicit else "reject",
            "criterion order-class formula explicit",
            sign_base.truth(criterion_order_class_formula_explicit),
            "Once only the extremizer matters, any strictly monotone reparametrization of the criterion yields the same selected Sigma_* and therefore belongs to the same criterion order class.",
        ),
        sign_base.row(
            "exact_beyond_current_written_action_selected_extension_convention_selector_selector_criterion_equivalence_class_formula_available_now",
            "pass"
            if exact_beyond_current_written_action_selected_extension_convention_selector_selector_criterion_equivalence_class_formula_available_now
            else "reject",
            "exact beyond-current-written-action selector criterion equivalence-class formula available now",
            sign_base.truth(
                exact_beyond_current_written_action_selected_extension_convention_selector_selector_criterion_equivalence_class_formula_available_now
            ),
            "The theorem stack now fixes the criterion only up to the equivalence class [W] of criteria that generate the same extremizer ordering.",
        ),
        sign_base.row(
            "exact_beyond_current_written_action_selected_extension_convention_selector_selector_criterion_monotone_equivalence_theorem_available_now",
            "pass"
            if exact_beyond_current_written_action_selected_extension_convention_selector_selector_criterion_monotone_equivalence_theorem_available_now
            else "reject",
            "exact beyond-current-written-action selector criterion monotone-equivalence theorem available now",
            sign_base.truth(
                exact_beyond_current_written_action_selected_extension_convention_selector_selector_criterion_monotone_equivalence_theorem_available_now
            ),
            "Any strictly monotone reparametrization phi of Omega^(W) preserves argext, so the current theory cannot distinguish criteria that differ only by order-preserving relabeling.",
        ),
        sign_base.row(
            "exact_beyond_current_written_action_selected_extension_convention_selector_selector_criterion_no_go_theorem_available_now",
            "pass"
            if exact_beyond_current_written_action_selected_extension_convention_selector_selector_criterion_no_go_theorem_available_now
            else "reject",
            "exact beyond-current-written-action selector criterion no-go theorem available now",
            sign_base.truth(
                exact_beyond_current_written_action_selected_extension_convention_selector_selector_criterion_no_go_theorem_available_now
            ),
            "The current theory therefore fixes only an order class of criteria and still does not supply one canonical concrete criterion.",
        ),
        sign_base.row(
            "exact_minimal_selected_extension_convention_selector_selector_representative_requirement_theorem_available_now",
            "pass"
            if exact_minimal_selected_extension_convention_selector_selector_representative_requirement_theorem_available_now
            else "reject",
            "exact minimal selector representative requirement theorem available now",
            sign_base.truth(
                exact_minimal_selected_extension_convention_selector_selector_representative_requirement_theorem_available_now
            ),
            "The honest next blocker is therefore not another criterion-family restatement but an extra representative rule that chooses one concrete criterion inside the monotone-equivalence class.",
        ),
        sign_base.row(
            "exact_beyond_current_written_action_selected_extension_convention_selector_selector_criterion_available_now",
            "pass" if exact_beyond_current_written_action_selected_extension_convention_selector_selector_criterion_available_now else "reject",
            "exact beyond-current-written-action selector criterion available now",
            sign_base.truth(exact_beyond_current_written_action_selected_extension_convention_selector_selector_criterion_available_now),
            "This branch closes the criterion order-class theorem and its no-go, not one concrete selected criterion.",
        ),
        sign_base.row(
            "updated_pack_beyond_current_written_action_selected_extension_convention_selector_selector_representative_primary_followup_required",
            "pass"
            if updated_pack_beyond_current_written_action_selected_extension_convention_selector_selector_representative_primary_followup_required
            else "reject",
            "updated-pack beyond-current-written-action selector representative primary followup required",
            sign_base.truth(
                updated_pack_beyond_current_written_action_selected_extension_convention_selector_selector_representative_primary_followup_required
            ),
            "The honest next blocker is to derive what extra representative rule could choose one concrete criterion from the monotone-equivalent class.",
        ),
        sign_base.row(
            "updated_pack_corrected_pack_refresh_secondary_hold_retained",
            "pass" if updated_pack_corrected_pack_refresh_secondary_hold_retained else "reject",
            "updated-pack corrected pack-refresh secondary hold retained",
            sign_base.truth(updated_pack_corrected_pack_refresh_secondary_hold_retained),
            "Pack-refresh stays secondary because it still cannot choose one concrete selector criterion.",
        ),
        sign_base.row(
            "updated_pack_same_tag_pack_refresh_reentry_admissible_now",
            "pass" if updated_pack_same_tag_pack_refresh_reentry_admissible_now else "reject",
            "updated-pack same-tag pack-refresh reentry admissible now",
            sign_base.truth(updated_pack_same_tag_pack_refresh_reentry_admissible_now),
            "Same-tag reentry remains closed because the blocker is theorem-side criterion representative choice, not bookkeeping syntax.",
        ),
        sign_base.row(
            "updated_pack_beyond_current_written_action_selected_extension_convention_selector_selector_criterion_breakthrough_passed_now",
            "pass"
            if updated_pack_beyond_current_written_action_selected_extension_convention_selector_selector_criterion_breakthrough_passed_now
            else "reject",
            "updated-pack beyond-current-written-action selector criterion breakthrough passed now",
            sign_base.truth(
                updated_pack_beyond_current_written_action_selected_extension_convention_selector_selector_criterion_breakthrough_passed_now
            ),
            "This branch sharpens the criterion lane but still does not choose one concrete criterion, one concrete selector candidate, or one selected extension.",
        ),
        sign_base.row(
            "blind_vector_observable_gate_still_blocked",
            "pass" if blind_blocked else "reject",
            "blind-vector observable gate still blocked",
            sign_base.truth(blind_blocked),
            "Blind-vector direct computation still waits on a concrete criterion, concrete candidate, and selected extension.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "retained_scalar_residual_rel": float(prior_gate_summary["retained_scalar_residual_rel"]),
        "updated_pack_beyond_current_written_action_selected_extension_convention_selector_selector_criterion_audit_selected": audit_selected,
        "retry_gate_computation_mode_selected": retry_mode,
        "failure_matrix_non_surrogate_guard_preserved": non_surrogate_guard,
        "exact_beyond_current_written_action_selected_extension_convention_selector_selector_candidate_family_no_go_available_now": selector_candidate_family_no_go_available,
        "exact_beyond_current_written_action_selected_extension_convention_selector_selector_candidate_family_formula_available_now": selector_candidate_family_formula_available,
        "exact_beyond_current_written_action_selected_extension_convention_selector_dual_sector_variational_selector_formula_available_now": dual_sector_variational_selector_formula_available,
        "exact_minimal_selected_extension_convention_selector_selector_criterion_requirement_theorem_available_now": selector_criterion_requirement_available,
        "criterion_order_class_formula_explicit": criterion_order_class_formula_explicit,
        "exact_beyond_current_written_action_selected_extension_convention_selector_selector_criterion_equivalence_class_formula_available_now": exact_beyond_current_written_action_selected_extension_convention_selector_selector_criterion_equivalence_class_formula_available_now,
        "exact_beyond_current_written_action_selected_extension_convention_selector_selector_criterion_monotone_equivalence_theorem_available_now": exact_beyond_current_written_action_selected_extension_convention_selector_selector_criterion_monotone_equivalence_theorem_available_now,
        "exact_beyond_current_written_action_selected_extension_convention_selector_selector_criterion_no_go_theorem_available_now": exact_beyond_current_written_action_selected_extension_convention_selector_selector_criterion_no_go_theorem_available_now,
        "exact_minimal_selected_extension_convention_selector_selector_representative_requirement_theorem_available_now": exact_minimal_selected_extension_convention_selector_selector_representative_requirement_theorem_available_now,
        "exact_beyond_current_written_action_selected_extension_convention_selector_selector_criterion_available_now": exact_beyond_current_written_action_selected_extension_convention_selector_selector_criterion_available_now,
        "updated_pack_beyond_current_written_action_selected_extension_convention_selector_selector_representative_primary_followup_required": updated_pack_beyond_current_written_action_selected_extension_convention_selector_selector_representative_primary_followup_required,
        "updated_pack_corrected_pack_refresh_secondary_hold_retained": updated_pack_corrected_pack_refresh_secondary_hold_retained,
        "updated_pack_same_tag_pack_refresh_reentry_admissible_now": updated_pack_same_tag_pack_refresh_reentry_admissible_now,
        "updated_pack_beyond_current_written_action_selected_extension_convention_selector_selector_criterion_breakthrough_passed_now": updated_pack_beyond_current_written_action_selected_extension_convention_selector_selector_criterion_breakthrough_passed_now,
        "blind_vector_observable_gate_still_blocked": blind_blocked,
        "selected_primary_completion_lane": "updated_pack_beyond_current_written_action_selected_extension_convention_selector_selector_representative_theorem_audit",
        "selected_secondary_completion_lane": "updated_pack_corrected_pack_refresh_return_sync",
        "selected_reserve_completion_lane": "farther_hybrid_extra_q_range_only",
        "selected_next_generation_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_beyond_current_written_action_selected_extension_convention_selector_selector_representative_theorem_audit",
        "recommended_next_route_or_none": "8.7.56.4871",
        "selected_followup_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_beyond_current_written_action_selected_extension_convention_selector_selector_representative_gate",
        "selected_followup_route_or_none": "8.7.56.4875",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.4865",
        STEP_NAME + " declaration gate",
        {
            "source_files": {
                "prior_gate": sign_base.display_path(PRIOR_GATE),
                "prior_audit": sign_base.display_path(PRIOR_AUDIT),
            },
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "8.7.56.4871",
                "followup_route": "8.7.56.4875",
            },
        },
        rows,
        summary,
        {
            "overall_status": "vector_qball_form_factor_updated_pack_beyond_current_written_action_selected_extension_convention_selector_selector_criterion_declared",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {"formulae": build_formulae()},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(
        f"[done] {STEP_TAG} updated-pack beyond-current-written-action selector criterion theorem completed"
    )
    print(f"[done] declaration: {declaration_paths['json']}")


if __name__ == "__main__":
    main()
