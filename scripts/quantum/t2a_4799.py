#!/usr/bin/env python3
"""Generate 8.7.56.4799-.4802 selected-extension-convention-selector-selected-extension artifacts."""

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
        "8.7.56.4795-4798",
        "updated_pack_beyond_current_written_action_selected_extension_convention_selector_selected_candidate_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
PRIOR_SELECTED_CANDIDATE_AUDIT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.4791-4794",
        "updated_pack_beyond_current_written_action_selected_extension_convention_selector_selected_candidate_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
PRIOR_SELECTED_EXTENSION_AUDIT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.4719-4722",
        "updated_pack_beyond_current_written_action_selected_extension_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.4799-4802"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack beyond-current-"
    "written-action selected extension convention selector selected extension theorem audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_beyond_current_written_action_selected_extension_convention_selector_selected_extension_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "beyond_current_written_action_selected_extension_convention_selector_"
    "selected_candidate_no_go_theorem_derived_selected_extension_convention_"
    "selector_selected_extension_primary_hybrid_reserve_secondary_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "beyond_current_written_action_selected_extension_convention_selector_"
    "selected_extension_no_go_theorem_derived_selected_extension_convention_"
    "selector_selected_extension_family_primary_pack_refresh_secondary_gate"
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


# 関数: selector-selected-extension theorem で使う式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used in the selected-extension-convention selector selected-extension audit."""
    return {
        "selected_candidate_family": (
            "Cand_sel_conv_ext[B_sel_conv_ext;B_conv_ext;W,K] := { "
            "C_ext,*^(B_conv_ext;W,K,A_conv_ext) | "
            "A_conv_ext in Rep_sel_conv_ext[B_sel_conv_ext;W,K] }"
        ),
        "selected_extension_formula": (
            "Sigma_sel_conv_ext,*^(B_sel_conv_ext;B_conv_ext;W,K,A_conv_ext) := "
            "C_ext,*^(B_conv_ext;W,K,A_conv_ext)(Ext_meas[W,K]) = "
            "Sigma_*^(W;K,chi_*^(B_conv_ext;W,K,A_conv_ext))"
        ),
        "selected_extension_no_go": (
            "without one concrete selector representative A_conv_ext, the current "
            "theorem stack still cannot choose one canonical selected extension"
        ),
    }


# 関数: `.4799-.4802` を実行する。

def main() -> None:
    """Execute the selected-extension-convention selector selected-extension theorem audit."""
    for path in (PRIOR_GATE, PRIOR_SELECTED_CANDIDATE_AUDIT, PRIOR_SELECTED_EXTENSION_AUDIT):
        sign_base.require(path)

    prior_gate_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    prior_selected_candidate_summary = sign_base.read_json(
        PRIOR_SELECTED_CANDIDATE_AUDIT
    )["summary"]
    prior_selected_extension_summary = sign_base.read_json(
        PRIOR_SELECTED_EXTENSION_AUDIT
    )["summary"]

    audit_selected = bool(
        prior_gate_summary[
            "gate_b_updated_pack_beyond_current_written_action_selected_extension_convention_selector_selected_extension_promoted_next"
        ]
        and prior_gate_summary["pack_update_required_now"]
    )
    retry_mode = bool(prior_gate_summary["retry_gate_computation_mode_selected"])
    non_surrogate_guard = bool(
        prior_gate_summary["failure_matrix_non_surrogate_guard_preserved"]
    )
    selector_selected_candidate_no_go_available = bool(
        prior_gate_summary[
            "gate_a_updated_pack_exact_beyond_current_written_action_selected_extension_convention_selector_selected_candidate_no_go_available_now"
        ]
    )
    selector_selected_candidate_formula_available = bool(
        prior_selected_candidate_summary[
            "exact_beyond_current_written_action_selected_extension_convention_selector_selected_candidate_formula_available_now"
        ]
    )
    selector_selected_candidate_family_available = bool(
        prior_selected_candidate_summary[
            "exact_beyond_current_written_action_selected_extension_convention_selector_selected_candidate_family_formula_available_now"
        ]
    )
    selected_extension_formula_available = bool(
        prior_selected_extension_summary[
            "exact_beyond_current_written_action_selected_extension_formula_available_now"
        ]
    )
    selector_selected_extension_requirement_available = bool(
        prior_selected_candidate_summary[
            "exact_minimal_selected_extension_convention_selector_selected_extension_requirement_theorem_available_now"
        ]
    )
    selector_selected_extension_formula_explicit = bool(
        audit_selected
        and retry_mode
        and non_surrogate_guard
        and selector_selected_candidate_no_go_available
        and selector_selected_candidate_formula_available
        and selector_selected_candidate_family_available
        and selected_extension_formula_available
        and selector_selected_extension_requirement_available
    )
    exact_beyond_current_written_action_selected_extension_convention_selector_selected_extension_formula_available_now = bool(
        selector_selected_extension_formula_explicit
    )
    exact_beyond_current_written_action_selected_extension_convention_selector_selected_extension_no_go_theorem_available_now = bool(
        selector_selected_extension_formula_explicit
    )
    exact_minimal_selected_extension_convention_selector_selected_extension_family_requirement_theorem_available_now = bool(
        selector_selected_extension_formula_explicit
    )
    exact_beyond_current_written_action_selected_extension_convention_selector_selected_extension_available_now = False
    updated_pack_beyond_current_written_action_selected_extension_convention_selector_selected_extension_family_primary_followup_required = bool(
        exact_minimal_selected_extension_convention_selector_selected_extension_family_requirement_theorem_available_now
    )
    updated_pack_corrected_pack_refresh_secondary_hold_retained = bool(
        updated_pack_beyond_current_written_action_selected_extension_convention_selector_selected_extension_family_primary_followup_required
    )
    updated_pack_same_tag_pack_refresh_reentry_admissible_now = False
    updated_pack_beyond_current_written_action_selected_extension_convention_selector_selected_extension_breakthrough_passed_now = False
    blind_blocked = bool(prior_gate_summary["blind_vector_observable_gate_still_blocked"])

    rows = [
        sign_base.row(
            "updated_pack_beyond_current_written_action_selected_extension_convention_selector_selected_extension_audit_selected",
            "pass" if audit_selected else "reject",
            "updated-pack beyond-current-written-action selected extension convention selector selected extension audit selected",
            sign_base.truth(audit_selected),
            "This branch is worth running only after selector selected-candidate underdetermination is already closed and same-tag reentry remains shut.",
        ),
        sign_base.row(
            "retry_gate_computation_mode_selected",
            "pass" if retry_mode else "reject",
            "retry gate computation mode selected",
            sign_base.truth(retry_mode),
            "This turn must close a new theorem object rather than restate selected-candidate underdetermination in new words.",
        ),
        sign_base.row(
            "failure_matrix_non_surrogate_guard_preserved",
            "pass" if non_surrogate_guard else "reject",
            "failure-matrix non-surrogate guard preserved",
            sign_base.truth(non_surrogate_guard),
            "The selector selected-extension theorem is admissible only if it does not reopen the exhausted surrogate family.",
        ),
        sign_base.row(
            "gate_a_updated_pack_exact_beyond_current_written_action_selected_extension_convention_selector_selected_candidate_no_go_available_now",
            "pass" if selector_selected_candidate_no_go_available else "reject",
            "gate A exact beyond-current-written-action selected extension convention selector selected candidate no-go available now",
            sign_base.truth(selector_selected_candidate_no_go_available),
            "The selected-extension theorem starts only after the current theory already closes that no concrete selected convention candidate has been chosen.",
        ),
        sign_base.row(
            "exact_beyond_current_written_action_selected_extension_convention_selector_selected_candidate_formula_available_now",
            "pass" if selector_selected_candidate_formula_available else "reject",
            "exact beyond-current-written-action selected extension convention selector selected candidate formula available now",
            sign_base.truth(selector_selected_candidate_formula_available),
            "The selected-extension theorem uses the already closed explicit map from selector representative A_conv_ext to convention candidate C_ext,*.",
        ),
        sign_base.row(
            "exact_beyond_current_written_action_selected_extension_convention_selector_selected_candidate_family_formula_available_now",
            "pass" if selector_selected_candidate_family_available else "reject",
            "exact beyond-current-written-action selected extension convention selector selected candidate family formula available now",
            sign_base.truth(selector_selected_candidate_family_available),
            "The selected-extension theorem uses the already closed family Cand_sel_conv_ext[B_sel_conv_ext;B_conv_ext;W,K].",
        ),
        sign_base.row(
            "exact_beyond_current_written_action_selected_extension_formula_available_now",
            "pass" if selected_extension_formula_available else "reject",
            "exact beyond-current-written-action selected extension formula available now",
            sign_base.truth(selected_extension_formula_available),
            "The selected-extension theorem still targets the already closed downstream object Sigma_*^(W;K,chi).",
        ),
        sign_base.row(
            "exact_minimal_selected_extension_convention_selector_selected_extension_requirement_theorem_available_now",
            "pass" if selector_selected_extension_requirement_available else "reject",
            "exact minimal selected extension convention selector selected extension requirement theorem available now",
            sign_base.truth(selector_selected_extension_requirement_available),
            "The prior branch already fixed that the honest next blocker is which selected extension follows from the unresolved selected-candidate family.",
        ),
        sign_base.row(
            "selector_selected_extension_formula_explicit",
            "pass" if selector_selected_extension_formula_explicit else "reject",
            "selected-extension convention selector selected-extension formula explicit",
            sign_base.truth(selector_selected_extension_formula_explicit),
            "Once the selector-selected-candidate family and downstream Sigma_* object are explicit, the honest next object is the induced selected extension under each admissible selector representative.",
        ),
        sign_base.row(
            "exact_beyond_current_written_action_selected_extension_convention_selector_selected_extension_formula_available_now",
            "pass"
            if exact_beyond_current_written_action_selected_extension_convention_selector_selected_extension_formula_available_now
            else "reject",
            "exact beyond-current-written-action selected extension convention selector selected extension formula available now",
            sign_base.truth(
                exact_beyond_current_written_action_selected_extension_convention_selector_selected_extension_formula_available_now
            ),
            "The theorem stack now fixes the explicit map from each admissible selector representative A_conv_ext to the induced selected extension Sigma_sel_conv_ext,*. ",
        ),
        sign_base.row(
            "exact_beyond_current_written_action_selected_extension_convention_selector_selected_extension_no_go_theorem_available_now",
            "pass"
            if exact_beyond_current_written_action_selected_extension_convention_selector_selected_extension_no_go_theorem_available_now
            else "reject",
            "exact beyond-current-written-action selected extension convention selector selected extension no-go theorem available now",
            sign_base.truth(
                exact_beyond_current_written_action_selected_extension_convention_selector_selected_extension_no_go_theorem_available_now
            ),
            "Because current theory still does not choose one concrete selector representative A_conv_ext, it still cannot choose one canonical selected extension.",
        ),
        sign_base.row(
            "exact_minimal_selected_extension_convention_selector_selected_extension_family_requirement_theorem_available_now",
            "pass"
            if exact_minimal_selected_extension_convention_selector_selected_extension_family_requirement_theorem_available_now
            else "reject",
            "exact minimal selected extension convention selector selected extension family requirement theorem available now",
            sign_base.truth(
                exact_minimal_selected_extension_convention_selector_selected_extension_family_requirement_theorem_available_now
            ),
            "The honest next blocker is therefore the full selected-extension family induced by the unresolved selector-selected-candidate family.",
        ),
        sign_base.row(
            "exact_beyond_current_written_action_selected_extension_convention_selector_selected_extension_available_now",
            "pass"
            if exact_beyond_current_written_action_selected_extension_convention_selector_selected_extension_available_now
            else "reject",
            "exact beyond-current-written-action selected extension convention selector selected extension available now",
            sign_base.truth(
                exact_beyond_current_written_action_selected_extension_convention_selector_selected_extension_available_now
            ),
            "This branch closes the selected-extension formula and no-go, not one concrete selected extension itself.",
        ),
        sign_base.row(
            "updated_pack_beyond_current_written_action_selected_extension_convention_selector_selected_extension_family_primary_followup_required",
            "pass"
            if updated_pack_beyond_current_written_action_selected_extension_convention_selector_selected_extension_family_primary_followup_required
            else "reject",
            "updated-pack beyond-current-written-action selected extension convention selector selected extension family primary followup required",
            sign_base.truth(
                updated_pack_beyond_current_written_action_selected_extension_convention_selector_selected_extension_family_primary_followup_required
            ),
            "The honest next blocker is the selected-extension family induced by unresolved selector representatives, not another same-tag route-sync.",
        ),
        sign_base.row(
            "updated_pack_corrected_pack_refresh_secondary_hold_retained",
            "pass" if updated_pack_corrected_pack_refresh_secondary_hold_retained else "reject",
            "updated-pack corrected pack-refresh secondary hold retained",
            sign_base.truth(updated_pack_corrected_pack_refresh_secondary_hold_retained),
            "Pack-refresh remains secondary because the blocker is still theorem-side selected-extension-family completion.",
        ),
        sign_base.row(
            "updated_pack_same_tag_pack_refresh_reentry_admissible_now",
            "pass" if updated_pack_same_tag_pack_refresh_reentry_admissible_now else "reject",
            "updated-pack same-tag pack-refresh reentry admissible now",
            sign_base.truth(updated_pack_same_tag_pack_refresh_reentry_admissible_now),
            "Same-tag reentry remains closed because the blocker is theorem-side selected-extension-family completion, not bookkeeping syntax.",
        ),
        sign_base.row(
            "updated_pack_beyond_current_written_action_selected_extension_convention_selector_selected_extension_breakthrough_passed_now",
            "pass"
            if updated_pack_beyond_current_written_action_selected_extension_convention_selector_selected_extension_breakthrough_passed_now
            else "reject",
            "updated-pack beyond-current-written-action selected extension convention selector selected extension breakthrough passed now",
            sign_base.truth(
                updated_pack_beyond_current_written_action_selected_extension_convention_selector_selected_extension_breakthrough_passed_now
            ),
            "This branch sharpens selected-extension underdetermination but still does not choose one concrete selected extension.",
        ),
        sign_base.row(
            "blind_vector_observable_gate_still_blocked",
            "pass" if blind_blocked else "reject",
            "blind-vector observable gate still blocked",
            sign_base.truth(blind_blocked),
            "Blind-vector direct computation still waits on one concrete selected extension.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "retained_scalar_residual_rel": float(
            prior_gate_summary["retained_scalar_residual_rel"]
        ),
        "updated_pack_beyond_current_written_action_selected_extension_convention_selector_selected_extension_audit_selected": audit_selected,
        "retry_gate_computation_mode_selected": retry_mode,
        "failure_matrix_non_surrogate_guard_preserved": non_surrogate_guard,
        "gate_a_updated_pack_exact_beyond_current_written_action_selected_extension_convention_selector_selected_candidate_no_go_available_now": selector_selected_candidate_no_go_available,
        "exact_beyond_current_written_action_selected_extension_convention_selector_selected_candidate_formula_available_now": selector_selected_candidate_formula_available,
        "exact_beyond_current_written_action_selected_extension_convention_selector_selected_candidate_family_formula_available_now": selector_selected_candidate_family_available,
        "exact_beyond_current_written_action_selected_extension_formula_available_now": selected_extension_formula_available,
        "exact_minimal_selected_extension_convention_selector_selected_extension_requirement_theorem_available_now": selector_selected_extension_requirement_available,
        "selector_selected_extension_formula_explicit": selector_selected_extension_formula_explicit,
        "exact_beyond_current_written_action_selected_extension_convention_selector_selected_extension_formula_available_now": exact_beyond_current_written_action_selected_extension_convention_selector_selected_extension_formula_available_now,
        "exact_beyond_current_written_action_selected_extension_convention_selector_selected_extension_no_go_theorem_available_now": exact_beyond_current_written_action_selected_extension_convention_selector_selected_extension_no_go_theorem_available_now,
        "exact_minimal_selected_extension_convention_selector_selected_extension_family_requirement_theorem_available_now": exact_minimal_selected_extension_convention_selector_selected_extension_family_requirement_theorem_available_now,
        "exact_beyond_current_written_action_selected_extension_convention_selector_selected_extension_available_now": exact_beyond_current_written_action_selected_extension_convention_selector_selected_extension_available_now,
        "updated_pack_beyond_current_written_action_selected_extension_convention_selector_selected_extension_family_primary_followup_required": updated_pack_beyond_current_written_action_selected_extension_convention_selector_selected_extension_family_primary_followup_required,
        "updated_pack_corrected_pack_refresh_secondary_hold_retained": updated_pack_corrected_pack_refresh_secondary_hold_retained,
        "updated_pack_same_tag_pack_refresh_reentry_admissible_now": updated_pack_same_tag_pack_refresh_reentry_admissible_now,
        "blind_vector_observable_gate_still_blocked": blind_blocked,
        "updated_pack_beyond_current_written_action_selected_extension_convention_selector_selected_extension_breakthrough_passed_now": updated_pack_beyond_current_written_action_selected_extension_convention_selector_selected_extension_breakthrough_passed_now,
        "selected_primary_completion_lane": "updated_pack_beyond_current_written_action_selected_extension_convention_selector_selected_extension_family_theorem_audit",
        "selected_secondary_completion_lane": "updated_pack_corrected_pack_refresh_return_sync",
        "selected_reserve_completion_lane": "farther_hybrid_extra_q_range_only",
        "selected_next_generation_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_beyond_current_written_action_selected_extension_convention_selector_selected_extension_family_theorem_audit",
        "recommended_next_route_or_none": "8.7.56.4807",
        "selected_followup_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_beyond_current_written_action_selected_extension_convention_selector_selected_extension_gate",
        "selected_followup_route_or_none": "8.7.56.4803",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.4801",
        STEP_NAME + " declaration gate",
        {
            "source_files": {
                "prior_gate": sign_base.display_path(PRIOR_GATE),
                "prior_selected_candidate_audit": sign_base.display_path(
                    PRIOR_SELECTED_CANDIDATE_AUDIT
                ),
                "prior_selected_extension_audit": sign_base.display_path(
                    PRIOR_SELECTED_EXTENSION_AUDIT
                ),
            },
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "8.7.56.4807",
                "followup_route": "8.7.56.4803",
            },
        },
        rows,
        summary,
        {
            "overall_status": "vector_qball_form_factor_updated_pack_beyond_current_written_action_selected_extension_convention_selector_selected_extension_declared",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {"formulae": build_formulae()},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(
        json.dumps(
            {
                "json": declaration_paths["json"],
                "classification": BRANCH_CLASS,
                "breakthrough_passed_now": False,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
