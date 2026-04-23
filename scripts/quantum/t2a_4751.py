#!/usr/bin/env python3
"""Generate 8.7.56.4751-.4754 selected-extension-convention-candidate theorem artifacts."""

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
        "8.7.56.4747-4750",
        "updated_pack_beyond_current_written_action_selected_extension_convention_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
PRIOR_PULLBACK_AUDIT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.4743-4746",
        "updated_pack_beyond_current_written_action_selected_extension_selector_pullback_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
PRIOR_CHART_CONVENTION_AUDIT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.4695-4698",
        "updated_pack_beyond_current_written_action_selector_measure_chart_convention_audit",
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

STEP_TAG = "8.7.56.4751-4754"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack beyond-current-"
    "written-action selected extension convention candidate theorem audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_beyond_current_written_action_selected_extension_convention_candidate_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "beyond_current_written_action_selected_extension_convention_requirement_"
    "theorem_derived_selected_extension_convention_candidate_primary_hybrid_"
    "reserve_secondary_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "beyond_current_written_action_selected_extension_convention_candidate_"
    "reduction_theorem_derived_selected_extension_convention_representative_"
    "primary_pack_refresh_secondary_gate"
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


# 関数: selected-extension-convention-candidate theorem で使う式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used in the selected-extension-convention-candidate audit."""
    return {
        "chart_convention_family": (
            "Conv_meas_chart_family[W,K] := { chi | chi : Im(K^(W)) -> J_chi "
            "strictly monotone bijection }"
        ),
        "selected_extension_family": (
            "Ext_meas[W,K] := { Sigma_*^(W;K,chi) | "
            "chi in Conv_meas_chart_family[W,K] }"
        ),
        "selected_extension_convention_candidate_family": (
            "Cand_conv_ext[W,K] := { C_ext^(W;K,chi) | "
            "chi in Conv_meas_chart_family[W,K] }"
        ),
        "selected_extension_convention_candidate": (
            "C_ext^(W;K,chi) : Ext_meas[W,K] -> Ext_meas[W,K], "
            "C_ext^(W;K,chi)(Ext_meas[W,K]) := Sigma_*^(W;K,chi)"
        ),
        "candidate_reduction": (
            "choosing one convention candidate C_ext^(W;K,chi) is theorem-side "
            "equivalent to choosing one concrete chart-convention representative chi"
        ),
    }


# 関数: `.4751-.4754` を実行する。

def main() -> None:
    """Execute the selected-extension-convention-candidate theorem audit."""
    for path in (
        PRIOR_GATE,
        PRIOR_PULLBACK_AUDIT,
        PRIOR_CHART_CONVENTION_AUDIT,
        PRIOR_SELECTED_EXTENSION_AUDIT,
    ):
        sign_base.require(path)

    prior_gate_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    prior_pullback_summary = sign_base.read_json(PRIOR_PULLBACK_AUDIT)["summary"]
    prior_chart_convention_summary = sign_base.read_json(PRIOR_CHART_CONVENTION_AUDIT)[
        "summary"
    ]
    prior_selected_extension_summary = sign_base.read_json(PRIOR_SELECTED_EXTENSION_AUDIT)[
        "summary"
    ]

    audit_selected = bool(
        prior_gate_summary[
            "gate_b_updated_pack_beyond_current_written_action_selected_extension_convention_candidate_promoted_next"
        ]
        and prior_gate_summary["pack_update_required_now"]
    )
    retry_mode = bool(prior_gate_summary["retry_gate_computation_mode_selected"])
    non_surrogate_guard = bool(
        prior_gate_summary["failure_matrix_non_surrogate_guard_preserved"]
    )
    selector_no_new_information_available = bool(
        prior_gate_summary[
            "gate_a_updated_pack_exact_beyond_current_written_action_selected_extension_selector_no_new_information_theorem_available_now"
        ]
    )
    selected_extension_family_available = bool(
        prior_selected_extension_summary[
            "exact_beyond_current_written_action_selected_extension_formula_available_now"
        ]
    )
    chart_convention_family_available = bool(
        prior_chart_convention_summary[
            "exact_beyond_current_written_action_selector_measure_chart_convention_family_formula_available_now"
        ]
    )
    convention_requirement_available = bool(
        prior_pullback_summary[
            "exact_minimal_selected_extension_convention_requirement_theorem_available_now"
        ]
    )
    candidate_formula_explicit = bool(
        audit_selected
        and retry_mode
        and non_surrogate_guard
        and selector_no_new_information_available
        and selected_extension_family_available
        and chart_convention_family_available
        and convention_requirement_available
    )
    exact_beyond_current_written_action_selected_extension_convention_candidate_family_formula_available_now = bool(
        candidate_formula_explicit
    )
    exact_beyond_current_written_action_selected_extension_convention_candidate_chart_convention_reduction_theorem_available_now = bool(
        candidate_formula_explicit
    )
    exact_beyond_current_written_action_selected_extension_convention_candidate_no_go_theorem_available_now = bool(
        candidate_formula_explicit
    )
    exact_minimal_selected_extension_convention_representative_requirement_theorem_available_now = bool(
        candidate_formula_explicit
    )
    exact_beyond_current_written_action_selected_extension_convention_candidate_available_now = False
    updated_pack_beyond_current_written_action_selected_extension_convention_representative_primary_followup_required = bool(
        exact_minimal_selected_extension_convention_representative_requirement_theorem_available_now
    )
    updated_pack_corrected_pack_refresh_secondary_hold_retained = bool(
        updated_pack_beyond_current_written_action_selected_extension_convention_representative_primary_followup_required
    )
    updated_pack_same_tag_pack_refresh_reentry_admissible_now = False
    updated_pack_beyond_current_written_action_selected_extension_convention_candidate_breakthrough_passed_now = False
    blind_blocked = bool(prior_gate_summary["blind_vector_observable_gate_still_blocked"])

    rows = [
        sign_base.row(
            "updated_pack_beyond_current_written_action_selected_extension_convention_candidate_audit_selected",
            "pass" if audit_selected else "reject",
            "updated-pack beyond-current-written-action selected extension convention candidate audit selected",
            sign_base.truth(audit_selected),
            "This branch is worth running only after the selector no-new-information theorem already shuts the internal selector ladder and same-tag reentry remains closed.",
        ),
        sign_base.row(
            "retry_gate_computation_mode_selected",
            "pass" if retry_mode else "reject",
            "retry gate computation mode selected",
            sign_base.truth(retry_mode),
            "This turn must close a new theorem object rather than rename the same selector obstruction.",
        ),
        sign_base.row(
            "failure_matrix_non_surrogate_guard_preserved",
            "pass" if non_surrogate_guard else "reject",
            "failure-matrix non-surrogate guard preserved",
            sign_base.truth(non_surrogate_guard),
            "The convention-candidate theorem is admissible only if it does not reopen the exhausted surrogate family.",
        ),
        sign_base.row(
            "gate_a_updated_pack_exact_beyond_current_written_action_selected_extension_selector_no_new_information_theorem_available_now",
            "pass" if selector_no_new_information_available else "reject",
            "gate A exact beyond-current-written-action selected extension selector no-new-information theorem available now",
            sign_base.truth(selector_no_new_information_available),
            "The candidate theorem starts only after the internal selector lane is closed as informationally redundant.",
        ),
        sign_base.row(
            "exact_beyond_current_written_action_selected_extension_formula_available_now",
            "pass" if selected_extension_family_available else "reject",
            "exact beyond-current-written-action selected extension formula available now",
            sign_base.truth(selected_extension_family_available),
            "The candidate theorem uses the already closed selected-extension objects Sigma_*^(W;K,chi).",
        ),
        sign_base.row(
            "exact_beyond_current_written_action_selector_measure_chart_convention_family_formula_available_now",
            "pass" if chart_convention_family_available else "reject",
            "exact beyond-current-written-action selector measure chart convention family formula available now",
            sign_base.truth(chart_convention_family_available),
            "The candidate theorem uses the already closed family of chart-convention representatives chi.",
        ),
        sign_base.row(
            "exact_minimal_selected_extension_convention_requirement_theorem_available_now",
            "pass" if convention_requirement_available else "reject",
            "exact minimal selected extension convention requirement theorem available now",
            sign_base.truth(convention_requirement_available),
            "The prior branch already fixed that an external convention is required once the selector lane is shown to add no new information.",
        ),
        sign_base.row(
            "candidate_formula_explicit",
            "pass" if candidate_formula_explicit else "reject",
            "selected-extension convention candidate formula explicit",
            sign_base.truth(candidate_formula_explicit),
            "Once chi and Sigma_*^(W;K,chi) are explicit, the honest next object is the family of convention candidates that map the admissible family Ext_meas[W,K] to one selected extension.",
        ),
        sign_base.row(
            "exact_beyond_current_written_action_selected_extension_convention_candidate_family_formula_available_now",
            "pass"
            if exact_beyond_current_written_action_selected_extension_convention_candidate_family_formula_available_now
            else "reject",
            "exact beyond-current-written-action selected extension convention candidate family formula available now",
            sign_base.truth(
                exact_beyond_current_written_action_selected_extension_convention_candidate_family_formula_available_now
            ),
            "The theorem stack now fixes the literal family Cand_conv_ext[W,K] of convention candidates C_ext^(W;K,chi).",
        ),
        sign_base.row(
            "exact_beyond_current_written_action_selected_extension_convention_candidate_chart_convention_reduction_theorem_available_now",
            "pass"
            if exact_beyond_current_written_action_selected_extension_convention_candidate_chart_convention_reduction_theorem_available_now
            else "reject",
            "exact beyond-current-written-action selected extension convention candidate chart-convention reduction theorem available now",
            sign_base.truth(
                exact_beyond_current_written_action_selected_extension_convention_candidate_chart_convention_reduction_theorem_available_now
            ),
            "Choosing one convention candidate C_ext^(W;K,chi) is now fixed to be theorem-side equivalent to choosing one concrete chart-convention representative chi.",
        ),
        sign_base.row(
            "exact_beyond_current_written_action_selected_extension_convention_candidate_no_go_theorem_available_now",
            "pass"
            if exact_beyond_current_written_action_selected_extension_convention_candidate_no_go_theorem_available_now
            else "reject",
            "exact beyond-current-written-action selected extension convention candidate no-go theorem available now",
            sign_base.truth(
                exact_beyond_current_written_action_selected_extension_convention_candidate_no_go_theorem_available_now
            ),
            "Because current theory still does not choose one concrete chart-convention representative chi, it still cannot choose one canonical selected-extension convention candidate.",
        ),
        sign_base.row(
            "exact_minimal_selected_extension_convention_representative_requirement_theorem_available_now",
            "pass"
            if exact_minimal_selected_extension_convention_representative_requirement_theorem_available_now
            else "reject",
            "exact minimal selected extension convention representative requirement theorem available now",
            sign_base.truth(
                exact_minimal_selected_extension_convention_representative_requirement_theorem_available_now
            ),
            "The honest next blocker is therefore not another candidate-family theorem but what representative convention on chi could be concretely fixed.",
        ),
        sign_base.row(
            "exact_beyond_current_written_action_selected_extension_convention_candidate_available_now",
            "pass"
            if exact_beyond_current_written_action_selected_extension_convention_candidate_available_now
            else "reject",
            "exact beyond-current-written-action selected extension convention candidate available now",
            sign_base.truth(
                exact_beyond_current_written_action_selected_extension_convention_candidate_available_now
            ),
            "This branch closes the candidate-family and reduction theorem, not one concrete convention candidate itself.",
        ),
        sign_base.row(
            "updated_pack_beyond_current_written_action_selected_extension_convention_representative_primary_followup_required",
            "pass"
            if updated_pack_beyond_current_written_action_selected_extension_convention_representative_primary_followup_required
            else "reject",
            "updated-pack beyond-current-written-action selected extension convention representative primary followup required",
            sign_base.truth(
                updated_pack_beyond_current_written_action_selected_extension_convention_representative_primary_followup_required
            ),
            "The blocker is now a concrete representative convention on chart representatives, not another abstract selector layer.",
        ),
        sign_base.row(
            "updated_pack_corrected_pack_refresh_secondary_hold_retained",
            "pass" if updated_pack_corrected_pack_refresh_secondary_hold_retained else "reject",
            "updated-pack corrected pack-refresh secondary hold retained",
            sign_base.truth(updated_pack_corrected_pack_refresh_secondary_hold_retained),
            "Pack-refresh stays secondary because the blocker is theorem-side representative selection, not bookkeeping.",
        ),
        sign_base.row(
            "updated_pack_same_tag_pack_refresh_reentry_admissible_now",
            "pass" if updated_pack_same_tag_pack_refresh_reentry_admissible_now else "reject",
            "updated-pack same-tag pack-refresh reentry admissible now",
            sign_base.truth(updated_pack_same_tag_pack_refresh_reentry_admissible_now),
            "Same-tag reentry remains closed because this branch is meant to compress the blocker, not to reopen low-value loop maintenance.",
        ),
        sign_base.row(
            "updated_pack_beyond_current_written_action_selected_extension_convention_candidate_breakthrough_passed_now",
            "pass" if updated_pack_beyond_current_written_action_selected_extension_convention_candidate_breakthrough_passed_now else "reject",
            "updated-pack beyond-current-written-action selected extension convention candidate breakthrough passed now",
            sign_base.truth(updated_pack_beyond_current_written_action_selected_extension_convention_candidate_breakthrough_passed_now),
            "This branch sharpens the blocker to representative choice but still does not choose one concrete selected extension.",
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
        "updated_pack_beyond_current_written_action_selected_extension_convention_candidate_audit_selected": audit_selected,
        "retry_gate_computation_mode_selected": retry_mode,
        "failure_matrix_non_surrogate_guard_preserved": non_surrogate_guard,
        "gate_a_updated_pack_exact_beyond_current_written_action_selected_extension_selector_no_new_information_theorem_available_now": selector_no_new_information_available,
        "exact_beyond_current_written_action_selected_extension_formula_available_now": selected_extension_family_available,
        "exact_beyond_current_written_action_selector_measure_chart_convention_family_formula_available_now": chart_convention_family_available,
        "exact_minimal_selected_extension_convention_requirement_theorem_available_now": convention_requirement_available,
        "candidate_formula_explicit": candidate_formula_explicit,
        "exact_beyond_current_written_action_selected_extension_convention_candidate_family_formula_available_now": exact_beyond_current_written_action_selected_extension_convention_candidate_family_formula_available_now,
        "exact_beyond_current_written_action_selected_extension_convention_candidate_chart_convention_reduction_theorem_available_now": exact_beyond_current_written_action_selected_extension_convention_candidate_chart_convention_reduction_theorem_available_now,
        "exact_beyond_current_written_action_selected_extension_convention_candidate_no_go_theorem_available_now": exact_beyond_current_written_action_selected_extension_convention_candidate_no_go_theorem_available_now,
        "exact_minimal_selected_extension_convention_representative_requirement_theorem_available_now": exact_minimal_selected_extension_convention_representative_requirement_theorem_available_now,
        "exact_beyond_current_written_action_selected_extension_convention_candidate_available_now": exact_beyond_current_written_action_selected_extension_convention_candidate_available_now,
        "updated_pack_beyond_current_written_action_selected_extension_convention_representative_primary_followup_required": updated_pack_beyond_current_written_action_selected_extension_convention_representative_primary_followup_required,
        "updated_pack_corrected_pack_refresh_secondary_hold_retained": updated_pack_corrected_pack_refresh_secondary_hold_retained,
        "updated_pack_same_tag_pack_refresh_reentry_admissible_now": updated_pack_same_tag_pack_refresh_reentry_admissible_now,
        "updated_pack_beyond_current_written_action_selected_extension_convention_candidate_breakthrough_passed_now": updated_pack_beyond_current_written_action_selected_extension_convention_candidate_breakthrough_passed_now,
        "blind_vector_observable_gate_still_blocked": blind_blocked,
        "selected_primary_completion_lane": "updated_pack_beyond_current_written_action_selected_extension_convention_representative_theorem_audit",
        "selected_secondary_completion_lane": "updated_pack_corrected_pack_refresh_return_sync",
        "selected_reserve_completion_lane": "farther_hybrid_extra_q_range_only",
        "selected_next_generation_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_beyond_current_written_action_selected_extension_convention_representative_theorem_audit",
        "recommended_next_route_or_none": "8.7.56.4755",
        "selected_followup_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_beyond_current_written_action_selected_extension_convention_representative_gate",
        "selected_followup_route_or_none": "8.7.56.4755",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.4753",
        STEP_NAME + " declaration gate",
        {
            "source_files": {
                "prior_gate": sign_base.display_path(PRIOR_GATE),
                "prior_pullback_audit": sign_base.display_path(PRIOR_PULLBACK_AUDIT),
                "prior_chart_convention_audit": sign_base.display_path(PRIOR_CHART_CONVENTION_AUDIT),
                "prior_selected_extension_audit": sign_base.display_path(PRIOR_SELECTED_EXTENSION_AUDIT),
            },
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "8.7.56.4755",
                "followup_route": "8.7.56.4755",
            },
        },
        rows,
        summary,
        {
            "overall_status": "vector_qball_form_factor_updated_pack_beyond_current_written_action_selected_extension_convention_candidate_declared",
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
