#!/usr/bin/env python3
"""Generate 8.7.56.4903-.4906 selected-candidate downstream value-audit artifacts."""

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
        "8.7.56.4899-4902",
        "updated_pack_beyond_current_written_action_selected_extension_convention_selector_selector_chart_representative_gate",
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
        "8.7.56.4799-4802",
        "updated_pack_beyond_current_written_action_selected_extension_convention_selector_selected_extension_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
PRIOR_SELECTED_EXTENSION_FAMILY_AUDIT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.4807-4810",
        "updated_pack_beyond_current_written_action_selected_extension_convention_selector_selected_extension_family_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.4903-4906"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack beyond-current-"
    "written-action selected extension convention selector selected candidate "
    "value audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_beyond_current_written_action_selected_extension_convention_selector_selected_candidate_value_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "beyond_current_written_action_selected_extension_convention_selector_"
    "selector_chart_representative_finite_anchor_no_go_theorem_derived_selected_"
    "extension_convention_selector_selected_candidate_value_audit_primary_"
    "hybrid_reserve_secondary_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "beyond_current_written_action_selected_extension_convention_selector_"
    "downstream_rerun_no_new_object_theorem_derived_selector_chart_representative_"
    "concrete_rule_primary_pack_refresh_secondary_gate"
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


# 関数: value audit で使う式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used in the selected-candidate downstream value audit."""
    return {
        "prior_selected_candidate_formula": (
            "C_ext,*^(B_conv_ext;W,K,A_conv_ext) := "
            "C_ext^(W;K,chi_*^(B_conv_ext;W,K,A_conv_ext))"
        ),
        "prior_selected_extension_formula": (
            "Sigma_sel_conv_ext,*^(B_sel_conv_ext;B_conv_ext;W,K,A_conv_ext) := "
            "Sigma_*^(W;K,chi_*^(B_conv_ext;W,K,A_conv_ext))"
        ),
        "prior_selected_extension_family": (
            "Ext_sel_conv_ext[B_sel_conv_ext;B_conv_ext;W,K] := { "
            "Sigma_sel_conv_ext,*^(B_sel_conv_ext;B_conv_ext;W,K,A_conv_ext) | "
            "A_conv_ext in Rep_sel_conv_ext[B_sel_conv_ext;W,K] }"
        ),
        "value_no_go": (
            "if chart-representative finite-anchor no-go still leaves "
            "A_conv_ext unresolved, replaying selected-candidate / selected-extension "
            "/ selected-extension-family theorem branches adds no new exact object"
        ),
    }


# 関数: `.4903-.4906` を実行する。

def main() -> None:
    """Execute the selected-candidate downstream value audit."""
    for path in (
        PRIOR_GATE,
        PRIOR_SELECTED_CANDIDATE_AUDIT,
        PRIOR_SELECTED_EXTENSION_AUDIT,
        PRIOR_SELECTED_EXTENSION_FAMILY_AUDIT,
    ):
        sign_base.require(path)

    prior_gate_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    prior_selected_candidate_summary = sign_base.read_json(
        PRIOR_SELECTED_CANDIDATE_AUDIT
    )["summary"]
    prior_selected_extension_summary = sign_base.read_json(
        PRIOR_SELECTED_EXTENSION_AUDIT
    )["summary"]
    prior_selected_extension_family_summary = sign_base.read_json(
        PRIOR_SELECTED_EXTENSION_FAMILY_AUDIT
    )["summary"]

    audit_selected = bool(
        prior_gate_summary[
            "gate_b_updated_pack_beyond_current_written_action_selected_extension_convention_selector_selected_candidate_value_audit_promoted_next"
        ]
        and prior_gate_summary["pack_update_required_now"]
    )
    retry_mode = bool(prior_gate_summary["retry_gate_computation_mode_selected"])
    non_surrogate_guard = bool(
        prior_gate_summary["failure_matrix_non_surrogate_guard_preserved"]
    )
    chart_representative_no_go_available = bool(
        prior_gate_summary[
            "gate_a_updated_pack_exact_beyond_current_written_action_selected_extension_convention_selector_selector_chart_representative_finite_anchor_no_go_available_now"
        ]
    )
    prior_selected_candidate_formula_available = bool(
        prior_selected_candidate_summary[
            "exact_beyond_current_written_action_selected_extension_convention_selector_selected_candidate_formula_available_now"
        ]
    )
    prior_selected_extension_formula_available = bool(
        prior_selected_extension_summary[
            "exact_beyond_current_written_action_selected_extension_convention_selector_selected_extension_formula_available_now"
        ]
    )
    prior_selected_extension_family_formula_available = bool(
        prior_selected_extension_family_summary[
            "exact_beyond_current_written_action_selected_extension_convention_selector_selected_extension_family_formula_available_now"
        ]
    )
    selected_candidate_rerun_adds_new_exact_object_now = False
    selected_extension_rerun_adds_new_exact_object_now = False
    selected_extension_family_rerun_adds_new_exact_object_now = False
    exact_beyond_current_written_action_selected_extension_convention_selector_downstream_rerun_no_new_object_theorem_available_now = bool(
        audit_selected
        and retry_mode
        and non_surrogate_guard
        and chart_representative_no_go_available
        and prior_selected_candidate_formula_available
        and prior_selected_extension_formula_available
        and prior_selected_extension_family_formula_available
    )
    exact_beyond_current_written_action_selector_chart_representative_concrete_rule_requirement_theorem_available_now = bool(
        exact_beyond_current_written_action_selected_extension_convention_selector_downstream_rerun_no_new_object_theorem_available_now
    )
    updated_pack_beyond_current_written_action_selector_chart_representative_concrete_rule_primary_followup_required = bool(
        exact_beyond_current_written_action_selector_chart_representative_concrete_rule_requirement_theorem_available_now
    )
    updated_pack_same_tag_selected_extension_convention_selector_downstream_rerun_admissible_now = False
    updated_pack_corrected_pack_refresh_secondary_hold_retained = bool(
        updated_pack_beyond_current_written_action_selector_chart_representative_concrete_rule_primary_followup_required
    )
    blind_blocked = bool(prior_gate_summary["blind_vector_observable_gate_still_blocked"])
    value_audit_confirms_repetition_if_replayed_now = bool(
        exact_beyond_current_written_action_selected_extension_convention_selector_downstream_rerun_no_new_object_theorem_available_now
    )

    rows = [
        sign_base.row(
            "updated_pack_beyond_current_written_action_selected_extension_convention_selector_selected_candidate_value_audit_selected",
            "pass" if audit_selected else "reject",
            "updated-pack beyond-current-written-action selected-candidate value audit selected",
            sign_base.truth(audit_selected),
            "This branch is worth running only after chart-representative finite-anchor no-go is closed and the user requested explicit repetition control.",
        ),
        sign_base.row(
            "retry_gate_computation_mode_selected",
            "pass" if retry_mode else "reject",
            "retry gate computation mode selected",
            sign_base.truth(retry_mode),
            "This turn must decide whether rerunning older downstream branches would add a new exact object or only restate already closed formulas.",
        ),
        sign_base.row(
            "failure_matrix_non_surrogate_guard_preserved",
            "pass" if non_surrogate_guard else "reject",
            "failure-matrix non-surrogate guard preserved",
            sign_base.truth(non_surrogate_guard),
            "The value audit is admissible only if it does not reopen the exhausted surrogate family.",
        ),
        sign_base.row(
            "gate_a_updated_pack_exact_beyond_current_written_action_selected_extension_convention_selector_selector_chart_representative_finite_anchor_no_go_available_now",
            "pass" if chart_representative_no_go_available else "reject",
            "gate A exact beyond-current-written-action selector-selector chart representative finite-anchor no-go available now",
            sign_base.truth(chart_representative_no_go_available),
            "The downstream replay can be judged only after finite chart anchors already fail to choose one concrete selector chart representative.",
        ),
        sign_base.row(
            "prior_selected_candidate_formula_available",
            "pass" if prior_selected_candidate_formula_available else "reject",
            "prior selected-candidate formula available",
            sign_base.truth(prior_selected_candidate_formula_available),
            "The earlier theorem stack already fixed the selected-candidate formula under unresolved selector representatives.",
        ),
        sign_base.row(
            "prior_selected_extension_formula_available",
            "pass" if prior_selected_extension_formula_available else "reject",
            "prior selected-extension formula available",
            sign_base.truth(prior_selected_extension_formula_available),
            "The earlier theorem stack already fixed the induced selected-extension formula under unresolved selector representatives.",
        ),
        sign_base.row(
            "prior_selected_extension_family_formula_available",
            "pass" if prior_selected_extension_family_formula_available else "reject",
            "prior selected-extension family formula available",
            sign_base.truth(prior_selected_extension_family_formula_available),
            "The earlier theorem stack already fixed the selected-extension family under unresolved selector representatives.",
        ),
        sign_base.row(
            "selected_candidate_rerun_adds_new_exact_object_now",
            "pass" if selected_candidate_rerun_adds_new_exact_object_now else "reject",
            "selected-candidate rerun adds new exact object now",
            sign_base.truth(selected_candidate_rerun_adds_new_exact_object_now),
            "Replaying the already closed selected-candidate theorem without a new concrete chart representative would only rename the same unresolved dependence on A_conv_ext.",
        ),
        sign_base.row(
            "selected_extension_rerun_adds_new_exact_object_now",
            "pass" if selected_extension_rerun_adds_new_exact_object_now else "reject",
            "selected-extension rerun adds new exact object now",
            sign_base.truth(selected_extension_rerun_adds_new_exact_object_now),
            "Replaying the already closed selected-extension theorem without a new concrete chart representative would only restate the same unresolved dependence on A_conv_ext.",
        ),
        sign_base.row(
            "selected_extension_family_rerun_adds_new_exact_object_now",
            "pass" if selected_extension_family_rerun_adds_new_exact_object_now else "reject",
            "selected-extension-family rerun adds new exact object now",
            sign_base.truth(selected_extension_family_rerun_adds_new_exact_object_now),
            "Replaying the already closed selected-extension-family theorem without a new concrete chart representative would only restate the same unresolved family parameterization by A_conv_ext.",
        ),
        sign_base.row(
            "exact_beyond_current_written_action_selected_extension_convention_selector_downstream_rerun_no_new_object_theorem_available_now",
            "pass"
            if exact_beyond_current_written_action_selected_extension_convention_selector_downstream_rerun_no_new_object_theorem_available_now
            else "reject",
            "exact beyond-current-written-action selected-extension-convention selector downstream rerun no-new-object theorem available now",
            sign_base.truth(
                exact_beyond_current_written_action_selected_extension_convention_selector_downstream_rerun_no_new_object_theorem_available_now
            ),
            "The current route now fixes that replaying the earlier selected-candidate / selected-extension / selected-extension-family branches would add no new exact object until one concrete chart representative is supplied.",
        ),
        sign_base.row(
            "exact_beyond_current_written_action_selector_chart_representative_concrete_rule_requirement_theorem_available_now",
            "pass"
            if exact_beyond_current_written_action_selector_chart_representative_concrete_rule_requirement_theorem_available_now
            else "reject",
            "exact beyond-current-written-action selector chart representative concrete-rule requirement theorem available now",
            sign_base.truth(
                exact_beyond_current_written_action_selector_chart_representative_concrete_rule_requirement_theorem_available_now
            ),
            "The honest next blocker is therefore a concrete chart-representative rule, not a replay of already closed downstream theorem syntax.",
        ),
        sign_base.row(
            "value_audit_confirms_repetition_if_replayed_now",
            "pass" if value_audit_confirms_repetition_if_replayed_now else "reject",
            "value audit confirms repetition if replayed now",
            sign_base.truth(value_audit_confirms_repetition_if_replayed_now),
            "Continuing the old downstream chain immediately would be low-value repetition under the current chart-representative blocker.",
        ),
        sign_base.row(
            "updated_pack_beyond_current_written_action_selector_chart_representative_concrete_rule_primary_followup_required",
            "pass"
            if updated_pack_beyond_current_written_action_selector_chart_representative_concrete_rule_primary_followup_required
            else "reject",
            "updated-pack selector chart representative concrete-rule primary followup required",
            sign_base.truth(
                updated_pack_beyond_current_written_action_selector_chart_representative_concrete_rule_primary_followup_required
            ),
            "The honest next blocker is to state which concrete rule could canonically choose one chart representative, not to replay already closed selected-candidate syntax.",
        ),
        sign_base.row(
            "updated_pack_same_tag_selected_extension_convention_selector_downstream_rerun_admissible_now",
            "pass"
            if updated_pack_same_tag_selected_extension_convention_selector_downstream_rerun_admissible_now
            else "reject",
            "updated-pack same-tag selected-extension-convention-selector downstream rerun admissible now",
            sign_base.truth(
                updated_pack_same_tag_selected_extension_convention_selector_downstream_rerun_admissible_now
            ),
            "Same-tag downstream rerun remains closed because it would be repetition under the current blocker.",
        ),
        sign_base.row(
            "updated_pack_corrected_pack_refresh_secondary_hold_retained",
            "pass" if updated_pack_corrected_pack_refresh_secondary_hold_retained else "reject",
            "updated-pack corrected pack-refresh secondary hold retained",
            sign_base.truth(updated_pack_corrected_pack_refresh_secondary_hold_retained),
            "Pack-refresh stays secondary because the blocker is theorem-side chart-representative rule completion.",
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
        "retained_scalar_residual_rel": float(prior_gate_summary["retained_scalar_residual_rel"]),
        "updated_pack_beyond_current_written_action_selected_extension_convention_selector_selected_candidate_value_audit_selected": audit_selected,
        "retry_gate_computation_mode_selected": retry_mode,
        "failure_matrix_non_surrogate_guard_preserved": non_surrogate_guard,
        "gate_a_updated_pack_exact_beyond_current_written_action_selected_extension_convention_selector_selector_chart_representative_finite_anchor_no_go_available_now": chart_representative_no_go_available,
        "prior_selected_candidate_formula_available": prior_selected_candidate_formula_available,
        "prior_selected_extension_formula_available": prior_selected_extension_formula_available,
        "prior_selected_extension_family_formula_available": prior_selected_extension_family_formula_available,
        "selected_candidate_rerun_adds_new_exact_object_now": selected_candidate_rerun_adds_new_exact_object_now,
        "selected_extension_rerun_adds_new_exact_object_now": selected_extension_rerun_adds_new_exact_object_now,
        "selected_extension_family_rerun_adds_new_exact_object_now": selected_extension_family_rerun_adds_new_exact_object_now,
        "exact_beyond_current_written_action_selected_extension_convention_selector_downstream_rerun_no_new_object_theorem_available_now": exact_beyond_current_written_action_selected_extension_convention_selector_downstream_rerun_no_new_object_theorem_available_now,
        "exact_beyond_current_written_action_selector_chart_representative_concrete_rule_requirement_theorem_available_now": exact_beyond_current_written_action_selector_chart_representative_concrete_rule_requirement_theorem_available_now,
        "value_audit_confirms_repetition_if_replayed_now": value_audit_confirms_repetition_if_replayed_now,
        "updated_pack_beyond_current_written_action_selector_chart_representative_concrete_rule_primary_followup_required": updated_pack_beyond_current_written_action_selector_chart_representative_concrete_rule_primary_followup_required,
        "updated_pack_same_tag_selected_extension_convention_selector_downstream_rerun_admissible_now": updated_pack_same_tag_selected_extension_convention_selector_downstream_rerun_admissible_now,
        "updated_pack_corrected_pack_refresh_secondary_hold_retained": updated_pack_corrected_pack_refresh_secondary_hold_retained,
        "blind_vector_observable_gate_still_blocked": blind_blocked,
        "selected_primary_completion_lane": "updated_pack_beyond_current_written_action_selector_chart_representative_concrete_rule_theorem_audit",
        "selected_secondary_completion_lane": "updated_pack_corrected_pack_refresh_return_sync",
        "selected_reserve_completion_lane": "farther_hybrid_extra_q_range_only",
        "selected_next_generation_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_beyond_current_written_action_selector_chart_representative_concrete_rule_theorem_audit",
        "recommended_next_route_or_none": "8.7.56.4911",
        "selected_followup_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_beyond_current_written_action_selected_extension_convention_selector_selected_candidate_value_gate",
        "selected_followup_route_or_none": "8.7.56.4907",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.4905",
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
                "prior_selected_extension_family_audit": sign_base.display_path(
                    PRIOR_SELECTED_EXTENSION_FAMILY_AUDIT
                ),
            },
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "8.7.56.4911",
                "followup_route": "8.7.56.4907",
            },
        },
        rows,
        summary,
        {
            "overall_status": "vector_qball_form_factor_updated_pack_beyond_current_written_action_selected_extension_convention_selector_selected_candidate_value_audit_declared",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {"formulae": build_formulae()},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(
        f"[done] {STEP_TAG} updated-pack beyond-current-written-action selected-candidate value audit completed"
    )
    print(f"[done] declaration: {declaration_paths['json']}")


if __name__ == "__main__":
    main()
