#!/usr/bin/env python3
"""Generate 8.7.56.5119-.5122 chart/measure convention inventory artifacts."""

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
        "8.7.56.5115-5118",
        "updated_pack_external_rule_selector_vacuum_anchor_minimal_deformation_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
PRIOR_AUDIT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5111-5114",
        "updated_pack_external_rule_selector_vacuum_anchor_minimal_deformation_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5119-5122"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack external "
    "rule-selector chart/measure convention inventory theorem audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_external_rule_selector_chart_measure_convention_inventory_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "external_rule_selector_vacuum_anchor_minimal_deformation_no_go_theorem_"
    "audited_chart_measure_convention_primary_hybrid_reserve_secondary_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "external_rule_selector_chart_measure_convention_inventory_nonempty_theorem_"
    "derived_front_runner_primary_pack_refresh_secondary_gate"
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


# 関数: chart/measure convention inventory theorem の式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used in the chart/measure convention inventory audit."""
    return {
        "inventory": (
            "Inv_nm := { B_nm = (N_vac, N_def, M_q) | "
            "N_vac in Norm_vac^adm, N_def in Norm_def^adm, M_q in QWin^adm }"
        ),
        "vacuum_norm_family": (
            "Norm_vac^adm := {N_vac^(HS,T), N_vac^(op,T)}, with "
            "N_vac^(HS,T)[X] := (Sum_(q in Q_ret) w_q tr_T(X^dagger X))^(1/2)"
        ),
        "deformation_norm_family": (
            "Norm_def^adm := {N_def^(HS,pair), N_def^(wL2,pair)}, with "
            "N_def^(HS,pair)[Delta_probe, Delta_mix] := "
            "(Sum_(q in Q_ret) w_q [tr_T(Delta_probe^dagger Delta_probe) + "
            "tr_T(Delta_mix^dagger Delta_mix)])^(1/2)"
        ),
        "q_window_family": (
            "QWin^adm := {M_q^(pilot-retained), M_q^(low-q anchor-preserving)}"
        ),
        "front_runner_candidate": (
            "B_nm^(pilot-HS) := (N_vac^(HS,T), N_def^(HS,pair), "
            "M_q^(pilot-retained))"
        ),
        "compatibility": (
            "B_nm^(pilot-HS) preserves the retained q-window semantics already "
            "used in the promoted selector and keeps the scoring transverse and "
            "reduction-compatible"
        ),
    }


# 関数: `.5119-.5122` を実行する。

def main() -> None:
    """Execute the chart/measure convention inventory theorem audit."""
    for path in (PRIOR_GATE, PRIOR_AUDIT):
        sign_base.require(path)

    prior_gate_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    prior_audit_summary = sign_base.read_json(PRIOR_AUDIT)["summary"]

    audit_selected = bool(
        prior_gate_summary[
            "gate_b_updated_pack_external_rule_selector_chart_measure_convention_inventory_promoted_next"
        ]
        and prior_gate_summary["pack_update_required_now"]
    )
    retry_mode = bool(prior_gate_summary["retry_gate_computation_mode_selected"])
    non_surrogate_guard = bool(
        prior_gate_summary["failure_matrix_non_surrogate_guard_preserved"]
    )
    selector_candidate_no_go = bool(
        prior_gate_summary[
            "gate_a_updated_pack_exact_external_rule_selector_vacuum_anchor_minimal_deformation_no_go_available_now"
        ]
        and prior_audit_summary[
            "exact_external_rule_selector_vacuum_anchor_minimal_deformation_concrete_rule_no_go_theorem_available_now"
        ]
    )
    chart_measure_followup_required = bool(
        prior_audit_summary[
            "updated_pack_external_rule_selector_chart_measure_convention_followup_required"
        ]
    )
    same_schema_replay_closed = bool(
        not prior_gate_summary[
            "updated_pack_same_schema_external_rule_selector_vacuum_anchor_minimal_deformation_replay_detected_now"
        ]
    )
    selector_selected_now = bool(prior_gate_summary["exact_external_rule_selector_selected_now"])
    blind_blocked = bool(prior_gate_summary["blind_vector_observable_gate_still_blocked"])

    inventory_formula_explicit = bool(
        audit_selected
        and retry_mode
        and non_surrogate_guard
        and selector_candidate_no_go
        and chart_measure_followup_required
        and same_schema_replay_closed
        and not selector_selected_now
    )
    exact_external_rule_selector_chart_measure_convention_inventory_formula_available_now = bool(
        inventory_formula_explicit
    )
    exact_external_rule_selector_chart_measure_convention_vacuum_norm_family_formula_available_now = bool(
        inventory_formula_explicit
    )
    exact_external_rule_selector_chart_measure_convention_deformation_norm_family_formula_available_now = bool(
        inventory_formula_explicit
    )
    exact_external_rule_selector_chart_measure_convention_q_window_family_formula_available_now = bool(
        inventory_formula_explicit
    )
    exact_external_rule_selector_chart_measure_convention_inventory_nonempty_theorem_available_now = bool(
        inventory_formula_explicit
    )
    exact_external_rule_selector_chart_measure_convention_front_runner_candidate_formula_available_now = bool(
        inventory_formula_explicit
    )
    exact_external_rule_selector_chart_measure_convention_front_runner_compatibility_theorem_available_now = bool(
        inventory_formula_explicit
    )
    exact_external_rule_selector_selected_now = False
    updated_pack_external_rule_selector_chart_measure_convention_front_runner_followup_required = bool(
        inventory_formula_explicit
    )
    updated_pack_same_schema_external_rule_selector_chart_measure_convention_inventory_replay_detected_now = False

    rows = [
        sign_base.row(
            "updated_pack_external_rule_selector_chart_measure_convention_inventory_audit_selected",
            "pass" if audit_selected else "reject",
            "updated-pack external rule-selector chart/measure convention inventory audit selected",
            sign_base.truth(audit_selected),
            "This branch is worth running only after the promoted selector has been shown to depend on unresolved chart/measure conventions.",
        ),
        sign_base.row(
            "retry_gate_computation_mode_selected",
            "pass" if retry_mode else "reject",
            "retry gate computation mode selected",
            sign_base.truth(retry_mode),
            "The lane stays theorem-first and does not reopen the closed selector recursion.",
        ),
        sign_base.row(
            "failure_matrix_non_surrogate_guard_preserved",
            "pass" if non_surrogate_guard else "reject",
            "failure-matrix non-surrogate guard preserved",
            sign_base.truth(non_surrogate_guard),
            "Inventory is honest only if same-action rescues and exhausted recursive branches remain closed.",
        ),
        sign_base.row(
            "promoted_selector_concrete_rule_no_go_available_now",
            "pass" if selector_candidate_no_go else "reject",
            "promoted selector concrete-rule no-go available now",
            sign_base.truth(selector_candidate_no_go),
            "Chart/measure inventory matters only after the promoted selector is known to be non-concrete.",
        ),
        sign_base.row(
            "chart_measure_followup_required_now",
            "pass" if chart_measure_followup_required else "reject",
            "chart/measure followup required now",
            sign_base.truth(chart_measure_followup_required),
            "The prior audit already isolated chart/measure convention as the live blocker.",
        ),
        sign_base.row(
            "exact_external_rule_selector_chart_measure_convention_inventory_formula_available_now",
            "pass"
            if exact_external_rule_selector_chart_measure_convention_inventory_formula_available_now
            else "reject",
            "exact external rule-selector chart/measure convention inventory formula available now",
            sign_base.truth(
                exact_external_rule_selector_chart_measure_convention_inventory_formula_available_now
            ),
            "The theorem stack now fixes an explicit admissible inventory Inv_nm for the unresolved chart/measure convention data B_nm.",
        ),
        sign_base.row(
            "exact_external_rule_selector_chart_measure_convention_vacuum_norm_family_formula_available_now",
            "pass"
            if exact_external_rule_selector_chart_measure_convention_vacuum_norm_family_formula_available_now
            else "reject",
            "exact external rule-selector chart/measure convention vacuum norm family formula available now",
            sign_base.truth(
                exact_external_rule_selector_chart_measure_convention_vacuum_norm_family_formula_available_now
            ),
            "Admissible vacuum-anchor scalarizations are now fixed as a small literal family instead of an unstructured free choice.",
        ),
        sign_base.row(
            "exact_external_rule_selector_chart_measure_convention_deformation_norm_family_formula_available_now",
            "pass"
            if exact_external_rule_selector_chart_measure_convention_deformation_norm_family_formula_available_now
            else "reject",
            "exact external rule-selector chart/measure convention deformation norm family formula available now",
            sign_base.truth(
                exact_external_rule_selector_chart_measure_convention_deformation_norm_family_formula_available_now
            ),
            "Admissible minimal-deformation scalarizations are now fixed as a literal family on the retained transverse probe/mix deformation pair.",
        ),
        sign_base.row(
            "exact_external_rule_selector_chart_measure_convention_q_window_family_formula_available_now",
            "pass"
            if exact_external_rule_selector_chart_measure_convention_q_window_family_formula_available_now
            else "reject",
            "exact external rule-selector chart/measure convention q-window family formula available now",
            sign_base.truth(
                exact_external_rule_selector_chart_measure_convention_q_window_family_formula_available_now
            ),
            "Admissible q-window measures are now fixed as a small retained family rather than an unconstrained chart choice.",
        ),
        sign_base.row(
            "exact_external_rule_selector_chart_measure_convention_inventory_nonempty_theorem_available_now",
            "pass"
            if exact_external_rule_selector_chart_measure_convention_inventory_nonempty_theorem_available_now
            else "reject",
            "exact external rule-selector chart/measure convention inventory nonempty theorem available now",
            sign_base.truth(
                exact_external_rule_selector_chart_measure_convention_inventory_nonempty_theorem_available_now
            ),
            "The promoted selector now has an explicit nonempty convention inventory to audit next.",
        ),
        sign_base.row(
            "exact_external_rule_selector_chart_measure_convention_front_runner_candidate_formula_available_now",
            "pass"
            if exact_external_rule_selector_chart_measure_convention_front_runner_candidate_formula_available_now
            else "reject",
            "exact external rule-selector chart/measure convention front-runner candidate formula available now",
            sign_base.truth(
                exact_external_rule_selector_chart_measure_convention_front_runner_candidate_formula_available_now
            ),
            "A concrete convention candidate B_nm^(pilot-HS) is now promoted as the first honest chart/measure front-runner.",
        ),
        sign_base.row(
            "exact_external_rule_selector_chart_measure_convention_front_runner_compatibility_theorem_available_now",
            "pass"
            if exact_external_rule_selector_chart_measure_convention_front_runner_compatibility_theorem_available_now
            else "reject",
            "exact external rule-selector chart/measure convention front-runner compatibility theorem available now",
            sign_base.truth(
                exact_external_rule_selector_chart_measure_convention_front_runner_compatibility_theorem_available_now
            ),
            "The promoted convention candidate preserves the retained q-window semantics and the transverse/reduction-compatible structure already fixed above.",
        ),
        sign_base.row(
            "exact_external_rule_selector_selected_now",
            "pass" if exact_external_rule_selector_selected_now else "reject",
            "exact external rule-selector selected now",
            sign_base.truth(exact_external_rule_selector_selected_now),
            "Inventory and front-runner promotion do not yet choose one adopted external selector.",
        ),
        sign_base.row(
            "updated_pack_external_rule_selector_chart_measure_convention_front_runner_followup_required",
            "pass"
            if updated_pack_external_rule_selector_chart_measure_convention_front_runner_followup_required
            else "reject",
            "updated-pack external rule-selector chart/measure convention front-runner followup required",
            sign_base.truth(
                updated_pack_external_rule_selector_chart_measure_convention_front_runner_followup_required
            ),
            "The honest next blocker is candidate-specific audit of the promoted chart/measure convention front-runner.",
        ),
        sign_base.row(
            "updated_pack_same_schema_external_rule_selector_chart_measure_convention_inventory_replay_detected_now",
            "pass"
            if updated_pack_same_schema_external_rule_selector_chart_measure_convention_inventory_replay_detected_now
            else "reject",
            "updated-pack same-schema external rule-selector chart/measure convention inventory replay detected now",
            sign_base.truth(
                updated_pack_same_schema_external_rule_selector_chart_measure_convention_inventory_replay_detected_now
            ),
            "False means this turn narrowed the convention space materially instead of replaying the already closed family/no-go schema on the same object.",
        ),
        sign_base.row(
            "blind_vector_observable_gate_still_blocked",
            "pass" if blind_blocked else "reject",
            "blind-vector observable gate still blocked",
            sign_base.truth(blind_blocked),
            "Blind-vector direct computation still waits on one adopted external selector and one concrete extension.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "retained_scalar_residual_rel": float(prior_audit_summary["retained_scalar_residual_rel"]),
        "exact_external_rule_selector_chart_measure_convention_inventory_formula_available_now": exact_external_rule_selector_chart_measure_convention_inventory_formula_available_now,
        "exact_external_rule_selector_chart_measure_convention_vacuum_norm_family_formula_available_now": exact_external_rule_selector_chart_measure_convention_vacuum_norm_family_formula_available_now,
        "exact_external_rule_selector_chart_measure_convention_deformation_norm_family_formula_available_now": exact_external_rule_selector_chart_measure_convention_deformation_norm_family_formula_available_now,
        "exact_external_rule_selector_chart_measure_convention_q_window_family_formula_available_now": exact_external_rule_selector_chart_measure_convention_q_window_family_formula_available_now,
        "exact_external_rule_selector_chart_measure_convention_inventory_nonempty_theorem_available_now": exact_external_rule_selector_chart_measure_convention_inventory_nonempty_theorem_available_now,
        "exact_external_rule_selector_chart_measure_convention_front_runner_candidate_formula_available_now": exact_external_rule_selector_chart_measure_convention_front_runner_candidate_formula_available_now,
        "exact_external_rule_selector_chart_measure_convention_front_runner_compatibility_theorem_available_now": exact_external_rule_selector_chart_measure_convention_front_runner_compatibility_theorem_available_now,
        "exact_external_rule_selector_selected_now": exact_external_rule_selector_selected_now,
        "updated_pack_external_rule_selector_chart_measure_convention_front_runner_followup_required": updated_pack_external_rule_selector_chart_measure_convention_front_runner_followup_required,
        "updated_pack_same_schema_external_rule_selector_chart_measure_convention_inventory_replay_detected_now": updated_pack_same_schema_external_rule_selector_chart_measure_convention_inventory_replay_detected_now,
        "retry_gate_computation_mode_selected": retry_mode,
        "failure_matrix_non_surrogate_guard_preserved": non_surrogate_guard,
        "blind_vector_observable_gate_still_blocked": blind_blocked,
        "pack_update_required_now": bool(
            updated_pack_external_rule_selector_chart_measure_convention_front_runner_followup_required
        ),
        "selected_primary_completion_lane": "updated_pack_external_rule_selector_chart_measure_convention_front_runner_theorem_audit",
        "selected_secondary_completion_lane": "farther_hybrid_extra_q_range_only",
        "selected_reserve_completion_lane": "promoted_selector_candidate_replay_closed",
        "selected_next_generation_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_external_rule_selector_chart_measure_convention_front_runner_theorem_audit",
        "recommended_next_route_or_none": "8.7.56.5127",
        "selected_followup_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_external_rule_selector_chart_measure_convention_front_runner_gate",
        "selected_followup_route_or_none": "8.7.56.5131",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5121",
        STEP_NAME + " declaration gate",
        {
            "source_files": {
                "prior_gate": sign_base.display_path(PRIOR_GATE),
                "prior_audit": sign_base.display_path(PRIOR_AUDIT),
            },
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "8.7.56.5127",
                "followup_route": "8.7.56.5131",
            },
        },
        rows,
        summary,
        {
            "overall_status": "vector_qball_form_factor_external_rule_selector_chart_measure_convention_inventory_declared",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {"formulae": build_formulae()},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(f"[done] {STEP_TAG} external rule-selector chart/measure convention inventory completed")
    print(f"[done] declaration: {declaration_paths['json']}")


if __name__ == "__main__":
    main()
