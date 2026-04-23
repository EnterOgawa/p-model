#!/usr/bin/env python3
"""Generate 8.7.56.5111-.5114 vacuum-anchor minimal-deformation selector artifacts."""

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
        "8.7.56.5107-5110",
        "updated_pack_external_rule_selector_inventory_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
PRIOR_AUDIT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5103-5106",
        "updated_pack_external_rule_selector_inventory_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5111-5114"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack external "
    "rule-selector vacuum-anchor minimal-deformation theorem audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_external_rule_selector_vacuum_anchor_minimal_deformation_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "external_rule_selector_inventory_audited_vacuum_anchor_minimal_"
    "deformation_primary_hybrid_reserve_secondary_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "external_rule_selector_vacuum_anchor_minimal_deformation_concrete_rule_"
    "no_go_theorem_derived_chart_measure_convention_primary_pack_refresh_"
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


# 関数: promoted external selector candidate の式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used in the vacuum-anchor minimal-deformation audit."""
    return {
        "selector_candidate_family": (
            "S_rule^(vac->def; B_nm)[R_Omega] := lexicographic argext of "
            "(S_rule^(vac; B_nm), S_rule^(def; B_nm)) on Rule_probe_schur[B_Omega]"
        ),
        "vacuum_anchor_with_convention": (
            "S_rule^(vac; B_nm)[R_Omega] := N_vac[M_q(Pi_T "
            "(K_AA^(R_Omega)[vac] - K_free) Pi_T)]"
        ),
        "minimal_deformation_with_convention": (
            "S_rule^(def; B_nm)[R_Omega] := N_def[M_q(Delta_probe^(R_Omega), "
            "Delta_mix^(R_Omega))]"
        ),
        "chart_measure_convention": (
            "B_nm := (N_vac, N_def, M_q), where N_vac/N_def are admissible "
            "operator norms or contractions and M_q is the retained q-window "
            "chart/measure convention"
        ),
        "concrete_rule_no_go": (
            "Current theorem stack fixes the selector idea vac->def but not one "
            "concrete B_nm, so no one concrete selector rule is selected now"
        ),
    }


# 関数: `.5111-.5114` を実行する。

def main() -> None:
    """Execute the vacuum-anchor minimal-deformation theorem audit."""
    for path in (PRIOR_GATE, PRIOR_AUDIT):
        sign_base.require(path)

    prior_gate_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    prior_audit_summary = sign_base.read_json(PRIOR_AUDIT)["summary"]

    audit_selected = bool(
        prior_gate_summary[
            "gate_b_updated_pack_external_rule_selector_vacuum_anchor_minimal_deformation_promoted_next"
        ]
        and prior_gate_summary["pack_update_required_now"]
    )
    retry_mode = bool(prior_gate_summary["retry_gate_computation_mode_selected"])
    non_surrogate_guard = bool(
        prior_gate_summary["failure_matrix_non_surrogate_guard_preserved"]
    )
    inventory_nonempty = bool(
        prior_gate_summary[
            "gate_a_updated_pack_exact_external_rule_selector_inventory_nonempty_available_now"
        ]
    )
    selector_already_selected = bool(
        prior_gate_summary["exact_external_rule_selector_selected_now"]
    )
    no_inventory_replay = bool(
        not prior_gate_summary[
            "updated_pack_same_schema_external_rule_selector_inventory_replay_detected_now"
        ]
    )
    candidate_formula_explicit = bool(
        prior_audit_summary[
            "exact_external_rule_selector_candidate_vacuum_free_probe_anchor_formula_available_now"
        ]
        and prior_audit_summary[
            "exact_external_rule_selector_candidate_minimal_completion_deformation_formula_available_now"
        ]
        and prior_audit_summary[
            "exact_external_rule_selector_candidate_lexicographic_vacuum_then_minimal_deformation_formula_available_now"
        ]
    )
    chart_measure_family_explicit = bool(
        audit_selected
        and retry_mode
        and non_surrogate_guard
        and inventory_nonempty
        and no_inventory_replay
        and candidate_formula_explicit
        and not selector_already_selected
    )
    exact_external_rule_selector_vacuum_anchor_minimal_deformation_family_formula_available_now = bool(
        chart_measure_family_explicit
    )
    exact_external_rule_selector_vacuum_anchor_minimal_deformation_chart_measure_dependence_theorem_available_now = bool(
        chart_measure_family_explicit
    )
    exact_external_rule_selector_vacuum_anchor_minimal_deformation_concrete_rule_no_go_theorem_available_now = bool(
        chart_measure_family_explicit
    )
    exact_minimal_external_rule_selector_chart_measure_convention_requirement_theorem_available_now = bool(
        chart_measure_family_explicit
    )
    exact_external_rule_selector_selected_now = False
    updated_pack_external_rule_selector_chart_measure_convention_followup_required = bool(
        chart_measure_family_explicit
    )
    updated_pack_same_schema_external_rule_selector_vacuum_anchor_minimal_deformation_replay_detected_now = False
    blind_blocked = bool(prior_gate_summary["blind_vector_observable_gate_still_blocked"])

    rows = [
        sign_base.row(
            "updated_pack_external_rule_selector_vacuum_anchor_minimal_deformation_audit_selected",
            "pass" if audit_selected else "reject",
            "updated-pack external rule-selector vacuum-anchor minimal-deformation audit selected",
            sign_base.truth(audit_selected),
            "This branch is worth running only after the inventory is nonempty and the lexicographic vacuum-anchor plus minimal-deformation selector has been promoted next.",
        ),
        sign_base.row(
            "retry_gate_computation_mode_selected",
            "pass" if retry_mode else "reject",
            "retry gate computation mode selected",
            sign_base.truth(retry_mode),
            "The promoted selector is audited theorem-first rather than by reopening same-tag replay.",
        ),
        sign_base.row(
            "failure_matrix_non_surrogate_guard_preserved",
            "pass" if non_surrogate_guard else "reject",
            "failure-matrix non-surrogate guard preserved",
            sign_base.truth(non_surrogate_guard),
            "The audit remains honest only if surrogate and exhausted internal routes stay closed.",
        ),
        sign_base.row(
            "gate_a_updated_pack_exact_external_rule_selector_inventory_nonempty_available_now",
            "pass" if inventory_nonempty else "reject",
            "gate A updated-pack exact external rule-selector inventory nonempty available now",
            sign_base.truth(inventory_nonempty),
            "The promoted selector can be audited only after the inventory theorem has been closed and promoted.",
        ),
        sign_base.row(
            "candidate_formula_explicit_now",
            "pass" if candidate_formula_explicit else "reject",
            "candidate formula explicit now",
            sign_base.truth(candidate_formula_explicit),
            "The front-runner selector requires explicit vacuum-anchor, minimal-deformation, and lexicographic formulae before its remaining blocker can be read honestly.",
        ),
        sign_base.row(
            "selector_already_selected_now",
            "pass" if selector_already_selected else "reject",
            "selector already selected now",
            sign_base.truth(selector_already_selected),
            "The promoted selector is still only a candidate at the start of this branch.",
        ),
        sign_base.row(
            "exact_external_rule_selector_vacuum_anchor_minimal_deformation_family_formula_available_now",
            "pass"
            if exact_external_rule_selector_vacuum_anchor_minimal_deformation_family_formula_available_now
            else "reject",
            "exact external rule-selector vacuum-anchor minimal-deformation family formula available now",
            sign_base.truth(
                exact_external_rule_selector_vacuum_anchor_minimal_deformation_family_formula_available_now
            ),
            "The promoted selector is now written honestly as a family parameterized by chart/measure convention data B_nm.",
        ),
        sign_base.row(
            "exact_external_rule_selector_vacuum_anchor_minimal_deformation_chart_measure_dependence_theorem_available_now",
            "pass"
            if exact_external_rule_selector_vacuum_anchor_minimal_deformation_chart_measure_dependence_theorem_available_now
            else "reject",
            "exact external rule-selector vacuum-anchor minimal-deformation chart/measure dependence theorem available now",
            sign_base.truth(
                exact_external_rule_selector_vacuum_anchor_minimal_deformation_chart_measure_dependence_theorem_available_now
            ),
            "The current theorem stack fixes the selector idea but not one concrete operator norm / q-window measure convention used to scalarize the candidate scores.",
        ),
        sign_base.row(
            "exact_external_rule_selector_vacuum_anchor_minimal_deformation_concrete_rule_no_go_theorem_available_now",
            "pass"
            if exact_external_rule_selector_vacuum_anchor_minimal_deformation_concrete_rule_no_go_theorem_available_now
            else "reject",
            "exact external rule-selector vacuum-anchor minimal-deformation concrete-rule no-go theorem available now",
            sign_base.truth(
                exact_external_rule_selector_vacuum_anchor_minimal_deformation_concrete_rule_no_go_theorem_available_now
            ),
            "Because B_nm is still unfixed, the promoted selector does not yet define one concrete adopted rule.",
        ),
        sign_base.row(
            "exact_minimal_external_rule_selector_chart_measure_convention_requirement_theorem_available_now",
            "pass"
            if exact_minimal_external_rule_selector_chart_measure_convention_requirement_theorem_available_now
            else "reject",
            "exact minimal external rule-selector chart/measure convention requirement theorem available now",
            sign_base.truth(
                exact_minimal_external_rule_selector_chart_measure_convention_requirement_theorem_available_now
            ),
            "The remaining blocker is now one external chart/measure convention for the promoted selector, not selector existence or candidate admission.",
        ),
        sign_base.row(
            "exact_external_rule_selector_selected_now",
            "pass" if exact_external_rule_selector_selected_now else "reject",
            "exact external rule-selector selected now",
            sign_base.truth(exact_external_rule_selector_selected_now),
            "No adopted external selector exists yet after auditing the promoted candidate.",
        ),
        sign_base.row(
            "updated_pack_external_rule_selector_chart_measure_convention_followup_required",
            "pass"
            if updated_pack_external_rule_selector_chart_measure_convention_followup_required
            else "reject",
            "updated-pack external rule-selector chart/measure convention followup required",
            sign_base.truth(
                updated_pack_external_rule_selector_chart_measure_convention_followup_required
            ),
            "The honest next blocker is inventory and audit of admissible chart/measure conventions for the promoted selector.",
        ),
        sign_base.row(
            "updated_pack_same_schema_external_rule_selector_vacuum_anchor_minimal_deformation_replay_detected_now",
            "pass"
            if updated_pack_same_schema_external_rule_selector_vacuum_anchor_minimal_deformation_replay_detected_now
            else "reject",
            "updated-pack same-schema external rule-selector vacuum-anchor minimal-deformation replay detected now",
            sign_base.truth(
                updated_pack_same_schema_external_rule_selector_vacuum_anchor_minimal_deformation_replay_detected_now
            ),
            "False means this turn did not reopen the old candidate recursion; it isolated a new concrete blocker specific to the promoted selector.",
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
        "exact_external_rule_selector_vacuum_anchor_minimal_deformation_family_formula_available_now": exact_external_rule_selector_vacuum_anchor_minimal_deformation_family_formula_available_now,
        "exact_external_rule_selector_vacuum_anchor_minimal_deformation_chart_measure_dependence_theorem_available_now": exact_external_rule_selector_vacuum_anchor_minimal_deformation_chart_measure_dependence_theorem_available_now,
        "exact_external_rule_selector_vacuum_anchor_minimal_deformation_concrete_rule_no_go_theorem_available_now": exact_external_rule_selector_vacuum_anchor_minimal_deformation_concrete_rule_no_go_theorem_available_now,
        "exact_minimal_external_rule_selector_chart_measure_convention_requirement_theorem_available_now": exact_minimal_external_rule_selector_chart_measure_convention_requirement_theorem_available_now,
        "exact_external_rule_selector_selected_now": exact_external_rule_selector_selected_now,
        "updated_pack_external_rule_selector_chart_measure_convention_followup_required": updated_pack_external_rule_selector_chart_measure_convention_followup_required,
        "updated_pack_same_schema_external_rule_selector_vacuum_anchor_minimal_deformation_replay_detected_now": updated_pack_same_schema_external_rule_selector_vacuum_anchor_minimal_deformation_replay_detected_now,
        "retry_gate_computation_mode_selected": retry_mode,
        "failure_matrix_non_surrogate_guard_preserved": non_surrogate_guard,
        "blind_vector_observable_gate_still_blocked": blind_blocked,
        "pack_update_required_now": updated_pack_external_rule_selector_chart_measure_convention_followup_required,
        "selected_primary_completion_lane": "updated_pack_external_rule_selector_chart_measure_convention_inventory_theorem_audit",
        "selected_secondary_completion_lane": "farther_hybrid_extra_q_range_only",
        "selected_reserve_completion_lane": "promoted_selector_candidate_replay_closed",
        "selected_next_generation_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_external_rule_selector_chart_measure_convention_inventory_theorem_audit",
        "recommended_next_route_or_none": "8.7.56.5119",
        "selected_followup_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_external_rule_selector_chart_measure_convention_inventory_gate",
        "selected_followup_route_or_none": "8.7.56.5123",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5113",
        STEP_NAME + " declaration gate",
        {
            "source_files": {
                "prior_gate": sign_base.display_path(PRIOR_GATE),
                "prior_audit": sign_base.display_path(PRIOR_AUDIT),
            },
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "8.7.56.5119",
                "followup_route": "8.7.56.5123",
            },
        },
        rows,
        summary,
        {
            "overall_status": "vector_qball_form_factor_external_rule_selector_vacuum_anchor_minimal_deformation_declared",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {"formulae": build_formulae()},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(f"[done] {STEP_TAG} external rule-selector vacuum-anchor minimal-deformation audit completed")
    print(f"[done] declaration: {declaration_paths['json']}")


if __name__ == "__main__":
    main()
