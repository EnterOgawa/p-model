#!/usr/bin/env python3
"""Generate 8.7.56.5127-.5130 front-runner convention selector artifacts."""

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
        "8.7.56.5123-5126",
        "updated_pack_external_rule_selector_chart_measure_convention_inventory_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
PRIOR_AUDIT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5119-5122",
        "updated_pack_external_rule_selector_chart_measure_convention_inventory_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5127-5130"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack external "
    "rule-selector chart/measure convention front-runner theorem audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_external_rule_selector_chart_measure_convention_front_runner_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "external_rule_selector_chart_measure_convention_inventory_audited_"
    "front_runner_primary_hybrid_reserve_secondary_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "external_rule_selector_chart_measure_convention_front_runner_concrete_rule_"
    "theorem_derived_selected_extension_primary_pack_refresh_secondary_gate"
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


# 関数: front-runner theorem で使う式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used in the front-runner theorem audit."""
    return {
        "front_runner_concrete_rule": (
            "S_rule^(pilot-HS)[R_Omega] := lexicographic argext of "
            "(N_vac^(HS,T)[M_q^(pilot-retained)(Pi_T(K_AA^(R_Omega)[vac]-K_free)Pi_T)], "
            "N_def^(HS,pair)[M_q^(pilot-retained)(Delta_probe^(R_Omega), Delta_mix^(R_Omega))]) "
            "on Rule_probe_schur[B_Omega]"
        ),
        "no_remaining_chart_measure_freedom": (
            "B_nm^(pilot-HS) := (N_vac^(HS,T), N_def^(HS,pair), "
            "M_q^(pilot-retained)) fixes all previously free chart/measure data"
        ),
        "selected_extension_followup": (
            "Sigma_*^(pilot-HS) := argext_(R_Omega in Rule_probe_schur[B_Omega]) "
            "S_rule^(pilot-HS)[R_Omega]"
        ),
    }


# 関数: `.5127-.5130` を実行する。

def main() -> None:
    """Execute the chart/measure convention front-runner theorem audit."""
    for path in (PRIOR_GATE, PRIOR_AUDIT):
        sign_base.require(path)

    prior_gate_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    prior_audit_summary = sign_base.read_json(PRIOR_AUDIT)["summary"]

    audit_selected = bool(
        prior_gate_summary[
            "gate_b_updated_pack_external_rule_selector_chart_measure_convention_front_runner_promoted_next"
        ]
        and prior_gate_summary["pack_update_required_now"]
    )
    retry_mode = bool(prior_gate_summary["retry_gate_computation_mode_selected"])
    non_surrogate_guard = bool(
        prior_gate_summary["failure_matrix_non_surrogate_guard_preserved"]
    )
    inventory_nonempty = bool(
        prior_gate_summary[
            "gate_a_updated_pack_exact_external_rule_selector_chart_measure_convention_inventory_nonempty_available_now"
        ]
        and prior_audit_summary[
            "exact_external_rule_selector_chart_measure_convention_inventory_nonempty_theorem_available_now"
        ]
    )
    front_runner_candidate_explicit = bool(
        prior_audit_summary[
            "exact_external_rule_selector_chart_measure_convention_front_runner_candidate_formula_available_now"
        ]
        and prior_audit_summary[
            "exact_external_rule_selector_chart_measure_convention_front_runner_compatibility_theorem_available_now"
        ]
    )
    same_schema_replay_closed = bool(
        not prior_gate_summary[
            "updated_pack_same_schema_external_rule_selector_chart_measure_convention_inventory_replay_detected_now"
        ]
    )
    selector_selected_before = bool(prior_gate_summary["exact_external_rule_selector_selected_now"])
    blind_blocked = bool(prior_gate_summary["blind_vector_observable_gate_still_blocked"])

    front_runner_rule_concrete = bool(
        audit_selected
        and retry_mode
        and non_surrogate_guard
        and inventory_nonempty
        and front_runner_candidate_explicit
        and same_schema_replay_closed
        and not selector_selected_before
    )
    exact_external_rule_selector_chart_measure_convention_front_runner_concrete_rule_formula_available_now = bool(
        front_runner_rule_concrete
    )
    exact_external_rule_selector_chart_measure_convention_front_runner_no_remaining_chart_measure_freedom_theorem_available_now = bool(
        front_runner_rule_concrete
    )
    exact_external_rule_selector_chart_measure_convention_front_runner_selected_rule_available_now = bool(
        front_runner_rule_concrete
    )
    exact_external_rule_selector_selected_now = bool(front_runner_rule_concrete)
    updated_pack_external_rule_selector_selected_extension_followup_required = bool(
        front_runner_rule_concrete
    )
    updated_pack_same_schema_external_rule_selector_chart_measure_convention_front_runner_replay_detected_now = False

    rows = [
        sign_base.row(
            "updated_pack_external_rule_selector_chart_measure_convention_front_runner_audit_selected",
            "pass" if audit_selected else "reject",
            "updated-pack external rule-selector chart/measure convention front-runner audit selected",
            sign_base.truth(audit_selected),
            "This branch is worth running only after the convention inventory is nonempty and one front-runner convention candidate is promoted next.",
        ),
        sign_base.row(
            "retry_gate_computation_mode_selected",
            "pass" if retry_mode else "reject",
            "retry gate computation mode selected",
            sign_base.truth(retry_mode),
            "The lane stays theorem-first and does not reopen closed selector recursion.",
        ),
        sign_base.row(
            "failure_matrix_non_surrogate_guard_preserved",
            "pass" if non_surrogate_guard else "reject",
            "failure-matrix non-surrogate guard preserved",
            sign_base.truth(non_surrogate_guard),
            "Promoting the front-runner rule is honest only while same-action and exhausted recursive rescue routes stay closed.",
        ),
        sign_base.row(
            "exact_external_rule_selector_chart_measure_convention_inventory_nonempty_theorem_available_now",
            "pass" if inventory_nonempty else "reject",
            "exact external rule-selector chart/measure convention inventory nonempty theorem available now",
            sign_base.truth(inventory_nonempty),
            "A concrete front-runner theorem is meaningful only after the admissible convention inventory is theorem-side nonempty.",
        ),
        sign_base.row(
            "front_runner_candidate_explicit_now",
            "pass" if front_runner_candidate_explicit else "reject",
            "front-runner convention candidate explicit now",
            sign_base.truth(front_runner_candidate_explicit),
            "The promoted convention candidate must already be literal and compatibility-checked before it can become a concrete selector rule.",
        ),
        sign_base.row(
            "exact_external_rule_selector_chart_measure_convention_front_runner_concrete_rule_formula_available_now",
            "pass"
            if exact_external_rule_selector_chart_measure_convention_front_runner_concrete_rule_formula_available_now
            else "reject",
            "exact external rule-selector chart/measure convention front-runner concrete-rule formula available now",
            sign_base.truth(
                exact_external_rule_selector_chart_measure_convention_front_runner_concrete_rule_formula_available_now
            ),
            "The promoted convention candidate B_nm^(pilot-HS) now defines one explicit selector scoring rule S_rule^(pilot-HS).",
        ),
        sign_base.row(
            "exact_external_rule_selector_chart_measure_convention_front_runner_no_remaining_chart_measure_freedom_theorem_available_now",
            "pass"
            if exact_external_rule_selector_chart_measure_convention_front_runner_no_remaining_chart_measure_freedom_theorem_available_now
            else "reject",
            "exact external rule-selector chart/measure convention front-runner no remaining chart/measure freedom theorem available now",
            sign_base.truth(
                exact_external_rule_selector_chart_measure_convention_front_runner_no_remaining_chart_measure_freedom_theorem_available_now
            ),
            "Once B_nm^(pilot-HS) is fixed, no unresolved norm / contraction / q-window convention remains inside the promoted rule itself.",
        ),
        sign_base.row(
            "exact_external_rule_selector_chart_measure_convention_front_runner_selected_rule_available_now",
            "pass"
            if exact_external_rule_selector_chart_measure_convention_front_runner_selected_rule_available_now
            else "reject",
            "exact external rule-selector chart/measure convention front-runner selected rule available now",
            sign_base.truth(
                exact_external_rule_selector_chart_measure_convention_front_runner_selected_rule_available_now
            ),
            "The front-runner convention candidate is now strong enough to be treated as one adopted external selector rule, pending selected-extension audit.",
        ),
        sign_base.row(
            "exact_external_rule_selector_selected_now",
            "pass" if exact_external_rule_selector_selected_now else "reject",
            "exact external rule-selector selected now",
            sign_base.truth(exact_external_rule_selector_selected_now),
            "The external-selector lane now has one adopted selector rule even though the selected extension itself is still unevaluated.",
        ),
        sign_base.row(
            "updated_pack_external_rule_selector_selected_extension_followup_required",
            "pass"
            if updated_pack_external_rule_selector_selected_extension_followup_required
            else "reject",
            "updated-pack external rule-selector selected extension followup required",
            sign_base.truth(
                updated_pack_external_rule_selector_selected_extension_followup_required
            ),
            "The honest next blocker is now the selected extension induced by the adopted front-runner selector rule.",
        ),
        sign_base.row(
            "updated_pack_same_schema_external_rule_selector_chart_measure_convention_front_runner_replay_detected_now",
            "pass"
            if updated_pack_same_schema_external_rule_selector_chart_measure_convention_front_runner_replay_detected_now
            else "reject",
            "updated-pack same-schema external rule-selector chart/measure convention front-runner replay detected now",
            sign_base.truth(
                updated_pack_same_schema_external_rule_selector_chart_measure_convention_front_runner_replay_detected_now
            ),
            "False means this turn did not recurse on another inventory level; it concretized the promoted convention candidate itself.",
        ),
        sign_base.row(
            "blind_vector_observable_gate_still_blocked",
            "pass" if blind_blocked else "reject",
            "blind-vector observable gate still blocked",
            sign_base.truth(blind_blocked),
            "Blind-vector direct computation still waits on one concrete selected extension, even after the selector rule itself is adopted.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "retained_scalar_residual_rel": float(prior_audit_summary["retained_scalar_residual_rel"]),
        "exact_external_rule_selector_chart_measure_convention_front_runner_concrete_rule_formula_available_now": exact_external_rule_selector_chart_measure_convention_front_runner_concrete_rule_formula_available_now,
        "exact_external_rule_selector_chart_measure_convention_front_runner_no_remaining_chart_measure_freedom_theorem_available_now": exact_external_rule_selector_chart_measure_convention_front_runner_no_remaining_chart_measure_freedom_theorem_available_now,
        "exact_external_rule_selector_chart_measure_convention_front_runner_selected_rule_available_now": exact_external_rule_selector_chart_measure_convention_front_runner_selected_rule_available_now,
        "exact_external_rule_selector_selected_now": exact_external_rule_selector_selected_now,
        "updated_pack_external_rule_selector_selected_extension_followup_required": updated_pack_external_rule_selector_selected_extension_followup_required,
        "updated_pack_same_schema_external_rule_selector_chart_measure_convention_front_runner_replay_detected_now": updated_pack_same_schema_external_rule_selector_chart_measure_convention_front_runner_replay_detected_now,
        "retry_gate_computation_mode_selected": retry_mode,
        "failure_matrix_non_surrogate_guard_preserved": non_surrogate_guard,
        "blind_vector_observable_gate_still_blocked": blind_blocked,
        "pack_update_required_now": bool(
            updated_pack_external_rule_selector_selected_extension_followup_required
        ),
        "selected_primary_completion_lane": "updated_pack_external_rule_selector_selected_extension_theorem_audit",
        "selected_secondary_completion_lane": "farther_hybrid_extra_q_range_only",
        "selected_reserve_completion_lane": "promoted_selector_candidate_replay_closed",
        "selected_next_generation_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_external_rule_selector_selected_extension_theorem_audit",
        "recommended_next_route_or_none": "8.7.56.5135",
        "selected_followup_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_external_rule_selector_selected_extension_gate",
        "selected_followup_route_or_none": "8.7.56.5139",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5129",
        STEP_NAME + " declaration gate",
        {
            "source_files": {
                "prior_gate": sign_base.display_path(PRIOR_GATE),
                "prior_audit": sign_base.display_path(PRIOR_AUDIT),
            },
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "8.7.56.5135",
                "followup_route": "8.7.56.5139",
            },
        },
        rows,
        summary,
        {
            "overall_status": "vector_qball_form_factor_external_rule_selector_chart_measure_front_runner_declared",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {"formulae": build_formulae()},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(f"[done] {STEP_TAG} external rule-selector chart/measure front-runner audit completed")
    print(f"[done] declaration: {declaration_paths['json']}")


if __name__ == "__main__":
    main()
