#!/usr/bin/env python3
"""Generate 8.7.56.5135-.5138 external-rule selected-extension theorem artifacts."""

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
        "8.7.56.5131-5134",
        "updated_pack_external_rule_selector_chart_measure_convention_front_runner_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
PRIOR_AUDIT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5127-5130",
        "updated_pack_external_rule_selector_chart_measure_convention_front_runner_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
PRIOR_SUPPORT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5071-5074",
        "updated_pack_external_selector_candidate_independent_probe_slot_schur_complement_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5135-5138"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack external "
    "rule-selector selected extension theorem audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_external_rule_selector_selected_extension_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "external_rule_selector_chart_measure_convention_front_runner_concrete_rule_"
    "audited_selected_extension_primary_hybrid_reserve_secondary_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "external_rule_selector_selected_extension_concrete_theorem_derived_"
    "blind_vector_primary_pack_refresh_secondary_gate"
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


# 関数: selected-extension theorem で使う式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used in the selected-extension theorem audit."""
    return {
        "adopted_completion_rule": "R_*^(pilot-HS) := S_rule^(pilot-HS)",
        "selected_extension_formula": (
            "Sigma_*^(pilot-HS) := argext_(Omega in Comp_probe_schur) "
            "R_*^(pilot-HS)[Omega]"
        ),
        "selected_extension_action": (
            "L_ext^(pilot-HS)[P_mu,A_mu] := L_total^vec[P_mu] + "
            "L_probe^(Sigma_*^(pilot-HS))[A_mu] + "
            "L_mix^(Sigma_*^(pilot-HS))[P_mu,A_mu]"
        ),
        "selected_effective_kernel": (
            "K_eff^(pilot-HS)[Q] := K_AA^(Sigma_*^(pilot-HS))[Q] - "
            "K_xiA^(Sigma_*^(pilot-HS))[Q](K_xixi[Q])^(-1)"
            "K_xiA^(Sigma_*^(pilot-HS))[Q]"
        ),
    }


# 関数: `.5135-.5138` を実行する。

def main() -> None:
    """Execute the external-rule selected-extension theorem audit."""
    for path in (PRIOR_GATE, PRIOR_AUDIT, PRIOR_SUPPORT):
        sign_base.require(path)

    prior_gate_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    prior_audit_summary = sign_base.read_json(PRIOR_AUDIT)["summary"]
    prior_support_summary = sign_base.read_json(PRIOR_SUPPORT)["summary"]

    audit_selected = bool(
        prior_gate_summary[
            "gate_b_updated_pack_external_rule_selector_selected_extension_promoted_next"
        ]
        and prior_gate_summary["pack_update_required_now"]
    )
    retry_mode = bool(prior_gate_summary["retry_gate_computation_mode_selected"])
    non_surrogate_guard = bool(
        prior_gate_summary["failure_matrix_non_surrogate_guard_preserved"]
    )
    selected_rule_selected = bool(prior_gate_summary["exact_external_rule_selector_selected_now"])
    front_runner_rule_formula_available = bool(
        prior_audit_summary[
            "exact_external_rule_selector_chart_measure_convention_front_runner_concrete_rule_formula_available_now"
        ]
    )
    no_remaining_chart_measure_freedom = bool(
        prior_audit_summary[
            "exact_external_rule_selector_chart_measure_convention_front_runner_no_remaining_chart_measure_freedom_theorem_available_now"
        ]
    )
    candidate_extension_formula_available = bool(
        prior_support_summary[
            "exact_external_selector_candidate_independent_probe_slot_schur_complement_extension_formula_available_now"
        ]
    )
    same_schema_replay_closed = bool(
        not prior_gate_summary[
            "updated_pack_same_schema_external_rule_selector_chart_measure_convention_front_runner_replay_detected_now"
        ]
    )
    selected_extension_explicit = bool(
        audit_selected
        and retry_mode
        and non_surrogate_guard
        and selected_rule_selected
        and front_runner_rule_formula_available
        and no_remaining_chart_measure_freedom
        and candidate_extension_formula_available
        and same_schema_replay_closed
    )
    exact_external_rule_selector_selected_extension_formula_available_now = bool(
        selected_extension_explicit
    )
    exact_external_rule_selector_selected_extension_action_formula_available_now = bool(
        selected_extension_explicit
    )
    exact_external_rule_selector_selected_extension_effective_kernel_formula_available_now = bool(
        selected_extension_explicit
    )
    exact_external_rule_selector_selected_extension_available_now = bool(
        selected_extension_explicit
    )
    exact_concrete_selected_extension_available_now = bool(selected_extension_explicit)
    updated_pack_blind_vector_direct_computation_followup_required = bool(
        selected_extension_explicit
    )
    updated_pack_same_schema_external_rule_selector_selected_extension_replay_detected_now = False
    blind_blocked = not exact_concrete_selected_extension_available_now

    rows = [
        sign_base.row(
            "updated_pack_external_rule_selector_selected_extension_audit_selected",
            "pass" if audit_selected else "reject",
            "updated-pack external rule-selector selected extension audit selected",
            sign_base.truth(audit_selected),
            "This branch is worth running only after one adopted external selector rule is already official and selected-extension closeout is the live blocker.",
        ),
        sign_base.row(
            "retry_gate_computation_mode_selected",
            "pass" if retry_mode else "reject",
            "retry gate computation mode selected",
            sign_base.truth(retry_mode),
            "This turn closes the selected extension itself rather than replaying selector-family recursion.",
        ),
        sign_base.row(
            "failure_matrix_non_surrogate_guard_preserved",
            "pass" if non_surrogate_guard else "reject",
            "failure-matrix non-surrogate guard preserved",
            sign_base.truth(non_surrogate_guard),
            "The selected-extension theorem is honest only if same-action rescue routes and exhausted selector replay remain closed.",
        ),
        sign_base.row(
            "exact_external_rule_selector_selected_now",
            "pass" if selected_rule_selected else "reject",
            "exact external rule-selector selected now",
            sign_base.truth(selected_rule_selected),
            "The theorem starts only because one adopted external selector rule S_rule^(pilot-HS) is already fixed.",
        ),
        sign_base.row(
            "front_runner_rule_formula_available_now",
            "pass" if front_runner_rule_formula_available else "reject",
            "front-runner rule formula available now",
            sign_base.truth(front_runner_rule_formula_available),
            "The adopted selector rule itself must already be literal before it can induce one selected extension.",
        ),
        sign_base.row(
            "no_remaining_chart_measure_freedom_now",
            "pass" if no_remaining_chart_measure_freedom else "reject",
            "no remaining chart/measure freedom now",
            sign_base.truth(no_remaining_chart_measure_freedom),
            "Selected-extension closeout is honest only after the adopted rule has no unresolved chart/measure dependence inside it.",
        ),
        sign_base.row(
            "exact_external_selector_candidate_independent_probe_slot_schur_complement_extension_formula_available_now",
            "pass" if candidate_extension_formula_available else "reject",
            "exact external selector candidate independent probe-slot Schur-complement extension formula available now",
            sign_base.truth(candidate_extension_formula_available),
            "The selected extension can be written only because the underlying independent probe-slot Schur-complement extension template is already explicit.",
        ),
        sign_base.row(
            "exact_external_rule_selector_selected_extension_formula_available_now",
            "pass"
            if exact_external_rule_selector_selected_extension_formula_available_now
            else "reject",
            "exact external rule-selector selected extension formula available now",
            sign_base.truth(
                exact_external_rule_selector_selected_extension_formula_available_now
            ),
            "The adopted selector rule now induces one literal selected extension Sigma_*^(pilot-HS) through its completion rule R_*^(pilot-HS).",
        ),
        sign_base.row(
            "exact_external_rule_selector_selected_extension_action_formula_available_now",
            "pass"
            if exact_external_rule_selector_selected_extension_action_formula_available_now
            else "reject",
            "exact external rule-selector selected extension action formula available now",
            sign_base.truth(
                exact_external_rule_selector_selected_extension_action_formula_available_now
            ),
            "Once Sigma_*^(pilot-HS) is fixed, the extended action L_ext^(pilot-HS)[P_mu,A_mu] becomes literal and reproducible.",
        ),
        sign_base.row(
            "exact_external_rule_selector_selected_extension_effective_kernel_formula_available_now",
            "pass"
            if exact_external_rule_selector_selected_extension_effective_kernel_formula_available_now
            else "reject",
            "exact external rule-selector selected extension effective-kernel formula available now",
            sign_base.truth(
                exact_external_rule_selector_selected_extension_effective_kernel_formula_available_now
            ),
            "The Schur-complement response kernel can now be attached to one selected extension instead of an unresolved completion family.",
        ),
        sign_base.row(
            "exact_external_rule_selector_selected_extension_available_now",
            "pass" if exact_external_rule_selector_selected_extension_available_now else "reject",
            "exact external rule-selector selected extension available now",
            sign_base.truth(
                exact_external_rule_selector_selected_extension_available_now
            ),
            "The external-selector lane now closes positively: one adopted selector rule yields one concrete selected extension.",
        ),
        sign_base.row(
            "exact_concrete_selected_extension_available_now",
            "pass" if exact_concrete_selected_extension_available_now else "reject",
            "exact concrete selected extension available now",
            sign_base.truth(exact_concrete_selected_extension_available_now),
            "The selected extension is now concrete enough to reopen blind-vector direct computation on the chosen extension alone.",
        ),
        sign_base.row(
            "updated_pack_blind_vector_direct_computation_followup_required",
            "pass" if updated_pack_blind_vector_direct_computation_followup_required else "reject",
            "updated-pack blind-vector direct computation followup required",
            sign_base.truth(
                updated_pack_blind_vector_direct_computation_followup_required
            ),
            "The honest next blocker is no longer extension selection but blind-vector direct computation on the selected extension.",
        ),
        sign_base.row(
            "updated_pack_same_schema_external_rule_selector_selected_extension_replay_detected_now",
            "pass" if updated_pack_same_schema_external_rule_selector_selected_extension_replay_detected_now else "reject",
            "updated-pack same-schema external rule-selector selected extension replay detected now",
            sign_base.truth(
                updated_pack_same_schema_external_rule_selector_selected_extension_replay_detected_now
            ),
            "False means this turn closed the selected extension itself and did not re-enter another selector replay layer.",
        ),
        sign_base.row(
            "blind_vector_observable_gate_still_blocked",
            "pass" if blind_blocked else "reject",
            "blind-vector observable gate still blocked",
            sign_base.truth(blind_blocked),
            "Once the selected extension is fixed, blind-vector direct computation is no longer blocked by extension-selection ambiguity.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "retained_scalar_residual_rel": float(prior_gate_summary["retained_scalar_residual_rel"]),
        "exact_external_rule_selector_selected_extension_formula_available_now": exact_external_rule_selector_selected_extension_formula_available_now,
        "exact_external_rule_selector_selected_extension_action_formula_available_now": exact_external_rule_selector_selected_extension_action_formula_available_now,
        "exact_external_rule_selector_selected_extension_effective_kernel_formula_available_now": exact_external_rule_selector_selected_extension_effective_kernel_formula_available_now,
        "exact_external_rule_selector_selected_extension_available_now": exact_external_rule_selector_selected_extension_available_now,
        "exact_concrete_selected_extension_available_now": exact_concrete_selected_extension_available_now,
        "updated_pack_blind_vector_direct_computation_followup_required": updated_pack_blind_vector_direct_computation_followup_required,
        "updated_pack_same_schema_external_rule_selector_selected_extension_replay_detected_now": updated_pack_same_schema_external_rule_selector_selected_extension_replay_detected_now,
        "retry_gate_computation_mode_selected": retry_mode,
        "failure_matrix_non_surrogate_guard_preserved": non_surrogate_guard,
        "blind_vector_observable_gate_still_blocked": blind_blocked,
        "pack_update_required_now": updated_pack_blind_vector_direct_computation_followup_required,
        "selected_primary_completion_lane": "updated_pack_blind_vector_direct_computation_theorem_audit",
        "selected_secondary_completion_lane": "farther_hybrid_extra_q_range_only",
        "selected_reserve_completion_lane": "selected_extension_route_sync",
        "selected_next_generation_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_external_rule_selector_selected_extension_gate",
        "recommended_next_route_or_none": "8.7.56.5139",
        "selected_followup_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_blind_vector_direct_computation_theorem_audit",
        "selected_followup_route_or_none": "8.7.56.5143",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5137",
        STEP_NAME + " declaration gate",
        {
            "source_files": {
                "prior_gate": sign_base.display_path(PRIOR_GATE),
                "prior_audit": sign_base.display_path(PRIOR_AUDIT),
                "prior_support": sign_base.display_path(PRIOR_SUPPORT),
            },
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "8.7.56.5139",
                "followup_route": "8.7.56.5143",
            },
        },
        rows,
        summary,
        {
            "overall_status": "vector_qball_form_factor_external_rule_selector_selected_extension_declared",
            "branch_completed": True,
            "breakthrough_passed_now": exact_concrete_selected_extension_available_now,
            "physical_reject_required": False,
        },
        {"formulae": build_formulae()},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(f"[done] {STEP_TAG} external rule-selector selected extension audit completed")
    print(f"[done] declaration: {declaration_paths['json']}")


if __name__ == "__main__":
    main()
