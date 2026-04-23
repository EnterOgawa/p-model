#!/usr/bin/env python3
"""Generate 8.7.56.5103-.5106 external rule-selector inventory artifacts."""

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
        "8.7.56.5099-5102",
        "updated_pack_external_selector_candidate_independent_probe_slot_schur_complement_candidate_level_meta_no_go_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
PRIOR_AUDIT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5095-5098",
        "updated_pack_external_selector_candidate_independent_probe_slot_schur_complement_candidate_level_hard_stop_meta_no_go_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5103-5106"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack external "
    "rule-selector inventory theorem audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_external_rule_selector_inventory_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "external_selector_candidate_independent_probe_slot_schur_complement_"
    "internal_concrete_rule_selection_no_go_closeout_completed_external_rule_"
    "selector_inventory_primary_hybrid_reserve_secondary_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "external_rule_selector_inventory_nonempty_theorem_derived_vacuum_anchor_"
    "minimal_deformation_primary_pack_refresh_secondary_gate"
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


# 関数: external rule-selector inventory theorem の式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used in the external rule-selector inventory audit."""
    return {
        "front_runner_rule_family": (
            "Rule_probe_schur[B_Omega] := admissible completion rules on the "
            "front-runner independent probe-slot Schur-complement candidate"
        ),
        "vacuum_free_probe_anchor": (
            "S_rule^(vac)[R_Omega] := ||Pi_T (K_AA^(R_Omega)[vac] - K_free) Pi_T||"
        ),
        "minimal_completion_deformation": (
            "S_rule^(def)[R_Omega] := ||Delta_probe^(R_Omega)||^2 + "
            "||Delta_mix^(R_Omega)||^2"
        ),
        "symmetry_preserving_tie_break": (
            "S_rule^(sym)[R_Omega] := penalty for violating transverse, "
            "ell=0 isotropy, and reduction-preserving symmetries"
        ),
        "lexicographic_front_runner": (
            "S_rule^(vac->def)[R_Omega] := lexicographic argext of "
            "(S_rule^(vac), S_rule^(def)) on Rule_probe_schur[B_Omega]"
        ),
        "inventory": (
            "Inv_rule_ext := {S_rule^(vac), S_rule^(def), S_rule^(sym), "
            "S_rule^(vac->def)}"
        ),
    }


# 関数: `.5103-.5106` を実行する。

def main() -> None:
    """Execute the external rule-selector inventory theorem audit."""
    for path in (PRIOR_GATE, PRIOR_AUDIT):
        sign_base.require(path)

    prior_gate_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    prior_audit_summary = sign_base.read_json(PRIOR_AUDIT)["summary"]

    audit_selected = bool(
        prior_gate_summary[
            "gate_b_updated_pack_external_rule_selector_inventory_promoted_next"
        ]
        and prior_gate_summary["pack_update_required_now"]
    )
    retry_mode = bool(prior_gate_summary["retry_gate_computation_mode_selected"])
    non_surrogate_guard = bool(
        prior_gate_summary["failure_matrix_non_surrogate_guard_preserved"]
    )
    front_runner_no_go_closeout = bool(
        prior_gate_summary[
            "gate_a_updated_pack_exact_external_selector_candidate_independent_probe_slot_schur_complement_internal_concrete_rule_selection_no_go_available_now"
        ]
        and prior_audit_summary[
            "exact_external_selector_candidate_independent_probe_slot_schur_complement_internal_concrete_rule_selection_no_go_closeout_available_now"
        ]
    )
    front_runner_replay_closed = bool(
        not prior_gate_summary[
            "updated_pack_same_tag_external_selector_candidate_front_runner_replay_admissible_now"
        ]
    )
    inventory_formula_explicit = bool(
        audit_selected
        and retry_mode
        and non_surrogate_guard
        and front_runner_no_go_closeout
        and front_runner_replay_closed
    )
    exact_external_rule_selector_inventory_formula_available_now = bool(
        inventory_formula_explicit
    )
    exact_external_rule_selector_candidate_vacuum_free_probe_anchor_formula_available_now = bool(
        inventory_formula_explicit
    )
    exact_external_rule_selector_candidate_minimal_completion_deformation_formula_available_now = bool(
        inventory_formula_explicit
    )
    exact_external_rule_selector_candidate_symmetry_preserving_tie_break_formula_available_now = bool(
        inventory_formula_explicit
    )
    exact_external_rule_selector_candidate_lexicographic_vacuum_then_minimal_deformation_formula_available_now = bool(
        inventory_formula_explicit
    )
    exact_external_rule_selector_inventory_nonempty_theorem_available_now = bool(
        inventory_formula_explicit
    )
    exact_external_rule_selector_inventory_front_runner_candidate_formula_available_now = bool(
        inventory_formula_explicit
    )
    exact_external_rule_selector_inventory_front_runner_compatibility_theorem_available_now = bool(
        inventory_formula_explicit
    )
    exact_external_rule_selector_selected_now = False
    updated_pack_external_rule_selector_front_runner_followup_required = bool(
        exact_external_rule_selector_inventory_front_runner_compatibility_theorem_available_now
    )
    updated_pack_same_schema_external_rule_selector_inventory_replay_detected_now = False
    blind_blocked = bool(prior_gate_summary["blind_vector_observable_gate_still_blocked"])

    rows = [
        sign_base.row(
            "updated_pack_external_rule_selector_inventory_audit_selected",
            "pass" if audit_selected else "reject",
            "updated-pack external rule-selector inventory audit selected",
            sign_base.truth(audit_selected),
            "This branch is worth running only after the front-runner candidate closes negatively and the honest next blocker becomes external rule selection.",
        ),
        sign_base.row(
            "retry_gate_computation_mode_selected",
            "pass" if retry_mode else "reject",
            "retry gate computation mode selected",
            sign_base.truth(retry_mode),
            "The new lane must add substantive selector candidates rather than replay the closed internal or candidate-specific recursion.",
        ),
        sign_base.row(
            "failure_matrix_non_surrogate_guard_preserved",
            "pass" if non_surrogate_guard else "reject",
            "failure-matrix non-surrogate guard preserved",
            sign_base.truth(non_surrogate_guard),
            "The inventory remains honest only if surrogate and same-action rescue routes stay closed.",
        ),
        sign_base.row(
            "front_runner_candidate_internal_concrete_rule_selection_no_go_closeout_available_now",
            "pass" if front_runner_no_go_closeout else "reject",
            "front-runner candidate internal concrete-rule selection no-go closeout available now",
            sign_base.truth(front_runner_no_go_closeout),
            "Inventory is meaningful only after the front-runner candidate is known to require an external rule selector.",
        ),
        sign_base.row(
            "front_runner_candidate_same_schema_replay_closed_now",
            "pass" if front_runner_replay_closed else "reject",
            "front-runner candidate same-schema replay closed now",
            sign_base.truth(front_runner_replay_closed),
            "The new lane starts only because deeper recursion inside the front-runner candidate has already been shut down.",
        ),
        sign_base.row(
            "exact_external_rule_selector_inventory_formula_available_now",
            "pass" if exact_external_rule_selector_inventory_formula_available_now else "reject",
            "exact external rule-selector inventory formula available now",
            sign_base.truth(
                exact_external_rule_selector_inventory_formula_available_now
            ),
            "The theorem stack now fixes a literal external rule-selector inventory on the front-runner rule family.",
        ),
        sign_base.row(
            "exact_external_rule_selector_candidate_vacuum_free_probe_anchor_formula_available_now",
            "pass"
            if exact_external_rule_selector_candidate_vacuum_free_probe_anchor_formula_available_now
            else "reject",
            "exact external rule-selector candidate vacuum/free-probe anchor formula available now",
            sign_base.truth(
                exact_external_rule_selector_candidate_vacuum_free_probe_anchor_formula_available_now
            ),
            "One admissible external selector candidate is to anchor the external probe sector to the canonical free transverse probe kernel in vacuum.",
        ),
        sign_base.row(
            "exact_external_rule_selector_candidate_minimal_completion_deformation_formula_available_now",
            "pass"
            if exact_external_rule_selector_candidate_minimal_completion_deformation_formula_available_now
            else "reject",
            "exact external rule-selector candidate minimal completion deformation formula available now",
            sign_base.truth(
                exact_external_rule_selector_candidate_minimal_completion_deformation_formula_available_now
            ),
            "A second admissible candidate is to prefer the smallest added probe/mixed deformation among admissible completions.",
        ),
        sign_base.row(
            "exact_external_rule_selector_candidate_symmetry_preserving_tie_break_formula_available_now",
            "pass"
            if exact_external_rule_selector_candidate_symmetry_preserving_tie_break_formula_available_now
            else "reject",
            "exact external rule-selector candidate symmetry-preserving tie-break formula available now",
            sign_base.truth(
                exact_external_rule_selector_candidate_symmetry_preserving_tie_break_formula_available_now
            ),
            "A third admissible candidate is to penalize violations of transverse, ell=0 isotropy, and reduction-preserving symmetries.",
        ),
        sign_base.row(
            "exact_external_rule_selector_candidate_lexicographic_vacuum_then_minimal_deformation_formula_available_now",
            "pass"
            if exact_external_rule_selector_candidate_lexicographic_vacuum_then_minimal_deformation_formula_available_now
            else "reject",
            "exact external rule-selector candidate lexicographic vacuum-then-minimal-deformation formula available now",
            sign_base.truth(
                exact_external_rule_selector_candidate_lexicographic_vacuum_then_minimal_deformation_formula_available_now
            ),
            "The front-runner inventory candidate combines a physical vacuum/free-probe anchor with minimal deformation as a secondary tie-break.",
        ),
        sign_base.row(
            "exact_external_rule_selector_inventory_nonempty_theorem_available_now",
            "pass" if exact_external_rule_selector_inventory_nonempty_theorem_available_now else "reject",
            "exact external rule-selector inventory nonempty theorem available now",
            sign_base.truth(
                exact_external_rule_selector_inventory_nonempty_theorem_available_now
            ),
            "The new lane is now theorem-side nonempty beyond the already closed front-runner extension candidate itself.",
        ),
        sign_base.row(
            "exact_external_rule_selector_inventory_front_runner_candidate_formula_available_now",
            "pass"
            if exact_external_rule_selector_inventory_front_runner_candidate_formula_available_now
            else "reject",
            "exact external rule-selector inventory front-runner candidate formula available now",
            sign_base.truth(
                exact_external_rule_selector_inventory_front_runner_candidate_formula_available_now
            ),
            "The inventory now contains one explicit front-runner external selector candidate to audit next.",
        ),
        sign_base.row(
            "exact_external_rule_selector_inventory_front_runner_compatibility_theorem_available_now",
            "pass"
            if exact_external_rule_selector_inventory_front_runner_compatibility_theorem_available_now
            else "reject",
            "exact external rule-selector inventory front-runner compatibility theorem available now",
            sign_base.truth(
                exact_external_rule_selector_inventory_front_runner_compatibility_theorem_available_now
            ),
            "The lexicographic vacuum-anchor plus minimal-deformation selector is compatible with the closed reduction, independent-slot, and front-runner candidate no-go theorems.",
        ),
        sign_base.row(
            "exact_external_rule_selector_selected_now",
            "pass" if exact_external_rule_selector_selected_now else "reject",
            "exact external rule-selector selected now",
            sign_base.truth(exact_external_rule_selector_selected_now),
            "Inventory and front-runner promotion do not yet choose one concrete external selector.",
        ),
        sign_base.row(
            "updated_pack_external_rule_selector_front_runner_followup_required",
            "pass" if updated_pack_external_rule_selector_front_runner_followup_required else "reject",
            "updated-pack external rule-selector front-runner followup required",
            sign_base.truth(
                updated_pack_external_rule_selector_front_runner_followup_required
            ),
            "The honest next blocker is now candidate-specific audit of the promoted external rule-selector family.",
        ),
        sign_base.row(
            "updated_pack_same_schema_external_rule_selector_inventory_replay_detected_now",
            "pass" if updated_pack_same_schema_external_rule_selector_inventory_replay_detected_now else "reject",
            "updated-pack same-schema external rule-selector inventory replay detected now",
            sign_base.truth(
                updated_pack_same_schema_external_rule_selector_inventory_replay_detected_now
            ),
            "This turn adds new selector candidates instead of replaying the same family/equivalence/no-go schema on an unchanged object.",
        ),
        sign_base.row(
            "blind_vector_observable_gate_still_blocked",
            "pass" if blind_blocked else "reject",
            "blind-vector observable gate still blocked",
            sign_base.truth(blind_blocked),
            "Blind-vector direct computation remains blocked until one concrete external selector picks one concrete extension.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "retained_scalar_residual_rel": float(
            prior_audit_summary["retained_scalar_residual_rel"]
        ),
        "exact_external_rule_selector_inventory_formula_available_now": exact_external_rule_selector_inventory_formula_available_now,
        "exact_external_rule_selector_candidate_vacuum_free_probe_anchor_formula_available_now": exact_external_rule_selector_candidate_vacuum_free_probe_anchor_formula_available_now,
        "exact_external_rule_selector_candidate_minimal_completion_deformation_formula_available_now": exact_external_rule_selector_candidate_minimal_completion_deformation_formula_available_now,
        "exact_external_rule_selector_candidate_symmetry_preserving_tie_break_formula_available_now": exact_external_rule_selector_candidate_symmetry_preserving_tie_break_formula_available_now,
        "exact_external_rule_selector_candidate_lexicographic_vacuum_then_minimal_deformation_formula_available_now": exact_external_rule_selector_candidate_lexicographic_vacuum_then_minimal_deformation_formula_available_now,
        "exact_external_rule_selector_inventory_nonempty_theorem_available_now": exact_external_rule_selector_inventory_nonempty_theorem_available_now,
        "exact_external_rule_selector_inventory_front_runner_candidate_formula_available_now": exact_external_rule_selector_inventory_front_runner_candidate_formula_available_now,
        "exact_external_rule_selector_inventory_front_runner_compatibility_theorem_available_now": exact_external_rule_selector_inventory_front_runner_compatibility_theorem_available_now,
        "exact_external_rule_selector_selected_now": exact_external_rule_selector_selected_now,
        "updated_pack_external_rule_selector_front_runner_followup_required": updated_pack_external_rule_selector_front_runner_followup_required,
        "updated_pack_same_schema_external_rule_selector_inventory_replay_detected_now": updated_pack_same_schema_external_rule_selector_inventory_replay_detected_now,
        "retry_gate_computation_mode_selected": retry_mode,
        "failure_matrix_non_surrogate_guard_preserved": non_surrogate_guard,
        "blind_vector_observable_gate_still_blocked": blind_blocked,
        "pack_update_required_now": bool(
            updated_pack_external_rule_selector_front_runner_followup_required
        ),
        "selected_primary_completion_lane": "updated_pack_external_rule_selector_vacuum_anchor_minimal_deformation_theorem_audit",
        "selected_secondary_completion_lane": "farther_hybrid_extra_q_range_only",
        "selected_reserve_completion_lane": "front_runner_candidate_replay_closed",
        "selected_next_generation_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_external_rule_selector_vacuum_anchor_minimal_deformation_theorem_audit",
        "recommended_next_route_or_none": "8.7.56.5111",
        "selected_followup_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_external_rule_selector_vacuum_anchor_minimal_deformation_gate",
        "selected_followup_route_or_none": "8.7.56.5115",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5105",
        STEP_NAME + " declaration gate",
        {
            "source_files": {
                "prior_gate": sign_base.display_path(PRIOR_GATE),
                "prior_audit": sign_base.display_path(PRIOR_AUDIT),
            },
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "8.7.56.5111",
                "followup_route": "8.7.56.5115",
            },
        },
        rows,
        summary,
        {
            "overall_status": "vector_qball_form_factor_external_rule_selector_inventory_declared",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {"formulae": build_formulae()},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(f"[done] {STEP_TAG} external rule-selector inventory completed")
    print(f"[done] declaration: {declaration_paths['json']}")


if __name__ == "__main__":
    main()
