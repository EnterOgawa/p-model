#!/usr/bin/env python3
"""Generate 8.7.56.5071-.5074 Schur-complement candidate theorem artifacts."""

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
PRIOR_AUDIT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5063-5066",
        "updated_pack_external_selector_candidate_inventory_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
PRIOR_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5067-5070",
        "updated_pack_external_selector_candidate_inventory_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5071-5074"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack external "
    "selector candidate independent probe-slot Schur-complement theorem audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_external_selector_candidate_independent_probe_slot_schur_complement_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "external_selector_candidate_inventory_independent_probe_slot_schur_"
    "complement_extension_audited_candidate_selection_primary_hybrid_reserve_"
    "secondary_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "external_selector_candidate_independent_probe_slot_schur_complement_"
    "nonuniqueness_no_go_theorem_derived_concrete_rule_primary_pack_refresh_"
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


# 関数: Schur-complement candidate audit で使う式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used in the Schur-complement candidate audit."""
    return {
        "candidate_family_label": (
            "C_ext^(probe-schur;Omega) := independent probe slot + "
            "Schur-complement response convention with completion data Omega"
        ),
        "admissible_completion_family": (
            "L_probe^(Omega)[A_mu] = L_probe^(0)[A_mu] + Delta_probe^(Omega)[A_mu], "
            "L_mix^(Omega)[P_mu,A_mu] = L_mix^(0)[P_mu,A_mu] + Delta_mix^(Omega)[P_mu,A_mu], "
            "Delta_mix^(Omega)[P_mu,0] = Delta_mix^(Omega)[0,A_mu] = 0"
        ),
        "p_sector_anchor": (
            "K_xixi[Q] = delta^2 S_total^vec / (delta xi_mu delta xi_nu) "
            "|_(P=Q, A=0)"
        ),
        "candidate_kernel_family": (
            "K_xiA^(Omega)[Q] = delta^2 S_ext^(Omega) / (delta xi_mu delta A_nu)|_(Q), "
            "K_AA^(Omega)[Q] = delta^2 S_ext^(Omega) / (delta A_mu delta A_nu)|_(Q)"
        ),
        "effective_kernel_family": (
            "K_eff^(Omega)[Q] = K_AA^(Omega)[Q] - "
            "K_xiA^(Omega)[Q](K_xixi[Q])^(-1)K_xiA^(Omega)[Q]"
        ),
        "nonuniqueness_no_go": (
            "Omega_1 != Omega_2 with the same reduction condition can yield "
            "K_eff^(Omega_1) != K_eff^(Omega_2), so current theory does not "
            "canonically choose one Schur-complement candidate"
        ),
    }


# 関数: `.5071-.5074` を実行する。

def main() -> None:
    """Execute the Schur-complement candidate theorem audit."""
    for path in (PRIOR_AUDIT, PRIOR_GATE):
        sign_base.require(path)

    prior_audit_summary = sign_base.read_json(PRIOR_AUDIT)["summary"]
    prior_gate_summary = sign_base.read_json(PRIOR_GATE)["summary"]

    audit_selected = bool(
        prior_gate_summary[
            "gate_b_updated_pack_external_selector_candidate_independent_probe_slot_schur_complement_promoted_next"
        ]
        and prior_gate_summary["pack_update_required_now"]
    )
    retry_mode = bool(prior_gate_summary["retry_gate_computation_mode_selected"])
    non_surrogate_guard = bool(
        prior_gate_summary["failure_matrix_non_surrogate_guard_preserved"]
    )
    inventory_nonempty_available = bool(
        prior_gate_summary[
            "gate_a_updated_pack_exact_external_selector_candidate_inventory_nonempty_available_now"
        ]
    )
    candidate_formula_available = bool(
        prior_audit_summary[
            "exact_external_selector_candidate_independent_probe_slot_schur_complement_extension_formula_available_now"
        ]
    )
    same_action_no_go_available = bool(
        prior_audit_summary[
            "exact_external_selector_candidate_same_action_three_field_reinterpretation_no_go_theorem_available_now"
        ]
    )
    explicit_extension_template_available = bool(
        prior_audit_summary[
            "exact_beyond_current_written_action_explicit_nonadditive_probe_extension_template_available_now"
        ]
    )
    mixed_kernel_formula_available = bool(
        prior_audit_summary["exact_corrected_mixed_probe_response_kernel_formula_available_now"]
    )
    pure_kernel_formula_available = bool(
        prior_audit_summary["exact_corrected_pure_probe_response_kernel_formula_available_now"]
    )
    kernel_rank_match_available = bool(
        prior_audit_summary["exact_corrected_kernel_rank_match_available_now"]
    )
    candidate_family_explicit = bool(
        audit_selected
        and retry_mode
        and non_surrogate_guard
        and inventory_nonempty_available
        and candidate_formula_available
        and same_action_no_go_available
        and explicit_extension_template_available
        and mixed_kernel_formula_available
        and pure_kernel_formula_available
        and kernel_rank_match_available
    )
    exact_external_selector_candidate_independent_probe_slot_schur_complement_kernel_family_formula_available_now = bool(
        candidate_family_explicit
    )
    exact_external_selector_candidate_independent_probe_slot_schur_complement_p_sector_anchor_formula_available_now = bool(
        candidate_family_explicit
    )
    exact_external_selector_candidate_independent_probe_slot_schur_complement_nonuniqueness_no_go_theorem_available_now = bool(
        candidate_family_explicit
    )
    exact_minimal_external_selector_candidate_independent_probe_slot_schur_complement_concrete_rule_requirement_theorem_available_now = bool(
        candidate_family_explicit
    )
    exact_external_selector_candidate_independent_probe_slot_schur_complement_selected_now = False
    updated_pack_external_selector_candidate_independent_probe_slot_schur_complement_concrete_rule_followup_required = bool(
        exact_external_selector_candidate_independent_probe_slot_schur_complement_nonuniqueness_no_go_theorem_available_now
    )
    updated_pack_same_tag_external_selector_inventory_replay_admissible_now = False
    blind_blocked = bool(prior_gate_summary["blind_vector_observable_gate_still_blocked"])

    rows = [
        sign_base.row(
            "updated_pack_external_selector_candidate_independent_probe_slot_schur_complement_audit_selected",
            "pass" if audit_selected else "reject",
            "updated-pack external selector candidate independent probe-slot Schur-complement audit selected",
            sign_base.truth(audit_selected),
            "This branch is worth running only after the external-selector inventory is nonempty and the Schur-complement candidate is the promoted next blocker.",
        ),
        sign_base.row(
            "retry_gate_computation_mode_selected",
            "pass" if retry_mode else "reject",
            "retry gate computation mode selected",
            sign_base.truth(retry_mode),
            "This turn must add a substantive theorem object on the new candidate lane rather than replay the closed internal lane.",
        ),
        sign_base.row(
            "failure_matrix_non_surrogate_guard_preserved",
            "pass" if non_surrogate_guard else "reject",
            "failure-matrix non-surrogate guard preserved",
            sign_base.truth(non_surrogate_guard),
            "The Schur-complement candidate audit remains honest only if it does not reopen the exhausted surrogate family.",
        ),
        sign_base.row(
            "exact_external_selector_axiom_or_convention_candidate_inventory_nonempty_theorem_available_now",
            "pass" if inventory_nonempty_available else "reject",
            "exact external selector axiom or convention candidate inventory nonempty theorem available now",
            sign_base.truth(inventory_nonempty_available),
            "The candidate-specific audit starts only after the external-selector lane is theorem-side nonempty.",
        ),
        sign_base.row(
            "exact_external_selector_candidate_independent_probe_slot_schur_complement_extension_formula_available_now",
            "pass" if candidate_formula_available else "reject",
            "exact external selector candidate independent probe-slot Schur-complement extension formula available now",
            sign_base.truth(candidate_formula_available),
            "The expert proposal is already admitted as an extension candidate class before this branch asks whether it can be made concrete.",
        ),
        sign_base.row(
            "exact_external_selector_candidate_same_action_three_field_reinterpretation_no_go_theorem_available_now",
            "pass" if same_action_no_go_available else "reject",
            "exact external selector candidate same-action three-field reinterpretation no-go theorem available now",
            sign_base.truth(same_action_no_go_available),
            "The candidate stays honest only if it is treated as an extension rather than as a rescue of the closed same-action lane.",
        ),
        sign_base.row(
            "exact_beyond_current_written_action_explicit_nonadditive_probe_extension_template_available_now",
            "pass" if explicit_extension_template_available else "reject",
            "exact beyond-current-written-action explicit nonadditive probe-extension template available now",
            sign_base.truth(explicit_extension_template_available),
            "The candidate-specific audit is meaningful only because a two-slot extension template already exists theorem-side.",
        ),
        sign_base.row(
            "exact_corrected_mixed_probe_response_kernel_formula_available_now",
            "pass" if mixed_kernel_formula_available else "reject",
            "exact corrected mixed probe-response kernel formula available now",
            sign_base.truth(mixed_kernel_formula_available),
            "The Schur-complement proposal depends on an already closed mixed response slot rather than inventing a new internal response algebra.",
        ),
        sign_base.row(
            "exact_corrected_pure_probe_response_kernel_formula_available_now",
            "pass" if pure_kernel_formula_available else "reject",
            "exact corrected pure probe-response kernel formula available now",
            sign_base.truth(pure_kernel_formula_available),
            "The proposal also depends on an already closed pure probe kernel slot.",
        ),
        sign_base.row(
            "exact_corrected_kernel_rank_match_available_now",
            "pass" if kernel_rank_match_available else "reject",
            "exact corrected kernel rank match available now",
            sign_base.truth(kernel_rank_match_available),
            "The candidate-specific effective kernel is worth auditing only because the observable rank-matched scalar slot has already been theorem-side fixed.",
        ),
        sign_base.row(
            "exact_external_selector_candidate_independent_probe_slot_schur_complement_kernel_family_formula_available_now",
            "pass" if exact_external_selector_candidate_independent_probe_slot_schur_complement_kernel_family_formula_available_now else "reject",
            "exact external selector candidate independent probe-slot Schur-complement kernel family formula available now",
            sign_base.truth(
                exact_external_selector_candidate_independent_probe_slot_schur_complement_kernel_family_formula_available_now
            ),
            "Once the extension is written as L_total[P] + L_probe^(Omega)[A] + L_mix^(Omega)[P,A], the Schur-complement candidate closes only as a family K_eff^(Omega) parameterized by unresolved probe and mixed completions.",
        ),
        sign_base.row(
            "exact_external_selector_candidate_independent_probe_slot_schur_complement_p_sector_anchor_formula_available_now",
            "pass" if exact_external_selector_candidate_independent_probe_slot_schur_complement_p_sector_anchor_formula_available_now else "reject",
            "exact external selector candidate independent probe-slot Schur-complement P-sector anchor formula available now",
            sign_base.truth(
                exact_external_selector_candidate_independent_probe_slot_schur_complement_p_sector_anchor_formula_available_now
            ),
            "The reduction condition keeps the P-only fluctuation Hessian K_xixi anchored to the already closed internal sector, while K_xiA and K_AA remain extension-dependent.",
        ),
        sign_base.row(
            "exact_external_selector_candidate_independent_probe_slot_schur_complement_nonuniqueness_no_go_theorem_available_now",
            "pass" if exact_external_selector_candidate_independent_probe_slot_schur_complement_nonuniqueness_no_go_theorem_available_now else "reject",
            "exact external selector candidate independent probe-slot Schur-complement nonuniqueness no-go theorem available now",
            sign_base.truth(
                exact_external_selector_candidate_independent_probe_slot_schur_complement_nonuniqueness_no_go_theorem_available_now
            ),
            "Different admissible probe/mixed completions can yield different K_eff^(Omega), so the current theorem stack still does not choose one concrete Schur-complement candidate.",
        ),
        sign_base.row(
            "exact_minimal_external_selector_candidate_independent_probe_slot_schur_complement_concrete_rule_requirement_theorem_available_now",
            "pass" if exact_minimal_external_selector_candidate_independent_probe_slot_schur_complement_concrete_rule_requirement_theorem_available_now else "reject",
            "exact minimal external selector candidate independent probe-slot Schur-complement concrete-rule requirement theorem available now",
            sign_base.truth(
                exact_minimal_external_selector_candidate_independent_probe_slot_schur_complement_concrete_rule_requirement_theorem_available_now
            ),
            "The honest next blocker is therefore not generic candidate inventory but a concrete rule that chooses one admissible Schur-complement completion Omega.",
        ),
        sign_base.row(
            "exact_external_selector_candidate_independent_probe_slot_schur_complement_selected_now",
            "pass" if exact_external_selector_candidate_independent_probe_slot_schur_complement_selected_now else "reject",
            "exact external selector candidate independent probe-slot Schur-complement selected now",
            sign_base.truth(
                exact_external_selector_candidate_independent_probe_slot_schur_complement_selected_now
            ),
            "This branch narrows the candidate but does not yet adopt one concrete rule or one concrete extension.",
        ),
        sign_base.row(
            "updated_pack_external_selector_candidate_independent_probe_slot_schur_complement_concrete_rule_followup_required",
            "pass" if updated_pack_external_selector_candidate_independent_probe_slot_schur_complement_concrete_rule_followup_required else "reject",
            "updated-pack external selector candidate independent probe-slot Schur-complement concrete-rule followup required",
            sign_base.truth(
                updated_pack_external_selector_candidate_independent_probe_slot_schur_complement_concrete_rule_followup_required
            ),
            "The honest next blocker is now which concrete rule on the Schur-complement completion family should be admitted.",
        ),
        sign_base.row(
            "updated_pack_same_tag_external_selector_inventory_replay_admissible_now",
            "pass" if updated_pack_same_tag_external_selector_inventory_replay_admissible_now else "reject",
            "updated-pack same-tag external selector inventory replay admissible now",
            sign_base.truth(updated_pack_same_tag_external_selector_inventory_replay_admissible_now),
            "Generic inventory replay is no longer admissible because the candidate-specific blocker has now been localized.",
        ),
        sign_base.row(
            "blind_vector_observable_gate_still_blocked",
            "pass" if blind_blocked else "reject",
            "blind-vector observable gate still blocked",
            sign_base.truth(blind_blocked),
            "Blind-vector direct computation still waits until one concrete rule selects one concrete external extension.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "retained_scalar_residual_rel": float(prior_gate_summary["retained_scalar_residual_rel"]),
        "retry_gate_computation_mode_selected": retry_mode,
        "failure_matrix_non_surrogate_guard_preserved": non_surrogate_guard,
        "exact_external_selector_axiom_or_convention_candidate_inventory_nonempty_theorem_available_now": inventory_nonempty_available,
        "exact_external_selector_candidate_independent_probe_slot_schur_complement_extension_formula_available_now": candidate_formula_available,
        "exact_external_selector_candidate_same_action_three_field_reinterpretation_no_go_theorem_available_now": same_action_no_go_available,
        "exact_beyond_current_written_action_explicit_nonadditive_probe_extension_template_available_now": explicit_extension_template_available,
        "exact_corrected_mixed_probe_response_kernel_formula_available_now": mixed_kernel_formula_available,
        "exact_corrected_pure_probe_response_kernel_formula_available_now": pure_kernel_formula_available,
        "exact_corrected_kernel_rank_match_available_now": kernel_rank_match_available,
        "exact_external_selector_candidate_independent_probe_slot_schur_complement_kernel_family_formula_available_now": exact_external_selector_candidate_independent_probe_slot_schur_complement_kernel_family_formula_available_now,
        "exact_external_selector_candidate_independent_probe_slot_schur_complement_p_sector_anchor_formula_available_now": exact_external_selector_candidate_independent_probe_slot_schur_complement_p_sector_anchor_formula_available_now,
        "exact_external_selector_candidate_independent_probe_slot_schur_complement_nonuniqueness_no_go_theorem_available_now": exact_external_selector_candidate_independent_probe_slot_schur_complement_nonuniqueness_no_go_theorem_available_now,
        "exact_minimal_external_selector_candidate_independent_probe_slot_schur_complement_concrete_rule_requirement_theorem_available_now": exact_minimal_external_selector_candidate_independent_probe_slot_schur_complement_concrete_rule_requirement_theorem_available_now,
        "exact_external_selector_candidate_independent_probe_slot_schur_complement_selected_now": exact_external_selector_candidate_independent_probe_slot_schur_complement_selected_now,
        "updated_pack_external_selector_candidate_independent_probe_slot_schur_complement_concrete_rule_followup_required": updated_pack_external_selector_candidate_independent_probe_slot_schur_complement_concrete_rule_followup_required,
        "updated_pack_same_tag_external_selector_inventory_replay_admissible_now": updated_pack_same_tag_external_selector_inventory_replay_admissible_now,
        "blind_vector_observable_gate_still_blocked": blind_blocked,
        "pack_update_required_now": updated_pack_external_selector_candidate_independent_probe_slot_schur_complement_concrete_rule_followup_required,
        "selected_primary_completion_lane": "updated_pack_external_selector_candidate_independent_probe_slot_schur_complement_concrete_rule_theorem_audit",
        "selected_secondary_completion_lane": "farther_hybrid_extra_q_range_only",
        "selected_reserve_completion_lane": "same_tag_external_selector_inventory_replay_closed",
        "selected_next_generation_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_external_selector_candidate_independent_probe_slot_schur_complement_concrete_rule_theorem_audit",
        "recommended_next_route_or_none": "8.7.56.5075",
        "selected_followup_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_external_selector_candidate_independent_probe_slot_schur_complement_concrete_rule_gate",
        "selected_followup_route_or_none": "8.7.56.5079",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5073",
        STEP_NAME + " declaration gate",
        {
            "source_files": {
                "prior_audit": sign_base.display_path(PRIOR_AUDIT),
                "prior_gate": sign_base.display_path(PRIOR_GATE),
            },
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "8.7.56.5075",
                "followup_route": "8.7.56.5079",
            },
        },
        rows,
        summary,
        {
            "overall_status": "vector_qball_form_factor_external_selector_candidate_independent_probe_slot_schur_complement_declared",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {"formulae": build_formulae()},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(
        f"[done] {STEP_TAG} external selector candidate Schur-complement theorem completed"
    )
    print(f"[done] declaration: {declaration_paths['json']}")


if __name__ == "__main__":
    main()
