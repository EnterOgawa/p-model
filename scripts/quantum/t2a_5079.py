#!/usr/bin/env python3
"""Generate 8.7.56.5079-.5082 Schur-complement concrete-rule theorem artifacts."""

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
        "8.7.56.5071-5074",
        "updated_pack_external_selector_candidate_independent_probe_slot_schur_complement_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
PRIOR_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5075-5078",
        "updated_pack_external_selector_candidate_independent_probe_slot_schur_complement_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5079-5082"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack external "
    "selector candidate independent probe-slot Schur-complement concrete-rule "
    "theorem audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_external_selector_candidate_independent_probe_slot_schur_complement_concrete_rule_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "external_selector_candidate_independent_probe_slot_schur_complement_"
    "nonuniqueness_no_go_theorem_audited_concrete_rule_primary_hybrid_reserve_"
    "secondary_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "external_selector_candidate_independent_probe_slot_schur_complement_"
    "concrete_rule_no_go_theorem_derived_selector_primary_pack_refresh_"
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


# 関数: Schur-complement concrete-rule theorem で使う式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used in the Schur-complement concrete-rule audit."""
    return {
        "completion_family": (
            "Comp_probe_schur := { Omega | L_ext^(Omega)[P_mu,A_mu] = "
            "L_total^vec[P_mu] + L_probe^(Omega)[A_mu] + L_mix^(Omega)[P_mu,A_mu], "
            "L_ext^(Omega)[P_mu,0] = L_total^vec[P_mu] }"
        ),
        "finite_anchor_completion_data": "B_Omega := {(Omega_i, r_i)}_(i=1)^N",
        "concrete_rule_family": (
            "Rule_probe_schur[B_Omega] := { R_Omega | "
            "R_Omega : Comp_probe_schur -> R, R_Omega[Omega_i] = r_i for all i }"
        ),
        "selector_family": (
            "Sel_probe_schur[B_Omega] := { J_Omega | "
            "J_Omega : Rule_probe_schur[B_Omega] -> R }"
        ),
        "candidate_formula": (
            "R_*^(B_Omega;J_Omega) := "
            "argext_(R_Omega in Rule_probe_schur[B_Omega]) J_Omega[R_Omega]"
        ),
        "concrete_rule_no_go": (
            "current theorem stack fixes only an admissible family of concrete "
            "rules on Omega and not one canonical rule R_*"
        ),
    }


# 関数: `.5079-.5082` を実行する。

def main() -> None:
    """Execute the Schur-complement concrete-rule theorem audit."""
    for path in (PRIOR_AUDIT, PRIOR_GATE):
        sign_base.require(path)

    prior_audit_summary = sign_base.read_json(PRIOR_AUDIT)["summary"]
    prior_gate_summary = sign_base.read_json(PRIOR_GATE)["summary"]

    audit_selected = bool(
        prior_gate_summary[
            "gate_b_updated_pack_external_selector_candidate_independent_probe_slot_schur_complement_concrete_rule_promoted_next"
        ]
        and prior_gate_summary["pack_update_required_now"]
    )
    retry_mode = bool(prior_gate_summary["retry_gate_computation_mode_selected"])
    non_surrogate_guard = bool(
        prior_gate_summary["failure_matrix_non_surrogate_guard_preserved"]
    )
    candidate_nonuniqueness_no_go_available = bool(
        prior_gate_summary[
            "gate_a_updated_pack_exact_external_selector_candidate_independent_probe_slot_schur_complement_nonuniqueness_no_go_available_now"
        ]
    )
    kernel_family_available = bool(
        prior_audit_summary[
            "exact_external_selector_candidate_independent_probe_slot_schur_complement_kernel_family_formula_available_now"
        ]
    )
    p_sector_anchor_available = bool(
        prior_audit_summary[
            "exact_external_selector_candidate_independent_probe_slot_schur_complement_p_sector_anchor_formula_available_now"
        ]
    )
    concrete_rule_requirement_available = bool(
        prior_audit_summary[
            "exact_minimal_external_selector_candidate_independent_probe_slot_schur_complement_concrete_rule_requirement_theorem_available_now"
        ]
    )
    candidate_selected_now = bool(
        prior_audit_summary[
            "exact_external_selector_candidate_independent_probe_slot_schur_complement_selected_now"
        ]
    )
    concrete_rule_formula_explicit = bool(
        audit_selected
        and retry_mode
        and non_surrogate_guard
        and candidate_nonuniqueness_no_go_available
        and kernel_family_available
        and p_sector_anchor_available
        and concrete_rule_requirement_available
        and not candidate_selected_now
    )
    exact_external_selector_candidate_independent_probe_slot_schur_complement_concrete_rule_family_formula_available_now = bool(
        concrete_rule_formula_explicit
    )
    exact_external_selector_candidate_independent_probe_slot_schur_complement_concrete_rule_candidate_formula_available_now = bool(
        concrete_rule_formula_explicit
    )
    exact_external_selector_candidate_independent_probe_slot_schur_complement_concrete_rule_no_go_theorem_available_now = bool(
        concrete_rule_formula_explicit
    )
    exact_minimal_external_selector_candidate_independent_probe_slot_schur_complement_selector_requirement_theorem_available_now = bool(
        concrete_rule_formula_explicit
    )
    exact_external_selector_candidate_independent_probe_slot_schur_complement_concrete_rule_available_now = False
    updated_pack_external_selector_candidate_independent_probe_slot_schur_complement_selector_followup_required = bool(
        exact_minimal_external_selector_candidate_independent_probe_slot_schur_complement_selector_requirement_theorem_available_now
    )
    updated_pack_same_tag_external_selector_candidate_replay_admissible_now = False
    blind_blocked = bool(prior_gate_summary["blind_vector_observable_gate_still_blocked"])

    rows = [
        sign_base.row(
            "updated_pack_external_selector_candidate_independent_probe_slot_schur_complement_concrete_rule_audit_selected",
            "pass" if audit_selected else "reject",
            "updated-pack external selector candidate independent probe-slot Schur-complement concrete-rule audit selected",
            sign_base.truth(audit_selected),
            "This branch is worth running only after candidate-specific nonuniqueness closes and concrete-rule choice becomes the honest next blocker.",
        ),
        sign_base.row(
            "retry_gate_computation_mode_selected",
            "pass" if retry_mode else "reject",
            "retry gate computation mode selected",
            sign_base.truth(retry_mode),
            "This turn must narrow the front-runner candidate theorem-side rather than replay generic inventory or the closed internal lane.",
        ),
        sign_base.row(
            "failure_matrix_non_surrogate_guard_preserved",
            "pass" if non_surrogate_guard else "reject",
            "failure-matrix non-surrogate guard preserved",
            sign_base.truth(non_surrogate_guard),
            "The concrete-rule audit stays honest only if it does not reopen exhausted surrogate or same-action routes.",
        ),
        sign_base.row(
            "gate_a_updated_pack_exact_external_selector_candidate_independent_probe_slot_schur_complement_nonuniqueness_no_go_available_now",
            "pass" if candidate_nonuniqueness_no_go_available else "reject",
            "gate A exact external selector candidate independent probe-slot Schur-complement nonuniqueness no-go available now",
            sign_base.truth(candidate_nonuniqueness_no_go_available),
            "The concrete-rule theorem starts only after the candidate-specific nonuniqueness theorem is already official.",
        ),
        sign_base.row(
            "exact_external_selector_candidate_independent_probe_slot_schur_complement_kernel_family_formula_available_now",
            "pass" if kernel_family_available else "reject",
            "exact external selector candidate independent probe-slot Schur-complement kernel family formula available now",
            sign_base.truth(kernel_family_available),
            "The concrete-rule theorem uses the already closed family K_eff^(Omega) as its starting object.",
        ),
        sign_base.row(
            "exact_external_selector_candidate_independent_probe_slot_schur_complement_p_sector_anchor_formula_available_now",
            "pass" if p_sector_anchor_available else "reject",
            "exact external selector candidate independent probe-slot Schur-complement P-sector anchor formula available now",
            sign_base.truth(p_sector_anchor_available),
            "The reduction-anchored P sector remains fixed while the rule acts only on unresolved completion freedom.",
        ),
        sign_base.row(
            "exact_minimal_external_selector_candidate_independent_probe_slot_schur_complement_concrete_rule_requirement_theorem_available_now",
            "pass" if concrete_rule_requirement_available else "reject",
            "exact minimal external selector candidate independent probe-slot Schur-complement concrete-rule requirement theorem available now",
            sign_base.truth(concrete_rule_requirement_available),
            "The prior branch already fixed that concrete-rule choice is the honest next blocker.",
        ),
        sign_base.row(
            "concrete_rule_formula_explicit",
            "pass" if concrete_rule_formula_explicit else "reject",
            "concrete-rule formula explicit",
            sign_base.truth(concrete_rule_formula_explicit),
            "Finite completion anchors B_Omega and the induced concrete-rule family can now be written literally once an auxiliary selector J_Omega is introduced.",
        ),
        sign_base.row(
            "exact_external_selector_candidate_independent_probe_slot_schur_complement_concrete_rule_family_formula_available_now",
            "pass"
            if exact_external_selector_candidate_independent_probe_slot_schur_complement_concrete_rule_family_formula_available_now
            else "reject",
            "exact external selector candidate independent probe-slot Schur-complement concrete-rule family formula available now",
            sign_base.truth(
                exact_external_selector_candidate_independent_probe_slot_schur_complement_concrete_rule_family_formula_available_now
            ),
            "The theorem stack now fixes the admissible family of concrete rules acting on the unresolved completion data Omega.",
        ),
        sign_base.row(
            "exact_external_selector_candidate_independent_probe_slot_schur_complement_concrete_rule_candidate_formula_available_now",
            "pass"
            if exact_external_selector_candidate_independent_probe_slot_schur_complement_concrete_rule_candidate_formula_available_now
            else "reject",
            "exact external selector candidate independent probe-slot Schur-complement concrete-rule candidate formula available now",
            sign_base.truth(
                exact_external_selector_candidate_independent_probe_slot_schur_complement_concrete_rule_candidate_formula_available_now
            ),
            "With an auxiliary selector J_Omega, the theorem stack can now state the induced rule candidate R_*^(B_Omega;J_Omega) explicitly.",
        ),
        sign_base.row(
            "exact_external_selector_candidate_independent_probe_slot_schur_complement_concrete_rule_no_go_theorem_available_now",
            "pass"
            if exact_external_selector_candidate_independent_probe_slot_schur_complement_concrete_rule_no_go_theorem_available_now
            else "reject",
            "exact external selector candidate independent probe-slot Schur-complement concrete-rule no-go theorem available now",
            sign_base.truth(
                exact_external_selector_candidate_independent_probe_slot_schur_complement_concrete_rule_no_go_theorem_available_now
            ),
            "Current theory still does not choose one canonical concrete rule; it fixes only the family generated by J_Omega over admissible completions.",
        ),
        sign_base.row(
            "exact_minimal_external_selector_candidate_independent_probe_slot_schur_complement_selector_requirement_theorem_available_now",
            "pass"
            if exact_minimal_external_selector_candidate_independent_probe_slot_schur_complement_selector_requirement_theorem_available_now
            else "reject",
            "exact minimal external selector candidate independent probe-slot Schur-complement selector requirement theorem available now",
            sign_base.truth(
                exact_minimal_external_selector_candidate_independent_probe_slot_schur_complement_selector_requirement_theorem_available_now
            ),
            "The honest next blocker is therefore not another completion-family replay but which selector on the concrete-rule family can canonically choose one rule.",
        ),
        sign_base.row(
            "exact_external_selector_candidate_independent_probe_slot_schur_complement_concrete_rule_available_now",
            "pass"
            if exact_external_selector_candidate_independent_probe_slot_schur_complement_concrete_rule_available_now
            else "reject",
            "exact external selector candidate independent probe-slot Schur-complement concrete-rule available now",
            sign_base.truth(
                exact_external_selector_candidate_independent_probe_slot_schur_complement_concrete_rule_available_now
            ),
            "This branch narrows the candidate to a concrete-rule family but does not yet choose one canonical rule.",
        ),
        sign_base.row(
            "updated_pack_external_selector_candidate_independent_probe_slot_schur_complement_selector_followup_required",
            "pass"
            if updated_pack_external_selector_candidate_independent_probe_slot_schur_complement_selector_followup_required
            else "reject",
            "updated-pack external selector candidate independent probe-slot Schur-complement selector followup required",
            sign_base.truth(
                updated_pack_external_selector_candidate_independent_probe_slot_schur_complement_selector_followup_required
            ),
            "The next honest blocker is the selector on the concrete-rule family, not same-tag replay.",
        ),
        sign_base.row(
            "updated_pack_same_tag_external_selector_candidate_replay_admissible_now",
            "pass"
            if updated_pack_same_tag_external_selector_candidate_replay_admissible_now
            else "reject",
            "updated-pack same-tag external selector candidate replay admissible now",
            sign_base.truth(
                updated_pack_same_tag_external_selector_candidate_replay_admissible_now
            ),
            "Replaying the candidate-family theorem without a concrete-rule selector would add no new substantive closeout.",
        ),
        sign_base.row(
            "blind_vector_observable_gate_still_blocked",
            "pass" if blind_blocked else "reject",
            "blind-vector observable gate still blocked",
            sign_base.truth(blind_blocked),
            "Blind-vector direct computation still waits on one concrete external selector and one concrete extension.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "retained_scalar_residual_rel": float(prior_audit_summary["retained_scalar_residual_rel"]),
        "retry_gate_computation_mode_selected": retry_mode,
        "failure_matrix_non_surrogate_guard_preserved": non_surrogate_guard,
        "exact_external_selector_candidate_independent_probe_slot_schur_complement_kernel_family_formula_available_now": kernel_family_available,
        "exact_external_selector_candidate_independent_probe_slot_schur_complement_p_sector_anchor_formula_available_now": p_sector_anchor_available,
        "exact_external_selector_candidate_independent_probe_slot_schur_complement_nonuniqueness_no_go_theorem_available_now": candidate_nonuniqueness_no_go_available,
        "exact_external_selector_candidate_independent_probe_slot_schur_complement_concrete_rule_family_formula_available_now": exact_external_selector_candidate_independent_probe_slot_schur_complement_concrete_rule_family_formula_available_now,
        "exact_external_selector_candidate_independent_probe_slot_schur_complement_concrete_rule_candidate_formula_available_now": exact_external_selector_candidate_independent_probe_slot_schur_complement_concrete_rule_candidate_formula_available_now,
        "exact_external_selector_candidate_independent_probe_slot_schur_complement_concrete_rule_no_go_theorem_available_now": exact_external_selector_candidate_independent_probe_slot_schur_complement_concrete_rule_no_go_theorem_available_now,
        "exact_minimal_external_selector_candidate_independent_probe_slot_schur_complement_selector_requirement_theorem_available_now": exact_minimal_external_selector_candidate_independent_probe_slot_schur_complement_selector_requirement_theorem_available_now,
        "exact_external_selector_candidate_independent_probe_slot_schur_complement_concrete_rule_available_now": exact_external_selector_candidate_independent_probe_slot_schur_complement_concrete_rule_available_now,
        "updated_pack_external_selector_candidate_independent_probe_slot_schur_complement_selector_followup_required": updated_pack_external_selector_candidate_independent_probe_slot_schur_complement_selector_followup_required,
        "updated_pack_same_tag_external_selector_candidate_replay_admissible_now": updated_pack_same_tag_external_selector_candidate_replay_admissible_now,
        "blind_vector_observable_gate_still_blocked": blind_blocked,
        "selected_primary_completion_lane": "updated_pack_external_selector_candidate_independent_probe_slot_schur_complement_selector_theorem_audit",
        "selected_secondary_completion_lane": "farther_hybrid_extra_q_range_only",
        "selected_reserve_completion_lane": "same_tag_external_selector_candidate_replay_closed",
        "selected_next_generation_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_external_selector_candidate_independent_probe_slot_schur_complement_selector_theorem_audit",
        "recommended_next_route_or_none": "8.7.56.5087",
        "selected_followup_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_external_selector_candidate_independent_probe_slot_schur_complement_selector_gate",
        "selected_followup_route_or_none": "8.7.56.5091",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5081",
        STEP_NAME + " declaration gate",
        {
            "source_files": {
                "prior_audit": sign_base.display_path(PRIOR_AUDIT),
                "prior_gate": sign_base.display_path(PRIOR_GATE),
            },
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "8.7.56.5087",
                "followup_route": "8.7.56.5091",
            },
        },
        rows,
        summary,
        {
            "overall_status": "vector_qball_form_factor_external_selector_candidate_independent_probe_slot_schur_complement_concrete_rule_theorem_audited",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {"formulae": build_formulae()},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(f"[done] {STEP_TAG} Schur-complement concrete-rule theorem audit completed")
    print(f"[done] declaration: {declaration_paths['json']}")


if __name__ == "__main__":
    main()
