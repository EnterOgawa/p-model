#!/usr/bin/env python3
"""Generate 8.7.56.5143-.5146 blind-vector direct-computation audit artifacts."""

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
        "8.7.56.5139-5142",
        "updated_pack_external_rule_selector_selected_extension_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
PRIOR_AUDIT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5135-5138",
        "updated_pack_external_rule_selector_selected_extension_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
PHASE3_EVAL = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_unified_closure_"
    "blind_vector_observable_gate_numeric_evaluation_metrics.json"
)
SCALAR_TARGET = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_qball_projection_overlap_coupled_tail_"
    "reconciliation_review_numeric_evaluation_metrics.json"
)

STEP_TAG = "8.7.56.5143-5146"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack blind-vector "
    "direct computation theorem audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_blind_vector_direct_computation_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "external_rule_selector_selected_extension_audited_blind_vector_primary_"
    "hybrid_reserve_secondary_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "blind_vector_direct_computation_contract_derived_numeric_evaluation_"
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


# 関数: blind-vector direct computation で使う式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used in the blind-vector direct-computation audit."""
    return {
        "selected_effective_kernel": (
            "K_eff^(pilot-HS)[Q] := K_AA^(Sigma_*^(pilot-HS))[Q] - "
            "K_xiA^(Sigma_*^(pilot-HS))[Q](K_xixi[Q])^(-1)"
            "K_xiA^(Sigma_*^(pilot-HS))[Q]"
        ),
        "transverse_scalar": (
            "Z_eff^(pilot-HS,T)(q) := (1/2) tr(Pi_T(q) K_eff^(pilot-HS)(q) Pi_T(q))"
        ),
        "blind_form_factor": (
            "F_blind^(pilot-HS)(q) := Z_eff^(pilot-HS,T)(q) / Z_eff^(pilot-HS,T)(0)"
        ),
        "blind_alpha": "alpha_blind^(pilot-HS)(q) := (F_blind^(pilot-HS)(q)^2) / (4 pi)",
        "residual_origin": (
            "delta_alpha_sel^(pilot-HS) := "
            "alpha_blind^(pilot-HS)(q_theory) - alpha_exact(q_theory)"
        ),
    }


# 関数: `.5143-.5146` を実行する。

def main() -> None:
    """Execute the blind-vector direct-computation theorem audit."""
    for path in (PRIOR_GATE, PRIOR_AUDIT, PHASE3_EVAL, SCALAR_TARGET):
        sign_base.require(path)

    prior_gate_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    prior_audit_summary = sign_base.read_json(PRIOR_AUDIT)["summary"]
    phase3_summary = sign_base.read_json(PHASE3_EVAL)["summary"]
    scalar_summary = sign_base.read_json(SCALAR_TARGET)["summary"]

    audit_selected = bool(
        prior_gate_summary["gate_b_updated_pack_blind_vector_direct_computation_promoted_next"]
        and not prior_gate_summary["blind_vector_observable_gate_still_blocked"]
    )
    retry_mode = bool(prior_audit_summary["retry_gate_computation_mode_selected"])
    non_surrogate_guard = bool(
        prior_audit_summary["failure_matrix_non_surrogate_guard_preserved"]
    )
    selected_extension_available = bool(
        prior_audit_summary["exact_concrete_selected_extension_available_now"]
    )
    retained_blind_checkpoint_contract = all(
        key in phase3_summary
        for key in (
            "blind_F_at_zero",
            "blind_F_at_q_theory",
            "blind_F_at_m0",
            "blind_alpha_at_q_theory",
        )
    )
    retained_scalar_reference_target = all(
        key in scalar_summary
        for key in (
            "q_theory_over_m0",
            "alpha_exact_at_q_theory",
            "alpha_exact_relative_error_vs_target",
        )
    )
    exact_blind_vector_selected_extension_transverse_scalar_formula_available_now = bool(
        audit_selected
        and retry_mode
        and non_surrogate_guard
        and selected_extension_available
        and prior_audit_summary[
            "exact_external_rule_selector_selected_extension_effective_kernel_formula_available_now"
        ]
    )
    exact_blind_vector_selected_extension_form_factor_formula_available_now = bool(
        exact_blind_vector_selected_extension_transverse_scalar_formula_available_now
    )
    exact_blind_vector_selected_extension_alpha_formula_available_now = bool(
        exact_blind_vector_selected_extension_form_factor_formula_available_now
    )
    exact_blind_vector_selected_extension_checkpoint_contract_available_now = bool(
        exact_blind_vector_selected_extension_alpha_formula_available_now
        and retained_blind_checkpoint_contract
        and retained_scalar_reference_target
    )
    exact_blind_vector_selected_extension_residual_origin_discriminator_formula_available_now = bool(
        exact_blind_vector_selected_extension_checkpoint_contract_available_now
    )
    direct_blind_vector_computation_primary_admissible_now = bool(
        exact_blind_vector_selected_extension_residual_origin_discriminator_formula_available_now
    )
    updated_pack_blind_vector_numeric_evaluation_followup_required = bool(
        direct_blind_vector_computation_primary_admissible_now
    )
    updated_pack_same_schema_blind_vector_direct_computation_replay_detected_now = False
    blind_vector_observable_gate_still_blocked = bool(
        not direct_blind_vector_computation_primary_admissible_now
    )
    farther_hybrid_continuation_reopen_required_now = False

    rows = [
        sign_base.row(
            "updated_pack_blind_vector_direct_computation_audit_selected",
            "pass" if audit_selected else "reject",
            "updated-pack blind-vector direct computation audit selected",
            sign_base.truth(audit_selected),
            "This branch is worth running only after one concrete selected extension is already fixed and blind-vector direct computation is the live blocker.",
        ),
        sign_base.row(
            "retry_gate_computation_mode_selected",
            "pass" if retry_mode else "reject",
            "retry gate computation mode selected",
            sign_base.truth(retry_mode),
            "The blind-vector lane now moves to computation-side contracts rather than theorem-family descent.",
        ),
        sign_base.row(
            "failure_matrix_non_surrogate_guard_preserved",
            "pass" if non_surrogate_guard else "reject",
            "failure-matrix non-surrogate guard preserved",
            sign_base.truth(non_surrogate_guard),
            "The new computation contract stays honest only if exhausted surrogate and replay families remain closed.",
        ),
        sign_base.row(
            "exact_concrete_selected_extension_available_now",
            "pass" if selected_extension_available else "reject",
            "exact concrete selected extension available now",
            sign_base.truth(selected_extension_available),
            "Blind-vector direct computation can only start after one concrete selected extension Sigma_*^(pilot-HS) is already official.",
        ),
        sign_base.row(
            "retained_blind_vector_checkpoint_contract_available_now",
            "pass" if retained_blind_checkpoint_contract else "reject",
            "retained blind-vector checkpoint contract available now",
            sign_base.truth(retained_blind_checkpoint_contract),
            "The old q=0 / q=q_theory / q=m0 checkpoint contract remains available as a literal computation contract.",
        ),
        sign_base.row(
            "retained_scalar_reference_target_available_now",
            "pass" if retained_scalar_reference_target else "reject",
            "retained scalar reference target available now",
            sign_base.truth(retained_scalar_reference_target),
            "Residual-origin discrimination still uses the retained scalar exact alpha at q_theory and its frozen 1.926% mismatch as the reference target.",
        ),
        sign_base.row(
            "exact_blind_vector_selected_extension_transverse_scalar_formula_available_now",
            "pass"
            if exact_blind_vector_selected_extension_transverse_scalar_formula_available_now
            else "reject",
            "exact blind-vector selected-extension transverse scalar formula available now",
            sign_base.truth(
                exact_blind_vector_selected_extension_transverse_scalar_formula_available_now
            ),
            "The selected extension now yields one literal transverse scalarization of K_eff^(pilot-HS) instead of an unresolved family-valued response.",
        ),
        sign_base.row(
            "exact_blind_vector_selected_extension_form_factor_formula_available_now",
            "pass"
            if exact_blind_vector_selected_extension_form_factor_formula_available_now
            else "reject",
            "exact blind-vector selected-extension form-factor formula available now",
            sign_base.truth(
                exact_blind_vector_selected_extension_form_factor_formula_available_now
            ),
            "The blind-vector form factor can now be defined directly on the selected extension by normalizing the transverse scalar at q = 0.",
        ),
        sign_base.row(
            "exact_blind_vector_selected_extension_alpha_formula_available_now",
            "pass"
            if exact_blind_vector_selected_extension_alpha_formula_available_now
            else "reject",
            "exact blind-vector selected-extension alpha formula available now",
            sign_base.truth(
                exact_blind_vector_selected_extension_alpha_formula_available_now
            ),
            "The selected-extension blind alpha formula is now literal rather than blocked behind selector ambiguity.",
        ),
        sign_base.row(
            "exact_blind_vector_selected_extension_checkpoint_contract_available_now",
            "pass"
            if exact_blind_vector_selected_extension_checkpoint_contract_available_now
            else "reject",
            "exact blind-vector selected-extension checkpoint contract available now",
            sign_base.truth(
                exact_blind_vector_selected_extension_checkpoint_contract_available_now
            ),
            "The selected extension inherits the old q checkpoints and keeps them aligned with the retained scalar target at q_theory.",
        ),
        sign_base.row(
            "exact_blind_vector_selected_extension_residual_origin_discriminator_formula_available_now",
            "pass"
            if exact_blind_vector_selected_extension_residual_origin_discriminator_formula_available_now
            else "reject",
            "exact blind-vector selected-extension residual-origin discriminator formula available now",
            sign_base.truth(
                exact_blind_vector_selected_extension_residual_origin_discriminator_formula_available_now
            ),
            "Residual-origin discrimination can now be written directly as the selected-extension blind alpha shift relative to the retained scalar exact alpha at q_theory.",
        ),
        sign_base.row(
            "direct_blind_vector_computation_primary_admissible_now",
            "pass" if direct_blind_vector_computation_primary_admissible_now else "reject",
            "direct blind-vector computation primary admissible now",
            sign_base.truth(direct_blind_vector_computation_primary_admissible_now),
            "Once the selected extension, transverse scalarization, and checkpoint contract are all literal, the honest next task is numeric evaluation rather than another theorem-family descent.",
        ),
        sign_base.row(
            "updated_pack_blind_vector_numeric_evaluation_followup_required",
            "pass" if updated_pack_blind_vector_numeric_evaluation_followup_required else "reject",
            "updated-pack blind-vector numeric evaluation followup required",
            sign_base.truth(updated_pack_blind_vector_numeric_evaluation_followup_required),
            "The next blocker is no longer definitional but the actual numeric evaluation of the selected-extension blind-vector observable.",
        ),
        sign_base.row(
            "updated_pack_same_schema_blind_vector_direct_computation_replay_detected_now",
            "pass" if updated_pack_same_schema_blind_vector_direct_computation_replay_detected_now else "reject",
            "updated-pack same-schema blind-vector direct computation replay detected now",
            sign_base.truth(updated_pack_same_schema_blind_vector_direct_computation_replay_detected_now),
            "False means this turn did not reopen selector recursion or same-tag replay and instead cut a new computation contract.",
        ),
        sign_base.row(
            "blind_vector_observable_gate_still_blocked",
            "pass" if blind_vector_observable_gate_still_blocked else "reject",
            "blind-vector observable gate still blocked",
            sign_base.truth(blind_vector_observable_gate_still_blocked),
            "Selector ambiguity no longer blocks the blind-vector gate once the selected-extension computation contract is fixed.",
        ),
        sign_base.row(
            "farther_hybrid_continuation_reopen_required_now",
            "pass" if farther_hybrid_continuation_reopen_required_now else "reject",
            "farther hybrid continuation reopen required now",
            sign_base.truth(farther_hybrid_continuation_reopen_required_now),
            "Extra q-range evidence remains reserve-only because the next honest task is still direct computation on the fixed selected extension.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "q_theory_over_m0": float(scalar_summary["q_theory_over_m0"]),
        "retained_scalar_alpha_at_q_theory": float(
            scalar_summary["alpha_exact_at_q_theory"]
        ),
        "retained_scalar_residual_rel": float(
            scalar_summary["alpha_exact_relative_error_vs_target"]
        ),
        "retained_prior_blind_alpha_at_q_theory": float(
            phase3_summary["blind_alpha_at_q_theory"]
        ),
        "exact_blind_vector_selected_extension_transverse_scalar_formula_available_now": exact_blind_vector_selected_extension_transverse_scalar_formula_available_now,
        "exact_blind_vector_selected_extension_form_factor_formula_available_now": exact_blind_vector_selected_extension_form_factor_formula_available_now,
        "exact_blind_vector_selected_extension_alpha_formula_available_now": exact_blind_vector_selected_extension_alpha_formula_available_now,
        "exact_blind_vector_selected_extension_checkpoint_contract_available_now": exact_blind_vector_selected_extension_checkpoint_contract_available_now,
        "exact_blind_vector_selected_extension_residual_origin_discriminator_formula_available_now": exact_blind_vector_selected_extension_residual_origin_discriminator_formula_available_now,
        "direct_blind_vector_computation_primary_admissible_now": direct_blind_vector_computation_primary_admissible_now,
        "updated_pack_blind_vector_numeric_evaluation_followup_required": updated_pack_blind_vector_numeric_evaluation_followup_required,
        "updated_pack_same_schema_blind_vector_direct_computation_replay_detected_now": updated_pack_same_schema_blind_vector_direct_computation_replay_detected_now,
        "retry_gate_computation_mode_selected": retry_mode,
        "failure_matrix_non_surrogate_guard_preserved": non_surrogate_guard,
        "blind_vector_observable_gate_still_blocked": blind_vector_observable_gate_still_blocked,
        "farther_hybrid_continuation_reopen_required_now": farther_hybrid_continuation_reopen_required_now,
        "pack_update_required_now": updated_pack_blind_vector_numeric_evaluation_followup_required,
        "selected_primary_completion_lane": "updated_pack_blind_vector_numeric_evaluation_audit",
        "selected_secondary_completion_lane": "updated_pack_residual_origin_refresh_after_selected_extension_blind_vector",
        "selected_reserve_completion_lane": "farther_hybrid_extra_q_range_only",
        "selected_next_generation_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_blind_vector_direct_computation_gate",
        "recommended_next_route_or_none": "8.7.56.5147",
        "selected_followup_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_blind_vector_numeric_evaluation_audit",
        "selected_followup_route_or_none": "8.7.56.5151",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5145",
        STEP_NAME + " declaration gate",
        {
            "source_files": {
                "prior_gate": sign_base.display_path(PRIOR_GATE),
                "prior_audit": sign_base.display_path(PRIOR_AUDIT),
                "phase3_eval": sign_base.display_path(PHASE3_EVAL),
                "scalar_target_eval": sign_base.display_path(SCALAR_TARGET),
            },
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "8.7.56.5147",
                "followup_route": "8.7.56.5151",
            },
        },
        rows,
        summary,
        {
            "overall_status": "vector_qball_form_factor_blind_vector_direct_computation_contract_declared",
            "branch_completed": True,
            "direct_computation_contract_ready_now": direct_blind_vector_computation_primary_admissible_now,
            "physical_reject_required": False,
        },
        {"formulae": build_formulae()},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(f"[done] {STEP_TAG} blind-vector direct computation audit completed")
    print(f"[done] declaration: {declaration_paths['json']}")


if __name__ == "__main__":
    main()
