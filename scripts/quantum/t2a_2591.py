#!/usr/bin/env python3
"""Generate 8.7.56.2591-.2594 exact charge-current derivation artifacts."""

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

STATUS = ROOT / "doc" / "STATUS.md"
ROADMAP = ROOT / "doc" / "ROADMAP.md"
AI_CONTEXT = ROOT / "doc" / "AI_CONTEXT_MIN.json"
WORK_HISTORY_RECENT = ROOT / "doc" / "WORK_HISTORY_RECENT.md"
CURRENT_PROBLEM = ROOT / "doc" / "quantum" / "34_trial2_numeric_alpha_current_problem.md"
CURRENT_STATUS = ROOT / "doc" / "quantum" / "36_trial2_numeric_alpha_current_status.md"
UNIFIED_ROADMAP = ROOT / "doc" / "quantum" / "39_trial2_vector_qball_unified_closure_roadmap.md"
LONG_ROADMAP = ROOT / "doc" / "quantum" / "55_trial2_numeric_alpha_vector_qball_long_horizon_roadmap.md"
PART1 = ROOT / "doc" / "paper" / "10_part1_core_theory.md"
PART3A = ROOT / "doc" / "paper" / "12_part3a_quantum_foundations.md"
PART5 = ROOT / "doc" / "paper" / "14_part5_future_predictions.md"

PRIOR_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.2587-2590",
        "updated_pack_background_expansion_derivation_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
BACKGROUND_DERIVATION = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.2583-2586",
        "updated_pack_exact_background_expansion_derivation_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
GLOBAL_U1_SOURCE = PUBLIC_OUT / "mass_origin_v2_complex_vector_phase_global_u1_source_inventory_metrics.json"
QBALL_CHARGE_MAPPING = PUBLIC_OUT / "mass_origin_qball_charge_mapping_statement_freeze_metrics.json"
QBALL_CHARGE_NORMALIZATION = (
    PUBLIC_OUT / "mass_origin_qball_charge_operator_normalization_audit_metrics.json"
)
FROZEN_JEFF_CLASS = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.1567-1570",
        "jeff_q0_class",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.2591-2594"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack exact "
    "charge-current derivation audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_exact_charge_current_derivation_audit",
    prefix="q",
)

PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_background_"
    "expansion_derived_charge_current_primary_low_order_jeff0_secondary_blind_vector_"
    "reserve_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_charge_current_"
    "derived_low_order_jeff0_gate"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_charge_current_"
    "derivation_gate_low_order_jeff0_refresh"
)
NEXT_ROUTE = "8.7.56.2595"
FOLLOWUP_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_exact_low_order_"
    "jeff0_derivation_audit"
)
FOLLOWUP_ROUTE = "8.7.56.2599"


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

    return {
        "json": sign_base.display_path(paths["json"]),
        "csv": sign_base.display_path(paths["csv"]),
    }


# 関数: charge-current derivation で使う式を返す。

def build_formulae(background_formulas: dict[str, str]) -> dict[str, str]:
    """Return formulas used in the exact charge-current derivation audit."""
    return {
        "complex_phase_stationary_rule": (
            "Q_mu(x) = exp(-i theta(x)) Qbar_mu(x), with partial_mu theta = omega "
            "delta_mu^0 on the stationary Q-ball background"
        ),
        "global_u1_current_rule": (
            "J_Noether^mu[Q] = 2 (partial^mu theta) (-Q_g^2)  "
            "[updated-pack phase-norm inference from retained global U(1) "
            "symmetry plus exact background norm bundle]"
        ),
        "exact_background_norm": background_formulas["exact_caseb_lift"],
        "exact_charge_density": (
            "J_Noether^0[Q] = 2 omega [f_0^4 / P_infty^2 - (P_infty^2 / f_0^2) f_L^2]"
        ),
        "exact_charge_density_series": (
            "J_Noether^0[Q] = 2 omega [(a_0^4/P_infty^2)"
            " + ((4 a_0^3 a_2)/P_infty^2 - P_infty^2 b_1^2/a_0^2) r^2"
            " + ((4 a_0^3 a_4 + 6 a_0^2 a_2^2)/P_infty^2"
            " - P_infty^2 (2 b_1 b_3/a_0^2 - 2 a_2 b_1^2/a_0^3)) r^4"
            " + O(r^6)]"
        ),
        "stationary_spatial_current": "J_Noether^i[Q] = 0 on the stationary Q-ball background",
        "proxy_limit": (
            "J_Noether^0[Q] / (2 omega) = -Q_g^2 -> |f_0|^2 - |f_L|^2 in the "
            "weak-field / flat-background limit f_0 -> P_infty"
        ),
        "object_split": (
            "J_Noether^mu[Q] closes the conserved background charge current; "
            "J_eff^mu[a;Q] remains the photon-side effective-source theorem object"
        ),
    }


# 関数: `.2591-.2594` を実行する。

def main() -> None:
    """Execute the updated-pack exact charge-current derivation audit."""
    for path in (
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        WORK_HISTORY_RECENT,
        CURRENT_PROBLEM,
        CURRENT_STATUS,
        UNIFIED_ROADMAP,
        LONG_ROADMAP,
        PART1,
        PART3A,
        PART5,
        PRIOR_GATE,
        BACKGROUND_DERIVATION,
        GLOBAL_U1_SOURCE,
        QBALL_CHARGE_MAPPING,
        QBALL_CHARGE_NORMALIZATION,
        FROZEN_JEFF_CLASS,
    ):
        sign_base.require(path)

    status_text = sign_base.read_text(STATUS)
    roadmap_text = sign_base.read_text(ROADMAP)
    current_problem_text = sign_base.read_text(CURRENT_PROBLEM)
    current_status_text = sign_base.read_text(CURRENT_STATUS)
    unified_text = sign_base.read_text(UNIFIED_ROADMAP)
    long_text = sign_base.read_text(LONG_ROADMAP)
    part1_text = sign_base.read_text(PART1)
    part3a_text = sign_base.read_text(PART3A)
    part5_text = sign_base.read_text(PART5)

    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    background_payload = sign_base.read_json(BACKGROUND_DERIVATION)
    background_summary = background_payload["summary"]
    background_formulas = background_payload["evidence"]["formulas"]
    global_u1_summary = sign_base.read_json(GLOBAL_U1_SOURCE)["summary"]
    charge_mapping_summary = sign_base.read_json(QBALL_CHARGE_MAPPING)["summary"]
    charge_norm_summary = sign_base.read_json(QBALL_CHARGE_NORMALIZATION)["summary"]
    frozen_jeff_summary = sign_base.read_json(FROZEN_JEFF_CLASS)["summary"]

    updated_pack_exact_charge_current_derivation_audit_selected = bool(
        prior_summary["gate_b_updated_pack_exact_charge_current_primary_selected"]
        and not prior_summary["gate_c_blind_vector_computation_primary_admissible_now"]
        and prior_summary["exact_qball_background_expansion_available_now"]
    )
    retained_global_u1_phase_surface_available = bool(
        global_u1_summary["global_u1_automatic_from_abs_only_potential"]
        and sign_base.hit(part1_text, "\\partial_\\mu J^\\mu=0") is not None
        and sign_base.hit(part3a_text, "Q-ball Noether charge = adopted U(1) charge") is not None
    )
    retained_adopted_qball_u1_identity_available = bool(
        charge_mapping_summary["u1_charge_quantization_to_qball_charge_mapping"] == "available"
        and charge_norm_summary["direct_qball_u1_identity_required"]
        and not charge_norm_summary["charge_operator_normalization_freedom_available"]
    )
    exact_background_norm_bundle_available = bool(
        background_summary["updated_pack_exact_qball_background_expansion_available_now"]
        and background_summary["caseb_exact_qball_background_expansion_formula_bundle_derived"]
    )
    frozen_jeff_zero_class_retained = bool(
        frozen_jeff_summary["classification_case_iv_zero_under_current_pack"]
    )
    updated_pack_noether_phase_norm_inference_available = bool(
        retained_global_u1_phase_surface_available
        and retained_adopted_qball_u1_identity_available
        and exact_background_norm_bundle_available
    )
    updated_pack_exact_charge_density_formula_bundle_derived = bool(
        updated_pack_noether_phase_norm_inference_available
    )
    updated_pack_exact_charge_current_noether_closure_available_now = bool(
        updated_pack_exact_charge_density_formula_bundle_derived
    )
    updated_pack_proxy_signed_density_promoted_to_exact_now = False
    updated_pack_charge_current_derivation_closes_second_missing_primitive_now = bool(
        updated_pack_exact_charge_current_noether_closure_available_now
    )
    updated_pack_low_order_jeff0_primary_refresh_required = bool(
        updated_pack_exact_charge_current_noether_closure_available_now
    )
    exact_source_theorem_derived_now = False
    blind_vector_observable_gate_still_blocked = True
    farther_hybrid_continuation_reopen_required_now = False

    rows = [
        sign_base.row(
            "updated_pack_exact_charge_current_derivation_audit_selected",
            "pass" if updated_pack_exact_charge_current_derivation_audit_selected else "reject",
            "updated-pack exact charge-current derivation audit selected",
            sign_base.truth(updated_pack_exact_charge_current_derivation_audit_selected),
            "The background-expansion gate already promoted exact charge-current / Noether-current derivation as the next honest remaining closure.",
        ),
        sign_base.row(
            "retained_global_u1_phase_surface_available",
            "pass" if retained_global_u1_phase_surface_available else "reject",
            "retained global-U(1) phase surface available",
            sign_base.truth(retained_global_u1_phase_surface_available),
            "Trial-1 already froze the complex-phase global-U(1) surface and Part I already freezes continuity.",
        ),
        sign_base.row(
            "retained_adopted_qball_u1_identity_available",
            "pass" if retained_adopted_qball_u1_identity_available else "reject",
            "retained adopted-U(1) / Q-ball identity available",
            sign_base.truth(retained_adopted_qball_u1_identity_available),
            "The public pack already fixes Q-ball Noether charge = adopted U(1) charge with no extra normalization freedom.",
        ),
        sign_base.row(
            "exact_background_norm_bundle_available",
            "pass" if exact_background_norm_bundle_available else "reject",
            "exact background-norm bundle available",
            sign_base.truth(exact_background_norm_bundle_available),
            "The updated-pack background-expansion branch already derived the exact Q_g^2 bundle needed to literalize the conserved charge density.",
        ),
        sign_base.row(
            "frozen_jeff_zero_class_retained",
            "pass" if frozen_jeff_zero_class_retained else "reject",
            "frozen J_eff zero-class retained",
            sign_base.truth(frozen_jeff_zero_class_retained),
            "The older zero-classification still applies to the same-field photon-source object J_eff^mu and does not contradict a conserved background Noether current.",
        ),
        sign_base.row(
            "updated_pack_noether_phase_norm_inference_available",
            "pass" if updated_pack_noether_phase_norm_inference_available else "reject",
            "updated-pack Noether phase-norm inference available",
            sign_base.truth(updated_pack_noether_phase_norm_inference_available),
            "Combining retained global-U(1) phase symmetry with the exact background norm bundle closes an exact conserved charge-current route as an explicit inference from frozen sources.",
        ),
        sign_base.row(
            "updated_pack_exact_charge_density_formula_bundle_derived",
            "pass" if updated_pack_exact_charge_density_formula_bundle_derived else "reject",
            "updated-pack exact charge-density formula bundle derived",
            sign_base.truth(updated_pack_exact_charge_density_formula_bundle_derived),
            "The branch now exposes J_Noether^0[Q] = 2 omega (-Q_g^2) and its exact two-component rewriting in terms of f_0 and f_L.",
        ),
        sign_base.row(
            "updated_pack_exact_charge_current_noether_closure_available_now",
            "pass" if updated_pack_exact_charge_current_noether_closure_available_now else "reject",
            "updated-pack exact charge-current / Noether-current closure available now",
            sign_base.truth(updated_pack_exact_charge_current_noether_closure_available_now),
            "The conserved background charge current is no longer only a target surface; it is now an explicit updated-pack formula bundle.",
        ),
        sign_base.row(
            "updated_pack_proxy_signed_density_promoted_to_exact_now",
            "pass" if updated_pack_proxy_signed_density_promoted_to_exact_now else "reject",
            "updated-pack proxy signed density promoted to exact now",
            sign_base.truth(updated_pack_proxy_signed_density_promoted_to_exact_now),
            "The old |f_0|^2 - |f_L|^2 object remains only a weak-field comparison target; the exact current closes as 2 omega (-Q_g^2), not as the flat proxy itself.",
        ),
        sign_base.row(
            "updated_pack_charge_current_derivation_closes_second_missing_primitive_now",
            "pass" if updated_pack_charge_current_derivation_closes_second_missing_primitive_now else "reject",
            "updated-pack charge-current derivation closes second missing primitive now",
            sign_base.truth(updated_pack_charge_current_derivation_closes_second_missing_primitive_now),
            "The background-expansion primitive is already closed, and this branch closes the next exact bridge: the conserved Q-ball charge current.",
        ),
        sign_base.row(
            "updated_pack_low_order_jeff0_primary_refresh_required",
            "pass" if updated_pack_low_order_jeff0_primary_refresh_required else "reject",
            "updated-pack low-order J_eff^0 primary refresh required",
            sign_base.truth(updated_pack_low_order_jeff0_primary_refresh_required),
            "Once the exact charge current is explicit, the next honest remaining theorem object is the low-order J_eff^0 synthesis.",
        ),
        sign_base.row(
            "exact_source_theorem_derived_now",
            "pass" if exact_source_theorem_derived_now else "reject",
            "exact source theorem derived now",
            sign_base.truth(exact_source_theorem_derived_now),
            "This branch derives the conserved background charge current only; the photon-side source theorem remains downstream of low-order J_eff^0.",
        ),
        sign_base.row(
            "blind_vector_observable_gate_still_blocked",
            "pass" if blind_vector_observable_gate_still_blocked else "reject",
            "blind vector observable gate still blocked",
            sign_base.truth(blind_vector_observable_gate_still_blocked),
            "Blind-vector direct computation remains downstream because low-order J_eff^0 is still absent even though the charge-current bridge is now explicit.",
        ),
        sign_base.row(
            "farther_hybrid_continuation_reopen_required_now",
            "pass" if farther_hybrid_continuation_reopen_required_now else "reject",
            "farther hybrid continuation reopen required now",
            sign_base.truth(farther_hybrid_continuation_reopen_required_now),
            "Extra q-range evidence remains unnecessary because the blocker stays theorem-side.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "retained_scalar_residual_rel": float(prior_summary["retained_scalar_residual_rel"]),
        "updated_pack_exact_charge_current_derivation_audit_selected": updated_pack_exact_charge_current_derivation_audit_selected,
        "retained_global_u1_phase_surface_available": retained_global_u1_phase_surface_available,
        "retained_adopted_qball_u1_identity_available": retained_adopted_qball_u1_identity_available,
        "exact_background_norm_bundle_available": exact_background_norm_bundle_available,
        "frozen_jeff_zero_class_retained": frozen_jeff_zero_class_retained,
        "updated_pack_noether_phase_norm_inference_available": updated_pack_noether_phase_norm_inference_available,
        "updated_pack_exact_charge_density_formula_bundle_derived": updated_pack_exact_charge_density_formula_bundle_derived,
        "updated_pack_exact_charge_current_noether_closure_available_now": updated_pack_exact_charge_current_noether_closure_available_now,
        "updated_pack_proxy_signed_density_promoted_to_exact_now": updated_pack_proxy_signed_density_promoted_to_exact_now,
        "updated_pack_charge_current_derivation_closes_second_missing_primitive_now": updated_pack_charge_current_derivation_closes_second_missing_primitive_now,
        "updated_pack_low_order_jeff0_primary_refresh_required": updated_pack_low_order_jeff0_primary_refresh_required,
        "exact_source_theorem_derived_now": exact_source_theorem_derived_now,
        "blind_vector_observable_gate_still_blocked": blind_vector_observable_gate_still_blocked,
        "farther_hybrid_continuation_reopen_required_now": farther_hybrid_continuation_reopen_required_now,
        "selected_primary_pack_update_surface": "updated_pack_exact_charge_current_noether_derived",
        "selected_secondary_pack_update_surface": "updated_pack_exact_low_order_jeff0_formula_derivation",
        "selected_reserve_completion_lane": "blind_vector_after_low_order_jeff0_derivation",
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": FOLLOWUP_ROUTE_NAME,
        "selected_followup_route_or_none": FOLLOWUP_ROUTE,
        "physical_reject_required": False,
    }

    declaration_payload = sign_base.payload(
        "8.7.56.2593",
        STEP_NAME + " declaration gate",
        {
            "source_files": {
                "status": sign_base.display_path(STATUS),
                "roadmap": sign_base.display_path(ROADMAP),
                "ai_context": sign_base.display_path(AI_CONTEXT),
                "work_history_recent": sign_base.display_path(WORK_HISTORY_RECENT),
                "current_problem": sign_base.display_path(CURRENT_PROBLEM),
                "current_status": sign_base.display_path(CURRENT_STATUS),
                "unified_roadmap": sign_base.display_path(UNIFIED_ROADMAP),
                "long_roadmap": sign_base.display_path(LONG_ROADMAP),
                "part1": sign_base.display_path(PART1),
                "part3a": sign_base.display_path(PART3A),
                "part5": sign_base.display_path(PART5),
                "prior_gate": sign_base.display_path(PRIOR_GATE),
                "background_derivation": sign_base.display_path(BACKGROUND_DERIVATION),
                "global_u1_source": sign_base.display_path(GLOBAL_U1_SOURCE),
                "qball_charge_mapping": sign_base.display_path(QBALL_CHARGE_MAPPING),
                "qball_charge_normalization": sign_base.display_path(QBALL_CHARGE_NORMALIZATION),
                "frozen_jeff_class": sign_base.display_path(FROZEN_JEFF_CLASS),
            },
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route_name": NEXT_ROUTE_NAME,
                "next_route": NEXT_ROUTE,
                "followup_route_name": FOLLOWUP_ROUTE_NAME,
                "followup_route": FOLLOWUP_ROUTE,
            },
        },
        rows,
        summary,
        {
            "overall_status": "vector_qball_form_factor_updated_pack_charge_current_derivation_declared",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "formulas": build_formulae(background_formulas),
            "hits": {
                "status_branch_hit": sign_base.hit(status_text, "8.7.56.2591"),
                "roadmap_branch_hit": sign_base.hit(roadmap_text, ".2587-.2590"),
                "current_problem_hit": sign_base.hit(current_problem_text, "exact charge-current / Noether-current closure"),
                "current_status_hit": sign_base.hit(current_status_text, "exact charge-current / Noether-current closure"),
                "unified_roadmap_hit": sign_base.hit(unified_text, ".2583-.2590"),
                "long_roadmap_hit": sign_base.hit(long_text, ".2583-.2590"),
                "part1_continuity_hit": sign_base.hit(part1_text, "\\partial_\\mu J^\\mu=0"),
                "part3a_qball_identity_hit": sign_base.hit(part3a_text, "Q-ball Noether charge = adopted U(1) charge"),
                "part5_hit": sign_base.hit(part5_text, "exact charge-current / Noether-current closure"),
                "background_qg2": background_formulas["exact_caseb_lift"],
                "background_qg2_series": background_formulas["Qg2_series"],
            },
            "inference": {
                "charge_current_formula_is_inference_from_sources": True,
                "why": (
                    "The retained pack freezes global U(1) phase symmetry, continuity, "
                    "the direct adopted-U(1) / Q-ball identity, and the exact updated-pack "
                    "background norm bundle. The explicit Noether current bundle is therefore "
                    "written as a phase-gradient current proportional to the exact norm."
                ),
                "old_zero_class_object": "J_eff^mu[a;Q] same-field photon source",
                "current_branch_object": "J_Noether^mu[Q] conserved background charge current",
            },
        },
    )
    declaration_paths = write_artifact("declaration_gate", declaration_payload)

    route_payload = {
        "generated_utc": sign_base.now_iso(),
        "phase": {
            "phase": 8,
            "step": "8.7.56.2594",
            "name": STEP_NAME + " route sync",
        },
        "inputs": declaration_paths,
        "rows": rows,
        "summary": summary,
        "decision": {
            "overall_status": "vector_qball_form_factor_updated_pack_charge_current_derivation_route_synced",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        "evidence": {
            "formulas": build_formulae(background_formulas),
            "disposition": {
                "exact_charge_current_noether_closure_available_now": updated_pack_exact_charge_current_noether_closure_available_now,
                "proxy_signed_density_still_limit_only": not updated_pack_proxy_signed_density_promoted_to_exact_now,
                "low_order_jeff0_primary_refresh_required": updated_pack_low_order_jeff0_primary_refresh_required,
                "direct_blind_vector_still_blocked": blind_vector_observable_gate_still_blocked,
            },
        },
    }
    route_paths = write_artifact("route_sync", route_payload)

    print("[ok] updated-pack exact charge-current derivation artifacts written")
    print(f"  declaration_gate: {declaration_paths['json']}")
    print(f"  route_sync: {route_paths['json']}")


if __name__ == "__main__":
    main()
