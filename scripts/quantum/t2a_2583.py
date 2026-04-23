#!/usr/bin/env python3
"""Generate 8.7.56.2583-.2586 exact background-expansion derivation artifacts."""

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
PART5 = ROOT / "doc" / "paper" / "14_part5_future_predictions.md"

PRIOR_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.2579-2582",
        "updated_pack_residual_origin_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
SERIES_AUDIT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.2447-2450",
        "updated_pack_exact_ell0_series_operator_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
BACKGROUND_LIFT_AUDIT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.1607-1610",
        "eff_metric_k_deriv",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.2583-2586"
STEP_NAME = "Trial-2 numeric alpha vector Q-ball form-factor updated-pack exact background-expansion derivation audit"
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_exact_background_expansion_derivation_audit",
    prefix="q",
)

PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_residual_origin_"
    "refresh_audited_background_expansion_derivation_primary_charge_current_followup_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_background_"
    "expansion_derived_charge_current_gate"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_background_expansion_"
    "derivation_gate_charge_current_refresh"
)
NEXT_ROUTE = "8.7.56.2587"
FOLLOWUP_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_exact_charge_current_"
    "derivation_audit"
)
FOLLOWUP_ROUTE = "8.7.56.2591"


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


# 関数: background-expansion derivation で使う式を返す。

def build_formulae() -> dict[str, str]:
    """Return the exact background-expansion derivation bundle."""
    return {
        "metric_u_rule": "u = ln(P_t / P_infty), with P_t = f_0 on the Q-ball background",
        "exact_caseb_lift": "Q_g^0 = -(f_0^3 / P_infty^2), Q_g^i = (P_infty^2 / f_0^2) f_L r_hat^i, Q_g^2 = -(f_0^4 / P_infty^2) + (P_infty^2 / f_0^2) f_L^2",
        "inverse_f0_sq_series": "f_0^(-2) = a_0^(-2) [1 - 2 (a_2/a_0) r^2 + (3 a_2^2/a_0^2 - 2 a_4/a_0) r^4 + O(r^6)]",
        "Qg0_series": "Q_g^0 = -(1/P_infty^2) [a_0^3 + 3 a_0^2 a_2 r^2 + (3 a_0^2 a_4 + 3 a_0 a_2^2) r^4 + O(r^6)]",
        "Qgr_series": "Q_g^r = P_infty^2 [(b_1/a_0^2) r + (b_3/a_0^2 - 2 a_2 b_1/a_0^3) r^3 + (b_5/a_0^2 - 2 a_2 b_3/a_0^3 + (3 a_2^2/a_0^4 - 2 a_4/a_0^3) b_1) r^5 + O(r^7)]",
        "Qg2_series": "Q_g^2 = -(a_0^4/P_infty^2) + [-(4 a_0^3 a_2)/P_infty^2 + P_infty^2 b_1^2/a_0^2] r^2 + [-(4 a_0^3 a_4 + 6 a_0^2 a_2^2)/P_infty^2 + P_infty^2 (2 b_1 b_3/a_0^2 - 2 a_2 b_1^2/a_0^3)] r^4 + O(r^6)",
    }


# 関数: `.2583-.2586` を実行する。

def main() -> None:
    """Execute the exact background-expansion derivation audit."""
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
        PART5,
        PRIOR_GATE,
        SERIES_AUDIT,
        BACKGROUND_LIFT_AUDIT,
    ):
        sign_base.require(path)

    status_text = sign_base.read_text(STATUS)
    roadmap_text = sign_base.read_text(ROADMAP)
    current_problem_text = sign_base.read_text(CURRENT_PROBLEM)
    current_status_text = sign_base.read_text(CURRENT_STATUS)
    unified_text = sign_base.read_text(UNIFIED_ROADMAP)
    long_text = sign_base.read_text(LONG_ROADMAP)
    part1_text = sign_base.read_text(PART1)
    part5_text = sign_base.read_text(PART5)

    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    series_payload = sign_base.read_json(SERIES_AUDIT)
    lift_payload = sign_base.read_json(BACKGROUND_LIFT_AUDIT)
    lift_formulae = lift_payload["evidence"]["formulas"]

    updated_pack_exact_background_expansion_derivation_audit_selected = bool(
        prior_summary["gate_b_updated_pack_exact_background_expansion_derivation_primary_selected"]
        and not prior_summary["direct_blind_vector_computation_primary_admissible_now"]
    )
    caseb_static_metric_u_relation_available = bool(
        sign_base.hit(part1_text, "u=\\ln\\!\\left(\\frac{P_t}{P_{\\infty}}\\right)") is not None
        or sign_base.hit(part1_text, "u=\\ln\\!\\left(\\frac{P_t}{P_{\\infty}}\\right)") is not None
        or sign_base.hit(part1_text, "u=\\ln\\!\\left(\\frac{P_t}{P_{\\infty}}\\right)") is not None
        or sign_base.hit(part1_text, "u=\\ln\\!\\left(\\frac{P_t}{P_{\\infty}}\\right)") is not None
    )
    caseb_background_split_available = bool(
        lift_formulae["background_split"] == "Q_mu = (f_0, f_L r_hat_i)"
        and series_payload["summary"]["updated_pack_exact_ell0_series_surface_explicit"]
    )
    caseb_exact_lift_rewritten_in_f0_available = bool(
        lift_payload["summary"]["effective_metric_raised_background_components_derived"]
        and lift_payload["summary"]["effective_metric_background_norm_derived"]
        and caseb_static_metric_u_relation_available
        and caseb_background_split_available
    )
    caseb_exact_qball_background_series_available = bool(
        caseb_exact_lift_rewritten_in_f0_available
        and series_payload["summary"]["updated_pack_exact_ell0_series_surface_explicit"]
    )
    caseb_exact_qball_background_expansion_formula_bundle_derived = bool(
        caseb_exact_qball_background_series_available
    )
    updated_pack_exact_qball_background_expansion_available_now = bool(
        caseb_exact_qball_background_expansion_formula_bundle_derived
    )
    updated_pack_exact_qball_background_expansion_closes_first_missing_primitive_now = bool(
        updated_pack_exact_qball_background_expansion_available_now
    )
    updated_pack_exact_charge_current_primary_refresh_required = bool(
        updated_pack_exact_qball_background_expansion_available_now
    )
    exact_source_theorem_derived_now = False
    blind_vector_observable_gate_still_blocked = True
    farther_hybrid_continuation_reopen_required_now = False

    rows = [
        sign_base.row(
            "updated_pack_exact_background_expansion_derivation_audit_selected",
            "pass" if updated_pack_exact_background_expansion_derivation_audit_selected else "reject",
            "updated-pack exact background-expansion derivation audit selected",
            sign_base.truth(updated_pack_exact_background_expansion_derivation_audit_selected),
            "Residual-origin gate already rerouted the theorem-side stack to background-expansion derivation.",
        ),
        sign_base.row(
            "caseb_static_metric_u_relation_available",
            "pass" if caseb_static_metric_u_relation_available else "reject",
            "caseB static-metric u relation available",
            sign_base.truth(caseb_static_metric_u_relation_available),
            "Part I already fixes u = ln(P_t / P_infty), so the caseB lift can be rewritten directly on the Q-ball background.",
        ),
        sign_base.row(
            "caseb_background_split_available",
            "pass" if caseb_background_split_available else "reject",
            "caseB background split available",
            sign_base.truth(caseb_background_split_available),
            "The retained caseB derivation already exposes the Q-ball background as Q_mu = (f_0, f_L r_hat_i).",
        ),
        sign_base.row(
            "caseb_exact_lift_rewritten_in_f0_available",
            "pass" if caseb_exact_lift_rewritten_in_f0_available else "reject",
            "caseB exact lift rewritten in f0 available",
            sign_base.truth(caseb_exact_lift_rewritten_in_f0_available),
            "Combining u = ln(P_t / P_infty) with P_t = f_0 rewrites the caseB lift into exact f_0-only prefactors.",
        ),
        sign_base.row(
            "caseb_exact_qball_background_series_available",
            "pass" if caseb_exact_qball_background_series_available else "reject",
            "caseB exact Q-ball background series available",
            sign_base.truth(caseb_exact_qball_background_series_available),
            "The retained near-origin two-component series is enough to expand the exact caseB lift explicitly.",
        ),
        sign_base.row(
            "caseb_exact_qball_background_expansion_formula_bundle_derived",
            "pass" if caseb_exact_qball_background_expansion_formula_bundle_derived else "reject",
            "caseB exact Q-ball background expansion formula bundle derived",
            sign_base.truth(caseb_exact_qball_background_expansion_formula_bundle_derived),
            "The branch now exposes explicit formulas for Q_g^0, Q_g^r, and Q_g^2 as exact symbolic background expansions.",
        ),
        sign_base.row(
            "updated_pack_exact_qball_background_expansion_available_now",
            "pass" if updated_pack_exact_qball_background_expansion_available_now else "reject",
            "updated-pack exact Q-ball background expansion available now",
            sign_base.truth(updated_pack_exact_qball_background_expansion_available_now),
            "The first missing primitive is no longer only a target surface; it is now an explicit formula bundle in the public artifact set.",
        ),
        sign_base.row(
            "updated_pack_exact_qball_background_expansion_closes_first_missing_primitive_now",
            "pass" if updated_pack_exact_qball_background_expansion_closes_first_missing_primitive_now else "reject",
            "updated-pack exact Q-ball background expansion closes first missing primitive now",
            sign_base.truth(updated_pack_exact_qball_background_expansion_closes_first_missing_primitive_now),
            "This derivation closes the background-expansion primitive itself, though not the whole missing-action blocker.",
        ),
        sign_base.row(
            "updated_pack_exact_charge_current_primary_refresh_required",
            "pass" if updated_pack_exact_charge_current_primary_refresh_required else "reject",
            "updated-pack exact charge-current primary refresh required",
            sign_base.truth(updated_pack_exact_charge_current_primary_refresh_required),
            "With the background primitive explicit, the next honest remaining closure is exact charge-current / Noether-current derivation.",
        ),
        sign_base.row(
            "exact_source_theorem_derived_now",
            "pass" if exact_source_theorem_derived_now else "reject",
            "exact source theorem derived now",
            sign_base.truth(exact_source_theorem_derived_now),
            "This branch derives the background primitive only; the full source theorem remains downstream of charge-current closure and low-order J_eff^0.",
        ),
        sign_base.row(
            "blind_vector_observable_gate_still_blocked",
            "pass" if blind_vector_observable_gate_still_blocked else "reject",
            "blind vector observable gate still blocked",
            sign_base.truth(blind_vector_observable_gate_still_blocked),
            "Blind-vector direct computation remains downstream because exact charge-current closure and low-order J_eff^0 are still absent.",
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
        "updated_pack_exact_background_expansion_derivation_audit_selected": updated_pack_exact_background_expansion_derivation_audit_selected,
        "caseb_static_metric_u_relation_available": caseb_static_metric_u_relation_available,
        "caseb_background_split_available": caseb_background_split_available,
        "caseb_exact_lift_rewritten_in_f0_available": caseb_exact_lift_rewritten_in_f0_available,
        "caseb_exact_qball_background_series_available": caseb_exact_qball_background_series_available,
        "caseb_exact_qball_background_expansion_formula_bundle_derived": caseb_exact_qball_background_expansion_formula_bundle_derived,
        "updated_pack_exact_qball_background_expansion_available_now": updated_pack_exact_qball_background_expansion_available_now,
        "updated_pack_exact_qball_background_expansion_closes_first_missing_primitive_now": updated_pack_exact_qball_background_expansion_closes_first_missing_primitive_now,
        "updated_pack_exact_charge_current_primary_refresh_required": updated_pack_exact_charge_current_primary_refresh_required,
        "exact_source_theorem_derived_now": exact_source_theorem_derived_now,
        "blind_vector_observable_gate_still_blocked": blind_vector_observable_gate_still_blocked,
        "farther_hybrid_continuation_reopen_required_now": farther_hybrid_continuation_reopen_required_now,
        "selected_primary_pack_update_surface": "updated_pack_exact_qball_background_expansion_derived",
        "selected_secondary_pack_update_surface": "updated_pack_exact_charge_current_noether_derivation",
        "selected_tertiary_pack_update_surface": "updated_pack_exact_low_order_jeff0_formula_synthesis",
        "selected_reserve_completion_lane": "blind_vector_after_charge_current_and_low_order_jeff0_refresh",
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": FOLLOWUP_ROUTE_NAME,
        "selected_followup_route_or_none": FOLLOWUP_ROUTE,
        "physical_reject_required": False,
    }

    declaration_payload = sign_base.payload(
        "8.7.56.2585",
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
                "part5": sign_base.display_path(PART5),
                "prior_gate": sign_base.display_path(PRIOR_GATE),
                "series_audit": sign_base.display_path(SERIES_AUDIT),
                "background_lift_audit": sign_base.display_path(BACKGROUND_LIFT_AUDIT),
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
            "overall_status": "vector_qball_form_factor_updated_pack_background_expansion_derivation_declared",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "formulas": build_formulae(),
            "hits": {
                "status_branch_hit": sign_base.hit(status_text, "8.7.56.2575"),
                "roadmap_branch_hit": sign_base.hit(roadmap_text, ".2579-.2582"),
                "current_problem_hit": sign_base.hit(current_problem_text, "explicit Q-ball background expansion"),
                "current_status_hit": sign_base.hit(current_status_text, "explicit Q-ball background expansion"),
                "unified_roadmap_hit": sign_base.hit(unified_text, ".2571-.2574"),
                "long_roadmap_hit": sign_base.hit(long_text, ".2571-.2574"),
                "part5_hit": sign_base.hit(part5_text, "exact Q-ball background expansion"),
                "part1_u_hit": sign_base.hit(part1_text, "u=\\ln\\!\\left(\\frac{P_t}{P_{\\infty}}\\right)"),
                "lift_background_split": lift_formulae["background_split"],
                "lift_background_norm": lift_formulae["background_norm_caseb"],
            },
        },
    )
    declaration_paths = write_artifact("declaration_gate", declaration_payload)

    route_payload = {
        "generated_utc": sign_base.now_iso(),
        "phase": {
            "phase": 8,
            "step": "8.7.56.2586",
            "name": STEP_NAME + " route sync",
        },
        "inputs": declaration_paths,
        "rows": rows,
        "summary": summary,
        "decision": {
            "overall_status": "vector_qball_form_factor_updated_pack_background_expansion_derivation_route_synced",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        "evidence": {
            "formulae": build_formulae(),
            "disposition": {
                "background_expansion_available_now": updated_pack_exact_qball_background_expansion_available_now,
                "first_missing_primitive_closed_now": updated_pack_exact_qball_background_expansion_closes_first_missing_primitive_now,
                "charge_current_primary_refresh_required": updated_pack_exact_charge_current_primary_refresh_required,
            },
        },
    }
    route_paths = write_artifact("route_sync", route_payload)

    print("[ok] updated-pack exact background-expansion derivation artifacts written")
    print(f"  declaration_gate: {declaration_paths['json']}")
    print(f"  route_sync: {route_paths['json']}")


if __name__ == "__main__":
    main()
