#!/usr/bin/env python3
"""Generate 8.7.56.2607-.2610 updated-pack exact source-theorem closeout artifacts."""

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
PART5 = ROOT / "doc" / "paper" / "14_part5_future_predictions.md"

PRIOR_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.2603-2606",
        "updated_pack_low_order_jeff0_derivation_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
SOURCE_AUDIT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.2455-2458",
        "updated_pack_exact_effective_source_theorem_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
SOURCE_RULE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.2463-2466",
        "updated_pack_source_rule_audit",
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
CHARGE_DERIVATION = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.2591-2594",
        "updated_pack_exact_charge_current_derivation_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
LOW_ORDER_DERIVATION = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.2599-2602",
        "updated_pack_exact_low_order_jeff0_derivation_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
FROZEN_JEFF_CLASS = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.1567-1570",
        "jeff_q0_class",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.2607-2610"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack exact "
    "source-theorem closeout audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_exact_source_theorem_closeout_audit",
    prefix="q",
)

PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_low_order_"
    "jeff0_derived_exact_source_theorem_primary_blind_vector_reserve_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_exact_source_"
    "theorem_no_go_derived_exact_ell0_operator_gate"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_exact_source_"
    "theorem_gate_exact_ell0_operator_refresh"
)
NEXT_ROUTE = "8.7.56.2611"
FOLLOWUP_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_exact_ell0_"
    "action_level_operator_refresh_audit"
)
FOLLOWUP_ROUTE = "8.7.56.2615"


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


# 関数: closeout で使う式を返す。

def build_formulae(
    source_formulas: dict[str, str],
    background_formulas: dict[str, str],
    charge_formulas: dict[str, str],
    low_formulas: dict[str, str],
    jeff_class_formulas: dict[str, str],
) -> dict[str, str]:
    """Return formulas used in the updated-pack exact source-theorem closeout audit."""
    return {
        "effective_source_surface": source_formulas["effective_source_surface"],
        "exact_background_bundle": background_formulas["exact_caseb_lift"],
        "exact_charge_density": charge_formulas["exact_charge_density"],
        "same_field_formula": low_formulas["derived_low_order_jeff0"],
        "object_split": charge_formulas["object_split"],
        "zero_class_rule": jeff_class_formulas["classification_rule"],
        "exact_source_theorem": (
            "Under the current updated-pack same-field photon-source route, "
            "J_eff^0[a;Q]_low-order,same-field = 0, while J_Noether^0[Q] is a "
            "distinct conserved background current. Therefore the proxy "
            "|f_0|^2 - |f_L|^2 route is exact no-go unless microscopic matter "
            "or rotational source functionals are reopened."
        ),
    }


# 関数: `.2607-.2610` を実行する。

def main() -> None:
    """Execute the updated-pack exact source-theorem closeout audit."""
    for path in (
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        WORK_HISTORY_RECENT,
        CURRENT_PROBLEM,
        CURRENT_STATUS,
        UNIFIED_ROADMAP,
        LONG_ROADMAP,
        PART5,
        PRIOR_GATE,
        SOURCE_AUDIT,
        SOURCE_RULE,
        BACKGROUND_DERIVATION,
        CHARGE_DERIVATION,
        LOW_ORDER_DERIVATION,
        FROZEN_JEFF_CLASS,
    ):
        sign_base.require(path)

    status_text = sign_base.read_text(STATUS)
    roadmap_text = sign_base.read_text(ROADMAP)
    current_problem_text = sign_base.read_text(CURRENT_PROBLEM)
    current_status_text = sign_base.read_text(CURRENT_STATUS)
    unified_text = sign_base.read_text(UNIFIED_ROADMAP)
    long_text = sign_base.read_text(LONG_ROADMAP)
    part5_text = sign_base.read_text(PART5)

    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    source_payload = sign_base.read_json(SOURCE_AUDIT)
    source_summary = source_payload["summary"]
    source_formulas = source_payload["evidence"]["formulas"]
    source_rule_payload = sign_base.read_json(SOURCE_RULE)
    source_rule_summary = source_rule_payload["summary"]
    background_payload = sign_base.read_json(BACKGROUND_DERIVATION)
    background_summary = background_payload["summary"]
    background_formulas = background_payload["evidence"]["formulas"]
    charge_payload = sign_base.read_json(CHARGE_DERIVATION)
    charge_summary = charge_payload["summary"]
    charge_formulas = charge_payload["evidence"]["formulas"]
    low_payload = sign_base.read_json(LOW_ORDER_DERIVATION)
    low_summary = low_payload["summary"]
    low_formulas = low_payload["evidence"]["formulas"]
    jeff_class_payload = sign_base.read_json(FROZEN_JEFF_CLASS)
    jeff_class_summary = jeff_class_payload["summary"]
    jeff_class_formulas = jeff_class_payload["evidence"]["formulas"]

    updated_pack_exact_source_theorem_closeout_audit_selected = bool(
        prior_summary["gate_b_updated_pack_exact_source_theorem_closeout_primary_selected"]
        and not prior_summary["gate_c_blind_vector_computation_primary_admissible_now"]
    )
    updated_pack_step_c_surface_explicit = bool(
        source_summary["updated_pack_step_c_surface_explicit"]
    )
    updated_pack_source_rule_no_go_surface_explicit = bool(
        source_rule_summary["updated_pack_source_rule_no_go_surface_explicit"]
        and source_rule_summary["updated_pack_low_order_jeff0_discriminator_surface_explicit"]
    )
    updated_pack_exact_qball_background_expansion_available_now = bool(
        background_summary["updated_pack_exact_qball_background_expansion_available_now"]
    )
    updated_pack_exact_charge_current_noether_closure_available_now = bool(
        charge_summary["updated_pack_exact_charge_current_noether_closure_available_now"]
    )
    updated_pack_exact_low_order_jeff0_formula_available_now = bool(
        low_summary["updated_pack_exact_low_order_jeff0_formula_available_now"]
    )
    same_field_on_shell_zero_retained = bool(
        jeff_class_summary["same_field_on_shell_zero_retained"]
    )
    microscopic_matter_functional_available = bool(
        jeff_class_summary["microscopic_matter_functional_available"]
    )
    microscopic_rotational_functional_available = bool(
        jeff_class_summary["microscopic_rotational_functional_available"]
    )
    updated_pack_same_field_source_zero_fixed = bool(
        same_field_on_shell_zero_retained
        and updated_pack_exact_low_order_jeff0_formula_available_now
        and low_summary["updated_pack_low_order_jeff0_same_field_zero_formula_derived"]
    )
    updated_pack_current_pack_nonzero_source_functional_available = bool(
        microscopic_matter_functional_available
        or microscopic_rotational_functional_available
    )
    updated_pack_exact_source_theorem_support_verdict_passed = False
    updated_pack_exact_source_theorem_no_go_verdict_passed = bool(
        updated_pack_exact_source_theorem_closeout_audit_selected
        and updated_pack_step_c_surface_explicit
        and updated_pack_source_rule_no_go_surface_explicit
        and updated_pack_exact_qball_background_expansion_available_now
        and updated_pack_exact_charge_current_noether_closure_available_now
        and updated_pack_exact_low_order_jeff0_formula_available_now
        and updated_pack_same_field_source_zero_fixed
        and not updated_pack_current_pack_nonzero_source_functional_available
        and low_summary["updated_pack_proxy_signed_density_no_go_verdict_passed"]
    )
    updated_pack_exact_source_theorem_derived_now = bool(
        updated_pack_exact_source_theorem_no_go_verdict_passed
    )
    updated_pack_exact_source_theorem_closes_current_theorem_lane_now = bool(
        updated_pack_exact_source_theorem_derived_now
    )
    updated_pack_exact_ell0_action_level_operator_refresh_required = bool(
        updated_pack_exact_source_theorem_derived_now
    )
    residual_origin_theorem_explained_now = False
    blind_vector_observable_gate_still_blocked = True
    farther_hybrid_continuation_reopen_required_now = False

    rows = [
        sign_base.row(
            "updated_pack_exact_source_theorem_closeout_audit_selected",
            "pass" if updated_pack_exact_source_theorem_closeout_audit_selected else "reject",
            "updated-pack exact source-theorem closeout audit selected",
            sign_base.truth(updated_pack_exact_source_theorem_closeout_audit_selected),
            "The low-order J_eff^0 derivation gate already promoted exact source-theorem closeout as the next honest theorem lane.",
        ),
        sign_base.row(
            "updated_pack_step_c_surface_explicit",
            "pass" if updated_pack_step_c_surface_explicit else "reject",
            "updated-pack Step C source/current surface explicit",
            sign_base.truth(updated_pack_step_c_surface_explicit),
            "The theorem still rests on the same explicit photon-side source surface L ⊃ a_mu J_eff^mu[P^Qball].",
        ),
        sign_base.row(
            "updated_pack_source_rule_no_go_surface_explicit",
            "pass" if updated_pack_source_rule_no_go_surface_explicit else "reject",
            "updated-pack source-rule no-go surface explicit",
            sign_base.truth(updated_pack_source_rule_no_go_surface_explicit),
            "The route-local no-go discriminator is already explicit at the low-order J_eff^0 surface.",
        ),
        sign_base.row(
            "updated_pack_exact_qball_background_expansion_available_now",
            "pass" if updated_pack_exact_qball_background_expansion_available_now else "reject",
            "updated-pack exact Q-ball background expansion available now",
            sign_base.truth(updated_pack_exact_qball_background_expansion_available_now),
            "The exact caseB Q_g^mu / Q_g^2 bundle is now derived rather than only targeted.",
        ),
        sign_base.row(
            "updated_pack_exact_charge_current_noether_closure_available_now",
            "pass" if updated_pack_exact_charge_current_noether_closure_available_now else "reject",
            "updated-pack exact charge-current / Noether-current closure available now",
            sign_base.truth(updated_pack_exact_charge_current_noether_closure_available_now),
            "The conserved background Noether current is now explicit rather than a placeholder bridge.",
        ),
        sign_base.row(
            "updated_pack_exact_low_order_jeff0_formula_available_now",
            "pass" if updated_pack_exact_low_order_jeff0_formula_available_now else "reject",
            "updated-pack exact low-order J_eff^0 formula available now",
            sign_base.truth(updated_pack_exact_low_order_jeff0_formula_available_now),
            "The same-field photon-source formula is now explicit and zero under the current updated-pack.",
        ),
        sign_base.row(
            "updated_pack_same_field_source_zero_fixed",
            "pass" if updated_pack_same_field_source_zero_fixed else "reject",
            "updated-pack same-field source zero fixed",
            sign_base.truth(updated_pack_same_field_source_zero_fixed),
            "The frozen same-field on-shell zero class now survives the updated-pack derivation stack as an exact low-order source formula.",
        ),
        sign_base.row(
            "updated_pack_current_pack_nonzero_source_functional_available",
            "pass" if updated_pack_current_pack_nonzero_source_functional_available else "reject",
            "updated-pack current-pack nonzero source functional available",
            sign_base.truth(updated_pack_current_pack_nonzero_source_functional_available),
            "A theorem-level nonzero source would require microscopic matter or rotational functionals, which remain absent under the current pack.",
        ),
        sign_base.row(
            "updated_pack_exact_source_theorem_support_verdict_passed",
            "pass" if updated_pack_exact_source_theorem_support_verdict_passed else "reject",
            "updated-pack exact source-theorem support verdict passed",
            sign_base.truth(updated_pack_exact_source_theorem_support_verdict_passed),
            "The proxy-support branch does not survive because the exact same-field source formula is zero rather than signed density.",
        ),
        sign_base.row(
            "updated_pack_exact_source_theorem_no_go_verdict_passed",
            "pass" if updated_pack_exact_source_theorem_no_go_verdict_passed else "reject",
            "updated-pack exact source-theorem no-go verdict passed",
            sign_base.truth(updated_pack_exact_source_theorem_no_go_verdict_passed),
            "With background expansion, exact charge current, and low-order same-field J_eff^0 all explicit, the current updated-pack theorem closes on the zero / no-go branch.",
        ),
        sign_base.row(
            "updated_pack_exact_source_theorem_derived_now",
            "pass" if updated_pack_exact_source_theorem_derived_now else "reject",
            "updated-pack exact source theorem derived now",
            sign_base.truth(updated_pack_exact_source_theorem_derived_now),
            "The theorem is now explicit: the same-field photon-side source remains zero under the current pack, while the exact conserved Noether current is a different object.",
        ),
        sign_base.row(
            "updated_pack_exact_source_theorem_closes_current_theorem_lane_now",
            "pass" if updated_pack_exact_source_theorem_closes_current_theorem_lane_now else "reject",
            "updated-pack exact source-theorem closes current theorem lane now",
            sign_base.truth(updated_pack_exact_source_theorem_closes_current_theorem_lane_now),
            "The theorem-side blocker is no longer 'missing exact source theorem'; it is now the downstream action-level operator / pack-update question.",
        ),
        sign_base.row(
            "updated_pack_exact_ell0_action_level_operator_refresh_required",
            "pass" if updated_pack_exact_ell0_action_level_operator_refresh_required else "reject",
            "updated-pack exact ell=0 action-level operator refresh required",
            sign_base.truth(updated_pack_exact_ell0_action_level_operator_refresh_required),
            "Once the source theorem is closed as current-pack no-go, the remaining honest mainline returns to the exact ell=0 action-level operator gap.",
        ),
        sign_base.row(
            "residual_origin_theorem_explained_now",
            "pass" if residual_origin_theorem_explained_now else "reject",
            "residual origin theorem explained now",
            sign_base.truth(residual_origin_theorem_explained_now),
            "The theorem closes as no-go under the current pack, so the retained 1.9% scalar residual is still not explained theorem-level.",
        ),
        sign_base.row(
            "blind_vector_observable_gate_still_blocked",
            "pass" if blind_vector_observable_gate_still_blocked else "reject",
            "blind vector observable gate still blocked",
            sign_base.truth(blind_vector_observable_gate_still_blocked),
            "Blind-vector direct computation remains reserve-only because the current-pack source theorem closes on no vector correction.",
        ),
        sign_base.row(
            "farther_hybrid_continuation_reopen_required_now",
            "pass" if farther_hybrid_continuation_reopen_required_now else "reject",
            "farther hybrid continuation reopen required now",
            sign_base.truth(farther_hybrid_continuation_reopen_required_now),
            "Extra q-range evidence remains unnecessary because the theorem-side lane now closes as no-go.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "retained_scalar_residual_rel": float(prior_summary["retained_scalar_residual_rel"]),
        "updated_pack_exact_source_theorem_closeout_audit_selected": updated_pack_exact_source_theorem_closeout_audit_selected,
        "updated_pack_step_c_surface_explicit": updated_pack_step_c_surface_explicit,
        "updated_pack_source_rule_no_go_surface_explicit": updated_pack_source_rule_no_go_surface_explicit,
        "updated_pack_exact_qball_background_expansion_available_now": updated_pack_exact_qball_background_expansion_available_now,
        "updated_pack_exact_charge_current_noether_closure_available_now": updated_pack_exact_charge_current_noether_closure_available_now,
        "updated_pack_exact_low_order_jeff0_formula_available_now": updated_pack_exact_low_order_jeff0_formula_available_now,
        "same_field_on_shell_zero_retained": same_field_on_shell_zero_retained,
        "microscopic_matter_functional_available": microscopic_matter_functional_available,
        "microscopic_rotational_functional_available": microscopic_rotational_functional_available,
        "updated_pack_same_field_source_zero_fixed": updated_pack_same_field_source_zero_fixed,
        "updated_pack_current_pack_nonzero_source_functional_available": updated_pack_current_pack_nonzero_source_functional_available,
        "updated_pack_exact_source_theorem_support_verdict_passed": updated_pack_exact_source_theorem_support_verdict_passed,
        "updated_pack_exact_source_theorem_no_go_verdict_passed": updated_pack_exact_source_theorem_no_go_verdict_passed,
        "updated_pack_exact_source_theorem_derived_now": updated_pack_exact_source_theorem_derived_now,
        "updated_pack_exact_source_theorem_closes_current_theorem_lane_now": updated_pack_exact_source_theorem_closes_current_theorem_lane_now,
        "updated_pack_exact_ell0_action_level_operator_refresh_required": updated_pack_exact_ell0_action_level_operator_refresh_required,
        "residual_origin_theorem_explained_now": residual_origin_theorem_explained_now,
        "blind_vector_observable_gate_still_blocked": blind_vector_observable_gate_still_blocked,
        "farther_hybrid_continuation_reopen_required_now": farther_hybrid_continuation_reopen_required_now,
        "selected_primary_pack_update_surface": "updated_pack_exact_source_theorem_no_go_derived",
        "selected_secondary_pack_update_surface": "updated_pack_exact_ell0_action_level_operator_refresh",
        "selected_reserve_completion_lane": "blind_vector_after_exact_ell0_operator_refresh",
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": FOLLOWUP_ROUTE_NAME,
        "selected_followup_route_or_none": FOLLOWUP_ROUTE,
        "physical_reject_required": False,
    }

    declaration_payload = sign_base.payload(
        "8.7.56.2609",
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
                "part5": sign_base.display_path(PART5),
                "prior_gate": sign_base.display_path(PRIOR_GATE),
                "source_audit": sign_base.display_path(SOURCE_AUDIT),
                "source_rule": sign_base.display_path(SOURCE_RULE),
                "background_derivation": sign_base.display_path(BACKGROUND_DERIVATION),
                "charge_derivation": sign_base.display_path(CHARGE_DERIVATION),
                "low_order_derivation": sign_base.display_path(LOW_ORDER_DERIVATION),
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
            "overall_status": "vector_qball_form_factor_updated_pack_exact_source_theorem_closeout_declared",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "formulas": build_formulae(
                source_formulas,
                background_formulas,
                charge_formulas,
                low_formulas,
                jeff_class_formulas,
            ),
            "hits": {
                "status_branch_hit": sign_base.hit(status_text, "8.7.56.2607"),
                "roadmap_branch_hit": sign_base.hit(roadmap_text, ".2587-.2590"),
                "current_problem_hit": sign_base.hit(current_problem_text, "exact source-theorem closeout"),
                "current_status_hit": sign_base.hit(current_status_text, "exact source-theorem closeout"),
                "unified_roadmap_hit": sign_base.hit(unified_text, ".2583-.2590"),
                "long_roadmap_hit": sign_base.hit(long_text, ".2587-.2590"),
                "part5_same_field_zero_hit": sign_base.hit(part5_text, "same-field on-shell zero"),
                "part5_no_go_hit": sign_base.hit(part5_text, "proxy route は no-go"),
            },
            "inference": {
                "exact_source_theorem_is_current_pack_no_go": True,
                "why": (
                    "The updated-pack now makes the background bundle, conserved "
                    "Noether current, and low-order same-field J_eff^0 explicit. "
                    "Because J_eff^0 remains zero while J_Noether^0 is a distinct "
                    "object and microscopic nonzero source functionals remain absent, "
                    "the current-pack exact source theorem closes on the no-go branch."
                ),
            },
        },
    )
    declaration_paths = write_artifact("declaration_gate", declaration_payload)

    route_payload = {
        "generated_utc": sign_base.now_iso(),
        "phase": {
            "phase": 8,
            "step": "8.7.56.2610",
            "name": STEP_NAME + " route sync",
        },
        "inputs": declaration_paths,
        "rows": rows,
        "summary": summary,
        "decision": {
            "overall_status": "vector_qball_form_factor_updated_pack_exact_source_theorem_closeout_route_synced",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        "evidence": {
            "formulas": build_formulae(
                source_formulas,
                background_formulas,
                charge_formulas,
                low_formulas,
                jeff_class_formulas,
            ),
            "disposition": {
                "exact_source_theorem_derived_now": updated_pack_exact_source_theorem_derived_now,
                "exact_source_theorem_no_go_verdict_passed": updated_pack_exact_source_theorem_no_go_verdict_passed,
                "residual_origin_theorem_explained_now": residual_origin_theorem_explained_now,
                "direct_blind_vector_still_blocked": blind_vector_observable_gate_still_blocked,
            },
        },
    }
    route_paths = write_artifact("route_sync", route_payload)

    print("[ok] updated-pack exact source-theorem closeout artifacts written")
    print(f"  declaration_gate: {declaration_paths['json']}")
    print(f"  route_sync: {route_paths['json']}")


if __name__ == "__main__":
    main()
