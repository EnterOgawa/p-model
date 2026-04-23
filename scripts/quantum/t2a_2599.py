#!/usr/bin/env python3
"""Generate 8.7.56.2599-.2602 updated-pack exact low-order J_eff^0 derivation artifacts."""

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
        "8.7.56.2595-2598",
        "updated_pack_charge_current_derivation_gate",
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
FROZEN_JEFF_SPLIT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.1563-1566",
        "direct_jeff_deriv",
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

STEP_TAG = "8.7.56.2599-2602"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack exact "
    "low-order J_eff^0 derivation audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_exact_low_order_jeff0_derivation_audit",
    prefix="q",
)

PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_charge_current_"
    "derived_low_order_jeff0_primary_blind_vector_reserve_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_low_order_"
    "jeff0_derived_exact_source_theorem_gate"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_low_order_jeff0_"
    "derivation_gate_exact_source_theorem_refresh"
)
NEXT_ROUTE = "8.7.56.2603"
FOLLOWUP_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_exact_source_"
    "theorem_closeout_audit"
)
FOLLOWUP_ROUTE = "8.7.56.2607"


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


# 関数: low-order J_eff^0 derivation で使う式を返す。
def build_formulae(
    frozen_formulas: dict[str, str],
    charge_formulas: dict[str, str],
) -> dict[str, str]:
    """Return formulas used in the updated-pack low-order J_eff^0 derivation audit."""
    return {
        "same_field_source_object": frozen_formulas["jeff_charge_density"],
        "same_field_on_shell_zero": frozen_formulas["same_field_on_shell"],
        "exact_noether_density": charge_formulas["exact_charge_density"],
        "object_split": charge_formulas["object_split"],
        "derived_low_order_jeff0": (
            "J_eff^0[a;Q]_low-order,same-field = 0 under the current updated-pack, "
            "while J_Noether^0[Q] remains a distinct conserved background current"
        ),
        "proxy_no_go": (
            "J_eff^0[a;Q]_low-order,same-field = 0 != |f_0|^2 - |f_L|^2 "
            "(except trivial vanishing limit), so proxy strong support is exact no-go"
        ),
    }


# 関数: `.2599-.2602` を実行する。
def main() -> None:
    """Execute the updated-pack exact low-order J_eff^0 derivation audit."""
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
        CHARGE_DERIVATION,
        FROZEN_JEFF_SPLIT,
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
    charge_payload = sign_base.read_json(CHARGE_DERIVATION)
    charge_summary = charge_payload["summary"]
    charge_formulas = charge_payload["evidence"]["formulas"]
    frozen_split_payload = sign_base.read_json(FROZEN_JEFF_SPLIT)
    frozen_split_summary = frozen_split_payload["summary"]
    frozen_formulas = frozen_split_payload["evidence"]["formulas"]
    frozen_class_payload = sign_base.read_json(FROZEN_JEFF_CLASS)
    frozen_class_summary = frozen_class_payload["summary"]

    updated_pack_exact_low_order_jeff0_derivation_audit_selected = bool(
        prior_summary["gate_b_updated_pack_exact_low_order_jeff0_primary_selected"]
        and prior_summary["exact_charge_current_noether_closure_available_now"]
        and not prior_summary["gate_c_blind_vector_computation_primary_admissible_now"]
    )
    frozen_jeff_same_field_zero_retained = bool(
        frozen_split_summary["same_field_on_shell_zero_retained"]
        and frozen_class_summary["classification_case_iv_zero_under_current_pack"]
    )
    updated_pack_exact_charge_current_noether_closure_available_now = bool(
        charge_summary["updated_pack_exact_charge_current_noether_closure_available_now"]
    )
    updated_pack_noether_current_object_split_explicit = bool(
        charge_summary["frozen_jeff_zero_class_retained"]
        and sign_base.hit(charge_formulas["object_split"], "J_Noether^mu[Q] closes") is not None
    )
    updated_pack_low_order_jeff0_same_field_zero_formula_derived = bool(
        frozen_jeff_same_field_zero_retained
        and updated_pack_exact_charge_current_noether_closure_available_now
        and updated_pack_noether_current_object_split_explicit
    )
    updated_pack_exact_low_order_jeff0_formula_available_now = bool(
        updated_pack_low_order_jeff0_same_field_zero_formula_derived
    )
    updated_pack_proxy_signed_density_support_verdict_passed = False
    updated_pack_proxy_signed_density_no_go_verdict_passed = bool(
        updated_pack_exact_low_order_jeff0_formula_available_now
    )
    updated_pack_low_order_jeff0_derivation_closes_third_missing_primitive_now = bool(
        updated_pack_exact_low_order_jeff0_formula_available_now
    )
    updated_pack_exact_source_theorem_closeout_primary_refresh_required = bool(
        updated_pack_exact_low_order_jeff0_formula_available_now
    )
    exact_source_theorem_derived_now = False
    blind_vector_observable_gate_still_blocked = True
    farther_hybrid_continuation_reopen_required_now = False

    rows = [
        sign_base.row(
            "updated_pack_exact_low_order_jeff0_derivation_audit_selected",
            "pass" if updated_pack_exact_low_order_jeff0_derivation_audit_selected else "reject",
            "updated-pack exact low-order J_eff^0 derivation audit selected",
            sign_base.truth(updated_pack_exact_low_order_jeff0_derivation_audit_selected),
            "The charge-current derivation gate already promoted low-order J_eff^0 as the next honest theorem object.",
        ),
        sign_base.row(
            "frozen_jeff_same_field_zero_retained",
            "pass" if frozen_jeff_same_field_zero_retained else "reject",
            "frozen J_eff same-field zero retained",
            sign_base.truth(frozen_jeff_same_field_zero_retained),
            "The older direct J_eff branch already fixed the same-field photon-source object to zero under the current pack.",
        ),
        sign_base.row(
            "updated_pack_exact_charge_current_noether_closure_available_now",
            "pass" if updated_pack_exact_charge_current_noether_closure_available_now else "reject",
            "updated-pack exact charge-current / Noether-current closure available now",
            sign_base.truth(updated_pack_exact_charge_current_noether_closure_available_now),
            "The updated-pack branch now exposes the exact conserved background Noether current explicitly.",
        ),
        sign_base.row(
            "updated_pack_noether_current_object_split_explicit",
            "pass" if updated_pack_noether_current_object_split_explicit else "reject",
            "updated-pack Noether current / J_eff object split explicit",
            sign_base.truth(updated_pack_noether_current_object_split_explicit),
            "The charge-current derivation already states that J_Noether^mu[Q] and J_eff^mu[a;Q] are different theorem objects.",
        ),
        sign_base.row(
            "updated_pack_low_order_jeff0_same_field_zero_formula_derived",
            "pass" if updated_pack_low_order_jeff0_same_field_zero_formula_derived else "reject",
            "updated-pack low-order J_eff^0 same-field zero formula derived",
            sign_base.truth(updated_pack_low_order_jeff0_same_field_zero_formula_derived),
            "Combining the retained same-field zero with the explicit object split closes the low-order photon-source formula as zero under the current updated-pack.",
        ),
        sign_base.row(
            "updated_pack_exact_low_order_jeff0_formula_available_now",
            "pass" if updated_pack_exact_low_order_jeff0_formula_available_now else "reject",
            "updated-pack exact low-order J_eff^0 formula available now",
            sign_base.truth(updated_pack_exact_low_order_jeff0_formula_available_now),
            "The exact low-order formula is now explicit for the same-field photon-source object: it is zero under the current updated-pack.",
        ),
        sign_base.row(
            "updated_pack_proxy_signed_density_support_verdict_passed",
            "pass" if updated_pack_proxy_signed_density_support_verdict_passed else "reject",
            "updated-pack proxy signed-density support verdict passed",
            sign_base.truth(updated_pack_proxy_signed_density_support_verdict_passed),
            "The proxy |f_0|^2 - |f_L|^2 does not survive as the exact J_eff^0 formula because the exact source object remains zero under the same-field current pack.",
        ),
        sign_base.row(
            "updated_pack_proxy_signed_density_no_go_verdict_passed",
            "pass" if updated_pack_proxy_signed_density_no_go_verdict_passed else "reject",
            "updated-pack proxy signed-density no-go verdict passed",
            sign_base.truth(updated_pack_proxy_signed_density_no_go_verdict_passed),
            "Once the exact same-field J_eff^0 formula is explicit and zero, the proxy strong-support route becomes theorem-level no-go under the current pack.",
        ),
        sign_base.row(
            "updated_pack_low_order_jeff0_derivation_closes_third_missing_primitive_now",
            "pass" if updated_pack_low_order_jeff0_derivation_closes_third_missing_primitive_now else "reject",
            "updated-pack low-order J_eff^0 derivation closes third missing primitive now",
            sign_base.truth(updated_pack_low_order_jeff0_derivation_closes_third_missing_primitive_now),
            "The third missing object is no longer absent: low-order J_eff^0 is now fixed as a zero same-field photon-source formula under the current updated-pack.",
        ),
        sign_base.row(
            "updated_pack_exact_source_theorem_closeout_primary_refresh_required",
            "pass" if updated_pack_exact_source_theorem_closeout_primary_refresh_required else "reject",
            "updated-pack exact source-theorem closeout primary refresh required",
            sign_base.truth(updated_pack_exact_source_theorem_closeout_primary_refresh_required),
            "With background expansion, charge current, and low-order J_eff^0 now explicit, the next honest lane is exact source-theorem closeout.",
        ),
        sign_base.row(
            "exact_source_theorem_derived_now",
            "pass" if exact_source_theorem_derived_now else "reject",
            "exact source theorem derived now",
            sign_base.truth(exact_source_theorem_derived_now),
            "This branch closes low-order J_eff^0 only; the full source-theorem closeout remains the next branch.",
        ),
        sign_base.row(
            "blind_vector_observable_gate_still_blocked",
            "pass" if blind_vector_observable_gate_still_blocked else "reject",
            "blind vector observable gate still blocked",
            sign_base.truth(blind_vector_observable_gate_still_blocked),
            "Blind-vector direct computation remains downstream until the exact source theorem itself is synchronized.",
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
        "updated_pack_exact_low_order_jeff0_derivation_audit_selected": updated_pack_exact_low_order_jeff0_derivation_audit_selected,
        "frozen_jeff_same_field_zero_retained": frozen_jeff_same_field_zero_retained,
        "updated_pack_exact_charge_current_noether_closure_available_now": updated_pack_exact_charge_current_noether_closure_available_now,
        "updated_pack_noether_current_object_split_explicit": updated_pack_noether_current_object_split_explicit,
        "updated_pack_low_order_jeff0_same_field_zero_formula_derived": updated_pack_low_order_jeff0_same_field_zero_formula_derived,
        "updated_pack_exact_low_order_jeff0_formula_available_now": updated_pack_exact_low_order_jeff0_formula_available_now,
        "updated_pack_proxy_signed_density_support_verdict_passed": updated_pack_proxy_signed_density_support_verdict_passed,
        "updated_pack_proxy_signed_density_no_go_verdict_passed": updated_pack_proxy_signed_density_no_go_verdict_passed,
        "updated_pack_low_order_jeff0_derivation_closes_third_missing_primitive_now": updated_pack_low_order_jeff0_derivation_closes_third_missing_primitive_now,
        "updated_pack_exact_source_theorem_closeout_primary_refresh_required": updated_pack_exact_source_theorem_closeout_primary_refresh_required,
        "exact_source_theorem_derived_now": exact_source_theorem_derived_now,
        "blind_vector_observable_gate_still_blocked": blind_vector_observable_gate_still_blocked,
        "farther_hybrid_continuation_reopen_required_now": farther_hybrid_continuation_reopen_required_now,
        "selected_primary_pack_update_surface": "updated_pack_exact_low_order_jeff0_same_field_zero_derived",
        "selected_secondary_pack_update_surface": "updated_pack_exact_source_theorem_closeout",
        "selected_reserve_completion_lane": "blind_vector_after_source_theorem_closeout",
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": FOLLOWUP_ROUTE_NAME,
        "selected_followup_route_or_none": FOLLOWUP_ROUTE,
        "physical_reject_required": False,
    }

    declaration_payload = sign_base.payload(
        "8.7.56.2601",
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
                "charge_derivation": sign_base.display_path(CHARGE_DERIVATION),
                "frozen_jeff_split": sign_base.display_path(FROZEN_JEFF_SPLIT),
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
            "overall_status": "vector_qball_form_factor_updated_pack_low_order_jeff0_derivation_declared",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "formulas": build_formulae(frozen_formulas, charge_formulas),
            "hits": {
                "status_branch_hit": sign_base.hit(status_text, "8.7.56.2591"),
                "roadmap_branch_hit": sign_base.hit(roadmap_text, ".2587-.2590"),
                "current_problem_hit": sign_base.hit(current_problem_text, "exact low-order `J_{\\rm eff}^0` formula"),
                "current_status_hit": sign_base.hit(current_status_text, "exact low-order `J_{\\rm eff}^0` formula"),
                "unified_roadmap_hit": sign_base.hit(unified_text, ".2583-.2590"),
                "long_roadmap_hit": sign_base.hit(long_text, ".2587-.2590"),
                "part5_same_field_zero_hit": sign_base.hit(part5_text, "same-field on-shell zero"),
                "part5_charge_gap_hit": sign_base.hit(part5_text, "exact charge-current / Noether-current closure"),
            },
            "inference": {
                "exact_low_order_formula_is_same_field_object": True,
                "why": (
                    "The updated-pack branch derives an exact conserved background Noether "
                    "current, but the photon-side source object remains the same-field J_eff. "
                    "Because the older same-field zero classification still holds and no new "
                    "microscopic matter/rotational source functional was added here, the exact "
                    "low-order same-field J_eff^0 formula remains zero under the current pack."
                ),
            },
        },
    )
    declaration_paths = write_artifact("declaration_gate", declaration_payload)

    route_payload = {
        "generated_utc": sign_base.now_iso(),
        "phase": {
            "phase": 8,
            "step": "8.7.56.2602",
            "name": STEP_NAME + " route sync",
        },
        "inputs": declaration_paths,
        "rows": rows,
        "summary": summary,
        "decision": {
            "overall_status": "vector_qball_form_factor_updated_pack_low_order_jeff0_derivation_route_synced",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        "evidence": {
            "formulas": build_formulae(frozen_formulas, charge_formulas),
            "disposition": {
                "exact_low_order_jeff0_formula_available_now": updated_pack_exact_low_order_jeff0_formula_available_now,
                "proxy_signed_density_no_go_verdict_passed": updated_pack_proxy_signed_density_no_go_verdict_passed,
                "source_theorem_closeout_primary_required": updated_pack_exact_source_theorem_closeout_primary_refresh_required,
                "direct_blind_vector_still_blocked": blind_vector_observable_gate_still_blocked,
            },
        },
    }
    route_paths = write_artifact("route_sync", route_payload)

    print("[ok] updated-pack exact low-order J_eff^0 derivation artifacts written")
    print(f"  declaration_gate: {declaration_paths['json']}")
    print(f"  route_sync: {route_paths['json']}")


if __name__ == "__main__":
    main()
