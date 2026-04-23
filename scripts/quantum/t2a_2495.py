#!/usr/bin/env python3
"""Generate 8.7.56.2495-.2498 updated-pack exact charge-current refresh audit artifacts."""

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
        "8.7.56.2491-2494",
        "updated_pack_qball_background_expansion_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
BACKGROUND_AUDIT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.2487-2490",
        "updated_pack_exact_qball_background_expansion_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
JEFF_PREREQ_AUDIT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.2471-2474",
        "updated_pack_exact_jeff_prerequisite_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
OLD_CHARGE_AUDIT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.1491-1494",
        "charge_current_closure",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

NEXT_STEPS = Path(r"C:\Users\ogawa\Downloads\trial2_vector_qball_next_steps_20260327.md")

STEP_TAG = "8.7.56.2495-2498"
STEP_NAME = "Trial-2 numeric alpha vector Q-ball form-factor updated-pack exact charge-current / Noether-current refresh audit"
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_exact_charge_current_refresh_audit",
    prefix="q",
)

PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_background_"
    "expansion_audited_charge_current_primary_low_order_jeff0_secondary_"
    "blind_vector_reserve_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_charge_"
    "current_audited_low_order_jeff0_gate"
)
NEXT_ROUTE_NAME = "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_charge_current_gate_low_order_jeff0_refresh"
NEXT_ROUTE = "8.7.56.2499"
FOLLOWUP_ROUTE_NAME = "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_exact_low_order_jeff0_refresh_audit"
FOLLOWUP_ROUTE = "8.7.56.2503"


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


# 関数: charge-current refresh audit で使う式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used in the updated-pack charge-current refresh audit."""
    return {
        "source_theorem_target": "L \\supset a_mu J_eff^mu[P^Qball]",
        "continuity_surface": "partial_mu J^mu = 0",
        "adopted_u1_identity": "Q-ball Noether charge = adopted U(1) charge",
        "proxy_density": "J_eff^0 low-order ?= |f_0|^2 - |f_L|^2",
        "refresh_order": "charge-current / Noether-current closure -> low-order J_eff^0 synthesis -> blind vector refresh",
    }


# 関数: `.2495-.2498` を実行する。

def main() -> None:
    """Execute the updated-pack exact charge-current refresh audit."""
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
        BACKGROUND_AUDIT,
        JEFF_PREREQ_AUDIT,
        OLD_CHARGE_AUDIT,
        NEXT_STEPS,
    ):
        sign_base.require(path)

    status_text = sign_base.read_text(STATUS)
    roadmap_text = sign_base.read_text(ROADMAP)
    current_problem_text = sign_base.read_text(CURRENT_PROBLEM)
    current_status_text = sign_base.read_text(CURRENT_STATUS)
    unified_text = sign_base.read_text(UNIFIED_ROADMAP)
    long_text = sign_base.read_text(LONG_ROADMAP)
    part5_text = sign_base.read_text(PART5)
    next_steps_text = sign_base.read_text(NEXT_STEPS)

    prior_gate_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    background_audit_summary = sign_base.read_json(BACKGROUND_AUDIT)["summary"]
    jeff_prereq_summary = sign_base.read_json(JEFF_PREREQ_AUDIT)["summary"]
    old_charge_summary = sign_base.read_json(OLD_CHARGE_AUDIT)["summary"]

    updated_pack_exact_charge_current_refresh_audit_selected = bool(
        prior_gate_summary["gate_b_updated_pack_exact_charge_current_primary_selected"]
        and not prior_gate_summary["gate_c_blind_vector_computation_primary_admissible_now"]
    )
    retained_generic_u1_continuity_surface_available = bool(
        old_charge_summary["generic_u1_continuity_surface_available"]
    )
    retained_qball_charge_mapping_statement_available = bool(
        old_charge_summary["qball_charge_mapping_statement_available"]
    )
    retained_direct_qball_u1_identity_required = bool(
        old_charge_summary["direct_qball_u1_identity_required"]
    )
    retained_proxy_signed_density_available = bool(
        old_charge_summary["proxy_signed_density_available"]
    )
    retained_proxy_signed_density_only = bool(old_charge_summary["proxy_signed_density_only"])
    updated_pack_exact_charge_current_refresh_target_surface_explicit = bool(
        background_audit_summary["updated_pack_exact_qball_background_expansion_target_surface_explicit"]
        and retained_generic_u1_continuity_surface_available
        and retained_qball_charge_mapping_statement_available
        and sign_base.hit(next_steps_text, "### Step C.") is not None
        and sign_base.hit(next_steps_text, "J_eff^0") is not None
    )
    updated_pack_exact_charge_current_refresh_machine_readable_now = bool(
        updated_pack_exact_charge_current_refresh_target_surface_explicit
        and background_audit_summary["updated_pack_exact_qball_background_expansion_machine_readable_now"]
        and old_charge_summary["noether_current_gap_retained"]
        and not old_charge_summary["exact_charge_current_noether_closure_available"]
    )
    updated_pack_exact_charge_current_noether_closure_available_now = bool(
        old_charge_summary["exact_charge_current_noether_closure_available"]
    )
    updated_pack_exact_charge_current_refresh_closes_missing_action_blocker_now = False
    updated_pack_low_order_jeff0_secondary_refresh_required = bool(
        updated_pack_exact_charge_current_refresh_machine_readable_now
        and not jeff_prereq_summary["updated_pack_exact_low_order_jeff0_formula_available"]
    )
    blind_vector_observable_gate_still_blocked = bool(
        not prior_gate_summary["gate_c_blind_vector_computation_primary_admissible_now"]
    )
    farther_hybrid_continuation_reopen_required_now = False

    rows = [
        sign_base.row(
            "updated_pack_exact_charge_current_refresh_audit_selected",
            "pass" if updated_pack_exact_charge_current_refresh_audit_selected else "reject",
            "updated-pack exact charge-current / Noether-current refresh audit selected",
            sign_base.truth(updated_pack_exact_charge_current_refresh_audit_selected),
            "The background-expansion gate already promoted exact charge-current / Noether-current refresh as the next honest remaining closure.",
        ),
        sign_base.row(
            "retained_generic_u1_continuity_surface_available",
            "pass" if retained_generic_u1_continuity_surface_available else "reject",
            "retained generic U(1) continuity surface available",
            sign_base.truth(retained_generic_u1_continuity_surface_available),
            "Generic continuity remains part of the retained public pack and can be reused in the updated refresh lane.",
        ),
        sign_base.row(
            "retained_qball_charge_mapping_statement_available",
            "pass" if retained_qball_charge_mapping_statement_available else "reject",
            "retained Q-ball / adopted-U(1) charge mapping statement available",
            sign_base.truth(retained_qball_charge_mapping_statement_available),
            "The adopted-U(1) charge mapping for the Q-ball family is already frozen and remains reusable under the updated pack.",
        ),
        sign_base.row(
            "retained_direct_qball_u1_identity_required",
            "pass" if retained_direct_qball_u1_identity_required else "reject",
            "retained direct Q-ball / adopted-U(1) identity required",
            sign_base.truth(retained_direct_qball_u1_identity_required),
            "The older charge-operator normalization audit already fixed that extra multiplicative freedom is not available here.",
        ),
        sign_base.row(
            "retained_proxy_signed_density_available",
            "pass" if retained_proxy_signed_density_available else "reject",
            "retained proxy signed density available",
            sign_base.truth(retained_proxy_signed_density_available),
            "The proxy |f_0|^2 - |f_L|^2 surface remains available as a hint while the exact current theorem is still being refreshed.",
        ),
        sign_base.row(
            "retained_proxy_signed_density_only",
            "pass" if retained_proxy_signed_density_only else "reject",
            "retained proxy signed density still proxy-only",
            sign_base.truth(retained_proxy_signed_density_only),
            "The updated pack still does not promote the proxy signed density into an exact action-level charge-current theorem.",
        ),
        sign_base.row(
            "updated_pack_exact_charge_current_refresh_target_surface_explicit",
            "pass" if updated_pack_exact_charge_current_refresh_target_surface_explicit else "reject",
            "updated-pack exact charge-current refresh target surface explicit",
            sign_base.truth(updated_pack_exact_charge_current_refresh_target_surface_explicit),
            "Step C plus the retained continuity and adopted-U(1) identity already identify the exact closure target surface explicitly.",
        ),
        sign_base.row(
            "updated_pack_exact_charge_current_refresh_machine_readable_now",
            "pass" if updated_pack_exact_charge_current_refresh_machine_readable_now else "reject",
            "updated-pack exact charge-current refresh machine-readable now",
            sign_base.truth(updated_pack_exact_charge_current_refresh_machine_readable_now),
            "The missing closure is now localized on top of the explicit background-expansion target rather than left as a flat historical gap.",
        ),
        sign_base.row(
            "updated_pack_exact_charge_current_noether_closure_available_now",
            "pass" if updated_pack_exact_charge_current_noether_closure_available_now else "reject",
            "updated-pack exact charge-current / Noether-current closure available now",
            sign_base.truth(updated_pack_exact_charge_current_noether_closure_available_now),
            "The canon still does not close the restored vector branch into an exact charge-current / Noether-current theorem.",
        ),
        sign_base.row(
            "updated_pack_exact_charge_current_refresh_closes_missing_action_blocker_now",
            "pass" if updated_pack_exact_charge_current_refresh_closes_missing_action_blocker_now else "reject",
            "updated-pack exact charge-current refresh closes missing-action blocker now",
            sign_base.truth(updated_pack_exact_charge_current_refresh_closes_missing_action_blocker_now),
            "This audit makes the exact closure target explicit, but the theorem itself is still absent.",
        ),
        sign_base.row(
            "updated_pack_low_order_jeff0_secondary_refresh_required",
            "pass" if updated_pack_low_order_jeff0_secondary_refresh_required else "reject",
            "updated-pack low-order J_eff^0 secondary refresh required",
            sign_base.truth(updated_pack_low_order_jeff0_secondary_refresh_required),
            "Once exact charge-current closure is isolated, low-order J_eff^0 synthesis remains the next dependent exact object.",
        ),
        sign_base.row(
            "blind_vector_observable_gate_still_blocked",
            "pass" if blind_vector_observable_gate_still_blocked else "reject",
            "blind vector observable gate still blocked",
            sign_base.truth(blind_vector_observable_gate_still_blocked),
            "Blind vector computation remains downstream until charge-current closure and low-order J_eff^0 synthesis move first.",
        ),
        sign_base.row(
            "farther_hybrid_continuation_reopen_required_now",
            "pass" if farther_hybrid_continuation_reopen_required_now else "reject",
            "farther hybrid continuation reopen required now",
            sign_base.truth(farther_hybrid_continuation_reopen_required_now),
            "Extra q-range evidence remains unnecessary because the blocker is still theorem-side and now sharpened to charge-current closure.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "retained_scalar_residual_rel": float(prior_gate_summary["retained_scalar_residual_rel"]),
        "updated_pack_exact_charge_current_refresh_audit_selected": updated_pack_exact_charge_current_refresh_audit_selected,
        "retained_generic_u1_continuity_surface_available": retained_generic_u1_continuity_surface_available,
        "retained_qball_charge_mapping_statement_available": retained_qball_charge_mapping_statement_available,
        "retained_direct_qball_u1_identity_required": retained_direct_qball_u1_identity_required,
        "retained_proxy_signed_density_available": retained_proxy_signed_density_available,
        "retained_proxy_signed_density_only": retained_proxy_signed_density_only,
        "updated_pack_exact_charge_current_refresh_target_surface_explicit": updated_pack_exact_charge_current_refresh_target_surface_explicit,
        "updated_pack_exact_charge_current_refresh_machine_readable_now": updated_pack_exact_charge_current_refresh_machine_readable_now,
        "updated_pack_exact_charge_current_noether_closure_available_now": updated_pack_exact_charge_current_noether_closure_available_now,
        "updated_pack_exact_charge_current_refresh_closes_missing_action_blocker_now": updated_pack_exact_charge_current_refresh_closes_missing_action_blocker_now,
        "updated_pack_low_order_jeff0_secondary_refresh_required": updated_pack_low_order_jeff0_secondary_refresh_required,
        "blind_vector_observable_gate_still_blocked": blind_vector_observable_gate_still_blocked,
        "farther_hybrid_continuation_reopen_required_now": farther_hybrid_continuation_reopen_required_now,
        "selected_primary_pack_update_surface": "updated_pack_exact_charge_current_noether_refresh",
        "selected_secondary_pack_update_surface": "updated_pack_exact_low_order_jeff0_formula_synthesis",
        "selected_reserve_completion_lane": "blind_vector_after_low_order_jeff0_refresh",
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": FOLLOWUP_ROUTE_NAME,
        "selected_followup_route_or_none": FOLLOWUP_ROUTE,
        "physical_reject_required": False,
    }

    declaration_payload = sign_base.payload(
        "8.7.56.2497",
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
                "background_audit": sign_base.display_path(BACKGROUND_AUDIT),
                "jeff_prereq_audit": sign_base.display_path(JEFF_PREREQ_AUDIT),
                "old_charge_audit": sign_base.display_path(OLD_CHARGE_AUDIT),
                "next_steps": sign_base.display_path(NEXT_STEPS),
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
            "overall_status": "vector_qball_form_factor_updated_pack_charge_current_refresh_audit_declared",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "formulas": build_formulae(),
            "hits": {
                "status_branch_hit": sign_base.hit(status_text, "8.7.56.2495"),
                "roadmap_branch_hit": sign_base.hit(roadmap_text, ".2495-.2498"),
                "current_problem_hit": sign_base.hit(current_problem_text, "updated-pack exact charge-current / Noether-current refresh audit"),
                "current_status_hit": sign_base.hit(current_status_text, "updated-pack exact charge-current / Noether-current refresh audit"),
                "unified_roadmap_hit": sign_base.hit(unified_text, ".2487-.2494"),
                "long_roadmap_hit": sign_base.hit(long_text, ".2495-.2498"),
                "part5_hit": sign_base.hit(part5_text, "exact charge-current / Noether-current refresh audit"),
                "step_c_hit": sign_base.hit(next_steps_text, "### Step C."),
                "jeff0_hit": sign_base.hit(next_steps_text, "J_eff^0"),
                "continuity_hit": sign_base.hit(
                    sign_base.read_text(OLD_CHARGE_AUDIT),
                    "generic U(1) continuity",
                ),
                "proxy_hit": sign_base.hit(next_steps_text, "|f_0|^2 - |f_L|^2"),
            },
        },
    )
    declaration_paths = write_artifact("declaration_gate", declaration_payload)

    route_payload = {
        "generated_utc": sign_base.now_iso(),
        "phase": {
            "phase": 8,
            "step": "8.7.56.2498",
            "name": STEP_NAME + " route sync",
        },
        "inputs": declaration_paths,
        "rows": rows,
        "summary": summary,
        "decision": {
            "overall_status": "vector_qball_form_factor_updated_pack_charge_current_refresh_route_synced",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        "evidence": {
            "formulas": build_formulae(),
            "disposition": {
                "charge_current_target_surface_explicit": updated_pack_exact_charge_current_refresh_target_surface_explicit,
                "charge_current_refresh_machine_readable_now": updated_pack_exact_charge_current_refresh_machine_readable_now,
                "low_order_jeff0_secondary_required": updated_pack_low_order_jeff0_secondary_refresh_required,
                "blind_vector_still_downstream": blind_vector_observable_gate_still_blocked,
            },
        },
    }
    route_paths = write_artifact("route_sync", route_payload)

    print("[ok] updated-pack exact charge-current refresh audit artifacts written")
    print(f"  declaration_gate: {declaration_paths['json']}")
    print(f"  route_sync: {route_paths['json']}")


if __name__ == "__main__":
    main()
