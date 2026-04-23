#!/usr/bin/env python3
"""Generate 8.7.56.2559-.2562 updated-pack low-order J_eff^0 closeout audit artifacts."""

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
        "8.7.56.2555-2558",
        "updated_pack_charge_current_closeout_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
BACKGROUND_CLOSEOUT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.2543-2546",
        "updated_pack_background_expansion_closeout_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
CHARGE_CLOSEOUT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.2551-2554",
        "updated_pack_charge_current_closeout_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
LOW_ORDER_REFRESH = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.2503-2506",
        "updated_pack_exact_low_order_jeff0_refresh_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.2559-2562"
STEP_NAME = "Trial-2 numeric alpha vector Q-ball form-factor updated-pack low-order J_eff^0 closeout audit"
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_low_order_jeff0_closeout_audit",
    prefix="q",
)

PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_charge_current_"
    "closeout_audited_low_order_jeff0_primary_blind_vector_reserve_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_low_order_"
    "jeff0_closeout_audited_blind_vector_gate"
)
NEXT_ROUTE_NAME = "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_low_order_jeff0_closeout_gate_blind_vector_refresh"
NEXT_ROUTE = "8.7.56.2563"
FOLLOWUP_ROUTE_NAME = "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_blind_vector_refresh_audit"
FOLLOWUP_ROUTE = "8.7.56.2567"


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


# 関数: low-order closeout audit で使う式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used in the updated-pack low-order J_eff^0 closeout audit."""
    return {
        "closeout_order": "background expansion closeout -> charge-current closeout -> low-order J_eff^0 closeout -> blind-vector refresh",
        "support_rule": "J_eff^0 low-order == |f_0|^2 - |f_L|^2 => proxy strong support",
        "no_go_rule": "J_eff^0 low-order != |f_0|^2 - |f_L|^2 => proxy route no-go",
        "why": "Once the first two closeout objects are explicit, the remaining theorem-side object is the exact low-order J_eff^0 formula.",
    }


# 関数: `.2559-.2562` を実行する。

def main() -> None:
    """Execute the updated-pack low-order J_eff^0 closeout audit."""
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
        BACKGROUND_CLOSEOUT,
        CHARGE_CLOSEOUT,
        LOW_ORDER_REFRESH,
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
    background_summary = sign_base.read_json(BACKGROUND_CLOSEOUT)["summary"]
    charge_summary = sign_base.read_json(CHARGE_CLOSEOUT)["summary"]
    low_refresh_summary = sign_base.read_json(LOW_ORDER_REFRESH)["summary"]

    updated_pack_low_order_jeff0_closeout_audit_selected = bool(
        prior_summary["gate_b_updated_pack_low_order_jeff0_closeout_primary_selected"]
        and not prior_summary["gate_c_blind_vector_computation_primary_admissible_now"]
    )
    retained_proxy_signed_density_available = bool(
        low_refresh_summary["retained_proxy_signed_density_available"]
    )
    retained_proxy_signed_density_only = bool(
        low_refresh_summary["retained_proxy_signed_density_only"]
    )
    updated_pack_low_order_jeff0_support_surface_explicit = bool(
        low_refresh_summary["updated_pack_low_order_jeff0_support_surface_explicit"]
    )
    updated_pack_low_order_jeff0_no_go_surface_explicit = bool(
        low_refresh_summary["updated_pack_low_order_jeff0_no_go_surface_explicit"]
    )
    updated_pack_background_expansion_closeout_surface_available = bool(
        background_summary["updated_pack_background_expansion_closeout_target_surface_explicit"]
    )
    updated_pack_charge_current_closeout_surface_available = bool(
        charge_summary["updated_pack_charge_current_closeout_target_surface_explicit"]
    )
    updated_pack_low_order_jeff0_closeout_target_surface_explicit = bool(
        updated_pack_low_order_jeff0_closeout_audit_selected
        and retained_proxy_signed_density_available
        and updated_pack_low_order_jeff0_support_surface_explicit
        and updated_pack_low_order_jeff0_no_go_surface_explicit
        and updated_pack_background_expansion_closeout_surface_available
        and updated_pack_charge_current_closeout_surface_available
    )
    updated_pack_low_order_jeff0_closeout_machine_readable_now = bool(
        updated_pack_low_order_jeff0_closeout_target_surface_explicit
        and background_summary["updated_pack_background_expansion_closeout_machine_readable_now"]
        and charge_summary["updated_pack_charge_current_closeout_machine_readable_now"]
        and low_refresh_summary["updated_pack_low_order_jeff0_refresh_machine_readable_now"]
        and not low_refresh_summary["updated_pack_exact_low_order_jeff0_formula_available_now"]
    )
    updated_pack_exact_low_order_jeff0_formula_available_now = bool(
        low_refresh_summary["updated_pack_exact_low_order_jeff0_formula_available_now"]
    )
    updated_pack_exact_charge_current_noether_closure_available_now = bool(
        charge_summary["updated_pack_exact_charge_current_noether_closure_available_now"]
    )
    updated_pack_exact_qball_background_expansion_available_now = bool(
        background_summary["updated_pack_exact_qball_background_expansion_available_now"]
    )
    updated_pack_low_order_jeff0_closeout_strong_support_verdict_derivable_now = bool(
        updated_pack_exact_low_order_jeff0_formula_available_now
        and updated_pack_exact_charge_current_noether_closure_available_now
        and updated_pack_exact_qball_background_expansion_available_now
    )
    updated_pack_low_order_jeff0_closeout_no_go_verdict_derivable_now = bool(
        updated_pack_low_order_jeff0_closeout_strong_support_verdict_derivable_now
    )
    updated_pack_blind_vector_refresh_followup_required = bool(
        updated_pack_low_order_jeff0_closeout_machine_readable_now
        and not updated_pack_exact_low_order_jeff0_formula_available_now
    )
    updated_pack_low_order_jeff0_closeout_closes_missing_action_blocker_now = False
    blind_vector_observable_gate_still_blocked = bool(
        not prior_summary["gate_c_blind_vector_computation_primary_admissible_now"]
    )
    farther_hybrid_continuation_reopen_required_now = bool(
        prior_summary["farther_hybrid_continuation_reopen_required_now"]
    )

    rows = [
        sign_base.row(
            "updated_pack_low_order_jeff0_closeout_audit_selected",
            "pass" if updated_pack_low_order_jeff0_closeout_audit_selected else "reject",
            "updated-pack low-order J_eff^0 closeout audit selected",
            sign_base.truth(updated_pack_low_order_jeff0_closeout_audit_selected),
            "The charge-current closeout gate already promoted low-order J_eff^0 closeout as the next honest exact object.",
        ),
        sign_base.row(
            "retained_proxy_signed_density_available",
            "pass" if retained_proxy_signed_density_available else "reject",
            "retained proxy signed density available",
            sign_base.truth(retained_proxy_signed_density_available),
            "The proxy |f_0|^2 - |f_L|^2 surface remains the comparison target for theorem-level low-order closeout.",
        ),
        sign_base.row(
            "retained_proxy_signed_density_only",
            "pass" if retained_proxy_signed_density_only else "reject",
            "retained proxy signed density still proxy-only",
            sign_base.truth(retained_proxy_signed_density_only),
            "The proxy surface is still not promoted to an exact action-level theorem inside the current pack.",
        ),
        sign_base.row(
            "updated_pack_low_order_jeff0_support_surface_explicit",
            "pass" if updated_pack_low_order_jeff0_support_surface_explicit else "reject",
            "updated-pack low-order J_eff^0 support surface explicit",
            sign_base.truth(updated_pack_low_order_jeff0_support_surface_explicit),
            "The strong-support branch remains explicit from the earlier low-order discriminator audit.",
        ),
        sign_base.row(
            "updated_pack_low_order_jeff0_no_go_surface_explicit",
            "pass" if updated_pack_low_order_jeff0_no_go_surface_explicit else "reject",
            "updated-pack low-order J_eff^0 no-go surface explicit",
            sign_base.truth(updated_pack_low_order_jeff0_no_go_surface_explicit),
            "The no-go branch also remains explicit, so closeout does not rely on vague proxy prose.",
        ),
        sign_base.row(
            "updated_pack_background_expansion_closeout_surface_available",
            "pass" if updated_pack_background_expansion_closeout_surface_available else "reject",
            "updated-pack background-expansion closeout surface available",
            sign_base.truth(updated_pack_background_expansion_closeout_surface_available),
            "The first exact closeout object is already explicit, so low-order closeout can sit on top of it without reopening the first primitive.",
        ),
        sign_base.row(
            "updated_pack_charge_current_closeout_surface_available",
            "pass" if updated_pack_charge_current_closeout_surface_available else "reject",
            "updated-pack charge-current closeout surface available",
            sign_base.truth(updated_pack_charge_current_closeout_surface_available),
            "The second exact closeout object is already explicit, so the remaining theorem-side target is low-order J_eff^0 itself.",
        ),
        sign_base.row(
            "updated_pack_low_order_jeff0_closeout_target_surface_explicit",
            "pass" if updated_pack_low_order_jeff0_closeout_target_surface_explicit else "reject",
            "updated-pack low-order J_eff^0 closeout target surface explicit",
            sign_base.truth(updated_pack_low_order_jeff0_closeout_target_surface_explicit),
            "Background-expansion closeout, charge-current closeout, and the support/no-go split now meet on one low-order closeout target.",
        ),
        sign_base.row(
            "updated_pack_low_order_jeff0_closeout_machine_readable_now",
            "pass" if updated_pack_low_order_jeff0_closeout_machine_readable_now else "reject",
            "updated-pack low-order J_eff^0 closeout machine-readable now",
            sign_base.truth(updated_pack_low_order_jeff0_closeout_machine_readable_now),
            "The remaining theorem-side formula gap is now localized on one explicit closeout object rather than a generic synthesis label.",
        ),
        sign_base.row(
            "updated_pack_exact_low_order_jeff0_formula_available_now",
            "pass" if updated_pack_exact_low_order_jeff0_formula_available_now else "reject",
            "updated-pack exact low-order J_eff^0 formula available now",
            sign_base.truth(updated_pack_exact_low_order_jeff0_formula_available_now),
            "The canon still lacks the first-principles low-order J_eff^0 formula itself.",
        ),
        sign_base.row(
            "updated_pack_exact_charge_current_noether_closure_available_now",
            "pass" if updated_pack_exact_charge_current_noether_closure_available_now else "reject",
            "updated-pack exact charge-current / Noether-current closure available now",
            sign_base.truth(updated_pack_exact_charge_current_noether_closure_available_now),
            "Low-order closeout still depends on the absent exact charge-current / Noether-current theorem object.",
        ),
        sign_base.row(
            "updated_pack_exact_qball_background_expansion_available_now",
            "pass" if updated_pack_exact_qball_background_expansion_available_now else "reject",
            "updated-pack exact Q-ball background expansion available now",
            sign_base.truth(updated_pack_exact_qball_background_expansion_available_now),
            "Low-order closeout still depends on the absent exact Q-ball background expansion.",
        ),
        sign_base.row(
            "updated_pack_low_order_jeff0_closeout_strong_support_verdict_derivable_now",
            "pass" if updated_pack_low_order_jeff0_closeout_strong_support_verdict_derivable_now else "reject",
            "updated-pack low-order J_eff^0 closeout strong-support verdict derivable now",
            sign_base.truth(updated_pack_low_order_jeff0_closeout_strong_support_verdict_derivable_now),
            "A theorem-level support verdict requires the exact low-order formula plus the exact background-expansion and charge-current objects together.",
        ),
        sign_base.row(
            "updated_pack_low_order_jeff0_closeout_no_go_verdict_derivable_now",
            "pass" if updated_pack_low_order_jeff0_closeout_no_go_verdict_derivable_now else "reject",
            "updated-pack low-order J_eff^0 closeout no-go verdict derivable now",
            sign_base.truth(updated_pack_low_order_jeff0_closeout_no_go_verdict_derivable_now),
            "A theorem-level no-go verdict requires the same exact stack, not merely the absence of support prose.",
        ),
        sign_base.row(
            "updated_pack_blind_vector_refresh_followup_required",
            "pass" if updated_pack_blind_vector_refresh_followup_required else "reject",
            "updated-pack blind-vector refresh followup required",
            sign_base.truth(updated_pack_blind_vector_refresh_followup_required),
            "Once low-order closeout is explicit and machine-readable, the honest downstream lane is blind-vector refresh.",
        ),
        sign_base.row(
            "updated_pack_low_order_jeff0_closeout_closes_missing_action_blocker_now",
            "pass" if updated_pack_low_order_jeff0_closeout_closes_missing_action_blocker_now else "reject",
            "updated-pack low-order J_eff^0 closeout closes missing-action blocker now",
            sign_base.truth(updated_pack_low_order_jeff0_closeout_closes_missing_action_blocker_now),
            "This audit localizes the low-order formula gap honestly, but the theorem object itself remains absent.",
        ),
        sign_base.row(
            "blind_vector_observable_gate_still_blocked",
            "pass" if blind_vector_observable_gate_still_blocked else "reject",
            "blind vector observable gate still blocked",
            sign_base.truth(blind_vector_observable_gate_still_blocked),
            "Direct blind-vector computation remains blocked until the exact low-order formula stack moves first.",
        ),
        sign_base.row(
            "farther_hybrid_continuation_reopen_required_now",
            "pass" if farther_hybrid_continuation_reopen_required_now else "reject",
            "farther hybrid continuation reopen required now",
            sign_base.truth(farther_hybrid_continuation_reopen_required_now),
            "Extra q-range evidence remains unnecessary because the blocker is still theorem-side and now sharpened to low-order closeout.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "retained_scalar_residual_rel": float(prior_summary["retained_scalar_residual_rel"]),
        "updated_pack_low_order_jeff0_closeout_audit_selected": updated_pack_low_order_jeff0_closeout_audit_selected,
        "retained_proxy_signed_density_available": retained_proxy_signed_density_available,
        "retained_proxy_signed_density_only": retained_proxy_signed_density_only,
        "updated_pack_low_order_jeff0_support_surface_explicit": updated_pack_low_order_jeff0_support_surface_explicit,
        "updated_pack_low_order_jeff0_no_go_surface_explicit": updated_pack_low_order_jeff0_no_go_surface_explicit,
        "updated_pack_background_expansion_closeout_surface_available": updated_pack_background_expansion_closeout_surface_available,
        "updated_pack_charge_current_closeout_surface_available": updated_pack_charge_current_closeout_surface_available,
        "updated_pack_low_order_jeff0_closeout_target_surface_explicit": updated_pack_low_order_jeff0_closeout_target_surface_explicit,
        "updated_pack_low_order_jeff0_closeout_machine_readable_now": updated_pack_low_order_jeff0_closeout_machine_readable_now,
        "updated_pack_exact_low_order_jeff0_formula_available_now": updated_pack_exact_low_order_jeff0_formula_available_now,
        "updated_pack_exact_charge_current_noether_closure_available_now": updated_pack_exact_charge_current_noether_closure_available_now,
        "updated_pack_exact_qball_background_expansion_available_now": updated_pack_exact_qball_background_expansion_available_now,
        "updated_pack_low_order_jeff0_closeout_strong_support_verdict_derivable_now": updated_pack_low_order_jeff0_closeout_strong_support_verdict_derivable_now,
        "updated_pack_low_order_jeff0_closeout_no_go_verdict_derivable_now": updated_pack_low_order_jeff0_closeout_no_go_verdict_derivable_now,
        "updated_pack_blind_vector_refresh_followup_required": updated_pack_blind_vector_refresh_followup_required,
        "updated_pack_low_order_jeff0_closeout_closes_missing_action_blocker_now": updated_pack_low_order_jeff0_closeout_closes_missing_action_blocker_now,
        "blind_vector_observable_gate_still_blocked": blind_vector_observable_gate_still_blocked,
        "farther_hybrid_continuation_reopen_required_now": farther_hybrid_continuation_reopen_required_now,
        "selected_primary_pack_update_surface": "updated_pack_low_order_jeff0_closeout",
        "selected_secondary_pack_update_surface": "updated_pack_blind_vector_refresh",
        "selected_reserve_completion_lane": "farther_hybrid_extra_q_range_only",
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": FOLLOWUP_ROUTE_NAME,
        "selected_followup_route_or_none": FOLLOWUP_ROUTE,
        "physical_reject_required": False,
    }

    declaration_payload = sign_base.payload(
        "8.7.56.2561",
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
                "background_closeout": sign_base.display_path(BACKGROUND_CLOSEOUT),
                "charge_closeout": sign_base.display_path(CHARGE_CLOSEOUT),
                "low_order_refresh": sign_base.display_path(LOW_ORDER_REFRESH),
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
            "overall_status": "vector_qball_form_factor_updated_pack_low_order_jeff0_closeout_audit_declared",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "formulas": build_formulae(),
            "hits": {
                "status_branch_hit": sign_base.hit(status_text, ".2559-.2562"),
                "roadmap_branch_hit": sign_base.hit(roadmap_text, ".2559-.2562"),
                "current_problem_hit": sign_base.hit(current_problem_text, "updated-pack low-order J_eff^0 closeout audit"),
                "current_status_hit": sign_base.hit(current_status_text, "updated-pack low-order J_eff^0 closeout audit"),
                "unified_roadmap_hit": sign_base.hit(unified_text, ".2551-.2558"),
                "long_roadmap_hit": sign_base.hit(long_text, ".2551-.2558"),
                "part5_hit": sign_base.hit(part5_text, "charge-current closeout lane"),
            },
        },
    )
    declaration_paths = write_artifact("declaration_gate", declaration_payload)

    route_payload = {
        "generated_utc": sign_base.now_iso(),
        "phase": {
            "phase": 8,
            "step": "8.7.56.2562",
            "name": STEP_NAME + " route sync",
        },
        "inputs": declaration_paths,
        "rows": rows,
        "summary": summary,
        "decision": {
            "overall_status": "vector_qball_form_factor_updated_pack_low_order_jeff0_closeout_route_synced",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        "evidence": {
            "formulas": build_formulae(),
            "disposition": {
                "low_order_jeff0_closeout_surface_explicit": updated_pack_low_order_jeff0_closeout_target_surface_explicit,
                "low_order_jeff0_closeout_machine_readable_now": updated_pack_low_order_jeff0_closeout_machine_readable_now,
                "blind_vector_followup_required": updated_pack_blind_vector_refresh_followup_required,
                "blind_vector_still_blocked": blind_vector_observable_gate_still_blocked,
            },
        },
    }
    route_paths = write_artifact("route_sync", route_payload)

    print("[ok] updated-pack low-order J_eff^0 closeout audit artifacts written")
    print(f"  declaration_gate: {declaration_paths['json']}")
    print(f"  route_sync: {route_paths['json']}")


if __name__ == "__main__":
    main()
