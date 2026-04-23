#!/usr/bin/env python3
"""Generate 8.7.56.2503-.2506 updated-pack exact low-order J_eff^0 refresh audit artifacts."""

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
        "8.7.56.2499-2502",
        "updated_pack_charge_current_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
SOURCE_RULE_AUDIT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.2463-2466",
        "updated_pack_source_rule_audit",
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
CHARGE_REFRESH_AUDIT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.2495-2498",
        "updated_pack_exact_charge_current_refresh_audit",
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

STEP_TAG = "8.7.56.2503-2506"
STEP_NAME = "Trial-2 numeric alpha vector Q-ball form-factor updated-pack exact low-order J_eff^0 refresh audit"
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_exact_low_order_jeff0_refresh_audit",
    prefix="q",
)

PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_charge_"
    "current_audited_low_order_jeff0_primary_blind_vector_reserve_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_low_order_"
    "jeff0_audited_blind_vector_gate"
)
NEXT_ROUTE_NAME = "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_low_order_jeff0_gate_blind_vector_refresh"
NEXT_ROUTE = "8.7.56.2507"
FOLLOWUP_ROUTE_NAME = "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_blind_vector_refresh_audit"
FOLLOWUP_ROUTE = "8.7.56.2511"


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


# 関数: low-order J_eff^0 refresh audit で使う式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used in the updated-pack low-order J_eff^0 refresh audit."""
    return {
        "support_rule": "J_eff^0 low-order == |f_0|^2 - |f_L|^2 => proxy strong support",
        "no_go_rule": "J_eff^0 low-order != |f_0|^2 - |f_L|^2 => proxy route no-go",
        "source_target": "L \\supset a_mu J_eff^mu[P^Qball]",
        "refresh_order": "low-order J_eff^0 synthesis -> blind vector refresh -> residual-origin refresh",
    }


# 関数: `.2503-.2506` を実行する。

def main() -> None:
    """Execute the updated-pack exact low-order J_eff^0 refresh audit."""
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
        SOURCE_RULE_AUDIT,
        JEFF_PREREQ_AUDIT,
        CHARGE_REFRESH_AUDIT,
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

    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    source_rule_summary = sign_base.read_json(SOURCE_RULE_AUDIT)["summary"]
    jeff_prereq_summary = sign_base.read_json(JEFF_PREREQ_AUDIT)["summary"]
    charge_refresh_summary = sign_base.read_json(CHARGE_REFRESH_AUDIT)["summary"]
    old_charge_summary = sign_base.read_json(OLD_CHARGE_AUDIT)["summary"]

    updated_pack_exact_low_order_jeff0_refresh_audit_selected = bool(
        prior_summary["gate_b_updated_pack_low_order_jeff0_primary_selected"]
        and not prior_summary["gate_c_blind_vector_computation_primary_admissible_now"]
    )
    retained_proxy_signed_density_available = bool(
        charge_refresh_summary["retained_proxy_signed_density_available"]
    )
    retained_proxy_signed_density_only = bool(
        charge_refresh_summary["retained_proxy_signed_density_only"]
    )
    updated_pack_low_order_jeff0_support_surface_explicit = bool(
        source_rule_summary["updated_pack_source_rule_support_surface_explicit"]
        and sign_base.hit(next_steps_text, "もし `J_eff^0` が低次で `|f_0|^2 - |f_L|^2` に落ちるなら") is not None
    )
    updated_pack_low_order_jeff0_no_go_surface_explicit = bool(
        source_rule_summary["updated_pack_source_rule_no_go_surface_explicit"]
        and sign_base.hit(next_steps_text, "そうならないなら、現在の proxy route は no-go") is not None
    )
    updated_pack_low_order_jeff0_refresh_target_surface_explicit = bool(
        charge_refresh_summary["updated_pack_exact_charge_current_refresh_target_surface_explicit"]
        and source_rule_summary["updated_pack_low_order_jeff0_discriminator_surface_explicit"]
        and updated_pack_low_order_jeff0_support_surface_explicit
        and updated_pack_low_order_jeff0_no_go_surface_explicit
        and sign_base.hit(next_steps_text, "J_eff^0") is not None
    )
    updated_pack_low_order_jeff0_refresh_machine_readable_now = bool(
        updated_pack_low_order_jeff0_refresh_target_surface_explicit
        and charge_refresh_summary["updated_pack_exact_charge_current_refresh_machine_readable_now"]
        and prior_summary["gate_b_updated_pack_low_order_jeff0_primary_selected"]
        and not jeff_prereq_summary["updated_pack_exact_low_order_jeff0_formula_available"]
    )
    updated_pack_exact_low_order_jeff0_formula_available_now = bool(
        jeff_prereq_summary["updated_pack_exact_low_order_jeff0_formula_available"]
    )
    updated_pack_exact_charge_current_noether_closure_available_now = bool(
        old_charge_summary["exact_charge_current_noether_closure_available"]
    )
    updated_pack_exact_qball_background_expansion_available_now = bool(
        jeff_prereq_summary["updated_pack_explicit_qball_background_expansion_available"]
    )
    updated_pack_low_order_jeff0_strong_support_verdict_derivable_now = bool(
        updated_pack_exact_low_order_jeff0_formula_available_now
        and updated_pack_exact_charge_current_noether_closure_available_now
        and updated_pack_exact_qball_background_expansion_available_now
    )
    updated_pack_low_order_jeff0_no_go_verdict_derivable_now = bool(
        updated_pack_low_order_jeff0_strong_support_verdict_derivable_now
    )
    updated_pack_blind_vector_refresh_followup_required = bool(
        updated_pack_low_order_jeff0_refresh_machine_readable_now
        and not updated_pack_exact_low_order_jeff0_formula_available_now
    )
    updated_pack_exact_low_order_jeff0_refresh_closes_missing_action_blocker_now = False
    blind_vector_observable_gate_still_blocked = bool(
        not prior_summary["gate_c_blind_vector_computation_primary_admissible_now"]
    )
    farther_hybrid_continuation_reopen_required_now = bool(
        prior_summary["farther_hybrid_continuation_reopen_required_now"]
    )

    rows = [
        sign_base.row(
            "updated_pack_exact_low_order_jeff0_refresh_audit_selected",
            "pass" if updated_pack_exact_low_order_jeff0_refresh_audit_selected else "reject",
            "updated-pack exact low-order J_eff^0 refresh audit selected",
            sign_base.truth(updated_pack_exact_low_order_jeff0_refresh_audit_selected),
            "The charge-current gate already promoted low-order J_eff^0 synthesis as the next honest exact object.",
        ),
        sign_base.row(
            "retained_proxy_signed_density_available",
            "pass" if retained_proxy_signed_density_available else "reject",
            "retained proxy signed density available",
            sign_base.truth(retained_proxy_signed_density_available),
            "The proxy |f_0|^2 - |f_L|^2 surface remains available as the comparison target for low-order J_eff^0.",
        ),
        sign_base.row(
            "retained_proxy_signed_density_only",
            "pass" if retained_proxy_signed_density_only else "reject",
            "retained proxy signed density still proxy-only",
            sign_base.truth(retained_proxy_signed_density_only),
            "The current pack still does not promote the proxy signed density into an exact action-level theorem.",
        ),
        sign_base.row(
            "updated_pack_low_order_jeff0_support_surface_explicit",
            "pass" if updated_pack_low_order_jeff0_support_surface_explicit else "reject",
            "updated-pack low-order J_eff^0 support surface explicit",
            sign_base.truth(updated_pack_low_order_jeff0_support_surface_explicit),
            "The refreshed next-steps pack states the strong-support branch explicitly at low-order J_eff^0.",
        ),
        sign_base.row(
            "updated_pack_low_order_jeff0_no_go_surface_explicit",
            "pass" if updated_pack_low_order_jeff0_no_go_surface_explicit else "reject",
            "updated-pack low-order J_eff^0 no-go surface explicit",
            sign_base.truth(updated_pack_low_order_jeff0_no_go_surface_explicit),
            "The same pack also states the route-local no-go branch explicitly rather than leaving it implicit.",
        ),
        sign_base.row(
            "updated_pack_low_order_jeff0_refresh_target_surface_explicit",
            "pass" if updated_pack_low_order_jeff0_refresh_target_surface_explicit else "reject",
            "updated-pack low-order J_eff^0 refresh target surface explicit",
            sign_base.truth(updated_pack_low_order_jeff0_refresh_target_surface_explicit),
            "Step C plus the support/no-go split are enough to identify the missing low-order J_eff^0 theorem target explicitly.",
        ),
        sign_base.row(
            "updated_pack_low_order_jeff0_refresh_machine_readable_now",
            "pass" if updated_pack_low_order_jeff0_refresh_machine_readable_now else "reject",
            "updated-pack low-order J_eff^0 refresh machine-readable now",
            sign_base.truth(updated_pack_low_order_jeff0_refresh_machine_readable_now),
            "The missing formula is now localized on top of an explicit discriminator surface rather than a vague theorem phrase.",
        ),
        sign_base.row(
            "updated_pack_exact_low_order_jeff0_formula_available_now",
            "pass" if updated_pack_exact_low_order_jeff0_formula_available_now else "reject",
            "updated-pack exact low-order J_eff^0 formula available now",
            sign_base.truth(updated_pack_exact_low_order_jeff0_formula_available_now),
            "The canon still lacks a first-principles low-order J_eff^0 formula rather than a proxy reading.",
        ),
        sign_base.row(
            "updated_pack_exact_charge_current_noether_closure_available_now",
            "pass" if updated_pack_exact_charge_current_noether_closure_available_now else "reject",
            "updated-pack exact charge-current / Noether-current closure available now",
            sign_base.truth(updated_pack_exact_charge_current_noether_closure_available_now),
            "The exact charge-current theorem gap remains open, so low-order J_eff^0 cannot be promoted to an exact verdict yet.",
        ),
        sign_base.row(
            "updated_pack_exact_qball_background_expansion_available_now",
            "pass" if updated_pack_exact_qball_background_expansion_available_now else "reject",
            "updated-pack exact Q-ball background expansion available now",
            sign_base.truth(updated_pack_exact_qball_background_expansion_available_now),
            "The low-order formula still depends on the absent exact background expansion of the Q-ball branch.",
        ),
        sign_base.row(
            "updated_pack_low_order_jeff0_strong_support_verdict_derivable_now",
            "pass" if updated_pack_low_order_jeff0_strong_support_verdict_derivable_now else "reject",
            "updated-pack low-order J_eff^0 strong-support verdict derivable now",
            sign_base.truth(updated_pack_low_order_jeff0_strong_support_verdict_derivable_now),
            "Strong support requires the exact low-order formula, exact charge-current closure, and exact background expansion together.",
        ),
        sign_base.row(
            "updated_pack_low_order_jeff0_no_go_verdict_derivable_now",
            "pass" if updated_pack_low_order_jeff0_no_go_verdict_derivable_now else "reject",
            "updated-pack low-order J_eff^0 no-go verdict derivable now",
            sign_base.truth(updated_pack_low_order_jeff0_no_go_verdict_derivable_now),
            "A theorem-level no-go also requires the same exact low-order formula stack, not merely the absence of support prose.",
        ),
        sign_base.row(
            "updated_pack_blind_vector_refresh_followup_required",
            "pass" if updated_pack_blind_vector_refresh_followup_required else "reject",
            "updated-pack blind-vector refresh followup required",
            sign_base.truth(updated_pack_blind_vector_refresh_followup_required),
            "Once the low-order J_eff^0 formula gap is explicit and machine-readable, the honest downstream lane is blind-vector refresh.",
        ),
        sign_base.row(
            "updated_pack_exact_low_order_jeff0_refresh_closes_missing_action_blocker_now",
            "pass" if updated_pack_exact_low_order_jeff0_refresh_closes_missing_action_blocker_now else "reject",
            "updated-pack exact low-order J_eff^0 refresh closes missing-action blocker now",
            sign_base.truth(updated_pack_exact_low_order_jeff0_refresh_closes_missing_action_blocker_now),
            "This audit makes the formula gap explicit, but the theorem itself remains absent.",
        ),
        sign_base.row(
            "blind_vector_observable_gate_still_blocked",
            "pass" if blind_vector_observable_gate_still_blocked else "reject",
            "blind vector observable gate still blocked",
            sign_base.truth(blind_vector_observable_gate_still_blocked),
            "Direct blind vector computation remains blocked until the low-order J_eff^0 lane moves first.",
        ),
        sign_base.row(
            "farther_hybrid_continuation_reopen_required_now",
            "pass" if farther_hybrid_continuation_reopen_required_now else "reject",
            "farther hybrid continuation reopen required now",
            sign_base.truth(farther_hybrid_continuation_reopen_required_now),
            "Extra q-range evidence remains unnecessary because the blocker is still theorem-side and now sharpened to low-order J_eff^0.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "retained_scalar_residual_rel": float(prior_summary["retained_scalar_residual_rel"]),
        "updated_pack_exact_low_order_jeff0_refresh_audit_selected": updated_pack_exact_low_order_jeff0_refresh_audit_selected,
        "retained_proxy_signed_density_available": retained_proxy_signed_density_available,
        "retained_proxy_signed_density_only": retained_proxy_signed_density_only,
        "updated_pack_low_order_jeff0_support_surface_explicit": updated_pack_low_order_jeff0_support_surface_explicit,
        "updated_pack_low_order_jeff0_no_go_surface_explicit": updated_pack_low_order_jeff0_no_go_surface_explicit,
        "updated_pack_low_order_jeff0_refresh_target_surface_explicit": updated_pack_low_order_jeff0_refresh_target_surface_explicit,
        "updated_pack_low_order_jeff0_refresh_machine_readable_now": updated_pack_low_order_jeff0_refresh_machine_readable_now,
        "updated_pack_exact_low_order_jeff0_formula_available_now": updated_pack_exact_low_order_jeff0_formula_available_now,
        "updated_pack_exact_charge_current_noether_closure_available_now": updated_pack_exact_charge_current_noether_closure_available_now,
        "updated_pack_exact_qball_background_expansion_available_now": updated_pack_exact_qball_background_expansion_available_now,
        "updated_pack_low_order_jeff0_strong_support_verdict_derivable_now": updated_pack_low_order_jeff0_strong_support_verdict_derivable_now,
        "updated_pack_low_order_jeff0_no_go_verdict_derivable_now": updated_pack_low_order_jeff0_no_go_verdict_derivable_now,
        "updated_pack_blind_vector_refresh_followup_required": updated_pack_blind_vector_refresh_followup_required,
        "updated_pack_exact_low_order_jeff0_refresh_closes_missing_action_blocker_now": updated_pack_exact_low_order_jeff0_refresh_closes_missing_action_blocker_now,
        "blind_vector_observable_gate_still_blocked": blind_vector_observable_gate_still_blocked,
        "farther_hybrid_continuation_reopen_required_now": farther_hybrid_continuation_reopen_required_now,
        "selected_primary_pack_update_surface": "updated_pack_exact_low_order_jeff0_refresh",
        "selected_secondary_pack_update_surface": "updated_pack_blind_vector_refresh",
        "selected_reserve_completion_lane": "farther_hybrid_extra_q_range_only",
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": FOLLOWUP_ROUTE_NAME,
        "selected_followup_route_or_none": FOLLOWUP_ROUTE,
        "physical_reject_required": False,
    }

    declaration_payload = sign_base.payload(
        "8.7.56.2505",
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
                "source_rule_audit": sign_base.display_path(SOURCE_RULE_AUDIT),
                "jeff_prereq_audit": sign_base.display_path(JEFF_PREREQ_AUDIT),
                "charge_refresh_audit": sign_base.display_path(CHARGE_REFRESH_AUDIT),
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
            "overall_status": "vector_qball_form_factor_updated_pack_low_order_jeff0_refresh_audit_declared",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "formulas": build_formulae(),
            "hits": {
                "status_branch_hit": sign_base.hit(status_text, "8.7.56.2503"),
                "roadmap_branch_hit": sign_base.hit(roadmap_text, ".2503-.2506"),
                "current_problem_hit": sign_base.hit(current_problem_text, "updated-pack exact low-order J_eff^0 refresh audit"),
                "current_status_hit": sign_base.hit(current_status_text, "updated-pack exact low-order J_eff^0 refresh audit"),
                "unified_roadmap_hit": sign_base.hit(unified_text, ".2495-.2502"),
                "long_roadmap_hit": sign_base.hit(long_text, ".2503-.2506"),
                "part5_hit": sign_base.hit(part5_text, "exact low-order `J_{\\rm eff}^0` refresh audit"),
                "jeff0_hit": sign_base.hit(next_steps_text, "J_eff^0"),
                "support_hit": sign_base.hit(next_steps_text, "もし `J_eff^0` が低次で `|f_0|^2 - |f_L|^2` に落ちるなら"),
                "no_go_hit": sign_base.hit(next_steps_text, "そうならないなら、現在の proxy route は no-go"),
                "proxy_hit": sign_base.hit(next_steps_text, "|f_0|^2 - |f_L|^2"),
            },
        },
    )
    declaration_paths = write_artifact("declaration_gate", declaration_payload)

    route_payload = {
        "generated_utc": sign_base.now_iso(),
        "phase": {
            "phase": 8,
            "step": "8.7.56.2506",
            "name": STEP_NAME + " route sync",
        },
        "inputs": declaration_paths,
        "rows": rows,
        "summary": summary,
        "decision": {
            "overall_status": "vector_qball_form_factor_updated_pack_low_order_jeff0_refresh_route_synced",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        "evidence": {
            "formulas": build_formulae(),
            "disposition": {
                "low_order_jeff0_surface_explicit": updated_pack_low_order_jeff0_refresh_target_surface_explicit,
                "low_order_jeff0_machine_readable_now": updated_pack_low_order_jeff0_refresh_machine_readable_now,
                "blind_vector_followup_required": updated_pack_blind_vector_refresh_followup_required,
                "blind_vector_still_blocked": blind_vector_observable_gate_still_blocked,
            },
        },
    }
    route_paths = write_artifact("route_sync", route_payload)

    print("[ok] updated-pack exact low-order J_eff^0 refresh audit artifacts written")
    print(f"  declaration_gate: {declaration_paths['json']}")
    print(f"  route_sync: {route_paths['json']}")


if __name__ == "__main__":
    main()
