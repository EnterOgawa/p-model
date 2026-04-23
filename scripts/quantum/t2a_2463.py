#!/usr/bin/env python3
"""Generate 8.7.56.2463-.2466 updated-pack source-rule audit artifacts."""

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
        "8.7.56.2459-2462",
        "updated_pack_exact_effective_source_theorem_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
PRIOR_AUDIT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.2455-2458",
        "updated_pack_exact_effective_source_theorem_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
OLD_SOURCE_AUDIT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.1487-1490",
        "effective_source_theorem",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

NEXT_STEPS = Path(r"C:\Users\ogawa\Downloads\trial2_vector_qball_next_steps_20260327.md")

STEP_TAG = "8.7.56.2463-2466"
STEP_NAME = "Trial-2 numeric alpha vector Q-ball form-factor updated-pack source-rule audit"
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_source_rule_audit",
    prefix="q",
)

PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_source_rule_"
    "primary_blind_vector_reserve_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_source_rule_"
    "audited_exact_jeff_prerequisite_gate"
)
NEXT_ROUTE_NAME = "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_source_rule_gate_blind_vector_refresh"
NEXT_ROUTE = "8.7.56.2467"
FOLLOWUP_ROUTE_NAME = "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_exact_jeff_prerequisite_audit"
FOLLOWUP_ROUTE = "8.7.56.2471"


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


# 関数: updated-pack source-rule audit で使う式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used in the updated-pack source-rule audit."""
    return {
        "source_rule_support": "J_eff^0 low-order == |f_0|^2 - |f_L|^2 => proxy strong support",
        "source_rule_no_go": "J_eff^0 low-order != |f_0|^2 - |f_L|^2 => proxy route no-go",
        "prerequisite_stack": "exact Q-ball background expansion + exact J_eff^mu formula + exact charge-current closure",
        "ordering": "source-rule audit -> source-rule gate -> exact J_eff prerequisite audit",
    }


# 関数: `.2463-.2466` を実行する。

def main() -> None:
    """Execute the updated-pack source-rule audit."""
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
        PRIOR_AUDIT,
        OLD_SOURCE_AUDIT,
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
    prior_audit_summary = sign_base.read_json(PRIOR_AUDIT)["summary"]
    old_source_summary = sign_base.read_json(OLD_SOURCE_AUDIT)["summary"]

    updated_pack_source_rule_audit_selected = bool(
        prior_gate_summary["gate_b_updated_pack_source_rule_refresh_selected"]
        and not prior_gate_summary["gate_c_blind_vector_computation_primary_admissible_now"]
    )
    updated_pack_source_rule_support_surface_explicit = bool(
        prior_audit_summary["updated_pack_proxy_support_rule_explicit"]
        and sign_base.hit(next_steps_text, "もし `J_eff^0` が低次で `|f_0|^2 - |f_L|^2` に落ちるなら") is not None
    )
    updated_pack_source_rule_no_go_surface_explicit = bool(
        prior_audit_summary["updated_pack_proxy_no_go_rule_explicit"]
        and sign_base.hit(next_steps_text, "そうならないなら、現在の proxy route は no-go") is not None
    )
    updated_pack_low_order_jeff0_discriminator_surface_explicit = bool(
        prior_audit_summary["updated_pack_step_c_surface_explicit"]
        and updated_pack_source_rule_support_surface_explicit
        and updated_pack_source_rule_no_go_surface_explicit
    )
    current_canon_explicit_qball_background_expansion_available = bool(
        prior_audit_summary["current_canon_explicit_qball_background_expansion_available"]
    )
    updated_pack_exact_low_order_jeff0_formula_available = bool(
        prior_audit_summary["current_canon_explicit_effective_source_formula_available"]
    )
    updated_pack_exact_charge_current_closure_available = bool(
        not old_source_summary["exact_charge_current_noether_closure_required"]
    )
    updated_pack_source_rule_support_verdict_derivable_now = bool(
        updated_pack_low_order_jeff0_discriminator_surface_explicit
        and current_canon_explicit_qball_background_expansion_available
        and updated_pack_exact_low_order_jeff0_formula_available
        and updated_pack_exact_charge_current_closure_available
    )
    updated_pack_source_rule_no_go_verdict_derivable_now = bool(
        updated_pack_source_rule_support_verdict_derivable_now
    )
    updated_pack_source_rule_discriminator_surface_official_fixable_now = bool(
        updated_pack_source_rule_audit_selected
        and updated_pack_low_order_jeff0_discriminator_surface_explicit
    )
    updated_pack_source_rule_audit_closes_missing_action_blocker_now = False
    updated_pack_exact_jeff_prerequisite_followup_required = bool(
        updated_pack_source_rule_discriminator_surface_official_fixable_now
        and not updated_pack_source_rule_audit_closes_missing_action_blocker_now
    )
    blind_vector_observable_gate_still_blocked = bool(
        prior_audit_summary["blind_vector_computation_primary_admissible_now"] is False
    )
    farther_hybrid_continuation_reopen_required_now = False

    rows = [
        sign_base.row(
            "updated_pack_source_rule_audit_selected",
            "pass" if updated_pack_source_rule_audit_selected else "reject",
            "updated-pack source-rule audit selected",
            sign_base.truth(updated_pack_source_rule_audit_selected),
            "The prior gate already selected source-rule refresh as the next honest mainline before any blind vector retry.",
        ),
        sign_base.row(
            "updated_pack_source_rule_support_surface_explicit",
            "pass" if updated_pack_source_rule_support_surface_explicit else "reject",
            "updated-pack source-rule support surface explicit",
            sign_base.truth(updated_pack_source_rule_support_surface_explicit),
            "The refreshed next-steps pack states the exact strong-support discriminator explicitly at low-order J_eff^0.",
        ),
        sign_base.row(
            "updated_pack_source_rule_no_go_surface_explicit",
            "pass" if updated_pack_source_rule_no_go_surface_explicit else "reject",
            "updated-pack source-rule no-go surface explicit",
            sign_base.truth(updated_pack_source_rule_no_go_surface_explicit),
            "The same pack also states the route-local no-go branch explicitly rather than leaving it implicit.",
        ),
        sign_base.row(
            "updated_pack_low_order_jeff0_discriminator_surface_explicit",
            "pass" if updated_pack_low_order_jeff0_discriminator_surface_explicit else "reject",
            "updated-pack low-order J_eff^0 discriminator surface explicit",
            sign_base.truth(updated_pack_low_order_jeff0_discriminator_surface_explicit),
            "Step C plus the support/no-go split are enough to fix the missing discriminator surface publicly.",
        ),
        sign_base.row(
            "current_canon_explicit_qball_background_expansion_available",
            "pass" if current_canon_explicit_qball_background_expansion_available else "reject",
            "current canon explicit Q-ball background expansion available",
            sign_base.truth(current_canon_explicit_qball_background_expansion_available),
            "The discriminator cannot be derived theorem-level while the explicit Q-ball background expansion remains absent.",
        ),
        sign_base.row(
            "updated_pack_exact_low_order_jeff0_formula_available",
            "pass" if updated_pack_exact_low_order_jeff0_formula_available else "reject",
            "updated-pack exact low-order J_eff^0 formula available",
            sign_base.truth(updated_pack_exact_low_order_jeff0_formula_available),
            "The canon still lacks a first-principles low-order J_eff^0 formula rather than a proxy reading.",
        ),
        sign_base.row(
            "updated_pack_exact_charge_current_closure_available",
            "pass" if updated_pack_exact_charge_current_closure_available else "reject",
            "updated-pack exact charge-current closure available",
            sign_base.truth(updated_pack_exact_charge_current_closure_available),
            "The old exact source-theorem lane already fixed that exact charge-current / Noether-current closure is still absent.",
        ),
        sign_base.row(
            "updated_pack_source_rule_support_verdict_derivable_now",
            "pass" if updated_pack_source_rule_support_verdict_derivable_now else "reject",
            "updated-pack source-rule support verdict derivable now",
            sign_base.truth(updated_pack_source_rule_support_verdict_derivable_now),
            "Strong support requires exact low-order J_eff^0, not just a proxy discriminator sentence.",
        ),
        sign_base.row(
            "updated_pack_source_rule_no_go_verdict_derivable_now",
            "pass" if updated_pack_source_rule_no_go_verdict_derivable_now else "reject",
            "updated-pack source-rule no-go verdict derivable now",
            sign_base.truth(updated_pack_source_rule_no_go_verdict_derivable_now),
            "A theorem-level no-go also requires the same exact low-order J_eff^0 object, not merely the absence of support prose.",
        ),
        sign_base.row(
            "updated_pack_source_rule_discriminator_surface_official_fixable_now",
            "pass" if updated_pack_source_rule_discriminator_surface_official_fixable_now else "reject",
            "updated-pack source-rule discriminator surface official-fixable now",
            sign_base.truth(updated_pack_source_rule_discriminator_surface_official_fixable_now),
            "What can be fixed now is the public discriminator surface and the remaining prerequisite stack, not the final exact verdict.",
        ),
        sign_base.row(
            "updated_pack_source_rule_audit_closes_missing_action_blocker_now",
            "pass" if updated_pack_source_rule_audit_closes_missing_action_blocker_now else "reject",
            "updated-pack source-rule audit closes missing-action blocker now",
            sign_base.truth(updated_pack_source_rule_audit_closes_missing_action_blocker_now),
            "The missing-action blocker stays open because exact J_eff prerequisites are still missing.",
        ),
        sign_base.row(
            "updated_pack_exact_jeff_prerequisite_followup_required",
            "pass" if updated_pack_exact_jeff_prerequisite_followup_required else "reject",
            "updated-pack exact J_eff prerequisite followup required",
            sign_base.truth(updated_pack_exact_jeff_prerequisite_followup_required),
            "After the source-rule surface is fixed, the honest next blocker is the exact J_eff prerequisite stack itself.",
        ),
        sign_base.row(
            "blind_vector_observable_gate_still_blocked",
            "pass" if blind_vector_observable_gate_still_blocked else "reject",
            "blind vector observable gate still blocked",
            sign_base.truth(blind_vector_observable_gate_still_blocked),
            "Blind vector computation remains downstream because the source rule has not become theorem-level yet.",
        ),
        sign_base.row(
            "farther_hybrid_continuation_reopen_required_now",
            "pass" if farther_hybrid_continuation_reopen_required_now else "reject",
            "farther hybrid continuation reopen required now",
            sign_base.truth(farther_hybrid_continuation_reopen_required_now),
            "Extra q-range evidence remains unnecessary because the blocker is still a theorem-surface prerequisite stack.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "retained_scalar_residual_rel": float(prior_gate_summary["retained_scalar_residual_rel"]),
        "updated_pack_source_rule_audit_selected": updated_pack_source_rule_audit_selected,
        "updated_pack_source_rule_support_surface_explicit": updated_pack_source_rule_support_surface_explicit,
        "updated_pack_source_rule_no_go_surface_explicit": updated_pack_source_rule_no_go_surface_explicit,
        "updated_pack_low_order_jeff0_discriminator_surface_explicit": updated_pack_low_order_jeff0_discriminator_surface_explicit,
        "current_canon_explicit_qball_background_expansion_available": current_canon_explicit_qball_background_expansion_available,
        "updated_pack_exact_low_order_jeff0_formula_available": updated_pack_exact_low_order_jeff0_formula_available,
        "updated_pack_exact_charge_current_closure_available": updated_pack_exact_charge_current_closure_available,
        "updated_pack_source_rule_support_verdict_derivable_now": updated_pack_source_rule_support_verdict_derivable_now,
        "updated_pack_source_rule_no_go_verdict_derivable_now": updated_pack_source_rule_no_go_verdict_derivable_now,
        "updated_pack_source_rule_discriminator_surface_official_fixable_now": updated_pack_source_rule_discriminator_surface_official_fixable_now,
        "updated_pack_source_rule_audit_closes_missing_action_blocker_now": updated_pack_source_rule_audit_closes_missing_action_blocker_now,
        "updated_pack_exact_jeff_prerequisite_followup_required": updated_pack_exact_jeff_prerequisite_followup_required,
        "blind_vector_observable_gate_still_blocked": blind_vector_observable_gate_still_blocked,
        "farther_hybrid_continuation_reopen_required_now": farther_hybrid_continuation_reopen_required_now,
        "selected_primary_pack_update_surface": "updated_pack_exact_jeff_prerequisite_audit",
        "selected_secondary_pack_update_surface": "blind_vector_computation_after_exact_jeff_prerequisite_gate",
        "selected_reserve_completion_lane": "farther_hybrid_extra_q_range_only",
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": FOLLOWUP_ROUTE_NAME,
        "selected_followup_route_or_none": FOLLOWUP_ROUTE,
        "physical_reject_required": False,
    }

    declaration_payload = sign_base.payload(
        "8.7.56.2465",
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
                "prior_audit": sign_base.display_path(PRIOR_AUDIT),
                "old_source_audit": sign_base.display_path(OLD_SOURCE_AUDIT),
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
            "overall_status": "vector_qball_form_factor_updated_pack_source_rule_audit_declared",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "formulas": build_formulae(),
            "hits": {
                "status_branch_hit": sign_base.hit(status_text, "8.7.56.2463"),
                "roadmap_branch_hit": sign_base.hit(roadmap_text, ".2463-.2466"),
                "current_problem_hit": sign_base.hit(current_problem_text, "updated-pack source-rule audit"),
                "current_status_hit": sign_base.hit(current_status_text, "updated-pack source-rule audit"),
                "unified_roadmap_hit": sign_base.hit(unified_text, ".2463-.2466"),
                "long_roadmap_hit": sign_base.hit(long_text, ".2463-.2466"),
                "part5_hit": sign_base.hit(part5_text, ".2463-.2466"),
                "support_rule_hit": sign_base.hit(next_steps_text, "もし `J_eff^0` が低次で `|f_0|^2 - |f_L|^2` に落ちるなら"),
                "no_go_rule_hit": sign_base.hit(next_steps_text, "そうならないなら、現在の proxy route は no-go"),
            },
        },
    )
    declaration_paths = write_artifact("declaration_gate", declaration_payload)

    route_payload = {
        "generated_utc": sign_base.now_iso(),
        "phase": {
            "phase": 8,
            "step": "8.7.56.2466",
            "name": STEP_NAME + " route sync",
        },
        "inputs": declaration_paths,
        "rows": rows,
        "summary": summary,
        "decision": {
            "overall_status": "vector_qball_form_factor_updated_pack_source_rule_audit_route_synced",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        "evidence": {
            "formulae": build_formulae(),
            "disposition": {
                "exact_support_verdict_available_now": updated_pack_source_rule_support_verdict_derivable_now,
                "exact_no_go_verdict_available_now": updated_pack_source_rule_no_go_verdict_derivable_now,
                "source_rule_surface_fixed_now": updated_pack_source_rule_discriminator_surface_official_fixable_now,
                "exact_jeff_prerequisite_followup_required": updated_pack_exact_jeff_prerequisite_followup_required,
            },
        },
    }
    route_paths = write_artifact("route_sync", route_payload)

    print("[ok] updated-pack source-rule audit artifacts written")
    print(f"  declaration_gate: {declaration_paths['json']}")
    print(f"  route_sync: {route_paths['json']}")


if __name__ == "__main__":
    main()
