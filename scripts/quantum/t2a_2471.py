#!/usr/bin/env python3
"""Generate 8.7.56.2471-.2474 updated-pack exact J_eff prerequisite audit artifacts."""

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
        "8.7.56.2467-2470",
        "updated_pack_source_rule_gate",
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
SOURCE_RULE_AUDIT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.2463-2466",
        "updated_pack_source_rule_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
CHARGE_AUDIT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.1491-1494",
        "charge_current_closure",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.2471-2474"
STEP_NAME = "Trial-2 numeric alpha vector Q-ball form-factor updated-pack exact J_eff prerequisite audit"
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_exact_jeff_prerequisite_audit",
    prefix="q",
)

PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_exact_jeff_"
    "prerequisite_primary_blind_vector_reserve_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_exact_jeff_"
    "prerequisite_audited_source_theorem_refresh_gate"
)
NEXT_ROUTE_NAME = "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_exact_jeff_prerequisite_gate_source_theorem_refresh"
NEXT_ROUTE = "8.7.56.2475"
FOLLOWUP_ROUTE_NAME = "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_exact_source_theorem_refresh_audit"
FOLLOWUP_ROUTE = "8.7.56.2479"


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


# 関数: updated-pack exact J_eff prerequisite audit で使う式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used in the updated-pack exact J_eff prerequisite audit."""
    return {
        "prerequisite_stack": "exact Q-ball background expansion + exact low-order J_eff^0 formula + exact charge-current / Noether-current closure",
        "source_surface": "L \\supset a_mu J_eff^mu[P^Qball]",
        "ordering": "exact J_eff prerequisite audit -> prerequisite gate -> exact source-theorem refresh audit",
    }


# 関数: `.2471-.2474` を実行する。

def main() -> None:
    """Execute the updated-pack exact J_eff prerequisite audit."""
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
        SOURCE_RULE_AUDIT,
        CHARGE_AUDIT,
    ):
        sign_base.require(path)

    status_text = sign_base.read_text(STATUS)
    roadmap_text = sign_base.read_text(ROADMAP)
    current_problem_text = sign_base.read_text(CURRENT_PROBLEM)
    current_status_text = sign_base.read_text(CURRENT_STATUS)
    unified_text = sign_base.read_text(UNIFIED_ROADMAP)
    long_text = sign_base.read_text(LONG_ROADMAP)
    part5_text = sign_base.read_text(PART5)

    prior_gate_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    source_audit_summary = sign_base.read_json(SOURCE_AUDIT)["summary"]
    source_rule_summary = sign_base.read_json(SOURCE_RULE_AUDIT)["summary"]
    charge_audit_summary = sign_base.read_json(CHARGE_AUDIT)["summary"]

    updated_pack_exact_jeff_prerequisite_audit_selected = bool(
        prior_gate_summary["gate_b_updated_pack_exact_jeff_prerequisite_selected"]
        and not prior_gate_summary["gate_c_blind_vector_computation_primary_admissible_now"]
    )
    retained_explicit_matter_current_surface_available = bool(
        source_audit_summary["retained_explicit_matter_current_surface_available"]
    )
    updated_pack_step_c_surface_explicit = bool(
        source_audit_summary["updated_pack_step_c_surface_explicit"]
    )
    updated_pack_explicit_qball_background_expansion_available = bool(
        charge_audit_summary["explicit_qball_background_expansion_available"]
    )
    updated_pack_exact_low_order_jeff0_formula_available = bool(
        source_rule_summary["updated_pack_exact_low_order_jeff0_formula_available"]
    )
    updated_pack_exact_charge_current_noether_closure_available = bool(
        charge_audit_summary["exact_charge_current_noether_closure_available"]
    )
    updated_pack_exact_jeff_prerequisite_stack_machine_readable_now = bool(
        retained_explicit_matter_current_surface_available
        and updated_pack_step_c_surface_explicit
        and "explicit_qball_background_expansion_available" in charge_audit_summary
        and "updated_pack_exact_low_order_jeff0_formula_available" in source_rule_summary
        and "exact_charge_current_noether_closure_available" in charge_audit_summary
    )
    updated_pack_exact_jeff_prerequisite_stack_fully_localized_now = bool(
        updated_pack_exact_jeff_prerequisite_stack_machine_readable_now
        and not updated_pack_explicit_qball_background_expansion_available
        and not updated_pack_exact_low_order_jeff0_formula_available
        and not updated_pack_exact_charge_current_noether_closure_available
    )
    updated_pack_exact_jeff_prerequisite_stack_closes_missing_action_blocker_now = False
    updated_pack_exact_source_theorem_refresh_required = bool(
        updated_pack_exact_jeff_prerequisite_stack_fully_localized_now
        and not updated_pack_exact_jeff_prerequisite_stack_closes_missing_action_blocker_now
    )
    blind_vector_observable_gate_still_blocked = bool(
        not prior_gate_summary["gate_c_blind_vector_computation_primary_admissible_now"]
    )
    farther_hybrid_continuation_reopen_required_now = False

    rows = [
        sign_base.row(
            "updated_pack_exact_jeff_prerequisite_audit_selected",
            "pass" if updated_pack_exact_jeff_prerequisite_audit_selected else "reject",
            "updated-pack exact J_eff prerequisite audit selected",
            sign_base.truth(updated_pack_exact_jeff_prerequisite_audit_selected),
            "The source-rule gate already promoted exact J_eff prerequisites as the next honest blocker.",
        ),
        sign_base.row(
            "retained_explicit_matter_current_surface_available",
            "pass" if retained_explicit_matter_current_surface_available else "reject",
            "retained explicit matter-current surface available",
            sign_base.truth(retained_explicit_matter_current_surface_available),
            "The matter-current interaction surface is already explicit, so the blocker sits downstream inside J_eff construction itself.",
        ),
        sign_base.row(
            "updated_pack_step_c_surface_explicit",
            "pass" if updated_pack_step_c_surface_explicit else "reject",
            "updated-pack Step C source surface explicit",
            sign_base.truth(updated_pack_step_c_surface_explicit),
            "Step C already states that the remaining object is the exact effective source/current coupled to the photon branch.",
        ),
        sign_base.row(
            "updated_pack_explicit_qball_background_expansion_available",
            "pass" if updated_pack_explicit_qball_background_expansion_available else "reject",
            "updated-pack explicit Q-ball background expansion available",
            sign_base.truth(updated_pack_explicit_qball_background_expansion_available),
            "The retained prerequisite gap still starts with the absent explicit Q-ball background expansion.",
        ),
        sign_base.row(
            "updated_pack_exact_low_order_jeff0_formula_available",
            "pass" if updated_pack_exact_low_order_jeff0_formula_available else "reject",
            "updated-pack exact low-order J_eff^0 formula available",
            sign_base.truth(updated_pack_exact_low_order_jeff0_formula_available),
            "Without an exact low-order J_eff^0 formula the source-rule verdict remains proxy-only.",
        ),
        sign_base.row(
            "updated_pack_exact_charge_current_noether_closure_available",
            "pass" if updated_pack_exact_charge_current_noether_closure_available else "reject",
            "updated-pack exact charge-current / Noether-current closure available",
            sign_base.truth(updated_pack_exact_charge_current_noether_closure_available),
            "The old charge-current audit already fixed that exact Noether-current closure remains absent.",
        ),
        sign_base.row(
            "updated_pack_exact_jeff_prerequisite_stack_machine_readable_now",
            "pass" if updated_pack_exact_jeff_prerequisite_stack_machine_readable_now else "reject",
            "updated-pack exact J_eff prerequisite stack machine-readable now",
            sign_base.truth(updated_pack_exact_jeff_prerequisite_stack_machine_readable_now),
            "All prerequisite surfaces and their present absence/presence states are now machine-readable in one branch.",
        ),
        sign_base.row(
            "updated_pack_exact_jeff_prerequisite_stack_fully_localized_now",
            "pass" if updated_pack_exact_jeff_prerequisite_stack_fully_localized_now else "reject",
            "updated-pack exact J_eff prerequisite stack fully localized now",
            sign_base.truth(updated_pack_exact_jeff_prerequisite_stack_fully_localized_now),
            "The blocker is no longer a vague missing theorem phrase; it localizes to three explicit absent prerequisite objects.",
        ),
        sign_base.row(
            "updated_pack_exact_jeff_prerequisite_stack_closes_missing_action_blocker_now",
            "pass" if updated_pack_exact_jeff_prerequisite_stack_closes_missing_action_blocker_now else "reject",
            "updated-pack exact J_eff prerequisite stack closes missing-action blocker now",
            sign_base.truth(updated_pack_exact_jeff_prerequisite_stack_closes_missing_action_blocker_now),
            "Localizing the prerequisite stack does not itself derive the theorem or close the residual-origin blocker.",
        ),
        sign_base.row(
            "updated_pack_exact_source_theorem_refresh_required",
            "pass" if updated_pack_exact_source_theorem_refresh_required else "reject",
            "updated-pack exact source-theorem refresh required",
            sign_base.truth(updated_pack_exact_source_theorem_refresh_required),
            "Once the prerequisite stack is fixed, the honest next move is to refresh the exact source-theorem lane around that stack.",
        ),
        sign_base.row(
            "blind_vector_observable_gate_still_blocked",
            "pass" if blind_vector_observable_gate_still_blocked else "reject",
            "blind vector observable gate still blocked",
            sign_base.truth(blind_vector_observable_gate_still_blocked),
            "Blind vector computation remains downstream because the exact theorem prerequisites are still absent.",
        ),
        sign_base.row(
            "farther_hybrid_continuation_reopen_required_now",
            "pass" if farther_hybrid_continuation_reopen_required_now else "reject",
            "farther hybrid continuation reopen required now",
            sign_base.truth(farther_hybrid_continuation_reopen_required_now),
            "Extra q-range evidence remains unnecessary because the blocker is still theorem-side and localized.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "retained_scalar_residual_rel": float(prior_gate_summary["retained_scalar_residual_rel"]),
        "updated_pack_exact_jeff_prerequisite_audit_selected": updated_pack_exact_jeff_prerequisite_audit_selected,
        "retained_explicit_matter_current_surface_available": retained_explicit_matter_current_surface_available,
        "updated_pack_step_c_surface_explicit": updated_pack_step_c_surface_explicit,
        "updated_pack_explicit_qball_background_expansion_available": updated_pack_explicit_qball_background_expansion_available,
        "updated_pack_exact_low_order_jeff0_formula_available": updated_pack_exact_low_order_jeff0_formula_available,
        "updated_pack_exact_charge_current_noether_closure_available": updated_pack_exact_charge_current_noether_closure_available,
        "updated_pack_exact_jeff_prerequisite_stack_machine_readable_now": updated_pack_exact_jeff_prerequisite_stack_machine_readable_now,
        "updated_pack_exact_jeff_prerequisite_stack_fully_localized_now": updated_pack_exact_jeff_prerequisite_stack_fully_localized_now,
        "updated_pack_exact_jeff_prerequisite_stack_closes_missing_action_blocker_now": updated_pack_exact_jeff_prerequisite_stack_closes_missing_action_blocker_now,
        "updated_pack_exact_source_theorem_refresh_required": updated_pack_exact_source_theorem_refresh_required,
        "blind_vector_observable_gate_still_blocked": blind_vector_observable_gate_still_blocked,
        "farther_hybrid_continuation_reopen_required_now": farther_hybrid_continuation_reopen_required_now,
        "selected_primary_pack_update_surface": "updated_pack_exact_source_theorem_refresh_audit",
        "selected_secondary_pack_update_surface": "blind_vector_computation_after_exact_source_theorem_gate",
        "selected_reserve_completion_lane": "farther_hybrid_extra_q_range_only",
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": FOLLOWUP_ROUTE_NAME,
        "selected_followup_route_or_none": FOLLOWUP_ROUTE,
        "physical_reject_required": False,
    }

    declaration_payload = sign_base.payload(
        "8.7.56.2473",
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
                "source_rule_audit": sign_base.display_path(SOURCE_RULE_AUDIT),
                "charge_audit": sign_base.display_path(CHARGE_AUDIT),
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
            "overall_status": "vector_qball_form_factor_updated_pack_exact_jeff_prerequisite_audit_declared",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "formulas": build_formulae(),
            "hits": {
                "status_branch_hit": sign_base.hit(status_text, "8.7.56.2471"),
                "roadmap_branch_hit": sign_base.hit(roadmap_text, ".2471-.2474"),
                "current_problem_hit": sign_base.hit(current_problem_text, "updated-pack exact J_eff prerequisite audit"),
                "current_status_hit": sign_base.hit(current_status_text, "updated-pack exact J_eff prerequisite audit"),
                "unified_roadmap_hit": sign_base.hit(unified_text, ".2471-.2474"),
                "long_roadmap_hit": sign_base.hit(long_text, ".2471-.2474"),
                "part5_hit": sign_base.hit(part5_text, ".2471-.2474"),
                "background_expansion_hit": sign_base.hit(current_status_text, "Q-ball background expansion"),
                "effective_source_formula_hit": sign_base.hit(current_status_text, "effective source formula"),
                "charge_current_hit": sign_base.hit(current_problem_text, "exact charge-current / Noether-current closure"),
            },
        },
    )
    declaration_paths = write_artifact("declaration_gate", declaration_payload)

    route_payload = {
        "generated_utc": sign_base.now_iso(),
        "phase": {
            "phase": 8,
            "step": "8.7.56.2474",
            "name": STEP_NAME + " route sync",
        },
        "inputs": declaration_paths,
        "rows": rows,
        "summary": summary,
        "decision": {
            "overall_status": "vector_qball_form_factor_updated_pack_exact_jeff_prerequisite_audit_route_synced",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        "evidence": {
            "formulae": build_formulae(),
            "disposition": {
                "exact_jeff_prerequisite_stack_fully_localized_now": updated_pack_exact_jeff_prerequisite_stack_fully_localized_now,
                "exact_source_theorem_refresh_required": updated_pack_exact_source_theorem_refresh_required,
                "blind_vector_still_downstream": blind_vector_observable_gate_still_blocked,
            },
        },
    }
    route_paths = write_artifact("route_sync", route_payload)

    print("[ok] updated-pack exact J_eff prerequisite audit artifacts written")
    print(f"  declaration_gate: {declaration_paths['json']}")
    print(f"  route_sync: {route_paths['json']}")


if __name__ == "__main__":
    main()
