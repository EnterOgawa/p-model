#!/usr/bin/env python3
"""Generate 8.7.56.2479-.2482 updated-pack exact source-theorem refresh audit artifacts."""

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
        "8.7.56.2475-2478",
        "updated_pack_exact_jeff_prerequisite_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
PRIOR_AUDIT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.2471-2474",
        "updated_pack_exact_jeff_prerequisite_audit",
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

NEXT_STEPS = Path(r"C:\Users\ogawa\Downloads\trial2_vector_qball_next_steps_20260327.md")

STEP_TAG = "8.7.56.2479-2482"
STEP_NAME = "Trial-2 numeric alpha vector Q-ball form-factor updated-pack exact source-theorem refresh audit"
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_exact_source_theorem_refresh_audit",
    prefix="q",
)

PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_exact_source_"
    "theorem_refresh_primary_blind_vector_reserve_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_exact_source_"
    "theorem_refresh_audited_background_expansion_gate"
)
NEXT_ROUTE_NAME = "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_exact_source_theorem_gate_blind_vector_refresh"
NEXT_ROUTE = "8.7.56.2483"
FOLLOWUP_ROUTE_NAME = "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_exact_qball_background_expansion_audit"
FOLLOWUP_ROUTE = "8.7.56.2487"


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


# 関数: source-theorem refresh audit で使う式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used in the updated-pack exact source-theorem refresh audit."""
    return {
        "source_theorem_surface": "L \\supset a_mu J_eff^mu[P^Qball]",
        "refresh_order": "explicit Q-ball background expansion -> exact charge-current / Noether-current closure -> exact low-order J_eff^0 formula -> blind vector refresh",
        "why_order": "background expansion is the missing primitive, charge-current closure has partial continuity / adopted-U(1) surfaces already retained, and low-order J_eff^0 is the synthesis object that depends on both",
    }


# 関数: `.2479-.2482` を実行する。

def main() -> None:
    """Execute the updated-pack exact source-theorem refresh audit."""
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
        SOURCE_RULE_AUDIT,
        CHARGE_AUDIT,
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
    source_rule_summary = sign_base.read_json(SOURCE_RULE_AUDIT)["summary"]
    charge_audit_summary = sign_base.read_json(CHARGE_AUDIT)["summary"]

    updated_pack_exact_source_theorem_refresh_audit_selected = bool(
        prior_gate_summary["gate_b_updated_pack_exact_source_theorem_refresh_selected"]
        and not prior_gate_summary["gate_c_blind_vector_computation_primary_admissible_now"]
    )
    updated_pack_exact_source_theorem_target_surface_explicit = bool(
        prior_audit_summary["retained_explicit_matter_current_surface_available"]
        and prior_audit_summary["updated_pack_step_c_surface_explicit"]
        and sign_base.hit(next_steps_text, "### Step C.") is not None
        and sign_base.hit(next_steps_text, "J_eff^μ") is not None
    )
    updated_pack_exact_source_theorem_refresh_surface_explicit_now = bool(
        updated_pack_exact_source_theorem_target_surface_explicit
        and prior_audit_summary["updated_pack_exact_jeff_prerequisite_stack_machine_readable_now"]
        and source_rule_summary["updated_pack_source_rule_support_surface_explicit"]
        and source_rule_summary["updated_pack_source_rule_no_go_surface_explicit"]
    )
    updated_pack_background_expansion_primary_refresh_supported = bool(
        updated_pack_exact_source_theorem_refresh_surface_explicit_now
        and not prior_audit_summary["updated_pack_explicit_qball_background_expansion_available"]
    )
    updated_pack_charge_current_secondary_refresh_supported = bool(
        updated_pack_exact_source_theorem_refresh_surface_explicit_now
        and charge_audit_summary["generic_u1_continuity_surface_available"]
        and charge_audit_summary["qball_charge_mapping_statement_available"]
        and not prior_audit_summary["updated_pack_exact_charge_current_noether_closure_available"]
    )
    updated_pack_low_order_jeff0_formula_tertiary_synthesis_supported = bool(
        updated_pack_background_expansion_primary_refresh_supported
        and updated_pack_charge_current_secondary_refresh_supported
        and not prior_audit_summary["updated_pack_exact_low_order_jeff0_formula_available"]
    )
    updated_pack_exact_source_theorem_refresh_order_stable = bool(
        updated_pack_background_expansion_primary_refresh_supported
        and updated_pack_charge_current_secondary_refresh_supported
        and updated_pack_low_order_jeff0_formula_tertiary_synthesis_supported
    )
    updated_pack_exact_source_theorem_derived_now = False
    updated_pack_exact_source_theorem_refresh_closes_missing_action_blocker_now = False
    blind_vector_observable_gate_still_blocked = bool(
        not prior_gate_summary["gate_c_blind_vector_computation_primary_admissible_now"]
    )
    farther_hybrid_continuation_reopen_required_now = False

    rows = [
        sign_base.row(
            "updated_pack_exact_source_theorem_refresh_audit_selected",
            "pass" if updated_pack_exact_source_theorem_refresh_audit_selected else "reject",
            "updated-pack exact source-theorem refresh audit selected",
            sign_base.truth(updated_pack_exact_source_theorem_refresh_audit_selected),
            "The prerequisite gate already promoted exact source-theorem refresh as the next honest mainline.",
        ),
        sign_base.row(
            "updated_pack_exact_source_theorem_target_surface_explicit",
            "pass" if updated_pack_exact_source_theorem_target_surface_explicit else "reject",
            "updated-pack exact source-theorem target surface explicit",
            sign_base.truth(updated_pack_exact_source_theorem_target_surface_explicit),
            "Step C plus the retained matter-current interaction already state the theorem target surface explicitly.",
        ),
        sign_base.row(
            "updated_pack_exact_source_theorem_refresh_surface_explicit_now",
            "pass" if updated_pack_exact_source_theorem_refresh_surface_explicit_now else "reject",
            "updated-pack exact source-theorem refresh surface explicit now",
            sign_base.truth(updated_pack_exact_source_theorem_refresh_surface_explicit_now),
            "The theorem target, source-rule discriminator, and localized prerequisite stack now form one explicit refresh surface.",
        ),
        sign_base.row(
            "updated_pack_background_expansion_primary_refresh_supported",
            "pass" if updated_pack_background_expansion_primary_refresh_supported else "reject",
            "updated-pack background-expansion primary refresh supported",
            sign_base.truth(updated_pack_background_expansion_primary_refresh_supported),
            "The absent Q-ball background expansion is the missing primitive that must be surfaced before the exact theorem can move.",
        ),
        sign_base.row(
            "updated_pack_charge_current_secondary_refresh_supported",
            "pass" if updated_pack_charge_current_secondary_refresh_supported else "reject",
            "updated-pack charge-current secondary refresh supported",
            sign_base.truth(updated_pack_charge_current_secondary_refresh_supported),
            "Generic continuity and adopted-U(1) mapping are already retained, so charge-current closure is the next dependent exact closure rather than the first missing primitive.",
        ),
        sign_base.row(
            "updated_pack_low_order_jeff0_formula_tertiary_synthesis_supported",
            "pass" if updated_pack_low_order_jeff0_formula_tertiary_synthesis_supported else "reject",
            "updated-pack low-order J_eff^0 formula tertiary synthesis supported",
            sign_base.truth(updated_pack_low_order_jeff0_formula_tertiary_synthesis_supported),
            "Low-order J_eff^0 is the synthesis object that depends on both the background expansion and exact charge-current closure.",
        ),
        sign_base.row(
            "updated_pack_exact_source_theorem_refresh_order_stable",
            "pass" if updated_pack_exact_source_theorem_refresh_order_stable else "reject",
            "updated-pack exact source-theorem refresh order stable",
            sign_base.truth(updated_pack_exact_source_theorem_refresh_order_stable),
            "The refresh lane now has a stable first honest move instead of a flat three-item blocker list.",
        ),
        sign_base.row(
            "updated_pack_exact_source_theorem_derived_now",
            "pass" if updated_pack_exact_source_theorem_derived_now else "reject",
            "updated-pack exact source theorem derived now",
            sign_base.truth(updated_pack_exact_source_theorem_derived_now),
            "Ordering the refresh lane does not itself derive the theorem or collapse the residual-origin blocker.",
        ),
        sign_base.row(
            "updated_pack_exact_source_theorem_refresh_closes_missing_action_blocker_now",
            "pass" if updated_pack_exact_source_theorem_refresh_closes_missing_action_blocker_now else "reject",
            "updated-pack exact source-theorem refresh closes missing-action blocker now",
            sign_base.truth(updated_pack_exact_source_theorem_refresh_closes_missing_action_blocker_now),
            "The blocker remains open because the first prerequisite object is still absent from the public canon.",
        ),
        sign_base.row(
            "blind_vector_observable_gate_still_blocked",
            "pass" if blind_vector_observable_gate_still_blocked else "reject",
            "blind vector observable gate still blocked",
            sign_base.truth(blind_vector_observable_gate_still_blocked),
            "Blind vector computation remains downstream until the exact source-theorem refresh lane has moved beyond the primitive prerequisites.",
        ),
        sign_base.row(
            "farther_hybrid_continuation_reopen_required_now",
            "pass" if farther_hybrid_continuation_reopen_required_now else "reject",
            "farther hybrid continuation reopen required now",
            sign_base.truth(farther_hybrid_continuation_reopen_required_now),
            "Extra q-range evidence remains unnecessary because the blocker is still theorem-side and now ordered.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "retained_scalar_residual_rel": float(prior_gate_summary["retained_scalar_residual_rel"]),
        "updated_pack_exact_source_theorem_refresh_audit_selected": updated_pack_exact_source_theorem_refresh_audit_selected,
        "updated_pack_exact_source_theorem_target_surface_explicit": updated_pack_exact_source_theorem_target_surface_explicit,
        "updated_pack_exact_source_theorem_refresh_surface_explicit_now": updated_pack_exact_source_theorem_refresh_surface_explicit_now,
        "updated_pack_background_expansion_primary_refresh_supported": updated_pack_background_expansion_primary_refresh_supported,
        "updated_pack_charge_current_secondary_refresh_supported": updated_pack_charge_current_secondary_refresh_supported,
        "updated_pack_low_order_jeff0_formula_tertiary_synthesis_supported": updated_pack_low_order_jeff0_formula_tertiary_synthesis_supported,
        "updated_pack_exact_source_theorem_refresh_order_stable": updated_pack_exact_source_theorem_refresh_order_stable,
        "updated_pack_exact_source_theorem_derived_now": updated_pack_exact_source_theorem_derived_now,
        "updated_pack_exact_source_theorem_refresh_closes_missing_action_blocker_now": updated_pack_exact_source_theorem_refresh_closes_missing_action_blocker_now,
        "blind_vector_observable_gate_still_blocked": blind_vector_observable_gate_still_blocked,
        "farther_hybrid_continuation_reopen_required_now": farther_hybrid_continuation_reopen_required_now,
        "selected_primary_pack_update_surface": "updated_pack_exact_qball_background_expansion_audit",
        "selected_secondary_pack_update_surface": "updated_pack_exact_charge_current_noether_refresh",
        "selected_tertiary_pack_update_surface": "updated_pack_exact_low_order_jeff0_formula_synthesis",
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": FOLLOWUP_ROUTE_NAME,
        "selected_followup_route_or_none": FOLLOWUP_ROUTE,
        "physical_reject_required": False,
    }

    declaration_payload = sign_base.payload(
        "8.7.56.2481",
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
                "source_rule_audit": sign_base.display_path(SOURCE_RULE_AUDIT),
                "charge_audit": sign_base.display_path(CHARGE_AUDIT),
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
            "overall_status": "vector_qball_form_factor_updated_pack_exact_source_theorem_refresh_audit_declared",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "formulas": build_formulae(),
            "hits": {
                "status_branch_hit": sign_base.hit(status_text, ".2479-.2482"),
                "roadmap_branch_hit": sign_base.hit(roadmap_text, ".2479-.2482"),
                "current_problem_hit": sign_base.hit(current_problem_text, "updated-pack exact source-theorem refresh audit"),
                "current_status_hit": sign_base.hit(current_status_text, "updated-pack exact source-theorem refresh audit"),
                "unified_roadmap_hit": sign_base.hit(unified_text, ".2479-.2482"),
                "long_roadmap_hit": sign_base.hit(long_text, ".2479-.2482"),
                "part5_hit": sign_base.hit(part5_text, ".2479-.2482"),
                "step_c_hit": sign_base.hit(next_steps_text, "### Step C."),
                "jeff_hit": sign_base.hit(next_steps_text, "J_eff^μ"),
                "continuity_hit": sign_base.hit(sign_base.read_text(CHARGE_AUDIT), "generic U(1) continuity"),
            },
        },
    )
    declaration_paths = write_artifact("declaration_gate", declaration_payload)

    route_payload = {
        "generated_utc": sign_base.now_iso(),
        "phase": {
            "phase": 8,
            "step": "8.7.56.2482",
            "name": STEP_NAME + " route sync",
        },
        "inputs": declaration_paths,
        "rows": rows,
        "summary": summary,
        "decision": {
            "overall_status": "vector_qball_form_factor_updated_pack_exact_source_theorem_refresh_route_synced",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        "evidence": {
            "formulas": build_formulae(),
            "disposition": {
                "background_expansion_primary": updated_pack_background_expansion_primary_refresh_supported,
                "charge_current_secondary": updated_pack_charge_current_secondary_refresh_supported,
                "low_order_jeff0_tertiary": updated_pack_low_order_jeff0_formula_tertiary_synthesis_supported,
                "blind_vector_still_downstream": blind_vector_observable_gate_still_blocked,
            },
        },
    }
    route_paths = write_artifact("route_sync", route_payload)

    print("[ok] updated-pack exact source-theorem refresh audit artifacts written")
    print(f"  declaration_gate: {declaration_paths['json']}")
    print(f"  route_sync: {route_paths['json']}")


if __name__ == "__main__":
    main()
