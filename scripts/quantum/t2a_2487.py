#!/usr/bin/env python3
"""Generate 8.7.56.2487-.2490 updated-pack exact Q-ball background expansion audit artifacts."""

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
        "8.7.56.2483-2486",
        "updated_pack_exact_source_theorem_gate",
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
SOURCE_THEOREM_AUDIT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.2455-2458",
        "updated_pack_exact_effective_source_theorem_audit",
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

NEXT_STEPS = Path(r"C:\Users\ogawa\Downloads\trial2_vector_qball_next_steps_20260327.md")

STEP_TAG = "8.7.56.2487-2490"
STEP_NAME = "Trial-2 numeric alpha vector Q-ball form-factor updated-pack exact Q-ball background expansion audit"
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_exact_qball_background_expansion_audit",
    prefix="q",
)

PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_background_"
    "expansion_primary_charge_current_secondary_blind_vector_reserve_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_background_"
    "expansion_audited_charge_current_gate"
)
NEXT_ROUTE_NAME = "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_qball_background_expansion_gate_charge_current_refresh"
NEXT_ROUTE = "8.7.56.2491"
FOLLOWUP_ROUTE_NAME = "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_exact_charge_current_noether_refresh_audit"
FOLLOWUP_ROUTE = "8.7.56.2495"


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


# 関数: background-expansion audit で使う式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used in the updated-pack Q-ball background-expansion audit."""
    return {
        "two_component_series": "f_0(r)=a_0+a_2 r^2 + ...,  f_L(r)=b_1 r + b_3 r^3 + ...",
        "caseb_background_lift": "Q_g^0=-e^{2u} f_0,  Q_g^i=e^{-2u} f_L r_hat^i,  Q_g^2=-e^{2u} f_0^2 + e^{-2u} f_L^2",
        "source_theorem_target": "L \\supset a_mu J_eff^mu[P^Qball]",
        "missing_primitive": "P_mu = P_mu^Qball + a_mu + ...",
        "refresh_order": "background expansion -> charge-current / Noether-current closure -> low-order J_eff^0 synthesis -> blind vector refresh",
    }


# 関数: `.2487-.2490` を実行する。

def main() -> None:
    """Execute the updated-pack exact Q-ball background-expansion audit."""
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
        SOURCE_THEOREM_AUDIT,
        SERIES_AUDIT,
        BACKGROUND_LIFT_AUDIT,
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
    source_theorem_summary = sign_base.read_json(SOURCE_THEOREM_AUDIT)["summary"]
    series_summary = sign_base.read_json(SERIES_AUDIT)["summary"]
    background_lift_summary = sign_base.read_json(BACKGROUND_LIFT_AUDIT)["summary"]

    updated_pack_exact_qball_background_expansion_audit_selected = bool(
        prior_gate_summary["gate_b_updated_pack_exact_qball_background_expansion_primary_selected"]
        and not prior_gate_summary["gate_c_blind_vector_computation_primary_admissible_now"]
    )
    updated_pack_exact_qball_background_expansion_target_surface_explicit = bool(
        series_summary["updated_pack_exact_ell0_series_surface_explicit"]
        and source_theorem_summary["updated_pack_step_c_surface_explicit"]
        and sign_base.hit(next_steps_text, "### Step A.") is not None
        and sign_base.hit(next_steps_text, "f_0(r)=a_0+a_2 r^2") is not None
        and sign_base.hit(next_steps_text, "f_L(r)=b_1 r + b_3 r^3") is not None
    )
    retained_caseb_background_lift_surface_available = bool(
        background_lift_summary["effective_metric_raised_background_components_derived"]
        and background_lift_summary["effective_metric_background_norm_derived"]
    )
    updated_pack_exact_qball_background_expansion_machine_readable_now = bool(
        updated_pack_exact_qball_background_expansion_target_surface_explicit
        and retained_caseb_background_lift_surface_available
        and prior_audit_summary["updated_pack_exact_jeff_prerequisite_stack_fully_localized_now"]
        and not prior_audit_summary["updated_pack_explicit_qball_background_expansion_available"]
    )
    updated_pack_exact_qball_background_expansion_available_now = bool(
        prior_audit_summary["updated_pack_explicit_qball_background_expansion_available"]
    )
    updated_pack_exact_qball_background_expansion_closes_missing_action_blocker_now = False
    updated_pack_exact_charge_current_secondary_refresh_required = bool(
        updated_pack_exact_qball_background_expansion_machine_readable_now
        and not prior_audit_summary["updated_pack_exact_charge_current_noether_closure_available"]
    )
    blind_vector_observable_gate_still_blocked = bool(
        not prior_gate_summary["gate_c_blind_vector_computation_primary_admissible_now"]
    )
    farther_hybrid_continuation_reopen_required_now = False

    rows = [
        sign_base.row(
            "updated_pack_exact_qball_background_expansion_audit_selected",
            "pass" if updated_pack_exact_qball_background_expansion_audit_selected else "reject",
            "updated-pack exact Q-ball background expansion audit selected",
            sign_base.truth(updated_pack_exact_qball_background_expansion_audit_selected),
            "The ordered source-theorem gate already promoted background expansion as the first honest primitive.",
        ),
        sign_base.row(
            "updated_pack_exact_qball_background_expansion_target_surface_explicit",
            "pass" if updated_pack_exact_qball_background_expansion_target_surface_explicit else "reject",
            "updated-pack exact Q-ball background expansion target surface explicit",
            sign_base.truth(updated_pack_exact_qball_background_expansion_target_surface_explicit),
            "Step A plus the retained source-theorem lane already identify the missing Q-ball background expansion target surface.",
        ),
        sign_base.row(
            "retained_caseb_background_lift_surface_available",
            "pass" if retained_caseb_background_lift_surface_available else "reject",
            "retained caseB background-lift surface available",
            sign_base.truth(retained_caseb_background_lift_surface_available),
            "The old effective-metric branch already fixed the raised background components and contracted Q-ball norm used by the expansion audit.",
        ),
        sign_base.row(
            "updated_pack_exact_qball_background_expansion_machine_readable_now",
            "pass" if updated_pack_exact_qball_background_expansion_machine_readable_now else "reject",
            "updated-pack exact Q-ball background expansion machine-readable now",
            sign_base.truth(updated_pack_exact_qball_background_expansion_machine_readable_now),
            "The missing primitive is now pinned to an explicit theorem target plus retained background-lift formulas rather than a vague blocker phrase.",
        ),
        sign_base.row(
            "updated_pack_exact_qball_background_expansion_available_now",
            "pass" if updated_pack_exact_qball_background_expansion_available_now else "reject",
            "updated-pack exact Q-ball background expansion available now",
            sign_base.truth(updated_pack_exact_qball_background_expansion_available_now),
            "The canon still does not expose the full explicit Q-ball background expansion as one public theorem surface.",
        ),
        sign_base.row(
            "updated_pack_exact_qball_background_expansion_closes_missing_action_blocker_now",
            "pass" if updated_pack_exact_qball_background_expansion_closes_missing_action_blocker_now else "reject",
            "updated-pack exact Q-ball background expansion closes missing-action blocker now",
            sign_base.truth(updated_pack_exact_qball_background_expansion_closes_missing_action_blocker_now),
            "This audit localizes the primitive honestly, but the explicit expansion itself is still absent.",
        ),
        sign_base.row(
            "updated_pack_exact_charge_current_secondary_refresh_required",
            "pass" if updated_pack_exact_charge_current_secondary_refresh_required else "reject",
            "updated-pack exact charge-current secondary refresh required",
            sign_base.truth(updated_pack_exact_charge_current_secondary_refresh_required),
            "Once the background-expansion primitive is isolated, charge-current / Noether-current closure remains the next exact closure to refresh.",
        ),
        sign_base.row(
            "blind_vector_observable_gate_still_blocked",
            "pass" if blind_vector_observable_gate_still_blocked else "reject",
            "blind vector observable gate still blocked",
            sign_base.truth(blind_vector_observable_gate_still_blocked),
            "Blind vector computation remains downstream because the theorem stack is still missing the explicit expansion and exact charge-current closure.",
        ),
        sign_base.row(
            "farther_hybrid_continuation_reopen_required_now",
            "pass" if farther_hybrid_continuation_reopen_required_now else "reject",
            "farther hybrid continuation reopen required now",
            sign_base.truth(farther_hybrid_continuation_reopen_required_now),
            "Extra q-range evidence remains unnecessary because the blocker is still theorem-side and now sharpened to the background-expansion primitive.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "retained_scalar_residual_rel": float(prior_gate_summary["retained_scalar_residual_rel"]),
        "updated_pack_exact_qball_background_expansion_audit_selected": updated_pack_exact_qball_background_expansion_audit_selected,
        "updated_pack_exact_qball_background_expansion_target_surface_explicit": updated_pack_exact_qball_background_expansion_target_surface_explicit,
        "retained_caseb_background_lift_surface_available": retained_caseb_background_lift_surface_available,
        "updated_pack_exact_qball_background_expansion_machine_readable_now": updated_pack_exact_qball_background_expansion_machine_readable_now,
        "updated_pack_exact_qball_background_expansion_available_now": updated_pack_exact_qball_background_expansion_available_now,
        "updated_pack_exact_qball_background_expansion_closes_missing_action_blocker_now": updated_pack_exact_qball_background_expansion_closes_missing_action_blocker_now,
        "updated_pack_exact_charge_current_secondary_refresh_required": updated_pack_exact_charge_current_secondary_refresh_required,
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
        "8.7.56.2489",
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
                "source_theorem_audit": sign_base.display_path(SOURCE_THEOREM_AUDIT),
                "series_audit": sign_base.display_path(SERIES_AUDIT),
                "background_lift_audit": sign_base.display_path(BACKGROUND_LIFT_AUDIT),
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
            "overall_status": "vector_qball_form_factor_updated_pack_qball_background_expansion_audit_declared",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "formulas": build_formulae(),
            "hits": {
                "status_branch_hit": sign_base.hit(status_text, "8.7.56.2487"),
                "roadmap_branch_hit": sign_base.hit(roadmap_text, ".2487-.2490"),
                "current_problem_hit": sign_base.hit(current_problem_text, "updated-pack exact Q-ball background expansion audit"),
                "current_status_hit": sign_base.hit(current_status_text, "updated-pack exact Q-ball background expansion audit"),
                "unified_roadmap_hit": sign_base.hit(unified_text, ".2479-.2486"),
                "long_roadmap_hit": sign_base.hit(long_text, ".2487-.2490"),
                "part5_hit": sign_base.hit(part5_text, "background expansion primary"),
                "step_a_hit": sign_base.hit(next_steps_text, "### Step A."),
                "f0_series_hit": sign_base.hit(next_steps_text, "f_0(r)=a_0+a_2 r^2"),
                "fl_series_hit": sign_base.hit(next_steps_text, "f_L(r)=b_1 r + b_3 r^3"),
                "background_lift_hit": sign_base.hit(
                    sign_base.read_text(BACKGROUND_LIFT_AUDIT),
                    "Q_g^2 = -e^{2u} f_0^2 + e^{-2u} f_L^2",
                ),
            },
        },
    )
    declaration_paths = write_artifact("declaration_gate", declaration_payload)

    route_payload = {
        "generated_utc": sign_base.now_iso(),
        "phase": {
            "phase": 8,
            "step": "8.7.56.2490",
            "name": STEP_NAME + " route sync",
        },
        "inputs": declaration_paths,
        "rows": rows,
        "summary": summary,
        "decision": {
            "overall_status": "vector_qball_form_factor_updated_pack_qball_background_expansion_route_synced",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        "evidence": {
            "formulas": build_formulae(),
            "disposition": {
                "background_expansion_target_surface_explicit": updated_pack_exact_qball_background_expansion_target_surface_explicit,
                "background_expansion_machine_readable_now": updated_pack_exact_qball_background_expansion_machine_readable_now,
                "charge_current_refresh_required": updated_pack_exact_charge_current_secondary_refresh_required,
                "blind_vector_still_downstream": blind_vector_observable_gate_still_blocked,
            },
        },
    }
    route_paths = write_artifact("route_sync", route_payload)

    print("[ok] updated-pack exact Q-ball background expansion audit artifacts written")
    print(f"  declaration_gate: {declaration_paths['json']}")
    print(f"  route_sync: {route_paths['json']}")


if __name__ == "__main__":
    main()
