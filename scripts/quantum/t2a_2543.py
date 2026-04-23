#!/usr/bin/env python3
"""Generate 8.7.56.2543-.2546 updated-pack background-expansion closeout audit artifacts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.quantum.t2a_1843 as sign_base
from scripts.utils.windows_length_policy import build_compact_artifact_stem, build_metrics_paths

PUB = ROOT / "output" / "public" / "quantum"
STATUS = ROOT / "doc" / "STATUS.md"
ROADMAP = ROOT / "doc" / "ROADMAP.md"
AI = ROOT / "doc" / "AI_CONTEXT_MIN.json"
RECENT = ROOT / "doc" / "WORK_HISTORY_RECENT.md"
P34 = ROOT / "doc" / "quantum" / "34_trial2_numeric_alpha_current_problem.md"
P36 = ROOT / "doc" / "quantum" / "36_trial2_numeric_alpha_current_status.md"
P39 = ROOT / "doc" / "quantum" / "39_trial2_vector_qball_unified_closure_roadmap.md"
P55 = ROOT / "doc" / "quantum" / "55_trial2_numeric_alpha_vector_qball_long_horizon_roadmap.md"
P14 = ROOT / "doc" / "paper" / "14_part5_future_predictions.md"
PRIOR = build_metrics_paths(
    PUB,
    build_compact_artifact_stem("8.7.56.2539-2542", "updated_pack_exact_source_theorem_gate", prefix="q"),
    "declaration_gate",
)["json"]
CLOSEOUT = build_metrics_paths(
    PUB,
    build_compact_artifact_stem("8.7.56.2535-2538", "updated_pack_exact_source_theorem_closeout_audit", prefix="q"),
    "declaration_gate",
)["json"]
BG = build_metrics_paths(
    PUB,
    build_compact_artifact_stem("8.7.56.2487-2490", "updated_pack_exact_qball_background_expansion_audit", prefix="q"),
    "declaration_gate",
)["json"]
CHARGE = build_metrics_paths(
    PUB,
    build_compact_artifact_stem("8.7.56.2495-2498", "updated_pack_exact_charge_current_refresh_audit", prefix="q"),
    "declaration_gate",
)["json"]
LOW = build_metrics_paths(
    PUB,
    build_compact_artifact_stem("8.7.56.2503-2506", "updated_pack_exact_low_order_jeff0_refresh_audit", prefix="q"),
    "declaration_gate",
)["json"]
SERIES = build_metrics_paths(
    PUB,
    build_compact_artifact_stem("8.7.56.2447-2450", "updated_pack_exact_ell0_series_operator_audit", prefix="q"),
    "declaration_gate",
)["json"]
BACKGROUND_LIFT = build_metrics_paths(
    PUB,
    build_compact_artifact_stem("8.7.56.1607-1610", "eff_metric_k_deriv", prefix="q"),
    "declaration_gate",
)["json"]
TAG = "8.7.56.2543-2546"
NAME = "Trial-2 numeric alpha vector Q-ball form-factor updated-pack background-expansion closeout audit"
STEM = build_compact_artifact_stem(TAG, "updated_pack_background_expansion_closeout_audit", prefix="q")
PRIOR_CLASS = "vector_qball_form_factor_residual_origin_missing_action_updated_pack_exact_source_theorem_closeout_audited_background_expansion_primary_charge_current_secondary_low_order_jeff0_tertiary_blind_vector_reserve_next"
BRANCH_CLASS = "vector_qball_form_factor_residual_origin_missing_action_updated_pack_background_expansion_closeout_audited_charge_current_primary_low_order_jeff0_secondary_gate"
NEXT_NAME = "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_background_expansion_closeout_gate_charge_current_refresh"
NEXT = "8.7.56.2547"
FOLLOW_NAME = "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_charge_current_closeout_audit"
FOLLOW = "8.7.56.2551"


# 関数: JSON/CSV artifact を書き出す。
def write_artifact(kind: str, data: dict) -> dict[str, str]:
    """Write one JSON payload and one CSV summary."""
    PUB.mkdir(parents=True, exist_ok=True)
    paths = build_metrics_paths(PUB, STEM, kind)
    paths["json"].write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with paths["csv"].open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["row_id", "status", "metric", "value", "note"])
        writer.writeheader()
        writer.writerows(data["rows"])

    return {"json": sign_base.display_path(paths["json"]), "csv": sign_base.display_path(paths["csv"])}


# 関数: background-expansion closeout audit で使う式を返す。

def formulas() -> dict[str, str]:
    """Return formulas used in the background-expansion closeout audit."""
    return {
        "two_component_series": "f_0(r)=a_0+a_2 r^2 + ...,  f_L(r)=b_1 r + b_3 r^3 + ...",
        "caseb_background_lift": "Q_g^0=-e^{2u} f_0,  Q_g^i=e^{-2u} f_L r_hat^i,  Q_g^2=-e^{2u} f_0^2 + e^{-2u} f_L^2",
        "closeout_stack": "exact source-theorem closeout -> background-expansion closeout -> charge-current closeout -> low-order J_eff^0 synthesis",
        "why": "background expansion remains the first missing primitive, charge-current closure is the next exact bridge, and low-order J_eff^0 remains the dependent synthesis object.",
    }


# 関数: `.2543-.2546` を実行する。

def main() -> None:
    """Execute the updated-pack background-expansion closeout audit."""
    for path in (
        STATUS,
        ROADMAP,
        AI,
        RECENT,
        P34,
        P36,
        P39,
        P55,
        P14,
        PRIOR,
        CLOSEOUT,
        BG,
        CHARGE,
        LOW,
        SERIES,
        BACKGROUND_LIFT,
    ):
        sign_base.require(path)

    status_text = sign_base.read_text(STATUS)
    roadmap_text = sign_base.read_text(ROADMAP)
    p = sign_base.read_json(PRIOR)["summary"]
    closeout = sign_base.read_json(CLOSEOUT)["summary"]
    bg = sign_base.read_json(BG)["summary"]
    charge = sign_base.read_json(CHARGE)["summary"]
    low = sign_base.read_json(LOW)["summary"]
    series = sign_base.read_json(SERIES)["summary"]
    background_lift = sign_base.read_json(BACKGROUND_LIFT)["summary"]

    selected = bool(
        p["gate_b_updated_pack_background_expansion_primary_selected"]
        and not p["gate_c_blind_vector_computation_primary_admissible_now"]
    )
    target_surface = bool(
        selected
        and closeout["updated_pack_background_expansion_primary_closeout_supported"]
        and bg["updated_pack_exact_qball_background_expansion_target_surface_explicit"]
        and series["updated_pack_exact_ell0_series_surface_explicit"]
    )
    retained_caseb = bool(
        bg["retained_caseb_background_lift_surface_available"]
        and background_lift["effective_metric_raised_background_components_derived"]
        and background_lift["effective_metric_background_norm_derived"]
    )
    machine = bool(
        target_surface
        and retained_caseb
        and bg["updated_pack_exact_qball_background_expansion_machine_readable_now"]
        and not bg["updated_pack_exact_qball_background_expansion_available_now"]
    )
    available = bool(bg["updated_pack_exact_qball_background_expansion_available_now"])
    closes = False
    charge_primary = bool(
        machine
        and charge["updated_pack_exact_charge_current_refresh_target_surface_explicit"]
        and charge["updated_pack_exact_charge_current_refresh_machine_readable_now"]
        and not charge["updated_pack_exact_charge_current_noether_closure_available_now"]
    )
    low_secondary = bool(
        charge_primary
        and low["updated_pack_low_order_jeff0_refresh_target_surface_explicit"]
        and low["updated_pack_low_order_jeff0_refresh_machine_readable_now"]
        and not low["updated_pack_exact_low_order_jeff0_formula_available_now"]
    )
    blind_blocked = bool(not p["gate_c_blind_vector_computation_primary_admissible_now"])
    hybrid = bool(p["farther_hybrid_continuation_reopen_required_now"])

    rows = [
        sign_base.row(
            "updated_pack_background_expansion_closeout_audit_selected",
            "pass" if selected else "reject",
            "updated-pack background-expansion closeout audit selected",
            sign_base.truth(selected),
            "The exact source-theorem gate already promoted background expansion as the next honest closeout object.",
        ),
        sign_base.row(
            "updated_pack_background_expansion_closeout_target_surface_explicit",
            "pass" if target_surface else "reject",
            "updated-pack background-expansion closeout target surface explicit",
            sign_base.truth(target_surface),
            "The source-theorem closeout stack and the retained Step A surface now meet on one background-expansion closeout target.",
        ),
        sign_base.row(
            "retained_caseb_background_lift_surface_available",
            "pass" if retained_caseb else "reject",
            "retained caseB background-lift surface available",
            sign_base.truth(retained_caseb),
            "The old effective-metric branch still supplies the raised background components and contracted Q-ball norm needed by the closeout audit.",
        ),
        sign_base.row(
            "updated_pack_background_expansion_closeout_machine_readable_now",
            "pass" if machine else "reject",
            "updated-pack background-expansion closeout machine-readable now",
            sign_base.truth(machine),
            "The first missing primitive is now pinned to one explicit theorem target plus retained background-lift formulas.",
        ),
        sign_base.row(
            "updated_pack_exact_qball_background_expansion_available_now",
            "pass" if available else "reject",
            "updated-pack exact Q-ball background expansion available now",
            sign_base.truth(available),
            "The canon still does not expose the full explicit Q-ball background expansion as one derived public theorem object.",
        ),
        sign_base.row(
            "updated_pack_background_expansion_closeout_closes_missing_action_blocker_now",
            "pass" if closes else "reject",
            "updated-pack background-expansion closeout closes missing-action blocker now",
            sign_base.truth(closes),
            "This audit localizes the first missing primitive honestly, but the explicit expansion itself is still absent.",
        ),
        sign_base.row(
            "updated_pack_charge_current_primary_closeout_required",
            "pass" if charge_primary else "reject",
            "updated-pack charge-current primary closeout required",
            sign_base.truth(charge_primary),
            "Once the background-expansion primitive is isolated, charge-current / Noether-current closure remains the next exact closeout object.",
        ),
        sign_base.row(
            "updated_pack_low_order_jeff0_secondary_closeout_required",
            "pass" if low_secondary else "reject",
            "updated-pack low-order J_eff^0 secondary closeout required",
            sign_base.truth(low_secondary),
            "Low-order J_eff^0 remains the dependent synthesis object after the background-expansion and charge-current closeout objects.",
        ),
        sign_base.row(
            "blind_vector_observable_gate_still_blocked",
            "pass" if blind_blocked else "reject",
            "blind vector observable gate still blocked",
            sign_base.truth(blind_blocked),
            "Blind-vector direct computation remains downstream because the theorem stack is still missing the explicit expansion and exact charge-current closure.",
        ),
        sign_base.row(
            "farther_hybrid_continuation_reopen_required_now",
            "pass" if hybrid else "reject",
            "farther hybrid continuation reopen required now",
            sign_base.truth(hybrid),
            "Extra q-range evidence remains unnecessary because the blocker is still theorem-side and sharpened to the background-expansion primitive.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "retained_scalar_residual_rel": float(p["retained_scalar_residual_rel"]),
        "updated_pack_background_expansion_closeout_audit_selected": selected,
        "updated_pack_background_expansion_closeout_target_surface_explicit": target_surface,
        "retained_caseb_background_lift_surface_available": retained_caseb,
        "updated_pack_background_expansion_closeout_machine_readable_now": machine,
        "updated_pack_exact_qball_background_expansion_available_now": available,
        "updated_pack_background_expansion_closeout_closes_missing_action_blocker_now": closes,
        "updated_pack_charge_current_primary_closeout_required": charge_primary,
        "updated_pack_low_order_jeff0_secondary_closeout_required": low_secondary,
        "blind_vector_observable_gate_still_blocked": blind_blocked,
        "farther_hybrid_continuation_reopen_required_now": hybrid,
        "selected_primary_pack_update_surface": "updated_pack_background_expansion_closeout_audit",
        "selected_secondary_pack_update_surface": "updated_pack_charge_current_closeout_audit",
        "selected_tertiary_pack_update_surface": "updated_pack_exact_low_order_jeff0_formula_synthesis",
        "selected_reserve_completion_lane": "blind_vector_after_background_expansion_and_charge_current_closeout",
        "selected_next_generation_route": NEXT_NAME,
        "recommended_next_route_or_none": NEXT,
        "selected_followup_route": FOLLOW_NAME,
        "selected_followup_route_or_none": FOLLOW,
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.2545",
        NAME + " declaration gate",
        {
            "source_files": {
                "status": sign_base.display_path(STATUS),
                "roadmap": sign_base.display_path(ROADMAP),
                "ai_context": sign_base.display_path(AI),
                "work_history_recent": sign_base.display_path(RECENT),
                "prior_gate": sign_base.display_path(PRIOR),
                "source_theorem_closeout_audit": sign_base.display_path(CLOSEOUT),
                "background_audit": sign_base.display_path(BG),
                "charge_audit": sign_base.display_path(CHARGE),
                "low_order_audit": sign_base.display_path(LOW),
                "series_audit": sign_base.display_path(SERIES),
                "background_lift_audit": sign_base.display_path(BACKGROUND_LIFT),
            },
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route_name": NEXT_NAME,
                "next_route": NEXT,
                "followup_route_name": FOLLOW_NAME,
                "followup_route": FOLLOW,
            },
        },
        rows,
        summary,
        {
            "overall_status": "vector_qball_form_factor_updated_pack_background_expansion_closeout_audit_declared",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_NAME],
        },
        {
            "formulas": formulas(),
            "hits": {
                "status_branch_hit": sign_base.hit(status_text, ".2539-.2542"),
                "roadmap_branch_hit": sign_base.hit(roadmap_text, ".2539-.2542"),
            },
        },
    )
    paths = write_artifact("declaration_gate", payload)
    route = {
        "generated_utc": sign_base.now_iso(),
        "phase": {"phase": 8, "step": "8.7.56.2546", "name": NAME + " route sync"},
        "inputs": paths,
        "rows": rows,
        "summary": summary,
        "decision": {
            "overall_status": "vector_qball_form_factor_updated_pack_background_expansion_closeout_route_synced",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_NAME],
        },
        "evidence": {
            "formulas": formulas(),
            "disposition": {
                "background_expansion_closeout_surface_explicit": target_surface,
                "background_expansion_closeout_machine_readable_now": machine,
                "charge_current_primary_required": charge_primary,
                "blind_vector_still_downstream": blind_blocked,
            },
        },
    }
    route_paths = write_artifact("route_sync", route)
    print("[ok] updated-pack background-expansion closeout audit artifacts written")
    print(f"  declaration_gate: {paths['json']}")
    print(f"  route_sync: {route_paths['json']}")


if __name__ == "__main__":
    main()
