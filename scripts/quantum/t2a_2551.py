#!/usr/bin/env python3
"""Generate 8.7.56.2551-.2554 updated-pack charge-current closeout audit artifacts."""

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
    build_compact_artifact_stem("8.7.56.2547-2550", "updated_pack_background_expansion_closeout_gate", prefix="q"),
    "declaration_gate",
)["json"]
BG_CLOSEOUT = build_metrics_paths(
    PUB,
    build_compact_artifact_stem("8.7.56.2543-2546", "updated_pack_background_expansion_closeout_audit", prefix="q"),
    "declaration_gate",
)["json"]
CHARGE_REFRESH = build_metrics_paths(
    PUB,
    build_compact_artifact_stem("8.7.56.2495-2498", "updated_pack_exact_charge_current_refresh_audit", prefix="q"),
    "declaration_gate",
)["json"]
LOW_REFRESH = build_metrics_paths(
    PUB,
    build_compact_artifact_stem("8.7.56.2503-2506", "updated_pack_exact_low_order_jeff0_refresh_audit", prefix="q"),
    "declaration_gate",
)["json"]
TAG = "8.7.56.2551-2554"
NAME = "Trial-2 numeric alpha vector Q-ball form-factor updated-pack charge-current closeout audit"
STEM = build_compact_artifact_stem(TAG, "updated_pack_charge_current_closeout_audit", prefix="q")
PRIOR_CLASS = "vector_qball_form_factor_residual_origin_missing_action_updated_pack_background_expansion_closeout_audited_charge_current_primary_low_order_jeff0_secondary_blind_vector_reserve_next"
BRANCH_CLASS = "vector_qball_form_factor_residual_origin_missing_action_updated_pack_charge_current_closeout_audited_low_order_jeff0_primary_gate"
NEXT_NAME = "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_charge_current_closeout_gate_low_order_jeff0_refresh"
NEXT = "8.7.56.2555"
FOLLOW_NAME = "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_low_order_jeff0_closeout_audit"
FOLLOW = "8.7.56.2559"


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


# 関数: charge-current closeout audit で使う式を返す。

def formulas() -> dict[str, str]:
    """Return formulas used in the charge-current closeout audit."""
    return {
        "continuity_surface": "partial_mu J^mu = 0",
        "charge_mapping": "Q_ball Noether charge = adopted U(1) charge",
        "closeout_stack": "background expansion closeout -> charge-current / Noether-current closeout -> low-order J_eff^0 closeout -> blind vector reserve",
        "why": "Once the first primitive is isolated, the next exact bridge is charge-current / Noether-current closure, while low-order J_eff^0 remains downstream.",
    }


# 関数: `.2551-.2554` を実行する。

def main() -> None:
    """Execute the updated-pack charge-current closeout audit."""
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
        BG_CLOSEOUT,
        CHARGE_REFRESH,
        LOW_REFRESH,
    ):
        sign_base.require(path)

    status_text = sign_base.read_text(STATUS)
    roadmap_text = sign_base.read_text(ROADMAP)
    prior = sign_base.read_json(PRIOR)["summary"]
    bg_closeout = sign_base.read_json(BG_CLOSEOUT)["summary"]
    charge_refresh = sign_base.read_json(CHARGE_REFRESH)["summary"]
    low_refresh = sign_base.read_json(LOW_REFRESH)["summary"]

    selected = bool(
        prior["gate_b_updated_pack_charge_current_primary_selected"]
        and not prior["gate_c_blind_vector_computation_primary_admissible_now"]
    )
    retained_continuity = bool(charge_refresh["retained_generic_u1_continuity_surface_available"])
    retained_mapping = bool(charge_refresh["retained_qball_charge_mapping_statement_available"])
    retained_identity = bool(charge_refresh["retained_direct_qball_u1_identity_required"])
    retained_proxy = bool(charge_refresh["retained_proxy_signed_density_available"])
    target_surface = bool(
        selected
        and bg_closeout["updated_pack_background_expansion_closeout_target_surface_explicit"]
        and charge_refresh["updated_pack_exact_charge_current_refresh_target_surface_explicit"]
        and retained_continuity
        and retained_mapping
        and retained_identity
    )
    machine = bool(
        target_surface
        and bg_closeout["updated_pack_background_expansion_closeout_machine_readable_now"]
        and charge_refresh["updated_pack_exact_charge_current_refresh_machine_readable_now"]
        and not charge_refresh["updated_pack_exact_charge_current_noether_closure_available_now"]
    )
    available = bool(charge_refresh["updated_pack_exact_charge_current_noether_closure_available_now"])
    closes = False
    low_primary = bool(
        machine
        and low_refresh["updated_pack_low_order_jeff0_refresh_target_surface_explicit"]
        and low_refresh["updated_pack_low_order_jeff0_refresh_machine_readable_now"]
        and not low_refresh["updated_pack_exact_low_order_jeff0_formula_available_now"]
    )
    blind_blocked = bool(not prior["gate_c_blind_vector_computation_primary_admissible_now"])
    hybrid = bool(prior["farther_hybrid_continuation_reopen_required_now"])

    rows = [
        sign_base.row(
            "updated_pack_charge_current_closeout_audit_selected",
            "pass" if selected else "reject",
            "updated-pack charge-current closeout audit selected",
            sign_base.truth(selected),
            "The background-expansion gate already promoted charge-current closeout as the next honest exact bridge.",
        ),
        sign_base.row(
            "retained_generic_u1_continuity_surface_available",
            "pass" if retained_continuity else "reject",
            "retained generic U(1) continuity surface available",
            sign_base.truth(retained_continuity),
            "The closeout lane still reuses the retained continuity surface instead of reopening a new operator family.",
        ),
        sign_base.row(
            "retained_qball_charge_mapping_statement_available",
            "pass" if retained_mapping else "reject",
            "retained Q-ball / adopted-U(1) charge mapping statement available",
            sign_base.truth(retained_mapping),
            "The adopted-U(1) charge mapping remains available as the exact-closeout target statement.",
        ),
        sign_base.row(
            "retained_direct_qball_u1_identity_required",
            "pass" if retained_identity else "reject",
            "retained direct Q-ball / adopted-U(1) identity required",
            sign_base.truth(retained_identity),
            "The closeout audit still forbids extra normalization freedom beyond the retained direct identity requirement.",
        ),
        sign_base.row(
            "retained_proxy_signed_density_available",
            "pass" if retained_proxy else "reject",
            "retained proxy signed density available",
            sign_base.truth(retained_proxy),
            "The proxy |f_0|^2 - |f_L|^2 surface remains available only as a downstream comparison target.",
        ),
        sign_base.row(
            "updated_pack_charge_current_closeout_target_surface_explicit",
            "pass" if target_surface else "reject",
            "updated-pack charge-current closeout target surface explicit",
            sign_base.truth(target_surface),
            "The background-expansion closeout stack and the retained charge-current ingredients now meet on one exact closeout target.",
        ),
        sign_base.row(
            "updated_pack_charge_current_closeout_machine_readable_now",
            "pass" if machine else "reject",
            "updated-pack charge-current closeout machine-readable now",
            sign_base.truth(machine),
            "The next exact bridge is now localized as one explicit closeout object rather than a diffuse refresh label.",
        ),
        sign_base.row(
            "updated_pack_exact_charge_current_noether_closure_available_now",
            "pass" if available else "reject",
            "updated-pack exact charge-current / Noether-current closure available now",
            sign_base.truth(available),
            "The canon still does not expose the exact charge-current / Noether-current theorem as a derived object.",
        ),
        sign_base.row(
            "updated_pack_charge_current_closeout_closes_missing_action_blocker_now",
            "pass" if closes else "reject",
            "updated-pack charge-current closeout closes missing-action blocker now",
            sign_base.truth(closes),
            "This audit localizes the charge-current bridge honestly, but the exact closure itself is still absent.",
        ),
        sign_base.row(
            "updated_pack_low_order_jeff0_primary_closeout_required",
            "pass" if low_primary else "reject",
            "updated-pack low-order J_eff^0 primary closeout required",
            sign_base.truth(low_primary),
            "Once charge-current closeout is isolated, low-order J_eff^0 becomes the next honest exact closeout object.",
        ),
        sign_base.row(
            "blind_vector_observable_gate_still_blocked",
            "pass" if blind_blocked else "reject",
            "blind vector observable gate still blocked",
            sign_base.truth(blind_blocked),
            "Blind-vector direct computation remains downstream until the charge-current and low-order closeout objects move first.",
        ),
        sign_base.row(
            "farther_hybrid_continuation_reopen_required_now",
            "pass" if hybrid else "reject",
            "farther hybrid continuation reopen required now",
            sign_base.truth(hybrid),
            "Extra q-range evidence remains unnecessary because the blocker is still theorem-side and now sharpened to charge-current closeout.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "retained_scalar_residual_rel": float(prior["retained_scalar_residual_rel"]),
        "updated_pack_charge_current_closeout_audit_selected": selected,
        "retained_generic_u1_continuity_surface_available": retained_continuity,
        "retained_qball_charge_mapping_statement_available": retained_mapping,
        "retained_direct_qball_u1_identity_required": retained_identity,
        "retained_proxy_signed_density_available": retained_proxy,
        "updated_pack_charge_current_closeout_target_surface_explicit": target_surface,
        "updated_pack_charge_current_closeout_machine_readable_now": machine,
        "updated_pack_exact_charge_current_noether_closure_available_now": available,
        "updated_pack_charge_current_closeout_closes_missing_action_blocker_now": closes,
        "updated_pack_low_order_jeff0_primary_closeout_required": low_primary,
        "blind_vector_observable_gate_still_blocked": blind_blocked,
        "farther_hybrid_continuation_reopen_required_now": hybrid,
        "selected_primary_pack_update_surface": "updated_pack_charge_current_closeout_audit",
        "selected_secondary_pack_update_surface": "updated_pack_low_order_jeff0_closeout_audit",
        "selected_reserve_completion_lane": "blind_vector_after_charge_current_and_low_order_closeout",
        "selected_next_generation_route": NEXT_NAME,
        "recommended_next_route_or_none": NEXT,
        "selected_followup_route": FOLLOW_NAME,
        "selected_followup_route_or_none": FOLLOW,
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.2553",
        NAME + " declaration gate",
        {
            "source_files": {
                "status": sign_base.display_path(STATUS),
                "roadmap": sign_base.display_path(ROADMAP),
                "ai_context": sign_base.display_path(AI),
                "work_history_recent": sign_base.display_path(RECENT),
                "prior_gate": sign_base.display_path(PRIOR),
                "background_closeout_audit": sign_base.display_path(BG_CLOSEOUT),
                "charge_refresh_audit": sign_base.display_path(CHARGE_REFRESH),
                "low_order_refresh_audit": sign_base.display_path(LOW_REFRESH),
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
            "overall_status": "vector_qball_form_factor_updated_pack_charge_current_closeout_audit_declared",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_NAME],
        },
        {
            "formulas": formulas(),
            "hits": {
                "status_branch_hit": sign_base.hit(status_text, ".2547-.2550"),
                "roadmap_branch_hit": sign_base.hit(roadmap_text, ".2547-.2550"),
            },
        },
    )
    paths = write_artifact("declaration_gate", payload)
    route = {
        "generated_utc": sign_base.now_iso(),
        "phase": {"phase": 8, "step": "8.7.56.2554", "name": NAME + " route sync"},
        "inputs": paths,
        "rows": rows,
        "summary": summary,
        "decision": {
            "overall_status": "vector_qball_form_factor_updated_pack_charge_current_closeout_route_synced",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_NAME],
        },
        "evidence": {
            "formulas": formulas(),
            "disposition": {
                "charge_current_closeout_surface_explicit": target_surface,
                "charge_current_closeout_machine_readable_now": machine,
                "low_order_jeff0_primary_required": low_primary,
                "blind_vector_still_downstream": blind_blocked,
            },
        },
    }
    route_paths = write_artifact("route_sync", route)
    print("[ok] updated-pack charge-current closeout audit artifacts written")
    print(f"  declaration_gate: {paths['json']}")
    print(f"  route_sync: {route_paths['json']}")


if __name__ == "__main__":
    main()
