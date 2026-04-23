#!/usr/bin/env python3
"""Generate 8.7.56.2535-.2538 updated-pack exact source-theorem closeout audit artifacts."""

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
    build_compact_artifact_stem("8.7.56.2531-2534", "updated_pack_theorem_gate", prefix="q"),
    "declaration_gate",
)["json"]
REFRESH = build_metrics_paths(
    PUB,
    build_compact_artifact_stem("8.7.56.2479-2482", "updated_pack_exact_source_theorem_refresh_audit", prefix="q"),
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
TAG = "8.7.56.2535-2538"
NAME = "Trial-2 numeric alpha vector Q-ball form-factor updated-pack exact source-theorem closeout audit"
STEM = build_compact_artifact_stem(TAG, "updated_pack_exact_source_theorem_closeout_audit", prefix="q")
PRIOR_CLASS = "vector_qball_form_factor_residual_origin_missing_action_updated_pack_theorem_refresh_audited_exact_source_theorem_closeout_primary_blind_vector_reserve_next"
BRANCH_CLASS = "vector_qball_form_factor_residual_origin_missing_action_updated_pack_exact_source_theorem_closeout_audited_background_expansion_primary_charge_current_secondary_low_order_jeff0_tertiary_gate"
NEXT_NAME = "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_exact_source_theorem_gate_blind_vector_reserve_refresh"
NEXT = "8.7.56.2539"
FOLLOW_NAME = "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_background_expansion_closeout_audit"
FOLLOW = "8.7.56.2543"


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


# 関数: closeout audit で使う式を返す。

def formulas() -> dict[str, str]:
    """Return formulas used in the exact source-theorem closeout audit."""
    return {
        "closeout_surface": "residual-origin discriminator -> theorem-refresh surface -> exact source-theorem closeout",
        "closeout_order": "background expansion -> charge-current / Noether-current closure -> low-order J_eff^0 synthesis",
        "why": "background expansion is still the first missing primitive, charge-current closure is the next exact bridge, and low-order J_eff^0 remains the dependent synthesis object.",
    }


# 関数: `.2535-.2538` を実行する。

def main() -> None:
    """Execute the updated-pack exact source-theorem closeout audit."""
    for path in (STATUS, ROADMAP, AI, RECENT, P34, P36, P39, P55, P14, PRIOR, REFRESH, BG, CHARGE, LOW):
        sign_base.require(path)

    status_text = sign_base.read_text(STATUS)
    roadmap_text = sign_base.read_text(ROADMAP)
    p = sign_base.read_json(PRIOR)["summary"]
    refresh = sign_base.read_json(REFRESH)["summary"]
    bg = sign_base.read_json(BG)["summary"]
    charge = sign_base.read_json(CHARGE)["summary"]
    low = sign_base.read_json(LOW)["summary"]

    selected = bool(
        p["gate_b_updated_pack_exact_source_theorem_closeout_primary_selected"]
        and not p["gate_c_blind_vector_computation_primary_admissible_now"]
    )
    surface = bool(
        selected
        and p["gate_a_updated_pack_theorem_refresh_surface_explicit"]
        and refresh["updated_pack_exact_source_theorem_refresh_surface_explicit_now"]
        and refresh["updated_pack_exact_source_theorem_refresh_order_stable"]
    )
    background_primary = bool(
        surface
        and bg["updated_pack_exact_qball_background_expansion_target_surface_explicit"]
        and bg["updated_pack_exact_qball_background_expansion_machine_readable_now"]
        and not bg["updated_pack_exact_qball_background_expansion_available_now"]
    )
    charge_secondary = bool(
        surface
        and charge["updated_pack_exact_charge_current_refresh_target_surface_explicit"]
        and charge["updated_pack_exact_charge_current_refresh_machine_readable_now"]
        and not charge["updated_pack_exact_charge_current_noether_closure_available_now"]
    )
    low_tertiary = bool(
        surface
        and low["updated_pack_low_order_jeff0_refresh_target_surface_explicit"]
        and low["updated_pack_low_order_jeff0_refresh_machine_readable_now"]
        and not low["updated_pack_exact_low_order_jeff0_formula_available_now"]
    )
    machine = bool(surface and background_primary and charge_secondary and low_tertiary)
    order_stable = bool(background_primary and charge_secondary and low_tertiary)
    exact_source = False
    closes = False
    blind_blocked = bool(not p["gate_c_blind_vector_computation_primary_admissible_now"])
    hybrid = bool(p["farther_hybrid_continuation_reopen_required_now"])

    rows = [
        sign_base.row(
            "updated_pack_exact_source_theorem_closeout_audit_selected",
            "pass" if selected else "reject",
            "updated-pack exact source-theorem closeout audit selected",
            sign_base.truth(selected),
            "The theorem gate already promoted exact source-theorem closeout as the next honest mainline.",
        ),
        sign_base.row(
            "updated_pack_exact_source_theorem_closeout_target_surface_explicit",
            "pass" if surface else "reject",
            "updated-pack exact source-theorem closeout target surface explicit",
            sign_base.truth(surface),
            "The residual-origin discriminator, theorem-refresh surface, and exact-source ordering now meet on one closeout surface.",
        ),
        sign_base.row(
            "updated_pack_background_expansion_primary_closeout_supported",
            "pass" if background_primary else "reject",
            "updated-pack background-expansion primary closeout supported",
            sign_base.truth(background_primary),
            "Background expansion remains the first missing primitive inside exact source-theorem closeout.",
        ),
        sign_base.row(
            "updated_pack_charge_current_secondary_closeout_supported",
            "pass" if charge_secondary else "reject",
            "updated-pack charge-current secondary closeout supported",
            sign_base.truth(charge_secondary),
            "Charge-current / Noether-current closure remains the next exact bridge after the missing primitive.",
        ),
        sign_base.row(
            "updated_pack_low_order_jeff0_tertiary_closeout_supported",
            "pass" if low_tertiary else "reject",
            "updated-pack low-order J_eff^0 tertiary closeout supported",
            sign_base.truth(low_tertiary),
            "Low-order J_eff^0 remains the dependent synthesis object after background expansion and charge-current closure.",
        ),
        sign_base.row(
            "updated_pack_exact_source_theorem_closeout_machine_readable_now",
            "pass" if machine else "reject",
            "updated-pack exact source-theorem closeout machine-readable now",
            sign_base.truth(machine),
            "The closeout lane now has one explicit machine-readable dependency stack.",
        ),
        sign_base.row(
            "updated_pack_exact_source_theorem_closeout_order_stable",
            "pass" if order_stable else "reject",
            "updated-pack exact source-theorem closeout order stable",
            sign_base.truth(order_stable),
            "The closeout route is now ordered rather than a flat theorem-side blocker list.",
        ),
        sign_base.row(
            "exact_source_theorem_derived_now",
            "pass" if exact_source else "reject",
            "exact source theorem derived now",
            sign_base.truth(exact_source),
            "The audit still organizes the closeout lane but does not derive the theorem itself.",
        ),
        sign_base.row(
            "updated_pack_exact_source_theorem_closeout_closes_missing_action_blocker_now",
            "pass" if closes else "reject",
            "updated-pack exact source-theorem closeout closes missing-action blocker now",
            sign_base.truth(closes),
            "The missing-action blocker remains open because every exact theorem primitive is still absent as a derived object.",
        ),
        sign_base.row(
            "blind_vector_observable_gate_still_blocked",
            "pass" if blind_blocked else "reject",
            "blind vector observable gate still blocked",
            sign_base.truth(blind_blocked),
            "Blind-vector direct computation remains downstream of exact source-theorem closeout.",
        ),
        sign_base.row(
            "farther_hybrid_continuation_reopen_required_now",
            "pass" if hybrid else "reject",
            "farther hybrid continuation reopen required now",
            sign_base.truth(hybrid),
            "Extra q-range evidence remains unnecessary because the blocker is still theorem-side and now explicitly ordered.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "retained_scalar_residual_rel": float(p["retained_scalar_residual_rel"]),
        "updated_pack_exact_source_theorem_closeout_audit_selected": selected,
        "updated_pack_exact_source_theorem_closeout_target_surface_explicit": surface,
        "updated_pack_background_expansion_primary_closeout_supported": background_primary,
        "updated_pack_charge_current_secondary_closeout_supported": charge_secondary,
        "updated_pack_low_order_jeff0_tertiary_closeout_supported": low_tertiary,
        "updated_pack_exact_source_theorem_closeout_machine_readable_now": machine,
        "updated_pack_exact_source_theorem_closeout_order_stable": order_stable,
        "exact_source_theorem_derived_now": exact_source,
        "updated_pack_exact_source_theorem_closeout_closes_missing_action_blocker_now": closes,
        "blind_vector_observable_gate_still_blocked": blind_blocked,
        "farther_hybrid_continuation_reopen_required_now": hybrid,
        "selected_primary_pack_update_surface": "updated_pack_background_expansion_closeout_audit",
        "selected_secondary_pack_update_surface": "updated_pack_exact_charge_current_noether_refresh",
        "selected_tertiary_pack_update_surface": "updated_pack_exact_low_order_jeff0_formula_synthesis",
        "selected_reserve_completion_lane": "blind_vector_after_exact_source_theorem_closeout",
        "selected_next_generation_route": NEXT_NAME,
        "recommended_next_route_or_none": NEXT,
        "selected_followup_route": FOLLOW_NAME,
        "selected_followup_route_or_none": FOLLOW,
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.2537",
        NAME + " declaration gate",
        {
            "source_files": {
                "status": sign_base.display_path(STATUS),
                "roadmap": sign_base.display_path(ROADMAP),
                "ai_context": sign_base.display_path(AI),
                "work_history_recent": sign_base.display_path(RECENT),
                "prior_gate": sign_base.display_path(PRIOR),
                "source_theorem_refresh_audit": sign_base.display_path(REFRESH),
                "background_audit": sign_base.display_path(BG),
                "charge_audit": sign_base.display_path(CHARGE),
                "low_order_audit": sign_base.display_path(LOW),
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
            "overall_status": "vector_qball_form_factor_updated_pack_exact_source_theorem_closeout_audit_declared",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_NAME],
        },
        {
            "formulas": formulas(),
            "hits": {
                "status_branch_hit": sign_base.hit(status_text, ".2531-.2534"),
                "roadmap_branch_hit": sign_base.hit(roadmap_text, ".2531-.2534"),
            },
        },
    )
    paths = write_artifact("declaration_gate", payload)
    route = {
        "generated_utc": sign_base.now_iso(),
        "phase": {"phase": 8, "step": "8.7.56.2538", "name": NAME + " route sync"},
        "inputs": paths,
        "rows": rows,
        "summary": summary,
        "decision": {
            "overall_status": "vector_qball_form_factor_updated_pack_exact_source_theorem_closeout_route_synced",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_NAME],
        },
        "evidence": {
            "formulas": formulas(),
            "disposition": {
                "background_expansion_primary": background_primary,
                "charge_current_secondary": charge_secondary,
                "low_order_jeff0_tertiary": low_tertiary,
                "blind_vector_still_blocked": blind_blocked,
            },
        },
    }
    route_paths = write_artifact("route_sync", route)
    print("[ok] updated-pack exact source-theorem closeout audit artifacts written")
    print(f"  declaration_gate: {paths['json']}")
    print(f"  route_sync: {route_paths['json']}")


if __name__ == "__main__":
    main()
