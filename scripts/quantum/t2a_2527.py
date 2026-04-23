#!/usr/bin/env python3
"""Generate 8.7.56.2527-.2530 updated-pack theorem refresh audit artifacts."""

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
PRIOR = build_metrics_paths(PUB, build_compact_artifact_stem("8.7.56.2523-2526", "updated_pack_residual_origin_gate", prefix="q"), "declaration_gate")["json"]
SRC = build_metrics_paths(PUB, build_compact_artifact_stem("8.7.56.2479-2482", "updated_pack_exact_source_theorem_refresh_audit", prefix="q"), "declaration_gate")["json"]
BG = build_metrics_paths(PUB, build_compact_artifact_stem("8.7.56.2487-2490", "updated_pack_exact_qball_background_expansion_audit", prefix="q"), "declaration_gate")["json"]
CHARGE = build_metrics_paths(PUB, build_compact_artifact_stem("8.7.56.2495-2498", "updated_pack_exact_charge_current_refresh_audit", prefix="q"), "declaration_gate")["json"]
LOW = build_metrics_paths(PUB, build_compact_artifact_stem("8.7.56.2503-2506", "updated_pack_exact_low_order_jeff0_refresh_audit", prefix="q"), "declaration_gate")["json"]
TAG = "8.7.56.2527-2530"
NAME = "Trial-2 numeric alpha vector Q-ball form-factor updated-pack theorem refresh audit"
STEM = build_compact_artifact_stem(TAG, "updated_pack_theorem_refresh_audit", prefix="q")
PRIOR_CLASS = "vector_qball_form_factor_residual_origin_missing_action_updated_pack_residual_origin_audited_theorem_refresh_primary_blind_vector_reserve_next"
BRANCH_CLASS = "vector_qball_form_factor_residual_origin_missing_action_updated_pack_theorem_refresh_audited_exact_source_theorem_closeout_gate"
NEXT_NAME = "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_theorem_gate_blind_vector_reserve_refresh"
NEXT = "8.7.56.2531"
FOLLOW_NAME = "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_exact_source_theorem_closeout_audit"
FOLLOW = "8.7.56.2535"


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


# 関数: theorem refresh audit で使う式を返す。

def formulas() -> dict[str, str]:
    """Return formulas used in the theorem refresh audit."""
    return {
        "surface": "residual-origin discriminator -> exact source-theorem closeout -> blind-vector reserve",
        "closeout_order": "background expansion -> charge-current closure -> low-order J_eff^0 synthesis",
        "why": "background expansion is the missing primitive, charge-current closure is the next exact bridge, and low-order J_eff^0 depends on both",
    }


# 関数: `.2527-.2530` を実行する。

def main() -> None:
    """Execute the updated-pack theorem refresh audit."""
    for path in (STATUS, ROADMAP, AI, RECENT, P34, P36, P39, P55, P14, PRIOR, SRC, BG, CHARGE, LOW):
        sign_base.require(path)

    status_text = sign_base.read_text(STATUS)
    roadmap_text = sign_base.read_text(ROADMAP)
    p = sign_base.read_json(PRIOR)["summary"]
    src = sign_base.read_json(SRC)["summary"]
    bg = sign_base.read_json(BG)["summary"]
    charge = sign_base.read_json(CHARGE)["summary"]
    low = sign_base.read_json(LOW)["summary"]

    selected = bool(p["gate_b_updated_pack_theorem_refresh_primary_selected"] and not p["gate_c_residual_origin_closeout_primary_admissible_now"])
    surface = bool(selected and src["updated_pack_exact_source_theorem_refresh_surface_explicit_now"] and p["gate_a_updated_pack_residual_origin_refresh_surface_explicit"])
    bg_dep = bool(bg["updated_pack_exact_qball_background_expansion_target_surface_explicit"] and not bg["updated_pack_exact_qball_background_expansion_available_now"])
    charge_dep = bool(charge["updated_pack_exact_charge_current_refresh_target_surface_explicit"] and not charge["updated_pack_exact_charge_current_noether_closure_available_now"])
    low_dep = bool(low["updated_pack_low_order_jeff0_refresh_target_surface_explicit"] and not low["updated_pack_exact_low_order_jeff0_formula_available_now"])
    machine = bool(surface and src["updated_pack_exact_source_theorem_refresh_order_stable"] and bg["updated_pack_exact_qball_background_expansion_machine_readable_now"] and charge["updated_pack_exact_charge_current_refresh_machine_readable_now"] and low["updated_pack_low_order_jeff0_refresh_machine_readable_now"])
    closeout_primary = bool(machine and bg_dep and charge_dep and low_dep and not src["updated_pack_exact_source_theorem_derived_now"])
    blind_follow = bool(low["updated_pack_blind_vector_refresh_followup_required"] and not p["direct_blind_vector_computation_primary_admissible_now"])
    exact_source = False
    direct_blind = bool(p["direct_blind_vector_computation_primary_admissible_now"])
    closes = False
    hybrid = bool(p["farther_hybrid_continuation_reopen_required_now"])

    rows = [
        sign_base.row("updated_pack_theorem_refresh_audit_selected", "pass" if selected else "reject", "updated-pack theorem refresh audit selected", sign_base.truth(selected), "Residual-origin gate already promoted theorem refresh as the next honest mainline."),
        sign_base.row("updated_pack_theorem_refresh_target_surface_explicit", "pass" if surface else "reject", "updated-pack theorem refresh target surface explicit", sign_base.truth(surface), "Residual-origin and exact source-theorem refresh surfaces now meet on one target surface."),
        sign_base.row("updated_pack_background_expansion_dependency_explicit", "pass" if bg_dep else "reject", "updated-pack background-expansion dependency explicit", sign_base.truth(bg_dep), "Background expansion remains the missing primitive inside theorem closeout."),
        sign_base.row("updated_pack_charge_current_dependency_explicit", "pass" if charge_dep else "reject", "updated-pack charge-current dependency explicit", sign_base.truth(charge_dep), "Charge-current closure remains the next dependent exact bridge."),
        sign_base.row("updated_pack_low_order_jeff0_dependency_explicit", "pass" if low_dep else "reject", "updated-pack low-order J_eff^0 dependency explicit", sign_base.truth(low_dep), "Low-order J_eff^0 remains the synthesis object."),
        sign_base.row("updated_pack_theorem_refresh_machine_readable_now", "pass" if machine else "reject", "updated-pack theorem refresh machine-readable now", sign_base.truth(machine), "The theorem-refresh lane now has one machine-readable closeout surface."),
        sign_base.row("updated_pack_exact_source_theorem_closeout_primary_supported", "pass" if closeout_primary else "reject", "updated-pack exact source-theorem closeout primary supported", sign_base.truth(closeout_primary), "The honest next closeout lane is now the exact source theorem itself."),
        sign_base.row("updated_pack_blind_vector_reserve_followup_required", "pass" if blind_follow else "reject", "updated-pack blind-vector reserve followup required", sign_base.truth(blind_follow), "Blind-vector evaluation remains downstream of theorem closeout."),
        sign_base.row("exact_source_theorem_derived_now", "pass" if exact_source else "reject", "exact source theorem derived now", sign_base.truth(exact_source), "Theorem refresh audit still does not derive the exact theorem."),
        sign_base.row("updated_pack_theorem_refresh_closes_missing_action_blocker_now", "pass" if closes else "reject", "updated-pack theorem refresh closes missing-action blocker now", sign_base.truth(closes), "The refresh lane is explicit, but the blocker itself is still open."),
        sign_base.row("farther_hybrid_continuation_reopen_required_now", "pass" if hybrid else "reject", "farther hybrid continuation reopen required now", sign_base.truth(hybrid), "Extra q-range remains reserve-only because the blocker is still theorem-side."),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "retained_scalar_residual_rel": float(p["retained_scalar_residual_rel"]),
        "updated_pack_theorem_refresh_audit_selected": selected,
        "updated_pack_theorem_refresh_target_surface_explicit": surface,
        "updated_pack_background_expansion_dependency_explicit": bg_dep,
        "updated_pack_charge_current_dependency_explicit": charge_dep,
        "updated_pack_low_order_jeff0_dependency_explicit": low_dep,
        "updated_pack_theorem_refresh_machine_readable_now": machine,
        "updated_pack_exact_source_theorem_closeout_primary_supported": closeout_primary,
        "updated_pack_blind_vector_reserve_followup_required": blind_follow,
        "exact_source_theorem_derived_now": exact_source,
        "direct_blind_vector_computation_primary_admissible_now": direct_blind,
        "updated_pack_theorem_refresh_closes_missing_action_blocker_now": closes,
        "farther_hybrid_continuation_reopen_required_now": hybrid,
        "selected_primary_pack_update_surface": "updated_pack_exact_source_theorem_closeout_after_theorem_refresh",
        "selected_secondary_pack_update_surface": "updated_pack_blind_vector_revisit_after_theorem_closeout",
        "selected_reserve_completion_lane": "farther_hybrid_extra_q_range_only",
        "selected_next_generation_route": NEXT_NAME,
        "recommended_next_route_or_none": NEXT,
        "selected_followup_route": FOLLOW_NAME,
        "selected_followup_route_or_none": FOLLOW,
        "physical_reject_required": False,
    }

    payload = sign_base.payload("8.7.56.2529", NAME + " declaration gate", {"source_files": {"status": sign_base.display_path(STATUS), "roadmap": sign_base.display_path(ROADMAP), "ai_context": sign_base.display_path(AI), "work_history_recent": sign_base.display_path(RECENT), "prior_gate": sign_base.display_path(PRIOR), "source_refresh_audit": sign_base.display_path(SRC), "background_audit": sign_base.display_path(BG), "charge_audit": sign_base.display_path(CHARGE), "low_order_audit": sign_base.display_path(LOW)}, "routes": {"prior_route": PRIOR_CLASS, "current_route": BRANCH_CLASS, "next_route_name": NEXT_NAME, "next_route": NEXT, "followup_route_name": FOLLOW_NAME, "followup_route": FOLLOW}}, rows, summary, {"overall_status": "vector_qball_form_factor_updated_pack_theorem_refresh_declared", "branch_completed": True, "next_required_artifacts": [NEXT_NAME]}, {"formulas": formulas(), "hits": {"status_branch_hit": sign_base.hit(status_text, "8.7.56.2527"), "roadmap_branch_hit": sign_base.hit(roadmap_text, ".2523-.2526")}})
    paths = write_artifact("declaration_gate", payload)
    route = {"generated_utc": sign_base.now_iso(), "phase": {"phase": 8, "step": "8.7.56.2530", "name": NAME + " route sync"}, "inputs": paths, "rows": rows, "summary": summary, "decision": {"overall_status": "vector_qball_form_factor_updated_pack_theorem_refresh_route_synced", "branch_completed": True, "next_required_artifacts": [NEXT_NAME]}, "evidence": {"formulae": formulas(), "disposition": {"theorem_refresh_surface_explicit": surface, "theorem_refresh_machine_readable_now": machine, "exact_source_theorem_closeout_primary_supported": closeout_primary, "blind_vector_reserve_followup_required": blind_follow}}}
    route_paths = write_artifact("route_sync", route)
    print("[ok] updated-pack theorem refresh audit artifacts written")
    print(f"  declaration_gate: {paths['json']}")
    print(f"  route_sync: {route_paths['json']}")


if __name__ == "__main__":
    main()
