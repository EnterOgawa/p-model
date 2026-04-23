#!/usr/bin/env python3
"""Generate 8.7.56.2519-.2522 residual-origin refresh audit artifacts."""

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
NOTE = Path(r"C:\Users\ogawa\Downloads\trial2_vector_qball_next_steps_20260327.md")
PRIOR_A = build_metrics_paths(PUB, build_compact_artifact_stem("8.7.56.2511-2514", "updated_pack_blind_vector_refresh_audit", prefix="q"), "declaration_gate")["json"]
PRIOR_G = build_metrics_paths(PUB, build_compact_artifact_stem("8.7.56.2515-2518", "updated_pack_blind_vector_gate", prefix="q"), "declaration_gate")["json"]
TAG = "8.7.56.2519-2522"
NAME = "Trial-2 numeric alpha vector Q-ball form-factor updated-pack residual-origin refresh audit"
STEM = build_compact_artifact_stem(TAG, "updated_pack_residual_origin_refresh_audit", prefix="q")
PRIOR_CLASS = "vector_qball_form_factor_residual_origin_missing_action_updated_pack_blind_vector_audited_residual_origin_primary_theorem_refresh_next"
BRANCH_CLASS = "vector_qball_form_factor_residual_origin_missing_action_updated_pack_residual_origin_audited_theorem_refresh_gate"
NEXT_NAME = "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_residual_origin_gate_theorem_refresh"
NEXT = "8.7.56.2523"
FOLLOW_NAME = "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_theorem_refresh_audit"
FOLLOW = "8.7.56.2527"


# Function: write JSON/CSV artifacts.
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


# Function: provide formulas kept in the audit artifact.

def formulas() -> dict[str, str]:
    """Return formulas used in the residual-origin refresh audit."""
    return {
        "focus": "use q=0, q=q_theory, q=m0, pass/no-go gates, and consistency guards to localize residual origin",
        "pass_gate": "blind F_vector(q_theory) improves the retained scalar residual while F_vector(0)=1 stays fixed",
        "no_go_gate": "exact source theorem gives no vector correction or the proxy density fails as an exact current",
    }


# Function: execute the .2519-.2522 branch.

def main() -> None:
    """Execute the updated-pack residual-origin refresh audit."""
    for path in (STATUS, ROADMAP, AI, RECENT, P34, P36, P39, P55, P14, NOTE, PRIOR_A, PRIOR_G):
        sign_base.require(path)

    status_text = sign_base.read_text(STATUS)
    roadmap_text = sign_base.read_text(ROADMAP)
    note_text = sign_base.read_text(NOTE)
    a = sign_base.read_json(PRIOR_A)["summary"]
    g = sign_base.read_json(PRIOR_G)["summary"]
    selected = bool(g["gate_b_updated_pack_residual_origin_refresh_primary_selected"] and not g["gate_c_blind_vector_computation_primary_admissible_now"])
    q0 = bool(a["blind_vector_q0_checkpoint_explicit"])
    qth = bool(a["blind_vector_q_theory_checkpoint_explicit"] and a["blind_vector_residual_improvement_target_explicit"] and sign_base.hit(note_text, "blind `F_vector(q_theory)` が scalar 残差を改善する") is not None)
    m0 = bool(a["blind_vector_m0_checkpoint_explicit"])
    source_no_go = bool(sign_base.hit(note_text, "exact source theorem が vector correction を与えない") is not None)
    proxy_no_go = bool(sign_base.hit(note_text, "proxy `|f_0|^2 - |f_L|^2` が exact current と一致しない") is not None)
    step_e = bool(sign_base.hit(note_text, "### Step E. universality / consistency を通す") is not None)
    low_q = bool(sign_base.hit(note_text, "low-q Coulomb tail") is not None)
    soft = bool(sign_base.hit(note_text, "soft-photon / Thomson limit") is not None)
    surface = bool(selected and q0 and qth and m0 and source_no_go and proxy_no_go and step_e and low_q and soft)
    machine = bool(surface and g["selected_secondary_pack_update_surface"] == "updated_pack_theorem_refresh_after_residual_origin")
    direct_blind = bool(g["gate_c_blind_vector_computation_primary_admissible_now"])
    primary = bool(machine and not direct_blind and not g["exact_source_theorem_derived_now"])
    theorem_followup = bool(primary)
    closes = False
    hybrid = bool(g["farther_hybrid_continuation_reopen_required_now"])

    rows = [
        sign_base.row("updated_pack_residual_origin_refresh_audit_selected", "pass" if selected else "reject", "updated-pack residual-origin refresh audit selected", sign_base.truth(selected), "Selected because blind-vector direct computation remains blocked."),
        sign_base.row("residual_origin_q0_normalization_guard_explicit", "pass" if q0 else "reject", "residual-origin q=0 guard explicit", sign_base.truth(q0), "Keep q=0 normalization explicit."),
        sign_base.row("residual_origin_q_theory_improvement_discriminator_explicit", "pass" if qth else "reject", "residual-origin q_theory discriminator explicit", sign_base.truth(qth), "Tie the residual-origin question to the retained 1.9% checkpoint."),
        sign_base.row("residual_origin_m0_tail_guard_explicit", "pass" if m0 else "reject", "residual-origin q=m0 guard explicit", sign_base.truth(m0), "Retain the m0 checkpoint as part of the same surface."),
        sign_base.row("residual_origin_exact_source_theorem_no_go_explicit", "pass" if source_no_go else "reject", "exact source-theorem no-go explicit", sign_base.truth(source_no_go), "The no-go branch is explicit."),
        sign_base.row("residual_origin_proxy_exact_current_no_go_explicit", "pass" if proxy_no_go else "reject", "proxy-vs-exact-current no-go explicit", sign_base.truth(proxy_no_go), "The proxy-current mismatch branch is explicit."),
        sign_base.row("universality_consistency_guard_explicit", "pass" if (step_e and low_q and soft) else "reject", "universality/consistency guard explicit", sign_base.truth(step_e and low_q and soft), "Residual-origin refresh still sits under Step E guards."),
        sign_base.row("updated_pack_residual_origin_refresh_target_surface_explicit", "pass" if surface else "reject", "updated-pack residual-origin target surface explicit", sign_base.truth(surface), "The discriminator surface is now explicit."),
        sign_base.row("updated_pack_residual_origin_refresh_machine_readable_now", "pass" if machine else "reject", "updated-pack residual-origin machine-readable now", sign_base.truth(machine), "The residual-origin lane is now localized on a concrete surface."),
        sign_base.row("residual_origin_refresh_supports_missing_action_primary_now", "pass" if primary else "reject", "residual-origin supports missing-action primary now", sign_base.truth(primary), "The lane still points back to the theorem-side blocker."),
        sign_base.row("updated_pack_theorem_refresh_followup_required", "pass" if theorem_followup else "reject", "updated-pack theorem refresh followup required", sign_base.truth(theorem_followup), "The honest followup is theorem refresh."),
        sign_base.row("updated_pack_residual_origin_refresh_closes_missing_action_blocker_now", "pass" if closes else "reject", "updated-pack residual-origin closes blocker now", sign_base.truth(closes), "Residual-origin refresh does not close the blocker by itself."),
        sign_base.row("farther_hybrid_continuation_reopen_required_now", "pass" if hybrid else "reject", "farther hybrid continuation reopen required now", sign_base.truth(hybrid), "Extra q-range remains reserve-only."),
    ]
    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "retained_scalar_residual_rel": float(g["retained_scalar_residual_rel"]),
        "updated_pack_residual_origin_refresh_audit_selected": selected,
        "residual_origin_q0_normalization_guard_explicit": q0,
        "residual_origin_q_theory_improvement_discriminator_explicit": qth,
        "residual_origin_m0_tail_guard_explicit": m0,
        "residual_origin_exact_source_theorem_no_go_explicit": source_no_go,
        "residual_origin_proxy_exact_current_no_go_explicit": proxy_no_go,
        "universality_consistency_step_explicit": step_e,
        "low_q_coulomb_guard_explicit": low_q,
        "soft_photon_limit_guard_explicit": soft,
        "updated_pack_residual_origin_refresh_target_surface_explicit": surface,
        "updated_pack_residual_origin_refresh_machine_readable_now": machine,
        "direct_blind_vector_computation_primary_admissible_now": direct_blind,
        "residual_origin_refresh_supports_missing_action_primary_now": primary,
        "updated_pack_theorem_refresh_followup_required": theorem_followup,
        "updated_pack_residual_origin_refresh_closes_missing_action_blocker_now": closes,
        "farther_hybrid_continuation_reopen_required_now": hybrid,
        "selected_primary_pack_update_surface": "residual_origin_refresh_after_blind_vector",
        "selected_secondary_pack_update_surface": "updated_pack_theorem_refresh_after_residual_origin",
        "selected_reserve_completion_lane": "farther_hybrid_extra_q_range_only",
        "selected_next_generation_route": NEXT_NAME,
        "recommended_next_route_or_none": NEXT,
        "selected_followup_route": FOLLOW_NAME,
        "selected_followup_route_or_none": FOLLOW,
        "physical_reject_required": False,
    }
    payload = sign_base.payload("8.7.56.2521", NAME + " declaration gate", {"source_files": {"status": sign_base.display_path(STATUS), "roadmap": sign_base.display_path(ROADMAP), "ai_context": sign_base.display_path(AI), "work_history_recent": sign_base.display_path(RECENT), "prior_audit": sign_base.display_path(PRIOR_A), "prior_gate": sign_base.display_path(PRIOR_G), "next_steps": sign_base.display_path(NOTE)}, "routes": {"prior_route": PRIOR_CLASS, "current_route": BRANCH_CLASS, "next_route_name": NEXT_NAME, "next_route": NEXT, "followup_route_name": FOLLOW_NAME, "followup_route": FOLLOW}}, rows, summary, {"overall_status": "vector_qball_form_factor_updated_pack_residual_origin_refresh_audit_declared", "branch_completed": True, "next_required_artifacts": [NEXT_NAME]}, {"formulas": formulas(), "hits": {"status_branch_hit": sign_base.hit(status_text, "8.7.56.2519"), "roadmap_branch_hit": sign_base.hit(roadmap_text, ".2515-.2518"), "q_theory_pass_hit": sign_base.hit(note_text, "blind `F_vector(q_theory)` が scalar 残差を改善する"), "source_no_go_hit": sign_base.hit(note_text, "exact source theorem が vector correction を与えない"), "proxy_no_go_hit": sign_base.hit(note_text, "proxy `|f_0|^2 - |f_L|^2` が exact current と一致しない")}})
    paths = write_artifact("declaration_gate", payload)
    route = {"generated_utc": sign_base.now_iso(), "phase": {"phase": 8, "step": "8.7.56.2522", "name": NAME + " route sync"}, "inputs": paths, "rows": rows, "summary": summary, "decision": {"overall_status": "vector_qball_form_factor_updated_pack_residual_origin_refresh_route_synced", "branch_completed": True, "next_required_artifacts": [NEXT_NAME]}, "evidence": {"formulae": formulas(), "disposition": {"residual_origin_refresh_surface_explicit": surface, "residual_origin_refresh_machine_readable_now": machine, "theorem_refresh_followup_required": theorem_followup, "direct_blind_vector_still_blocked": not direct_blind}}}
    route_paths = write_artifact("route_sync", route)
    print("[ok] updated-pack residual-origin refresh audit artifacts written")
    print(f"  declaration_gate: {paths['json']}")
    print(f"  route_sync: {route_paths['json']}")


if __name__ == "__main__":
    main()
