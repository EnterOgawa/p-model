#!/usr/bin/env python3
"""Generate 8.7.56.2531-.2534 updated-pack theorem gate artifacts."""

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
PRIOR = build_metrics_paths(PUB, build_compact_artifact_stem("8.7.56.2527-2530", "updated_pack_theorem_refresh_audit", prefix="q"), "declaration_gate")["json"]
TAG = "8.7.56.2531-2534"
NAME = "Trial-2 numeric alpha vector Q-ball form-factor updated-pack theorem gate"
STEM = build_compact_artifact_stem(TAG, "updated_pack_theorem_gate", prefix="q")
PRIOR_CLASS = "vector_qball_form_factor_residual_origin_missing_action_updated_pack_theorem_refresh_audited_exact_source_theorem_closeout_gate"
BRANCH_CLASS = "vector_qball_form_factor_residual_origin_missing_action_updated_pack_theorem_refresh_audited_exact_source_theorem_closeout_primary_blind_vector_reserve_next"
NEXT_NAME = "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_exact_source_theorem_closeout_audit"
NEXT = "8.7.56.2535"
FOLLOW_NAME = "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_exact_source_theorem_gate_blind_vector_reserve_refresh"
FOLLOW = "8.7.56.2539"


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


# 関数: theorem gate で使う式を返す。

def formulas() -> dict[str, str]:
    """Return formulas used in the theorem gate."""
    return {
        "gate_a": "Gate A = updated-pack theorem-refresh closeout surface explicit and machine-readable",
        "gate_b": "Gate B = exact source-theorem closeout promoted as the next primary lane",
        "gate_c": "Gate C = blind-vector direct computation primary admissible now",
    }


# 関数: `.2531-.2534` を実行する。

def main() -> None:
    """Execute the updated-pack theorem gate."""
    for path in (STATUS, ROADMAP, AI, RECENT, P34, P36, P39, P55, P14, PRIOR):
        sign_base.require(path)

    status_text = sign_base.read_text(STATUS)
    roadmap_text = sign_base.read_text(ROADMAP)
    p = sign_base.read_json(PRIOR)["summary"]

    gate_a = bool(p["updated_pack_theorem_refresh_target_surface_explicit"] and p["updated_pack_theorem_refresh_machine_readable_now"])
    gate_b = bool(p["updated_pack_exact_source_theorem_closeout_primary_supported"])
    gate_c = bool(p["direct_blind_vector_computation_primary_admissible_now"])
    exact_source = bool(p["exact_source_theorem_derived_now"])
    hybrid = bool(p["farther_hybrid_continuation_reopen_required_now"])
    old_retry = False

    rows = [
        sign_base.row("gate_a_updated_pack_theorem_refresh_surface_explicit", "pass" if gate_a else "reject", "Gate A updated-pack theorem-refresh surface explicit", sign_base.truth(gate_a), "The theorem-refresh audit already localized one explicit machine-readable closeout surface."),
        sign_base.row("gate_b_updated_pack_exact_source_theorem_closeout_primary_selected", "pass" if gate_b else "reject", "Gate B updated-pack exact source-theorem closeout primary selected", sign_base.truth(gate_b), "The honest next lane is now exact source-theorem closeout itself."),
        sign_base.row("gate_c_blind_vector_computation_primary_admissible_now", "pass" if gate_c else "reject", "Gate C blind-vector computation primary admissible now", sign_base.truth(gate_c), "Blind-vector direct computation remains downstream of theorem closeout."),
        sign_base.row("exact_source_theorem_derived_now", "pass" if exact_source else "reject", "exact source theorem derived now", sign_base.truth(exact_source), "This gate only promotes closeout; it does not derive the theorem."),
        sign_base.row("farther_hybrid_continuation_reopen_required_now", "pass" if hybrid else "reject", "farther hybrid continuation reopen required now", sign_base.truth(hybrid), "Extra q-range remains reserve-only."),
        sign_base.row("old_density_proxy_eigenvalue_retry_admissible_now", "pass" if old_retry else "reject", "old density/proxy/eigenvalue retry admissible now", sign_base.truth(old_retry), "Exhausted pre-update retry families remain closed."),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "retained_scalar_residual_rel": float(p["retained_scalar_residual_rel"]),
        "gate_a_updated_pack_theorem_refresh_surface_explicit": gate_a,
        "gate_b_updated_pack_exact_source_theorem_closeout_primary_selected": gate_b,
        "gate_c_blind_vector_computation_primary_admissible_now": gate_c,
        "exact_source_theorem_derived_now": exact_source,
        "farther_hybrid_continuation_reopen_required_now": hybrid,
        "old_density_proxy_eigenvalue_retry_admissible_now": old_retry,
        "hybrid_supporting_evidence_reopen_required": hybrid,
        "selected_primary_pack_update_surface": "updated_pack_exact_source_theorem_closeout_after_theorem_refresh",
        "selected_secondary_pack_update_surface": "updated_pack_blind_vector_revisit_after_theorem_closeout",
        "selected_reserve_completion_lane": "farther_hybrid_extra_q_range_only",
        "selected_next_generation_route": NEXT_NAME,
        "recommended_next_route_or_none": NEXT,
        "selected_followup_route": FOLLOW_NAME,
        "selected_followup_route_or_none": FOLLOW,
        "physical_reject_required": False,
    }

    payload = sign_base.payload("8.7.56.2533", NAME + " declaration gate", {"source_files": {"status": sign_base.display_path(STATUS), "roadmap": sign_base.display_path(ROADMAP), "ai_context": sign_base.display_path(AI), "work_history_recent": sign_base.display_path(RECENT), "prior_audit": sign_base.display_path(PRIOR)}, "routes": {"prior_route": PRIOR_CLASS, "current_route": BRANCH_CLASS, "next_route_name": NEXT_NAME, "next_route": NEXT, "followup_route_name": FOLLOW_NAME, "followup_route": FOLLOW}}, rows, summary, {"overall_status": "vector_qball_form_factor_updated_pack_theorem_gate_declared", "branch_completed": True, "next_required_artifacts": [NEXT_NAME]}, {"formulas": formulas(), "hits": {"status_branch_hit": sign_base.hit(status_text, "8.7.56.2527"), "roadmap_branch_hit": sign_base.hit(roadmap_text, ".2523-.2526")}})
    paths = write_artifact("declaration_gate", payload)
    route = {"generated_utc": sign_base.now_iso(), "phase": {"phase": 8, "step": "8.7.56.2534", "name": NAME + " route sync"}, "inputs": paths, "rows": rows, "summary": summary, "decision": {"overall_status": "vector_qball_form_factor_updated_pack_theorem_gate_route_synced", "branch_completed": True, "next_required_artifacts": [NEXT_NAME]}, "evidence": {"formulae": formulas(), "disposition": {"theorem_refresh_surface_explicit": gate_a, "exact_source_theorem_closeout_primary_selected": gate_b, "direct_blind_vector_still_blocked": not gate_c}}}
    route_paths = write_artifact("route_sync", route)
    print("[ok] updated-pack theorem gate artifacts written")
    print(f"  declaration_gate: {paths['json']}")
    print(f"  route_sync: {route_paths['json']}")


if __name__ == "__main__":
    main()
