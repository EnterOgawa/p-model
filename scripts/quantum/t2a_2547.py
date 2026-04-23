#!/usr/bin/env python3
"""Generate 8.7.56.2547-.2550 updated-pack background-expansion gate artifacts."""

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
    build_compact_artifact_stem("8.7.56.2543-2546", "updated_pack_background_expansion_closeout_audit", prefix="q"),
    "declaration_gate",
)["json"]
TAG = "8.7.56.2547-2550"
NAME = "Trial-2 numeric alpha vector Q-ball form-factor updated-pack background-expansion gate"
STEM = build_compact_artifact_stem(TAG, "updated_pack_background_expansion_closeout_gate", prefix="q")
PRIOR_CLASS = "vector_qball_form_factor_residual_origin_missing_action_updated_pack_background_expansion_closeout_audited_charge_current_primary_low_order_jeff0_secondary_gate"
BRANCH_CLASS = "vector_qball_form_factor_residual_origin_missing_action_updated_pack_background_expansion_closeout_audited_charge_current_primary_low_order_jeff0_secondary_blind_vector_reserve_next"
NEXT_NAME = "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_charge_current_closeout_audit"
NEXT = "8.7.56.2551"
FOLLOW_NAME = "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_charge_current_closeout_gate_low_order_jeff0_refresh"
FOLLOW = "8.7.56.2555"


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


# 関数: gate で使う式を返す。

def formulas() -> dict[str, str]:
    """Return formulas used in the background-expansion closeout gate."""
    return {
        "gate_a": "Gate A = background-expansion closeout surface explicit and machine-readable",
        "gate_b": "Gate B = charge-current closeout promoted as the next primary lane",
        "gate_c": "Gate C = blind-vector direct computation primary admissible now",
        "ordered_closeout": "background expansion -> charge-current / Noether-current closure -> low-order J_eff^0 synthesis -> blind vector refresh",
    }


# 関数: `.2547-.2550` を実行する。

def main() -> None:
    """Execute the updated-pack background-expansion decision gate."""
    for path in (STATUS, ROADMAP, AI, RECENT, P34, P36, P39, P55, P14, PRIOR):
        sign_base.require(path)

    status_text = sign_base.read_text(STATUS)
    roadmap_text = sign_base.read_text(ROADMAP)
    p = sign_base.read_json(PRIOR)["summary"]

    gate_a = bool(
        p["updated_pack_background_expansion_closeout_target_surface_explicit"]
        and p["updated_pack_background_expansion_closeout_machine_readable_now"]
    )
    gate_b = bool(
        p["updated_pack_charge_current_primary_closeout_required"]
        and not p["updated_pack_exact_qball_background_expansion_available_now"]
    )
    gate_c = bool(not p["blind_vector_observable_gate_still_blocked"])
    exact_expansion = bool(p["updated_pack_exact_qball_background_expansion_available_now"])
    exact_source = False
    hybrid = bool(p["farther_hybrid_continuation_reopen_required_now"])
    old_retry = False

    rows = [
        sign_base.row(
            "gate_a_updated_pack_background_expansion_closeout_surface_explicit",
            "pass" if gate_a else "reject",
            "Gate A updated-pack background-expansion closeout surface explicit",
            sign_base.truth(gate_a),
            "The closeout audit already compressed the first missing primitive into one explicit dependency stack.",
        ),
        sign_base.row(
            "gate_b_updated_pack_charge_current_primary_selected",
            "pass" if gate_b else "reject",
            "Gate B updated-pack charge-current primary selected",
            sign_base.truth(gate_b),
            "Because the explicit background expansion is still absent, the next honest remaining closeout object is exact charge-current / Noether-current closure.",
        ),
        sign_base.row(
            "gate_c_blind_vector_computation_primary_admissible_now",
            "pass" if gate_c else "reject",
            "Gate C blind-vector computation primary admissible now",
            sign_base.truth(gate_c),
            "Blind-vector direct computation remains downstream until the background-expansion and charge-current closeout objects move first.",
        ),
        sign_base.row(
            "exact_qball_background_expansion_available_now",
            "pass" if exact_expansion else "reject",
            "exact Q-ball background expansion available now",
            sign_base.truth(exact_expansion),
            "The gate synchronizes that the primitive is explicit as a closeout target but still absent as a derived theorem object.",
        ),
        sign_base.row(
            "exact_source_theorem_derived_now",
            "pass" if exact_source else "reject",
            "exact source theorem derived now",
            sign_base.truth(exact_source),
            "Neither the background-expansion closeout audit nor this gate derives the exact source theorem by itself.",
        ),
        sign_base.row(
            "farther_hybrid_continuation_reopen_required_now",
            "pass" if hybrid else "reject",
            "farther hybrid continuation reopen required now",
            sign_base.truth(hybrid),
            "Extra q-range evidence remains reserve-only because the blocker is still theorem-side.",
        ),
        sign_base.row(
            "old_density_proxy_eigenvalue_retry_admissible_now",
            "pass" if old_retry else "reject",
            "old density/proxy/eigenvalue retry admissible now",
            sign_base.truth(old_retry),
            "The background-expansion closeout route still does not reopen exhausted pre-update retry families.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "retained_scalar_residual_rel": float(p["retained_scalar_residual_rel"]),
        "gate_a_updated_pack_background_expansion_closeout_surface_explicit": gate_a,
        "gate_b_updated_pack_charge_current_primary_selected": gate_b,
        "gate_c_blind_vector_computation_primary_admissible_now": gate_c,
        "exact_qball_background_expansion_available_now": exact_expansion,
        "exact_source_theorem_derived_now": exact_source,
        "farther_hybrid_continuation_reopen_required_now": hybrid,
        "old_density_proxy_eigenvalue_retry_admissible_now": old_retry,
        "hybrid_supporting_evidence_reopen_required": hybrid,
        "selected_primary_pack_update_surface": "updated_pack_charge_current_closeout_audit",
        "selected_secondary_pack_update_surface": "updated_pack_exact_low_order_jeff0_formula_synthesis",
        "selected_reserve_completion_lane": "blind_vector_after_charge_current_and_low_order_jeff0_closeout",
        "selected_next_generation_route": NEXT_NAME,
        "recommended_next_route_or_none": NEXT,
        "selected_followup_route": FOLLOW_NAME,
        "selected_followup_route_or_none": FOLLOW,
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.2549",
        NAME + " declaration gate",
        {
            "source_files": {
                "status": sign_base.display_path(STATUS),
                "roadmap": sign_base.display_path(ROADMAP),
                "ai_context": sign_base.display_path(AI),
                "work_history_recent": sign_base.display_path(RECENT),
                "prior_audit": sign_base.display_path(PRIOR),
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
            "overall_status": "vector_qball_form_factor_updated_pack_background_expansion_gate_declared",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_NAME],
        },
        {
            "formulas": formulas(),
            "hits": {
                "status_branch_hit": sign_base.hit(status_text, ".2543-.2546"),
                "roadmap_branch_hit": sign_base.hit(roadmap_text, ".2543-.2546"),
            },
        },
    )
    paths = write_artifact("declaration_gate", payload)
    route = {
        "generated_utc": sign_base.now_iso(),
        "phase": {"phase": 8, "step": "8.7.56.2550", "name": NAME + " route sync"},
        "inputs": paths,
        "rows": rows,
        "summary": summary,
        "decision": {
            "overall_status": "vector_qball_form_factor_updated_pack_background_expansion_gate_route_synced",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_NAME],
        },
        "evidence": {
            "formulas": formulas(),
            "disposition": {
                "background_expansion_closeout_surface_explicit": gate_a,
                "charge_current_primary_selected": gate_b,
                "blind_vector_still_downstream": not gate_c,
            },
        },
    }
    route_paths = write_artifact("route_sync", route)
    print("[ok] updated-pack background-expansion gate artifacts written")
    print(f"  declaration_gate: {paths['json']}")
    print(f"  route_sync: {route_paths['json']}")


if __name__ == "__main__":
    main()
