#!/usr/bin/env python3
"""Generate 8.7.56.1987-.1990 boundary local-jet generalization gate artifacts."""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime
from datetime import timezone
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

PRIOR_GATE = (
    PUBLIC_OUT
    / "q_8_7_56_1983_1986_boundary_local_jet_higher_q_ext_audit_declaration_gate_metrics.json"
)

STEP_TAG = "8.7.56.1987-1990"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor boundary local-jet "
    "generalization decision gate / registry"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "boundary_local_jet_generalization_gate",
    prefix="q",
)

PRIOR_CLASS = (
    "vector_qball_form_factor_box_edge_local_jet_extension_to_40_retained_"
    "asymptotic_phase_drift_generalization_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_box_edge_local_jet_extension_to_40_retained_"
    "asymptotic_phase_drift_audit_next"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_boundary_local_jet_"
    "asymptotic_phase_drift_audit"
)
NEXT_ROUTE = "8.7.56.1991"
FOLLOWUP_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_boundary_local_jet_"
    "asymptotic_phase_drift_decision_gate_registry"
)
FOLLOWUP_ROUTE = "8.7.56.1995"


# 関数: 現在UTC時刻を返す。
def now_iso() -> str:
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


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


# 関数: closeout 用の公式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used in the generalization decision gate."""
    return {
        "retained_rule": "G_jet(q)=(-h0 q^2 + h2) cos(q R_box) + h1 q sin(q R_box)=0",
        "finite_extension_read": "retain the boundary local-jet rule on 12<=q/m0<=40 when signed reconstruction remains stable",
        "next_open_surface": "asymptotic phase-drift correction beyond the retained extension window",
    }


# 関数: `.1987-.1990` を実行する。

def main() -> None:
    """Execute the boundary local-jet generalization decision gate."""
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
    ):
        sign_base.require(path)

    status_text = sign_base.read_text(STATUS)
    roadmap_text = sign_base.read_text(ROADMAP)
    current_problem_text = sign_base.read_text(CURRENT_PROBLEM)
    current_status_text = sign_base.read_text(CURRENT_STATUS)
    unified_text = sign_base.read_text(UNIFIED_ROADMAP)
    long_text = sign_base.read_text(LONG_ROADMAP)
    part5_text = sign_base.read_text(PART5)

    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    inventory_ready = bool(prior_summary["higher_q_extension_supported"])

    gate_a_extension_retained = bool(prior_summary["higher_q_extension_supported"])
    gate_b_asymptotic_phase_drift_selected = bool(
        prior_summary["asymptotic_generalization_beyond_40_not_yet_supported"]
    )
    gate_c_current_rule_blocked = False
    same_level_box_edge_retry_admissible = False
    asymptotic_phase_drift_audit_admissible_now = True
    physical_reject_required = False

    rows = [
        sign_base.row("inventory_ready", "pass" if inventory_ready else "reject", "generalization decision inventory ready", sign_base.truth(inventory_ready), "The decision gate starts only after the higher-q extension audit has separated the retained finite extension from the later asymptotic drift."),
        sign_base.row("gate_a_extension_retained", "pass" if gate_a_extension_retained else "reject", "Gate A finite higher-q extension retained", sign_base.truth(gate_a_extension_retained), "The retained boundary local-jet theorem survives on 12<=q/m0<=40 and should stay on the mainline."),
        sign_base.row("gate_b_asymptotic_phase_drift_selected", "watch" if gate_b_asymptotic_phase_drift_selected else "pass", "Gate B asymptotic phase drift selected", sign_base.truth(gate_b_asymptotic_phase_drift_selected), "The next open theorem question is no longer threshold dependence but phase drift at later asymptotic q."),
        sign_base.row("gate_c_current_rule_blocked", "reject" if not gate_c_current_rule_blocked else "pass", "Gate C current rule blocked", sign_base.truth(gate_c_current_rule_blocked), "The current rule is not globally rejected because it still retains a stable finite higher-q extension window."),
        sign_base.row("same_level_box_edge_retry_admissible", "reject", "same-level box-edge retry admissible", sign_base.truth(same_level_box_edge_retry_admissible), "The next route is theorem-level asymptotic phase-drift analysis, not a same-level rerun of the already retained boundary local-jet extension."),
        sign_base.row("asymptotic_phase_drift_audit_admissible_now", "pass", "asymptotic phase-drift audit admissible now", sign_base.truth(asymptotic_phase_drift_audit_admissible_now), "The monitor and stress windows now justify a dedicated phase-drift theorem branch."),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "gate_a_extension_retained": gate_a_extension_retained,
        "gate_b_asymptotic_phase_drift_selected": gate_b_asymptotic_phase_drift_selected,
        "gate_c_current_rule_blocked": gate_c_current_rule_blocked,
        "same_level_box_edge_retry_admissible": same_level_box_edge_retry_admissible,
        "asymptotic_phase_drift_audit_admissible_now": asymptotic_phase_drift_audit_admissible_now,
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": FOLLOWUP_ROUTE_NAME,
        "selected_followup_route_or_none": FOLLOWUP_ROUTE,
        "physical_reject_required": physical_reject_required,
    }

    declaration_payload = sign_base.payload(
        "8.7.56.1989",
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
            },
            "constants": {
                "next_route_name": NEXT_ROUTE_NAME,
                "next_route": NEXT_ROUTE,
                "followup_route_name": FOLLOWUP_ROUTE_NAME,
                "followup_route": FOLLOWUP_ROUTE,
            },
        },
        rows,
        summary,
        {
            "overall_status": "vector_qball_form_factor_boundary_local_jet_generalization_gate_declared",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "formulas": build_formulae(),
            "hits": {
                "status_branch_hit": sign_base.hit(status_text, "8.7.56.1987"),
                "roadmap_branch_hit": sign_base.hit(roadmap_text, "8.7.56.1987-.1990"),
                "current_problem_hit": sign_base.hit(current_problem_text, "higher_q_boundary_local_jet_extension_admissible"),
                "current_status_hit": sign_base.hit(current_status_text, "boundary local-jet higher-q extension audit"),
                "unified_roadmap_hit": sign_base.hit(unified_text, ".1987-.1990"),
                "long_roadmap_hit": sign_base.hit(long_text, "boundary local-jet higher-q extension audit"),
                "part5_hit": sign_base.hit(part5_text, ".1975-.1982"),
            },
        },
    )

    route_payload = sign_base.payload(
        "8.7.56.1990",
        STEP_NAME + " route sync",
        declaration_payload["inputs"],
        [
            sign_base.row("gate_a_extension_retained", "pass" if gate_a_extension_retained else "reject", "Gate A finite higher-q extension retained", sign_base.truth(gate_a_extension_retained), "The finite higher-q extension to 12<=q/m0<=40 remains on the official mainline."),
            sign_base.row("gate_b_asymptotic_phase_drift_selected", "watch" if gate_b_asymptotic_phase_drift_selected else "pass", "Gate B asymptotic phase drift selected", sign_base.truth(gate_b_asymptotic_phase_drift_selected), "The next route asks for a theorem that corrects the later phase drift without discarding the retained finite extension window."),
            sign_base.row("next_route_fixed", "pass", "next route fixed", 1.0, "The next official branch is the boundary local-jet asymptotic phase-drift audit."),
        ],
        {
            "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
            "selected_next_generation_route": NEXT_ROUTE_NAME,
            "recommended_next_route_or_none": NEXT_ROUTE,
            "selected_followup_route": FOLLOWUP_ROUTE_NAME,
            "selected_followup_route_or_none": FOLLOWUP_ROUTE,
            "physical_reject_required": physical_reject_required,
        },
        {
            "overall_status": "vector_qball_form_factor_boundary_local_jet_generalization_route_synced",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {"formulas": build_formulae()},
    )

    write_artifact("declaration_gate", declaration_payload)
    write_artifact("route_sync", route_payload)

    print("[ok] 8.7.56.1987-.1990 boundary local-jet generalization gate artifacts generated")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
