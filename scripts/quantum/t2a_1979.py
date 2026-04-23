#!/usr/bin/env python3
"""Generate 8.7.56.1979-.1982 box-edge local-jet closeout artifacts."""

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
    / "q_8_7_56_1975_1978_box_edge_local_jet_signed_rule_declaration_gate_metrics.json"
)

STEP_TAG = "8.7.56.1979-1982"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor box-free tail closeout / "
    "substantive pack-update registry"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "box_edge_local_jet_closeout_registry",
    prefix="q",
)

PRIOR_CLASS = (
    "vector_qball_form_factor_box_edge_local_jet_signed_rule_derived_closeout_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_box_edge_local_jet_signed_rule_retained_higher_q_"
    "generalization_next"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_boundary_local_jet_higher_q_"
    "extension_audit"
)
NEXT_ROUTE = "8.7.56.1983"
FOLLOWUP_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_boundary_local_jet_generalization_"
    "decision_gate_registry"
)
FOLLOWUP_ROUTE = "8.7.56.1987"


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
    """Return formulas used in the closeout sync."""
    return {
        "retained_exact_interval": "F_exact(q)=sigma_exact(q)|F_exact(q)| for 0<=q/m0<=4",
        "box_edge_local_jet_rule": "G_jet(q)=(-h0 q^2 + h2) cos(q R_box) + h1 q sin(q R_box)=0",
        "hybrid_signed_rule": "sigma_hybrid(q)=sigma_exact(q) for 0<=q<=4, and sigma_hybrid(q)=(-1)^{N_<4+N_jet(q)} for q>4",
        "next_open_surface": "higher-q stability / further extension of the box-edge local-jet sign rule",
    }


# 関数: `.1979-.1982` を実行する。

def main() -> None:
    """Execute the box-edge local-jet closeout and registry sync."""
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
    inventory_ready = bool(prior_summary["gate_a_new_signed_rule_selected"])

    box_edge_local_jet_signed_rule_retained = True
    higher_q_boundary_local_jet_extension_admissible = True
    genuinely_new_signed_rule_required_now = False
    substantive_pack_update_required_now = False
    physical_reject_required = False

    rows = [
        sign_base.row(
            "inventory_ready",
            "pass" if inventory_ready else "reject",
            "box-edge local-jet closeout inventory ready",
            sign_base.truth(inventory_ready),
            "The closeout sync starts only after the new signed-rule reactivation has selected the boundary-local-jet theorem.",
        ),
        sign_base.row(
            "box_edge_local_jet_signed_rule_retained",
            "pass",
            "box-edge local-jet signed rule retained",
            sign_base.truth(box_edge_local_jet_signed_rule_retained),
            "The threshold-selected tail family is now superseded by the boundary-local-jet signed rule inside the current retained pack.",
        ),
        sign_base.row(
            "same_level_threshold_scan_admissible",
            "reject",
            "same-level threshold scan admissible",
            sign_base.truth(False),
            "The old threshold-selected `r_match` scans are no longer honest once the local-jet rule is retained.",
        ),
        sign_base.row(
            "same_level_box_free_tail_retry_admissible",
            "reject",
            "same-level box-free tail retry admissible",
            sign_base.truth(False),
            "The threshold-dependent box-free family has been replaced and should not be reopened at the same level.",
        ),
        sign_base.row(
            "higher_q_boundary_local_jet_extension_admissible",
            "pass",
            "higher-q boundary local-jet extension admissible",
            sign_base.truth(higher_q_boundary_local_jet_extension_admissible),
            "The next honest question is further extension of the retained box-edge local-jet rule, not a return to threshold matching.",
        ),
        sign_base.row(
            "genuinely_new_signed_rule_required_now",
            "reject" if not genuinely_new_signed_rule_required_now else "pass",
            "genuinely new signed rule required now",
            sign_base.truth(genuinely_new_signed_rule_required_now),
            "The current pack already supplies one theorem-level new rule, so an additional signed-rule break is not yet required.",
        ),
        sign_base.row(
            "substantive_pack_update_required_now",
            "reject" if not substantive_pack_update_required_now else "pass",
            "substantive pack update required now",
            sign_base.truth(substantive_pack_update_required_now),
            "Pack update remains reserve because the current retained pack already closes the threshold-dependence obstruction.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "box_edge_local_jet_signed_rule_retained": box_edge_local_jet_signed_rule_retained,
        "higher_q_boundary_local_jet_extension_admissible": higher_q_boundary_local_jet_extension_admissible,
        "genuinely_new_signed_rule_required_now": genuinely_new_signed_rule_required_now,
        "substantive_pack_update_required_now": substantive_pack_update_required_now,
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": FOLLOWUP_ROUTE_NAME,
        "selected_followup_route_or_none": FOLLOWUP_ROUTE,
        "physical_reject_required": physical_reject_required,
    }

    declaration_payload = sign_base.payload(
        "8.7.56.1981",
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
            "overall_status": "vector_qball_form_factor_box_edge_local_jet_closeout_declared",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "formulas": build_formulae(),
            "hits": {
                "status_branch_hit": sign_base.hit(status_text, "8.7.56.1979"),
                "roadmap_branch_hit": sign_base.hit(roadmap_text, "8.7.56.1979-.1982"),
                "current_problem_hit": sign_base.hit(current_problem_text, "new signed observable rule"),
                "current_status_hit": sign_base.hit(current_status_text, "box_free_tail_completion"),
                "unified_roadmap_hit": sign_base.hit(unified_text, ".1979-.1982"),
                "long_roadmap_hit": sign_base.hit(long_text, ".1979-.1982"),
                "part5_hit": sign_base.hit(part5_text, "0<=q/m0<=4"),
            },
        },
    )

    route_payload = sign_base.payload(
        "8.7.56.1982",
        STEP_NAME + " route sync",
        declaration_payload["inputs"],
        [
            sign_base.row(
                "box_edge_local_jet_signed_rule_retained",
                "pass",
                "box-edge local-jet signed rule retained",
                sign_base.truth(box_edge_local_jet_signed_rule_retained),
                "The current pack has an explicit theorem-level replacement for the threshold-selected tail family.",
            ),
            sign_base.row(
                "higher_q_boundary_local_jet_extension_admissible",
                "pass",
                "higher-q boundary local-jet extension admissible",
                sign_base.truth(higher_q_boundary_local_jet_extension_admissible),
                "The next mainline is further extension, not a return to threshold scans or dormant refresh loops.",
            ),
            sign_base.row(
                "next_route_fixed",
                "pass",
                "next route fixed",
                1.0,
                "The next official branch is the boundary local-jet higher-q extension audit.",
            ),
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
            "overall_status": "vector_qball_form_factor_box_edge_local_jet_route_synced",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {"formulas": build_formulae()},
    )

    write_artifact("declaration_gate", declaration_payload)
    write_artifact("route_sync", route_payload)

    print("[ok] 8.7.56.1979-.1982 box-edge local-jet closeout artifacts generated")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
