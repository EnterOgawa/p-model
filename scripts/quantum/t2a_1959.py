#!/usr/bin/env python3
"""Generate 8.7.56.1959-.1962 extended-interval decision-gate artifacts.

`.1955-.1958` widened the exact sign-parity audit from `0 <= q/m0 <= 2` to
`0 <= q/m0 <= 4`. This branch freezes the resulting disposition. The key
question is whether the current theorem remained exact on the wider interval
(`Gate A`), encountered an honest beyond-interval obstruction (`Gate B`), or
blocked the current rule already on the audited interval (`Gate C`).
"""

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

import scripts.quantum.t2a_1847 as closeout_base
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

FURTHER_GATE = (
    PUBLIC_OUT
    / "q_8_7_56_1955_1958_further_ext_interval_sign_phase_audit_declaration_gate_metrics.json"
)
PRIOR_CLOSEOUT_GATE = (
    PUBLIC_OUT
    / "q_8_7_56_1951_1954_ext_interval_closeout_registry_declaration_gate_metrics.json"
)

STEP_TAG = "8.7.56.1959-1962"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor extended-interval decision "
    "gate / theorem stability sync"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "ext_interval_decision_gate",
    prefix="q",
)

PRIOR_CLASS = (
    "vector_qball_form_factor_retained_interval_extension_real_branch_sign_parity_0_to_4_"
    "derived_decision_gate_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_global_exact_alpha_signed_form_factor_extended_interval_0_to_4_"
    "promotion_retained_asymptotic_generalization_next"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_asymptotic_sign_parity_generalization_audit"
)
NEXT_ROUTE = "8.7.56.1963"
FOLLOWUP_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_post_generalization_closeout_registry"
)
FOLLOWUP_ROUTE = "8.7.56.1967"
EXTENDED_Q_MAX = 4.0


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
        "json": closeout_base.display_path(paths["json"]),
        "csv": closeout_base.display_path(paths["csv"]),
    }


# 関数: decision-gate 用の公式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas for the extended-interval decision gate."""
    return {
        "gate_a": "exact extension retained on the audited interval",
        "gate_b": "finite-interval exact but beyond-interval obstruction detected",
        "gate_c": "current sign-parity rule already blocked on the audited interval",
        "current_retained_interval": "0 <= q/m0 <= 4",
    }


# 関数: `.1959-.1962` を実行する。

def main() -> None:
    """Execute the extended-interval decision gate / theorem stability sync."""
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
        FURTHER_GATE,
        PRIOR_CLOSEOUT_GATE,
    ):
        closeout_base.require(path)

    status_text = closeout_base.read_text(STATUS)
    roadmap_text = closeout_base.read_text(ROADMAP)
    current_problem_text = closeout_base.read_text(CURRENT_PROBLEM)
    current_status_text = closeout_base.read_text(CURRENT_STATUS)
    unified_text = closeout_base.read_text(UNIFIED_ROADMAP)
    long_text = closeout_base.read_text(LONG_ROADMAP)
    part5_text = closeout_base.read_text(PART5)

    further_summary = closeout_base.read_json(FURTHER_GATE)["summary"]
    prior_closeout_summary = closeout_base.read_json(PRIOR_CLOSEOUT_GATE)["summary"]

    inventory_ready = all(
        (
            bool(prior_closeout_summary["exact_alpha_promotion_retained"]),
            bool(prior_closeout_summary["exact_signed_form_factor_promotion_retained"]),
            bool(further_summary["further_interval_extension_surface_present"]),
        )
    )
    gate_a_exact_extension_selected = bool(further_summary["gate_a_exact_extension_selected"])
    gate_b_finite_interval_obstruction_selected = bool(
        further_summary["beyond_interval_obstruction_detected"]
    )
    gate_c_current_rule_blocked_selected = bool(further_summary["current_rule_blocked"])
    exact_alpha_promotion_retained = bool(prior_closeout_summary["exact_alpha_promotion_retained"])
    exact_signed_form_factor_promotion_retained = gate_a_exact_extension_selected
    asymptotic_generalization_admissible = gate_a_exact_extension_selected
    same_level_old_retry_admissible = False
    physical_reject_required = False

    formulas = build_formulae()

    rows = [
        closeout_base.row(
            "inventory_ready",
            "pass" if inventory_ready else "reject",
            "extended-interval decision inventory ready",
            closeout_base.truth(inventory_ready),
            "The decision gate starts only after the 0<=q/m0<=4 theorem audit has been computed.",
        ),
        closeout_base.row(
            "gate_a_exact_extension_selected",
            "pass" if gate_a_exact_extension_selected else "reject",
            "Gate A exact extension retained",
            closeout_base.truth(gate_a_exact_extension_selected),
            "The current theorem survives on the wider interval without changing the observable rule.",
        ),
        closeout_base.row(
            "gate_b_finite_interval_obstruction_selected",
            "reject" if not gate_b_finite_interval_obstruction_selected else "pass",
            "Gate B finite-interval exact / beyond-interval obstruction",
            closeout_base.truth(gate_b_finite_interval_obstruction_selected),
            "This gate stays closed because no obstruction was detected on the present wider interval.",
        ),
        closeout_base.row(
            "gate_c_current_rule_blocked_selected",
            "reject" if not gate_c_current_rule_blocked_selected else "pass",
            "Gate C current rule blocked",
            closeout_base.truth(gate_c_current_rule_blocked_selected),
            "The present sign-parity rule remains valid on the audited interval.",
        ),
        closeout_base.row(
            "exact_alpha_promotion_retained",
            "pass" if exact_alpha_promotion_retained else "reject",
            "exact alpha promotion retained",
            closeout_base.truth(exact_alpha_promotion_retained),
            "The amplitude-side theorem remains untouched by the wider sign audit.",
        ),
        closeout_base.row(
            "exact_signed_form_factor_promotion_retained",
            "pass" if exact_signed_form_factor_promotion_retained else "reject",
            "exact signed form-factor promotion retained on 0<=q/m0<=4",
            closeout_base.truth(exact_signed_form_factor_promotion_retained),
            "The signed promotion now remains exact on the widened retained interval.",
        ),
        closeout_base.row(
            "extended_interval_over_m0",
            "watch",
            "current retained interval upper edge q_max/m0",
            EXTENDED_Q_MAX,
            "This decision gate freezes 0<=q/m0<=4 as the current retained interval.",
        ),
        closeout_base.row(
            "asymptotic_generalization_admissible",
            "pass" if asymptotic_generalization_admissible else "watch",
            "asymptotic sign-parity generalization admissible",
            closeout_base.truth(asymptotic_generalization_admissible),
            "Because Gate A passed, the next honest step is to generalize the theorem rather than reopen a new signed observable rule immediately.",
        ),
        closeout_base.row(
            "same_level_old_retry_admissible",
            "reject",
            "same-level old retry admissible",
            closeout_base.truth(same_level_old_retry_admissible),
            "Old dormant, density, proxy, and eigenvalue retries remain closed.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "gate_a_exact_extension_selected": gate_a_exact_extension_selected,
        "gate_b_finite_interval_obstruction_selected": gate_b_finite_interval_obstruction_selected,
        "gate_c_current_rule_blocked_selected": gate_c_current_rule_blocked_selected,
        "exact_alpha_promotion_retained": exact_alpha_promotion_retained,
        "exact_signed_form_factor_promotion_retained": exact_signed_form_factor_promotion_retained,
        "extended_interval_over_m0": EXTENDED_Q_MAX,
        "asymptotic_generalization_admissible": asymptotic_generalization_admissible,
        "same_level_old_retry_admissible": same_level_old_retry_admissible,
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": FOLLOWUP_ROUTE_NAME,
        "selected_followup_route_or_none": FOLLOWUP_ROUTE,
        "physical_reject_required": physical_reject_required,
    }

    decision = {
        "overall_status": "vector_qball_form_factor_extended_interval_decision_gate_declared",
        "branch_completed": True,
        "next_required_artifacts": [NEXT_ROUTE_NAME],
    }

    evidence = {
        "formulas": formulas,
        "hits": {
            "status_branch_hit": closeout_base.hit(status_text, "8.7.56.1955"),
            "roadmap_branch_hit": closeout_base.hit(roadmap_text, "8.7.56.1955-.1958"),
            "current_problem_hit": closeout_base.hit(current_problem_text, "extended_interval_over_m0 = 2.0"),
            "current_status_hit": closeout_base.hit(current_status_text, "extended_interval_over_m0 = 2.0"),
            "unified_roadmap_hit": closeout_base.hit(unified_text, "116. `.1955-.1958`"),
            "long_roadmap_hit": closeout_base.hit(long_text, "8.7.56.1955-.1958"),
            "part5_hit": closeout_base.hit(part5_text, "0<=q/m0<=2"),
        },
    }

    declaration_payload = closeout_base.payload(
        "8.7.56.1961",
        STEP_NAME + " declaration gate",
        {
            "source_files": {
                "status": closeout_base.display_path(STATUS),
                "roadmap": closeout_base.display_path(ROADMAP),
                "ai_context": closeout_base.display_path(AI_CONTEXT),
                "work_history_recent": closeout_base.display_path(WORK_HISTORY_RECENT),
                "current_problem": closeout_base.display_path(CURRENT_PROBLEM),
                "current_status": closeout_base.display_path(CURRENT_STATUS),
                "unified_roadmap": closeout_base.display_path(UNIFIED_ROADMAP),
                "long_roadmap": closeout_base.display_path(LONG_ROADMAP),
                "part5": closeout_base.display_path(PART5),
                "further_gate": closeout_base.display_path(FURTHER_GATE),
                "prior_closeout_gate": closeout_base.display_path(PRIOR_CLOSEOUT_GATE),
            },
            "constants": {
                "extended_interval_over_m0": EXTENDED_Q_MAX,
                "next_route_name": NEXT_ROUTE_NAME,
                "next_route": NEXT_ROUTE,
                "followup_route_name": FOLLOWUP_ROUTE_NAME,
                "followup_route": FOLLOWUP_ROUTE,
            },
        },
        rows,
        summary,
        decision,
        evidence,
    )

    route_payload = closeout_base.payload(
        "8.7.56.1962",
        STEP_NAME + " route sync",
        declaration_payload["inputs"],
        [
            closeout_base.row(
                "gate_a_exact_extension_selected",
                "pass" if gate_a_exact_extension_selected else "reject",
                "Gate A exact extension retained",
                closeout_base.truth(gate_a_exact_extension_selected),
                "The mainline stays on theorem extension because the current rule still works on the widened interval.",
            ),
            closeout_base.row(
                "asymptotic_generalization_admissible",
                "pass" if asymptotic_generalization_admissible else "watch",
                "asymptotic generalization admissible",
                closeout_base.truth(asymptotic_generalization_admissible),
                "The next honest move is to seek a broader generalization theorem rather than a replacement signed rule.",
            ),
            closeout_base.row(
                "next_route_fixed",
                "pass",
                "next route fixed",
                1.0,
                "The next official branch is the asymptotic sign-parity generalization audit.",
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
            "overall_status": "vector_qball_form_factor_extended_interval_route_synced",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {"formulas": formulas},
    )

    write_artifact("declaration_gate", declaration_payload)
    write_artifact("route_sync", route_payload)

    print("[ok] 8.7.56.1959-.1962 extended-interval decision gate artifacts generated")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
