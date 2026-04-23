#!/usr/bin/env python3
"""Generate 8.7.56.1967-.1970 generalization decision-gate artifacts.

The asymptotic audit showed that the current sign-parity theorem remains exact
on `0 <= q/m0 <= 4` but that its high-q continuation is controlled by the
finite solver-box boundary term rather than a box-free canonical theorem.
This branch freezes that Gate-B disposition and resets the next route toward
box-free tail completion / cutoff removal.
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

ASYMPTOTIC_GATE = (
    PUBLIC_OUT / "q_8_7_56_1963_1966_asymp_sign_parity_audit_declaration_gate_metrics.json"
)
PRIOR_GATE = (
    PUBLIC_OUT / "q_8_7_56_1959_1962_ext_interval_decision_gate_declaration_gate_metrics.json"
)

STEP_TAG = "8.7.56.1967-1970"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor generalization decision "
    "gate / closeout registry"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "asymp_generalization_gate",
    prefix="q",
)

PRIOR_CLASS = (
    "vector_qball_form_factor_asymptotic_sign_parity_box_boundary_obstruction_"
    "decision_gate_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_extended_interval_exact_box_boundary_asymptotic_"
    "obstruction_tail_completion_or_pack_update_next"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_conditional_box_free_tail_completion_"
    "or_substantive_pack_update_reactivation"
)
NEXT_ROUTE = "8.7.56.1971"
FOLLOWUP_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_conditional_new_signed_observable_rule_"
    "reactivation_after_box_free_tail_audit"
)
FOLLOWUP_ROUTE = "8.7.56.1975"


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
    """Return formulas for the asymptotic decision gate."""
    return {
        "gate_a": "box-free asymptotic continuation retained",
        "gate_b": "finite-interval exact but asymptotic obstruction detected",
        "gate_c": "current continuation rule blocked on the audited finite interval",
        "obstruction_signature": "high-q zeros lock to pi/R_box and the leading boundary term -(h(R_box)/(N q^2)) cos(q R_box)",
    }


# 関数: `.1967-.1970` を実行する。

def main() -> None:
    """Execute the asymptotic generalization decision gate / closeout registry."""
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
        ASYMPTOTIC_GATE,
        PRIOR_GATE,
    ):
        closeout_base.require(path)

    status_text = closeout_base.read_text(STATUS)
    roadmap_text = closeout_base.read_text(ROADMAP)
    current_problem_text = closeout_base.read_text(CURRENT_PROBLEM)
    current_status_text = closeout_base.read_text(CURRENT_STATUS)
    unified_text = closeout_base.read_text(UNIFIED_ROADMAP)
    long_text = closeout_base.read_text(LONG_ROADMAP)
    part5_text = closeout_base.read_text(PART5)

    asymptotic_summary = closeout_base.read_json(ASYMPTOTIC_GATE)["summary"]
    prior_summary = closeout_base.read_json(PRIOR_GATE)["summary"]

    inventory_ready = all(
        (
            bool(prior_summary["exact_alpha_promotion_retained"]),
            bool(prior_summary["exact_signed_form_factor_promotion_retained"]),
            bool(asymptotic_summary["box_boundary_asymptotic_supported"]),
        )
    )

    gate_a_asymptotic_continuation_retained = bool(
        asymptotic_summary["asymptotic_continuation_retained"]
    )
    gate_b_finite_interval_exact_asymptotic_obstruction_selected = bool(
        asymptotic_summary["finite_interval_exact_but_asymptotic_obstruction_detected"]
    )
    gate_c_current_continuation_rule_blocked = bool(
        asymptotic_summary["current_continuation_rule_blocked"]
    )

    exact_alpha_promotion_retained = bool(prior_summary["exact_alpha_promotion_retained"])
    exact_signed_form_factor_promotion_retained = bool(
        prior_summary["exact_signed_form_factor_promotion_retained"]
    )
    box_free_tail_completion_admissible_now = gate_b_finite_interval_exact_asymptotic_obstruction_selected
    same_level_old_retry_admissible = False
    physical_reject_required = False

    formulas = build_formulae()

    rows = [
        closeout_base.row(
            "inventory_ready",
            "pass" if inventory_ready else "reject",
            "generalization decision inventory ready",
            closeout_base.truth(inventory_ready),
            "The closeout starts only after the asymptotic audit has isolated the solver-box signature.",
        ),
        closeout_base.row(
            "gate_a_asymptotic_continuation_retained",
            "reject" if not gate_a_asymptotic_continuation_retained else "pass",
            "Gate A asymptotic continuation retained",
            closeout_base.truth(gate_a_asymptotic_continuation_retained),
            "Current evidence does not retain a box-free asymptotic theorem under the present solver-box pack.",
        ),
        closeout_base.row(
            "gate_b_finite_interval_exact_asymptotic_obstruction_selected",
            "pass" if gate_b_finite_interval_exact_asymptotic_obstruction_selected else "reject",
            "Gate B finite-interval exact but asymptotic obstruction selected",
            closeout_base.truth(gate_b_finite_interval_exact_asymptotic_obstruction_selected),
            "The exact theorem survives on 0<=q/m0<=4, but large-q continuation is controlled by the solver-box edge rather than a canonical box-free theorem.",
        ),
        closeout_base.row(
            "gate_c_current_continuation_rule_blocked",
            "reject" if not gate_c_current_continuation_rule_blocked else "pass",
            "Gate C current continuation rule blocked",
            closeout_base.truth(gate_c_current_continuation_rule_blocked),
            "The finite-interval theorem itself is not blocked; the obstruction only appears in the asymptotic generalization.",
        ),
        closeout_base.row(
            "exact_alpha_promotion_retained",
            "pass" if exact_alpha_promotion_retained else "reject",
            "exact alpha promotion retained",
            closeout_base.truth(exact_alpha_promotion_retained),
            "The amplitude-side exact promotion remains intact while the asymptotic issue is localized separately.",
        ),
        closeout_base.row(
            "exact_signed_form_factor_promotion_retained",
            "pass" if exact_signed_form_factor_promotion_retained else "reject",
            "exact signed form-factor promotion retained on 0<=q/m0<=4",
            closeout_base.truth(exact_signed_form_factor_promotion_retained),
            "The signed theorem remains exact on the retained finite interval even though the asymptotic continuation is obstructed.",
        ),
        closeout_base.row(
            "box_free_tail_completion_admissible_now",
            "pass" if box_free_tail_completion_admissible_now else "watch",
            "box-free tail completion admissible now",
            closeout_base.truth(box_free_tail_completion_admissible_now),
            "Because Gate B is selected, the next honest theorem work is cutoff removal / box-free tail completion rather than another same-level retry.",
        ),
        closeout_base.row(
            "same_level_old_retry_admissible",
            "reject",
            "same-level old retry admissible",
            closeout_base.truth(same_level_old_retry_admissible),
            "Old dormant loops, old surrogate retries, and same-level sign rewrites remain blocked.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "gate_a_asymptotic_continuation_retained": gate_a_asymptotic_continuation_retained,
        "gate_b_finite_interval_exact_asymptotic_obstruction_selected": gate_b_finite_interval_exact_asymptotic_obstruction_selected,
        "gate_c_current_continuation_rule_blocked": gate_c_current_continuation_rule_blocked,
        "exact_alpha_promotion_retained": exact_alpha_promotion_retained,
        "exact_signed_form_factor_promotion_retained": exact_signed_form_factor_promotion_retained,
        "box_free_tail_completion_admissible_now": box_free_tail_completion_admissible_now,
        "same_level_old_retry_admissible": same_level_old_retry_admissible,
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": FOLLOWUP_ROUTE_NAME,
        "selected_followup_route_or_none": FOLLOWUP_ROUTE,
        "physical_reject_required": physical_reject_required,
    }

    decision = {
        "overall_status": "vector_qball_form_factor_asymptotic_generalization_closeout_declared",
        "branch_completed": True,
        "next_required_artifacts": [NEXT_ROUTE_NAME],
    }

    evidence = {
        "formulas": formulas,
        "selected_obstruction_signature": {
            "solver_box_edge_over_m0": asymptotic_summary["solver_box_edge_over_m0"],
            "asymptotic_spacing_theory": asymptotic_summary["asymptotic_spacing_theory"],
            "mean_high_q_spacing": asymptotic_summary["mean_high_q_spacing"],
            "spacing_rel_gap_vs_theory": asymptotic_summary["spacing_rel_gap_vs_theory"],
            "leading_fit_max_rel_error": asymptotic_summary["leading_fit_max_rel_error"],
        },
        "hits": {
            "status_branch_hit": closeout_base.hit(status_text, "8.7.56.1967"),
            "roadmap_branch_hit": closeout_base.hit(roadmap_text, "8.7.56.1967-.1970"),
            "current_problem_hit": closeout_base.hit(current_problem_text, "0<=q/m0<=4"),
            "current_status_hit": closeout_base.hit(current_status_text, "asymptotic_generalization_admissible"),
            "unified_roadmap_hit": closeout_base.hit(unified_text, "119. `.1967-.1970`"),
            "long_roadmap_hit": closeout_base.hit(long_text, "1967-.1970"),
            "part5_hit": closeout_base.hit(part5_text, "0<=q/m0<=4"),
        },
    }

    declaration_payload = closeout_base.payload(
        "8.7.56.1969",
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
                "asymptotic_gate": closeout_base.display_path(ASYMPTOTIC_GATE),
                "prior_gate": closeout_base.display_path(PRIOR_GATE),
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
        decision,
        evidence,
    )

    route_payload = closeout_base.payload(
        "8.7.56.1970",
        STEP_NAME + " route sync",
        declaration_payload["inputs"],
        [
            closeout_base.row(
                "gate_b_finite_interval_exact_asymptotic_obstruction_selected",
                "pass" if gate_b_finite_interval_exact_asymptotic_obstruction_selected else "reject",
                "Gate B finite-interval exact but asymptotic obstruction selected",
                closeout_base.truth(gate_b_finite_interval_exact_asymptotic_obstruction_selected),
                "The official read is that the theorem remains exact on 0<=q/m0<=4 but fails to become a box-free asymptotic theorem under the current pack.",
            ),
            closeout_base.row(
                "box_free_tail_completion_admissible_now",
                "pass" if box_free_tail_completion_admissible_now else "watch",
                "box-free tail completion admissible now",
                closeout_base.truth(box_free_tail_completion_admissible_now),
                "The next honest route is box-free tail completion / cutoff removal or a substantive pack update that supplies it.",
            ),
            closeout_base.row(
                "next_route_fixed",
                "pass",
                "next route fixed",
                1.0,
                "The next official branch is the conditional box-free tail completion or substantive pack-update reactivation.",
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
            "overall_status": "vector_qball_form_factor_asymptotic_generalization_route_synced",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {"formulas": formulas},
    )

    write_artifact("declaration_gate", declaration_payload)
    write_artifact("route_sync", route_payload)

    print("[ok] 8.7.56.1967-.1970 generalization decision gate artifacts generated")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
