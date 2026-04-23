#!/usr/bin/env python3
"""Generate 8.7.56.1951-.1954 extended-interval closeout artifacts.

`.1947-.1950` replaced the twelfth dormant loop with a retained-interval
extension computation. The exact amplitude theorem was already global in alpha,
and the real-branch sign-parity theorem is now numerically retained on
`0 <= q/m0 <= 2`. This branch freezes that route reset as the new official
closeout state and blocks further same-level dormant retries.
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

EXTENSION_GATE = (
    PUBLIC_OUT
    / "q_8_7_56_1947_1950_ext_interval_sign_phase_reactivation_declaration_gate_metrics.json"
)
PRIOR_CLOSEOUT_GATE = (
    PUBLIC_OUT
    / "q_8_7_56_1847_1850_signed_source_phase_closeout_wait_restore_declaration_gate_metrics.json"
)

STEP_TAG = "8.7.56.1951-1954"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor extended-interval closeout "
    "/ route reset"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "ext_interval_closeout_registry",
    prefix="q",
)

PRIOR_CLASS = (
    "vector_qball_form_factor_retained_interval_extension_real_branch_sign_parity_0_to_2_"
    "derived_closeout_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_global_exact_alpha_signed_form_factor_extended_interval_0_to_2_"
    "promotion_closeout_registry_completed"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_conditional_further_retained_interval_"
    "extension_or_new_signed_observable_rule_reactivation"
)
NEXT_ROUTE = "8.7.56.1955"
FOLLOWUP_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_post_extension_wait_restore_registry_refresh"
)
FOLLOWUP_ROUTE = "8.7.56.1959"
EXTENDED_Q_MAX = 2.0


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


# 関数: 拡張 closeout 用の公式を返す。

def build_formulae() -> dict[str, str]:
    """Return the retained extended-interval formulas."""
    return {
        "retained_exact_alpha_rule": "alpha_exact(q) = |F_exact(q)|^2 / (4 pi)",
        "retained_signed_rule": "F_exact(q) = sigma_F(q) |F_exact(q)|",
        "retained_interval": "0 <= q/m0 <= 2",
        "primary_reopen_surface": "further retained-interval extension beyond q/m0<=2 under the same real-branch parity theorem",
        "secondary_reopen_surface": "new signed observable rule beyond the extended-interval pack",
        "reserve_reopen_surface": "substantive pack update or genuinely new external input guiding a post-extension surface",
    }


# 関数: `.1951-.1954` を実行する。

def main() -> None:
    """Execute the extended-interval closeout / route-reset branch."""
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
        EXTENSION_GATE,
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

    extension_summary = closeout_base.read_json(EXTENSION_GATE)["summary"]
    prior_closeout_summary = closeout_base.read_json(PRIOR_CLOSEOUT_GATE)["summary"]

    inventory_ready = all(
        (
            bool(prior_closeout_summary["exact_alpha_promotion_retained"]),
            bool(prior_closeout_summary["global_signed_form_factor_promotion_retained"]),
            bool(extension_summary["extended_exact_signed_form_factor_promotion_selected"]),
        )
    )
    retry_gate_route_reset_selected = bool(extension_summary["retry_gate_triggered"])
    extended_interval_promotion_retained = bool(
        extension_summary["extended_exact_signed_form_factor_promotion_selected"]
    )
    exact_alpha_promotion_retained = bool(prior_closeout_summary["exact_alpha_promotion_retained"])
    exact_signed_form_factor_promotion_retained = extended_interval_promotion_retained
    same_level_post_dormant_retry_admissible = False
    physical_reject_required = False

    formulas = build_formulae()

    rows = [
        closeout_base.row(
            "inventory_ready",
            "pass" if inventory_ready else "reject",
            "extended-interval closeout inventory ready",
            closeout_base.truth(inventory_ready),
            "The branch starts only after the doubled-interval theorem has been derived on top of the already-retained exact alpha promotion.",
        ),
        closeout_base.row(
            "retry_gate_route_reset_selected",
            "pass" if retry_gate_route_reset_selected else "reject",
            "retry-gate route reset selected",
            closeout_base.truth(retry_gate_route_reset_selected),
            "The repeated dormant loop was replaced by computation because the retry gate forced an honest route reset.",
        ),
        closeout_base.row(
            "exact_alpha_promotion_retained",
            "pass" if exact_alpha_promotion_retained else "reject",
            "exact alpha promotion retained",
            closeout_base.truth(exact_alpha_promotion_retained),
            "The global amplitude theorem remains the canonical alpha-side read.",
        ),
        closeout_base.row(
            "exact_signed_form_factor_promotion_retained",
            "pass" if exact_signed_form_factor_promotion_retained else "reject",
            "extended exact signed form-factor promotion retained",
            closeout_base.truth(exact_signed_form_factor_promotion_retained),
            "The sign-side theorem now covers the extended audit interval 0<=q/m0<=2 rather than only 0<=q/m0<=1.",
        ),
        closeout_base.row(
            "extended_interval_over_m0",
            "watch",
            "retained extended interval upper edge q_max/m0",
            EXTENDED_Q_MAX,
            "This branch freezes the doubled interval as the new retained audit range.",
        ),
        closeout_base.row(
            "same_level_post_dormant_retry_admissible",
            "reject",
            "same-level post-dormant retry admissible",
            closeout_base.truth(same_level_post_dormant_retry_admissible),
            "The dormant family was superseded by the interval-extension theorem, so same-level dormant retry is no longer honest.",
        ),
        closeout_base.row(
            "physical_reject_required",
            "reject",
            "physical reject required",
            closeout_base.truth(physical_reject_required),
            "The family closes positively on the extended retained interval; no physical reject flag is needed.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "retry_gate_route_reset_selected": retry_gate_route_reset_selected,
        "exact_alpha_promotion_retained": exact_alpha_promotion_retained,
        "exact_signed_form_factor_promotion_retained": exact_signed_form_factor_promotion_retained,
        "extended_interval_over_m0": EXTENDED_Q_MAX,
        "extended_interval_promotion_retained": extended_interval_promotion_retained,
        "same_level_post_dormant_retry_admissible": same_level_post_dormant_retry_admissible,
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": FOLLOWUP_ROUTE_NAME,
        "selected_followup_route_or_none": FOLLOWUP_ROUTE,
        "physical_reject_required": physical_reject_required,
    }

    decision = {
        "overall_status": "vector_qball_form_factor_extended_interval_closeout_declared",
        "branch_completed": True,
        "next_required_artifacts": [NEXT_ROUTE_NAME],
    }

    evidence = {
        "formulas": formulas,
        "hits": {
            "status_branch_hit": closeout_base.hit(status_text, "8.7.56.1951"),
            "roadmap_branch_hit": closeout_base.hit(roadmap_text, "8.7.56.1951-.1954"),
            "current_problem_hit": closeout_base.hit(current_problem_text, "retained-interval extension"),
            "current_status_hit": closeout_base.hit(current_status_text, "eleventh post-dormant"),
            "unified_roadmap_hit": closeout_base.hit(unified_text, "115. `.1951-.1954`"),
            "long_roadmap_hit": closeout_base.hit(long_text, "8.7.56.1947"),
            "part5_hit": closeout_base.hit(part5_text, "signed source-phase theorem"),
        },
    }

    declaration_payload = closeout_base.payload(
        "8.7.56.1953",
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
                "extension_gate": closeout_base.display_path(EXTENSION_GATE),
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
        "8.7.56.1954",
        STEP_NAME + " route sync",
        declaration_payload["inputs"],
        [
            closeout_base.row(
                "extended_interval_promotion_retained",
                "pass" if extended_interval_promotion_retained else "reject",
                "extended interval promotion retained",
                closeout_base.truth(extended_interval_promotion_retained),
                "The exact signed form-factor promotion is now retained on the doubled audit interval.",
            ),
            closeout_base.row(
                "same_level_post_dormant_retry_admissible",
                "reject",
                "same-level post-dormant retry admissible",
                closeout_base.truth(False),
                "The old dormant family has been superseded by the interval-extension theorem.",
            ),
            closeout_base.row(
                "next_route_fixed",
                "pass",
                "next route fixed",
                1.0,
                "The next official branch is the conditional further interval-extension / new signed observable rule reactivation.",
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

    print("[ok] 8.7.56.1951-.1954 extended-interval closeout artifacts generated")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
