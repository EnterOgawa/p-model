#!/usr/bin/env python3
"""Generate 8.7.56.5883-.5886 self-consistent 4D selector gate artifacts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.quantum.t2a_1843 as sign_base
from scripts.quantum.trial2_4d_self_consistent_selector_exact_goal_backend import (
    build_trial2_4d_self_consistent_selector_exact_goal_pack,
)
from scripts.utils.windows_length_policy import build_compact_artifact_stem
from scripts.utils.windows_length_policy import build_metrics_paths


PUBLIC_OUT = ROOT / "output" / "public" / "quantum"
PRIOR_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5879-5882",
        "updated_pack_trial2_4d_self_consistent_selector_exact_goal_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5883-5886"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack Trial-2 "
    "4D self-consistent selector gate"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_trial2_4d_self_consistent_selector_gate",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_4d_self_consistent_selector_audited_current_family_negative_partial_gate"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_4d_self_consistent_selector_negative_closeout_completed_zero_residual_"
    "exact_goal_unavailable_current_pack_conditional_reopen_only_next"
)


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

    return {"json": sign_base.display_path(paths["json"])}


# 関数: gate で固定する式 bundle を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas fixed by the self-consistent selector gate."""
    return {
        "best_self_consistent_pair": (
            "alpha_qstar^(4D,all_mix)(beta) = alpha_R8^(4D,half_mix)(beta)"
        ),
        "expert_minimal_pair": (
            "alpha_qstar^(4D,vertex)(beta) = alpha_R8^(4D,mass_sq)(beta)"
        ),
        "gate_rule": (
            "The route advances only if a deterministic self-consistent 4D pair beats the retained external-probe candidate."
        ),
    }


# 関数: `.5883-.5886` を実行する。

def main() -> None:
    """Execute the self-consistent 4D selector gate."""
    sign_base.require(PRIOR_GATE)

    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    pack = build_trial2_4d_self_consistent_selector_exact_goal_pack()
    best_row = dict(pack["best_row"])
    expert_row = dict(pack["expert_minimal_row"])

    route_selected = (
        str(prior_summary["trial2_numeric_alpha_problem_classification"]) == PRIOR_CLASS
    )
    family_available = bool(
        pack["exact_trial2_4d_self_consistent_selector_family_available_now"]
    )
    positive_partial = bool(
        pack["exact_trial2_4d_self_consistent_selector_positive_partial_now"]
    )
    beats_current_best = bool(
        pack["exact_trial2_4d_self_consistent_selector_beats_current_best_now"]
    )
    zero_residual_available = bool(
        pack["exact_trial2_4d_self_consistent_selector_zero_residual_exact_goal_available_now"]
    )
    negative_closeout = bool(
        pack["updated_pack_trial2_4d_self_consistent_selector_negative_closeout_now"]
    )
    no_unconditional_next = bool(negative_closeout and not zero_residual_available)

    rows = [
        sign_base.row(
            "updated_pack_trial2_4d_self_consistent_selector_audit_selected_now",
            "pass" if route_selected else "reject",
            "updated-pack Trial-2 4D self-consistent selector audit selected now",
            sign_base.truth(route_selected),
            "The gate starts only after the deterministic self-consistent family has been audited inside the current pack.",
        ),
        sign_base.row(
            "exact_trial2_4d_self_consistent_selector_family_available_now",
            "pass" if family_available else "reject",
            "exact Trial-2 4D self-consistent selector family available now",
            sign_base.truth(family_available),
            "The route is a real computation surface, not a theorem-side placeholder.",
        ),
        sign_base.row(
            "exact_trial2_4d_self_consistent_selector_positive_partial_now",
            "pass" if positive_partial else "reject",
            "exact Trial-2 4D self-consistent selector positive partial now",
            sign_base.truth(positive_partial),
            "The best deterministic self-consistent pair still improves the canonical 4D row.",
        ),
        sign_base.row(
            "exact_trial2_4d_self_consistent_selector_beats_current_best_now",
            "pass" if beats_current_best else "reject",
            "exact Trial-2 4D self-consistent selector beats current best now",
            sign_base.truth(beats_current_best),
            "Passing would mean the self-consistent 4D root supersedes the retained external-probe current-vertex candidate.",
        ),
        sign_base.row(
            "exact_trial2_4d_self_consistent_selector_zero_residual_exact_goal_available_now",
            "pass" if zero_residual_available else "reject",
            "exact Trial-2 4D self-consistent selector zero-residual exact goal available now",
            sign_base.truth(zero_residual_available),
            "Passing would mean the self-consistent 4D route already closes alpha = 1/137 inside the current deterministic family.",
        ),
        sign_base.row(
            "updated_pack_trial2_4d_self_consistent_selector_negative_closeout_now",
            "pass" if negative_closeout else "reject",
            "updated-pack Trial-2 4D self-consistent selector negative closeout now",
            sign_base.truth(negative_closeout),
            "Because the best deterministic self-consistent pair remains weaker than the retained current-best candidate, the route closes negatively as a main residual-closing branch.",
        ),
        sign_base.row(
            "updated_pack_trial2_no_unconditional_next_branch_now",
            "pass" if no_unconditional_next else "reject",
            "updated-pack Trial-2 no unconditional next branch now",
            sign_base.truth(no_unconditional_next),
            "The current pack still needs a genuinely new exact-constant extraction route or a genuinely new selected-extension-native source/computation branch before reopening.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "best_pair_q_label": str(best_row["q_label"]),
        "best_pair_r_label": str(best_row["r_label"]),
        "best_pair_beta_root": float(best_row["root_beta"]),
        "best_pair_alpha": float(best_row["alpha_self_consistent_at_root"]),
        "best_pair_rel_error_vs_exact_goal": float(
            best_row["alpha_self_consistent_rel_error_vs_exact_goal"]
        ),
        "best_pair_improvement_factor_vs_canonical": float(
            pack["best_pair_improvement_factor_vs_canonical"]
        ),
        "best_pair_improvement_factor_vs_current_best": float(
            pack["best_pair_improvement_factor_vs_current_best"]
        ),
        "expert_pair_q_label": str(expert_row["q_label"]),
        "expert_pair_r_label": str(expert_row["r_label"]),
        "expert_pair_beta_root": float(expert_row["root_beta"]),
        "expert_pair_alpha": float(expert_row["alpha_self_consistent_at_root"]),
        "expert_pair_rel_error_vs_exact_goal": float(
            expert_row["alpha_self_consistent_rel_error_vs_exact_goal"]
        ),
        "selected_next_generation_route": "none",
        "recommended_next_route_or_none": "No unconditional next official branch",
        "selected_followup_route": "conditional_reopen_only",
        "selected_followup_route_or_none": (
            "genuinely new exact-constant extraction route or genuinely new selected-extension-native source/computation branch only"
        ),
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5885",
        STEP_NAME + " declaration gate",
        {
            "source_files": {"prior_gate": sign_base.display_path(PRIOR_GATE)},
            "formulae": build_formulae(),
        },
        rows,
        summary,
        {
            "overall_status": "trial2_4d_self_consistent_selector_negative_closeout",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {
            "pack": pack,
        },
    )
    outputs = write_artifact("declaration_gate", payload)
    print("[done] 8.7.56.5883-5886 self-consistent 4D selector gate completed")
    print(f"[done] declaration: {outputs['json']}")


if __name__ == "__main__":
    main()
