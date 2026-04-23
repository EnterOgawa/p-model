#!/usr/bin/env python3
"""Generate 8.7.56.5799-.5802 zero-residual final theorem gate artifacts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.quantum.t2a_1843 as sign_base
from scripts.utils.windows_length_policy import build_compact_artifact_stem
from scripts.utils.windows_length_policy import build_metrics_paths


PUBLIC_OUT = ROOT / "output" / "public" / "quantum"
PRIOR_AUDIT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5795-5798",
        "updated_pack_trial2_exact_alpha_closed_form_extraction_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5799-5802"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack Trial-2 "
    "zero-residual final theorem gate"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_trial2_zero_residual_final_theorem_gate",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_exact_alpha_finite_invariant_form_completed_"
    "zero_residual_final_theorem_primary_conditional_hold_secondary_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_exact_alpha_finite_invariant_form_completed_"
    "zero_residual_exact_constant_unavailable_current_pack_"
    "conditional_reopen_only_next"
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


# 関数: gate で使う式 bundle を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used by the zero-residual final theorem gate."""
    return {
        "exact_alpha_formula": (
            "alpha(beta_symbolic_root) = [4(g + eps - b) - q] "
            "[2(5 + beta^2) + 10 g - q - 4 b] / [36 (1 + beta^2)^2]"
        ),
        "exact_goal": "alpha_goal = 1 / 137",
        "gate_rule": "Zero-residual final theorem passes only if alpha_exact_symbolic = 1 / 137 exactly.",
    }


# 関数: `.5799-.5802` を実行する。

def main() -> None:
    """Execute the zero-residual final theorem gate."""
    sign_base.require(PRIOR_AUDIT)

    prior_summary = sign_base.read_json(PRIOR_AUDIT)["summary"]
    route_selected = (
        str(prior_summary["trial2_numeric_alpha_problem_classification"]) == PRIOR_CLASS
    )
    finite_invariant_form_completed = bool(
        prior_summary["exact_trial2_finite_invariant_alpha_form_retained_now"]
    )
    exact_constant_unavailable = not bool(
        prior_summary["exact_trial2_constant_extraction_one_over_137_available_now"]
    )
    observed_zero_residual_unavailable = not bool(
        prior_summary["exact_trial2_observed_target_zero_residual_available_now"]
    )
    conditional_reopen_only = bool(
        finite_invariant_form_completed
        and exact_constant_unavailable
        and observed_zero_residual_unavailable
    )

    rows = [
        sign_base.row(
            "gate_a_trial2_zero_residual_final_theorem_selected_now",
            "pass" if route_selected else "reject",
            "gate A Trial-2 zero-residual final theorem selected now",
            sign_base.truth(route_selected),
            "The final theorem gate starts only after exact-alpha closed-form extraction has been honestly classified.",
        ),
        sign_base.row(
            "gate_b_trial2_finite_invariant_alpha_form_completed_now",
            "pass" if finite_invariant_form_completed else "reject",
            "gate B Trial-2 finite invariant alpha form completed now",
            sign_base.truth(finite_invariant_form_completed),
            "The current pack does retain one exact finite-invariant alpha formula at the symbolic root.",
        ),
        sign_base.row(
            "gate_c_trial2_exact_constant_one_over_137_unavailable_now",
            "pass" if exact_constant_unavailable else "reject",
            "gate C Trial-2 exact constant one-over-137 unavailable now",
            sign_base.truth(exact_constant_unavailable),
            "The current pack does not yet collapse the exact alpha formula to the exact constant 1/137 with zero residual.",
        ),
        sign_base.row(
            "gate_d_trial2_conditional_reopen_only_now",
            "pass" if conditional_reopen_only else "reject",
            "gate D Trial-2 conditional reopen only now",
            sign_base.truth(conditional_reopen_only),
            "With finite-invariant exact alpha fixed but zero-residual constant extraction unavailable, no unconditional next official branch remains inside the current pack.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "alpha_goal_exact_one_over_137": float(prior_summary["alpha_goal_exact_one_over_137"]),
        "alpha_exact_symbolic": float(prior_summary["alpha_exact_symbolic"]),
        "one_over_alpha_exact_symbolic": float(prior_summary["one_over_alpha_exact_symbolic"]),
        "alpha_exact_symbolic_rel_error_vs_exact_goal": float(
            prior_summary["alpha_exact_symbolic_rel_error_vs_exact_goal"]
        ),
        "alpha_exact_symbolic_rel_error_vs_observed_target": float(
            prior_summary["alpha_exact_symbolic_rel_error_vs_observed_target"]
        ),
        "trial2_zero_residual_final_theorem_available_now": False,
        "no_unconditional_next_official_branch_now": bool(conditional_reopen_only),
        "selected_next_generation_route": "none_unconditional",
        "recommended_next_route_or_none": "none",
        "selected_followup_route": "conditional_reopen_only",
        "selected_followup_route_or_none": "reopen_only_if_genuinely_new_exact_constant_route_materializes",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5801",
        STEP_NAME + " declaration gate",
        {
            "source_files": {"prior_audit": sign_base.display_path(PRIOR_AUDIT)},
            "formulae": build_formulae(),
        },
        rows,
        summary,
        {
            "overall_status": "trial2_zero_residual_final_theorem_gate_completed",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {
            "alpha_exact_symbolic": float(prior_summary["alpha_exact_symbolic"]),
            "one_over_alpha_exact_symbolic": float(
                prior_summary["one_over_alpha_exact_symbolic"]
            ),
            "alpha_exact_symbolic_rel_error_vs_exact_goal": float(
                prior_summary["alpha_exact_symbolic_rel_error_vs_exact_goal"]
            ),
        },
    )
    outputs = write_artifact("declaration_gate", payload)
    print("[done] 8.7.56.5799-5802 Trial-2 zero-residual final theorem gate completed")
    print(f"[done] declaration: {outputs['json']}")


if __name__ == "__main__":
    main()
