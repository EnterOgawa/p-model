#!/usr/bin/env python3
"""Generate 8.7.56.5795-.5798 exact-alpha extraction artifacts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.quantum.t2a_1843 as sign_base
from scripts.quantum.trial2_exact_alpha_closed_form_extraction_backend import (
    build_trial2_exact_alpha_closed_form_extraction_pack,
)
from scripts.utils.windows_length_policy import build_compact_artifact_stem
from scripts.utils.windows_length_policy import build_metrics_paths


PUBLIC_OUT = ROOT / "output" / "public" / "quantum"
PRIOR_AUDIT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5791-5794",
        "updated_pack_trial2_invariant_reduction_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5795-5798"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack Trial-2 "
    "exact-alpha closed-form extraction audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_trial2_exact_alpha_closed_form_extraction_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_invariant_reduction_audited_"
    "exact_alpha_extraction_primary_zero_residual_secondary_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_exact_alpha_finite_invariant_form_completed_"
    "zero_residual_final_theorem_primary_conditional_hold_secondary_next"
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


# 関数: exact-alpha extraction の式 bundle を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas fixed by the exact-alpha extraction audit."""
    return {
        "exact_alpha_formula": (
            "alpha(beta) = [4(g + eps - b) - q] "
            "[2(5 + beta^2) + 10 g - q - 4 b] / [36 (1 + beta^2)^2]"
        ),
        "exact_goal": "alpha_goal = 1 / 137",
        "symbolic_root_eval": "alpha_exact = alpha(beta_symbolic_root)",
    }


# 関数: `.5795-.5798` を実行する。

def main() -> None:
    """Execute the exact-alpha closed-form extraction audit."""
    sign_base.require(PRIOR_AUDIT)

    prior_summary = sign_base.read_json(PRIOR_AUDIT)["summary"]
    pack = build_trial2_exact_alpha_closed_form_extraction_pack()

    route_selected = (
        str(prior_summary["trial2_numeric_alpha_problem_classification"]) == PRIOR_CLASS
    )
    finite_form_retained = bool(
        pack["exact_trial2_finite_invariant_alpha_form_retained_now"]
    )
    exact_one_over_137_unavailable = not bool(
        pack["exact_trial2_constant_extraction_one_over_137_available_now"]
    )
    observed_zero_residual_unavailable = not bool(
        pack["exact_trial2_observed_target_zero_residual_available_now"]
    )
    zero_residual_gate_required = bool(
        pack["updated_pack_trial2_zero_residual_final_theorem_gate_required_now"]
    )

    rows = [
        sign_base.row(
            "updated_pack_trial2_exact_alpha_extraction_selected_now",
            "pass" if route_selected else "reject",
            "updated-pack Trial-2 exact-alpha extraction selected now",
            sign_base.truth(route_selected),
            "This audit starts only after finite invariant reduction is official and exact constant extraction is the declared live blocker.",
        ),
        sign_base.row(
            "exact_trial2_finite_invariant_alpha_form_retained_now",
            "pass" if finite_form_retained else "reject",
            "exact Trial-2 finite invariant alpha form retained now",
            sign_base.truth(finite_form_retained),
            "Pass means alpha(beta) is now carried by one exact finite-invariant formula at the symbolic root.",
        ),
        sign_base.row(
            "exact_trial2_constant_extraction_one_over_137_unavailable_now",
            "pass" if exact_one_over_137_unavailable else "reject",
            "exact Trial-2 constant extraction one-over-137 unavailable now",
            sign_base.truth(exact_one_over_137_unavailable),
            "The current finite-invariant exact alpha formula does not yet collapse to the exact constant 1/137 with zero residual.",
        ),
        sign_base.row(
            "exact_trial2_observed_target_zero_residual_unavailable_now",
            "pass" if observed_zero_residual_unavailable else "reject",
            "exact Trial-2 observed-target zero residual unavailable now",
            sign_base.truth(observed_zero_residual_unavailable),
            "The exact symbolic evaluation also does not yet hit the observed alpha target with zero residual.",
        ),
        sign_base.row(
            "updated_pack_trial2_zero_residual_final_theorem_gate_required_now",
            "pass" if zero_residual_gate_required else "reject",
            "updated-pack Trial-2 zero-residual final theorem gate required now",
            sign_base.truth(zero_residual_gate_required),
            "The next honest step is no longer invariant reduction but a final theorem gate that classifies the exact-constant extraction verdict.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "alpha_goal_exact_one_over_137": float(pack["alpha_goal_exact_one_over_137"]),
        "alpha_target_observed": float(pack["alpha_target_observed"]),
        "beta_symbolic_root": float(pack["beta_symbolic_root"]),
        "alpha_exact_symbolic": float(pack["alpha_exact_symbolic"]),
        "one_over_alpha_exact_symbolic": float(pack["one_over_alpha_exact_symbolic"]),
        "alpha_exact_symbolic_rel_error_vs_exact_goal": float(
            pack["alpha_exact_symbolic_rel_error_vs_exact_goal"]
        ),
        "alpha_exact_symbolic_rel_error_vs_observed_target": float(
            pack["alpha_exact_symbolic_rel_error_vs_observed_target"]
        ),
        "exact_trial2_finite_invariant_alpha_form_retained_now": bool(
            finite_form_retained
        ),
        "exact_trial2_constant_extraction_one_over_137_available_now": False,
        "exact_trial2_observed_target_zero_residual_available_now": False,
        "updated_pack_trial2_zero_residual_final_theorem_gate_required_now": bool(
            zero_residual_gate_required
        ),
        "selected_next_generation_route": "trial2_zero_residual_final_theorem_gate",
        "recommended_next_route_or_none": ".5799-.5802",
        "selected_followup_route": "none_or_reopen_after_gate",
        "selected_followup_route_or_none": "none",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5797",
        STEP_NAME + " declaration gate",
        {
            "source_files": {"prior_audit": sign_base.display_path(PRIOR_AUDIT)},
            "formulae": build_formulae(),
        },
        rows,
        summary,
        {
            "overall_status": "trial2_exact_alpha_closed_form_extraction_audited",
            "branch_completed": True,
            "breakthrough_passed_now": finite_form_retained,
            "physical_reject_required": False,
        },
        {
            "beta_symbolic_root": float(pack["beta_symbolic_root"]),
            "alpha_exact_symbolic": float(pack["alpha_exact_symbolic"]),
            "one_over_alpha_exact_symbolic": float(pack["one_over_alpha_exact_symbolic"]),
            "alpha_exact_symbolic_rel_error_vs_exact_goal": float(
                pack["alpha_exact_symbolic_rel_error_vs_exact_goal"]
            ),
        },
    )
    outputs = write_artifact("declaration_gate", payload)
    print("[done] 8.7.56.5795-5798 Trial-2 exact-alpha closed-form extraction audit completed")
    print(f"[done] declaration: {outputs['json']}")


if __name__ == "__main__":
    main()
