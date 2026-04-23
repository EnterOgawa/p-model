#!/usr/bin/env python3
"""Generate 8.7.56.5835-.5838 Trial-2 4D exact-constant selector theorem artifacts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.quantum.t2a_1843 as sign_base
from scripts.quantum.trial2_4d_exact_constant_selector_theorem_backend import (
    build_trial2_4d_exact_constant_selector_theorem_pack,
)
from scripts.utils.windows_length_policy import build_compact_artifact_stem
from scripts.utils.windows_length_policy import build_metrics_paths


PUBLIC_OUT = ROOT / "output" / "public" / "quantum"
PRIOR_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5831-5834",
        "updated_pack_trial2_4d_residual_absorption_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5835-5838"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack Trial-2 "
    "4D exact-constant selector theorem audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_trial2_4d_exact_constant_selector_theorem_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_4d_residual_absorption_gate_completed_"
    "exact_constant_selector_theorem_primary_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_4d_exact_constant_selector_theorem_audited_"
    "exact_goal_closeout_primary_next"
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


# 関数: route で固定する式 bundle を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas fixed by the 4D exact-constant selector theorem audit."""
    return {
        "canonical_selector": "selector_can := leading_nontrivial_time_component = (ell, s) = (1, ±1)",
        "canonical_rule": "alpha_4D,can(beta_*) := alpha_3D(beta_*) / M_4D(beta_*, 1, ±1)^2",
        "theorem_reading": (
            "The selector theorem is available once the same row is uniquely best "
            "overall, uniquely best within the leading selector, uniquely best "
            "within the mass-squared family, and supported by dominant "
            "time-component weight."
        ),
    }


# 関数: `.5835-.5838` を実行する。

def main() -> None:
    """Execute the 4D exact-constant selector theorem audit."""
    sign_base.require(PRIOR_GATE)

    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    pack = build_trial2_4d_exact_constant_selector_theorem_pack()

    route_selected = (
        str(prior_summary["trial2_numeric_alpha_problem_classification"]) == PRIOR_CLASS
    )
    canonical_formula_available = bool(
        pack["exact_trial2_4d_canonical_exact_alpha_correction_formula_available_now"]
    )
    selector_theorem_available = bool(
        pack["exact_trial2_4d_exact_constant_selector_theorem_available_now"]
    )
    zero_residual_unavailable = not bool(
        pack["exact_trial2_4d_zero_residual_exact_goal_available_now"]
    )
    exact_goal_closeout_required = bool(
        pack["exact_trial2_4d_exact_goal_closeout_gate_required_now"]
    )

    rows = [
        sign_base.row(
            "updated_pack_trial2_4d_residual_absorption_gate_selected_now",
            "pass" if route_selected else "reject",
            "updated-pack Trial-2 4D residual-absorption gate selected now",
            sign_base.truth(route_selected),
            "The selector theorem audit starts only after the 4D family has already been canonized as one best residual absorber.",
        ),
        sign_base.row(
            "exact_trial2_4d_canonical_exact_alpha_correction_formula_available_now",
            "pass" if canonical_formula_available else "reject",
            "exact Trial-2 4D canonical exact-alpha correction formula available now",
            sign_base.truth(canonical_formula_available),
            "The current deterministic family now fixes one canonical correction formula alpha_4D,can = alpha_3D / M_4D^2.",
        ),
        sign_base.row(
            "exact_trial2_4d_exact_constant_selector_theorem_available_now",
            "pass" if selector_theorem_available else "reject",
            "exact Trial-2 4D exact-constant selector theorem available now",
            sign_base.truth(selector_theorem_available),
            "The leading time-component selector and inverse-squared mass rule now form one target-free canonical theorem inside the current 4D family.",
        ),
        sign_base.row(
            "exact_trial2_4d_zero_residual_exact_goal_unavailable_now",
            "pass" if zero_residual_unavailable else "reject",
            "exact Trial-2 4D zero-residual exact goal unavailable now",
            sign_base.truth(zero_residual_unavailable),
            "The canonical selector theorem still does not collapse the exact goal to zero residual inside the current pack.",
        ),
        sign_base.row(
            "updated_pack_trial2_4d_exact_goal_closeout_gate_required_now",
            "pass" if exact_goal_closeout_required else "reject",
            "updated-pack Trial-2 4D exact-goal closeout gate required now",
            sign_base.truth(exact_goal_closeout_required),
            "The next honest blocker is no longer selector ambiguity but the final exact-goal closeout verdict itself.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "canonical_selector_label": str(pack["canonical_selector_label"]),
        "canonical_formula_label": str(pack["canonical_formula_label"]),
        "canonical_corrected_alpha": float(pack["canonical_corrected_alpha"]),
        "canonical_rel_error_vs_exact_goal": float(
            pack["canonical_rel_error_vs_exact_goal"]
        ),
        "canonical_rel_error_vs_observed_target": float(
            pack["canonical_rel_error_vs_observed_target"]
        ),
        "canonical_mass_factor": float(pack["canonical_mass_factor"]),
        "canonical_charge_factor": float(pack["canonical_charge_factor"]),
        "exact_trial2_4d_exact_constant_selector_theorem_available_now": bool(
            selector_theorem_available
        ),
        "exact_trial2_4d_zero_residual_exact_goal_available_now": bool(
            pack["exact_trial2_4d_zero_residual_exact_goal_available_now"]
        ),
        "selected_next_generation_route": "trial2_4d_exact_goal_closeout_gate",
        "recommended_next_route_or_none": ".5839-.5842",
        "selected_followup_route": "trial2_4d_exact_goal_closeout_gate",
        "selected_followup_route_or_none": ".5839-.5842",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5837",
        STEP_NAME + " declaration gate",
        {
            "source_files": {"prior_gate": sign_base.display_path(PRIOR_GATE)},
            "formulae": build_formulae(),
        },
        rows,
        summary,
        {
            "overall_status": "trial2_4d_exact_constant_selector_theorem_audited",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {
            "best_row": pack["best_row"],
            "leading_selector_row": pack["leading_selector_row"],
            "next_nonzero_time_row": pack["next_nonzero_time_row"],
        },
    )
    outputs = write_artifact("declaration_gate", payload)
    print(
        "[done] 8.7.56.5835-5838 Trial-2 4D exact-constant selector theorem audit completed"
    )
    print(f"[done] declaration: {outputs['json']}")


if __name__ == "__main__":
    main()
