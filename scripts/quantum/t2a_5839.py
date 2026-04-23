#!/usr/bin/env python3
"""Generate 8.7.56.5839-.5842 Trial-2 4D exact-goal closeout artifacts."""

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
        "8.7.56.5835-5838",
        "updated_pack_trial2_4d_exact_constant_selector_theorem_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5839-5842"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack Trial-2 "
    "4D exact-goal closeout gate"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_trial2_4d_exact_goal_closeout_gate",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_4d_exact_constant_selector_theorem_audited_"
    "exact_goal_closeout_primary_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_4d_canonical_exact_alpha_correction_completed_"
    "zero_residual_exact_goal_unavailable_current_pack_conditional_reopen_only_next"
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
    """Return formulas fixed by the 4D exact-goal closeout gate."""
    return {
        "canonical_formula": "alpha_4D,can(beta_*) = alpha_3D(beta_*) / M_4D(beta_*, 1, ±1)^2",
        "closeout_reading": (
            "If the canonical 4D selector theorem is available but "
            "|alpha_4D,can - 1/137| > 0, the honest verdict is canonical 4D "
            "correction completed / zero-residual exact-goal unavailable in the "
            "current pack."
        ),
    }


# 関数: `.5839-.5842` を実行する。

def main() -> None:
    """Execute the 4D exact-goal closeout gate."""
    sign_base.require(PRIOR_GATE)

    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    pack = build_trial2_4d_exact_constant_selector_theorem_pack()

    route_selected = (
        str(prior_summary["trial2_numeric_alpha_problem_classification"]) == PRIOR_CLASS
    )
    canonical_formula_available = bool(
        pack["exact_trial2_4d_canonical_exact_alpha_correction_formula_available_now"]
    )
    zero_residual_unavailable = not bool(
        pack["exact_trial2_4d_zero_residual_exact_goal_available_now"]
    )
    closeout_negative = bool(canonical_formula_available and zero_residual_unavailable)
    conditional_reopen_only = bool(closeout_negative)

    rows = [
        sign_base.row(
            "updated_pack_trial2_4d_exact_constant_selector_theorem_selected_now",
            "pass" if route_selected else "reject",
            "updated-pack Trial-2 4D exact-constant selector theorem selected now",
            sign_base.truth(route_selected),
            "The exact-goal closeout gate starts only after the canonical 4D selector theorem has already been synchronized.",
        ),
        sign_base.row(
            "exact_trial2_4d_canonical_exact_alpha_correction_formula_available_now",
            "pass" if canonical_formula_available else "reject",
            "exact Trial-2 4D canonical exact-alpha correction formula available now",
            sign_base.truth(canonical_formula_available),
            "The current pack now has one canonical 4D correction formula anchored target-free inside the deterministic selector family.",
        ),
        sign_base.row(
            "exact_trial2_4d_zero_residual_exact_goal_unavailable_now",
            "pass" if zero_residual_unavailable else "reject",
            "exact Trial-2 4D zero-residual exact goal unavailable now",
            sign_base.truth(zero_residual_unavailable),
            "Even the canonical 4D correction does not collapse the exact goal 1/137 to zero residual in the current pack.",
        ),
        sign_base.row(
            "exact_trial2_4d_exact_goal_closeout_negative_closeout_now",
            "pass" if closeout_negative else "reject",
            "exact Trial-2 4D exact-goal closeout negative closeout now",
            sign_base.truth(closeout_negative),
            "Therefore the 4D exact-goal route closes honestly as canonicalized but still nonzero-residual.",
        ),
        sign_base.row(
            "updated_pack_trial2_exact_goal_conditional_reopen_only_now",
            "pass" if conditional_reopen_only else "reject",
            "updated-pack Trial-2 exact goal conditional reopen only now",
            sign_base.truth(conditional_reopen_only),
            "No unconditional next official branch remains inside the current exact-goal pack; reopen requires a genuinely new exact-constant extraction route or a genuinely new selected-extension-native computation branch.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "canonical_selector_label": str(pack["canonical_selector_label"]),
        "canonical_formula_label": str(pack["canonical_formula_label"]),
        "alpha_4d_canonical": float(pack["canonical_corrected_alpha"]),
        "alpha_4d_canonical_rel_error_vs_exact_goal": float(
            pack["canonical_rel_error_vs_exact_goal"]
        ),
        "alpha_4d_canonical_rel_error_vs_observed_target": float(
            pack["canonical_rel_error_vs_observed_target"]
        ),
        "exact_trial2_4d_zero_residual_exact_goal_available_now": bool(
            pack["exact_trial2_4d_zero_residual_exact_goal_available_now"]
        ),
        "selected_next_generation_route": "none",
        "recommended_next_route_or_none": "No unconditional next official branch",
        "selected_followup_route": "conditional_reopen_only",
        "selected_followup_route_or_none": (
            "genuinely new exact-constant extraction route or genuinely new selected-extension-native computation branch only"
        ),
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5841",
        STEP_NAME + " declaration gate",
        {
            "source_files": {"prior_gate": sign_base.display_path(PRIOR_GATE)},
            "formulae": build_formulae(),
        },
        rows,
        summary,
        {
            "overall_status": "trial2_4d_exact_goal_closeout_negative_closeout",
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
    print("[done] 8.7.56.5839-5842 Trial-2 4D exact-goal closeout gate completed")
    print(f"[done] declaration: {outputs['json']}")


if __name__ == "__main__":
    main()
