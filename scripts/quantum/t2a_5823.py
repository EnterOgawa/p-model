#!/usr/bin/env python3
"""Generate 8.7.56.5823-.5826 Trial-2 4D augmentation artifacts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.quantum.t2a_1843 as sign_base
from scripts.quantum.trial2_4d_time_component_augmentation_backend import (
    build_trial2_4d_time_component_augmentation_pack,
)
from scripts.utils.windows_length_policy import build_compact_artifact_stem
from scripts.utils.windows_length_policy import build_metrics_paths


PUBLIC_OUT = ROOT / "output" / "public" / "quantum"
PRIOR_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5819-5822",
        "updated_pack_trial2_q_elimination_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5823-5826"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack Trial-2 "
    "4D time-component augmentation audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_trial2_4d_time_component_augmentation_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_q_elimination_negative_closeout_completed_"
    "three_d_internal_exactification_exhausted_4d_time_component_primary_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_4d_time_component_augmentation_audited_"
    "leading_mass_sq_correction_primary_exact_alpha_refresh_next"
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
    """Return formulas fixed by the 4D augmentation audit."""
    return {
        "baseline_alpha_formula": (
            "alpha_3D(beta_*) = [4(g + eps - b) - q][2(5 + beta^2) + 10 g - q - 4 b] "
            "/ [36 (1 + beta^2)^2]"
        ),
        "time_component_selector": (
            "leading nontrivial 4D selector := (ell, s) = (1, ±1)"
        ),
        "charge_factor_rule": (
            "C_4D(beta, ell, s) := coupled_charge_factor(beta, ell, s)"
        ),
        "mass_factor_rule": (
            "M_4D(beta, ell, s) := coupled_mass_factor(beta, ell, s)"
        ),
        "primary_correction_rule": (
            "alpha_4D,lead(beta_*) := alpha_3D(beta_*) / M_4D(beta_*, 1, ±1)^2"
        ),
    }


# 関数: `.5823-.5826` を実行する。

def main() -> None:
    """Execute the 4D time-component augmentation audit."""
    sign_base.require(PRIOR_GATE)

    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    pack = build_trial2_4d_time_component_augmentation_pack()
    primary = pack["leading_primary_formula_row"]

    route_selected = (
        str(prior_summary["trial2_numeric_alpha_problem_classification"]) == PRIOR_CLASS
    )
    selector_family_available = bool(
        pack["exact_trial2_4d_selector_family_machine_readable_now"]
    )
    leading_selector_available = bool(
        pack["exact_trial2_4d_leading_time_component_selector_available_now"]
    )
    correction_required = bool(pack["exact_trial2_4d_exact_alpha_correction_required_now"])

    rows = [
        sign_base.row(
            "updated_pack_trial2_q_negative_closeout_selected_now",
            "pass" if route_selected else "reject",
            "updated-pack Trial-2 q negative closeout selected now",
            sign_base.truth(route_selected),
            "The 4D augmentation audit starts only after the remaining honest 3D internal route has closed negatively.",
        ),
        sign_base.row(
            "exact_trial2_4d_selector_family_machine_readable_now",
            "pass" if selector_family_available else "reject",
            "exact Trial-2 4D selector family machine-readable now",
            sign_base.truth(selector_family_available),
            "The current pack now materializes one target-free 4D selector family from retained vector-Q-ball coupled factors.",
        ),
        sign_base.row(
            "exact_trial2_4d_leading_time_component_selector_available_now",
            "pass" if leading_selector_available else "reject",
            "exact Trial-2 leading time-component selector available now",
            sign_base.truth(leading_selector_available),
            "The lowest nontrivial time-component-active selector is fixed as the leading mode (ell=1, |s|=1).",
        ),
        sign_base.row(
            "exact_trial2_4d_primary_correction_formula_explicit_now",
            "pass" if leading_selector_available else "reject",
            "exact Trial-2 primary 4D correction formula explicit now",
            sign_base.truth(leading_selector_available),
            "Because alpha maps quadratically to the corrected form-factor amplitude, the first natural 4D correction is the inverse squared mass-factor rule.",
        ),
        sign_base.row(
            "updated_pack_trial2_4d_exact_alpha_correction_required_now",
            "pass" if correction_required else "reject",
            "updated-pack Trial-2 4D exact-alpha correction required now",
            sign_base.truth(correction_required),
            "The selector family is now explicit, so the next honest move is to evaluate whether the leading correction actually absorbs the exact-goal residual.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "beta_symbolic_root": float(pack["beta_symbolic_root"]),
        "alpha_exact_symbolic": float(pack["alpha_exact_symbolic"]),
        "baseline_rel_error_vs_exact_goal": float(
            pack["alpha_exact_symbolic_rel_error_vs_exact_goal"]
        ),
        "leading_selector_label": str(pack["leading_selector_label"]),
        "leading_primary_formula_label": str(pack["leading_primary_formula_label"]),
        "leading_selector_ell": int(primary["ell"]),
        "leading_selector_s": int(primary["s"]),
        "leading_primary_corrected_alpha": float(primary["corrected_alpha"]),
        "leading_primary_rel_error_vs_exact_goal": float(
            primary["corrected_alpha_rel_error_vs_exact_goal"]
        ),
        "exact_trial2_4d_exact_alpha_correction_required_now": bool(correction_required),
        "selected_next_generation_route": "trial2_4d_time_component_exact_alpha_correction_audit",
        "recommended_next_route_or_none": ".5827-.5830",
        "selected_followup_route": "trial2_4d_residual_absorption_gate",
        "selected_followup_route_or_none": ".5831-.5834",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5825",
        STEP_NAME + " declaration gate",
        {
            "source_files": {"prior_gate": sign_base.display_path(PRIOR_GATE)},
            "formulae": build_formulae(),
        },
        rows,
        summary,
        {
            "overall_status": "trial2_4d_time_component_augmentation_audited",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {
            "leading_primary_formula_row": primary,
            "best_formula_row": pack["best_formula_row"],
            "retained_exact_ladder_anchor_row": pack["retained_exact_ladder_anchor_row"],
        },
    )
    outputs = write_artifact("declaration_gate", payload)
    print("[done] 8.7.56.5823-5826 Trial-2 4D time-component augmentation audit completed")
    print(f"[done] declaration: {outputs['json']}")


if __name__ == "__main__":
    main()
