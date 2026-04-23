#!/usr/bin/env python3
"""Generate 8.7.56.5875-.5878 deterministic weight theorem-source artifacts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.quantum.t2a_1843 as sign_base
from scripts.quantum.trial2_external_probe_weight_theorem_source_backend import (
    build_trial2_external_probe_weight_theorem_source_pack,
)
from scripts.utils.windows_length_policy import build_compact_artifact_stem
from scripts.utils.windows_length_policy import build_metrics_paths


PUBLIC_OUT = ROOT / "output" / "public" / "quantum"
PRIOR_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5871-5874",
        "updated_pack_trial2_sign_flip_interpolation_diagnostic",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5875-5878"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack Trial-2 "
    "theorem source for deterministic external-probe weight"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_trial2_external_probe_weight_theorem_source",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_sign_flip_local_bracket_positive_diagnostic_completed_"
    "weight_source_primary_exact_goal_secondary_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_4d_external_probe_weight_theorem_source_negative_closeout_completed_"
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


# 関数: theorem-source gate で固定する式 bundle を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas fixed by the deterministic weight theorem-source audit."""
    return {
        "nonzero_time_weight": (
            "eta_vertex = w_(1,±1) / sum_(s!=0) w_(ell,s)"
        ),
        "all_selector_weight": (
            "eta_all = w_(1,±1) / sum_(all selectors) w_(ell,s)"
        ),
        "theorem_source_test": (
            "A theorem source would need a frozen-action law selecting one normalization set and collapsing eta_vertex to eta_*."
        ),
    }


# 関数: `.5875-.5878` を実行する。

def main() -> None:
    """Execute the deterministic external-probe weight theorem-source audit."""
    sign_base.require(PRIOR_GATE)

    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    pack = build_trial2_external_probe_weight_theorem_source_pack()

    route_selected = (
        str(prior_summary["trial2_numeric_alpha_problem_classification"]) == PRIOR_CLASS
    )
    computation_source_available = bool(
        pack["exact_trial2_external_probe_weight_computation_source_available_now"]
    )
    normalization_set_sensitive = bool(
        pack["exact_trial2_external_probe_weight_normalization_set_sensitive_now"]
    )
    exact_goal_side_flips = bool(
        pack["exact_trial2_external_probe_weight_exact_goal_side_flips_now"]
    )
    theorem_source_available = bool(
        pack["exact_trial2_external_probe_weight_theorem_source_available_now"]
    )
    theorem_source_negative_closeout = bool(
        computation_source_available
        and normalization_set_sensitive
        and exact_goal_side_flips
        and not theorem_source_available
    )

    rows = [
        sign_base.row(
            "updated_pack_trial2_sign_flip_positive_diagnostic_selected_now",
            "pass" if route_selected else "reject",
            "updated-pack Trial-2 sign-flip positive diagnostic selected now",
            sign_base.truth(route_selected),
            "The theorem-source audit starts only after the local sign-flip geometry has been fixed as a valid secondary diagnostic.",
        ),
        sign_base.row(
            "exact_trial2_external_probe_weight_computation_source_available_now",
            "pass" if computation_source_available else "reject",
            "exact Trial-2 external-probe weight computation source available now",
            sign_base.truth(computation_source_available),
            "The current pack does provide one deterministic computation source for eta_vertex as the normalized leading nonzero-time weight.",
        ),
        sign_base.row(
            "exact_trial2_external_probe_weight_normalization_set_sensitive_now",
            "pass" if normalization_set_sensitive else "reject",
            "exact Trial-2 external-probe weight normalization-set sensitive now",
            sign_base.truth(normalization_set_sensitive),
            "Changing only the normalization set from nonzero-time to all selectors materially moves the deterministic weight candidate.",
        ),
        sign_base.row(
            "exact_trial2_external_probe_weight_exact_goal_side_flips_now",
            "pass" if exact_goal_side_flips else "reject",
            "exact Trial-2 external-probe weight exact-goal side flips now",
            sign_base.truth(exact_goal_side_flips),
            "The two deterministic normalization choices land on opposite sides of the exact goal, so the missing piece is the selector law itself.",
        ),
        sign_base.row(
            "exact_trial2_external_probe_weight_theorem_source_unavailable_now",
            "pass" if not theorem_source_available else "reject",
            "exact Trial-2 external-probe weight theorem source unavailable now",
            sign_base.truth(not theorem_source_available),
            "The current pack has no frozen-action theorem selecting eta_vertex over nearby deterministic normalizations and collapsing it to eta_*.",
        ),
        sign_base.row(
            "updated_pack_trial2_external_probe_weight_theorem_source_negative_closeout_now",
            "pass" if theorem_source_negative_closeout else "reject",
            "updated-pack Trial-2 external-probe weight theorem source negative closeout now",
            sign_base.truth(theorem_source_negative_closeout),
            "Therefore the deterministic weight route closes honestly: strong computation source exists, but theorem source and zero-residual exact-goal closeout remain unavailable in the current pack.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "eta_exact_goal_interpolant": float(pack["eta_exact_goal_interpolant"]),
        "eta_nonzero_time_weight_candidate": float(
            pack["eta_nonzero_time_weight_candidate"]
        ),
        "eta_all_selector_weight_candidate": float(
            pack["eta_all_selector_weight_candidate"]
        ),
        "eta_nonzero_time_minus_all_selector": float(
            pack["eta_nonzero_time_minus_all_selector"]
        ),
        "alpha_nonzero_time_weight_candidate": float(
            pack["alpha_nonzero_time_weight_candidate"]
        ),
        "alpha_nonzero_time_rel_error_vs_exact_goal": float(
            pack["alpha_nonzero_time_rel_error_vs_exact_goal"]
        ),
        "alpha_all_selector_weight_candidate": float(
            pack["alpha_all_selector_weight_candidate"]
        ),
        "alpha_all_selector_rel_error_vs_exact_goal": float(
            pack["alpha_all_selector_rel_error_vs_exact_goal"]
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
        "8.7.56.5877",
        STEP_NAME + " declaration gate",
        {
            "source_files": {"prior_gate": sign_base.display_path(PRIOR_GATE)},
            "formulae": build_formulae(),
        },
        rows,
        summary,
        {
            "overall_status": "trial2_external_probe_weight_theorem_source_negative_closeout",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {
            "pack": pack,
        },
    )
    outputs = write_artifact("declaration_gate", payload)
    print("[done] 8.7.56.5875-5878 external-probe weight theorem-source audit completed")
    print(f"[done] declaration: {outputs['json']}")


if __name__ == "__main__":
    main()
