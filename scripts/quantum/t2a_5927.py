#!/usr/bin/env python3
"""Generate 8.7.56.5927-.5930 second-rerun gate artifacts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.quantum.t2a_1843 as sign_base
from scripts.quantum.trial2_second_independent_observable_rerun_gate_backend import (
    build_trial2_second_independent_rerun_gate_pack,
)
from scripts.utils.windows_length_policy import build_compact_artifact_stem
from scripts.utils.windows_length_policy import build_metrics_paths


PUBLIC_OUT = ROOT / "output" / "public" / "quantum"
PRIOR_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5923-5926",
        "updated_pack_trial2_lamb_absolute_alpha_formula_materialization_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5927-5930"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack Trial-2 "
    "second independent observable rerun gate"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_trial2_second_independent_observable_rerun_gate",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_lamb_absolute_formula_negative_closeout_completed_"
    "second_observable_primary_multi_observable_secondary_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_second_independent_observable_rerun_unavailable_completed_"
    "multi_observable_gate_primary_conditional_reopen_secondary_next"
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


# 関数: `.5927-.5930` の rule bundle を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas / rules fixed by the second-rerun gate."""
    return {
        "second_rerun_rule": (
            "a second rerun exists only if one surface beyond the first Hydrogen "
            "1S-2S baseline is independent, alpha explicit, and rerun ready"
        ),
        "lamb_rule": (
            "Lamb remains structurally retained but not rerun ready while its "
            "absolute alpha formula is unavailable"
        ),
        "weak_rule": (
            "weak beta-decay remains reserve while no public fine-structure-alpha "
            "input exists in the current pack"
        ),
    }


# 関数: `.5927-.5930` を実行する。

def main() -> None:
    """Execute the second independent-observable rerun gate."""
    sign_base.require(PRIOR_GATE)

    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    pack = build_trial2_second_independent_rerun_gate_pack()
    summary_pack = pack["summary"]

    route_selected = (
        str(prior_summary["trial2_numeric_alpha_problem_classification"]) == PRIOR_CLASS
    )
    second_unavailable = not bool(summary_pack["second_independent_observable_rerun_available_now"])
    only_one_surface = int(summary_pack["current_actual_surface_count_now"]) == 1
    lamb_unavailable = not bool(summary_pack["lamb_absolute_formula_ready_now"])
    weak_unavailable = not bool(summary_pack["weak_explicit_formula_ready_now"])

    rows = [
        sign_base.row(
            "updated_pack_trial2_lamb_negative_closeout_selected_now",
            "pass" if route_selected else "reject",
            "updated-pack Trial-2 Lamb negative closeout selected now",
            sign_base.truth(route_selected),
            "The second-rerun gate starts only after Lamb absolute-formula materialization closes negatively.",
        ),
        sign_base.row(
            "trial2_second_independent_observable_rerun_unavailable_now",
            "pass" if second_unavailable else "reject",
            "Trial-2 second independent observable rerun unavailable now",
            sign_base.truth(second_unavailable),
            "No second independent alpha-explicit rerun-ready surface exists in the current public pack.",
        ),
        sign_base.row(
            "trial2_current_actual_surface_count_is_one_now",
            "pass" if only_one_surface else "reject",
            "Trial-2 current actual surface count is one now",
            sign_base.truth(only_one_surface),
            "Hydrogen 1S-2S gross structure remains the lone actual rerun surface.",
        ),
        sign_base.row(
            "trial2_lamb_absolute_formula_still_unavailable_now",
            "pass" if lamb_unavailable else "reject",
            "Trial-2 Lamb absolute formula still unavailable now",
            sign_base.truth(lamb_unavailable),
            "Lamb cannot yet supply the missing second independent surface.",
        ),
        sign_base.row(
            "trial2_weak_explicit_formula_still_unavailable_now",
            "pass" if weak_unavailable else "reject",
            "Trial-2 weak explicit formula still unavailable now",
            sign_base.truth(weak_unavailable),
            "Weak beta-decay cannot yet supply the missing second independent surface.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "second_independent_observable_rerun_available_now": False,
        "current_actual_surface_count_now": int(summary_pack["current_actual_surface_count_now"]),
        "retained_only_surface_id": str(summary_pack["retained_only_surface_id"]),
        "selected_next_generation_route": "trial2_first_multi_observable_comparison_gate",
        "recommended_next_route_or_none": ".5931-.5934",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5929",
        STEP_NAME + " declaration gate",
        {
            "source_files": {"prior_gate": sign_base.display_path(PRIOR_GATE)},
            "formulae": build_formulae(),
            "retained_only_surface": pack["retained_only_surface"],
        },
        rows,
        summary,
        {
            "overall_status": "trial2_second_independent_observable_rerun_unavailable_completed",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {
            "current_actual_surface_count_now": int(summary_pack["current_actual_surface_count_now"]),
            "lamb_absolute_formula_ready_now": bool(summary_pack["lamb_absolute_formula_ready_now"]),
            "weak_explicit_formula_ready_now": bool(summary_pack["weak_explicit_formula_ready_now"]),
        },
    )
    artifacts = write_artifact("declaration_gate", payload)
    print("[ok] second rerun gate:", artifacts["json"])


if __name__ == "__main__":
    main()
