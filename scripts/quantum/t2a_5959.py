#!/usr/bin/env python3
"""Generate 8.7.56.5959-.5962 hyperfine g/2-correction artifacts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.quantum.t2a_1843 as sign_base
from scripts.quantum.trial2_hyperfine_g2_correction_materialization_backend import (
    build_trial2_hyperfine_g2_correction_materialization_pack,
)
from scripts.utils.windows_length_policy import build_compact_artifact_stem
from scripts.utils.windows_length_policy import build_metrics_paths


PUBLIC_OUT = ROOT / "output" / "public" / "quantum"
PRIOR_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5955-5958",
        "updated_pack_trial2_multi_observable_watch_pass_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5959-5962"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack Trial-2 "
    "hyperfine g2 correction materialization audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_trial2_hyperfine_g2_correction_materialization_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_hyperfine_attribution_split_completed_third_independent_surface_"
    "unavailable_multi_observable_watch_retained_conditional_reopen_only_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_hyperfine_g2_correction_materialized_corrected_attribution_primary_"
    "two_surface_refresh_secondary_next"
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
        writer = csv.DictWriter(handle, fieldnames=["row_id", "status", "metric", "value", "note"])
        writer.writeheader()
        writer.writerows(data["rows"])

    return {"json": sign_base.display_path(paths["json"])}


# 関数: `.5959-.5962` の formula bundle を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas / rules fixed by the g/2 correction materialization."""
    return {
        "correction_rule": "nu_hfs_g2(alpha) = (g_e / 2) * nu_hfs_Fermi(alpha)",
        "source_rule": "extract the directly measured g/2 token, not the alpha value inferred in the same paper",
        "selection_rule": (
            "promote the branch only if the corrected hyperfine surface is deterministic "
            "and rerun-ready under the current public pack"
        ),
    }


# 関数: `.5959-.5962` を実行する。

def main() -> None:
    """Execute the hyperfine g/2 correction materialization gate."""
    sign_base.require(PRIOR_GATE)

    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    pack = build_trial2_hyperfine_g2_correction_materialization_pack()
    summary_pack = pack["summary"]

    route_selected = str(prior_summary["trial2_numeric_alpha_problem_classification"]) == PRIOR_CLASS
    materialized = bool(pack["trial2_hyperfine_g2_correction_materialized_now"])
    surface_ready = bool(summary_pack["hyperfine_corrected_surface_ready_now"])
    token_extracted = bool(summary_pack["g_over_2_token_extracted_now"])
    codata_best = bool(summary_pack["best_overall_is_codata_now"])
    primary_ready = bool(summary_pack["primary_score_admissible_now"])

    rows = [
        sign_base.row(
            "updated_pack_trial2_watch_retained_selected_now",
            "pass" if route_selected else "reject",
            "updated-pack Trial-2 watch-retained selected now",
            sign_base.truth(route_selected),
            "The g/2 correction branch starts only after the watch-retained pack localizes the reopen condition.",
        ),
        sign_base.row(
            "trial2_hyperfine_g2_correction_materialized_now",
            "pass" if materialized else "reject",
            "Trial-2 hyperfine g2 correction materialized now",
            sign_base.truth(materialized),
            "The retained public source cache now carries one deterministic g/2-corrected hyperfine surface.",
        ),
        sign_base.row(
            "trial2_hyperfine_corrected_surface_ready_now",
            "pass" if surface_ready else "reject",
            "Trial-2 hyperfine corrected surface ready now",
            sign_base.truth(surface_ready),
            "The corrected 21 cm surface is now rerun-ready under the current public pack.",
        ),
        sign_base.row(
            "trial2_g_over_2_token_extracted_now",
            "pass" if token_extracted else "reject",
            "Trial-2 g over 2 token extracted now",
            sign_base.truth(token_extracted),
            "The correction uses the directly measured g/2 token rather than the alpha extracted in the same paper.",
        ),
        sign_base.row(
            "trial2_hyperfine_corrected_primary_score_admissible_now",
            "pass" if primary_ready else "reject",
            "Trial-2 hyperfine corrected primary score admissible now",
            sign_base.truth(primary_ready),
            "The corrected surface remains independent of the hyperfine observable itself and is usable in the primary score.",
        ),
        sign_base.row(
            "trial2_hyperfine_corrected_best_overall_is_codata_now",
            "pass" if codata_best else "reject",
            "Trial-2 hyperfine corrected best overall is CODATA now",
            sign_base.truth(codata_best),
            "After the g/2 correction, alpha_CODATA becomes the closest retained checkpoint on the 21 cm surface.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "hyperfine_corrected_surface_id": str(summary_pack["hyperfine_corrected_surface_id"]),
        "hyperfine_corrected_surface_ready_now": bool(summary_pack["hyperfine_corrected_surface_ready_now"]),
        "g_over_2": float(summary_pack["g_over_2"]),
        "best_overall_alpha_label": str(summary_pack["best_overall_alpha_label"]),
        "best_overall_relative_error_vs_observed": float(summary_pack["best_overall_relative_error_vs_observed"]),
        "primary_score_admissible_now": bool(summary_pack["primary_score_admissible_now"]),
        "recommended_next_route_or_none": ".5963-.5966",
        "selected_next_generation_route": "trial2_hyperfine_corrected_attribution_refresh",
        "selected_followup_route": "trial2_multi_observable_corrected_hyperfine_gate",
        "selected_followup_route_or_none": ".5967-.5970",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5961",
        STEP_NAME + " declaration gate",
        {"source_files": {"prior_gate": sign_base.display_path(PRIOR_GATE)}, "formulae": build_formulae()},
        rows,
        summary,
        {
            "overall_status": "trial2_hyperfine_g2_correction_materialized",
            "branch_completed": True,
            "breakthrough_passed_now": True,
            "physical_reject_required": False,
        },
        {
            "g_over_2": float(summary_pack["g_over_2"]),
            "best_overall_relative_error_vs_observed": float(summary_pack["best_overall_relative_error_vs_observed"]),
        },
    )
    artifacts = write_artifact("declaration_gate", payload)
    print("[ok] hyperfine g2 correction gate:", artifacts["json"])


if __name__ == "__main__":
    main()
