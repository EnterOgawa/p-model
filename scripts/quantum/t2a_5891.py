#!/usr/bin/env python3
"""Generate 8.7.56.5891-.5894 independent-observable filter artifacts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.quantum.t2a_1843 as sign_base
from scripts.quantum.trial2_alpha_observable_sensitivity_inventory_backend import (
    build_trial2_alpha_observable_sensitivity_inventory_pack,
)
from scripts.utils.windows_length_policy import build_compact_artifact_stem
from scripts.utils.windows_length_policy import build_metrics_paths


PUBLIC_OUT = ROOT / "output" / "public" / "quantum"
PRIOR_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5887-5890",
        "updated_pack_trial2_alpha_observable_sensitivity_inventory_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5891-5894"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack Trial-2 "
    "independent observable filter / exclusion gate"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_trial2_independent_observable_filter_gate",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_alpha_observable_sensitivity_inventory_audited_"
    "independent_filter_gate_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_alpha_observable_inventory_filter_completed_"
    "primary_rerun_gate_next"
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


# 関数: exclusion gate で固定する rule bundle を返す。

def build_formulae() -> dict[str, str]:
    """Return exclusion-gate rules."""
    return {
        "primary_exclusion_rule": (
            "exclude observables that directly reuse CODATA-input style alpha extraction "
            "such as electron g-2 and atom-recoil alpha determination"
        ),
        "primary_admission_rule": (
            "primary score requires independence, explicit alpha leverage, and current rerun readiness"
        ),
        "reserve_rule": "excluded explicit-alpha surfaces remain reserve diagnostics rather than being discarded",
    }


# 関数: `.5891-.5894` を実行する。

def main() -> None:
    """Execute the independent observable filter / exclusion gate."""
    sign_base.require(PRIOR_GATE)

    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    pack = build_trial2_alpha_observable_sensitivity_inventory_pack()
    candidates = list(pack["candidates"])

    route_selected = (
        str(prior_summary["trial2_numeric_alpha_problem_classification"]) == PRIOR_CLASS
    )
    excluded_rows = [row for row in candidates if row["codata_input_overlap_now"]]
    primary_ready_rows = [row for row in candidates if row["primary_score_admissible_now"]]
    independent_rows = [row for row in candidates if not row["codata_input_overlap_now"]]

    exclusion_gate_complete = bool(route_selected and excluded_rows)
    de_broglie_excluded = any(
        row["observable_id"] == "de_broglie_alpha_consistency" for row in excluded_rows
    )
    primary_ready_still_unavailable = len(primary_ready_rows) == 0
    independent_rows_retained = len(independent_rows) > 0
    rerun_gate_required = bool(primary_ready_still_unavailable and independent_rows_retained)

    rows = [
        sign_base.row(
            "updated_pack_trial2_alpha_observable_sensitivity_inventory_selected_now",
            "pass" if route_selected else "reject",
            "updated-pack Trial-2 alpha observable sensitivity inventory selected now",
            sign_base.truth(route_selected),
            "The exclusion gate only starts after the observable inventory has fixed which current surfaces actually carry alpha and which do not.",
        ),
        sign_base.row(
            "trial2_alpha_exclusion_gate_complete_now",
            "pass" if exclusion_gate_complete else "reject",
            "Trial-2 alpha exclusion gate complete now",
            sign_base.truth(exclusion_gate_complete),
            "The primary scoreboard now separates CODATA-input-style explicit-alpha surfaces from genuinely independent observable candidates.",
        ),
        sign_base.row(
            "trial2_alpha_de_broglie_primary_excluded_now",
            "pass" if de_broglie_excluded else "reject",
            "Trial-2 alpha de Broglie primary excluded now",
            sign_base.truth(de_broglie_excluded),
            "The recoil-vs-g-2 alpha comparison is retained only as a reserve diagnostic because it reuses extraction-side alpha inputs.",
        ),
        sign_base.row(
            "trial2_alpha_primary_ready_still_unavailable_now",
            "pass" if primary_ready_still_unavailable else "reject",
            "Trial-2 alpha primary-ready still unavailable now",
            sign_base.truth(primary_ready_still_unavailable),
            "After applying the independence filter, no honest primary rerun surface remains available in the current public pack.",
        ),
        sign_base.row(
            "trial2_alpha_independent_candidate_rows_retained_now",
            "pass" if independent_rows_retained else "reject",
            "Trial-2 alpha independent candidate rows retained now",
            sign_base.truth(independent_rows_retained),
            "Independent observables are still present, so the blocker is not lack of targets but lack of explicit alpha-ready observable maps.",
        ),
        sign_base.row(
            "updated_pack_trial2_primary_observable_rerun_gate_required_now",
            "pass" if rerun_gate_required else "reject",
            "updated-pack Trial-2 primary observable rerun gate required now",
            sign_base.truth(rerun_gate_required),
            "The next honest blocker is to decide whether an actual top-priority rerun exists now or whether observable-map materialization must be promoted first.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "excluded_primary_count": int(len(excluded_rows)),
        "independent_candidate_count": int(len(independent_rows)),
        "primary_ready_count_after_filter": int(len(primary_ready_rows)),
        "selected_excluded_observable_ids": [
            str(row["observable_id"]) for row in excluded_rows
        ],
        "selected_independent_observable_ids": [
            str(row["observable_id"]) for row in independent_rows
        ],
        "selected_next_generation_route": "trial2_primary_observable_rerun_gate",
        "recommended_next_route_or_none": ".5895-.5898",
        "selected_followup_route": "trial2_primary_observable_rerun_gate",
        "selected_followup_route_or_none": ".5895-.5898",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5893",
        STEP_NAME + " declaration gate",
        {
            "source_files": {"prior_gate": sign_base.display_path(PRIOR_GATE)},
            "formulae": build_formulae(),
            "excluded_primary_count": int(len(excluded_rows)),
        },
        rows,
        summary,
        {
            "overall_status": "trial2_independent_observable_filter_completed",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {
            "excluded_primary_count": int(len(excluded_rows)),
            "independent_candidate_count": int(len(independent_rows)),
            "primary_ready_count_after_filter": int(len(primary_ready_rows)),
        },
    )
    artifacts = write_artifact("declaration_gate", payload)
    print("[ok] exclusion gate:", artifacts["json"])


if __name__ == "__main__":
    main()
