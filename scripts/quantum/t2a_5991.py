#!/usr/bin/env python3
"""Generate 8.7.56.5991-.5994 Hydrogen-only watch refresh artifacts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.quantum.t2a_1843 as sign_base
from scripts.quantum.trial2_hydrogen_only_watch_gate_refresh_backend import (
    build_trial2_hydrogen_only_watch_gate_refresh_pack,
)
from scripts.utils.windows_length_policy import build_compact_artifact_stem
from scripts.utils.windows_length_policy import build_metrics_paths


PUBLIC_OUT = ROOT / "output" / "public" / "quantum"
PRIOR_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5987-5990",
        "updated_pack_trial2_non_hydrogen_surface_gate_refresh",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5991-5994"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack Trial-2 "
    "Hydrogen-only watch gate refresh"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_trial2_hydrogen_only_watch_gate_refresh",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_non_hydrogen_surface_unavailable_completed_"
    "hydrogen_only_watch_primary_conditional_reopen_secondary_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_non_hydrogen_surface_unavailable_completed_"
    "hydrogen_only_codata_lead_watch_retained_conditional_reopen_only_next"
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


# 関数: `.5991-.5994` の rule bundle を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas / rules fixed by the Hydrogen-only watch refresh."""
    return {
        "aggregate_rule": (
            "retain the three-surface CODATA-lead aggregate exactly as is while "
            "the actual rerun table remains Hydrogen-only"
        ),
        "watch_rule": (
            "Hydrogen-only watch is honest when CODATA sweeps the actual table but "
            "no non-Hydrogen alpha-explicit surface exists"
        ),
        "reopen_rule": (
            "no unconditional next official branch; reopen only through a genuinely "
            "new non-Hydrogen surface, deeper hyperfine correction, or new selected extension"
        ),
    }


# 関数: `.5991-.5994` を実行する。

def main() -> None:
    """Execute the Hydrogen-only watch gate refresh."""
    sign_base.require(PRIOR_GATE)

    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    pack = build_trial2_hydrogen_only_watch_gate_refresh_pack()
    summary_pack = pack["summary"]

    route_selected = (
        str(prior_summary["trial2_numeric_alpha_problem_classification"]) == PRIOR_CLASS
    )
    codata_sweep = bool(summary_pack["codata_sweep_verdict_now"])
    non_hydrogen_unavailable = int(summary_pack["non_hydrogen_actual_surface_count_now"]) == 0
    watch_retained = bool(summary_pack["hydrogen_only_watch_retained_now"])
    no_unconditional_next = bool(summary_pack["no_unconditional_next_official_branch_now"])

    rows = [
        sign_base.row(
            "updated_pack_trial2_non_hydrogen_gate_selected_now",
            "pass" if route_selected else "reject",
            "updated-pack Trial-2 non-Hydrogen gate selected now",
            sign_base.truth(route_selected),
            "The final refresh starts only after the non-Hydrogen surface gate is cut.",
        ),
        sign_base.row(
            "trial2_three_surface_codata_sweep_retained_now",
            "pass" if codata_sweep else "reject",
            "Trial-2 three-surface CODATA sweep retained now",
            sign_base.truth(codata_sweep),
            "The current actual three-surface table still leans to alpha_CODATA on every retained Hydrogen surface.",
        ),
        sign_base.row(
            "trial2_non_hydrogen_actual_surface_still_unavailable_now",
            "pass" if non_hydrogen_unavailable else "reject",
            "Trial-2 non-Hydrogen actual surface still unavailable now",
            sign_base.truth(non_hydrogen_unavailable),
            "No Helium, Lamb, or weak-sector route currently enters the actual rerun table.",
        ),
        sign_base.row(
            "trial2_hydrogen_only_codata_lead_watch_retained_now",
            "pass" if watch_retained else "reject",
            "Trial-2 Hydrogen-only CODATA-lead watch retained now",
            sign_base.truth(watch_retained),
            "The honest verdict stays watch rather than pass/reject because the actual table is still Hydrogen-only.",
        ),
        sign_base.row(
            "trial2_no_unconditional_next_official_branch_now",
            "pass" if no_unconditional_next else "reject",
            "Trial-2 no unconditional next official branch now",
            sign_base.truth(no_unconditional_next),
            "Current pack re-enters conditional reopen only.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "hydrogen_actual_surface_count_now": int(summary_pack["hydrogen_actual_surface_count_now"]),
        "pmodel_win_count_now": int(summary_pack["pmodel_win_count_now"]),
        "codata_win_count_now": int(summary_pack["codata_win_count_now"]),
        "codata_sweep_verdict_now": bool(summary_pack["codata_sweep_verdict_now"]),
        "non_hydrogen_actual_surface_count_now": int(
            summary_pack["non_hydrogen_actual_surface_count_now"]
        ),
        "hydrogen_only_watch_retained_now": bool(
            summary_pack["hydrogen_only_watch_retained_now"]
        ),
        "recommended_next_route_or_none": "none",
        "selected_next_generation_route": "conditional_reopen_only",
        "selected_followup_route": (
            "new_non_hydrogen_surface_or_full_hyperfine_precision_or_"
            "new_lamb_helium_weak_formula_or_selected_extension"
        ),
        "selected_followup_route_or_none": "conditional",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5993",
        STEP_NAME + " declaration gate",
        {
            "source_files": {"prior_gate": sign_base.display_path(PRIOR_GATE)},
            "formulae": build_formulae(),
        },
        rows,
        summary,
        {
            "overall_status": "trial2_hydrogen_only_codata_lead_watch_retained",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {
            "hydrogen_actual_surface_count_now": int(summary_pack["hydrogen_actual_surface_count_now"]),
            "non_hydrogen_actual_surface_count_now": int(
                summary_pack["non_hydrogen_actual_surface_count_now"]
            ),
            "codata_win_count_now": int(summary_pack["codata_win_count_now"]),
        },
    )
    artifacts = write_artifact("declaration_gate", payload)
    print("[ok] hydrogen-only watch gate:", artifacts["json"])


if __name__ == "__main__":
    main()
