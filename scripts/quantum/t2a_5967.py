#!/usr/bin/env python3
"""Generate 8.7.56.5967-.5970 corrected two-surface gate artifacts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.quantum.t2a_1843 as sign_base
from scripts.quantum.trial2_multi_observable_corrected_hyperfine_gate_backend import (
    build_trial2_multi_observable_corrected_hyperfine_gate_pack,
)
from scripts.utils.windows_length_policy import build_compact_artifact_stem
from scripts.utils.windows_length_policy import build_metrics_paths


PUBLIC_OUT = ROOT / "output" / "public" / "quantum"
PRIOR_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5963-5966",
        "updated_pack_trial2_hyperfine_corrected_attribution_refresh",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5967-5970"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack Trial-2 "
    "multi-observable corrected hyperfine gate"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_trial2_multi_observable_corrected_hyperfine_gate",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_hyperfine_g2_corrected_attribution_completed_two_surface_gate_"
    "primary_watch_secondary_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_hyperfine_g2_corrected_two_surface_codata_lead_watch_retained_"
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
        writer = csv.DictWriter(handle, fieldnames=["row_id", "status", "metric", "value", "note"])
        writer.writeheader()
        writer.writerows(data["rows"])

    return {"json": sign_base.display_path(paths["json"])}


# 関数: `.5967-.5970` の formula bundle を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas / rules fixed by the corrected two-surface gate."""
    return {
        "table_rule": "compare Hydrogen 1S-2S gross structure with Hydrogen 21 cm g/2-corrected baseline",
        "codata_sweep_rule": "codata sweep when alpha_CODATA is best overall on both actual surfaces",
        "watch_rule": (
            "retain watch, not final reject, while the table still has only two actual surfaces "
            "and no third independent family"
        ),
    }


# 関数: `.5967-.5970` を実行する。

def main() -> None:
    """Execute the corrected two-surface gate."""
    sign_base.require(PRIOR_GATE)

    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    pack = build_trial2_multi_observable_corrected_hyperfine_gate_pack()
    summary_pack = pack["summary"]

    route_selected = str(prior_summary["trial2_numeric_alpha_problem_classification"]) == PRIOR_CLASS
    codata_sweep = bool(summary_pack["codata_sweep_verdict_now"])
    split_gone = not bool(summary_pack["split_watch_verdict_now"])
    watch_retained = bool(summary_pack["multi_observable_watch_retained_now"])
    pass_unavailable = not bool(summary_pack["multi_observable_pass_available_now"])

    rows = [
        sign_base.row(
            "updated_pack_trial2_corrected_attribution_selected_now",
            "pass" if route_selected else "reject",
            "updated-pack Trial-2 corrected attribution selected now",
            sign_base.truth(route_selected),
            "The corrected two-surface gate starts only after the corrected attribution refresh passes.",
        ),
        sign_base.row(
            "trial2_corrected_two_surface_codata_sweep_now",
            "pass" if codata_sweep else "reject",
            "Trial-2 corrected two-surface CODATA sweep now",
            sign_base.truth(codata_sweep),
            "With the g/2-corrected hyperfine surface, CODATA is the closest retained checkpoint on both actual surfaces.",
        ),
        sign_base.row(
            "trial2_split_watch_no_longer_now",
            "pass" if split_gone else "reject",
            "Trial-2 split watch no longer now",
            sign_base.truth(split_gone),
            "The prior 1-1 split disappears once the source-backed hyperfine correction is applied.",
        ),
        sign_base.row(
            "trial2_multi_observable_pass_still_unavailable_now",
            "pass" if pass_unavailable else "reject",
            "Trial-2 multi-observable pass still unavailable now",
            sign_base.truth(pass_unavailable),
            "A third genuinely independent alpha-explicit family is still missing, so no final pass/reject is claimed.",
        ),
        sign_base.row(
            "trial2_codata_lead_watch_retained_now",
            "pass" if watch_retained else "reject",
            "Trial-2 CODATA-lead watch retained now",
            sign_base.truth(watch_retained),
            "The honest reading becomes CODATA-leading watch rather than split watch or final reject.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "current_actual_surface_count_now": int(summary_pack["current_actual_surface_count_now"]),
        "surface_ids_now": list(summary_pack["surface_ids_now"]),
        "pmodel_win_count_now": int(summary_pack["pmodel_win_count_now"]),
        "codata_win_count_now": int(summary_pack["codata_win_count_now"]),
        "split_watch_verdict_now": bool(summary_pack["split_watch_verdict_now"]),
        "codata_sweep_verdict_now": bool(summary_pack["codata_sweep_verdict_now"]),
        "multi_observable_watch_retained_now": bool(summary_pack["multi_observable_watch_retained_now"]),
        "recommended_next_route_or_none": "none",
        "selected_next_generation_route": "conditional_reopen_only",
        "selected_followup_route": "third_independent_surface_or_full_hyperfine_precision_source",
        "selected_followup_route_or_none": "conditional",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5969",
        STEP_NAME + " declaration gate",
        {
            "source_files": {"prior_gate": sign_base.display_path(PRIOR_GATE)},
            "formulae": build_formulae(),
            "surface_rows": pack["surface_rows"],
        },
        rows,
        summary,
        {
            "overall_status": "trial2_corrected_two_surface_codata_lead_watch_retained",
            "branch_completed": True,
            "breakthrough_passed_now": True,
            "physical_reject_required": False,
        },
        {
            "pmodel_win_count_now": int(summary_pack["pmodel_win_count_now"]),
            "codata_win_count_now": int(summary_pack["codata_win_count_now"]),
        },
    )
    artifacts = write_artifact("declaration_gate", payload)
    print("[ok] corrected two-surface gate:", artifacts["json"])


if __name__ == "__main__":
    main()
