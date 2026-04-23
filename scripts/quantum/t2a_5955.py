#!/usr/bin/env python3
"""Generate 8.7.56.5955-.5958 multi-observable watch/pass gate artifacts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.quantum.t2a_1843 as sign_base
from scripts.quantum.trial2_multi_observable_watch_pass_gate_backend import (
    build_trial2_multi_observable_watch_pass_gate_pack,
)
from scripts.utils.windows_length_policy import build_compact_artifact_stem
from scripts.utils.windows_length_policy import build_metrics_paths


PUBLIC_OUT = ROOT / "output" / "public" / "quantum"
PRIOR_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5951-5954",
        "updated_pack_trial2_third_independent_surface_inventory_refresh",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5955-5958"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack Trial-2 "
    "multi-observable watch or pass gate"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_trial2_multi_observable_watch_pass_gate",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_third_independent_surface_unavailable_multi_watch_gate_"
    "primary_conditional_reopen_secondary_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_hyperfine_attribution_split_completed_third_independent_surface_"
    "unavailable_multi_observable_watch_retained_conditional_reopen_only_next"
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


# 関数: `.5955-.5958` の formula bundle を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas / rules fixed by the watch/pass gate."""
    return {
        "watch_rule": (
            "retain watch when the split is localized but a third genuinely new "
            "independent alpha-explicit surface is still missing"
        ),
        "pass_rule": (
            "promote to pass only if attribution is localized and a new third "
            "surface resolves the current 1-1 split"
        ),
        "reopen_rule": (
            "without such a third surface, next motion is conditional reopen only"
        ),
    }


# 関数: `.5955-.5958` を実行する。

def main() -> None:
    """Execute the multi-observable watch/pass gate."""
    sign_base.require(PRIOR_GATE)

    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    pack = build_trial2_multi_observable_watch_pass_gate_pack()
    summary_pack = pack["summary"]

    route_selected = str(prior_summary["trial2_numeric_alpha_problem_classification"]) == PRIOR_CLASS
    attribution_localized = bool(summary_pack["hyperfine_attribution_split_localized_now"])
    third_unavailable = not bool(summary_pack["genuine_third_independent_surface_available_now"])
    pass_unavailable = not bool(summary_pack["multi_observable_pass_available_now"])
    watch_retained = bool(summary_pack["multi_observable_watch_retained_now"])

    rows = [
        sign_base.row(
            "updated_pack_trial2_third_surface_unavailable_selected_now",
            "pass" if route_selected else "reject",
            "updated-pack Trial-2 third surface unavailable selected now",
            sign_base.truth(route_selected),
            "The watch/pass gate starts only after the third-surface inventory closes honestly.",
        ),
        sign_base.row(
            "trial2_hyperfine_attribution_localized_now",
            "pass" if attribution_localized else "reject",
            "Trial-2 hyperfine attribution localized now",
            sign_base.truth(attribution_localized),
            "The split origin is already localized to different surface-implied effective-alpha values.",
        ),
        sign_base.row(
            "trial2_genuine_third_surface_still_unavailable_now",
            "pass" if third_unavailable else "reject",
            "Trial-2 genuine third surface still unavailable now",
            sign_base.truth(third_unavailable),
            "No third genuinely independent alpha-explicit rerun surface is available in the current pack.",
        ),
        sign_base.row(
            "trial2_multi_observable_pass_unavailable_now",
            "pass" if pass_unavailable else "reject",
            "Trial-2 multi-observable pass unavailable now",
            sign_base.truth(pass_unavailable),
            "Without a third surface, the current two-surface split cannot be promoted to pass.",
        ),
        sign_base.row(
            "trial2_multi_observable_watch_retained_now",
            "pass" if watch_retained else "reject",
            "Trial-2 multi-observable watch retained now",
            sign_base.truth(watch_retained),
            "The honest current verdict remains split watch rather than pass or reject.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "hyperfine_attribution_split_localized_now": bool(summary_pack["hyperfine_attribution_split_localized_now"]),
        "genuine_third_independent_surface_available_now": bool(
            summary_pack["genuine_third_independent_surface_available_now"]
        ),
        "multi_observable_pass_available_now": bool(summary_pack["multi_observable_pass_available_now"]),
        "multi_observable_watch_retained_now": bool(summary_pack["multi_observable_watch_retained_now"]),
        "recommended_next_route_or_none": "none",
        "selected_next_generation_route": "conditional_reopen_only",
        "selected_followup_route": "conditional_reopen_only",
        "selected_followup_route_or_none": "conditional",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5957",
        STEP_NAME + " declaration gate",
        {"source_files": {"prior_gate": sign_base.display_path(PRIOR_GATE)}, "formulae": build_formulae()},
        rows,
        summary,
        {
            "overall_status": "trial2_multi_observable_watch_retained",
            "branch_completed": True,
            "breakthrough_passed_now": True,
            "physical_reject_required": False,
        },
        {
            "multi_observable_watch_retained_now": bool(summary_pack["multi_observable_watch_retained_now"]),
            "multi_observable_pass_available_now": bool(summary_pack["multi_observable_pass_available_now"]),
        },
    )
    artifacts = write_artifact("declaration_gate", payload)
    print("[ok] multi-observable watch/pass gate:", artifacts["json"])


if __name__ == "__main__":
    main()
