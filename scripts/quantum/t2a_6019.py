#!/usr/bin/env python3
"""Generate 8.7.56.6019-.6022 native third-surface gate artifacts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.quantum.t2a_1843 as sign_base
from scripts.quantum.trial2_native_third_surface_gate_backend import (
    build_trial2_native_third_surface_gate_pack,
)
from scripts.utils.windows_length_policy import build_compact_artifact_stem
from scripts.utils.windows_length_policy import build_metrics_paths


PUBLIC_OUT = ROOT / "output" / "public" / "quantum"

STEP_TAG = "8.7.56.6019-6022"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor native third-surface gate"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "trial2_native_third_surface_gate",
    prefix="q",
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_native_third_surface_negative_closeout_completed_native_two_surface_"
    "split_watch_primary_native_watch_gate_secondary_next"
)


# 関数: JSON/CSV artifact を書き出す。
def write_artifact(kind: str, data: dict) -> dict[str, str]:
    """Write one JSON payload and one rows CSV."""
    PUBLIC_OUT.mkdir(parents=True, exist_ok=True)
    paths = build_metrics_paths(PUBLIC_OUT, STEM, kind)
    paths["json"].write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with paths["csv"].open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["row_id", "status", "metric", "value", "note"])
        writer.writeheader()
        writer.writerows(data["rows"])

    return {"json": sign_base.display_path(paths["json"])}


# 関数: `.6019-.6022` の formula bundle を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas / rules fixed by the native third-surface gate."""
    return {
        "native_table_rule": "primary table may count only native actual surfaces under the absolute condition",
        "third_surface_rule": "a third surface is native only if its relativistic bridge is public-canonical",
        "watch_rule": "retain split watch when native wins and CODATA wins are tied on the actual native table",
    }


# 関数: `.6019-.6022` を実行する。

def main() -> None:
    """Execute the native third-surface gate."""
    pack = build_trial2_native_third_surface_gate_pack()
    summary_pack = pack["summary"]

    rows = [
        sign_base.row(
            "trial2_native_actual_surface_count_is_two_now",
            "pass" if int(summary_pack["native_actual_surface_count_now"]) == 2 else "reject",
            "Trial-2 native actual surface count is two now",
            float(summary_pack["native_actual_surface_count_now"]),
            "Only 1S-2S and tree-level 21 cm remain in the native primary table.",
        ),
        sign_base.row(
            "trial2_native_genuine_third_surface_available_now",
            "pass" if summary_pack["native_genuine_third_surface_available_now"] else "reject",
            "Trial-2 native genuine third surface available now",
            sign_base.truth(summary_pack["native_genuine_third_surface_available_now"]),
            "The retained Halpha diagnostic cannot count as native until its relativistic bridge materializes.",
        ),
        sign_base.row(
            "trial2_native_split_watch_retained_now",
            "pass" if summary_pack["native_split_watch_retained_now"] else "reject",
            "Trial-2 native split watch retained now",
            sign_base.truth(summary_pack["native_split_watch_retained_now"]),
            "The current native table still gives one CODATA-side surface and one P-model-side surface.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "native_actual_surface_count_now": int(summary_pack["native_actual_surface_count_now"]),
        "native_surface_ids_now": list(summary_pack["native_surface_ids_now"]),
        "native_pmodel_win_count_now": int(summary_pack["native_pmodel_win_count_now"]),
        "native_codata_win_count_now": int(summary_pack["native_codata_win_count_now"]),
        "native_genuine_third_surface_available_now": bool(
            summary_pack["native_genuine_third_surface_available_now"]
        ),
        "native_split_watch_retained_now": bool(summary_pack["native_split_watch_retained_now"]),
        "selected_next_generation_route": "trial2_native_multi_observable_watch_pass_gate",
        "recommended_next_route_or_none": ".6023-.6026",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.6021",
        STEP_NAME + " declaration gate",
        {"formulae": build_formulae()},
        rows,
        summary,
        {
            "overall_status": "trial2_native_third_surface_negative_closeout",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {
            "native_pmodel_win_count_now": int(summary_pack["native_pmodel_win_count_now"]),
            "native_codata_win_count_now": int(summary_pack["native_codata_win_count_now"]),
        },
    )
    artifacts = write_artifact("declaration_gate", payload)
    print("[ok] native third-surface gate:", artifacts["json"])


if __name__ == "__main__":
    main()
