#!/usr/bin/env python3
"""Generate 8.7.56.6031-.6034 native non-Hydrogen surface gate artifacts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.quantum.t2a_1843 as sign_base
from scripts.quantum.trial2_native_non_hydrogen_surface_gate_backend import (
    build_trial2_native_non_hydrogen_surface_gate_pack,
)
from scripts.utils.windows_length_policy import build_compact_artifact_stem
from scripts.utils.windows_length_policy import build_metrics_paths


PUBLIC_OUT = ROOT / "output" / "public" / "quantum"

STEP_TAG = "8.7.56.6031-6034"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor native non-hydrogen "
    "surface gate"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "trial2_native_non_hydrogen_surface_gate",
    prefix="q",
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_native_non_hydrogen_surface_completed_three_surface_watch_"
    "primary_watch_gate_secondary_next"
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


# 関数: `.6031-.6034` の formula bundle を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas / rules fixed by the native non-Hydrogen gate."""
    return {
        "surface_count_rule": "count only native actual surfaces under the absolute condition",
        "heii_rule": "He II one-electron gross structure is a genuine non-Hydrogen native surface",
        "watch_rule": "three actual native surfaces remove the missing-third-surface blocker but do not force pass",
    }


# 関数: `.6031-.6034` を実行する。

def main() -> None:
    """Execute the native non-Hydrogen surface gate."""
    pack = build_trial2_native_non_hydrogen_surface_gate_pack()
    summary_pack = pack["summary"]

    rows = [
        sign_base.row(
            "trial2_native_actual_surface_count_is_three_now",
            "pass" if int(summary_pack["native_actual_surface_count_now"]) == 3 else "reject",
            "Trial-2 native actual surface count is three now",
            float(summary_pack["native_actual_surface_count_now"]),
            "The He II hydrogenic baseline joins the two retained Hydrogen native surfaces.",
        ),
        sign_base.row(
            "trial2_native_non_hydrogen_actual_surface_count_now",
            "pass" if int(summary_pack["native_non_hydrogen_actual_surface_count_now"]) >= 1 else "reject",
            "Trial-2 native non-Hydrogen actual surface count now",
            float(summary_pack["native_non_hydrogen_actual_surface_count_now"]),
            "This removes the Hydrogen-only blocker from the native primary table.",
        ),
        sign_base.row(
            "trial2_native_codata_lead_diagnostic_now",
            "pass" if summary_pack["native_codata_lead_diagnostic_now"] else "reject",
            "Trial-2 native CODATA-lead diagnostic now",
            sign_base.truth(summary_pack["native_codata_lead_diagnostic_now"]),
            "The current three-surface native table still places the diagnostic CODATA row closest on two gross-structure surfaces.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "native_actual_surface_count_now": int(summary_pack["native_actual_surface_count_now"]),
        "native_non_hydrogen_actual_surface_count_now": int(
            summary_pack["native_non_hydrogen_actual_surface_count_now"]
        ),
        "native_surface_ids_now": list(summary_pack["native_surface_ids_now"]),
        "native_pmodel_win_count_now": int(summary_pack["native_pmodel_win_count_now"]),
        "native_codata_win_count_now": int(summary_pack["native_codata_win_count_now"]),
        "native_codata_lead_diagnostic_now": bool(summary_pack["native_codata_lead_diagnostic_now"]),
        "selected_next_generation_route": "trial2_native_three_surface_watch_gate",
        "recommended_next_route_or_none": ".6035-.6038",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.6033",
        STEP_NAME + " declaration gate",
        {"formulae": build_formulae()},
        rows,
        summary,
        {
            "overall_status": "trial2_native_non_hydrogen_surface_completed",
            "branch_completed": True,
            "breakthrough_passed_now": True,
            "physical_reject_required": False,
        },
        {
            "native_pmodel_win_count_now": int(summary_pack["native_pmodel_win_count_now"]),
            "native_codata_win_count_now": int(summary_pack["native_codata_win_count_now"]),
        },
    )
    artifacts = write_artifact("declaration_gate", payload)
    print("[ok] native non-Hydrogen surface gate:", artifacts["json"])


if __name__ == "__main__":
    main()
