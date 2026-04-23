#!/usr/bin/env python3
"""Generate 8.7.56.6051-.6054 native local non-Hydrogen gate artifacts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.quantum.t2a_1843 as sign_base
from scripts.quantum.trial2_native_local_nonhydrogen_gate_backend import (
    build_trial2_native_local_nonhydrogen_gate_pack,
)
from scripts.utils.windows_length_policy import build_compact_artifact_stem
from scripts.utils.windows_length_policy import build_metrics_paths


PUBLIC_OUT = ROOT / "output" / "public" / "quantum"

STEP_TAG = "8.7.56.6051-6054"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor native local non-Hydrogen gate"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "trial2_native_local_nonhydrogen_gate",
    prefix="q",
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_native_three_surface_watch_retained_local_non_hydrogen_candidates_"
    "exhausted_conditional_reopen_only_next"
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


# 関数: `.6051-.6054` の formula bundle を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas / rules fixed by the local non-Hydrogen gate."""
    return {
        "watch_rule": "retain watch because the current native table still does not show pass-grade dominance",
        "local_nonhydrogen_rule": "the local non-Hydrogen pool is exhausted once He II same-family replay is cut and He I simple screening fails honestly",
        "next_rule": "further progress now requires a genuinely new relativistic bridge, precision correction, or non-Hydrogen family extension",
    }


# 関数: `.6051-.6054` を実行する。

def main() -> None:
    """Execute the native local non-Hydrogen gate."""
    pack = build_trial2_native_local_nonhydrogen_gate_pack()
    summary_pack = pack["summary"]

    rows = [
        sign_base.row(
            "trial2_stronger_local_nonhydrogen_route_available_now",
            "pass" if summary_pack["stronger_local_non_hydrogen_route_available_now"] else "reject",
            "Trial-2 stronger local non-Hydrogen route available now",
            sign_base.truth(summary_pack["stronger_local_non_hydrogen_route_available_now"]),
            "A false value means the retained local He I / He II candidate pool is exhausted.",
        ),
        sign_base.row(
            "trial2_native_multi_observable_watch_retained_now",
            "pass" if summary_pack["native_multi_observable_watch_retained_now"] else "reject",
            "Trial-2 native multi-observable watch retained now",
            sign_base.truth(summary_pack["native_multi_observable_watch_retained_now"]),
            "The aggregate native verdict remains watch after closing the local non-Hydrogen side routes.",
        ),
        sign_base.row(
            "trial2_no_unconditional_next_official_branch_now",
            "pass" if summary_pack["no_unconditional_next_official_branch_now"] else "reject",
            "Trial-2 no unconditional next official branch now",
            sign_base.truth(summary_pack["no_unconditional_next_official_branch_now"]),
            "Further progress now requires a genuinely new bridge, precision correction, or family extension.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "native_actual_surface_count_now": int(summary_pack["native_actual_surface_count_now"]),
        "native_pmodel_win_count_now": int(summary_pack["native_pmodel_win_count_now"]),
        "native_codata_win_count_now": int(summary_pack["native_codata_win_count_now"]),
        "native_multi_observable_watch_retained_now": bool(
            summary_pack["native_multi_observable_watch_retained_now"]
        ),
        "heii_family_stronger_than_46867_route_available_now": bool(
            summary_pack["heii_family_stronger_than_46867_route_available_now"]
        ),
        "helium_simple_screening_surface_ready_now": bool(
            summary_pack["helium_simple_screening_surface_ready_now"]
        ),
        "stronger_local_non_hydrogen_route_available_now": bool(
            summary_pack["stronger_local_non_hydrogen_route_available_now"]
        ),
        "no_unconditional_next_official_branch_now": bool(
            summary_pack["no_unconditional_next_official_branch_now"]
        ),
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.6053",
        STEP_NAME + " declaration gate",
        {"formulae": build_formulae()},
        rows,
        summary,
        {
            "overall_status": "trial2_native_three_surface_watch_retained_local_nonhydrogen_exhausted",
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
    print("[ok] native local non-Hydrogen gate:", artifacts["json"])


if __name__ == "__main__":
    main()
