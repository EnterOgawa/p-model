#!/usr/bin/env python3
"""Generate 8.7.56.6047-.6050 native He I simple-screening audit artifacts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.quantum.t2a_1843 as sign_base
from scripts.quantum.trial2_native_helium_simple_screening_audit_backend import (
    build_trial2_native_helium_simple_screening_audit_pack,
)
from scripts.utils.windows_length_policy import build_compact_artifact_stem
from scripts.utils.windows_length_policy import build_metrics_paths


PUBLIC_OUT = ROOT / "output" / "public" / "quantum"

STEP_TAG = "8.7.56.6047-6050"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor native helium simple "
    "screening audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "trial2_native_helium_simple_screening_audit",
    prefix="q",
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_native_helium_simple_screening_negative_closeout_completed_local_"
    "non_hydrogen_gate_primary_new_family_reserve_secondary_next"
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


# 関数: `.6047-.6050` の formula bundle を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas / rules fixed by the He I simple-screening audit."""
    return {
        "surrogate_rule": "scan one constant Z_eff surrogate on top of the already-public reduced-mass Coulomb shell",
        "physical_rule": "neutral-helium simple screening is physically admissible only if the inferred constant Z_eff does not exceed the nuclear charge ceiling Z=2",
        "closeout_rule": "if the strongest physical simple-screening surrogate still needs an inferred pair map and keeps a material residual, the He I route remains non-native",
    }


# 関数: `.6047-.6050` を実行する。

def main() -> None:
    """Execute the native He I simple-screening audit."""
    pack = build_trial2_native_helium_simple_screening_audit_pack()
    summary_pack = pack["summary"]

    rows = [
        sign_base.row(
            "trial2_helium_unrestricted_physical_admissible_now",
            "pass" if summary_pack["unrestricted_physical_admissible_now"] else "reject",
            "Trial-2 He I unrestricted simple-screening surrogate physically admissible now",
            sign_base.truth(summary_pack["unrestricted_physical_admissible_now"]),
            "The strongest unrestricted constant-Z_eff surrogate is not honest if it already requires Z_eff > 2.",
        ),
        sign_base.row(
            "trial2_helium_physical_simple_screening_subpercent_now",
            "pass" if summary_pack["physical_subpercent_now"] else "reject",
            "Trial-2 He I physical simple-screening surrogate subpercent now",
            sign_base.truth(summary_pack["physical_subpercent_now"]),
            "A physical simple-screening surrogate must keep the fitted residual comfortably below the few-percent regime.",
        ),
        sign_base.row(
            "trial2_helium_simple_screening_surface_ready_now",
            "pass" if summary_pack["helium_simple_screening_surface_ready_now"] else "reject",
            "Trial-2 He I simple-screening surface ready now",
            sign_base.truth(summary_pack["helium_simple_screening_surface_ready_now"]),
            "Without an admissible constant-screening surrogate, the retained He I cache cannot enter the native primary table.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "helium_selected_line_count_now": int(summary_pack["helium_selected_line_count_now"]),
        "unrestricted_constant_zeff_now": float(summary_pack["unrestricted_constant_zeff_now"]),
        "unrestricted_max_relative_residual_now": float(summary_pack["unrestricted_max_relative_residual_now"]),
        "unrestricted_relative_spread_now": float(summary_pack["unrestricted_relative_spread_now"]),
        "unrestricted_physical_admissible_now": bool(summary_pack["unrestricted_physical_admissible_now"]),
        "physical_constant_zeff_now": float(summary_pack["physical_constant_zeff_now"]),
        "physical_max_relative_residual_now": float(summary_pack["physical_max_relative_residual_now"]),
        "physical_relative_spread_now": float(summary_pack["physical_relative_spread_now"]),
        "physical_subpercent_now": bool(summary_pack["physical_subpercent_now"]),
        "helium_simple_screening_surface_ready_now": bool(summary_pack["helium_simple_screening_surface_ready_now"]),
        "selected_next_generation_route": "trial2_native_local_nonhydrogen_gate",
        "recommended_next_route_or_none": ".6051-.6054",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.6049",
        STEP_NAME + " declaration gate",
        {"formulae": build_formulae()},
        rows,
        summary,
        {
            "overall_status": "trial2_native_helium_simple_screening_negative_closeout",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {
            "unrestricted_constant_zeff_now": float(summary_pack["unrestricted_constant_zeff_now"]),
            "physical_constant_zeff_now": float(summary_pack["physical_constant_zeff_now"]),
        },
    )
    artifacts = write_artifact("declaration_gate", payload)
    print("[ok] native He I simple-screening audit:", artifacts["json"])


if __name__ == "__main__":
    main()
