#!/usr/bin/env python3
"""Generate 8.7.56.6039-.6042 native He II family-strength artifacts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.quantum.t2a_1843 as sign_base
from scripts.quantum.trial2_native_helium_ion_family_strength_audit_backend import (
    build_trial2_native_helium_ion_family_strength_audit_pack,
)
from scripts.utils.windows_length_policy import build_compact_artifact_stem
from scripts.utils.windows_length_policy import build_metrics_paths


PUBLIC_OUT = ROOT / "output" / "public" / "quantum"

STEP_TAG = "8.7.56.6039-6042"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor native helium-ion "
    "family strength audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "trial2_native_helium_ion_family_strength_audit",
    prefix="q",
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_native_heii_same_family_negative_closeout_completed_post_heii_"
    "watch_gate_primary_precision_reserve_secondary_next"
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


# 関数: `.6039-.6042` の formula bundle を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas / rules fixed by the He II family audit."""
    return {
        "family_rule": "all retained He II lines stay inside the same hydrogenic gross-structure alpha^2 family",
        "selection_rule": "infer n-pairs from the CODATA row only to classify the retained family, not to tune alpha_P",
        "closeout_rule": "if no retained He II line beats the selected 468.67 nm route on the same native shell, same-family replay is exhausted",
    }


# 関数: `.6039-.6042` を実行する。

def main() -> None:
    """Execute the native He II family-strength audit."""
    pack = build_trial2_native_helium_ion_family_strength_audit_pack()
    summary_pack = pack["summary"]

    rows = [
        sign_base.row(
            "trial2_heii_family_line_count_now",
            "pass" if int(summary_pack["heii_family_line_count_now"]) >= 3 else "reject",
            "Trial-2 He II family line count now",
            float(summary_pack["heii_family_line_count_now"]),
            "The retained He II cache already exposes three one-electron hydrogenic lines.",
        ),
        sign_base.row(
            "trial2_heii_family_strongest_line_is_46867_now",
            "pass" if summary_pack["heii_family_strongest_pmodel_line_id_now"] == "He_II_468.67nm" else "reject",
            "Trial-2 He II family strongest line is 468.67 nm now",
            1.0 if summary_pack["heii_family_strongest_pmodel_line_id_now"] == "He_II_468.67nm" else 0.0,
            "No retained He II line improves on the already selected 468.67 nm route under the same native shell.",
        ),
        sign_base.row(
            "trial2_heii_family_stronger_same_family_route_available_now",
            "pass" if summary_pack["heii_family_stronger_than_46867_route_available_now"] else "reject",
            "Trial-2 He II family stronger same-family route available now",
            sign_base.truth(summary_pack["heii_family_stronger_than_46867_route_available_now"]),
            "A false value means same-family He II replay is exhausted.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "heii_family_line_count_now": int(summary_pack["heii_family_line_count_now"]),
        "heii_family_codata_win_count_now": int(summary_pack["heii_family_codata_win_count_now"]),
        "heii_family_pmodel_win_count_now": int(summary_pack["heii_family_pmodel_win_count_now"]),
        "heii_family_strongest_pmodel_line_id_now": str(summary_pack["heii_family_strongest_pmodel_line_id_now"]),
        "heii_family_stronger_than_46867_route_available_now": bool(
            summary_pack["heii_family_stronger_than_46867_route_available_now"]
        ),
        "selected_next_generation_route": "trial2_native_post_heii_watch_gate",
        "recommended_next_route_or_none": ".6043-.6046",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.6041",
        STEP_NAME + " declaration gate",
        {"formulae": build_formulae()},
        rows,
        summary,
        {
            "overall_status": "trial2_native_heii_same_family_negative_closeout",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {
            "heii_family_strongest_pmodel_relative_error_vs_observed_now": float(
                summary_pack["heii_family_strongest_pmodel_relative_error_vs_observed_now"]
            ),
        },
    )
    artifacts = write_artifact("declaration_gate", payload)
    print("[ok] native He II family strength audit:", artifacts["json"])


if __name__ == "__main__":
    main()
