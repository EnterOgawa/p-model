#!/usr/bin/env python3
"""Audit alpha-observable materialization inside the QED-vacuum baseline pack.

Purpose:
    Trial-2 mainline has moved from exact-constant collapse to observable-level
    comparison. The first honest question inside that new mainline is whether
    the retained QED-vacuum baseline pack already exposes one independent,
    alpha-explicit rerun surface.

    This backend does not create a new physical formula. It only classifies the
    current public implementation and fixes which subsurfaces are:

    1. alpha inactive under the current idealized implementation,
    2. structurally alpha sensitive but still missing an explicit rerun map,
    3. excluded because they reuse CODATA-input style alpha extraction.

Inputs:
    - scripts/quantum/qed_vacuum_precision.py
    - output/public/quantum/qed_vacuum_precision_metrics.json

Outputs:
    - One in-memory audit pack consumed by `.5899-.5902` wrappers
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


PUBLIC_OUT = ROOT / "output" / "public" / "quantum"

ALPHA_P_FROZEN = 0.007302943961943229
ALPHA_P_4D_CAN = 0.0072988143426522215
ALPHA_P_4D_VERTEX = 0.007299279720153683
ALPHA_CODATA = 0.0072973525643


# 関数: JSON artifact を可能なら読み込む。
def read_json_if_exists(path: Path) -> dict | None:
    """Read one UTF-8 JSON payload when it exists."""
    if not path.exists():
        return None

    return json.loads(path.read_text(encoding="utf-8"))


# 関数: 1 つの file-info row を返す。

def build_file_info(path: Path) -> dict:
    """Return one compact file-info dictionary."""
    return {
        "path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "exists": bool(path.exists()),
    }


# 関数: QED-vacuum subsurface rows を返す。

def build_qed_vacuum_rows() -> list[dict]:
    """Return retained QED-vacuum subsurface rows."""
    metrics_path = PUBLIC_OUT / "qed_vacuum_precision_metrics.json"
    payload = read_json_if_exists(metrics_path)
    sources = payload.get("sources", []) if isinstance(payload, dict) else []
    topic_map = {
        str(item.get("topic")): item
        for item in sources
        if isinstance(item, dict) and str(item.get("topic", "")).strip()
    }

    alpha_recoil = None
    alpha_g2 = None
    alpha_delta_z = None
    if isinstance(payload, dict):
        alpha_precision = payload.get("alpha_precision")
        if isinstance(alpha_precision, dict):
            recoil = alpha_precision.get("recoil")
            g2 = alpha_precision.get("g2")
            derived = alpha_precision.get("derived")
            if isinstance(recoil, dict):
                alpha_recoil = float(recoil["alpha_inv"])

            if isinstance(g2, dict):
                alpha_g2 = float(g2["alpha_inv"])

            if isinstance(derived, dict):
                alpha_delta_z = float(derived["z_score"])

    one_s_two_s_fractional_sigma = None
    one_s_two_s_entry = topic_map.get("hydrogen_1s2s_frequency")
    if isinstance(one_s_two_s_entry, dict):
        extracted = one_s_two_s_entry.get("extracted_value")
        if isinstance(extracted, dict):
            one_s_two_s_fractional_sigma = float(extracted["fractional_sigma"])

    return [
        {
            "subsurface_id": "casimir_ideal_conductor_baseline",
            "label": "Casimir ideal-conductor baseline",
            "alpha_dependency_kind": "inactive_under_current_ideal_baseline",
            "current_alpha_rerun_ready_now": False,
            "independent_observable_now": True,
            "primary_score_admissible_now": False,
            "selected_primary_target_now": False,
            "notes": (
                "The current implementation uses ideal-conductor PFA formulas "
                "with hbar and c only. Under this baseline Casimir is not an "
                "explicit alpha lever yet."
            ),
            "key_metric": None,
            "key_metric_label": "not_available",
        },
        {
            "subsurface_id": "lamb_shift_scaling_surface",
            "label": "Lamb-shift scaling surface",
            "alpha_dependency_kind": "structurally_alpha_sensitive_scaling_only",
            "current_alpha_rerun_ready_now": False,
            "independent_observable_now": True,
            "primary_score_admissible_now": False,
            "selected_primary_target_now": True,
            "notes": (
                "The current script retains Z^4 and Z^6 scaling surfaces plus "
                "nuclear-size examples, but it does not yet expose an absolute "
                "alpha-to-observable map for reruns."
            ),
            "key_metric": None,
            "key_metric_label": "not_available",
        },
        {
            "subsurface_id": "hydrogen_1s2s_precision_surface",
            "label": "Hydrogen 1S-2S precision surface",
            "alpha_dependency_kind": "structurally_alpha_sensitive_reference_only",
            "current_alpha_rerun_ready_now": False,
            "independent_observable_now": True,
            "primary_score_admissible_now": False,
            "selected_primary_target_now": True,
            "notes": (
                "The current script fixes the experimental 1S-2S reference and "
                "its uncertainty, but it does not yet carry a deterministic "
                "alpha-dependent prediction formula."
            ),
            "key_metric": one_s_two_s_fractional_sigma,
            "key_metric_label": "fractional_sigma_1s2s",
        },
        {
            "subsurface_id": "recoil_g2_alpha_crosscheck",
            "label": "Recoil-vs-g-2 alpha cross-check",
            "alpha_dependency_kind": "explicit_alpha_but_extraction_overlap",
            "current_alpha_rerun_ready_now": True,
            "independent_observable_now": False,
            "primary_score_admissible_now": False,
            "selected_primary_target_now": False,
            "notes": (
                "This is the only explicit alpha surface in the current QED "
                "vacuum pack, but it directly reuses recoil and electron g-2 "
                "alpha extractions and is therefore excluded from the primary score."
            ),
            "key_metric": alpha_delta_z,
            "key_metric_label": "recoil_vs_g2_z",
            "observed_values": {
                "alpha_inv_recoil": alpha_recoil,
                "alpha_inv_g2": alpha_g2,
            },
        },
    ]


# 関数: `.5899-.5902` 用の audit pack を束ねる。

def build_trial2_qed_vacuum_alpha_materialization_pack() -> dict:
    """Return the retained QED-vacuum materialization audit pack."""
    rows = build_qed_vacuum_rows()
    independent_rows = [row for row in rows if row["independent_observable_now"]]
    selected_primary_rows = [row for row in rows if row["selected_primary_target_now"]]
    rerun_ready_rows = [row for row in rows if row["current_alpha_rerun_ready_now"]]
    primary_ready_rows = [row for row in rows if row["primary_score_admissible_now"]]
    structurally_alpha_sensitive_rows = [
        row
        for row in rows
        if row["alpha_dependency_kind"]
        in {
            "structurally_alpha_sensitive_scaling_only",
            "structurally_alpha_sensitive_reference_only",
        }
    ]

    return {
        "alpha_constants": {
            "alpha_P_frozen": ALPHA_P_FROZEN,
            "alpha_P_4D_can": ALPHA_P_4D_CAN,
            "alpha_P_4D_vertex": ALPHA_P_4D_VERTEX,
            "alpha_CODATA": ALPHA_CODATA,
        },
        "subsurfaces": rows,
        "summary": {
            "qed_subsurface_count": len(rows),
            "qed_independent_subsurface_count": len(independent_rows),
            "qed_structurally_alpha_sensitive_count": len(structurally_alpha_sensitive_rows),
            "qed_selected_primary_target_count": len(selected_primary_rows),
            "qed_rerun_ready_count": len(rerun_ready_rows),
            "qed_primary_ready_count": len(primary_ready_rows),
            "selected_qed_primary_target_ids": [
                str(row["subsurface_id"]) for row in selected_primary_rows
            ],
        },
        "trial2_qed_vacuum_materialization_complete_now": True,
        "trial2_qed_vacuum_positive_partial_now": bool(len(selected_primary_rows) > 0),
        "trial2_qed_vacuum_primary_ready_now": bool(len(primary_ready_rows) > 0),
        "trial2_qed_vacuum_rerun_ready_now": bool(len(rerun_ready_rows) > 0),
    }


# 関数: backend 単体実行時に compact summary を表示する。

def main() -> None:
    """Run the QED-vacuum materialization audit directly."""
    pack = build_trial2_qed_vacuum_alpha_materialization_pack()
    summary = pack["summary"]
    print("[trial2_qed_vacuum_alpha_materialization_backend]")
    print(f"  qed_subsurface_count = {summary['qed_subsurface_count']}")
    print(
        "  qed_structurally_alpha_sensitive_count = "
        f"{summary['qed_structurally_alpha_sensitive_count']}"
    )
    print(f"  qed_primary_ready_count = {summary['qed_primary_ready_count']}")
    print(
        "  selected_qed_primary_target_ids = "
        f"{summary['selected_qed_primary_target_ids']}"
    )


if __name__ == "__main__":
    main()
