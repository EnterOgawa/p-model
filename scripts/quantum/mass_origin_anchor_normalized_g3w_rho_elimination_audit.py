#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_origin_anchor_normalized_g3w_rho_elimination_audit.py

Step 8.7.55.2.250:
Audit whether rho_* can already be eliminated from the preferred
anchor-normalized g_3w route without introducing a new free parameter.

Inputs:
  - output/public/quantum/mass_origin_anchor_normalized_g3w_source_inventory_metrics.json
  - output/public/quantum/mass_origin_anchor_normalized_g3w_anchor_audit_metrics.json
  - output/public/quantum/mass_origin_anchor_local_curvature_bridge_metrics.json

Outputs:
  - output/public/quantum/mass_origin_anchor_normalized_g3w_rho_elimination_audit_metrics.json
  - output/public/quantum/mass_origin_anchor_normalized_g3w_rho_elimination_audit_rows.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]

SOURCE_INVENTORY_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_anchor_normalized_g3w_source_inventory_metrics.json"
ANCHOR_AUDIT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_anchor_normalized_g3w_anchor_audit_metrics.json"
CURVATURE_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_anchor_local_curvature_bridge_metrics.json"
OUT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_anchor_normalized_g3w_rho_elimination_audit_metrics.json"
OUT_CSV = ROOT / "output" / "public" / "quantum" / "mass_origin_anchor_normalized_g3w_rho_elimination_audit_rows.csv"
DEFAULT_STEP_TAG = "8.7.55.2.250"


# 関数: `_utc_now_iso` の入出力契約と処理意図を定義する。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_parse_args` の入出力契約と処理意図を定義する。

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit whether rho_* can already be eliminated from the anchor-normalized g_3w route.",
    )
    parser.add_argument(
        "--step-tag",
        default=DEFAULT_STEP_TAG,
        help="Roadmap step tag to stamp into the output payload.",
    )
    return parser.parse_args()


# 関数: `_require_path` の入出力契約と処理意図を定義する。

def _require_path(path: Path) -> None:
    # 条件分岐: `not path.exists()` を満たす経路を評価する。
    if not path.exists():
        raise SystemExit(f"[fail] missing required input: {path}")


# 関数: `_read_json` の入出力契約と処理意図を定義する。

def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


# 関数: `_relative_str` の入出力契約と処理意図を定義する。

def _relative_str(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


# 関数: `_build_rows` の入出力契約と処理意図を定義する。

def _build_rows(
    *,
    rho_star_elimination_rule_available: bool,
    rho_star_elimination_kind_or_none: str | None,
    rho_star_elimination_without_new_free_parameters: bool,
    missing_inputs: List[str],
) -> List[Dict[str, Any]]:
    return [
        {
            "row_id": "anchor_normalized_g3w_rho_elimination_audit_complete",
            "status": "pass",
            "metric": "rho_* elimination audit complete",
            "value": 1.0,
            "note": "This step tests whether rho_* can already be removed from the preferred anchor-normalized g_3w route.",
        },
        {
            "row_id": "anchor_normalized_g3w_rho_elimination_rule_available",
            "status": "pass" if rho_star_elimination_rule_available else "reject",
            "metric": "rho_* elimination rule available",
            "value": 1.0 if rho_star_elimination_rule_available else 0.0,
            "note": (
                f"The current public pack already supports {rho_star_elimination_kind_or_none}."
                if rho_star_elimination_rule_available
                else "The current public pack still exposes rho_* only as a reference-point symbol, not as an eliminable public observable or ratio."
            ),
        },
        {
            "row_id": "anchor_normalized_g3w_rho_elimination_without_new_free_parameters",
            "status": "pass" if rho_star_elimination_without_new_free_parameters else "reject",
            "metric": "rho_* elimination stays inside no-new-free-parameter envelope",
            "value": 1.0 if rho_star_elimination_without_new_free_parameters else 0.0,
            "note": (
                "The current public pack already exposes a same-sector proxy that removes rho_* without a new fit."
                if rho_star_elimination_without_new_free_parameters
                else "Any elimination would still require a missing same-sector proxy or reference-ratio rule."
            ),
        },
        {
            "row_id": "anchor_normalized_g3w_rho_elimination_missing_inputs",
            "status": "missing" if missing_inputs else "pass",
            "metric": "remaining missing inputs for rho_* elimination",
            "value": float(len(missing_inputs)),
            "note": f"Missing inputs: {missing_inputs}.",
        },
    ]


# 関数: `_build_payload` の入出力契約と処理意図を定義する。

def _build_payload(step_tag: str) -> Dict[str, Any]:
    for path in (SOURCE_INVENTORY_JSON, ANCHOR_AUDIT_JSON, CURVATURE_JSON):
        _require_path(path)

    inventory = _read_json(SOURCE_INVENTORY_JSON)
    anchor_audit = _read_json(ANCHOR_AUDIT_JSON)
    curvature = _read_json(CURVATURE_JSON)

    inventory_summary = inventory.get("summary", {})
    anchor_summary = anchor_audit.get("summary", {})
    curvature_summary = curvature.get("summary", {})

    anchor_normalization_rule_available = bool(anchor_summary.get("anchor_normalization_rule_available", False))
    reference_point_symbol = curvature_summary.get("reference_point_symbol")
    rho_star_elimination_rule_available = False
    rho_star_elimination_kind_or_none = None
    rho_star_elimination_without_new_free_parameters = False
    missing_inputs = [
        "chi_star_or_same_sector_proxy",
        "rho_star_to_reference_ratio_rule",
    ]

    rows = _build_rows(
        rho_star_elimination_rule_available=rho_star_elimination_rule_available,
        rho_star_elimination_kind_or_none=rho_star_elimination_kind_or_none,
        rho_star_elimination_without_new_free_parameters=rho_star_elimination_without_new_free_parameters,
        missing_inputs=missing_inputs,
    )

    return {
        "generated_utc": _utc_now_iso(),
        "phase": {
            "phase": 8,
            "step": step_tag,
            "name": "rho_star elimination audit",
        },
        "inputs": {
            "mass_origin_anchor_normalized_g3w_source_inventory_json": _relative_str(SOURCE_INVENTORY_JSON),
            "mass_origin_anchor_normalized_g3w_anchor_audit_json": _relative_str(ANCHOR_AUDIT_JSON),
            "mass_origin_anchor_local_curvature_bridge_json": _relative_str(CURVATURE_JSON),
        },
        "intent": "Determine whether the preferred anchor-normalized g_3w route can already remove rho_* without a new parameter.",
        "formulas": {
            "route_with_rho_star": "R_3 = rho_* (2 g_3w / V''(rho_*)) = 2 rho_*^3 g_3w / (M_chi^2 omega_*^2)",
            "elimination_rule": "rho_star_elimination_rule_available iff the public pack already exposes a same-sector proxy or a reference-ratio rule that replaces rho_*",
            "current_absence": "the public pack names rho_* = |P|_* as a reference-point symbol but does not yet expose chi_* or an equivalent same-sector proxy value",
        },
        "rows": rows,
        "summary": {
            "anchor_normalization_rule_available": anchor_normalization_rule_available,
            "reference_point_symbol": reference_point_symbol,
            "rho_star_elimination_rule_available": rho_star_elimination_rule_available,
            "rho_star_elimination_kind_or_none": rho_star_elimination_kind_or_none,
            "rho_star_elimination_without_new_free_parameters": rho_star_elimination_without_new_free_parameters,
            "remaining_missing_inputs": missing_inputs,
        },
        "decision": {
            "overall_status": "anchor_normalized_g3w_rho_elimination_frozen_absent",
            "keep_mass_origin_branch_blocked": True,
            "anchor_normalization_rule_available": anchor_normalization_rule_available,
            "rho_star_elimination_rule_available": rho_star_elimination_rule_available,
            "rho_star_elimination_kind_or_none": rho_star_elimination_kind_or_none,
            "rho_star_elimination_without_new_free_parameters": rho_star_elimination_without_new_free_parameters,
            "hand_off_to_8_7_55_2_83": False,
            "next_required_artifacts": [
                "chi_star_or_same_sector_proxy",
                "rho_star_to_reference_ratio_rule",
                "anchor_normalized_g3w_public_value",
                "r3_target",
                "single_public_vpp_shape",
                "positive_particle_sector_chi_p_to_vpp_public_artifact",
                "solver_ready_row_promoted_to_pass",
            ],
        },
        "evidence": {
            "source_inventory_summary": inventory_summary,
            "anchor_audit_summary": anchor_summary,
            "curvature_summary": curvature_summary,
        },
    }


# 関数: `_write_csv` の入出力契約と処理意図を定義する。

def _write_csv(rows: List[Dict[str, Any]]) -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["row_id", "status", "metric", "value", "note"])
        writer.writeheader()
        writer.writerows(rows)


# 関数: `main` の入出力契約と処理意図を定義する。

def main() -> None:
    args = _parse_args()
    payload = _build_payload(args.step_tag)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_csv(payload["rows"])
    print(f"[ok] wrote {OUT_JSON}")
    print(f"[ok] wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
