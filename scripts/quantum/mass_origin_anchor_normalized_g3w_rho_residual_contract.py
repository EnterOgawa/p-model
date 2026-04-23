#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_origin_anchor_normalized_g3w_rho_residual_contract.py

Step 8.7.55.2.253:
Freeze the residual branch contract after the preferred anchor-normalized g_3w
route closes its symbolic anchor normalization but still fails on rho_*.

Inputs:
  - output/public/quantum/mass_origin_anchor_normalized_g3w_value_closure_metrics.json
  - output/public/quantum/mass_origin_anchor_local_shape_gate_refresh_metrics.json
  - output/public/quantum/mass_origin_anchor_normalized_g3w_anchor_audit_metrics.json

Outputs:
  - output/public/quantum/mass_origin_anchor_normalized_g3w_rho_residual_contract_metrics.json
  - output/public/quantum/mass_origin_anchor_normalized_g3w_rho_residual_contract_rows.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]

VALUE_CLOSURE_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_anchor_normalized_g3w_value_closure_metrics.json"
SHAPE_REFRESH_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_anchor_local_shape_gate_refresh_metrics.json"
ANCHOR_AUDIT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_anchor_normalized_g3w_anchor_audit_metrics.json"
OUT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_anchor_normalized_g3w_rho_residual_contract_metrics.json"
OUT_CSV = ROOT / "output" / "public" / "quantum" / "mass_origin_anchor_normalized_g3w_rho_residual_contract_rows.csv"
DEFAULT_STEP_TAG = "8.7.55.2.253"


# 関数: `_utc_now_iso` の入出力契約と処理意図を定義する。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_parse_args` の入出力契約と処理意図を定義する。

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze the residual rho_* branch contract after the anchor-normalized g_3w route refresh.",
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

def _build_rows(*, remaining_route_items: List[str], split_contract_ready: bool) -> List[Dict[str, Any]]:
    return [
        {
            "row_id": "anchor_normalized_g3w_rho_residual_contract_complete",
            "status": "pass",
            "metric": "anchor-normalized g_3w rho residual contract complete",
            "value": 1.0,
            "note": "This step freezes the residual branch after the preferred g_3w route narrows to the rho_* blocker.",
        },
        {
            "row_id": "anchor_normalized_g3w_rho_residual_items",
            "status": "watch",
            "metric": "remaining residual route items",
            "value": float(len(remaining_route_items)),
            "note": f"Remaining route items: {remaining_route_items}.",
        },
        {
            "row_id": "anchor_normalized_g3w_rho_residual_split_contract_ready",
            "status": "pass" if split_contract_ready else "reject",
            "metric": "rho residual split contract ready",
            "value": 1.0 if split_contract_ready else 0.0,
            "note": (
                "The next branch can now audit same-sector chi_* proxy / rho_* ratio routes separately."
                if split_contract_ready
                else "The residual rho_* branch is not yet formalized."
            ),
        },
    ]


# 関数: `_build_payload` の入出力契約と処理意図を定義する。

def _build_payload(step_tag: str) -> Dict[str, Any]:
    for path in (VALUE_CLOSURE_JSON, SHAPE_REFRESH_JSON, ANCHOR_AUDIT_JSON):
        _require_path(path)

    value_closure = _read_json(VALUE_CLOSURE_JSON)
    shape_refresh = _read_json(SHAPE_REFRESH_JSON)
    anchor_audit = _read_json(ANCHOR_AUDIT_JSON)

    value_summary = value_closure.get("summary", {})
    shape_summary = shape_refresh.get("summary", {})
    anchor_summary = anchor_audit.get("summary", {})

    remaining_route_items = [
        "chi_star_or_same_sector_proxy",
        "rho_star_to_reference_ratio_rule",
        "anchor_normalized_g3w_public_value",
    ]
    split_contract_ready = True
    rows = _build_rows(remaining_route_items=remaining_route_items, split_contract_ready=split_contract_ready)

    return {
        "generated_utc": _utc_now_iso(),
        "phase": {
            "phase": 8,
            "step": step_tag,
            "name": "anchor-normalized g3w rho residual contract freeze",
        },
        "inputs": {
            "mass_origin_anchor_normalized_g3w_value_closure_json": _relative_str(VALUE_CLOSURE_JSON),
            "mass_origin_anchor_local_shape_gate_refresh_json": _relative_str(SHAPE_REFRESH_JSON),
            "mass_origin_anchor_normalized_g3w_anchor_audit_json": _relative_str(ANCHOR_AUDIT_JSON),
        },
        "intent": "Freeze the residual branch contract after the preferred g_3w route narrows to the rho_* elimination blocker.",
        "formulas": {
            "residual_rule": "after symbolic anchor normalization succeeds, the residual branch focuses on replacing rho_* by a public same-sector proxy or a reference-ratio rule",
            "closure_rule": "anchor_normalized_g3w_public_value closes only after a rho_* replacement route is formalized",
        },
        "rows": rows,
        "summary": {
            "preferred_r3_route_or_none": "g3w_anchor_normalized_route",
            "anchor_normalization_rule_available": anchor_summary.get("anchor_normalization_rule_available", False),
            "rho_star_elimination_rule_available": False,
            "anchor_normalized_g3w_public_value_available": value_summary.get("anchor_normalized_g3w_public_value_available", False),
            "remaining_route_items": remaining_route_items,
            "split_contract_ready": split_contract_ready,
        },
        "decision": {
            "overall_status": "anchor_normalized_g3w_rho_residual_contract_frozen",
            "keep_mass_origin_branch_blocked": True,
            "anchor_normalization_rule_available": anchor_summary.get("anchor_normalization_rule_available", False),
            "rho_star_elimination_rule_available": False,
            "anchor_normalized_g3w_public_value_available": value_summary.get("anchor_normalized_g3w_public_value_available", False),
            "hand_off_to_8_7_55_2_83": shape_summary.get("hand_off_to_8_7_55_2_83", False),
            "remaining_route_items": remaining_route_items,
            "split_contract_ready": split_contract_ready,
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
            "value_closure_summary": value_summary,
            "shape_gate_refresh_summary": shape_summary,
            "anchor_audit_summary": anchor_summary,
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
