#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_origin_chi_star_proxy_route_contract.py

Step 8.7.55.2.259:
Freeze the next residual branch contract after the anchor-normalized g_3w retry
route narrows to the single missing anchor-coordinate datum
`chi_star_or_same_sector_proxy`.

Inputs:
  - output/public/quantum/mass_origin_anchor_normalized_g3w_chi_proxy_audit_metrics.json
  - output/public/quantum/mass_origin_anchor_normalized_g3w_reference_ratio_audit_metrics.json
  - output/public/quantum/mass_origin_anchor_local_shape_gate_retry_refresh_metrics.json

Outputs:
  - output/public/quantum/mass_origin_chi_star_proxy_route_contract_metrics.json
  - output/public/quantum/mass_origin_chi_star_proxy_route_contract_rows.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]

CHI_PROXY_AUDIT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_anchor_normalized_g3w_chi_proxy_audit_metrics.json"
REFERENCE_RATIO_AUDIT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_anchor_normalized_g3w_reference_ratio_audit_metrics.json"
SHAPE_GATE_RETRY_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_anchor_local_shape_gate_retry_refresh_metrics.json"
OUT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_chi_star_proxy_route_contract_metrics.json"
OUT_CSV = ROOT / "output" / "public" / "quantum" / "mass_origin_chi_star_proxy_route_contract_rows.csv"
DEFAULT_STEP_TAG = "8.7.55.2.259"


# 関数: `_utc_now_iso` の入出力契約と処理意図を定義する。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_parse_args` の入出力契約と処理意図を定義する。

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze the next residual branch contract for the missing chi_* or same-sector proxy datum.",
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

def _build_rows(*, required_route_items: List[str], split_contract_ready: bool) -> List[Dict[str, Any]]:
    return [
        {
            "row_id": "chi_star_proxy_route_contract_complete",
            "status": "pass",
            "metric": "chi_* or same-sector proxy residual contract complete",
            "value": 1.0,
            "note": "This step freezes the next residual branch after the retry route narrows to the missing anchor-coordinate datum.",
        },
        {
            "row_id": "chi_star_proxy_route_contract_required_items",
            "status": "watch",
            "metric": "required route items for chi_* proxy residual branch",
            "value": float(len(required_route_items)),
            "note": f"Required route items: {required_route_items}.",
        },
        {
            "row_id": "chi_star_proxy_route_contract_split_ready",
            "status": "pass" if split_contract_ready else "reject",
            "metric": "chi_* proxy residual split contract ready",
            "value": 1.0 if split_contract_ready else 0.0,
            "note": (
                "The next branch can now audit concrete source candidates for the missing chi_* or same-sector proxy datum."
                if split_contract_ready
                else "The chi_* proxy residual branch is not yet formalized."
            ),
        },
    ]


# 関数: `_build_payload` の入出力契約と処理意図を定義する。

def _build_payload(step_tag: str) -> Dict[str, Any]:
    for path in (CHI_PROXY_AUDIT_JSON, REFERENCE_RATIO_AUDIT_JSON, SHAPE_GATE_RETRY_JSON):
        _require_path(path)

    chi_proxy_audit = _read_json(CHI_PROXY_AUDIT_JSON)
    reference_ratio_audit = _read_json(REFERENCE_RATIO_AUDIT_JSON)
    shape_gate_retry = _read_json(SHAPE_GATE_RETRY_JSON)

    chi_proxy_summary = chi_proxy_audit.get("summary", {})
    reference_ratio_summary = reference_ratio_audit.get("summary", {})
    shape_gate_retry_summary = shape_gate_retry.get("summary", {})

    required_route_items = [
        "chi_definition",
        "rho_star_reference_point_symbol",
        "chi_star_or_same_sector_proxy",
        "same_sector_equivalence_rule",
        "no_new_free_parameter_wording",
    ]
    split_contract_ready = True
    rows = _build_rows(required_route_items=required_route_items, split_contract_ready=split_contract_ready)

    return {
        "generated_utc": _utc_now_iso(),
        "phase": {
            "phase": 8,
            "step": step_tag,
            "name": "chi_star or same-sector proxy route contract freeze",
        },
        "inputs": {
            "mass_origin_anchor_normalized_g3w_chi_proxy_audit_json": _relative_str(CHI_PROXY_AUDIT_JSON),
            "mass_origin_anchor_normalized_g3w_reference_ratio_audit_json": _relative_str(REFERENCE_RATIO_AUDIT_JSON),
            "mass_origin_anchor_local_shape_gate_retry_refresh_json": _relative_str(SHAPE_GATE_RETRY_JSON),
        },
        "intent": "Freeze the next residual branch contract after the anchor-normalized g_3w retry route narrows to the single missing anchor-coordinate datum chi_star_or_same_sector_proxy.",
        "formulas": {
            "residual_rule": "rho_* / P_ref = exp(chi_*) can close only after a public chi_* datum or an equivalent same-sector proxy rule is available",
            "closure_rule": "anchor_normalized_g3w_public_value closes only after the chi_* or same-sector proxy datum is formalized without a new fit",
        },
        "rows": rows,
        "summary": {
            "missing_anchor_coordinate_datum": "chi_star_or_same_sector_proxy",
            "required_route_items": required_route_items,
            "split_contract_ready": split_contract_ready,
        },
        "decision": {
            "overall_status": "chi_star_proxy_route_contract_frozen",
            "keep_mass_origin_branch_blocked": True,
            "missing_anchor_coordinate_datum": "chi_star_or_same_sector_proxy",
            "split_contract_ready": split_contract_ready,
            "hand_off_to_8_7_55_2_83": False,
            "next_required_artifacts": [
                "chi_star_or_same_sector_proxy",
                "anchor_normalized_g3w_public_value",
                "r3_target",
                "single_public_vpp_shape",
                "positive_particle_sector_chi_p_to_vpp_public_artifact",
                "solver_ready_row_promoted_to_pass",
            ],
        },
        "evidence": {
            "chi_proxy_summary": chi_proxy_summary,
            "reference_ratio_summary": reference_ratio_summary,
            "shape_gate_retry_summary": shape_gate_retry_summary,
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
