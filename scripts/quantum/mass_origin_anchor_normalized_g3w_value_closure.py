#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_origin_anchor_normalized_g3w_value_closure.py

Step 8.7.55.2.251:
Bundle the anchor-normalized g_3w route audits and decide whether the route now
supplies a public anchor-normalized value and therefore an R_3 target.

Inputs:
  - output/public/quantum/mass_origin_anchor_normalized_g3w_route_contract_metrics.json
  - output/public/quantum/mass_origin_anchor_normalized_g3w_source_inventory_metrics.json
  - output/public/quantum/mass_origin_anchor_normalized_g3w_anchor_audit_metrics.json
  - output/public/quantum/mass_origin_anchor_normalized_g3w_rho_elimination_audit_metrics.json

Outputs:
  - output/public/quantum/mass_origin_anchor_normalized_g3w_value_closure_metrics.json
  - output/public/quantum/mass_origin_anchor_normalized_g3w_value_closure_rows.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]

ROUTE_CONTRACT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_anchor_normalized_g3w_route_contract_metrics.json"
SOURCE_INVENTORY_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_anchor_normalized_g3w_source_inventory_metrics.json"
ANCHOR_AUDIT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_anchor_normalized_g3w_anchor_audit_metrics.json"
RHO_AUDIT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_anchor_normalized_g3w_rho_elimination_audit_metrics.json"
OUT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_anchor_normalized_g3w_value_closure_metrics.json"
OUT_CSV = ROOT / "output" / "public" / "quantum" / "mass_origin_anchor_normalized_g3w_value_closure_rows.csv"
DEFAULT_STEP_TAG = "8.7.55.2.251"


# 関数: `_utc_now_iso` の入出力契約と処理意図を定義する。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_parse_args` の入出力契約と処理意図を定義する。

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Close or reject the anchor-normalized g_3w public-value route.",
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
    anchor_normalization_rule_available: bool,
    rho_star_elimination_rule_available: bool,
    anchor_normalized_g3w_public_value_available: bool,
    r3_target_available: bool,
    nonclosure_reason: str | None,
) -> List[Dict[str, Any]]:
    return [
        {
            "row_id": "anchor_normalized_g3w_value_closure_complete",
            "status": "pass",
            "metric": "anchor-normalized g_3w value closure complete",
            "value": 1.0,
            "note": "This step bundles the route audits and decides whether the preferred g_3w route can already promote a public value.",
        },
        {
            "row_id": "anchor_normalized_g3w_value_closure_anchor_rule_available",
            "status": "pass" if anchor_normalization_rule_available else "reject",
            "metric": "anchor normalization rule available at closure step",
            "value": 1.0 if anchor_normalization_rule_available else 0.0,
            "note": (
                "The route already has a symbolic anchor-normalization rule."
                if anchor_normalization_rule_available
                else "The route still lacks even the symbolic anchor-normalization rule."
            ),
        },
        {
            "row_id": "anchor_normalized_g3w_value_closure_rho_rule_available",
            "status": "pass" if rho_star_elimination_rule_available else "reject",
            "metric": "rho_* elimination rule available at closure step",
            "value": 1.0 if rho_star_elimination_rule_available else 0.0,
            "note": (
                "The route already removes rho_* and is therefore ready to promote a public value."
                if rho_star_elimination_rule_available
                else "The route still cannot remove rho_* from the anchor-normalized g_3w expression."
            ),
        },
        {
            "row_id": "anchor_normalized_g3w_value_closure_public_value_available",
            "status": "pass" if anchor_normalized_g3w_public_value_available else "missing",
            "metric": "anchor-normalized public g_3w value available",
            "value": 1.0 if anchor_normalized_g3w_public_value_available else 0.0,
            "note": (
                "A public anchor-normalized g_3w value is now available."
                if anchor_normalized_g3w_public_value_available
                else f"The route remains non-closing: {nonclosure_reason}."
            ),
        },
        {
            "row_id": "anchor_normalized_g3w_value_closure_r3_target_available",
            "status": "pass" if r3_target_available else "reject",
            "metric": "R_3 target available after g_3w route closure",
            "value": 1.0 if r3_target_available else 0.0,
            "note": (
                "The g_3w route now promotes R_3_target."
                if r3_target_available
                else "R_3_target remains unavailable because the public anchor-normalized g_3w value is still missing."
            ),
        },
    ]


# 関数: `_build_payload` の入出力契約と処理意図を定義する。

def _build_payload(step_tag: str) -> Dict[str, Any]:
    for path in (ROUTE_CONTRACT_JSON, SOURCE_INVENTORY_JSON, ANCHOR_AUDIT_JSON, RHO_AUDIT_JSON):
        _require_path(path)

    route_contract = _read_json(ROUTE_CONTRACT_JSON)
    inventory = _read_json(SOURCE_INVENTORY_JSON)
    anchor_audit = _read_json(ANCHOR_AUDIT_JSON)
    rho_audit = _read_json(RHO_AUDIT_JSON)

    route_summary = route_contract.get("summary", {})
    inventory_summary = inventory.get("summary", {})
    anchor_summary = anchor_audit.get("summary", {})
    rho_summary = rho_audit.get("summary", {})

    anchor_normalization_rule_available = bool(anchor_summary.get("anchor_normalization_rule_available", False))
    rho_star_elimination_rule_available = bool(rho_summary.get("rho_star_elimination_rule_available", False))
    anchor_normalized_g3w_public_value_available = bool(anchor_normalization_rule_available and rho_star_elimination_rule_available)
    r3_target_available = anchor_normalized_g3w_public_value_available
    r3_target_value_or_none = None
    nonclosure_reason = None

    # 条件分岐: `not anchor_normalized_g3w_public_value_available` を満たす経路を評価する。
    if not anchor_normalized_g3w_public_value_available:
        nonclosure_reason = "rho_star_elimination_rule_absent"

    rows = _build_rows(
        anchor_normalization_rule_available=anchor_normalization_rule_available,
        rho_star_elimination_rule_available=rho_star_elimination_rule_available,
        anchor_normalized_g3w_public_value_available=anchor_normalized_g3w_public_value_available,
        r3_target_available=r3_target_available,
        nonclosure_reason=nonclosure_reason,
    )

    return {
        "generated_utc": _utc_now_iso(),
        "phase": {
            "phase": 8,
            "step": step_tag,
            "name": "anchor-normalized g3w public value closure",
        },
        "inputs": {
            "mass_origin_anchor_normalized_g3w_route_contract_json": _relative_str(ROUTE_CONTRACT_JSON),
            "mass_origin_anchor_normalized_g3w_source_inventory_json": _relative_str(SOURCE_INVENTORY_JSON),
            "mass_origin_anchor_normalized_g3w_anchor_audit_json": _relative_str(ANCHOR_AUDIT_JSON),
            "mass_origin_anchor_normalized_g3w_rho_elimination_audit_json": _relative_str(RHO_AUDIT_JSON),
        },
        "intent": "Promote or reject a public anchor-normalized g_3w value and the corresponding R_3 target.",
        "formulas": {
            "closure_rule": "anchor_normalized_g3w_public_value_available iff both the symbolic anchor-normalization rule and the rho_* elimination rule are public canonical",
            "r3_promotion_rule": "r3_target_available iff anchor_normalized_g3w_public_value_available",
        },
        "rows": rows,
        "summary": {
            "preferred_r3_route_or_none": route_summary.get("preferred_r3_route_or_none"),
            "required_g3w_route_sources": inventory_summary.get("required_g3w_route_sources", []),
            "anchor_normalization_rule_available": anchor_normalization_rule_available,
            "rho_star_elimination_rule_available": rho_star_elimination_rule_available,
            "anchor_normalized_g3w_public_value_available": anchor_normalization_rule_available and rho_star_elimination_rule_available,
            "r3_target_available": r3_target_available,
            "r3_target_value_or_none": r3_target_value_or_none,
            "g3w_route_nonclosure_reason_or_none": nonclosure_reason,
        },
        "decision": {
            "overall_status": "anchor_normalized_g3w_value_closure_frozen",
            "keep_mass_origin_branch_blocked": True,
            "anchor_normalization_rule_available": anchor_normalization_rule_available,
            "rho_star_elimination_rule_available": rho_star_elimination_rule_available,
            "anchor_normalized_g3w_public_value_available": anchor_normalization_rule_available and rho_star_elimination_rule_available,
            "r3_target_available": r3_target_available,
            "r3_target_value_or_none": r3_target_value_or_none,
            "g3w_route_nonclosure_reason_or_none": nonclosure_reason,
            "hand_off_to_8_7_55_2_83": False,
            "next_required_artifacts": [
                "rho_star_elimination_rule",
                "anchor_normalized_g3w_public_value",
                "r3_target",
                "single_public_vpp_shape",
                "positive_particle_sector_chi_p_to_vpp_public_artifact",
                "solver_ready_row_promoted_to_pass",
            ],
        },
        "evidence": {
            "route_contract_summary": route_summary,
            "source_inventory_summary": inventory_summary,
            "anchor_audit_summary": anchor_summary,
            "rho_audit_summary": rho_summary,
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
