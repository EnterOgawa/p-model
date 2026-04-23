#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_origin_anchor_normalized_g3w_chi_proxy_inventory.py

Step 8.7.55.2.254:
Inventory the required / present / missing public-canonical sources for the
chi_* or same-sector proxy route inside the residual rho_* elimination branch.

Inputs:
  - output/public/quantum/mass_origin_anchor_normalized_g3w_rho_elimination_audit_metrics.json
  - output/public/quantum/mass_origin_anchor_normalized_g3w_rho_residual_contract_metrics.json

Outputs:
  - output/public/quantum/mass_origin_anchor_normalized_g3w_chi_proxy_inventory_metrics.json
  - output/public/quantum/mass_origin_anchor_normalized_g3w_chi_proxy_inventory_rows.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]

RHO_AUDIT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_anchor_normalized_g3w_rho_elimination_audit_metrics.json"
RHO_RESIDUAL_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_anchor_normalized_g3w_rho_residual_contract_metrics.json"
OUT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_anchor_normalized_g3w_chi_proxy_inventory_metrics.json"
OUT_CSV = ROOT / "output" / "public" / "quantum" / "mass_origin_anchor_normalized_g3w_chi_proxy_inventory_rows.csv"
DEFAULT_STEP_TAG = "8.7.55.2.254"


# 関数: `_utc_now_iso` の入出力契約と処理意図を定義する。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_parse_args` の入出力契約と処理意図を定義する。

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inventory public sources for the chi_* or same-sector proxy route.",
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
    required_sources: List[str],
    present_sources: List[str],
    missing_sources: List[str],
    first_route_to_close_or_none: str | None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = [
        {
            "row_id": "anchor_normalized_g3w_chi_proxy_inventory_complete",
            "status": "pass",
            "metric": "chi_* or same-sector proxy inventory complete",
            "value": 1.0,
            "note": "This step freezes the required, present, and missing source list for the chi_* proxy route.",
        },
        {
            "row_id": "anchor_normalized_g3w_chi_proxy_inventory_required_count",
            "status": "pass",
            "metric": "required source count for chi_* proxy route",
            "value": float(len(required_sources)),
            "note": f"Required sources: {required_sources}.",
        },
        {
            "row_id": "anchor_normalized_g3w_chi_proxy_inventory_present_count",
            "status": "pass",
            "metric": "present source count for chi_* proxy route",
            "value": float(len(present_sources)),
            "note": f"Present sources: {present_sources}.",
        },
        {
            "row_id": "anchor_normalized_g3w_chi_proxy_inventory_missing_count",
            "status": "watch" if missing_sources else "pass",
            "metric": "missing source count for chi_* proxy route",
            "value": float(len(missing_sources)),
            "note": f"Missing sources: {missing_sources}.",
        },
    ]

    for source_name in required_sources:
        is_present = source_name in present_sources
        rows.append(
            {
                "row_id": f"anchor_normalized_g3w_chi_proxy_source_{source_name}",
                "status": "pass" if is_present else "watch",
                "metric": f"source availability for {source_name}",
                "value": 1.0 if is_present else 0.0,
                "note": (
                    f"{source_name} is already available in the current public canonical pack."
                    if is_present
                    else f"{source_name} is still missing from the current public canonical pack."
                ),
            }
        )

    rows.append(
        {
            "row_id": "anchor_normalized_g3w_chi_proxy_inventory_first_route",
            "status": "watch" if first_route_to_close_or_none else "reject",
            "metric": "first residual source to audit after chi_* proxy inventory",
            "value": 1.0 if first_route_to_close_or_none else 0.0,
            "note": (
                f"The next closure attempt starts from {first_route_to_close_or_none}."
                if first_route_to_close_or_none
                else "No preferred chi_* proxy source can be chosen from the inventory."
            ),
        }
    )
    return rows


# 関数: `_build_payload` の入出力契約と処理意図を定義する。

def _build_payload(step_tag: str) -> Dict[str, Any]:
    for path in (RHO_AUDIT_JSON, RHO_RESIDUAL_JSON):
        _require_path(path)

    rho_audit = _read_json(RHO_AUDIT_JSON)
    rho_residual = _read_json(RHO_RESIDUAL_JSON)

    rho_summary = rho_audit.get("summary", {})
    rho_decision = rho_audit.get("decision", {})
    rho_evidence = rho_audit.get("evidence", {})
    curvature_summary = rho_evidence.get("curvature_summary", {})
    residual_summary = rho_residual.get("summary", {})
    residual_decision = rho_residual.get("decision", {})

    required_sources = [
        "chi_definition",
        "same_sector_contract",
        "chi_p_local_static_identification_wording",
        "rho_star_reference_point_symbol",
        "chi_star_or_same_sector_proxy",
    ]

    present_sources: List[str] = []

    # 条件分岐: `curvature_summary.get(\"chi_definition_frozen\", False)` を満たす経路を評価する。
    if curvature_summary.get("chi_definition_frozen", False):
        present_sources.append("chi_definition")

    # 条件分岐: `curvature_summary.get(\"same_sector_contract_frozen\", False)` を満たす経路を評価する。

    if curvature_summary.get("same_sector_contract_frozen", False):
        present_sources.append("same_sector_contract")

    # 条件分岐: `curvature_summary.get(\"chi_p_local_static_identification_wording_ready\", False)` を満たす経路を評価する。

    if curvature_summary.get("chi_p_local_static_identification_wording_ready", False):
        present_sources.append("chi_p_local_static_identification_wording")

    # 条件分岐: `bool(rho_summary.get(\"reference_point_symbol\"))` を満たす経路を評価する。

    if bool(rho_summary.get("reference_point_symbol")):
        present_sources.append("rho_star_reference_point_symbol")

    remaining_route_items = [str(item) for item in residual_summary.get("remaining_route_items", [])]
    missing_sources = [item for item in required_sources if item not in present_sources]

    # 条件分岐: `"chi_star_or_same_sector_proxy" not in remaining_route_items` を満たす経路を評価する。
    if "chi_star_or_same_sector_proxy" not in remaining_route_items and "chi_star_or_same_sector_proxy" in missing_sources:
        missing_sources.remove("chi_star_or_same_sector_proxy")
        present_sources.append("chi_star_or_same_sector_proxy")

    first_route_to_close_or_none = "chi_star_or_same_sector_proxy"
    rows = _build_rows(
        required_sources=required_sources,
        present_sources=present_sources,
        missing_sources=missing_sources,
        first_route_to_close_or_none=first_route_to_close_or_none,
    )

    return {
        "generated_utc": _utc_now_iso(),
        "phase": {
            "phase": 8,
            "step": step_tag,
            "name": "chi_star or same-sector proxy source inventory",
        },
        "inputs": {
            "mass_origin_anchor_normalized_g3w_rho_elimination_audit_json": _relative_str(RHO_AUDIT_JSON),
            "mass_origin_anchor_normalized_g3w_rho_residual_contract_json": _relative_str(RHO_RESIDUAL_JSON),
        },
        "intent": "Freeze the required / present / missing public-canonical sources for the chi_* or same-sector proxy route inside the residual rho_* branch.",
        "formulas": {
            "inventory_rule": "the chi_* proxy route stays open until the public pack exposes chi = ln(rho / P_ref), the same-sector contract, the chi_P local-static wording, the rho_* reference symbol, and a public chi_* or same-sector proxy value",
            "proxy_need": "rho_* elimination can start either from an explicit chi_* datum or from an equivalent same-sector proxy that fixes the anchor branch coordinate without a new fit",
            "current_absence": "the current public pack already freezes chi-space language and the rho_* reference point, but it does not yet expose a public chi_* or same-sector proxy datum",
        },
        "rows": rows,
        "summary": {
            "required_chi_proxy_sources": required_sources,
            "present_chi_proxy_sources": present_sources,
            "missing_chi_proxy_sources": missing_sources,
            "first_route_to_close_or_none": first_route_to_close_or_none,
            "chi_proxy_inventory_ready": True,
        },
        "decision": {
            "overall_status": "anchor_normalized_g3w_chi_proxy_inventory_frozen",
            "keep_mass_origin_branch_blocked": True,
            "preferred_r3_route_or_none": residual_summary.get("preferred_r3_route_or_none"),
            "required_chi_proxy_sources": required_sources,
            "present_chi_proxy_sources": present_sources,
            "missing_chi_proxy_sources": missing_sources,
            "first_route_to_close_or_none": first_route_to_close_or_none,
            "chi_proxy_route_still_missing": "chi_star_or_same_sector_proxy" in missing_sources,
            "chi_proxy_inventory_ready": True,
            "hand_off_to_8_7_55_2_83": bool(residual_decision.get("hand_off_to_8_7_55_2_83", False)),
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
            "rho_elimination_summary": rho_summary,
            "rho_elimination_decision": rho_decision,
            "curvature_summary": curvature_summary,
            "rho_residual_summary": residual_summary,
            "rho_residual_decision": residual_decision,
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
