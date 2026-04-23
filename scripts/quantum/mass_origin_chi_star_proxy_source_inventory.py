#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_origin_chi_star_proxy_source_inventory.py

Step 8.7.55.2.260:
Inventory the current public-canonical source candidates for the missing
anchor-coordinate datum `chi_star_or_same_sector_proxy`.

Inputs:
  - output/public/quantum/mass_origin_anchor_normalized_g3w_rho_elimination_audit_metrics.json
  - output/public/quantum/mass_origin_anchor_normalized_g3w_chi_proxy_inventory_metrics.json
  - output/public/quantum/mass_origin_anchor_normalized_g3w_chi_proxy_audit_metrics.json
  - output/public/quantum/mass_origin_anchor_normalized_g3w_reference_ratio_audit_metrics.json
  - output/public/quantum/mass_origin_anchor_normalized_g3w_value_retry_metrics.json
  - output/public/quantum/mass_origin_anchor_local_shape_gate_retry_refresh_metrics.json
  - output/public/quantum/mass_origin_chi_star_proxy_route_contract_metrics.json

Outputs:
  - output/public/quantum/mass_origin_chi_star_proxy_source_inventory_metrics.json
  - output/public/quantum/mass_origin_chi_star_proxy_source_inventory_rows.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]

RHO_ELIMINATION_AUDIT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_anchor_normalized_g3w_rho_elimination_audit_metrics.json"
CHI_PROXY_INVENTORY_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_anchor_normalized_g3w_chi_proxy_inventory_metrics.json"
CHI_PROXY_AUDIT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_anchor_normalized_g3w_chi_proxy_audit_metrics.json"
REFERENCE_RATIO_AUDIT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_anchor_normalized_g3w_reference_ratio_audit_metrics.json"
G3W_VALUE_RETRY_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_anchor_normalized_g3w_value_retry_metrics.json"
SHAPE_GATE_RETRY_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_anchor_local_shape_gate_retry_refresh_metrics.json"
CHI_STAR_PROXY_CONTRACT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_chi_star_proxy_route_contract_metrics.json"
OUT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_chi_star_proxy_source_inventory_metrics.json"
OUT_CSV = ROOT / "output" / "public" / "quantum" / "mass_origin_chi_star_proxy_source_inventory_rows.csv"
DEFAULT_STEP_TAG = "8.7.55.2.260"


# 関数: `_utc_now_iso` の入出力契約と処理意図を定義する。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_parse_args` の入出力契約と処理意図を定義する。

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inventory current public source candidates for the chi_* or same-sector proxy route.",
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
            "row_id": "chi_star_proxy_source_inventory_complete",
            "status": "pass",
            "metric": "chi_* or same-sector proxy source inventory complete",
            "value": 1.0,
            "note": "This step inventories the concrete public-canonical source candidates for the missing chi_* or same-sector proxy datum.",
        },
        {
            "row_id": "chi_star_proxy_source_inventory_required_count",
            "status": "pass",
            "metric": "required source count for chi_* proxy route",
            "value": float(len(required_sources)),
            "note": f"Required sources: {required_sources}.",
        },
        {
            "row_id": "chi_star_proxy_source_inventory_present_count",
            "status": "pass",
            "metric": "present source count for chi_* proxy route",
            "value": float(len(present_sources)),
            "note": f"Present sources: {present_sources}.",
        },
        {
            "row_id": "chi_star_proxy_source_inventory_missing_count",
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
                "row_id": f"chi_star_proxy_source_{source_name}",
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
            "row_id": "chi_star_proxy_source_inventory_first_route",
            "status": "watch" if first_route_to_close_or_none else "reject",
            "metric": "first residual source to close after chi_* proxy source inventory",
            "value": 1.0 if first_route_to_close_or_none else 0.0,
            "note": (
                f"The next closure attempt starts from {first_route_to_close_or_none}."
                if first_route_to_close_or_none
                else "No preferred source candidate can be chosen from the current chi_* proxy inventory."
            ),
        }
    )
    return rows


# 関数: `_build_payload` の入出力契約と処理意図を定義する。

def _build_payload(step_tag: str) -> Dict[str, Any]:
    for path in (
        RHO_ELIMINATION_AUDIT_JSON,
        CHI_PROXY_INVENTORY_JSON,
        CHI_PROXY_AUDIT_JSON,
        REFERENCE_RATIO_AUDIT_JSON,
        G3W_VALUE_RETRY_JSON,
        SHAPE_GATE_RETRY_JSON,
        CHI_STAR_PROXY_CONTRACT_JSON,
    ):
        _require_path(path)

    rho_elimination_audit = _read_json(RHO_ELIMINATION_AUDIT_JSON)
    chi_proxy_inventory = _read_json(CHI_PROXY_INVENTORY_JSON)
    chi_proxy_audit = _read_json(CHI_PROXY_AUDIT_JSON)
    reference_ratio_audit = _read_json(REFERENCE_RATIO_AUDIT_JSON)
    g3w_value_retry = _read_json(G3W_VALUE_RETRY_JSON)
    shape_gate_retry = _read_json(SHAPE_GATE_RETRY_JSON)
    chi_star_proxy_contract = _read_json(CHI_STAR_PROXY_CONTRACT_JSON)

    rho_elimination_summary = rho_elimination_audit.get("summary", {})
    rho_elimination_evidence = rho_elimination_audit.get("evidence", {})
    source_inventory_summary = rho_elimination_evidence.get("source_inventory_summary", {})
    curvature_summary = rho_elimination_evidence.get("curvature_summary", {})

    chi_proxy_inventory_summary = chi_proxy_inventory.get("summary", {})
    chi_proxy_audit_summary = chi_proxy_audit.get("summary", {})
    reference_ratio_summary = reference_ratio_audit.get("summary", {})
    g3w_value_retry_summary = g3w_value_retry.get("summary", {})
    shape_gate_retry_summary = shape_gate_retry.get("summary", {})
    chi_star_proxy_contract_summary = chi_star_proxy_contract.get("summary", {})
    chi_star_proxy_contract_decision = chi_star_proxy_contract.get("decision", {})

    required_sources = [
        "chi_definition",
        "rho_star_reference_point_symbol",
        "same_sector_contract",
        "no_new_free_parameter_wording",
        "same_sector_equivalence_rule",
        "chi_star_or_same_sector_proxy",
    ]

    present_sources: List[str] = []

    # 条件分岐: `curvature_summary.get("chi_definition_frozen", False)` を満たす経路を評価する。
    if curvature_summary.get("chi_definition_frozen", False):
        present_sources.append("chi_definition")

    # 条件分岐: `bool(rho_elimination_summary.get("reference_point_symbol"))` を満たす経路を評価する。

    if bool(rho_elimination_summary.get("reference_point_symbol")):
        present_sources.append("rho_star_reference_point_symbol")

    chi_proxy_present_sources = [str(item) for item in chi_proxy_inventory_summary.get("present_chi_proxy_sources", [])]

    # 条件分岐: `"same_sector_contract" in chi_proxy_present_sources` を満たす経路を評価する。
    if "same_sector_contract" in chi_proxy_present_sources:
        present_sources.append("same_sector_contract")

    g3w_present_sources = [str(item) for item in source_inventory_summary.get("present_g3w_route_sources", [])]

    # 条件分岐: `"no_new_free_parameter_wording" in g3w_present_sources` を満たす経路を評価する。
    if "no_new_free_parameter_wording" in g3w_present_sources:
        present_sources.append("no_new_free_parameter_wording")

    missing_sources = [item for item in required_sources if item not in present_sources]
    first_route_to_close_or_none = "same_sector_equivalence_rule"
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
            "mass_origin_anchor_normalized_g3w_rho_elimination_audit_json": _relative_str(RHO_ELIMINATION_AUDIT_JSON),
            "mass_origin_anchor_normalized_g3w_chi_proxy_inventory_json": _relative_str(CHI_PROXY_INVENTORY_JSON),
            "mass_origin_anchor_normalized_g3w_chi_proxy_audit_json": _relative_str(CHI_PROXY_AUDIT_JSON),
            "mass_origin_anchor_normalized_g3w_reference_ratio_audit_json": _relative_str(REFERENCE_RATIO_AUDIT_JSON),
            "mass_origin_anchor_normalized_g3w_value_retry_json": _relative_str(G3W_VALUE_RETRY_JSON),
            "mass_origin_anchor_local_shape_gate_retry_refresh_json": _relative_str(SHAPE_GATE_RETRY_JSON),
            "mass_origin_chi_star_proxy_route_contract_json": _relative_str(CHI_STAR_PROXY_CONTRACT_JSON),
        },
        "intent": "Inventory the current public-canonical source candidates for the missing chi_* or same-sector proxy datum inside the preferred anchor-normalized g_3w route.",
        "formulas": {
            "inventory_rule": "the chi_* proxy route can close only after the public pack exposes chi-space language, the rho_* reference point, the same-sector contract, the no-new-free-parameter envelope, and either a same-sector equivalence rule or an explicit chi_* proxy datum",
            "proxy_candidate_rule": "a same-sector proxy can replace chi_* only if the public pack states an explicit equivalence rule without adding a new fit parameter",
            "current_absence": "the current public pack already freezes chi definition, rho_* reference-point language, the same-sector contract, and the no-new-free-parameter envelope, but it does not yet expose the same-sector equivalence rule or the anchor-coordinate datum itself",
        },
        "rows": rows,
        "summary": {
            "required_proxy_route_sources": required_sources,
            "present_proxy_route_sources": present_sources,
            "missing_proxy_route_sources": missing_sources,
            "first_route_to_close_or_none": first_route_to_close_or_none,
            "proxy_source_inventory_ready": True,
        },
        "decision": {
            "overall_status": "chi_star_proxy_source_inventory_frozen",
            "keep_mass_origin_branch_blocked": True,
            "missing_anchor_coordinate_datum": chi_star_proxy_contract_summary.get("missing_anchor_coordinate_datum"),
            "required_proxy_route_sources": required_sources,
            "present_proxy_route_sources": present_sources,
            "missing_proxy_route_sources": missing_sources,
            "first_route_to_close_or_none": first_route_to_close_or_none,
            "proxy_source_inventory_ready": True,
            "hand_off_to_8_7_55_2_83": bool(chi_star_proxy_contract_decision.get("hand_off_to_8_7_55_2_83", False)),
            "next_required_artifacts": [
                "same_sector_equivalence_rule",
                "chi_star_or_same_sector_proxy",
                "anchor_normalized_g3w_public_value",
                "r3_target",
                "single_public_vpp_shape",
                "positive_particle_sector_chi_p_to_vpp_public_artifact",
                "solver_ready_row_promoted_to_pass",
            ],
        },
        "evidence": {
            "rho_elimination_summary": rho_elimination_summary,
            "curvature_summary": curvature_summary,
            "g3w_source_inventory_summary": source_inventory_summary,
            "chi_proxy_inventory_summary": chi_proxy_inventory_summary,
            "chi_proxy_audit_summary": chi_proxy_audit_summary,
            "reference_ratio_summary": reference_ratio_summary,
            "g3w_value_retry_summary": g3w_value_retry_summary,
            "shape_gate_retry_summary": shape_gate_retry_summary,
            "chi_star_proxy_contract_summary": chi_star_proxy_contract_summary,
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
