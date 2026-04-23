#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_origin_same_sector_mapping_equation_source_audit.py

Step 8.7.55.2.94:
Audit whether an explicit same-sector chi_P -> V''(|P|_*) mapping equation
can already be promoted into public canonical form from the current public
row pack without introducing new free parameters.

Inputs:
  - output/public/quantum/mass_origin_same_sector_chi_to_vpp_contract_metrics.json
  - output/public/quantum/mass_origin_positive_particle_sector_chi_to_vpp_metrics.json
  - output/public/quantum/mass_origin_same_sector_tiebreak_target_source_contract_metrics.json
  - output/public/quantum/mass_origin_same_sector_tiebreak_target_source_inventory_metrics.json

Outputs:
  - output/public/quantum/mass_origin_same_sector_mapping_equation_source_metrics.json
  - output/public/quantum/mass_origin_same_sector_mapping_equation_source_rows.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]

CONTRACT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_same_sector_chi_to_vpp_contract_metrics.json"
PROMOTION_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_positive_particle_sector_chi_to_vpp_metrics.json"
SOURCE_CONTRACT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_same_sector_tiebreak_target_source_contract_metrics.json"
SOURCE_INVENTORY_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_same_sector_tiebreak_target_source_inventory_metrics.json"
OUT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_same_sector_mapping_equation_source_metrics.json"
OUT_CSV = ROOT / "output" / "public" / "quantum" / "mass_origin_same_sector_mapping_equation_source_rows.csv"
DEFAULT_STEP_TAG = "8.7.55.2.94"


# 関数: `_utc_now_iso` の入出力契約と処理意図を定義する。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_parse_args` の入出力契約と処理意図を定義する。

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit whether an explicit same-sector chi_P -> V''(|P|_*) mapping equation is already promotable.",
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


# 関数: `_build_payload` の入出力契約と処理意図を定義する。

def _build_payload(step_tag: str) -> Dict[str, Any]:
    for path in (CONTRACT_JSON, PROMOTION_JSON, SOURCE_CONTRACT_JSON, SOURCE_INVENTORY_JSON):
        _require_path(path)

    contract = _read_json(CONTRACT_JSON)
    promotion = _read_json(PROMOTION_JSON)
    source_contract = _read_json(SOURCE_CONTRACT_JSON)
    source_inventory = _read_json(SOURCE_INVENTORY_JSON)

    contract_summary = contract.get("summary", {})
    promotion_summary = promotion.get("summary", {})
    source_contract_summary = source_contract.get("summary", {})
    source_inventory_summary = source_inventory.get("summary", {})

    allowed_source_kind_ids = [str(item) for item in source_contract_summary.get("allowed_source_kind_ids", [])]
    candidate_source_row_ids = [str(item) for item in source_inventory_summary.get("candidate_source_row_ids", [])]
    explicit_mapping_candidate_listed = "explicit_mapping_equation" in candidate_source_row_ids
    explicit_mapping_source_kind_allowed = "explicit_mapping_equation" in allowed_source_kind_ids
    explicit_mapping_equation_available = bool(contract_summary.get("explicit_mapping_equation_available", False))
    promotion_explicit_mapping_equation_present = bool(promotion_summary.get("explicit_mapping_equation_present", False))
    same_particle_sector_only = bool(contract_summary.get("same_particle_sector_only", False))
    mapping_without_new_free_parameters = bool(
        explicit_mapping_equation_available
        and same_particle_sector_only
        and source_contract_summary.get("bridge_without_new_free_parameters_required", False)
    )
    mapping_source_kind_or_none = "explicit_mapping_equation" if explicit_mapping_equation_available else None
    missing_mapping_requirements: List[str] = []

    # 条件分岐: `not explicit_mapping_source_kind_allowed` を満たす経路を評価する。
    if not explicit_mapping_source_kind_allowed:
        missing_mapping_requirements.append("explicit_mapping_source_kind_not_allowed")

    # 条件分岐: `not explicit_mapping_candidate_listed` を満たす経路を評価する。

    if not explicit_mapping_candidate_listed:
        missing_mapping_requirements.append("explicit_mapping_equation_not_in_inventory")

    # 条件分岐: `not explicit_mapping_equation_available` を満たす経路を評価する。

    if not explicit_mapping_equation_available:
        missing_mapping_requirements.append("explicit_mapping_equation")

    rows = [
        {
            "row_id": "same_sector_mapping_equation_source_audit_complete",
            "status": "pass",
            "metric": "explicit mapping-equation source audit complete",
            "value": 1.0,
            "note": "This step isolates the explicit mapping-equation route from the broader target-source branch and checks whether the current public row pack already supports promotion.",
        },
        {
            "row_id": "same_sector_mapping_equation_source_kind_allowed",
            "status": "pass" if explicit_mapping_source_kind_allowed else "reject",
            "metric": "explicit mapping equation remains an allowed source kind",
            "value": 1.0 if explicit_mapping_source_kind_allowed else 0.0,
            "note": f"Allowed source kinds are {allowed_source_kind_ids}.",
        },
        {
            "row_id": "same_sector_mapping_equation_candidate_listed",
            "status": "pass" if explicit_mapping_candidate_listed else "reject",
            "metric": "explicit mapping equation candidate is listed in the source inventory",
            "value": 1.0 if explicit_mapping_candidate_listed else 0.0,
            "note": f"Candidate source rows are {candidate_source_row_ids}.",
        },
        {
            "row_id": "same_sector_mapping_equation_present_in_contract",
            "status": "pass" if explicit_mapping_equation_available else "missing",
            "metric": "explicit same-sector chi_P -> V''(|P|_*) mapping equation already available",
            "value": 1.0 if explicit_mapping_equation_available else 0.0,
            "note": (
                "The same-sector contract already contains a public canonical mapping equation."
                if explicit_mapping_equation_available
                else "The same-sector contract still freezes the mapping equation as absent; only the placeholder route is admissible."
            ),
        },
        {
            "row_id": "same_sector_mapping_equation_present_in_promotion_stack",
            "status": "pass" if promotion_explicit_mapping_equation_present else "missing",
            "metric": "promotion stack sees the explicit mapping equation as present",
            "value": 1.0 if promotion_explicit_mapping_equation_present else 0.0,
            "note": (
                "The existing positive-particle-sector promotion artifact already sees the mapping equation as present."
                if promotion_explicit_mapping_equation_present
                else "The existing positive-particle-sector promotion artifact still records the explicit mapping equation as missing."
            ),
        },
        {
            "row_id": "same_sector_mapping_equation_without_new_free_parameters",
            "status": "pass" if mapping_without_new_free_parameters else "reject",
            "metric": "explicit mapping equation route closes without new free parameters",
            "value": 1.0 if mapping_without_new_free_parameters else 0.0,
            "note": (
                "The explicit mapping equation is public and satisfies the no-new-free-parameter contract."
                if mapping_without_new_free_parameters
                else "The route cannot satisfy the no-new-free-parameter requirement because the explicit mapping equation itself is not public canonical yet."
            ),
        },
        {
            "row_id": "same_sector_mapping_equation_route_available",
            "status": "pass" if explicit_mapping_equation_available else "watch",
            "metric": "explicit mapping-equation route is available for target closure",
            "value": 1.0 if explicit_mapping_equation_available else 0.0,
            "note": (
                "The target-source branch may now use the explicit mapping equation route."
                if explicit_mapping_equation_available
                else f"The explicit mapping-equation route remains non-closing because the missing requirements are {missing_mapping_requirements}."
            ),
        },
    ]

    return {
        "generated_utc": _utc_now_iso(),
        "phase": {
            "phase": 8,
            "step": step_tag,
            "name": "explicit mapping-equation source audit",
        },
        "inputs": {
            "mass_origin_same_sector_chi_to_vpp_contract_json": _relative_str(CONTRACT_JSON),
            "mass_origin_positive_particle_sector_chi_to_vpp_json": _relative_str(PROMOTION_JSON),
            "mass_origin_same_sector_tiebreak_target_source_contract_json": _relative_str(SOURCE_CONTRACT_JSON),
            "mass_origin_same_sector_tiebreak_target_source_inventory_json": _relative_str(SOURCE_INVENTORY_JSON),
        },
        "intent": "Determine whether the current public canonical rows already support an explicit same-sector chi_P -> V''(|P|_*) mapping equation without introducing new free parameters.",
        "formulas": {
            "mapping_route_rule": "explicit_mapping_equation_available iff the same-sector contract and promotion stack both expose the explicit chi_P -> V''(|P|_*) equation as a public canonical row",
            "mapping_no_new_parameter_rule": "mapping_without_new_free_parameters iff the explicit mapping equation is available and still satisfies the same-sector no-new-free-parameter contract",
        },
        "rows": rows,
        "summary": {
            "explicit_mapping_source_kind_allowed": explicit_mapping_source_kind_allowed,
            "explicit_mapping_candidate_listed": explicit_mapping_candidate_listed,
            "explicit_mapping_equation_available": explicit_mapping_equation_available,
            "mapping_source_kind_or_none": mapping_source_kind_or_none,
            "mapping_without_new_free_parameters": mapping_without_new_free_parameters,
            "missing_mapping_requirements": missing_mapping_requirements,
            "tiebreak_invariant_name": source_contract_summary.get("tiebreak_invariant_name"),
        },
        "decision": {
            "overall_status": (
                "same_sector_mapping_equation_source_frozen_available"
                if explicit_mapping_equation_available
                else "same_sector_mapping_equation_source_frozen_absent"
            ),
            "keep_mass_origin_branch_blocked": True,
            "explicit_mapping_equation_available": explicit_mapping_equation_available,
            "mapping_source_kind_or_none": mapping_source_kind_or_none,
            "mapping_without_new_free_parameters": mapping_without_new_free_parameters,
            "missing_mapping_requirements": missing_mapping_requirements,
        },
        "evidence": {
            "contract_summary": contract_summary,
            "promotion_summary": promotion_summary,
            "target_source_contract_summary": source_contract_summary,
            "target_source_inventory_summary": source_inventory_summary,
        },
    }


# 関数: `_write_csv` の入出力契約と処理意図を定義する。

def _write_csv(rows: List[Dict[str, Any]]) -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
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
