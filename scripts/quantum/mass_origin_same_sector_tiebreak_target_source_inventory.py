#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_origin_same_sector_tiebreak_target_source_inventory.py

Step 8.7.55.2.92:
Inventory admissible public source rows for the missing same-sector target
value of the derivative-ratio tie-break invariant.

Inputs:
  - output/public/quantum/mass_origin_same_sector_chi_to_vpp_contract_metrics.json
  - output/public/quantum/mass_origin_same_sector_tiebreak_target_bridge_metrics.json
  - output/public/quantum/mass_origin_same_sector_tiebreak_target_source_contract_metrics.json

Outputs:
  - output/public/quantum/mass_origin_same_sector_tiebreak_target_source_inventory_metrics.json
  - output/public/quantum/mass_origin_same_sector_tiebreak_target_source_inventory_rows.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]

CHI_CONTRACT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_same_sector_chi_to_vpp_contract_metrics.json"
TARGET_BRIDGE_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_same_sector_tiebreak_target_bridge_metrics.json"
SOURCE_CONTRACT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_same_sector_tiebreak_target_source_contract_metrics.json"
OUT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_same_sector_tiebreak_target_source_inventory_metrics.json"
OUT_CSV = ROOT / "output" / "public" / "quantum" / "mass_origin_same_sector_tiebreak_target_source_inventory_rows.csv"
DEFAULT_STEP_TAG = "8.7.55.2.92"


# 関数: `_utc_now_iso` の入出力契約と処理意図を定義する。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_parse_args` の入出力契約と処理意図を定義する。

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inventory admissible public source rows for the same-sector tie-break target value.",
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
    for path in (CHI_CONTRACT_JSON, TARGET_BRIDGE_JSON, SOURCE_CONTRACT_JSON):
        _require_path(path)

    chi_contract = _read_json(CHI_CONTRACT_JSON)
    target_bridge = _read_json(TARGET_BRIDGE_JSON)
    source_contract = _read_json(SOURCE_CONTRACT_JSON)

    chi_summary = chi_contract.get("summary", {})
    bridge_summary = target_bridge.get("summary", {})
    source_contract_summary = source_contract.get("summary", {})

    shell_row_ids = list(chi_summary.get("existing_shell_family_row_ids", []))
    candidate_source_row_ids = ["explicit_mapping_equation"] + shell_row_ids
    candidate_source_kind_ids = list(source_contract_summary.get("allowed_source_kind_ids", []))
    candidate_source_kind_by_row_id = {
        "explicit_mapping_equation": "explicit_mapping_equation",
        **{row_id: "surviving_shell_anchor_pack" for row_id in shell_row_ids},
    }
    candidate_source_count = len(candidate_source_row_ids)
    candidate_source_present_count = len(shell_row_ids)
    currently_derived_target_value = bool(bridge_summary.get("target_value_available", False))
    inventory_status = (
        "same_sector_tiebreak_target_source_inventory_currently_derived"
        if currently_derived_target_value
        else "same_sector_tiebreak_target_source_inventory_frozen_target_missing"
    )

    rows = [
        {
            "row_id": "same_sector_tiebreak_target_source_inventory_complete",
            "status": "pass",
            "metric": "same-sector tie-break target-source inventory complete",
            "value": 1.0,
            "source_kind": "aggregate",
            "source_row_id": "aggregate",
            "note": "This step inventories admissible public source rows only; derivability is deferred to the next audits.",
        },
        {
            "row_id": "same_sector_tiebreak_target_source_explicit_mapping_equation_candidate",
            "status": "watch",
            "metric": "explicit mapping equation remains an admissible source candidate",
            "value": 0.0,
            "source_kind": "explicit_mapping_equation",
            "source_row_id": "explicit_mapping_equation",
            "note": "The source contract allows an explicit same-sector mapping equation, but the row is not present yet in the public canonical pack.",
        },
        {
            "row_id": "same_sector_tiebreak_target_source_shell_quantization_fit_kappa",
            "status": "inventory",
            "metric": "shell_quantization_fit_kappa listed as shell-anchor source candidate",
            "value": 1.0,
            "source_kind": "surviving_shell_anchor_pack",
            "source_row_id": "shell_quantization_fit_kappa",
            "note": "This row is present in the same-sector contract evidence and remains an admissible shell-anchor source candidate.",
        },
        {
            "row_id": "same_sector_tiebreak_target_source_shell_quantization_fit_kz_over_kn",
            "status": "inventory",
            "metric": "shell_quantization_fit_kz_over_kn listed as shell-anchor source candidate",
            "value": 1.0,
            "source_kind": "surviving_shell_anchor_pack",
            "source_row_id": "shell_quantization_fit_kz_over_kn",
            "note": "This row is present in the same-sector contract evidence and remains an admissible shell-anchor source candidate.",
        },
        {
            "row_id": "same_sector_tiebreak_target_source_candidate_count",
            "status": "inventory",
            "metric": "candidate source row count",
            "value": float(candidate_source_count),
            "source_kind": "aggregate",
            "source_row_id": "aggregate",
            "note": f"Candidate source rows are {candidate_source_row_ids}.",
        },
        {
            "row_id": "same_sector_tiebreak_target_source_present_count",
            "status": "inventory",
            "metric": "present source row count",
            "value": float(candidate_source_present_count),
            "source_kind": "aggregate",
            "source_row_id": "aggregate",
            "note": f"Present rows in the current public pack are {shell_row_ids}; the explicit mapping equation remains absent.",
        },
        {
            "row_id": "same_sector_tiebreak_target_source_current_target_value_available",
            "status": "pass" if currently_derived_target_value else "watch",
            "metric": "current target value already derivable from inventoried sources",
            "value": 1.0 if currently_derived_target_value else 0.0,
            "source_kind": "aggregate",
            "source_row_id": str(bridge_summary.get("target_source_kind_or_none") or "none"),
            "note": (
                f"Current target source kind is {bridge_summary.get('target_source_kind_or_none')}."
                if currently_derived_target_value
                else "The inventory is frozen, but no inventoried source currently closes the target-value bridge."
            ),
        },
    ]

    return {
        "generated_utc": _utc_now_iso(),
        "phase": {
            "phase": 8,
            "step": step_tag,
            "name": "same-sector tie-break target-source inventory",
        },
        "inputs": {
            "mass_origin_same_sector_chi_to_vpp_contract_json": _relative_str(CHI_CONTRACT_JSON),
            "mass_origin_same_sector_tiebreak_target_bridge_json": _relative_str(TARGET_BRIDGE_JSON),
            "mass_origin_same_sector_tiebreak_target_source_contract_json": _relative_str(SOURCE_CONTRACT_JSON),
        },
        "intent": "List admissible public source rows for the missing same-sector target value of the derivative-ratio invariant.",
        "formulas": {
            "inventory_rule": "candidate_source_row_ids are the public rows or row placeholders whose source kind is allowed by the target-source contract",
            "inventory_scope_rule": "inventory does not imply derivability; it only enumerates admissible public source candidates before shell-anchor and explicit-mapping audits",
        },
        "rows": rows,
        "summary": {
            "candidate_source_row_ids": candidate_source_row_ids,
            "candidate_source_kind_ids": candidate_source_kind_ids,
            "candidate_source_kind_by_row_id": candidate_source_kind_by_row_id,
            "candidate_source_count": candidate_source_count,
            "candidate_source_present_count": candidate_source_present_count,
            "inventory_status": inventory_status,
            "current_target_value_available": currently_derived_target_value,
            "current_target_source_kind_or_none": bridge_summary.get("target_source_kind_or_none"),
        },
        "decision": {
            "overall_status": inventory_status,
            "keep_mass_origin_branch_blocked": True,
            "candidate_source_row_ids": candidate_source_row_ids,
            "candidate_source_kind_ids": candidate_source_kind_ids,
            "candidate_source_count": candidate_source_count,
            "inventory_status": inventory_status,
            "current_target_value_available": currently_derived_target_value,
        },
        "evidence": {
            "chi_contract_summary": chi_summary,
            "target_bridge_summary": bridge_summary,
            "target_source_contract_summary": source_contract_summary,
        },
    }


# 関数: `_write_csv` の入出力契約と処理意図を定義する。

def _write_csv(rows: List[Dict[str, Any]]) -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["row_id", "status", "metric", "value", "source_kind", "source_row_id", "note"],
        )
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
