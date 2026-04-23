#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_origin_same_sector_tiebreak_target_source_contract.py

Step 8.7.55.2.91:
Freeze the public-contract constraints for any future same-sector target value
of the derivative-ratio tie-break invariant.

Inputs:
  - output/public/quantum/mass_origin_same_sector_chi_to_vpp_contract_metrics.json
  - output/public/quantum/mass_origin_same_sector_vpp_tiebreak_invariant_metrics.json
  - output/public/quantum/mass_origin_same_sector_tiebreak_target_bridge_metrics.json
  - output/public/quantum/mass_origin_tiebreak_branch_disposition_metrics.json

Outputs:
  - output/public/quantum/mass_origin_same_sector_tiebreak_target_source_contract_metrics.json
  - output/public/quantum/mass_origin_same_sector_tiebreak_target_source_contract_rows.csv
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
TIEBREAK_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_same_sector_vpp_tiebreak_invariant_metrics.json"
BRIDGE_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_same_sector_tiebreak_target_bridge_metrics.json"
DISPOSITION_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_tiebreak_branch_disposition_metrics.json"
OUT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_same_sector_tiebreak_target_source_contract_metrics.json"
OUT_CSV = ROOT / "output" / "public" / "quantum" / "mass_origin_same_sector_tiebreak_target_source_contract_rows.csv"
DEFAULT_STEP_TAG = "8.7.55.2.91"


# 関数: `_utc_now_iso` の入出力契約と処理意図を定義する。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_parse_args` の入出力契約と処理意図を定義する。

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze the same-sector target-source contract for the tie-break invariant.",
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
    for path in (CONTRACT_JSON, TIEBREAK_JSON, BRIDGE_JSON, DISPOSITION_JSON):
        _require_path(path)

    contract = _read_json(CONTRACT_JSON)
    tiebreak = _read_json(TIEBREAK_JSON)
    bridge = _read_json(BRIDGE_JSON)
    disposition = _read_json(DISPOSITION_JSON)

    contract_summary = contract.get("summary", {})
    tiebreak_summary = tiebreak.get("summary", {})
    bridge_summary = bridge.get("summary", {})
    disposition_summary = disposition.get("summary", {})

    allowed_source_kind_ids = [
        "explicit_mapping_equation",
        "surviving_shell_anchor_pack",
    ]
    forbidden_source_kind_ids = [
        "cross_sector_proxy",
        "interface_only_spread",
        "phenomenological_backsolve",
    ]
    current_target_value_available = bool(bridge_summary.get("target_value_available", False))
    current_target_source_kind_or_none = bridge_summary.get("target_source_kind_or_none")

    rows = [
        {
            "row_id": "same_sector_tiebreak_target_source_contract_complete",
            "status": "pass",
            "metric": "same-sector tie-break target-source contract complete",
            "value": 1.0,
            "note": "This contract freezes which public source classes are admissible for any future target value of the derivative-ratio invariant.",
        },
        {
            "row_id": "same_sector_tiebreak_target_source_same_particle_sector_only",
            "status": "pass" if contract_summary.get("same_particle_sector_only", False) else "reject",
            "metric": "same-particle-sector only",
            "value": 1.0 if contract_summary.get("same_particle_sector_only", False) else 0.0,
            "note": "Any future target value must stay inside the same particle sector and may not be satisfied by interface-only or cross-sector substitutes.",
        },
        {
            "row_id": "same_sector_tiebreak_target_source_dimensionless_required",
            "status": "pass",
            "metric": "target value must remain dimensionless",
            "value": 1.0,
            "note": "The target value belongs to the invariant absP_star_times_vppp_over_vpp and therefore must remain dimensionless at the public-contract layer.",
        },
        {
            "row_id": "same_sector_tiebreak_target_source_allowed_kind_count",
            "status": "inventory",
            "metric": "allowed public source kind count",
            "value": float(len(allowed_source_kind_ids)),
            "note": f"Allowed source kinds are {allowed_source_kind_ids}.",
        },
        {
            "row_id": "same_sector_tiebreak_target_source_forbidden_kind_count",
            "status": "inventory",
            "metric": "forbidden source kind count",
            "value": float(len(forbidden_source_kind_ids)),
            "note": f"Forbidden source kinds are {forbidden_source_kind_ids}.",
        },
        {
            "row_id": "same_sector_tiebreak_target_source_no_new_free_parameters_required",
            "status": "pass",
            "metric": "no-new-free-parameter bridge required",
            "value": 1.0,
            "note": "A future target value is admissible only if it comes entirely from already-frozen same-sector ingredients without introducing a new fit parameter.",
        },
        {
            "row_id": "same_sector_tiebreak_target_source_current_value_available",
            "status": "pass" if current_target_value_available else "watch",
            "metric": "current target value already available",
            "value": 1.0 if current_target_value_available else 0.0,
            "note": (
                f"Current target source kind is {current_target_source_kind_or_none}."
                if current_target_value_available
                else "No admissible target value is currently available; the next branch must derive it from one of the allowed source kinds."
            ),
        },
    ]

    return {
        "generated_utc": _utc_now_iso(),
        "phase": {
            "phase": 8,
            "step": step_tag,
            "name": "same-sector tie-break target-source contract",
        },
        "inputs": {
            "mass_origin_same_sector_chi_to_vpp_contract_json": _relative_str(CONTRACT_JSON),
            "mass_origin_same_sector_vpp_tiebreak_invariant_json": _relative_str(TIEBREAK_JSON),
            "mass_origin_same_sector_tiebreak_target_bridge_json": _relative_str(BRIDGE_JSON),
            "mass_origin_tiebreak_branch_disposition_json": _relative_str(DISPOSITION_JSON),
        },
        "intent": "Freeze the admissible public source classes for any future same-sector target value of the tie-break invariant.",
        "formulas": {
            "source_contract_rule": "same_sector_tiebreak_target_value may come only from an explicit same-sector mapping equation or from the surviving shell anchor pack",
            "dimensionless_rule": "the target value must stay dimensionless because it belongs to absP_star_times_vppp_over_vpp",
            "no_new_parameter_rule": "the target source is admissible only if it introduces no new free parameters",
        },
        "rows": rows,
        "summary": {
            "same_particle_sector_only": bool(contract_summary.get("same_particle_sector_only", False)),
            "tiebreak_invariant_name": tiebreak_summary.get("tiebreak_invariant_name"),
            "target_value_dimensionless_required": True,
            "allowed_source_kind_ids": allowed_source_kind_ids,
            "allowed_source_kind_count": len(allowed_source_kind_ids),
            "forbidden_source_kind_ids": forbidden_source_kind_ids,
            "bridge_without_new_free_parameters_required": True,
            "current_target_value_available": current_target_value_available,
            "current_target_source_kind_or_none": current_target_source_kind_or_none,
            "remaining_missing_artifacts": disposition_summary.get("remaining_missing_artifacts", []),
        },
        "decision": {
            "overall_status": "same_sector_tiebreak_target_source_contract_frozen",
            "keep_mass_origin_branch_blocked": True,
            "same_particle_sector_only": bool(contract_summary.get("same_particle_sector_only", False)),
            "allowed_source_kind_ids": allowed_source_kind_ids,
            "forbidden_source_kind_ids": forbidden_source_kind_ids,
            "current_target_value_available": current_target_value_available,
            "current_target_source_kind_or_none": current_target_source_kind_or_none,
            "remaining_missing_artifacts": disposition_summary.get("remaining_missing_artifacts", []),
        },
        "evidence": {
            "contract_summary": contract_summary,
            "tiebreak_summary": tiebreak_summary,
            "bridge_summary": bridge_summary,
            "disposition_summary": disposition_summary,
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
