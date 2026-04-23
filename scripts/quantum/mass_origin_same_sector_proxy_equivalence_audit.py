#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_origin_same_sector_proxy_equivalence_audit.py

Step 8.7.55.2.261:
Audit whether the current public canonical pack already fixes a same-sector
proxy equivalence rule for the missing anchor-coordinate datum without a new
fit parameter.

Inputs:
  - output/public/quantum/mass_origin_chi_star_proxy_source_inventory_metrics.json

Outputs:
  - output/public/quantum/mass_origin_same_sector_proxy_equivalence_audit_metrics.json
  - output/public/quantum/mass_origin_same_sector_proxy_equivalence_audit_rows.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]

SOURCE_INVENTORY_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_chi_star_proxy_source_inventory_metrics.json"
OUT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_same_sector_proxy_equivalence_audit_metrics.json"
OUT_CSV = ROOT / "output" / "public" / "quantum" / "mass_origin_same_sector_proxy_equivalence_audit_rows.csv"
DEFAULT_STEP_TAG = "8.7.55.2.261"


# 関数: `_utc_now_iso` の入出力契約と処理意図を定義する。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_parse_args` の入出力契約と処理意図を定義する。

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit whether a same-sector proxy equivalence rule is already public canonical.",
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
    same_sector_proxy_rule_available: bool,
    same_sector_proxy_without_new_free_parameters: bool,
    missing_inputs: List[str],
) -> List[Dict[str, Any]]:
    return [
        {
            "row_id": "same_sector_proxy_equivalence_audit_complete",
            "status": "pass",
            "metric": "same-sector proxy equivalence audit complete",
            "value": 1.0,
            "note": "This step audits whether a same-sector proxy equivalence rule is already public canonical.",
        },
        {
            "row_id": "same_sector_proxy_rule_available",
            "status": "pass" if same_sector_proxy_rule_available else "reject",
            "metric": "same-sector proxy equivalence rule available",
            "value": 1.0 if same_sector_proxy_rule_available else 0.0,
            "note": (
                "A same-sector proxy equivalence rule is now public canonical."
                if same_sector_proxy_rule_available
                else f"Missing inputs: {missing_inputs}."
            ),
        },
        {
            "row_id": "same_sector_proxy_without_new_free_parameters",
            "status": "pass" if same_sector_proxy_without_new_free_parameters else "reject",
            "metric": "same-sector proxy equivalence rule stays inside no-new-free-parameter envelope",
            "value": 1.0 if same_sector_proxy_without_new_free_parameters else 0.0,
            "note": (
                "The same-sector proxy equivalence rule closes without a new fit parameter."
                if same_sector_proxy_without_new_free_parameters
                else "The current public pack still lacks an explicit same-sector equivalence rule."
            ),
        },
    ]


# 関数: `_build_payload` の入出力契約と処理意図を定義する。

def _build_payload(step_tag: str) -> Dict[str, Any]:
    _require_path(SOURCE_INVENTORY_JSON)

    source_inventory = _read_json(SOURCE_INVENTORY_JSON)
    source_summary = source_inventory.get("summary", {})
    present_sources = [str(item) for item in source_summary.get("present_proxy_route_sources", [])]
    missing_sources = [str(item) for item in source_summary.get("missing_proxy_route_sources", [])]

    same_sector_equivalence_rule_available = "same_sector_equivalence_rule" not in missing_sources
    no_new_free_parameter_wording_available = "no_new_free_parameter_wording" in present_sources
    same_sector_proxy_rule_available = same_sector_equivalence_rule_available
    same_sector_proxy_kind_or_none = (
        "same_sector_proxy_equivalence_rule" if same_sector_proxy_rule_available else None
    )
    same_sector_proxy_without_new_free_parameters = bool(
        same_sector_proxy_rule_available and no_new_free_parameter_wording_available
    )
    missing_inputs = []

    # 条件分岐: `not same_sector_equivalence_rule_available` を満たす経路を評価する。
    if not same_sector_equivalence_rule_available:
        missing_inputs.append("same_sector_equivalence_rule")

    rows = _build_rows(
        same_sector_proxy_rule_available=same_sector_proxy_rule_available,
        same_sector_proxy_without_new_free_parameters=same_sector_proxy_without_new_free_parameters,
        missing_inputs=missing_inputs,
    )

    return {
        "generated_utc": _utc_now_iso(),
        "phase": {
            "phase": 8,
            "step": step_tag,
            "name": "same-sector proxy equivalence audit",
        },
        "inputs": {
            "mass_origin_chi_star_proxy_source_inventory_json": _relative_str(SOURCE_INVENTORY_JSON),
        },
        "intent": "Audit whether the current public canonical pack already fixes a same-sector proxy equivalence rule for the missing anchor-coordinate datum.",
        "formulas": {
            "equivalence_rule": "same_sector_proxy_rule_available iff the public pack explicitly states a no-new-free-parameter equivalence between chi_* and a same-sector proxy coordinate",
            "current_absence": "the current source inventory still lacks the same-sector equivalence rule itself, so the proxy route cannot yet promote a canonical anchor-coordinate replacement",
        },
        "rows": rows,
        "summary": {
            "same_sector_proxy_rule_available": same_sector_proxy_rule_available,
            "same_sector_proxy_kind_or_none": same_sector_proxy_kind_or_none,
            "same_sector_proxy_without_new_free_parameters": same_sector_proxy_without_new_free_parameters,
            "same_sector_proxy_missing_inputs": missing_inputs,
        },
        "decision": {
            "overall_status": "same_sector_proxy_equivalence_audit_frozen_absent",
            "keep_mass_origin_branch_blocked": True,
            "same_sector_proxy_rule_available": same_sector_proxy_rule_available,
            "same_sector_proxy_kind_or_none": same_sector_proxy_kind_or_none,
            "same_sector_proxy_without_new_free_parameters": same_sector_proxy_without_new_free_parameters,
            "hand_off_to_8_7_55_2_83": False,
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
            "chi_star_proxy_source_inventory_summary": source_summary,
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
