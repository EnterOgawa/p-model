#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_origin_chi_star_proxy_closure_retry.py

Step 8.7.55.2.262:
Retry the closure of the missing anchor-coordinate datum after the same-sector
proxy equivalence audit.

Inputs:
  - output/public/quantum/mass_origin_anchor_normalized_g3w_chi_proxy_audit_metrics.json
  - output/public/quantum/mass_origin_chi_star_proxy_source_inventory_metrics.json
  - output/public/quantum/mass_origin_same_sector_proxy_equivalence_audit_metrics.json

Outputs:
  - output/public/quantum/mass_origin_chi_star_proxy_closure_retry_metrics.json
  - output/public/quantum/mass_origin_chi_star_proxy_closure_retry_rows.csv
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
SOURCE_INVENTORY_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_chi_star_proxy_source_inventory_metrics.json"
PROXY_EQUIVALENCE_AUDIT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_same_sector_proxy_equivalence_audit_metrics.json"
OUT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_chi_star_proxy_closure_retry_metrics.json"
OUT_CSV = ROOT / "output" / "public" / "quantum" / "mass_origin_chi_star_proxy_closure_retry_rows.csv"
DEFAULT_STEP_TAG = "8.7.55.2.262"


# 関数: `_utc_now_iso` の入出力契約と処理意図を定義する。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_parse_args` の入出力契約と処理意図を定義する。

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retry the chi_* or same-sector proxy closure after the equivalence audit.",
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
    chi_star_or_same_sector_proxy_available: bool,
    nonclosure_reason: str | None,
) -> List[Dict[str, Any]]:
    return [
        {
            "row_id": "chi_star_proxy_closure_retry_complete",
            "status": "pass",
            "metric": "chi_* or same-sector proxy closure retry complete",
            "value": 1.0,
            "note": "This step retries the closure of the missing anchor-coordinate datum after the same-sector proxy audit.",
        },
        {
            "row_id": "chi_star_proxy_closure_retry_available",
            "status": "pass" if chi_star_or_same_sector_proxy_available else "reject",
            "metric": "chi_* or same-sector proxy available after retry",
            "value": 1.0 if chi_star_or_same_sector_proxy_available else 0.0,
            "note": (
                "The missing anchor-coordinate datum is now public canonical."
                if chi_star_or_same_sector_proxy_available
                else f"The retry remains non-closing: {nonclosure_reason}."
            ),
        },
    ]


# 関数: `_build_payload` の入出力契約と処理意図を定義する。

def _build_payload(step_tag: str) -> Dict[str, Any]:
    for path in (CHI_PROXY_AUDIT_JSON, SOURCE_INVENTORY_JSON, PROXY_EQUIVALENCE_AUDIT_JSON):
        _require_path(path)

    chi_proxy_audit = _read_json(CHI_PROXY_AUDIT_JSON)
    source_inventory = _read_json(SOURCE_INVENTORY_JSON)
    proxy_equivalence_audit = _read_json(PROXY_EQUIVALENCE_AUDIT_JSON)

    chi_proxy_summary = chi_proxy_audit.get("summary", {})
    source_summary = source_inventory.get("summary", {})
    proxy_equivalence_summary = proxy_equivalence_audit.get("summary", {})

    missing_sources = [str(item) for item in source_summary.get("missing_proxy_route_sources", [])]
    same_sector_proxy_rule_available = bool(proxy_equivalence_summary.get("same_sector_proxy_rule_available", False))
    anchor_coordinate_datum_available = "chi_star_or_same_sector_proxy" not in missing_sources
    chi_star_or_same_sector_proxy_available = bool(
        same_sector_proxy_rule_available and anchor_coordinate_datum_available
    )
    nonclosure_reason = None

    # 条件分岐: `not same_sector_proxy_rule_available` を満たす経路を評価する。
    if not same_sector_proxy_rule_available:
        nonclosure_reason = "same_sector_equivalence_rule_absent"

    # 条件分岐: `same_sector_proxy_rule_available and not anchor_coordinate_datum_available` を満たす経路を評価する。

    if same_sector_proxy_rule_available and not anchor_coordinate_datum_available:
        nonclosure_reason = "chi_star_or_same_sector_proxy_absent"

    rows = _build_rows(
        chi_star_or_same_sector_proxy_available=chi_star_or_same_sector_proxy_available,
        nonclosure_reason=nonclosure_reason,
    )

    return {
        "generated_utc": _utc_now_iso(),
        "phase": {
            "phase": 8,
            "step": step_tag,
            "name": "chi_star or same-sector proxy closure retry",
        },
        "inputs": {
            "mass_origin_anchor_normalized_g3w_chi_proxy_audit_json": _relative_str(CHI_PROXY_AUDIT_JSON),
            "mass_origin_chi_star_proxy_source_inventory_json": _relative_str(SOURCE_INVENTORY_JSON),
            "mass_origin_same_sector_proxy_equivalence_audit_json": _relative_str(PROXY_EQUIVALENCE_AUDIT_JSON),
        },
        "intent": "Retry the closure of the missing anchor-coordinate datum after the same-sector proxy equivalence audit.",
        "formulas": {
            "retry_rule": "chi_star_or_same_sector_proxy_available iff the same-sector equivalence rule is public canonical and the anchor-coordinate datum itself is public canonical",
            "current_blocker": "the current source inventory still lacks the same-sector equivalence rule and the anchor-coordinate datum itself, so the retry remains non-closing",
        },
        "rows": rows,
        "summary": {
            "chi_star_or_same_sector_proxy_available": chi_star_or_same_sector_proxy_available,
            "proxy_route_retry_nonclosure_reason_or_none": nonclosure_reason,
            "same_sector_proxy_rule_available": same_sector_proxy_rule_available,
        },
        "decision": {
            "overall_status": "chi_star_proxy_closure_retry_frozen_absent",
            "keep_mass_origin_branch_blocked": True,
            "chi_star_or_same_sector_proxy_available": chi_star_or_same_sector_proxy_available,
            "proxy_route_retry_nonclosure_reason_or_none": nonclosure_reason,
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
            "chi_proxy_summary": chi_proxy_summary,
            "chi_star_proxy_source_inventory_summary": source_summary,
            "same_sector_proxy_equivalence_audit_summary": proxy_equivalence_summary,
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
