#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_origin_chi_star_proxy_second_retry.py

Step 8.7.55.2.268:
Retry the chi_* / same-sector proxy closure after the same-sector equivalence
wording audit.

Inputs:
  - output/public/quantum/mass_origin_chi_star_proxy_closure_retry_metrics.json
  - output/public/quantum/mass_origin_same_sector_equivalence_wording_audit_metrics.json

Outputs:
  - output/public/quantum/mass_origin_chi_star_proxy_second_retry_metrics.json
  - output/public/quantum/mass_origin_chi_star_proxy_second_retry_rows.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]

PROXY_CLOSURE_RETRY_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_chi_star_proxy_closure_retry_metrics.json"
EQUIVALENCE_WORDING_AUDIT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_same_sector_equivalence_wording_audit_metrics.json"
OUT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_chi_star_proxy_second_retry_metrics.json"
OUT_CSV = ROOT / "output" / "public" / "quantum" / "mass_origin_chi_star_proxy_second_retry_rows.csv"
DEFAULT_STEP_TAG = "8.7.55.2.268"


# 関数: `_utc_now_iso` の入出力契約と処理意図を定義する。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_parse_args` の入出力契約と処理意図を定義する。

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retry the chi_* / same-sector proxy closure after the same-sector equivalence wording audit.",
    )
    parser.add_argument("--step-tag", default=DEFAULT_STEP_TAG, help="Roadmap step tag to stamp into the output payload.")
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

def _build_rows(*, chi_star_or_same_sector_proxy_available: bool, nonclosure_reason: str | None) -> List[Dict[str, Any]]:
    return [
        {
            "row_id": "chi_star_proxy_second_retry_complete",
            "status": "pass",
            "metric": "chi_* or same-sector proxy second retry complete",
            "value": 1.0,
            "note": "This step retries the chi_* proxy closure after the same-sector equivalence wording audit.",
        },
        {
            "row_id": "chi_star_proxy_second_retry_available",
            "status": "pass" if chi_star_or_same_sector_proxy_available else "reject",
            "metric": "chi_* or same-sector proxy available after second retry",
            "value": 1.0 if chi_star_or_same_sector_proxy_available else 0.0,
            "note": (
                "The missing anchor-coordinate datum is now public canonical."
                if chi_star_or_same_sector_proxy_available
                else f"The second retry remains non-closing: {nonclosure_reason}."
            ),
        },
    ]


# 関数: `_build_payload` の入出力契約と処理意図を定義する。

def _build_payload(step_tag: str) -> Dict[str, Any]:
    for path in (PROXY_CLOSURE_RETRY_JSON, EQUIVALENCE_WORDING_AUDIT_JSON):
        _require_path(path)

    proxy_closure_retry = _read_json(PROXY_CLOSURE_RETRY_JSON)
    equivalence_wording_audit = _read_json(EQUIVALENCE_WORDING_AUDIT_JSON)

    proxy_closure_summary = proxy_closure_retry.get("summary", {})
    equivalence_wording_summary = equivalence_wording_audit.get("summary", {})

    same_sector_equivalence_rule_available = bool(
        equivalence_wording_summary.get("same_sector_equivalence_rule_available", False)
    )
    chi_star_or_same_sector_proxy_available = same_sector_equivalence_rule_available
    missing_inputs = [str(item) for item in equivalence_wording_summary.get("equivalence_rule_missing_inputs", [])]
    nonclosure_reason = None

    # 条件分岐: `not same_sector_equivalence_rule_available` を満たす経路を評価する。
    if not same_sector_equivalence_rule_available:
        nonclosure_reason = "same_sector_equivalence_statement_absent"

        # 条件分岐: `"same_sector_equivalence_statement" not in missing_inputs` を満たす経路を評価する。
        if "same_sector_equivalence_statement" not in missing_inputs:
            nonclosure_reason = "same_sector_equivalence_rule_absent"

    rows = _build_rows(
        chi_star_or_same_sector_proxy_available=chi_star_or_same_sector_proxy_available,
        nonclosure_reason=nonclosure_reason,
    )

    return {
        "generated_utc": _utc_now_iso(),
        "phase": {"phase": 8, "step": step_tag, "name": "chi_star proxy closure second retry"},
        "inputs": {
            "mass_origin_chi_star_proxy_closure_retry_json": _relative_str(PROXY_CLOSURE_RETRY_JSON),
            "mass_origin_same_sector_equivalence_wording_audit_json": _relative_str(EQUIVALENCE_WORDING_AUDIT_JSON),
        },
        "intent": "Retry the chi_* / same-sector proxy closure after the same-sector equivalence wording audit.",
        "formulas": {
            "second_retry_rule": "chi_star_or_same_sector_proxy_available iff the same-sector equivalence rule becomes public canonical",
            "current_blocker": "the wording audit still lacks a promotable same-sector equivalence statement, so the chi_* proxy route cannot yet close",
        },
        "rows": rows,
        "summary": {
            "chi_star_or_same_sector_proxy_available": chi_star_or_same_sector_proxy_available,
            "proxy_second_retry_nonclosure_reason_or_none": nonclosure_reason,
            "same_sector_equivalence_rule_available": same_sector_equivalence_rule_available,
        },
        "decision": {
            "overall_status": "chi_star_proxy_second_retry_frozen_absent",
            "keep_mass_origin_branch_blocked": True,
            "chi_star_or_same_sector_proxy_available": chi_star_or_same_sector_proxy_available,
            "proxy_second_retry_nonclosure_reason_or_none": nonclosure_reason,
            "hand_off_to_8_7_55_2_83": False,
            "next_required_artifacts": [
                "same_sector_equivalence_statement",
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
            "chi_star_proxy_closure_retry_summary": proxy_closure_summary,
            "same_sector_equivalence_wording_audit_summary": equivalence_wording_summary,
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
