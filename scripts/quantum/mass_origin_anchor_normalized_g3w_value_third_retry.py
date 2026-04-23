#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_origin_anchor_normalized_g3w_value_third_retry.py

Step 8.7.55.2.269:
Retry the preferred anchor-normalized g_3w route after the chi_* proxy second
retry.

Inputs:
  - output/public/quantum/mass_origin_anchor_normalized_g3w_value_second_retry_metrics.json
  - output/public/quantum/mass_origin_chi_star_proxy_second_retry_metrics.json

Outputs:
  - output/public/quantum/mass_origin_anchor_normalized_g3w_value_third_retry_metrics.json
  - output/public/quantum/mass_origin_anchor_normalized_g3w_value_third_retry_rows.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]

VALUE_SECOND_RETRY_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_anchor_normalized_g3w_value_second_retry_metrics.json"
PROXY_SECOND_RETRY_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_chi_star_proxy_second_retry_metrics.json"
OUT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_anchor_normalized_g3w_value_third_retry_metrics.json"
OUT_CSV = ROOT / "output" / "public" / "quantum" / "mass_origin_anchor_normalized_g3w_value_third_retry_rows.csv"
DEFAULT_STEP_TAG = "8.7.55.2.269"


# 関数: `_utc_now_iso` の入出力契約と処理意図を定義する。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_parse_args` の入出力契約と処理意図を定義する。

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retry the anchor-normalized g_3w public value after the chi_* proxy second retry.",
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

def _build_rows(
    *,
    anchor_normalized_g3w_public_value_available: bool,
    r3_target_available: bool,
    nonclosure_reason: str | None,
) -> List[Dict[str, Any]]:
    return [
        {
            "row_id": "anchor_normalized_g3w_value_third_retry_complete",
            "status": "pass",
            "metric": "anchor-normalized g_3w value third retry complete",
            "value": 1.0,
            "note": "This step retries the preferred g_3w route after the chi_* proxy second retry.",
        },
        {
            "row_id": "anchor_normalized_g3w_value_third_retry_public_value_available",
            "status": "pass" if anchor_normalized_g3w_public_value_available else "reject",
            "metric": "anchor-normalized public g_3w value available after third retry",
            "value": 1.0 if anchor_normalized_g3w_public_value_available else 0.0,
            "note": (
                "A public anchor-normalized g_3w value is now available."
                if anchor_normalized_g3w_public_value_available
                else f"The third retry remains non-closing: {nonclosure_reason}."
            ),
        },
        {
            "row_id": "anchor_normalized_g3w_value_third_retry_r3_target_available",
            "status": "pass" if r3_target_available else "reject",
            "metric": "R_3 target available after third retry",
            "value": 1.0 if r3_target_available else 0.0,
            "note": (
                "The third retry now promotes a public canonical R_3 target."
                if r3_target_available
                else "R_3 target remains unavailable because the anchor-normalized g_3w public value is still missing."
            ),
        },
    ]


# 関数: `_build_payload` の入出力契約と処理意図を定義する。

def _build_payload(step_tag: str) -> Dict[str, Any]:
    for path in (VALUE_SECOND_RETRY_JSON, PROXY_SECOND_RETRY_JSON):
        _require_path(path)

    value_second_retry = _read_json(VALUE_SECOND_RETRY_JSON)
    proxy_second_retry = _read_json(PROXY_SECOND_RETRY_JSON)

    value_second_retry_summary = value_second_retry.get("summary", {})
    proxy_second_retry_summary = proxy_second_retry.get("summary", {})

    chi_star_or_same_sector_proxy_available = bool(
        proxy_second_retry_summary.get("chi_star_or_same_sector_proxy_available", False)
    )
    anchor_normalized_g3w_public_value_available = chi_star_or_same_sector_proxy_available
    r3_target_available = anchor_normalized_g3w_public_value_available
    r3_target_value_or_none = None
    nonclosure_reason = proxy_second_retry_summary.get("proxy_second_retry_nonclosure_reason_or_none")
    rows = _build_rows(
        anchor_normalized_g3w_public_value_available=anchor_normalized_g3w_public_value_available,
        r3_target_available=r3_target_available,
        nonclosure_reason=nonclosure_reason,
    )

    return {
        "generated_utc": _utc_now_iso(),
        "phase": {"phase": 8, "step": step_tag, "name": "anchor-normalized g3w public value third retry"},
        "inputs": {
            "mass_origin_anchor_normalized_g3w_value_second_retry_json": _relative_str(VALUE_SECOND_RETRY_JSON),
            "mass_origin_chi_star_proxy_second_retry_json": _relative_str(PROXY_SECOND_RETRY_JSON),
        },
        "intent": "Retry the preferred anchor-normalized g_3w route after the chi_* proxy second retry.",
        "formulas": {
            "third_retry_rule": "anchor_normalized_g3w_public_value_available iff the chi_* or same-sector proxy second retry succeeds",
            "r3_rule": "r3_target_available iff anchor_normalized_g3w_public_value_available",
        },
        "rows": rows,
        "summary": {
            "anchor_normalized_g3w_public_value_available": anchor_normalized_g3w_public_value_available,
            "r3_target_available": r3_target_available,
            "r3_target_value_or_none": r3_target_value_or_none,
            "g3w_third_retry_nonclosure_reason_or_none": nonclosure_reason,
        },
        "decision": {
            "overall_status": "anchor_normalized_g3w_value_third_retry_frozen_absent",
            "keep_mass_origin_branch_blocked": True,
            "anchor_normalized_g3w_public_value_available": anchor_normalized_g3w_public_value_available,
            "r3_target_available": r3_target_available,
            "r3_target_value_or_none": r3_target_value_or_none,
            "g3w_third_retry_nonclosure_reason_or_none": nonclosure_reason,
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
            "g3w_second_retry_summary": value_second_retry_summary,
            "chi_star_proxy_second_retry_summary": proxy_second_retry_summary,
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
