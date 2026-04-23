#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_origin_reflective_cavity_rule_twenty_seventh_retry.py

Step 8.7.55.2.707:
Retry closure of the geometric reflective cavity rule after the twenty-seventh
boundary-radius/domain-proxy retry.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]

CURRENT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_reflective_cavity_rule_twenty_sixth_retry_metrics.json"
DEPENDENCY_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_boundary_radius_proxy_twenty_seventh_retry_metrics.json"
OUT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_reflective_cavity_rule_twenty_seventh_retry_metrics.json"
OUT_CSV = ROOT / "output" / "public" / "quantum" / "mass_origin_reflective_cavity_rule_twenty_seventh_retry_rows.csv"
DEFAULT_STEP_TAG = "8.7.55.2.707"


# 関数: 現在UTC時刻を ISO 8601 文字列で返す。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: CLI 引数を解釈する。

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retry reflective cavity rule closure for the twenty-seventh time.")
    parser.add_argument("--step-tag", default=DEFAULT_STEP_TAG, help="Roadmap step tag to stamp into the output payload.")
    return parser.parse_args()


# 関数: 必須入力の存在を検査する。

def _require_path(path: Path) -> None:
    if not path.exists():
        raise SystemExit(f"[fail] missing required input: {path}")


# 関数: JSON ファイルを辞書として読む。

def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


# 関数: リポジトリ相対パスへ正規化する。

def _relative_str(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


# 関数: rows を構成する。

def _build_rows(*, cavity_ready: bool, nonclosure_reason: str | None) -> List[Dict[str, Any]]:
    return [
        {
            "row_id": "reflective_cavity_rule_twenty_seventh_retry_complete",
            "status": "pass",
            "metric": "reflective cavity rule twenty-seventh retry complete",
            "value": 1.0,
            "note": "This step retries closure of the geometric reflective cavity rule after the twenty-seventh boundary-proxy retry.",
        },
        {
            "row_id": "reflective_cavity_rule_twenty_seventh_retry_available",
            "status": "pass" if cavity_ready else "reject",
            "metric": "reflective cavity rule available after twenty-seventh retry",
            "value": 1.0 if cavity_ready else 0.0,
            "note": "The reflective cavity rule is available after the twenty-seventh retry." if cavity_ready else f"The cavity retry remains blocked: {nonclosure_reason}.",
        },
    ]


# 関数: metrics payload 全体を構成する。

def _build_payload(step_tag: str) -> Dict[str, Any]:
    for path in (CURRENT_JSON, DEPENDENCY_JSON):
        _require_path(path)

    current_payload = _read_json(CURRENT_JSON)
    dependency_payload = _read_json(DEPENDENCY_JSON)
    current_summary = current_payload.get("summary", {})
    dependency_summary = dependency_payload.get("summary", {})
    cavity_ready = bool(dependency_summary.get("boundary_radius_or_domain_available", False))
    nonclosure_reason = dependency_summary.get("boundary_proxy_twenty_seventh_retry_nonclosure_reason_or_none")
    rows = _build_rows(cavity_ready=cavity_ready, nonclosure_reason=nonclosure_reason)
    return {
        "generated_utc": _utc_now_iso(),
        "phase": {"phase": 8, "step": step_tag, "name": "reflective cavity rule twenty-seventh retry"},
        "inputs": {
            "mass_origin_reflective_cavity_rule_twenty_sixth_retry_json": _relative_str(CURRENT_JSON),
            "mass_origin_boundary_radius_proxy_twenty_seventh_retry_json": _relative_str(DEPENDENCY_JSON),
        },
        "intent": "Retry whether the geometric reflective cavity rule can now close after the twenty-sixth boundary-radius/domain-proxy retry.",
        "formulas": {
            "retry_rule": "geometric_boundary_promotion_rule_available iff the twenty-sixth boundary-radius/domain-proxy retry promotes a boundary radius or domain proxy",
        },
        "rows": rows,
        "summary": {
            "candidate_binding_route_id": "geometric_reflective_boundary",
            "boundary_radius_or_domain_available": cavity_ready,
            "geometric_boundary_promotion_rule_available": cavity_ready,
            "discrete_shell_cavity_ready": cavity_ready,
            "cavity_rule_twenty_seventh_retry_nonclosure_reason_or_none": nonclosure_reason,
        },
        "decision": {
            "overall_status": "reflective_cavity_rule_twenty_seventh_retry_frozen_absent",
            "keep_mass_origin_branch_blocked": True,
            "candidate_binding_route_id": "geometric_reflective_boundary",
            "boundary_radius_or_domain_available": cavity_ready,
            "geometric_boundary_promotion_rule_available": cavity_ready,
            "discrete_shell_cavity_ready": cavity_ready,
            "cavity_rule_twenty_seventh_retry_nonclosure_reason_or_none": nonclosure_reason,
            "hand_off_to_8_7_55_2_84": False,
        },
        "evidence": {
            "mass_origin_reflective_cavity_rule_twenty_sixth_retry_summary": current_summary,
            "mass_origin_boundary_radius_proxy_twenty_seventh_retry_summary": dependency_summary,
        },
    }


# 関数: rows を CSV 出力する。

def _write_csv(rows: List[Dict[str, Any]]) -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["row_id", "status", "metric", "value", "note"])
        writer.writeheader()
        writer.writerows(rows)


# 関数: JSON を整形出力する。

def _write_json(payload: Dict[str, Any]) -> None:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


# 関数: エントリポイントとして step を実行する。

def main() -> None:
    args = _parse_args()
    payload = _build_payload(args.step_tag)
    _write_json(payload)
    _write_csv(payload["rows"])
    print(f"[ok] wrote {OUT_JSON}")
    print(f"[ok] wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
