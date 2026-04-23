#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_origin_boundary_radius_proxy_fourth_retry.py

Step 8.7.55.2.422:
Retry closure of the boundary-radius or domain proxy after the shell-
quantization domain relation fourth retry.

Inputs:
  - output/public/quantum/mass_origin_shell_quantization_domain_statement_route_contract_metrics.json
  - output/public/quantum/mass_origin_shell_quantization_domain_relation_fourth_retry_metrics.json
  - output/public/quantum/mass_origin_boundary_radius_proxy_third_retry_metrics.json

Outputs:
  - output/public/quantum/mass_origin_boundary_radius_proxy_fourth_retry_metrics.json
  - output/public/quantum/mass_origin_boundary_radius_proxy_fourth_retry_rows.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]

ROUTE_CONTRACT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_shell_quantization_domain_statement_route_contract_metrics.json"
RELATION_RETRY_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_shell_quantization_domain_relation_fourth_retry_metrics.json"
THIRD_PROXY_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_boundary_radius_proxy_third_retry_metrics.json"
OUT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_boundary_radius_proxy_fourth_retry_metrics.json"
OUT_CSV = ROOT / "output" / "public" / "quantum" / "mass_origin_boundary_radius_proxy_fourth_retry_rows.csv"
DEFAULT_STEP_TAG = "8.7.55.2.422"


# 関数: 現在UTC時刻を ISO 8601 文字列で返す。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: CLI 引数を解釈する。

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retry closure of the boundary-radius proxy a fourth time.")
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

def _build_rows(
    *,
    boundary_radius_or_domain_available: bool,
    boundary_proxy_kind_or_none: str | None,
    boundary_proxy_without_new_free_parameters: bool,
    nonclosure_reason_or_none: str | None,
) -> List[Dict[str, Any]]:
    return [
        {
            "row_id": "boundary_radius_proxy_fourth_retry_complete",
            "status": "pass",
            "metric": "boundary-radius proxy fourth retry complete",
            "value": 1.0,
            "note": "This step retries closure of the cavity-radius or domain proxy after the shell-domain statement branch.",
        },
        {
            "row_id": "boundary_radius_or_domain_available",
            "status": "pass" if boundary_radius_or_domain_available else "reject",
            "metric": "boundary radius or domain proxy available after fourth retry",
            "value": 1.0 if boundary_radius_or_domain_available else 0.0,
            "note": (
                f"The current public pack already supplies the proxy kind {boundary_proxy_kind_or_none}."
                if boundary_radius_or_domain_available
                else "The current public pack still lacks the shell-quantization domain-statement literal that would define a cavity radius or domain proxy."
            ),
        },
        {
            "row_id": "boundary_proxy_without_new_free_parameters",
            "status": "pass" if boundary_proxy_without_new_free_parameters else "reject",
            "metric": "boundary proxy stays inside no-new-free-parameter envelope after fourth retry",
            "value": 1.0 if boundary_proxy_without_new_free_parameters else 0.0,
            "note": (
                "The cavity proxy is already fixed without introducing an extra scale."
                if boundary_proxy_without_new_free_parameters
                else "Any cavity proxy would still require a new scale or an unfrozen shell-domain statement literal."
            ),
        },
        {
            "row_id": "boundary_proxy_fourth_retry_nonclosure_reason",
            "status": "watch" if nonclosure_reason_or_none else "pass",
            "metric": "boundary-radius proxy fourth-retry non-closure reason",
            "value": 0.0 if nonclosure_reason_or_none else 1.0,
            "note": (
                f"Current non-closure reason: {nonclosure_reason_or_none}."
                if nonclosure_reason_or_none
                else "The boundary-radius proxy is fully closed."
            ),
        },
    ]


# 関数: metrics payload 全体を構成する。

def _build_payload(step_tag: str) -> Dict[str, Any]:
    for path in (ROUTE_CONTRACT_JSON, RELATION_RETRY_JSON, THIRD_PROXY_JSON):
        _require_path(path)

    route_contract = _read_json(ROUTE_CONTRACT_JSON)
    relation_retry = _read_json(RELATION_RETRY_JSON)
    third_proxy = _read_json(THIRD_PROXY_JSON)

    route_contract_summary = route_contract.get("summary", {})
    relation_retry_summary = relation_retry.get("summary", {})
    third_proxy_summary = third_proxy.get("summary", {})

    boundary_radius_or_domain_available = bool(relation_retry_summary.get("shell_quantization_to_domain_relation_available", False))
    boundary_proxy_kind_or_none = "shell_quantization_domain_proxy" if boundary_radius_or_domain_available else None
    boundary_proxy_without_new_free_parameters = boundary_radius_or_domain_available
    nonclosure_reason_or_none = None if boundary_radius_or_domain_available else "shell_quantization_domain_statement_literal_absent"
    rows = _build_rows(
        boundary_radius_or_domain_available=boundary_radius_or_domain_available,
        boundary_proxy_kind_or_none=boundary_proxy_kind_or_none,
        boundary_proxy_without_new_free_parameters=boundary_proxy_without_new_free_parameters,
        nonclosure_reason_or_none=nonclosure_reason_or_none,
    )

    return {
        "generated_utc": _utc_now_iso(),
        "phase": {
            "phase": 8,
            "step": step_tag,
            "name": "boundary-radius proxy fourth retry",
        },
        "inputs": {
            "mass_origin_shell_quantization_domain_statement_route_contract_json": _relative_str(ROUTE_CONTRACT_JSON),
            "mass_origin_shell_quantization_domain_relation_fourth_retry_json": _relative_str(RELATION_RETRY_JSON),
            "mass_origin_boundary_radius_proxy_third_retry_json": _relative_str(THIRD_PROXY_JSON),
        },
        "intent": "Retry closure of the cavity-radius or domain proxy after the shell-domain statement branch.",
        "formulas": {
            "closure_rule": "the boundary-radius proxy closes only if the shell-quantization domain relation closes without introducing a new scale",
            "current_absence": "the public shell-family coefficients still lack the shell-quantization domain-statement literal needed to define a cavity radius or domain proxy",
        },
        "rows": rows,
        "summary": {
            "candidate_binding_route_id": route_contract_summary.get("selected_residual_binding_route_or_none"),
            "boundary_radius_or_domain_available": boundary_radius_or_domain_available,
            "boundary_proxy_kind_or_none": boundary_proxy_kind_or_none,
            "boundary_proxy_without_new_free_parameters": boundary_proxy_without_new_free_parameters,
            "boundary_proxy_fourth_retry_nonclosure_reason_or_none": nonclosure_reason_or_none,
        },
        "decision": {
            "overall_status": (
                "boundary_radius_proxy_fourth_retry_closed"
                if boundary_radius_or_domain_available
                else "boundary_radius_proxy_fourth_retry_still_blocked"
            ),
            "keep_mass_origin_branch_blocked": not boundary_radius_or_domain_available,
            "candidate_binding_route_id": route_contract_summary.get("selected_residual_binding_route_or_none"),
            "boundary_radius_or_domain_available": boundary_radius_or_domain_available,
            "boundary_proxy_kind_or_none": boundary_proxy_kind_or_none,
            "boundary_proxy_without_new_free_parameters": boundary_proxy_without_new_free_parameters,
            "boundary_proxy_fourth_retry_nonclosure_reason_or_none": nonclosure_reason_or_none,
            "hand_off_to_8_7_55_2_84": False,
            "next_required_artifacts": (
                ["shell_quantization_reflective_cavity_rule", "discrete_spectrum_fifth_reopen_refresh"]
                if boundary_radius_or_domain_available
                else ["shell_quantization_domain_statement_literal", "shell_quantization_domain_statement", "boundary_radius_or_domain_proxy"]
            ),
        },
        "evidence": {
            "shell_quantization_domain_statement_route_contract_summary": route_contract_summary,
            "shell_quantization_domain_relation_fourth_retry_summary": relation_retry_summary,
            "boundary_radius_proxy_third_retry_summary": third_proxy_summary,
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
