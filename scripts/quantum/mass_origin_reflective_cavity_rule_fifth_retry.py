#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_origin_reflective_cavity_rule_fifth_retry.py

Step 8.7.55.2.431:
Retry closure of the geometric reflective cavity rule after the fifth
boundary-radius/domain-proxy retry.

Inputs:
  - output/public/quantum/mass_origin_shell_quantization_domain_statement_literal_route_contract_metrics.json
  - output/public/quantum/mass_origin_boundary_radius_proxy_fifth_retry_metrics.json
  - output/public/quantum/mass_origin_reflective_cavity_rule_fourth_retry_metrics.json

Outputs:
  - output/public/quantum/mass_origin_reflective_cavity_rule_fifth_retry_metrics.json
  - output/public/quantum/mass_origin_reflective_cavity_rule_fifth_retry_rows.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]

ROUTE_CONTRACT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_shell_quantization_domain_statement_literal_route_contract_metrics.json"
BOUNDARY_RETRY_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_boundary_radius_proxy_fifth_retry_metrics.json"
PREVIOUS_CAVITY_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_reflective_cavity_rule_fourth_retry_metrics.json"
OUT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_reflective_cavity_rule_fifth_retry_metrics.json"
OUT_CSV = ROOT / "output" / "public" / "quantum" / "mass_origin_reflective_cavity_rule_fifth_retry_rows.csv"
DEFAULT_STEP_TAG = "8.7.55.2.431"


# 関数: 現在UTC時刻を ISO 8601 文字列で返す。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: CLI 引数を解釈する。

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retry closure of the geometric reflective cavity rule a fifth time.")
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
    cavity_ready: bool,
    nonclosure_reason_or_none: str | None,
) -> List[Dict[str, Any]]:
    return [
        {
            "row_id": "reflective_cavity_rule_fifth_retry_complete",
            "status": "pass",
            "metric": "reflective cavity rule fifth retry complete",
            "value": 1.0,
            "note": "This step retries closure of the geometric reflective cavity rule after the fifth boundary-proxy retry.",
        },
        {
            "row_id": "reflective_cavity_rule_fifth_retry_available",
            "status": "pass" if cavity_ready else "reject",
            "metric": "reflective cavity rule available after fifth retry",
            "value": 1.0 if cavity_ready else 0.0,
            "note": (
                "The geometric reflective cavity rule is now public canonical."
                if cavity_ready
                else f"The cavity retry remains blocked: {nonclosure_reason_or_none}."
            ),
        },
    ]


# 関数: metrics payload 全体を構成する。

def _build_payload(step_tag: str) -> Dict[str, Any]:
    for path in (ROUTE_CONTRACT_JSON, BOUNDARY_RETRY_JSON, PREVIOUS_CAVITY_JSON):
        _require_path(path)

    route_contract = _read_json(ROUTE_CONTRACT_JSON)
    boundary_retry = _read_json(BOUNDARY_RETRY_JSON)
    previous_cavity = _read_json(PREVIOUS_CAVITY_JSON)

    route_contract_summary = route_contract.get("summary", {})
    boundary_retry_summary = boundary_retry.get("summary", {})
    previous_cavity_summary = previous_cavity.get("summary", {})

    boundary_available = bool(boundary_retry_summary.get("boundary_radius_or_domain_available", False))
    cavity_ready = bool(boundary_available)
    nonclosure_reason_or_none = None if cavity_ready else "shell_quantization_domain_statement_phrase_fragment_absent"
    rows = _build_rows(cavity_ready=cavity_ready, nonclosure_reason_or_none=nonclosure_reason_or_none)

    return {
        "generated_utc": _utc_now_iso(),
        "phase": {
            "phase": 8,
            "step": step_tag,
            "name": "reflective cavity rule fifth retry",
        },
        "inputs": {
            "mass_origin_shell_quantization_domain_statement_literal_route_contract_json": _relative_str(ROUTE_CONTRACT_JSON),
            "mass_origin_boundary_radius_proxy_fifth_retry_json": _relative_str(BOUNDARY_RETRY_JSON),
            "mass_origin_reflective_cavity_rule_fourth_retry_json": _relative_str(PREVIOUS_CAVITY_JSON),
        },
        "intent": "Retry closure of the geometric reflective cavity rule after the fifth boundary-radius/domain-proxy retry.",
        "formulas": {
            "closure_rule": "the reflective cavity rule closes only if the boundary radius/domain proxy closes without introducing a new geometric scale",
            "current_absence": "the boundary proxy still lacks the shell-quantization domain-statement phrase fragment needed to promote the reflective cavity rule",
        },
        "rows": rows,
        "summary": {
            "candidate_binding_route_id": route_contract_summary.get("selected_residual_binding_route_or_none"),
            "boundary_radius_or_domain_available": boundary_available,
            "geometric_boundary_promotion_rule_available": cavity_ready,
            "discrete_shell_cavity_ready": cavity_ready,
            "cavity_rule_fifth_retry_nonclosure_reason_or_none": nonclosure_reason_or_none,
        },
        "decision": {
            "overall_status": (
                "reflective_cavity_rule_fifth_retry_closed"
                if cavity_ready
                else "reflective_cavity_rule_fifth_retry_still_blocked"
            ),
            "keep_mass_origin_branch_blocked": not cavity_ready,
            "candidate_binding_route_id": route_contract_summary.get("selected_residual_binding_route_or_none"),
            "boundary_radius_or_domain_available": boundary_available,
            "geometric_boundary_promotion_rule_available": cavity_ready,
            "discrete_shell_cavity_ready": cavity_ready,
            "cavity_rule_fifth_retry_nonclosure_reason_or_none": nonclosure_reason_or_none,
            "hand_off_to_8_7_55_2_84": False,
            "next_required_artifacts": (
                ["discrete_spectrum_reopen_refresh"]
                if cavity_ready
                else ["shell_quantization_domain_statement_phrase_fragment", "shell_quantization_reflective_cavity_rule", "boundary_radius_or_domain_proxy"]
            ),
        },
        "evidence": {
            "shell_quantization_domain_statement_literal_route_contract_summary": route_contract_summary,
            "boundary_radius_proxy_fifth_retry_summary": boundary_retry_summary,
            "reflective_cavity_rule_fourth_retry_summary": previous_cavity_summary,
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
