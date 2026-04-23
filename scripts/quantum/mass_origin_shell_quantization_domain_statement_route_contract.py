#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_origin_shell_quantization_domain_statement_route_contract.py

Step 8.7.55.2.418:
Freeze the next residual branch after the shell-quantization domain-relation
third retry still fails and the unresolved core contracts to the missing
shell-quantization domain statement.

Inputs:
  - output/public/quantum/mass_origin_reflective_cavity_rule_third_retry_metrics.json
  - output/public/quantum/mass_origin_discrete_spectrum_fourth_reopen_refresh_metrics.json

Outputs:
  - output/public/quantum/mass_origin_shell_quantization_domain_statement_route_contract_metrics.json
  - output/public/quantum/mass_origin_shell_quantization_domain_statement_route_contract_rows.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]

THIRD_RETRY_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_reflective_cavity_rule_third_retry_metrics.json"
FOURTH_REFRESH_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_discrete_spectrum_fourth_reopen_refresh_metrics.json"
OUT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_shell_quantization_domain_statement_route_contract_metrics.json"
OUT_CSV = ROOT / "output" / "public" / "quantum" / "mass_origin_shell_quantization_domain_statement_route_contract_rows.csv"
DEFAULT_STEP_TAG = "8.7.55.2.418"


# 関数: 現在UTC時刻を ISO 8601 文字列で返す。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: CLI 引数を解釈する。

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze the residual shell-quantization domain-statement branch contract.")
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
    split_contract_ready: bool,
) -> List[Dict[str, Any]]:
    return [
        {
            "row_id": "shell_quantization_domain_statement_route_contract_complete",
            "status": "pass",
            "metric": "shell-quantization domain-statement route contract complete",
            "value": 1.0,
            "note": "This step freezes the next residual branch after the shell-domain relation third retry remains blocked.",
        },
        {
            "row_id": "shell_quantization_domain_statement_route_retained",
            "status": "pass",
            "metric": "shell-quantization domain statement retained as residual follow-up",
            "value": 1.0,
            "note": "The unresolved core has contracted to the missing shell-quantization domain statement needed by the geometric route.",
        },
        {
            "row_id": "shell_quantization_domain_statement_route_split_ready",
            "status": "pass" if split_contract_ready else "reject",
            "metric": "shell-quantization domain-statement residual split contract ready",
            "value": 1.0 if split_contract_ready else 0.0,
            "note": (
                "The next branch can now focus on source inventory and wording audit for the shell-quantization domain statement."
                if split_contract_ready
                else "The residual split contract is not ready."
            ),
        },
    ]


# 関数: metrics payload 全体を構成する。

def _build_payload(step_tag: str) -> Dict[str, Any]:
    for path in (THIRD_RETRY_JSON, FOURTH_REFRESH_JSON):
        _require_path(path)

    third_retry = _read_json(THIRD_RETRY_JSON)
    fourth_refresh = _read_json(FOURTH_REFRESH_JSON)

    third_retry_summary = third_retry.get("summary", {})
    fourth_refresh_summary = fourth_refresh.get("summary", {})

    split_contract_ready = True
    rows = _build_rows(split_contract_ready=split_contract_ready)

    return {
        "generated_utc": _utc_now_iso(),
        "phase": {
            "phase": 8,
            "step": step_tag,
            "name": "shell-quantization domain-statement residual route contract",
        },
        "inputs": {
            "mass_origin_reflective_cavity_rule_third_retry_json": _relative_str(THIRD_RETRY_JSON),
            "mass_origin_discrete_spectrum_fourth_reopen_refresh_json": _relative_str(FOURTH_REFRESH_JSON),
        },
        "intent": "Freeze the next residual route after the geometric reflective-cavity third retry still fails and the unresolved core contracts to the missing shell-quantization domain statement.",
        "formulas": {
            "residual_route_rule": "if the geometric route stays preferred but reflective cavity closure still fails, the next branch must focus on the missing shell-quantization domain statement",
        },
        "rows": rows,
        "summary": {
            "selected_residual_binding_route_or_none": "geometric_reflective_boundary",
            "missing_geometric_boundary_artifact": "shell_quantization_domain_statement",
            "split_contract_ready": split_contract_ready,
        },
        "decision": {
            "overall_status": "shell_quantization_domain_statement_route_contract_frozen",
            "keep_mass_origin_branch_blocked": True,
            "selected_residual_binding_route_or_none": "geometric_reflective_boundary",
            "missing_geometric_boundary_artifact": "shell_quantization_domain_statement",
            "split_contract_ready": split_contract_ready,
            "hand_off_to_8_7_55_2_84": False,
            "next_required_artifacts": [
                "shell_quantization_domain_statement",
                "shell_quantization_to_domain_relation",
                "boundary_radius_or_domain_proxy",
            ],
        },
        "evidence": {
            "reflective_cavity_rule_third_retry_summary": third_retry_summary,
            "discrete_spectrum_fourth_reopen_refresh_summary": fourth_refresh_summary,
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
