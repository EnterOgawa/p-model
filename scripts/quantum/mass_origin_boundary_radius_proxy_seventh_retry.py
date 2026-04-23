#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_origin_boundary_radius_proxy_seventh_retry.py

Step 8.7.55.2.449:
Retry boundary-radius/domain-proxy closure for the seventh time after the
seventh shell-quantization domain-relation retry.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]

BOUNDARY_PROXY_SIXTH_RETRY_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_boundary_radius_proxy_sixth_retry_metrics.json"
DOMAIN_RELATION_SEVENTH_RETRY_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_shell_quantization_domain_relation_seventh_retry_metrics.json"
OUT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_boundary_radius_proxy_seventh_retry_metrics.json"
OUT_CSV = ROOT / "output" / "public" / "quantum" / "mass_origin_boundary_radius_proxy_seventh_retry_rows.csv"
DEFAULT_STEP_TAG = "8.7.55.2.449"


# 関数: 現在UTC時刻を ISO 8601 文字列で返す。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: CLI 引数を解釈する。

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retry boundary-radius/domain-proxy closure for the seventh time.")
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

def _build_rows(*, proxy_available: bool, nonclosure_reason: str | None) -> List[Dict[str, Any]]:
    return [
        {
            "row_id": "boundary_radius_proxy_seventh_retry_complete",
            "status": "pass",
            "metric": "boundary radius/domain proxy seventh retry complete",
            "value": 1.0,
            "note": "This step retries boundary radius/domain proxy closure after the seventh shell-domain relation retry.",
        },
        {
            "row_id": "boundary_radius_proxy_seventh_retry_available",
            "status": "pass" if proxy_available else "reject",
            "metric": "boundary radius/domain proxy available after seventh retry",
            "value": 1.0 if proxy_available else 0.0,
            "note": (
                "The boundary radius/domain proxy is available after seventh retry."
                if proxy_available
                else f"The seventh retry remains non-closing: {nonclosure_reason}."
            ),
        },
    ]


# 関数: metrics payload 全体を構成する。

def _build_payload(step_tag: str) -> Dict[str, Any]:
    for path in (BOUNDARY_PROXY_SIXTH_RETRY_JSON, DOMAIN_RELATION_SEVENTH_RETRY_JSON):
        _require_path(path)

    boundary_proxy_sixth_retry = _read_json(BOUNDARY_PROXY_SIXTH_RETRY_JSON)
    domain_relation_seventh_retry = _read_json(DOMAIN_RELATION_SEVENTH_RETRY_JSON)

    boundary_proxy_sixth_retry_summary = boundary_proxy_sixth_retry.get("summary", {})
    domain_relation_seventh_retry_summary = domain_relation_seventh_retry.get("summary", {})

    proxy_available = bool(domain_relation_seventh_retry_summary.get("shell_quantization_to_domain_relation_available", False))
    nonclosure_reason = domain_relation_seventh_retry_summary.get("domain_relation_seventh_retry_nonclosure_reason_or_none")
    rows = _build_rows(proxy_available=proxy_available, nonclosure_reason=nonclosure_reason)

    return {
        "generated_utc": _utc_now_iso(),
        "phase": {"phase": 8, "step": step_tag, "name": "boundary radius/domain proxy seventh retry"},
        "inputs": {
            "mass_origin_boundary_radius_proxy_sixth_retry_json": _relative_str(BOUNDARY_PROXY_SIXTH_RETRY_JSON),
            "mass_origin_shell_quantization_domain_relation_seventh_retry_json": _relative_str(
                DOMAIN_RELATION_SEVENTH_RETRY_JSON
            ),
        },
        "intent": "Retry whether the boundary radius/domain proxy can now close after the seventh shell-domain relation retry.",
        "formulas": {
            "seventh_retry_rule": "boundary_radius_or_domain_available iff the seventh shell-domain relation retry promotes a shell-quantization to domain relation",
        },
        "rows": rows,
        "summary": {
            "candidate_binding_route_id": "geometric_reflective_boundary",
            "boundary_radius_or_domain_available": proxy_available,
            "boundary_proxy_kind_or_none": "boundary_radius_or_domain_proxy" if proxy_available else None,
            "boundary_proxy_seventh_retry_nonclosure_reason_or_none": nonclosure_reason,
        },
        "decision": {
            "overall_status": "boundary_radius_proxy_seventh_retry_frozen_absent",
            "keep_mass_origin_branch_blocked": True,
            "candidate_binding_route_id": "geometric_reflective_boundary",
            "boundary_radius_or_domain_available": proxy_available,
            "boundary_proxy_kind_or_none": "boundary_radius_or_domain_proxy" if proxy_available else None,
            "boundary_proxy_seventh_retry_nonclosure_reason_or_none": nonclosure_reason,
            "hand_off_to_8_7_55_2_84": False,
            "next_required_artifacts": [
                "shell_quantization_domain_statement_token_atom",
                "boundary_radius_or_domain_proxy",
                "shell_quantization_reflective_cavity_rule",
            ],
        },
        "evidence": {
            "mass_origin_boundary_radius_proxy_sixth_retry_summary": boundary_proxy_sixth_retry_summary,
            "mass_origin_shell_quantization_domain_relation_seventh_retry_summary": domain_relation_seventh_retry_summary,
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
