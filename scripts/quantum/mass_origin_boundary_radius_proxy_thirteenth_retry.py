#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_origin_boundary_radius_proxy_thirteenth_retry.py

Step 8.7.55.2.524:
Retry whether a boundary-radius/domain proxy can be promoted after the
thirteenth shell-quantization to domain relation retry.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]

CURRENT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_boundary_radius_proxy_twelfth_retry_metrics.json"
DEPENDENCY_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_shell_quantization_domain_relation_thirteenth_retry_metrics.json"
OUT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_boundary_radius_proxy_thirteenth_retry_metrics.json"
OUT_CSV = ROOT / "output" / "public" / "quantum" / "mass_origin_boundary_radius_proxy_thirteenth_retry_rows.csv"
DEFAULT_STEP_TAG = "8.7.55.2.524"


# 関数: 現在UTC時刻を ISO 8601 文字列で返す。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: CLI 引数を解釈する。

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retry boundary-radius/domain proxy promotion for the thirteenth time.")
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

def _build_rows(*, proxy_ready: bool, nonclosure_reason: str | None) -> List[Dict[str, Any]]:
    return [
        {
            "row_id": "boundary_radius_proxy_thirteenth_retry_complete",
            "status": "pass",
            "metric": "boundary-radius proxy thirteenth retry complete",
            "value": 1.0,
            "note": "This step retries boundary-radius/domain proxy promotion after the thirteenth domain-relation retry.",
        },
        {
            "row_id": "boundary_radius_proxy_thirteenth_retry_available",
            "status": "pass" if proxy_ready else "reject",
            "metric": "boundary-radius or domain proxy available after thirteenth retry",
            "value": 1.0 if proxy_ready else 0.0,
            "note": "The boundary proxy is available after the thirteenth retry." if proxy_ready else f"The thirteenth retry remains blocked: {nonclosure_reason}.",
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
    proxy_ready = bool(dependency_summary.get("shell_quantization_to_domain_relation_available", False))
    nonclosure_reason = None if proxy_ready else "shell_quantization_domain_statement_terminal_atom_absent"
    rows = _build_rows(proxy_ready=proxy_ready, nonclosure_reason=nonclosure_reason)
    return {
        "generated_utc": _utc_now_iso(),
        "phase": {"phase": 8, "step": step_tag, "name": "boundary-radius proxy thirteenth retry"},
        "inputs": {
            "mass_origin_boundary_radius_proxy_twelfth_retry_json": _relative_str(CURRENT_JSON),
            "mass_origin_shell_quantization_domain_relation_thirteenth_retry_json": _relative_str(DEPENDENCY_JSON),
        },
        "intent": "Retry whether a boundary-radius/domain proxy can now close after the thirteenth shell-quantization to domain relation retry.",
        "formulas": {
            "retry_rule": "boundary_radius_or_domain_available iff shell_quantization_to_domain_relation_available",
        },
        "rows": rows,
        "summary": {
            "boundary_radius_or_domain_available": proxy_ready,
            "boundary_proxy_kind_or_none": "geometric_domain_proxy" if proxy_ready else None,
            "boundary_proxy_thirteenth_retry_nonclosure_reason_or_none": nonclosure_reason,
        },
        "decision": {
            "overall_status": "boundary_radius_proxy_thirteenth_retry_frozen_absent",
            "keep_mass_origin_branch_blocked": True,
            "boundary_radius_or_domain_available": proxy_ready,
            "boundary_proxy_kind_or_none": "geometric_domain_proxy" if proxy_ready else None,
            "boundary_proxy_thirteenth_retry_nonclosure_reason_or_none": nonclosure_reason,
            "hand_off_to_8_7_55_2_84": False,
        },
        "evidence": {
            "mass_origin_boundary_radius_proxy_twelfth_retry_summary": current_summary,
            "mass_origin_shell_quantization_domain_relation_thirteenth_retry_summary": dependency_summary,
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
