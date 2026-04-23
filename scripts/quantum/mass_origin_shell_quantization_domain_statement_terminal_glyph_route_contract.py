#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_origin_shell_quantization_domain_statement_terminal_glyph_route_contract.py

Step 8.7.55.2.736:
Freeze the next residual route after the shell-quantization
domain-statement-terminal-atom branch still fails and the unresolved
geometric-binding core contracts to the missing shell-quantization
domain-statement terminal glyph.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]

REFLECTIVE_CAVITY_TWENTY_NINTH_RETRY_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_reflective_cavity_rule_twenty_ninth_retry_metrics.json"
DISCRETE_SPECTRUM_THIRTIETH_REOPEN_REFRESH_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_discrete_spectrum_thirtieth_reopen_refresh_metrics.json"
OUT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_shell_quantization_domain_statement_terminal_glyph_route_contract_metrics.json"
OUT_CSV = ROOT / "output" / "public" / "quantum" / "mass_origin_shell_quantization_domain_statement_terminal_glyph_route_contract_rows.csv"
DEFAULT_STEP_TAG = "8.7.55.2.736"


# 関数: 現在UTC時刻を ISO 8601 文字列で返す。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: CLI 引数を解釈する。

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze the next residual route after the shell-quantization domain-statement-terminal-atom branch still fails.")
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

def _build_rows(*, required_route_items: List[str], split_contract_ready: bool) -> List[Dict[str, Any]]:
    return [
        {
            "row_id": "shell_quantization_domain_statement_terminal_glyph_route_contract_complete",
            "status": "pass",
            "metric": "shell-quantization domain-statement-terminal-glyph route contract complete",
            "value": 1.0,
            "note": "This step freezes the next residual branch after the shell-domain statement-terminal-atom route remains blocked.",
        },
        {
            "row_id": "shell_quantization_domain_statement_terminal_glyph_route_contract_required_items",
            "status": "watch",
            "metric": "required route items for shell-quantization domain-statement-terminal-glyph branch",
            "value": float(len(required_route_items)),
            "note": f"Required route items: {required_route_items}.",
        },
        {
            "row_id": "shell_quantization_domain_statement_terminal_glyph_route_contract_split_ready",
            "status": "pass" if split_contract_ready else "reject",
            "metric": "shell-quantization domain-statement-terminal-glyph residual split contract ready",
            "value": 1.0 if split_contract_ready else 0.0,
            "note": (
                "The next branch can now audit concrete public-canonical source candidates for the missing shell-quantization domain-statement terminal glyph."
                if split_contract_ready
                else "The shell-quantization domain-statement-terminal-glyph residual branch is not yet formalized."
            ),
        },
    ]


# 関数: metrics payload 全体を構成する。

def _build_payload(step_tag: str) -> Dict[str, Any]:
    for path in (REFLECTIVE_CAVITY_TWENTY_NINTH_RETRY_JSON, DISCRETE_SPECTRUM_THIRTIETH_REOPEN_REFRESH_JSON):
        _require_path(path)

    reflective_cavity_twenty_ninth_retry = _read_json(REFLECTIVE_CAVITY_TWENTY_NINTH_RETRY_JSON)
    discrete_spectrum_thirtieth_reopen_refresh = _read_json(DISCRETE_SPECTRUM_THIRTIETH_REOPEN_REFRESH_JSON)

    reflective_cavity_twenty_ninth_retry_summary = reflective_cavity_twenty_ninth_retry.get("summary", {})
    discrete_spectrum_thirtieth_reopen_refresh_summary = discrete_spectrum_thirtieth_reopen_refresh.get("summary", {})

    required_route_items = [
        "shell_quantization_family_public_candidate",
        "geometric_domain_symbol_note",
        "boundary_condition_quantization_note",
        "shell_quantization_domain_statement_terminal_glyph",
    ]
    split_contract_ready = True
    rows = _build_rows(required_route_items=required_route_items, split_contract_ready=split_contract_ready)

    return {
        "generated_utc": _utc_now_iso(),
        "phase": {"phase": 8, "step": step_tag, "name": "shell-quantization domain-statement-terminal-glyph route contract freeze"},
        "inputs": {
            "mass_origin_reflective_cavity_rule_twenty_ninth_retry_json": _relative_str(REFLECTIVE_CAVITY_TWENTY_NINTH_RETRY_JSON),
            "mass_origin_discrete_spectrum_thirtieth_reopen_refresh_json": _relative_str(DISCRETE_SPECTRUM_THIRTIETH_REOPEN_REFRESH_JSON),
        },
        "intent": "Freeze the next residual branch contract after the shell-quantization domain-statement-terminal-atom branch still fails and the unresolved geometric-binding core contracts to the missing shell-quantization domain-statement terminal glyph.",
        "rows": rows,
        "summary": {
            "selected_residual_binding_route_or_none": "geometric_reflective_boundary",
            "missing_geometric_boundary_artifact": "shell_quantization_domain_statement_terminal_glyph",
            "required_route_items": required_route_items,
            "split_contract_ready": split_contract_ready,
        },
        "decision": {
            "overall_status": "shell_quantization_domain_statement_terminal_glyph_route_contract_frozen",
            "keep_mass_origin_branch_blocked": True,
            "selected_residual_binding_route_or_none": "geometric_reflective_boundary",
            "missing_geometric_boundary_artifact": "shell_quantization_domain_statement_terminal_glyph",
            "split_contract_ready": split_contract_ready,
            "hand_off_to_8_7_55_2_84": False,
        },
        "evidence": {
            "mass_origin_reflective_cavity_rule_twenty_ninth_retry_summary": reflective_cavity_twenty_ninth_retry_summary,
            "mass_origin_discrete_spectrum_thirtieth_reopen_refresh_summary": discrete_spectrum_thirtieth_reopen_refresh_summary,
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
