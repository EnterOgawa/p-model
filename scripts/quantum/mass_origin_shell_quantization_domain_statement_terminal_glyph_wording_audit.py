#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_origin_shell_quantization_domain_statement_terminal_glyph_wording_audit.py

Step 8.7.55.2.738:
Audit whether the missing shell-quantization domain-statement terminal glyph can
already be promoted to a public canonical wording without a new free parameter.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]

TERMINAL_GLYPH_SOURCE_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_shell_quantization_domain_statement_terminal_glyph_source_inventory_metrics.json"
OUT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_shell_quantization_domain_statement_terminal_glyph_wording_audit_metrics.json"
OUT_CSV = ROOT / "output" / "public" / "quantum" / "mass_origin_shell_quantization_domain_statement_terminal_glyph_wording_audit_rows.csv"
DEFAULT_STEP_TAG = "8.7.55.2.738"


# 関数: 現在UTC時刻を ISO 8601 文字列で返す。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: CLI 引数を解釈する。

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit shell-quantization domain-statement terminal-glyph wording availability.")
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

def _build_rows(terminal_glyph_available: bool, missing_inputs: List[str]) -> List[Dict[str, Any]]:
    return [
        {
            "row_id": "shell_quantization_domain_statement_terminal_glyph_wording_audit_complete",
            "status": "pass",
            "metric": "shell-quantization domain-statement-terminal-glyph wording audit complete",
            "value": 1.0,
            "note": "This step audits whether the terminal glyph can already be promoted to public canonical wording.",
        },
        {
            "row_id": "shell_quantization_domain_statement_terminal_glyph_available",
            "status": "pass" if terminal_glyph_available else "reject",
            "metric": "shell-quantization domain-statement terminal glyph available",
            "value": 1.0 if terminal_glyph_available else 0.0,
            "note": "The terminal glyph is available." if terminal_glyph_available else f"Missing inputs: {missing_inputs}.",
        },
    ]


# 関数: metrics payload 全体を構成する。

def _build_payload(step_tag: str) -> Dict[str, Any]:
    _require_path(TERMINAL_GLYPH_SOURCE_JSON)
    terminal_glyph_source = _read_json(TERMINAL_GLYPH_SOURCE_JSON)
    terminal_glyph_source_summary = terminal_glyph_source.get("summary", {})

    terminal_glyph_available = False
    terminal_glyph_kind = None
    without_new_free_parameters = False
    missing_inputs = ["shell_quantization_domain_statement_symbol_fragment"]
    rows = _build_rows(terminal_glyph_available, missing_inputs)
    return {
        "generated_utc": _utc_now_iso(),
        "phase": {"phase": 8, "step": step_tag, "name": "shell-quantization domain-statement-terminal-glyph wording audit"},
        "inputs": {
            "mass_origin_shell_quantization_domain_statement_terminal_glyph_source_inventory_json": _relative_str(TERMINAL_GLYPH_SOURCE_JSON),
        },
        "intent": "Audit whether the missing shell-quantization domain-statement terminal glyph can already be promoted without a new free parameter.",
        "formulas": {
            "audit_rule": "shell_quantization_domain_statement_terminal_glyph_available iff the public pack exposes a shell-quantization domain-statement symbol fragment",
        },
        "rows": rows,
        "summary": {
            "shell_quantization_domain_statement_terminal_glyph_available": terminal_glyph_available,
            "domain_statement_terminal_glyph_kind_or_none": terminal_glyph_kind,
            "domain_statement_terminal_glyph_without_new_free_parameters": without_new_free_parameters,
            "domain_statement_terminal_glyph_missing_inputs": missing_inputs,
        },
        "decision": {
            "overall_status": "shell_quantization_domain_statement_terminal_glyph_wording_audit_frozen_absent",
            "keep_mass_origin_branch_blocked": True,
            "shell_quantization_domain_statement_terminal_glyph_available": terminal_glyph_available,
            "domain_statement_terminal_glyph_kind_or_none": terminal_glyph_kind,
            "domain_statement_terminal_glyph_without_new_free_parameters": without_new_free_parameters,
            "domain_statement_terminal_glyph_missing_inputs": missing_inputs,
            "hand_off_to_8_7_55_2_84": False,
        },
        "evidence": {
            "shell_quantization_domain_statement_terminal_glyph_source_inventory_summary": terminal_glyph_source_summary,
        },
    }


# 関数: rows を CSV 出力する。

def _write_csv(rows: List[Dict[str, Any]]) -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["row_id", "status", "metric", "value", "note"])
        writer.writeheader()
        writer.writerows(rows)


# 関数: エントリポイントとして payload を生成して保存する。

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
