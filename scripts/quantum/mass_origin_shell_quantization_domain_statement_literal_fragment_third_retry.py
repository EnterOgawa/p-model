#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_origin_shell_quantization_domain_statement_literal_fragment_third_retry.py

Step 8.7.55.2.467:
Retry shell-quantization domain-statement-literal-fragment closure after the
token-atom second retry.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]

LITERAL_FRAGMENT_SECOND_RETRY_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_shell_quantization_domain_statement_literal_fragment_second_retry_metrics.json"
TOKEN_ATOM_SECOND_RETRY_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_shell_quantization_domain_statement_token_atom_second_retry_metrics.json"
OUT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_shell_quantization_domain_statement_literal_fragment_third_retry_metrics.json"
OUT_CSV = ROOT / "output" / "public" / "quantum" / "mass_origin_shell_quantization_domain_statement_literal_fragment_third_retry_rows.csv"
DEFAULT_STEP_TAG = "8.7.55.2.467"


# 関数: 現在UTC時刻を ISO 8601 文字列で返す。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: CLI 引数を解釈する。

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retry shell-quantization domain-statement-literal-fragment closure for the third time.")
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

def _build_rows(*, available: bool, nonclosure_reason: str | None) -> List[Dict[str, Any]]:
    return [
        {
            "row_id": "shell_quantization_domain_statement_literal_fragment_third_retry_complete",
            "status": "pass",
            "metric": "shell-quantization domain-statement-literal-fragment third retry complete",
            "value": 1.0,
            "note": "This step retries literal-fragment closure after the token-atom second retry.",
        },
        {
            "row_id": "shell_quantization_domain_statement_literal_fragment_third_retry_available",
            "status": "pass" if available else "reject",
            "metric": "shell-quantization domain-statement literal fragment available after third retry",
            "value": 1.0 if available else 0.0,
            "note": (
                "The shell-quantization domain-statement literal fragment is available after third retry."
                if available
                else f"The third retry remains non-closing: {nonclosure_reason}."
            ),
        },
    ]


# 関数: metrics payload 全体を構成する。

def _build_payload(step_tag: str) -> Dict[str, Any]:
    for path in (LITERAL_FRAGMENT_SECOND_RETRY_JSON, TOKEN_ATOM_SECOND_RETRY_JSON):
        _require_path(path)

    literal_fragment_second_retry = _read_json(LITERAL_FRAGMENT_SECOND_RETRY_JSON)
    token_atom_second_retry = _read_json(TOKEN_ATOM_SECOND_RETRY_JSON)

    literal_fragment_second_retry_summary = literal_fragment_second_retry.get("summary", {})
    token_atom_second_retry_summary = token_atom_second_retry.get("summary", {})

    available = bool(token_atom_second_retry_summary.get("shell_quantization_domain_statement_token_atom_available", False))
    nonclosure_reason = "shell_quantization_domain_statement_symbol_fragment_absent"
    rows = _build_rows(available=available, nonclosure_reason=nonclosure_reason)

    return {
        "generated_utc": _utc_now_iso(),
        "phase": {"phase": 8, "step": step_tag, "name": "shell-quantization domain-statement-literal-fragment third retry"},
        "inputs": {
            "mass_origin_shell_quantization_domain_statement_literal_fragment_second_retry_json": _relative_str(
                LITERAL_FRAGMENT_SECOND_RETRY_JSON
            ),
            "mass_origin_shell_quantization_domain_statement_token_atom_second_retry_json": _relative_str(
                TOKEN_ATOM_SECOND_RETRY_JSON
            ),
        },
        "intent": "Retry whether the shell-quantization domain-statement literal fragment can now close after the token-atom second retry.",
        "formulas": {
            "third_retry_rule": "shell_quantization_domain_statement_literal_fragment_available iff the token-atom second retry promotes a shell-quantization domain-statement token atom",
        },
        "rows": rows,
        "summary": {
            "shell_quantization_domain_statement_literal_fragment_available": available,
            "domain_statement_literal_fragment_kind_or_none": "shell_quantization_domain_statement_literal_fragment" if available else None,
            "domain_statement_literal_fragment_third_retry_nonclosure_reason_or_none": nonclosure_reason,
        },
        "decision": {
            "overall_status": "shell_quantization_domain_statement_literal_fragment_third_retry_frozen_absent",
            "keep_mass_origin_branch_blocked": True,
            "shell_quantization_domain_statement_literal_fragment_available": available,
            "domain_statement_literal_fragment_kind_or_none": "shell_quantization_domain_statement_literal_fragment" if available else None,
            "domain_statement_literal_fragment_third_retry_nonclosure_reason_or_none": nonclosure_reason,
            "hand_off_to_8_7_55_2_84": False,
            "next_required_artifacts": [
                "shell_quantization_domain_statement_symbol_fragment",
                "shell_quantization_domain_statement_literal_fragment",
                "shell_quantization_domain_statement_phrase_fragment",
                "shell_quantization_domain_statement_literal",
                "shell_quantization_domain_statement",
                "shell_quantization_to_domain_relation",
            ],
        },
        "evidence": {
            "shell_quantization_domain_statement_literal_fragment_second_retry_summary": literal_fragment_second_retry_summary,
            "shell_quantization_domain_statement_token_atom_second_retry_summary": token_atom_second_retry_summary,
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
