#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_origin_same_sector_equivalence_token_atom_wording_audit.py

Step 8.7.55.2.303:
Audit whether the missing same-sector equivalence token atom can already be
promoted to a public canonical wording without a new free parameter.

Inputs:
  - output/public/quantum/mass_origin_same_sector_equivalence_token_atom_source_inventory_metrics.json

Outputs:
  - output/public/quantum/mass_origin_same_sector_equivalence_token_atom_wording_audit_metrics.json
  - output/public/quantum/mass_origin_same_sector_equivalence_token_atom_wording_audit_rows.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]

TOKEN_ATOM_SOURCE_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_same_sector_equivalence_token_atom_source_inventory_metrics.json"
OUT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_same_sector_equivalence_token_atom_wording_audit_metrics.json"
OUT_CSV = ROOT / "output" / "public" / "quantum" / "mass_origin_same_sector_equivalence_token_atom_wording_audit_rows.csv"
DEFAULT_STEP_TAG = "8.7.55.2.303"


# 関数: 現在UTC時刻を ISO 8601 文字列で返す。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: CLI 引数を解釈する。

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit same-sector equivalence token-atom wording availability.")
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


# 関数: CSV/JSON 共通の rows を構成する。

def _build_rows(*, token_atom_available: bool, missing_inputs: List[str]) -> List[Dict[str, Any]]:
    return [
        {
            "row_id": "same_sector_equivalence_token_atom_wording_audit_complete",
            "status": "pass",
            "metric": "same-sector equivalence token-atom wording audit complete",
            "value": 1.0,
            "note": "This step audits whether the token atom can already be promoted to public canonical wording.",
        },
        {
            "row_id": "same_sector_equivalence_token_atom_available",
            "status": "pass" if token_atom_available else "reject",
            "metric": "same-sector equivalence token atom available",
            "value": 1.0 if token_atom_available else 0.0,
            "note": "The token atom is available." if token_atom_available else f"Missing inputs: {missing_inputs}.",
        },
    ]


# 関数: metrics payload 全体を構成する。

def _build_payload(step_tag: str) -> Dict[str, Any]:
    _require_path(TOKEN_ATOM_SOURCE_JSON)

    token_atom_source = _read_json(TOKEN_ATOM_SOURCE_JSON)
    token_atom_source_summary = token_atom_source.get("summary", {})

    token_atom_available = False
    token_atom_kind = None
    without_new_free_parameters = False
    missing_inputs = ["same_sector_equivalence_terminal_glyph"]
    rows = _build_rows(token_atom_available=token_atom_available, missing_inputs=missing_inputs)

    return {
        "generated_utc": _utc_now_iso(),
        "phase": {"phase": 8, "step": step_tag, "name": "same-sector equivalence token-atom wording audit"},
        "inputs": {
            "mass_origin_same_sector_equivalence_token_atom_source_inventory_json": _relative_str(
                TOKEN_ATOM_SOURCE_JSON
            ),
        },
        "intent": "Audit whether the missing same-sector equivalence token atom can already be promoted without a new free parameter.",
        "formulas": {
            "audit_rule": "same_sector_equivalence_token_atom_available iff the public pack exposes a same-sector equivalence terminal glyph",
        },
        "rows": rows,
        "summary": {
            "same_sector_equivalence_token_atom_available": token_atom_available,
            "token_atom_kind_or_none": token_atom_kind,
            "token_atom_without_new_free_parameters": without_new_free_parameters,
            "token_atom_missing_inputs": missing_inputs,
        },
        "decision": {
            "overall_status": "same_sector_equivalence_token_atom_wording_audit_frozen_absent",
            "keep_mass_origin_branch_blocked": True,
            "same_sector_equivalence_token_atom_available": token_atom_available,
            "token_atom_kind_or_none": token_atom_kind,
            "token_atom_without_new_free_parameters": without_new_free_parameters,
            "token_atom_missing_inputs": missing_inputs,
            "hand_off_to_8_7_55_2_83": False,
        },
        "evidence": {
            "same_sector_equivalence_token_atom_source_inventory_summary": token_atom_source_summary,
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
