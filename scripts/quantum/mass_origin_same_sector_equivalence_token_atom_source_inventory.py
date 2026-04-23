#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_origin_same_sector_equivalence_token_atom_source_inventory.py

Step 8.7.55.2.302:
Inventory which public-canonical source candidates are already present for the
missing same-sector equivalence token atom.

Inputs:
  - output/public/quantum/mass_origin_same_sector_equivalence_literal_fragment_source_inventory_metrics.json
  - output/public/quantum/mass_origin_same_sector_equivalence_token_atom_route_contract_metrics.json

Outputs:
  - output/public/quantum/mass_origin_same_sector_equivalence_token_atom_source_inventory_metrics.json
  - output/public/quantum/mass_origin_same_sector_equivalence_token_atom_source_inventory_rows.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]

LITERAL_FRAGMENT_SOURCE_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_same_sector_equivalence_literal_fragment_source_inventory_metrics.json"
TOKEN_ATOM_ROUTE_CONTRACT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_same_sector_equivalence_token_atom_route_contract_metrics.json"
OUT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_same_sector_equivalence_token_atom_source_inventory_metrics.json"
OUT_CSV = ROOT / "output" / "public" / "quantum" / "mass_origin_same_sector_equivalence_token_atom_source_inventory_rows.csv"
DEFAULT_STEP_TAG = "8.7.55.2.302"


# 関数: 現在UTC時刻を ISO 8601 文字列で返す。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: CLI 引数を解釈する。

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inventory source candidates for the missing same-sector equivalence token atom.")
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

def _build_rows(*, required_sources: List[str], present_sources: List[str], missing_sources: List[str], inventory_ready: bool, first_route: str) -> List[Dict[str, Any]]:
    return [
        {
            "row_id": "same_sector_equivalence_token_atom_source_inventory_complete",
            "status": "pass",
            "metric": "same-sector equivalence token-atom source inventory complete",
            "value": 1.0,
            "note": "This step inventories current public-canonical source candidates for the missing same-sector equivalence token atom.",
        },
        {
            "row_id": "same_sector_equivalence_token_atom_source_inventory_required_count",
            "status": "watch",
            "metric": "required same-sector equivalence token-atom source count",
            "value": float(len(required_sources)),
            "note": f"Required token-atom sources: {required_sources}.",
        },
        {
            "row_id": "same_sector_equivalence_token_atom_source_inventory_present_count",
            "status": "watch",
            "metric": "present same-sector equivalence token-atom source count",
            "value": float(len(present_sources)),
            "note": f"Present token-atom sources: {present_sources}.",
        },
        {
            "row_id": "same_sector_equivalence_token_atom_source_inventory_missing_count",
            "status": "reject" if missing_sources else "pass",
            "metric": "missing same-sector equivalence token-atom source count",
            "value": float(len(missing_sources)),
            "note": f"Missing token-atom sources: {missing_sources}.",
        },
        {
            "row_id": "same_sector_equivalence_token_atom_source_inventory_first_route",
            "status": "watch",
            "metric": "first route to close after token-atom source inventory",
            "value": 1.0,
            "note": f"The next closure attempt starts from {first_route}.",
        },
        {
            "row_id": "same_sector_equivalence_token_atom_source_inventory_ready",
            "status": "pass" if inventory_ready else "reject",
            "metric": "same-sector equivalence token-atom source inventory ready",
            "value": 1.0 if inventory_ready else 0.0,
            "note": "The token-atom source inventory is formalized." if inventory_ready else "The token-atom source inventory is not yet formalized.",
        },
    ]


# 関数: metrics payload 全体を構成する。

def _build_payload(step_tag: str) -> Dict[str, Any]:
    for path in (LITERAL_FRAGMENT_SOURCE_JSON, TOKEN_ATOM_ROUTE_CONTRACT_JSON):
        _require_path(path)

    literal_fragment_source = _read_json(LITERAL_FRAGMENT_SOURCE_JSON)
    token_atom_route_contract = _read_json(TOKEN_ATOM_ROUTE_CONTRACT_JSON)

    literal_fragment_source_summary = literal_fragment_source.get("summary", {})
    token_atom_route_contract_summary = token_atom_route_contract.get("summary", {})

    required_sources = [
        "chi_definition",
        "same_sector_contract",
        "same_sector_equivalence_token_atom",
        "equivalence_relation_operator",
        "no_new_free_parameter_wording",
    ]
    present_sources = [
        "chi_definition",
        "same_sector_contract",
        "no_new_free_parameter_wording",
    ]
    missing_sources = [
        "same_sector_equivalence_token_atom",
        "equivalence_relation_operator",
    ]
    first_route = "same_sector_equivalence_terminal_glyph"
    inventory_ready = True
    rows = _build_rows(
        required_sources=required_sources,
        present_sources=present_sources,
        missing_sources=missing_sources,
        inventory_ready=inventory_ready,
        first_route=first_route,
    )

    return {
        "generated_utc": _utc_now_iso(),
        "phase": {"phase": 8, "step": step_tag, "name": "same-sector equivalence token-atom source inventory"},
        "inputs": {
            "mass_origin_same_sector_equivalence_literal_fragment_source_inventory_json": _relative_str(
                LITERAL_FRAGMENT_SOURCE_JSON
            ),
            "mass_origin_same_sector_equivalence_token_atom_route_contract_json": _relative_str(
                TOKEN_ATOM_ROUTE_CONTRACT_JSON
            ),
        },
        "intent": "Inventory how much of the missing same-sector equivalence token atom can already be sourced from the current public canonical pack.",
        "rows": rows,
        "summary": {
            "required_token_atom_sources": required_sources,
            "present_token_atom_sources": present_sources,
            "missing_token_atom_sources": missing_sources,
            "first_route_to_close_or_none": first_route,
            "token_atom_source_inventory_ready": inventory_ready,
        },
        "decision": {
            "overall_status": "same_sector_equivalence_token_atom_source_inventory_frozen",
            "keep_mass_origin_branch_blocked": True,
            "required_token_atom_sources": required_sources,
            "present_token_atom_sources": present_sources,
            "missing_token_atom_sources": missing_sources,
            "first_route_to_close_or_none": first_route,
            "token_atom_source_inventory_ready": inventory_ready,
            "hand_off_to_8_7_55_2_83": False,
        },
        "evidence": {
            "same_sector_equivalence_literal_fragment_source_inventory_summary": literal_fragment_source_summary,
            "same_sector_equivalence_token_atom_route_contract_summary": token_atom_route_contract_summary,
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
