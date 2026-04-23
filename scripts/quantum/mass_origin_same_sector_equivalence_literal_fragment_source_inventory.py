#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_origin_same_sector_equivalence_literal_fragment_source_inventory.py

Step 8.7.55.2.293:
Inventory the current public-canonical source candidates for the missing
same-sector equivalence literal fragment.

Inputs:
  - output/public/quantum/mass_origin_same_sector_equivalence_phrase_fragment_source_inventory_metrics.json
  - output/public/quantum/mass_origin_same_sector_equivalence_literal_fragment_route_contract_metrics.json

Outputs:
  - output/public/quantum/mass_origin_same_sector_equivalence_literal_fragment_source_inventory_metrics.json
  - output/public/quantum/mass_origin_same_sector_equivalence_literal_fragment_source_inventory_rows.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]

PHRASE_FRAGMENT_SOURCE_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_same_sector_equivalence_phrase_fragment_source_inventory_metrics.json"
LITERAL_FRAGMENT_ROUTE_CONTRACT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_same_sector_equivalence_literal_fragment_route_contract_metrics.json"
OUT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_same_sector_equivalence_literal_fragment_source_inventory_metrics.json"
OUT_CSV = ROOT / "output" / "public" / "quantum" / "mass_origin_same_sector_equivalence_literal_fragment_source_inventory_rows.csv"
DEFAULT_STEP_TAG = "8.7.55.2.293"


# 関数: 現在UTC時刻を ISO 8601 文字列で返す。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: CLI 引数を解釈する。

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inventory source candidates for the missing same-sector equivalence literal fragment.",
    )
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


# 関数: CSV/JSON 共有の rows を構成する。

def _build_rows(
    *,
    required_sources: List[str],
    present_sources: List[str],
    missing_sources: List[str],
    first_route_to_close_or_none: str | None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = [
        {
            "row_id": "same_sector_equivalence_literal_fragment_source_inventory_complete",
            "status": "pass",
            "metric": "same-sector equivalence literal-fragment source inventory complete",
            "value": 1.0,
            "note": "This step inventories concrete public-canonical sources for the missing same-sector equivalence literal fragment.",
        },
        {
            "row_id": "same_sector_equivalence_literal_fragment_source_inventory_required_count",
            "status": "pass",
            "metric": "required source count for same-sector equivalence literal-fragment route",
            "value": float(len(required_sources)),
            "note": f"Required sources: {required_sources}.",
        },
        {
            "row_id": "same_sector_equivalence_literal_fragment_source_inventory_present_count",
            "status": "pass",
            "metric": "present source count for same-sector equivalence literal-fragment route",
            "value": float(len(present_sources)),
            "note": f"Present sources: {present_sources}.",
        },
        {
            "row_id": "same_sector_equivalence_literal_fragment_source_inventory_missing_count",
            "status": "watch",
            "metric": "missing source count for same-sector equivalence literal-fragment route",
            "value": float(len(missing_sources)),
            "note": f"Missing sources: {missing_sources}.",
        },
    ]

    for source in required_sources:
        source_present = source in present_sources
        rows.append(
            {
                "row_id": f"same_sector_equivalence_literal_fragment_source_{source}",
                "status": "pass" if source_present else "watch",
                "metric": f"source availability for {source}",
                "value": 1.0 if source_present else 0.0,
                "note": (
                    f"{source} is already available in the current public canonical pack."
                    if source_present
                    else f"{source} is still missing from the current public canonical pack."
                ),
            }
        )

    rows.append(
        {
            "row_id": "same_sector_equivalence_literal_fragment_source_inventory_first_route",
            "status": "watch",
            "metric": "first residual source to close after literal-fragment source inventory",
            "value": 1.0 if first_route_to_close_or_none else 0.0,
            "note": f"The next closure attempt starts from {first_route_to_close_or_none}.",
        }
    )
    return rows


# 関数: metrics payload 全体を構成する。

def _build_payload(step_tag: str) -> Dict[str, Any]:
    for path in (PHRASE_FRAGMENT_SOURCE_JSON, LITERAL_FRAGMENT_ROUTE_CONTRACT_JSON):
        _require_path(path)

    phrase_fragment_source = _read_json(PHRASE_FRAGMENT_SOURCE_JSON)
    literal_fragment_route_contract = _read_json(LITERAL_FRAGMENT_ROUTE_CONTRACT_JSON)

    phrase_fragment_source_summary = phrase_fragment_source.get("summary", {})
    literal_fragment_route_contract_summary = literal_fragment_route_contract.get("summary", {})

    required_sources = [
        "chi_definition",
        "same_sector_contract",
        "same_sector_equivalence_literal_fragment",
        "equivalence_relation_operator",
        "no_new_free_parameter_wording",
    ]
    present_from_phrase_fragment = [
        str(item) for item in phrase_fragment_source_summary.get("present_phrase_fragment_sources", [])
    ]
    present_sources = [source for source in required_sources if source in present_from_phrase_fragment]
    missing_sources = [source for source in required_sources if source not in present_sources]
    first_route_to_close_or_none = "same_sector_equivalence_token_atom"
    rows = _build_rows(
        required_sources=required_sources,
        present_sources=present_sources,
        missing_sources=missing_sources,
        first_route_to_close_or_none=first_route_to_close_or_none,
    )

    return {
        "generated_utc": _utc_now_iso(),
        "phase": {"phase": 8, "step": step_tag, "name": "same-sector equivalence literal fragment source inventory"},
        "inputs": {
            "mass_origin_same_sector_equivalence_phrase_fragment_source_inventory_json": _relative_str(
                PHRASE_FRAGMENT_SOURCE_JSON
            ),
            "mass_origin_same_sector_equivalence_literal_fragment_route_contract_json": _relative_str(
                LITERAL_FRAGMENT_ROUTE_CONTRACT_JSON
            ),
        },
        "intent": "Inventory source candidates that could instantiate the missing same-sector equivalence literal fragment without a new fit parameter.",
        "formulas": {
            "inventory_rule": "the literal-fragment route can close only after the public pack exposes chi definition, the same-sector contract, an explicit same-sector equivalence literal fragment, an equivalence relation operator, and the no-new-free-parameter envelope",
            "current_absence": "the current pack still lacks the same-sector equivalence literal fragment and the relation operator, so the first fine-grained closure attempt starts from the missing same-sector equivalence token atom",
        },
        "rows": rows,
        "summary": {
            "required_literal_fragment_sources": required_sources,
            "present_literal_fragment_sources": present_sources,
            "missing_literal_fragment_sources": missing_sources,
            "first_route_to_close_or_none": first_route_to_close_or_none,
            "literal_fragment_source_inventory_ready": True,
        },
        "decision": {
            "overall_status": "same_sector_equivalence_literal_fragment_source_inventory_frozen",
            "keep_mass_origin_branch_blocked": True,
            "required_literal_fragment_sources": required_sources,
            "present_literal_fragment_sources": present_sources,
            "missing_literal_fragment_sources": missing_sources,
            "first_route_to_close_or_none": first_route_to_close_or_none,
            "literal_fragment_source_inventory_ready": True,
            "hand_off_to_8_7_55_2_83": False,
            "next_required_artifacts": [
                "same_sector_equivalence_literal_fragment",
                "same_sector_equivalence_phrase_fragment",
                "same_sector_equivalence_literal",
                "same_sector_equivalence_statement",
                "same_sector_equivalence_rule",
                "chi_star_or_same_sector_proxy",
                "anchor_normalized_g3w_public_value",
                "r3_target",
                "single_public_vpp_shape",
                "positive_particle_sector_chi_p_to_vpp_public_artifact",
                "solver_ready_row_promoted_to_pass",
            ],
        },
        "evidence": {
            "same_sector_equivalence_phrase_fragment_source_inventory_summary": phrase_fragment_source_summary,
            "same_sector_equivalence_literal_fragment_route_contract_summary": literal_fragment_route_contract_summary,
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
