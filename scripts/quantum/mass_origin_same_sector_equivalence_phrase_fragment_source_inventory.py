#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_origin_same_sector_equivalence_phrase_fragment_source_inventory.py

Step 8.7.55.2.285:
Inventory the current public-canonical source candidates for the missing
same-sector equivalence phrase fragment.

Inputs:
  - output/public/quantum/mass_origin_same_sector_equivalence_literal_source_inventory_metrics.json
  - output/public/quantum/mass_origin_same_sector_equivalence_phrase_fragment_route_contract_metrics.json

Outputs:
  - output/public/quantum/mass_origin_same_sector_equivalence_phrase_fragment_source_inventory_metrics.json
  - output/public/quantum/mass_origin_same_sector_equivalence_phrase_fragment_source_inventory_rows.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]

LITERAL_SOURCE_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_same_sector_equivalence_literal_source_inventory_metrics.json"
PHRASE_FRAGMENT_ROUTE_CONTRACT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_same_sector_equivalence_phrase_fragment_route_contract_metrics.json"
OUT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_same_sector_equivalence_phrase_fragment_source_inventory_metrics.json"
OUT_CSV = ROOT / "output" / "public" / "quantum" / "mass_origin_same_sector_equivalence_phrase_fragment_source_inventory_rows.csv"
DEFAULT_STEP_TAG = "8.7.55.2.285"


# 関数: `_utc_now_iso` の入出力契約と処理意図を定義する。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_parse_args` の入出力契約と処理意図を定義する。

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inventory source candidates for the missing same-sector equivalence phrase fragment.",
    )
    parser.add_argument("--step-tag", default=DEFAULT_STEP_TAG, help="Roadmap step tag to stamp into the output payload.")
    return parser.parse_args()


# 関数: `_require_path` の入出力契約と処理意図を定義する。

def _require_path(path: Path) -> None:
    if not path.exists():
        raise SystemExit(f"[fail] missing required input: {path}")


# 関数: `_read_json` の入出力契約と処理意図を定義する。

def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


# 関数: `_relative_str` の入出力契約と処理意図を定義する。

def _relative_str(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


# 関数: `_build_rows` の入出力契約と処理意図を定義する。

def _build_rows(
    *,
    required_sources: List[str],
    present_sources: List[str],
    missing_sources: List[str],
    first_route_to_close_or_none: str | None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = [
        {
            "row_id": "same_sector_equivalence_phrase_fragment_source_inventory_complete",
            "status": "pass",
            "metric": "same-sector equivalence phrase-fragment source inventory complete",
            "value": 1.0,
            "note": "This step inventories concrete public-canonical sources for the missing same-sector equivalence phrase fragment.",
        },
        {
            "row_id": "same_sector_equivalence_phrase_fragment_source_inventory_required_count",
            "status": "pass",
            "metric": "required source count for same-sector equivalence phrase-fragment route",
            "value": float(len(required_sources)),
            "note": f"Required sources: {required_sources}.",
        },
        {
            "row_id": "same_sector_equivalence_phrase_fragment_source_inventory_present_count",
            "status": "pass",
            "metric": "present source count for same-sector equivalence phrase-fragment route",
            "value": float(len(present_sources)),
            "note": f"Present sources: {present_sources}.",
        },
        {
            "row_id": "same_sector_equivalence_phrase_fragment_source_inventory_missing_count",
            "status": "watch",
            "metric": "missing source count for same-sector equivalence phrase-fragment route",
            "value": float(len(missing_sources)),
            "note": f"Missing sources: {missing_sources}.",
        },
    ]

    for source in required_sources:
        source_present = source in present_sources
        rows.append(
            {
                "row_id": f"same_sector_equivalence_phrase_fragment_source_{source}",
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
            "row_id": "same_sector_equivalence_phrase_fragment_source_inventory_first_route",
            "status": "watch",
            "metric": "first residual source to close after phrase-fragment source inventory",
            "value": 1.0 if first_route_to_close_or_none else 0.0,
            "note": f"The next closure attempt starts from {first_route_to_close_or_none}.",
        }
    )
    return rows


# 関数: `_build_payload` の入出力契約と処理意図を定義する。

def _build_payload(step_tag: str) -> Dict[str, Any]:
    for path in (LITERAL_SOURCE_JSON, PHRASE_FRAGMENT_ROUTE_CONTRACT_JSON):
        _require_path(path)

    literal_source = _read_json(LITERAL_SOURCE_JSON)
    phrase_fragment_route_contract = _read_json(PHRASE_FRAGMENT_ROUTE_CONTRACT_JSON)

    literal_source_summary = literal_source.get("summary", {})
    phrase_fragment_route_contract_summary = phrase_fragment_route_contract.get("summary", {})

    required_sources = [
        "chi_definition",
        "same_sector_contract",
        "same_sector_equivalence_phrase_fragment",
        "equivalence_relation_operator",
        "no_new_free_parameter_wording",
    ]
    present_from_literal = [str(item) for item in literal_source_summary.get("present_literal_sources", [])]
    present_sources = [source for source in required_sources if source in present_from_literal]
    missing_sources = [source for source in required_sources if source not in present_sources]
    first_route_to_close_or_none = "same_sector_equivalence_literal_fragment"
    rows = _build_rows(
        required_sources=required_sources,
        present_sources=present_sources,
        missing_sources=missing_sources,
        first_route_to_close_or_none=first_route_to_close_or_none,
    )

    return {
        "generated_utc": _utc_now_iso(),
        "phase": {"phase": 8, "step": step_tag, "name": "same-sector equivalence phrase-fragment source inventory"},
        "inputs": {
            "mass_origin_same_sector_equivalence_literal_source_inventory_json": _relative_str(LITERAL_SOURCE_JSON),
            "mass_origin_same_sector_equivalence_phrase_fragment_route_contract_json": _relative_str(
                PHRASE_FRAGMENT_ROUTE_CONTRACT_JSON
            ),
        },
        "intent": "Inventory source candidates that could instantiate the missing same-sector equivalence phrase fragment without a new fit parameter.",
        "formulas": {
            "inventory_rule": "the phrase-fragment route can close only after the public pack exposes chi definition, the same-sector contract, an explicit same-sector equivalence phrase fragment, an equivalence relation operator, and the no-new-free-parameter envelope",
            "current_absence": "the current pack still lacks the same-sector equivalence phrase fragment and the relation operator, so the first fine-grained closure attempt starts from the missing same-sector equivalence literal fragment",
        },
        "rows": rows,
        "summary": {
            "required_phrase_fragment_sources": required_sources,
            "present_phrase_fragment_sources": present_sources,
            "missing_phrase_fragment_sources": missing_sources,
            "first_route_to_close_or_none": first_route_to_close_or_none,
            "phrase_fragment_source_inventory_ready": True,
        },
        "decision": {
            "overall_status": "same_sector_equivalence_phrase_fragment_source_inventory_frozen",
            "keep_mass_origin_branch_blocked": True,
            "required_phrase_fragment_sources": required_sources,
            "present_phrase_fragment_sources": present_sources,
            "missing_phrase_fragment_sources": missing_sources,
            "first_route_to_close_or_none": first_route_to_close_or_none,
            "phrase_fragment_source_inventory_ready": True,
            "hand_off_to_8_7_55_2_83": False,
            "next_required_artifacts": [
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
            "same_sector_equivalence_literal_source_inventory_summary": literal_source_summary,
            "same_sector_equivalence_phrase_fragment_route_contract_summary": phrase_fragment_route_contract_summary,
        },
    }


# 関数: `_write_csv` の入出力契約と処理意図を定義する。

def _write_csv(rows: List[Dict[str, Any]]) -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["row_id", "status", "metric", "value", "note"])
        writer.writeheader()
        writer.writerows(rows)


# 関数: `main` の入出力契約と処理意図を定義する。

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
