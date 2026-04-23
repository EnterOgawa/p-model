#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_origin_same_sector_equivalence_literal_fragment_route_contract.py

Step 8.7.55.2.292:
Freeze the next residual branch contract after the sixth retry route narrows
the unresolved core to the missing same-sector equivalence literal fragment.

Inputs:
  - output/public/quantum/mass_origin_same_sector_equivalence_phrase_fragment_source_inventory_metrics.json
  - output/public/quantum/mass_origin_same_sector_equivalence_phrase_fragment_wording_audit_metrics.json
  - output/public/quantum/mass_origin_anchor_local_shape_gate_sixth_retry_refresh_metrics.json

Outputs:
  - output/public/quantum/mass_origin_same_sector_equivalence_literal_fragment_route_contract_metrics.json
  - output/public/quantum/mass_origin_same_sector_equivalence_literal_fragment_route_contract_rows.csv
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
PHRASE_FRAGMENT_WORDING_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_same_sector_equivalence_phrase_fragment_wording_audit_metrics.json"
SHAPE_GATE_SIXTH_RETRY_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_anchor_local_shape_gate_sixth_retry_refresh_metrics.json"
OUT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_same_sector_equivalence_literal_fragment_route_contract_metrics.json"
OUT_CSV = ROOT / "output" / "public" / "quantum" / "mass_origin_same_sector_equivalence_literal_fragment_route_contract_rows.csv"
DEFAULT_STEP_TAG = "8.7.55.2.292"


# 関数: `_utc_now_iso` の入出力契約と処理意図を定義する。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_parse_args` の入出力契約と処理意図を定義する。

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze the next residual branch contract for the missing same-sector equivalence literal fragment.",
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

def _build_rows(*, required_route_items: List[str], split_contract_ready: bool) -> List[Dict[str, Any]]:
    return [
        {
            "row_id": "same_sector_equivalence_literal_fragment_route_contract_complete",
            "status": "pass",
            "metric": "same-sector equivalence literal-fragment residual contract complete",
            "value": 1.0,
            "note": "This step freezes the next residual branch after the sixth retry route narrows to the missing same-sector equivalence literal fragment.",
        },
        {
            "row_id": "same_sector_equivalence_literal_fragment_route_contract_required_items",
            "status": "watch",
            "metric": "required route items for same-sector equivalence literal-fragment branch",
            "value": float(len(required_route_items)),
            "note": f"Required route items: {required_route_items}.",
        },
        {
            "row_id": "same_sector_equivalence_literal_fragment_route_contract_split_ready",
            "status": "pass" if split_contract_ready else "reject",
            "metric": "same-sector equivalence literal-fragment residual split contract ready",
            "value": 1.0 if split_contract_ready else 0.0,
            "note": (
                "The next branch can now audit concrete public-canonical source candidates for the missing same-sector equivalence literal fragment."
                if split_contract_ready
                else "The same-sector equivalence literal-fragment residual branch is not yet formalized."
            ),
        },
    ]


# 関数: `_build_payload` の入出力契約と処理意図を定義する。

def _build_payload(step_tag: str) -> Dict[str, Any]:
    for path in (PHRASE_FRAGMENT_SOURCE_JSON, PHRASE_FRAGMENT_WORDING_JSON, SHAPE_GATE_SIXTH_RETRY_JSON):
        _require_path(path)

    phrase_fragment_source = _read_json(PHRASE_FRAGMENT_SOURCE_JSON)
    phrase_fragment_wording = _read_json(PHRASE_FRAGMENT_WORDING_JSON)
    shape_gate_sixth_retry = _read_json(SHAPE_GATE_SIXTH_RETRY_JSON)

    phrase_fragment_source_summary = phrase_fragment_source.get("summary", {})
    phrase_fragment_wording_summary = phrase_fragment_wording.get("summary", {})
    shape_gate_sixth_retry_summary = shape_gate_sixth_retry.get("summary", {})

    required_route_items = [
        "chi_definition",
        "same_sector_contract",
        "same_sector_equivalence_literal_fragment",
        "equivalence_relation_operator",
        "no_new_free_parameter_wording",
    ]
    split_contract_ready = True
    rows = _build_rows(required_route_items=required_route_items, split_contract_ready=split_contract_ready)

    return {
        "generated_utc": _utc_now_iso(),
        "phase": {"phase": 8, "step": step_tag, "name": "same-sector equivalence literal fragment route contract freeze"},
        "inputs": {
            "mass_origin_same_sector_equivalence_phrase_fragment_source_inventory_json": _relative_str(
                PHRASE_FRAGMENT_SOURCE_JSON
            ),
            "mass_origin_same_sector_equivalence_phrase_fragment_wording_audit_json": _relative_str(
                PHRASE_FRAGMENT_WORDING_JSON
            ),
            "mass_origin_anchor_local_shape_gate_sixth_retry_refresh_json": _relative_str(
                SHAPE_GATE_SIXTH_RETRY_JSON
            ),
        },
        "intent": "Freeze the next residual branch contract after the sixth retry route narrows the unresolved core to the missing same-sector equivalence literal fragment.",
        "formulas": {
            "residual_rule": "same_sector_equivalence_phrase_fragment can close only after the public pack states a same-sector equivalence literal fragment together with a relation operator",
            "closure_rule": "same_sector_equivalence_literal, same_sector_equivalence_statement, same_sector_equivalence_rule, chi_star_or_same_sector_proxy, anchor_normalized_g3w_public_value, and R_3_target can only retry-close after the same-sector equivalence literal-fragment route succeeds",
        },
        "rows": rows,
        "summary": {
            "missing_same_sector_equivalence_literal_fragment_artifact": "same_sector_equivalence_literal_fragment",
            "required_route_items": required_route_items,
            "split_contract_ready": split_contract_ready,
        },
        "decision": {
            "overall_status": "same_sector_equivalence_literal_fragment_route_contract_frozen",
            "keep_mass_origin_branch_blocked": True,
            "missing_same_sector_equivalence_literal_fragment_artifact": "same_sector_equivalence_literal_fragment",
            "split_contract_ready": split_contract_ready,
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
            "same_sector_equivalence_phrase_fragment_wording_audit_summary": phrase_fragment_wording_summary,
            "anchor_local_shape_gate_sixth_retry_refresh_summary": shape_gate_sixth_retry_summary,
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
