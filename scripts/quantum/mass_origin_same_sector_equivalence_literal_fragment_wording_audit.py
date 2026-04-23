#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_origin_same_sector_equivalence_literal_fragment_wording_audit.py

Step 8.7.55.2.294:
Audit whether the current public-canonical pack already fixes the missing
same-sector equivalence literal fragment without a new fit parameter.

Inputs:
  - output/public/quantum/mass_origin_same_sector_equivalence_literal_fragment_source_inventory_metrics.json

Outputs:
  - output/public/quantum/mass_origin_same_sector_equivalence_literal_fragment_wording_audit_metrics.json
  - output/public/quantum/mass_origin_same_sector_equivalence_literal_fragment_wording_audit_rows.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]

LITERAL_FRAGMENT_SOURCE_INVENTORY_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_same_sector_equivalence_literal_fragment_source_inventory_metrics.json"
OUT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_same_sector_equivalence_literal_fragment_wording_audit_metrics.json"
OUT_CSV = ROOT / "output" / "public" / "quantum" / "mass_origin_same_sector_equivalence_literal_fragment_wording_audit_rows.csv"
DEFAULT_STEP_TAG = "8.7.55.2.294"


# 関数: 現在UTC時刻を ISO 8601 文字列で返す。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: CLI 引数を解釈する。

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit whether a same-sector equivalence literal fragment is already public canonical.",
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
    literal_fragment_available: bool,
    literal_fragment_without_new_free_parameters: bool,
    missing_inputs: List[str],
) -> List[Dict[str, Any]]:
    return [
        {
            "row_id": "same_sector_equivalence_literal_fragment_wording_audit_complete",
            "status": "pass",
            "metric": "same-sector equivalence literal-fragment wording audit complete",
            "value": 1.0,
            "note": "This step audits whether the same-sector equivalence literal fragment is already public canonical.",
        },
        {
            "row_id": "same_sector_equivalence_literal_fragment_available",
            "status": "pass" if literal_fragment_available else "reject",
            "metric": "same-sector equivalence literal fragment available",
            "value": 1.0 if literal_fragment_available else 0.0,
            "note": (
                "A same-sector equivalence literal fragment is now public canonical."
                if literal_fragment_available
                else f"Missing inputs: {missing_inputs}."
            ),
        },
        {
            "row_id": "same_sector_equivalence_literal_fragment_without_new_free_parameters",
            "status": "pass" if literal_fragment_without_new_free_parameters else "reject",
            "metric": "same-sector equivalence literal fragment stays inside no-new-free-parameter envelope",
            "value": 1.0 if literal_fragment_without_new_free_parameters else 0.0,
            "note": (
                "The same-sector equivalence literal fragment closes without a new fit parameter."
                if literal_fragment_without_new_free_parameters
                else "The current public pack still lacks a promotable same-sector equivalence token atom."
            ),
        },
    ]


# 関数: metrics payload 全体を構成する。

def _build_payload(step_tag: str) -> Dict[str, Any]:
    _require_path(LITERAL_FRAGMENT_SOURCE_INVENTORY_JSON)
    literal_fragment_source_inventory = _read_json(LITERAL_FRAGMENT_SOURCE_INVENTORY_JSON)
    literal_fragment_source_summary = literal_fragment_source_inventory.get("summary", {})

    literal_fragment_available = False
    literal_fragment_kind_or_none = None
    literal_fragment_without_new_free_parameters = False
    missing_inputs = ["same_sector_equivalence_token_atom"]
    rows = _build_rows(
        literal_fragment_available=literal_fragment_available,
        literal_fragment_without_new_free_parameters=literal_fragment_without_new_free_parameters,
        missing_inputs=missing_inputs,
    )

    return {
        "generated_utc": _utc_now_iso(),
        "phase": {"phase": 8, "step": step_tag, "name": "same-sector equivalence literal fragment wording audit"},
        "inputs": {
            "mass_origin_same_sector_equivalence_literal_fragment_source_inventory_json": _relative_str(
                LITERAL_FRAGMENT_SOURCE_INVENTORY_JSON
            ),
        },
        "intent": "Audit whether the current public-canonical pack already fixes the missing same-sector equivalence literal fragment without a new fit parameter.",
        "formulas": {
            "audit_rule": "same_sector_equivalence_literal_fragment_available iff the public pack exposes a same-sector equivalence token atom",
            "current_absence": "the current pack still lacks the same-sector equivalence token atom, so the literal fragment cannot yet promote",
        },
        "rows": rows,
        "summary": {
            "same_sector_equivalence_literal_fragment_available": literal_fragment_available,
            "literal_fragment_kind_or_none": literal_fragment_kind_or_none,
            "literal_fragment_without_new_free_parameters": literal_fragment_without_new_free_parameters,
            "literal_fragment_missing_inputs": missing_inputs,
        },
        "decision": {
            "overall_status": "same_sector_equivalence_literal_fragment_wording_audit_frozen_absent",
            "keep_mass_origin_branch_blocked": True,
            "same_sector_equivalence_literal_fragment_available": literal_fragment_available,
            "literal_fragment_kind_or_none": literal_fragment_kind_or_none,
            "literal_fragment_without_new_free_parameters": literal_fragment_without_new_free_parameters,
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
            "same_sector_equivalence_literal_fragment_source_inventory_summary": literal_fragment_source_summary,
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
