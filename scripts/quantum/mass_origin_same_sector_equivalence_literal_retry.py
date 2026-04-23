#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_origin_same_sector_equivalence_literal_retry.py

Step 8.7.55.2.287:
Reinject the phrase-fragment result into the literal route and determine
whether the same-sector equivalence literal now closes.

Inputs:
  - output/public/quantum/mass_origin_same_sector_equivalence_literal_wording_audit_metrics.json
  - output/public/quantum/mass_origin_same_sector_equivalence_phrase_fragment_wording_audit_metrics.json

Outputs:
  - output/public/quantum/mass_origin_same_sector_equivalence_literal_retry_metrics.json
  - output/public/quantum/mass_origin_same_sector_equivalence_literal_retry_rows.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]

LITERAL_WORDING_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_same_sector_equivalence_literal_wording_audit_metrics.json"
PHRASE_FRAGMENT_WORDING_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_same_sector_equivalence_phrase_fragment_wording_audit_metrics.json"
OUT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_same_sector_equivalence_literal_retry_metrics.json"
OUT_CSV = ROOT / "output" / "public" / "quantum" / "mass_origin_same_sector_equivalence_literal_retry_rows.csv"
DEFAULT_STEP_TAG = "8.7.55.2.287"


# 関数: `_utc_now_iso` の入出力契約と処理意図を定義する。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_parse_args` の入出力契約と処理意図を定義する。

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retry same-sector equivalence literal closure.")
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

def _build_rows(*, literal_available: bool, nonclosure_reason: str | None) -> List[Dict[str, Any]]:
    return [
        {
            "row_id": "same_sector_equivalence_literal_retry_complete",
            "status": "pass",
            "metric": "same-sector equivalence literal retry complete",
            "value": 1.0,
            "note": "This step retries literal closure after the phrase-fragment audit.",
        },
        {
            "row_id": "same_sector_equivalence_literal_retry_available",
            "status": "pass" if literal_available else "reject",
            "metric": "same-sector equivalence literal available after retry",
            "value": 1.0 if literal_available else 0.0,
            "note": (
                "The same-sector equivalence literal is available after retry."
                if literal_available
                else f"The retry remains non-closing: {nonclosure_reason}."
            ),
        },
    ]


# 関数: `_build_payload` の入出力契約と処理意図を定義する。

def _build_payload(step_tag: str) -> Dict[str, Any]:
    for path in (LITERAL_WORDING_JSON, PHRASE_FRAGMENT_WORDING_JSON):
        _require_path(path)

    literal_wording = _read_json(LITERAL_WORDING_JSON)
    phrase_fragment_wording = _read_json(PHRASE_FRAGMENT_WORDING_JSON)

    literal_wording_summary = literal_wording.get("summary", {})
    phrase_fragment_wording_summary = phrase_fragment_wording.get("summary", {})

    phrase_fragment_available = bool(
        phrase_fragment_wording_summary.get("same_sector_equivalence_phrase_fragment_available", False)
    )
    phrase_fragment_missing_inputs = [
        str(item) for item in phrase_fragment_wording_summary.get("phrase_fragment_missing_inputs", [])
    ]
    literal_available = phrase_fragment_available
    literal_kind_or_none = "same_sector_equivalence_literal" if literal_available else None
    nonclosure_reason = None

    if "same_sector_equivalence_literal_fragment" in phrase_fragment_missing_inputs:
        nonclosure_reason = "same_sector_equivalence_literal_fragment_absent"
    elif not phrase_fragment_available:
        nonclosure_reason = "same_sector_equivalence_phrase_fragment_unavailable"

    rows = _build_rows(literal_available=literal_available, nonclosure_reason=nonclosure_reason)

    return {
        "generated_utc": _utc_now_iso(),
        "phase": {"phase": 8, "step": step_tag, "name": "same-sector equivalence literal retry"},
        "inputs": {
            "mass_origin_same_sector_equivalence_literal_wording_audit_json": _relative_str(LITERAL_WORDING_JSON),
            "mass_origin_same_sector_equivalence_phrase_fragment_wording_audit_json": _relative_str(
                PHRASE_FRAGMENT_WORDING_JSON
            ),
        },
        "intent": "Retry whether the same-sector equivalence literal can now close after the phrase-fragment audit.",
        "formulas": {
            "retry_rule": "same_sector_equivalence_literal_available iff the phrase-fragment wording audit promotes a same-sector equivalence phrase fragment",
        },
        "rows": rows,
        "summary": {
            "same_sector_equivalence_literal_available": literal_available,
            "literal_kind_or_none": literal_kind_or_none,
            "literal_retry_nonclosure_reason_or_none": nonclosure_reason,
        },
        "decision": {
            "overall_status": "same_sector_equivalence_literal_retry_frozen_absent",
            "keep_mass_origin_branch_blocked": True,
            "same_sector_equivalence_literal_available": literal_available,
            "literal_kind_or_none": literal_kind_or_none,
            "literal_retry_nonclosure_reason_or_none": nonclosure_reason,
            "hand_off_to_8_7_55_2_83": False,
            "next_required_artifacts": [
                "same_sector_equivalence_literal_fragment",
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
            "same_sector_equivalence_literal_wording_audit_summary": literal_wording_summary,
            "same_sector_equivalence_phrase_fragment_wording_audit_summary": phrase_fragment_wording_summary,
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
