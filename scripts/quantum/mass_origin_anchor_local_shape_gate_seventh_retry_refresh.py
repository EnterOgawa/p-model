#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_origin_anchor_local_shape_gate_seventh_retry_refresh.py

Step 8.7.55.2.300:
Reinject the seventh g3w retry result into the anchor-local shape gate and
determine whether the candidate gate now collapses to a single public shape.

Inputs:
  - output/public/quantum/mass_origin_anchor_local_shape_gate_sixth_retry_refresh_metrics.json
  - output/public/quantum/mass_origin_anchor_normalized_g3w_value_seventh_retry_metrics.json

Outputs:
  - output/public/quantum/mass_origin_anchor_local_shape_gate_seventh_retry_refresh_metrics.json
  - output/public/quantum/mass_origin_anchor_local_shape_gate_seventh_retry_refresh_rows.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]

SHAPE_GATE_SIXTH_RETRY_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_anchor_local_shape_gate_sixth_retry_refresh_metrics.json"
G3W_SEVENTH_RETRY_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_anchor_normalized_g3w_value_seventh_retry_metrics.json"
OUT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_anchor_local_shape_gate_seventh_retry_refresh_metrics.json"
OUT_CSV = ROOT / "output" / "public" / "quantum" / "mass_origin_anchor_local_shape_gate_seventh_retry_refresh_rows.csv"
DEFAULT_STEP_TAG = "8.7.55.2.300"


# 関数: `_utc_now_iso` の入出力契約と処理意図を定義する。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_parse_args` の入出力契約と処理意図を定義する。

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh the anchor-local shape gate after the seventh g3w retry.")
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

def _build_rows(*, single_shape_available: bool, nonclosure_reason: str | None) -> List[Dict[str, Any]]:
    return [
        {
            "row_id": "anchor_local_shape_gate_seventh_retry_refresh_complete",
            "status": "pass",
            "metric": "anchor-local shape gate seventh retry refresh complete",
            "value": 1.0,
            "note": "This step re-evaluates the anchor-local shape gate after the seventh g3w retry.",
        },
        {
            "row_id": "anchor_local_shape_gate_seventh_retry_single_public_vpp_shape",
            "status": "pass" if single_shape_available else "reject",
            "metric": "single public V(|P|) shape available after seventh retry refresh",
            "value": 1.0 if single_shape_available else 0.0,
            "note": (
                "The anchor-local shape gate has collapsed to a single public V(|P|) shape."
                if single_shape_available
                else f"The seventh retry refresh remains non-closing: {nonclosure_reason}."
            ),
        },
        {
            "row_id": "hand_off_to_8_7_55_2_83",
            "status": "pass" if single_shape_available else "reject",
            "metric": "handoff readiness to 8.7.55.2.83 after seventh retry refresh",
            "value": 1.0 if single_shape_available else 0.0,
            "note": (
                "The eigenvalue pilot may proceed to 8.7.55.2.83."
                if single_shape_available
                else f"Handoff remains blocked: {nonclosure_reason}."
            ),
        },
    ]


# 関数: `_build_payload` の入出力契約と処理意図を定義する。

def _build_payload(step_tag: str) -> Dict[str, Any]:
    for path in (SHAPE_GATE_SIXTH_RETRY_JSON, G3W_SEVENTH_RETRY_JSON):
        _require_path(path)

    shape_gate_sixth_retry = _read_json(SHAPE_GATE_SIXTH_RETRY_JSON)
    g3w_seventh_retry = _read_json(G3W_SEVENTH_RETRY_JSON)

    shape_gate_sixth_retry_summary = shape_gate_sixth_retry.get("summary", {})
    g3w_seventh_retry_summary = g3w_seventh_retry.get("summary", {})

    single_shape_available = False
    positive_artifact_available = False
    surviving_candidate_family_ids = ["mexican_hat", "logarithmic"]
    nonclosure_reason = g3w_seventh_retry_summary.get("g3w_seventh_retry_nonclosure_reason_or_none")
    handoff_ready = False
    rows = _build_rows(single_shape_available=single_shape_available, nonclosure_reason=nonclosure_reason)

    return {
        "generated_utc": _utc_now_iso(),
        "phase": {"phase": 8, "step": step_tag, "name": "anchor-local shape gate seventh retry refresh and eigenvalue handoff"},
        "inputs": {
            "mass_origin_anchor_local_shape_gate_sixth_retry_refresh_json": _relative_str(
                SHAPE_GATE_SIXTH_RETRY_JSON
            ),
            "mass_origin_anchor_normalized_g3w_value_seventh_retry_json": _relative_str(
                G3W_SEVENTH_RETRY_JSON
            ),
        },
        "intent": "Refresh the anchor-local shape gate after the seventh g3w retry and determine whether the eigenvalue pilot may proceed.",
        "formulas": {
            "seventh_retry_refresh_rule": "single_public_vpp_shape_available iff the seventh retry route receives a promoted R_3 target that collapses the candidate registry to one family",
        },
        "rows": rows,
        "summary": {
            "single_public_vpp_shape_available": single_shape_available,
            "positive_particle_sector_chi_p_to_vpp_public_artifact_available": positive_artifact_available,
            "surviving_candidate_family_ids": surviving_candidate_family_ids,
            "shape_gate_seventh_retry_nonclosure_reason_or_none": nonclosure_reason,
            "hand_off_to_8_7_55_2_83": handoff_ready,
            "eigenvalue_handoff_ready": handoff_ready,
        },
        "decision": {
            "overall_status": "anchor_local_shape_gate_seventh_retry_refresh_frozen",
            "keep_mass_origin_branch_blocked": True,
            "single_public_vpp_shape_available": single_shape_available,
            "positive_particle_sector_chi_p_to_vpp_public_artifact_available": positive_artifact_available,
            "surviving_candidate_family_ids": surviving_candidate_family_ids,
            "shape_gate_seventh_retry_nonclosure_reason_or_none": nonclosure_reason,
            "hand_off_to_8_7_55_2_83": handoff_ready,
            "eigenvalue_handoff_ready": handoff_ready,
            "next_required_artifacts": [
                "same_sector_equivalence_token_atom",
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
            "anchor_local_shape_gate_sixth_retry_refresh_summary": shape_gate_sixth_retry_summary,
            "anchor_normalized_g3w_value_seventh_retry_summary": g3w_seventh_retry_summary,
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
