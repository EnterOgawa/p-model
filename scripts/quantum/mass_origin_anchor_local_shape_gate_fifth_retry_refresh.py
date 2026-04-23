#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_origin_anchor_local_shape_gate_fifth_retry_refresh.py

Step 8.7.55.2.283:
Refresh the anchor-local same-sector shape gate after the fifth retry of the
preferred anchor-normalized g_3w route.

Inputs:
  - output/public/quantum/mass_origin_anchor_local_shape_gate_fourth_retry_refresh_metrics.json
  - output/public/quantum/mass_origin_anchor_normalized_g3w_value_fifth_retry_metrics.json

Outputs:
  - output/public/quantum/mass_origin_anchor_local_shape_gate_fifth_retry_refresh_metrics.json
  - output/public/quantum/mass_origin_anchor_local_shape_gate_fifth_retry_refresh_rows.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]

SHAPE_GATE_FOURTH_RETRY_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_anchor_local_shape_gate_fourth_retry_refresh_metrics.json"
G3W_FIFTH_RETRY_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_anchor_normalized_g3w_value_fifth_retry_metrics.json"
OUT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_anchor_local_shape_gate_fifth_retry_refresh_metrics.json"
OUT_CSV = ROOT / "output" / "public" / "quantum" / "mass_origin_anchor_local_shape_gate_fifth_retry_refresh_rows.csv"
DEFAULT_STEP_TAG = "8.7.55.2.283"


# 関数: `_utc_now_iso` の入出力契約と処理意図を定義する。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_parse_args` の入出力契約と処理意図を定義する。

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refresh the anchor-local shape gate after the fifth g_3w retry.",
    )
    parser.add_argument("--step-tag", default=DEFAULT_STEP_TAG, help="Roadmap step tag to stamp into the output payload.")
    return parser.parse_args()


# 関数: `_require_path` の入出力契約と処理意図を定義する。

def _require_path(path: Path) -> None:
    # 条件分岐: `not path.exists()` を満たす経路を評価する。
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
    surviving_candidate_family_ids: List[str],
    single_public_vpp_shape_available: bool,
    positive_artifact_available: bool,
    eigenvalue_handoff_ready: bool,
    handoff: bool,
    nonclosure_reason: str | None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = [
        {
            "row_id": "anchor_local_shape_gate_fifth_retry_refresh_complete",
            "status": "pass",
            "metric": "anchor-local shape gate fifth retry refresh complete",
            "value": 1.0,
            "note": "This step re-evaluates the anchor-local shape gate after the fifth g_3w retry.",
        }
    ]

    for family in surviving_candidate_family_ids:
        rows.append(
            {
                "row_id": f"anchor_local_shape_gate_fifth_retry_candidate_{family}",
                "status": "watch",
                "metric": f"{family} remains in the fifth retry candidate set",
                "value": 1.0,
                "note": f"{family} remains live because the fifth retry gate still lacks a promoted R_3 target.",
            }
        )

    rows.extend(
        [
            {
                "row_id": "anchor_local_shape_gate_fifth_retry_single_public_vpp_shape",
                "status": "pass" if single_public_vpp_shape_available else "reject",
                "metric": "single public V(|P|) shape available after fifth retry refresh",
                "value": 1.0 if single_public_vpp_shape_available else 0.0,
                "note": (
                    "The fifth retry refresh now selects a unique same-sector shape."
                    if single_public_vpp_shape_available
                    else f"The fifth retry refresh remains non-closing: {nonclosure_reason}."
                ),
            },
            {
                "row_id": "anchor_local_shape_gate_fifth_retry_positive_artifact",
                "status": "pass" if positive_artifact_available else "reject",
                "metric": "positive same-sector public artifact available after fifth retry refresh",
                "value": 1.0 if positive_artifact_available else 0.0,
                "note": (
                    "The positive same-sector public artifact is now promotable."
                    if positive_artifact_available
                    else "The positive same-sector public artifact cannot promote before the fifth retry single-shape gate closes."
                ),
            },
            {
                "row_id": "anchor_local_shape_gate_fifth_retry_eigenvalue_handoff_ready",
                "status": "pass" if eigenvalue_handoff_ready else "reject",
                "metric": "eigenvalue handoff ready after fifth retry refresh",
                "value": 1.0 if eigenvalue_handoff_ready else 0.0,
                "note": (
                    "The branch is now ready to hand off into the eigenvalue pilot."
                    if eigenvalue_handoff_ready
                    else "The branch is still not ready for the eigenvalue pilot."
                ),
            },
            {
                "row_id": "hand_off_to_8_7_55_2_83",
                "status": "pass" if handoff else "reject",
                "metric": "handoff to 8.7.55.2.83-.84 allowed after fifth retry refresh",
                "value": 1.0 if handoff else 0.0,
                "note": (
                    "Handoff to the eigenvalue pilot is now allowed."
                    if handoff
                    else "Handoff remains blocked because the fifth retry route did not yet promote a public R_3 target."
                ),
            },
        ]
    )
    return rows


# 関数: `_build_payload` の入出力契約と処理意図を定義する。

def _build_payload(step_tag: str) -> Dict[str, Any]:
    for path in (SHAPE_GATE_FOURTH_RETRY_JSON, G3W_FIFTH_RETRY_JSON):
        _require_path(path)

    shape_gate_fourth_retry = _read_json(SHAPE_GATE_FOURTH_RETRY_JSON)
    g3w_fifth_retry = _read_json(G3W_FIFTH_RETRY_JSON)

    shape_gate_fourth_retry_summary = shape_gate_fourth_retry.get("summary", {})
    g3w_fifth_retry_summary = g3w_fifth_retry.get("summary", {})

    surviving_candidate_family_ids = [
        str(item) for item in shape_gate_fourth_retry_summary.get("surviving_candidate_family_ids", [])
    ]
    r3_target_available = bool(g3w_fifth_retry_summary.get("r3_target_available", False))
    single_public_vpp_shape_available = bool(r3_target_available and len(surviving_candidate_family_ids) == 1)
    positive_artifact_available = False
    eigenvalue_handoff_ready = False
    handoff = False
    nonclosure_reason = g3w_fifth_retry_summary.get("g3w_fifth_retry_nonclosure_reason_or_none")
    rows = _build_rows(
        surviving_candidate_family_ids=surviving_candidate_family_ids,
        single_public_vpp_shape_available=single_public_vpp_shape_available,
        positive_artifact_available=positive_artifact_available,
        eigenvalue_handoff_ready=eigenvalue_handoff_ready,
        handoff=handoff,
        nonclosure_reason=nonclosure_reason,
    )

    return {
        "generated_utc": _utc_now_iso(),
        "phase": {
            "phase": 8,
            "step": step_tag,
            "name": "anchor-local shape-gate fifth retry refresh and eigenvalue handoff",
        },
        "inputs": {
            "mass_origin_anchor_local_shape_gate_fourth_retry_refresh_json": _relative_str(
                SHAPE_GATE_FOURTH_RETRY_JSON
            ),
            "mass_origin_anchor_normalized_g3w_value_fifth_retry_json": _relative_str(G3W_FIFTH_RETRY_JSON),
        },
        "intent": "Refresh the anchor-local candidate gate after the fifth retry of the preferred g_3w route and determine whether the eigenvalue handoff can reopen.",
        "formulas": {
            "fifth_retry_refresh_rule": "single_public_vpp_shape_available iff the fifth retry route receives a promoted R_3 target that collapses the candidate registry to one family",
            "current_blocker": "the fifth retry route still lacks a public same-sector equivalence phrase fragment, so R_3_target remains unavailable and the refreshed gate stays non-closing",
        },
        "rows": rows,
        "summary": {
            "single_public_vpp_shape_available": single_public_vpp_shape_available,
            "positive_particle_sector_chi_p_to_vpp_public_artifact_available": positive_artifact_available,
            "surviving_candidate_family_ids": surviving_candidate_family_ids,
            "shape_gate_fifth_retry_nonclosure_reason_or_none": nonclosure_reason,
            "hand_off_to_8_7_55_2_83": handoff,
            "eigenvalue_handoff_ready": eigenvalue_handoff_ready,
        },
        "decision": {
            "overall_status": "anchor_local_shape_gate_fifth_retry_refresh_frozen",
            "keep_mass_origin_branch_blocked": True,
            "single_public_vpp_shape_available": single_public_vpp_shape_available,
            "positive_particle_sector_chi_p_to_vpp_public_artifact_available": positive_artifact_available,
            "surviving_candidate_family_ids": surviving_candidate_family_ids,
            "shape_gate_fifth_retry_nonclosure_reason_or_none": nonclosure_reason,
            "hand_off_to_8_7_55_2_83": handoff,
            "eigenvalue_handoff_ready": eigenvalue_handoff_ready,
            "remaining_fifth_retry_route_blockers": [
                "same_sector_equivalence_phrase_fragment",
                "same_sector_equivalence_literal",
                "same_sector_equivalence_statement",
                "same_sector_equivalence_rule",
                "chi_star_or_same_sector_proxy",
                "anchor_normalized_g3w_public_value",
            ],
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
            "shape_gate_fourth_retry_summary": shape_gate_fourth_retry_summary,
            "g3w_fifth_retry_summary": g3w_fifth_retry_summary,
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
