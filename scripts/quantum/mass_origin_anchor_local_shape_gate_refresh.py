#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_origin_anchor_local_shape_gate_refresh.py

Step 8.7.55.2.252:
Reinject the g_3w route closure result into the anchor-local same-sector shape
gate and decide whether the mass-origin branch can now hand off into the
eigenvalue pilot.

Inputs:
  - output/public/quantum/mass_origin_anchor_local_shape_gate_metrics.json
  - output/public/quantum/mass_origin_anchor_normalized_g3w_value_closure_metrics.json

Outputs:
  - output/public/quantum/mass_origin_anchor_local_shape_gate_refresh_metrics.json
  - output/public/quantum/mass_origin_anchor_local_shape_gate_refresh_rows.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]

SHAPE_GATE_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_anchor_local_shape_gate_metrics.json"
G3W_CLOSURE_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_anchor_normalized_g3w_value_closure_metrics.json"
OUT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_anchor_local_shape_gate_refresh_metrics.json"
OUT_CSV = ROOT / "output" / "public" / "quantum" / "mass_origin_anchor_local_shape_gate_refresh_rows.csv"
DEFAULT_STEP_TAG = "8.7.55.2.252"


# 関数: `_utc_now_iso` の入出力契約と処理意図を定義する。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_parse_args` の入出力契約と処理意図を定義する。

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refresh the anchor-local shape gate after the g_3w route closure audit.",
    )
    parser.add_argument(
        "--step-tag",
        default=DEFAULT_STEP_TAG,
        help="Roadmap step tag to stamp into the output payload.",
    )
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
            "row_id": "anchor_local_shape_gate_refresh_complete",
            "status": "pass",
            "metric": "anchor-local shape gate refresh complete",
            "value": 1.0,
            "note": "This step re-evaluates the anchor-local shape gate after the preferred g_3w route closure attempt.",
        },
    ]

    for family in surviving_candidate_family_ids:
        rows.append(
            {
                "row_id": f"anchor_local_shape_gate_refresh_candidate_{family}",
                "status": "watch",
                "metric": f"{family} remains in the refreshed candidate set",
                "value": 1.0,
                "note": f"{family} remains live because the refreshed gate still lacks a promoted R_3 target.",
            }
        )

    rows.extend(
        [
            {
                "row_id": "anchor_local_shape_gate_refresh_single_public_vpp_shape",
                "status": "pass" if single_public_vpp_shape_available else "reject",
                "metric": "single public V(|P|) shape available after refreshed gate",
                "value": 1.0 if single_public_vpp_shape_available else 0.0,
                "note": (
                    "The refreshed gate now selects a unique same-sector shape."
                    if single_public_vpp_shape_available
                    else f"The refreshed gate remains non-closing: {nonclosure_reason}."
                ),
            },
            {
                "row_id": "anchor_local_shape_gate_refresh_positive_artifact",
                "status": "pass" if positive_artifact_available else "reject",
                "metric": "positive same-sector public artifact available after refreshed gate",
                "value": 1.0 if positive_artifact_available else 0.0,
                "note": (
                    "The positive same-sector public artifact is now promotable."
                    if positive_artifact_available
                    else "The positive same-sector public artifact cannot promote before the refreshed single-shape gate closes."
                ),
            },
            {
                "row_id": "anchor_local_shape_gate_refresh_eigenvalue_handoff_ready",
                "status": "pass" if eigenvalue_handoff_ready else "reject",
                "metric": "eigenvalue handoff ready after refreshed gate",
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
                "metric": "handoff to 8.7.55.2.83-.84 allowed after refreshed gate",
                "value": 1.0 if handoff else 0.0,
                "note": (
                    "Handoff to the eigenvalue pilot is now allowed."
                    if handoff
                    else "Handoff remains blocked because the g_3w route did not yet promote a public R_3 target."
                ),
            },
        ]
    )
    return rows


# 関数: `_build_payload` の入出力契約と処理意図を定義する。

def _build_payload(step_tag: str) -> Dict[str, Any]:
    for path in (SHAPE_GATE_JSON, G3W_CLOSURE_JSON):
        _require_path(path)

    shape_gate = _read_json(SHAPE_GATE_JSON)
    g3w_closure = _read_json(G3W_CLOSURE_JSON)

    shape_gate_summary = shape_gate.get("summary", {})
    g3w_summary = g3w_closure.get("summary", {})

    surviving_candidate_family_ids = [str(item) for item in shape_gate_summary.get("surviving_candidate_family_ids", [])]
    r3_target_available = bool(g3w_summary.get("r3_target_available", False))
    single_public_vpp_shape_available = bool(r3_target_available and len(surviving_candidate_family_ids) == 1)
    positive_artifact_available = False
    eigenvalue_handoff_ready = False
    handoff = False
    nonclosure_reason = "g3w_route_rho_star_elimination_pending"

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
            "name": "anchor-local shape-gate refresh and eigenvalue handoff",
        },
        "inputs": {
            "mass_origin_anchor_local_shape_gate_json": _relative_str(SHAPE_GATE_JSON),
            "mass_origin_anchor_normalized_g3w_value_closure_json": _relative_str(G3W_CLOSURE_JSON),
        },
        "intent": "Refresh the anchor-local candidate gate after the preferred g_3w route and determine whether the eigenvalue handoff can reopen.",
        "formulas": {
            "refresh_rule": "single_public_vpp_shape_available iff the refreshed gate receives a promoted R_3 target that collapses the candidate registry to one family",
            "current_blocker": "the g_3w route still lacks rho_* elimination, so R_3_target remains unavailable and the refreshed gate stays non-closing",
        },
        "rows": rows,
        "summary": {
            "single_public_vpp_shape_available": single_public_vpp_shape_available,
            "positive_particle_sector_chi_p_to_vpp_public_artifact_available": positive_artifact_available,
            "surviving_candidate_family_ids": surviving_candidate_family_ids,
            "shape_gate_nonclosure_reason_or_none": nonclosure_reason,
            "hand_off_to_8_7_55_2_83": handoff,
            "eigenvalue_handoff_ready": eigenvalue_handoff_ready,
            "r3_target_available": r3_target_available,
        },
        "decision": {
            "overall_status": "anchor_local_shape_gate_refresh_frozen",
            "keep_mass_origin_branch_blocked": True,
            "single_public_vpp_shape_available": single_public_vpp_shape_available,
            "positive_particle_sector_chi_p_to_vpp_public_artifact_available": positive_artifact_available,
            "surviving_candidate_family_ids": surviving_candidate_family_ids,
            "shape_gate_nonclosure_reason_or_none": nonclosure_reason,
            "hand_off_to_8_7_55_2_83": handoff,
            "eigenvalue_handoff_ready": eigenvalue_handoff_ready,
            "remaining_g3w_route_blockers": [
                "rho_star_elimination_rule",
                "anchor_normalized_g3w_public_value",
            ],
            "next_required_artifacts": [
                "rho_star_elimination_rule",
                "anchor_normalized_g3w_public_value",
                "r3_target",
                "single_public_vpp_shape",
                "positive_particle_sector_chi_p_to_vpp_public_artifact",
                "solver_ready_row_promoted_to_pass",
            ],
        },
        "evidence": {
            "shape_gate_summary": shape_gate_summary,
            "g3w_closure_summary": g3w_summary,
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
