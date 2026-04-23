#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_origin_single_public_vpp_shape_closure_retry.py

Step 8.7.55.2.87:
Retry same-sector single-shape closure using the derivative-ratio tie-break
route and the target-bridge audit.

Inputs:
  - output/public/quantum/mass_origin_single_public_vpp_shape_closure_metrics.json
  - output/public/quantum/mass_origin_same_sector_vpp_tiebreak_invariant_metrics.json
  - output/public/quantum/mass_origin_same_sector_tiebreak_target_bridge_metrics.json

Outputs:
  - output/public/quantum/mass_origin_single_public_vpp_shape_closure_retry_metrics.json
  - output/public/quantum/mass_origin_single_public_vpp_shape_closure_retry_rows.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]

CLOSURE_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_single_public_vpp_shape_closure_metrics.json"
TIEBREAK_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_same_sector_vpp_tiebreak_invariant_metrics.json"
BRIDGE_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_same_sector_tiebreak_target_bridge_metrics.json"
OUT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_single_public_vpp_shape_closure_retry_metrics.json"
OUT_CSV = ROOT / "output" / "public" / "quantum" / "mass_origin_single_public_vpp_shape_closure_retry_rows.csv"
DEFAULT_STEP_TAG = "8.7.55.2.87"


# 関数: `_utc_now_iso` の入出力契約と処理意図を定義する。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_parse_args` の入出力契約と処理意図を定義する。

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retry single_public_vpp_shape closure using the same-sector tie-break route.",
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


# 関数: `_candidate_retry_status` の入出力契約と処理意図を定義する。

def _candidate_retry_status(
    candidate_id: str,
    initial_survivors: List[str],
    selected_candidate_id_or_none: str | None,
) -> tuple[str, float, str]:
    # 条件分岐: `candidate_id == selected_candidate_id_or_none` を満たす経路を評価する。
    if candidate_id == selected_candidate_id_or_none:
        return (
            "pass",
            1.0,
            f"{candidate_id} is uniquely selected by the tie-break retry and becomes the single public V(|P|) shape.",
        )

    # 条件分岐: `candidate_id in initial_survivors` を満たす経路を評価する。

    if candidate_id in initial_survivors:
        return (
            "watch",
            1.0,
            f"{candidate_id} remains live after the retry because the tie-break route exists but still lacks a public canonical target value.",
        )

    return (
        "reject",
        0.0,
        f"{candidate_id} remains excluded from the retry closure candidate set.",
    )


# 関数: `_build_payload` の入出力契約と処理意図を定義する。

def _build_payload(step_tag: str) -> Dict[str, Any]:
    for path in (CLOSURE_JSON, TIEBREAK_JSON, BRIDGE_JSON):
        _require_path(path)

    closure = _read_json(CLOSURE_JSON)
    tiebreak = _read_json(TIEBREAK_JSON)
    bridge = _read_json(BRIDGE_JSON)

    closure_summary = closure.get("summary", {})
    closure_decision = closure.get("decision", {})
    tiebreak_summary = tiebreak.get("summary", {})
    bridge_summary = bridge.get("summary", {})
    bridge_decision = bridge.get("decision", {})

    initial_survivors = [str(item) for item in closure_summary.get("surviving_candidate_ids", [])]
    rejected_candidate_ids = [str(item) for item in closure_summary.get("rejected_candidate_ids", [])]
    tie_break_route_available = bool(tiebreak_summary.get("tie_break_route_available", False))
    tiebreak_invariant_name = str(tiebreak_summary.get("tiebreak_invariant_name", ""))
    target_value_available = bool(bridge_summary.get("target_value_available", False))
    target_source_kind_or_none = bridge_summary.get("target_source_kind_or_none")
    bridge_without_new_free_parameters = bool(bridge_summary.get("bridge_without_new_free_parameters", False))
    candidate_match_count = int(bridge_summary.get("candidate_match_count", 0))
    matching_candidate_ids = [str(item) for item in bridge_summary.get("matching_candidate_ids", [])]

    selected_candidate_id_or_none: str | None = None
    single_public_vpp_shape_available = False
    curvature_coefficients_fixed = False
    three_wave_coefficients_fixed = False

    # 条件分岐: `not tie_break_route_available` を満たす経路を評価する。
    if not tie_break_route_available:
        closure_retry_status = "watch_retry_route_unavailable"
        nonclosure_reason_or_none = "same_sector_tiebreak_route_unavailable"

    # 条件分岐: `target_value_available and candidate_match_count == 1` を満たす経路を評価する。
    elif target_value_available and candidate_match_count == 1:
        selected_candidate_id_or_none = matching_candidate_ids[0]
        single_public_vpp_shape_available = True
        curvature_coefficients_fixed = True
        three_wave_coefficients_fixed = True
        closure_retry_status = "closed_single_same_sector_candidate_via_tiebreak_target"
        nonclosure_reason_or_none = None

    # 条件分岐: `target_value_available and candidate_match_count > 1` を満たす経路を評価する。
    elif target_value_available and candidate_match_count > 1:
        closure_retry_status = "watch_nonclosure_multiple_candidates_match_tiebreak_target"
        nonclosure_reason_or_none = "multiple_candidates_match_tiebreak_target"

    # 条件分岐: `target_value_available and candidate_match_count == 0` を満たす経路を評価する。
    elif target_value_available and candidate_match_count == 0:
        closure_retry_status = "watch_nonclosure_no_candidate_matches_tiebreak_target"
        nonclosure_reason_or_none = "no_candidate_matches_tiebreak_target"

    else:
        closure_retry_status = "watch_nonclosure_tiebreak_target_value_missing"
        nonclosure_reason_or_none = "same_sector_tiebreak_target_value_missing"

    retried_surviving_candidate_ids = (
        [selected_candidate_id_or_none]
        if selected_candidate_id_or_none
        else initial_survivors
    )
    retried_surviving_candidate_count = len(retried_surviving_candidate_ids)

    mexican_hat_status, mexican_hat_value, mexican_hat_note = _candidate_retry_status(
        "mexican_hat",
        initial_survivors,
        selected_candidate_id_or_none,
    )
    logarithmic_status, logarithmic_value, logarithmic_note = _candidate_retry_status(
        "logarithmic",
        initial_survivors,
        selected_candidate_id_or_none,
    )
    even_polynomial_status, even_polynomial_value, even_polynomial_note = _candidate_retry_status(
        "even_polynomial",
        initial_survivors,
        selected_candidate_id_or_none,
    )

    rows = [
        {
            "row_id": "single_vpp_shape_retry_complete",
            "status": "pass",
            "metric": "single_public_vpp_shape retry complete",
            "value": 1.0,
            "family": "aggregate",
            "note": "The retry artifact reevaluates single-shape closure after introducing the derivative-ratio tie-break route.",
        },
        {
            "row_id": "single_vpp_shape_retry_tiebreak_route_available",
            "status": "pass" if tie_break_route_available else "reject",
            "metric": "same-sector derivative-ratio tie-break route available for retry",
            "value": 1.0 if tie_break_route_available else 0.0,
            "family": "aggregate",
            "note": f"The retry uses invariant {tiebreak_invariant_name} when a public target value exists.",
        },
        {
            "row_id": "single_vpp_shape_retry_target_value_available",
            "status": "pass" if target_value_available else "watch",
            "metric": "public target value for tie-break retry available",
            "value": 1.0 if target_value_available else 0.0,
            "family": "aggregate",
            "note": (
                f"Target value source is {target_source_kind_or_none}."
                if target_value_available
                else "The retry cannot choose a unique family because the tie-break target value is still missing."
            ),
        },
        {
            "row_id": "single_vpp_shape_retry_candidate_match_count",
            "status": "inventory",
            "metric": "candidate count matching the retry tie-break target",
            "value": float(candidate_match_count),
            "family": "aggregate",
            "note": (
                f"Matching candidate ids are {matching_candidate_ids}."
                if matching_candidate_ids
                else "No matching candidate set is available because the public tie-break target is missing."
            ),
        },
        {
            "row_id": "single_vpp_shape_retry_bridge_without_new_free_parameters",
            "status": "pass" if bridge_without_new_free_parameters else "reject",
            "metric": "retry bridge closes without new free parameters",
            "value": 1.0 if bridge_without_new_free_parameters else 0.0,
            "family": "aggregate",
            "note": (
                "The retry uses only already-frozen same-sector ingredients."
                if bridge_without_new_free_parameters
                else "The retry still lacks a no-new-free-parameter bridge because no public target value is available."
            ),
        },
        {
            "row_id": "single_vpp_shape_retry_selected_candidate",
            "status": "pass" if selected_candidate_id_or_none else "watch",
            "metric": "selected same-sector public V(|P|) candidate after retry",
            "value": 1.0 if selected_candidate_id_or_none else 0.0,
            "family": selected_candidate_id_or_none or "none",
            "note": (
                f"Single public V(|P|) shape is fixed to {selected_candidate_id_or_none} after the tie-break retry."
                if selected_candidate_id_or_none
                else "No unique same-sector candidate is selected in the retry artifact."
            ),
        },
        {
            "row_id": "single_vpp_shape_retry_mexican_hat_state",
            "status": mexican_hat_status,
            "metric": "mexican_hat state after single-shape retry",
            "value": mexican_hat_value,
            "family": "mexican_hat",
            "note": mexican_hat_note,
        },
        {
            "row_id": "single_vpp_shape_retry_logarithmic_state",
            "status": logarithmic_status,
            "metric": "logarithmic state after single-shape retry",
            "value": logarithmic_value,
            "family": "logarithmic",
            "note": logarithmic_note,
        },
        {
            "row_id": "single_vpp_shape_retry_even_polynomial_state",
            "status": even_polynomial_status,
            "metric": "even_polynomial state after single-shape retry",
            "value": even_polynomial_value,
            "family": "even_polynomial",
            "note": even_polynomial_note if "even_polynomial" in rejected_candidate_ids else "even_polynomial is not part of the surviving candidate set.",
        },
        {
            "row_id": "single_vpp_shape_retry_nonclosure_reason",
            "status": "watch" if nonclosure_reason_or_none else "pass",
            "metric": "single_public_vpp_shape retry nonclosure reason fixed",
            "value": 0.0 if nonclosure_reason_or_none else 1.0,
            "family": "aggregate",
            "note": (
                f"Retry nonclosure reason is {nonclosure_reason_or_none}."
                if nonclosure_reason_or_none
                else "No retry nonclosure reason remains because the shape closes."
            ),
        },
    ]

    return {
        "generated_utc": _utc_now_iso(),
        "phase": {
            "phase": 8,
            "step": step_tag,
            "name": "single_public_vpp_shape closure retry from tie-break target",
        },
        "inputs": {
            "mass_origin_single_public_vpp_shape_closure_json": _relative_str(CLOSURE_JSON),
            "mass_origin_same_sector_vpp_tiebreak_invariant_json": _relative_str(TIEBREAK_JSON),
            "mass_origin_same_sector_tiebreak_target_bridge_json": _relative_str(BRIDGE_JSON),
        },
        "intent": "Retry single-shape closure after introducing a same-sector derivative-ratio tie-break route.",
        "formulas": {
            "retry_closure_rule": "single_public_vpp_shape_available iff the tie-break route is available, the public target value exists, and exactly one candidate matches it",
            "current_retry_nonclosure_rule": "retry remains non-closing when the tie-break target value is still missing or does not isolate a unique candidate",
        },
        "rows": rows,
        "summary": {
            "selected_candidate_id_or_none": selected_candidate_id_or_none,
            "single_public_vpp_shape_available": single_public_vpp_shape_available,
            "curvature_coefficients_fixed": curvature_coefficients_fixed,
            "three_wave_coefficients_fixed": three_wave_coefficients_fixed,
            "closure_retry_status": closure_retry_status,
            "nonclosure_reason_or_none": nonclosure_reason_or_none,
            "retried_surviving_candidate_count": retried_surviving_candidate_count,
            "retried_surviving_candidate_ids": retried_surviving_candidate_ids,
            "tie_break_route_available": tie_break_route_available,
            "target_value_available": target_value_available,
            "target_source_kind_or_none": target_source_kind_or_none,
            "candidate_match_count": candidate_match_count,
            "matching_candidate_ids": matching_candidate_ids,
        },
        "decision": {
            "overall_status": "single_public_vpp_shape_closure_retry_frozen",
            "keep_mass_origin_branch_blocked": True,
            "selected_candidate_id_or_none": selected_candidate_id_or_none,
            "single_public_vpp_shape_available": single_public_vpp_shape_available,
            "curvature_coefficients_fixed": curvature_coefficients_fixed,
            "three_wave_coefficients_fixed": three_wave_coefficients_fixed,
            "closure_retry_status": closure_retry_status,
            "nonclosure_reason_or_none": nonclosure_reason_or_none,
            "blocked_state_detail": str(bridge_decision.get("blocked_state_detail", closure_decision.get("blocked_state_detail", ""))),
            "next_required_artifacts": bridge_decision.get(
                "next_required_artifacts",
                [
                    "same_sector_tiebreak_target_value",
                    "single_public_vpp_shape",
                    "positive_particle_sector_chi_p_to_vpp_public_artifact",
                    "solver_ready_row_promoted_to_pass",
                ],
            ),
        },
        "evidence": {
            "closure_summary": closure_summary,
            "tiebreak_summary": tiebreak_summary,
            "bridge_summary": bridge_summary,
        },
    }


# 関数: `_write_csv` の入出力契約と処理意図を定義する。

def _write_csv(rows: List[Dict[str, Any]]) -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["row_id", "status", "metric", "value", "family", "note"],
        )
        writer.writeheader()
        writer.writerows(rows)


# 関数: `main` の入出力契約と処理意図を定義する。

def main() -> None:
    args = _parse_args()
    payload = _build_payload(str(args.step_tag))
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_csv(payload["rows"])
    print(json.dumps(payload["decision"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
