#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_origin_same_sector_vpp_tiebreak_invariant.py

Step 8.7.55.2.85:
Freeze a new same-sector tie-break route for the surviving V(|P|) families by
using the dimensionless derivative-ratio invariant

  R3 = |P|_* V'''(|P|_*) / V''(|P|_*)

which is already fixed by the candidate derivative formulas:

  - mexican_hat -> R3 = 3
  - logarithmic -> R3 = 1

Inputs:
  - output/public/quantum/mass_origin_single_vpp_candidate_derivative_metrics.json
  - output/public/quantum/mass_origin_single_public_vpp_shape_closure_metrics.json

Outputs:
  - output/public/quantum/mass_origin_same_sector_vpp_tiebreak_invariant_metrics.json
  - output/public/quantum/mass_origin_same_sector_vpp_tiebreak_invariant_rows.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]

DERIVATIVE_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_single_vpp_candidate_derivative_metrics.json"
CLOSURE_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_single_public_vpp_shape_closure_metrics.json"
OUT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_same_sector_vpp_tiebreak_invariant_metrics.json"
OUT_CSV = ROOT / "output" / "public" / "quantum" / "mass_origin_same_sector_vpp_tiebreak_invariant_rows.csv"
DEFAULT_STEP_TAG = "8.7.55.2.85"


# 関数: `_utc_now_iso` の入出力契約と処理意図を定義する。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_parse_args` の入出力契約と処理意図を定義する。

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze a same-sector derivative-ratio tie-break route for the surviving V(|P|) families.",
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


# 関数: `_ratio_for_family` の入出力契約と処理意図を定義する。

def _ratio_for_family(family: str) -> float | None:
    # 条件分岐: `family == "mexican_hat"` を満たす経路を評価する。
    if family == "mexican_hat":
        return 3.0

    # 条件分岐: `family == "logarithmic"` を満たす経路を評価する。

    if family == "logarithmic":
        return 1.0

    return None


# 関数: `_build_payload` の入出力契約と処理意図を定義する。

def _build_payload(step_tag: str) -> Dict[str, Any]:
    for path in (DERIVATIVE_JSON, CLOSURE_JSON):
        _require_path(path)

    derivative = _read_json(DERIVATIVE_JSON)
    closure = _read_json(CLOSURE_JSON)

    derivative_summary = derivative.get("summary", {})
    closure_summary = closure.get("summary", {})
    closure_decision = closure.get("decision", {})

    surviving_candidate_ids = [str(item) for item in closure_summary.get("surviving_candidate_ids", [])]
    ratio_rows: List[Dict[str, Any]] = []
    fixed_value_rows = 0
    ratio_values: Dict[str, float] = {}

    for family in surviving_candidate_ids:
        ratio_value = _ratio_for_family(family)
        ratio_values[family] = ratio_value if ratio_value is not None else float("nan")

        # 条件分岐: `ratio_value is not None` を満たす経路を評価する。
        if ratio_value is not None:
            fixed_value_rows += 1

        ratio_rows.append(
            {
                "row_id": f"same_sector_tiebreak_ratio_{family}",
                "status": "pass" if ratio_value is not None else "watch",
                "metric": f"dimensionless same-sector tie-break ratio for {family}",
                "value": ratio_value if ratio_value is not None else 0.0,
                "note": (
                    f"R3 = |P|_* V'''(|P|_*) / V''(|P|_*) is fixed to {ratio_value} for {family}."
                    if ratio_value is not None
                    else f"R3 is not yet frozen to a family-independent value for {family}."
                ),
            }
        )

    fixed_values = [value for value in ratio_values.values() if isinstance(value, float) and value == value]
    unique_across_surviving_candidates = len(set(fixed_values)) == len(fixed_values) and len(fixed_values) == len(
        surviving_candidate_ids
    )
    tie_break_route_available = unique_across_surviving_candidates and len(surviving_candidate_ids) > 1
    discriminant_gap = max(fixed_values) - min(fixed_values) if len(fixed_values) >= 2 else 0.0
    selection_ready = False

    rows = [
        {
            "row_id": "same_sector_tiebreak_surviving_candidate_count",
            "status": "watch",
            "metric": "surviving candidate count seen by tie-break invariant step",
            "value": float(len(surviving_candidate_ids)),
            "note": f"Current surviving same-sector families are {surviving_candidate_ids}.",
        },
        *ratio_rows,
        {
            "row_id": "same_sector_tiebreak_invariant_unique_across_survivors",
            "status": "pass" if unique_across_surviving_candidates else "reject",
            "metric": "dimensionless derivative-ratio invariant separates surviving families",
            "value": 1.0 if unique_across_surviving_candidates else 0.0,
            "note": (
                "The invariant values are distinct across the surviving families and therefore define a concrete tie-break route."
                if unique_across_surviving_candidates
                else "The invariant does not yet separate all surviving families."
            ),
        },
        {
            "row_id": "same_sector_tiebreak_target_value_observed",
            "status": "watch",
            "metric": "same-sector public target value for derivative-ratio invariant already observed",
            "value": 0.0,
            "note": "A discriminant route exists, but no public canonical same-sector target value has yet been derived from chi_P or the shell-family anchor pack.",
        },
        {
            "row_id": "same_sector_tiebreak_route_available",
            "status": "pass" if tie_break_route_available else "reject",
            "metric": "new same-sector tie-break route available",
            "value": 1.0 if tie_break_route_available else 0.0,
            "note": (
                "The new route is to derive or measure the same-sector target value of R3 and use it to choose between the surviving families."
                if tie_break_route_available
                else "No concrete same-sector tie-break route is available yet."
            ),
        },
    ]

    return {
        "generated_utc": _utc_now_iso(),
        "phase": {
            "phase": 8,
            "step": step_tag,
            "name": "same-sector derivative-ratio tie-break invariant",
        },
        "inputs": {
            "mass_origin_single_vpp_candidate_derivative_json": _relative_str(DERIVATIVE_JSON),
            "mass_origin_single_public_vpp_shape_closure_json": _relative_str(CLOSURE_JSON),
        },
        "intent": "Freeze a concrete same-sector tie-break route for the surviving V(|P|) families using the dimensionless derivative-ratio invariant R3 = |P|_* V''' / V''.",
        "formulas": {
            "tiebreak_invariant": "R3 = |P|_* V'''(|P|_*) / V''(|P|_*)",
            "mexican_hat_value": "R3 = 3",
            "logarithmic_value": "R3 = 1",
        },
        "rows": rows,
        "summary": {
            "surviving_candidate_count": len(surviving_candidate_ids),
            "surviving_candidate_ids": surviving_candidate_ids,
            "tiebreak_invariant_name": "absP_star_times_vppp_over_vpp",
            "fixed_value_candidate_count": fixed_value_rows,
            "surviving_candidate_invariant_values": ratio_values,
            "invariant_unique_across_surviving_candidates": unique_across_surviving_candidates,
            "invariant_value_gap": discriminant_gap,
            "tie_break_route_available": tie_break_route_available,
            "selection_ready": selection_ready,
        },
        "decision": {
            "overall_status": "same_sector_vpp_tiebreak_invariant_frozen",
            "keep_mass_origin_branch_blocked": True,
            "tie_break_route_available": tie_break_route_available,
            "selection_ready": selection_ready,
            "invariant_unique_across_surviving_candidates": unique_across_surviving_candidates,
            "blocked_state_detail": str(closure_decision.get("blocked_state_detail", "")),
            "next_required_artifacts": [
                "same_sector_tiebreak_target_value",
                "single_public_vpp_shape",
                "positive_particle_sector_chi_p_to_vpp_public_artifact",
                "solver_ready_row_promoted_to_pass",
            ],
        },
        "evidence": {
            "derivative_summary": derivative_summary,
            "closure_summary": closure_summary,
        },
    }


# 関数: `_write_csv` の入出力契約と処理意図を定義する。

def _write_csv(rows: List[Dict[str, Any]]) -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
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
