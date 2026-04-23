#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_origin_single_public_vpp_shape_closure.py

Step 8.7.55.2.80:
Freeze the closure or non-closure of a single public V(|P|) shape after the
same-sector admissibility gate.

Inputs:
  - output/public/quantum/mass_origin_same_sector_candidate_admissibility_metrics.json

Outputs:
  - output/public/quantum/mass_origin_single_public_vpp_shape_closure_metrics.json
  - output/public/quantum/mass_origin_single_public_vpp_shape_closure_rows.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]

ADMISSIBILITY_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_same_sector_candidate_admissibility_metrics.json"
OUT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_single_public_vpp_shape_closure_metrics.json"
OUT_CSV = ROOT / "output" / "public" / "quantum" / "mass_origin_single_public_vpp_shape_closure_rows.csv"
DEFAULT_STEP_TAG = "8.7.55.2.80"


# 関数: `_utc_now_iso` の入出力契約と処理意図を定義する。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_parse_args` の入出力契約と処理意図を定義する。

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze closure or non-closure of a single public V(|P|) shape.",
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


# 関数: `_find_row_by_id` の入出力契約と処理意図を定義する。

def _find_row_by_id(rows: List[Dict[str, Any]], row_id: str) -> Dict[str, Any]:
    for row in rows:
        # 条件分岐: `str(row.get("row_id")) == row_id` を満たす経路を評価する。
        if str(row.get("row_id")) == row_id:
            return row

    raise KeyError(f"missing row_id: {row_id}")


# 関数: `_build_payload` の入出力契約と処理意図を定義する。

def _build_payload(step_tag: str) -> Dict[str, Any]:
    _require_path(ADMISSIBILITY_JSON)
    admissibility = _read_json(ADMISSIBILITY_JSON)

    summary = admissibility.get("summary", {})
    decision = admissibility.get("decision", {})
    rows = admissibility.get("rows", [])

    # 条件分岐: `not isinstance(rows, list)` を満たす経路を評価する。
    if not isinstance(rows, list):
        raise SystemExit(f"[fail] invalid rows in {ADMISSIBILITY_JSON}")

    surviving_candidate_ids = [str(item) for item in summary.get("surviving_candidate_ids", [])]
    rejected_candidate_ids = [str(item) for item in summary.get("rejected_candidate_ids", [])]
    surviving_candidate_count = int(summary.get("surviving_candidate_count", 0))
    candidate_family_count = int(summary.get("candidate_family_count", 0))
    single_shape_ready = bool(summary.get("single_shape_ready", False))
    admissibility_nonclosure_reason = str(summary.get("admissibility_nonclosure_reason", ""))

    selected_candidate_id_or_none: str | None = surviving_candidate_ids[0] if single_shape_ready else None
    single_public_vpp_shape_available = single_shape_ready
    curvature_coefficients_fixed = single_shape_ready
    three_wave_coefficients_fixed = single_shape_ready

    # 条件分岐: `single_shape_ready` を満たす経路を評価する。
    if single_shape_ready:
        closure_status = "closed_single_same_sector_candidate"
        nonclosure_reason_or_none = None

    else:
        closure_status = "watch_nonclosure_two_minimal_same_sector_families_remain"
        nonclosure_reason_or_none = admissibility_nonclosure_reason

    mexican_hat_row = _find_row_by_id(rows, "candidate_admissibility_mexican_hat")
    logarithmic_row = _find_row_by_id(rows, "candidate_admissibility_logarithmic")
    even_polynomial_row = _find_row_by_id(rows, "candidate_admissibility_even_polynomial")

    closure_rows = [
        {
            "row_id": "single_vpp_shape_closure_attempt_complete",
            "status": "pass",
            "metric": "single_public_vpp_shape closure attempt complete",
            "value": 1.0,
            "family": "aggregate",
            "note": "The closure artifact evaluates whether the admissible same-sector set can be reduced to one public V(|P|) shape.",
        },
        {
            "row_id": "single_vpp_shape_surviving_candidate_count",
            "status": "watch" if not single_shape_ready else "pass",
            "metric": "surviving admissible candidate count seen by closure step",
            "value": float(surviving_candidate_count),
            "family": "aggregate",
            "note": f"Surviving candidate ids are {surviving_candidate_ids}.",
        },
        {
            "row_id": "single_vpp_shape_selected_candidate",
            "status": "pass" if single_shape_ready else "watch",
            "metric": "selected same-sector public V(|P|) candidate",
            "value": 1.0 if single_shape_ready else 0.0,
            "family": selected_candidate_id_or_none or "none",
            "note": (
                f"Single public V(|P|) shape is fixed to {selected_candidate_id_or_none}."
                if single_shape_ready
                else "No unique same-sector candidate can be selected because the admissible set still contains more than one family."
            ),
        },
        {
            "row_id": "single_vpp_shape_mexican_hat_still_live",
            "status": str(mexican_hat_row.get("status", "watch")),
            "metric": "mexican_hat remains in closure candidate set",
            "value": float(mexican_hat_row.get("value", 0.0)),
            "family": "mexican_hat",
            "note": str(mexican_hat_row.get("note", "")),
        },
        {
            "row_id": "single_vpp_shape_logarithmic_still_live",
            "status": str(logarithmic_row.get("status", "watch")),
            "metric": "logarithmic remains in closure candidate set",
            "value": float(logarithmic_row.get("value", 0.0)),
            "family": "logarithmic",
            "note": str(logarithmic_row.get("note", "")),
        },
        {
            "row_id": "single_vpp_shape_even_polynomial_rejected",
            "status": str(even_polynomial_row.get("status", "reject")),
            "metric": "even_polynomial remains excluded from closure candidate set",
            "value": float(even_polynomial_row.get("value", 0.0)),
            "family": "even_polynomial",
            "note": str(even_polynomial_row.get("note", "")),
        },
        {
            "row_id": "single_vpp_shape_nonclosure_reason",
            "status": "watch" if not single_shape_ready else "pass",
            "metric": "single_public_vpp_shape nonclosure reason fixed",
            "value": 0.0 if not single_shape_ready else 1.0,
            "family": "aggregate",
            "note": (
                f"Nonclosure reason is {nonclosure_reason_or_none}."
                if nonclosure_reason_or_none
                else "No nonclosure reason remains because the shape is closed."
            ),
        },
    ]

    return {
        "generated_utc": _utc_now_iso(),
        "phase": {
            "phase": 8,
            "step": step_tag,
            "name": "single_public_vpp_shape closure attempt",
        },
        "inputs": {
            "mass_origin_same_sector_candidate_admissibility_json": _relative_str(ADMISSIBILITY_JSON),
        },
        "intent": "Freeze whether the surviving same-sector V(|P|) candidate set closes to a single public shape.",
        "formulas": {
            "closure_rule": "single_public_vpp_shape_available iff surviving_candidate_count == 1",
            "current_nonclosure_rule": "nonclosure persists when more than one admissible minimal same-sector family remains",
        },
        "rows": closure_rows,
        "summary": {
            "candidate_family_count": candidate_family_count,
            "surviving_candidate_count": surviving_candidate_count,
            "surviving_candidate_ids": surviving_candidate_ids,
            "rejected_candidate_ids": rejected_candidate_ids,
            "selected_candidate_id_or_none": selected_candidate_id_or_none,
            "single_public_vpp_shape_available": single_public_vpp_shape_available,
            "curvature_coefficients_fixed": curvature_coefficients_fixed,
            "three_wave_coefficients_fixed": three_wave_coefficients_fixed,
            "closure_status": closure_status,
            "nonclosure_reason_or_none": nonclosure_reason_or_none,
        },
        "decision": {
            "overall_status": "single_public_vpp_shape_closure_frozen",
            "keep_mass_origin_branch_blocked": True,
            "selected_candidate_id_or_none": selected_candidate_id_or_none,
            "single_public_vpp_shape_available": single_public_vpp_shape_available,
            "curvature_coefficients_fixed": curvature_coefficients_fixed,
            "three_wave_coefficients_fixed": three_wave_coefficients_fixed,
            "closure_status": closure_status,
            "nonclosure_reason_or_none": nonclosure_reason_or_none,
            "blocked_state_detail": str(decision.get("blocked_state_detail", "")),
            "next_required_artifacts": decision.get(
                "next_required_artifacts",
                [
                    "positive_particle_sector_chi_p_to_vpp_public_artifact",
                    "single_public_vpp_shape",
                    "solver_ready_row_promoted_to_pass",
                ],
            ),
        },
        "evidence": {
            "admissibility_summary": summary,
            "admissibility_decision": decision,
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
