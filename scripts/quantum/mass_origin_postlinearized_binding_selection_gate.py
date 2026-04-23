#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_origin_postlinearized_binding_selection_gate.py

Step 8.7.55.2.397:
Combine the three post-linearized binding-route audits and decide whether a
unique no-new-free-parameter reopening channel is available.

Inputs:
  - output/public/quantum/mass_origin_postlinearized_binding_route_contract_metrics.json
  - output/public/quantum/mass_origin_geometric_boundary_promotion_metrics.json
  - output/public/quantum/mass_origin_nonlinear_self_binding_metrics.json
  - output/public/quantum/mass_origin_complex_field_stabilization_metrics.json

Outputs:
  - output/public/quantum/mass_origin_postlinearized_binding_selection_gate_metrics.json
  - output/public/quantum/mass_origin_postlinearized_binding_selection_gate_rows.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]

CONTRACT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_postlinearized_binding_route_contract_metrics.json"
GEOMETRIC_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_geometric_boundary_promotion_metrics.json"
NONLINEAR_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_nonlinear_self_binding_metrics.json"
COMPLEX_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_complex_field_stabilization_metrics.json"
OUT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_postlinearized_binding_selection_gate_metrics.json"
OUT_CSV = ROOT / "output" / "public" / "quantum" / "mass_origin_postlinearized_binding_selection_gate_rows.csv"
DEFAULT_STEP_TAG = "8.7.55.2.397"


# 関数: 現在UTC時刻を ISO 8601 文字列で返す。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: CLI 引数を解釈する。

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select the post-linearized binding route, if any.")
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


# 関数: rows を構成する。

def _build_rows(
    *,
    geometric_ready: bool,
    nonlinear_ready: bool,
    complex_ready: bool,
    geometric_route_still_admissible: bool,
    selected_binding_route_or_none: str | None,
    discrete_spectrum_reopen_ready: bool,
) -> List[Dict[str, Any]]:
    return [
        {
            "row_id": "postlinearized_binding_selection_gate_complete",
            "status": "pass",
            "metric": "post-linearized binding selection gate complete",
            "value": 1.0,
            "note": "This step combines the geometric, nonlinear, and complex-field post-linearized binding-route audits.",
        },
        {
            "row_id": "geometric_binding_route_promoted",
            "status": "pass" if geometric_ready else "reject",
            "metric": "geometric reflective-boundary route promoted",
            "value": 1.0 if geometric_ready else 0.0,
            "note": "The geometric route is the only remaining no-new-free-parameter candidate if shell quantization can be lifted into a cavity rule.",
        },
        {
            "row_id": "nonlinear_binding_route_promoted",
            "status": "pass" if nonlinear_ready else "reject",
            "metric": "nonlinear self-binding route promoted",
            "value": 1.0 if nonlinear_ready else 0.0,
            "note": "The nonlinear route requires a public stable self-binding artifact before it can be selected.",
        },
        {
            "row_id": "complex_field_binding_route_promoted",
            "status": "pass" if complex_ready else "reject",
            "metric": "complex-field U(1) route promoted",
            "value": 1.0 if complex_ready else 0.0,
            "note": "The complex-field route requires a public canonical complex-field artifact before it can be selected.",
        },
        {
            "row_id": "geometric_route_still_admissible",
            "status": "pass" if geometric_route_still_admissible else "reject",
            "metric": "geometric route still admissible as residual no-new-free-parameter path",
            "value": 1.0 if geometric_route_still_admissible else 0.0,
            "note": (
                "Even though not yet promoted, the geometric route is still the only residual path that does not require new field content or extra conserved charge."
                if geometric_route_still_admissible
                else "No residual no-new-free-parameter geometric route remains."
            ),
        },
        {
            "row_id": "selected_binding_route_unique",
            "status": "pass" if selected_binding_route_or_none is not None else "reject",
            "metric": "selected binding route unique",
            "value": 1.0 if selected_binding_route_or_none is not None else 0.0,
            "note": (
                f"The unique selected route is {selected_binding_route_or_none}."
                if selected_binding_route_or_none is not None
                else "No post-linearized binding route has yet promoted into a unique reopening channel."
            ),
        },
        {
            "row_id": "discrete_spectrum_reopen_ready",
            "status": "pass" if discrete_spectrum_reopen_ready else "reject",
            "metric": "discrete-spectrum reopen ready",
            "value": 1.0 if discrete_spectrum_reopen_ready else 0.0,
            "note": (
                "A unique promoted binding channel is available for discrete-spectrum refresh."
                if discrete_spectrum_reopen_ready
                else "Discrete-spectrum refresh remains blocked because no promoted binding route exists."
            ),
        },
    ]


# 関数: metrics payload 全体を構成する。

def _build_payload(step_tag: str) -> Dict[str, Any]:
    for path in (CONTRACT_JSON, GEOMETRIC_JSON, NONLINEAR_JSON, COMPLEX_JSON):
        _require_path(path)

    contract = _read_json(CONTRACT_JSON)
    geometric = _read_json(GEOMETRIC_JSON)
    nonlinear = _read_json(NONLINEAR_JSON)
    complex_field = _read_json(COMPLEX_JSON)

    contract_summary = contract.get("summary", {})
    geometric_summary = geometric.get("summary", {})
    nonlinear_summary = nonlinear.get("summary", {})
    complex_summary = complex_field.get("summary", {})

    geometric_ready = bool(geometric_summary.get("geometric_boundary_promotion_rule_available", False))
    nonlinear_ready = bool(nonlinear_summary.get("nonlinear_self_binding_admissible", False))
    complex_ready = bool(complex_summary.get("complex_field_u1_stabilization_artifact_available", False))
    promoted_routes = [
        route
        for route, ready in (
            ("geometric_reflective_boundary", geometric_ready),
            ("nonlinear_self_binding", nonlinear_ready),
            ("complex_field_u1_stabilization", complex_ready),
        )
        if ready
    ]
    selected_binding_route_or_none = promoted_routes[0] if len(promoted_routes) == 1 else None
    binding_route_unique = selected_binding_route_or_none is not None
    no_new_free_parameter_binding_channel = binding_route_unique
    discrete_spectrum_reopen_ready = binding_route_unique
    geometric_route_still_admissible = bool(geometric_summary.get("geometric_route_still_admissible", False))
    residual_preferred_route_or_none = "geometric_reflective_boundary" if (not binding_route_unique and geometric_route_still_admissible) else None
    remaining_binding_blockers = []

    if not geometric_ready:
        remaining_binding_blockers.append("geometric_boundary_promotion_rule")

    if not nonlinear_ready:
        remaining_binding_blockers.append("nonlinear_self_binding_admissibility")

    if not complex_ready:
        remaining_binding_blockers.append("complex_field_u1_stabilization_artifact")

    selection_nonclosure_reason_or_none = None if binding_route_unique else "no_postlinearized_binding_route_promoted"

    rows = _build_rows(
        geometric_ready=geometric_ready,
        nonlinear_ready=nonlinear_ready,
        complex_ready=complex_ready,
        geometric_route_still_admissible=geometric_route_still_admissible,
        selected_binding_route_or_none=selected_binding_route_or_none,
        discrete_spectrum_reopen_ready=discrete_spectrum_reopen_ready,
    )

    return {
        "generated_utc": _utc_now_iso(),
        "phase": {
            "phase": 8,
            "step": step_tag,
            "name": "post-linearized binding selection gate",
        },
        "inputs": {
            "mass_origin_postlinearized_binding_route_contract_json": _relative_str(CONTRACT_JSON),
            "mass_origin_geometric_boundary_promotion_json": _relative_str(GEOMETRIC_JSON),
            "mass_origin_nonlinear_self_binding_json": _relative_str(NONLINEAR_JSON),
            "mass_origin_complex_field_stabilization_json": _relative_str(COMPLEX_JSON),
        },
        "intent": "Decide whether any post-linearized binding route has promoted into a unique no-new-free-parameter reopening channel.",
        "formulas": {
            "selection_rule": "discrete_spectrum_reopen_ready iff exactly one binding route promotes into a no-new-free-parameter reopening channel",
            "residual_route_rule": "if no route promotes but the geometric route remains admissible, the next residual branch should focus on shell-quantization-to-cavity promotion",
        },
        "rows": rows,
        "summary": {
            "candidate_binding_route_ids": contract_summary.get("candidate_binding_route_ids", []),
            "selected_binding_route_or_none": selected_binding_route_or_none,
            "binding_route_unique": binding_route_unique,
            "no_new_free_parameter_binding_channel": no_new_free_parameter_binding_channel,
            "discrete_spectrum_reopen_ready": discrete_spectrum_reopen_ready,
            "geometric_route_still_admissible": geometric_route_still_admissible,
            "residual_preferred_route_or_none": residual_preferred_route_or_none,
            "remaining_binding_blockers": remaining_binding_blockers,
            "selection_nonclosure_reason_or_none": selection_nonclosure_reason_or_none,
        },
        "decision": {
            "overall_status": "postlinearized_binding_selection_gate_frozen",
            "keep_mass_origin_branch_blocked": True,
            "selected_binding_route_or_none": selected_binding_route_or_none,
            "binding_route_unique": binding_route_unique,
            "no_new_free_parameter_binding_channel": no_new_free_parameter_binding_channel,
            "discrete_spectrum_reopen_ready": discrete_spectrum_reopen_ready,
            "geometric_route_still_admissible": geometric_route_still_admissible,
            "residual_preferred_route_or_none": residual_preferred_route_or_none,
            "remaining_binding_blockers": remaining_binding_blockers,
            "selection_nonclosure_reason_or_none": selection_nonclosure_reason_or_none,
        },
        "evidence": {
            "postlinearized_binding_route_contract_summary": contract_summary,
            "geometric_boundary_promotion_summary": geometric_summary,
            "nonlinear_self_binding_summary": nonlinear_summary,
            "complex_field_stabilization_summary": complex_summary,
        },
    }


# 関数: rows を CSV 出力する。

def _write_csv(rows: List[Dict[str, Any]]) -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["row_id", "status", "metric", "value", "note"])
        writer.writeheader()
        writer.writerows(rows)


# 関数: JSON を整形出力する。

def _write_json(payload: Dict[str, Any]) -> None:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


# 関数: エントリポイントとして step を実行する。

def main() -> None:
    args = _parse_args()
    payload = _build_payload(args.step_tag)
    _write_json(payload)
    _write_csv(payload["rows"])
    print(f"[ok] wrote {OUT_JSON}")
    print(f"[ok] wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
