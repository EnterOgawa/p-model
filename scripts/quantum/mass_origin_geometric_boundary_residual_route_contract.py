#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_origin_geometric_boundary_residual_route_contract.py

Step 8.7.55.2.399:
Freeze the next residual branch after the first post-linearized binding-route
selection fails and only the geometric reflective-boundary route remains
admissible without extending field content.

Inputs:
  - output/public/quantum/mass_origin_postlinearized_binding_selection_gate_metrics.json
  - output/public/quantum/mass_origin_discrete_spectrum_reopen_refresh_metrics.json

Outputs:
  - output/public/quantum/mass_origin_geometric_boundary_residual_route_contract_metrics.json
  - output/public/quantum/mass_origin_geometric_boundary_residual_route_contract_rows.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]

SELECTION_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_postlinearized_binding_selection_gate_metrics.json"
REFRESH_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_discrete_spectrum_reopen_refresh_metrics.json"
OUT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_geometric_boundary_residual_route_contract_metrics.json"
OUT_CSV = ROOT / "output" / "public" / "quantum" / "mass_origin_geometric_boundary_residual_route_contract_rows.csv"
DEFAULT_STEP_TAG = "8.7.55.2.399"


# 関数: 現在UTC時刻を ISO 8601 文字列で返す。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: CLI 引数を解釈する。

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze the residual geometric-boundary branch contract.")
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
    geometric_route_still_admissible: bool,
    split_contract_ready: bool,
) -> List[Dict[str, Any]]:
    return [
        {
            "row_id": "geometric_boundary_residual_route_contract_complete",
            "status": "pass",
            "metric": "geometric-boundary residual route contract complete",
            "value": 1.0,
            "note": "This step freezes the residual branch after the first post-linearized binding selection remains blocked.",
        },
        {
            "row_id": "geometric_route_retained_for_residual_followup",
            "status": "pass" if geometric_route_still_admissible else "reject",
            "metric": "geometric reflective-boundary route retained for residual follow-up",
            "value": 1.0 if geometric_route_still_admissible else 0.0,
            "note": (
                "The geometric route remains the only residual path that does not require a field extension or an extra conserved charge."
                if geometric_route_still_admissible
                else "No admissible residual geometric route remains."
            ),
        },
        {
            "row_id": "geometric_boundary_residual_split_ready",
            "status": "pass" if split_contract_ready else "reject",
            "metric": "geometric-boundary residual split contract ready",
            "value": 1.0 if split_contract_ready else 0.0,
            "note": (
                "The next branch can now focus on shell-quantization-to-cavity source inventory and reflective-boundary closure."
                if split_contract_ready
                else "The residual split contract is not ready."
            ),
        },
    ]


# 関数: metrics payload 全体を構成する。

def _build_payload(step_tag: str) -> Dict[str, Any]:
    for path in (SELECTION_JSON, REFRESH_JSON):
        _require_path(path)

    selection = _read_json(SELECTION_JSON)
    refresh = _read_json(REFRESH_JSON)

    selection_summary = selection.get("summary", {})
    refresh_summary = refresh.get("summary", {})

    geometric_route_still_admissible = bool(selection_summary.get("geometric_route_still_admissible", False))
    selected_residual_binding_route = "geometric_reflective_boundary" if geometric_route_still_admissible else None
    split_contract_ready = geometric_route_still_admissible

    rows = _build_rows(
        geometric_route_still_admissible=geometric_route_still_admissible,
        split_contract_ready=split_contract_ready,
    )

    return {
        "generated_utc": _utc_now_iso(),
        "phase": {
            "phase": 8,
            "step": step_tag,
            "name": "geometric-boundary residual route contract",
        },
        "inputs": {
            "mass_origin_postlinearized_binding_selection_gate_json": _relative_str(SELECTION_JSON),
            "mass_origin_discrete_spectrum_reopen_refresh_json": _relative_str(REFRESH_JSON),
        },
        "intent": "Freeze the residual route after post-linearized binding remains blocked and geometric reflective-boundary remains the only admissible no-new-free-parameter path.",
        "formulas": {
            "residual_route_rule": "if no route promotes but geometric reflective-boundary remains admissible, the next branch must focus on shell-quantization-to-cavity closure",
        },
        "rows": rows,
        "summary": {
            "selected_residual_binding_route_or_none": selected_residual_binding_route,
            "missing_geometric_boundary_artifact": "shell_quantization_reflective_cavity_rule",
            "excluded_binding_route_ids": [
                "nonlinear_self_binding",
                "complex_field_u1_stabilization",
            ],
            "split_contract_ready": split_contract_ready,
        },
        "decision": {
            "overall_status": "geometric_boundary_residual_route_contract_frozen",
            "keep_mass_origin_branch_blocked": True,
            "selected_residual_binding_route_or_none": selected_residual_binding_route,
            "missing_geometric_boundary_artifact": "shell_quantization_reflective_cavity_rule",
            "split_contract_ready": split_contract_ready,
            "hand_off_to_8_7_55_2_84": False,
            "next_required_artifacts": [
                "shell_quantization_reflective_cavity_rule",
                "discrete_spectrum_second_reopen_refresh",
            ],
        },
        "evidence": {
            "postlinearized_binding_selection_gate_summary": selection_summary,
            "discrete_spectrum_reopen_refresh_summary": refresh_summary,
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
