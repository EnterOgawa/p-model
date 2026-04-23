#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_origin_anchor_local_shape_gate_basis_closure_refresh.py

Step 8.7.55.2.392:
Refresh the anchor-local shape gate after the chi-space basis-closure route has
selected the mexican-hat family, and determine whether the mass-origin branch
may finally hand off into 8.7.55.2.83-.84.

Inputs:
  - output/public/quantum/mass_origin_same_sector_vpp_shape_gate_metrics.json
  - output/public/quantum/mass_origin_anchor_local_curvature_bridge_metrics.json
  - output/public/quantum/mass_origin_basis_closure_selection_gate_metrics.json
  - output/public/quantum/mass_origin_sigma3_r3_basis_closure_freeze_metrics.json
  - output/public/quantum/mass_origin_mexican_hat_parameter_freeze_metrics.json

Outputs:
  - output/public/quantum/mass_origin_anchor_local_shape_gate_basis_closure_refresh_metrics.json
  - output/public/quantum/mass_origin_anchor_local_shape_gate_basis_closure_refresh_rows.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]

SAME_SECTOR_GATE_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_same_sector_vpp_shape_gate_metrics.json"
CURVATURE_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_anchor_local_curvature_bridge_metrics.json"
SELECTION_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_basis_closure_selection_gate_metrics.json"
SIGMA3_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_sigma3_r3_basis_closure_freeze_metrics.json"
PARAM_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_mexican_hat_parameter_freeze_metrics.json"
OUT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_anchor_local_shape_gate_basis_closure_refresh_metrics.json"
OUT_CSV = ROOT / "output" / "public" / "quantum" / "mass_origin_anchor_local_shape_gate_basis_closure_refresh_rows.csv"
DEFAULT_STEP_TAG = "8.7.55.2.392"


# 関数: 現在UTC時刻を ISO 8601 文字列で返す。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: CLI 引数を解釈する。

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refresh the anchor-local shape gate after chi-space basis closure selected mexican hat.",
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


# 関数: rows を構成する。

def _build_rows(
    *,
    single_boundary_family_fixed: bool,
    single_shape_available: bool,
    positive_artifact_available: bool,
    handoff_ready: bool,
) -> List[Dict[str, Any]]:
    return [
        {
            "row_id": "anchor_local_shape_gate_basis_closure_refresh_complete",
            "status": "pass",
            "metric": "anchor-local shape gate basis-closure refresh complete",
            "value": 1.0,
            "note": "This step re-injects the chi-space basis-closure result into the anchor-local shape gate and re-evaluates the handoff into the mass-eigenmode pilot.",
        },
        {
            "row_id": "anchor_local_shape_gate_basis_closure_single_boundary_family",
            "status": "pass" if single_boundary_family_fixed else "reject",
            "metric": "single public boundary family preserved during refresh",
            "value": 1.0 if single_boundary_family_fixed else 0.0,
            "note": (
                "Shell quantization remains the sole public boundary family during the basis-closure refresh."
                if single_boundary_family_fixed
                else "The boundary family is no longer unique, so the basis-closure refresh cannot hand off."
            ),
        },
        {
            "row_id": "anchor_local_shape_gate_basis_closure_single_shape",
            "status": "pass" if single_shape_available else "reject",
            "metric": "single public V(|P|) shape available after basis closure refresh",
            "value": 1.0 if single_shape_available else 0.0,
            "note": (
                "The chi-space basis-closure route fixes mexican hat as the unique public V(|P|) family."
                if single_shape_available
                else "The chi-space basis-closure route did not yet produce a single public V(|P|) family."
            ),
        },
        {
            "row_id": "anchor_local_shape_gate_basis_closure_positive_artifact",
            "status": "pass" if positive_artifact_available else "reject",
            "metric": "positive particle-sector chi_P -> V''(|P|_*) artifact available",
            "value": 1.0 if positive_artifact_available else 0.0,
            "note": (
                "The anchor-local curvature bridge plus the selected mexican-hat family now make the positive particle-sector chi_P -> V''(|P|_*) artifact explicit."
                if positive_artifact_available
                else "The positive particle-sector chi_P -> V''(|P|_*) artifact is still missing."
            ),
        },
        {
            "row_id": "solver_ready_row_promoted_to_pass",
            "status": "pass" if handoff_ready else "reject",
            "metric": "solver-ready row promoted to pass",
            "value": 1.0 if handoff_ready else 0.0,
            "note": (
                "With a single public V(|P|) shape and an explicit same-sector curvature artifact, the branch is ready to enter the mass-eigenmode boundary pilot."
                if handoff_ready
                else "The solver-ready row cannot promote because the shape-gate refresh is still incomplete."
            ),
        },
        {
            "row_id": "hand_off_to_8_7_55_2_83",
            "status": "pass" if handoff_ready else "reject",
            "metric": "handoff to 8.7.55.2.83-.84 allowed after basis-closure refresh",
            "value": 1.0 if handoff_ready else 0.0,
            "note": (
                "The mass-origin branch may proceed to the mass-eigenmode boundary specification and discrete-spectrum pilot."
                if handoff_ready
                else "Handoff remains blocked after the basis-closure refresh."
            ),
        },
    ]


# 関数: metrics payload 全体を構成する。

def _build_payload(step_tag: str) -> Dict[str, Any]:
    for path in (SAME_SECTOR_GATE_JSON, CURVATURE_JSON, SELECTION_JSON, SIGMA3_JSON, PARAM_JSON):
        _require_path(path)

    same_sector_gate = _read_json(SAME_SECTOR_GATE_JSON)
    curvature = _read_json(CURVATURE_JSON)
    selection = _read_json(SELECTION_JSON)
    sigma3 = _read_json(SIGMA3_JSON)
    params = _read_json(PARAM_JSON)

    same_sector_summary = same_sector_gate.get("summary", {})
    curvature_summary = curvature.get("summary", {})
    selection_summary = selection.get("summary", {})
    sigma3_summary = sigma3.get("summary", {})
    params_summary = params.get("summary", {})

    single_boundary_family_fixed = bool(same_sector_summary.get("single_public_boundary_family_fixed", False))
    selected_family = selection_summary.get("selected_candidate_family_id_or_none")
    single_shape_available = selected_family == "mexican_hat"
    positive_artifact_available = bool(
        single_shape_available
        and curvature_summary.get("vpp_closed_without_new_free_parameters", False)
        and params_summary.get("eigenvalue_preflight_ready", False)
    )
    handoff_ready = bool(single_boundary_family_fixed and single_shape_available and positive_artifact_available)
    rows = _build_rows(
        single_boundary_family_fixed=single_boundary_family_fixed,
        single_shape_available=single_shape_available,
        positive_artifact_available=positive_artifact_available,
        handoff_ready=handoff_ready,
    )

    return {
        "generated_utc": _utc_now_iso(),
        "phase": {
            "phase": 8,
            "step": step_tag,
            "name": "anchor-local shape-gate refresh and Mexican-hat handoff",
        },
        "inputs": {
            "mass_origin_same_sector_vpp_shape_gate_json": _relative_str(SAME_SECTOR_GATE_JSON),
            "mass_origin_anchor_local_curvature_bridge_json": _relative_str(CURVATURE_JSON),
            "mass_origin_basis_closure_selection_gate_json": _relative_str(SELECTION_JSON),
            "mass_origin_sigma3_r3_basis_closure_freeze_json": _relative_str(SIGMA3_JSON),
            "mass_origin_mexican_hat_parameter_freeze_json": _relative_str(PARAM_JSON),
        },
        "intent": "Refresh the anchor-local shape gate after the chi-space basis-closure route and determine whether the mass-origin branch may hand off into 8.7.55.2.83-.84.",
        "formulas": {
            "single_shape_rule": "single_public_vpp_shape_available iff basis closure selects one candidate family",
            "positive_artifact_rule": "positive_particle_sector_chi_P_to_vpp_public_artifact_available iff the anchor-local curvature bridge is explicit and the selected family makes the same-sector curvature map concrete",
            "handoff_rule": "hand_off_to_8_7_55_2_83 iff single boundary family + single public shape + positive particle-sector chi_P -> V'' artifact all close together",
        },
        "rows": rows,
        "summary": {
            "single_public_boundary_family_fixed": single_boundary_family_fixed,
            "selected_candidate_family_id": selected_family,
            "rejected_candidate_family_ids": selection_summary.get("rejected_candidate_family_ids", []),
            "sigma3_target_value_or_none": sigma3_summary.get("sigma3_target_value_or_none"),
            "r3_target_value_or_none": sigma3_summary.get("r3_target_value_or_none"),
            "single_public_vpp_shape_available": single_shape_available,
            "positive_particle_sector_chi_p_to_vpp_public_artifact_available": positive_artifact_available,
            "solver_ready_row_promoted_to_pass": handoff_ready,
            "mass_origin_branch_reopen_ready": handoff_ready,
            "hand_off_to_8_7_55_2_83": handoff_ready,
            "eigenvalue_handoff_ready": handoff_ready,
        },
        "decision": {
            "overall_status": "anchor_local_shape_gate_basis_closure_refresh_frozen",
            "keep_mass_origin_branch_blocked": not handoff_ready,
            "single_public_boundary_family_fixed": single_boundary_family_fixed,
            "selected_candidate_family_id": selected_family,
            "single_public_vpp_shape_available": single_shape_available,
            "positive_particle_sector_chi_p_to_vpp_public_artifact_available": positive_artifact_available,
            "solver_ready_row_promoted_to_pass": handoff_ready,
            "mass_origin_branch_reopen_ready": handoff_ready,
            "proceed_to_no_free_parameter_mass_solver": handoff_ready,
            "hand_off_to_8_7_55_2_83": handoff_ready,
            "eigenvalue_handoff_ready": handoff_ready,
            "next_required_artifacts": [] if handoff_ready else ["basis_closure_failure_review"],
        },
        "evidence": {
            "same_sector_vpp_shape_gate_summary": same_sector_summary,
            "anchor_local_curvature_bridge_summary": curvature_summary,
            "basis_closure_selection_gate_summary": selection_summary,
            "sigma3_r3_basis_closure_freeze_summary": sigma3_summary,
            "mexican_hat_parameter_freeze_summary": params_summary,
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
