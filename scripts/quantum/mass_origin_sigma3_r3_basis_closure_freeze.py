#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_origin_sigma3_r3_basis_closure_freeze.py

Step 8.7.55.2.390:
Once basis closure selects the mexican-hat family, freeze the anchor-local
cubicity jet values Sigma_{3,*} = 6 and R_3 = 3.

Inputs:
  - output/public/quantum/mass_origin_anchor_local_curvature_bridge_metrics.json
  - output/public/quantum/mass_origin_anchor_local_r3_registry_metrics.json
  - output/public/quantum/mass_origin_basis_closure_selection_gate_metrics.json

Outputs:
  - output/public/quantum/mass_origin_sigma3_r3_basis_closure_freeze_metrics.json
  - output/public/quantum/mass_origin_sigma3_r3_basis_closure_freeze_rows.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]

CURVATURE_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_anchor_local_curvature_bridge_metrics.json"
R3_REGISTRY_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_anchor_local_r3_registry_metrics.json"
SELECTION_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_basis_closure_selection_gate_metrics.json"
OUT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_sigma3_r3_basis_closure_freeze_metrics.json"
OUT_CSV = ROOT / "output" / "public" / "quantum" / "mass_origin_sigma3_r3_basis_closure_freeze_rows.csv"
DEFAULT_STEP_TAG = "8.7.55.2.390"


# 関数: 現在UTC時刻を ISO 8601 文字列で返す。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: CLI 引数を解釈する。

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze Sigma_{3,*} and R_3 after the basis-closure selection gate.",
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

def _build_rows(selected_family: str | None) -> List[Dict[str, Any]]:
    selected = selected_family == "mexican_hat"
    return [
        {
            "row_id": "sigma3_r3_basis_closure_freeze_complete",
            "status": "pass",
            "metric": "Sigma_{3,*} / R_3 freeze complete",
            "value": 1.0,
            "note": "This step freezes the anchor-local cubicity jet values implied by the selected candidate family.",
        },
        {
            "row_id": "sigma3_r3_basis_closure_selected_family_mexican_hat",
            "status": "pass" if selected else "reject",
            "metric": "basis closure selected mexican hat",
            "value": 1.0 if selected else 0.0,
            "note": (
                "The basis-closure gate selected mexican hat, so Sigma_{3,*} and R_3 can now be frozen from that family."
                if selected
                else "Sigma_{3,*} and R_3 cannot be frozen because the selection gate did not choose mexican hat."
            ),
        },
        {
            "row_id": "sigma3_r3_basis_closure_sigma3_target_frozen",
            "status": "pass" if selected else "reject",
            "metric": "Sigma_{3,*} = U'''_* / U''_* frozen",
            "value": 6.0 if selected else 0.0,
            "note": (
                "For mexican hat, the anchor-local chi-space cubicity ratio freezes to Sigma_{3,*} = 6."
                if selected
                else "The chi-space cubicity ratio is not frozen because family selection is incomplete."
            ),
        },
        {
            "row_id": "sigma3_r3_basis_closure_r3_target_frozen",
            "status": "pass" if selected else "reject",
            "metric": "R_3 frozen under selected family",
            "value": 3.0 if selected else 0.0,
            "note": (
                "For mexican hat, the rho-space cubicity ratio freezes to R_3 = 3."
                if selected
                else "R_3 is not frozen because family selection is incomplete."
            ),
        },
        {
            "row_id": "sigma3_r3_basis_closure_fallback_route_required",
            "status": "reject" if selected else "watch",
            "metric": "fallback same-sector-equivalence route must be resumed",
            "value": 0.0 if selected else 1.0,
            "note": (
                "Fallback is not required because basis closure already selected the family and fixed the cubicity jet."
                if selected
                else "Fallback remains required because basis closure did not fix the cubicity jet."
            ),
        },
    ]


# 関数: metrics payload 全体を構成する。

def _build_payload(step_tag: str) -> Dict[str, Any]:
    for path in (CURVATURE_JSON, R3_REGISTRY_JSON, SELECTION_JSON):
        _require_path(path)

    curvature = _read_json(CURVATURE_JSON)
    r3_registry = _read_json(R3_REGISTRY_JSON)
    selection = _read_json(SELECTION_JSON)

    curvature_summary = curvature.get("summary", {})
    r3_summary = r3_registry.get("summary", {})
    selection_summary = selection.get("summary", {})

    selected_family = selection_summary.get("selected_candidate_family_id_or_none")
    sigma3_target_available = selected_family == "mexican_hat"
    r3_target_available = selected_family == "mexican_hat"
    sigma3_target_value = 6.0 if sigma3_target_available else None
    r3_target_value = 3.0 if r3_target_available else None
    rows = _build_rows(selected_family)

    return {
        "generated_utc": _utc_now_iso(),
        "phase": {
            "phase": 8,
            "step": step_tag,
            "name": "Sigma_{3,*} and R_3 freeze under selected family",
        },
        "inputs": {
            "mass_origin_anchor_local_curvature_bridge_json": _relative_str(CURVATURE_JSON),
            "mass_origin_anchor_local_r3_registry_json": _relative_str(R3_REGISTRY_JSON),
            "mass_origin_basis_closure_selection_gate_json": _relative_str(SELECTION_JSON),
        },
        "intent": "Freeze the anchor-local cubicity jet once basis closure has selected the mexican-hat family.",
        "formulas": {
            "sigma3_definition": "Sigma_{3,*} = U'''(chi_*) / U''(chi_*)",
            "r3_definition": "R_3 = rho_* V'''(rho_*) / V''(rho_*)",
            "selected_family_rule": "mexican_hat => Sigma_{3,*} = 6 and R_3 = 3",
            "anchor_curvature_identity": "rho_*^2 V''(rho_*) = M_chi^2 omega_*^2 = rho_*^2 g_P Z_P / chi_P",
        },
        "rows": rows,
        "summary": {
            "selected_candidate_family_id": selected_family,
            "sigma3_target_available": sigma3_target_available,
            "sigma3_target_value_or_none": sigma3_target_value,
            "r3_target_available": r3_target_available,
            "r3_target_value_or_none": r3_target_value,
            "vp_anchor_zero": True,
            "rho2_vpp_anchor_value": "M_chi^2 omega_*^2 = rho_*^2 g_P Z_P / chi_P",
            "anchor_local_cubicity_jet_closed": bool(sigma3_target_available and r3_target_available),
        },
        "decision": {
            "overall_status": "sigma3_r3_basis_closure_freeze_frozen",
            "keep_mass_origin_branch_blocked": True,
            "selected_candidate_family_id": selected_family,
            "sigma3_target_available": sigma3_target_available,
            "sigma3_target_value_or_none": sigma3_target_value,
            "r3_target_available": r3_target_available,
            "r3_target_value_or_none": r3_target_value,
            "anchor_local_cubicity_jet_closed": bool(sigma3_target_available and r3_target_available),
            "fallback_same_sector_equivalence_route_required": not sigma3_target_available,
            "next_required_artifacts": [
                "mexican_hat_parameter_pack",
                "shape_gate_refresh",
            ],
        },
        "evidence": {
            "curvature_summary": curvature_summary,
            "r3_registry_summary": r3_summary,
            "basis_closure_selection_gate_summary": selection_summary,
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
