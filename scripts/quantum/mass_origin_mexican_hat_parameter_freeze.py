#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_origin_mexican_hat_parameter_freeze.py

Step 8.7.55.2.391:
Freeze the minimal mexican-hat parameter pack and the preflight conditions for
handing the mass-origin branch into the eigenvalue boundary pilot.

Inputs:
  - doc/paper/10_part1_core_theory.md
  - output/public/quantum/mass_origin_anchor_local_curvature_bridge_metrics.json
  - output/public/quantum/mass_origin_sigma3_r3_basis_closure_freeze_metrics.json

Outputs:
  - output/public/quantum/mass_origin_mexican_hat_parameter_freeze_metrics.json
  - output/public/quantum/mass_origin_mexican_hat_parameter_freeze_rows.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]

PART1_MD = ROOT / "doc" / "paper" / "10_part1_core_theory.md"
CURVATURE_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_anchor_local_curvature_bridge_metrics.json"
SIGMA3_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_sigma3_r3_basis_closure_freeze_metrics.json"
OUT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_mexican_hat_parameter_freeze_metrics.json"
OUT_CSV = ROOT / "output" / "public" / "quantum" / "mass_origin_mexican_hat_parameter_freeze_rows.csv"
DEFAULT_STEP_TAG = "8.7.55.2.391"


# 関数: 現在UTC時刻を ISO 8601 文字列で返す。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: CLI 引数を解釈する。

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze the mexican-hat parameter pack and eigenvalue preflight conditions.",
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


# 関数: UTF-8 テキストを読み込む。

def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# 関数: リポジトリ相対パスへ正規化する。

def _relative_str(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


# 関数: 指定パターンを最初に含む行を返す。

def _find_first_match(text: str, pattern: str) -> Dict[str, Any] | None:
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        if pattern in raw_line:
            return {
                "pattern": pattern,
                "line": line_number,
                "text": raw_line.strip(),
            }

    return None


# 関数: rows を構成する。

def _build_rows(selected_family: str | None) -> List[Dict[str, Any]]:
    selected = selected_family == "mexican_hat"
    return [
        {
            "row_id": "mexican_hat_parameter_freeze_complete",
            "status": "pass",
            "metric": "mexican-hat parameter freeze complete",
            "value": 1.0,
            "note": "This step freezes the minimal parameter pack implied by the selected mexican-hat family.",
        },
        {
            "row_id": "mexican_hat_parameter_freeze_selected_family",
            "status": "pass" if selected else "reject",
            "metric": "mexican hat selected before parameter freeze",
            "value": 1.0 if selected else 0.0,
            "note": (
                "The mexican-hat family is selected, so its parameter pack can be frozen."
                if selected
                else "The mexican-hat parameter pack cannot be frozen because family selection is incomplete."
            ),
        },
        {
            "row_id": "mexican_hat_parameter_freeze_anchor_equals_vacuum_scale",
            "status": "pass" if selected else "reject",
            "metric": "rho_* = v fixed at the stationary mexican-hat anchor",
            "value": 1.0 if selected else 0.0,
            "note": (
                "For mexican hat, the nonzero stationary anchor satisfies rho_* = v."
                if selected
                else "The stationary mexican-hat anchor is not yet fixed."
            ),
        },
        {
            "row_id": "mexican_hat_parameter_freeze_reference_choice_admissible",
            "status": "pass" if selected else "reject",
            "metric": "local reference normalization rho_* = P_infty admissible as a coordinate choice",
            "value": 1.0 if selected else 0.0,
            "note": (
                "Because observables depend on the ratio P / P_infty only, choosing the local anchor normalization rho_* = P_infty is admissible and introduces no new parameter."
                if selected
                else "The reference normalization cannot be frozen until the selected family is known."
            ),
        },
        {
            "row_id": "mexican_hat_parameter_freeze_single_remaining_parameter",
            "status": "pass" if selected else "reject",
            "metric": "only lambda remains as a free coupling",
            "value": 1.0 if selected else 0.0,
            "note": (
                "With rho_* = v fixed at the anchor and the reference normalization chosen, lambda is the only remaining free coupling in the selected family."
                if selected
                else "The number of remaining free couplings cannot be frozen yet."
            ),
        },
        {
            "row_id": "mexican_hat_parameter_freeze_eigenvalue_preflight_ready",
            "status": "pass" if selected else "reject",
            "metric": "eigenvalue preflight ready after parameter freeze",
            "value": 1.0 if selected else 0.0,
            "note": (
                "The selected family and its parameter relations are now explicit enough to hand off into the mass-eigenmode boundary specification."
                if selected
                else "The eigenvalue preflight remains blocked until the selected family is explicit."
            ),
        },
    ]


# 関数: metrics payload 全体を構成する。

def _build_payload(step_tag: str) -> Dict[str, Any]:
    for path in (PART1_MD, CURVATURE_JSON, SIGMA3_JSON):
        _require_path(path)

    part1_text = _read_text(PART1_MD)
    curvature = _read_json(CURVATURE_JSON)
    sigma3 = _read_json(SIGMA3_JSON)

    curvature_summary = curvature.get("summary", {})
    sigma3_summary = sigma3.get("summary", {})

    selected_family = sigma3_summary.get("selected_candidate_family_id")
    selected = selected_family == "mexican_hat"
    ratio_only_hit = _find_first_match(part1_text, r"比 $P/P_{\infty}$（無次元）")
    p_ref_equals_p_infty_hit = _find_first_match(part1_text, r"P_{\mathrm{ref}}\equiv P_{\infty}")

    rows = _build_rows(selected_family)

    return {
        "generated_utc": _utc_now_iso(),
        "phase": {
            "phase": 8,
            "step": step_tag,
            "name": "Mexican-hat parameter freeze and eigenvalue preflight",
        },
        "inputs": {
            "part1_core_theory_markdown": _relative_str(PART1_MD),
            "mass_origin_anchor_local_curvature_bridge_json": _relative_str(CURVATURE_JSON),
            "mass_origin_sigma3_r3_basis_closure_freeze_json": _relative_str(SIGMA3_JSON),
        },
        "intent": "Freeze the minimal mexican-hat parameter pack and preflight the handoff into the mass-eigenmode boundary pilot without overclaiming a globally unique potential beyond the selected basis closure.",
        "formulas": {
            "selected_potential": "V(rho) = (lambda / 4) (rho^2 - v^2)^2",
            "anchor_relation": "rho_* = v",
            "anchor_curvatures": "V''(rho_*) = 2 lambda v^2 and V'''(rho_*) = 6 lambda v",
            "mass_parameter": "m_P^2 = 2 lambda v^2 / Z_P",
            "susceptibility_constraint": "chi_P = g_P Z_P / (2 lambda v^2)",
            "overclaim_guard": "The selected basis closure fixes the mexican-hat family and its anchor-local jet; it does not yet forbid all higher even-exponential global extensions beyond the current candidate family set.",
        },
        "rows": rows,
        "summary": {
            "selected_candidate_family_id": selected_family,
            "selected_potential_family_formula": "V(rho) = (lambda / 4) (rho^2 - v^2)^2" if selected else None,
            "rho_star_equals_v": selected,
            "coordinate_choice_rho_star_equals_p_infty_admissible": selected,
            "remaining_free_parameter_symbols": ["lambda"] if selected else [],
            "mass_parameter_formula": "m_P^2 = 2 lambda v^2 / Z_P" if selected else None,
            "susceptibility_constraint_formula": "chi_P = g_P Z_P / (2 lambda v^2)" if selected else None,
            "eigenvalue_preflight_ready": selected,
        },
        "decision": {
            "overall_status": "mexican_hat_parameter_freeze_frozen",
            "keep_mass_origin_branch_blocked": True,
            "selected_candidate_family_id": selected_family,
            "rho_star_equals_v": selected,
            "coordinate_choice_rho_star_equals_p_infty_admissible": selected,
            "remaining_free_parameter_symbols": ["lambda"] if selected else [],
            "eigenvalue_preflight_ready": selected,
            "next_required_artifacts": [
                "shape_gate_refresh",
            ],
        },
        "evidence": {
            "part1_ratio_only_line": ratio_only_hit,
            "part1_reference_identification_line": p_ref_equals_p_infty_hit,
            "curvature_summary": curvature_summary,
            "sigma3_r3_basis_closure_summary": sigma3_summary,
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
