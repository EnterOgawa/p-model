#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_origin_mass_eigenmode_boundary_spec.py

Step 8.7.55.2.83:
Freeze the linearized mass-eigenmode boundary specification after the
chi-space basis-closure route selected the mexican-hat family, and test
whether the current no-new-free-parameter pilot already supports a discrete
bound-state spectrum.

Inputs:
  - output/public/quantum/mass_origin_anchor_local_shape_gate_basis_closure_refresh_metrics.json
  - output/public/quantum/mass_origin_mexican_hat_parameter_freeze_metrics.json
  - output/public/quantum/mass_origin_same_sector_vpp_shape_gate_metrics.json
  - output/public/quantum/mass_origin_shell_quantization_canonicalization_metrics.json
  - doc/quantum/18_p_field_action_and_schrodinger_mapping.md

Outputs:
  - output/public/quantum/mass_origin_mass_eigenmode_boundary_metrics.json
  - output/public/quantum/mass_origin_mass_eigenmode_boundary_rows.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]

SHAPE_GATE_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_anchor_local_shape_gate_basis_closure_refresh_metrics.json"
PARAM_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_mexican_hat_parameter_freeze_metrics.json"
SAME_SECTOR_GATE_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_same_sector_vpp_shape_gate_metrics.json"
SHELL_CANON_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_shell_quantization_canonicalization_metrics.json"
MASS_NOTE_MD = ROOT / "doc" / "quantum" / "18_p_field_action_and_schrodinger_mapping.md"
OUT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_mass_eigenmode_boundary_metrics.json"
OUT_CSV = ROOT / "output" / "public" / "quantum" / "mass_origin_mass_eigenmode_boundary_rows.csv"
DEFAULT_STEP_TAG = "8.7.55.2.83"


# 関数: 現在UTC時刻を ISO 8601 文字列で返す。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: CLI 引数を解釈する。

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze the linearized mass-eigenmode boundary specification and discrete-spectrum pilot.",
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


# 関数: 指定した row_id を rows から検索する。

def _find_row_by_id(rows: List[Dict[str, Any]], row_id: str) -> Dict[str, Any]:
    for row in rows:
        if str(row.get("row_id")) == row_id:
            return row

    raise KeyError(f"missing row_id: {row_id}")


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

def _build_rows(
    *,
    linearized_mode_equation_frozen: bool,
    boundary_conditions_frozen: bool,
    bound_state_problem_well_posed: bool,
    discrete_spectrum_found: bool,
    pilot_mode_count: int,
) -> List[Dict[str, Any]]:
    return [
        {
            "row_id": "mass_eigenmode_boundary_spec_complete",
            "status": "pass",
            "metric": "mass-eigenmode boundary specification complete",
            "value": 1.0,
            "note": "This step freezes the linearized mexican-hat mass-eigenmode pilot and checks whether the current no-new-free-parameter branch already yields a discrete bound-state ladder.",
        },
        {
            "row_id": "linearized_mode_equation_frozen",
            "status": "pass" if linearized_mode_equation_frozen else "reject",
            "metric": "linearized mode equation frozen",
            "value": 1.0 if linearized_mode_equation_frozen else 0.0,
            "note": (
                "The selected mexican-hat family freezes the pilot equation (Box + m_P^2) eta = 0 around rho_* = v."
                if linearized_mode_equation_frozen
                else "The linearized mass-eigenmode equation is not yet frozen."
            ),
        },
        {
            "row_id": "boundary_conditions_frozen",
            "status": "pass" if boundary_conditions_frozen else "reject",
            "metric": "pilot boundary conditions frozen",
            "value": 1.0 if boundary_conditions_frozen else 0.0,
            "note": (
                "The pilot freezes regularity at r = 0 and square-integrable decay as r -> infinity, without importing any extra reflective cavity or charge-stabilized branch."
                if boundary_conditions_frozen
                else "The pilot boundary conditions are not yet frozen."
            ),
        },
        {
            "row_id": "bound_state_problem_well_posed",
            "status": "pass" if bound_state_problem_well_posed else "reject",
            "metric": "bound-state pilot is mathematically well posed",
            "value": 1.0 if bound_state_problem_well_posed else 0.0,
            "note": (
                "Given the frozen linearized operator and the minimal whole-space regularity conditions, the pilot bound-state test is well posed even though it does not yet generate a discrete ladder."
                if bound_state_problem_well_posed
                else "The bound-state pilot is not yet mathematically well posed."
            ),
        },
        {
            "row_id": "discrete_spectrum_found",
            "status": "pass" if discrete_spectrum_found else "reject",
            "metric": "discrete bound-state spectrum found in the linearized pilot",
            "value": 1.0 if discrete_spectrum_found else 0.0,
            "note": (
                f"The linearized pilot already yields {pilot_mode_count} discrete normalizable mode(s)."
                if discrete_spectrum_found
                else "No nontrivial discrete normalizable mode appears in the linearized mexican-hat pilot; the current branch exposes only the massive continuum threshold."
            ),
        },
        {
            "row_id": "pilot_mode_count",
            "status": "inventory" if discrete_spectrum_found else "watch",
            "metric": "pilot discrete mode count",
            "value": float(pilot_mode_count),
            "note": (
                "Count of discrete normalizable modes found in the current linearized pilot."
                if discrete_spectrum_found
                else "The current linearized pilot produces zero discrete modes because no confining cavity, nonlinear self-binding branch, or charge-stabilized complex-field route has been promoted."
            ),
        },
    ]


# 関数: metrics payload 全体を構成する。

def _build_payload(step_tag: str) -> Dict[str, Any]:
    for path in (SHAPE_GATE_JSON, PARAM_JSON, SAME_SECTOR_GATE_JSON, SHELL_CANON_JSON, MASS_NOTE_MD):
        _require_path(path)

    shape_gate = _read_json(SHAPE_GATE_JSON)
    params = _read_json(PARAM_JSON)
    same_sector_gate = _read_json(SAME_SECTOR_GATE_JSON)
    shell_canon = _read_json(SHELL_CANON_JSON)
    mass_note_text = _read_text(MASS_NOTE_MD)

    shape_gate_summary = shape_gate.get("summary", {})
    params_summary = params.get("summary", {})
    same_sector_summary = same_sector_gate.get("summary", {})
    shell_rows = shell_canon.get("rows", [])

    if not isinstance(shell_rows, list):
        raise SystemExit(f"[fail] invalid rows in {SHELL_CANON_JSON}")

    shell_family_row = _find_row_by_id(shell_rows, "shell_quantization_family_public_candidate")
    shell_kappa_row = _find_row_by_id(shell_rows, "shell_quantization_fit_kappa")
    shell_kz_row = _find_row_by_id(shell_rows, "shell_quantization_fit_kz_over_kn")

    handoff_ready = bool(shape_gate_summary.get("hand_off_to_8_7_55_2_83", False))
    selected_family = str(params_summary.get("selected_candidate_family_id", ""))
    linearized_mode_equation_frozen = handoff_ready and selected_family == "mexican_hat"
    boundary_conditions_frozen = linearized_mode_equation_frozen and bool(
        same_sector_summary.get("single_public_boundary_family_fixed", False)
    )
    bound_state_problem_well_posed = linearized_mode_equation_frozen and boundary_conditions_frozen
    discrete_spectrum_found = False
    pilot_mode_count = 0
    lowest_mode_frequency_available = discrete_spectrum_found

    rows = _build_rows(
        linearized_mode_equation_frozen=linearized_mode_equation_frozen,
        boundary_conditions_frozen=boundary_conditions_frozen,
        bound_state_problem_well_posed=bound_state_problem_well_posed,
        discrete_spectrum_found=discrete_spectrum_found,
        pilot_mode_count=pilot_mode_count,
    )

    no_linear_localization_line = _find_first_match(
        mass_note_text,
        "上の線形真空方程式（質量なしのスカラー波）だけでは、3次元で一般に局在が保たれず、初期局在は分散して広がる。",
    )
    oscillon_line = _find_first_match(mass_note_text, "oscillon/Q-ball")
    boundary_quantization_line = _find_first_match(mass_note_text, "境界条件による離散化")

    return {
        "generated_utc": _utc_now_iso(),
        "phase": {
            "phase": 8,
            "step": step_tag,
            "name": "mass-eigenmode boundary specification and discrete-spectrum pilot",
        },
        "inputs": {
            "mass_origin_anchor_local_shape_gate_basis_closure_refresh_json": _relative_str(SHAPE_GATE_JSON),
            "mass_origin_mexican_hat_parameter_freeze_json": _relative_str(PARAM_JSON),
            "mass_origin_same_sector_vpp_shape_gate_json": _relative_str(SAME_SECTOR_GATE_JSON),
            "mass_origin_shell_quantization_canonicalization_json": _relative_str(SHELL_CANON_JSON),
            "mass_origin_note_markdown": _relative_str(MASS_NOTE_MD),
        },
        "intent": "Freeze the linearized mexican-hat mass-eigenmode pilot and decide whether the current no-new-free-parameter branch already yields a discrete bound-state spectrum.",
        "formulas": {
            "selected_potential": "V(rho) = (lambda / 4) (rho^2 - v^2)^2",
            "linearized_field_expansion": "rho(x,t) = rho_* + eta(x,t), with rho_* = v and |eta| << rho_*",
            "linearized_time_domain_equation": "(Box + m_P^2) eta = 0",
            "mass_parameter_formula": "m_P^2 = 2 lambda v^2 / Z_P",
            "mode_ansatz": "eta(t,r,Omega) = exp(-i omega t) Y_{ell m}(Omega) u_ell(r) / r",
            "radial_pilot_equation": "-u_ell''(r) + ell(ell+1) u_ell(r) / r^2 + (m_P c / hbar)^2 u_ell(r) = (omega^2 / c^2) u_ell(r)",
            "pilot_boundary_conditions": "u_ell(0) = 0 and u_ell(r -> infinity) = 0",
            "continuum_threshold_frequency": "omega_gap = m_P c^2 / hbar",
            "nonclosure_rule": "Without an added confining cavity, nonlinear self-binding branch, or charge-stabilized complex-field route, the linearized mexican-hat pilot exposes a massive threshold but no discrete normalizable ladder.",
        },
        "rows": rows,
        "summary": {
            "selected_candidate_family_id": selected_family,
            "linearized_mode_equation_frozen": linearized_mode_equation_frozen,
            "boundary_conditions_frozen": boundary_conditions_frozen,
            "bound_state_problem_well_posed": bound_state_problem_well_posed,
            "discrete_spectrum_found": discrete_spectrum_found,
            "pilot_mode_count": pilot_mode_count,
            "lowest_mode_frequency_available": lowest_mode_frequency_available,
            "continuum_threshold_frequency_formula": "omega_gap = m_P c^2 / hbar",
            "bound_state_nonclosure_reason_or_none": "linearized_mexican_hat_without_confining_channel",
            "surviving_public_boundary_family": same_sector_summary.get("single_public_boundary_family_fixed", False)
            and "boundary_shell_quantization"
            or None,
            "shell_quantization_kappa": shell_kappa_row.get("value"),
            "shell_quantization_kz_over_kn": shell_kz_row.get("value"),
        },
        "decision": {
            "overall_status": "mass_eigenmode_boundary_spec_frozen_no_discrete_spectrum",
            "keep_mass_origin_branch_blocked": True,
            "linearized_mode_equation_frozen": linearized_mode_equation_frozen,
            "boundary_conditions_frozen": boundary_conditions_frozen,
            "bound_state_problem_well_posed": bound_state_problem_well_posed,
            "discrete_spectrum_found": discrete_spectrum_found,
            "pilot_mode_count": pilot_mode_count,
            "lowest_mode_frequency_available": lowest_mode_frequency_available,
            "proceed_to_mass_ratio_pilot": False,
            "hand_off_to_8_7_55_2_84": False,
            "next_required_artifacts": [
                "postlinearized_binding_channel",
                "discrete_spectrum_reopen_refresh",
            ],
        },
        "evidence": {
            "anchor_local_shape_gate_basis_closure_refresh_summary": shape_gate_summary,
            "mexican_hat_parameter_freeze_summary": params_summary,
            "same_sector_vpp_shape_gate_summary": same_sector_summary,
            "shell_quantization_family_row": shell_family_row,
            "shell_quantization_kappa_row": shell_kappa_row,
            "shell_quantization_kz_over_kn_row": shell_kz_row,
            "mass_origin_note_no_linear_localization_line": no_linear_localization_line,
            "mass_origin_note_oscillon_qball_line": oscillon_line,
            "mass_origin_note_boundary_quantization_line": boundary_quantization_line,
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
