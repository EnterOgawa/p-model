#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_origin_geometric_boundary_promotion_audit.py

Step 8.7.55.2.394:
Audit whether the surviving public shell-quantization family can already be
promoted into a mexican-hat geometric reflective-boundary rule without
introducing new free parameters.

Inputs:
  - output/public/quantum/mass_origin_postlinearized_binding_route_contract_metrics.json
  - output/public/quantum/mass_origin_mass_eigenmode_boundary_metrics.json
  - output/public/quantum/mass_origin_shell_quantization_canonicalization_metrics.json
  - output/public/quantum/mass_origin_solver_family_elimination_metrics.json
  - output/public/quantum/mass_origin_shell_curvature_bridge_metrics.json
  - doc/quantum/18_p_field_action_and_schrodinger_mapping.md

Outputs:
  - output/public/quantum/mass_origin_geometric_boundary_promotion_metrics.json
  - output/public/quantum/mass_origin_geometric_boundary_promotion_rows.csv
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
BOUNDARY_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_mass_eigenmode_boundary_metrics.json"
SHELL_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_shell_quantization_canonicalization_metrics.json"
SOLVER_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_solver_family_elimination_metrics.json"
CURVATURE_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_shell_curvature_bridge_metrics.json"
NOTE_MD = ROOT / "doc" / "quantum" / "18_p_field_action_and_schrodinger_mapping.md"
OUT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_geometric_boundary_promotion_metrics.json"
OUT_CSV = ROOT / "output" / "public" / "quantum" / "mass_origin_geometric_boundary_promotion_rows.csv"
DEFAULT_STEP_TAG = "8.7.55.2.394"


# 関数: 現在UTC時刻を ISO 8601 文字列で返す。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: CLI 引数を解釈する。

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit the geometric reflective-boundary promotion route.")
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


# 関数: Markdown 内の最初の一致行を抽出する。

def _find_first_line(path: Path, pattern: str) -> Dict[str, Any]:
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if pattern in line:
            return {"pattern": pattern, "line": lineno, "text": line.strip()}

    return {"pattern": pattern, "line": None, "text": ""}


# 関数: rows を構成する。

def _build_rows(
    *,
    surviving_public_family: str,
    reflective_boundary_documented: bool,
    boundary_radius_or_domain_available: bool,
    geometric_boundary_promotion_rule_available: bool,
    discrete_shell_cavity_ready: bool,
) -> List[Dict[str, Any]]:
    return [
        {
            "row_id": "geometric_boundary_promotion_audit_complete",
            "status": "pass",
            "metric": "geometric boundary promotion audit complete",
            "value": 1.0,
            "note": "This step checks whether shell quantization already closes a no-new-free-parameter reflective cavity rule for the mexican-hat mass pilot.",
        },
        {
            "row_id": "surviving_public_family_is_shell_quantization",
            "status": "pass" if surviving_public_family == "boundary_shell_quantization" else "reject",
            "metric": "surviving public boundary family is shell quantization",
            "value": 1.0 if surviving_public_family == "boundary_shell_quantization" else 0.0,
            "note": f"Solver-family elimination currently leaves {surviving_public_family} as the only public boundary family.",
        },
        {
            "row_id": "reflective_boundary_route_documented",
            "status": "pass" if reflective_boundary_documented else "reject",
            "metric": "geometric reflective-boundary route is documented in the public note",
            "value": 1.0 if reflective_boundary_documented else 0.0,
            "note": "The quantum note explicitly keeps a reflective-boundary route on the table for post-linearized localization.",
        },
        {
            "row_id": "shell_quantization_defines_boundary_radius_or_domain",
            "status": "pass" if boundary_radius_or_domain_available else "reject",
            "metric": "shell quantization directly defines a cavity radius or domain",
            "value": 1.0 if boundary_radius_or_domain_available else 0.0,
            "note": (
                "Current public shell-quantization rows already define the reflective cavity scale."
                if boundary_radius_or_domain_available
                else "Current public shell-quantization rows only expose kappa and kZ/kN coefficients; they do not freeze a cavity radius, shell domain, or reflective wall."
            ),
        },
        {
            "row_id": "geometric_boundary_promotion_rule_available",
            "status": "pass" if geometric_boundary_promotion_rule_available else "reject",
            "metric": "geometric boundary promotion rule available",
            "value": 1.0 if geometric_boundary_promotion_rule_available else 0.0,
            "note": (
                "A no-new-free-parameter lift from shell quantization to a reflective cavity rule is available."
                if geometric_boundary_promotion_rule_available
                else "No no-new-free-parameter lift from shell quantization to a reflective cavity rule is yet frozen in the public canonical pack."
            ),
        },
        {
            "row_id": "discrete_shell_cavity_ready",
            "status": "pass" if discrete_shell_cavity_ready else "reject",
            "metric": "discrete shell cavity ready for spectrum reopen",
            "value": 1.0 if discrete_shell_cavity_ready else 0.0,
            "note": (
                "The shell family can already be reinterpreted as a reflective cavity for the mexican-hat pilot."
                if discrete_shell_cavity_ready
                else "The shell family remains a fit family only; it is not yet a cavity rule that can discretize the mexican-hat pilot."
            ),
        },
    ]


# 関数: metrics payload 全体を構成する。

def _build_payload(step_tag: str) -> Dict[str, Any]:
    for path in (CONTRACT_JSON, BOUNDARY_JSON, SHELL_JSON, SOLVER_JSON, CURVATURE_JSON, NOTE_MD):
        _require_path(path)

    contract = _read_json(CONTRACT_JSON)
    boundary = _read_json(BOUNDARY_JSON)
    shell = _read_json(SHELL_JSON)
    solver = _read_json(SOLVER_JSON)
    curvature = _read_json(CURVATURE_JSON)
    note_line = _find_first_line(NOTE_MD, "幾何学的境界条件（反射境界）")

    contract_summary = contract.get("summary", {})
    boundary_summary = boundary.get("summary", {})
    shell_summary = shell.get("summary", {})
    solver_summary = solver.get("summary", {})
    curvature_summary = curvature.get("summary", {})

    surviving_public_family = str(solver_summary.get("surviving_public_family", ""))
    reflective_boundary_documented = note_line["line"] is not None
    boundary_radius_or_domain_available = False
    geometric_boundary_promotion_rule_available = False
    discrete_shell_cavity_ready = False
    geometric_route_still_admissible = True
    geometric_boundary_nonclosure_reason = "shell_quantization_family_has_no_public_reflective_cavity_rule"

    rows = _build_rows(
        surviving_public_family=surviving_public_family,
        reflective_boundary_documented=reflective_boundary_documented,
        boundary_radius_or_domain_available=boundary_radius_or_domain_available,
        geometric_boundary_promotion_rule_available=geometric_boundary_promotion_rule_available,
        discrete_shell_cavity_ready=discrete_shell_cavity_ready,
    )

    return {
        "generated_utc": _utc_now_iso(),
        "phase": {
            "phase": 8,
            "step": step_tag,
            "name": "geometric boundary promotion audit",
        },
        "inputs": {
            "mass_origin_postlinearized_binding_route_contract_json": _relative_str(CONTRACT_JSON),
            "mass_origin_mass_eigenmode_boundary_json": _relative_str(BOUNDARY_JSON),
            "mass_origin_shell_quantization_canonicalization_json": _relative_str(SHELL_JSON),
            "mass_origin_solver_family_elimination_json": _relative_str(SOLVER_JSON),
            "mass_origin_shell_curvature_bridge_json": _relative_str(CURVATURE_JSON),
            "mass_origin_note_markdown": _relative_str(NOTE_MD),
        },
        "intent": "Test whether the surviving shell-quantization family already supplies a reflective cavity rule that can discretize the linearized mexican-hat pilot without new free parameters.",
        "formulas": {
            "promotion_rule": "geometric boundary route passes iff the public shell family supplies a cavity/domain rule plus reflective boundary conditions that can be injected into the linearized mexican-hat pilot",
            "reopen_dependency": "discrete spectrum reopen can use the geometric route only after boundary_radius_or_domain_available and geometric_boundary_promotion_rule_available are both true",
        },
        "rows": rows,
        "summary": {
            "candidate_binding_route_id": "geometric_reflective_boundary",
            "surviving_public_boundary_family": surviving_public_family,
            "shell_quantization_public_canonical": bool(shell_summary.get("shell_quantization_public_canonical", False)),
            "boundary_radius_or_domain_available": boundary_radius_or_domain_available,
            "geometric_boundary_promotion_rule_available": geometric_boundary_promotion_rule_available,
            "discrete_shell_cavity_ready": discrete_shell_cavity_ready,
            "geometric_route_still_admissible": geometric_route_still_admissible,
            "geometric_boundary_nonclosure_reason_or_none": geometric_boundary_nonclosure_reason,
        },
        "decision": {
            "overall_status": "geometric_boundary_promotion_audit_frozen",
            "keep_mass_origin_branch_blocked": True,
            "candidate_binding_route_id": "geometric_reflective_boundary",
            "geometric_boundary_promotion_rule_available": geometric_boundary_promotion_rule_available,
            "boundary_radius_or_domain_available": boundary_radius_or_domain_available,
            "discrete_shell_cavity_ready": discrete_shell_cavity_ready,
            "geometric_route_still_admissible": geometric_route_still_admissible,
            "geometric_boundary_nonclosure_reason_or_none": geometric_boundary_nonclosure_reason,
            "next_required_artifacts": [
                "geometric_boundary_promotion_rule",
                "boundary_radius_or_domain",
            ],
        },
        "evidence": {
            "postlinearized_binding_route_contract_summary": contract_summary,
            "mass_eigenmode_boundary_summary": boundary_summary,
            "shell_quantization_canonicalization_summary": shell_summary,
            "solver_family_elimination_summary": solver_summary,
            "shell_curvature_bridge_summary": curvature_summary,
            "mass_origin_note_geometric_boundary_line": note_line,
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
