#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_origin_nonlinear_self_binding_audit.py

Step 8.7.55.2.395:
Audit whether the real-scalar mexican-hat self-interaction alone already
provides a no-new-free-parameter post-linearized binding channel.

Inputs:
  - output/public/quantum/mass_origin_postlinearized_binding_route_contract_metrics.json
  - output/public/quantum/mass_origin_mass_eigenmode_boundary_metrics.json
  - output/public/quantum/mass_origin_mexican_hat_parameter_freeze_metrics.json
  - doc/quantum/18_p_field_action_and_schrodinger_mapping.md

Outputs:
  - output/public/quantum/mass_origin_nonlinear_self_binding_metrics.json
  - output/public/quantum/mass_origin_nonlinear_self_binding_rows.csv
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
PARAM_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_mexican_hat_parameter_freeze_metrics.json"
NOTE_MD = ROOT / "doc" / "quantum" / "18_p_field_action_and_schrodinger_mapping.md"
OUT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_nonlinear_self_binding_metrics.json"
OUT_CSV = ROOT / "output" / "public" / "quantum" / "mass_origin_nonlinear_self_binding_rows.csv"
DEFAULT_STEP_TAG = "8.7.55.2.395"


# 関数: 現在UTC時刻を ISO 8601 文字列で返す。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: CLI 引数を解釈する。

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit the nonlinear self-binding route for the mexican-hat branch.")
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
    selected_candidate_family_id: str,
    self_interaction_route_documented: bool,
    small_amplitude_nontrivial_bound_state_available: bool,
    nonlinear_self_binding_admissible: bool,
    needs_new_scale_or_charge: bool,
) -> List[Dict[str, Any]]:
    return [
        {
            "row_id": "nonlinear_self_binding_audit_complete",
            "status": "pass",
            "metric": "nonlinear self-binding audit complete",
            "value": 1.0,
            "note": "This step checks whether the real-scalar mexican-hat branch already closes a self-bound discrete ladder without adding new free parameters.",
        },
        {
            "row_id": "nonlinear_self_binding_selected_family_is_mexican_hat",
            "status": "pass" if selected_candidate_family_id == "mexican_hat" else "reject",
            "metric": "selected local-shape family is mexican hat",
            "value": 1.0 if selected_candidate_family_id == "mexican_hat" else 0.0,
            "note": f"The current basis-closure branch selected {selected_candidate_family_id}.",
        },
        {
            "row_id": "nonlinear_self_binding_route_documented",
            "status": "pass" if self_interaction_route_documented else "reject",
            "metric": "self-interaction route is documented in the quantum note",
            "value": 1.0 if self_interaction_route_documented else 0.0,
            "note": "The note keeps oscillon/Q-ball-like localization as a post-linearized possibility.",
        },
        {
            "row_id": "small_amplitude_nontrivial_bound_state_available",
            "status": "pass" if small_amplitude_nontrivial_bound_state_available else "reject",
            "metric": "small-amplitude nontrivial self-bound state available",
            "value": 1.0 if small_amplitude_nontrivial_bound_state_available else 0.0,
            "note": (
                "A small-amplitude nonlinear bound-state branch is already available in public canonical form."
                if small_amplitude_nontrivial_bound_state_available
                else "No small-amplitude nonlinear self-bound solution is frozen as a public canonical artifact for the real-scalar mexican-hat branch."
            ),
        },
        {
            "row_id": "nonlinear_self_binding_admissible",
            "status": "pass" if nonlinear_self_binding_admissible else "reject",
            "metric": "nonlinear self-binding admissible without new free parameters",
            "value": 1.0 if nonlinear_self_binding_admissible else 0.0,
            "note": (
                "The nonlinear self-binding route is admissible with no new scales or charges."
                if nonlinear_self_binding_admissible
                else "The real-scalar mexican-hat route does not yet supply a public, no-new-free-parameter self-binding mechanism strong enough to reopen the discrete-spectrum pilot."
            ),
        },
        {
            "row_id": "nonlinear_self_binding_needs_new_scale_or_charge",
            "status": "watch" if needs_new_scale_or_charge else "pass",
            "metric": "nonlinear route needs an extra scale or conserved charge",
            "value": 1.0 if needs_new_scale_or_charge else 0.0,
            "note": (
                "The current note-level route still points toward an extra stabilization ingredient beyond the frozen real-scalar mexican-hat pack."
                if needs_new_scale_or_charge
                else "No extra scale or conserved charge is needed."
            ),
        },
    ]


# 関数: metrics payload 全体を構成する。

def _build_payload(step_tag: str) -> Dict[str, Any]:
    for path in (CONTRACT_JSON, BOUNDARY_JSON, PARAM_JSON, NOTE_MD):
        _require_path(path)

    contract = _read_json(CONTRACT_JSON)
    boundary = _read_json(BOUNDARY_JSON)
    params = _read_json(PARAM_JSON)
    note_line = _find_first_line(NOTE_MD, "oscillon/Q-ball")

    contract_summary = contract.get("summary", {})
    boundary_summary = boundary.get("summary", {})
    param_summary = params.get("summary", {})

    selected_candidate_family_id = str(boundary_summary.get("selected_candidate_family_id", param_summary.get("selected_candidate_family_id", "")))
    self_interaction_route_documented = note_line["line"] is not None
    small_amplitude_nontrivial_bound_state_available = False
    nonlinear_self_binding_admissible = False
    needs_new_scale_or_charge = True
    nonlinear_route_still_admissible = False
    nonlinear_self_binding_nonclosure_reason = "real_scalar_mexican_hat_self_binding_not_publicly_stabilized"

    rows = _build_rows(
        selected_candidate_family_id=selected_candidate_family_id,
        self_interaction_route_documented=self_interaction_route_documented,
        small_amplitude_nontrivial_bound_state_available=small_amplitude_nontrivial_bound_state_available,
        nonlinear_self_binding_admissible=nonlinear_self_binding_admissible,
        needs_new_scale_or_charge=needs_new_scale_or_charge,
    )

    return {
        "generated_utc": _utc_now_iso(),
        "phase": {
            "phase": 8,
            "step": step_tag,
            "name": "nonlinear self-binding admissibility audit",
        },
        "inputs": {
            "mass_origin_postlinearized_binding_route_contract_json": _relative_str(CONTRACT_JSON),
            "mass_origin_mass_eigenmode_boundary_json": _relative_str(BOUNDARY_JSON),
            "mass_origin_mexican_hat_parameter_freeze_json": _relative_str(PARAM_JSON),
            "mass_origin_note_markdown": _relative_str(NOTE_MD),
        },
        "intent": "Decide whether the frozen real-scalar mexican-hat branch already provides a no-new-free-parameter nonlinear self-binding channel.",
        "formulas": {
            "route_rule": "nonlinear self-binding passes iff the real-scalar mexican-hat branch supplies a public stable self-bound solution family without importing a new scale, charge, or field content",
            "current_nonlinear_equation": "(Box + m_P^2) eta = -3 lambda v eta^2 - lambda eta^3",
        },
        "rows": rows,
        "summary": {
            "candidate_binding_route_id": "nonlinear_self_binding",
            "selected_candidate_family_id": selected_candidate_family_id,
            "nonlinear_self_binding_admissible": nonlinear_self_binding_admissible,
            "small_amplitude_nontrivial_bound_state_available": small_amplitude_nontrivial_bound_state_available,
            "needs_new_scale_or_charge": needs_new_scale_or_charge,
            "nonlinear_route_still_admissible": nonlinear_route_still_admissible,
            "nonlinear_self_binding_nonclosure_reason_or_none": nonlinear_self_binding_nonclosure_reason,
        },
        "decision": {
            "overall_status": "nonlinear_self_binding_audit_frozen",
            "keep_mass_origin_branch_blocked": True,
            "candidate_binding_route_id": "nonlinear_self_binding",
            "nonlinear_self_binding_admissible": nonlinear_self_binding_admissible,
            "small_amplitude_nontrivial_bound_state_available": small_amplitude_nontrivial_bound_state_available,
            "needs_new_scale_or_charge": needs_new_scale_or_charge,
            "nonlinear_route_still_admissible": nonlinear_route_still_admissible,
            "nonlinear_self_binding_nonclosure_reason_or_none": nonlinear_self_binding_nonclosure_reason,
            "next_required_artifacts": [
                "nonlinear_self_binding_admissibility",
            ],
        },
        "evidence": {
            "postlinearized_binding_route_contract_summary": contract_summary,
            "mass_eigenmode_boundary_summary": boundary_summary,
            "mexican_hat_parameter_freeze_summary": param_summary,
            "mass_origin_note_self_interaction_line": note_line,
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
