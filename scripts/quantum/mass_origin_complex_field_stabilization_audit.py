#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_origin_complex_field_stabilization_audit.py

Step 8.7.55.2.396:
Audit whether the complex-field / U(1)-stabilized binding route can already be
promoted into a public canonical post-linearized artifact.

Inputs:
  - output/public/quantum/mass_origin_postlinearized_binding_route_contract_metrics.json
  - output/public/quantum/mass_origin_solver_family_elimination_metrics.json
  - doc/quantum/18_p_field_action_and_schrodinger_mapping.md

Outputs:
  - output/public/quantum/mass_origin_complex_field_stabilization_metrics.json
  - output/public/quantum/mass_origin_complex_field_stabilization_rows.csv
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
SOLVER_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_solver_family_elimination_metrics.json"
NOTE_MD = ROOT / "doc" / "quantum" / "18_p_field_action_and_schrodinger_mapping.md"
OUT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_complex_field_stabilization_metrics.json"
OUT_CSV = ROOT / "output" / "public" / "quantum" / "mass_origin_complex_field_stabilization_rows.csv"
DEFAULT_STEP_TAG = "8.7.55.2.396"


# 関数: 現在UTC時刻を ISO 8601 文字列で返す。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: CLI 引数を解釈する。

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit the complex-field / U(1) stabilization route.")
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


# 関数: 指定 row_id の row を抽出する。

def _find_row(rows: List[Dict[str, Any]], row_id: str) -> Dict[str, Any]:
    for row in rows:
        if str(row.get("row_id")) == row_id:
            return row

    return {}


# 関数: rows を構成する。

def _build_rows(
    *,
    complex_route_documented: bool,
    complex_field_u1_stabilization_artifact_available: bool,
    charge_stabilized_binding_ready: bool,
    doc_only_route_promotable: bool,
) -> List[Dict[str, Any]]:
    return [
        {
            "row_id": "complex_field_stabilization_audit_complete",
            "status": "pass",
            "metric": "complex-field stabilization audit complete",
            "value": 1.0,
            "note": "This step checks whether the note-only complex-field route can be promoted into a public canonical stabilization artifact.",
        },
        {
            "row_id": "complex_field_route_documented",
            "status": "pass" if complex_route_documented else "reject",
            "metric": "complex-field / U(1) route is documented in the note",
            "value": 1.0 if complex_route_documented else 0.0,
            "note": "The note explicitly records a complex-field / U(1)-stabilized Q-ball-like route.",
        },
        {
            "row_id": "complex_field_u1_stabilization_artifact_available",
            "status": "pass" if complex_field_u1_stabilization_artifact_available else "reject",
            "metric": "complex-field stabilization artifact available in public canonical form",
            "value": 1.0 if complex_field_u1_stabilization_artifact_available else 0.0,
            "note": (
                "A public complex-field stabilization artifact is available."
                if complex_field_u1_stabilization_artifact_available
                else "The complex-field stabilization route remains doc-only and has not been promoted into a public canonical artifact."
            ),
        },
        {
            "row_id": "charge_stabilized_binding_ready",
            "status": "pass" if charge_stabilized_binding_ready else "reject",
            "metric": "charge-stabilized binding ready",
            "value": 1.0 if charge_stabilized_binding_ready else 0.0,
            "note": (
                "A U(1)-protected charge-stabilized binding channel is ready to reopen the discrete-spectrum pilot."
                if charge_stabilized_binding_ready
                else "No U(1)-protected charge-stabilized binding channel is frozen in the public canonical pack."
            ),
        },
        {
            "row_id": "doc_only_route_promotable",
            "status": "pass" if doc_only_route_promotable else "reject",
            "metric": "doc-only complex-field route promotable without extending field content",
            "value": 1.0 if doc_only_route_promotable else 0.0,
            "note": (
                "The documented complex-field route is promotable without extending field content."
                if doc_only_route_promotable
                else "The route still requires a complex-field extension that is absent from the current public canonical action pack."
            ),
        },
    ]


# 関数: metrics payload 全体を構成する。

def _build_payload(step_tag: str) -> Dict[str, Any]:
    for path in (CONTRACT_JSON, SOLVER_JSON, NOTE_MD):
        _require_path(path)

    contract = _read_json(CONTRACT_JSON)
    solver = _read_json(SOLVER_JSON)
    note_line = _find_first_line(NOTE_MD, "複素場（位相）への拡張")

    contract_summary = contract.get("summary", {})
    solver_summary = solver.get("summary", {})
    complex_field_row = _find_row(solver.get("rows", []), "complex_field_family_elimination")

    complex_route_documented = note_line["line"] is not None
    complex_field_u1_stabilization_artifact_available = False
    charge_stabilized_binding_ready = False
    doc_only_route_promotable = False
    complex_route_still_admissible = False
    complex_field_route_nonclosure_reason = "complex_field_u1_route_doc_only"

    rows = _build_rows(
        complex_route_documented=complex_route_documented,
        complex_field_u1_stabilization_artifact_available=complex_field_u1_stabilization_artifact_available,
        charge_stabilized_binding_ready=charge_stabilized_binding_ready,
        doc_only_route_promotable=doc_only_route_promotable,
    )

    return {
        "generated_utc": _utc_now_iso(),
        "phase": {
            "phase": 8,
            "step": step_tag,
            "name": "complex-field / U(1) stabilization promotion audit",
        },
        "inputs": {
            "mass_origin_postlinearized_binding_route_contract_json": _relative_str(CONTRACT_JSON),
            "mass_origin_solver_family_elimination_json": _relative_str(SOLVER_JSON),
            "mass_origin_note_markdown": _relative_str(NOTE_MD),
        },
        "intent": "Decide whether the complex-field / U(1) stabilization route can be promoted from note-only status into the public canonical mass-origin branch.",
        "formulas": {
            "promotion_rule": "complex-field route passes iff a public canonical complex-field artifact plus a charge-stabilized binding rule are both available without reopening hidden field-content assumptions",
        },
        "rows": rows,
        "summary": {
            "candidate_binding_route_id": "complex_field_u1_stabilization",
            "complex_field_u1_stabilization_artifact_available": complex_field_u1_stabilization_artifact_available,
            "charge_stabilized_binding_ready": charge_stabilized_binding_ready,
            "doc_only_route_promotable": doc_only_route_promotable,
            "complex_route_still_admissible": complex_route_still_admissible,
            "complex_field_route_nonclosure_reason_or_none": complex_field_route_nonclosure_reason,
        },
        "decision": {
            "overall_status": "complex_field_stabilization_audit_frozen",
            "keep_mass_origin_branch_blocked": True,
            "candidate_binding_route_id": "complex_field_u1_stabilization",
            "complex_field_u1_stabilization_artifact_available": complex_field_u1_stabilization_artifact_available,
            "charge_stabilized_binding_ready": charge_stabilized_binding_ready,
            "doc_only_route_promotable": doc_only_route_promotable,
            "complex_route_still_admissible": complex_route_still_admissible,
            "complex_field_route_nonclosure_reason_or_none": complex_field_route_nonclosure_reason,
            "next_required_artifacts": [
                "complex_field_u1_stabilization_artifact",
            ],
        },
        "evidence": {
            "postlinearized_binding_route_contract_summary": contract_summary,
            "solver_family_elimination_summary": solver_summary,
            "solver_family_elimination_complex_field_row": complex_field_row,
            "mass_origin_note_complex_field_line": note_line,
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
