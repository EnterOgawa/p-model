#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_origin_postlinearized_binding_route_contract.py

Step 8.7.55.2.393:
Freeze the next branch contract after the linearized mexican-hat pilot shows a
well-posed mass-eigenmode problem but no discrete spectrum.

Inputs:
  - output/public/quantum/mass_origin_mass_eigenmode_boundary_metrics.json
  - output/public/quantum/mass_origin_solver_family_elimination_metrics.json
  - doc/quantum/18_p_field_action_and_schrodinger_mapping.md

Outputs:
  - output/public/quantum/mass_origin_postlinearized_binding_route_contract_metrics.json
  - output/public/quantum/mass_origin_postlinearized_binding_route_contract_rows.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]

BOUNDARY_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_mass_eigenmode_boundary_metrics.json"
ELIMINATION_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_solver_family_elimination_metrics.json"
MASS_NOTE_MD = ROOT / "doc" / "quantum" / "18_p_field_action_and_schrodinger_mapping.md"
OUT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_postlinearized_binding_route_contract_metrics.json"
OUT_CSV = ROOT / "output" / "public" / "quantum" / "mass_origin_postlinearized_binding_route_contract_rows.csv"
DEFAULT_STEP_TAG = "8.7.55.2.393"


# 関数: 現在UTC時刻を ISO 8601 文字列で返す。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: CLI 引数を解釈する。

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze the post-linearized binding-route contract after the linearized mexican-hat pilot.",
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

def _build_rows(*, required_route_items: List[str], split_contract_ready: bool) -> List[Dict[str, Any]]:
    return [
        {
            "row_id": "postlinearized_binding_route_contract_complete",
            "status": "pass",
            "metric": "post-linearized binding-route contract complete",
            "value": 1.0,
            "note": "This step freezes the next route after the linearized mexican-hat pilot proves well posed but non-discrete.",
        },
        {
            "row_id": "postlinearized_binding_route_contract_required_items",
            "status": "watch",
            "metric": "required route items for post-linearized binding branch",
            "value": float(len(required_route_items)),
            "note": f"Required route items: {required_route_items}.",
        },
        {
            "row_id": "postlinearized_binding_route_contract_split_ready",
            "status": "pass" if split_contract_ready else "reject",
            "metric": "post-linearized binding residual split contract ready",
            "value": 1.0 if split_contract_ready else 0.0,
            "note": (
                "The next branch can now audit whether a no-new-free-parameter binding channel exists beyond the linearized mexican-hat pilot."
                if split_contract_ready
                else "The post-linearized binding branch is not yet formalized."
            ),
        },
    ]


# 関数: metrics payload 全体を構成する。

def _build_payload(step_tag: str) -> Dict[str, Any]:
    for path in (BOUNDARY_JSON, ELIMINATION_JSON, MASS_NOTE_MD):
        _require_path(path)

    boundary = _read_json(BOUNDARY_JSON)
    elimination = _read_json(ELIMINATION_JSON)
    mass_note_text = _read_text(MASS_NOTE_MD)

    boundary_summary = boundary.get("summary", {})
    elimination_summary = elimination.get("summary", {})

    required_route_items = [
        "geometric_boundary_promotion_rule",
        "nonlinear_self_binding_admissibility",
        "complex_field_u1_stabilization_artifact",
        "no_new_free_parameter_binding_channel",
        "discrete_spectrum_reopen_refresh",
    ]
    split_contract_ready = True
    rows = _build_rows(required_route_items=required_route_items, split_contract_ready=split_contract_ready)

    boundary_line = _find_first_match(mass_note_text, "幾何学的境界条件（反射境界）")
    self_interaction_line = _find_first_match(mass_note_text, "有効質量項／自己相互作用")
    complex_field_line = _find_first_match(mass_note_text, "複素場（位相）への拡張")

    return {
        "generated_utc": _utc_now_iso(),
        "phase": {"phase": 8, "step": step_tag, "name": "post-linearized binding route contract freeze"},
        "inputs": {
            "mass_origin_mass_eigenmode_boundary_json": _relative_str(BOUNDARY_JSON),
            "mass_origin_solver_family_elimination_json": _relative_str(ELIMINATION_JSON),
            "mass_origin_note_markdown": _relative_str(MASS_NOTE_MD),
        },
        "intent": "Freeze the next route after the linearized mexican-hat pilot shows that the branch is well posed but still lacks a discrete bound-state ladder.",
        "formulas": {
            "linearized_nonclosure_rule": "linearized mexican hat on the whole-space pilot yields a massive threshold but no discrete ladder",
            "reopen_rule": "8.7.55.2.84 can activate only after a later refresh produces discrete_spectrum_found = true",
            "candidate_binding_route_ids": [
                "geometric_reflective_boundary",
                "nonlinear_self_binding",
                "complex_field_u1_stabilization",
            ],
        },
        "rows": rows,
        "summary": {
            "missing_postlinearized_binding_artifact": "discrete_binding_channel_beyond_linearized_mexican_hat",
            "candidate_binding_route_ids": [
                "geometric_reflective_boundary",
                "nonlinear_self_binding",
                "complex_field_u1_stabilization",
            ],
            "split_contract_ready": split_contract_ready,
        },
        "decision": {
            "overall_status": "postlinearized_binding_route_contract_frozen",
            "keep_mass_origin_branch_blocked": True,
            "missing_postlinearized_binding_artifact": "discrete_binding_channel_beyond_linearized_mexican_hat",
            "split_contract_ready": split_contract_ready,
            "hand_off_to_8_7_55_2_84": False,
            "next_required_artifacts": required_route_items,
        },
        "evidence": {
            "mass_eigenmode_boundary_summary": boundary_summary,
            "solver_family_elimination_summary": elimination_summary,
            "mass_origin_note_geometric_boundary_line": boundary_line,
            "mass_origin_note_self_interaction_line": self_interaction_line,
            "mass_origin_note_complex_field_line": complex_field_line,
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
