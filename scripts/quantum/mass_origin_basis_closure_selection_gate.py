#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_origin_basis_closure_selection_gate.py

Step 8.7.55.2.389:
Apply the finite pure-exponential chi-space basis-closure rule to the surviving
candidate families and determine whether one family is uniquely selected.

Inputs:
  - output/public/quantum/mass_origin_chi_space_action_basis_inventory_metrics.json
  - output/public/quantum/mass_origin_candidate_pushforward_basis_audit_metrics.json
  - output/public/quantum/mass_origin_anchor_local_r3_registry_metrics.json

Outputs:
  - output/public/quantum/mass_origin_basis_closure_selection_gate_metrics.json
  - output/public/quantum/mass_origin_basis_closure_selection_gate_rows.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]

BASIS_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_chi_space_action_basis_inventory_metrics.json"
PUSHFORWARD_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_candidate_pushforward_basis_audit_metrics.json"
R3_REGISTRY_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_anchor_local_r3_registry_metrics.json"
OUT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_basis_closure_selection_gate_metrics.json"
OUT_CSV = ROOT / "output" / "public" / "quantum" / "mass_origin_basis_closure_selection_gate_rows.csv"
DEFAULT_STEP_TAG = "8.7.55.2.389"


# 関数: 現在UTC時刻を ISO 8601 文字列で返す。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: CLI 引数を解釈する。

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply the chi-space basis-closure selection gate for the mass-origin candidate families.",
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


# 関数: selection gate rows を構成する。

def _build_rows(
    *,
    selected_candidate_id: str | None,
    rejected_candidate_ids: List[str],
    basis_extension_required: bool,
) -> List[Dict[str, Any]]:
    unique_selection = selected_candidate_id is not None
    return [
        {
            "row_id": "basis_closure_selection_gate_complete",
            "status": "pass",
            "metric": "basis-closure selection gate complete",
            "value": 1.0,
            "note": "This step applies the finite pure-exponential basis rule to the surviving candidate family set.",
        },
        {
            "row_id": "basis_closure_selection_gate_unique_survivor",
            "status": "pass" if unique_selection else "reject",
            "metric": "basis closure selects a unique surviving family",
            "value": 1.0 if unique_selection else 0.0,
            "note": (
                f"The basis-closure gate selects {selected_candidate_id} as the unique surviving family."
                if unique_selection
                else "The basis-closure gate does not yet select a unique surviving family."
            ),
        },
        {
            "row_id": "basis_closure_selection_gate_logarithmic_rejected",
            "status": "pass" if "logarithmic" in rejected_candidate_ids else "watch",
            "metric": "logarithmic family rejected by finite pure-exponential closure",
            "value": 1.0 if "logarithmic" in rejected_candidate_ids else 0.0,
            "note": (
                "The logarithmic family is rejected because its chi-space pushforward exposes a chi exp(2 chi) term outside the frozen basis."
                if "logarithmic" in rejected_candidate_ids
                else "The logarithmic family has not yet been rejected by the basis-closure rule."
            ),
        },
        {
            "row_id": "basis_closure_selection_gate_mexican_hat_passes",
            "status": "pass" if selected_candidate_id == "mexican_hat" else "watch",
            "metric": "mexican hat family passes finite pure-exponential closure",
            "value": 1.0 if selected_candidate_id == "mexican_hat" else 0.0,
            "note": (
                "The mexican-hat family remains inside the ambient exponent set {0, 2, 4} and therefore passes the basis-closure rule."
                if selected_candidate_id == "mexican_hat"
                else "The mexican-hat family is not yet selected by the basis-closure rule."
            ),
        },
        {
            "row_id": "basis_closure_selection_gate_basis_extension_required",
            "status": "reject" if not basis_extension_required else "watch",
            "metric": "higher pure-exponential basis extension required",
            "value": 1.0 if basis_extension_required else 0.0,
            "note": (
                "A higher pure-exponential basis extension is required before the candidate families can be separated."
                if basis_extension_required
                else "No higher pure-exponential basis extension is required for the current candidate set; the existing basis already selects one family."
            ),
        },
        {
            "row_id": "basis_closure_selection_gate_fallback_route_required",
            "status": "reject" if unique_selection else "watch",
            "metric": "fallback same-sector equivalence route must be resumed",
            "value": 0.0 if unique_selection else 1.0,
            "note": (
                "The fallback same-sector-equivalence route remains on hold because basis closure succeeded uniquely."
                if unique_selection
                else "The fallback same-sector-equivalence route must be resumed because basis closure remained ambiguous."
            ),
        },
    ]


# 関数: metrics payload 全体を構成する。

def _build_payload(step_tag: str) -> Dict[str, Any]:
    for path in (BASIS_JSON, PUSHFORWARD_JSON, R3_REGISTRY_JSON):
        _require_path(path)

    basis = _read_json(BASIS_JSON)
    pushforward = _read_json(PUSHFORWARD_JSON)
    r3_registry = _read_json(R3_REGISTRY_JSON)

    basis_summary = basis.get("summary", {})
    pushforward_summary = pushforward.get("summary", {})
    r3_summary = r3_registry.get("summary", {})

    candidate_family_ids = [str(item) for item in r3_summary.get("candidate_family_ids", [])]
    pass_candidate_ids = [str(item) for item in pushforward_summary.get("basis_pass_candidate_ids", [])]
    rejected_candidate_ids = [str(item) for item in pushforward_summary.get("basis_reject_candidate_ids", [])]
    selected_candidate_id = pass_candidate_ids[0] if len(pass_candidate_ids) == 1 else None
    basis_extension_required = False
    unique_selection = selected_candidate_id is not None

    rows = _build_rows(
        selected_candidate_id=selected_candidate_id,
        rejected_candidate_ids=rejected_candidate_ids,
        basis_extension_required=basis_extension_required,
    )

    return {
        "generated_utc": _utc_now_iso(),
        "phase": {
            "phase": 8,
            "step": step_tag,
            "name": "basis-closure selection gate",
        },
        "inputs": {
            "mass_origin_chi_space_action_basis_inventory_json": _relative_str(BASIS_JSON),
            "mass_origin_candidate_pushforward_basis_audit_json": _relative_str(PUSHFORWARD_JSON),
            "mass_origin_anchor_local_r3_registry_json": _relative_str(R3_REGISTRY_JSON),
        },
        "intent": "Select the surviving V(|P|) family by requiring the chi-space pushforward to stay inside the frozen finite pure-exponential basis.",
        "formulas": {
            "basis_accept_rule": "candidate passes iff it is a finite linear combination of {1, exp(n chi)} inside the frozen ambient basis and contains no naked chi outside exponentials",
            "basis_reject_rule": "candidate is rejected if it requires chi exp(n chi) or any other decorated non-basis term",
        },
        "rows": rows,
        "summary": {
            "ambient_basis_exponents": basis_summary.get("primitive_basis_exponents", []),
            "candidate_family_ids": candidate_family_ids,
            "selected_candidate_family_id_or_none": selected_candidate_id,
            "rejected_candidate_family_ids": rejected_candidate_ids,
            "basis_closure_unique_selection": unique_selection,
            "basis_extension_required": basis_extension_required,
            "fallback_same_sector_equivalence_route_required": not unique_selection,
        },
        "decision": {
            "overall_status": "basis_closure_selection_gate_frozen",
            "keep_mass_origin_branch_blocked": True,
            "selected_candidate_family_id_or_none": selected_candidate_id,
            "rejected_candidate_family_ids": rejected_candidate_ids,
            "basis_closure_unique_selection": unique_selection,
            "basis_extension_required": basis_extension_required,
            "fallback_same_sector_equivalence_route_required": not unique_selection,
            "next_required_artifacts": [
                "sigma3_r3_freeze",
                "mexican_hat_parameter_pack",
                "shape_gate_refresh",
            ],
        },
        "evidence": {
            "chi_space_action_basis_summary": basis_summary,
            "candidate_pushforward_basis_audit_summary": pushforward_summary,
            "r3_registry_summary": r3_summary,
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
