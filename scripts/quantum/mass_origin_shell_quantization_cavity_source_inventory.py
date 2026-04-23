#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_origin_shell_quantization_cavity_source_inventory.py

Step 8.7.55.2.400:
Inventory the current public-canonical source candidates that could promote
the surviving shell-quantization family into a reflective cavity rule for the
post-linearized mexican-hat mass pilot.

Inputs:
  - output/public/quantum/mass_origin_geometric_boundary_residual_route_contract_metrics.json
  - output/public/quantum/mass_origin_geometric_boundary_promotion_metrics.json
  - output/public/quantum/mass_origin_shell_quantization_canonicalization_metrics.json
  - output/public/quantum/mass_origin_mass_eigenmode_boundary_metrics.json
  - doc/quantum/18_p_field_action_and_schrodinger_mapping.md

Outputs:
  - output/public/quantum/mass_origin_shell_quantization_cavity_source_inventory_metrics.json
  - output/public/quantum/mass_origin_shell_quantization_cavity_source_inventory_rows.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]

ROUTE_CONTRACT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_geometric_boundary_residual_route_contract_metrics.json"
GEOMETRIC_PROMOTION_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_geometric_boundary_promotion_metrics.json"
SHELL_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_shell_quantization_canonicalization_metrics.json"
BOUNDARY_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_mass_eigenmode_boundary_metrics.json"
NOTE_MD = ROOT / "doc" / "quantum" / "18_p_field_action_and_schrodinger_mapping.md"
OUT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_shell_quantization_cavity_source_inventory_metrics.json"
OUT_CSV = ROOT / "output" / "public" / "quantum" / "mass_origin_shell_quantization_cavity_source_inventory_rows.csv"
DEFAULT_STEP_TAG = "8.7.55.2.400"


# 関数: 現在UTC時刻を ISO 8601 文字列で返す。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: CLI 引数を解釈する。

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inventory shell-quantization source candidates for a reflective cavity rule.",
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


# 関数: Markdown 内の最初の一致行を抽出する。

def _find_first_line(path: Path, pattern: str) -> Dict[str, Any]:
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if pattern in line:
            return {"pattern": pattern, "line": lineno, "text": line.strip()}

    return {"pattern": pattern, "line": None, "text": ""}


# 関数: rows を構成する。

def _build_rows(
    *,
    required_sources: List[str],
    present_sources: List[str],
    missing_sources: List[str],
    first_route_to_close_or_none: str | None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = [
        {
            "row_id": "shell_quantization_cavity_source_inventory_complete",
            "status": "pass",
            "metric": "shell-quantization cavity source inventory complete",
            "value": 1.0,
            "note": "This step inventories the public-canonical source candidates that could promote shell quantization into a reflective cavity rule.",
        },
        {
            "row_id": "shell_quantization_cavity_required_source_count",
            "status": "pass",
            "metric": "required source count for shell-quantization cavity route",
            "value": float(len(required_sources)),
            "note": f"Required sources: {required_sources}.",
        },
        {
            "row_id": "shell_quantization_cavity_present_source_count",
            "status": "pass",
            "metric": "present source count for shell-quantization cavity route",
            "value": float(len(present_sources)),
            "note": f"Present sources: {present_sources}.",
        },
        {
            "row_id": "shell_quantization_cavity_missing_source_count",
            "status": "watch" if missing_sources else "pass",
            "metric": "missing source count for shell-quantization cavity route",
            "value": float(len(missing_sources)),
            "note": f"Missing sources: {missing_sources}.",
        },
    ]

    for source_name in required_sources:
        is_present = source_name in present_sources
        rows.append(
            {
                "row_id": f"shell_quantization_cavity_source_{source_name}",
                "status": "pass" if is_present else "watch",
                "metric": f"source availability for {source_name}",
                "value": 1.0 if is_present else 0.0,
                "note": (
                    f"{source_name} is already public canonical."
                    if is_present
                    else f"{source_name} is still missing from the current public canonical pack."
                ),
            }
        )

    rows.append(
        {
            "row_id": "shell_quantization_cavity_first_route_to_close",
            "status": "watch" if first_route_to_close_or_none else "reject",
            "metric": "first residual source to close for shell-quantization cavity route",
            "value": 1.0 if first_route_to_close_or_none else 0.0,
            "note": (
                f"The first residual source to close is {first_route_to_close_or_none}."
                if first_route_to_close_or_none
                else "No residual source could be prioritized from the shell-quantization cavity inventory."
            ),
        }
    )
    return rows


# 関数: metrics payload 全体を構成する。

def _build_payload(step_tag: str) -> Dict[str, Any]:
    for path in (ROUTE_CONTRACT_JSON, GEOMETRIC_PROMOTION_JSON, SHELL_JSON, BOUNDARY_JSON, NOTE_MD):
        _require_path(path)

    route_contract = _read_json(ROUTE_CONTRACT_JSON)
    geometric_promotion = _read_json(GEOMETRIC_PROMOTION_JSON)
    shell = _read_json(SHELL_JSON)
    boundary = _read_json(BOUNDARY_JSON)
    note_boundary_quantization_line = _find_first_line(NOTE_MD, "境界条件による離散化")
    note_reflective_boundary_line = _find_first_line(NOTE_MD, "Dirichlet/Neumann")

    route_contract_summary = route_contract.get("summary", {})
    geometric_promotion_summary = geometric_promotion.get("summary", {})
    shell_summary = shell.get("summary", {})
    boundary_evidence = boundary.get("evidence", {})

    required_sources = [
        "shell_quantization_family_public_candidate",
        "shell_quantization_fit_kappa_row",
        "shell_quantization_fit_kz_over_kn_row",
        "boundary_condition_discretization_note",
        "reflective_boundary_operator_note",
        "boundary_radius_or_domain_proxy",
    ]
    present_sources: List[str] = []

    if bool(shell_summary.get("shell_quantization_public_canonical", False)):
        present_sources.append("shell_quantization_family_public_candidate")

    if "shell_quantization_kappa_row" in boundary_evidence:
        present_sources.append("shell_quantization_fit_kappa_row")

    if "shell_quantization_kz_over_kn_row" in boundary_evidence:
        present_sources.append("shell_quantization_fit_kz_over_kn_row")

    if note_boundary_quantization_line["line"] is not None:
        present_sources.append("boundary_condition_discretization_note")

    if note_reflective_boundary_line["line"] is not None:
        present_sources.append("reflective_boundary_operator_note")

    missing_sources = [item for item in required_sources if item not in present_sources]
    first_route_to_close_or_none = "boundary_radius_or_domain_proxy" if "boundary_radius_or_domain_proxy" in missing_sources else None

    rows = _build_rows(
        required_sources=required_sources,
        present_sources=present_sources,
        missing_sources=missing_sources,
        first_route_to_close_or_none=first_route_to_close_or_none,
    )

    return {
        "generated_utc": _utc_now_iso(),
        "phase": {
            "phase": 8,
            "step": step_tag,
            "name": "shell-quantization cavity source inventory",
        },
        "inputs": {
            "mass_origin_geometric_boundary_residual_route_contract_json": _relative_str(ROUTE_CONTRACT_JSON),
            "mass_origin_geometric_boundary_promotion_json": _relative_str(GEOMETRIC_PROMOTION_JSON),
            "mass_origin_shell_quantization_canonicalization_json": _relative_str(SHELL_JSON),
            "mass_origin_mass_eigenmode_boundary_json": _relative_str(BOUNDARY_JSON),
            "mass_origin_note_markdown": _relative_str(NOTE_MD),
        },
        "intent": "Inventory the current public-canonical source candidates that could promote the shell-quantization family into a reflective cavity rule for the mexican-hat post-linearized pilot.",
        "formulas": {
            "inventory_rule": "the shell-quantization cavity route can close only after the public pack contains shell-family rows, boundary-condition discretization wording, reflective-boundary wording, and a cavity radius or domain proxy that can be injected into the pilot without a new fit",
            "current_absence": "the current public pack already freezes shell-family coefficients and note-level boundary wording, but it still does not expose a cavity radius or domain proxy",
        },
        "rows": rows,
        "summary": {
            "required_cavity_route_sources": required_sources,
            "present_cavity_route_sources": present_sources,
            "missing_cavity_route_sources": missing_sources,
            "first_route_to_close_or_none": first_route_to_close_or_none,
            "cavity_source_inventory_ready": True,
        },
        "decision": {
            "overall_status": "shell_quantization_cavity_source_inventory_frozen",
            "keep_mass_origin_branch_blocked": True,
            "selected_residual_binding_route_or_none": route_contract_summary.get("selected_residual_binding_route_or_none"),
            "missing_geometric_boundary_artifact": route_contract_summary.get("missing_geometric_boundary_artifact"),
            "required_cavity_route_sources": required_sources,
            "present_cavity_route_sources": present_sources,
            "missing_cavity_route_sources": missing_sources,
            "first_route_to_close_or_none": first_route_to_close_or_none,
            "cavity_source_inventory_ready": True,
            "hand_off_to_8_7_55_2_84": False,
            "next_required_artifacts": [
                "boundary_radius_or_domain_proxy",
                "shell_quantization_reflective_cavity_rule",
                "discrete_spectrum_second_reopen_refresh",
            ],
        },
        "evidence": {
            "geometric_boundary_residual_route_contract_summary": route_contract_summary,
            "geometric_boundary_promotion_summary": geometric_promotion_summary,
            "shell_quantization_canonicalization_summary": shell_summary,
            "shell_quantization_family_row": boundary_evidence.get("shell_quantization_family_row", {}),
            "shell_quantization_kappa_row": boundary_evidence.get("shell_quantization_kappa_row", {}),
            "shell_quantization_kz_over_kn_row": boundary_evidence.get("shell_quantization_kz_over_kn_row", {}),
            "mass_origin_note_boundary_quantization_line": note_boundary_quantization_line,
            "mass_origin_note_reflective_boundary_line": note_reflective_boundary_line,
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
