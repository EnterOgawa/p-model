#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_origin_anchor_normalized_g3w_source_inventory.py

Step 8.7.55.2.248:
Inventory the required / present / missing public-canonical sources for the
preferred anchor-normalized g_3w route.

Inputs:
  - doc/paper/12_part3a_quantum_foundations.md
  - output/public/quantum/mass_origin_anchor_normalized_g3w_route_contract_metrics.json
  - output/public/quantum/mass_origin_anchor_local_shape_jet_metrics.json
  - output/public/quantum/entanglement_source_dynamics_three_wave_mixing_metrics.json

Outputs:
  - output/public/quantum/mass_origin_anchor_normalized_g3w_source_inventory_metrics.json
  - output/public/quantum/mass_origin_anchor_normalized_g3w_source_inventory_rows.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]

PART3A_MD = ROOT / "doc" / "paper" / "12_part3a_quantum_foundations.md"
ROUTE_CONTRACT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_anchor_normalized_g3w_route_contract_metrics.json"
SHAPE_JET_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_anchor_local_shape_jet_metrics.json"
THREE_WAVE_JSON = ROOT / "output" / "public" / "quantum" / "entanglement_source_dynamics_three_wave_mixing_metrics.json"
OUT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_anchor_normalized_g3w_source_inventory_metrics.json"
OUT_CSV = ROOT / "output" / "public" / "quantum" / "mass_origin_anchor_normalized_g3w_source_inventory_rows.csv"
DEFAULT_STEP_TAG = "8.7.55.2.248"


# 関数: `_utc_now_iso` の入出力契約と処理意図を定義する。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_parse_args` の入出力契約と処理意図を定義する。

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inventory the public sources for the anchor-normalized g_3w route.",
    )
    parser.add_argument(
        "--step-tag",
        default=DEFAULT_STEP_TAG,
        help="Roadmap step tag to stamp into the output payload.",
    )
    return parser.parse_args()


# 関数: `_require_path` の入出力契約と処理意図を定義する。

def _require_path(path: Path) -> None:
    # 条件分岐: `not path.exists()` を満たす経路を評価する。
    if not path.exists():
        raise SystemExit(f"[fail] missing required input: {path}")


# 関数: `_read_json` の入出力契約と処理意図を定義する。

def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


# 関数: `_read_text` の入出力契約と処理意図を定義する。

def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# 関数: `_relative_str` の入出力契約と処理意図を定義する。

def _relative_str(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


# 関数: `_find_first_match` の入出力契約と処理意図を定義する。

def _find_first_match(text: str, pattern: str) -> Dict[str, Any] | None:
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        # 条件分岐: `pattern in raw_line` を満たす経路を評価する。
        if pattern in raw_line:
            return {
                "pattern": pattern,
                "line": line_number,
                "text": raw_line.strip(),
            }

    return None


# 関数: `_build_rows` の入出力契約と処理意図を定義する。

def _build_rows(
    *,
    required_sources: List[str],
    present_sources: List[str],
    missing_sources: List[str],
    first_route_to_close_or_none: str | None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = [
        {
            "row_id": "anchor_normalized_g3w_source_inventory_complete",
            "status": "pass",
            "metric": "anchor-normalized g_3w source inventory complete",
            "value": 1.0,
            "note": "This step freezes the required, present, and missing public-canonical sources for the preferred g_3w route.",
        },
        {
            "row_id": "anchor_normalized_g3w_source_inventory_required_count",
            "status": "pass",
            "metric": "required source count for preferred g_3w route",
            "value": float(len(required_sources)),
            "note": f"Required sources: {required_sources}.",
        },
        {
            "row_id": "anchor_normalized_g3w_source_inventory_present_count",
            "status": "pass",
            "metric": "present source count for preferred g_3w route",
            "value": float(len(present_sources)),
            "note": f"Present sources: {present_sources}.",
        },
        {
            "row_id": "anchor_normalized_g3w_source_inventory_missing_count",
            "status": "watch" if missing_sources else "pass",
            "metric": "missing source count for preferred g_3w route",
            "value": float(len(missing_sources)),
            "note": f"Missing sources: {missing_sources}.",
        },
    ]

    for source_name in required_sources:
        is_present = source_name in present_sources
        rows.append(
            {
                "row_id": f"anchor_normalized_g3w_source_{source_name}",
                "status": "pass" if is_present else "watch",
                "metric": f"source availability for {source_name}",
                "value": 1.0 if is_present else 0.0,
                "note": (
                    f"{source_name} is already available in the current public canonical pack."
                    if is_present
                    else f"{source_name} is still missing from the current public canonical pack."
                ),
            }
        )

    rows.append(
        {
            "row_id": "anchor_normalized_g3w_source_inventory_first_route",
            "status": "watch" if first_route_to_close_or_none else "reject",
            "metric": "first residual route to audit after inventory freeze",
            "value": 1.0 if first_route_to_close_or_none else 0.0,
            "note": (
                f"The next closure attempt starts from {first_route_to_close_or_none}."
                if first_route_to_close_or_none
                else "No preferred first residual route can be chosen from the inventory."
            ),
        }
    )
    return rows


# 関数: `_build_payload` の入出力契約と処理意図を定義する。

def _build_payload(step_tag: str) -> Dict[str, Any]:
    for path in (PART3A_MD, ROUTE_CONTRACT_JSON, SHAPE_JET_JSON, THREE_WAVE_JSON):
        _require_path(path)

    part3a_text = _read_text(PART3A_MD)
    route_contract = _read_json(ROUTE_CONTRACT_JSON)
    shape_jet = _read_json(SHAPE_JET_JSON)
    three_wave = _read_json(THREE_WAVE_JSON)

    route_summary = route_contract.get("summary", {})
    shape_jet_summary = shape_jet.get("summary", {})
    three_wave_decision = three_wave.get("decision", {})

    g3w_formula_hit = _find_first_match(part3a_text, "g_{3\\mathrm{w}}\\equiv \\frac{1}{2}V_{*}^{(3)}")
    g3w_wording_hit = _find_first_match(part3a_text, "第1に、$g_{3\\mathrm{w}}$ は新しい P-model パラメータではなく")

    required_sources = [
        "public_g3w_formula",
        "no_new_free_parameter_wording",
        "anchor_curvature_identity",
        "anchor_local_r3_definition",
        "anchor_normalization_rule",
        "rho_star_elimination_rule",
        "public_anchor_normalized_g3w_value",
    ]

    present_sources: List[str] = []

    # 条件分岐: `route_summary.get("public_g3w_formula_available", False)` を満たす経路を評価する。
    if route_summary.get("public_g3w_formula_available", False) and g3w_formula_hit:
        present_sources.append("public_g3w_formula")

    # 条件分岐: `route_summary.get("g3w_not_new_parameter_wording_available", False)` を満たす経路を評価する。

    if route_summary.get("g3w_not_new_parameter_wording_available", False) and g3w_wording_hit:
        present_sources.append("no_new_free_parameter_wording")

    # 条件分岐: `shape_jet_summary.get("rho2_vpp_anchor_value")` を満たす経路を評価する。

    if shape_jet_summary.get("rho2_vpp_anchor_value"):
        present_sources.append("anchor_curvature_identity")

    # 条件分岐: `shape_jet_summary.get("vp_anchor_zero", False)` を満たす経路を評価する。

    if shape_jet_summary.get("vp_anchor_zero", False):
        present_sources.append("anchor_local_r3_definition")

    missing_sources = [item for item in required_sources if item not in present_sources]
    first_route_to_close_or_none = "anchor_normalization_rule"

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
            "name": "anchor-normalized g3w source inventory",
        },
        "inputs": {
            "part3a_quantum_foundations_markdown": _relative_str(PART3A_MD),
            "mass_origin_anchor_normalized_g3w_route_contract_json": _relative_str(ROUTE_CONTRACT_JSON),
            "mass_origin_anchor_local_shape_jet_json": _relative_str(SHAPE_JET_JSON),
            "entanglement_source_dynamics_three_wave_mixing_json": _relative_str(THREE_WAVE_JSON),
        },
        "intent": "Freeze the required / present / missing public-canonical sources for the preferred anchor-normalized g_3w route before the residual audits.",
        "formulas": {
            "route_definition": "R_3 = rho_* V'''(rho_*) / V''(rho_*) = 2 rho_* g_3w / V''(rho_*) = 2 rho_*^3 g_3w / (M_chi^2 omega_*^2)",
            "inventory_rule": "the preferred route stays open until public g_3w, no-new-parameter wording, anchor curvature identity, anchor normalization, rho_* elimination, and an anchor-normalized public value are all present",
            "first_route_rule": "anchor normalization is audited first because public g_3w and the anchor-local curvature identity are already present",
        },
        "rows": rows,
        "summary": {
            "required_g3w_route_sources": required_sources,
            "present_g3w_route_sources": present_sources,
            "missing_g3w_route_sources": missing_sources,
            "first_route_to_close_or_none": first_route_to_close_or_none,
            "anchor_normalization_can_be_audited": True,
            "rho_star_elimination_can_be_audited": True,
            "g3w_source_inventory_ready": True,
        },
        "decision": {
            "overall_status": "anchor_normalized_g3w_source_inventory_frozen",
            "keep_mass_origin_branch_blocked": True,
            "preferred_r3_route_or_none": route_summary.get("preferred_r3_route_or_none"),
            "required_g3w_route_sources": required_sources,
            "present_g3w_route_sources": present_sources,
            "missing_g3w_route_sources": missing_sources,
            "first_route_to_close_or_none": first_route_to_close_or_none,
            "route_admissible_without_new_free_parameters": bool(three_wave_decision.get("new_pmodel_free_parameters_introduced", False) is False),
            "g3w_source_inventory_ready": True,
            "hand_off_to_8_7_55_2_83": False,
            "next_required_artifacts": [
                "anchor_normalization_rule",
                "rho_star_elimination_rule",
                "anchor_normalized_g3w_public_value",
                "r3_target",
                "single_public_vpp_shape",
                "positive_particle_sector_chi_p_to_vpp_public_artifact",
                "solver_ready_row_promoted_to_pass",
            ],
        },
        "evidence": {
            "g3w_formula_line": g3w_formula_hit,
            "g3w_not_new_parameter_line": g3w_wording_hit,
            "route_contract_summary": route_summary,
            "shape_jet_summary": shape_jet_summary,
            "three_wave_decision": three_wave_decision,
        },
    }


# 関数: `_write_csv` の入出力契約と処理意図を定義する。

def _write_csv(rows: List[Dict[str, Any]]) -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["row_id", "status", "metric", "value", "note"])
        writer.writeheader()
        writer.writerows(rows)


# 関数: `main` の入出力契約と処理意図を定義する。

def main() -> None:
    args = _parse_args()
    payload = _build_payload(args.step_tag)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_csv(payload["rows"])
    print(f"[ok] wrote {OUT_JSON}")
    print(f"[ok] wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
