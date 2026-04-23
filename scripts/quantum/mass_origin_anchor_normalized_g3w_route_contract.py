#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_origin_anchor_normalized_g3w_route_contract.py

Step 8.7.55.2.247:
Freeze the new preferred route contract for closing the missing R_3 target via
an anchor-normalized public value of g_3w.

The prior anchor-local gate ends with a blocked handoff because the repository
still lacks a public canonical `anchor_normalized_g3w_public_value`. This step
freezes the minimum route contract for the new branch that will try to close
that datum without introducing a new free parameter.

Inputs:
  - doc/paper/12_part3a_quantum_foundations.md
  - output/public/quantum/mass_origin_r3_target_source_audit_metrics.json
  - output/public/quantum/mass_origin_anchor_local_shape_gate_metrics.json
  - output/public/quantum/entanglement_source_dynamics_three_wave_mixing_metrics.json

Outputs:
  - output/public/quantum/mass_origin_anchor_normalized_g3w_route_contract_metrics.json
  - output/public/quantum/mass_origin_anchor_normalized_g3w_route_contract_rows.csv
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
R3_ROUTE_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_r3_target_source_audit_metrics.json"
SHAPE_GATE_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_anchor_local_shape_gate_metrics.json"
THREE_WAVE_JSON = ROOT / "output" / "public" / "quantum" / "entanglement_source_dynamics_three_wave_mixing_metrics.json"
OUT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_anchor_normalized_g3w_route_contract_metrics.json"
OUT_CSV = ROOT / "output" / "public" / "quantum" / "mass_origin_anchor_normalized_g3w_route_contract_rows.csv"
DEFAULT_STEP_TAG = "8.7.55.2.247"


# 関数: `_utc_now_iso` の入出力契約と処理意図を定義する。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_parse_args` の入出力契約と処理意図を定義する。

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze the preferred anchor-normalized g_3w route contract for the missing R_3 target.",
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
    public_g3w_formula_available: bool,
    no_new_parameter_wording_available: bool,
    route_admissible: bool,
    anchor_normalization_rule_available: bool,
    rho_star_elimination_rule_available: bool,
    anchor_normalized_g3w_public_value_available: bool,
    preferred_r3_route_or_none: str | None,
) -> List[Dict[str, Any]]:
    return [
        {
            "row_id": "anchor_normalized_g3w_route_contract_complete",
            "status": "pass",
            "metric": "anchor-normalized g_3w route contract complete",
            "value": 1.0,
            "note": "This step freezes the minimum route contract for the preferred g_3w-based closure of R_3_target.",
        },
        {
            "row_id": "anchor_normalized_g3w_route_selected_as_preferred",
            "status": "pass" if preferred_r3_route_or_none == "g3w_anchor_normalized_route" else "reject",
            "metric": "g_3w route selected as the preferred R_3 target route",
            "value": 1.0 if preferred_r3_route_or_none == "g3w_anchor_normalized_route" else 0.0,
            "note": (
                "The R_3 target-source audit already selected the anchor-normalized g_3w route as the preferred next-closing route."
                if preferred_r3_route_or_none == "g3w_anchor_normalized_route"
                else "The preferred route is not currently the anchor-normalized g_3w route."
            ),
        },
        {
            "row_id": "anchor_normalized_g3w_public_formula_available",
            "status": "pass" if public_g3w_formula_available else "reject",
            "metric": "public g_3w formula already available",
            "value": 1.0 if public_g3w_formula_available else 0.0,
            "note": "Part III-A already exposes g_3w ≡ 1/2 V_*^(3)." if public_g3w_formula_available else "The public g_3w formula is not yet available.",
        },
        {
            "row_id": "anchor_normalized_g3w_not_new_parameter_wording_available",
            "status": "pass" if no_new_parameter_wording_available else "reject",
            "metric": "no-new-free-parameter wording for g_3w available",
            "value": 1.0 if no_new_parameter_wording_available else 0.0,
            "note": (
                "Part III-A already states that g_3w is not a new P-model parameter."
                if no_new_parameter_wording_available
                else "No public wording yet states that g_3w is not a new P-model parameter."
            ),
        },
        {
            "row_id": "anchor_normalized_g3w_route_admissible",
            "status": "pass" if route_admissible else "reject",
            "metric": "anchor-normalized g_3w route remains admissible",
            "value": 1.0 if route_admissible else 0.0,
            "note": (
                "The three-wave public artifact keeps the g_3w route inside the no-new-free-parameter envelope."
                if route_admissible
                else "The current public pack no longer supports a no-new-free-parameter g_3w route."
            ),
        },
        {
            "row_id": "anchor_normalized_g3w_anchor_normalization_rule_available",
            "status": "pass" if anchor_normalization_rule_available else "watch",
            "metric": "anchor normalization rule already public canonical",
            "value": 1.0 if anchor_normalization_rule_available else 0.0,
            "note": (
                "A public rule already normalizes g_3w at the anchor."
                if anchor_normalization_rule_available
                else "The preferred route still lacks a public canonical anchor normalization rule."
            ),
        },
        {
            "row_id": "anchor_normalized_g3w_rho_star_elimination_rule_available",
            "status": "pass" if rho_star_elimination_rule_available else "watch",
            "metric": "rho_* elimination rule already public canonical",
            "value": 1.0 if rho_star_elimination_rule_available else 0.0,
            "note": (
                "A public rule already eliminates rho_* from the g_3w route."
                if rho_star_elimination_rule_available
                else "The preferred route still lacks a public canonical rho_* elimination rule."
            ),
        },
        {
            "row_id": "anchor_normalized_g3w_public_value_available",
            "status": "pass" if anchor_normalized_g3w_public_value_available else "missing",
            "metric": "anchor-normalized public g_3w value available",
            "value": 1.0 if anchor_normalized_g3w_public_value_available else 0.0,
            "note": (
                "The route already carries a public anchor-normalized g_3w value."
                if anchor_normalized_g3w_public_value_available
                else "The preferred route is still blocked by the missing anchor_normalized_g3w_public_value."
            ),
        },
    ]


# 関数: `_build_payload` の入出力契約と処理意図を定義する。

def _build_payload(step_tag: str) -> Dict[str, Any]:
    for path in (PART3A_MD, R3_ROUTE_JSON, SHAPE_GATE_JSON, THREE_WAVE_JSON):
        _require_path(path)

    part3a_text = _read_text(PART3A_MD)
    r3_route = _read_json(R3_ROUTE_JSON)
    shape_gate = _read_json(SHAPE_GATE_JSON)
    three_wave = _read_json(THREE_WAVE_JSON)

    r3_route_summary = r3_route.get("summary", {})
    shape_gate_summary = shape_gate.get("summary", {})
    three_wave_decision = three_wave.get("decision", {})

    g3w_formula_hit = _find_first_match(part3a_text, "g_{3\\mathrm{w}}\\equiv \\frac{1}{2}V_{*}^{(3)}")
    g3w_not_new_parameter_hit = _find_first_match(part3a_text, "第1に、$g_{3\\mathrm{w}}$ は新しい P-model パラメータではなく")

    public_g3w_formula_available = bool(g3w_formula_hit)
    no_new_parameter_wording_available = bool(g3w_not_new_parameter_hit)
    route_admissible = bool(three_wave_decision.get("new_pmodel_free_parameters_introduced", False) is False)
    preferred_r3_route_or_none = r3_route_summary.get("preferred_r3_route_or_none")
    anchor_normalization_rule_available = False
    rho_star_elimination_rule_available = False
    anchor_normalized_g3w_public_value_available = False

    rows = _build_rows(
        public_g3w_formula_available=public_g3w_formula_available,
        no_new_parameter_wording_available=no_new_parameter_wording_available,
        route_admissible=route_admissible,
        anchor_normalization_rule_available=anchor_normalization_rule_available,
        rho_star_elimination_rule_available=rho_star_elimination_rule_available,
        anchor_normalized_g3w_public_value_available=anchor_normalized_g3w_public_value_available,
        preferred_r3_route_or_none=preferred_r3_route_or_none,
    )

    required_route_items = [
        "public_g3w_formula",
        "no_new_free_parameter_wording",
        "anchor_normalization_rule",
        "rho_star_elimination_rule",
        "public_anchor_normalized_g3w_value",
    ]

    return {
        "generated_utc": _utc_now_iso(),
        "phase": {
            "phase": 8,
            "step": step_tag,
            "name": "anchor-normalized g3w route contract freeze",
        },
        "inputs": {
            "part3a_quantum_foundations_markdown": _relative_str(PART3A_MD),
            "mass_origin_r3_target_source_audit_json": _relative_str(R3_ROUTE_JSON),
            "mass_origin_anchor_local_shape_gate_json": _relative_str(SHAPE_GATE_JSON),
            "entanglement_source_dynamics_three_wave_mixing_json": _relative_str(THREE_WAVE_JSON),
        },
        "intent": "Freeze the minimum contract for the preferred anchor-normalized g_3w route that could still supply the missing public canonical R_3 target.",
        "formulas": {
            "g3w_definition": "g_3w = 1/2 V_*^(3)",
            "route_goal": "R_3 = rho_* V'''(rho_*) / V''(rho_*) must be rewritten through an anchor-normalized public g_3w value with rho_* eliminated",
            "route_completion_rule": "anchor_normalized_g3w_public_value_available iff a public anchor normalization rule and a public rho_* elimination rule are both present",
        },
        "rows": rows,
        "summary": {
            "preferred_r3_route_or_none": preferred_r3_route_or_none,
            "remaining_r3_missing_datum_or_none": r3_route_summary.get("remaining_r3_missing_datum_or_none"),
            "required_route_items": required_route_items,
            "public_g3w_formula_available": public_g3w_formula_available,
            "g3w_not_new_parameter_wording_available": no_new_parameter_wording_available,
            "anchor_normalization_rule_available": anchor_normalization_rule_available,
            "rho_star_elimination_rule_available": rho_star_elimination_rule_available,
            "anchor_normalized_g3w_public_value_available": anchor_normalized_g3w_public_value_available,
            "shape_gate_handoff_ready": bool(shape_gate_summary.get("hand_off_to_8_7_55_2_83", False)),
            "split_contract_ready": True,
        },
        "decision": {
            "overall_status": "anchor_normalized_g3w_route_contract_frozen",
            "keep_mass_origin_branch_blocked": True,
            "preferred_r3_route_or_none": preferred_r3_route_or_none,
            "remaining_r3_missing_datum_or_none": r3_route_summary.get("remaining_r3_missing_datum_or_none"),
            "anchor_normalization_rule_available": anchor_normalization_rule_available,
            "rho_star_elimination_rule_available": rho_star_elimination_rule_available,
            "anchor_normalized_g3w_public_value_available": anchor_normalized_g3w_public_value_available,
            "split_contract_ready": True,
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
            "g3w_not_new_parameter_line": g3w_not_new_parameter_hit,
            "r3_route_summary": r3_route_summary,
            "shape_gate_summary": shape_gate_summary,
            "three_wave_decision": three_wave_decision,
        },
    }


# 関数: `_write_csv` の入出力契約と処理意図を定義する。

def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["row_id", "status", "metric", "value", "note"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


# 関数: `_write_json` の入出力契約と処理意図を定義する。

def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


# 関数: `main` の入出力契約と処理意図を定義する。

def main() -> None:
    args = _parse_args()
    payload = _build_payload(args.step_tag)
    _write_json(OUT_JSON, payload)
    _write_csv(OUT_CSV, payload["rows"])
    print(f"[ok] wrote {OUT_JSON}")
    print(f"[ok] wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
