#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_origin_explicit_mapping_literal_fragment_inventory.py

Step 8.7.55.2.105:
Freeze the required / present / missing literal fragments for the explicit
same-sector chi_P -> V''(|P|_*) mapping route using only the current public
canonical pack.

Inputs:
  - output/public/quantum/mass_origin_explicit_mapping_equation_lift_contract_metrics.json
  - output/public/quantum/mass_origin_explicit_mapping_equation_lift_metrics.json
  - output/public/quantum/mass_origin_same_sector_chi_to_vpp_contract_metrics.json
  - output/public/quantum/mass_origin_positive_particle_sector_chi_to_vpp_metrics.json

Outputs:
  - output/public/quantum/mass_origin_explicit_mapping_literal_fragment_inventory_metrics.json
  - output/public/quantum/mass_origin_explicit_mapping_literal_fragment_inventory_rows.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]

LIFT_CONTRACT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_explicit_mapping_equation_lift_contract_metrics.json"
LIFT_AUDIT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_explicit_mapping_equation_lift_metrics.json"
CHI_CONTRACT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_same_sector_chi_to_vpp_contract_metrics.json"
PROMOTION_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_positive_particle_sector_chi_to_vpp_metrics.json"
OUT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_explicit_mapping_literal_fragment_inventory_metrics.json"
OUT_CSV = ROOT / "output" / "public" / "quantum" / "mass_origin_explicit_mapping_literal_fragment_inventory_rows.csv"
DEFAULT_STEP_TAG = "8.7.55.2.105"


# 関数: `_utc_now_iso` の入出力契約と処理意図を定義する。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_parse_args` の入出力契約と処理意図を定義する。

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inventory explicit mapping-equation literal fragments already present in public canonical form.",
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


# 関数: `_relative_str` の入出力契約と処理意図を定義する。

def _relative_str(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


# 関数: `_ordered_required_fragment_ids` の入出力契約と処理意図を定義する。

def _ordered_required_fragment_ids() -> List[str]:
    return [
        "same_particle_sector_only",
        "chi_P_symbol",
        "Vpp_absP_star_symbol",
        "absP_star_reference_point",
        "sign_statement",
        "unit_statement",
        "shell_quantization_fit_kappa_row",
        "shell_quantization_fit_kz_over_kn_row",
        "no_new_free_parameter_note",
        "source_kind_explicit_mapping_equation",
        "explicit_mapping_equation_literal",
        "same_sector_equivalence_statement",
        "mapping_operator_or_relation",
    ]


# 関数: `_fragment_notes` の入出力契約と処理意図を定義する。

def _fragment_notes(
    chi_contract_summary: Dict[str, Any],
    promotion_summary: Dict[str, Any],
) -> Dict[str, str]:
    existing_shell_family_row_ids = [str(item) for item in chi_contract_summary.get("existing_shell_family_row_ids", [])]
    return {
        "same_particle_sector_only": "The mapping route is frozen to the same particle sector only.",
        "chi_P_symbol": f"The observable symbol is frozen to {chi_contract_summary.get('chi_symbol')}.",
        "Vpp_absP_star_symbol": f"The curvature symbol is frozen to {chi_contract_summary.get('curvature_symbol')}.",
        "absP_star_reference_point": f"The reference point is frozen to {chi_contract_summary.get('reference_point_symbol')}.",
        "sign_statement": "Sign wording is already frozen in the same-sector contract.",
        "unit_statement": "Unit wording is already frozen in the same-sector contract.",
        "shell_quantization_fit_kappa_row": "The surviving shell-family anchor row shell_quantization_fit_kappa is already public canonical.",
        "shell_quantization_fit_kz_over_kn_row": "The surviving shell-family anchor row shell_quantization_fit_kz_over_kn is already public canonical.",
        "no_new_free_parameter_note": "No-new-free-parameter wording is already available for promotion reuse.",
        "source_kind_explicit_mapping_equation": "The explicit_mapping_equation source kind remains admissible and listed.",
        "explicit_mapping_equation_literal": "The explicit chi_P -> V''(|P|_*) equation row itself is still absent.",
        "same_sector_equivalence_statement": "A same-sector equivalence statement is still absent from the current public canonical pack.",
        "mapping_operator_or_relation": "A literal mapping operator / relation between chi_P and V''(|P|_*) is still absent from the current public canonical pack.",
        "shell_family_row_ids": f"Current shell-family row ids are {existing_shell_family_row_ids}.",
        "promotion_missing": f"Promotion still misses {promotion_summary.get('missing_promotion_requirements', [])}.",
    }


# 関数: `_build_fragment_presence` の入出力契約と処理意図を定義する。

def _build_fragment_presence(
    chi_contract_summary: Dict[str, Any],
    promotion_summary: Dict[str, Any],
    lift_contract_summary: Dict[str, Any],
    lift_audit_evidence: Dict[str, Any],
) -> Dict[str, bool]:
    field_presence = lift_audit_evidence.get("field_presence", {})
    equation_slot_presence = lift_audit_evidence.get("equation_slot_presence", {})

    return {
        "same_particle_sector_only": bool(chi_contract_summary.get("same_particle_sector_only", False)),
        "chi_P_symbol": str(chi_contract_summary.get("chi_symbol")) == "chi_P",
        "Vpp_absP_star_symbol": str(chi_contract_summary.get("curvature_symbol")) == "V''(|P|_*)",
        "absP_star_reference_point": str(chi_contract_summary.get("reference_point_symbol")) == "|P|_*",
        "sign_statement": bool(chi_contract_summary.get("sign_convention_frozen", False)),
        "unit_statement": bool(chi_contract_summary.get("unit_contract_frozen", False)),
        "shell_quantization_fit_kappa_row": "shell_quantization_fit_kappa" in [
            str(item) for item in chi_contract_summary.get("existing_shell_family_row_ids", [])
        ],
        "shell_quantization_fit_kz_over_kn_row": "shell_quantization_fit_kz_over_kn" in [
            str(item) for item in chi_contract_summary.get("existing_shell_family_row_ids", [])
        ],
        "no_new_free_parameter_note": bool(promotion_summary.get("no_new_free_parameter_note_present", False)),
        "source_kind_explicit_mapping_equation": bool(field_presence.get("source_kind_explicit_mapping_equation", False))
        and bool(lift_contract_summary.get("mapping_equation_source_kind_allowed", False)),
        "explicit_mapping_equation_literal": bool(field_presence.get("explicit_mapping_equation", False)),
        "same_sector_equivalence_statement": bool(equation_slot_presence.get("same_sector_equivalence_statement_or_none", False)),
        "mapping_operator_or_relation": bool(equation_slot_presence.get("mapping_operator_or_relation", False)),
    }


# 関数: `_build_payload` の入出力契約と処理意図を定義する。

def _build_payload(step_tag: str) -> Dict[str, Any]:
    for path in (
        LIFT_CONTRACT_JSON,
        LIFT_AUDIT_JSON,
        CHI_CONTRACT_JSON,
        PROMOTION_JSON,
    ):
        _require_path(path)

    lift_contract = _read_json(LIFT_CONTRACT_JSON)
    lift_audit = _read_json(LIFT_AUDIT_JSON)
    chi_contract = _read_json(CHI_CONTRACT_JSON)
    promotion = _read_json(PROMOTION_JSON)

    lift_contract_summary = lift_contract.get("summary", {})
    lift_audit_summary = lift_audit.get("summary", {})
    lift_audit_evidence = lift_audit.get("evidence", {})
    chi_contract_summary = chi_contract.get("summary", {})
    promotion_summary = promotion.get("summary", {})

    required_literal_fragments = _ordered_required_fragment_ids()
    fragment_presence = _build_fragment_presence(
        chi_contract_summary=chi_contract_summary,
        promotion_summary=promotion_summary,
        lift_contract_summary=lift_contract_summary,
        lift_audit_evidence=lift_audit_evidence,
    )
    notes = _fragment_notes(
        chi_contract_summary=chi_contract_summary,
        promotion_summary=promotion_summary,
    )
    present_literal_fragments = [
        fragment_id
        for fragment_id in required_literal_fragments
        if fragment_presence.get(fragment_id, False)
    ]
    missing_literal_fragments = [
        fragment_id
        for fragment_id in required_literal_fragments
        if not fragment_presence.get(fragment_id, False)
    ]
    literal_fragment_inventory_ready = bool(
        lift_contract_summary.get("mapping_equation_lift_ready", False)
        and len(required_literal_fragments) == len(present_literal_fragments) + len(missing_literal_fragments)
    )

    rows: List[Dict[str, Any]] = [
        {
            "row_id": "explicit_mapping_literal_fragment_inventory_complete",
            "status": "pass",
            "metric": "explicit mapping-equation literal fragment inventory complete",
            "value": 1.0,
            "note": "This step freezes which literal fragments are already public canonical and which remain absent before literal lift closure.",
        },
        {
            "row_id": "explicit_mapping_literal_fragment_required_count",
            "status": "inventory",
            "metric": "required literal fragment count",
            "value": float(len(required_literal_fragments)),
            "note": f"Required literal fragments are {required_literal_fragments}.",
        },
        {
            "row_id": "explicit_mapping_literal_fragment_present_count",
            "status": "inventory",
            "metric": "present literal fragment count",
            "value": float(len(present_literal_fragments)),
            "note": f"Present literal fragments are {present_literal_fragments}.",
        },
        {
            "row_id": "explicit_mapping_literal_fragment_missing_count",
            "status": "inventory",
            "metric": "missing literal fragment count",
            "value": float(len(missing_literal_fragments)),
            "note": f"Missing literal fragments are {missing_literal_fragments}.",
        },
    ]

    for fragment_id in required_literal_fragments:
        is_present = fragment_presence.get(fragment_id, False)
        rows.append(
            {
                "row_id": f"explicit_mapping_literal_fragment_{fragment_id}",
                "status": "pass" if is_present else "watch",
                "metric": f"literal fragment {fragment_id} present in current public canonical pack",
                "value": 1.0 if is_present else 0.0,
                "note": notes.get(fragment_id, ""),
            }
        )

    rows.append(
        {
            "row_id": "explicit_mapping_literal_fragment_inventory_ready",
            "status": "pass" if literal_fragment_inventory_ready else "reject",
            "metric": "explicit mapping literal fragment inventory ready for closure",
            "value": 1.0 if literal_fragment_inventory_ready else 0.0,
            "note": (
                "The next step may attempt literal lift closure using the frozen required/present/missing fragment split."
                if literal_fragment_inventory_ready
                else "The fragment inventory is not internally complete enough to support literal lift closure."
            ),
        }
    )

    return {
        "generated_utc": _utc_now_iso(),
        "phase": {
            "phase": 8,
            "step": step_tag,
            "name": "explicit mapping-equation literal fragment inventory",
        },
        "inputs": {
            "mass_origin_explicit_mapping_equation_lift_contract_json": _relative_str(LIFT_CONTRACT_JSON),
            "mass_origin_explicit_mapping_equation_lift_json": _relative_str(LIFT_AUDIT_JSON),
            "mass_origin_same_sector_chi_to_vpp_contract_json": _relative_str(CHI_CONTRACT_JSON),
            "mass_origin_positive_particle_sector_chi_to_vpp_json": _relative_str(PROMOTION_JSON),
        },
        "intent": "Freeze the literal fragments, equation-slot wording anchors, and missing pieces for the explicit same-sector mapping route.",
        "formulas": {
            "fragment_inventory_rule": "required_literal_fragments are partitioned into present_literal_fragments and missing_literal_fragments using only the current public canonical pack",
            "closure_readiness_rule": "literal_fragment_inventory_ready iff the lift contract is already frozen and every required literal fragment is explicitly inventoried as present or missing",
        },
        "rows": rows,
        "summary": {
            "required_literal_fragments": required_literal_fragments,
            "present_literal_fragments": present_literal_fragments,
            "missing_literal_fragments": missing_literal_fragments,
            "required_literal_fragment_count": len(required_literal_fragments),
            "present_literal_fragment_count": len(present_literal_fragments),
            "missing_literal_fragment_count": len(missing_literal_fragments),
            "literal_fragment_inventory_ready": literal_fragment_inventory_ready,
            "lift_missing_requirements": lift_audit_summary.get("missing_lift_requirements", []),
            "promotion_missing_requirements": promotion_summary.get("missing_promotion_requirements", []),
        },
        "decision": {
            "overall_status": "explicit_mapping_literal_fragment_inventory_frozen",
            "keep_mass_origin_branch_blocked": True,
            "required_literal_fragments": required_literal_fragments,
            "present_literal_fragments": present_literal_fragments,
            "missing_literal_fragments": missing_literal_fragments,
            "literal_fragment_inventory_ready": literal_fragment_inventory_ready,
        },
        "evidence": {
            "lift_contract_summary": lift_contract_summary,
            "lift_audit_summary": lift_audit_summary,
            "chi_contract_summary": chi_contract_summary,
            "promotion_summary": promotion_summary,
            "field_presence": lift_audit_evidence.get("field_presence", {}),
            "equation_slot_presence": lift_audit_evidence.get("equation_slot_presence", {}),
        },
    }


# 関数: `_write_csv` の入出力契約と処理意図を定義する。

def _write_csv(rows: List[Dict[str, Any]]) -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
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
