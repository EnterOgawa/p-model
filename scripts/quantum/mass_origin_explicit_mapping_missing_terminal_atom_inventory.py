#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_origin_explicit_mapping_missing_terminal_atom_inventory.py

Step 8.7.55.2.226:
Inventory atom contexts and terminal glyph candidates for the explicit-mapping
missing terminal atoms that remain after the residual terminal-atom split.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]
PRIOR_RETRY_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_explicit_mapping_missing_symbol_fragment_closure_retry_metrics.json"
SPLIT_CONTRACT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_missing_terminal_atom_residual_split_contract_metrics.json"
OUT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_explicit_mapping_missing_terminal_atom_inventory_metrics.json"
OUT_CSV = ROOT / "output" / "public" / "quantum" / "mass_origin_explicit_mapping_missing_terminal_atom_inventory_rows.csv"
DEFAULT_STEP_TAG = "8.7.55.2.226"


# 関数: `_utc_now_iso` の入出力契約と処理意図を定義する。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_parse_args` の入出力契約と処理意図を定義する。

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inventory explicit-mapping missing terminal atoms.")
    parser.add_argument("--step-tag", default=DEFAULT_STEP_TAG)
    return parser.parse_args()


# 関数: `_require_path` の入出力契約と処理意図を定義する。

def _require_path(path: Path) -> None:
    if not path.exists():
        raise SystemExit(f"[fail] missing required input: {path}")


# 関数: `_read_json` の入出力契約と処理意図を定義する。

def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


# 関数: `_relative_str` の入出力契約と処理意図を定義する。

def _relative_str(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


# 関数: `_unique_preserve_order` の入出力契約と処理意図を定義する。

def _unique_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            ordered.append(item)

    return ordered


# 関数: `_atom_plan` の入出力契約と処理意図を定義する。

def _atom_plan() -> Dict[str, Dict[str, List[str] | str]]:
    return {
        "explicit_mapping_equation_terminal_atom": {
            "present_contexts": ["lhs_observable_context", "rhs_curvature_context", "reference_point_context"],
            "missing_glyphs": ["explicit_mapping_equation_terminal_glyph"],
            "note": "The observable, curvature, and reference-point contexts remain public, but the terminal equation glyph is absent.",
        },
        "same_sector_equivalence_terminal_atom": {
            "present_contexts": ["lhs_same_sector_context"],
            "missing_glyphs": ["same_sector_equivalence_terminal_glyph"],
            "note": "The same-sector context remains public, but the terminal equivalence glyph is absent.",
        },
        "mapping_operator_or_relation_terminal_atom": {
            "present_contexts": ["lhs_observable_context", "rhs_curvature_context"],
            "missing_glyphs": ["mapping_operator_or_relation_terminal_glyph"],
            "note": "The left-hand and right-hand contexts remain public, but the terminal operator / relation glyph is absent.",
        },
    }


# 関数: `_build_payload` の入出力契約と処理意図を定義する。

def _build_payload(step_tag: str) -> Dict[str, Any]:
    for path in (PRIOR_RETRY_JSON, SPLIT_CONTRACT_JSON):
        _require_path(path)

    prior_retry = _read_json(PRIOR_RETRY_JSON)
    split_contract = _read_json(SPLIT_CONTRACT_JSON)
    prior_retry_summary = prior_retry.get("summary", {})
    split_contract_summary = split_contract.get("summary", {})

    required_atoms = [str(item) for item in split_contract_summary.get("explicit_mapping_missing_terminal_atoms", [])]
    route_ok = bool(split_contract_summary.get("explicit_mapping_missing_terminal_atom_route_still_admissible", False))
    prior_retry_ok = bool(prior_retry_summary.get("explicit_mapping_missing_symbol_fragment_inventory_ready", False))

    atom_plan = _atom_plan()
    present_by_atom = {atom_id: list(atom_plan.get(atom_id, {}).get("present_contexts", [])) for atom_id in required_atoms}
    missing_by_atom = {atom_id: list(atom_plan.get(atom_id, {}).get("missing_glyphs", [])) for atom_id in required_atoms}
    present_contexts = _unique_preserve_order([context for atom_id in required_atoms for context in present_by_atom.get(atom_id, [])])
    missing_glyphs = _unique_preserve_order([glyph for atom_id in required_atoms for glyph in missing_by_atom.get(atom_id, [])])
    inventory_ready = bool(route_ok and prior_retry_ok and set(required_atoms).issubset(set(atom_plan.keys())))

    rows: List[Dict[str, Any]] = [
        {"row_id": "explicit_mapping_missing_terminal_atom_inventory_complete", "status": "pass", "metric": "explicit-mapping missing-terminal-atom inventory complete", "value": 1.0, "atom_id": "aggregate", "glyph_id": "aggregate", "note": "This step freezes present atom contexts and missing terminal glyphs for the explicit-mapping route."},
        {"row_id": "explicit_mapping_missing_terminal_atom_inventory_ready", "status": "pass" if inventory_ready else "reject", "metric": "explicit-mapping missing-terminal-atom inventory ready for closure retry", "value": 1.0 if inventory_ready else 0.0, "atom_id": "aggregate", "glyph_id": "aggregate", "note": "Each missing explicit-mapping terminal atom now has an explicit present-context / missing-terminal-glyph decomposition." if inventory_ready else "The current public canonical pack still cannot provide a stable terminal-glyph decomposition for every explicit-mapping terminal atom."},
        {"row_id": "explicit_mapping_missing_terminal_atom_glyph_count", "status": "inventory", "metric": "missing explicit-mapping terminal glyph count", "value": float(len(missing_glyphs)), "atom_id": "aggregate", "glyph_id": "aggregate", "note": f"Missing explicit-mapping terminal glyphs are {missing_glyphs}."},
    ]

    for atom_id in required_atoms:
        rows.append(
            {
                "row_id": f"explicit_mapping_missing_terminal_atom_{atom_id}",
                "status": "watch",
                "metric": f"terminal-glyph decomposition for explicit-mapping missing terminal atom {atom_id}",
                "value": float(len(present_by_atom.get(atom_id, []))),
                "atom_id": atom_id,
                "glyph_id": "aggregate",
                "note": (
                    f"Present contexts: {present_by_atom.get(atom_id, [])}. "
                    f"Missing terminal glyphs: {missing_by_atom.get(atom_id, [])}. "
                    f"{atom_plan.get(atom_id, {}).get('note', '')}"
                ),
            }
        )

    return {
        "generated_utc": _utc_now_iso(),
        "phase": {"phase": 8, "step": step_tag, "name": "explicit-mapping missing-terminal-atom inventory"},
        "inputs": {
            "mass_origin_explicit_mapping_missing_symbol_fragment_closure_retry_json": _relative_str(PRIOR_RETRY_JSON),
            "mass_origin_missing_terminal_atom_residual_split_contract_json": _relative_str(SPLIT_CONTRACT_JSON),
        },
        "rows": rows,
        "summary": {
            "required_explicit_mapping_missing_terminal_atoms": required_atoms,
            "present_explicit_mapping_missing_atom_contexts": present_contexts,
            "missing_explicit_mapping_missing_terminal_glyphs": missing_glyphs,
            "present_explicit_mapping_missing_atom_contexts_by_atom": present_by_atom,
            "missing_explicit_mapping_missing_terminal_glyphs_by_atom": missing_by_atom,
            "explicit_mapping_missing_terminal_atom_route_still_admissible": route_ok,
            "prior_explicit_mapping_missing_symbol_fragment_inventory_ready": prior_retry_ok,
            "explicit_mapping_missing_terminal_atom_inventory_ready": inventory_ready,
            "prior_explicit_mapping_missing_symbol_fragment_nonclosure_reason_or_none": prior_retry_summary.get("explicit_mapping_missing_symbol_fragment_nonclosure_reason_or_none"),
        },
        "decision": {
            "overall_status": "explicit_mapping_missing_terminal_atom_inventory_frozen",
            "keep_mass_origin_branch_blocked": True,
            "required_explicit_mapping_missing_terminal_atoms": required_atoms,
            "present_explicit_mapping_missing_atom_contexts": present_contexts,
            "missing_explicit_mapping_missing_terminal_glyphs": missing_glyphs,
            "explicit_mapping_missing_terminal_atom_inventory_ready": inventory_ready,
        },
    }


# 関数: `_write_csv` の入出力契約と処理意図を定義する。

def _write_csv(rows: List[Dict[str, Any]]) -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["row_id", "status", "metric", "value", "atom_id", "glyph_id", "note"])
        writer.writeheader()
        writer.writerows(rows)


# 関数: `main` の入出力契約と処理意図を定義する。

def main() -> None:
    args = _parse_args()
    payload = _build_payload(str(args.step_tag))
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_csv(payload["rows"])
    print(f"[ok] wrote {OUT_JSON}")
    print(f"[ok] wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
