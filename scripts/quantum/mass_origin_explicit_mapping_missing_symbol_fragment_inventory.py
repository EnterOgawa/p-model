#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_origin_explicit_mapping_missing_symbol_fragment_inventory.py

Step 8.7.55.2.238:
Inventory fragment contexts and terminal atom candidates for the
explicit-mapping missing symbol fragments that remain after the current
residual symbol-fragment split.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]

PRIOR_RETRY_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_explicit_mapping_missing_terminal_glyph_closure_retry_metrics.json"
SPLIT_CONTRACT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_missing_symbol_fragment_residual_split_contract_metrics.json"
OUT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_explicit_mapping_missing_symbol_fragment_inventory_metrics.json"
OUT_CSV = ROOT / "output" / "public" / "quantum" / "mass_origin_explicit_mapping_missing_symbol_fragment_inventory_rows.csv"
DEFAULT_STEP_TAG = "8.7.55.2.238"


# 関数: `_utc_now_iso` の入出力契約と処理意図を定義する。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_parse_args` の入出力契約と処理意図を定義する。

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inventory fragment contexts and terminal atoms for explicit-mapping missing symbol fragments.",
    )
    parser.add_argument("--step-tag", default=DEFAULT_STEP_TAG, help="Roadmap step tag to stamp into the output payload.")
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


# 関数: `_fragment_plan` の入出力契約と処理意図を定義する。

def _fragment_plan() -> Dict[str, Dict[str, List[str] | str]]:
    return {
        "explicit_mapping_equation_symbol_fragment": {
            "present_contexts": [
                "lhs_observable_context",
                "rhs_curvature_context",
                "reference_point_context",
            ],
            "missing_terminal_atoms": ["explicit_mapping_equation_terminal_atom"],
            "note": "The observable, curvature, and reference-point contexts remain public, but the terminal atom that prints the explicit mapping equation is absent.",
        },
        "same_sector_equivalence_symbol_fragment": {
            "present_contexts": [
                "lhs_same_sector_context",
            ],
            "missing_terminal_atoms": ["same_sector_equivalence_terminal_atom"],
            "note": "The same-sector context remains public, but the terminal atom that states equivalence is absent.",
        },
        "mapping_operator_or_relation_symbol_fragment": {
            "present_contexts": [
                "lhs_observable_context",
                "rhs_curvature_context",
            ],
            "missing_terminal_atoms": ["mapping_operator_or_relation_terminal_atom"],
            "note": "The left-hand and right-hand contexts remain public, but the terminal atom that fixes the mapping operator / relation is absent.",
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

    required_fragments = [str(item) for item in split_contract_summary.get("explicit_mapping_missing_symbol_fragments", [])]
    route_ok = bool(split_contract_summary.get("explicit_mapping_missing_symbol_fragment_route_still_admissible", False))
    prior_ready = bool(prior_retry_summary.get("explicit_mapping_missing_terminal_glyph_inventory_ready", False))
    fragment_plan = _fragment_plan()
    present_by_fragment = {
        fragment_id: list(fragment_plan.get(fragment_id, {}).get("present_contexts", []))
        for fragment_id in required_fragments
    }
    missing_by_fragment = {
        fragment_id: list(fragment_plan.get(fragment_id, {}).get("missing_terminal_atoms", []))
        for fragment_id in required_fragments
    }
    present_contexts = _unique_preserve_order(
        [
            context_id
            for fragment_id in required_fragments
            for context_id in present_by_fragment.get(fragment_id, [])
        ]
    )
    missing_atoms = _unique_preserve_order(
        [
            atom_id
            for fragment_id in required_fragments
            for atom_id in missing_by_fragment.get(fragment_id, [])
        ]
    )
    inventory_ready = bool(route_ok and prior_ready and set(required_fragments).issubset(set(fragment_plan.keys())))

    rows: List[Dict[str, Any]] = [
        {
            "row_id": "explicit_mapping_missing_symbol_fragment_inventory_complete",
            "status": "pass",
            "metric": "explicit-mapping missing-symbol-fragment inventory complete",
            "value": 1.0,
            "fragment_id": "aggregate",
            "atom_id": "aggregate",
            "note": "This step freezes present fragment contexts and missing terminal atoms for the explicit-mapping symbol fragments that remain absent.",
        },
        {
            "row_id": "explicit_mapping_missing_symbol_fragment_route_admissible",
            "status": "pass" if route_ok else "reject",
            "metric": "explicit-mapping missing-symbol-fragment route remains admissible",
            "value": 1.0 if route_ok else 0.0,
            "fragment_id": "aggregate",
            "atom_id": "aggregate",
            "note": (
                "The explicit-mapping missing-symbol-fragment route remains admissible after the residual symbol-fragment split."
                if route_ok
                else "The explicit-mapping missing-symbol-fragment route is no longer admissible, so fragment inventory cannot support closure retry."
            ),
        },
        {
            "row_id": "explicit_mapping_missing_symbol_fragment_inventory_ready",
            "status": "pass" if inventory_ready else "reject",
            "metric": "explicit-mapping missing-symbol-fragment inventory ready for closure retry",
            "value": 1.0 if inventory_ready else 0.0,
            "fragment_id": "aggregate",
            "atom_id": "aggregate",
            "note": (
                "Each missing explicit-mapping symbol fragment now has an explicit present-context / missing-terminal-atom decomposition."
                if inventory_ready
                else "The current public canonical pack still cannot provide a stable atom decomposition for every explicit-mapping missing symbol fragment."
            ),
        },
        {
            "row_id": "explicit_mapping_missing_symbol_fragment_context_count",
            "status": "inventory",
            "metric": "present explicit-mapping missing-symbol-fragment context count",
            "value": float(len(present_contexts)),
            "fragment_id": "aggregate",
            "atom_id": "aggregate",
            "note": f"Present explicit-mapping missing-symbol-fragment contexts are {present_contexts}.",
        },
        {
            "row_id": "explicit_mapping_missing_symbol_fragment_atom_count",
            "status": "inventory",
            "metric": "missing explicit-mapping terminal atom count",
            "value": float(len(missing_atoms)),
            "fragment_id": "aggregate",
            "atom_id": "aggregate",
            "note": f"Missing explicit-mapping terminal atoms are {missing_atoms}.",
        },
    ]

    for fragment_id in required_fragments:
        fragment_contexts = present_by_fragment.get(fragment_id, [])
        fragment_atoms = missing_by_fragment.get(fragment_id, [])
        fragment_note = str(fragment_plan.get(fragment_id, {}).get("note", ""))
        rows.append(
            {
                "row_id": f"explicit_mapping_missing_symbol_fragment_{fragment_id}",
                "status": "watch" if fragment_atoms else "pass",
                "metric": f"atom decomposition for explicit-mapping missing symbol fragment {fragment_id}",
                "value": float(len(fragment_contexts)),
                "fragment_id": fragment_id,
                "atom_id": "aggregate",
                "note": f"Present contexts: {fragment_contexts}. Missing terminal atoms: {fragment_atoms}. {fragment_note}",
            }
        )

        for context_id in fragment_contexts:
            rows.append(
                {
                    "row_id": f"explicit_mapping_missing_symbol_fragment_present_context_{fragment_id}_{context_id}",
                    "status": "inventory",
                    "metric": f"present context candidate for explicit-mapping missing symbol fragment {fragment_id}",
                    "value": 1.0,
                    "fragment_id": fragment_id,
                    "atom_id": context_id,
                    "note": f"{context_id} is already present in the current public canonical pack as context for {fragment_id}.",
                }
            )

        for atom_id in fragment_atoms:
            rows.append(
                {
                    "row_id": f"explicit_mapping_missing_symbol_fragment_missing_atom_{fragment_id}_{atom_id}",
                    "status": "watch",
                    "metric": f"missing terminal atom for explicit-mapping symbol fragment {fragment_id}",
                    "value": 0.0,
                    "fragment_id": fragment_id,
                    "atom_id": atom_id,
                    "note": f"{atom_id} is still absent from the current public canonical pack for {fragment_id}.",
                }
            )

    return {
        "generated_utc": _utc_now_iso(),
        "phase": {
            "phase": 8,
            "step": step_tag,
            "name": "explicit-mapping missing-symbol-fragment inventory",
        },
        "inputs": {
            "mass_origin_explicit_mapping_missing_terminal_glyph_closure_retry_json": _relative_str(PRIOR_RETRY_JSON),
            "mass_origin_missing_symbol_fragment_residual_split_contract_json": _relative_str(SPLIT_CONTRACT_JSON),
        },
        "rows": rows,
        "summary": {
            "required_explicit_mapping_missing_symbol_fragments": required_fragments,
            "present_explicit_mapping_missing_fragment_contexts": present_contexts,
            "missing_explicit_mapping_missing_terminal_atoms": missing_atoms,
            "present_explicit_mapping_missing_fragment_contexts_by_fragment": present_by_fragment,
            "missing_explicit_mapping_missing_terminal_atoms_by_fragment": missing_by_fragment,
            "explicit_mapping_missing_symbol_fragment_route_still_admissible": route_ok,
            "prior_explicit_mapping_missing_symbol_fragment_inventory_ready": prior_ready,
            "explicit_mapping_missing_symbol_fragment_inventory_ready": inventory_ready,
            "prior_explicit_mapping_missing_terminal_glyph_nonclosure_reason_or_none": prior_retry_summary.get(
                "explicit_mapping_missing_terminal_glyph_nonclosure_reason_or_none"
            ),
        },
        "decision": {
            "overall_status": "explicit_mapping_missing_symbol_fragment_inventory_frozen",
            "keep_mass_origin_branch_blocked": True,
            "required_explicit_mapping_missing_symbol_fragments": required_fragments,
            "present_explicit_mapping_missing_fragment_contexts": present_contexts,
            "missing_explicit_mapping_missing_terminal_atoms": missing_atoms,
            "explicit_mapping_missing_symbol_fragment_inventory_ready": inventory_ready,
        },
    }


# 関数: `_write_csv` の入出力契約と処理意図を定義する。

def _write_csv(rows: List[Dict[str, Any]]) -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["row_id", "status", "metric", "value", "fragment_id", "atom_id", "note"])
        writer.writeheader()
        writer.writerows(rows)


# 関数: `main` の入出力契約と処理意図を定義する。

def main() -> None:
    args = _parse_args()
    payload = _build_payload(step_tag=str(args.step_tag))
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_csv(payload["rows"])
    print(f"[ok] wrote {OUT_JSON}")
    print(f"[ok] wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
