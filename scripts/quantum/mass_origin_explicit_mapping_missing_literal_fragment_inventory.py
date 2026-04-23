#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_origin_explicit_mapping_missing_literal_fragment_inventory.py

Step 8.7.55.2.130:
Inventory fragment contexts and terminal atoms for the explicit-mapping
missing literal fragments that remain after the residual-source-literal split.

Inputs:
  - output/public/quantum/mass_origin_explicit_mapping_missing_source_closure_retry_metrics.json
  - output/public/quantum/mass_origin_missing_source_literal_residual_split_contract_metrics.json

Outputs:
  - output/public/quantum/mass_origin_explicit_mapping_missing_literal_fragment_inventory_metrics.json
  - output/public/quantum/mass_origin_explicit_mapping_missing_literal_fragment_inventory_rows.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]

PRIOR_RETRY_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_explicit_mapping_missing_source_closure_retry_metrics.json"
SPLIT_CONTRACT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_missing_source_literal_residual_split_contract_metrics.json"
OUT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_explicit_mapping_missing_literal_fragment_inventory_metrics.json"
OUT_CSV = ROOT / "output" / "public" / "quantum" / "mass_origin_explicit_mapping_missing_literal_fragment_inventory_rows.csv"
DEFAULT_STEP_TAG = "8.7.55.2.130"


# 関数: `_utc_now_iso` の入出力契約と処理意図を定義する。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_parse_args` の入出力契約と処理意図を定義する。

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inventory fragment contexts and terminal atoms for explicit-mapping missing literal fragments.",
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


# 関数: `_unique_preserve_order` の入出力契約と処理意図を定義する。

def _unique_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []

    for item in items:
        # 条件分岐: `item not in seen` を満たす経路を評価する。
        if item not in seen:
            seen.add(item)
            ordered.append(item)

    return ordered


# 関数: `_fragment_plan` の入出力契約と処理意図を定義する。

def _fragment_plan() -> Dict[str, Dict[str, List[str] | str]]:
    return {
        "explicit_mapping_equation_phrase_fragment": {
            "present_contexts": [
                "lhs_observable_context",
                "rhs_curvature_context",
                "reference_point_context",
            ],
            "missing_atoms": ["explicit_mapping_equation_token_atom"],
            "note": "The observable, curvature, and reference-point contexts are public, but the terminal equation token is absent.",
        },
        "same_sector_equivalence_statement_phrase_fragment": {
            "present_contexts": ["lhs_same_sector_context"],
            "missing_atoms": ["same_sector_equivalence_token_atom"],
            "note": "The same-sector context is public, but the terminal equivalence token is absent.",
        },
        "mapping_operator_or_relation_phrase_fragment": {
            "present_contexts": [
                "lhs_observable_context",
                "rhs_curvature_context",
            ],
            "missing_atoms": ["mapping_operator_or_relation_token_atom"],
            "note": "The left-hand and right-hand contexts are public, but the terminal operator / relation token is absent.",
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

    required_explicit_mapping_missing_literal_fragments = [
        str(item) for item in split_contract_summary.get("explicit_mapping_missing_literal_fragments", [])
    ]
    explicit_mapping_missing_literal_route_still_admissible = bool(
        split_contract_summary.get("explicit_mapping_missing_literal_route_still_admissible", False)
    )
    prior_explicit_mapping_missing_literal_inventory_ready = bool(
        prior_retry_summary.get("explicit_mapping_missing_literal_inventory_ready", False)
    )

    fragment_plan = _fragment_plan()
    present_contexts_by_fragment = {
        fragment_id: list(fragment_plan.get(fragment_id, {}).get("present_contexts", []))
        for fragment_id in required_explicit_mapping_missing_literal_fragments
    }
    missing_atoms_by_fragment = {
        fragment_id: list(fragment_plan.get(fragment_id, {}).get("missing_atoms", []))
        for fragment_id in required_explicit_mapping_missing_literal_fragments
    }
    present_explicit_mapping_missing_fragment_contexts = _unique_preserve_order(
        [
            context_id
            for fragment_id in required_explicit_mapping_missing_literal_fragments
            for context_id in present_contexts_by_fragment.get(fragment_id, [])
        ]
    )
    missing_explicit_mapping_missing_literal_atoms = _unique_preserve_order(
        [
            atom_id
            for fragment_id in required_explicit_mapping_missing_literal_fragments
            for atom_id in missing_atoms_by_fragment.get(fragment_id, [])
        ]
    )
    explicit_mapping_missing_fragment_inventory_ready = bool(
        explicit_mapping_missing_literal_route_still_admissible
        and prior_explicit_mapping_missing_literal_inventory_ready
        and set(required_explicit_mapping_missing_literal_fragments).issubset(set(fragment_plan.keys()))
    )

    rows: List[Dict[str, Any]] = [
        {
            "row_id": "explicit_mapping_missing_literal_fragment_inventory_complete",
            "status": "pass",
            "metric": "explicit-mapping missing-literal-fragment inventory complete",
            "value": 1.0,
            "fragment_id": "aggregate",
            "atom_id": "aggregate",
            "note": "This step freezes present fragment contexts and missing terminal atoms for the explicit-mapping literal fragments that remain absent.",
        },
        {
            "row_id": "explicit_mapping_missing_literal_fragment_inventory_route_admissible",
            "status": "pass" if explicit_mapping_missing_literal_route_still_admissible else "reject",
            "metric": "explicit-mapping missing-literal-fragment route remains admissible",
            "value": 1.0 if explicit_mapping_missing_literal_route_still_admissible else 0.0,
            "fragment_id": "aggregate",
            "atom_id": "aggregate",
            "note": (
                "The explicit-mapping missing-literal-fragment route remains admissible after the residual-source-literal split."
                if explicit_mapping_missing_literal_route_still_admissible
                else "The explicit-mapping missing-literal-fragment route is no longer admissible, so fragment inventory cannot support closure retry."
            ),
        },
        {
            "row_id": "explicit_mapping_missing_literal_fragment_inventory_ready",
            "status": "pass" if explicit_mapping_missing_fragment_inventory_ready else "reject",
            "metric": "explicit-mapping missing-literal-fragment inventory ready for closure retry",
            "value": 1.0 if explicit_mapping_missing_fragment_inventory_ready else 0.0,
            "fragment_id": "aggregate",
            "atom_id": "aggregate",
            "note": (
                "Each missing explicit-mapping literal fragment now has an explicit present-context / missing-atom decomposition."
                if explicit_mapping_missing_fragment_inventory_ready
                else "The current public canonical pack still cannot provide a stable atom decomposition for every explicit-mapping missing literal fragment."
            ),
        },
        {
            "row_id": "explicit_mapping_missing_literal_fragment_context_count",
            "status": "inventory",
            "metric": "present explicit-mapping missing-literal-fragment context count",
            "value": float(len(present_explicit_mapping_missing_fragment_contexts)),
            "fragment_id": "aggregate",
            "atom_id": "aggregate",
            "note": f"Present explicit-mapping missing-literal-fragment contexts are {present_explicit_mapping_missing_fragment_contexts}.",
        },
        {
            "row_id": "explicit_mapping_missing_literal_fragment_atom_count",
            "status": "inventory",
            "metric": "missing explicit-mapping terminal atom count",
            "value": float(len(missing_explicit_mapping_missing_literal_atoms)),
            "fragment_id": "aggregate",
            "atom_id": "aggregate",
            "note": f"Missing explicit-mapping terminal atoms are {missing_explicit_mapping_missing_literal_atoms}.",
        },
    ]

    for fragment_id in required_explicit_mapping_missing_literal_fragments:
        fragment_present_contexts = present_contexts_by_fragment.get(fragment_id, [])
        fragment_missing_atoms = missing_atoms_by_fragment.get(fragment_id, [])
        fragment_note = str(fragment_plan.get(fragment_id, {}).get("note", ""))
        fragment_closable_now = not fragment_missing_atoms
        rows.append(
            {
                "row_id": f"explicit_mapping_missing_literal_fragment_{fragment_id}",
                "status": "pass" if fragment_closable_now else "watch",
                "metric": f"atom decomposition for explicit-mapping missing literal fragment {fragment_id}",
                "value": float(len(fragment_present_contexts)),
                "fragment_id": fragment_id,
                "atom_id": "aggregate",
                "note": f"Present contexts: {fragment_present_contexts}. Missing atoms: {fragment_missing_atoms}. {fragment_note}",
            }
        )

        for context_id in fragment_present_contexts:
            rows.append(
                {
                    "row_id": f"explicit_mapping_missing_literal_fragment_present_context_{fragment_id}_{context_id}",
                    "status": "inventory",
                    "metric": f"present context candidate for explicit-mapping missing literal fragment {fragment_id}",
                    "value": 1.0,
                    "fragment_id": fragment_id,
                    "atom_id": context_id,
                    "note": f"{context_id} is already present in the current public canonical pack as context for {fragment_id}.",
                }
            )

        for atom_id in fragment_missing_atoms:
            rows.append(
                {
                    "row_id": f"explicit_mapping_missing_literal_fragment_missing_atom_{fragment_id}_{atom_id}",
                    "status": "watch",
                    "metric": f"missing terminal atom for explicit-mapping literal fragment {fragment_id}",
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
            "name": "explicit-mapping missing-literal-fragment inventory",
        },
        "inputs": {
            "mass_origin_explicit_mapping_missing_source_closure_retry_json": _relative_str(PRIOR_RETRY_JSON),
            "mass_origin_missing_source_literal_residual_split_contract_json": _relative_str(SPLIT_CONTRACT_JSON),
        },
        "intent": "Decompose the remaining explicit-mapping literal fragments into present fragment contexts and missing terminal atoms using only the current public canonical pack.",
        "formulas": {
            "fragment_inventory_rule": "each explicit-mapping missing literal fragment is decomposed into fragment contexts already implied in the public pack and terminal atoms that remain absent",
            "inventory_ready_rule": "explicit_mapping_missing_fragment_inventory_ready iff the explicit-mapping missing-literal-fragment route remains admissible, the prior literal inventory remains ready, and every residual fragment is explicitly decomposed into present contexts and missing atoms",
        },
        "rows": rows,
        "summary": {
            "required_explicit_mapping_missing_literal_fragments": required_explicit_mapping_missing_literal_fragments,
            "present_explicit_mapping_missing_fragment_contexts": present_explicit_mapping_missing_fragment_contexts,
            "missing_explicit_mapping_missing_literal_atoms": missing_explicit_mapping_missing_literal_atoms,
            "present_explicit_mapping_missing_fragment_contexts_by_fragment": present_contexts_by_fragment,
            "missing_explicit_mapping_missing_literal_atoms_by_fragment": missing_atoms_by_fragment,
            "explicit_mapping_missing_literal_route_still_admissible": explicit_mapping_missing_literal_route_still_admissible,
            "prior_explicit_mapping_missing_literal_inventory_ready": prior_explicit_mapping_missing_literal_inventory_ready,
            "explicit_mapping_missing_fragment_inventory_ready": explicit_mapping_missing_fragment_inventory_ready,
            "prior_explicit_mapping_missing_source_nonclosure_reason_or_none": prior_retry_summary.get(
                "explicit_mapping_missing_source_nonclosure_reason_or_none"
            ),
        },
        "decision": {
            "overall_status": "explicit_mapping_missing_literal_fragment_inventory_frozen",
            "keep_mass_origin_branch_blocked": True,
            "required_explicit_mapping_missing_literal_fragments": required_explicit_mapping_missing_literal_fragments,
            "present_explicit_mapping_missing_fragment_contexts": present_explicit_mapping_missing_fragment_contexts,
            "missing_explicit_mapping_missing_literal_atoms": missing_explicit_mapping_missing_literal_atoms,
            "explicit_mapping_missing_fragment_inventory_ready": explicit_mapping_missing_fragment_inventory_ready,
        },
        "evidence": {
            "explicit_mapping_missing_source_closure_retry_summary": prior_retry_summary,
            "missing_source_literal_residual_split_contract_summary": split_contract_summary,
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
