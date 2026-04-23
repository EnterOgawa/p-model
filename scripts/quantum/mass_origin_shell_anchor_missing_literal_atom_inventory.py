#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_origin_shell_anchor_missing_literal_atom_inventory.py

Step 8.7.55.2.134:
Inventory atom contexts and terminal glyph candidates for the shell-anchor
missing literal atoms that remain after the residual atom split.

Inputs:
  - output/public/quantum/mass_origin_shell_anchor_missing_literal_fragment_closure_retry_metrics.json
  - output/public/quantum/mass_origin_missing_literal_atom_residual_split_contract_metrics.json

Outputs:
  - output/public/quantum/mass_origin_shell_anchor_missing_literal_atom_inventory_metrics.json
  - output/public/quantum/mass_origin_shell_anchor_missing_literal_atom_inventory_rows.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]

PRIOR_RETRY_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_shell_anchor_missing_literal_fragment_closure_retry_metrics.json"
SPLIT_CONTRACT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_missing_literal_atom_residual_split_contract_metrics.json"
OUT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_shell_anchor_missing_literal_atom_inventory_metrics.json"
OUT_CSV = ROOT / "output" / "public" / "quantum" / "mass_origin_shell_anchor_missing_literal_atom_inventory_rows.csv"
DEFAULT_STEP_TAG = "8.7.55.2.134"


# 関数: `_utc_now_iso` の入出力契約と処理意図を定義する。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_parse_args` の入出力契約と処理意図を定義する。

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inventory atom contexts and terminal glyph candidates for shell-anchor missing literal atoms.",
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


# 関数: `_atom_plan` の入出力契約と処理意図を定義する。

def _atom_plan() -> Dict[str, Dict[str, List[str] | str]]:
    return {
        "pair_to_target_relation_token_atom": {
            "present_contexts": [
                "shell_anchor_pair_reference_context",
                "bridge_target_symbol::absP_star_times_vppp_over_vpp",
            ],
            "missing_glyphs": ["pair_to_target_relation_terminal_glyph"],
            "note": "The pair and target symbol remain public, but the terminal glyph that prints the relation is absent.",
        },
        "dimensionless_target_note_token_atom": {
            "present_contexts": ["bridge_target_symbol::absP_star_times_vppp_over_vpp"],
            "missing_glyphs": ["dimensionless_target_note_terminal_glyph"],
            "note": "The target symbol remains public, but the terminal glyph that marks it as dimensionless is absent.",
        },
        "no_new_free_parameter_token_atom": {
            "present_contexts": ["same_sector_statement_context"],
            "missing_glyphs": ["no_new_free_parameter_terminal_glyph"],
            "note": "The same-sector statement remains public, but the terminal glyph that freezes the no-new-free-parameter note is absent.",
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

    required_shell_anchor_missing_literal_atoms = [
        str(item) for item in split_contract_summary.get("shell_anchor_missing_literal_atoms", [])
    ]
    shell_anchor_missing_atom_route_still_admissible = bool(
        split_contract_summary.get("shell_anchor_missing_atom_route_still_admissible", False)
    )
    prior_shell_anchor_missing_atom_inventory_ready = bool(
        prior_retry_summary.get("shell_anchor_missing_fragment_inventory_ready", False)
    )

    atom_plan = _atom_plan()
    present_contexts_by_atom = {
        atom_id: list(atom_plan.get(atom_id, {}).get("present_contexts", []))
        for atom_id in required_shell_anchor_missing_literal_atoms
    }
    missing_glyphs_by_atom = {
        atom_id: list(atom_plan.get(atom_id, {}).get("missing_glyphs", []))
        for atom_id in required_shell_anchor_missing_literal_atoms
    }
    present_shell_anchor_missing_atom_contexts = _unique_preserve_order(
        [
            context_id
            for atom_id in required_shell_anchor_missing_literal_atoms
            for context_id in present_contexts_by_atom.get(atom_id, [])
        ]
    )
    missing_shell_anchor_missing_terminal_glyphs = _unique_preserve_order(
        [
            glyph_id
            for atom_id in required_shell_anchor_missing_literal_atoms
            for glyph_id in missing_glyphs_by_atom.get(atom_id, [])
        ]
    )
    shell_anchor_missing_atom_inventory_ready = bool(
        shell_anchor_missing_atom_route_still_admissible
        and prior_shell_anchor_missing_atom_inventory_ready
        and set(required_shell_anchor_missing_literal_atoms).issubset(set(atom_plan.keys()))
    )

    rows: List[Dict[str, Any]] = [
        {
            "row_id": "shell_anchor_missing_literal_atom_inventory_complete",
            "status": "pass",
            "metric": "shell-anchor missing-literal-atom inventory complete",
            "value": 1.0,
            "atom_id": "aggregate",
            "glyph_id": "aggregate",
            "note": "This step freezes present atom contexts and missing terminal glyphs for the shell-anchor literal atoms that remain absent.",
        },
        {
            "row_id": "shell_anchor_missing_literal_atom_inventory_route_admissible",
            "status": "pass" if shell_anchor_missing_atom_route_still_admissible else "reject",
            "metric": "shell-anchor missing-literal-atom route remains admissible",
            "value": 1.0 if shell_anchor_missing_atom_route_still_admissible else 0.0,
            "atom_id": "aggregate",
            "glyph_id": "aggregate",
            "note": (
                "The shell-anchor missing-literal-atom route remains admissible after the residual atom split."
                if shell_anchor_missing_atom_route_still_admissible
                else "The shell-anchor missing-literal-atom route is no longer admissible, so atom inventory cannot support closure retry."
            ),
        },
        {
            "row_id": "shell_anchor_missing_literal_atom_inventory_ready",
            "status": "pass" if shell_anchor_missing_atom_inventory_ready else "reject",
            "metric": "shell-anchor missing-literal-atom inventory ready for closure retry",
            "value": 1.0 if shell_anchor_missing_atom_inventory_ready else 0.0,
            "atom_id": "aggregate",
            "glyph_id": "aggregate",
            "note": (
                "Each missing shell-anchor literal atom now has an explicit present-context / missing-glyph decomposition."
                if shell_anchor_missing_atom_inventory_ready
                else "The current public canonical pack still cannot provide a stable glyph decomposition for every shell-anchor missing literal atom."
            ),
        },
        {
            "row_id": "shell_anchor_missing_literal_atom_context_count",
            "status": "inventory",
            "metric": "present shell-anchor missing-literal-atom context count",
            "value": float(len(present_shell_anchor_missing_atom_contexts)),
            "atom_id": "aggregate",
            "glyph_id": "aggregate",
            "note": f"Present shell-anchor missing-literal-atom contexts are {present_shell_anchor_missing_atom_contexts}.",
        },
        {
            "row_id": "shell_anchor_missing_literal_atom_glyph_count",
            "status": "inventory",
            "metric": "missing shell-anchor terminal glyph count",
            "value": float(len(missing_shell_anchor_missing_terminal_glyphs)),
            "atom_id": "aggregate",
            "glyph_id": "aggregate",
            "note": f"Missing shell-anchor terminal glyphs are {missing_shell_anchor_missing_terminal_glyphs}.",
        },
    ]

    for atom_id in required_shell_anchor_missing_literal_atoms:
        atom_present_contexts = present_contexts_by_atom.get(atom_id, [])
        atom_missing_glyphs = missing_glyphs_by_atom.get(atom_id, [])
        atom_note = str(atom_plan.get(atom_id, {}).get("note", ""))
        atom_closable_now = not atom_missing_glyphs
        rows.append(
            {
                "row_id": f"shell_anchor_missing_literal_atom_{atom_id}",
                "status": "pass" if atom_closable_now else "watch",
                "metric": f"glyph decomposition for shell-anchor missing literal atom {atom_id}",
                "value": float(len(atom_present_contexts)),
                "atom_id": atom_id,
                "glyph_id": "aggregate",
                "note": f"Present contexts: {atom_present_contexts}. Missing glyphs: {atom_missing_glyphs}. {atom_note}",
            }
        )

        for context_id in atom_present_contexts:
            rows.append(
                {
                    "row_id": f"shell_anchor_missing_literal_atom_present_context_{atom_id}_{context_id}",
                    "status": "inventory",
                    "metric": f"present context candidate for shell-anchor missing literal atom {atom_id}",
                    "value": 1.0,
                    "atom_id": atom_id,
                    "glyph_id": context_id,
                    "note": f"{context_id} is already present in the current public canonical pack as context for {atom_id}.",
                }
            )

        for glyph_id in atom_missing_glyphs:
            rows.append(
                {
                    "row_id": f"shell_anchor_missing_literal_atom_missing_glyph_{atom_id}_{glyph_id}",
                    "status": "watch",
                    "metric": f"missing terminal glyph for shell-anchor literal atom {atom_id}",
                    "value": 0.0,
                    "atom_id": atom_id,
                    "glyph_id": glyph_id,
                    "note": f"{glyph_id} is still absent from the current public canonical pack for {atom_id}.",
                }
            )

    return {
        "generated_utc": _utc_now_iso(),
        "phase": {
            "phase": 8,
            "step": step_tag,
            "name": "shell-anchor missing-literal-atom inventory",
        },
        "inputs": {
            "mass_origin_shell_anchor_missing_literal_fragment_closure_retry_json": _relative_str(PRIOR_RETRY_JSON),
            "mass_origin_missing_literal_atom_residual_split_contract_json": _relative_str(SPLIT_CONTRACT_JSON),
        },
        "intent": "Decompose the remaining shell-anchor literal atoms into present atom contexts and missing terminal glyphs using only the current public canonical pack.",
        "formulas": {
            "atom_inventory_rule": "each shell-anchor missing literal atom is decomposed into atom contexts already implied in the public pack and terminal glyphs that remain absent",
            "inventory_ready_rule": "shell_anchor_missing_atom_inventory_ready iff the shell-anchor missing-literal-atom route remains admissible, the prior fragment inventory remains ready, and every residual atom is explicitly decomposed into present contexts and missing glyphs",
        },
        "rows": rows,
        "summary": {
            "required_shell_anchor_missing_literal_atoms": required_shell_anchor_missing_literal_atoms,
            "present_shell_anchor_missing_atom_contexts": present_shell_anchor_missing_atom_contexts,
            "missing_shell_anchor_missing_terminal_glyphs": missing_shell_anchor_missing_terminal_glyphs,
            "present_shell_anchor_missing_atom_contexts_by_atom": present_contexts_by_atom,
            "missing_shell_anchor_missing_terminal_glyphs_by_atom": missing_glyphs_by_atom,
            "shell_anchor_missing_atom_route_still_admissible": shell_anchor_missing_atom_route_still_admissible,
            "prior_shell_anchor_missing_atom_inventory_ready": prior_shell_anchor_missing_atom_inventory_ready,
            "shell_anchor_missing_atom_inventory_ready": shell_anchor_missing_atom_inventory_ready,
            "prior_shell_anchor_missing_literal_fragment_nonclosure_reason_or_none": prior_retry_summary.get(
                "shell_anchor_missing_literal_fragment_nonclosure_reason_or_none"
            ),
        },
        "decision": {
            "overall_status": "shell_anchor_missing_literal_atom_inventory_frozen",
            "keep_mass_origin_branch_blocked": True,
            "required_shell_anchor_missing_literal_atoms": required_shell_anchor_missing_literal_atoms,
            "present_shell_anchor_missing_atom_contexts": present_shell_anchor_missing_atom_contexts,
            "missing_shell_anchor_missing_terminal_glyphs": missing_shell_anchor_missing_terminal_glyphs,
            "shell_anchor_missing_atom_inventory_ready": shell_anchor_missing_atom_inventory_ready,
        },
        "evidence": {
            "shell_anchor_missing_literal_fragment_closure_retry_summary": prior_retry_summary,
            "missing_literal_atom_residual_split_contract_summary": split_contract_summary,
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
    payload = _build_payload(step_tag=str(args.step_tag))
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_csv(payload["rows"])
    print(f"[ok] wrote {OUT_JSON}")
    print(f"[ok] wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
