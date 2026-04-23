#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_origin_shell_anchor_missing_terminal_glyph_inventory.py

Step 8.7.55.2.230:
Inventory glyph contexts and terminal symbol fragment candidates for the
shell-anchor missing terminal glyphs that remain after the current residual
missing-terminal-glyph split.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]
PRIOR_RETRY_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_shell_anchor_missing_terminal_atom_closure_retry_metrics.json"
SPLIT_CONTRACT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_missing_terminal_glyph_residual_split_contract_metrics.json"
OUT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_shell_anchor_missing_terminal_glyph_inventory_metrics.json"
OUT_CSV = ROOT / "output" / "public" / "quantum" / "mass_origin_shell_anchor_missing_terminal_glyph_inventory_rows.csv"
DEFAULT_STEP_TAG = "8.7.55.2.230"


# 関数: `_utc_now_iso` の入出力契約と処理意図を定義する。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_parse_args` の入出力契約と処理意図を定義する。

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inventory shell-anchor missing terminal glyphs.")
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


# 関数: `_glyph_plan` の入出力契約と処理意図を定義する。

def _glyph_plan() -> Dict[str, Dict[str, List[str] | str]]:
    return {
        "pair_to_target_relation_terminal_glyph": {
            "present_contexts": [
                "shell_anchor_pair_reference_context",
                "bridge_target_symbol::absP_star_times_vppp_over_vpp",
            ],
            "missing_fragments": ["pair_to_target_relation_symbol_fragment"],
            "note": "The pair and target symbol remain public, but the terminal symbol fragment that prints the shell-anchor relation is absent.",
        },
        "dimensionless_target_note_terminal_glyph": {
            "present_contexts": ["bridge_target_symbol::absP_star_times_vppp_over_vpp"],
            "missing_fragments": ["dimensionless_target_note_symbol_fragment"],
            "note": "The target symbol remains public, but the terminal symbol fragment that marks the target as dimensionless is absent.",
        },
        "no_new_free_parameter_terminal_glyph": {
            "present_contexts": ["same_sector_statement_context"],
            "missing_fragments": ["no_new_free_parameter_symbol_fragment"],
            "note": "The same-sector statement remains public, but the terminal symbol fragment that freezes the no-new-free-parameter note is absent.",
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

    required_glyphs = [str(item) for item in split_contract_summary.get("shell_anchor_missing_terminal_glyphs", [])]
    route_ok = bool(split_contract_summary.get("shell_anchor_missing_terminal_glyph_route_still_admissible", False))
    prior_retry_ok = bool(prior_retry_summary.get("shell_anchor_missing_terminal_atom_inventory_ready", False))

    glyph_plan = _glyph_plan()
    present_by_glyph = {
        glyph_id: list(glyph_plan.get(glyph_id, {}).get("present_contexts", []))
        for glyph_id in required_glyphs
    }
    missing_by_glyph = {
        glyph_id: list(glyph_plan.get(glyph_id, {}).get("missing_fragments", []))
        for glyph_id in required_glyphs
    }
    present_contexts = _unique_preserve_order(
        [context for glyph_id in required_glyphs for context in present_by_glyph.get(glyph_id, [])]
    )
    missing_fragments = _unique_preserve_order(
        [fragment for glyph_id in required_glyphs for fragment in missing_by_glyph.get(glyph_id, [])]
    )
    inventory_ready = bool(route_ok and prior_retry_ok and set(required_glyphs).issubset(set(glyph_plan.keys())))

    rows: List[Dict[str, Any]] = [
        {
            "row_id": "shell_anchor_missing_terminal_glyph_inventory_complete",
            "status": "pass",
            "metric": "shell-anchor missing-terminal-glyph inventory complete",
            "value": 1.0,
            "glyph_id": "aggregate",
            "fragment_id": "aggregate",
            "note": "This step freezes present glyph contexts and missing terminal symbol fragments for the shell-anchor route.",
        },
        {
            "row_id": "shell_anchor_missing_terminal_glyph_inventory_ready",
            "status": "pass" if inventory_ready else "reject",
            "metric": "shell-anchor missing-terminal-glyph inventory ready for closure retry",
            "value": 1.0 if inventory_ready else 0.0,
            "glyph_id": "aggregate",
            "fragment_id": "aggregate",
            "note": (
                "Each missing shell-anchor terminal glyph now has an explicit present-context / missing-symbol-fragment decomposition."
                if inventory_ready
                else "The current public canonical pack still cannot provide a stable symbol-fragment decomposition for every shell-anchor terminal glyph."
            ),
        },
        {
            "row_id": "shell_anchor_missing_terminal_glyph_fragment_count",
            "status": "inventory",
            "metric": "missing shell-anchor symbol fragment count",
            "value": float(len(missing_fragments)),
            "glyph_id": "aggregate",
            "fragment_id": "aggregate",
            "note": f"Missing shell-anchor symbol fragments are {missing_fragments}.",
        },
    ]

    for glyph_id in required_glyphs:
        glyph_present_contexts = present_by_glyph.get(glyph_id, [])
        glyph_missing_fragments = missing_by_glyph.get(glyph_id, [])
        glyph_note = str(glyph_plan.get(glyph_id, {}).get("note", ""))
        rows.append(
            {
                "row_id": f"shell_anchor_missing_terminal_glyph_{glyph_id}",
                "status": "watch",
                "metric": f"symbol-fragment decomposition for shell-anchor missing terminal glyph {glyph_id}",
                "value": float(len(glyph_present_contexts)),
                "glyph_id": glyph_id,
                "fragment_id": "aggregate",
                "note": f"Present contexts: {glyph_present_contexts}. Missing symbol fragments: {glyph_missing_fragments}. {glyph_note}",
            }
        )

    return {
        "generated_utc": _utc_now_iso(),
        "phase": {"phase": 8, "step": step_tag, "name": "shell-anchor missing-terminal-glyph inventory"},
        "inputs": {
            "mass_origin_shell_anchor_missing_terminal_atom_closure_retry_json": _relative_str(PRIOR_RETRY_JSON),
            "mass_origin_missing_terminal_glyph_residual_split_contract_json": _relative_str(SPLIT_CONTRACT_JSON),
        },
        "rows": rows,
        "summary": {
            "required_shell_anchor_missing_terminal_glyphs": required_glyphs,
            "present_shell_anchor_missing_glyph_contexts": present_contexts,
            "missing_shell_anchor_missing_symbol_fragments": missing_fragments,
            "present_shell_anchor_missing_glyph_contexts_by_glyph": present_by_glyph,
            "missing_shell_anchor_missing_symbol_fragments_by_glyph": missing_by_glyph,
            "shell_anchor_missing_terminal_glyph_route_still_admissible": route_ok,
            "prior_shell_anchor_missing_terminal_atom_inventory_ready": prior_retry_ok,
            "shell_anchor_missing_terminal_glyph_inventory_ready": inventory_ready,
            "prior_shell_anchor_missing_terminal_atom_nonclosure_reason_or_none": prior_retry_summary.get(
                "shell_anchor_missing_terminal_atom_nonclosure_reason_or_none"
            ),
        },
        "decision": {
            "overall_status": "shell_anchor_missing_terminal_glyph_inventory_frozen",
            "keep_mass_origin_branch_blocked": True,
            "required_shell_anchor_missing_terminal_glyphs": required_glyphs,
            "present_shell_anchor_missing_glyph_contexts": present_contexts,
            "missing_shell_anchor_missing_symbol_fragments": missing_fragments,
            "shell_anchor_missing_terminal_glyph_inventory_ready": inventory_ready,
        },
    }


# 関数: `_write_csv` の入出力契約と処理意図を定義する。

def _write_csv(rows: List[Dict[str, Any]]) -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["row_id", "status", "metric", "value", "glyph_id", "fragment_id", "note"])
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
