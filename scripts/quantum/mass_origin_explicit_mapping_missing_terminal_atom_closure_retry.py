#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_origin_explicit_mapping_missing_terminal_atom_closure_retry.py

Step 8.7.55.2.227:
Retry explicit-mapping closure using the frozen missing-terminal-atom
inventory produced after the residual terminal-atom split.
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
ATOM_INVENTORY_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_explicit_mapping_missing_terminal_atom_inventory_metrics.json"
OUT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_explicit_mapping_missing_terminal_atom_closure_retry_metrics.json"
OUT_CSV = ROOT / "output" / "public" / "quantum" / "mass_origin_explicit_mapping_missing_terminal_atom_closure_retry_rows.csv"
DEFAULT_STEP_TAG = "8.7.55.2.227"


# 関数: `_utc_now_iso` の入出力契約と処理意図を定義する。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_parse_args` の入出力契約と処理意図を定義する。

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retry explicit-mapping closure using the frozen missing-terminal-atom inventory.")
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


# 関数: `_build_payload` の入出力契約と処理意図を定義する。

def _build_payload(step_tag: str) -> Dict[str, Any]:
    for path in (PRIOR_RETRY_JSON, SPLIT_CONTRACT_JSON, ATOM_INVENTORY_JSON):
        _require_path(path)

    prior_retry = _read_json(PRIOR_RETRY_JSON)
    split_contract = _read_json(SPLIT_CONTRACT_JSON)
    atom_inventory = _read_json(ATOM_INVENTORY_JSON)
    prior_retry_summary = prior_retry.get("summary", {})
    split_contract_summary = split_contract.get("summary", {})
    atom_inventory_summary = atom_inventory.get("summary", {})

    route_ok = bool(split_contract_summary.get("explicit_mapping_missing_terminal_atom_route_still_admissible", False))
    inventory_ok = bool(atom_inventory_summary.get("explicit_mapping_missing_terminal_atom_inventory_ready", False))
    required_atoms = [str(item) for item in atom_inventory_summary.get("required_explicit_mapping_missing_terminal_atoms", [])]
    present_contexts = [str(item) for item in atom_inventory_summary.get("present_explicit_mapping_missing_atom_contexts", [])]
    missing_glyphs = [str(item) for item in atom_inventory_summary.get("missing_explicit_mapping_missing_terminal_glyphs", [])]
    equation_available = bool(route_ok and inventory_ok and not missing_glyphs)
    lifted_kind = "explicit_mapping_missing_terminal_atom_bridge" if equation_available else None
    no_new_parameters = bool(equation_available)
    nonclosure_reason = None if equation_available else "explicit_mapping_missing_terminal_glyphs_still_missing"

    rows: List[Dict[str, Any]] = [
        {"row_id": "explicit_mapping_missing_terminal_atom_closure_retry_complete", "status": "pass", "metric": "explicit-mapping missing-terminal-atom closure retry complete", "value": 1.0, "note": "This step retries explicit-mapping closure using the frozen missing-terminal-atom inventory."},
        {"row_id": "explicit_mapping_missing_terminal_atom_closure_retry_route_admissible", "status": "pass" if route_ok else "reject", "metric": "explicit-mapping missing-terminal-atom route remains admissible", "value": 1.0 if route_ok else 0.0, "note": "The explicit-mapping missing-terminal-atom route remains admissible after the residual terminal-atom split." if route_ok else "The explicit-mapping missing-terminal-atom route is no longer admissible, so closure retry is not meaningful."},
        {"row_id": "explicit_mapping_missing_terminal_atom_closure_retry_inventory_ready", "status": "pass" if inventory_ok else "reject", "metric": "explicit-mapping missing-terminal-atom inventory is ready for closure retry", "value": 1.0 if inventory_ok else 0.0, "note": "The missing-terminal-atom inventory is internally complete and can support closure retry." if inventory_ok else "The missing-terminal-atom inventory is not complete enough to support closure retry."},
        {"row_id": "explicit_mapping_missing_terminal_atom_closure_retry_missing_glyph_count", "status": "inventory", "metric": "explicit-mapping missing-terminal-atom closure retry missing glyph count", "value": float(len(missing_glyphs)), "note": f"Missing terminal glyphs at retry are {missing_glyphs}."},
        {"row_id": "explicit_mapping_missing_terminal_atom_closure_retry_equation_available", "status": "pass" if equation_available else "watch", "metric": "explicit mapping equation available after missing-terminal-atom closure retry", "value": 1.0 if equation_available else 0.0, "note": f"Lifted mapping equation kind is {lifted_kind}." if equation_available else "Closure retry remains non-closing because required explicit-mapping terminal glyphs are still missing."},
        {"row_id": "explicit_mapping_missing_terminal_atom_closure_retry_no_new_free_parameters", "status": "pass" if no_new_parameters else "reject", "metric": "explicit-mapping missing-terminal-atom closure retry closes without new free parameters", "value": 1.0 if no_new_parameters else 0.0, "note": "Closure retry yields an explicit mapping equation without introducing new free parameters." if no_new_parameters else "Closure retry did not yield an explicit mapping equation without new free parameters."},
    ]

    return {
        "generated_utc": _utc_now_iso(),
        "phase": {"phase": 8, "step": step_tag, "name": "explicit-mapping missing-terminal-atom closure retry"},
        "inputs": {
            "mass_origin_explicit_mapping_missing_symbol_fragment_closure_retry_json": _relative_str(PRIOR_RETRY_JSON),
            "mass_origin_missing_terminal_atom_residual_split_contract_json": _relative_str(SPLIT_CONTRACT_JSON),
            "mass_origin_explicit_mapping_missing_terminal_atom_inventory_json": _relative_str(ATOM_INVENTORY_JSON),
        },
        "rows": rows,
        "summary": {
            "required_explicit_mapping_missing_terminal_atoms": required_atoms,
            "present_explicit_mapping_missing_atom_contexts": present_contexts,
            "missing_explicit_mapping_missing_terminal_glyphs": missing_glyphs,
            "explicit_mapping_missing_terminal_atom_route_still_admissible": route_ok,
            "explicit_mapping_missing_terminal_atom_inventory_ready": inventory_ok,
            "explicit_mapping_equation_available": equation_available,
            "lifted_mapping_equation_kind_or_none": lifted_kind,
            "mapping_without_new_free_parameters": no_new_parameters,
            "explicit_mapping_missing_terminal_atom_nonclosure_reason_or_none": nonclosure_reason,
            "prior_explicit_mapping_missing_symbol_fragment_nonclosure_reason_or_none": prior_retry_summary.get("explicit_mapping_missing_symbol_fragment_nonclosure_reason_or_none"),
        },
        "decision": {
            "overall_status": "explicit_mapping_missing_terminal_atom_closure_retry_available" if equation_available else "explicit_mapping_missing_terminal_atom_closure_retry_frozen_absent",
            "keep_mass_origin_branch_blocked": True,
            "explicit_mapping_equation_available": equation_available,
            "lifted_mapping_equation_kind_or_none": lifted_kind,
            "mapping_without_new_free_parameters": no_new_parameters,
            "explicit_mapping_missing_terminal_atom_nonclosure_reason_or_none": nonclosure_reason,
            "missing_explicit_mapping_missing_terminal_glyphs": missing_glyphs,
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
    payload = _build_payload(str(args.step_tag))
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_csv(payload["rows"])
    print(f"[ok] wrote {OUT_JSON}")
    print(f"[ok] wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
