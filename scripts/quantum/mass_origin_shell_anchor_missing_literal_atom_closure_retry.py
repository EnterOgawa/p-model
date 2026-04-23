#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_origin_shell_anchor_missing_literal_atom_closure_retry.py

Step 8.7.55.2.135:
Retry shell-anchor closure using the frozen missing-literal-atom inventory
produced after the residual atom split.

Inputs:
  - output/public/quantum/mass_origin_shell_anchor_missing_literal_fragment_closure_retry_metrics.json
  - output/public/quantum/mass_origin_missing_literal_atom_residual_split_contract_metrics.json
  - output/public/quantum/mass_origin_shell_anchor_missing_literal_atom_inventory_metrics.json

Outputs:
  - output/public/quantum/mass_origin_shell_anchor_missing_literal_atom_closure_retry_metrics.json
  - output/public/quantum/mass_origin_shell_anchor_missing_literal_atom_closure_retry_rows.csv
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
ATOM_INVENTORY_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_shell_anchor_missing_literal_atom_inventory_metrics.json"
OUT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_shell_anchor_missing_literal_atom_closure_retry_metrics.json"
OUT_CSV = ROOT / "output" / "public" / "quantum" / "mass_origin_shell_anchor_missing_literal_atom_closure_retry_rows.csv"
DEFAULT_STEP_TAG = "8.7.55.2.135"


# 関数: `_utc_now_iso` の入出力契約と処理意図を定義する。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_parse_args` の入出力契約と処理意図を定義する。

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retry shell-anchor closure using the frozen missing-literal-atom inventory.",
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

    shell_anchor_missing_atom_route_still_admissible = bool(
        split_contract_summary.get("shell_anchor_missing_atom_route_still_admissible", False)
    )
    shell_anchor_missing_atom_inventory_ready = bool(
        atom_inventory_summary.get("shell_anchor_missing_atom_inventory_ready", False)
    )
    required_shell_anchor_missing_literal_atoms = [
        str(item) for item in atom_inventory_summary.get("required_shell_anchor_missing_literal_atoms", [])
    ]
    present_shell_anchor_missing_atom_contexts = [
        str(item) for item in atom_inventory_summary.get("present_shell_anchor_missing_atom_contexts", [])
    ]
    missing_shell_anchor_missing_terminal_glyphs = [
        str(item) for item in atom_inventory_summary.get("missing_shell_anchor_missing_terminal_glyphs", [])
    ]
    semantic_bridge_available = bool(
        shell_anchor_missing_atom_route_still_admissible
        and shell_anchor_missing_atom_inventory_ready
        and not missing_shell_anchor_missing_terminal_glyphs
    )
    bridge_relation_kind_or_none = (
        "shell_anchor_missing_literal_atom_bridge"
        if semantic_bridge_available
        else None
    )
    semantic_bridge_without_new_free_parameters = bool(semantic_bridge_available)
    shell_anchor_missing_literal_atom_nonclosure_reason_or_none = None

    # 条件分岐: `not semantic_bridge_available` を満たす経路を評価する。
    if not semantic_bridge_available:
        shell_anchor_missing_literal_atom_nonclosure_reason_or_none = "shell_anchor_missing_terminal_glyphs_still_missing"

    rows: List[Dict[str, Any]] = [
        {
            "row_id": "shell_anchor_missing_literal_atom_closure_retry_complete",
            "status": "pass",
            "metric": "shell-anchor missing-literal-atom closure retry complete",
            "value": 1.0,
            "note": "This step retries shell-anchor semantic-bridge closure using the frozen missing-literal-atom inventory.",
        },
        {
            "row_id": "shell_anchor_missing_literal_atom_closure_retry_route_admissible",
            "status": "pass" if shell_anchor_missing_atom_route_still_admissible else "reject",
            "metric": "shell-anchor missing-literal-atom route remains admissible",
            "value": 1.0 if shell_anchor_missing_atom_route_still_admissible else 0.0,
            "note": (
                "The shell-anchor missing-literal-atom route remains admissible after the residual atom split."
                if shell_anchor_missing_atom_route_still_admissible
                else "The shell-anchor missing-literal-atom route is no longer admissible, so closure retry is not meaningful."
            ),
        },
        {
            "row_id": "shell_anchor_missing_literal_atom_closure_retry_inventory_ready",
            "status": "pass" if shell_anchor_missing_atom_inventory_ready else "reject",
            "metric": "shell-anchor missing-literal-atom inventory is ready for closure retry",
            "value": 1.0 if shell_anchor_missing_atom_inventory_ready else 0.0,
            "note": (
                "The missing-literal-atom inventory is internally complete and can support closure retry."
                if shell_anchor_missing_atom_inventory_ready
                else "The missing-literal-atom inventory is not complete enough to support closure retry."
            ),
        },
        {
            "row_id": "shell_anchor_missing_literal_atom_closure_retry_missing_glyph_count",
            "status": "inventory",
            "metric": "shell-anchor missing-literal-atom closure retry missing glyph count",
            "value": float(len(missing_shell_anchor_missing_terminal_glyphs)),
            "note": f"Missing terminal glyphs at retry are {missing_shell_anchor_missing_terminal_glyphs}.",
        },
        {
            "row_id": "shell_anchor_missing_literal_atom_closure_retry_bridge_available",
            "status": "pass" if semantic_bridge_available else "watch",
            "metric": "shell-anchor semantic bridge available after missing-literal-atom closure retry",
            "value": 1.0 if semantic_bridge_available else 0.0,
            "note": (
                f"Bridge relation kind is {bridge_relation_kind_or_none}."
                if semantic_bridge_available
                else "Closure retry remains non-closing because required shell-anchor terminal glyphs are still missing."
            ),
        },
        {
            "row_id": "shell_anchor_missing_literal_atom_closure_retry_no_new_free_parameters",
            "status": "pass" if semantic_bridge_without_new_free_parameters else "reject",
            "metric": "shell-anchor missing-literal-atom closure retry closes without new free parameters",
            "value": 1.0 if semantic_bridge_without_new_free_parameters else 0.0,
            "note": (
                "Closure retry yields a same-sector semantic bridge without introducing new free parameters."
                if semantic_bridge_without_new_free_parameters
                else "Closure retry did not yield a no-new-free-parameter semantic bridge."
            ),
        },
    ]

    return {
        "generated_utc": _utc_now_iso(),
        "phase": {
            "phase": 8,
            "step": step_tag,
            "name": "shell-anchor missing-literal-atom closure retry",
        },
        "inputs": {
            "mass_origin_shell_anchor_missing_literal_fragment_closure_retry_json": _relative_str(PRIOR_RETRY_JSON),
            "mass_origin_missing_literal_atom_residual_split_contract_json": _relative_str(SPLIT_CONTRACT_JSON),
            "mass_origin_shell_anchor_missing_literal_atom_inventory_json": _relative_str(ATOM_INVENTORY_JSON),
        },
        "intent": "Retry shell-anchor semantic-bridge closure using the frozen missing-literal-atom inventory from the residual atom branch.",
        "formulas": {
            "closure_retry_rule": "semantic_bridge_available iff the shell-anchor missing-literal-atom route remains admissible, the frozen atom inventory is ready, and no required terminal glyphs remain missing",
            "no_new_parameter_rule": "semantic_bridge_without_new_free_parameters iff the semantic bridge is available after missing-literal-atom closure retry",
        },
        "rows": rows,
        "summary": {
            "required_shell_anchor_missing_literal_atoms": required_shell_anchor_missing_literal_atoms,
            "present_shell_anchor_missing_atom_contexts": present_shell_anchor_missing_atom_contexts,
            "missing_shell_anchor_missing_terminal_glyphs": missing_shell_anchor_missing_terminal_glyphs,
            "shell_anchor_missing_atom_route_still_admissible": shell_anchor_missing_atom_route_still_admissible,
            "shell_anchor_missing_atom_inventory_ready": shell_anchor_missing_atom_inventory_ready,
            "semantic_bridge_available": semantic_bridge_available,
            "bridge_relation_kind_or_none": bridge_relation_kind_or_none,
            "semantic_bridge_without_new_free_parameters": semantic_bridge_without_new_free_parameters,
            "shell_anchor_missing_literal_atom_nonclosure_reason_or_none": shell_anchor_missing_literal_atom_nonclosure_reason_or_none,
            "prior_shell_anchor_missing_literal_fragment_nonclosure_reason_or_none": prior_retry_summary.get(
                "shell_anchor_missing_literal_fragment_nonclosure_reason_or_none"
            ),
        },
        "decision": {
            "overall_status": (
                "shell_anchor_missing_literal_atom_closure_retry_available"
                if semantic_bridge_available
                else "shell_anchor_missing_literal_atom_closure_retry_frozen_absent"
            ),
            "keep_mass_origin_branch_blocked": True,
            "semantic_bridge_available": semantic_bridge_available,
            "bridge_relation_kind_or_none": bridge_relation_kind_or_none,
            "semantic_bridge_without_new_free_parameters": semantic_bridge_without_new_free_parameters,
            "shell_anchor_missing_literal_atom_nonclosure_reason_or_none": shell_anchor_missing_literal_atom_nonclosure_reason_or_none,
            "missing_shell_anchor_missing_terminal_glyphs": missing_shell_anchor_missing_terminal_glyphs,
        },
        "evidence": {
            "shell_anchor_missing_literal_fragment_closure_retry_summary": prior_retry_summary,
            "missing_literal_atom_residual_split_contract_summary": split_contract_summary,
            "shell_anchor_missing_literal_atom_inventory_summary": atom_inventory_summary,
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
    payload = _build_payload(step_tag=str(args.step_tag))
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_csv(payload["rows"])
    print(f"[ok] wrote {OUT_JSON}")
    print(f"[ok] wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
