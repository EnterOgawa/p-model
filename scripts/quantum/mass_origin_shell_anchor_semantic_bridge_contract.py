#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_origin_shell_anchor_semantic_bridge_contract.py

Step 8.7.55.2.103:
Freeze the admissible same-sector semantic bridge contract that would be
required to connect the surviving shell-anchor pair to the tie-break
invariant absP_star_times_vppp_over_vpp without introducing new free
parameters.

Inputs:
  - output/public/quantum/mass_origin_same_sector_tiebreak_target_source_contract_metrics.json
  - output/public/quantum/mass_origin_shell_anchor_target_synthesis_contract_metrics.json
  - output/public/quantum/mass_origin_shell_anchor_target_synthesis_metrics.json
  - output/public/quantum/mass_origin_split_source_branch_refresh_metrics.json

Outputs:
  - output/public/quantum/mass_origin_shell_anchor_semantic_bridge_contract_metrics.json
  - output/public/quantum/mass_origin_shell_anchor_semantic_bridge_contract_rows.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]

SOURCE_CONTRACT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_same_sector_tiebreak_target_source_contract_metrics.json"
SHELL_SYNTHESIS_CONTRACT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_shell_anchor_target_synthesis_contract_metrics.json"
SHELL_SYNTHESIS_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_shell_anchor_target_synthesis_metrics.json"
SPLIT_SOURCE_REFRESH_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_split_source_branch_refresh_metrics.json"
OUT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_shell_anchor_semantic_bridge_contract_metrics.json"
OUT_CSV = ROOT / "output" / "public" / "quantum" / "mass_origin_shell_anchor_semantic_bridge_contract_rows.csv"
DEFAULT_STEP_TAG = "8.7.55.2.103"


# 関数: `_utc_now_iso` の入出力契約と処理意図を定義する。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_parse_args` の入出力契約と処理意図を定義する。

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze the shell-anchor semantic bridge contract for the same-sector tie-break target value.",
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
    for path in (
        SOURCE_CONTRACT_JSON,
        SHELL_SYNTHESIS_CONTRACT_JSON,
        SHELL_SYNTHESIS_JSON,
        SPLIT_SOURCE_REFRESH_JSON,
    ):
        _require_path(path)

    source_contract = _read_json(SOURCE_CONTRACT_JSON)
    shell_synthesis_contract = _read_json(SHELL_SYNTHESIS_CONTRACT_JSON)
    shell_synthesis = _read_json(SHELL_SYNTHESIS_JSON)
    split_source_refresh = _read_json(SPLIT_SOURCE_REFRESH_JSON)

    source_contract_summary = source_contract.get("summary", {})
    shell_synthesis_contract_summary = shell_synthesis_contract.get("summary", {})
    shell_synthesis_summary = shell_synthesis.get("summary", {})
    split_source_refresh_summary = split_source_refresh.get("summary", {})

    required_bridge_subject_row_ids = [
        str(item) for item in shell_synthesis_contract_summary.get("required_shell_anchor_row_ids", [])
    ]
    required_bridge_target_symbol = str(source_contract_summary.get("tiebreak_invariant_name"))
    required_bridge_relation_slots = [
        "shell_anchor_pair_reference",
        "pair_to_target_relation",
        "same_sector_statement",
        "dimensionless_target_note",
        "no_new_free_parameter_note",
    ]
    forbidden_semantic_shortcuts = [
        "cross_sector_proxy_substitution",
        "interface_only_spread_substitution",
        "phenomenological_backsolve",
        "undeclared_shell_equivalent_substitution",
        "target_value_placeholder_substitution",
        "candidate_id_selection_without_bridge",
    ]
    shell_anchor_pair_complete = bool(shell_synthesis_summary.get("shell_anchor_pair_complete", False))
    semantic_bridge_route_admissible = bool(shell_synthesis_contract_summary.get("shell_anchor_route_still_admissible", False))
    current_semantic_bridge_available = False
    semantic_bridge_contract_ready = bool(shell_anchor_pair_complete and semantic_bridge_route_admissible)

    rows = [
        {
            "row_id": "shell_anchor_semantic_bridge_contract_complete",
            "status": "pass",
            "metric": "shell-anchor semantic bridge contract complete",
            "value": 1.0,
            "note": "This step freezes the minimal same-sector wording contract needed to connect the shell-anchor pair to the derivative-ratio invariant.",
        },
        {
            "row_id": "shell_anchor_semantic_bridge_route_admissible",
            "status": "pass" if semantic_bridge_route_admissible else "reject",
            "metric": "shell-anchor semantic bridge route remains admissible",
            "value": 1.0 if semantic_bridge_route_admissible else 0.0,
            "note": "The shell-anchor source class remains allowed under the same-sector no-new-free-parameter contract.",
        },
        {
            "row_id": "shell_anchor_semantic_bridge_pair_complete",
            "status": "pass" if shell_anchor_pair_complete else "reject",
            "metric": "required shell-anchor pair already complete",
            "value": 1.0 if shell_anchor_pair_complete else 0.0,
            "note": f"Required bridge subject rows are {required_bridge_subject_row_ids}.",
        },
        {
            "row_id": "shell_anchor_semantic_bridge_relation_slot_count",
            "status": "inventory",
            "metric": "required semantic bridge relation slot count",
            "value": float(len(required_bridge_relation_slots)),
            "note": f"Required relation slots are {required_bridge_relation_slots}.",
        },
        {
            "row_id": "shell_anchor_semantic_bridge_forbidden_shortcut_count",
            "status": "inventory",
            "metric": "forbidden semantic shortcut count",
            "value": float(len(forbidden_semantic_shortcuts)),
            "note": f"Forbidden semantic shortcuts are {forbidden_semantic_shortcuts}.",
        },
        {
            "row_id": "shell_anchor_semantic_bridge_currently_absent",
            "status": "watch",
            "metric": "shell-anchor semantic bridge currently available",
            "value": 1.0 if current_semantic_bridge_available else 0.0,
            "note": "The current public canonical pack still has no same-sector semantic bridge from the shell-anchor pair to absP_star_times_vppp_over_vpp.",
        },
        {
            "row_id": "shell_anchor_semantic_bridge_contract_ready",
            "status": "pass" if semantic_bridge_contract_ready else "reject",
            "metric": "shell-anchor semantic bridge contract ready for audit",
            "value": 1.0 if semantic_bridge_contract_ready else 0.0,
            "note": (
                "The next step may audit whether the current public canonical pack already satisfies this frozen semantic bridge contract."
                if semantic_bridge_contract_ready
                else "The shell-anchor pair or route admissibility is not stable enough to audit the semantic bridge."
            ),
        },
    ]

    return {
        "generated_utc": _utc_now_iso(),
        "phase": {
            "phase": 8,
            "step": step_tag,
            "name": "shell-anchor semantic bridge contract",
        },
        "inputs": {
            "mass_origin_same_sector_tiebreak_target_source_contract_json": _relative_str(SOURCE_CONTRACT_JSON),
            "mass_origin_shell_anchor_target_synthesis_contract_json": _relative_str(SHELL_SYNTHESIS_CONTRACT_JSON),
            "mass_origin_shell_anchor_target_synthesis_json": _relative_str(SHELL_SYNTHESIS_JSON),
            "mass_origin_split_source_branch_refresh_json": _relative_str(SPLIT_SOURCE_REFRESH_JSON),
        },
        "intent": "Freeze the minimal same-sector semantic bridge wording that would be required to connect the shell-anchor pair to absP_star_times_vppp_over_vpp without new parameters.",
        "formulas": {
            "semantic_bridge_rule": "semantic_bridge_available iff the surviving shell-anchor pair already carries a public same-sector relation that equates the frozen pair synthesis to absP_star_times_vppp_over_vpp",
            "shortcut_rule": "the semantic bridge may not be satisfied by cross-sector, interface-only, phenomenological, undeclared-shell-equivalent, placeholder, or candidate-name-only shortcuts",
        },
        "rows": rows,
        "summary": {
            "required_bridge_subject_row_ids": required_bridge_subject_row_ids,
            "required_bridge_target_symbol": required_bridge_target_symbol,
            "required_bridge_relation_slots": required_bridge_relation_slots,
            "forbidden_semantic_shortcuts": forbidden_semantic_shortcuts,
            "shell_anchor_pair_complete": shell_anchor_pair_complete,
            "semantic_bridge_route_admissible": semantic_bridge_route_admissible,
            "current_semantic_bridge_available": current_semantic_bridge_available,
            "semantic_bridge_contract_ready": semantic_bridge_contract_ready,
            "prior_source_level_blockers": split_source_refresh_summary.get("remaining_source_level_blockers", []),
        },
        "decision": {
            "overall_status": "shell_anchor_semantic_bridge_contract_frozen",
            "keep_mass_origin_branch_blocked": True,
            "required_bridge_subject_row_ids": required_bridge_subject_row_ids,
            "required_bridge_target_symbol": required_bridge_target_symbol,
            "required_bridge_relation_slots": required_bridge_relation_slots,
            "forbidden_semantic_shortcuts": forbidden_semantic_shortcuts,
            "semantic_bridge_contract_ready": semantic_bridge_contract_ready,
        },
        "evidence": {
            "target_source_contract_summary": source_contract_summary,
            "shell_synthesis_contract_summary": shell_synthesis_contract_summary,
            "shell_synthesis_summary": shell_synthesis_summary,
            "split_source_refresh_summary": split_source_refresh_summary,
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
