#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_origin_shell_anchor_missing_source_literal_inventory.py

Step 8.7.55.2.122:
Inventory literal contexts and terminal fragments for the shell-anchor
missing wording sources that remain after the residual-source split contract.

Inputs:
  - output/public/quantum/mass_origin_shell_anchor_missing_slot_closure_retry_metrics.json
  - output/public/quantum/mass_origin_missing_slot_source_residual_split_contract_metrics.json

Outputs:
  - output/public/quantum/mass_origin_shell_anchor_missing_source_literal_inventory_metrics.json
  - output/public/quantum/mass_origin_shell_anchor_missing_source_literal_inventory_rows.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]

PRIOR_RETRY_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_shell_anchor_missing_slot_closure_retry_metrics.json"
SPLIT_CONTRACT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_missing_slot_source_residual_split_contract_metrics.json"
OUT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_shell_anchor_missing_source_literal_inventory_metrics.json"
OUT_CSV = ROOT / "output" / "public" / "quantum" / "mass_origin_shell_anchor_missing_source_literal_inventory_rows.csv"
DEFAULT_STEP_TAG = "8.7.55.2.122"


# 関数: `_utc_now_iso` の入出力契約と処理意図を定義する。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_parse_args` の入出力契約と処理意図を定義する。

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inventory literal contexts and terminal fragments for shell-anchor missing wording sources.",
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


# 関数: `_literal_plan` の入出力契約と処理意図を定義する。

def _literal_plan() -> Dict[str, Dict[str, List[str] | str]]:
    return {
        "pair_to_target_relation_literal": {
            "present_contexts": [
                "shell_anchor_pair_reference_context",
                "bridge_target_symbol::absP_star_times_vppp_over_vpp",
            ],
            "missing_fragments": ["pair_to_target_relation_phrase_fragment"],
            "note": "The pair name and target symbol are already present, but the literal phrase that maps the pair onto the target is still absent.",
        },
        "dimensionless_target_note_literal": {
            "present_contexts": ["bridge_target_symbol::absP_star_times_vppp_over_vpp"],
            "missing_fragments": ["dimensionless_target_note_phrase_fragment"],
            "note": "The target symbol is already fixed, but the literal phrase stating that the target is dimensionless is still absent.",
        },
        "no_new_free_parameter_note_literal": {
            "present_contexts": ["same_sector_statement_context"],
            "missing_fragments": ["no_new_free_parameter_note_phrase_fragment"],
            "note": "The same-sector route is already declared, but the literal phrase that freezes the no-new-free-parameter claim is still absent.",
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

    required_shell_anchor_missing_literal_sources = [
        str(item) for item in split_contract_summary.get("shell_anchor_missing_wording_sources", [])
    ]
    shell_anchor_missing_source_route_still_admissible = bool(
        split_contract_summary.get("shell_anchor_missing_source_route_still_admissible", False)
    )
    shell_anchor_missing_slot_source_inventory_ready = bool(
        prior_retry_summary.get("shell_anchor_missing_slot_source_inventory_ready", False)
    )

    literal_plan = _literal_plan()
    present_contexts_by_source = {
        source_id: list(literal_plan.get(source_id, {}).get("present_contexts", []))
        for source_id in required_shell_anchor_missing_literal_sources
    }
    missing_fragments_by_source = {
        source_id: list(literal_plan.get(source_id, {}).get("missing_fragments", []))
        for source_id in required_shell_anchor_missing_literal_sources
    }
    present_shell_anchor_missing_literal_contexts = _unique_preserve_order(
        [
            context_id
            for source_id in required_shell_anchor_missing_literal_sources
            for context_id in present_contexts_by_source.get(source_id, [])
        ]
    )
    missing_shell_anchor_missing_literal_fragments = _unique_preserve_order(
        [
            fragment_id
            for source_id in required_shell_anchor_missing_literal_sources
            for fragment_id in missing_fragments_by_source.get(source_id, [])
        ]
    )
    shell_anchor_missing_literal_inventory_ready = bool(
        shell_anchor_missing_source_route_still_admissible
        and shell_anchor_missing_slot_source_inventory_ready
        and set(required_shell_anchor_missing_literal_sources).issubset(set(literal_plan.keys()))
    )

    rows: List[Dict[str, Any]] = [
        {
            "row_id": "shell_anchor_missing_source_literal_inventory_complete",
            "status": "pass",
            "metric": "shell-anchor missing-source literal inventory complete",
            "value": 1.0,
            "source_id": "aggregate",
            "fragment_id": "aggregate",
            "note": "This step freezes present contexts and missing terminal fragments for the shell-anchor wording sources that are still absent.",
        },
        {
            "row_id": "shell_anchor_missing_source_literal_inventory_route_admissible",
            "status": "pass" if shell_anchor_missing_source_route_still_admissible else "reject",
            "metric": "shell-anchor missing-source route remains admissible",
            "value": 1.0 if shell_anchor_missing_source_route_still_admissible else 0.0,
            "source_id": "aggregate",
            "fragment_id": "aggregate",
            "note": (
                "The shell-anchor missing-source route remains admissible after the residual-source split."
                if shell_anchor_missing_source_route_still_admissible
                else "The shell-anchor missing-source route is no longer admissible, so literal inventory cannot support closure retry."
            ),
        },
        {
            "row_id": "shell_anchor_missing_source_literal_inventory_ready",
            "status": "pass" if shell_anchor_missing_literal_inventory_ready else "reject",
            "metric": "shell-anchor missing-source literal inventory ready for closure retry",
            "value": 1.0 if shell_anchor_missing_literal_inventory_ready else 0.0,
            "source_id": "aggregate",
            "fragment_id": "aggregate",
            "note": (
                "Each missing shell-anchor wording source now has an explicit present-context / missing-fragment decomposition."
                if shell_anchor_missing_literal_inventory_ready
                else "The current public canonical pack still cannot provide a stable literal decomposition for every shell-anchor missing wording source."
            ),
        },
        {
            "row_id": "shell_anchor_missing_source_literal_context_count",
            "status": "inventory",
            "metric": "present shell-anchor missing-source literal context count",
            "value": float(len(present_shell_anchor_missing_literal_contexts)),
            "source_id": "aggregate",
            "fragment_id": "aggregate",
            "note": f"Present shell-anchor missing-source literal contexts are {present_shell_anchor_missing_literal_contexts}.",
        },
        {
            "row_id": "shell_anchor_missing_source_literal_fragment_count",
            "status": "inventory",
            "metric": "missing shell-anchor missing-source terminal fragment count",
            "value": float(len(missing_shell_anchor_missing_literal_fragments)),
            "source_id": "aggregate",
            "fragment_id": "aggregate",
            "note": f"Missing shell-anchor missing-source terminal fragments are {missing_shell_anchor_missing_literal_fragments}.",
        },
    ]

    for source_id in required_shell_anchor_missing_literal_sources:
        source_present_contexts = present_contexts_by_source.get(source_id, [])
        source_missing_fragments = missing_fragments_by_source.get(source_id, [])
        source_note = str(literal_plan.get(source_id, {}).get("note", ""))
        source_closable_now = not source_missing_fragments
        rows.append(
            {
                "row_id": f"shell_anchor_missing_source_literal_{source_id}",
                "status": "pass" if source_closable_now else "watch",
                "metric": f"literal decomposition for shell-anchor missing wording source {source_id}",
                "value": float(len(source_present_contexts)),
                "source_id": source_id,
                "fragment_id": "aggregate",
                "note": f"Present contexts: {source_present_contexts}. Missing fragments: {source_missing_fragments}. {source_note}",
            }
        )

        for context_id in source_present_contexts:
            rows.append(
                {
                    "row_id": f"shell_anchor_missing_source_literal_present_context_{source_id}_{context_id}",
                    "status": "inventory",
                    "metric": f"present context candidate for shell-anchor missing wording source {source_id}",
                    "value": 1.0,
                    "source_id": source_id,
                    "fragment_id": context_id,
                    "note": f"{context_id} is already present in the current public canonical pack as context for {source_id}.",
                }
            )

        for fragment_id in source_missing_fragments:
            rows.append(
                {
                    "row_id": f"shell_anchor_missing_source_literal_missing_fragment_{source_id}_{fragment_id}",
                    "status": "watch",
                    "metric": f"missing terminal fragment for shell-anchor missing wording source {source_id}",
                    "value": 0.0,
                    "source_id": source_id,
                    "fragment_id": fragment_id,
                    "note": f"{fragment_id} is still absent from the current public canonical pack for {source_id}.",
                }
            )

    return {
        "generated_utc": _utc_now_iso(),
        "phase": {
            "phase": 8,
            "step": step_tag,
            "name": "shell-anchor missing-source literal inventory",
        },
        "inputs": {
            "mass_origin_shell_anchor_missing_slot_closure_retry_json": _relative_str(PRIOR_RETRY_JSON),
            "mass_origin_missing_slot_source_residual_split_contract_json": _relative_str(SPLIT_CONTRACT_JSON),
        },
        "intent": "Decompose the remaining shell-anchor missing wording sources into present literal contexts and missing terminal fragments using only the current public canonical pack.",
        "formulas": {
            "literal_inventory_rule": "each shell-anchor missing wording source is decomposed into contextual phrases already implied in the public pack and terminal literal fragments that remain absent",
            "inventory_ready_rule": "shell_anchor_missing_literal_inventory_ready iff the shell-anchor missing-source route remains admissible, the prior source inventory remains ready, and every residual source is explicitly decomposed into present contexts and missing fragments",
        },
        "rows": rows,
        "summary": {
            "required_shell_anchor_missing_literal_sources": required_shell_anchor_missing_literal_sources,
            "present_shell_anchor_missing_literal_contexts": present_shell_anchor_missing_literal_contexts,
            "missing_shell_anchor_missing_literal_fragments": missing_shell_anchor_missing_literal_fragments,
            "present_shell_anchor_missing_literal_contexts_by_source": present_contexts_by_source,
            "missing_shell_anchor_missing_literal_fragments_by_source": missing_fragments_by_source,
            "shell_anchor_missing_source_route_still_admissible": shell_anchor_missing_source_route_still_admissible,
            "shell_anchor_missing_slot_source_inventory_ready": shell_anchor_missing_slot_source_inventory_ready,
            "shell_anchor_missing_literal_inventory_ready": shell_anchor_missing_literal_inventory_ready,
            "prior_shell_anchor_missing_slot_nonclosure_reason_or_none": prior_retry_summary.get(
                "shell_anchor_missing_slot_nonclosure_reason_or_none"
            ),
        },
        "decision": {
            "overall_status": "shell_anchor_missing_source_literal_inventory_frozen",
            "keep_mass_origin_branch_blocked": True,
            "required_shell_anchor_missing_literal_sources": required_shell_anchor_missing_literal_sources,
            "present_shell_anchor_missing_literal_contexts": present_shell_anchor_missing_literal_contexts,
            "missing_shell_anchor_missing_literal_fragments": missing_shell_anchor_missing_literal_fragments,
            "shell_anchor_missing_literal_inventory_ready": shell_anchor_missing_literal_inventory_ready,
        },
        "evidence": {
            "shell_anchor_missing_slot_closure_retry_summary": prior_retry_summary,
            "missing_slot_source_residual_split_contract_summary": split_contract_summary,
        },
    }


# 関数: `_write_csv` の入出力契約と処理意図を定義する。

def _write_csv(rows: List[Dict[str, Any]]) -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["row_id", "status", "metric", "value", "source_id", "fragment_id", "note"])
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
