#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_origin_shell_anchor_semantic_bridge_audit.py

Step 8.7.55.2.104:
Audit whether the current public canonical pack already carries the frozen
same-sector semantic bridge that would connect the surviving shell-anchor
pair to the tie-break target absP_star_times_vppp_over_vpp without adding
new free parameters.

Inputs:
  - output/public/quantum/mass_origin_shell_anchor_semantic_bridge_contract_metrics.json
  - output/public/quantum/mass_origin_shell_anchor_target_synthesis_metrics.json
  - output/public/quantum/mass_origin_same_sector_tiebreak_target_source_inventory_rows.csv
  - output/public/quantum/mass_origin_shell_quantization_canonicalization_rows.csv

Outputs:
  - output/public/quantum/mass_origin_shell_anchor_semantic_bridge_metrics.json
  - output/public/quantum/mass_origin_shell_anchor_semantic_bridge_rows.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]

CONTRACT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_shell_anchor_semantic_bridge_contract_metrics.json"
SHELL_SYNTHESIS_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_shell_anchor_target_synthesis_metrics.json"
SOURCE_INVENTORY_ROWS_CSV = ROOT / "output" / "public" / "quantum" / "mass_origin_same_sector_tiebreak_target_source_inventory_rows.csv"
SHELL_CANONICAL_ROWS_CSV = ROOT / "output" / "public" / "quantum" / "mass_origin_shell_quantization_canonicalization_rows.csv"
OUT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_shell_anchor_semantic_bridge_metrics.json"
OUT_CSV = ROOT / "output" / "public" / "quantum" / "mass_origin_shell_anchor_semantic_bridge_rows.csv"
DEFAULT_STEP_TAG = "8.7.55.2.104"


# 関数: `_utc_now_iso` の入出力契約と処理意図を定義する。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_parse_args` の入出力契約と処理意図を定義する。

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit whether the shell-anchor pair already carries the frozen same-sector semantic bridge.",
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


# 関数: `_read_csv_by_row_id` の入出力契約と処理意図を定義する。

def _read_csv_by_row_id(path: Path) -> Dict[str, Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return {str(row["row_id"]): {key: str(value) for key, value in row.items()} for row in reader}


# 関数: `_read_inventory_rows_by_source_row_id` の入出力契約と処理意図を定義する。

def _read_inventory_rows_by_source_row_id(path: Path) -> Dict[str, Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows: Dict[str, Dict[str, str]] = {}

        for row in reader:
            normalized_row = {key: str(value) for key, value in row.items()}
            source_kind = normalized_row.get("source_kind", "")
            source_row_id = normalized_row.get("source_row_id", "")

            # 条件分岐: `source_kind == "surviving_shell_anchor_pack" and source_row_id` を満たす経路を評価する。
            if source_kind == "surviving_shell_anchor_pack" and source_row_id:
                rows[source_row_id] = normalized_row

        return rows


# 関数: `_relative_str` の入出力契約と処理意図を定義する。

def _relative_str(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


# 関数: `_combined_row_text` の入出力契約と処理意図を定義する。

def _combined_row_text(row: Dict[str, str] | None) -> str:
    # 条件分岐: `row is None` を満たす経路を評価する。
    if row is None:
        return ""

    return " ".join(
        part.strip()
        for part in (
            row.get("row_id", ""),
            row.get("metric", ""),
            row.get("note", ""),
        )
        if part and part.strip()
    ).lower()


# 関数: `_build_payload` の入出力契約と処理意図を定義する。

def _build_payload(step_tag: str) -> Dict[str, Any]:
    for path in (
        CONTRACT_JSON,
        SHELL_SYNTHESIS_JSON,
        SOURCE_INVENTORY_ROWS_CSV,
        SHELL_CANONICAL_ROWS_CSV,
    ):
        _require_path(path)

    contract = _read_json(CONTRACT_JSON)
    shell_synthesis = _read_json(SHELL_SYNTHESIS_JSON)
    source_inventory_rows = _read_inventory_rows_by_source_row_id(SOURCE_INVENTORY_ROWS_CSV)
    shell_canonical_rows = _read_csv_by_row_id(SHELL_CANONICAL_ROWS_CSV)

    contract_summary = contract.get("summary", {})
    shell_synthesis_summary = shell_synthesis.get("summary", {})

    required_bridge_subject_row_ids = [str(item) for item in contract_summary.get("required_bridge_subject_row_ids", [])]
    required_bridge_relation_slots = [str(item) for item in contract_summary.get("required_bridge_relation_slots", [])]
    forbidden_semantic_shortcuts = [str(item) for item in contract_summary.get("forbidden_semantic_shortcuts", [])]
    required_bridge_target_symbol = str(contract_summary.get("required_bridge_target_symbol"))

    inventory_rows_present_count = 0
    canonical_rows_present_count = 0
    source_corpus_parts: List[str] = []
    bridge_corpus_row_ids: List[str] = []

    for row_id in required_bridge_subject_row_ids:
        inventory_row = source_inventory_rows.get(row_id)
        canonical_row = shell_canonical_rows.get(row_id)

        # 条件分岐: `inventory_row is not None` を満たす経路を評価する。
        if inventory_row is not None:
            inventory_rows_present_count += 1
            source_corpus_parts.append(_combined_row_text(inventory_row))
            bridge_corpus_row_ids.append(f"inventory:{row_id}")

        # 条件分岐: `canonical_row is not None` を満たす経路を評価する。

        if canonical_row is not None:
            canonical_rows_present_count += 1
            source_corpus_parts.append(_combined_row_text(canonical_row))
            bridge_corpus_row_ids.append(f"canonical:{row_id}")

    combined_corpus = " ".join(part for part in source_corpus_parts if part).lower()
    shell_anchor_pair_complete = bool(shell_synthesis_summary.get("shell_anchor_pair_complete", False))
    semantic_bridge_route_admissible = bool(contract_summary.get("semantic_bridge_route_admissible", False))

    relation_slot_presence = {
        "shell_anchor_pair_reference": bool(
            shell_anchor_pair_complete
            and inventory_rows_present_count == len(required_bridge_subject_row_ids)
            and canonical_rows_present_count == len(required_bridge_subject_row_ids)
        ),
        "pair_to_target_relation": required_bridge_target_symbol.lower() in combined_corpus,
        "same_sector_statement": "same-sector" in combined_corpus or "same sector" in combined_corpus,
        "dimensionless_target_note": "dimensionless" in combined_corpus,
        "no_new_free_parameter_note": "no-new-free-parameter" in combined_corpus or "no new free parameter" in combined_corpus,
    }
    present_relation_slot_count = sum(1 for present in relation_slot_presence.values() if present)
    missing_relation_slots = [slot for slot in required_bridge_relation_slots if not relation_slot_presence.get(slot, False)]

    forbidden_shortcut_hits = [
        shortcut_id
        for shortcut_id in forbidden_semantic_shortcuts
        if shortcut_id.lower() in combined_corpus
    ]
    semantic_bridge_available = bool(
        semantic_bridge_route_admissible
        and shell_anchor_pair_complete
        and present_relation_slot_count == len(required_bridge_relation_slots)
        and not forbidden_shortcut_hits
    )
    bridge_relation_kind_or_none = (
        "shell_anchor_pair_to_absP_star_times_vppp_over_vpp_same_sector_bridge"
        if semantic_bridge_available
        else None
    )
    semantic_bridge_without_new_free_parameters = bool(
        semantic_bridge_available and relation_slot_presence.get("no_new_free_parameter_note", False)
    )
    semantic_bridge_nonclosure_reason_or_none = None

    # 条件分岐: `not semantic_bridge_available` を満たす経路を評価する。
    if not semantic_bridge_available:
        semantic_bridge_nonclosure_reason_or_none = "shell_anchor_pair_has_no_public_same_sector_semantic_bridge"

    rows = [
        {
            "row_id": "shell_anchor_semantic_bridge_audit_complete",
            "status": "pass",
            "metric": "shell-anchor semantic bridge audit complete",
            "value": 1.0,
            "note": "This audit checks whether the frozen shell-anchor pair already carries the required same-sector wording bridge to the tie-break target symbol.",
        },
        {
            "row_id": "shell_anchor_semantic_bridge_route_admissible",
            "status": "pass" if semantic_bridge_route_admissible else "reject",
            "metric": "shell-anchor semantic bridge route remains admissible",
            "value": 1.0 if semantic_bridge_route_admissible else 0.0,
            "note": "The audit is only meaningful while the route remains admissible under the frozen contract.",
        },
        {
            "row_id": "shell_anchor_semantic_bridge_pair_complete",
            "status": "pass" if relation_slot_presence["shell_anchor_pair_reference"] else "reject",
            "metric": "shell-anchor pair reference is publicly complete",
            "value": 1.0 if relation_slot_presence["shell_anchor_pair_reference"] else 0.0,
            "note": f"Required bridge subject rows are {required_bridge_subject_row_ids}.",
        },
        {
            "row_id": "shell_anchor_semantic_bridge_present_relation_slot_count",
            "status": "inventory",
            "metric": "present semantic bridge relation slot count",
            "value": float(present_relation_slot_count),
            "note": f"Present relation slots are {sorted([slot for slot, present in relation_slot_presence.items() if present])}.",
        },
        {
            "row_id": "shell_anchor_semantic_bridge_missing_relation_slot_count",
            "status": "inventory",
            "metric": "missing semantic bridge relation slot count",
            "value": float(len(missing_relation_slots)),
            "note": f"Missing relation slots are {missing_relation_slots}.",
        },
        {
            "row_id": "shell_anchor_semantic_bridge_pair_to_target_relation",
            "status": "pass" if relation_slot_presence["pair_to_target_relation"] else "watch",
            "metric": "shell-anchor pair already publishes a direct relation to the tie-break target symbol",
            "value": 1.0 if relation_slot_presence["pair_to_target_relation"] else 0.0,
            "note": (
                f"The current shell-anchor row corpus explicitly references {required_bridge_target_symbol}."
                if relation_slot_presence["pair_to_target_relation"]
                else f"The current shell-anchor row corpus never names {required_bridge_target_symbol}, so the bridge remains non-closing."
            ),
        },
        {
            "row_id": "shell_anchor_semantic_bridge_same_sector_statement",
            "status": "pass" if relation_slot_presence["same_sector_statement"] else "watch",
            "metric": "shell-anchor row corpus already carries same-sector wording",
            "value": 1.0 if relation_slot_presence["same_sector_statement"] else 0.0,
            "note": (
                "The current shell-anchor row corpus already contains same-sector wording."
                if relation_slot_presence["same_sector_statement"]
                else "No same-sector wording is attached to the current shell-anchor row corpus."
            ),
        },
        {
            "row_id": "shell_anchor_semantic_bridge_dimensionless_note",
            "status": "pass" if relation_slot_presence["dimensionless_target_note"] else "watch",
            "metric": "shell-anchor row corpus already marks the target as dimensionless",
            "value": 1.0 if relation_slot_presence["dimensionless_target_note"] else 0.0,
            "note": (
                "The current shell-anchor row corpus already marks the target quantity as dimensionless."
                if relation_slot_presence["dimensionless_target_note"]
                else "No dimensionless-target note is attached to the current shell-anchor row corpus."
            ),
        },
        {
            "row_id": "shell_anchor_semantic_bridge_no_new_free_parameter_note",
            "status": "pass" if relation_slot_presence["no_new_free_parameter_note"] else "watch",
            "metric": "shell-anchor row corpus already states no new free parameters",
            "value": 1.0 if relation_slot_presence["no_new_free_parameter_note"] else 0.0,
            "note": (
                "The current shell-anchor row corpus already states that the bridge introduces no new free parameters."
                if relation_slot_presence["no_new_free_parameter_note"]
                else "No no-new-free-parameter note is attached to the current shell-anchor row corpus."
            ),
        },
        {
            "row_id": "shell_anchor_semantic_bridge_forbidden_shortcut_hit_count",
            "status": "inventory",
            "metric": "forbidden semantic shortcut hit count",
            "value": float(len(forbidden_shortcut_hits)),
            "note": (
                f"Forbidden shortcut hits are {forbidden_shortcut_hits}."
                if forbidden_shortcut_hits
                else "No forbidden semantic shortcuts are detected in the current shell-anchor row corpus."
            ),
        },
        {
            "row_id": "shell_anchor_semantic_bridge_available",
            "status": "pass" if semantic_bridge_available else "watch",
            "metric": "shell-anchor semantic bridge already available in public canonical form",
            "value": 1.0 if semantic_bridge_available else 0.0,
            "note": (
                f"Bridge relation kind is {bridge_relation_kind_or_none}."
                if semantic_bridge_available
                else f"The bridge remains absent because the missing relation slots are {missing_relation_slots}."
            ),
        },
        {
            "row_id": "shell_anchor_semantic_bridge_without_new_free_parameters",
            "status": "pass" if semantic_bridge_without_new_free_parameters else "reject",
            "metric": "shell-anchor semantic bridge closes without new free parameters",
            "value": 1.0 if semantic_bridge_without_new_free_parameters else 0.0,
            "note": (
                "The shell-anchor bridge is public, same-sector, and explicitly no-new-free-parameter."
                if semantic_bridge_without_new_free_parameters
                else "The bridge is not yet public-canonical complete, so a no-new-free-parameter closure cannot be claimed."
            ),
        },
    ]

    return {
        "generated_utc": _utc_now_iso(),
        "phase": {
            "phase": 8,
            "step": step_tag,
            "name": "shell-anchor semantic bridge audit",
        },
        "inputs": {
            "mass_origin_shell_anchor_semantic_bridge_contract_json": _relative_str(CONTRACT_JSON),
            "mass_origin_shell_anchor_target_synthesis_json": _relative_str(SHELL_SYNTHESIS_JSON),
            "mass_origin_same_sector_tiebreak_target_source_inventory_rows_csv": _relative_str(SOURCE_INVENTORY_ROWS_CSV),
            "mass_origin_shell_quantization_canonicalization_rows_csv": _relative_str(SHELL_CANONICAL_ROWS_CSV),
        },
        "intent": "Audit whether the current public canonical shell-anchor row corpus already satisfies the frozen same-sector semantic bridge contract.",
        "formulas": {
            "semantic_bridge_rule": "semantic_bridge_available iff the shell-anchor pair is publicly complete, every frozen relation slot is present in the current shell-anchor row corpus, and no forbidden shortcut is used",
            "no_new_parameter_rule": "semantic_bridge_without_new_free_parameters iff the semantic bridge is available and the corpus explicitly carries the no-new-free-parameter note",
        },
        "rows": rows,
        "summary": {
            "required_bridge_subject_row_ids": required_bridge_subject_row_ids,
            "required_bridge_target_symbol": required_bridge_target_symbol,
            "semantic_bridge_available": semantic_bridge_available,
            "bridge_relation_kind_or_none": bridge_relation_kind_or_none,
            "semantic_bridge_without_new_free_parameters": semantic_bridge_without_new_free_parameters,
            "required_relation_slot_count": len(required_bridge_relation_slots),
            "present_relation_slot_count": present_relation_slot_count,
            "missing_relation_slots": missing_relation_slots,
            "forbidden_shortcut_hits": forbidden_shortcut_hits,
            "semantic_bridge_nonclosure_reason_or_none": semantic_bridge_nonclosure_reason_or_none,
            "bridge_corpus_row_ids": bridge_corpus_row_ids,
        },
        "decision": {
            "overall_status": (
                "shell_anchor_semantic_bridge_available"
                if semantic_bridge_available
                else "shell_anchor_semantic_bridge_frozen_absent"
            ),
            "keep_mass_origin_branch_blocked": True,
            "semantic_bridge_available": semantic_bridge_available,
            "bridge_relation_kind_or_none": bridge_relation_kind_or_none,
            "semantic_bridge_without_new_free_parameters": semantic_bridge_without_new_free_parameters,
            "missing_relation_slots": missing_relation_slots,
            "forbidden_shortcut_hits": forbidden_shortcut_hits,
        },
        "evidence": {
            "contract_summary": contract_summary,
            "shell_synthesis_summary": shell_synthesis_summary,
            "source_inventory_rows": {
                row_id: source_inventory_rows.get(row_id)
                for row_id in required_bridge_subject_row_ids
            },
            "shell_anchor_canonical_rows": {
                row_id: shell_canonical_rows.get(row_id)
                for row_id in required_bridge_subject_row_ids
            },
            "relation_slot_presence": relation_slot_presence,
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
