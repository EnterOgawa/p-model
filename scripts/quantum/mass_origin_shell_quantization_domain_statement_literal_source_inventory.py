#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_origin_shell_quantization_domain_statement_literal_source_inventory.py

Step 8.7.55.2.426:
Inventory the current public-canonical source candidates for the missing
shell-quantization domain-statement literal.

Inputs:
  - output/public/quantum/mass_origin_shell_quantization_domain_statement_source_inventory_metrics.json
  - output/public/quantum/mass_origin_shell_quantization_domain_statement_literal_route_contract_metrics.json

Outputs:
  - output/public/quantum/mass_origin_shell_quantization_domain_statement_literal_source_inventory_metrics.json
  - output/public/quantum/mass_origin_shell_quantization_domain_statement_literal_source_inventory_rows.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]

STATEMENT_SOURCE_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_shell_quantization_domain_statement_source_inventory_metrics.json"
LITERAL_ROUTE_CONTRACT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_shell_quantization_domain_statement_literal_route_contract_metrics.json"
OUT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_shell_quantization_domain_statement_literal_source_inventory_metrics.json"
OUT_CSV = ROOT / "output" / "public" / "quantum" / "mass_origin_shell_quantization_domain_statement_literal_source_inventory_rows.csv"
DEFAULT_STEP_TAG = "8.7.55.2.426"


# 関数: 現在UTC時刻を ISO 8601 文字列で返す。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: CLI 引数を解釈する。

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inventory source candidates for the missing shell-quantization domain-statement literal.")
    parser.add_argument("--step-tag", default=DEFAULT_STEP_TAG, help="Roadmap step tag to stamp into the output payload.")
    return parser.parse_args()


# 関数: 必須入力の存在を検査する。

def _require_path(path: Path) -> None:
    if not path.exists():
        raise SystemExit(f"[fail] missing required input: {path}")


# 関数: JSON ファイルを辞書として読む。

def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


# 関数: リポジトリ相対パスへ正規化する。

def _relative_str(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


# 関数: rows を構成する。

def _build_rows(
    *,
    required_sources: List[str],
    present_sources: List[str],
    missing_sources: List[str],
    first_route_to_close_or_none: str | None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = [
        {
            "row_id": "shell_quantization_domain_statement_literal_source_inventory_complete",
            "status": "pass",
            "metric": "shell-quantization domain-statement-literal source inventory complete",
            "value": 1.0,
            "note": "This step inventories concrete public-canonical sources for the missing shell-quantization domain-statement literal.",
        },
        {
            "row_id": "shell_quantization_domain_statement_literal_required_source_count",
            "status": "pass",
            "metric": "required source count for shell-quantization domain-statement-literal route",
            "value": float(len(required_sources)),
            "note": f"Required sources: {required_sources}.",
        },
        {
            "row_id": "shell_quantization_domain_statement_literal_present_source_count",
            "status": "pass",
            "metric": "present source count for shell-quantization domain-statement-literal route",
            "value": float(len(present_sources)),
            "note": f"Present sources: {present_sources}.",
        },
        {
            "row_id": "shell_quantization_domain_statement_literal_missing_source_count",
            "status": "watch",
            "metric": "missing source count for shell-quantization domain-statement-literal route",
            "value": float(len(missing_sources)),
            "note": f"Missing sources: {missing_sources}.",
        },
    ]

    for source in required_sources:
        source_present = source in present_sources
        rows.append(
            {
                "row_id": f"shell_quantization_domain_statement_literal_source_{source}",
                "status": "pass" if source_present else "watch",
                "metric": f"source availability for {source}",
                "value": 1.0 if source_present else 0.0,
                "note": (
                    f"{source} is already available in the current public canonical pack."
                    if source_present
                    else f"{source} is still missing from the current public canonical pack."
                ),
            }
        )

    rows.append(
        {
            "row_id": "shell_quantization_domain_statement_literal_first_route_to_close",
            "status": "watch",
            "metric": "first residual source to close for shell-quantization domain-statement-literal route",
            "value": 1.0 if first_route_to_close_or_none else 0.0,
            "note": f"The next closure attempt starts from {first_route_to_close_or_none}.",
        }
    )
    return rows


# 関数: metrics payload 全体を構成する。

def _build_payload(step_tag: str) -> Dict[str, Any]:
    for path in (STATEMENT_SOURCE_JSON, LITERAL_ROUTE_CONTRACT_JSON):
        _require_path(path)

    statement_source = _read_json(STATEMENT_SOURCE_JSON)
    literal_route_contract = _read_json(LITERAL_ROUTE_CONTRACT_JSON)

    statement_source_summary = statement_source.get("summary", {})
    literal_route_contract_summary = literal_route_contract.get("summary", {})

    required_sources = [
        "shell_quantization_family_public_candidate",
        "shell_quantization_fit_kappa_row",
        "shell_quantization_fit_kz_over_kn_row",
        "geometric_domain_symbol_note",
        "boundary_condition_quantization_note",
        "shell_quantization_domain_statement_phrase_fragment",
    ]
    present_from_statement = [str(item) for item in statement_source_summary.get("present_domain_statement_sources", [])]
    present_sources = [source for source in required_sources if source in present_from_statement]
    missing_sources = [source for source in required_sources if source not in present_sources]
    first_route_to_close_or_none = "shell_quantization_domain_statement_phrase_fragment"
    rows = _build_rows(
        required_sources=required_sources,
        present_sources=present_sources,
        missing_sources=missing_sources,
        first_route_to_close_or_none=first_route_to_close_or_none,
    )

    return {
        "generated_utc": _utc_now_iso(),
        "phase": {
            "phase": 8,
            "step": step_tag,
            "name": "shell-quantization domain-statement-literal source inventory",
        },
        "inputs": {
            "mass_origin_shell_quantization_domain_statement_source_inventory_json": _relative_str(STATEMENT_SOURCE_JSON),
            "mass_origin_shell_quantization_domain_statement_literal_route_contract_json": _relative_str(LITERAL_ROUTE_CONTRACT_JSON),
        },
        "intent": "Inventory source candidates that could instantiate the missing shell-quantization domain-statement literal without a new fit parameter.",
        "formulas": {
            "inventory_rule": "the literal route can close only after the public pack exposes shell-family rows, geometric domain notes, boundary-condition quantization wording, and a shell-quantization domain-statement phrase fragment",
            "current_absence": "the current pack still lacks the shell-quantization domain-statement phrase fragment, so the first fine-grained closure attempt starts from that missing phrase fragment",
        },
        "rows": rows,
        "summary": {
            "required_domain_statement_literal_sources": required_sources,
            "present_domain_statement_literal_sources": present_sources,
            "missing_domain_statement_literal_sources": missing_sources,
            "first_route_to_close_or_none": first_route_to_close_or_none,
            "domain_statement_literal_source_inventory_ready": True,
        },
        "decision": {
            "overall_status": "shell_quantization_domain_statement_literal_source_inventory_frozen",
            "keep_mass_origin_branch_blocked": True,
            "selected_residual_binding_route_or_none": literal_route_contract_summary.get("selected_residual_binding_route_or_none"),
            "missing_geometric_boundary_artifact": literal_route_contract_summary.get("missing_geometric_boundary_artifact"),
            "required_domain_statement_literal_sources": required_sources,
            "present_domain_statement_literal_sources": present_sources,
            "missing_domain_statement_literal_sources": missing_sources,
            "first_route_to_close_or_none": first_route_to_close_or_none,
            "domain_statement_literal_source_inventory_ready": True,
            "hand_off_to_8_7_55_2_84": False,
            "next_required_artifacts": [
                "shell_quantization_domain_statement_phrase_fragment",
                "shell_quantization_domain_statement_literal",
                "shell_quantization_domain_statement",
                "shell_quantization_to_domain_relation",
            ],
        },
        "evidence": {
            "shell_quantization_domain_statement_source_inventory_summary": statement_source_summary,
            "shell_quantization_domain_statement_literal_route_contract_summary": literal_route_contract_summary,
        },
    }


# 関数: rows を CSV 出力する。

def _write_csv(rows: List[Dict[str, Any]]) -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["row_id", "status", "metric", "value", "note"])
        writer.writeheader()
        writer.writerows(rows)


# 関数: JSON を整形出力する。

def _write_json(payload: Dict[str, Any]) -> None:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


# 関数: エントリポイントとして step を実行する。

def main() -> None:
    args = _parse_args()
    payload = _build_payload(args.step_tag)
    _write_json(payload)
    _write_csv(payload["rows"])
    print(f"[ok] wrote {OUT_JSON}")
    print(f"[ok] wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
