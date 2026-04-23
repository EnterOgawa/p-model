#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_origin_same_sector_tiebreak_target_bridge.py

Step 8.7.55.2.86:
Audit whether the same-sector tie-break invariant target value can be derived
from the chi_P contract or the surviving shell anchor pack without adding new
free parameters.

Inputs:
  - output/public/quantum/mass_origin_same_sector_chi_to_vpp_contract_metrics.json
  - output/public/quantum/mass_origin_positive_particle_sector_chi_to_vpp_metrics.json
  - output/public/quantum/mass_origin_same_sector_vpp_tiebreak_invariant_metrics.json

Outputs:
  - output/public/quantum/mass_origin_same_sector_tiebreak_target_bridge_metrics.json
  - output/public/quantum/mass_origin_same_sector_tiebreak_target_bridge_rows.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]

CONTRACT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_same_sector_chi_to_vpp_contract_metrics.json"
PROMOTION_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_positive_particle_sector_chi_to_vpp_metrics.json"
TIEBREAK_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_same_sector_vpp_tiebreak_invariant_metrics.json"
OUT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_same_sector_tiebreak_target_bridge_metrics.json"
OUT_CSV = ROOT / "output" / "public" / "quantum" / "mass_origin_same_sector_tiebreak_target_bridge_rows.csv"
DEFAULT_STEP_TAG = "8.7.55.2.86"


# 関数: `_utc_now_iso` の入出力契約と処理意図を定義する。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_parse_args` の入出力契約と処理意図を定義する。

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit whether a same-sector target value exists for the derivative-ratio tie-break invariant.",
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
    for path in (CONTRACT_JSON, PROMOTION_JSON, TIEBREAK_JSON):
        _require_path(path)

    contract = _read_json(CONTRACT_JSON)
    promotion = _read_json(PROMOTION_JSON)
    tiebreak = _read_json(TIEBREAK_JSON)

    contract_summary = contract.get("summary", {})
    contract_decision = contract.get("decision", {})
    promotion_summary = promotion.get("summary", {})
    tiebreak_summary = tiebreak.get("summary", {})

    same_particle_sector_only = bool(contract_summary.get("same_particle_sector_only", False))
    explicit_mapping_equation_present = bool(promotion_summary.get("explicit_mapping_equation_present", False))
    shell_family_row_ids = [str(item) for item in contract_summary.get("existing_shell_family_row_ids", [])]
    shell_anchor_count = len(shell_family_row_ids)
    shell_family_numerical_rows_present = bool(promotion_summary.get("shell_family_numerical_rows_present", False))
    tie_break_route_available = bool(tiebreak_summary.get("tie_break_route_available", False))
    surviving_candidate_ids = [str(item) for item in tiebreak_summary.get("surviving_candidate_ids", [])]
    invariant_name = str(tiebreak_summary.get("tiebreak_invariant_name", ""))
    invariant_values = tiebreak_summary.get("surviving_candidate_invariant_values", {})

    target_from_explicit_mapping_equation = explicit_mapping_equation_present
    target_from_shell_anchor_pack = False

    # 条件分岐: `shell_family_numerical_rows_present and shell_anchor_count >= 2` を満たす経路を評価する。
    if shell_family_numerical_rows_present and shell_anchor_count >= 2:
        target_from_shell_anchor_pack = False

    # 条件分岐: `target_from_explicit_mapping_equation` を満たす経路を評価する。

    if target_from_explicit_mapping_equation:
        target_source_kind_or_none: str | None = "explicit_mapping_equation"

    # 条件分岐: `target_from_shell_anchor_pack` を満たす経路を評価する。
    elif target_from_shell_anchor_pack:
        target_source_kind_or_none = "surviving_shell_anchor_pack"

    else:
        target_source_kind_or_none = None

    target_value_available = target_source_kind_or_none is not None
    bridge_without_new_free_parameters = target_value_available and same_particle_sector_only
    matching_candidate_ids = surviving_candidate_ids if target_value_available else []
    candidate_match_count = len(matching_candidate_ids)

    rows = [
        {
            "row_id": "same_sector_tiebreak_same_particle_sector_contract",
            "status": "pass" if same_particle_sector_only else "reject",
            "metric": "same-particle-sector contract remains available for tie-break bridge",
            "value": 1.0 if same_particle_sector_only else 0.0,
            "note": "The tie-break bridge may only use same-sector ingredients already frozen in the chi_P -> V'' contract.",
        },
        {
            "row_id": "same_sector_tiebreak_route_available",
            "status": "pass" if tie_break_route_available else "reject",
            "metric": "derivative-ratio tie-break route available",
            "value": 1.0 if tie_break_route_available else 0.0,
            "note": f"Current invariant `{invariant_name}` separates the surviving families with values {invariant_values}.",
        },
        {
            "row_id": "same_sector_tiebreak_target_from_explicit_mapping_equation",
            "status": "pass" if target_from_explicit_mapping_equation else "missing",
            "metric": "public same-sector target value derivable from explicit chi_P mapping equation",
            "value": 1.0 if target_from_explicit_mapping_equation else 0.0,
            "note": (
                "The explicit same-sector mapping equation is present and can in principle set the target invariant value."
                if target_from_explicit_mapping_equation
                else "The explicit chi_P -> V''(|P|_*) mapping equation is still absent, so the invariant target value cannot be derived from the observable-side contract."
            ),
        },
        {
            "row_id": "same_sector_tiebreak_target_from_shell_anchor_pack",
            "status": "pass" if target_from_shell_anchor_pack else "reject",
            "metric": "public same-sector target value derivable from surviving shell anchor pack",
            "value": 1.0 if target_from_shell_anchor_pack else 0.0,
            "note": (
                "The surviving shell anchor pack directly fixes the derivative-ratio target value."
                if target_from_shell_anchor_pack
                else f"The surviving shell anchors {shell_family_row_ids} provide kappa-like numerical anchors only; no public canonical row turns them into a target value of {invariant_name}."
            ),
        },
        {
            "row_id": "same_sector_tiebreak_target_value_available",
            "status": "pass" if target_value_available else "watch",
            "metric": "public same-sector target value for tie-break invariant available",
            "value": 1.0 if target_value_available else 0.0,
            "note": (
                f"Target value source is {target_source_kind_or_none}."
                if target_value_available
                else "A tie-break route exists, but no public canonical target value is yet available to choose between the surviving families."
            ),
        },
        {
            "row_id": "same_sector_tiebreak_bridge_without_new_free_parameters",
            "status": "pass" if bridge_without_new_free_parameters else "reject",
            "metric": "tie-break bridge closes without new free parameters",
            "value": 1.0 if bridge_without_new_free_parameters else 0.0,
            "note": (
                "The target value is derived from already-frozen same-sector ingredients only."
                if bridge_without_new_free_parameters
                else "No no-free-parameter bridge is available yet because the target value itself is still missing."
            ),
        },
        {
            "row_id": "same_sector_tiebreak_candidate_match_count",
            "status": "inventory",
            "metric": "candidate count matching the public target value",
            "value": float(candidate_match_count),
            "note": (
                f"Matching candidate ids are {matching_candidate_ids}."
                if matching_candidate_ids
                else "No candidate can be selected because no public target value exists yet."
            ),
        },
    ]

    return {
        "generated_utc": _utc_now_iso(),
        "phase": {
            "phase": 8,
            "step": step_tag,
            "name": "same-sector tie-break target bridge audit",
        },
        "inputs": {
            "mass_origin_same_sector_chi_to_vpp_contract_json": _relative_str(CONTRACT_JSON),
            "mass_origin_positive_particle_sector_chi_to_vpp_json": _relative_str(PROMOTION_JSON),
            "mass_origin_same_sector_vpp_tiebreak_invariant_json": _relative_str(TIEBREAK_JSON),
        },
        "intent": "Audit whether the derivative-ratio tie-break route can be grounded in a public same-sector target value without adding new free parameters.",
        "formulas": {
            "target_bridge_rule": "target_value_available iff either the explicit chi_P mapping equation or the surviving shell anchor pack yields a public canonical value for the invariant",
            "no_new_parameter_rule": "bridge_without_new_free_parameters iff the target value comes entirely from already-frozen same-sector ingredients",
        },
        "rows": rows,
        "summary": {
            "target_value_available": target_value_available,
            "target_source_kind_or_none": target_source_kind_or_none,
            "bridge_without_new_free_parameters": bridge_without_new_free_parameters,
            "candidate_match_count": candidate_match_count,
            "matching_candidate_ids": matching_candidate_ids,
            "shell_anchor_count": shell_anchor_count,
            "tie_break_route_available": tie_break_route_available,
            "tiebreak_invariant_name": invariant_name,
        },
        "decision": {
            "overall_status": "same_sector_tiebreak_target_bridge_frozen",
            "keep_mass_origin_branch_blocked": True,
            "target_value_available": target_value_available,
            "target_source_kind_or_none": target_source_kind_or_none,
            "bridge_without_new_free_parameters": bridge_without_new_free_parameters,
            "candidate_match_count": candidate_match_count,
            "matching_candidate_ids": matching_candidate_ids,
            "blocked_state_detail": str(contract_decision.get("blocked_state_detail", "")),
            "next_required_artifacts": [
                "same_sector_tiebreak_target_value",
                "single_public_vpp_shape",
                "positive_particle_sector_chi_p_to_vpp_public_artifact",
                "solver_ready_row_promoted_to_pass",
            ],
        },
        "evidence": {
            "contract_summary": contract_summary,
            "promotion_summary": promotion_summary,
            "tiebreak_summary": tiebreak_summary,
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
