#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_origin_missing_symbol_fragment_residual_branch_refresh.py

Step 8.7.55.2.240:
Reinject the residual missing-symbol-fragment route results into the blocked
second-route stack and refreeze whether handoff to 8.7.55.2.83-.84 is allowed.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

ROOT = Path(__file__).resolve().parents[2]

PRIOR_REFRESH_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_missing_terminal_glyph_residual_branch_refresh_metrics.json"
SPLIT_CONTRACT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_missing_symbol_fragment_residual_split_contract_metrics.json"
SHELL_ANCHOR_RETRY_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_shell_anchor_missing_symbol_fragment_closure_retry_metrics.json"
EXPLICIT_MAPPING_RETRY_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_explicit_mapping_missing_symbol_fragment_closure_retry_metrics.json"
OUT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_missing_symbol_fragment_residual_branch_refresh_metrics.json"
OUT_CSV = ROOT / "output" / "public" / "quantum" / "mass_origin_missing_symbol_fragment_residual_branch_refresh_rows.csv"
DEFAULT_STEP_TAG = "8.7.55.2.240"


# 関数: `_utc_now_iso` の入出力契約と処理意図を定義する。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_parse_args` の入出力契約と処理意図を定義する。

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refresh blocked second-route handoff eligibility after missing-symbol-fragment residual retries.",
    )
    parser.add_argument("--step-tag", default=DEFAULT_STEP_TAG, help="Roadmap step tag to stamp into the output payload.")
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


# 関数: `_ordered_unique` の入出力契約と処理意図を定義する。

def _ordered_unique(values: Iterable[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []

    for value in values:
        if value and value not in seen:
            seen.add(value)
            ordered.append(value)

    return ordered


# 関数: `_build_payload` の入出力契約と処理意図を定義する。

def _build_payload(step_tag: str) -> Dict[str, Any]:
    for path in (PRIOR_REFRESH_JSON, SPLIT_CONTRACT_JSON, SHELL_ANCHOR_RETRY_JSON, EXPLICIT_MAPPING_RETRY_JSON):
        _require_path(path)

    prior_refresh = _read_json(PRIOR_REFRESH_JSON)
    split_contract = _read_json(SPLIT_CONTRACT_JSON)
    shell_anchor_retry = _read_json(SHELL_ANCHOR_RETRY_JSON)
    explicit_mapping_retry = _read_json(EXPLICIT_MAPPING_RETRY_JSON)
    prior_refresh_summary = prior_refresh.get("summary", {})
    split_contract_summary = split_contract.get("summary", {})
    shell_anchor_retry_summary = shell_anchor_retry.get("summary", {})
    explicit_mapping_retry_summary = explicit_mapping_retry.get("summary", {})

    target_available = bool(shell_anchor_retry_summary.get("semantic_bridge_available", False))
    target_source_kind = "shell_anchor_missing_symbol_fragment_bridge" if target_available else None
    target_no_new = bool(shell_anchor_retry_summary.get("semantic_bridge_without_new_free_parameters", False))
    boundary_fixed = bool(prior_refresh_summary.get("single_public_boundary_family_fixed", False))
    shape_available = bool(prior_refresh_summary.get("single_public_vpp_shape_available", False) and target_available)
    selected_candidate = prior_refresh_summary.get("selected_candidate_id_or_none") if shape_available else None
    public_artifact = bool(explicit_mapping_retry_summary.get("explicit_mapping_equation_available", False) and shape_available)
    solver_ready = bool(boundary_fixed and target_available and shape_available and public_artifact)
    reopen_ready = solver_ready
    handoff = bool(reopen_ready)

    remaining_artifacts: List[str] = []
    if not target_available:
        remaining_artifacts.append("same_sector_tiebreak_target_value")

    if not shape_available:
        remaining_artifacts.append("single_public_vpp_shape")

    if not public_artifact:
        remaining_artifacts.append("positive_particle_sector_chi_p_to_vpp_public_artifact")

    if not solver_ready:
        remaining_artifacts.append("solver_ready_row_promoted_to_pass")

    remaining_artifacts = _ordered_unique(remaining_artifacts)

    remaining_blockers = _ordered_unique(
        [
            str(shell_anchor_retry_summary.get("shell_anchor_missing_symbol_fragment_nonclosure_reason_or_none") or ""),
            str(explicit_mapping_retry_summary.get("explicit_mapping_missing_symbol_fragment_nonclosure_reason_or_none") or ""),
        ]
    )

    rows: List[Dict[str, Any]] = [
        {
            "row_id": "missing_symbol_fragment_residual_branch_refresh_complete",
            "status": "pass",
            "metric": "missing-symbol-fragment residual branch refresh complete",
            "value": 1.0,
            "note": "This refresh reinjects symbol-fragment residual retries into the blocked second-route stack and refreezes .83-.84 handoff eligibility.",
        },
        {
            "row_id": "hand_off_to_8_7_55_2_83",
            "status": "pass" if handoff else "reject",
            "metric": "handoff to 8.7.55.2.83-.84 allowed after missing-symbol-fragment residual refresh",
            "value": 1.0 if handoff else 0.0,
            "note": (
                "Handoff to the discrete-spectrum pilot is now allowed."
                if handoff
                else "Handoff remains blocked because missing-symbol-fragment residual reinjection did not reopen the branch."
            ),
        },
        {
            "row_id": "missing_symbol_fragment_residual_branch_refresh_source_level_blocker_count",
            "status": "inventory",
            "metric": "remaining source-level blocker count after missing-symbol-fragment residual refresh",
            "value": float(len(remaining_blockers)),
            "note": f"Remaining source-level blockers are {remaining_blockers}.",
        },
    ]

    return {
        "generated_utc": _utc_now_iso(),
        "phase": {
            "phase": 8,
            "step": step_tag,
            "name": "missing-symbol-fragment residual branch refresh / handoff",
        },
        "inputs": {
            "mass_origin_missing_terminal_glyph_residual_branch_refresh_json": _relative_str(PRIOR_REFRESH_JSON),
            "mass_origin_missing_symbol_fragment_residual_split_contract_json": _relative_str(SPLIT_CONTRACT_JSON),
            "mass_origin_shell_anchor_missing_symbol_fragment_closure_retry_json": _relative_str(SHELL_ANCHOR_RETRY_JSON),
            "mass_origin_explicit_mapping_missing_symbol_fragment_closure_retry_json": _relative_str(EXPLICIT_MAPPING_RETRY_JSON),
        },
        "rows": rows,
        "summary": {
            "same_sector_tiebreak_target_value_available": target_available,
            "target_source_kind_or_none": target_source_kind,
            "target_value_bridge_without_new_free_parameters": target_no_new,
            "single_public_boundary_family_fixed": boundary_fixed,
            "single_public_vpp_shape_available": shape_available,
            "selected_candidate_id_or_none": selected_candidate,
            "positive_particle_sector_chi_p_to_vpp_public_artifact_available": public_artifact,
            "solver_ready_row_promoted_to_pass": solver_ready,
            "mass_origin_branch_reopen_ready": reopen_ready,
            "hand_off_to_8_7_55_2_83": handoff,
            "remaining_missing_artifacts": remaining_artifacts,
            "remaining_source_level_blockers": remaining_blockers,
            "symbol_fragment_split_contract_ready": split_contract_summary.get("split_contract_ready"),
        },
        "decision": {
            "overall_status": (
                "missing_symbol_fragment_residual_branch_refresh_reopen_ready"
                if handoff
                else "missing_symbol_fragment_residual_branch_refresh_still_blocked"
            ),
            "keep_mass_origin_branch_blocked": True,
            "same_sector_tiebreak_target_value_available": target_available,
            "single_public_vpp_shape_available": shape_available,
            "positive_particle_sector_chi_p_to_vpp_public_artifact_available": public_artifact,
            "mass_origin_branch_reopen_ready": reopen_ready,
            "hand_off_to_8_7_55_2_83": handoff,
            "remaining_missing_artifacts": remaining_artifacts,
            "remaining_source_level_blockers": remaining_blockers,
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
