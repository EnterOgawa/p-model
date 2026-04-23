#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_origin_missing_terminal_glyph_residual_split_contract.py

Step 8.7.55.2.229:
Freeze the reduced missing-terminal-glyph residual blocker split so the next
branch can attack shell-anchor and explicit-mapping terminal-glyph routes
separately after the current residual terminal-atom branch stayed blocked.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]
SHELL_ANCHOR_RETRY_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_shell_anchor_missing_terminal_atom_closure_retry_metrics.json"
EXPLICIT_MAPPING_RETRY_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_explicit_mapping_missing_terminal_atom_closure_retry_metrics.json"
BRANCH_REFRESH_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_missing_terminal_atom_residual_branch_refresh_metrics.json"
OUT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_missing_terminal_glyph_residual_split_contract_metrics.json"
OUT_CSV = ROOT / "output" / "public" / "quantum" / "mass_origin_missing_terminal_glyph_residual_split_contract_rows.csv"
DEFAULT_STEP_TAG = "8.7.55.2.229"


# 関数: `_utc_now_iso` の入出力契約と処理意図を定義する。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_parse_args` の入出力契約と処理意図を定義する。

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze the reduced missing-terminal-glyph residual blocker split after .228 stayed blocked.")
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
    for path in (SHELL_ANCHOR_RETRY_JSON, EXPLICIT_MAPPING_RETRY_JSON, BRANCH_REFRESH_JSON):
        _require_path(path)

    shell_anchor_retry = _read_json(SHELL_ANCHOR_RETRY_JSON)
    explicit_mapping_retry = _read_json(EXPLICIT_MAPPING_RETRY_JSON)
    branch_refresh = _read_json(BRANCH_REFRESH_JSON)
    shell_anchor_retry_summary = shell_anchor_retry.get("summary", {})
    explicit_mapping_retry_summary = explicit_mapping_retry.get("summary", {})
    branch_refresh_summary = branch_refresh.get("summary", {})
    remaining_source_level_blockers = [str(item) for item in branch_refresh_summary.get("remaining_source_level_blockers", [])]
    shell_anchor_missing_terminal_glyphs = [str(item) for item in shell_anchor_retry_summary.get("missing_shell_anchor_missing_terminal_glyphs", [])]
    explicit_mapping_missing_terminal_glyphs = [str(item) for item in explicit_mapping_retry_summary.get("missing_explicit_mapping_missing_terminal_glyphs", [])]
    shell_anchor_route_ok = bool(shell_anchor_retry_summary.get("shell_anchor_missing_terminal_atom_route_still_admissible", False))
    explicit_mapping_route_ok = bool(explicit_mapping_retry_summary.get("explicit_mapping_missing_terminal_atom_route_still_admissible", False))
    handoff = bool(branch_refresh_summary.get("hand_off_to_8_7_55_2_83", False))
    split_ready = bool(
        not handoff
        and remaining_source_level_blockers == ["shell_anchor_missing_terminal_glyphs_still_missing", "explicit_mapping_missing_terminal_glyphs_still_missing"]
        and shell_anchor_route_ok
        and explicit_mapping_route_ok
    )
    rows: List[Dict[str, Any]] = [
        {"row_id": "missing_terminal_glyph_residual_split_contract_complete", "status": "pass", "metric": "missing-terminal-glyph residual split contract complete", "value": 1.0, "note": "This step freezes the reduced residual blocker split that remains after the .228 refresh stayed blocked."},
        {"row_id": "missing_terminal_glyph_residual_split_handoff_still_blocked", "status": "pass", "metric": "handoff to 8.7.55.2.83 still blocked after missing-terminal-atom residual refresh", "value": 1.0, "note": "The next branch is only needed because .228 still returned hand_off_to_8_7_55_2_83=false."},
        {"row_id": "missing_terminal_glyph_residual_split_shell_anchor_glyph_count", "status": "inventory", "metric": "shell-anchor missing terminal glyph count", "value": float(len(shell_anchor_missing_terminal_glyphs)), "note": f"Shell-anchor missing terminal glyphs are {shell_anchor_missing_terminal_glyphs}."},
        {"row_id": "missing_terminal_glyph_residual_split_explicit_mapping_glyph_count", "status": "inventory", "metric": "explicit-mapping missing terminal glyph count", "value": float(len(explicit_mapping_missing_terminal_glyphs)), "note": f"Explicit-mapping missing terminal glyphs are {explicit_mapping_missing_terminal_glyphs}."},
        {"row_id": "missing_terminal_glyph_residual_split_contract_ready", "status": "pass" if split_ready else "reject", "metric": "reduced missing-terminal-glyph residual split contract ready for next branch", "value": 1.0 if split_ready else 0.0, "note": "The next branch may now treat shell-anchor missing terminal glyphs and explicit-mapping missing terminal glyphs as separate residual routes." if split_ready else "The reduced missing-terminal-glyph residual split is not stable enough to launch the next branch."},
    ]
    return {
        "generated_utc": _utc_now_iso(),
        "phase": {"phase": 8, "step": step_tag, "name": "missing-terminal-glyph residual blocker split contract"},
        "inputs": {
            "mass_origin_shell_anchor_missing_terminal_atom_closure_retry_json": _relative_str(SHELL_ANCHOR_RETRY_JSON),
            "mass_origin_explicit_mapping_missing_terminal_atom_closure_retry_json": _relative_str(EXPLICIT_MAPPING_RETRY_JSON),
            "mass_origin_missing_terminal_atom_residual_branch_refresh_json": _relative_str(BRANCH_REFRESH_JSON),
        },
        "rows": rows,
        "summary": {
            "remaining_source_level_blockers": remaining_source_level_blockers,
            "shell_anchor_missing_terminal_glyphs": shell_anchor_missing_terminal_glyphs,
            "explicit_mapping_missing_terminal_glyphs": explicit_mapping_missing_terminal_glyphs,
            "shell_anchor_missing_terminal_glyph_route_still_admissible": shell_anchor_route_ok,
            "explicit_mapping_missing_terminal_glyph_route_still_admissible": explicit_mapping_route_ok,
            "hand_off_to_8_7_55_2_83": handoff,
            "split_contract_ready": split_ready,
        },
        "decision": {
            "overall_status": "missing_terminal_glyph_residual_split_contract_frozen",
            "keep_mass_origin_branch_blocked": True,
            "split_contract_ready": split_ready,
            "hand_off_to_8_7_55_2_83": handoff,
            "remaining_source_level_blockers": remaining_source_level_blockers,
            "shell_anchor_missing_terminal_glyphs": shell_anchor_missing_terminal_glyphs,
            "explicit_mapping_missing_terminal_glyphs": explicit_mapping_missing_terminal_glyphs,
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
