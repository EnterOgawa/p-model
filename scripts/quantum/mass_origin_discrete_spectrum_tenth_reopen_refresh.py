#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_origin_discrete_spectrum_tenth_reopen_refresh.py

Step 8.7.55.2.474:
Refresh the discrete-spectrum reopen gate after the ninth reflective-cavity
retry is re-injected into the mexican-hat pilot.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]

MASS_EIGENMODE_BOUNDARY_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_mass_eigenmode_boundary_metrics.json"
REFLECTIVE_CAVITY_NINTH_RETRY_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_reflective_cavity_rule_ninth_retry_metrics.json"
OUT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_discrete_spectrum_tenth_reopen_refresh_metrics.json"
OUT_CSV = ROOT / "output" / "public" / "quantum" / "mass_origin_discrete_spectrum_tenth_reopen_refresh_rows.csv"
DEFAULT_STEP_TAG = "8.7.55.2.474"

# 関数: 現在UTC時刻を ISO 8601 文字列で返す。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

# 関数: CLI 引数を解釈する。

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh the discrete-spectrum reopen gate after the ninth reflective-cavity retry.")
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

def _build_rows(*, discrete_spectrum_found: bool, handoff: bool, nonclosure_reason: str | None) -> List[Dict[str, Any]]:
    return [
        {"row_id": "discrete_spectrum_tenth_reopen_refresh_complete", "status": "pass", "metric": "discrete-spectrum tenth reopen refresh complete", "value": 1.0, "note": "This step refreshes the discrete-spectrum gate after the ninth reflective-cavity retry."},
        {"row_id": "discrete_spectrum_found", "status": "pass" if discrete_spectrum_found else "reject", "metric": "discrete normalizable spectrum found after tenth reopen refresh", "value": 1.0 if discrete_spectrum_found else 0.0, "note": "At least one discrete normalizable mode is now fixed by the current public canonical pack." if discrete_spectrum_found else f"The mexican-hat pilot remains non-closing: {nonclosure_reason}."},
        {"row_id": "hand_off_to_8_7_55_2_84", "status": "pass" if handoff else "reject", "metric": "handoff to 8.7.55.2.84 available after tenth reopen refresh", "value": 1.0 if handoff else 0.0, "note": "The discrete-spectrum branch is now ready for the mass-ratio pilot." if handoff else "The discrete-spectrum branch remains blocked and cannot hand off to the mass-ratio pilot."},
    ]

# 関数: metrics payload 全体を構成する。

def _build_payload(step_tag: str) -> Dict[str, Any]:
    for path in (MASS_EIGENMODE_BOUNDARY_JSON, REFLECTIVE_CAVITY_NINTH_RETRY_JSON):
        _require_path(path)

    mass_eigenmode_boundary = _read_json(MASS_EIGENMODE_BOUNDARY_JSON)
    reflective_cavity_ninth_retry = _read_json(REFLECTIVE_CAVITY_NINTH_RETRY_JSON)
    mass_eigenmode_summary = mass_eigenmode_boundary.get("summary", {})
    reflective_cavity_summary = reflective_cavity_ninth_retry.get("summary", {})
    discrete_shell_cavity_ready = bool(reflective_cavity_summary.get("discrete_shell_cavity_ready", False))
    discrete_spectrum_found = bool(discrete_shell_cavity_ready and mass_eigenmode_summary.get("discrete_spectrum_found", False))
    pilot_mode_count = int(mass_eigenmode_summary.get("pilot_mode_count", 0)) if discrete_spectrum_found else 0
    lowest_mode_frequency_available = bool(discrete_spectrum_found and mass_eigenmode_summary.get("lowest_mode_frequency_available", False))
    selected_binding_route_or_none = "geometric_reflective_boundary" if discrete_shell_cavity_ready else None
    nonclosure_reason_or_none = None if discrete_shell_cavity_ready else "shell_quantization_reflective_cavity_rule_still_unavailable"
    handoff = bool(discrete_spectrum_found and lowest_mode_frequency_available)
    remaining_binding_blockers = [] if handoff else ["shell_quantization_domain_statement_symbol_fragment"]
    rows = _build_rows(discrete_spectrum_found=discrete_spectrum_found, handoff=handoff, nonclosure_reason=nonclosure_reason_or_none)
    return {
        "generated_utc": _utc_now_iso(),
        "phase": {"phase": 8, "step": step_tag, "name": "discrete-spectrum tenth reopen refresh"},
        "inputs": {"mass_origin_mass_eigenmode_boundary_json": _relative_str(MASS_EIGENMODE_BOUNDARY_JSON), "mass_origin_reflective_cavity_rule_ninth_retry_json": _relative_str(REFLECTIVE_CAVITY_NINTH_RETRY_JSON)},
        "intent": "Refresh the discrete-spectrum reopen gate after re-injecting the ninth reflective-cavity retry into the mexican-hat pilot.",
        "formulas": {"refresh_rule": "hand_off_to_8_7_55_2_84 iff the reflective cavity route closes and the refreshed pilot yields at least one discrete normalizable mode", "current_absence": "the reflective cavity route still lacks the shell-quantization domain-statement symbol fragment, so the mexican-hat pilot remains a continuum-threshold problem with no discrete handoff"},
        "rows": rows,
        "summary": {"selected_candidate_family_id": "mexican_hat", "selected_binding_route_or_none": selected_binding_route_or_none, "discrete_spectrum_found": discrete_spectrum_found, "pilot_mode_count": pilot_mode_count, "lowest_mode_frequency_available": lowest_mode_frequency_available, "bound_state_nonclosure_reason_or_none": nonclosure_reason_or_none, "hand_off_to_8_7_55_2_84": handoff, "remaining_binding_blockers": remaining_binding_blockers},
        "decision": {"overall_status": "discrete_spectrum_tenth_reopen_refresh_pass" if handoff else "discrete_spectrum_tenth_reopen_refreshed_still_blocked", "keep_mass_origin_branch_blocked": not handoff, "selected_binding_route_or_none": selected_binding_route_or_none, "discrete_spectrum_found": discrete_spectrum_found, "pilot_mode_count": pilot_mode_count, "lowest_mode_frequency_available": lowest_mode_frequency_available, "bound_state_nonclosure_reason_or_none": nonclosure_reason_or_none, "hand_off_to_8_7_55_2_84": handoff, "next_required_artifacts": remaining_binding_blockers},
        "evidence": {"mass_origin_mass_eigenmode_boundary_summary": mass_eigenmode_summary, "mass_origin_reflective_cavity_rule_ninth_retry_summary": reflective_cavity_summary},
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
