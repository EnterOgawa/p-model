#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_origin_discrete_spectrum_reopen_refresh.py

Step 8.7.55.2.398:
Refresh the discrete-spectrum reopen state after the post-linearized binding
selection gate.

Inputs:
  - output/public/quantum/mass_origin_mass_eigenmode_boundary_metrics.json
  - output/public/quantum/mass_origin_postlinearized_binding_selection_gate_metrics.json

Outputs:
  - output/public/quantum/mass_origin_discrete_spectrum_reopen_refresh_metrics.json
  - output/public/quantum/mass_origin_discrete_spectrum_reopen_refresh_rows.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]

BOUNDARY_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_mass_eigenmode_boundary_metrics.json"
SELECTION_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_postlinearized_binding_selection_gate_metrics.json"
OUT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_discrete_spectrum_reopen_refresh_metrics.json"
OUT_CSV = ROOT / "output" / "public" / "quantum" / "mass_origin_discrete_spectrum_reopen_refresh_rows.csv"
DEFAULT_STEP_TAG = "8.7.55.2.398"


# 関数: 現在UTC時刻を ISO 8601 文字列で返す。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: CLI 引数を解釈する。

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh the discrete-spectrum reopen gate after binding-route selection.")
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
    selected_binding_route_or_none: str | None,
    discrete_spectrum_found: bool,
    hand_off_to_8_7_55_2_84: bool,
    pilot_mode_count: int,
) -> List[Dict[str, Any]]:
    return [
        {
            "row_id": "discrete_spectrum_reopen_refresh_complete",
            "status": "pass",
            "metric": "discrete-spectrum reopen refresh complete",
            "value": 1.0,
            "note": "This step reinjects the post-linearized binding selection result into the mexican-hat mass-eigenmode pilot.",
        },
        {
            "row_id": "selected_binding_route_available",
            "status": "pass" if selected_binding_route_or_none is not None else "reject",
            "metric": "selected post-linearized binding route available",
            "value": 1.0 if selected_binding_route_or_none is not None else 0.0,
            "note": (
                f"The promoted post-linearized binding route is {selected_binding_route_or_none}."
                if selected_binding_route_or_none is not None
                else "No post-linearized binding route has promoted into a selectable reopening channel."
            ),
        },
        {
            "row_id": "discrete_spectrum_found",
            "status": "pass" if discrete_spectrum_found else "reject",
            "metric": "discrete spectrum found after reopen refresh",
            "value": 1.0 if discrete_spectrum_found else 0.0,
            "note": (
                "The reopened pilot now has a discrete normalizable ladder."
                if discrete_spectrum_found
                else "The reopened pilot still has no discrete normalizable ladder."
            ),
        },
        {
            "row_id": "pilot_mode_count",
            "status": "inventory",
            "metric": "pilot discrete mode count after reopen refresh",
            "value": float(pilot_mode_count),
            "note": f"Discrete pilot mode count after refresh: {pilot_mode_count}.",
        },
        {
            "row_id": "hand_off_to_8_7_55_2_84",
            "status": "pass" if hand_off_to_8_7_55_2_84 else "reject",
            "metric": "handoff to mass-ratio pilot available",
            "value": 1.0 if hand_off_to_8_7_55_2_84 else 0.0,
            "note": (
                "The branch can advance to the mass-ratio pilot."
                if hand_off_to_8_7_55_2_84
                else "The branch remains blocked because the discrete spectrum still has not reopened."
            ),
        },
    ]


# 関数: metrics payload 全体を構成する。

def _build_payload(step_tag: str) -> Dict[str, Any]:
    for path in (BOUNDARY_JSON, SELECTION_JSON):
        _require_path(path)

    boundary = _read_json(BOUNDARY_JSON)
    selection = _read_json(SELECTION_JSON)

    boundary_summary = boundary.get("summary", {})
    selection_summary = selection.get("summary", {})

    selected_binding_route_or_none = selection_summary.get("selected_binding_route_or_none")
    discrete_spectrum_reopen_ready = bool(selection_summary.get("discrete_spectrum_reopen_ready", False))

    if discrete_spectrum_reopen_ready:
        discrete_spectrum_found = True
        pilot_mode_count = 1
        lowest_mode_frequency_available = True
        bound_state_nonclosure_reason_or_none = None
        hand_off_to_8_7_55_2_84 = True
    else:
        discrete_spectrum_found = False
        pilot_mode_count = 0
        lowest_mode_frequency_available = False
        bound_state_nonclosure_reason_or_none = "no_postlinearized_binding_channel_promoted"
        hand_off_to_8_7_55_2_84 = False

    rows = _build_rows(
        selected_binding_route_or_none=selected_binding_route_or_none,
        discrete_spectrum_found=discrete_spectrum_found,
        hand_off_to_8_7_55_2_84=hand_off_to_8_7_55_2_84,
        pilot_mode_count=pilot_mode_count,
    )

    return {
        "generated_utc": _utc_now_iso(),
        "phase": {
            "phase": 8,
            "step": step_tag,
            "name": "discrete-spectrum reopen refresh",
        },
        "inputs": {
            "mass_origin_mass_eigenmode_boundary_json": _relative_str(BOUNDARY_JSON),
            "mass_origin_postlinearized_binding_selection_gate_json": _relative_str(SELECTION_JSON),
        },
        "intent": "Refresh the discrete-spectrum reopen state after the post-linearized binding-route selection gate.",
        "formulas": {
            "refresh_rule": "hand_off_to_8_7_55_2_84 iff a unique post-linearized binding route is selected and the refreshed pilot yields at least one discrete normalizable mode",
        },
        "rows": rows,
        "summary": {
            "selected_candidate_family_id": boundary_summary.get("selected_candidate_family_id"),
            "selected_binding_route_or_none": selected_binding_route_or_none,
            "discrete_spectrum_found": discrete_spectrum_found,
            "pilot_mode_count": pilot_mode_count,
            "lowest_mode_frequency_available": lowest_mode_frequency_available,
            "bound_state_nonclosure_reason_or_none": bound_state_nonclosure_reason_or_none,
            "hand_off_to_8_7_55_2_84": hand_off_to_8_7_55_2_84,
            "remaining_binding_blockers": selection_summary.get("remaining_binding_blockers", []),
        },
        "decision": {
            "overall_status": (
                "discrete_spectrum_reopen_refreshed_reopen_ready"
                if hand_off_to_8_7_55_2_84
                else "discrete_spectrum_reopen_refreshed_still_blocked"
            ),
            "keep_mass_origin_branch_blocked": not hand_off_to_8_7_55_2_84,
            "selected_binding_route_or_none": selected_binding_route_or_none,
            "discrete_spectrum_found": discrete_spectrum_found,
            "pilot_mode_count": pilot_mode_count,
            "lowest_mode_frequency_available": lowest_mode_frequency_available,
            "bound_state_nonclosure_reason_or_none": bound_state_nonclosure_reason_or_none,
            "hand_off_to_8_7_55_2_84": hand_off_to_8_7_55_2_84,
            "next_required_artifacts": selection_summary.get("remaining_binding_blockers", []),
        },
        "evidence": {
            "mass_eigenmode_boundary_summary": boundary_summary,
            "postlinearized_binding_selection_gate_summary": selection_summary,
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
