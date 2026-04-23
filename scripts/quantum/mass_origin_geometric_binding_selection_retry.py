#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_origin_geometric_binding_selection_retry.py

Step 8.7.55.2.403:
Reinject the reflective-cavity retry result into the post-linearized binding
selection gate and decide whether the geometric route can now be selected as a
unique reopening channel.

Inputs:
  - output/public/quantum/mass_origin_geometric_boundary_residual_route_contract_metrics.json
  - output/public/quantum/mass_origin_reflective_cavity_rule_retry_metrics.json

Outputs:
  - output/public/quantum/mass_origin_geometric_binding_selection_retry_metrics.json
  - output/public/quantum/mass_origin_geometric_binding_selection_retry_rows.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]

ROUTE_CONTRACT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_geometric_boundary_residual_route_contract_metrics.json"
REFLECTIVE_CAVITY_RETRY_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_reflective_cavity_rule_retry_metrics.json"
OUT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_geometric_binding_selection_retry_metrics.json"
OUT_CSV = ROOT / "output" / "public" / "quantum" / "mass_origin_geometric_binding_selection_retry_rows.csv"
DEFAULT_STEP_TAG = "8.7.55.2.403"


# 関数: 現在UTC時刻を ISO 8601 文字列で返す。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: CLI 引数を解釈する。

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retry geometric binding-route selection after reflective-cavity closure.")
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
    discrete_spectrum_reopen_ready: bool,
    remaining_binding_blockers: List[str],
) -> List[Dict[str, Any]]:
    return [
        {
            "row_id": "geometric_binding_selection_retry_complete",
            "status": "pass",
            "metric": "geometric binding selection retry complete",
            "value": 1.0,
            "note": "This step reinjects the reflective-cavity retry into the post-linearized binding selection gate.",
        },
        {
            "row_id": "selected_binding_route_unique",
            "status": "pass" if selected_binding_route_or_none is not None else "reject",
            "metric": "selected geometric binding route unique after retry",
            "value": 1.0 if selected_binding_route_or_none is not None else 0.0,
            "note": (
                f"The unique selected route is {selected_binding_route_or_none}."
                if selected_binding_route_or_none is not None
                else "The geometric route still has not promoted into a unique reopening channel."
            ),
        },
        {
            "row_id": "discrete_spectrum_reopen_ready",
            "status": "pass" if discrete_spectrum_reopen_ready else "reject",
            "metric": "discrete-spectrum reopen ready after geometric retry",
            "value": 1.0 if discrete_spectrum_reopen_ready else 0.0,
            "note": (
                "The geometric route now defines a unique no-new-free-parameter reopening channel."
                if discrete_spectrum_reopen_ready
                else "The geometric route remains blocked and discrete-spectrum reopen is still unavailable."
            ),
        },
        {
            "row_id": "remaining_binding_blocker_count",
            "status": "watch" if remaining_binding_blockers else "pass",
            "metric": "remaining binding blocker count after geometric retry",
            "value": float(len(remaining_binding_blockers)),
            "note": f"Remaining binding blockers: {remaining_binding_blockers}.",
        },
    ]


# 関数: metrics payload 全体を構成する。

def _build_payload(step_tag: str) -> Dict[str, Any]:
    for path in (ROUTE_CONTRACT_JSON, REFLECTIVE_CAVITY_RETRY_JSON):
        _require_path(path)

    route_contract = _read_json(ROUTE_CONTRACT_JSON)
    reflective_cavity_retry = _read_json(REFLECTIVE_CAVITY_RETRY_JSON)

    route_contract_summary = route_contract.get("summary", {})
    reflective_cavity_retry_summary = reflective_cavity_retry.get("summary", {})

    geometric_boundary_promotion_rule_available = bool(reflective_cavity_retry_summary.get("geometric_boundary_promotion_rule_available", False))
    selected_binding_route_or_none = "geometric_reflective_boundary" if geometric_boundary_promotion_rule_available else None
    binding_route_unique = selected_binding_route_or_none is not None
    no_new_free_parameter_binding_channel = binding_route_unique
    discrete_spectrum_reopen_ready = binding_route_unique
    geometric_route_still_admissible = True
    residual_preferred_route_or_none = "geometric_reflective_boundary"
    remaining_binding_blockers = [] if geometric_boundary_promotion_rule_available else ["boundary_radius_or_domain_proxy"]
    selection_retry_nonclosure_reason_or_none = None if geometric_boundary_promotion_rule_available else "shell_quantization_reflective_cavity_rule_unavailable"

    rows = _build_rows(
        selected_binding_route_or_none=selected_binding_route_or_none,
        discrete_spectrum_reopen_ready=discrete_spectrum_reopen_ready,
        remaining_binding_blockers=remaining_binding_blockers,
    )

    return {
        "generated_utc": _utc_now_iso(),
        "phase": {
            "phase": 8,
            "step": step_tag,
            "name": "geometric binding selection retry",
        },
        "inputs": {
            "mass_origin_geometric_boundary_residual_route_contract_json": _relative_str(ROUTE_CONTRACT_JSON),
            "mass_origin_reflective_cavity_rule_retry_json": _relative_str(REFLECTIVE_CAVITY_RETRY_JSON),
        },
        "intent": "Decide whether the geometric reflective-boundary route can now be selected as the unique post-linearized reopening channel.",
        "formulas": {
            "selection_retry_rule": "the geometric route is selected iff the reflective cavity rule closes without a new scale or extra field content",
            "current_absence": "the reflective cavity rule remains unavailable because no cavity radius or domain proxy is yet public canonical",
        },
        "rows": rows,
        "summary": {
            "selected_residual_binding_route_or_none": route_contract_summary.get("selected_residual_binding_route_or_none"),
            "selected_binding_route_or_none": selected_binding_route_or_none,
            "binding_route_unique": binding_route_unique,
            "no_new_free_parameter_binding_channel": no_new_free_parameter_binding_channel,
            "discrete_spectrum_reopen_ready": discrete_spectrum_reopen_ready,
            "geometric_route_still_admissible": geometric_route_still_admissible,
            "residual_preferred_route_or_none": residual_preferred_route_or_none,
            "remaining_binding_blockers": remaining_binding_blockers,
            "selection_retry_nonclosure_reason_or_none": selection_retry_nonclosure_reason_or_none,
        },
        "decision": {
            "overall_status": "geometric_binding_selection_retry_frozen",
            "keep_mass_origin_branch_blocked": not discrete_spectrum_reopen_ready,
            "selected_binding_route_or_none": selected_binding_route_or_none,
            "binding_route_unique": binding_route_unique,
            "no_new_free_parameter_binding_channel": no_new_free_parameter_binding_channel,
            "discrete_spectrum_reopen_ready": discrete_spectrum_reopen_ready,
            "geometric_route_still_admissible": geometric_route_still_admissible,
            "residual_preferred_route_or_none": residual_preferred_route_or_none,
            "remaining_binding_blockers": remaining_binding_blockers,
            "selection_retry_nonclosure_reason_or_none": selection_retry_nonclosure_reason_or_none,
        },
        "evidence": {
            "geometric_boundary_residual_route_contract_summary": route_contract_summary,
            "reflective_cavity_rule_retry_summary": reflective_cavity_retry_summary,
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
