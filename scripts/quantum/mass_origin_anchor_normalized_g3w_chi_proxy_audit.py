#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_origin_anchor_normalized_g3w_chi_proxy_audit.py

Step 8.7.55.2.255:
Audit whether the current public canonical pack already exposes a chi_* or
same-sector proxy rule that can be injected into the anchor-normalized g_3w
route without introducing a new free parameter.

Inputs:
  - output/public/quantum/mass_origin_anchor_normalized_g3w_chi_proxy_inventory_metrics.json

Outputs:
  - output/public/quantum/mass_origin_anchor_normalized_g3w_chi_proxy_audit_metrics.json
  - output/public/quantum/mass_origin_anchor_normalized_g3w_chi_proxy_audit_rows.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]

CHI_PROXY_INVENTORY_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_anchor_normalized_g3w_chi_proxy_inventory_metrics.json"
OUT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_anchor_normalized_g3w_chi_proxy_audit_metrics.json"
OUT_CSV = ROOT / "output" / "public" / "quantum" / "mass_origin_anchor_normalized_g3w_chi_proxy_audit_rows.csv"
DEFAULT_STEP_TAG = "8.7.55.2.255"


# 関数: `_utc_now_iso` の入出力契約と処理意図を定義する。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_parse_args` の入出力契約と処理意図を定義する。

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit whether a chi_* or same-sector proxy rule is already public canonical.",
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


# 関数: `_build_rows` の入出力契約と処理意図を定義する。

def _build_rows(
    *,
    chi_proxy_rule_available: bool,
    chi_proxy_kind_or_none: str | None,
    chi_proxy_without_new_free_parameters: bool,
    missing_inputs: List[str],
) -> List[Dict[str, Any]]:
    return [
        {
            "row_id": "anchor_normalized_g3w_chi_proxy_audit_complete",
            "status": "pass",
            "metric": "chi_* proxy audit complete",
            "value": 1.0,
            "note": "This step tests whether the current public pack already supports a usable chi_* or same-sector proxy rule.",
        },
        {
            "row_id": "anchor_normalized_g3w_chi_proxy_rule_available",
            "status": "pass" if chi_proxy_rule_available else "reject",
            "metric": "chi_* or same-sector proxy rule available",
            "value": 1.0 if chi_proxy_rule_available else 0.0,
            "note": (
                f"The current public pack already supports {chi_proxy_kind_or_none}."
                if chi_proxy_rule_available
                else "The current public pack still lacks an explicit chi_* datum or an equivalent same-sector proxy rule."
            ),
        },
        {
            "row_id": "anchor_normalized_g3w_chi_proxy_without_new_free_parameters",
            "status": "pass" if chi_proxy_without_new_free_parameters else "reject",
            "metric": "chi_* proxy rule stays inside no-new-free-parameter envelope",
            "value": 1.0 if chi_proxy_without_new_free_parameters else 0.0,
            "note": (
                "The current public pack already exposes a same-sector anchor-coordinate rule without a new fit."
                if chi_proxy_without_new_free_parameters
                else "Any chi_* proxy injection would still require a missing public anchor-coordinate datum."
            ),
        },
        {
            "row_id": "anchor_normalized_g3w_chi_proxy_missing_inputs",
            "status": "missing" if missing_inputs else "pass",
            "metric": "remaining missing inputs for chi_* proxy rule",
            "value": float(len(missing_inputs)),
            "note": f"Missing inputs: {missing_inputs}.",
        },
    ]


# 関数: `_build_payload` の入出力契約と処理意図を定義する。

def _build_payload(step_tag: str) -> Dict[str, Any]:
    _require_path(CHI_PROXY_INVENTORY_JSON)

    inventory = _read_json(CHI_PROXY_INVENTORY_JSON)
    inventory_summary = inventory.get("summary", {})
    inventory_decision = inventory.get("decision", {})
    inventory_evidence = inventory.get("evidence", {})
    curvature_summary = inventory_evidence.get("curvature_summary", {})
    rho_summary = inventory_evidence.get("rho_elimination_summary", {})
    residual_summary = inventory_evidence.get("rho_residual_summary", {})

    missing_inputs = [str(item) for item in inventory_summary.get("missing_chi_proxy_sources", [])]
    chi_proxy_rule_available = len(missing_inputs) == 0
    chi_proxy_kind_or_none = "explicit_chi_star_or_same_sector_proxy_rule" if chi_proxy_rule_available else None
    chi_proxy_without_new_free_parameters = chi_proxy_rule_available

    rows = _build_rows(
        chi_proxy_rule_available=chi_proxy_rule_available,
        chi_proxy_kind_or_none=chi_proxy_kind_or_none,
        chi_proxy_without_new_free_parameters=chi_proxy_without_new_free_parameters,
        missing_inputs=missing_inputs,
    )

    return {
        "generated_utc": _utc_now_iso(),
        "phase": {
            "phase": 8,
            "step": step_tag,
            "name": "chi_star proxy audit",
        },
        "inputs": {
            "mass_origin_anchor_normalized_g3w_chi_proxy_inventory_json": _relative_str(CHI_PROXY_INVENTORY_JSON),
        },
        "intent": "Determine whether the current public canonical pack already exposes a chi_* or same-sector proxy rule for the anchor-normalized g_3w route.",
        "formulas": {
            "proxy_injection_rule": "chi_proxy_rule_available iff the current public pack already exposes an explicit chi_* datum or an equivalent same-sector proxy value",
            "chi_space_bridge": "chi = ln(rho / P_ref), so rho_* elimination can proceed once the anchor branch coordinate chi_* (or an equivalent same-sector proxy) is fixed",
            "current_absence": "the current public pack freezes chi-space language and the rho_* reference symbol, but not the anchor-coordinate value itself",
        },
        "rows": rows,
        "summary": {
            "chi_proxy_rule_available": chi_proxy_rule_available,
            "chi_proxy_kind_or_none": chi_proxy_kind_or_none,
            "chi_proxy_without_new_free_parameters": chi_proxy_without_new_free_parameters,
            "chi_proxy_missing_inputs": missing_inputs,
        },
        "decision": {
            "overall_status": "anchor_normalized_g3w_chi_proxy_audit_frozen_absent",
            "keep_mass_origin_branch_blocked": True,
            "chi_proxy_rule_available": chi_proxy_rule_available,
            "chi_proxy_kind_or_none": chi_proxy_kind_or_none,
            "chi_proxy_without_new_free_parameters": chi_proxy_without_new_free_parameters,
            "hand_off_to_8_7_55_2_83": bool(inventory_decision.get("hand_off_to_8_7_55_2_83", False)),
            "next_required_artifacts": [
                "chi_star_or_same_sector_proxy",
                "rho_star_to_reference_ratio_rule",
                "anchor_normalized_g3w_public_value",
                "r3_target",
                "single_public_vpp_shape",
                "positive_particle_sector_chi_p_to_vpp_public_artifact",
                "solver_ready_row_promoted_to_pass",
            ],
        },
        "evidence": {
            "chi_proxy_inventory_summary": inventory_summary,
            "chi_proxy_inventory_decision": inventory_decision,
            "curvature_summary": curvature_summary,
            "rho_elimination_summary": rho_summary,
            "rho_residual_summary": residual_summary,
        },
    }


# 関数: `_write_csv` の入出力契約と処理意図を定義する。

def _write_csv(rows: List[Dict[str, Any]]) -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8", newline="") as handle:
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
