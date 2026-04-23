#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_origin_anchor_normalized_g3w_anchor_audit.py

Step 8.7.55.2.249:
Audit whether the current public canonical pack already supports an
anchor-normalization rule for g_3w without introducing a new free parameter.

Inputs:
  - output/public/quantum/mass_origin_anchor_normalized_g3w_source_inventory_metrics.json
  - output/public/quantum/mass_origin_anchor_local_shape_jet_metrics.json

Outputs:
  - output/public/quantum/mass_origin_anchor_normalized_g3w_anchor_audit_metrics.json
  - output/public/quantum/mass_origin_anchor_normalized_g3w_anchor_audit_rows.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]

SOURCE_INVENTORY_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_anchor_normalized_g3w_source_inventory_metrics.json"
SHAPE_JET_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_anchor_local_shape_jet_metrics.json"
OUT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_anchor_normalized_g3w_anchor_audit_metrics.json"
OUT_CSV = ROOT / "output" / "public" / "quantum" / "mass_origin_anchor_normalized_g3w_anchor_audit_rows.csv"
DEFAULT_STEP_TAG = "8.7.55.2.249"


# 関数: `_utc_now_iso` の入出力契約と処理意図を定義する。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_parse_args` の入出力契約と処理意図を定義する。

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit whether an anchor-normalization rule for g_3w is already public canonical.",
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
    anchor_normalization_rule_available: bool,
    anchor_normalization_kind_or_none: str | None,
    anchor_normalization_without_new_free_parameters: bool,
) -> List[Dict[str, Any]]:
    return [
        {
            "row_id": "anchor_normalized_g3w_anchor_audit_complete",
            "status": "pass",
            "metric": "anchor-normalized g_3w anchor audit complete",
            "value": 1.0,
            "note": "This step tests whether the current public pack already supports a symbolic anchor-normalization rule for g_3w.",
        },
        {
            "row_id": "anchor_normalized_g3w_anchor_rule_available",
            "status": "pass" if anchor_normalization_rule_available else "reject",
            "metric": "anchor normalization rule available",
            "value": 1.0 if anchor_normalization_rule_available else 0.0,
            "note": (
                f"The current public pack already supports the symbolic rule {anchor_normalization_kind_or_none}."
                if anchor_normalization_rule_available
                else "No public anchor-normalization rule for g_3w is yet available."
            ),
        },
        {
            "row_id": "anchor_normalized_g3w_anchor_rule_without_new_free_parameters",
            "status": "pass" if anchor_normalization_without_new_free_parameters else "reject",
            "metric": "anchor normalization rule stays inside no-new-free-parameter envelope",
            "value": 1.0 if anchor_normalization_without_new_free_parameters else 0.0,
            "note": (
                "The symbolic anchor normalization uses only the already-frozen g_3w and rho_*^2 V''(rho_*) identities."
                if anchor_normalization_without_new_free_parameters
                else "The anchor normalization would require an extra parameter or an extra observable."
            ),
        },
        {
            "row_id": "anchor_normalized_g3w_anchor_rule_still_needs_rho_star",
            "status": "watch" if anchor_normalization_rule_available else "reject",
            "metric": "anchor normalization alone still leaves rho_* unresolved",
            "value": 1.0 if anchor_normalization_rule_available else 0.0,
            "note": (
                "The symbolic anchor normalization reaches 2 g_3w / V''(rho_*) = V'''(rho_*) / V''(rho_*), but rho_* still has to be supplied before R_3 can close."
                if anchor_normalization_rule_available
                else "The route cannot yet reach the anchor-normalized symbolic quantity."
            ),
        },
    ]


# 関数: `_build_payload` の入出力契約と処理意図を定義する。

def _build_payload(step_tag: str) -> Dict[str, Any]:
    for path in (SOURCE_INVENTORY_JSON, SHAPE_JET_JSON):
        _require_path(path)

    inventory = _read_json(SOURCE_INVENTORY_JSON)
    shape_jet = _read_json(SHAPE_JET_JSON)

    inventory_summary = inventory.get("summary", {})
    shape_jet_summary = shape_jet.get("summary", {})

    present_sources = [str(item) for item in inventory_summary.get("present_g3w_route_sources", [])]
    anchor_normalization_rule_available = (
        "public_g3w_formula" in present_sources
        and "no_new_free_parameter_wording" in present_sources
        and "anchor_curvature_identity" in present_sources
        and "anchor_local_r3_definition" in present_sources
    )
    anchor_normalization_kind_or_none = "curvature_ratio_symbolic_rule" if anchor_normalization_rule_available else None
    anchor_normalization_without_new_free_parameters = anchor_normalization_rule_available
    rows = _build_rows(
        anchor_normalization_rule_available=anchor_normalization_rule_available,
        anchor_normalization_kind_or_none=anchor_normalization_kind_or_none,
        anchor_normalization_without_new_free_parameters=anchor_normalization_without_new_free_parameters,
    )

    return {
        "generated_utc": _utc_now_iso(),
        "phase": {
            "phase": 8,
            "step": step_tag,
            "name": "anchor normalization audit",
        },
        "inputs": {
            "mass_origin_anchor_normalized_g3w_source_inventory_json": _relative_str(SOURCE_INVENTORY_JSON),
            "mass_origin_anchor_local_shape_jet_json": _relative_str(SHAPE_JET_JSON),
        },
        "intent": "Determine whether the current public canonical formulas already define a symbolic anchor-normalized quantity built from g_3w and the anchor curvature scale.",
        "formulas": {
            "anchor_normalization_rule": "2 g_3w / V''(rho_*) = V'''(rho_*) / V''(rho_*)",
            "curvature_identity": "rho_*^2 V''(rho_*) = M_chi^2 omega_*^2",
            "audit_rule": "anchor_normalization_rule_available iff public g_3w, no-new-parameter wording, anchor curvature identity, and the anchor-local R_3 definition are all already present",
        },
        "rows": rows,
        "summary": {
            "anchor_normalization_rule_available": anchor_normalization_rule_available,
            "anchor_normalization_kind_or_none": anchor_normalization_kind_or_none,
            "anchor_normalization_without_new_free_parameters": anchor_normalization_without_new_free_parameters,
            "anchor_normalized_symbolic_rule": "2 g_3w / V''(rho_*) = V'''(rho_*) / V''(rho_*)" if anchor_normalization_rule_available else None,
            "rho_star_still_required_after_anchor_normalization": anchor_normalization_rule_available,
        },
        "decision": {
            "overall_status": "anchor_normalized_g3w_anchor_audit_frozen",
            "keep_mass_origin_branch_blocked": True,
            "anchor_normalization_rule_available": anchor_normalization_rule_available,
            "anchor_normalization_kind_or_none": anchor_normalization_kind_or_none,
            "anchor_normalization_without_new_free_parameters": anchor_normalization_without_new_free_parameters,
            "rho_star_still_required_after_anchor_normalization": anchor_normalization_rule_available,
            "hand_off_to_8_7_55_2_83": False,
            "next_required_artifacts": [
                "rho_star_elimination_rule",
                "anchor_normalized_g3w_public_value",
                "r3_target",
                "single_public_vpp_shape",
                "positive_particle_sector_chi_p_to_vpp_public_artifact",
                "solver_ready_row_promoted_to_pass",
            ],
        },
        "evidence": {
            "source_inventory_summary": inventory_summary,
            "shape_jet_summary": shape_jet_summary,
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
