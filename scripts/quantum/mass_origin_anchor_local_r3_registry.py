#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_origin_anchor_local_r3_registry.py

Step 8.7.55.2.243:
Freeze the canonical anchor-local definition of

  R_3 = rho_* V'''(rho_*) / V''(rho_*)

and register the currently surviving same-sector candidate families with their
local cubicity values:

  - mexican_hat -> R_3 = 3
  - logarithmic -> R_3 = 1

This step does not yet fix a public target value for R_3. It only freezes the
definition, the chi-space expression used by the new roadmap, and the local
candidate registry that later steps will audit against public target-source
routes.

Inputs:
  - doc/paper/12_part3a_quantum_foundations.md
  - output/public/quantum/mass_origin_anchor_local_curvature_bridge_metrics.json
  - output/public/quantum/mass_origin_same_sector_vpp_tiebreak_invariant_metrics.json
  - output/public/quantum/mass_origin_single_public_vpp_shape_closure_metrics.json

Outputs:
  - output/public/quantum/mass_origin_anchor_local_r3_registry_metrics.json
  - output/public/quantum/mass_origin_anchor_local_r3_registry_rows.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]

PART3A_MD = ROOT / "doc" / "paper" / "12_part3a_quantum_foundations.md"
CURVATURE_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_anchor_local_curvature_bridge_metrics.json"
TIEBREAK_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_same_sector_vpp_tiebreak_invariant_metrics.json"
CLOSURE_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_single_public_vpp_shape_closure_metrics.json"
OUT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_anchor_local_r3_registry_metrics.json"
OUT_CSV = ROOT / "output" / "public" / "quantum" / "mass_origin_anchor_local_r3_registry_rows.csv"
DEFAULT_STEP_TAG = "8.7.55.2.243"


# 関数: `_utc_now_iso` の入出力契約と処理意図を定義する。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_parse_args` の入出力契約と処理意図を定義する。

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze the anchor-local R3 definition and candidate registry for the mass-origin route.",
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


# 関数: `_read_text` の入出力契約と処理意図を定義する。

def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# 関数: `_relative_str` の入出力契約と処理意図を定義する。

def _relative_str(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


# 関数: `_find_first_match` の入出力契約と処理意図を定義する。

def _find_first_match(text: str, pattern: str) -> Dict[str, Any] | None:
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        # 条件分岐: `pattern in raw_line` を満たす経路を評価する。
        if pattern in raw_line:
            return {
                "pattern": pattern,
                "line": line_number,
                "text": raw_line.strip(),
            }

    return None


# 関数: `_candidate_r3_value` の入出力契約と処理意図を定義する。

def _candidate_r3_value(family: str, ratio_values: Dict[str, Any]) -> float | None:
    raw_value = ratio_values.get(family)

    # 条件分岐: `raw_value is not None` を満たす経路を評価する。
    if raw_value is not None:
        return float(raw_value)

    # 条件分岐: `family == "mexican_hat"` を満たす経路を評価する。

    if family == "mexican_hat":
        return 3.0

    # 条件分岐: `family == "logarithmic"` を満たす経路を評価する。

    if family == "logarithmic":
        return 1.0

    return None


# 関数: `_build_rows` の入出力契約と処理意図を定義する。

def _build_rows(
    *,
    curvature_summary: Dict[str, Any],
    tiebreak_summary: Dict[str, Any],
    closure_summary: Dict[str, Any],
    background_hit: Dict[str, Any] | None,
    derivative_hit: Dict[str, Any] | None,
) -> List[Dict[str, Any]]:
    bridge_ready = bool(curvature_summary.get("vpp_closed_without_new_free_parameters", False))
    surviving_candidate_ids = [str(item) for item in tiebreak_summary.get("surviving_candidate_ids", [])]
    registry_values = {
        family: _candidate_r3_value(
            family,
            tiebreak_summary.get("surviving_candidate_invariant_values", {}),
        )
        for family in surviving_candidate_ids
    }
    unique_values = [
        value
        for value in registry_values.values()
        if isinstance(value, float)
    ]
    registry_unique = len(set(unique_values)) == len(unique_values) and len(unique_values) == len(surviving_candidate_ids)
    anchor_stationary_condition_vp_zero = bool(background_hit)

    rows: List[Dict[str, Any]] = [
        {
            "row_id": "anchor_local_r3_bridge_prerequisite_ready",
            "status": "pass" if bridge_ready else "reject",
            "metric": "anchor-local curvature bridge prerequisite already frozen",
            "value": 1.0 if bridge_ready else 0.0,
            "note": (
                "Step 8.7.55.2.242 already closed V'' without new free parameters, so the route can now register the local cubicity variable R_3."
                if bridge_ready
                else "The R_3 registry cannot be frozen until the anchor-local curvature bridge is closed."
            ),
        },
        {
            "row_id": "anchor_local_r3_definition_frozen",
            "status": "pass",
            "metric": "R_3 = rho_* V'''(rho_*) / V''(rho_*) frozen",
            "value": 1.0,
            "note": "The local cubicity variable is frozen as R_3 = rho_* V'''(rho_*) / V''(rho_*).",
        },
        {
            "row_id": "anchor_local_r3_chi_space_formula_ready",
            "status": "pass",
            "metric": "R_3 = 2 d ln omega_* / dchi |_*-3 frozen",
            "value": 1.0,
            "note": "The chi-space expression is frozen as R_3 = 2 (d ln omega_* / dchi)|_* - 3 for the later target-source audit.",
        },
        {
            "row_id": "anchor_local_anchor_stationary_condition_vp_zero",
            "status": "pass" if anchor_stationary_condition_vp_zero else "watch",
            "metric": "anchor stationary condition V'(rho_*) = 0 frozen",
            "value": 1.0 if anchor_stationary_condition_vp_zero else 0.0,
            "note": (
                f"Part III-A linearizes around the static background line {background_hit['line']}, so the anchor is treated as the stationary background point with V'(rho_*) = 0."
                if background_hit
                else "The static background wording was not located, so the anchor-stationary interpretation remains watch."
            ),
        },
        {
            "row_id": "anchor_local_r3_candidate_family_count",
            "status": "watch",
            "metric": "surviving candidate family count in R_3 registry",
            "value": float(len(surviving_candidate_ids)),
            "note": f"The current surviving same-sector shape classes are {surviving_candidate_ids}.",
        },
    ]

    for family in surviving_candidate_ids:
        value = registry_values.get(family)
        rows.append(
            {
                "row_id": f"anchor_local_r3_candidate_value_{family}",
                "status": "pass" if value is not None else "watch",
                "metric": f"registered R_3 value for {family}",
                "value": value if value is not None else 0.0,
                "note": (
                    f"The candidate registry freezes R_3 = {value} for {family}."
                    if value is not None
                    else f"The candidate registry does not yet contain an R_3 value for {family}."
                ),
            }
        )

    rows.extend(
        [
            {
                "row_id": "anchor_local_r3_candidate_registry_unique",
                "status": "pass" if registry_unique else "reject",
                "metric": "registered R_3 values separate the surviving families",
                "value": 1.0 if registry_unique else 0.0,
                "note": (
                    "The registered R_3 values are distinct across the surviving families, so a later target value would immediately choose one family."
                    if registry_unique
                    else "The registered R_3 values do not yet separate all surviving families."
                ),
            },
            {
                "row_id": "anchor_local_r3_target_value_observed",
                "status": "watch",
                "metric": "public canonical target value for R_3 already available",
                "value": 0.0,
                "note": "This step freezes the definition and registry only; the public canonical target value for R_3 is still missing and is deferred to the next audit step.",
            },
            {
                "row_id": "anchor_local_single_public_shape_still_open",
                "status": "watch",
                "metric": "single_public_vpp_shape already available after R_3 registry freeze",
                "value": 1.0 if closure_summary.get("single_public_vpp_shape_available", False) else 0.0,
                "note": (
                    "The shape has already closed to a single public family."
                    if closure_summary.get("single_public_vpp_shape_available", False)
                    else "The shape is still open because the registry does not yet contain a target value that can select between the surviving families."
                ),
            },
            {
                "row_id": "anchor_local_r3_registry_ready",
                "status": "pass" if bridge_ready and registry_unique else "watch",
                "metric": "anchor-local R_3 registry ready for target-source audit",
                "value": 1.0 if bridge_ready and registry_unique else 0.0,
                "note": (
                    "The next step can now audit which public route can supply the missing R_3 target value."
                    if bridge_ready and registry_unique
                    else "The registry is not yet ready for the target-source audit."
                ),
            },
        ]
    )

    # 条件分岐: `derivative_hit` を満たす経路を評価する。
    if derivative_hit:
        rows.append(
            {
                "row_id": "anchor_local_derivative_notation_public_canonical",
                "status": "pass",
                "metric": "Part III-A already exposes V_*^(n) notation",
                "value": 1.0,
                "note": f"Part III-A exposes derivative notation at line {derivative_hit['line']}: {derivative_hit['text']}",
            }
        )

    return rows


# 関数: `_build_payload` の入出力契約と処理意図を定義する。

def _build_payload(step_tag: str) -> Dict[str, Any]:
    for path in (PART3A_MD, CURVATURE_JSON, TIEBREAK_JSON, CLOSURE_JSON):
        _require_path(path)

    part3a_text = _read_text(PART3A_MD)
    curvature = _read_json(CURVATURE_JSON)
    tiebreak = _read_json(TIEBREAK_JSON)
    closure = _read_json(CLOSURE_JSON)

    curvature_summary = curvature.get("summary", {})
    curvature_decision = curvature.get("decision", {})
    tiebreak_summary = tiebreak.get("summary", {})
    tiebreak_decision = tiebreak.get("decision", {})
    closure_summary = closure.get("summary", {})

    background_hit = _find_first_match(part3a_text, "静的背景解 $P_{*}(x)>0$")
    derivative_hit = _find_first_match(part3a_text, "V_{*}^{(n)}\\equiv")

    surviving_candidate_ids = [str(item) for item in tiebreak_summary.get("surviving_candidate_ids", [])]
    candidate_family_r3_values = {
        family: _candidate_r3_value(
            family,
            tiebreak_summary.get("surviving_candidate_invariant_values", {}),
        )
        for family in surviving_candidate_ids
    }
    anchor_stationary_condition_vp_zero = bool(background_hit)
    r3_definition_frozen = True
    r3_chi_space_formula_ready = True
    r3_target_available = False
    registry_unique = len(set(candidate_family_r3_values.values())) == len(candidate_family_r3_values) and bool(
        candidate_family_r3_values
    )

    rows = _build_rows(
        curvature_summary=curvature_summary,
        tiebreak_summary=tiebreak_summary,
        closure_summary=closure_summary,
        background_hit=background_hit,
        derivative_hit=derivative_hit,
    )

    return {
        "generated_utc": _utc_now_iso(),
        "phase": {
            "phase": 8,
            "step": step_tag,
            "name": "R3 definition and candidate registry freeze",
        },
        "inputs": {
            "part3a_quantum_foundations_markdown": _relative_str(PART3A_MD),
            "mass_origin_anchor_local_curvature_bridge_json": _relative_str(CURVATURE_JSON),
            "mass_origin_same_sector_vpp_tiebreak_invariant_json": _relative_str(TIEBREAK_JSON),
            "mass_origin_single_public_vpp_shape_closure_json": _relative_str(CLOSURE_JSON),
        },
        "intent": "Freeze the canonical anchor-local definition of R_3 and the local cubicity registry for the surviving same-sector V(|P|) candidates before auditing any public target-source route.",
        "formulas": {
            "rho_definition": "rho = |P|",
            "anchor_stationary_condition": "V'(rho_*) = 0",
            "r3_rho_space_definition": "R_3 = rho_* V'''(rho_*) / V''(rho_*)",
            "u3_translation": "U'''(chi_*) = rho_*^2 V''(rho_*) (3 + R_3)",
            "r3_chi_space_formula": "R_3 = 2 (d ln omega_* / dchi)|_* - 3",
            "candidate_registry_rule": "mexican_hat -> R_3 = 3; logarithmic -> R_3 = 1",
        },
        "rows": rows,
        "summary": {
            "r3_definition_frozen": r3_definition_frozen,
            "r3_chi_space_formula_ready": r3_chi_space_formula_ready,
            "anchor_stationary_condition_vp_zero": anchor_stationary_condition_vp_zero,
            "candidate_family_ids": surviving_candidate_ids,
            "candidate_family_r3_values": candidate_family_r3_values,
            "r3_registry_unique_across_surviving_candidates": registry_unique,
            "r3_target_available": r3_target_available,
            "single_public_vpp_shape_available": bool(closure_summary.get("single_public_vpp_shape_available", False)),
            "positive_particle_sector_chi_p_to_vpp_public_artifact_available": bool(
                curvature_summary.get("positive_particle_sector_chi_p_to_vpp_public_artifact_available", False)
            ),
        },
        "decision": {
            "overall_status": "anchor_local_r3_registry_frozen",
            "keep_mass_origin_branch_blocked": True,
            "r3_definition_frozen": r3_definition_frozen,
            "r3_chi_space_formula_ready": r3_chi_space_formula_ready,
            "anchor_stationary_condition_vp_zero": anchor_stationary_condition_vp_zero,
            "candidate_family_ids": surviving_candidate_ids,
            "candidate_family_r3_values": candidate_family_r3_values,
            "r3_target_available": r3_target_available,
            "single_public_vpp_shape_available": bool(closure_summary.get("single_public_vpp_shape_available", False)),
            "positive_particle_sector_chi_p_to_vpp_public_artifact_available": bool(
                curvature_summary.get("positive_particle_sector_chi_p_to_vpp_public_artifact_available", False)
            ),
            "blocked_state_detail": str(curvature_decision.get("blocked_state_detail", "")),
            "next_required_artifacts": [
                "r3_target",
                "single_public_vpp_shape",
                "positive_particle_sector_chi_p_to_vpp_public_artifact",
                "solver_ready_row_promoted_to_pass",
            ],
        },
        "evidence": {
            "part3a_static_background_line": background_hit,
            "part3a_derivative_notation_line": derivative_hit,
            "curvature_bridge_summary": curvature_summary,
            "curvature_bridge_decision": curvature_decision,
            "tiebreak_summary": tiebreak_summary,
            "tiebreak_decision": tiebreak_decision,
            "closure_summary": closure_summary,
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
