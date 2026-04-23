#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_origin_r3_target_source_audit.py

Step 8.7.55.2.244:
Audit the three minimum routes that could supply a public canonical target
value for the anchor-local cubicity variable

  R_3 = rho_* V'''(rho_*) / V''(rho_*)

The audit compares:
  1. shell-anchor slope route: S_* = (d ln omega_* / dchi)|_*
  2. anchor-normalized three-wave route: g_3w with rho_* eliminated
  3. non-Gaussian kurtosis route: experimental access to an R_3^2-sensitive
     extension of the already-closed phase-diffusion bridge

Inputs:
  - doc/paper/12_part3a_quantum_foundations.md
  - output/public/quantum/mass_origin_anchor_local_curvature_bridge_metrics.json
  - output/public/quantum/mass_origin_anchor_local_r3_registry_metrics.json
  - output/public/quantum/mass_origin_same_sector_tiebreak_target_bridge_metrics.json
  - output/public/quantum/mass_origin_shell_anchor_target_synthesis_metrics.json
  - output/public/quantum/entanglement_source_dynamics_three_wave_mixing_metrics.json
  - output/public/quantum/born_phase_diffusion_audit_metrics.json

Outputs:
  - output/public/quantum/mass_origin_r3_target_source_audit_metrics.json
  - output/public/quantum/mass_origin_r3_target_source_audit_rows.csv
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
R3_REGISTRY_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_anchor_local_r3_registry_metrics.json"
TIEBREAK_BRIDGE_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_same_sector_tiebreak_target_bridge_metrics.json"
SHELL_SYNTHESIS_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_shell_anchor_target_synthesis_metrics.json"
THREE_WAVE_JSON = ROOT / "output" / "public" / "quantum" / "entanglement_source_dynamics_three_wave_mixing_metrics.json"
BORN_JSON = ROOT / "output" / "public" / "quantum" / "born_phase_diffusion_audit_metrics.json"
OUT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_r3_target_source_audit_metrics.json"
OUT_CSV = ROOT / "output" / "public" / "quantum" / "mass_origin_r3_target_source_audit_rows.csv"
DEFAULT_STEP_TAG = "8.7.55.2.244"


# 関数: `_utc_now_iso` の入出力契約と処理意図を定義する。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_parse_args` の入出力契約と処理意図を定義する。

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit the minimum public routes that could fix the missing R3 target value.",
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


# 関数: `_build_rows` の入出力契約と処理意図を定義する。

def _build_rows(
    *,
    curvature_summary: Dict[str, Any],
    r3_registry_summary: Dict[str, Any],
    tiebreak_bridge_summary: Dict[str, Any],
    shell_synthesis_summary: Dict[str, Any],
    born_summary: Dict[str, Any],
    born_decision: Dict[str, Any],
    three_wave_decision: Dict[str, Any],
    g3w_formula_hit: Dict[str, Any] | None,
    g3w_not_new_parameter_hit: Dict[str, Any] | None,
    shell_anchor_slope_route_ready: bool,
    g3w_anchor_normalized_route_ready: bool,
    kurtosis_route_feasible: bool,
    preferred_r3_route_or_none: str | None,
    remaining_r3_missing_datum_or_none: str | None,
) -> List[Dict[str, Any]]:
    rows = [
        {
            "row_id": "r3_target_source_audit_complete",
            "status": "pass",
            "metric": "R_3 target-source audit complete",
            "value": 1.0,
            "note": "This step compares the shell-anchor slope route, the anchor-normalized g_3w route, and the non-Gaussian kurtosis route.",
        },
        {
            "row_id": "r3_target_source_prerequisite_registry_ready",
            "status": "pass" if curvature_summary.get("vpp_closed_without_new_free_parameters", False) and r3_registry_summary.get("r3_registry_unique_across_surviving_candidates", False) else "reject",
            "metric": "anchor-local curvature bridge and R_3 registry are ready",
            "value": 1.0 if curvature_summary.get("vpp_closed_without_new_free_parameters", False) and r3_registry_summary.get("r3_registry_unique_across_surviving_candidates", False) else 0.0,
            "note": "The target-source audit is only meaningful because V'' is already closed and the candidate registry is already unique across the surviving families.",
        },
        {
            "row_id": "r3_target_source_shell_anchor_route_admissible",
            "status": "pass" if bool(tiebreak_bridge_summary.get("tie_break_route_available", False)) else "reject",
            "metric": "shell-anchor route remains admissible in principle",
            "value": 1.0 if bool(tiebreak_bridge_summary.get("tie_break_route_available", False)) else 0.0,
            "note": "The shell-anchor pack still separates the surviving families in principle, but it must supply a same-sector slope datum before it can fix R_3.",
        },
        {
            "row_id": "r3_target_source_shell_anchor_slope_route_ready",
            "status": "pass" if shell_anchor_slope_route_ready else "reject",
            "metric": "shell-anchor slope route ready",
            "value": 1.0 if shell_anchor_slope_route_ready else 0.0,
            "note": (
                "The shell-anchor pack already carries a universal same-sector branch with a chi_* proxy, so S_* = (d ln omega_* / dchi)|_* can be read directly."
                if shell_anchor_slope_route_ready
                else "Existing shell-anchor artifacts still expose only kappa-like numeric anchors; they do not yet publish a universal same-sector branch or chi_* proxy that would turn the pack into S_* = (d ln omega_* / dchi)|_*."
            ),
        },
        {
            "row_id": "r3_target_source_g3w_formula_public_canonical",
            "status": "pass" if g3w_formula_hit and g3w_not_new_parameter_hit else "watch",
            "metric": "Part III-A already exposes g_3w = V_*^(3) / 2 as a non-new-parameter relation",
            "value": 1.0 if g3w_formula_hit and g3w_not_new_parameter_hit else 0.0,
            "note": (
                f"Part III-A exposes g_3w at line {g3w_formula_hit['line']} and states at line {g3w_not_new_parameter_hit['line']} that it is not a new P-model parameter."
                if g3w_formula_hit and g3w_not_new_parameter_hit
                else "The current public canonical pack does not yet expose both the g_3w formula and the no-new-parameter wording."
            ),
        },
        {
            "row_id": "r3_target_source_g3w_route_admissible",
            "status": "pass" if bool(three_wave_decision.get("new_pmodel_free_parameters_introduced", False) is False) else "reject",
            "metric": "g_3w route remains admissible without new model parameters",
            "value": 1.0 if bool(three_wave_decision.get("new_pmodel_free_parameters_introduced", False) is False) else 0.0,
            "note": "The public three-wave source dynamics already treats g_3w as shorthand for an existing derivative rather than a new fit parameter.",
        },
        {
            "row_id": "r3_target_source_g3w_anchor_normalized_route_ready",
            "status": "pass" if g3w_anchor_normalized_route_ready else "reject",
            "metric": "anchor-normalized g_3w route ready",
            "value": 1.0 if g3w_anchor_normalized_route_ready else 0.0,
            "note": (
                "A public anchor-normalized g_3w value is already available and rho_* has been eliminated, so the route can fix R_3 directly."
                if g3w_anchor_normalized_route_ready
                else "The g_3w formula is public, but the pack still lacks an anchor-normalized public value or an explicit rho_* elimination, so the route cannot yet fix R_3."
            ),
        },
        {
            "row_id": "r3_target_source_kurtosis_markov_regime_ready",
            "status": "pass" if born_decision.get("markov_closure_status") == "closed_after_carrier_removal" else "reject",
            "metric": "phase-diffusion / Markov regime already closed for a non-Gaussian extension",
            "value": 1.0 if born_decision.get("markov_closure_status") == "closed_after_carrier_removal" else 0.0,
            "note": "The carrier-removed phase-diffusion route is already closed, so a non-Gaussian extension would not reopen the basic Markov / susceptibility contract.",
        },
        {
            "row_id": "r3_target_source_kurtosis_route_feasible",
            "status": "pass" if kurtosis_route_feasible else "watch",
            "metric": "non-Gaussian kurtosis route feasible",
            "value": 1.0 if kurtosis_route_feasible else 0.0,
            "note": (
                "Inference from the already-closed phase-diffusion grid: probe rows with Gamma_deph T_obs > 1 exist, so an R_3^2-sensitive non-Gaussian extension is experimentally feasible in principle."
                if kurtosis_route_feasible
                else "The current public pack does not yet demonstrate an experimentally viable non-Gaussian regime for the kurtosis route."
            ),
        },
        {
            "row_id": "r3_target_source_preferred_route",
            "status": "watch" if preferred_r3_route_or_none else "reject",
            "metric": "preferred next route to close R_3 target",
            "value": 1.0 if preferred_r3_route_or_none else 0.0,
            "note": (
                f"The preferred next-closing route is {preferred_r3_route_or_none}."
                if preferred_r3_route_or_none
                else "No preferred route can yet be selected because every route is simultaneously blocked."
            ),
        },
        {
            "row_id": "r3_target_source_remaining_missing_datum",
            "status": "missing" if remaining_r3_missing_datum_or_none else "pass",
            "metric": "remaining missing datum after route comparison",
            "value": 0.0 if remaining_r3_missing_datum_or_none else 1.0,
            "note": (
                f"The remaining datum is {remaining_r3_missing_datum_or_none}."
                if remaining_r3_missing_datum_or_none
                else "No missing datum remains after the route comparison."
            ),
        },
    ]
    return rows


# 関数: `_build_payload` の入出力契約と処理意図を定義する。

def _build_payload(step_tag: str) -> Dict[str, Any]:
    for path in (
        PART3A_MD,
        CURVATURE_JSON,
        R3_REGISTRY_JSON,
        TIEBREAK_BRIDGE_JSON,
        SHELL_SYNTHESIS_JSON,
        THREE_WAVE_JSON,
        BORN_JSON,
    ):
        _require_path(path)

    part3a_text = _read_text(PART3A_MD)
    curvature = _read_json(CURVATURE_JSON)
    r3_registry = _read_json(R3_REGISTRY_JSON)
    tiebreak_bridge = _read_json(TIEBREAK_BRIDGE_JSON)
    shell_synthesis = _read_json(SHELL_SYNTHESIS_JSON)
    three_wave = _read_json(THREE_WAVE_JSON)
    born = _read_json(BORN_JSON)

    curvature_summary = curvature.get("summary", {})
    curvature_decision = curvature.get("decision", {})
    r3_registry_summary = r3_registry.get("summary", {})
    tiebreak_bridge_summary = tiebreak_bridge.get("summary", {})
    tiebreak_bridge_decision = tiebreak_bridge.get("decision", {})
    shell_synthesis_summary = shell_synthesis.get("summary", {})
    shell_synthesis_decision = shell_synthesis.get("decision", {})
    three_wave_summary = three_wave.get("summary", {})
    three_wave_decision = three_wave.get("decision", {})
    born_summary = born.get("summary", {})
    born_decision = born.get("decision", {})

    g3w_formula_hit = _find_first_match(part3a_text, "g_{3\\mathrm{w}}\\equiv \\frac{1}{2}V_{*}^{(3)}")
    g3w_not_new_parameter_hit = _find_first_match(part3a_text, "第1に、$g_{3\\mathrm{w}}$ は新しい P-model パラメータではなく")

    shell_anchor_slope_route_ready = False
    g3w_anchor_normalized_route_ready = False
    kurtosis_route_feasible = bool(born_summary.get("probe_rows_with_phase_mixing", 0)) > 0 and bool(
        born_decision.get("markov_closure_status") == "closed_after_carrier_removal"
    )

    preferred_r3_route_or_none: str | None = None
    remaining_r3_missing_datum_or_none: str | None = None

    # 条件分岐: `g3w_formula_hit and g3w_not_new_parameter_hit and not g3w_anchor_normalized_route_ready` を満たす経路を評価する。
    if g3w_formula_hit and g3w_not_new_parameter_hit and not g3w_anchor_normalized_route_ready:
        preferred_r3_route_or_none = "g3w_anchor_normalized_route"
        remaining_r3_missing_datum_or_none = "anchor_normalized_g3w_public_value"

    # 条件分岐: `preferred_r3_route_or_none is None and kurtosis_route_feasible` を満たす経路を評価する。

    if preferred_r3_route_or_none is None and kurtosis_route_feasible:
        preferred_r3_route_or_none = "kurtosis_route"
        remaining_r3_missing_datum_or_none = "anchor_normalized_phase_kurtosis_measurement"

    rows = _build_rows(
        curvature_summary=curvature_summary,
        r3_registry_summary=r3_registry_summary,
        tiebreak_bridge_summary=tiebreak_bridge_summary,
        shell_synthesis_summary=shell_synthesis_summary,
        born_summary=born_summary,
        born_decision=born_decision,
        three_wave_decision=three_wave_decision,
        g3w_formula_hit=g3w_formula_hit,
        g3w_not_new_parameter_hit=g3w_not_new_parameter_hit,
        shell_anchor_slope_route_ready=shell_anchor_slope_route_ready,
        g3w_anchor_normalized_route_ready=g3w_anchor_normalized_route_ready,
        kurtosis_route_feasible=kurtosis_route_feasible,
        preferred_r3_route_or_none=preferred_r3_route_or_none,
        remaining_r3_missing_datum_or_none=remaining_r3_missing_datum_or_none,
    )

    return {
        "generated_utc": _utc_now_iso(),
        "phase": {
            "phase": 8,
            "step": step_tag,
            "name": "R3 target-source audit",
        },
        "inputs": {
            "part3a_quantum_foundations_markdown": _relative_str(PART3A_MD),
            "mass_origin_anchor_local_curvature_bridge_json": _relative_str(CURVATURE_JSON),
            "mass_origin_anchor_local_r3_registry_json": _relative_str(R3_REGISTRY_JSON),
            "mass_origin_same_sector_tiebreak_target_bridge_json": _relative_str(TIEBREAK_BRIDGE_JSON),
            "mass_origin_shell_anchor_target_synthesis_json": _relative_str(SHELL_SYNTHESIS_JSON),
            "entanglement_source_dynamics_three_wave_mixing_json": _relative_str(THREE_WAVE_JSON),
            "born_phase_diffusion_audit_json": _relative_str(BORN_JSON),
        },
        "intent": "Audit which public canonical route can supply the missing R3 target value without introducing a new free parameter.",
        "formulas": {
            "shell_anchor_slope_rule": "shell_anchor_slope_route_ready iff the shell-anchor pack already carries a universal same-sector branch with a chi_* proxy that supports S_* = (d ln omega_* / dchi)|_*",
            "g3w_route_rule": "g3w_anchor_normalized_route_ready iff g_3w is already public, anchor-normalized, and rho_* has been eliminated without a new parameter",
            "kurtosis_route_rule": "kurtosis_route_feasible iff the closed phase-diffusion bridge already supplies a viable non-Gaussian regime sensitive to R_3^2",
            "preferred_route_rule": "preferred_r3_route_or_none selects the closest remaining no-new-free-parameter route; here preference is given to an anchor-normalized g_3w route when the formal relation is already public canonical",
        },
        "rows": rows,
        "summary": {
            "shell_anchor_slope_route_ready": shell_anchor_slope_route_ready,
            "g3w_anchor_normalized_route_ready": g3w_anchor_normalized_route_ready,
            "kurtosis_route_feasible": kurtosis_route_feasible,
            "preferred_r3_route_or_none": preferred_r3_route_or_none,
            "remaining_r3_missing_datum_or_none": remaining_r3_missing_datum_or_none,
            "r3_target_available": False,
            "single_public_vpp_shape_available": bool(r3_registry_summary.get("single_public_vpp_shape_available", False)),
            "positive_particle_sector_chi_p_to_vpp_public_artifact_available": bool(
                r3_registry_summary.get("positive_particle_sector_chi_p_to_vpp_public_artifact_available", False)
            ),
        },
        "decision": {
            "overall_status": "r3_target_source_audit_frozen",
            "keep_mass_origin_branch_blocked": True,
            "shell_anchor_slope_route_ready": shell_anchor_slope_route_ready,
            "g3w_anchor_normalized_route_ready": g3w_anchor_normalized_route_ready,
            "kurtosis_route_feasible": kurtosis_route_feasible,
            "preferred_r3_route_or_none": preferred_r3_route_or_none,
            "remaining_r3_missing_datum_or_none": remaining_r3_missing_datum_or_none,
            "r3_target_available": False,
            "single_public_vpp_shape_available": bool(r3_registry_summary.get("single_public_vpp_shape_available", False)),
            "positive_particle_sector_chi_p_to_vpp_public_artifact_available": bool(
                r3_registry_summary.get("positive_particle_sector_chi_p_to_vpp_public_artifact_available", False)
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
            "g3w_formula_line": g3w_formula_hit,
            "g3w_not_new_parameter_line": g3w_not_new_parameter_hit,
            "curvature_summary": curvature_summary,
            "r3_registry_summary": r3_registry_summary,
            "tiebreak_bridge_summary": tiebreak_bridge_summary,
            "tiebreak_bridge_decision": tiebreak_bridge_decision,
            "shell_synthesis_summary": shell_synthesis_summary,
            "shell_synthesis_decision": shell_synthesis_decision,
            "three_wave_summary": three_wave_summary,
            "three_wave_decision": three_wave_decision,
            "born_summary": born_summary,
            "born_decision": born_decision,
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
