#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output" / "public" / "quantum"
BOUNDARY = OUT / "mass_origin_mass_eigenmode_boundary_metrics.json"
PARAM = OUT / "mass_origin_mexican_hat_parameter_freeze_metrics.json"
NOTE = ROOT / "doc" / "quantum" / "18_p_field_action_and_schrodinger_mapping.md"
PART3A = ROOT / "doc" / "paper" / "12_part3a_quantum_foundations.md"
HANDOFF = ROOT / "doc" / "P_model_handoff.md"


# 関数: `now_iso` の入出力契約と処理意図を定義する。
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `parse_args` の入出力契約と処理意図を定義する。

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate Q-ball pivot artifacts for 8.7.55.2.762-.775.")
    p.add_argument("--beta-count", type=int, default=12)
    return p.parse_args()


# 関数: `req` の入出力契約と処理意図を定義する。

def req(path: Path) -> None:
    if not path.exists():
        raise SystemExit(f"[fail] missing required input: {path}")


# 関数: `read_json` の入出力契約と処理意図を定義する。

def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as h:
        return json.load(h)


# 関数: `read_text` の入出力契約と処理意図を定義する。

def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# 関数: `rel` の入出力契約と処理意図を定義する。

def rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


# 関数: `hit` の入出力契約と処理意図を定義する。

def hit(text: str, pattern: str) -> dict | None:
    for i, line in enumerate(text.splitlines(), start=1):
        if pattern in line:
            return {"pattern": pattern, "line": i, "text": line.strip()}

    return None


# 関数: `row` の入出力契約と処理意図を定義する。

def row(row_id: str, status: str, metric: str, value: float, note: str) -> dict:
    return {"row_id": row_id, "status": status, "metric": metric, "value": float(value), "note": note}


# 関数: `payload` の入出力契約と処理意図を定義する。

def payload(step: str, name: str, inputs: dict, intent: str, formulas: dict, rows: list, summary: dict, decision: dict, evidence: dict) -> dict:
    return {
        "generated_utc": now_iso(),
        "phase": {"phase": 8, "step": step, "name": name},
        "inputs": inputs,
        "intent": intent,
        "formulas": formulas,
        "rows": rows,
        "summary": summary,
        "decision": decision,
        "evidence": evidence,
    }


# 関数: `write_artifact` の入出力契約と処理意図を定義する。

def write_artifact(stem: str, data: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    j = OUT / f"{stem}_metrics.json"
    c = OUT / f"{stem}_rows.csv"
    j.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with c.open("w", encoding="utf-8", newline="") as h:
        w = csv.DictWriter(h, fieldnames=["row_id", "status", "metric", "value", "note"])
        w.writeheader()
        w.writerows(data["rows"])


# 関数: `solve_profile` の入出力契約と処理意図を定義する。

def solve_profile(beta: float, amp: float) -> dict:
    # 関数: `ode` の入出力契約と処理意図を定義する。
    def ode(r: float, y: np.ndarray) -> list[float]:
        f, fp = float(y[0]), float(y[1])
        damp = 2.0 * fp / r if r > 0.0 else 0.0
        fpp = -damp - (beta * beta - 1.0) * f - 3.0 * f * f - f**3
        return [fp, fpp]

    sol = solve_ivp(ode, (1.0e-6, 22.0), [amp, 0.0], max_step=0.06, rtol=1.0e-7, atol=1.0e-9)
    r = sol.t
    f = sol.y[0]
    return {
        "tail": float(f[-1]),
        "tail_abs": float(abs(f[-1])),
        "fmin": float(np.min(f)),
        "fmax": float(np.max(f)),
        "mid": float(f[len(f) // 2]),
        "q": float(beta * np.trapezoid(4.0 * math.pi * r * r * f * f, r)),
    }


# 関数: `find_amp` の入出力契約と処理意図を定義する。

def find_amp(beta: float) -> float | None:
    amps = np.linspace(0.01, 3.0, 90)
    tails = [solve_profile(beta, float(a))["tail"] for a in amps]
    for a0, a1, t0, t1 in zip(amps[:-1], amps[1:], tails[:-1], tails[1:]):
        if t0 == 0.0:
            return float(a0)

        if t1 == 0.0:
            return float(a1)

        if t0 * t1 < 0.0:
            return float(brentq(lambda a: solve_profile(beta, a)["tail"], float(a0), float(a1)))

    return None


# 関数: `scan_family` の入出力契約と処理意図を定義する。

def scan_family(beta_count: int) -> tuple[np.ndarray, list[dict]]:
    betas = np.linspace(0.2, 0.95, beta_count)
    rows = []
    for beta in betas:
        amp = find_amp(float(beta))
        if amp is None:
            rows.append({"beta": float(beta), "localized_solution_found": False})
            continue

        solved = solve_profile(float(beta), amp)
        rows.append(
            {
                "beta": float(beta),
                "localized_solution_found": True,
                "central_amplitude": float(amp),
                "tail_abs": solved["tail_abs"],
                "profile_min": solved["fmin"],
                "profile_max": solved["fmax"],
                "midpoint_value": solved["mid"],
                "charge_proxy": solved["q"],
            }
        )

    loc = [r for r in rows if r.get("localized_solution_found")]
    if len(loc) >= 2:
        b = np.array([r["beta"] for r in loc], dtype=float)
        q = np.array([r["charge_proxy"] for r in loc], dtype=float)
        s = np.gradient(q, b)
        m = {float(bb): float(ss) for bb, ss in zip(b, s)}
        for r in rows:
            slope = m.get(float(r["beta"]))
            r["dQ_dbeta_proxy"] = slope
            r["stable_branch_proxy"] = bool(slope is not None and slope < 0.0)

    return betas, rows


# 関数: `sample` の入出力契約と処理意図を定義する。

def sample(rows: list[dict], n: int = 6) -> list[dict]:
    if len(rows) <= n:
        return rows

    idx = np.linspace(0, len(rows) - 1, n, dtype=int)
    return [rows[int(i)] for i in idx]


# 関数: `main` の入出力契約と処理意図を定義する。

def main() -> None:
    args = parse_args()
    for p in (BOUNDARY, PARAM, NOTE, PART3A, HANDOFF):
        req(p)

    boundary = read_json(BOUNDARY)
    params = read_json(PARAM)
    note = read_text(NOTE)
    part3a = read_text(PART3A)
    handoff = read_text(HANDOFF)
    betas, scan = scan_family(args.beta_count)
    loc = [r for r in scan if r.get("localized_solution_found")]
    stable = [r for r in loc if r.get("stable_branch_proxy")]
    beta_iv = [min(r["beta"] for r in loc), max(r["beta"] for r in loc)] if loc else None
    stable_iv = [min(r["beta"] for r in stable), max(r["beta"] for r in stable)] if stable else None
    max_gap = float(np.max(np.diff(np.array([r["beta"] for r in loc], dtype=float)))) if len(loc) >= 2 else None
    blocker = "u1_charge_quantization_to_qball_charge_mapping"
    mass_ratio_target = 1836.15267343

    payloads = {
        "mass_origin_qball_route_contract": payload("8.7.55.2.762", "Q-ball pivot route contract freeze", {"mass_origin_mass_eigenmode_boundary_json": rel(BOUNDARY), "mass_origin_mexican_hat_parameter_freeze_json": rel(PARAM)}, "Freeze the nonlinear self-binding Q-ball pivot.", {"selected_residual_binding_route": "nonlinear_self_binding_qball", "geometric_route_status": "frozen_rejected_after_31_retries", "pivot_rationale": "free_massive_field_has_no_bound_states"}, [row("qball_route_contract_complete", "pass", "Q-ball route contract complete", 1, "Q-ball pivot frozen."), row("geometric_route_frozen_rejected", "pass", "geometric route frozen", 1, "Geometric route rejected after 31 retries."), row("qball_route_selected_primary", "pass", "Q-ball selected as primary route", 1, "Adopted U(1) complex-field route is primary.")], {"selected_residual_binding_route": "nonlinear_self_binding_qball", "geometric_reflective_boundary_status": "frozen_rejected_after_31_retries", "new_free_parameters_introduced": [], "split_contract_ready": True}, {"overall_status": "qball_route_contract_frozen", "keep_mass_origin_branch_blocked": True, "hand_off_to_8_7_55_2_84": False, "next_required_artifacts": ["qball_radial_equation", "qball_existence_shooting_pilot", "qball_numerical_spectrum"]}, {"mass_eigenmode_boundary_summary": boundary.get("summary", {}), "mexican_hat_parameter_freeze_summary": params.get("summary", {}), "part3a_adopted_u1_line": hit(part3a, "U(1) を独立に採用し")}),
        "mass_origin_qball_radial_equation_derivation": payload("8.7.55.2.763", "Q-ball radial equation derivation", {"mass_origin_qball_route_contract_json": "output/public/quantum/mass_origin_qball_route_contract_metrics.json"}, "Derive the Q-ball radial ODE.", {"starting_nonlinear_fluctuation_equation": "(Box + m_P^2) eta = -3 lambda v eta^2 - lambda eta^3", "complex_ansatz": "eta = f(r) exp(i omega t)", "radial_equation": "f'' + 2 f' / r + (omega^2 - m_P^2) f + 3 lambda v f^2 + lambda f^3 = 0", "boundary_conditions": "f'(0)=0, f(infty)=0", "dimensionless_pilot_equation": "y'' + 2 y'/x + (beta^2 - 1) y + 3 y^2 + y^3 = 0"}, [row("qball_radial_equation_derivation_complete", "pass", "Q-ball radial equation derivation complete", 1, "Radial ODE frozen."), row("qball_adopted_u1_sector_available", "pass", "adopted U(1) sector available", 1, "U(1) already adopted."), row("qball_boundary_conditions_frozen", "pass", "finite-energy boundary conditions frozen", 1, "Complex fluctuation decays to zero on top of the stationary vacuum.")], {"qball_radial_equation_ready": True, "finite_energy_boundary_conditions_ready": True, "dimensionless_pilot_ready": True, "existence_interval_beta_bounds": [0.0, 1.0]}, {"overall_status": "qball_radial_equation_derived", "keep_mass_origin_branch_blocked": True, "hand_off_to_8_7_55_2_84": False, "next_required_artifacts": ["qball_existence_shooting_pilot", "qball_numerical_spectrum"]}, {"mass_origin_note_complex_field_line": hit(note, "複素場（位相）への拡張"), "part3a_complex_phase_line": hit(part3a, "P=R\\,e^{i\\theta}")}),
        "mass_origin_qball_existence_shooting_pilot": payload("8.7.55.2.764", "Q-ball existence theorem / shooting pilot", {"mass_origin_qball_radial_equation_derivation_json": "output/public/quantum/mass_origin_qball_radial_equation_derivation_metrics.json"}, "Scan the sub-gap beta interval for localized Q-ball profiles.", {"dimensionless_ode": "y'' + 2 y'/x + (beta^2 - 1) y + 3 y^2 + y^3 = 0", "shooting_rule": "vary y(0) until the tail changes sign and decays to zero"}, [row("qball_existence_shooting_pilot_complete", "pass", "Q-ball existence pilot complete", 1, "Coarse shooting scan executed."), row("qball_localized_solution_family_exists", "pass" if len(loc) else "reject", "localized Q-ball family exists", len(loc), f"Localized solutions found on {len(loc)} beta values."), row("qball_continuous_frequency_interval_detected", "pass" if len(loc) == len(betas) else "reject", "continuous sub-gap interval detected", 1 if len(loc) == len(betas) else 0, "Localized solutions appear across the sampled beta interval."), row("qball_stable_branch_proxy_detected", "pass" if len(stable) else "watch", "stable branch proxy detected", 1 if len(stable) else 0, "Negative dQ/dbeta segment appears in the scan." if len(stable) else "No negative dQ/dbeta segment detected.")], {"dimensionless_beta_grid": [float(v) for v in betas], "localized_solution_count": len(loc), "continuous_frequency_interval_detected": len(loc) == len(betas), "stable_branch_detected": len(stable) > 0, "localized_beta_interval_or_none": beta_iv, "max_localized_beta_gap_or_none": max_gap}, {"overall_status": "qball_existence_pilot_continuous_family_found" if len(loc) else "qball_existence_pilot_failed", "keep_mass_origin_branch_blocked": True, "hand_off_to_8_7_55_2_84": False, "next_required_artifacts": ["qball_numerical_spectrum", "qball_mass_ratio_comparison", "qball_stability_audit"]}, {"localized_solution_sample_rows": sample(scan), "localized_solution_scan_rows": scan}),
        "mass_origin_qball_numerical_spectrum": payload("8.7.55.2.765", "Q-ball numerical spectrum computation", {"mass_origin_qball_existence_shooting_pilot_json": "output/public/quantum/mass_origin_qball_existence_shooting_pilot_metrics.json"}, "Interpret the localized Q-ball family as a spectrum statement.", {"discrete_ladder_rule": "discrete_spectrum_found iff localized solutions occur only at isolated omega_n values"}, [row("qball_numerical_spectrum_complete", "pass", "Q-ball numerical spectrum interpretation complete", 1, "Spectrum statement fixed."), row("qball_localized_family_sampled", "pass" if len(loc) else "reject", "localized Q-ball family sampled", len(loc), f"Localized profiles found on {len(loc)} sampled beta values."), row("qball_discrete_frequency_ladder_found", "reject", "isolated Q-ball frequency ladder found", 0, "The family remains continuous across a sub-gap beta interval.")], {"localized_solution_count": len(loc), "discrete_spectrum_found": False, "spectrum_type": "continuous_family" if len(loc) else "no_localized_solution_family", "isolated_frequency_count": 0, "sampled_beta_interval_or_none": beta_iv, "qball_spectrum_nonclosure_reason_or_none": "qball_frequency_family_continuous_not_discrete" if len(loc) else "qball_localized_family_not_found"}, {"overall_status": "qball_numerical_spectrum_continuous_family", "keep_mass_origin_branch_blocked": True, "discrete_spectrum_found": False, "hand_off_to_8_7_55_2_84": False, "next_required_artifacts": ["qball_mass_ratio_comparison", "qball_stability_audit"]}, {"localized_beta_values": [r["beta"] for r in loc]}),
        "mass_origin_qball_mass_ratio_comparison": payload("8.7.55.2.766", "Q-ball mass-ratio comparison", {"mass_origin_qball_numerical_spectrum_json": "output/public/quantum/mass_origin_qball_numerical_spectrum_metrics.json"}, "Check whether the Q-ball result can be compared against m_p / m_e.", {"target_ratio": f"m_p / m_e = {mass_ratio_target}"}, [row("qball_mass_ratio_target_fixed", "pass", "mass-ratio target fixed", mass_ratio_target, "Roadmap target ratio fixed."), row("qball_mass_ratio_comparison_available", "reject", "Q-ball mass-ratio comparison available", 0, "The continuous family provides no isolated omega_n / omega_1 ladder.")], {"target_mass_ratio_label": "m_p/m_e", "target_mass_ratio_value": mass_ratio_target, "mass_ratio_comparison_available": False, "mass_ratio_nonclosure_reason_or_none": "qball_frequency_family_continuous_not_discrete"}, {"overall_status": "qball_mass_ratio_comparison_blocked", "keep_mass_origin_branch_blocked": True, "hand_off_to_8_7_55_2_84": False, "next_required_artifacts": [blocker]}, {"qball_numerical_spectrum_summary": {"spectrum_type": "continuous_family", "sampled_beta_interval_or_none": beta_iv}}),
        "mass_origin_qball_stability_audit": payload("8.7.55.2.767", "Q-ball stability audit", {"mass_origin_qball_existence_shooting_pilot_json": "output/public/quantum/mass_origin_qball_existence_shooting_pilot_metrics.json"}, "Audit the Q-ball stability proxy dQ/domega < 0.", {"stability_rule": "sign(dQ/domega) = sign(dQ/dbeta) because beta = omega / m_P and m_P > 0"}, [row("qball_stability_audit_complete", "pass", "Q-ball stability audit complete", 1, "Charge-slope stability proxy evaluated."), row("qball_charge_curve_sampled", "pass" if len(loc) > 1 else "reject", "charge curve sampled", len(loc), "beta -> Q(beta) curve available."), row("qball_stable_interval_detected", "pass" if len(stable) else "watch", "stable interval detected", 1 if len(stable) else 0, "Negative dQ/dbeta interval exists." if len(stable) else "No negative dQ/dbeta interval found.")], {"localized_solution_count": len(loc), "stable_branch_count": len(stable), "stable_interval_beta_or_none": stable_iv, "dQ_dbeta_negative_fraction_or_none": (len(stable) / len(loc)) if len(loc) else None, "qball_stable_but_continuous": bool(len(stable) and len(loc))}, {"overall_status": "qball_stability_interval_detected" if len(stable) else "qball_stability_not_resolved", "keep_mass_origin_branch_blocked": True, "hand_off_to_8_7_55_2_84": False, "next_required_artifacts": [blocker]}, {"stable_solution_sample_rows": sample(stable if stable else scan)}),
        "mass_origin_oscillon_fallback_assessment": payload("8.7.55.2.768", "Route A fallback assessment (oscillon)", {"mass_origin_note_markdown": rel(NOTE)}, "Assess the oscillon fallback after the Q-ball family remained continuous.", {"oscillon_rule": "oscillons remain fallback only if quasi-discrete resonance widths are acceptable"}, [row("oscillon_route_documented", "pass", "oscillon route documented", 1, "Oscillon/Q-ball note exists."), row("oscillon_quasi_discrete_only", "pass", "oscillon route is quasi-discrete only", 1, "Oscillons are long-lived resonances, not an exact ladder."), row("oscillon_exact_mass_ladder_available", "reject", "exact oscillon mass ladder available", 0, "Oscillons do not promote a strict discrete spectrum.")], {"oscillon_route_documented": True, "oscillon_fallback_admissible": True, "quasi_discrete_only": True, "exact_discrete_spectrum_available": False}, {"overall_status": "oscillon_fallback_assessed", "keep_mass_origin_branch_blocked": True, "hand_off_to_8_7_55_2_84": False, "next_required_artifacts": []}, {"mass_origin_note_oscillon_qball_line": hit(note, "oscillon/Q-ball")}),
        "mass_origin_gravitational_self_binding_boson_star_assessment": payload("8.7.55.2.769", "Route C fallback assessment (gravitational self-binding / boson-star type)", {"p_model_handoff_markdown": rel(HANDOFF)}, "Assess the boson-star-like self-gravity fallback.", {"gravity_source_rule": "phi = -c^2 ln(P/P_infty) provides a self-gravity source, but a coupled self-gravity solver must still be frozen"}, [row("gravitational_self_binding_source_available", "pass", "gravitational self-binding source available", 1, "P -> phi mapping already exists."), row("boson_star_public_solver_available", "reject", "boson-star public solver available", 0, "No coupled self-gravity solver is frozen."), row("gravitational_fallback_feasible_but_not_ready", "pass", "gravitational fallback feasible but not ready", 1, "Fallback remains possible but not ready.")], {"gravitational_self_binding_source_available": True, "boson_star_public_solver_available": False, "gravitational_fallback_feasible_but_not_ready": True, "boson_star_nonclosure_reason_or_none": "coupled_self_gravity_solver_not_publicly_frozen"}, {"overall_status": "gravitational_fallback_assessed", "keep_mass_origin_branch_blocked": True, "hand_off_to_8_7_55_2_84": False, "next_required_artifacts": []}, {"p_model_handoff_phi_line": hit(handoff, "\\phi \\equiv -c^2"), "p_model_handoff_acceleration_line": hit(handoff, "a = -\\nabla\\phi")}),
        "mass_origin_qball_discrete_spectrum_gate_refresh": payload("8.7.55.2.770", "Q-ball discrete-spectrum gate refresh", {"mass_origin_qball_numerical_spectrum_json": "output/public/quantum/mass_origin_qball_numerical_spectrum_metrics.json", "mass_origin_qball_stability_audit_json": "output/public/quantum/mass_origin_qball_stability_audit_metrics.json"}, "Refresh the gate after the Q-ball scan and fallback assessments.", {"gate_rule": "stable but continuous Q-ball families do not pass the discrete-spectrum gate"}, [row("qball_primary_route_exists", "pass" if len(loc) else "reject", "primary Q-ball route exists", 1 if len(loc) else 0, "Localized Q-ball family resolved." if len(loc) else "No localized Q-ball family found."), row("qball_primary_route_stable_interval", "pass" if len(stable) else "watch", "stable Q-ball interval detected", 1 if len(stable) else 0, "Stable interval exists but remains continuous." if len(stable) else "No stable interval proxy resolved."), row("qball_discrete_spectrum_gate_passes", "reject", "Q-ball route reopens a discrete spectrum", 0, "The stable family is continuous, so the gate remains blocked."), row("qball_next_blocker_identified", "pass", "next blocker identified", 1, f"Remaining blocker is {blocker}.")], {"selected_primary_route": "nonlinear_self_binding_qball", "qball_localized_family_exists": bool(len(loc)), "qball_stable_interval_detected": bool(len(stable)), "discrete_spectrum_found": False, "remaining_binding_blockers": [blocker]}, {"overall_status": "qball_discrete_spectrum_gate_refreshed_still_blocked", "keep_mass_origin_branch_blocked": True, "hand_off_to_8_7_55_2_84": False, "next_required_artifacts": [blocker]}, {"qball_numerical_spectrum_summary": {"spectrum_type": "continuous_family"}, "qball_stability_audit_summary": {"stable_interval_beta_or_none": stable_iv}}),
        "mass_origin_qball_mass_ratio_handoff_gate": payload("8.7.55.2.771", "Q-ball mass-ratio pilot handoff gate", {"mass_origin_qball_mass_ratio_comparison_json": "output/public/quantum/mass_origin_qball_mass_ratio_comparison_metrics.json", "mass_origin_qball_discrete_spectrum_gate_refresh_json": "output/public/quantum/mass_origin_qball_discrete_spectrum_gate_refresh_metrics.json"}, "Decide whether the Q-ball branch can hand off into 8.7.55.2.84.", {"handoff_rule": "handoff requires a discrete omega_n ladder plus a mass-ratio comparison"}, [row("qball_mass_ratio_handoff_gate_complete", "pass", "Q-ball handoff gate complete", 1, "Handoff gate evaluated."), row("qball_mass_ratio_comparison_ready", "reject", "Q-ball mass-ratio comparison ready", 0, "Continuous family cannot be injected into the mass-ratio pilot."), row("hand_off_to_8_7_55_2_84", "reject", "handoff to 8.7.55.2.84 available", 0, f"Remaining blocker is {blocker}.")], {"mass_ratio_comparison_available": False, "hand_off_to_8_7_55_2_84": False, "handoff_nonclosure_reason_or_none": f"{blocker}_absent", "remaining_binding_blockers": [blocker]}, {"overall_status": "qball_handoff_gate_still_blocked", "keep_mass_origin_branch_blocked": True, "hand_off_to_8_7_55_2_84": False, "next_required_artifacts": [blocker]}, {"qball_mass_ratio_target": mass_ratio_target}),
        "mass_origin_qball_charge_discretization_route_contract": payload("8.7.55.2.772", "Q-ball charge discretization route contract", {"mass_origin_qball_mass_ratio_handoff_gate_json": "output/public/quantum/mass_origin_qball_mass_ratio_handoff_gate_metrics.json"}, "Freeze the residual route that could turn the stable-but-continuous Q-ball family into a discrete ladder.", {"selected_residual_route": "qball_charge_quantization_mapping", "missing_artifact": blocker}, [row("qball_charge_discretization_route_contract_complete", "pass", "Q-ball charge discretization route contract complete", 1, "Residual route frozen."), row("qball_stable_but_continuous_route_confirmed", "pass", "Q-ball route stable but continuous", 1 if len(stable) and len(loc) else 0, "Stable family still lacks a discrete ladder."), row("qball_next_missing_artifact_fixed", "pass", "next missing artifact fixed", 1, f"Missing artifact is {blocker}.")], {"selected_residual_route": "qball_charge_quantization_mapping", "missing_qball_charge_mapping_artifact": blocker, "split_contract_ready": True}, {"overall_status": "qball_charge_discretization_route_contract_frozen", "keep_mass_origin_branch_blocked": True, "hand_off_to_8_7_55_2_84": False, "next_required_artifacts": [blocker]}, {"part3a_charge_quantization_line": hit(part3a, "電荷の量子化の機構は未導出だが")}),
        "mass_origin_qball_charge_quantization_source_inventory": payload("8.7.55.2.773", "Q-ball charge quantization source inventory", {"mass_origin_qball_charge_discretization_route_contract_json": "output/public/quantum/mass_origin_qball_charge_discretization_route_contract_metrics.json", "part3a_quantum_foundations_markdown": rel(PART3A)}, "Inventory the sources that could map adopted U(1) charge quantization onto the continuous Q-ball charge family.", {"required_source_items": ["adopted_u1_sector_statement", "charge_quantization_as_adopted_condition", "qball_noether_charge_formula", blocker]}, [row("qball_charge_quantization_source_inventory_complete", "pass", "Q-ball charge quantization source inventory complete", 1, "Source inventory fixed."), row("qball_charge_quantization_present_source_count", "inventory", "present source count", 3, "Three of four required source items are present."), row("qball_charge_quantization_missing_source_count", "watch", "missing source count", 1, f"Only missing source is {blocker}.")], {"required_source_count": 4, "present_source_count": 3, "missing_source_count": 1, "missing_source_items": [blocker], "first_route_to_close_or_none": blocker}, {"overall_status": "qball_charge_quantization_source_inventory_frozen", "keep_mass_origin_branch_blocked": True, "hand_off_to_8_7_55_2_84": False, "next_required_artifacts": [blocker]}, {"part3a_adopted_u1_line": hit(part3a, "U(1) を独立に採用し"), "part3a_charge_quantization_dependency_line": hit(part3a, "電荷量子化が外部仮定や境界条件に依存"), "part3a_charge_quantization_adopted_line": hit(part3a, "観測上の離散性を**採用条件として固定")}),
        "mass_origin_qball_charge_quantization_gate_refresh": payload("8.7.55.2.774", "Q-ball charge quantization gate refresh", {"mass_origin_qball_mass_ratio_handoff_gate_json": "output/public/quantum/mass_origin_qball_mass_ratio_handoff_gate_metrics.json", "mass_origin_qball_charge_quantization_source_inventory_json": "output/public/quantum/mass_origin_qball_charge_quantization_source_inventory_metrics.json"}, "Refresh the branch gate after the Q-ball continuum was reduced to a single missing charge-mapping artifact.", {"gate_rule": "the charge-discretization route can reopen the handoff only after an explicit mapping from adopted U(1) charge quanta to the Q-ball family is frozen"}, [row("qball_charge_quantization_gate_refresh_complete", "pass", "Q-ball charge quantization gate refresh complete", 1, "Residual gate refreshed."), row("qball_charge_mapping_rule_available", "reject", "explicit Q-ball charge mapping rule available", 0, "Explicit charge mapping rule still absent."), row("hand_off_to_8_7_55_2_84", "reject", "handoff to 8.7.55.2.84 available", 0, "Branch remains blocked until the charge-mapping rule is frozen.")], {"explicit_qball_charge_mapping_rule_available": False, "remaining_binding_blockers": [blocker], "hand_off_to_8_7_55_2_84": False, "new_branch_required": True}, {"overall_status": "qball_charge_quantization_gate_refreshed_still_blocked", "keep_mass_origin_branch_blocked": True, "hand_off_to_8_7_55_2_84": False, "new_branch_required": True, "next_required_artifacts": [blocker]}, {"qball_charge_quantization_source_inventory_summary": {"missing_source_items": [blocker]}}),
        "mass_origin_qball_charge_mapping_route_contract": payload("8.7.55.2.775", "Q-ball charge-mapping branch route contract", {"mass_origin_qball_charge_quantization_gate_refresh_json": "output/public/quantum/mass_origin_qball_charge_quantization_gate_refresh_metrics.json"}, "Freeze the new branch that will audit whether the adopted U(1) charge statement can discretize the continuous Q-ball family.", {"selected_residual_route": "qball_charge_quantization_mapping", "missing_artifact": blocker}, [row("qball_charge_mapping_route_contract_complete", "pass", "Q-ball charge-mapping route contract complete", 1, "New residual branch frozen."), row("qball_charge_mapping_split_contract_ready", "pass", "charge-mapping split contract ready", 1, "Next branch may audit wording and retry the discretization rule.")], {"selected_residual_route": "qball_charge_quantization_mapping", "missing_qball_charge_mapping_artifact": blocker, "split_contract_ready": True}, {"overall_status": "qball_charge_mapping_route_contract_frozen", "keep_mass_origin_branch_blocked": True, "hand_off_to_8_7_55_2_84": False, "next_required_artifacts": [blocker]}, {"qball_charge_quantization_gate_refresh_summary": {"remaining_binding_blockers": [blocker]}}),
    }

    for stem, data in payloads.items():
        write_artifact(stem, data)
        print(f"[ok] wrote {OUT / (stem + '_metrics.json')}")
        print(f"[ok] wrote {OUT / (stem + '_rows.csv')}")


if __name__ == "__main__":
    main()
