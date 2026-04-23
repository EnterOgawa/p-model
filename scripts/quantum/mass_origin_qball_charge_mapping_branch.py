#!/usr/bin/env python3
"""
Generate Q-ball charge-mapping branch artifacts for 8.7.55.2.776-.781.

This branch accepts the adopted U(1) charge quantization statement as the
canonical discretization rule for the continuous Q-ball family, then checks
whether the resulting discrete ladder can reopen the mass-ratio pilot.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output" / "public" / "quantum"
PART3A = ROOT / "doc" / "paper" / "12_part3a_quantum_foundations.md"
QBALL_ROUTE = OUT / "mass_origin_qball_charge_mapping_route_contract_metrics.json"
QBALL_STABILITY = OUT / "mass_origin_qball_stability_audit_metrics.json"
QBALL_PIVOT = ROOT / "scripts" / "quantum" / "mass_origin_qball_pivot_branch.py"


# Function: return the current UTC timestamp in ISO 8601 format.
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# Function: parse reproducible CLI arguments for the branch generator.

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate Q-ball charge-mapping branch artifacts for 8.7.55.2.776-.781.")
    p.add_argument("--beta-count", type=int, default=60)
    p.add_argument("--beta-upper", type=float, default=0.999)
    return p.parse_args()


# Function: fail early when a required input artifact is missing.

def req(path: Path) -> None:
    if not path.exists():
        raise SystemExit(f"[fail] missing required input: {path}")


# Function: read JSON artifacts using UTF-8.

def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as h:
        return json.load(h)


# Function: read UTF-8 text sources used as canonical evidence.

def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# Function: convert an absolute path into repository-relative form.

def rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


# Function: find the first source line containing the requested pattern.

def hit(text: str, pattern: str) -> dict | None:
    for i, line in enumerate(text.splitlines(), start=1):
        if pattern in line:
            return {"pattern": pattern, "line": i, "text": line.strip()}

    return None


# Function: construct a single metrics row in canonical format.

def row(row_id: str, status: str, metric: str, value: float, note: str) -> dict:
    return {"row_id": row_id, "status": status, "metric": metric, "value": float(value), "note": note}


# Function: construct a full artifact payload with the shared schema.

def payload(step: str, name: str, inputs: dict, intent: str, formulas: dict, rows: list[dict], summary: dict, decision: dict, evidence: dict) -> dict:
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


# Function: write JSON and CSV artifacts side by side.

def write_artifact(stem: str, data: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    json_path = OUT / f"{stem}_metrics.json"
    csv_path = OUT / f"{stem}_rows.csv"
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as h:
        writer = csv.DictWriter(h, fieldnames=["row_id", "status", "metric", "value", "note"])
        writer.writeheader()
        writer.writerows(data["rows"])


# Function: load the previous pivot script as a reusable module.

def load_qball_module():
    spec = importlib.util.spec_from_file_location("wavep_qball_pivot", QBALL_PIVOT)
    if spec is None or spec.loader is None:
        raise SystemExit(f"[fail] unable to load module from {QBALL_PIVOT}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Function: solve the full radial profile needed for charge and energy integrals.

def solve_full_profile(beta: float, amp: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # 関数: `ode` の入出力契約と処理意図を定義する。
    def ode(r: float, y: np.ndarray) -> list[float]:
        f, fp = float(y[0]), float(y[1])
        damp = 2.0 * fp / r if r > 0.0 else 0.0
        fpp = -damp - (beta * beta - 1.0) * f - 3.0 * f * f - f**3
        return [fp, fpp]

    sol = solve_ivp(ode, (1.0e-6, 30.0), [amp, 0.0], max_step=0.03, rtol=1.0e-8, atol=1.0e-10)
    return sol.t, sol.y[0], sol.y[1]


# Function: evaluate the charge and energy proxies at a single beta value.

def build_charge_energy_evaluator(qball_module):
    # 関数: `evaluate` の入出力契約と処理意図を定義する。
    @lru_cache(maxsize=None)
    def evaluate(beta: float) -> dict | None:
        amp = qball_module.find_amp(float(beta))
        if amp is None:
            return None

        r, f, fp = solve_full_profile(float(beta), float(amp))
        charge = float(beta * np.trapezoid(4.0 * math.pi * r * r * f * f, r))
        energy = float(np.trapezoid(4.0 * math.pi * r * r * (0.5 * fp * fp + 0.5 * (1.0 + beta * beta) * f * f + f**3 + 0.25 * f**4), r))
        return {
            "beta": float(beta),
            "central_amplitude": float(amp),
            "charge_proxy": charge,
            "energy_proxy": energy,
            "tail_abs": float(abs(f[-1])),
        }

    return evaluate


# Function: scan the stable charge family on a refined beta grid.

def scan_charge_family(evaluate, beta_start: float, beta_upper: float, beta_count: int) -> list[dict]:
    rows = []
    betas = np.linspace(beta_start, beta_upper, beta_count)
    for beta in betas:
        solved = evaluate(float(beta))
        if solved is None:
            rows.append({"beta": float(beta), "localized_solution_found": False})
            continue

        rows.append(
            {
                "beta": float(beta),
                "localized_solution_found": True,
                "central_amplitude": solved["central_amplitude"],
                "charge_proxy": solved["charge_proxy"],
                "energy_proxy": solved["energy_proxy"],
                "tail_abs": solved["tail_abs"],
            }
        )

    localized = [r for r in rows if r.get("localized_solution_found")]
    if len(localized) >= 2:
        b = np.array([r["beta"] for r in localized], dtype=float)
        q = np.array([r["charge_proxy"] for r in localized], dtype=float)
        slopes = np.gradient(q, b)
        for r, slope in zip(localized, slopes):
            r["dQ_dbeta_proxy"] = float(slope)
            r["stable_branch_proxy"] = bool(slope < 0.0)

    return rows


# Function: invert the stable charge curve at integer Q/q = n values.

def invert_integer_modes(scan_rows: list[dict], evaluate) -> list[dict]:
    stable = [r for r in scan_rows if r.get("localized_solution_found") and r.get("stable_branch_proxy")]
    if len(stable) < 2:
        return []

    stable = sorted(stable, key=lambda r: float(r["beta"]))
    q_min = min(float(r["charge_proxy"]) for r in stable)
    q_max = max(float(r["charge_proxy"]) for r in stable)
    n_min = max(1, int(math.ceil(q_min)))
    n_max = int(math.floor(q_max))
    if n_max < n_min:
        return []

    modes = []
    for n in range(n_min, n_max + 1):
        beta_n = None
        for left, right in zip(stable[:-1], stable[1:]):
            q0 = float(left["charge_proxy"])
            q1 = float(right["charge_proxy"])
            if (q0 - n) * (q1 - n) > 0.0:
                continue

            b0 = float(left["beta"])
            b1 = float(right["beta"])
            if q1 == q0:
                beta_n = b0
            else:
                beta_n = b0 + (n - q0) * (b1 - b0) / (q1 - q0)

            break

        if beta_n is None:
            continue

        solved = evaluate(float(beta_n))
        if solved is None:
            continue

        modes.append(
            {
                "mode_index": int(n),
                "beta_n": float(beta_n),
                "charge_proxy": float(solved["charge_proxy"]),
                "energy_proxy": float(solved["energy_proxy"]),
                "central_amplitude": float(solved["central_amplitude"]),
            }
        )

    modes = sorted(modes, key=lambda r: int(r["mode_index"]))
    if modes:
        base_energy = float(modes[0]["energy_proxy"])
        for mode in modes:
            mode["mass_ratio_to_first"] = float(mode["energy_proxy"] / base_energy) if base_energy != 0.0 else math.nan

    return modes


# Function: compare the derived mass ratios against the canonical particle targets.

def compare_targets(modes: list[dict]) -> tuple[list[dict], dict | None]:
    targets = [
        {"label": "m_p/m_e", "value": 1836.15267343, "threshold": 0.10},
        {"label": "m_mu/m_e", "value": 206.7682830, "threshold": 0.10},
        {"label": "m_tau/m_e", "value": 3477.48, "threshold": 0.10},
        {"label": "m_n/m_p", "value": 1.00137842, "threshold": 0.001},
    ]
    if not modes:
        return [], None

    rows = []
    best = None
    for mode in modes[1:]:
        ratio = float(mode["mass_ratio_to_first"])
        for target in targets:
            rel_err = abs(ratio - float(target["value"])) / float(target["value"])
            rec = {
                "mode_index": int(mode["mode_index"]),
                "ratio_label": f"M_{int(mode['mode_index'])}/M_1",
                "ratio_value": ratio,
                "target_label": target["label"],
                "target_value": float(target["value"]),
                "relative_error": float(rel_err),
                "passes_threshold": bool(rel_err <= float(target["threshold"])),
            }
            rows.append(rec)
            if best is None or rec["relative_error"] < best["relative_error"]:
                best = rec

    return rows, best


# Function: return a compact sample of rows for artifact evidence blocks.

def sample(rows: list[dict], n: int = 6) -> list[dict]:
    if len(rows) <= n:
        return rows

    idx = np.linspace(0, len(rows) - 1, n, dtype=int)
    return [rows[int(i)] for i in idx]


# Function: drive the branch and emit all public artifacts.

def main() -> None:
    args = parse_args()
    for path in (PART3A, QBALL_ROUTE, QBALL_STABILITY, QBALL_PIVOT):
        req(path)

    part3a = read_text(PART3A)
    route = read_json(QBALL_ROUTE)
    stability = read_json(QBALL_STABILITY)
    qball_module = load_qball_module()
    evaluate = build_charge_energy_evaluator(qball_module)
    stable_start = float(stability["summary"]["stable_interval_beta_or_none"][0])
    scan_rows = scan_charge_family(evaluate, stable_start, float(args.beta_upper), int(args.beta_count))
    stable_rows = [r for r in scan_rows if r.get("localized_solution_found") and r.get("stable_branch_proxy")]
    modes = invert_integer_modes(scan_rows, evaluate)
    ratio_rows, best_match = compare_targets(modes)
    discrete_found = len(modes) >= 2
    handoff = bool(best_match is not None and best_match["passes_threshold"])
    case_label = "case_a_handoff_pass" if handoff else ("case_b_discrete_ladder_but_ratio_mismatch" if discrete_found else "case_c_insufficient_discrete_modes")
    next_route = "qball_ratio_mismatch_resolution" if discrete_found and not handoff else "oscillon_fallback_reopen"

    mapping_line = hit(part3a, "観測上の離散性を**採用条件として固定")
    adopted_u1_line = hit(part3a, "U(1) を独立に採用し")
    dependency_line = hit(part3a, "電荷量子化が外部仮定や境界条件に依存")

    payloads = {
        "mass_origin_qball_charge_mapping_statement_freeze": payload(
            "8.7.55.2.776",
            "Q-ball charge mapping canonical statement freeze",
            {
                "mass_origin_qball_charge_mapping_route_contract_json": rel(QBALL_ROUTE),
                "part3a_quantum_foundations_markdown": rel(PART3A),
            },
            "Freeze the adopted U(1) charge statement as the canonical discretization rule for the Q-ball family.",
            {
                "canonical_statement": "The Q-ball Noether charge coincides with the adopted U(1) charge. Charge quantization Q_n = n q, already frozen as an observational adoption in Part III-A sec. 2.6.2, discretizes the continuous Q-ball family into a countable mass ladder {M_n} with no new free parameter.",
                "mapping_source": "Part III-A sec. 2.6.2 adopted charge quantization",
            },
            [
                row("qball_charge_mapping_statement_complete", "pass", "charge mapping canonical statement frozen", 1, "Direct mapping statement frozen."),
                row("u1_charge_quantization_to_qball_charge_mapping_available", "pass", "charge mapping available", 1, "Adopted U(1) charge quantization is treated as the discretization rule."),
                row("qball_charge_mapping_new_free_parameters", "pass", "new free parameters introduced", 0, "No new free parameters introduced."),
            ],
            {
                "u1_charge_quantization_to_qball_charge_mapping": "available",
                "new_free_parameters_introduced": [],
                "mapping_source": "Part III-A sec. 2.6.2 adopted charge quantization",
                "charge_quantum_normalization": "elementary_charge_unit_q",
            },
            {
                "overall_status": "qball_charge_mapping_statement_frozen",
                "keep_mass_origin_branch_blocked": True,
                "hand_off_to_8_7_55_2_84": False,
                "next_required_artifacts": [
                    "qball_charge_discrete_frequency_inversion",
                    "qball_discrete_mass_spectrum",
                    "qball_charge_mapped_mass_ratio_comparison",
                ],
            },
            {
                "part3a_adopted_u1_line": adopted_u1_line,
                "part3a_charge_quantization_dependency_line": dependency_line,
                "part3a_charge_quantization_adopted_line": mapping_line,
            },
        ),
        "mass_origin_qball_charge_discrete_frequency_inversion": payload(
            "8.7.55.2.777",
            "Q-ball charge-discrete frequency inversion",
            {
                "mass_origin_qball_charge_mapping_statement_freeze_json": "output/public/quantum/mass_origin_qball_charge_mapping_statement_freeze_metrics.json",
                "mass_origin_qball_stability_audit_json": rel(QBALL_STABILITY),
            },
            "Invert the refined stable Q(beta) curve at integer Q/q = n values.",
            {
                "charge_discretization_rule": "Q_n = n q",
                "inverse_rule": "beta_n = Q^{-1}(n q) on the monotone stable branch",
                "refined_beta_scan_interval": [stable_start, float(args.beta_upper)],
            },
            [
                row("qball_charge_inversion_complete", "pass", "charge inversion complete", 1, "Stable Q(beta) curve inverted at integer charges."),
                row("qball_integer_mode_count", "pass" if discrete_found else "watch", "integer charge modes found", len(modes), f"{len(modes)} integer modes found on the refined stable branch."),
                row("qball_discrete_spectrum_found", "pass" if discrete_found else "reject", "discrete spectrum found", 1 if discrete_found else 0, "At least two integer charge modes exist." if discrete_found else "Fewer than two integer charge modes exist."),
            ],
            {
                "refined_stable_beta_interval_or_none": [min(r["beta"] for r in stable_rows), max(r["beta"] for r in stable_rows)] if stable_rows else None,
                "refined_stable_mode_count": len(stable_rows),
                "discrete_mode_count": len(modes),
                "discrete_mode_indices": [int(m["mode_index"]) for m in modes],
                "discrete_spectrum_found": discrete_found,
            },
            {
                "overall_status": "qball_charge_discrete_ladder_found" if discrete_found else "qball_charge_discrete_ladder_insufficient",
                "keep_mass_origin_branch_blocked": True,
                "hand_off_to_8_7_55_2_84": False,
                "next_required_artifacts": ["qball_discrete_mass_spectrum", "qball_charge_mapped_mass_ratio_comparison"],
            },
            {
                "refined_charge_curve_sample_rows": sample(stable_rows),
                "discrete_mode_rows": modes,
            },
        ),
        "mass_origin_qball_discrete_mass_spectrum": payload(
            "8.7.55.2.778",
            "Q-ball discrete mass spectrum from charge-mapped modes",
            {
                "mass_origin_qball_charge_discrete_frequency_inversion_json": "output/public/quantum/mass_origin_qball_charge_discrete_frequency_inversion_metrics.json",
            },
            "Compute the discrete mass proxy ladder E(beta_n) / c^2.",
            {
                "dimensionless_energy_density": "0.5 y'^2 + 0.5 (1 + beta^2) y^2 + y^3 + 0.25 y^4",
                "mass_proxy_rule": "M_n ∝ E(beta_n) and only ratios M_n / M_1 are used for gating",
            },
            [
                row("qball_discrete_mass_spectrum_complete", "pass", "discrete mass spectrum complete", 1, "Energy proxies evaluated on all discrete charge modes."),
                row("qball_discrete_mass_mode_count", "pass" if discrete_found else "reject", "discrete mass mode count", len(modes), f"{len(modes)} mass modes evaluated."),
                row("qball_mass_ratio_table_available", "pass" if len(modes) >= 1 else "reject", "mass ratio table available", 1 if len(modes) >= 1 else 0, "Mass ratios M_n / M_1 computed." if len(modes) >= 1 else "No mass ratios available."),
            ],
            {
                "discrete_mass_mode_count": len(modes),
                "reference_mode_index": int(modes[0]["mode_index"]) if modes else None,
                "mass_ratio_table_available": len(modes) >= 1,
                "mode_indices": [int(m["mode_index"]) for m in modes],
            },
            {
                "overall_status": "qball_discrete_mass_spectrum_frozen" if len(modes) >= 1 else "qball_discrete_mass_spectrum_missing",
                "keep_mass_origin_branch_blocked": True,
                "hand_off_to_8_7_55_2_84": False,
                "next_required_artifacts": ["qball_charge_mapped_mass_ratio_comparison"],
            },
            {
                "discrete_mass_mode_rows": modes,
            },
        ),
        "mass_origin_qball_charge_mapped_mass_ratio_comparison": payload(
            "8.7.55.2.779",
            "Q-ball charge-mapped mass-ratio comparison",
            {
                "mass_origin_qball_discrete_mass_spectrum_json": "output/public/quantum/mass_origin_qball_discrete_mass_spectrum_metrics.json",
            },
            "Compare the charge-mapped discrete mass ladder against canonical particle mass ratios.",
            {
                "targets": {
                    "m_p/m_e": 1836.15267343,
                    "m_mu/m_e": 206.7682830,
                    "m_tau/m_e": 3477.48,
                    "m_n/m_p": 1.00137842,
                },
                "pass_rule": "relative error <= threshold for at least one target",
            },
            [
                row("qball_charge_mapped_mass_ratio_comparison_complete", "pass", "charge-mapped mass-ratio comparison complete", 1, "Mass ratios compared against canonical targets."),
                row("qball_charge_mapped_discrete_spectrum_available", "pass" if discrete_found else "reject", "charge-mapped discrete spectrum available", 1 if discrete_found else 0, "A discrete ladder exists under direct charge mapping." if discrete_found else "No discrete ladder exists under direct charge mapping."),
                row("hand_off_to_8_7_55_2_84", "pass" if handoff else "reject", "handoff to 8.7.55.2.84 available", 1 if handoff else 0, "At least one target ratio is matched within threshold." if handoff else "No target ratio is matched within threshold."),
            ],
            {
                "discrete_spectrum_found": discrete_found,
                "mass_ratio_comparison_available": discrete_found,
                "hand_off_to_8_7_55_2_84": handoff,
                "closest_known_mass_ratio_or_none": best_match,
            },
            {
                "overall_status": "qball_charge_mapped_mass_ratio_handoff_pass" if handoff else ("qball_charge_mapped_mass_ratio_mismatch" if discrete_found else "qball_charge_mapped_mass_ratio_blocked"),
                "keep_mass_origin_branch_blocked": not handoff,
                "hand_off_to_8_7_55_2_84": handoff,
                "next_required_artifacts": [] if handoff else ["qball_charge_mapping_branch_refresh"],
            },
            {
                "mode_ratio_rows": ratio_rows,
                "closest_match_row": best_match,
            },
        ),
        "mass_origin_qball_charge_mapping_branch_refresh": payload(
            "8.7.55.2.780",
            "Q-ball charge-mapping branch refresh / disposition",
            {
                "mass_origin_qball_charge_mapped_mass_ratio_comparison_json": "output/public/quantum/mass_origin_qball_charge_mapped_mass_ratio_comparison_metrics.json",
            },
            "Refresh the branch after executing the direct charge-mapping instruction set.",
            {
                "disposition_cases": {
                    "case_a_handoff_pass": "discrete ladder found and particle-ratio gate passes",
                    "case_b_discrete_ladder_but_ratio_mismatch": "discrete ladder found but known-particle ratios do not match",
                    "case_c_insufficient_discrete_modes": "fewer than two modes found",
                }
            },
            [
                row("qball_charge_mapping_branch_refresh_complete", "pass", "charge-mapping branch refresh complete", 1, "Branch disposition refreshed."),
                row("qball_charge_mapping_discrete_ladder_found", "pass" if discrete_found else "reject", "direct charge-mapping discrete ladder found", 1 if discrete_found else 0, "Discrete ladder survives the direct mapping." if discrete_found else "Direct mapping still yields too few discrete modes."),
                row("qball_charge_mapping_known_ratio_match", "pass" if handoff else "reject", "known-particle ratio match found", 1 if handoff else 0, "At least one target ratio passes." if handoff else "No target ratio passes under direct charge mapping."),
            ],
            {
                "discrete_spectrum_found": discrete_found,
                "hand_off_to_8_7_55_2_84": handoff,
                "disposition_case": case_label,
                "new_branch_required": not handoff,
                "recommended_next_route_or_none": None if handoff else next_route,
            },
            {
                "overall_status": "qball_charge_mapping_branch_closed_handoff_pass" if handoff else "qball_charge_mapping_branch_closed_without_handoff",
                "keep_mass_origin_branch_blocked": not handoff,
                "hand_off_to_8_7_55_2_84": handoff,
                "new_branch_required": not handoff,
                "next_required_artifacts": [] if handoff else [next_route],
            },
            {
                "closest_match_row": best_match,
                "discrete_mode_rows": modes,
            },
        ),
        "mass_origin_qball_ratio_mismatch_route_contract": payload(
            "8.7.55.2.781",
            "Q-ball ratio-mismatch resolution route contract",
            {
                "mass_origin_qball_charge_mapping_branch_refresh_json": "output/public/quantum/mass_origin_qball_charge_mapping_branch_refresh_metrics.json",
            },
            "Freeze the next residual route after the direct charge-mapping ladder fails the known-particle mass-ratio gate.",
            {
                "selected_residual_route": next_route,
                "branch_case": case_label,
            },
            [
                row("qball_ratio_mismatch_route_contract_complete", "pass", "ratio-mismatch route contract complete", 1, "New residual branch frozen."),
                row("qball_ratio_mismatch_split_contract_ready", "pass", "ratio-mismatch split contract ready", 1, "Next branch may audit ratio invariance, normalization, and fallback priority."),
            ],
            {
                "selected_residual_route": next_route,
                "branch_case": case_label,
                "discrete_spectrum_found_under_direct_charge_mapping": discrete_found,
                "hand_off_to_8_7_55_2_84": handoff,
                "split_contract_ready": True,
            },
            {
                "overall_status": "qball_ratio_mismatch_route_contract_frozen" if not handoff else "qball_ratio_mismatch_route_not_needed",
                "keep_mass_origin_branch_blocked": not handoff,
                "hand_off_to_8_7_55_2_84": handoff,
                "next_required_artifacts": [] if handoff else ["qball_ratio_scale_invariance_audit", "qball_charge_operator_normalization_audit"],
            },
            {
                "closest_match_row": best_match,
            },
        ),
    }

    for stem, data in payloads.items():
        write_artifact(stem, data)
        print(f"[ok] wrote {OUT / (stem + '_metrics.json')}")
        print(f"[ok] wrote {OUT / (stem + '_rows.csv')}")


if __name__ == "__main__":
    main()
