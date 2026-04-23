#!/usr/bin/env python3
"""
Generate Trial-3 two-component spectrum / WZ comparison artifacts for 8.7.56.332-.334.

This branch executes the first honest full-spectrum pass after the two-component
coupled-Q-ball pivot. The goal is not to force a premature weak-sector closeout,
but to freeze whether the post-photon two-component nontransverse route lifts
the absolute ceiling and, if so, whether it also yields distinct W/Z anchors and
an acceptable Weinberg-angle proxy under the current no-new-parameter canon.
"""

from __future__ import annotations

import csv
import heapq
import importlib.util
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output" / "public" / "quantum"
STATUS = ROOT / "doc" / "STATUS.md"
ROADMAP = ROOT / "doc" / "ROADMAP.md"
AI_CONTEXT = ROOT / "doc" / "AI_CONTEXT_MIN.json"

ADVICE_CANDIDATES = (
    Path(r"C:\Users\ogawa\Downloads\pmodel_v2_trial3_two_component_pivot.md"),
    ROOT / "doc" / "quantum" / "pmodel_v2_trial3_two_component_pivot.md",
)

HELPER_BRANCH = ROOT / "scripts" / "quantum" / "mass_origin_v2_t3_post_ell18_amplitude_branch.py"
PIVOT_BRANCH = ROOT / "scripts" / "quantum" / "mass_origin_v2_trial3_two_component_pivot_branch.py"
NUMERICAL_BRANCH = ROOT / "scripts" / "quantum" / "mass_origin_vector_qball_numerical_solver_branch.py"
FULL_BRANCH = ROOT / "scripts" / "quantum" / "mass_origin_vector_qball_full_coupled_solver_branch.py"

PIVOT_ROUTE = OUT / "mass_origin_v2_trial3_two_component_pivot_route_contract_metrics.json"
PIVOT_ODE = OUT / "mass_origin_v2_trial3_two_component_coupled_radial_ode_derivation_metrics.json"
PIVOT_IMPLEMENTATION = OUT / "mass_origin_v2_trial3_two_component_shooting_solver_implementation_metrics.json"
POST_PHOTON_PRESERVATION = OUT / "mass_origin_v2_post_photon_vector_mass_ratio_preservation_audit_metrics.json"
SCALAR_SPECTRUM = OUT / "mass_origin_qball_discrete_mass_spectrum_metrics.json"
VECTOR_SPIN = OUT / "mass_origin_vector_qball_spin_orbit_freeze_audit_metrics.json"

ELL_VALUES = tuple(range(0, 31))
BETA_GRID = (0.50, 0.80, 0.95)
AMP0_GRID = (0.1, 2.0, 8.0)
AMPL_GRID = (2.0, 8.0)
TAIL_RATIO_THRESHOLD = 0.20
R_MAX = 25.0
MAX_STEP = 0.10
RTOL = 1.0e-7
ATOL = 1.0e-9
PAIR_NEAR_PASS_THRESHOLD = 0.15

W_MASS_MEV = 80369.0
Z_MASS_MEV = 91187.6
ELECTRON_MASS_MEV = 0.51099895
W_TARGET = W_MASS_MEV / ELECTRON_MASS_MEV
Z_TARGET = Z_MASS_MEV / ELECTRON_MASS_MEV
WZ_RATIO_TARGET = W_MASS_MEV / Z_MASS_MEV
SIN2_THETA_W_TARGET = 1.0 - WZ_RATIO_TARGET * WZ_RATIO_TARGET
PASS_THRESHOLD = 0.10


# 関数: 現在の UTC 時刻を ISO 8601 形式で返す。
def now_iso() -> str:
    """Return the current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


# 関数: 必須入力 artifact の存在を確認する。

def req(path: Path) -> None:
    """Abort immediately when a required input artifact is missing."""
    if not path.exists():
        raise SystemExit(f"[fail] missing required input: {path}")


# 関数: UTF-8 JSON artifact を辞書として読む。

def read_json(path: Path) -> dict:
    """Load a UTF-8 JSON artifact into a Python dictionary."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


# 関数: UTF-8 テキスト source を文字列として読む。

def read_text(path: Path) -> str:
    """Read a UTF-8 text source into memory."""
    return path.read_text(encoding="utf-8")


# 関数: 外部メモを repo 内候補へフォールバックして任意入力として解決する。

def resolve_optional_advice() -> tuple[Path | None, str]:
    """Return the first available expert-note path and text, or empty text."""
    for path in ADVICE_CANDIDATES:
        if path.exists():
            return path, read_text(path)

    return None, ""


# 関数: 絶対パスを repo 相対表記へ変換する。

def rel(path: Path) -> str:
    """Return a repo-relative POSIX-style path string."""
    return str(path.relative_to(ROOT)).replace("\\", "/")


# 関数: source 内で最初に一致した pattern の行情報を返す。

def hit(text: str, pattern: str) -> dict | None:
    """Return the first line match for a substring pattern, if any."""
    for line_no, line in enumerate(text.splitlines(), start=1):
        if pattern in line:
            return {"pattern": pattern, "line": line_no, "text": line.strip()}

    return None


# 関数: 共通 schema の metrics row を組み立てる。

def row(row_id: str, status: str, metric: str, value: float, note: str) -> dict:
    """Build a standard metrics row payload."""
    return {
        "row_id": row_id,
        "status": status,
        "metric": metric,
        "value": float(value),
        "note": note,
    }


# 関数: 共通 schema の payload を組み立てる。

def payload(
    step: str,
    name: str,
    inputs: dict,
    intent: str,
    formulas: dict,
    rows: list[dict],
    summary: dict,
    decision: dict,
    evidence: dict,
) -> dict:
    """Build the standard JSON metrics payload used across the roadmap."""
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


# 関数: JSON artifact と rows CSV を side-by-side で保存する。

def write_artifact(stem: str, data: dict) -> None:
    """Write the metrics payload as JSON and as a rows CSV sidecar."""
    OUT.mkdir(parents=True, exist_ok=True)
    json_path = OUT / f"{stem}_metrics.json"
    csv_path = OUT / f"{stem}_rows.csv"
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["row_id", "status", "metric", "value", "note"])
        writer.writeheader()
        writer.writerows(data["rows"])


# 関数: long row table から要約サンプルだけを返す。

def sample(rows: list[dict], count: int = 12) -> list[dict]:
    """Return a sparse sample of long tables for compact evidence payloads."""
    if len(rows) <= count:
        return rows

    step = max(1, len(rows) // count)
    sampled = [rows[index] for index in range(0, len(rows), step)]
    return sampled[:count]


# 関数: local Python module を動的 import する。

def load_module(path: Path, module_name: str):
    """Load a local Python module from a filesystem path."""
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"[fail] unable to import module: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# 関数: two-component coupled radial profile を current pilot contract で解く。

def solve_two_component_profile(
    pivot,
    numerical,
    beta: float,
    ell: int,
    amp0: float,
    amp_l: float,
) -> dict:
    """Integrate the pilot two-component profile and return compact observables."""
    r0 = 1.0e-4
    y0 = [float(amp0), 0.0, float(amp_l) * r0, float(amp_l)]

    # 関数: current pilot coupled radial ODE の右辺を返す。
    def ode(radius: float, y: np.ndarray) -> list[float]:
        f0, f0_prime, f_l, f_l_prime = [float(value) for value in y]
        rr = max(float(radius), 1.0e-6)
        k_proxy = math.sqrt(max(float(ell * (ell + 1)), 0.0)) / rr
        coupling = float(beta) * k_proxy
        rho = math.sqrt(max(f0 * f0 + f_l * f_l, 0.0))
        nonlinear_coeff = 3.0 * rho + rho * rho
        f0_double_prime = (
            -(2.0 / rr) * f0_prime
            - (float(beta * beta) - float(pivot.RADIAL_MASS_SQUARED)) * f0
            - coupling * f_l
            - nonlinear_coeff * f0
        )
        f_l_double_prime = (
            -(2.0 / rr) * f_l_prime
            + (float(ell * (ell + 1)) / (rr * rr)) * f_l
            - (float(beta * beta) - float(pivot.LONGITUDINAL_DIRECT_MASS_SQUARED)) * f_l
            - coupling * f0
            - nonlinear_coeff * f_l
        )
        return [f0_prime, f0_double_prime, f_l_prime, f_l_double_prime]

    sol = solve_ivp(
        ode,
        (r0, float(R_MAX)),
        y0,
        max_step=float(MAX_STEP),
        rtol=float(RTOL),
        atol=float(ATOL),
    )
    radius = sol.t
    f0_values = sol.y[0]
    f0_prime_values = sol.y[1]
    f_l_values = sol.y[2]
    f_l_prime_values = sol.y[3]
    rr = np.maximum(radius, 1.0e-6)
    rho = np.sqrt(np.maximum(f0_values * f0_values + f_l_values * f_l_values, 0.0))
    k_proxy = math.sqrt(max(float(ell * (ell + 1)), 0.0)) / rr
    coupling_density = float(beta) * k_proxy * f0_values * f_l_values
    energy_density = (
        0.5 * (f0_prime_values * f0_prime_values + f_l_prime_values * f_l_prime_values)
        + 0.5 * (float(pivot.RADIAL_MASS_SQUARED) + float(beta * beta)) * f0_values * f0_values
        + 0.5
        * (
            float(beta * beta)
            + float(ell * (ell + 1)) / (rr * rr)
            + float(pivot.LONGITUDINAL_DIRECT_MASS_SQUARED)
        )
        * f_l_values
        * f_l_values
        + coupling_density
        + rho**3
        + 0.25 * rho**4
    )
    charge_density = float(beta) * (f0_values * f0_values + f_l_values * f_l_values)
    charge_proxy = float(np.trapezoid(4.0 * math.pi * radius * radius * charge_density, radius))
    energy_proxy = float(np.trapezoid(4.0 * math.pi * radius * radius * energy_density, radius))
    tail_norm = math.sqrt(float(f0_values[-1] * f0_values[-1] + f_l_values[-1] * f_l_values[-1]))
    input_norm = math.sqrt(float(amp0 * amp0 + amp_l * amp_l))
    k0 = int(numerical.count_radial_nodes(np.asarray(f0_values, dtype=float)))
    k_l = int(numerical.count_radial_nodes(np.asarray(f_l_values, dtype=float)))
    return {
        "success": bool(sol.success),
        "beta": float(beta),
        "ell": int(ell),
        "amp0": float(amp0),
        "amp_l": float(amp_l),
        "component_ratio_amp_l_to_amp0": None if amp0 == 0.0 else float(amp_l / amp0),
        "tail_norm": float(tail_norm),
        "input_norm": float(input_norm),
        "tail_to_input_ratio": None if input_norm == 0.0 else float(tail_norm / input_norm),
        "charge_proxy": float(charge_proxy),
        "energy_proxy": float(energy_proxy),
        "node_count_k0": int(k0),
        "node_count_kL": int(k_l),
        "node_count_k": max(int(k0), int(k_l)),
        "max_abs_f0": float(np.max(np.abs(f0_values))),
        "max_abs_fL": float(np.max(np.abs(f_l_values))),
        "final_f0": float(f0_values[-1]),
        "final_fL": float(f_l_values[-1]),
        "r_max": float(R_MAX),
        "max_step": float(MAX_STEP),
    }


# 関数: coarse two-component spectrum scan を ell=0..30 へ展開する。

def run_two_component_scan(pivot, numerical) -> tuple[list[dict], dict[str, dict]]:
    """Run the first-pass two-component scan and return localized rows plus sector summaries."""
    localized_rows: list[dict] = []
    sector_summary: dict[str, dict] = {}
    evaluations_per_sector = len(BETA_GRID) * len(AMP0_GRID) * len(AMPL_GRID)

    for ell in ELL_VALUES:
        sector_candidates = 0
        sector_best_tail = None
        sector_best_candidate = None
        localized_beta_values: list[float] = []

        for beta in BETA_GRID:
            best_by_k: dict[int, dict] = {}
            for amp0 in AMP0_GRID:
                for amp_l in AMPL_GRID:
                    solved = solve_two_component_profile(
                        pivot,
                        numerical,
                        float(beta),
                        int(ell),
                        float(amp0),
                        float(amp_l),
                    )
                    tail_ratio = solved["tail_to_input_ratio"]
                    if not solved["success"] or tail_ratio is None:
                        continue

                    if sector_best_tail is None or float(tail_ratio) < float(sector_best_tail):
                        sector_best_tail = float(tail_ratio)
                        sector_best_candidate = solved

                    if float(tail_ratio) > float(TAIL_RATIO_THRESHOLD):
                        continue

                    k_value = int(solved["node_count_k"])
                    current = best_by_k.get(k_value)
                    if current is None or float(tail_ratio) < float(current["tail_to_input_ratio"]):
                        best_by_k[k_value] = solved

            if not best_by_k:
                continue

            localized_beta_values.append(float(beta))
            for branch_index, k_value in enumerate(sorted(best_by_k), start=1):
                localized = dict(best_by_k[k_value])
                localized["localized_solution_found"] = True
                localized["solution_branch_index"] = int(branch_index)
                localized_rows.append(localized)
                sector_candidates += 1

        sector_summary[str(int(ell))] = {
            "evaluations": int(evaluations_per_sector),
            "localized_solution_count": int(sector_candidates),
            "localized_beta_values": [float(value) for value in localized_beta_values],
            "best_tail_ratio_or_none": None if sector_best_tail is None else float(sector_best_tail),
            "best_candidate_or_none": sector_best_candidate,
        }

    return localized_rows, sector_summary


# 関数: localized charge curve を整数 n に補間して base mode table を作る。

def interpolate_two_component_modes(localized_rows: list[dict]) -> tuple[list[dict], dict[str, dict]]:
    """Interpolate integer charge targets from localized two-component scan rows."""
    grouped: dict[tuple[int, int], list[dict]] = {}
    for localized in localized_rows:
        ell = int(localized["ell"])
        k_value = int(localized["node_count_k"])
        grouped.setdefault((ell, k_value), []).append(localized)

    base_modes: list[dict] = []
    mode_summary: dict[str, dict] = {}
    for (ell, k_value), rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda item: float(item["beta"]))
        if len(rows) < 2:
            mode_summary[f"{ell}:{k_value}"] = {
                "point_count": len(rows),
                "integer_mode_count": 0,
                "charge_window_or_none": None,
            }
            continue

        q_min = int(math.ceil(min(float(item["charge_proxy"]) for item in rows)))
        q_max = int(math.floor(max(float(item["charge_proxy"]) for item in rows)))
        if q_max < q_min:
            mode_summary[f"{ell}:{k_value}"] = {
                "point_count": len(rows),
                "integer_mode_count": 0,
                "charge_window_or_none": None,
            }
            continue

        integer_mode_count = 0
        for charge_index in range(q_min, q_max + 1):
            beta_n = None
            energy_n = None
            for left, right in zip(rows[:-1], rows[1:]):
                q_left = float(left["charge_proxy"])
                q_right = float(right["charge_proxy"])
                if (q_left - charge_index) * (q_right - charge_index) > 0.0:
                    continue

                if q_right == q_left:
                    fraction = 0.0
                else:
                    fraction = (charge_index - q_left) / (q_right - q_left)

                beta_n = float(left["beta"]) + fraction * (float(right["beta"]) - float(left["beta"]))
                energy_n = float(left["energy_proxy"]) + fraction * (
                    float(right["energy_proxy"]) - float(left["energy_proxy"])
                )
                break

            if beta_n is None or energy_n is None:
                continue

            integer_mode_count += 1
            base_modes.append(
                {
                    "n": int(charge_index),
                    "k": int(k_value),
                    "ell": int(ell),
                    "beta_n": float(beta_n),
                    "charge_proxy_target": float(charge_index),
                    "base_mass_proxy": float(energy_n),
                    "node_count_k": int(k_value),
                    "origin": "two_component_first_pass",
                }
            )

        mode_summary[f"{ell}:{k_value}"] = {
            "point_count": len(rows),
            "integer_mode_count": int(integer_mode_count),
            "charge_window_or_none": [int(q_min), int(q_max)],
        }

    base_modes = sorted(base_modes, key=lambda item: (int(item["ell"]), int(item["k"]), int(item["n"])))
    return base_modes, mode_summary


# 関数: high-mass candidate table から最良の W/Z pair を探索する。

def best_ratio_pair_fast(rows: list[dict], top_count: int = 1500) -> dict | None:
    """Return the best W/Z ratio pair among the heaviest candidate rows."""
    if len(rows) < 2:
        return None

    candidates = heapq.nlargest(top_count, rows, key=lambda item: float(item["mass_ratio_to_scalar_base"]))
    best = None
    for index, left in enumerate(candidates):
        left_ratio = float(left["mass_ratio_to_scalar_base"])
        for right in candidates[index + 1 :]:
            right_ratio = float(right["mass_ratio_to_scalar_base"])
            heavier_row = left if left_ratio >= right_ratio else right
            lighter_row = right if left_ratio >= right_ratio else left
            heavier = max(left_ratio, right_ratio)
            lighter = min(left_ratio, right_ratio)
            ratio_value = lighter / heavier
            ratio_error = abs(ratio_value - WZ_RATIO_TARGET) / WZ_RATIO_TARGET
            sin2_value = 1.0 - ratio_value * ratio_value
            sin2_error = abs(sin2_value - SIN2_THETA_W_TARGET) / SIN2_THETA_W_TARGET
            record = {
                "lighter_state": {
                    "n": int(lighter_row["n"]),
                    "k": int(lighter_row["k"]),
                    "ell": int(lighter_row["ell"]),
                    "s": int(lighter_row["s"]),
                    "mass_ratio_to_electron": float(lighter),
                },
                "heavier_state": {
                    "n": int(heavier_row["n"]),
                    "k": int(heavier_row["k"]),
                    "ell": int(heavier_row["ell"]),
                    "s": int(heavier_row["s"]),
                    "mass_ratio_to_electron": float(heavier),
                },
                "mw_mz_ratio_value": float(ratio_value),
                "mw_mz_ratio_relative_error": float(ratio_error),
                "sin2_theta_w_value": float(sin2_value),
                "sin2_theta_w_relative_error": float(sin2_error),
                "passes_threshold": bool(ratio_error <= PASS_THRESHOLD and sin2_error <= PASS_THRESHOLD),
            }
            key = (record["mw_mz_ratio_relative_error"], record["sin2_theta_w_relative_error"])
            if best is None or key < (
                best["mw_mz_ratio_relative_error"],
                best["sin2_theta_w_relative_error"],
            ):
                best = record

    return best


# 関数: Trial-3 two-component spectrum / WZ comparison branch を実行する。

def main() -> None:
    """Execute the first full-spectrum two-component Trial-3 branch."""
    for path in (
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        HELPER_BRANCH,
        PIVOT_BRANCH,
        NUMERICAL_BRANCH,
        FULL_BRANCH,
        PIVOT_ROUTE,
        PIVOT_ODE,
        PIVOT_IMPLEMENTATION,
        POST_PHOTON_PRESERVATION,
        SCALAR_SPECTRUM,
        VECTOR_SPIN,
    ):
        req(path)

    advice_path, advice_text = resolve_optional_advice()
    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    ai_context = read_json(AI_CONTEXT)
    helper_text = read_text(HELPER_BRANCH)
    numerical_text = read_text(NUMERICAL_BRANCH)
    full_text = read_text(FULL_BRANCH)

    helper = load_module(HELPER_BRANCH, "trial3_two_component_helper")
    pivot = load_module(PIVOT_BRANCH, "trial3_two_component_pivot")
    numerical = load_module(NUMERICAL_BRANCH, "trial3_two_component_numerical")
    full = load_module(FULL_BRANCH, "trial3_two_component_full")

    pivot_route = read_json(PIVOT_ROUTE)
    pivot_ode = read_json(PIVOT_ODE)
    pivot_implementation = read_json(PIVOT_IMPLEMENTATION)
    post_photon_preservation = read_json(POST_PHOTON_PRESERVATION)
    scalar_spectrum = read_json(SCALAR_SPECTRUM)
    vector_spin = read_json(VECTOR_SPIN)

    scalar_modes = list(scalar_spectrum["evidence"]["discrete_mass_mode_rows"])
    lambda_rot = float(vector_spin["summary"]["lambda_rot_value"])
    normalization_scale = float(post_photon_preservation["summary"]["absolute_mass_normalization_scale_factor"])
    historic_ceiling = float(pivot_route["summary"]["current_same_family_ceiling_to_electron"])

    common_inputs = {
        "expert_note_markdown": str(advice_path) if advice_path else str(ADVICE_CANDIDATES[0]),
        "expert_note_available": advice_path is not None,
        "expert_note_candidates": [str(path) for path in ADVICE_CANDIDATES],
        "status_markdown": rel(STATUS),
        "roadmap_markdown": rel(ROADMAP),
        "ai_context_json": rel(AI_CONTEXT),
        "mass_origin_v2_trial3_two_component_pivot_route_contract_json": rel(PIVOT_ROUTE),
        "mass_origin_v2_trial3_two_component_coupled_radial_ode_derivation_json": rel(PIVOT_ODE),
        "mass_origin_v2_trial3_two_component_shooting_solver_implementation_json": rel(PIVOT_IMPLEMENTATION),
        "mass_origin_v2_post_photon_vector_mass_ratio_preservation_audit_json": rel(POST_PHOTON_PRESERVATION),
        "mass_origin_qball_discrete_mass_spectrum_json": rel(SCALAR_SPECTRUM),
        "mass_origin_vector_qball_spin_orbit_freeze_audit_json": rel(VECTOR_SPIN),
        "mass_origin_v2_t3_post_ell18_amplitude_branch_py": rel(HELPER_BRANCH),
        "mass_origin_vector_qball_numerical_solver_branch_py": rel(NUMERICAL_BRANCH),
        "mass_origin_vector_qball_full_coupled_solver_branch_py": rel(FULL_BRANCH),
    }

    localized_rows, sector_summary = run_two_component_scan(pivot, numerical)
    base_modes, mode_summary = interpolate_two_component_modes(localized_rows)
    exact_rows = full.build_exact_ladder(scalar_modes, base_modes, lambda_rot)
    normalized_vector_rows = helper.normalize_vector_rows(
        [row_data for row_data in exact_rows if int(row_data["ell"]) > 0],
        normalization_scale,
    )
    best_w = helper.closest_state(normalized_vector_rows, W_TARGET)
    best_z = helper.closest_state(normalized_vector_rows, Z_TARGET)
    best_pair = best_ratio_pair_fast(normalized_vector_rows)
    max_row = max(normalized_vector_rows, key=lambda item: float(item["mass_ratio_to_scalar_base"]))
    localized_ell_values = sorted({int(row_data["ell"]) for row_data in localized_rows})
    k_positive_mode_count = len([mode for mode in base_modes if int(mode["k"]) > 0])
    anchor_collapsed = bool(
        best_w
        and best_z
        and best_w["n"] == best_z["n"]
        and best_w["k"] == best_z["k"]
        and best_w["ell"] == best_z["ell"]
        and best_w["s"] == best_z["s"]
    )
    ceiling_lifted = float(max_row["mass_ratio_to_scalar_base"]) > historic_ceiling
    ceiling_surpasses_w = float(max_row["mass_ratio_to_scalar_base"]) >= W_TARGET
    ceiling_surpasses_z = float(max_row["mass_ratio_to_scalar_base"]) >= Z_TARGET
    w_anchor_pass = bool(best_w and best_w["passes_threshold"])
    z_anchor_pass = bool(best_z and best_z["passes_threshold"])
    mw_mz_ratio_pass = bool(best_pair and best_pair["mw_mz_ratio_relative_error"] <= PASS_THRESHOLD)
    sin2_theta_w_pass = bool(best_pair and best_pair["sin2_theta_w_relative_error"] <= PASS_THRESHOLD)
    best_pair_near_pass = bool(best_pair and best_pair["mw_mz_ratio_relative_error"] <= PAIR_NEAR_PASS_THRESHOLD)
    trial3_closeable = bool(w_anchor_pass and z_anchor_pass and mw_mz_ratio_pass and sin2_theta_w_pass and not anchor_collapsed)

    if trial3_closeable:
        case_label = "case_a_two_component_wz_pack_closed"
        selected_residual_route = None
        missing_v2_artifact = None
        recommended_next_route = None
    elif ceiling_lifted:
        case_label = "case_b_two_component_ceiling_lifted_without_distinct_wz_closeout"
        selected_residual_route = "trial3_two_component_distinct_w_z_anchor_split_identification"
        missing_v2_artifact = "trial3_two_component_distinct_w_z_anchor_split_pack"
        recommended_next_route = "8.7.56.335"
    else:
        case_label = "case_c_two_component_no_gain_under_first_pass_scan"
        selected_residual_route = "trial3_two_component_localization_gain_reaudit_identification"
        missing_v2_artifact = "trial3_two_component_localization_gain_reaudit_pack"
        recommended_next_route = "8.7.56.335"

    spectrum = payload(
        "8.7.56.332",
        "Trial-3 two-component spectrum computation",
        common_inputs,
        "Execute the first full ell=0..30 two-component coupled-Q-ball scan and freeze the localized spectrum / integer-mode table under the current no-new-parameter canon.",
        {
            "scan_rule": "scan ell = 0..30 over beta in {0.50, 0.80, 0.95}, amp0 in {0.1, 2.0, 8.0}, ampL in {2.0, 8.0}, keep best-by-k profiles with tail_to_input_ratio <= 0.20",
            "integer_mode_rule": "for each localized (ell,k) branch, interpolate charge_proxy(beta) to integer n targets and inherit base_mass_proxy from linear interpolation in beta",
            "exact_ladder_rule": "reuse the frozen full-coupled exact ladder builder and apply the post-photon normalization scale sqrt(2) only at the absolute-mass level",
        },
        [
            row("trial3_two_component_spectrum_computation_complete", "pass", "Trial-3 two-component spectrum computation complete", 1, "The first full ell=0..30 two-component scan is frozen."),
            row("trial3_two_component_localized_solution_count", "pass" if localized_rows else "reject", "localized solution count under two-component scan", len(localized_rows), "The first-pass two-component scan must produce localized sectors to be scientifically meaningful."),
            row("trial3_two_component_integer_mode_count", "pass" if base_modes else "reject", "integer mode count under two-component scan", len(base_modes), "Localized sectors must interpolate to integer charge modes before exact-family comparison."),
            row("trial3_two_component_exact_vector_row_count", "pass" if normalized_vector_rows else "reject", "exact vector row count under two-component scan", len(normalized_vector_rows), "The two-component scan must rebuild a non-empty exact-family vector table."),
            row("trial3_two_component_ceiling_lifted_vs_single_component", "pass" if ceiling_lifted else "reject", "two-component ceiling lifts beyond one-component physical ceiling", 1 if ceiling_lifted else 0, "The first honest question is whether the two-component route lifts the stalled one-component ceiling at all."),
        ],
        {
            "scan_grid": {
                "ell_values": [int(value) for value in ELL_VALUES],
                "beta_grid": [float(value) for value in BETA_GRID],
                "amp0_grid": [float(value) for value in AMP0_GRID],
                "amp_l_grid": [float(value) for value in AMPL_GRID],
            },
            "tail_ratio_threshold": float(TAIL_RATIO_THRESHOLD),
            "localized_solution_count_total": len(localized_rows),
            "localized_ell_values": localized_ell_values,
            "base_mode_count_total": len(base_modes),
            "exact_vector_row_count_total": len(normalized_vector_rows),
            "k_positive_mode_count": int(k_positive_mode_count),
            "historic_single_component_ceiling_to_electron": historic_ceiling,
            "two_component_rebuilt_ceiling_to_electron": float(max_row["mass_ratio_to_scalar_base"]),
            "ceiling_lifted_vs_single_component": ceiling_lifted,
            "ceiling_surpasses_w_threshold": ceiling_surpasses_w,
            "ceiling_surpasses_z_threshold": ceiling_surpasses_z,
            "best_ceiling_row_or_none": {
                "n": int(max_row["n"]),
                "k": int(max_row["k"]),
                "ell": int(max_row["ell"]),
                "s": int(max_row["s"]),
                "mass_ratio_to_electron": float(max_row["mass_ratio_to_scalar_base"]),
            },
            "next_required_route": "trial3_two_component_wz_target_comparison",
        },
        {
            "overall_status": "trial3_two_component_spectrum_computed",
            "advance_to_8_7_56_333": True,
            "next_required_artifacts": ["trial3_two_component_wz_target_comparison"],
        },
        {
            "advice_step3_line": hit(advice_text, "### Step 3: 2成分スペクトルの計算"),
            "advice_full_scan_line": hit(advice_text, "solver を ell=0..30 に走らせる"),
            "status_next_step_line": hit(status_text, "current official next step は `8.7.56.332`"),
            "roadmap_branch_line": hit(roadmap_text, "`8.7.56.329-.334` 試練3 two-component coupled-Q-ball pivot branch"),
            "helper_normalize_line": hit(helper_text, "def normalize_vector_rows(rows: list[dict], scale_factor: float) -> list[dict]:"),
            "numerical_node_counter_line": hit(numerical_text, "def count_radial_nodes(field: np.ndarray) -> int:"),
            "full_exact_builder_line": hit(full_text, "def build_exact_ladder("),
            "sector_summary": sector_summary,
            "mode_summary": mode_summary,
            "localized_row_sample": sample(localized_rows, 16),
            "base_mode_sample": sample(base_modes, 16),
            "exact_row_sample": sample(
                sorted(normalized_vector_rows, key=lambda item: float(item["mass_ratio_to_scalar_base"]), reverse=True),
                16,
            ),
        },
    )

    wz_comparison = payload(
        "8.7.56.333",
        "Trial-3 two-component W/Z target comparison",
        common_inputs,
        "Compare the rebuilt two-component exact-family table against W/Z targets and freeze whether the first-pass route closes absolute anchors, pair shape, and the Weinberg-angle proxy.",
        {
            "target_rule": "W/e = 80369 / 0.51099895 and Z/e = 91187.6 / 0.51099895",
            "pair_rule": "pick the pair minimizing (relative_error(M_W/M_Z), relative_error(sin^2(theta_W))) over the heaviest candidate table slice",
            "collapse_rule": "if the best W and best Z anchors collapse onto the same state, the next honest blocker is distinct anchor splitting rather than absolute ceiling",
        },
        [
            row("trial3_two_component_wz_target_comparison_complete", "pass", "Trial-3 two-component W/Z target comparison complete", 1, "The first-pass two-component W/Z comparison is frozen."),
            row("trial3_two_component_w_anchor_pass", "pass" if w_anchor_pass else "reject", "two-component W/e anchor passes", 1 if w_anchor_pass else 0, "The W anchor must land within the fixed relative-error threshold."),
            row("trial3_two_component_z_anchor_pass", "pass" if z_anchor_pass else "reject", "two-component Z/e anchor passes", 1 if z_anchor_pass else 0, "The Z anchor must land within the fixed relative-error threshold."),
            row("trial3_two_component_mw_mz_ratio_pass", "pass" if mw_mz_ratio_pass else "reject", "two-component M_W/M_Z ratio passes", 1 if mw_mz_ratio_pass else 0, "A viable weak-sector pair must close the mass ratio together with the anchors."),
            row("trial3_two_component_sin2_theta_w_pass", "pass" if sin2_theta_w_pass else "reject", "two-component sin^2(theta_W) passes", 1 if sin2_theta_w_pass else 0, "The Weinberg-angle proxy must close together with the W/Z pair."),
            row("trial3_two_component_anchor_collapsed", "reject" if anchor_collapsed else "pass", "best W and Z anchors collapse onto the same state", 1 if anchor_collapsed else 0, "Collapsed anchors indicate that the route has lifted the ceiling but still lacks distinct weak-boson splitting."),
        ],
        {
            "best_w_row_or_none": best_w,
            "best_z_row_or_none": best_z,
            "best_pair_or_none": best_pair,
            "w_gap_factor_or_none": None if best_w is None else W_TARGET / float(best_w["ratio_value"]),
            "z_gap_factor_or_none": None if best_z is None else Z_TARGET / float(best_z["ratio_value"]),
            "anchor_collapsed": anchor_collapsed,
            "best_pair_near_pass": best_pair_near_pass,
            "trial3_closeable_under_first_pass_scan": trial3_closeable,
            "case_label": case_label,
            "next_required_route": "trial3_two_component_declaration_gate",
        },
        {
            "overall_status": "trial3_two_component_wz_target_compared",
            "advance_to_8_7_56_334": True,
            "next_required_artifacts": ["trial3_two_component_declaration_gate"],
        },
        {
            "pivot_route_summary": pivot_route["summary"],
            "pivot_ode_summary": pivot_ode["summary"],
            "pivot_implementation_summary": pivot_implementation["summary"],
            "best_w_row_or_none": best_w,
            "best_z_row_or_none": best_z,
            "best_pair_or_none": best_pair,
            "max_row_or_none": {
                "n": int(max_row["n"]),
                "k": int(max_row["k"]),
                "ell": int(max_row["ell"]),
                "s": int(max_row["s"]),
                "mass_ratio_to_electron": float(max_row["mass_ratio_to_scalar_base"]),
            },
        },
    )

    declaration = payload(
        "8.7.56.334",
        "Trial-3 two-component declaration gate",
        common_inputs,
        "Freeze the first-pass judgment of the two-component route and refresh the Trial-2 / Trial-4 side disposition based on whether the current blocker is absolute ceiling or distinct W/Z splitting.",
        {
            "closeout_rule": "close Trial-3 only if distinct W and Z anchors, the pair ratio, and sin^2(theta_W) all pass together under the no-new-parameter two-component canon",
            "residual_rule": "if the ceiling lifts but W/Z anchors collapse onto the same state, the next honest blocker is distinct W/Z anchor splitting rather than further one-component ceiling work",
            "reserve_rule": "Trial-2 paper-side sync remains unlocked reserve retained while the scientifically honest Trial-3 route is still open",
        },
        [
            row("trial3_two_component_declaration_gate_complete", "pass", "Trial-3 two-component declaration gate complete", 1, "The first-pass two-component declaration gate is frozen."),
            row("trial3_two_component_branch_closeable", "pass" if trial3_closeable else "reject", "two-component Trial-3 branch closeable under first-pass scan", 1 if trial3_closeable else 0, "The branch closes only if anchors, pair ratio, and Weinberg-angle proxy all pass without collapse."),
            row("trial3_two_component_residual_route_required", "reject" if trial3_closeable else "pass", "two-component residual route required after first-pass scan", 0 if trial3_closeable else 1, "A residual route is still required unless the first-pass two-component scan already closes the weak-sector pack."),
            row("trial2_paper_side_sync_reserve_retained_after_two_component_gate", "pass", "Trial-2 paper-side sync reserve retained after two-component gate", 1, "Trial-2 paper sync remains unlocked reserve work while the two-component Trial-3 route stays open."),
            row("trial4_deferred_retained_after_two_component_gate", "pass", "Trial-4 deferred retained after two-component gate", 1, "Trial-4 remains deferred while Trial-3 still has an honest current-canon search axis."),
        ],
        {
            "trial3_current_branch_closeable": trial3_closeable,
            "case_label": case_label,
            "selected_residual_route": selected_residual_route,
            "missing_v2_artifact": missing_v2_artifact,
            "recommended_next_route_or_none": recommended_next_route,
            "trial2_paper_side_sync_state": "unlocked_reserve_retained",
            "trial4_deferred": True,
        },
        {
            "overall_status": "trial3_two_component_declaration_gate_frozen",
            "trial3_branch_closeable": trial3_closeable,
            "advance_to_next_branch": not trial3_closeable,
            "next_required_artifacts": [] if trial3_closeable else [recommended_next_route],
        },
        {
            "status_current_step_before_branch": ai_context.get("current_step") or ai_context.get("focus") or ai_context.get("next"),
            "spectrum_summary": spectrum["summary"],
            "wz_comparison_summary": wz_comparison["summary"],
            "advice_split_line": hit(advice_text, "W / Z は 1成分 ladder の high-ell 延長ではなく、2成分 coupled mode で割れる可能性が高い。"),
            "advice_step4_line": hit(advice_text, "### Step 4: W/Z 同定と split 判定"),
        },
    )

    write_artifact("mass_origin_v2_trial3_two_component_spectrum_computation", spectrum)
    write_artifact("mass_origin_v2_trial3_two_component_wz_target_comparison", wz_comparison)
    write_artifact("mass_origin_v2_trial3_two_component_declaration_gate", declaration)

    print("[done] Trial-3 two-component spectrum artifacts written:")
    print(" - mass_origin_v2_trial3_two_component_spectrum_computation_metrics.json")
    print(" - mass_origin_v2_trial3_two_component_wz_target_comparison_metrics.json")
    print(" - mass_origin_v2_trial3_two_component_declaration_gate_metrics.json")


# 関数: CLI から branch を実行する。

def run_cli() -> None:
    """CLI entry point for the Trial-3 two-component spectrum branch."""
    main()


if __name__ == "__main__":
    run_cli()
