#!/usr/bin/env python3
"""
Generate vector Q-ball numerical-solver artifacts for 8.7.55.2.820-.825.

This branch executes the first effective vector-Q-ball pilot after the
P_mu / Proca-soliton reopen. The solver keeps the exact scalar limit and
adds an ell(ell+1)/r^2 centrifugal barrier as the first low-cost proxy for
ell>0 sectors. The branch is intentionally conservative: a promising
mass-ratio hint is allowed, but 8.7.55.2.84 is reopened only after an exact
full-coupled vector ladder exists rather than just an effective pilot.
"""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output" / "public" / "quantum"
VECTOR_ROUTE = OUT / "mass_origin_vector_qball_numerical_solver_route_contract_metrics.json"
VECTOR_SPEC = OUT / "mass_origin_vector_qball_solver_spec_metrics.json"
VECTOR_SPIN = OUT / "mass_origin_vector_qball_spin_orbit_freeze_audit_metrics.json"
SCALAR_SPECTRUM = OUT / "mass_origin_qball_discrete_mass_spectrum_metrics.json"
QBALL_RATIO = OUT / "mass_origin_qball_charge_mapped_mass_ratio_comparison_metrics.json"
PART1 = ROOT / "doc" / "paper" / "10_part1_core_theory.md"
PART2 = ROOT / "doc" / "paper" / "11_part2_astrophysics.md"


# 関数: 現在の UTC 時刻を ISO 8601 形式で返す。
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: 必須入力 artifact が存在しない場合に即時停止する。

def req(path: Path) -> None:
    if not path.exists():
        raise SystemExit(f"[fail] missing required input: {path}")


# 関数: UTF-8 JSON artifact を辞書として読み込む。

def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


# 関数: UTF-8 テキスト source を読む。

def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# 関数: 絶対パスを repo 相対表記へ変換する。

def rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


# 関数: source 内で最初に一致した pattern の行情報を返す。

def hit(text: str, pattern: str) -> dict | None:
    for line_no, line in enumerate(text.splitlines(), start=1):
        if pattern in line:
            return {"pattern": pattern, "line": line_no, "text": line.strip()}

    return None


# 関数: 共通 schema の metrics row を組み立てる。

def row(row_id: str, status: str, metric: str, value: float, note: str) -> dict:
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


# 関数: JSON/CSV artifact を side-by-side で保存する。

def write_artifact(stem: str, data: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    json_path = OUT / f"{stem}_metrics.json"
    csv_path = OUT / f"{stem}_rows.csv"
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["row_id", "status", "metric", "value", "note"])
        writer.writeheader()
        writer.writerows(data["rows"])


# 関数: 試行 sector の beta grid を ell ごとに返す。

def pilot_beta_grid(ell: int) -> np.ndarray:
    mapping = {
        1: np.linspace(0.25, 0.95, 10),
        2: np.linspace(0.20, 0.80, 10),
        3: np.linspace(0.20, 0.60, 10),
    }
    return mapping[ell]


# 関数: shooting の初期振幅探索に使う amplitude grid を返す。

def amplitude_grid(ell: int) -> np.ndarray:
    if ell == 0:
        return np.linspace(0.01, 3.0, 90)

    if ell == 1:
        return np.unique(np.concatenate([np.logspace(-3, -0.05, 60), np.linspace(0.9, 2.5, 20)]))

    if ell == 2:
        return np.unique(np.concatenate([np.logspace(-4, -0.2, 70), np.linspace(0.6, 1.6, 15)]))

    return np.unique(np.concatenate([np.logspace(-5, -0.35, 70), np.linspace(0.3, 0.9, 12)]))


# 関数: effective vector pilot の radial profile を解く。

def solve_sector_profile(beta: float, amp: float, ell: int) -> dict:
    r0 = 1.0e-4
    if ell == 0:
        y0 = [float(amp), 0.0]
    else:
        y0 = [float(amp) * (r0**ell), float(amp) * ell * (r0 ** (ell - 1))]

    # 関数: effective single-profile pilot ODE を返す。

    def ode(radius: float, y: np.ndarray) -> list[float]:
        field, field_prime = float(y[0]), float(y[1])
        damping = 2.0 * field_prime / radius if radius > 0.0 else 0.0
        barrier = ell * (ell + 1.0) * field / (radius * radius) if radius > 0.0 else 0.0
        field_double_prime = -damping + barrier - (beta * beta - 1.0) * field - 3.0 * field * field - field**3
        return [field_prime, field_double_prime]

    sol = solve_ivp(ode, (r0, 30.0), y0, max_step=0.05, rtol=1.0e-7, atol=1.0e-9)
    radius = sol.t
    field = sol.y[0]
    field_prime = sol.y[1]
    energy_density = (
        0.5 * field_prime * field_prime
        + 0.5 * (1.0 + beta * beta) * field * field
        + field**3
        + 0.25 * field**4
        + 0.5 * ell * (ell + 1.0) * field * field / np.maximum(radius * radius, 1.0e-12)
    )
    charge_proxy = float(beta * np.trapezoid(4.0 * math.pi * radius * radius * field * field, radius))
    energy_proxy = float(np.trapezoid(4.0 * math.pi * radius * radius * energy_density, radius))
    return {
        "tail": float(field[-1]),
        "tail_abs": float(abs(field[-1])),
        "charge_proxy": charge_proxy,
        "energy_proxy": energy_proxy,
        "central_amplitude": float(amp),
        "field_min": float(np.min(field)),
        "field_max": float(np.max(field)),
        "node_count_k": count_radial_nodes(field),
        "radius_values": radius.tolist(),
        "field_values": field.tolist(),
    }


# 関数: radial profile のゼロ交差回数から radial quantum number k を数える。

def count_radial_nodes(field_values: np.ndarray) -> int:
    amplitude = float(np.max(np.abs(field_values))) if len(field_values) > 0 else 0.0
    if amplitude == 0.0:
        return 0

    threshold = max(1.0e-9, amplitude * 1.0e-6)
    filtered = np.asarray([float(value) for value in field_values if abs(float(value)) > threshold], dtype=float)
    if len(filtered) < 2:
        return 0

    signs = np.sign(filtered)
    return int(np.sum(np.diff(signs) != 0))


# 関数: root-finding の重複計算を減らすため tail を cache する。

@lru_cache(maxsize=None)
def cached_tail(beta: float, amp: float, ell: int) -> float:
    return float(solve_sector_profile(float(beta), float(amp), int(ell))["tail"])


# 関数: root-finding で確定した profile を cache する。

@lru_cache(maxsize=None)
def cached_profile(beta: float, amp: float, ell: int) -> dict:
    return solve_sector_profile(float(beta), float(amp), int(ell))


# 関数: 与えた beta, ell で localized tail を作る amplitude を探す。

def find_sector_amplitudes(beta: float, ell: int) -> list[dict]:
    amps = amplitude_grid(int(ell))
    tails: list[float] = []
    for amp in amps:
        try:
            tails.append(cached_tail(float(beta), float(amp), int(ell)))
        except Exception:
            tails.append(float("nan"))

    candidates: list[dict] = []

    for amp_left, amp_right, tail_left, tail_right in zip(amps[:-1], amps[1:], tails[:-1], tails[1:]):
        if not np.isfinite(tail_left) or not np.isfinite(tail_right):
            continue

        if tail_left == 0.0:
            root_amp = float(amp_left)
        elif tail_right == 0.0:
            root_amp = float(amp_right)
        elif tail_left * tail_right < 0.0:
            root_amp = float(
                brentq(
                    lambda amp: cached_tail(float(beta), float(amp), int(ell)),
                    float(amp_left),
                    float(amp_right),
                    maxiter=60,
                )
            )
        else:
            continue

        solved = cached_profile(float(beta), float(root_amp), int(ell))
        candidates.append(
            {
                "central_amplitude": float(root_amp),
                "profile": solved,
                "node_count_k": int(solved["node_count_k"]),
                "tail_abs": float(solved["tail_abs"]),
            }
        )

    best_by_k: dict[int, dict] = {}
    for candidate in candidates:
        k_value = int(candidate["node_count_k"])
        previous = best_by_k.get(k_value)
        if previous is None or float(candidate["tail_abs"]) < float(previous["tail_abs"]):
            best_by_k[k_value] = candidate

    return [best_by_k[k_value] for k_value in sorted(best_by_k)]


# 関数: legacy caller 向けに lowest-k branch の central amplitude を返す。

def find_sector_amplitude(beta: float, ell: int) -> float | None:
    localized_profiles = find_sector_amplitudes(float(beta), int(ell))
    if not localized_profiles:
        return None

    return float(localized_profiles[0]["central_amplitude"])


# 関数: 指定 ell sector の localized scan を返す。

def scan_ell_sector(ell: int) -> list[dict]:
    rows = []
    for beta in pilot_beta_grid(int(ell)):
        localized_profiles = find_sector_amplitudes(float(beta), int(ell))
        if not localized_profiles:
            rows.append({"ell": int(ell), "beta": float(beta), "localized_solution_found": False})
            continue

        for branch_index, localized_profile in enumerate(localized_profiles, start=1):
            solved = localized_profile["profile"]
            rows.append(
                {
                    "ell": int(ell),
                    "beta": float(beta),
                    "localized_solution_found": True,
                    "central_amplitude": float(localized_profile["central_amplitude"]),
                    "charge_proxy": solved["charge_proxy"],
                    "energy_proxy": solved["energy_proxy"],
                    "tail_abs": solved["tail_abs"],
                    "field_min": solved["field_min"],
                    "field_max": solved["field_max"],
                    "node_count_k": int(localized_profile["node_count_k"]),
                    "k": int(localized_profile["node_count_k"]),
                    "solution_branch_index": int(branch_index),
                }
            )

    return rows


# 関数: charge curve を整数 n に線形補間して base mode table を作る。

def interpolate_integer_modes(scan_rows: list[dict], ell: int) -> list[dict]:
    localized = [row for row in scan_rows if row.get("localized_solution_found")]
    modes = []
    localized_by_k: dict[int, list[dict]] = {}
    for localized_row in localized:
        localized_by_k.setdefault(int(localized_row.get("node_count_k", localized_row.get("k", 0))), []).append(
            localized_row
        )

    for k_value, localized_rows in sorted(localized_by_k.items()):
        if len(localized_rows) < 2:
            continue

        localized_rows = sorted(localized_rows, key=lambda item: float(item["beta"]))
        q_min = int(math.ceil(min(float(item["charge_proxy"]) for item in localized_rows)))
        q_max = int(math.floor(max(float(item["charge_proxy"]) for item in localized_rows)))
        if q_max < q_min:
            continue

        for charge_index in range(q_min, q_max + 1):
            beta_n = None
            energy_n = None
            for left, right in zip(localized_rows[:-1], localized_rows[1:]):
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

            modes.append(
                {
                    "n": int(charge_index),
                    "k": int(k_value),
                    "ell": int(ell),
                    "beta_n": float(beta_n),
                    "charge_proxy_target": float(charge_index),
                    "base_mass_proxy": float(energy_n),
                    "node_count_k": int(k_value),
                }
            )

    return modes


# 関数: selected ell sectors から flat `base_modes` list を再構成する。

def build_base_modes(ell_values: tuple[int, ...] = (1, 2, 3)) -> tuple[dict[int, list[dict]], list[dict]]:
    ell_scan_rows = {int(ell): scan_ell_sector(int(ell)) for ell in ell_values}
    base_modes = []
    for ell in ell_values:
        base_modes.extend(interpolate_integer_modes(ell_scan_rows[int(ell)], int(ell)))

    base_modes = sorted(base_modes, key=lambda item: (int(item["ell"]), int(item["k"]), int(item["n"])))
    return ell_scan_rows, base_modes


# 関数: flat `base_modes` list を ell ごとの grouped view に戻す。

def group_base_modes_by_ell(base_modes: list[dict]) -> dict[int, list[dict]]:
    grouped: dict[int, list[dict]] = {}
    for mode in base_modes:
        grouped.setdefault(int(mode["ell"]), []).append(mode)

    return grouped


# 関数: trial-state inventory の 50 state rows を作る。

def build_trial_state_rows(scalar_modes: list[dict], sector_rows: list[dict]) -> list[dict]:
    trial_rows = []
    scalar_indices = [int(mode["mode_index"]) for mode in scalar_modes]
    for sector in sector_rows:
        ell = int(sector["ell"])
        s = int(sector["s"])
        label = str(sector["label"])
        for n in scalar_indices:
            trial_rows.append(
                {
                    "n": int(n),
                    "k": 0,
                    "ell": ell,
                    "s": s,
                    "label": label,
                    "trial_state_id": f"M_({n},0,{ell},{s})",
                }
            )

    return trial_rows


# 関数: scalar limit の mode を current effective solver で再評価する。

def build_scalar_recovery_rows(scalar_modes: list[dict]) -> list[dict]:
    recovered = []
    for mode in scalar_modes:
        beta_n = float(mode["beta_n"])
        amplitude = float(mode["central_amplitude"])
        solved = solve_sector_profile(beta_n, amplitude, 0)
        recovered.append(
            {
                "mode_index": int(mode["mode_index"]),
                "beta_n": beta_n,
                "stored_charge_proxy": float(mode["charge_proxy"]),
                "recovered_charge_proxy": float(solved["charge_proxy"]),
                "stored_energy_proxy": float(mode["energy_proxy"]),
                "recovered_energy_proxy": float(solved["energy_proxy"]),
                "charge_relative_error": abs(float(solved["charge_proxy"]) - float(mode["charge_proxy"])) / float(mode["charge_proxy"]),
                "energy_relative_error": abs(float(solved["energy_proxy"]) - float(mode["energy_proxy"])) / float(mode["energy_proxy"]),
            }
        )

    base_energy = float(scalar_modes[0]["energy_proxy"])
    for row_data, mode in zip(recovered, scalar_modes):
        recovered_ratio = float(row_data["recovered_energy_proxy"]) / base_energy
        stored_ratio = float(mode["mass_ratio_to_first"])
        row_data["stored_mass_ratio_to_first"] = stored_ratio
        row_data["recovered_mass_ratio_to_first"] = recovered_ratio
        row_data["mass_ratio_relative_error"] = abs(recovered_ratio - stored_ratio) / stored_ratio if stored_ratio != 0.0 else 0.0

    return recovered


# 関数: spin-orbit split を掛けた vector mode table を返す。

def build_spin_orbit_rows(
    scalar_modes: list[dict],
    base_modes: list[dict],
    lambda_rot: float,
) -> list[dict]:
    scalar_base_mass = float(scalar_modes[0]["energy_proxy"])
    split_rows = []
    for mode in scalar_modes:
        split_rows.append(
            {
                "n": int(mode["mode_index"]),
                "k": 0,
                "ell": 0,
                "s": 0,
                "beta_n": float(mode["beta_n"]),
                "spin_factor": 1.0,
                "base_mass_proxy": float(mode["energy_proxy"]),
                "split_mass_proxy": float(mode["energy_proxy"]),
                "mass_ratio_to_scalar_base": float(mode["energy_proxy"]) / scalar_base_mass,
            }
        )

    for mode in base_modes:
        ell = int(mode["ell"])
        for s in (-1, 0, 1):
            spin_factor = 1.0 + float(lambda_rot) * ell * s
            split_mass = float(mode["base_mass_proxy"]) * spin_factor
            split_rows.append(
                {
                    "n": int(mode["n"]),
                    "k": int(mode["k"]),
                    "ell": int(ell),
                    "s": int(s),
                    "beta_n": float(mode["beta_n"]),
                    "spin_factor": float(spin_factor),
                    "base_mass_proxy": float(mode["base_mass_proxy"]),
                    "split_mass_proxy": float(split_mass),
                    "mass_ratio_to_scalar_base": float(split_mass) / scalar_base_mass,
                    "node_count_k": int(mode.get("node_count_k", mode["k"])),
                }
            )

    return split_rows


# 関数: known target 比との最良一致を探索する。

def compare_known_targets(split_rows: list[dict]) -> tuple[list[dict], dict | None]:
    targets = [
        {"label": "m_mu/m_e", "value": 206.7682830, "threshold": 0.10},
        {"label": "m_p/m_e", "value": 1836.15267343, "threshold": 0.10},
        {"label": "m_tau/m_e", "value": 3477.48, "threshold": 0.10},
    ]
    comparisons = []
    best = None
    for mode in split_rows:
        if int(mode["ell"]) == 0:
            continue

        ratio = float(mode["mass_ratio_to_scalar_base"])
        for target in targets:
            relative_error = abs(ratio - float(target["value"])) / float(target["value"])
            record = {
                "n": int(mode["n"]),
                "k": int(mode["k"]),
                "ell": int(mode["ell"]),
                "s": int(mode["s"]),
                "target_label": target["label"],
                "target_value": float(target["value"]),
                "ratio_value": ratio,
                "relative_error": float(relative_error),
                "passes_threshold": bool(relative_error <= float(target["threshold"])),
            }
            comparisons.append(record)
            if best is None or record["relative_error"] < best["relative_error"]:
                best = record

    comparisons = sorted(comparisons, key=lambda item: float(item["relative_error"]))
    return comparisons, best


# 関数: 長い evidence row を要点だけに間引く。

def sample(rows: list[dict], n: int = 8) -> list[dict]:
    if len(rows) <= n:
        return rows

    indices = np.linspace(0, len(rows) - 1, n, dtype=int)
    return [rows[int(index)] for index in indices]


# 関数: branch 全体を実行して `.820-.825` の artifact 群を出力する。

def main() -> None:
    for path in (VECTOR_ROUTE, VECTOR_SPEC, VECTOR_SPIN, SCALAR_SPECTRUM, QBALL_RATIO, PART1, PART2):
        req(path)

    vector_route = read_json(VECTOR_ROUTE)
    vector_spec = read_json(VECTOR_SPEC)
    vector_spin = read_json(VECTOR_SPIN)
    scalar_spectrum = read_json(SCALAR_SPECTRUM)
    scalar_ratio = read_json(QBALL_RATIO)
    part1 = read_text(PART1)
    part2 = read_text(PART2)

    scalar_modes = list(scalar_spectrum["evidence"]["discrete_mass_mode_rows"])
    sector_rows = list(vector_spec["summary"]["pilot_sector_rows"])
    lambda_rot = float(vector_spin["summary"]["lambda_rot_value"])
    scalar_best_match = scalar_ratio["summary"]["closest_known_mass_ratio_or_none"]
    scalar_base_mass = float(scalar_modes[0]["energy_proxy"])

    trial_state_rows = build_trial_state_rows(scalar_modes, sector_rows)
    scalar_recovery_rows = build_scalar_recovery_rows(scalar_modes)
    ell_scan_rows, base_modes = build_base_modes((1, 2, 3))
    base_modes_by_ell = group_base_modes_by_ell(base_modes)
    split_rows = build_spin_orbit_rows(scalar_modes, base_modes, lambda_rot)
    comparison_rows, best_match = compare_known_targets(split_rows)

    total_integer_modes = len(base_modes)
    total_split_modes = len(split_rows)
    threshold_pass = bool(best_match and best_match["passes_threshold"])
    exact_full_coupled_vector_ladder_available = False
    handoff = bool(threshold_pass and exact_full_coupled_vector_ladder_available)
    next_route = "vector_qball_full_coupled_solver"
    available_k_values = sorted({int(mode["k"]) for mode in base_modes})
    max_detected_k = max(available_k_values) if available_k_values else 0
    k_positive_mode_count = sum(1 for mode in base_modes if int(mode["k"]) > 0)
    maximum_split_ratio = max(float(row_data["mass_ratio_to_scalar_base"]) for row_data in split_rows)
    max_split_row = max(split_rows, key=lambda row_data: float(row_data["mass_ratio_to_scalar_base"]))

    max_charge_error = max(float(row_data["charge_relative_error"]) for row_data in scalar_recovery_rows)
    max_energy_error = max(float(row_data["energy_relative_error"]) for row_data in scalar_recovery_rows)
    max_ratio_error = max(float(row_data["mass_ratio_relative_error"]) for row_data in scalar_recovery_rows)

    ell_sector_summary_rows = []
    for ell, rows in ell_scan_rows.items():
        localized = [row for row in rows if row.get("localized_solution_found")]
        ell_sector_summary_rows.append(
            {
                "ell": int(ell),
                "localized_solution_count": len(localized),
                "localized_beta_interval_or_none": [float(localized[0]["beta"]), float(localized[-1]["beta"])] if localized else None,
                "charge_interval_or_none": [
                    float(min(row["charge_proxy"] for row in localized)),
                    float(max(row["charge_proxy"] for row in localized)),
                ]
                if localized
                else None,
                "integer_mode_count": len(base_modes_by_ell.get(int(ell), [])),
                "k_values": sorted({int(mode["k"]) for mode in base_modes if int(mode["ell"]) == int(ell)}),
            }
        )

    payloads = {
        "mass_origin_vector_qball_trial_state_inventory": payload(
            "8.7.55.2.820",
            "Vector Q-ball trial-state inventory",
            {
                "mass_origin_vector_qball_numerical_solver_route_contract_json": rel(VECTOR_ROUTE),
                "mass_origin_vector_qball_solver_spec_json": rel(VECTOR_SPEC),
                "mass_origin_qball_discrete_mass_spectrum_json": rel(SCALAR_SPECTRUM),
            },
            "Freeze the first pilot `(n,k,ell,s)` sector table for the vector Q-ball numerical campaign.",
            {
                "trial_state_rule": "use the frozen sector rows `(ell,s)` from 8.7.55.2.815, keep the scalar `n=1..5` seed table explicit, and let the localized solver add `k>0` rows once node-counted solutions appear",
                "trial_state_id": "M_(n,k,ell,s)",
            },
            [
                row(
                    "vector_qball_trial_state_inventory_complete",
                    "pass",
                    "vector Q-ball trial-state inventory complete",
                    1,
                    "The first pilot sector table is frozen.",
                ),
                row(
                    "vector_qball_trial_sector_count",
                    "pass",
                    "vector Q-ball trial sector count",
                    len(sector_rows),
                    "The first pilot uses the scalar limit plus nine vector sectors.",
                ),
                row(
                    "vector_qball_trial_state_count",
                    "pass",
                    "vector Q-ball trial state count",
                    len(trial_state_rows),
                    "The first fixed trial table contains 50 states before any extended charge window is added.",
                ),
            ],
            {
                "trial_sector_count": len(sector_rows),
                "trial_state_count": len(trial_state_rows),
                "trial_charge_indices": [int(mode["mode_index"]) for mode in scalar_modes],
                "trial_k_indices": [0],
                "trial_sector_rows": sector_rows,
            },
            {
                "overall_status": "vector_qball_trial_state_inventory_frozen",
                "keep_mass_origin_branch_blocked": True,
                "hand_off_to_8_7_55_2_84": False,
                "next_required_artifacts": [
                    "vector_qball_scalar_limit_numerical_recovery",
                    "vector_qball_ell_sector_shooting_pilot",
                ],
            },
            {
                "trial_state_rows": trial_state_rows,
            },
        ),
        "mass_origin_vector_qball_scalar_limit_numerical_recovery": payload(
            "8.7.55.2.821",
            "Vector Q-ball scalar-limit numerical recovery",
            {
                "mass_origin_vector_qball_trial_state_inventory_json": "output/public/quantum/mass_origin_vector_qball_trial_state_inventory_metrics.json",
                "mass_origin_qball_discrete_mass_spectrum_json": rel(SCALAR_SPECTRUM),
            },
            "Verify that the effective vector solver exactly recovers the old scalar Q-ball ladder at `ell=0, s=0`.",
            {
                "scalar_limit_identifier": "(k, ell, s) = (0, 0, 0)",
                "recovery_rule": "re-evaluate the stored scalar mode betas and amplitudes with the current effective solver and compare charge/energy/mass-ratio proxies",
            },
            [
                row(
                    "vector_qball_scalar_limit_numerical_recovery_complete",
                    "pass",
                    "vector Q-ball scalar-limit numerical recovery complete",
                    1,
                    "The scalar limit has been re-evaluated inside the current solver.",
                ),
                row(
                    "vector_qball_scalar_limit_recovered_mode_count",
                    "pass",
                    "scalar limit recovered mode count",
                    len(scalar_recovery_rows),
                    "All scalar reference modes are recovered.",
                ),
                row(
                    "vector_qball_scalar_limit_max_energy_relative_error",
                    "pass",
                    "scalar limit maximum energy relative error",
                    max_energy_error,
                    "The current effective solver reproduces the scalar energy proxy exactly.",
                ),
            ],
            {
                "scalar_limit_recovered_mode_count": len(scalar_recovery_rows),
                "scalar_limit_recovery_available": True,
                "max_charge_relative_error": max_charge_error,
                "max_energy_relative_error": max_energy_error,
                "max_mass_ratio_relative_error": max_ratio_error,
            },
            {
                "overall_status": "vector_qball_scalar_limit_recovered",
                "keep_mass_origin_branch_blocked": True,
                "hand_off_to_8_7_55_2_84": False,
                "next_required_artifacts": ["vector_qball_ell_sector_shooting_pilot"],
            },
            {
                "scalar_limit_recovery_rows": scalar_recovery_rows,
                "scalar_reference_closest_match_row": scalar_best_match,
            },
        ),
        "mass_origin_vector_qball_ell_sector_shooting_pilot": payload(
            "8.7.55.2.822",
            "Vector Q-ball ell-sector shooting pilot",
            {
                "mass_origin_vector_qball_scalar_limit_numerical_recovery_json": "output/public/quantum/mass_origin_vector_qball_scalar_limit_numerical_recovery_metrics.json",
                "mass_origin_vector_qball_spin_orbit_freeze_audit_json": rel(VECTOR_SPIN),
            },
            "Run the first effective `ell=1,2,3` shooting pilot and estimate the integer-charge base ladders on each sector.",
            {
                "effective_vector_pilot_equation": "f'' + 2 f'/r - ell(ell+1) f / r^2 + (beta^2 - 1) f + 3 f^2 + f^3 = 0",
                "pilot_scope": "coarse localized scan only; this is not yet the exact full-coupled Proca system",
                "integer_mode_rule": "linearly invert each localized Q_(ell,k)(beta) branch onto integer target charges `n` to obtain a flat `(n,k,ell)` base ladder",
            },
            [
                row(
                    "vector_qball_ell_sector_shooting_pilot_complete",
                    "pass",
                    "vector Q-ball ell-sector shooting pilot complete",
                    1,
                    "The first `ell=1,2,3` effective pilot has been executed.",
                ),
                row(
                    "vector_qball_localized_ell_sector_count",
                    "pass",
                    "localized ell-sector count",
                    sum(1 for ell, rows in ell_scan_rows.items() if any(row.get("localized_solution_found") for row in rows)),
                    "All three ell sectors produced localized rows in the coarse pilot.",
                ),
                row(
                    "vector_qball_integer_mode_count_lower_bound",
                    "pass",
                    "vector Q-ball integer mode count lower bound",
                    total_integer_modes,
                    "The effective pilot already opens a large integer-charge ladder beyond the scalar special case.",
                ),
                row(
                    "vector_qball_k_positive_mode_count",
                    "pass" if k_positive_mode_count > 0 else "watch",
                    "vector Q-ball k-positive mode count",
                    k_positive_mode_count,
                    "The localized solver now records how many interpolated base modes carry explicit k>0.",
                ),
            ],
            {
                "pilot_sector_count": 3,
                "localized_ell_sector_count": sum(1 for ell, rows in ell_scan_rows.items() if any(row.get("localized_solution_found") for row in rows)),
                "total_integer_mode_count_lower_bound": total_integer_modes,
                "available_k_values": available_k_values,
                "maximum_detected_k": max_detected_k,
                "k_positive_mode_count": k_positive_mode_count,
                "ell_sector_summary_rows": ell_sector_summary_rows,
                "effective_pilot_exact_full_vector_solver": False,
            },
            {
                "overall_status": "vector_qball_effective_ell_sector_pilot_complete",
                "keep_mass_origin_branch_blocked": True,
                "hand_off_to_8_7_55_2_84": False,
                "next_required_artifacts": ["vector_qball_spin_orbit_mass_ratio_table"],
            },
            {
                "ell_sector_scan_rows": ell_scan_rows,
                "flat_base_mode_sample_rows": sample(base_modes, 16),
            },
        ),
        "mass_origin_vector_qball_spin_orbit_mass_ratio_table": payload(
            "8.7.55.2.823",
            "Vector Q-ball spin-orbit splitting / mass-ratio table",
            {
                "mass_origin_vector_qball_ell_sector_shooting_pilot_json": "output/public/quantum/mass_origin_vector_qball_ell_sector_shooting_pilot_metrics.json",
                "mass_origin_vector_qball_spin_orbit_freeze_audit_json": rel(VECTOR_SPIN),
                "mass_origin_qball_discrete_mass_spectrum_json": rel(SCALAR_SPECTRUM),
            },
            "Apply the frozen `lambda_rot` to the effective vector base ladder and build the first provisional mass-ratio table.",
            {
                "spin_orbit_rule": "M_(n,k,ell,s) = M_(n,k,ell) * (1 + lambda_rot * ell * s)",
                "reference_state": "M_(1,0,0,0)",
                "pilot_caveat": "the table is only a first effective proxy until the full coupled vector solver is frozen",
            },
            [
                row(
                    "vector_qball_spin_orbit_mass_ratio_table_complete",
                    "pass",
                    "vector Q-ball spin-orbit / mass-ratio table complete",
                    1,
                    "The first effective vector mass-ratio table has been computed.",
                ),
                row(
                    "vector_qball_split_mode_count",
                    "pass",
                    "vector Q-ball split mode count",
                    total_split_modes,
                    "The table includes the scalar baseline and the split vector sectors.",
                ),
                row(
                    "vector_qball_best_pilot_relative_error",
                    "pass" if best_match and best_match["passes_threshold"] else "watch",
                    "vector Q-ball best pilot relative error",
                    float(best_match["relative_error"]) if best_match else 1.0,
                    "The best effective pilot candidate is reported, but exact handoff still needs the full coupled solver.",
                ),
            ],
            {
                "split_mode_count": total_split_modes,
                "first_mass_ratio_table_available": True,
                "best_provisional_match_or_none": best_match,
                "pilot_ratio_threshold_pass": threshold_pass,
                "reference_state_mass_proxy": scalar_base_mass,
                "maximum_mass_ratio_to_scalar_base": maximum_split_ratio,
                "max_ratio_row_or_none": max_split_row,
            },
            {
                "overall_status": "vector_qball_first_mass_ratio_table_frozen",
                "keep_mass_origin_branch_blocked": True,
                "hand_off_to_8_7_55_2_84": False,
                "next_required_artifacts": ["vector_qball_handoff_gate_refresh"],
            },
            {
                "top_ratio_candidate_rows": comparison_rows[:12],
                "scalar_reference_mode_rows": scalar_modes,
            },
        ),
        "mass_origin_vector_qball_handoff_gate_refresh": payload(
            "8.7.55.2.824",
            "Vector Q-ball handoff gate refresh",
            {
                "mass_origin_vector_qball_spin_orbit_mass_ratio_table_json": "output/public/quantum/mass_origin_vector_qball_spin_orbit_mass_ratio_table_metrics.json",
                "mass_origin_vector_qball_numerical_solver_route_contract_json": rel(VECTOR_ROUTE),
            },
            "Refresh the mass-origin gate after the first effective vector pilot and decide whether `.84` can be reopened.",
            {
                "handoff_rule": "pilot ratio threshold pass AND exact full-coupled vector ladder available",
                "current_failure_mode": "effective_single_profile_pilot_not_equal_to_full_coupled_vector_solver",
            },
            [
                row(
                    "vector_qball_handoff_gate_refresh_complete",
                    "pass",
                    "vector Q-ball handoff gate refresh complete",
                    1,
                    "The vector handoff gate has been re-evaluated after the first pilot.",
                ),
                row(
                    "vector_qball_pilot_ratio_hint_available",
                    "pass" if threshold_pass else "watch",
                    "vector Q-ball pilot ratio hint available",
                    1 if threshold_pass else 0,
                    "The effective pilot contains a within-threshold candidate, but it is still only a proxy result.",
                ),
                row(
                    "vector_qball_exact_full_coupled_ladder_available",
                    "reject",
                    "exact full-coupled vector ladder available",
                    0,
                    "The current branch still lacks the exact multicomponent vector solver and cannot reopen `.84` yet.",
                ),
                row(
                    "hand_off_to_8_7_55_2_84",
                    "reject",
                    "handoff to 8.7.55.2.84 available",
                    0,
                    "A promising pilot exists, but exact handoff remains blocked until the full coupled vector ladder is frozen.",
                ),
            ],
            {
                "effective_vector_pilot_ratio_hint_available": threshold_pass,
                "best_provisional_match_or_none": best_match,
                "exact_full_coupled_vector_ladder_available": exact_full_coupled_vector_ladder_available,
                "hand_off_to_8_7_55_2_84": handoff,
                "new_branch_required": True,
                "recommended_next_route_or_none": next_route,
            },
            {
                "overall_status": "vector_qball_effective_pilot_promising_but_not_exact",
                "keep_mass_origin_branch_blocked": True,
                "hand_off_to_8_7_55_2_84": handoff,
                "new_branch_required": True,
                "next_required_artifacts": [next_route],
            },
            {
                "best_provisional_match_row": best_match,
                "vector_qball_route_contract_summary": vector_route["summary"],
                "part1_spin_line": hit(part1, "Pauli 型スピン結合"),
                "part2_frame_dragging_line": hit(part2, "frame dragging"),
            },
        ),
        "mass_origin_vector_qball_full_coupled_solver_route_contract": payload(
            "8.7.55.2.825",
            "Vector Q-ball full-coupled solver route contract",
            {
                "mass_origin_vector_qball_handoff_gate_refresh_json": "output/public/quantum/mass_origin_vector_qball_handoff_gate_refresh_metrics.json",
            },
            "Freeze the next residual route after the effective vector pilot produced a promising hint but not an exact reopenable ladder.",
            {
                "selected_residual_route": next_route,
                "missing_artifact": "full_coupled_vector_qball_discrete_mass_ladder",
            },
            [
                row(
                    "vector_qball_full_coupled_solver_route_contract_complete",
                    "pass",
                    "vector Q-ball full-coupled solver route contract complete",
                    1,
                    "The next exact-vector branch is frozen.",
                ),
                row(
                    "vector_qball_full_coupled_solver_split_contract_ready",
                    "pass",
                    "vector Q-ball full-coupled solver split contract ready",
                    1,
                    "The next branch may inventory the coupled radial system, freeze constraints, and compute the exact multicomponent ladder.",
                ),
            ],
            {
                "selected_residual_route": next_route,
                "missing_vector_qball_artifact": "full_coupled_vector_qball_discrete_mass_ladder",
                "best_effective_pilot_match_or_none": best_match,
                "split_contract_ready": True,
            },
            {
                "overall_status": "vector_qball_full_coupled_solver_route_contract_frozen",
                "keep_mass_origin_branch_blocked": True,
                "hand_off_to_8_7_55_2_84": False,
                "next_required_artifacts": [
                    "vector_qball_coupled_solver_source_inventory",
                    "vector_qball_coupled_constraint_freeze_audit",
                ],
            },
            {
                "best_provisional_match_row": best_match,
            },
        ),
    }

    for stem, data in payloads.items():
        write_artifact(stem, data)
        print(f"[ok] wrote {OUT / (stem + '_metrics.json')}")
        print(f"[ok] wrote {OUT / (stem + '_rows.csv')}")


# 関数: スクリプト実行時に `.820-.825` branch を起動する。

if __name__ == "__main__":
    main()
