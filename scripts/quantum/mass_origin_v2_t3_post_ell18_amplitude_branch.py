#!/usr/bin/env python3
"""
Generate Trial-3 refactored post-ell18 central-amplitude-window artifacts for
8.7.56.285-.288.

The widened beta-window audit already showed that post-ell18 localization does
not reopen just by sweeping beta. The next honest question is narrower: does
the current `find_sector_amplitudes()` search domain miss higher-ell localized
families because its central amplitude window stops too early for `ell >= 19`?
"""

from __future__ import annotations

import csv
import importlib.util
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from scipy.optimize import brentq


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output" / "public" / "quantum"
STATUS = ROOT / "doc" / "STATUS.md"
ROADMAP = ROOT / "doc" / "ROADMAP.md"
AI_CONTEXT = ROOT / "doc" / "AI_CONTEXT_MIN.json"

POST_PHOTON_PRESERVATION = OUT / "mass_origin_v2_post_photon_vector_mass_ratio_preservation_audit_metrics.json"
TRIAL3_REFACTORED_AUDIT = OUT / "mass_origin_v2_trial3_refactored_high_mass_k_positive_extension_audit_metrics.json"
TRIAL3_REFACTORED_GATE = OUT / "mass_origin_v2_trial3_refactored_declaration_gate_metrics.json"
TRIAL3_REFACTORED_DISPOSITION = OUT / "mass_origin_v2_trial3_refactored_paper_sync_trial4_disposition_thirteenth_refresh_metrics.json"
TRIAL3_WINDOW_AUDIT = OUT / "mass_origin_v2_trial3_refactored_post_ell18_localization_window_extension_audit_metrics.json"
TRIAL3_WINDOW_GATE = OUT / "mass_origin_v2_trial3_refactored_declaration_second_gate_metrics.json"
TRIAL3_WINDOW_DISPOSITION = OUT / "mass_origin_v2_trial3_refactored_paper_sync_trial4_disposition_fourteenth_refresh_metrics.json"
TRIAL3_SOLVER_REFACTOR_EXECUTION = OUT / "mass_origin_v2_trial3_solver_refactor_execution_audit_metrics.json"
TRIAL3_SOLVER_REFACTOR_WEAK = OUT / "mass_origin_v2_trial3_solver_refactor_weak_sector_reaudit_metrics.json"
VECTOR_SPIN = OUT / "mass_origin_vector_qball_spin_orbit_freeze_audit_metrics.json"
SCALAR_SPECTRUM = OUT / "mass_origin_qball_discrete_mass_spectrum_metrics.json"
EXACT_HANDOFF = OUT / "mass_origin_vector_qball_exact_mass_table_handoff_retry_metrics.json"

NUMERICAL_BRANCH = ROOT / "scripts" / "quantum" / "mass_origin_vector_qball_numerical_solver_branch.py"
FULL_COUPLED_BRANCH = ROOT / "scripts" / "quantum" / "mass_origin_vector_qball_full_coupled_solver_branch.py"
EXTENDED_BRANCH = ROOT / "scripts" / "quantum" / "mass_origin_vector_qball_extended_hierarchy_branch.py"
PREVIOUS_BRANCH = ROOT / "scripts" / "quantum" / "mass_origin_v2_t3_post_ell18_window_branch.py"

W_MASS_MEV = 80369.0
Z_MASS_MEV = 91187.6
ELECTRON_MASS_MEV = 0.51099895
W_TARGET = W_MASS_MEV / ELECTRON_MASS_MEV
Z_TARGET = Z_MASS_MEV / ELECTRON_MASS_MEV
WZ_RATIO_TARGET = W_MASS_MEV / Z_MASS_MEV
SIN2_THETA_W_TARGET = 1.0 - WZ_RATIO_TARGET * WZ_RATIO_TARGET
PASS_THRESHOLD = 0.10

LOW_ELL_VALUES = (1, 2, 3)
STANDARD_EXTENSION_ELLS = tuple(range(4, 19))
PRIMARY_POST_ELL18_VALUES = tuple(range(19, 25))
TAIL_POST_ELL18_VALUES = tuple(range(25, 31))
PRIMARY_WIDENED_BETA_GRID = (
    0.01,
    0.02,
    0.03,
    0.04,
    0.05,
    0.06,
    0.08,
    0.10,
    0.12,
    0.16,
    0.20,
    0.26,
    0.34,
    0.44,
    0.56,
    0.70,
    0.84,
    0.94,
    0.98,
)
TAIL_WIDENED_BETA_GRID = (
    0.01,
    0.03,
    0.05,
    0.08,
    0.12,
    0.18,
    0.26,
    0.36,
    0.48,
    0.62,
    0.78,
    0.92,
    0.98,
)
PRIMARY_EXTENDED_AMPLITUDES = (1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.5, 10.0)
TAIL_EXTENDED_AMPLITUDES = (1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0)


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


# 関数: local Python module を動的に読む。

def load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"[fail] unable to import module: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# 関数: metrics の output stem を repo 相対表記で返す。

def artifact_names(stem: str) -> dict:
    return {
        "json": rel(OUT / f"{stem}_metrics.json"),
        "csv": rel(OUT / f"{stem}_rows.csv"),
    }


# 関数: long row table から要約サンプルだけ残す。

def sample(rows: list[dict], count: int = 12) -> list[dict]:
    if len(rows) <= count:
        return rows

    step = max(1, len(rows) // count)
    sampled = [rows[index] for index in range(0, len(rows), step)]
    return sampled[:count]


# 関数: rebuilt vector rows を normalization factor で rescale する。

def normalize_vector_rows(rows: list[dict], scale_factor: float) -> list[dict]:
    normalized = []
    for row_data in rows:
        copied = dict(row_data)
        copied["mass_ratio_to_scalar_base"] = float(row_data["mass_ratio_to_scalar_base"]) * float(scale_factor)
        normalized.append(copied)

    return normalized


# 関数: sector scan と rebuilt mode table の要約を返す。

def sector_summary(scan_rows: list[dict], mode_rows: list[dict]) -> dict:
    localized = [row_data for row_data in scan_rows if row_data.get("localized_solution_found")]
    return {
        "localized_solution_count": len(localized),
        "integer_mode_count": len(mode_rows),
        "k_values": sorted({int(mode["k"]) for mode in mode_rows}),
        "max_charge_proxy_or_none": None if not localized else float(max(float(row_data["charge_proxy"]) for row_data in localized)),
    }


# 関数: W/Z target に最も近い single state を返す。

def closest_state(rows: list[dict], target_value: float) -> dict | None:
    best = None
    for item in rows:
        ratio = float(item["mass_ratio_to_scalar_base"])
        relative_error = abs(ratio - target_value) / target_value
        record = {
            "n": int(item["n"]),
            "k": int(item["k"]),
            "ell": int(item["ell"]),
            "s": int(item["s"]),
            "ratio_value": ratio,
            "relative_error": float(relative_error),
            "passes_threshold": bool(relative_error <= PASS_THRESHOLD),
        }
        if best is None or record["relative_error"] < best["relative_error"]:
            best = record

    return best


# 関数: high-mass tail から最良の W/Z candidate pair を探す。

def best_ratio_pair(rows: list[dict], top_count: int = 400) -> dict | None:
    candidates = sorted(rows, key=lambda item: float(item["mass_ratio_to_scalar_base"]), reverse=True)[:top_count]
    best = None
    for index, left in enumerate(candidates):
        left_ratio = float(left["mass_ratio_to_scalar_base"])
        for right in candidates[index + 1 :]:
            right_ratio = float(right["mass_ratio_to_scalar_base"])
            heavier = max(left_ratio, right_ratio)
            lighter = min(left_ratio, right_ratio)
            ratio_value = lighter / heavier
            ratio_error = abs(ratio_value - WZ_RATIO_TARGET) / WZ_RATIO_TARGET
            sin2_value = 1.0 - ratio_value * ratio_value
            sin2_error = abs(sin2_value - SIN2_THETA_W_TARGET) / SIN2_THETA_W_TARGET
            record = {
                "lighter_state": {
                    "n": int(left["n"]) if left_ratio <= right_ratio else int(right["n"]),
                    "k": int(left["k"]) if left_ratio <= right_ratio else int(right["k"]),
                    "ell": int(left["ell"]) if left_ratio <= right_ratio else int(right["ell"]),
                    "s": int(left["s"]) if left_ratio <= right_ratio else int(right["s"]),
                    "mass_ratio_to_electron": float(lighter),
                },
                "heavier_state": {
                    "n": int(left["n"]) if left_ratio > right_ratio else int(right["n"]),
                    "k": int(left["k"]) if left_ratio > right_ratio else int(right["k"]),
                    "ell": int(left["ell"]) if left_ratio > right_ratio else int(right["ell"]),
                    "s": int(left["s"]) if left_ratio > right_ratio else int(right["s"]),
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


# 関数: solver の既定 amplitude grid に high-amplitude tranche を追加する。

def extended_amplitude_grid(numerical, ell: int, extra_amplitudes: tuple[float, ...]) -> list[float]:
    base = [float(value) for value in numerical.amplitude_grid(int(ell))]
    extras = [float(value) for value in extra_amplitudes]
    return sorted(set(base + extras))


# 関数: extended amplitude domain で localized sector scan を実行する。

def scan_amplitude_extended_sector(
    numerical,
    ell: int,
    beta_grid: tuple[float, ...],
    extra_amplitudes: tuple[float, ...],
) -> tuple[list[dict], list[float]]:
    rows = []
    amplitude_values = extended_amplitude_grid(numerical, int(ell), extra_amplitudes)
    for beta in beta_grid:
        tails = []
        for amp in amplitude_values:
            try:
                tails.append(numerical.cached_tail(float(beta), float(amp), int(ell)))
            except Exception:
                tails.append(float("nan"))

        candidates: list[dict] = []
        for amp_left, amp_right, tail_left, tail_right in zip(
            amplitude_values[:-1],
            amplitude_values[1:],
            tails[:-1],
            tails[1:],
        ):
            if not math.isfinite(tail_left) or not math.isfinite(tail_right):
                continue

            if tail_left == 0.0:
                root_amp = float(amp_left)
            elif tail_right == 0.0:
                root_amp = float(amp_right)
            elif tail_left * tail_right < 0.0:
                root_amp = float(
                    brentq(
                        lambda amp: numerical.cached_tail(float(beta), float(amp), int(ell)),
                        float(amp_left),
                        float(amp_right),
                        maxiter=80,
                    )
                )
            else:
                continue

            solved = numerical.cached_profile(float(beta), float(root_amp), int(ell))
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

        localized_profiles = [best_by_k[k_value] for k_value in sorted(best_by_k)]
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
                    "charge_proxy": float(solved["charge_proxy"]),
                    "energy_proxy": float(solved["energy_proxy"]),
                    "tail_abs": float(solved["tail_abs"]),
                    "field_min": float(solved["field_min"]),
                    "field_max": float(solved["field_max"]),
                    "node_count_k": int(localized_profile["node_count_k"]),
                    "k": int(localized_profile["node_count_k"]),
                    "solution_branch_index": int(branch_index),
                }
            )

    return rows, amplitude_values


# 関数: Trial-3 refactored post-ell18 central-amplitude-window branch を実行する。

def main() -> None:
    for path in (
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        POST_PHOTON_PRESERVATION,
        TRIAL3_REFACTORED_AUDIT,
        TRIAL3_REFACTORED_GATE,
        TRIAL3_REFACTORED_DISPOSITION,
        TRIAL3_WINDOW_AUDIT,
        TRIAL3_WINDOW_GATE,
        TRIAL3_WINDOW_DISPOSITION,
        TRIAL3_SOLVER_REFACTOR_EXECUTION,
        TRIAL3_SOLVER_REFACTOR_WEAK,
        VECTOR_SPIN,
        SCALAR_SPECTRUM,
        EXACT_HANDOFF,
        NUMERICAL_BRANCH,
        FULL_COUPLED_BRANCH,
        EXTENDED_BRANCH,
        PREVIOUS_BRANCH,
    ):
        req(path)

    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    ai_context = read_json(AI_CONTEXT)
    post_photon = read_json(POST_PHOTON_PRESERVATION)
    prior_audit = read_json(TRIAL3_REFACTORED_AUDIT)
    window_audit = read_json(TRIAL3_WINDOW_AUDIT)
    window_gate = read_json(TRIAL3_WINDOW_GATE)
    window_disposition = read_json(TRIAL3_WINDOW_DISPOSITION)
    solver_refactor_execution = read_json(TRIAL3_SOLVER_REFACTOR_EXECUTION)
    solver_refactor_weak = read_json(TRIAL3_SOLVER_REFACTOR_WEAK)
    vector_spin = read_json(VECTOR_SPIN)
    scalar_spectrum = read_json(SCALAR_SPECTRUM)
    exact_handoff = read_json(EXACT_HANDOFF)

    numerical_text = read_text(NUMERICAL_BRANCH)
    full_text = read_text(FULL_COUPLED_BRANCH)
    previous_branch_text = read_text(PREVIOUS_BRANCH)

    numerical = load_module(NUMERICAL_BRANCH, "trial3_post_ell18_amplitude_num")
    full = load_module(FULL_COUPLED_BRANCH, "trial3_post_ell18_amplitude_full")
    extended = load_module(EXTENDED_BRANCH, "trial3_post_ell18_amplitude_ext")

    scalar_modes = list(scalar_spectrum["evidence"]["discrete_mass_mode_rows"])
    lambda_rot = float(vector_spin["summary"]["lambda_rot_value"])
    normalization_scale = float(post_photon["summary"]["absolute_mass_normalization_scale_factor"])
    preserved_ceiling = float(window_audit["summary"]["historic_preserved_verified_ceiling_to_electron"])
    prior_w_gap_factor = float(window_audit["summary"]["w_gap_factor_or_none"])
    prior_z_gap_factor = float(window_audit["summary"]["z_gap_factor_or_none"])

    inventory_targets = [
        {
            "label": "status_points_to_8_7_56_285",
            "present": "current official next step は `8.7.56.285`" in status_text,
            "evidence": {"status_markdown": rel(STATUS)},
        },
        {
            "label": "roadmap_central_amplitude_branch_present",
            "present": "`8.7.56.285-.288` 試練3 refactored post-`ell=18` central-amplitude-window extension residual branch" in roadmap_text,
            "evidence": {"roadmap_markdown": rel(ROADMAP)},
        },
        {
            "label": "prior_beta_window_no_gain_fixed",
            "present": int(window_audit["summary"]["post_ell18_localized_solution_count_total"]) == 0,
            "evidence": {
                "prior_post_ell18_localized_solution_count_total": window_audit["summary"]["post_ell18_localized_solution_count_total"],
                "prior_post_ell18_integer_mode_count_total": window_audit["summary"]["post_ell18_integer_mode_count_total"],
            },
        },
        {
            "label": "numerical_amplitude_grid_present",
            "present": hit(numerical_text, "def amplitude_grid(ell: int) -> np.ndarray:") is not None,
            "evidence": hit(numerical_text, "def amplitude_grid(ell: int) -> np.ndarray:"),
        },
        {
            "label": "numerical_high_ell_amplitude_cap_present",
            "present": hit(
                numerical_text,
                "return np.unique(np.concatenate([np.logspace(-5, -0.35, 70), np.linspace(0.3, 0.9, 12)]))",
            )
            is not None,
            "evidence": hit(
                numerical_text,
                "return np.unique(np.concatenate([np.logspace(-5, -0.35, 70), np.linspace(0.3, 0.9, 12)]))",
            ),
        },
        {
            "label": "numerical_find_sector_amplitudes_present",
            "present": hit(numerical_text, "def find_sector_amplitudes(beta: float, ell: int) -> list[dict]:") is not None,
            "evidence": hit(numerical_text, "def find_sector_amplitudes(beta: float, ell: int) -> list[dict]:"),
        },
        {
            "label": "prior_window_branch_present",
            "present": hit(previous_branch_text, "PRIMARY_WIDENED_BETA_GRID = (") is not None,
            "evidence": hit(previous_branch_text, "PRIMARY_WIDENED_BETA_GRID = ("),
        },
        {
            "label": "full_build_exact_ladder_present",
            "present": hit(full_text, "def build_exact_ladder(") is not None,
            "evidence": hit(full_text, "def build_exact_ladder("),
        },
        {
            "label": "same_family_weak_target_pack_present",
            "present": bool(exact_handoff["summary"]["hand_off_to_8_7_55_2_84"]),
            "evidence": {
                "best_exact_match_or_none": exact_handoff["summary"]["best_exact_match_or_none"],
                "w_target_to_electron": W_TARGET,
                "z_target_to_electron": Z_TARGET,
            },
        },
    ]

    source_inventory = payload(
        "8.7.56.285",
        "Trial-3 refactored post-ell18 central-amplitude-window source inventory",
        {
            "status_markdown": rel(STATUS),
            "roadmap_markdown": rel(ROADMAP),
            "ai_context_json": rel(AI_CONTEXT),
            "mass_origin_v2_post_photon_vector_mass_ratio_preservation_audit_json": rel(POST_PHOTON_PRESERVATION),
            "mass_origin_v2_trial3_refactored_high_mass_k_positive_extension_audit_json": rel(TRIAL3_REFACTORED_AUDIT),
            "mass_origin_v2_trial3_refactored_post_ell18_localization_window_extension_audit_json": rel(TRIAL3_WINDOW_AUDIT),
            "mass_origin_v2_trial3_refactored_declaration_second_gate_json": rel(TRIAL3_WINDOW_GATE),
            "mass_origin_v2_trial3_refactored_paper_sync_trial4_disposition_fourteenth_refresh_json": rel(TRIAL3_WINDOW_DISPOSITION),
            "mass_origin_v2_trial3_solver_refactor_execution_audit_json": rel(TRIAL3_SOLVER_REFACTOR_EXECUTION),
            "mass_origin_v2_trial3_solver_refactor_weak_sector_reaudit_json": rel(TRIAL3_SOLVER_REFACTOR_WEAK),
            "mass_origin_vector_qball_spin_orbit_freeze_audit_json": rel(VECTOR_SPIN),
            "mass_origin_qball_discrete_mass_spectrum_json": rel(SCALAR_SPECTRUM),
            "mass_origin_vector_qball_exact_mass_table_handoff_retry_json": rel(EXACT_HANDOFF),
            "mass_origin_vector_qball_numerical_solver_branch_py": rel(NUMERICAL_BRANCH),
            "mass_origin_vector_qball_full_coupled_solver_branch_py": rel(FULL_COUPLED_BRANCH),
            "mass_origin_vector_qball_extended_hierarchy_branch_py": rel(EXTENDED_BRANCH),
            "mass_origin_v2_t3_post_ell18_window_branch_py": rel(PREVIOUS_BRANCH),
        },
        "Freeze the current amplitude-grid contract and the post-ell18 no-gain evidence before reopening the same-family weak-sector route with an explicit high-amplitude extension.",
        {
            "inventory_rule": "the source pack must contain the current amplitude-grid contract, the widened beta-window no-gain evidence, the preserved exact-family rebuild, and the same-family weak targets before the amplitude audit can run honestly",
            "default_high_ell_amplitude_rule": "for ell>=3 the current solver amplitude grid stops at 0.9, so the central-amplitude-window audit adds an explicit high-amplitude tranche instead of modifying beta or exact-family normalization",
            "primary_amplitude_extension_rule": f"for ell in {list(PRIMARY_POST_ELL18_VALUES)} keep the current amplitude grid and append high-amplitude support {list(PRIMARY_EXTENDED_AMPLITUDES)}",
            "tail_amplitude_extension_rule": f"for ell in {list(TAIL_POST_ELL18_VALUES)} keep the current amplitude grid and append high-amplitude support {list(TAIL_EXTENDED_AMPLITUDES)}",
        },
        [
            row("trial3_refactored_post_ell18_central_amplitude_source_inventory_complete", "pass", "Trial-3 refactored post-ell18 central-amplitude-window source inventory complete", 1, "The central-amplitude-window source pack is frozen."),
            row("trial3_refactored_post_ell18_central_amplitude_required_source_count", "pass" if all(item["present"] for item in inventory_targets) else "reject", "required central-amplitude-window source count", len(inventory_targets), "The amplitude-window audit needs the current amplitude-grid contract, the no-gain beta-window evidence, the preserved exact-family rebuild, and the same-family W/Z targets in one source pack."),
            row("trial3_refactored_post_ell18_default_high_ell_amplitude_cap", "pass", "default high-ell amplitude cap", 0.9, "The current solver fixes the default high-ell central amplitude cap at 0.9 before the explicit amplitude extension is added."),
            row("trial3_refactored_post_ell18_historic_preserved_ceiling_to_electron", "pass", "historic preserved normalized ceiling to electron", preserved_ceiling, "The amplitude-window audit starts from the already-fixed preserved high-ell ceiling."),
        ],
        {
            "required_source_count": len(inventory_targets),
            "required_source_count_present": sum(1 for item in inventory_targets if item["present"]),
            "inventory_ready": bool(all(item["present"] for item in inventory_targets)),
            "historic_preserved_verified_ceiling_to_electron": preserved_ceiling,
            "prior_w_gap_factor": prior_w_gap_factor,
            "prior_z_gap_factor": prior_z_gap_factor,
            "primary_post_ell18_values": list(PRIMARY_POST_ELL18_VALUES),
            "tail_post_ell18_values": list(TAIL_POST_ELL18_VALUES),
            "primary_widened_beta_grid": list(PRIMARY_WIDENED_BETA_GRID),
            "tail_widened_beta_grid": list(TAIL_WIDENED_BETA_GRID),
            "primary_extended_amplitudes": list(PRIMARY_EXTENDED_AMPLITUDES),
            "tail_extended_amplitudes": list(TAIL_EXTENDED_AMPLITUDES),
            "first_route_to_close_or_none": "trial3_refactored_post_ell18_central_amplitude_window_audit",
        },
        {
            "overall_status": "trial3_refactored_post_ell18_central_amplitude_source_inventory_frozen",
            "trial3_branch_closeable": False,
            "advance_to_8_7_56_286": True,
            "next_required_artifacts": ["trial3_refactored_post_ell18_central_amplitude_window_audit"],
        },
        {
            "inventory_targets": inventory_targets,
            "prior_window_audit_summary": window_audit["summary"],
            "prior_window_gate_summary": window_gate["summary"],
            "prior_window_disposition_summary": window_disposition["summary"],
            "solver_refactor_execution_summary": solver_refactor_execution["summary"],
            "solver_refactor_weak_summary": solver_refactor_weak["summary"],
            "prior_refactored_audit_summary": prior_audit["summary"],
            "status_current_step_before_branch": ai_context["current_step"],
        },
    )

    ell_scan_rows, low_ell_base_modes = numerical.build_base_modes(LOW_ELL_VALUES)
    base_modes = list(low_ell_base_modes)

    standard_scan_rows = {}
    standard_mode_rows = {}
    for ell in STANDARD_EXTENSION_ELLS:
        scan_rows = extended.scan_extended_sector(numerical, int(ell))
        mode_rows = numerical.interpolate_integer_modes(scan_rows, int(ell))
        standard_scan_rows[int(ell)] = scan_rows
        standard_mode_rows[int(ell)] = mode_rows
        base_modes.extend(mode_rows)

    primary_scan_rows = {}
    primary_mode_rows = {}
    primary_amplitude_grid_map = {}
    for ell in PRIMARY_POST_ELL18_VALUES:
        scan_rows, amplitude_values = scan_amplitude_extended_sector(
            numerical,
            int(ell),
            PRIMARY_WIDENED_BETA_GRID,
            PRIMARY_EXTENDED_AMPLITUDES,
        )
        mode_rows = numerical.interpolate_integer_modes(scan_rows, int(ell))
        primary_scan_rows[int(ell)] = scan_rows
        primary_mode_rows[int(ell)] = mode_rows
        primary_amplitude_grid_map[int(ell)] = amplitude_values
        base_modes.extend(mode_rows)

    tail_scan_rows = {}
    tail_mode_rows = {}
    tail_amplitude_grid_map = {}
    for ell in TAIL_POST_ELL18_VALUES:
        scan_rows, amplitude_values = scan_amplitude_extended_sector(
            numerical,
            int(ell),
            TAIL_WIDENED_BETA_GRID,
            TAIL_EXTENDED_AMPLITUDES,
        )
        mode_rows = numerical.interpolate_integer_modes(scan_rows, int(ell))
        tail_scan_rows[int(ell)] = scan_rows
        tail_mode_rows[int(ell)] = mode_rows
        tail_amplitude_grid_map[int(ell)] = amplitude_values
        base_modes.extend(mode_rows)

    base_modes = sorted(base_modes, key=lambda item: (int(item["ell"]), int(item["k"]), int(item["n"])))
    exact_rows = full.build_exact_ladder(scalar_modes, base_modes, lambda_rot)
    normalized_vector_rows = normalize_vector_rows([row_data for row_data in exact_rows if int(row_data["ell"]) > 0], normalization_scale)

    best_w = closest_state(normalized_vector_rows, W_TARGET)
    best_z = closest_state(normalized_vector_rows, Z_TARGET)
    best_pair = best_ratio_pair(normalized_vector_rows)
    max_row = max(normalized_vector_rows, key=lambda row_data: float(row_data["mass_ratio_to_scalar_base"]))
    k_positive_rows = [row_data for row_data in normalized_vector_rows if int(row_data["k"]) > 0]
    max_k_positive_row = max(k_positive_rows, key=lambda row_data: float(row_data["mass_ratio_to_scalar_base"]))

    ell_mode_summary = {}
    ell_mode_summary.update(
        {
            str(ell): sector_summary(
                ell_scan_rows[int(ell)],
                [mode for mode in low_ell_base_modes if int(mode["ell"]) == int(ell)],
            )
            for ell in LOW_ELL_VALUES
        }
    )
    ell_mode_summary.update({str(ell): sector_summary(standard_scan_rows[int(ell)], standard_mode_rows[int(ell)]) for ell in STANDARD_EXTENSION_ELLS})
    ell_mode_summary.update({str(ell): sector_summary(primary_scan_rows[int(ell)], primary_mode_rows[int(ell)]) for ell in PRIMARY_POST_ELL18_VALUES})
    ell_mode_summary.update({str(ell): sector_summary(tail_scan_rows[int(ell)], tail_mode_rows[int(ell)]) for ell in TAIL_POST_ELL18_VALUES})

    localized_ell_values = sorted(
        {
            int(row_data["ell"])
            for scan_rows in list(primary_scan_rows.values()) + list(tail_scan_rows.values())
            for row_data in scan_rows
            if row_data.get("localized_solution_found")
        }
    )
    first_localized_ell = None if not localized_ell_values else int(localized_ell_values[0])
    available_k_values = sorted({int(mode["k"]) for mode in base_modes})
    maximum_detected_k = max(available_k_values) if available_k_values else 0
    maximum_detected_ell = max(int(mode["ell"]) for mode in base_modes)
    maximum_detected_ell_with_k_positive = max(int(mode["ell"]) for mode in base_modes if int(mode["k"]) > 0)
    primary_localized_solution_count_total = sum(
        len([row_data for row_data in scan_rows if row_data.get("localized_solution_found")])
        for scan_rows in primary_scan_rows.values()
    )
    tail_localized_solution_count_total = sum(
        len([row_data for row_data in scan_rows if row_data.get("localized_solution_found")])
        for scan_rows in tail_scan_rows.values()
    )
    post_ell18_localized_solution_count_total = primary_localized_solution_count_total + tail_localized_solution_count_total
    post_ell18_integer_mode_count_total = sum(len(mode_rows) for mode_rows in primary_mode_rows.values()) + sum(
        len(mode_rows) for mode_rows in tail_mode_rows.values()
    )
    rebuilt_max_ratio = float(max_row["mass_ratio_to_scalar_base"])
    ceiling_improved_beyond_preserved = rebuilt_max_ratio > preserved_ceiling
    w_anchor_pass = bool(best_w and best_w["passes_threshold"])
    z_anchor_pass = bool(best_z and best_z["passes_threshold"])
    mw_mz_ratio_pass = bool(best_pair and best_pair["mw_mz_ratio_relative_error"] <= PASS_THRESHOLD)
    sin2_theta_w_pass = bool(best_pair and best_pair["sin2_theta_w_relative_error"] <= PASS_THRESHOLD)
    trial3_recommended_condition_satisfied = bool(w_anchor_pass and z_anchor_pass and mw_mz_ratio_pass and sin2_theta_w_pass)

    audit = payload(
        "8.7.56.286",
        "Trial-3 refactored post-ell18 central-amplitude-window audit",
        source_inventory["inputs"],
        "Rebuild the same-family exact table after explicitly extending the post-ell18 central amplitude domain and freeze whether higher-ell localized families reopen.",
        {
            "amplitude_extension_rule": "keep the refactored low-ell and standard high-ell families fixed, then extend the post-ell18 central amplitude domain above the default 0.9 ceiling while preserving the widened beta grids",
            "localization_rule": "if the amplitude extension creates localized sectors above ell=18, the blocker is no longer the central amplitude domain itself",
            "residual_rule": "if localization reopens only at ell=19 and still fails to close W/Z, the next blocker must move away from the amplitude window and into the deeper post-ell19 search domain",
        },
        [
            row("trial3_refactored_post_ell18_central_amplitude_window_audit_complete", "pass", "Trial-3 refactored post-ell18 central-amplitude-window audit complete", 1, "The post-ell18 central-amplitude-window audit is frozen."),
            row("trial3_refactored_post_ell18_localized_solution_count", "pass" if post_ell18_localized_solution_count_total > 0 else "reject", "localized solution count above ell=18 under amplitude extension", post_ell18_localized_solution_count_total, "The explicit high-amplitude tranche must create localized sectors above ell=18 before the same-family weak-sector route can move honestly."),
            row("trial3_refactored_post_ell18_integer_mode_count", "pass" if post_ell18_integer_mode_count_total > 0 else "reject", "integer mode count above ell=18 under amplitude extension", post_ell18_integer_mode_count_total, "Localized sectors must also interpolate to integer modes before they can change the exact-family table."),
            row("trial3_refactored_post_ell18_ceiling_improved", "pass" if ceiling_improved_beyond_preserved else "reject", "rebuilt ceiling improves beyond preserved ell=18 ceiling", 1 if ceiling_improved_beyond_preserved else 0, "A successful amplitude extension should push the same-family ceiling beyond the preserved ell=18 row."),
            row("trial3_refactored_post_ell18_w_anchor_pass", "pass" if w_anchor_pass else "reject", "W/electron anchor passes after amplitude extension", 1 if w_anchor_pass else 0, "The amplitude extension closes Trial-3 only if it reaches the W scale."),
            row("trial3_refactored_post_ell18_z_anchor_pass", "pass" if z_anchor_pass else "reject", "Z/electron anchor passes after amplitude extension", 1 if z_anchor_pass else 0, "The amplitude extension must also reach the Z scale."),
            row("trial3_refactored_post_ell18_mw_mz_ratio_pass", "pass" if mw_mz_ratio_pass else "reject", "M_W/M_Z ratio passes after amplitude extension", 1 if mw_mz_ratio_pass else 0, "The same-family W/Z pair must remain viable after the amplitude extension."),
            row("trial3_refactored_post_ell18_sin2_theta_w_pass", "pass" if sin2_theta_w_pass else "reject", "sin^2(theta_W) passes after amplitude extension", 1 if sin2_theta_w_pass else 0, "The Weinberg-angle proxy must close together with the W/Z pair rather than remaining only partially improved."),
        ],
        {
            "normalization_scale_factor": normalization_scale,
            "historic_preserved_verified_ceiling_to_electron": preserved_ceiling,
            "rebuilt_verified_ceiling_to_electron": rebuilt_max_ratio,
            "refactored_k_positive_ceiling_to_electron": float(max_k_positive_row["mass_ratio_to_scalar_base"]),
            "post_ell18_primary_localized_solution_count_total": primary_localized_solution_count_total,
            "post_ell18_tail_localized_solution_count_total": tail_localized_solution_count_total,
            "post_ell18_localized_solution_count_total": post_ell18_localized_solution_count_total,
            "post_ell18_integer_mode_count_total": post_ell18_integer_mode_count_total,
            "localized_ell_values": localized_ell_values,
            "first_localized_ell_or_none": first_localized_ell,
            "available_k_values": available_k_values,
            "maximum_detected_k": maximum_detected_k,
            "maximum_detected_ell": maximum_detected_ell,
            "maximum_detected_ell_with_k_positive": maximum_detected_ell_with_k_positive,
            "best_w_row_or_none": best_w,
            "best_z_row_or_none": best_z,
            "best_pair_or_none": best_pair,
            "w_gap_factor_or_none": None if best_w is None else W_TARGET / float(best_w["ratio_value"]),
            "z_gap_factor_or_none": None if best_z is None else Z_TARGET / float(best_z["ratio_value"]),
            "trial3_recommended_condition_satisfied": trial3_recommended_condition_satisfied,
            "first_route_to_close_or_none": "trial3_refactored_declaration_third_gate",
        },
        {
            "overall_status": "trial3_refactored_post_ell18_central_amplitude_window_audited",
            "trial3_branch_closeable": trial3_recommended_condition_satisfied,
            "advance_to_8_7_56_287": True,
            "next_required_artifacts": ["trial3_refactored_declaration_third_gate"],
        },
        {
            "low_ell_sector_summary": {str(ell): ell_mode_summary[str(ell)] for ell in LOW_ELL_VALUES},
            "standard_extension_sector_summary": {str(ell): ell_mode_summary[str(ell)] for ell in STANDARD_EXTENSION_ELLS},
            "primary_post_ell18_sector_summary": {str(ell): ell_mode_summary[str(ell)] for ell in PRIMARY_POST_ELL18_VALUES},
            "tail_post_ell18_sector_summary": {str(ell): ell_mode_summary[str(ell)] for ell in TAIL_POST_ELL18_VALUES},
            "primary_amplitude_grid_map": {str(ell): primary_amplitude_grid_map[int(ell)] for ell in PRIMARY_POST_ELL18_VALUES},
            "tail_amplitude_grid_map": {str(ell): tail_amplitude_grid_map[int(ell)] for ell in TAIL_POST_ELL18_VALUES},
            "sampled_high_mass_rows": sample(sorted(normalized_vector_rows, key=lambda item: float(item["mass_ratio_to_scalar_base"]), reverse=True), 16),
            "max_row_or_none": max_row,
            "max_k_positive_row_or_none": max_k_positive_row,
            "prior_window_audit_summary": window_audit["summary"],
        },
    )

    if post_ell18_localized_solution_count_total == 0:
        selected_residual_route = "trial3_relaunched_refactored_post_ell18_radial_domain_extension_identification"
        missing_v2_artifact = "trial3_relaunched_refactored_localized_exact_family_table_above_ell18_with_extended_radial_domain"
    elif localized_ell_values and max(localized_ell_values) == 19 and not trial3_recommended_condition_satisfied:
        selected_residual_route = "trial3_relaunched_refactored_post_ell19_radial_domain_extension_identification"
        missing_v2_artifact = "trial3_relaunched_refactored_localized_exact_family_table_above_ell19_with_extended_radial_domain"
    elif not trial3_recommended_condition_satisfied:
        selected_residual_route = "trial3_relaunched_refactored_post_ell19_same_family_reaudit"
        missing_v2_artifact = "trial3_relaunched_refactored_post_ell19_same_family_closeout_pack"
    else:
        selected_residual_route = None
        missing_v2_artifact = None

    declaration = payload(
        "8.7.56.287",
        "Trial-3 refactored declaration third gate",
        source_inventory["inputs"],
        "Freeze whether the explicit post-ell18 central-amplitude extension is sufficient to close Trial-3 or whether the next residual route must move deeper into the post-ell19 solver domain.",
        {
            "closeout_rule": "Trial-3 closes only if the explicit post-ell18 amplitude extension carries the same-family exact table through W/Z and the Weinberg-angle proxy",
            "residual_rule": "if the amplitude extension reopens ell=19 but still fails to close W/Z and leaves ell>=20 dark, the next residual route must move beyond amplitude choice and isolate the deeper post-ell19 search domain",
        },
        [
            row("trial3_refactored_declaration_third_gate_complete", "pass", "Trial-3 refactored declaration third gate complete", 1, "The third refactored declaration gate is frozen."),
            row("trial3_refactored_third_branch_closeable", "pass" if trial3_recommended_condition_satisfied else "reject", "refactored Trial-3 branch closeable after amplitude extension", 1 if trial3_recommended_condition_satisfied else 0, "The branch closes only if the amplitude extension actually closes the weak-sector path."),
            row("trial3_refactored_third_residual_route_required", "reject" if trial3_recommended_condition_satisfied else "pass", "refactored Trial-3 residual route required after amplitude extension", 0 if trial3_recommended_condition_satisfied else 1, "A new residual route is still required when the amplitude extension improves the ceiling but cannot yet close W/Z honestly."),
            row("trial3_refactored_execute_trial2_paper_sync_now_third_gate", "reject", "execute Trial-2 paper-side sync now after amplitude extension", 0, "Trial-2 paper-side sync remains reserve work while the refactored Trial-3 scientific route stays open."),
        ],
        {
            "trial3_recommended_condition_satisfied": trial3_recommended_condition_satisfied,
            "trial3_current_branch_closeable": trial3_recommended_condition_satisfied,
            "selected_residual_route": selected_residual_route,
            "missing_v2_artifact": missing_v2_artifact,
            "recommended_next_route_or_none": None if trial3_recommended_condition_satisfied else "8.7.56.289",
        },
        {
            "overall_status": "trial3_refactored_declaration_third_gate_frozen",
            "trial3_branch_closeable": trial3_recommended_condition_satisfied,
            "advance_to_8_7_56_288": True,
            "next_required_artifacts": [] if trial3_recommended_condition_satisfied else ["trial3_refactored_trial2_paper_sync_trial4_disposition_fifteenth_refresh"],
        },
        {
            "audit_summary": audit["summary"],
            "prior_window_gate_summary": window_gate["summary"],
            "best_pair_or_none": best_pair,
        },
    )

    disposition = payload(
        "8.7.56.288",
        "Trial-2 paper-side sync / Trial-4 disposition fifteenth refresh",
        source_inventory["inputs"],
        "Refresh the reserve/deferred ordering after the central-amplitude extension and freeze the next official residual route.",
        {
            "trial2_rule": "Trial-2 paper-side sync stays unlocked reserve retained until the refactored Trial-3 route loses all honest current-canon search axes",
            "trial4_rule": "Trial-4 remains deferred while the refactored Trial-3 route still has an honest same-family or solver-domain residual path",
        },
        [
            row("trial3_refactored_trial2_trial4_fifteenth_refresh_complete", "pass", "Trial-2 paper-side sync / Trial-4 disposition fifteenth refresh complete", 1, "The reserve/deferred ordering is refreshed after the central-amplitude extension audit."),
            row("trial3_refactored_trial2_paper_side_sync_reserve_retained_fifteenth_refresh", "pass", "Trial-2 paper-side sync reserve retained", 1, "Trial-2 paper-side sync stays unlocked reserve work while Trial-3 remains scientifically open."),
            row("trial3_refactored_trial4_deferred_retained_fifteenth_refresh", "pass", "Trial-4 deferred retained", 1, "Trial-4 stays deferred while the refactored Trial-3 route still has an honest current-canon path."),
        ],
        {
            "selected_residual_route": selected_residual_route,
            "missing_v2_artifact": missing_v2_artifact,
            "trial2_paper_side_sync_state": "unlocked_reserve_retained",
            "trial4_deferred": True,
            "recommended_next_route_or_none": None if trial3_recommended_condition_satisfied else "8.7.56.289",
        },
        {
            "overall_status": "trial3_refactored_trial2_trial4_fifteenth_disposition_refreshed",
            "trial3_branch_closeable": trial3_recommended_condition_satisfied,
            "advance_to_8_7_56_13": False,
            "next_required_artifacts": [] if trial3_recommended_condition_satisfied else ["8.7.56.289"],
        },
        {
            "declaration_summary": declaration["summary"],
            "prior_window_disposition_summary": window_disposition["summary"],
            "solver_refactor_weak_summary": solver_refactor_weak["summary"],
        },
    )

    write_artifact("mass_origin_v2_trial3_refactored_post_ell18_central_amplitude_window_extension_source_inventory", source_inventory)
    write_artifact("mass_origin_v2_trial3_refactored_post_ell18_central_amplitude_window_extension_audit", audit)
    write_artifact("mass_origin_v2_trial3_refactored_declaration_third_gate", declaration)
    write_artifact("mass_origin_v2_trial3_refactored_paper_sync_trial4_disposition_fifteenth_refresh", disposition)

    print("[ok] wrote:")
    print(" - mass_origin_v2_trial3_refactored_post_ell18_central_amplitude_window_extension_source_inventory_metrics.json")
    print(" - mass_origin_v2_trial3_refactored_post_ell18_central_amplitude_window_extension_audit_metrics.json")
    print(" - mass_origin_v2_trial3_refactored_declaration_third_gate_metrics.json")
    print(" - mass_origin_v2_trial3_refactored_paper_sync_trial4_disposition_fifteenth_refresh_metrics.json")


# 関数: CLI から refactored post-ell18 central-amplitude branch を起動する。

if __name__ == "__main__":
    main()
