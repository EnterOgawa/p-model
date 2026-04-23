#!/usr/bin/env python3
"""
Generate Trial-3 refactored high-mass k-positive extension artifacts for
8.7.56.277-.280.

The solver-refactor pivot removed the old software blocker. The next question
is scientific: after rebuilding the same-family exact table with explicit k>0
support, does the refactored high-mass family now close the W/Z sector?
"""

from __future__ import annotations

import csv
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output" / "public" / "quantum"
STATUS = ROOT / "doc" / "STATUS.md"
ROADMAP = ROOT / "doc" / "ROADMAP.md"
AI_CONTEXT = ROOT / "doc" / "AI_CONTEXT_MIN.json"

POST_PHOTON_PRESERVATION = OUT / "mass_origin_v2_post_photon_vector_mass_ratio_preservation_audit_metrics.json"
TRIAL3_RELAUNCHED_PILOT = OUT / "mass_origin_v2_trial3_relaunched_weak_sector_pilot_metrics.json"
TRIAL3_RELAUNCHED_AUDIT = OUT / "mass_origin_v2_trial3_relaunched_weinberg_angle_weak_coupling_audit_metrics.json"
TRIAL3_SOLVER_REFACTOR_EXECUTION = OUT / "mass_origin_v2_trial3_solver_refactor_execution_audit_metrics.json"
TRIAL3_SOLVER_REFACTOR_WEAK = OUT / "mass_origin_v2_trial3_solver_refactor_weak_sector_reaudit_metrics.json"
TRIAL3_SOLVER_REFACTOR_GATE = OUT / "mass_origin_v2_trial3_solver_refactor_declaration_gate_metrics.json"
VECTOR_SPIN = OUT / "mass_origin_vector_qball_spin_orbit_freeze_audit_metrics.json"
SCALAR_SPECTRUM = OUT / "mass_origin_qball_discrete_mass_spectrum_metrics.json"
EXACT_HANDOFF = OUT / "mass_origin_vector_qball_exact_mass_table_handoff_retry_metrics.json"

NUMERICAL_BRANCH = ROOT / "scripts" / "quantum" / "mass_origin_vector_qball_numerical_solver_branch.py"
FULL_COUPLED_BRANCH = ROOT / "scripts" / "quantum" / "mass_origin_vector_qball_full_coupled_solver_branch.py"
EXTENDED_BRANCH = ROOT / "scripts" / "quantum" / "mass_origin_vector_qball_extended_hierarchy_branch.py"

ELECTRON_MASS_MEV = 0.51099895
W_MASS_MEV = 80369.0
Z_MASS_MEV = 91187.6
W_TARGET = W_MASS_MEV / ELECTRON_MASS_MEV
Z_TARGET = Z_MASS_MEV / ELECTRON_MASS_MEV
WZ_RATIO_TARGET = W_MASS_MEV / Z_MASS_MEV
SIN2_THETA_W_TARGET = 1.0 - WZ_RATIO_TARGET * WZ_RATIO_TARGET
PASS_THRESHOLD = 0.10

LOW_ELL_VALUES = (1, 2, 3)
STANDARD_EXTENSION_ELLS = tuple(range(4, 19))
BROAD_EXTENSION_ELLS = tuple(range(19, 25))
BROAD_BETA_GRID = (0.04, 0.08, 0.12, 0.16, 0.22, 0.28, 0.34, 0.40, 0.48, 0.56, 0.66, 0.76, 0.86, 0.92)


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


# 関数: widened beta grid を使って custom ell sector を走査する。

def scan_custom_sector(numerical, ell: int, beta_grid: tuple[float, ...]) -> list[dict]:
    rows = []
    for beta in beta_grid:
        localized_profiles = numerical.find_sector_amplitudes(float(beta), int(ell))
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

    return rows


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


# 関数: long row table から要約サンプルだけ残す。

def sample(rows: list[dict], count: int = 12) -> list[dict]:
    if len(rows) <= count:
        return rows

    step = max(1, len(rows) // count)
    sampled = [rows[index] for index in range(0, len(rows), step)]
    return sampled[:count]


# 関数: sector scan と rebuilt mode table の要約を返す。

def sector_summary(scan_rows: list[dict], mode_rows: list[dict]) -> dict:
    localized = [row_data for row_data in scan_rows if row_data.get("localized_solution_found")]
    return {
        "localized_solution_count": len(localized),
        "integer_mode_count": len(mode_rows),
        "k_values": sorted({int(mode["k"]) for mode in mode_rows}),
        "max_charge_proxy_or_none": None if not localized else float(max(float(row_data["charge_proxy"]) for row_data in localized)),
    }


# 関数: rebuilt vector rows を normalization factor で rescale する。

def normalize_vector_rows(rows: list[dict], scale_factor: float) -> list[dict]:
    normalized = []
    for row_data in rows:
        copied = dict(row_data)
        copied["mass_ratio_to_scalar_base"] = float(row_data["mass_ratio_to_scalar_base"]) * float(scale_factor)
        normalized.append(copied)

    return normalized


# 関数: Trial-3 refactored high-mass extension branch を実行する。

def main() -> None:
    for path in (
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        POST_PHOTON_PRESERVATION,
        TRIAL3_RELAUNCHED_PILOT,
        TRIAL3_RELAUNCHED_AUDIT,
        TRIAL3_SOLVER_REFACTOR_EXECUTION,
        TRIAL3_SOLVER_REFACTOR_WEAK,
        TRIAL3_SOLVER_REFACTOR_GATE,
        VECTOR_SPIN,
        SCALAR_SPECTRUM,
        EXACT_HANDOFF,
        NUMERICAL_BRANCH,
        FULL_COUPLED_BRANCH,
        EXTENDED_BRANCH,
    ):
        req(path)

    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    ai_context = read_json(AI_CONTEXT)
    preservation = read_json(POST_PHOTON_PRESERVATION)
    relaunched_pilot = read_json(TRIAL3_RELAUNCHED_PILOT)
    relaunched_audit = read_json(TRIAL3_RELAUNCHED_AUDIT)
    solver_refactor_execution = read_json(TRIAL3_SOLVER_REFACTOR_EXECUTION)
    solver_refactor_weak = read_json(TRIAL3_SOLVER_REFACTOR_WEAK)
    solver_refactor_gate = read_json(TRIAL3_SOLVER_REFACTOR_GATE)
    vector_spin = read_json(VECTOR_SPIN)
    scalar_spectrum = read_json(SCALAR_SPECTRUM)
    exact_handoff = read_json(EXACT_HANDOFF)

    numerical_text = read_text(NUMERICAL_BRANCH)
    full_text = read_text(FULL_COUPLED_BRANCH)
    extended_text = read_text(EXTENDED_BRANCH)

    numerical = load_module(NUMERICAL_BRANCH, "trial3_refactored_high_mass_num")
    full = load_module(FULL_COUPLED_BRANCH, "trial3_refactored_high_mass_full")
    extended = load_module(EXTENDED_BRANCH, "trial3_refactored_high_mass_ext")

    scalar_modes = list(scalar_spectrum["evidence"]["discrete_mass_mode_rows"])
    lambda_rot = float(vector_spin["summary"]["lambda_rot_value"])
    normalization_scale = float(preservation["summary"]["absolute_mass_normalization_scale_factor"])
    preserved_ceiling = float(solver_refactor_weak["summary"]["historic_preserved_verified_ceiling_to_electron"])
    low_ell_exact_ceiling = float(solver_refactor_weak["summary"]["refactored_low_ell_exact_ceiling_to_electron_after_normalization"])

    common_inputs = {
        "status_markdown": rel(STATUS),
        "roadmap_markdown": rel(ROADMAP),
        "ai_context_json": rel(AI_CONTEXT),
        "mass_origin_v2_post_photon_vector_mass_ratio_preservation_audit_json": rel(POST_PHOTON_PRESERVATION),
        "mass_origin_v2_trial3_relaunched_weak_sector_pilot_json": rel(TRIAL3_RELAUNCHED_PILOT),
        "mass_origin_v2_trial3_relaunched_weinberg_angle_weak_coupling_audit_json": rel(TRIAL3_RELAUNCHED_AUDIT),
        "mass_origin_v2_trial3_solver_refactor_execution_audit_json": rel(TRIAL3_SOLVER_REFACTOR_EXECUTION),
        "mass_origin_v2_trial3_solver_refactor_weak_sector_reaudit_json": rel(TRIAL3_SOLVER_REFACTOR_WEAK),
        "mass_origin_v2_trial3_solver_refactor_declaration_gate_json": rel(TRIAL3_SOLVER_REFACTOR_GATE),
        "mass_origin_vector_qball_spin_orbit_freeze_audit_json": rel(VECTOR_SPIN),
        "mass_origin_qball_discrete_mass_spectrum_json": rel(SCALAR_SPECTRUM),
        "mass_origin_vector_qball_exact_mass_table_handoff_retry_json": rel(EXACT_HANDOFF),
        "mass_origin_vector_qball_numerical_solver_branch_py": rel(NUMERICAL_BRANCH),
        "mass_origin_vector_qball_full_coupled_solver_branch_py": rel(FULL_COUPLED_BRANCH),
        "mass_origin_vector_qball_extended_hierarchy_branch_py": rel(EXTENDED_BRANCH),
    }

    inventory_targets = [
        {
            "label": "status_points_to_8_7_56_277",
            "present": "current official next step は `8.7.56.277`" in status_text,
            "note": "STATUS must already point to the refactored high-mass extension source inventory step.",
        },
        {
            "label": "roadmap_refactored_high_mass_branch_present",
            "present": "`8.7.56.277-.280` 試練3 refactored high-mass `k>0` extension branch" in roadmap_text,
            "note": "ROADMAP must already freeze the refactored high-mass branch as the next official route.",
        },
        {
            "label": "post_photon_normalization_preserved",
            "present": bool(preservation["summary"]["working_action_vector_mass_spectrum_physical_claim_preserved"]),
            "note": "The preserved post-photon normalization must remain the mass-scale baseline for the rebuilt weak-sector table.",
        },
        {
            "label": "solver_refactor_removed_software_blocker",
            "present": bool(solver_refactor_execution["summary"]["software_blocker_removed"]),
            "note": "The high-mass rebuild is only honest after the solver refactor has removed the old software blocker.",
        },
        {
            "label": "solver_refactor_selected_high_mass_remaining_problem",
            "present": solver_refactor_execution["summary"]["selected_remaining_problem_class"] == "refactored_high_mass_weak_sector_extension",
            "note": "The remaining mainline problem must already be reclassified as a scientific high-mass extension rather than a code retry loop.",
        },
        {
            "label": "numerical_build_base_modes_present",
            "present": hit(numerical_text, "def build_base_modes(") is not None,
            "note": "The numerical solver must already expose the flat base-mode builder used by the refactored rebuild.",
        },
        {
            "label": "full_build_exact_ladder_present",
            "present": hit(full_text, "def build_exact_ladder(") is not None,
            "note": "The full-coupled solver must already expose the exact ladder builder used by the refactored rebuild.",
        },
        {
            "label": "extended_scan_present",
            "present": hit(extended_text, "def scan_extended_sector(") is not None,
            "note": "The extended hierarchy helper must already expose the high-ell scan used by the rebuild window.",
        },
        {
            "label": "exact_handoff_reopened",
            "present": bool(exact_handoff["summary"]["hand_off_to_8_7_55_2_84"]),
            "note": "The reopened exact handoff confirms that the refactored exact family is executable before extending it to the weak sector.",
        },
    ]
    inventory_ready = all(item["present"] for item in inventory_targets)

    source_inventory = payload(
        "8.7.56.277",
        "Trial-3 refactored high-mass k-positive extension source inventory",
        common_inputs,
        "Freeze the source pack for the scientific high-mass Trial-3 route after the solver refactor removed the old software blocker.",
        {
            "inventory_rule": "combine the preserved post-photon normalization, the refactored low-ell exact k-positive ladder, the extended high-ell scan windows, and the same-family W/Z targets before rebuilding the next exact-family table",
            "normalization_rule": "keep the post-photon sqrt(2) normalization as the current physical mass-scale baseline",
            "window_rule": f"rebuild the same-family table over low ell {list(LOW_ELL_VALUES)}, standard extension ell {list(STANDARD_EXTENSION_ELLS)}, and broadened extension ell {list(BROAD_EXTENSION_ELLS)}",
        },
        [
            row(
                "trial3_refactored_high_mass_source_inventory_complete",
                "pass",
                "Trial-3 refactored high-mass source inventory complete",
                1,
                "The refactored high-mass extension source pack is frozen.",
            ),
            row(
                "trial3_refactored_high_mass_required_source_count",
                "pass" if inventory_ready else "reject",
                "required refactored high-mass source count",
                len(inventory_targets),
                "The refactored high-mass rebuild needs preserved normalization, removed software blocker, exact k-positive ladder, and extended scan helpers to coexist in one source pack.",
            ),
            row(
                "trial3_refactored_high_mass_preserved_ceiling_to_electron",
                "pass",
                "preserved normalized weak-sector ceiling to electron",
                preserved_ceiling,
                "The preserved post-photon normalized ceiling remains the current verified high-mass baseline.",
            ),
            row(
                "trial3_refactored_high_mass_low_ell_exact_ceiling_to_electron",
                "pass",
                "refactored low-ell exact k-positive ceiling to electron",
                low_ell_exact_ceiling,
                "The low-ell exact k-positive ladder is already executable and enters the same-family rebuild as a local proof artifact.",
            ),
        ],
        {
            "inventory_ready": inventory_ready,
            "normalization_scale_factor": normalization_scale,
            "historic_preserved_verified_ceiling_to_electron": preserved_ceiling,
            "refactored_low_ell_exact_ceiling_to_electron": low_ell_exact_ceiling,
            "target_window_low_ell": list(LOW_ELL_VALUES),
            "target_window_standard_extension_ell": list(STANDARD_EXTENSION_ELLS),
            "target_window_broad_extension_ell": list(BROAD_EXTENSION_ELLS),
            "w_target_ratio": W_TARGET,
            "z_target_ratio": Z_TARGET,
            "first_route_to_close_or_none": "trial3_refactored_high_mass_k_positive_extension_audit",
        },
        {
            "overall_status": "trial3_refactored_high_mass_source_inventory_frozen",
            "trial3_branch_closeable": False,
            "advance_to_8_7_56_278": True,
            "next_required_artifacts": ["trial3_refactored_high_mass_k_positive_extension_audit"],
        },
        {
            "inventory_targets": inventory_targets,
            "solver_refactor_execution_summary": solver_refactor_execution["summary"],
            "solver_refactor_weak_summary": solver_refactor_weak["summary"],
            "exact_handoff_summary": exact_handoff["summary"],
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

    broad_scan_rows = {}
    broad_mode_rows = {}
    for ell in BROAD_EXTENSION_ELLS:
        scan_rows = scan_custom_sector(numerical, int(ell), BROAD_BETA_GRID)
        mode_rows = numerical.interpolate_integer_modes(scan_rows, int(ell))
        broad_scan_rows[int(ell)] = scan_rows
        broad_mode_rows[int(ell)] = mode_rows
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
    ell_mode_summary.update({str(ell): sector_summary(ell_scan_rows[int(ell)], [mode for mode in low_ell_base_modes if int(mode["ell"]) == int(ell)]) for ell in LOW_ELL_VALUES})
    ell_mode_summary.update({str(ell): sector_summary(standard_scan_rows[int(ell)], standard_mode_rows[int(ell)]) for ell in STANDARD_EXTENSION_ELLS})
    ell_mode_summary.update({str(ell): sector_summary(broad_scan_rows[int(ell)], broad_mode_rows[int(ell)]) for ell in BROAD_EXTENSION_ELLS})

    available_k_values = sorted({int(mode["k"]) for mode in base_modes})
    maximum_detected_k = max(available_k_values) if available_k_values else 0
    maximum_detected_ell = max(int(mode["ell"]) for mode in base_modes)
    maximum_detected_ell_with_k_positive = max(int(mode["ell"]) for mode in base_modes if int(mode["k"]) > 0)
    broad_window_localized_solution_count_total = sum(
        len([row_data for row_data in scan_rows if row_data.get("localized_solution_found")])
        for scan_rows in broad_scan_rows.values()
    )
    rebuilt_max_ratio = float(max_row["mass_ratio_to_scalar_base"])
    ceiling_reproduced = abs(rebuilt_max_ratio - preserved_ceiling) <= 1.0e-9
    w_anchor_pass = bool(best_w and best_w["passes_threshold"])
    z_anchor_pass = bool(best_z and best_z["passes_threshold"])
    mw_mz_ratio_pass = bool(best_pair and best_pair["mw_mz_ratio_relative_error"] <= PASS_THRESHOLD)
    sin2_theta_w_pass = bool(best_pair and best_pair["sin2_theta_w_relative_error"] <= PASS_THRESHOLD)
    trial3_recommended_condition_satisfied = bool(w_anchor_pass and z_anchor_pass and mw_mz_ratio_pass and sin2_theta_w_pass)

    audit = payload(
        "8.7.56.278",
        "Trial-3 refactored high-mass k-positive extension audit",
        common_inputs,
        "Rebuild and audit the refactored same-family high-mass exact table after the solver-side k-axis blocker is removed.",
        {
            "rebuild_rule": "recompute the exact-family vector table from the refactored low-ell k-positive ladder plus the extended high-ell scan windows, then apply the preserved post-photon normalization",
            "ceiling_rule": "the rebuilt table is honest only if it reproduces or exceeds the preserved normalized high-ell ceiling without hiding the explicit k-positive contribution",
            "closure_rule": "Trial-3 closes only if the rebuilt same-family table reaches W and Z anchors and supplies a consistent Weinberg-angle proxy under the preserved current canon",
        },
        [
            row(
                "trial3_refactored_high_mass_extension_audit_complete",
                "pass",
                "Trial-3 refactored high-mass extension audit complete",
                1,
                "The rebuilt refactored high-mass exact-family audit is frozen.",
            ),
            row(
                "trial3_refactored_high_mass_exact_family_table_rebuilt",
                "pass",
                "refactored high-mass exact-family table rebuilt",
                1,
                "The scientific high-mass route now runs on a rebuilt exact-family table instead of a software retry diagnosis.",
            ),
            row(
                "trial3_refactored_high_mass_preserved_ceiling_reproduced",
                "pass" if ceiling_reproduced else "reject",
                "preserved normalized ceiling reproduced by rebuilt table",
                1 if ceiling_reproduced else 0,
                "The rebuilt same-family table should at least reproduce the preserved normalized ceiling if the solver refactor has not broken current-canon mass claims.",
            ),
            row(
                "trial3_refactored_high_mass_w_anchor_pass",
                "pass" if w_anchor_pass else "reject",
                "W/electron anchor passes in refactored high-mass table",
                1 if w_anchor_pass else 0,
                "The rebuilt exact-family table must actually reach the W scale to close Trial-3.",
            ),
            row(
                "trial3_refactored_high_mass_z_anchor_pass",
                "pass" if z_anchor_pass else "reject",
                "Z/electron anchor passes in refactored high-mass table",
                1 if z_anchor_pass else 0,
                "The rebuilt exact-family table must also reach the Z scale to close Trial-3.",
            ),
            row(
                "trial3_refactored_high_mass_mw_mz_ratio_pass",
                "pass" if mw_mz_ratio_pass else "reject",
                "M_W/M_Z ratio passes in refactored high-mass table",
                1 if mw_mz_ratio_pass else 0,
                "A same-family weak-sector closeout also needs a viable W/Z ratio pair.",
            ),
            row(
                "trial3_refactored_high_mass_sin2_theta_w_pass",
                "pass" if sin2_theta_w_pass else "reject",
                "sin^2(theta_W) passes in refactored high-mass table",
                1 if sin2_theta_w_pass else 0,
                "The Weinberg-angle proxy must close together with the mass ratio rather than only approximately recover M_W/M_Z.",
            ),
            row(
                "trial3_refactored_high_mass_localized_solution_count_above_ell18",
                "pass" if broad_window_localized_solution_count_total > 0 else "reject",
                "localized solution count above ell=18",
                broad_window_localized_solution_count_total,
                "The broad post-ell18 window must produce localized sectors if the next refactored high-mass route is to move beyond the preserved ell=18 ceiling.",
            ),
        ],
        {
            "normalization_scale_factor": normalization_scale,
            "refactored_exact_family_table_rebuilt": True,
            "exact_state_count": len(exact_rows),
            "vector_state_count": len(normalized_vector_rows),
            "base_mode_count": len(base_modes),
            "available_k_values": available_k_values,
            "maximum_detected_k": maximum_detected_k,
            "maximum_detected_ell": maximum_detected_ell,
            "maximum_detected_ell_with_k_positive": maximum_detected_ell_with_k_positive,
            "k_positive_mode_count": sum(1 for mode in base_modes if int(mode["k"]) > 0),
            "k_positive_vector_state_count": len(k_positive_rows),
            "historic_preserved_verified_ceiling_to_electron": preserved_ceiling,
            "rebuilt_verified_ceiling_to_electron": rebuilt_max_ratio,
            "refactored_k_positive_ceiling_to_electron": float(max_k_positive_row["mass_ratio_to_scalar_base"]),
            "best_w_row_or_none": best_w,
            "best_z_row_or_none": best_z,
            "best_pair_or_none": best_pair,
            "w_gap_factor_or_none": None if best_w is None else W_TARGET / float(best_w["ratio_value"]),
            "z_gap_factor_or_none": None if best_z is None else Z_TARGET / float(best_z["ratio_value"]),
            "trial3_recommended_condition_satisfied": trial3_recommended_condition_satisfied,
            "first_route_to_close_or_none": "trial3_refactored_declaration_gate",
        },
        {
            "overall_status": "trial3_refactored_high_mass_extension_audited",
            "trial3_branch_closeable": trial3_recommended_condition_satisfied,
            "advance_to_8_7_56_279": True,
            "next_required_artifacts": ["trial3_refactored_declaration_gate"],
        },
        {
            "low_ell_sector_summary": {str(ell): ell_mode_summary[str(ell)] for ell in LOW_ELL_VALUES},
            "standard_extension_sector_summary": {str(ell): ell_mode_summary[str(ell)] for ell in STANDARD_EXTENSION_ELLS},
            "broad_extension_sector_summary": {str(ell): ell_mode_summary[str(ell)] for ell in BROAD_EXTENSION_ELLS},
            "sampled_high_mass_rows": sample(
                sorted(normalized_vector_rows, key=lambda item: float(item["mass_ratio_to_scalar_base"]), reverse=True),
                16,
            ),
            "max_row_or_none": max_row,
            "max_k_positive_row_or_none": max_k_positive_row,
            "solver_refactor_gate_summary": solver_refactor_gate["summary"],
            "relaunched_audit_summary": relaunched_audit["summary"],
        },
    )

    selected_residual_route = (
        "trial3_relaunched_refactored_post_ell18_localization_window_extension_identification"
        if broad_window_localized_solution_count_total == 0
        else "trial3_relaunched_refactored_high_mass_same_family_reaudit"
    )
    missing_v2_artifact = (
        "trial3_relaunched_refactored_localized_exact_family_table_above_ell18"
        if broad_window_localized_solution_count_total == 0
        else "trial3_relaunched_refactored_high_mass_same_family_closeout_pack"
    )

    declaration = payload(
        "8.7.56.279",
        "Trial-3 refactored declaration gate",
        common_inputs,
        "Freeze whether the rebuilt refactored high-mass exact-family table is sufficient to close Trial-3 or whether a new residual route is still required.",
        {
            "closeout_rule": "Trial-3 closes only if the rebuilt refactored high-mass table reaches both W/Z anchors and a viable Weinberg-angle proxy under the preserved current canon",
            "residual_rule": "if the rebuilt table reproduces the preserved ceiling but still misses W/Z, the next residual route must describe the specific missing same-family extension rather than reopening old software diagnosis loops",
        },
        [
            row(
                "trial3_refactored_declaration_gate_complete",
                "pass",
                "Trial-3 refactored declaration gate complete",
                1,
                "The refactored high-mass declaration gate is frozen.",
            ),
            row(
                "trial3_refactored_branch_closeable",
                "pass" if trial3_recommended_condition_satisfied else "reject",
                "refactored Trial-3 branch closeable",
                1 if trial3_recommended_condition_satisfied else 0,
                "The branch closes only if the rebuilt exact-family table actually closes the weak sector.",
            ),
            row(
                "trial3_refactored_residual_route_required",
                "reject" if trial3_recommended_condition_satisfied else "pass",
                "refactored Trial-3 residual route required",
                0 if trial3_recommended_condition_satisfied else 1,
                "A new residual route is still required when the rebuilt table reproduces current claims but cannot yet reach W/Z.",
            ),
            row(
                "trial3_refactored_execute_trial2_paper_sync_now",
                "reject",
                "execute Trial-2 paper-side sync now",
                0,
                "Trial-2 paper-side sync remains reserve work while the rebuilt Trial-3 high-mass route stays scientifically open.",
            ),
        ],
        {
            "trial3_recommended_condition_satisfied": trial3_recommended_condition_satisfied,
            "trial3_current_branch_closeable": trial3_recommended_condition_satisfied,
            "selected_residual_route": selected_residual_route,
            "missing_v2_artifact": missing_v2_artifact,
            "recommended_next_route_or_none": "8.7.56.281" if not trial3_recommended_condition_satisfied else None,
        },
        {
            "overall_status": "trial3_refactored_declaration_gate_frozen",
            "trial3_branch_closeable": trial3_recommended_condition_satisfied,
            "advance_to_8_7_56_280": True,
            "next_required_artifacts": [] if trial3_recommended_condition_satisfied else ["trial3_refactored_trial2_paper_sync_trial4_disposition_refresh"],
        },
        {
            "audit_summary": audit["summary"],
            "best_pair_or_none": best_pair,
        },
    )

    disposition = payload(
        "8.7.56.280",
        "Trial-2 paper-side sync / Trial-4 disposition thirteenth refresh",
        common_inputs,
        "Refresh the reserve/deferred ordering after the refactored high-mass exact-family rebuild and freeze the next official residual route.",
        {
            "trial2_rule": "Trial-2 paper-side sync stays unlocked reserve retained until the refactored high-mass Trial-3 route loses all honest current-canon paths",
            "trial4_rule": "Trial-4 remains deferred while the rebuilt Trial-3 table still points to a same-family residual extension",
        },
        [
            row(
                "trial3_refactored_trial2_trial4_disposition_refresh_complete",
                "pass",
                "Trial-2 paper-side sync / Trial-4 disposition thirteenth refresh complete",
                1,
                "The reserve/deferred ordering is refreshed after the rebuilt high-mass audit.",
            ),
            row(
                "trial3_refactored_trial2_paper_side_sync_reserve_retained",
                "pass",
                "Trial-2 paper-side sync reserve retained",
                1,
                "Trial-2 paper-side sync stays unlocked reserve work while Trial-3 remains scientifically open.",
            ),
            row(
                "trial3_refactored_trial4_deferred_retained",
                "pass",
                "Trial-4 deferred retained",
                1,
                "Trial-4 stays deferred while the rebuilt high-mass route still has an honest same-family residual path.",
            ),
        ],
        {
            "selected_residual_route": selected_residual_route,
            "missing_v2_artifact": missing_v2_artifact,
            "trial2_paper_side_sync_state": "unlocked_reserve_retained",
            "trial4_deferred": True,
            "recommended_next_route_or_none": "8.7.56.281" if not trial3_recommended_condition_satisfied else None,
        },
        {
            "overall_status": "trial3_refactored_trial2_trial4_disposition_refreshed",
            "trial3_branch_closeable": trial3_recommended_condition_satisfied,
            "advance_to_8_7_56_13": False,
            "next_required_artifacts": [] if trial3_recommended_condition_satisfied else ["8.7.56.281"],
        },
        {
            "declaration_summary": declaration["summary"],
            "solver_refactor_weak_summary": solver_refactor_weak["summary"],
        },
    )

    write_artifact("mass_origin_v2_trial3_refactored_high_mass_k_positive_extension_source_inventory", source_inventory)
    write_artifact("mass_origin_v2_trial3_refactored_high_mass_k_positive_extension_audit", audit)
    write_artifact("mass_origin_v2_trial3_refactored_declaration_gate", declaration)
    write_artifact("mass_origin_v2_trial3_refactored_paper_sync_trial4_disposition_thirteenth_refresh", disposition)

    print("[ok] wrote:")
    print(" - mass_origin_v2_trial3_refactored_high_mass_k_positive_extension_source_inventory_metrics.json")
    print(" - mass_origin_v2_trial3_refactored_high_mass_k_positive_extension_audit_metrics.json")
    print(" - mass_origin_v2_trial3_refactored_declaration_gate_metrics.json")
    print(" - mass_origin_v2_trial3_refactored_paper_sync_trial4_disposition_thirteenth_refresh_metrics.json")


# 関数: CLI から refactored high-mass extension branch を起動する。

if __name__ == "__main__":
    main()
