#!/usr/bin/env python3
"""
Generate Trial-3 two-component charge-window-extension pivot artifacts for 8.7.56.355-.358.

This branch adopts the expert recommendation that the current two-component weak-sector
blocker is no longer a wording or broad-theory problem, but a solver search-range issue
inside the ratio-compatible anchor family `(k, ell, s) = (17, 1, 1)`. The branch extends
that family's charge window from the currently frozen `q_max = 40682` to `q_max = 60000`,
recomputes the exact family table, and then freezes a stricter question: do the resulting
absolute W/Z anchors close inside an admissible current-canon regime, or only after the
continued `beta_n` trajectory crosses unity and enters the clipped polarization branch of
the present full-coupled builder?
"""

from __future__ import annotations

import csv
import importlib.util
import json
import math
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output" / "public" / "quantum"
STATUS = ROOT / "doc" / "STATUS.md"
ROADMAP = ROOT / "doc" / "ROADMAP.md"
AI_CONTEXT = ROOT / "doc" / "AI_CONTEXT_MIN.json"

ADVICE = Path(r"C:\Users\ogawa\Downloads\pmodel_v2_trial3_charge_window_extension.md")

SPECTRUM_BRANCH = ROOT / "scripts" / "quantum" / "mass_origin_v2_trial3_two_component_spectrum_branch.py"
PIVOT_BRANCH = ROOT / "scripts" / "quantum" / "mass_origin_v2_trial3_two_component_pivot_branch.py"
NUMERICAL_BRANCH = ROOT / "scripts" / "quantum" / "mass_origin_vector_qball_numerical_solver_branch.py"
FULL_BRANCH = ROOT / "scripts" / "quantum" / "mass_origin_vector_qball_full_coupled_solver_branch.py"
HELPER_BRANCH = ROOT / "scripts" / "quantum" / "mass_origin_v2_t3_post_ell18_amplitude_branch.py"

POST_PHOTON_PRESERVATION = OUT / "mass_origin_v2_post_photon_vector_mass_ratio_preservation_audit_metrics.json"
VECTOR_SPIN = OUT / "mass_origin_vector_qball_spin_orbit_freeze_audit_metrics.json"
SCALAR_SPECTRUM = OUT / "mass_origin_qball_discrete_mass_spectrum_metrics.json"
UPPER_WINDOW_SOURCE = OUT / "mass_origin_v2_trial3_two_component_ratio_compatible_anchor_family_upper_charge_window_extension_source_inventory_metrics.json"
UPPER_WINDOW_AUDIT = OUT / "mass_origin_v2_trial3_two_component_ratio_compatible_anchor_family_upper_charge_window_extension_audit_metrics.json"
UPPER_WINDOW_GATE = OUT / "mass_origin_v2_trial3_two_component_declaration_sixth_gate_metrics.json"
UPPER_WINDOW_DISPOSITION = OUT / "mass_origin_v2_trial3_two_component_paper_sync_trial4_disposition_thirty_first_refresh_metrics.json"
ABS_SUPPORT_AUDIT = OUT / "mass_origin_v2_trial3_two_component_ratio_compatible_anchor_family_absolute_anchor_support_audit_metrics.json"
FLOOR_AUDIT = OUT / "mass_origin_v2_trial3_two_component_ratio_compatible_anchor_family_floor_lowering_audit_metrics.json"

ANCHOR_FAMILY = {"k": 17, "ell": 1, "s": 1}
ELL_VALUE = 1
CURRENT_Q_WINDOW = (39532, 40682)
EXTENDED_Q_MAX = 60000
TRIAL2_RESERVE_STATE = "unlocked_reserve_retained"
NEXT_ROUTE = "8.7.56.359"


# 関数: 現在の UTC 時刻を ISO 8601 形式で返す。
def now_iso() -> str:
    """Return the current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


# 関数: 必須入力 artifact の存在を確認する。

def req(path: Path) -> None:
    """Abort when a required input artifact is missing."""
    if not path.exists():
        raise SystemExit(f"[fail] missing required input: {path}")


# 関数: UTF-8 JSON artifact を辞書として読む。

def read_json(path: Path) -> dict:
    """Load a UTF-8 JSON artifact into a dictionary."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


# 関数: UTF-8 テキスト source を文字列として読む。

def read_text(path: Path) -> str:
    """Read a UTF-8 text source into memory."""
    return path.read_text(encoding="utf-8")


# 関数: 絶対パスを repo 相対表記へ変換する。

def rel(path: Path) -> str:
    """Return a repo-relative POSIX-style path string."""
    return str(path.relative_to(ROOT)).replace("\\", "/")


# 関数: source 内で最初に一致した pattern の行情報を返す。

def hit(text: str, pattern: str) -> dict | None:
    """Return the first line hit for a substring pattern, if any."""
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


# 関数: target family `(k, ell) = (17, 1)` の localized support を再走査する。

def run_targeted_family_localized_scan(spectrum, pivot, numerical) -> list[dict]:
    """Recompute the localized support rows for the anchor family's `(ell=1, k=17)` sector."""
    localized_rows: list[dict] = []
    for beta in spectrum.BETA_GRID:
        best_by_k: dict[int, dict] = {}
        for amp0 in spectrum.AMP0_GRID:
            for amp_l in spectrum.AMPL_GRID:
                solved = spectrum.solve_two_component_profile(
                    pivot,
                    numerical,
                    float(beta),
                    int(ELL_VALUE),
                    float(amp0),
                    float(amp_l),
                )
                tail_ratio = solved["tail_to_input_ratio"]
                if not solved["success"] or tail_ratio is None:
                    continue

                if float(tail_ratio) > float(spectrum.TAIL_RATIO_THRESHOLD):
                    continue

                k_value = int(solved["node_count_k"])
                current = best_by_k.get(k_value)
                if current is None or float(tail_ratio) < float(current["tail_to_input_ratio"]):
                    best_by_k[k_value] = solved

        for branch_index, k_value in enumerate(sorted(best_by_k), start=1):
            localized = dict(best_by_k[k_value])
            localized["localized_solution_found"] = True
            localized["solution_branch_index"] = int(branch_index)
            localized_rows.append(localized)

    family_rows = [
        row_data
        for row_data in localized_rows
        if int(row_data["ell"]) == int(ANCHOR_FAMILY["ell"])
        and int(row_data["node_count_k"]) == int(ANCHOR_FAMILY["k"])
    ]
    return sorted(family_rows, key=lambda item: float(item["beta"]))


# 関数: localized 2点を terminal segment continuation として `q_max=60000` まで延長する。

def extend_charge_window_family(localized_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Extend the ratio-compatible family to the advised `q_max` using the terminal localized segment."""
    if len(localized_rows) < 2:
        raise SystemExit("[fail] targeted family lacks the two localized support points required for continuation")

    left = localized_rows[-2]
    right = localized_rows[-1]
    q_left = float(left["charge_proxy"])
    q_right = float(right["charge_proxy"])
    if q_right == q_left:
        raise SystemExit("[fail] localized charge_proxy support is degenerate and cannot define a continuation segment")

    q_min = int(math.ceil(min(float(item["charge_proxy"]) for item in localized_rows)))
    base_modes: list[dict] = []
    continuation_rows: list[dict] = []
    for charge_index in range(q_min, int(EXTENDED_Q_MAX) + 1):
        fraction = (charge_index - q_left) / (q_right - q_left)
        beta_n = float(left["beta"]) + fraction * (float(right["beta"]) - float(left["beta"]))
        energy_n = float(left["energy_proxy"]) + fraction * (
            float(right["energy_proxy"]) - float(left["energy_proxy"])
        )
        continuation = {
            "n": int(charge_index),
            "beta_n": float(beta_n),
            "base_mass_proxy": float(energy_n),
            "inside_original_window": bool(charge_index <= int(CURRENT_Q_WINDOW[1])),
            "continued_beyond_original_qmax": bool(charge_index > int(CURRENT_Q_WINDOW[1])),
        }
        continuation_rows.append(continuation)
        base_modes.append(
            {
                "n": int(charge_index),
                "k": int(ANCHOR_FAMILY["k"]),
                "ell": int(ANCHOR_FAMILY["ell"]),
                "beta_n": float(beta_n),
                "charge_proxy_target": float(charge_index),
                "base_mass_proxy": float(energy_n),
                "node_count_k": int(ANCHOR_FAMILY["k"]),
                "origin": "trial3_two_component_charge_window_extension",
            }
        )

    return base_modes, continuation_rows


# 関数: family signature に一致する rows だけを抽出する。

def filter_family(rows: list[dict], family: dict, beta_upper_bound: float | None = None) -> list[dict]:
    """Return rows that match the fixed family signature, with an optional beta upper bound."""
    filtered = [
        row_data
        for row_data in rows
        if int(row_data["k"]) == int(family["k"])
        and int(row_data["ell"]) == int(family["ell"])
        and int(row_data["s"]) == int(family["s"])
    ]
    if beta_upper_bound is not None:
        filtered = [row_data for row_data in filtered if float(row_data["beta_n"]) <= float(beta_upper_bound)]

    return filtered


# 関数: target state の exact row を lookup する。

def find_state_row(rows: list[dict], state: dict | None) -> dict | None:
    """Look up the exact row backing a compact state summary."""
    if state is None:
        return None

    for row_data in rows:
        if (
            int(row_data["n"]) == int(state["n"])
            and int(row_data["k"]) == int(state["k"])
            and int(row_data["ell"]) == int(state["ell"])
            and int(row_data["s"]) == int(state["s"])
        ):
            return row_data

    return None


# 関数: state row を compact evidence へ圧縮する。

def compact_state(row_data: dict | None, summary: dict | None) -> dict | None:
    """Merge the compact state summary with the underlying builder-side metadata."""
    if row_data is None or summary is None:
        return None

    return {
        "n": int(summary["n"]),
        "k": int(summary["k"]),
        "ell": int(summary["ell"]),
        "s": int(summary["s"]),
        "ratio_value": float(summary.get("ratio_value", summary.get("mass_ratio_to_electron"))),
        "relative_error": float(summary.get("relative_error", 0.0)),
        "passes_threshold": bool(summary.get("passes_threshold", False)),
        "beta_n": float(row_data["beta_n"]),
        "polarization_weight": float(row_data["polarization_weight"]),
        "coupled_charge_factor": float(row_data["coupled_charge_factor"]),
        "coupled_mass_factor": float(row_data["coupled_mass_factor"]),
        "mass_ratio_to_scalar_base": float(row_data["mass_ratio_to_scalar_base"]),
    }


# 関数: charge-window-extension pivot branch を実行する。

def main() -> None:
    """Execute the advised charge-window-extension pivot and freeze the next honest blocker."""
    for path in (
        ADVICE,
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        SPECTRUM_BRANCH,
        PIVOT_BRANCH,
        NUMERICAL_BRANCH,
        FULL_BRANCH,
        HELPER_BRANCH,
        POST_PHOTON_PRESERVATION,
        VECTOR_SPIN,
        SCALAR_SPECTRUM,
        UPPER_WINDOW_SOURCE,
        UPPER_WINDOW_AUDIT,
        UPPER_WINDOW_GATE,
        UPPER_WINDOW_DISPOSITION,
        ABS_SUPPORT_AUDIT,
        FLOOR_AUDIT,
    ):
        req(path)

    advice_text = read_text(ADVICE)
    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    ai_context = read_json(AI_CONTEXT)
    spectrum_text = read_text(SPECTRUM_BRANCH)
    full_text = read_text(FULL_BRANCH)

    spectrum = load_module(SPECTRUM_BRANCH, "trial3_t2_charge_window_spectrum")
    pivot = load_module(PIVOT_BRANCH, "trial3_t2_charge_window_pivot")
    numerical = load_module(NUMERICAL_BRANCH, "trial3_t2_charge_window_numerical")
    full = load_module(FULL_BRANCH, "trial3_t2_charge_window_full")
    helper = load_module(HELPER_BRANCH, "trial3_t2_charge_window_helper")

    post_photon_preservation = read_json(POST_PHOTON_PRESERVATION)
    vector_spin = read_json(VECTOR_SPIN)
    scalar_spectrum = read_json(SCALAR_SPECTRUM)
    upper_window_source = read_json(UPPER_WINDOW_SOURCE)
    upper_window_audit = read_json(UPPER_WINDOW_AUDIT)
    upper_window_gate = read_json(UPPER_WINDOW_GATE)
    upper_window_disposition = read_json(UPPER_WINDOW_DISPOSITION)
    abs_support_audit = read_json(ABS_SUPPORT_AUDIT)
    floor_audit = read_json(FLOOR_AUDIT)

    localized_rows = run_targeted_family_localized_scan(spectrum, pivot, numerical)
    base_modes, continuation_rows = extend_charge_window_family(localized_rows)

    scalar_modes = list(scalar_spectrum["evidence"]["discrete_mass_mode_rows"])
    lambda_rot = float(vector_spin["summary"]["lambda_rot_value"])
    normalization_scale = float(post_photon_preservation["summary"]["absolute_mass_normalization_scale_factor"])

    exact_rows = full.build_exact_ladder(scalar_modes, base_modes, lambda_rot)
    normalized_vector_rows = helper.normalize_vector_rows(
        [row_data for row_data in exact_rows if int(row_data["ell"]) > 0],
        normalization_scale,
    )

    anchor_family_rows = filter_family(normalized_vector_rows, ANCHOR_FAMILY)
    subunity_anchor_family_rows = filter_family(normalized_vector_rows, ANCHOR_FAMILY, beta_upper_bound=1.0)

    exact_best_w_summary = helper.closest_state(anchor_family_rows, spectrum.W_TARGET)
    exact_best_z_summary = helper.closest_state(anchor_family_rows, spectrum.Z_TARGET)
    subunity_best_w_summary = helper.closest_state(subunity_anchor_family_rows, spectrum.W_TARGET)
    subunity_best_z_summary = helper.closest_state(subunity_anchor_family_rows, spectrum.Z_TARGET)
    subunity_best_pair_summary = spectrum.best_ratio_pair_fast(subunity_anchor_family_rows)

    exact_best_w_row = find_state_row(anchor_family_rows, exact_best_w_summary)
    exact_best_z_row = find_state_row(anchor_family_rows, exact_best_z_summary)
    subunity_best_w_row = find_state_row(subunity_anchor_family_rows, subunity_best_w_summary)
    subunity_best_z_row = find_state_row(subunity_anchor_family_rows, subunity_best_z_summary)
    subunity_pair_lighter_row = find_state_row(
        subunity_anchor_family_rows,
        None if subunity_best_pair_summary is None else subunity_best_pair_summary["lighter_state"],
    )
    subunity_pair_heavier_row = find_state_row(
        subunity_anchor_family_rows,
        None if subunity_best_pair_summary is None else subunity_best_pair_summary["heavier_state"],
    )

    exact_best_w = compact_state(exact_best_w_row, exact_best_w_summary)
    exact_best_z = compact_state(exact_best_z_row, exact_best_z_summary)
    subunity_best_w = compact_state(subunity_best_w_row, subunity_best_w_summary)
    subunity_best_z = compact_state(subunity_best_z_row, subunity_best_z_summary)
    subunity_best_pair = None
    if subunity_best_pair_summary and subunity_pair_lighter_row and subunity_pair_heavier_row:
        subunity_best_pair = {
            "lighter_state": compact_state(subunity_pair_lighter_row, subunity_best_pair_summary["lighter_state"]),
            "heavier_state": compact_state(subunity_pair_heavier_row, subunity_best_pair_summary["heavier_state"]),
            "mw_mz_ratio_value": float(subunity_best_pair_summary["mw_mz_ratio_value"]),
            "mw_mz_ratio_relative_error": float(subunity_best_pair_summary["mw_mz_ratio_relative_error"]),
            "sin2_theta_w_value": float(subunity_best_pair_summary["sin2_theta_w_value"]),
            "sin2_theta_w_relative_error": float(subunity_best_pair_summary["sin2_theta_w_relative_error"]),
            "passes_threshold": bool(subunity_best_pair_summary["passes_threshold"]),
        }

    solver_range_blocker_removed = bool(
        exact_best_w_summary
        and exact_best_w_summary["passes_threshold"]
        and exact_best_z_summary
        and exact_best_z_summary["passes_threshold"]
    )
    subunity_w_anchor_pass = bool(subunity_best_w_summary and subunity_best_w_summary["passes_threshold"])
    subunity_z_anchor_pass = bool(subunity_best_z_summary and subunity_best_z_summary["passes_threshold"])
    subunity_pair_pass = bool(subunity_best_pair and subunity_best_pair["passes_threshold"])
    exact_anchor_support_requires_beta_above_unity = bool(
        (exact_best_w and float(exact_best_w["beta_n"]) > 1.0)
        or (exact_best_z and float(exact_best_z["beta_n"]) > 1.0)
    )
    clipped_polarization_used_for_exact_anchor_support = bool(
        (exact_best_w and float(exact_best_w["beta_n"]) > 1.0 and float(exact_best_w["polarization_weight"]) == 0.0)
        or (exact_best_z and float(exact_best_z["beta_n"]) > 1.0 and float(exact_best_z["polarization_weight"]) == 0.0)
    )
    beta_above_unity_anchor_support_admissible_under_current_canon = False
    branch_closeable = bool(
        solver_range_blocker_removed
        and subunity_pair_pass
        and beta_above_unity_anchor_support_admissible_under_current_canon
    )

    current_q_min_line = hit(
        spectrum_text,
        'q_min = int(math.ceil(min(float(item["charge_proxy"]) for item in rows)))',
    )
    current_q_max_line = hit(
        spectrum_text,
        'q_max = int(math.floor(max(float(item["charge_proxy"]) for item in rows)))',
    )
    interpolation_loop_line = hit(spectrum_text, "for charge_index in range(q_min, q_max + 1):")
    clip_rule_line = hit(full_text, "localization = math.sqrt(max(0.0, 1.0 - beta_n * beta_n))")
    polarization_weight_line = hit(full_text, "def polarization_weight(beta_n: float, ell: int, s: int) -> float:")

    common_inputs = {
        "expert_note_markdown": str(ADVICE),
        "status_markdown": rel(STATUS),
        "roadmap_markdown": rel(ROADMAP),
        "ai_context_json": rel(AI_CONTEXT),
        "mass_origin_v2_trial3_two_component_ratio_compatible_anchor_family_upper_charge_window_extension_source_inventory_json": rel(
            UPPER_WINDOW_SOURCE
        ),
        "mass_origin_v2_trial3_two_component_ratio_compatible_anchor_family_upper_charge_window_extension_audit_json": rel(
            UPPER_WINDOW_AUDIT
        ),
        "mass_origin_v2_trial3_two_component_declaration_sixth_gate_json": rel(UPPER_WINDOW_GATE),
        "mass_origin_v2_trial3_two_component_paper_sync_trial4_disposition_thirty_first_refresh_json": rel(
            UPPER_WINDOW_DISPOSITION
        ),
        "mass_origin_v2_trial3_two_component_ratio_compatible_anchor_family_absolute_anchor_support_audit_json": rel(
            ABS_SUPPORT_AUDIT
        ),
        "mass_origin_v2_trial3_two_component_ratio_compatible_anchor_family_floor_lowering_audit_json": rel(
            FLOOR_AUDIT
        ),
        "mass_origin_v2_trial3_two_component_spectrum_branch_py": rel(SPECTRUM_BRANCH),
        "mass_origin_vector_qball_full_coupled_solver_branch_py": rel(FULL_BRANCH),
        "mass_origin_v2_trial3_two_component_pivot_branch_py": rel(PIVOT_BRANCH),
    }

    source_inventory = payload(
        "8.7.56.355",
        "Trial-3 two-component charge-window-extension pivot source inventory",
        common_inputs,
        "Adopt the expert charge-window-extension advice as a solver-side pivot and freeze the current window rule, the proposed q_max=60000 extension, the ratio-compatible family support points, and the present builder's beta>1 clip rule in one pack.",
        {
            "extension_rule": "keep the ratio-compatible family (17,1,1) fixed and extend q_max from the current localized-charge-span window [39532, 40682] to q_max = 60000 using the terminal localized segment",
            "admissibility_rule": "any exact-anchor gain obtained beyond beta_n = 1 must be judged against the current polarization_weight(beta_n, ell, s) clip rule before Trial-3 closeout can be declared",
        },
        [
            row(
                "trial3_t2_charge_window_pivot_source_inventory_complete",
                "pass",
                "Trial-3 two-component charge-window-extension pivot source inventory complete",
                1,
                "The charge-window-extension pivot source pack is frozen.",
            ),
            row(
                "trial3_t2_charge_window_advice_present",
                "pass",
                "charge-window extension advice present",
                1,
                "The expert recommendation is present and can be frozen as the pivot source.",
            ),
            row(
                "trial3_t2_anchor_family_localized_support_points_present",
                "pass" if len(localized_rows) >= 2 else "reject",
                "anchor-family localized support points present",
                len(localized_rows),
                "The family continuation requires the localized support points that define the terminal segment.",
            ),
            row(
                "trial3_t2_charge_window_extension_target_present",
                "pass",
                "charge-window extension target present",
                float(EXTENDED_Q_MAX),
                "The pivot uses the advised q_max = 60000 extension target.",
            ),
            row(
                "trial3_t2_beta_clip_rule_present",
                "pass" if clip_rule_line else "reject",
                "beta>1 clip rule present in current full-coupled builder",
                1 if clip_rule_line else 0,
                "The current builder-side admissibility question depends on whether anchor support enters the beta>1 clipped branch.",
            ),
        ],
        {
            "anchor_family_or_none": dict(ANCHOR_FAMILY),
            "current_charge_window_or_none": [int(value) for value in CURRENT_Q_WINDOW],
            "extended_charge_window_or_none": [int(CURRENT_Q_WINDOW[0]), int(EXTENDED_Q_MAX)],
            "localized_support_point_count": len(localized_rows),
            "current_generation_rule_present": bool(current_q_min_line and current_q_max_line and interpolation_loop_line),
            "beta_clip_rule_present": bool(clip_rule_line and polarization_weight_line),
            "next_required_route": "trial3_t2_charge_window_pivot_execution_audit",
        },
        {
            "overall_status": "trial3_t2_charge_window_pivot_inventory_frozen",
            "advance_to_8_7_56_356": True,
            "next_required_artifacts": ["trial3_t2_charge_window_pivot_execution_audit"],
        },
        {
            "status_next_step_line": hit(status_text, "current official next step は `8.7.56.355`"),
            "roadmap_branch_line": hit(
                roadmap_text,
                "`8.7.56.355-.358` 試練3 two-component ratio-compatible anchor-family upper-charge-proxy-continuation residual branch",
            ),
            "advice_qmax_line": hit(advice_text, "**q_max = 60000**"),
            "advice_solver_line": hit(advice_text, "solver の charge window 上限を `q_max = 60000` に変更"),
            "advice_retry_line": hit(advice_text, "retry loop は不要"),
            "current_q_min_line": current_q_min_line,
            "current_q_max_line": current_q_max_line,
            "interpolation_loop_line": interpolation_loop_line,
            "polarization_weight_line": polarization_weight_line,
            "clip_rule_line": clip_rule_line,
            "upper_window_audit_summary": upper_window_audit["summary"],
            "localized_support_rows": localized_rows,
        },
    )

    execution_audit = payload(
        "8.7.56.356",
        "Trial-3 two-component charge-window-extension pivot execution audit",
        common_inputs,
        "Execute the advised q_max=60000 extension on the ratio-compatible family and freeze whether the resulting W/Z anchor support closes honestly or only after beta_n crosses unity and activates the clipped polarization branch.",
        {
            "continuation_rule": "continue the anchor-family terminal localized segment linearly in charge_proxy -> (beta_n, base_mass_proxy) space from q = 40682 out to q = 60000",
            "subunity_rule": "separately audit the beta_n <= 1 subset to determine whether the absolute-anchor support survives without invoking the clipped polarization branch",
        },
        [
            row(
                "trial3_t2_charge_window_pivot_execution_complete",
                "pass",
                "Trial-3 two-component charge-window-extension pivot execution complete",
                1,
                "The advised q_max=60000 continuation has been executed.",
            ),
            row(
                "trial3_t2_solver_range_blocker_removed",
                "pass" if solver_range_blocker_removed else "reject",
                "solver range blocker removed by charge-window extension",
                1 if solver_range_blocker_removed else 0,
                "The advised extension succeeds if it numerically reaches both W and Z anchors inside the fixed anchor family.",
            ),
            row(
                "trial3_t2_same_family_subunity_pair_preserved",
                "pass" if subunity_pair_pass else "reject",
                "same-family beta<=1 pair preserved under charge-window extension",
                1 if subunity_pair_pass else 0,
                "The ratio-compatible pair must remain available even when the audit excludes beta_n > 1 states.",
            ),
            row(
                "trial3_t2_same_family_subunity_w_anchor_pass",
                "pass" if subunity_w_anchor_pass else "reject",
                "same-family beta<=1 W anchor passes",
                1 if subunity_w_anchor_pass else 0,
                "If the beta<=1 family already closes W, no beta>1 admissibility issue remains for the absolute anchors.",
            ),
            row(
                "trial3_t2_same_family_subunity_z_anchor_pass",
                "pass" if subunity_z_anchor_pass else "reject",
                "same-family beta<=1 Z anchor passes",
                1 if subunity_z_anchor_pass else 0,
                "The beta<=1 subset may already support Z even if W still needs further continuation.",
            ),
            row(
                "trial3_t2_exact_anchor_support_requires_beta_above_unity",
                "reject" if exact_anchor_support_requires_beta_above_unity else "pass",
                "exact same-family anchor support requires beta_n > 1",
                1 if exact_anchor_support_requires_beta_above_unity else 0,
                "A beta-above-unity dependence means the present gain is not automatically an honest current-canon closeout.",
            ),
            row(
                "trial3_t2_clipped_polarization_used_for_exact_anchor_support",
                "reject" if clipped_polarization_used_for_exact_anchor_support else "pass",
                "clipped polarization branch used for exact same-family anchor support",
                1 if clipped_polarization_used_for_exact_anchor_support else 0,
                "The present builder clips sqrt(1-beta_n^2) to zero above unity, so exact-anchor closure there needs an admissibility judgment.",
            ),
        ],
        {
            "anchor_family_or_none": dict(ANCHOR_FAMILY),
            "current_charge_window_or_none": [int(value) for value in CURRENT_Q_WINDOW],
            "extended_charge_window_or_none": [int(CURRENT_Q_WINDOW[0]), int(EXTENDED_Q_MAX)],
            "same_family_row_count": len(anchor_family_rows),
            "same_family_beta_leq_one_row_count": len(subunity_anchor_family_rows),
            "solver_range_blocker_removed": solver_range_blocker_removed,
            "same_family_subunity_pair_preserved": subunity_pair_pass,
            "same_family_subunity_w_anchor_pass": subunity_w_anchor_pass,
            "same_family_subunity_z_anchor_pass": subunity_z_anchor_pass,
            "exact_anchor_support_requires_beta_above_unity": exact_anchor_support_requires_beta_above_unity,
            "clipped_polarization_used_for_exact_anchor_support": clipped_polarization_used_for_exact_anchor_support,
            "next_required_route": "trial3_t2_charge_window_pivot_declaration_seventh_gate",
        },
        {
            "overall_status": "trial3_t2_charge_window_pivot_audited",
            "advance_to_8_7_56_357": True,
            "next_required_artifacts": ["trial3_t2_charge_window_pivot_declaration_seventh_gate"],
        },
        {
            "upper_window_source_summary": upper_window_source["summary"],
            "abs_support_audit_summary": abs_support_audit["summary"],
            "floor_audit_summary": floor_audit["summary"],
            "exact_best_w_or_none": exact_best_w,
            "exact_best_z_or_none": exact_best_z,
            "subunity_best_w_or_none": subunity_best_w,
            "subunity_best_z_or_none": subunity_best_z,
            "subunity_best_pair_or_none": subunity_best_pair,
            "continuation_row_sample": sample(continuation_rows, 16),
        },
    )

    declaration_gate = payload(
        "8.7.56.357",
        "Trial-3 two-component declaration seventh gate",
        common_inputs,
        "Freeze whether the advised charge-window-extension pivot actually closes Trial-3, or whether the honest next blocker has moved from search-range to beta-above-unity anchor-support admissibility under the current full-coupled builder.",
        {
            "closeout_rule": "close Trial-3 only if the charge-window extension closes the same-family W/Z anchors and pair without leaving an unresolved current-canon admissibility question",
            "residual_rule": "if the extension works numerically but the winning anchors require beta_n > 1 and the clipped polarization branch, the next blocker is anchor-support admissibility rather than further charge-window expansion",
        },
        [
            row(
                "trial3_t2_declaration_seventh_gate_complete",
                "pass",
                "Trial-3 two-component declaration seventh gate complete",
                1,
                "The charge-window-extension pivot gate is frozen.",
            ),
            row(
                "trial3_t2_branch_closeable_seventh_gate",
                "pass" if branch_closeable else "reject",
                "two-component branch closeable after charge-window-extension pivot",
                1 if branch_closeable else 0,
                "The branch closes only if the solver-range gain is also admissible under the current canon.",
            ),
            row(
                "trial3_t2_residual_route_required_seventh_gate",
                "reject" if branch_closeable else "pass",
                "two-component residual route still required after charge-window-extension pivot",
                0 if branch_closeable else 1,
                "A residual route remains required while exact-anchor support depends on an unresolved beta>1 clipped regime.",
            ),
        ],
        {
            "solver_range_blocker_removed": solver_range_blocker_removed,
            "same_family_subunity_pair_preserved": subunity_pair_pass,
            "same_family_subunity_w_anchor_pass": subunity_w_anchor_pass,
            "same_family_subunity_z_anchor_pass": subunity_z_anchor_pass,
            "exact_anchor_support_requires_beta_above_unity": exact_anchor_support_requires_beta_above_unity,
            "beta_above_unity_anchor_support_admissible_under_current_canon": beta_above_unity_anchor_support_admissible_under_current_canon,
            "trial3_current_branch_closeable": branch_closeable,
            "selected_residual_route": (
                None
                if branch_closeable
                else "trial3_two_component_beta_above_unity_anchor_support_admissibility_identification"
            ),
            "missing_v2_artifact": (
                None
                if branch_closeable
                else "trial3_two_component_beta_above_unity_anchor_support_admissibility_pack"
            ),
            "recommended_next_route_or_none": None if branch_closeable else NEXT_ROUTE,
        },
        {
            "overall_status": "trial3_t2_declaration_seventh_gate_frozen",
            "trial3_branch_closeable": branch_closeable,
            "advance_to_8_7_56_358": True,
            "next_required_artifacts": [] if branch_closeable else [NEXT_ROUTE],
        },
        {
            "execution_summary": execution_audit["summary"],
            "current_ai_context_step": ai_context["current_step"],
        },
    )

    disposition = payload(
        "8.7.56.358",
        "Trial-2 paper-side sync / Trial-4 disposition thirty-second refresh",
        common_inputs,
        "Refresh the reserve/deferred ordering after the charge-window-extension pivot moves the main blocker from search-range to beta-above-unity anchor-support admissibility.",
        {
            "trial2_rule": "Trial-2 paper-side sync remains unlocked reserve retained while the two-component Trial-3 route still has an honest admissibility blocker to resolve",
            "trial4_rule": "Trial-4 remains deferred while the two-component Trial-3 route is still scientifically live",
        },
        [
            row(
                "trial3_t2_trial2_trial4_thirty_second_refresh_complete",
                "pass",
                "Trial-2 paper-side sync / Trial-4 disposition thirty-second refresh complete",
                1,
                "The reserve/deferred ordering is refreshed after the charge-window-extension pivot.",
            ),
            row(
                "trial3_t2_trial2_reserve_retained_thirty_second_refresh",
                "pass",
                "Trial-2 paper-side sync reserve retained",
                1,
                "Trial-2 paper sync remains reserve work while the two-component admissibility route is still open.",
            ),
            row(
                "trial3_t2_trial4_deferred_retained_thirty_second_refresh",
                "pass",
                "Trial-4 deferred retained",
                1,
                "Trial-4 stays deferred while the two-component admissibility route remains live.",
            ),
        ],
        {
            "selected_residual_route": declaration_gate["summary"]["selected_residual_route"],
            "missing_v2_artifact": declaration_gate["summary"]["missing_v2_artifact"],
            "trial2_paper_side_sync_state": TRIAL2_RESERVE_STATE,
            "trial4_deferred": True,
            "recommended_next_route_or_none": declaration_gate["summary"]["recommended_next_route_or_none"],
        },
        {
            "overall_status": "trial3_t2_trial2_trial4_thirty_second_refresh_frozen",
            "trial3_branch_closeable": branch_closeable,
            "advance_to_next_branch": not branch_closeable,
            "next_required_artifacts": [] if branch_closeable else [NEXT_ROUTE],
        },
        {
            "declaration_summary": declaration_gate["summary"],
            "upper_window_disposition_summary": upper_window_disposition["summary"],
        },
    )

    write_artifact("mass_origin_v2_t3_t2_charge_window_pivot_source_inventory", source_inventory)
    write_artifact("mass_origin_v2_t3_t2_charge_window_pivot_execution_audit", execution_audit)
    write_artifact("mass_origin_v2_t3_t2_charge_window_pivot_declaration_gate", declaration_gate)
    write_artifact("mass_origin_v2_t3_t2_paper_sync_trial4_disp_32nd_refresh", disposition)

    print("[done] Trial-3 two-component charge-window-extension pivot artifacts written:")
    print(" - mass_origin_v2_t3_t2_charge_window_pivot_source_inventory_metrics.json")
    print(" - mass_origin_v2_t3_t2_charge_window_pivot_execution_audit_metrics.json")
    print(" - mass_origin_v2_t3_t2_charge_window_pivot_declaration_gate_metrics.json")
    print(" - mass_origin_v2_t3_t2_paper_sync_trial4_disp_32nd_refresh_metrics.json")


# 関数: CLI から branch を実行する。

def run_cli() -> None:
    """CLI entry point for the Trial-3 charge-window-extension pivot branch."""
    main()


if __name__ == "__main__":
    run_cli()
