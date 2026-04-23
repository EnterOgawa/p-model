#!/usr/bin/env python3
"""
Generate Trial-3 two-component ratio-compatible anchor-family absolute-anchor-support
artifacts for 8.7.56.343-.346.

This branch follows the family-bridge audit. The current canon already contains a
ratio-compatible internal pair inside the collapsed anchor family `(17,1,1)`, so
the remaining question is narrower: why that same family still cannot support the
absolute W/Z anchors.
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

ADVICE = Path(r"C:\Users\ogawa\Downloads\pmodel_v2_trial3_two_component_pivot.md")

SPECTRUM_BRANCH = ROOT / "scripts" / "quantum" / "mass_origin_v2_trial3_two_component_spectrum_branch.py"
HELPER_BRANCH = ROOT / "scripts" / "quantum" / "mass_origin_v2_t3_post_ell18_amplitude_branch.py"
PIVOT_BRANCH = ROOT / "scripts" / "quantum" / "mass_origin_v2_trial3_two_component_pivot_branch.py"
NUMERICAL_BRANCH = ROOT / "scripts" / "quantum" / "mass_origin_vector_qball_numerical_solver_branch.py"
FULL_BRANCH = ROOT / "scripts" / "quantum" / "mass_origin_vector_qball_full_coupled_solver_branch.py"

PIVOT_ROUTE = OUT / "mass_origin_v2_trial3_two_component_pivot_route_contract_metrics.json"
PIVOT_ODE = OUT / "mass_origin_v2_trial3_two_component_coupled_radial_ode_derivation_metrics.json"
PIVOT_IMPLEMENTATION = OUT / "mass_origin_v2_trial3_two_component_shooting_solver_implementation_metrics.json"
SPECTRUM_METRICS = OUT / "mass_origin_v2_trial3_two_component_spectrum_computation_metrics.json"
WZ_COMPARISON = OUT / "mass_origin_v2_trial3_two_component_wz_target_comparison_metrics.json"
DECLARATION_GATE = OUT / "mass_origin_v2_trial3_two_component_declaration_gate_metrics.json"
FAMILY_BRIDGE_INVENTORY = OUT / "mass_origin_v2_trial3_two_component_anchor_pair_family_bridge_source_inventory_metrics.json"
FAMILY_BRIDGE_AUDIT = OUT / "mass_origin_v2_trial3_two_component_anchor_pair_family_bridge_audit_metrics.json"
FAMILY_BRIDGE_GATE = OUT / "mass_origin_v2_trial3_two_component_declaration_third_gate_metrics.json"
FAMILY_BRIDGE_DISPOSITION = OUT / "mass_origin_v2_trial3_two_component_paper_sync_trial4_disposition_twenty_eighth_refresh_metrics.json"
POST_PHOTON_PRESERVATION = OUT / "mass_origin_v2_post_photon_vector_mass_ratio_preservation_audit_metrics.json"
SCALAR_SPECTRUM = OUT / "mass_origin_qball_discrete_mass_spectrum_metrics.json"
VECTOR_SPIN = OUT / "mass_origin_vector_qball_spin_orbit_freeze_audit_metrics.json"

TRIAL2_RESERVE_STATE = "unlocked_reserve_retained"
NEXT_ROUTE = "8.7.56.347"


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


# 関数: state row から family signature を抜き出す。

def family_id(row_data: dict | None) -> dict | None:
    """Return the (k, ell, s) family signature for a state row."""
    if row_data is None:
        return None

    return {
        "k": int(row_data["k"]),
        "ell": int(row_data["ell"]),
        "s": int(row_data["s"]),
    }


# 関数: family signature を持つ rows だけを抽出する。

def filter_family(rows: list[dict], family: dict | None) -> list[dict]:
    """Return all rows that match a given family signature."""
    if family is None:
        return []

    return [
        row_data
        for row_data in rows
        if int(row_data["k"]) == int(family["k"])
        and int(row_data["ell"]) == int(family["ell"])
        and int(row_data["s"]) == int(family["s"])
    ]


# 関数: target 以下の最大 row を返す。

def nearest_lower(rows: list[dict], target_value: float) -> dict | None:
    """Return the largest row at or below the target ratio."""
    eligible = [row_data for row_data in rows if float(row_data["mass_ratio_to_scalar_base"]) <= float(target_value)]
    if not eligible:
        return None

    return max(eligible, key=lambda row_data: float(row_data["mass_ratio_to_scalar_base"]))


# 関数: target 以上の最小 row を返す。

def nearest_upper(rows: list[dict], target_value: float) -> dict | None:
    """Return the smallest row at or above the target ratio."""
    eligible = [row_data for row_data in rows if float(row_data["mass_ratio_to_scalar_base"]) >= float(target_value)]
    if not eligible:
        return None

    return min(eligible, key=lambda row_data: float(row_data["mass_ratio_to_scalar_base"]))


# 関数: Trial-3 two-component absolute-anchor-support residual branch を実行する。

def main() -> None:
    """Freeze the absolute-anchor-support audit after the family-bridge collapse."""
    for path in (
        ADVICE,
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        SPECTRUM_BRANCH,
        HELPER_BRANCH,
        PIVOT_BRANCH,
        NUMERICAL_BRANCH,
        FULL_BRANCH,
        PIVOT_ROUTE,
        PIVOT_ODE,
        PIVOT_IMPLEMENTATION,
        SPECTRUM_METRICS,
        WZ_COMPARISON,
        DECLARATION_GATE,
        FAMILY_BRIDGE_INVENTORY,
        FAMILY_BRIDGE_AUDIT,
        FAMILY_BRIDGE_GATE,
        FAMILY_BRIDGE_DISPOSITION,
        POST_PHOTON_PRESERVATION,
        SCALAR_SPECTRUM,
        VECTOR_SPIN,
    ):
        req(path)

    advice_text = read_text(ADVICE)
    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    ai_context = read_json(AI_CONTEXT)
    spectrum_branch_text = read_text(SPECTRUM_BRANCH)
    helper_text = read_text(HELPER_BRANCH)

    helper = load_module(HELPER_BRANCH, "trial3_two_component_abs_anchor_helper")
    spectrum_branch = load_module(SPECTRUM_BRANCH, "trial3_two_component_abs_anchor_spectrum")
    pivot = load_module(PIVOT_BRANCH, "trial3_two_component_abs_anchor_pivot")
    numerical = load_module(NUMERICAL_BRANCH, "trial3_two_component_abs_anchor_numerical")
    full = load_module(FULL_BRANCH, "trial3_two_component_abs_anchor_full")

    pivot_route = read_json(PIVOT_ROUTE)
    pivot_ode = read_json(PIVOT_ODE)
    pivot_implementation = read_json(PIVOT_IMPLEMENTATION)
    spectrum_metrics = read_json(SPECTRUM_METRICS)
    wz_metrics = read_json(WZ_COMPARISON)
    declaration_gate = read_json(DECLARATION_GATE)
    family_bridge_inventory = read_json(FAMILY_BRIDGE_INVENTORY)
    family_bridge_audit = read_json(FAMILY_BRIDGE_AUDIT)
    family_bridge_gate = read_json(FAMILY_BRIDGE_GATE)
    family_bridge_disposition = read_json(FAMILY_BRIDGE_DISPOSITION)
    post_photon_preservation = read_json(POST_PHOTON_PRESERVATION)
    scalar_spectrum = read_json(SCALAR_SPECTRUM)
    vector_spin = read_json(VECTOR_SPIN)

    scalar_modes = list(scalar_spectrum["evidence"]["discrete_mass_mode_rows"])
    lambda_rot = float(vector_spin["summary"]["lambda_rot_value"])
    normalization_scale = float(post_photon_preservation["summary"]["absolute_mass_normalization_scale_factor"])

    localized_rows, sector_summary = spectrum_branch.run_two_component_scan(pivot, numerical)
    base_modes, mode_summary = spectrum_branch.interpolate_two_component_modes(localized_rows)
    exact_rows = full.build_exact_ladder(scalar_modes, base_modes, lambda_rot)
    normalized_vector_rows = helper.normalize_vector_rows(
        [row_data for row_data in exact_rows if int(row_data["ell"]) > 0],
        normalization_scale,
    )

    best_w = helper.closest_state(normalized_vector_rows, spectrum_branch.W_TARGET)
    best_z = helper.closest_state(normalized_vector_rows, spectrum_branch.Z_TARGET)
    best_pair = spectrum_branch.best_ratio_pair_fast(normalized_vector_rows)
    anchor_family = family_id(best_w)
    pair_family = family_id(best_pair["lighter_state"]) if best_pair else None
    anchor_family_rows = filter_family(normalized_vector_rows, anchor_family)
    pair_family_rows = filter_family(normalized_vector_rows, pair_family)

    anchor_family_best_pair = spectrum_branch.best_ratio_pair_fast(anchor_family_rows)
    anchor_family_floor_row = min(anchor_family_rows, key=lambda row_data: float(row_data["mass_ratio_to_scalar_base"]))
    anchor_family_ceiling_row = max(anchor_family_rows, key=lambda row_data: float(row_data["mass_ratio_to_scalar_base"]))
    anchor_family_lower_w = nearest_lower(anchor_family_rows, spectrum_branch.W_TARGET)
    anchor_family_lower_z = nearest_lower(anchor_family_rows, spectrum_branch.Z_TARGET)
    anchor_family_upper_w = nearest_upper(anchor_family_rows, spectrum_branch.W_TARGET)
    anchor_family_upper_z = nearest_upper(anchor_family_rows, spectrum_branch.Z_TARGET)

    floor_value = float(anchor_family_floor_row["mass_ratio_to_scalar_base"])
    floor_above_w_target = bool(floor_value > float(spectrum_branch.W_TARGET))
    floor_above_z_target = bool(floor_value > float(spectrum_branch.Z_TARGET))
    lower_w_candidate_available = anchor_family_lower_w is not None
    lower_z_candidate_available = anchor_family_lower_z is not None
    absolute_anchor_support_available = bool(
        lower_w_candidate_available
        and lower_z_candidate_available
        and anchor_family_upper_w is not None
        and anchor_family_upper_z is not None
    )

    if floor_above_w_target and floor_above_z_target:
        selected_residual_route = "trial3_two_component_ratio_compatible_anchor_family_floor_lowering_identification"
        missing_v2_artifact = "trial3_two_component_ratio_compatible_anchor_family_floor_lowering_pack"
        next_open_question = "ratio-compatible anchor family exists, but its minimum mass floor still sits above both W and Z targets"
    elif not absolute_anchor_support_available:
        selected_residual_route = "trial3_two_component_ratio_compatible_anchor_family_lower_target_crossing_identification"
        missing_v2_artifact = "trial3_two_component_ratio_compatible_anchor_family_lower_target_crossing_pack"
        next_open_question = "ratio-compatible anchor family still lacks target-crossing support for one or both absolute anchors"
    else:
        selected_residual_route = None
        missing_v2_artifact = None
        next_open_question = "ratio-compatible anchor family already supports the absolute W/Z anchors"

    common_inputs = {
        "expert_note_markdown": str(ADVICE),
        "status_markdown": rel(STATUS),
        "roadmap_markdown": rel(ROADMAP),
        "ai_context_json": rel(AI_CONTEXT),
        "mass_origin_v2_trial3_two_component_pivot_route_contract_json": rel(PIVOT_ROUTE),
        "mass_origin_v2_trial3_two_component_coupled_radial_ode_derivation_json": rel(PIVOT_ODE),
        "mass_origin_v2_trial3_two_component_shooting_solver_implementation_json": rel(PIVOT_IMPLEMENTATION),
        "mass_origin_v2_trial3_two_component_spectrum_computation_json": rel(SPECTRUM_METRICS),
        "mass_origin_v2_trial3_two_component_wz_target_comparison_json": rel(WZ_COMPARISON),
        "mass_origin_v2_trial3_two_component_declaration_gate_json": rel(DECLARATION_GATE),
        "mass_origin_v2_trial3_two_component_anchor_pair_family_bridge_inventory_json": rel(FAMILY_BRIDGE_INVENTORY),
        "mass_origin_v2_trial3_two_component_anchor_pair_family_bridge_audit_json": rel(FAMILY_BRIDGE_AUDIT),
        "mass_origin_v2_trial3_two_component_declaration_third_gate_json": rel(FAMILY_BRIDGE_GATE),
        "mass_origin_v2_trial3_two_component_paper_sync_trial4_disposition_twenty_eighth_refresh_json": rel(FAMILY_BRIDGE_DISPOSITION),
        "mass_origin_v2_trial3_two_component_spectrum_branch_py": rel(SPECTRUM_BRANCH),
        "mass_origin_v2_t3_post_ell18_amplitude_branch_py": rel(HELPER_BRANCH),
        "mass_origin_vector_qball_full_coupled_solver_branch_py": rel(FULL_BRANCH),
    }

    source_inventory = payload(
        "8.7.56.343",
        "Trial-3 two-component ratio-compatible anchor-family absolute-anchor-support source inventory",
        common_inputs,
        "Collect the ratio-compatible anchor family, its near-exact internal pair, the absolute W/Z anchor miss, and the ceiling-only pair family evidence into one pack before the absolute-anchor-support audit.",
        {
            "inventory_rule": "the absolute-anchor-support audit starts only after the family-bridge audit has already shown that the anchor family itself is ratio-compatible",
            "support_focus": "the residual is local absolute-anchor support inside the anchor family, not a missing bridge to some other family",
        },
        [
            row("trial3_two_component_ratio_compatible_anchor_family_absolute_anchor_support_source_inventory_complete", "pass", "Trial-3 two-component ratio-compatible anchor-family absolute-anchor-support source inventory complete", 1, "The absolute-anchor-support source pack is frozen."),
            row("trial3_two_component_ratio_compatible_anchor_family_present", "pass" if anchor_family_rows else "reject", "ratio-compatible anchor family present", len(anchor_family_rows), "The anchor family itself is the current mainline candidate."),
            row("trial3_two_component_anchor_family_internal_pair_present", "pass" if anchor_family_best_pair else "reject", "anchor family internal pair present", 1 if anchor_family_best_pair else 0, "The current mainline candidate must already carry the near-exact pair shape."),
            row("trial3_two_component_anchor_family_absolute_anchor_miss_present", "pass", "anchor family absolute-anchor miss present", 1, "The same family still misses the absolute W/Z anchors and therefore remains a live residual."),
            row("trial3_two_component_pair_family_ceiling_only_pack_present", "pass" if pair_family_rows else "reject", "pair family ceiling-only evidence present", len(pair_family_rows), "The previous pair family remains as comparison evidence even though it no longer supplies the mainline bridge."),
        ],
        {
            "anchor_family_or_none": anchor_family,
            "pair_family_or_none": pair_family,
            "anchor_family_row_count": len(anchor_family_rows),
            "pair_family_row_count": len(pair_family_rows),
            "anchor_family_best_pair_or_none": anchor_family_best_pair,
            "anchor_family_floor_row_or_none": anchor_family_floor_row,
            "anchor_family_ceiling_row_or_none": anchor_family_ceiling_row,
            "best_w_row_or_none": best_w,
            "best_z_row_or_none": best_z,
            "next_required_route": "trial3_two_component_ratio_compatible_anchor_family_absolute_anchor_support_audit",
        },
        {
            "overall_status": "trial3_two_component_ratio_compatible_anchor_family_absolute_anchor_support_inventory_frozen",
            "advance_to_8_7_56_344": True,
            "next_required_artifacts": ["trial3_two_component_ratio_compatible_anchor_family_absolute_anchor_support_audit"],
        },
        {
            "status_next_step_line": hit(status_text, "current official next step は `8.7.56.343`"),
            "roadmap_placeholder_line": hit(roadmap_text, "`8.7.56.343-.346` 試練3 two-component ratio-compatible anchor-family absolute-anchor-support residual branch"),
            "family_bridge_summary": family_bridge_audit["summary"],
            "wz_comparison_summary": wz_metrics["summary"],
        },
    )

    audit = payload(
        "8.7.56.344",
        "Trial-3 two-component ratio-compatible anchor-family absolute-anchor-support audit",
        common_inputs,
        "Audit whether the ratio-compatible anchor family already contains any state below the W/Z thresholds, or whether its minimum mass floor itself blocks the absolute anchors.",
        {
            "floor_rule": "if the family minimum already sits above both W and Z targets, the blocker is a floor-lowering problem rather than a missing family bridge",
            "crossing_rule": "absolute-anchor support requires target-crossing support inside the same ratio-compatible family, not just a near-exact internal pair",
        },
        [
            row("trial3_two_component_ratio_compatible_anchor_family_absolute_anchor_support_audit_complete", "pass", "Trial-3 two-component ratio-compatible anchor-family absolute-anchor-support audit complete", 1, "The absolute-anchor-support audit is frozen."),
            row("trial3_two_component_anchor_family_floor_above_w_target", "reject" if floor_above_w_target else "pass", "anchor family minimum floor above W target", 1 if floor_above_w_target else 0, "If the family floor already exceeds the W target, no lower in-family W anchor can exist under the current canon."),
            row("trial3_two_component_anchor_family_floor_above_z_target", "reject" if floor_above_z_target else "pass", "anchor family minimum floor above Z target", 1 if floor_above_z_target else 0, "If the family floor already exceeds the Z target, no lower in-family Z anchor can exist under the current canon."),
            row("trial3_two_component_anchor_family_lower_w_candidate_available", "pass" if lower_w_candidate_available else "reject", "anchor family lower-than-W candidate available", 1 if lower_w_candidate_available else 0, "Absolute W support requires at least one in-family state at or below the W target."),
            row("trial3_two_component_anchor_family_lower_z_candidate_available", "pass" if lower_z_candidate_available else "reject", "anchor family lower-than-Z candidate available", 1 if lower_z_candidate_available else 0, "Absolute Z support requires at least one in-family state at or below the Z target."),
            row("trial3_two_component_anchor_family_absolute_anchor_support_available", "pass" if absolute_anchor_support_available else "reject", "anchor family absolute-anchor support available", 1 if absolute_anchor_support_available else 0, "The branch closes only if the ratio-compatible family also supports absolute W/Z crossings."),
        ],
        {
            "anchor_family_or_none": anchor_family,
            "anchor_family_floor_row_or_none": anchor_family_floor_row,
            "anchor_family_ceiling_row_or_none": anchor_family_ceiling_row,
            "anchor_family_best_pair_or_none": anchor_family_best_pair,
            "anchor_family_lower_w_or_none": anchor_family_lower_w,
            "anchor_family_lower_z_or_none": anchor_family_lower_z,
            "anchor_family_upper_w_or_none": anchor_family_upper_w,
            "anchor_family_upper_z_or_none": anchor_family_upper_z,
            "floor_above_w_target": floor_above_w_target,
            "floor_above_z_target": floor_above_z_target,
            "lower_w_candidate_available": lower_w_candidate_available,
            "lower_z_candidate_available": lower_z_candidate_available,
            "absolute_anchor_support_available": absolute_anchor_support_available,
            "next_required_route": "trial3_two_component_declaration_fourth_gate",
        },
        {
            "overall_status": "trial3_two_component_ratio_compatible_anchor_family_absolute_anchor_support_audited",
            "advance_to_8_7_56_345": True,
            "next_required_artifacts": ["trial3_two_component_declaration_fourth_gate"],
        },
        {
            "advice_two_component_line": hit(advice_text, "two-component"),
            "spectrum_branch_target_line": hit(spectrum_branch_text, "W_TARGET = W_MASS_MEV / ELECTRON_MASS_MEV"),
            "helper_closest_state_line": hit(helper_text, "def closest_state(rows: list[dict], target_value: float) -> dict | None:"),
            "family_bridge_gate_summary": family_bridge_gate["summary"],
            "family_bridge_disposition_summary": family_bridge_disposition["summary"],
            "sector_summary_sample": sample(
                [{"ell": key, **value} for key, value in sector_summary.items()],
                8,
            ),
        },
    )

    branch_closeable = absolute_anchor_support_available

    declaration = payload(
        "8.7.56.345",
        "Trial-3 two-component declaration fourth gate",
        common_inputs,
        "Freeze whether the ratio-compatible anchor family already closes the absolute W/Z anchors or whether the honest blocker has collapsed further to floor lowering inside that family.",
        {
            "closeout_rule": "Trial-3 closes only if the ratio-compatible anchor family supports both the pair shape and the absolute W/Z anchors",
            "residual_rule": "if the family floor stays above both targets, the next blocker is floor lowering inside that same family",
        },
        [
            row("trial3_two_component_declaration_fourth_gate_complete", "pass", "Trial-3 two-component declaration fourth gate complete", 1, "The fourth declaration gate is frozen."),
            row("trial3_two_component_branch_closeable_fourth_gate", "pass" if branch_closeable else "reject", "two-component branch closeable after absolute-anchor-support audit", 1 if branch_closeable else 0, "The branch closes only if the ratio-compatible family also closes the absolute anchors."),
            row("trial3_two_component_residual_route_required_fourth_gate", "reject" if branch_closeable else "pass", "two-component residual route still required after absolute-anchor-support audit", 0 if branch_closeable else 1, "A further residual route is required while the family floor still blocks the absolute anchors."),
        ],
        {
            "trial3_current_branch_closeable": branch_closeable,
            "selected_residual_route": selected_residual_route,
            "missing_v2_artifact": missing_v2_artifact,
            "recommended_next_route_or_none": None if branch_closeable else NEXT_ROUTE,
            "next_open_question": next_open_question,
        },
        {
            "overall_status": "trial3_two_component_declaration_fourth_gate_frozen",
            "trial3_branch_closeable": branch_closeable,
            "advance_to_8_7_56_346": True,
            "next_required_artifacts": [] if branch_closeable else [NEXT_ROUTE],
        },
        {
            "audit_summary": audit["summary"],
            "family_bridge_summary": family_bridge_audit["summary"],
            "current_ai_context_step": ai_context["current_step"],
        },
    )

    disposition = payload(
        "8.7.56.346",
        "Trial-2 paper-side sync / Trial-4 disposition twenty-ninth refresh",
        common_inputs,
        "Refresh the reserve/deferred ordering after the absolute-anchor-support audit collapses to the anchor-family floor problem.",
        {
            "trial2_rule": "Trial-2 paper-side sync stays unlocked reserve retained while the two-component Trial-3 route still has an honest current-canon residual path",
            "trial4_rule": "Trial-4 remains deferred while the two-component Trial-3 route is still scientifically live",
        },
        [
            row("trial3_two_component_trial2_trial4_twenty_ninth_refresh_complete", "pass", "Trial-2 paper-side sync / Trial-4 disposition twenty-ninth refresh complete", 1, "The reserve/deferred ordering is refreshed after the absolute-anchor-support audit."),
            row("trial3_two_component_trial2_reserve_retained_twenty_ninth_refresh", "pass", "Trial-2 paper-side sync reserve retained", 1, "Trial-2 paper sync remains reserve work while the two-component route still has an honest residual path."),
            row("trial3_two_component_trial4_deferred_retained_twenty_ninth_refresh", "pass", "Trial-4 deferred retained", 1, "Trial-4 stays deferred while the two-component route still has an honest residual path."),
        ],
        {
            "selected_residual_route": selected_residual_route,
            "missing_v2_artifact": missing_v2_artifact,
            "trial2_paper_side_sync_state": TRIAL2_RESERVE_STATE,
            "trial4_deferred": True,
            "recommended_next_route_or_none": None if branch_closeable else NEXT_ROUTE,
        },
        {
            "overall_status": "trial3_two_component_trial2_trial4_twenty_ninth_refresh_frozen",
            "trial3_branch_closeable": branch_closeable,
            "advance_to_next_branch": not branch_closeable,
            "next_required_artifacts": [] if branch_closeable else [NEXT_ROUTE],
        },
        {
            "declaration_summary": declaration["summary"],
            "family_bridge_disposition_summary": family_bridge_disposition["summary"],
            "wz_comparison_summary": wz_metrics["summary"],
        },
    )

    write_artifact("mass_origin_v2_trial3_two_component_ratio_compatible_anchor_family_absolute_anchor_support_source_inventory", source_inventory)
    write_artifact("mass_origin_v2_trial3_two_component_ratio_compatible_anchor_family_absolute_anchor_support_audit", audit)
    write_artifact("mass_origin_v2_trial3_two_component_declaration_fourth_gate", declaration)
    write_artifact("mass_origin_v2_trial3_two_component_paper_sync_trial4_disposition_twenty_ninth_refresh", disposition)

    print("[done] Trial-3 two-component absolute-anchor-support artifacts written:")
    print(" - mass_origin_v2_trial3_two_component_ratio_compatible_anchor_family_absolute_anchor_support_source_inventory_metrics.json")
    print(" - mass_origin_v2_trial3_two_component_ratio_compatible_anchor_family_absolute_anchor_support_audit_metrics.json")
    print(" - mass_origin_v2_trial3_two_component_declaration_fourth_gate_metrics.json")
    print(" - mass_origin_v2_trial3_two_component_paper_sync_trial4_disposition_twenty_ninth_refresh_metrics.json")


# 関数: CLI から branch を実行する。

def run_cli() -> None:
    """CLI entry point for the Trial-3 two-component absolute-anchor-support branch."""
    main()


if __name__ == "__main__":
    run_cli()
