#!/usr/bin/env python3
"""
Generate Trial-3 two-component anchor/pair family-bridge artifacts for 8.7.56.339-.342.

This branch revisits the first two-component weak-sector table after the distinct
anchor-split audit. The goal is to determine whether the collapsed anchor family
and the near-pass pair family are truly disconnected, or whether one of those
families already carries the missing bridge candidate and only lacks a more local
absolute-anchor closeout.
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
ANCHOR_SPLIT_INVENTORY = OUT / "mass_origin_v2_trial3_two_component_distinct_w_z_anchor_split_source_inventory_metrics.json"
ANCHOR_SPLIT_AUDIT = OUT / "mass_origin_v2_trial3_two_component_distinct_w_z_anchor_split_audit_metrics.json"
ANCHOR_SPLIT_GATE = OUT / "mass_origin_v2_trial3_two_component_declaration_second_gate_metrics.json"
ANCHOR_SPLIT_DISPOSITION = OUT / "mass_origin_v2_trial3_two_component_paper_sync_trial4_disposition_twenty_seventh_refresh_metrics.json"
POST_PHOTON_PRESERVATION = OUT / "mass_origin_v2_post_photon_vector_mass_ratio_preservation_audit_metrics.json"
SCALAR_SPECTRUM = OUT / "mass_origin_qball_discrete_mass_spectrum_metrics.json"
VECTOR_SPIN = OUT / "mass_origin_vector_qball_spin_orbit_freeze_audit_metrics.json"

TRIAL2_RESERVE_STATE = "unlocked_reserve_retained"
NEXT_ROUTE = "8.7.56.343"


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


# 関数: family signature を mode_summary key へ変換する。

def family_mode_key(family: dict | None) -> str | None:
    """Return the interpolate_two_component_modes key for a family signature."""
    if family is None:
        return None

    return f"{int(family['ell'])}:{int(family['k'])}"


# 関数: 一致する family の rows だけを抽出する。

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


# 関数: Trial-3 two-component anchor/pair family-bridge residual branch を実行する。

def main() -> None:
    """Freeze the family-bridge audit after the distinct-anchor split residual."""
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
        ANCHOR_SPLIT_INVENTORY,
        ANCHOR_SPLIT_AUDIT,
        ANCHOR_SPLIT_GATE,
        ANCHOR_SPLIT_DISPOSITION,
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

    helper = load_module(HELPER_BRANCH, "trial3_two_component_bridge_helper")
    spectrum_branch = load_module(SPECTRUM_BRANCH, "trial3_two_component_bridge_spectrum")
    pivot = load_module(PIVOT_BRANCH, "trial3_two_component_bridge_pivot")
    numerical = load_module(NUMERICAL_BRANCH, "trial3_two_component_bridge_numerical")
    full = load_module(FULL_BRANCH, "trial3_two_component_bridge_full")

    pivot_route = read_json(PIVOT_ROUTE)
    pivot_ode = read_json(PIVOT_ODE)
    pivot_implementation = read_json(PIVOT_IMPLEMENTATION)
    spectrum_metrics = read_json(SPECTRUM_METRICS)
    wz_metrics = read_json(WZ_COMPARISON)
    declaration_gate = read_json(DECLARATION_GATE)
    anchor_split_inventory = read_json(ANCHOR_SPLIT_INVENTORY)
    anchor_split_audit = read_json(ANCHOR_SPLIT_AUDIT)
    anchor_split_gate = read_json(ANCHOR_SPLIT_GATE)
    anchor_split_disposition = read_json(ANCHOR_SPLIT_DISPOSITION)
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

    anchor_family_best_w = helper.closest_state(anchor_family_rows, spectrum_branch.W_TARGET)
    anchor_family_best_z = helper.closest_state(anchor_family_rows, spectrum_branch.Z_TARGET)
    pair_family_best_w = helper.closest_state(pair_family_rows, spectrum_branch.W_TARGET)
    pair_family_best_z = helper.closest_state(pair_family_rows, spectrum_branch.Z_TARGET)
    anchor_family_best_pair = spectrum_branch.best_ratio_pair_fast(anchor_family_rows)
    pair_family_best_pair = spectrum_branch.best_ratio_pair_fast(pair_family_rows)

    anchor_family_internal_pair_available = bool(
        anchor_family_best_pair and anchor_family_best_pair["passes_threshold"]
    )
    anchor_family_absolute_anchor_support_available = bool(
        anchor_family_best_w
        and anchor_family_best_w["passes_threshold"]
        and anchor_family_best_z
        and anchor_family_best_z["passes_threshold"]
    )
    pair_family_absolute_anchor_support_available = bool(
        pair_family_best_w
        and pair_family_best_w["passes_threshold"]
        and pair_family_best_z
        and pair_family_best_z["passes_threshold"]
    )
    ratio_compatible_anchor_family_exists = anchor_family_internal_pair_available
    bridge_candidate_present = ratio_compatible_anchor_family_exists

    if bridge_candidate_present and not anchor_family_absolute_anchor_support_available:
        selected_residual_route = (
            "trial3_two_component_ratio_compatible_anchor_family_absolute_anchor_support_identification"
        )
        missing_v2_artifact = "trial3_two_component_ratio_compatible_anchor_family_absolute_anchor_support_pack"
        next_open_question = (
            "ratio-compatible anchor family exists but still lacks absolute W/Z anchor support under the current canon"
        )
    elif not bridge_candidate_present:
        selected_residual_route = "trial3_two_component_anchor_pair_family_bridge_candidate_identification"
        missing_v2_artifact = "trial3_two_component_anchor_pair_family_bridge_candidate_pack"
        next_open_question = "no ratio-compatible family bridge candidate is visible in the current two-component table"
    else:
        selected_residual_route = None
        missing_v2_artifact = None
        next_open_question = "two-component family bridge closes under the current canon"

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
        "mass_origin_v2_trial3_two_component_distinct_w_z_anchor_split_inventory_json": rel(ANCHOR_SPLIT_INVENTORY),
        "mass_origin_v2_trial3_two_component_distinct_w_z_anchor_split_audit_json": rel(ANCHOR_SPLIT_AUDIT),
        "mass_origin_v2_trial3_two_component_declaration_second_gate_json": rel(ANCHOR_SPLIT_GATE),
        "mass_origin_v2_trial3_two_component_paper_sync_trial4_disposition_twenty_seventh_refresh_json": rel(ANCHOR_SPLIT_DISPOSITION),
        "mass_origin_v2_trial3_two_component_spectrum_branch_py": rel(SPECTRUM_BRANCH),
        "mass_origin_v2_t3_post_ell18_amplitude_branch_py": rel(HELPER_BRANCH),
        "mass_origin_vector_qball_full_coupled_solver_branch_py": rel(FULL_BRANCH),
    }

    source_inventory = payload(
        "8.7.56.339",
        "Trial-3 two-component anchor/pair family-bridge source inventory",
        common_inputs,
        "Collect the collapsed anchor family, the near-pass pair family, the localized ell=0..5 evidence, and the remaining weak-sector failures into one pack before the family-bridge audit.",
        {
            "inventory_rule": "the family-bridge audit starts only after the distinct-anchor split audit is already frozen",
            "bridge_focus": "the residual is whether one family already carries the ratio/anchor bridge, not whether the absolute ceiling is high enough in general",
        },
        [
            row("trial3_two_component_anchor_pair_family_bridge_source_inventory_complete", "pass", "Trial-3 two-component anchor/pair family-bridge source inventory complete", 1, "The family-bridge source pack is frozen."),
            row("trial3_two_component_collapsed_anchor_family_present", "pass" if anchor_family_rows else "reject", "collapsed anchor family present", len(anchor_family_rows), "The bridge audit must start from the actual collapsed anchor family pinned by the W/Z neighborhoods."),
            row("trial3_two_component_near_pass_pair_family_present", "pass" if pair_family_rows else "reject", "near-pass pair family present", len(pair_family_rows), "The bridge audit must also include the family that carries the near-pass ratio evidence."),
            row("trial3_two_component_localized_low_ell_pack_present", "pass" if localized_rows else "reject", "localized low-ell pack present", len(localized_rows), "The family-bridge audit is valid only if the low-ell localized two-component table is already frozen."),
            row("trial3_two_component_remaining_weak_failure_pack_present", "pass", "remaining weak-sector failure pack present", 1, "The audit keeps the unresolved W/Z absolute-anchor and weak-coupling failures in the same pack."),
        ],
        {
            "localized_solution_count_total": len(localized_rows),
            "localized_ell_values": spectrum_metrics["summary"]["localized_ell_values"],
            "anchor_family_or_none": anchor_family,
            "pair_family_or_none": pair_family,
            "anchor_family_row_count": len(anchor_family_rows),
            "pair_family_row_count": len(pair_family_rows),
            "best_w_row_or_none": best_w,
            "best_z_row_or_none": best_z,
            "best_pair_or_none": best_pair,
            "next_required_route": "trial3_two_component_anchor_pair_family_bridge_audit",
        },
        {
            "overall_status": "trial3_two_component_anchor_pair_family_bridge_inventory_frozen",
            "advance_to_8_7_56_340": True,
            "next_required_artifacts": ["trial3_two_component_anchor_pair_family_bridge_audit"],
        },
        {
            "status_next_step_line": hit(status_text, "current official next step は `8.7.56.339`"),
            "roadmap_branch_line": hit(roadmap_text, "`8.7.56.339-.342` 試練3 two-component anchor/pair family-bridge residual branch"),
            "prior_open_question_line": hit(status_text, "`trial3_two_component_anchor_pair_family_bridge_identification`"),
            "spectrum_summary": spectrum_metrics["summary"],
            "anchor_split_summary": anchor_split_audit["summary"],
            "wz_comparison_summary": wz_metrics["summary"],
        },
    )

    audit = payload(
        "8.7.56.340",
        "Trial-3 two-component anchor/pair family-bridge audit",
        common_inputs,
        "Audit whether the current two-component exact-family table already contains a bridge candidate between the collapsed anchor family and the near-pass pair family, or whether the blocker has already collapsed to a more local absolute-anchor support problem.",
        {
            "anchor_family_rule": "if the collapsed anchor family itself already yields a near-pass W/Z pair, the bridge generality collapses to whatever still prevents that family from closing the absolute anchors",
            "pair_family_rule": "if the near-pass pair family cannot host the absolute W/Z anchors, it is no longer the mainline bridge candidate under the current canon",
        },
        [
            row("trial3_two_component_anchor_pair_family_bridge_audit_complete", "pass", "Trial-3 two-component anchor/pair family-bridge audit complete", 1, "The family-bridge audit is frozen."),
            row("trial3_two_component_ratio_compatible_anchor_family_exists", "pass" if ratio_compatible_anchor_family_exists else "reject", "ratio-compatible anchor family exists", 1 if ratio_compatible_anchor_family_exists else 0, "The collapsed anchor family itself becomes the bridge candidate if it already reproduces the W/Z pair shape."),
            row("trial3_two_component_anchor_family_absolute_anchor_support_available", "pass" if anchor_family_absolute_anchor_support_available else "reject", "anchor family absolute W/Z anchor support available", 1 if anchor_family_absolute_anchor_support_available else 0, "The anchor family closes the branch only if its best W and Z anchors also pass the absolute-target thresholds."),
            row("trial3_two_component_pair_family_absolute_anchor_support_available", "pass" if pair_family_absolute_anchor_support_available else "reject", "pair family absolute W/Z anchor support available", 1 if pair_family_absolute_anchor_support_available else 0, "If the pair family cannot host the absolute anchors, it does not supply the missing bridge under the current canon."),
            row("trial3_two_component_bridge_candidate_present", "pass" if bridge_candidate_present else "reject", "bridge candidate present in current canon", 1 if bridge_candidate_present else 0, "The question is whether a concrete bridge candidate already exists, not whether the ceiling is generically lifted."),
        ],
        {
            "anchor_family_or_none": anchor_family,
            "pair_family_or_none": pair_family,
            "anchor_family_mode_key_or_none": family_mode_key(anchor_family),
            "pair_family_mode_key_or_none": family_mode_key(pair_family),
            "anchor_family_mode_summary_or_none": mode_summary.get(family_mode_key(anchor_family) or ""),
            "pair_family_mode_summary_or_none": mode_summary.get(family_mode_key(pair_family) or ""),
            "anchor_family_best_w_row_or_none": anchor_family_best_w,
            "anchor_family_best_z_row_or_none": anchor_family_best_z,
            "anchor_family_best_pair_or_none": anchor_family_best_pair,
            "pair_family_best_w_row_or_none": pair_family_best_w,
            "pair_family_best_z_row_or_none": pair_family_best_z,
            "pair_family_best_pair_or_none": pair_family_best_pair,
            "ratio_compatible_anchor_family_exists": ratio_compatible_anchor_family_exists,
            "anchor_family_absolute_anchor_support_available": anchor_family_absolute_anchor_support_available,
            "pair_family_absolute_anchor_support_available": pair_family_absolute_anchor_support_available,
            "bridge_candidate_present": bridge_candidate_present,
            "next_required_route": "trial3_two_component_declaration_third_gate",
        },
        {
            "overall_status": "trial3_two_component_anchor_pair_family_bridge_audited",
            "advance_to_8_7_56_341": True,
            "next_required_artifacts": ["trial3_two_component_declaration_third_gate"],
        },
        {
            "advice_two_component_line": hit(advice_text, "two-component"),
            "spectrum_branch_pair_line": hit(spectrum_branch_text, "def best_ratio_pair_fast(rows: list[dict], top_count: int = 1500) -> dict | None:"),
            "helper_closest_state_line": hit(helper_text, "def closest_state(rows: list[dict], target_value: float) -> dict | None:"),
            "anchor_split_gate_summary": anchor_split_gate["summary"],
            "anchor_split_disposition_summary": anchor_split_disposition["summary"],
            "sector_summary_sample": sample(
                [{"ell": key, **value} for key, value in sector_summary.items()],
                8,
            ),
        },
    )

    if bridge_candidate_present and not anchor_family_absolute_anchor_support_available:
        branch_closeable = False
    elif bridge_candidate_present and anchor_family_absolute_anchor_support_available:
        branch_closeable = True
    else:
        branch_closeable = False

    declaration = payload(
        "8.7.56.341",
        "Trial-3 two-component declaration third gate",
        common_inputs,
        "Freeze whether the family-bridge audit already closes the two-component route or whether the honest blocker has collapsed to the ratio-compatible anchor family's missing absolute-anchor support.",
        {
            "closeout_rule": "Trial-3 closes only if the same current-canon family supports both the W/Z pair shape and the absolute W/Z anchors",
            "residual_rule": "if the anchor family already carries the pair shape but still misses the absolute anchors, the next blocker is local absolute-anchor support, not bridge generality",
        },
        [
            row("trial3_two_component_declaration_third_gate_complete", "pass", "Trial-3 two-component declaration third gate complete", 1, "The third declaration gate is frozen."),
            row("trial3_two_component_branch_closeable_third_gate", "pass" if branch_closeable else "reject", "two-component branch closeable after family-bridge audit", 1 if branch_closeable else 0, "The branch closes only if the current bridge candidate also passes the absolute W/Z anchors."),
            row("trial3_two_component_residual_route_required_third_gate", "reject" if branch_closeable else "pass", "two-component residual route still required after family-bridge audit", 0 if branch_closeable else 1, "A further residual route is required while the ratio-compatible anchor family still misses the absolute anchors."),
        ],
        {
            "trial3_current_branch_closeable": branch_closeable,
            "selected_residual_route": selected_residual_route,
            "missing_v2_artifact": missing_v2_artifact,
            "recommended_next_route_or_none": None if branch_closeable else NEXT_ROUTE,
            "next_open_question": next_open_question,
        },
        {
            "overall_status": "trial3_two_component_declaration_third_gate_frozen",
            "trial3_branch_closeable": branch_closeable,
            "advance_to_8_7_56_342": True,
            "next_required_artifacts": [] if branch_closeable else [NEXT_ROUTE],
        },
        {
            "audit_summary": audit["summary"],
            "prior_gate_summary": anchor_split_gate["summary"],
            "current_ai_context_step": ai_context["current_step"],
        },
    )

    disposition = payload(
        "8.7.56.342",
        "Trial-2 paper-side sync / Trial-4 disposition twenty-eighth refresh",
        common_inputs,
        "Refresh the reserve/deferred ordering after the family-bridge audit collapses to the ratio-compatible anchor family's missing absolute-anchor support.",
        {
            "trial2_rule": "Trial-2 paper-side sync stays unlocked reserve retained while the two-component Trial-3 route still has an honest current-canon residual path",
            "trial4_rule": "Trial-4 remains deferred while the two-component Trial-3 route is still scientifically live",
        },
        [
            row("trial3_two_component_trial2_trial4_twenty_eighth_refresh_complete", "pass", "Trial-2 paper-side sync / Trial-4 disposition twenty-eighth refresh complete", 1, "The reserve/deferred ordering is refreshed after the family-bridge audit."),
            row("trial3_two_component_trial2_reserve_retained_twenty_eighth_refresh", "pass", "Trial-2 paper-side sync reserve retained", 1, "Trial-2 paper sync remains reserve work while the two-component route still has an honest residual path."),
            row("trial3_two_component_trial4_deferred_retained_twenty_eighth_refresh", "pass", "Trial-4 deferred retained", 1, "Trial-4 stays deferred while the two-component route still has a current-canon residual path."),
        ],
        {
            "selected_residual_route": selected_residual_route,
            "missing_v2_artifact": missing_v2_artifact,
            "trial2_paper_side_sync_state": TRIAL2_RESERVE_STATE,
            "trial4_deferred": True,
            "recommended_next_route_or_none": None if branch_closeable else NEXT_ROUTE,
        },
        {
            "overall_status": "trial3_two_component_trial2_trial4_twenty_eighth_refresh_frozen",
            "trial3_branch_closeable": branch_closeable,
            "advance_to_next_branch": not branch_closeable,
            "next_required_artifacts": [] if branch_closeable else [NEXT_ROUTE],
        },
        {
            "declaration_summary": declaration["summary"],
            "anchor_split_disposition_summary": anchor_split_disposition["summary"],
            "wz_comparison_summary": wz_metrics["summary"],
        },
    )

    write_artifact("mass_origin_v2_trial3_two_component_anchor_pair_family_bridge_source_inventory", source_inventory)
    write_artifact("mass_origin_v2_trial3_two_component_anchor_pair_family_bridge_audit", audit)
    write_artifact("mass_origin_v2_trial3_two_component_declaration_third_gate", declaration)
    write_artifact("mass_origin_v2_trial3_two_component_paper_sync_trial4_disposition_twenty_eighth_refresh", disposition)

    print("[done] Trial-3 two-component family-bridge artifacts written:")
    print(" - mass_origin_v2_trial3_two_component_anchor_pair_family_bridge_source_inventory_metrics.json")
    print(" - mass_origin_v2_trial3_two_component_anchor_pair_family_bridge_audit_metrics.json")
    print(" - mass_origin_v2_trial3_two_component_declaration_third_gate_metrics.json")
    print(" - mass_origin_v2_trial3_two_component_paper_sync_trial4_disposition_twenty_eighth_refresh_metrics.json")


# 関数: CLI から branch を実行する。

def run_cli() -> None:
    """CLI entry point for the Trial-3 two-component family-bridge branch."""
    main()


if __name__ == "__main__":
    run_cli()
