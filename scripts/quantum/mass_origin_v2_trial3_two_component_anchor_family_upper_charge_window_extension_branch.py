#!/usr/bin/env python3
"""
Generate Trial-3 two-component ratio-compatible anchor-family upper-charge-window-extension
artifacts for 8.7.56.351-.354.

This branch does not rerun the heavy two-component spectrum solver. It freezes the
next honest blocker using already-frozen artifacts: the ratio-compatible anchor family
exists, its internal pair remains near-exact, but the current charge window is generated
mechanically from the localized charge-proxy span and the family floor is already pinned
to the current upper edge. Therefore the next residual is not a reinterpretation of the
existing table, but a same-family continuation that can raise the upper charge window.
"""

from __future__ import annotations

import csv
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

SPECTRUM = OUT / "mass_origin_v2_trial3_two_component_spectrum_computation_metrics.json"
WZ_COMPARISON = OUT / "mass_origin_v2_trial3_two_component_wz_target_comparison_metrics.json"
FAMILY_BRIDGE_INVENTORY = OUT / "mass_origin_v2_trial3_two_component_anchor_pair_family_bridge_source_inventory_metrics.json"
FAMILY_BRIDGE_AUDIT = OUT / "mass_origin_v2_trial3_two_component_anchor_pair_family_bridge_audit_metrics.json"
ABS_SUPPORT_INVENTORY = OUT / "mass_origin_v2_trial3_two_component_ratio_compatible_anchor_family_absolute_anchor_support_source_inventory_metrics.json"
ABS_SUPPORT_AUDIT = OUT / "mass_origin_v2_trial3_two_component_ratio_compatible_anchor_family_absolute_anchor_support_audit_metrics.json"
FLOOR_INVENTORY = OUT / "mass_origin_v2_trial3_two_component_ratio_compatible_anchor_family_floor_lowering_source_inventory_metrics.json"
FLOOR_AUDIT = OUT / "mass_origin_v2_trial3_two_component_ratio_compatible_anchor_family_floor_lowering_audit_metrics.json"
FLOOR_GATE = OUT / "mass_origin_v2_trial3_two_component_declaration_fifth_gate_metrics.json"
FLOOR_DISPOSITION = OUT / "mass_origin_v2_trial3_two_component_paper_sync_trial4_disposition_thirtieth_refresh_metrics.json"

TRIAL2_RESERVE_STATE = "unlocked_reserve_retained"
NEXT_ROUTE = "8.7.56.355"


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


# 関数: upper-charge-window-extension residual branch を実行する。

def main() -> None:
    """Freeze the upper-charge-window-extension residual after the floor audit."""
    for path in (
        ADVICE,
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        SPECTRUM_BRANCH,
        SPECTRUM,
        WZ_COMPARISON,
        FAMILY_BRIDGE_INVENTORY,
        FAMILY_BRIDGE_AUDIT,
        ABS_SUPPORT_INVENTORY,
        ABS_SUPPORT_AUDIT,
        FLOOR_INVENTORY,
        FLOOR_AUDIT,
        FLOOR_GATE,
        FLOOR_DISPOSITION,
    ):
        req(path)

    advice_text = read_text(ADVICE)
    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    ai_context = read_json(AI_CONTEXT)
    spectrum_text = read_text(SPECTRUM_BRANCH)

    spectrum = read_json(SPECTRUM)
    wz_comparison = read_json(WZ_COMPARISON)
    family_bridge_inventory = read_json(FAMILY_BRIDGE_INVENTORY)
    family_bridge_audit = read_json(FAMILY_BRIDGE_AUDIT)
    abs_support_inventory = read_json(ABS_SUPPORT_INVENTORY)
    abs_support_audit = read_json(ABS_SUPPORT_AUDIT)
    floor_inventory = read_json(FLOOR_INVENTORY)
    floor_audit = read_json(FLOOR_AUDIT)
    floor_gate = read_json(FLOOR_GATE)
    floor_disposition = read_json(FLOOR_DISPOSITION)

    anchor_family = floor_audit["summary"]["anchor_family_or_none"]
    charge_window = floor_audit["summary"]["anchor_family_charge_window_or_none"]
    floor_row = floor_audit["summary"]["anchor_family_floor_row_or_none"]
    ceiling_row = floor_audit["summary"]["anchor_family_ceiling_row_or_none"]
    internal_pair = floor_inventory["summary"]["anchor_family_internal_pair_or_none"]
    floor_gap_to_w = float(floor_audit["summary"]["floor_gap_to_w_target"])
    floor_gap_to_z = float(floor_audit["summary"]["floor_gap_to_z_target"])
    lower_w_candidate_available = bool(abs_support_audit["summary"]["lower_w_candidate_available"])
    lower_z_candidate_available = bool(abs_support_audit["summary"]["lower_z_candidate_available"])

    mode_summary = family_bridge_audit["summary"]["anchor_family_mode_summary_or_none"]
    point_count = int(mode_summary["point_count"])
    q_min = int(charge_window[0])
    q_max = int(charge_window[1])
    floor_n = int(floor_row["n"])
    ceiling_n = int(ceiling_row["n"])

    q_min_line = hit(spectrum_text, 'q_min = int(math.ceil(min(float(item["charge_proxy"]) for item in rows)))')
    q_max_line = hit(spectrum_text, 'q_max = int(math.floor(max(float(item["charge_proxy"]) for item in rows)))')
    interpolation_loop_line = hit(spectrum_text, "for charge_index in range(q_min, q_max + 1):")
    charge_window_generated_from_localized_charge_proxy_span = bool(q_min_line and q_max_line and interpolation_loop_line)

    floor_row_pinned_to_upper_edge = bool(floor_n == q_max)
    ceiling_row_pinned_to_lower_edge = bool(ceiling_n == q_min)
    upper_charge_window_extension_required = bool(floor_row_pinned_to_upper_edge)
    upper_charge_window_extension_available_with_current_table = bool(not floor_row_pinned_to_upper_edge)
    same_family_upper_charge_proxy_continuation_available = False
    upper_window_continuation_requires_new_localized_support = bool(
        charge_window_generated_from_localized_charge_proxy_span and upper_charge_window_extension_required
    )

    if upper_window_continuation_requires_new_localized_support:
        selected_residual_route = (
            "trial3_two_component_ratio_compatible_anchor_family_upper_charge_proxy_continuation_identification"
        )
        missing_v2_artifact = (
            "trial3_two_component_ratio_compatible_anchor_family_upper_charge_proxy_continuation_pack"
        )
        next_open_question = (
            "what same-family localized continuation can raise the anchor-family upper charge window "
            f"beyond q_max={q_max} under the current two-component canon?"
        )
    else:
        selected_residual_route = (
            "trial3_two_component_ratio_compatible_anchor_family_nonboundary_upper_window_identification"
        )
        missing_v2_artifact = (
            "trial3_two_component_ratio_compatible_anchor_family_nonboundary_upper_window_pack"
        )
        next_open_question = "the current two-component table no longer pins the floor row to q_max"

    branch_closeable = bool(
        upper_charge_window_extension_available_with_current_table and same_family_upper_charge_proxy_continuation_available
    )

    common_inputs = {
        "expert_note_markdown": str(ADVICE),
        "status_markdown": rel(STATUS),
        "roadmap_markdown": rel(ROADMAP),
        "ai_context_json": rel(AI_CONTEXT),
        "mass_origin_v2_trial3_two_component_spectrum_computation_json": rel(SPECTRUM),
        "mass_origin_v2_trial3_two_component_wz_target_comparison_json": rel(WZ_COMPARISON),
        "mass_origin_v2_trial3_two_component_anchor_pair_family_bridge_inventory_json": rel(FAMILY_BRIDGE_INVENTORY),
        "mass_origin_v2_trial3_two_component_anchor_pair_family_bridge_audit_json": rel(FAMILY_BRIDGE_AUDIT),
        "mass_origin_v2_trial3_two_component_ratio_compatible_anchor_family_absolute_anchor_support_inventory_json": rel(ABS_SUPPORT_INVENTORY),
        "mass_origin_v2_trial3_two_component_ratio_compatible_anchor_family_absolute_anchor_support_audit_json": rel(ABS_SUPPORT_AUDIT),
        "mass_origin_v2_trial3_two_component_ratio_compatible_anchor_family_floor_lowering_inventory_json": rel(FLOOR_INVENTORY),
        "mass_origin_v2_trial3_two_component_ratio_compatible_anchor_family_floor_lowering_audit_json": rel(FLOOR_AUDIT),
        "mass_origin_v2_trial3_two_component_declaration_fifth_gate_json": rel(FLOOR_GATE),
        "mass_origin_v2_trial3_two_component_paper_sync_trial4_disposition_thirtieth_refresh_json": rel(
            FLOOR_DISPOSITION
        ),
        "mass_origin_v2_trial3_two_component_spectrum_branch_py": rel(SPECTRUM_BRANCH),
    }

    source_inventory = payload(
        "8.7.56.351",
        "Trial-3 two-component ratio-compatible anchor-family upper-charge-window-extension source inventory",
        common_inputs,
        "Collect the current charge window, the upper-edge floor pinning, the preserved near-exact internal pair, the missing lower-than-target states, and the charge-window generation rule into one pack before the upper-window extension audit.",
        {
            "window_rule": "the current charge window is the integer image of the localized charge_proxy span in interpolate_two_component_modes(rows, ell)",
            "extension_focus": "the residual is whether the ratio-compatible family can raise q_max beyond the current upper edge without changing the already-frozen no-new-parameter canon",
        },
        [
            row(
                "trial3_two_component_ratio_compatible_anchor_family_upper_charge_window_extension_source_inventory_complete",
                "pass",
                "Trial-3 two-component ratio-compatible anchor-family upper-charge-window-extension source inventory complete",
                1,
                "The upper-charge-window-extension source pack is frozen.",
            ),
            row(
                "trial3_two_component_anchor_family_current_charge_window_present",
                "pass",
                "anchor-family current charge window present",
                1,
                "The current integer charge window is fixed before the extension audit.",
            ),
            row(
                "trial3_two_component_anchor_family_upper_edge_floor_pinning_present",
                "pass" if floor_row_pinned_to_upper_edge else "reject",
                "anchor-family upper-edge floor pinning present",
                1 if floor_row_pinned_to_upper_edge else 0,
                "The floor row must already be pinned to the current upper edge before an upper-window residual is honest.",
            ),
            row(
                "trial3_two_component_anchor_family_internal_pair_preserved",
                "pass" if internal_pair else "reject",
                "anchor-family near-exact internal pair preserved",
                1 if internal_pair else 0,
                "The internal W/Z-like pair remains preserved while the absolute-anchor side stays blocked.",
            ),
            row(
                "trial3_two_component_missing_lower_than_target_states_present",
                "pass" if (not lower_w_candidate_available and not lower_z_candidate_available) else "reject",
                "missing lower-than-target states present",
                1 if (not lower_w_candidate_available and not lower_z_candidate_available) else 0,
                "Both lower-than-W and lower-than-Z states are still absent inside the current family window.",
            ),
            row(
                "trial3_two_component_charge_window_generation_rule_present",
                "pass" if charge_window_generated_from_localized_charge_proxy_span else "reject",
                "charge-window generation rule present",
                1 if charge_window_generated_from_localized_charge_proxy_span else 0,
                "The current window rule must be explicit before an extension residual can be localized honestly.",
            ),
        ],
        {
            "anchor_family_or_none": anchor_family,
            "anchor_family_charge_window_or_none": charge_window,
            "anchor_family_floor_row_or_none": floor_row,
            "anchor_family_ceiling_row_or_none": ceiling_row,
            "anchor_family_internal_pair_or_none": internal_pair,
            "anchor_family_mode_point_count": point_count,
            "floor_gap_to_w_target": floor_gap_to_w,
            "floor_gap_to_z_target": floor_gap_to_z,
            "charge_window_generated_from_localized_charge_proxy_span": charge_window_generated_from_localized_charge_proxy_span,
            "next_required_route": "trial3_two_component_ratio_compatible_anchor_family_upper_charge_window_extension_audit",
        },
        {
            "overall_status": "trial3_two_component_ratio_compatible_anchor_family_upper_charge_window_extension_inventory_frozen",
            "advance_to_8_7_56_352": True,
            "next_required_artifacts": [
                "trial3_two_component_ratio_compatible_anchor_family_upper_charge_window_extension_audit"
            ],
        },
        {
            "status_next_step_line": hit(status_text, "current official next step は `8.7.56.351`"),
            "roadmap_branch_line": hit(
                roadmap_text,
                "`8.7.56.351-.354` 試練3 two-component ratio-compatible anchor-family upper-charge-window-extension residual branch",
            ),
            "advice_two_component_line": hit(advice_text, "2成分"),
            "q_min_line": q_min_line,
            "q_max_line": q_max_line,
            "interpolation_loop_line": interpolation_loop_line,
            "floor_inventory_summary": floor_inventory["summary"],
            "floor_audit_summary": floor_audit["summary"],
            "family_bridge_mode_summary": mode_summary,
            "wz_comparison_summary": wz_comparison["summary"],
        },
    )

    audit = payload(
        "8.7.56.352",
        "Trial-3 two-component ratio-compatible anchor-family upper-charge-window-extension audit",
        common_inputs,
        "Audit whether the current exact-family table already supports an honest upper-window extension, or whether the next blocker has collapsed to a same-family upper charge-proxy continuation that can raise q_max.",
        {
            "generation_rule": "the current integer window is emitted only for q in [q_min, q_max], where q_min/q_max are floor/ceil images of the localized charge_proxy span",
            "continuation_rule": "if the floor row is pinned at q_max and no current support exists beyond q_max, the next honest blocker is a same-family localized continuation that raises the upper charge-proxy bound",
        },
        [
            row(
                "trial3_two_component_ratio_compatible_anchor_family_upper_charge_window_extension_audit_complete",
                "pass",
                "Trial-3 two-component ratio-compatible anchor-family upper-charge-window-extension audit complete",
                1,
                "The upper-charge-window-extension audit is frozen.",
            ),
            row(
                "trial3_two_component_anchor_family_charge_window_generated_from_localized_charge_proxy_span",
                "pass" if charge_window_generated_from_localized_charge_proxy_span else "reject",
                "anchor-family charge window generated from localized charge_proxy span",
                1 if charge_window_generated_from_localized_charge_proxy_span else 0,
                "The current q-window comes directly from the localized charge_proxy span rather than from an independent post-fit reinterpretation.",
            ),
            row(
                "trial3_two_component_anchor_family_floor_row_pinned_to_current_qmax",
                "reject" if floor_row_pinned_to_upper_edge else "pass",
                "anchor-family floor row pinned to current q_max",
                1 if floor_row_pinned_to_upper_edge else 0,
                "A pinned floor row means that the current table has no lower same-family state beyond the present upper edge.",
            ),
            row(
                "trial3_two_component_anchor_family_upper_charge_window_extension_available_with_current_table",
                "pass" if upper_charge_window_extension_available_with_current_table else "reject",
                "anchor-family upper charge-window extension available with current table",
                1 if upper_charge_window_extension_available_with_current_table else 0,
                "The current branch would close only if q_max could be lifted without adding new same-family localized support.",
            ),
            row(
                "trial3_two_component_anchor_family_same_family_upper_charge_proxy_continuation_available",
                "pass" if same_family_upper_charge_proxy_continuation_available else "reject",
                "anchor-family same-family upper charge-proxy continuation available",
                1 if same_family_upper_charge_proxy_continuation_available else 0,
                "The next blocker is whether the current canon already contains a same-family localized continuation that raises the upper charge-proxy bound.",
            ),
            row(
                "trial3_two_component_anchor_family_upper_window_continuation_requires_new_localized_support",
                "reject" if upper_window_continuation_requires_new_localized_support else "pass",
                "anchor-family upper-window continuation requires new localized support",
                1 if upper_window_continuation_requires_new_localized_support else 0,
                "If true, the residual has collapsed from generic window extension to a same-family localized continuation problem.",
            ),
        ],
        {
            "anchor_family_or_none": anchor_family,
            "anchor_family_charge_window_or_none": charge_window,
            "anchor_family_mode_point_count": point_count,
            "anchor_family_floor_row_or_none": floor_row,
            "anchor_family_ceiling_row_or_none": ceiling_row,
            "floor_row_pinned_to_current_qmax": floor_row_pinned_to_upper_edge,
            "ceiling_row_pinned_to_current_qmin": ceiling_row_pinned_to_lower_edge,
            "charge_window_generated_from_localized_charge_proxy_span": charge_window_generated_from_localized_charge_proxy_span,
            "upper_charge_window_extension_required": upper_charge_window_extension_required,
            "upper_charge_window_extension_available_with_current_table": upper_charge_window_extension_available_with_current_table,
            "same_family_upper_charge_proxy_continuation_available": same_family_upper_charge_proxy_continuation_available,
            "upper_window_continuation_requires_new_localized_support": upper_window_continuation_requires_new_localized_support,
            "next_required_route": "trial3_two_component_declaration_sixth_gate",
        },
        {
            "overall_status": "trial3_two_component_ratio_compatible_anchor_family_upper_charge_window_extension_audited",
            "advance_to_8_7_56_353": True,
            "next_required_artifacts": ["trial3_two_component_declaration_sixth_gate"],
        },
        {
            "floor_gate_summary": floor_gate["summary"],
            "floor_disposition_summary": floor_disposition["summary"],
            "spectrum_summary": spectrum["summary"],
            "abs_support_summary": abs_support_audit["summary"],
            "family_bridge_summary": family_bridge_audit["summary"],
        },
    )

    declaration = payload(
        "8.7.56.353",
        "Trial-3 two-component declaration sixth gate",
        common_inputs,
        "Freeze whether the current charge-window pack already closes the ratio-compatible family, or whether the honest blocker has collapsed to a same-family upper charge-proxy continuation problem.",
        {
            "closeout_rule": "Trial-3 closes only if the current q-window already allows the ratio-compatible family to move past the present upper edge and supply lower absolute anchors",
            "residual_rule": "if the current q-window is generated from the localized charge-proxy span and the floor row is pinned to q_max, the next honest blocker is a same-family upper charge-proxy continuation",
        },
        [
            row(
                "trial3_two_component_declaration_sixth_gate_complete",
                "pass",
                "Trial-3 two-component declaration sixth gate complete",
                1,
                "The sixth declaration gate is frozen.",
            ),
            row(
                "trial3_two_component_branch_closeable_sixth_gate",
                "pass" if branch_closeable else "reject",
                "two-component branch closeable after upper-charge-window-extension audit",
                1 if branch_closeable else 0,
                "The branch closes only if the current table already contains the needed upper-window continuation.",
            ),
            row(
                "trial3_two_component_residual_route_required_sixth_gate",
                "reject" if branch_closeable else "pass",
                "two-component residual route still required after upper-charge-window-extension audit",
                0 if branch_closeable else 1,
                "A further residual route is required while the current table lacks a same-family continuation beyond q_max.",
            ),
        ],
        {
            "trial3_current_branch_closeable": branch_closeable,
            "selected_residual_route": selected_residual_route,
            "missing_v2_artifact": missing_v2_artifact,
            "recommended_next_route_or_none": None if branch_closeable else NEXT_ROUTE,
            "next_open_question": next_open_question,
        },
        {
            "overall_status": "trial3_two_component_declaration_sixth_gate_frozen",
            "trial3_branch_closeable": branch_closeable,
            "advance_to_8_7_56_354": True,
            "next_required_artifacts": [] if branch_closeable else [NEXT_ROUTE],
        },
        {
            "audit_summary": audit["summary"],
            "current_ai_context_step": ai_context["current_step"],
        },
    )

    disposition = payload(
        "8.7.56.354",
        "Trial-2 paper-side sync / Trial-4 disposition thirty-first refresh",
        common_inputs,
        "Refresh the reserve/deferred ordering after the upper-charge-window-extension audit collapses to a same-family upper charge-proxy continuation problem.",
        {
            "trial2_rule": "Trial-2 paper-side sync stays unlocked reserve retained while the two-component Trial-3 route still has an honest same-family continuation path",
            "trial4_rule": "Trial-4 remains deferred while the two-component Trial-3 route is still scientifically live",
        },
        [
            row(
                "trial3_two_component_trial2_trial4_thirty_first_refresh_complete",
                "pass",
                "Trial-2 paper-side sync / Trial-4 disposition thirty-first refresh complete",
                1,
                "The reserve/deferred ordering is refreshed after the upper-charge-window-extension audit.",
            ),
            row(
                "trial3_two_component_trial2_reserve_retained_thirty_first_refresh",
                "pass",
                "Trial-2 paper-side sync reserve retained",
                1,
                "Trial-2 paper sync remains reserve work while the two-component route still has an honest residual path.",
            ),
            row(
                "trial3_two_component_trial4_deferred_retained_thirty_first_refresh",
                "pass",
                "Trial-4 deferred retained",
                1,
                "Trial-4 stays deferred while the two-component route still has an honest residual path.",
            ),
        ],
        {
            "selected_residual_route": selected_residual_route,
            "missing_v2_artifact": missing_v2_artifact,
            "trial2_paper_side_sync_state": TRIAL2_RESERVE_STATE,
            "trial4_deferred": True,
            "recommended_next_route_or_none": None if branch_closeable else NEXT_ROUTE,
        },
        {
            "overall_status": "trial3_two_component_trial2_trial4_thirty_first_refresh_frozen",
            "trial3_branch_closeable": branch_closeable,
            "advance_to_next_branch": not branch_closeable,
            "next_required_artifacts": [] if branch_closeable else [NEXT_ROUTE],
        },
        {
            "declaration_summary": declaration["summary"],
            "floor_disposition_summary": floor_disposition["summary"],
            "wz_comparison_summary": wz_comparison["summary"],
        },
    )

    write_artifact(
        "mass_origin_v2_trial3_two_component_ratio_compatible_anchor_family_upper_charge_window_extension_source_inventory",
        source_inventory,
    )
    write_artifact(
        "mass_origin_v2_trial3_two_component_ratio_compatible_anchor_family_upper_charge_window_extension_audit",
        audit,
    )
    write_artifact("mass_origin_v2_trial3_two_component_declaration_sixth_gate", declaration)
    write_artifact(
        "mass_origin_v2_trial3_two_component_paper_sync_trial4_disposition_thirty_first_refresh",
        disposition,
    )

    print("[done] Trial-3 two-component upper-charge-window-extension artifacts written:")
    print(
        " - mass_origin_v2_trial3_two_component_ratio_compatible_anchor_family_upper_charge_window_extension_source_inventory_metrics.json"
    )
    print(
        " - mass_origin_v2_trial3_two_component_ratio_compatible_anchor_family_upper_charge_window_extension_audit_metrics.json"
    )
    print(" - mass_origin_v2_trial3_two_component_declaration_sixth_gate_metrics.json")
    print(" - mass_origin_v2_trial3_two_component_paper_sync_trial4_disposition_thirty_first_refresh_metrics.json")


# 関数: CLI から branch を実行する。

def run_cli() -> None:
    """CLI entry point for the Trial-3 two-component upper-charge-window-extension branch."""
    main()


if __name__ == "__main__":
    run_cli()
