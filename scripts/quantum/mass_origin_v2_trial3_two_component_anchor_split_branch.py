#!/usr/bin/env python3
"""
Generate Trial-3 two-component distinct W/Z anchor-split artifacts for 8.7.56.335-.338.
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
POST_PHOTON_PRESERVATION = OUT / "mass_origin_v2_post_photon_vector_mass_ratio_preservation_audit_metrics.json"
SCALAR_SPECTRUM = OUT / "mass_origin_qball_discrete_mass_spectrum_metrics.json"
VECTOR_SPIN = OUT / "mass_origin_vector_qball_spin_orbit_freeze_audit_metrics.json"

TRIAL2_RESERVE_STATE = "unlocked_reserve_retained"


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


# 関数: target に最も近い候補 rows を返す。

def nearest_candidates(rows: list[dict], target_value: float, count: int = 12) -> list[dict]:
    """Return the top nearest rows to a target mass ratio."""
    ranked = sorted(
        rows,
        key=lambda item: abs(float(item["mass_ratio_to_scalar_base"]) - float(target_value)) / float(target_value),
    )
    candidates = []
    for item in ranked[:count]:
        ratio = float(item["mass_ratio_to_scalar_base"])
        candidates.append(
            {
                "n": int(item["n"]),
                "k": int(item["k"]),
                "ell": int(item["ell"]),
                "s": int(item["s"]),
                "ratio_value": ratio,
                "relative_error": abs(ratio - float(target_value)) / float(target_value),
            }
        )

    return candidates


# 関数: Trial-3 two-component distinct anchor-split branch を実行する。

def main() -> None:
    """Freeze the first distinct-anchor split audit after the two-component pivot."""
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
    full_text = read_text(FULL_BRANCH)

    helper = load_module(HELPER_BRANCH, "trial3_two_component_anchor_helper")
    spectrum_branch = load_module(SPECTRUM_BRANCH, "trial3_two_component_anchor_spectrum")
    pivot = load_module(PIVOT_BRANCH, "trial3_two_component_anchor_pivot")
    numerical = load_module(NUMERICAL_BRANCH, "trial3_two_component_anchor_numerical")
    full = load_module(FULL_BRANCH, "trial3_two_component_anchor_full")

    pivot_route = read_json(PIVOT_ROUTE)
    pivot_ode = read_json(PIVOT_ODE)
    pivot_implementation = read_json(PIVOT_IMPLEMENTATION)
    spectrum_metrics = read_json(SPECTRUM_METRICS)
    wz_metrics = read_json(WZ_COMPARISON)
    declaration_gate = read_json(DECLARATION_GATE)
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

    near_w = nearest_candidates(normalized_vector_rows, spectrum_branch.W_TARGET, 12)
    near_z = nearest_candidates(normalized_vector_rows, spectrum_branch.Z_TARGET, 12)
    near_w_families = sorted({(int(item["k"]), int(item["ell"]), int(item["s"])) for item in near_w})
    near_z_families = sorted({(int(item["k"]), int(item["ell"]), int(item["s"])) for item in near_z})
    anchor_family = family_id(best_w)
    pair_lighter_family = family_id(best_pair["lighter_state"]) if best_pair else None
    pair_heavier_family = family_id(best_pair["heavier_state"]) if best_pair else None
    pair_family_same = bool(pair_lighter_family == pair_heavier_family)
    anchor_collapsed = bool(
        best_w
        and best_z
        and best_w["n"] == best_z["n"]
        and best_w["k"] == best_z["k"]
        and best_w["ell"] == best_z["ell"]
        and best_w["s"] == best_z["s"]
    )
    anchor_family_reused_across_targets = bool(family_id(best_w) == family_id(best_z))
    anchor_pair_family_bridge_available = bool(anchor_family and pair_heavier_family and anchor_family == pair_heavier_family)
    distinct_anchor_split_available = bool(not anchor_collapsed and not anchor_family_reused_across_targets)
    best_pair_near_pass = bool(
        best_pair and float(best_pair["mw_mz_ratio_relative_error"]) <= spectrum_branch.PAIR_NEAR_PASS_THRESHOLD
    )

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
        "mass_origin_v2_trial3_two_component_spectrum_branch_py": rel(SPECTRUM_BRANCH),
        "mass_origin_v2_t3_post_ell18_amplitude_branch_py": rel(HELPER_BRANCH),
        "mass_origin_vector_qball_full_coupled_solver_branch_py": rel(FULL_BRANCH),
    }

    source_inventory = payload(
        "8.7.56.335",
        "Trial-3 two-component distinct W/Z anchor-split source inventory",
        common_inputs,
        "Collect the already-frozen two-component localized sectors, ceiling lift, collapsed best W/Z anchor, and near-pass pair evidence into one pack before the split audit.",
        {
            "inventory_rule": "the anchor-split audit starts from the frozen first-pass two-component spectrum and must not reopen the one-component ceiling branch",
            "split_focus": "the residual is distinct W/Z anchor splitting, not absolute ceiling or one-component same-family retries",
        },
        [
            row("trial3_two_component_distinct_anchor_split_source_inventory_complete", "pass", "Trial-3 two-component distinct anchor-split source inventory complete", 1, "The current two-component split residual pack is frozen."),
            row("trial3_two_component_localized_family_pack_present", "pass" if localized_rows else "reject", "two-component localized family pack present", len(localized_rows), "The split audit must start from an already localized two-component family."),
            row("trial3_two_component_ceiling_lift_pack_present", "pass" if spectrum_metrics["summary"]["ceiling_lifted_vs_single_component"] else "reject", "two-component ceiling lift pack present", 1 if spectrum_metrics["summary"]["ceiling_lifted_vs_single_component"] else 0, "The split audit only makes sense after the absolute ceiling is already lifted."),
            row("trial3_two_component_collapsed_anchor_pack_present", "pass" if anchor_collapsed else "reject", "collapsed best W/Z anchor pack present", 1 if anchor_collapsed else 0, "The current residual specifically begins from the observed W/Z anchor collapse."),
            row("trial3_two_component_near_pass_pair_pack_present", "pass" if best_pair_near_pass else "reject", "near-pass pair pack present", 1 if best_pair_near_pass else 0, "The pair side is already near-pass, so the main question is where the split itself is missing."),
        ],
        {
            "localized_solution_count_total": len(localized_rows),
            "localized_ell_values": spectrum_metrics["summary"]["localized_ell_values"],
            "rebuilt_ceiling_to_electron": spectrum_metrics["summary"]["two_component_rebuilt_ceiling_to_electron"],
            "best_w_row_or_none": best_w,
            "best_z_row_or_none": best_z,
            "best_pair_or_none": best_pair,
            "anchor_collapsed": anchor_collapsed,
            "best_pair_near_pass": best_pair_near_pass,
            "next_required_route": "trial3_two_component_distinct_w_z_anchor_split_audit",
        },
        {
            "overall_status": "trial3_two_component_distinct_anchor_split_inventory_frozen",
            "advance_to_8_7_56_336": True,
            "next_required_artifacts": ["trial3_two_component_distinct_w_z_anchor_split_audit"],
        },
        {
            "advice_step4_line": hit(advice_text, "### Step 4: W/Z target との照合"),
            "status_next_step_line": hit(status_text, "current official next step は `8.7.56.335`"),
            "roadmap_branch_line": hit(roadmap_text, "`8.7.56.335-.338` 試練3 two-component distinct W/Z anchor-split residual branch"),
            "spectrum_branch_run_scan_line": hit(spectrum_branch_text, "def run_two_component_scan("),
            "helper_normalize_line": hit(helper_text, "def normalize_vector_rows(rows: list[dict], scale_factor: float) -> list[dict]:"),
            "full_exact_builder_line": hit(full_text, "def build_exact_ladder("),
            "pivot_route_summary": pivot_route["summary"],
            "spectrum_summary": spectrum_metrics["summary"],
            "wz_comparison_summary": wz_metrics["summary"],
            "declaration_summary": declaration_gate["summary"],
        },
    )

    audit = payload(
        "8.7.56.336",
        "Trial-3 two-component distinct W/Z anchor-split audit",
        common_inputs,
        "Audit whether the two-component exact-family table already contains a distinct W/Z split or whether the best anchors remain trapped inside one collapsed anchor family disconnected from the near-pass pair family.",
        {
            "collapsed_family_rule": "if the nearest W and Z target neighborhoods both live inside the same (k, ell, s) family, distinct anchor splitting is still absent",
            "bridge_rule": "if the near-pass pair lives in a different family from the collapsed anchor family, the next residual is an anchor/pair family bridge rather than an absolute-ceiling search",
        },
        [
            row("trial3_two_component_distinct_anchor_split_audit_complete", "pass", "Trial-3 two-component distinct anchor-split audit complete", 1, "The split audit is frozen."),
            row("trial3_two_component_anchor_collapsed", "reject" if anchor_collapsed else "pass", "best W/Z anchors collapse onto one state", 1 if anchor_collapsed else 0, "A collapsed anchor state means distinct W/Z splitting is not yet available."),
            row("trial3_two_component_anchor_family_reused_across_targets", "reject" if anchor_family_reused_across_targets else "pass", "nearest W/Z neighborhoods reuse the same family signature", 1 if anchor_family_reused_across_targets else 0, "Top W and Z candidates still come from the same (k, ell, s) family."),
            row("trial3_two_component_pair_family_same", "pass" if pair_family_same else "watch", "best pair comes from one common family", 1 if pair_family_same else 0, "The near-pass pair is itself essentially one family, not two independently split anchors."),
            row("trial3_two_component_anchor_pair_family_bridge_available", "pass" if anchor_pair_family_bridge_available else "reject", "collapsed anchor family already bridges to near-pass pair family", 1 if anchor_pair_family_bridge_available else 0, "If this bridge is absent, the next blocker is not ceiling but family-bridge construction."),
            row("trial3_two_component_distinct_anchor_split_available", "pass" if distinct_anchor_split_available else "reject", "distinct W/Z anchor split available", 1 if distinct_anchor_split_available else 0, "The current canon closes only if W and Z separate into distinct anchor states."),
        ],
        {
            "best_w_row_or_none": best_w,
            "best_z_row_or_none": best_z,
            "best_pair_or_none": best_pair,
            "near_w_candidates": near_w,
            "near_z_candidates": near_z,
            "anchor_family_or_none": anchor_family,
            "near_w_family_signatures": near_w_families,
            "near_z_family_signatures": near_z_families,
            "pair_lighter_family_or_none": pair_lighter_family,
            "pair_heavier_family_or_none": pair_heavier_family,
            "anchor_collapsed": anchor_collapsed,
            "anchor_family_reused_across_targets": anchor_family_reused_across_targets,
            "pair_family_same": pair_family_same,
            "anchor_pair_family_bridge_available": anchor_pair_family_bridge_available,
            "distinct_w_z_anchor_split_available": distinct_anchor_split_available,
            "next_required_route": "trial3_two_component_declaration_second_gate",
        },
        {
            "overall_status": "trial3_two_component_distinct_anchor_split_audited",
            "advance_to_8_7_56_337": True,
            "next_required_artifacts": ["trial3_two_component_declaration_second_gate"],
        },
        {
            "sector_summary": sector_summary,
            "mode_summary_sample": sample(
                [{"family": key, **value} for key, value in mode_summary.items()],
                16,
            ),
            "normalized_row_count": len(normalized_vector_rows),
            "spectrum_summary": spectrum_metrics["summary"],
            "wz_comparison_summary": wz_metrics["summary"],
        },
    )

    if distinct_anchor_split_available:
        selected_residual_route = None
        missing_v2_artifact = None
        recommended_next_route = None
    elif not anchor_pair_family_bridge_available:
        selected_residual_route = "trial3_two_component_anchor_pair_family_bridge_identification"
        missing_v2_artifact = "trial3_two_component_anchor_pair_family_bridge_pack"
        recommended_next_route = "8.7.56.339"
    else:
        selected_residual_route = "trial3_two_component_collapsed_anchor_family_separation_identification"
        missing_v2_artifact = "trial3_two_component_collapsed_anchor_family_separation_pack"
        recommended_next_route = "8.7.56.339"

    declaration = payload(
        "8.7.56.337",
        "Trial-3 two-component declaration second gate",
        common_inputs,
        "Freeze whether the first distinct-anchor audit already closes the two-component route or whether the next honest blocker is an anchor/pair family bridge.",
        {
            "closeout_rule": "Trial-3 closes only if distinct W and Z anchors are actually separated under the current two-component canon",
            "residual_rule": "if W/Z anchors remain collapsed and the near-pass pair lives in a different family, the next blocker is the missing anchor/pair family bridge",
        },
        [
            row("trial3_two_component_declaration_second_gate_complete", "pass", "Trial-3 two-component declaration second gate complete", 1, "The second declaration gate is frozen."),
            row("trial3_two_component_branch_closeable_second_gate", "pass" if distinct_anchor_split_available else "reject", "two-component branch closeable after first split audit", 1 if distinct_anchor_split_available else 0, "The branch closes only if distinct W/Z anchors already separate."),
            row("trial3_two_component_residual_route_required_second_gate", "reject" if distinct_anchor_split_available else "pass", "two-component residual route still required after split audit", 0 if distinct_anchor_split_available else 1, "A further residual route is required while the anchor split stays collapsed."),
        ],
        {
            "trial3_current_branch_closeable": distinct_anchor_split_available,
            "selected_residual_route": selected_residual_route,
            "missing_v2_artifact": missing_v2_artifact,
            "recommended_next_route_or_none": recommended_next_route,
        },
        {
            "overall_status": "trial3_two_component_declaration_second_gate_frozen",
            "trial3_branch_closeable": distinct_anchor_split_available,
            "advance_to_8_7_56_338": True,
            "next_required_artifacts": [] if distinct_anchor_split_available else [recommended_next_route],
        },
        {
            "audit_summary": audit["summary"],
            "status_current_step_before_branch": ai_context["current_step"],
            "advice_case_b_line": hit(advice_text, "**Case B: ceiling は上がるが W/Z に届かない**"),
            "advice_case_c_line": hit(advice_text, "**Case C: 2成分でも ceiling が動かない**"),
        },
    )

    disposition = payload(
        "8.7.56.338",
        "Trial-2 paper-side sync / Trial-4 disposition twenty-seventh refresh",
        common_inputs,
        "Refresh the reserve/deferred ordering after the two-component distinct-anchor audit.",
        {
            "trial2_rule": "Trial-2 paper-side sync stays unlocked reserve retained while the two-component Trial-3 route still has an honest current-canon residual path",
            "trial4_rule": "Trial-4 remains deferred while the Trial-3 two-component route is still scientifically live",
        },
        [
            row("trial3_two_component_trial2_trial4_twenty_seventh_refresh_complete", "pass", "Trial-2 paper-side sync / Trial-4 disposition twenty-seventh refresh complete", 1, "The reserve/deferred ordering is refreshed after the split audit."),
            row("trial3_two_component_trial2_reserve_retained_twenty_seventh_refresh", "pass", "Trial-2 paper-side sync reserve retained", 1, "Trial-2 paper sync remains reserve work while the two-component split route stays open."),
            row("trial3_two_component_trial4_deferred_retained_twenty_seventh_refresh", "pass", "Trial-4 deferred retained", 1, "Trial-4 stays deferred while the two-component route still has an honest residual path."),
        ],
        {
            "selected_residual_route": selected_residual_route,
            "missing_v2_artifact": missing_v2_artifact,
            "trial2_paper_side_sync_state": TRIAL2_RESERVE_STATE,
            "trial4_deferred": True,
            "recommended_next_route_or_none": recommended_next_route,
        },
        {
            "overall_status": "trial3_two_component_trial2_trial4_twenty_seventh_refresh_frozen",
            "trial3_branch_closeable": distinct_anchor_split_available,
            "advance_to_next_branch": not distinct_anchor_split_available,
            "next_required_artifacts": [] if distinct_anchor_split_available else [recommended_next_route],
        },
        {
            "declaration_summary": declaration["summary"],
            "current_status_summary": declaration_gate["summary"],
            "current_wz_summary": wz_metrics["summary"],
        },
    )

    write_artifact("mass_origin_v2_trial3_two_component_distinct_w_z_anchor_split_source_inventory", source_inventory)
    write_artifact("mass_origin_v2_trial3_two_component_distinct_w_z_anchor_split_audit", audit)
    write_artifact("mass_origin_v2_trial3_two_component_declaration_second_gate", declaration)
    write_artifact("mass_origin_v2_trial3_two_component_paper_sync_trial4_disposition_twenty_seventh_refresh", disposition)

    print("[done] Trial-3 two-component anchor-split artifacts written:")
    print(" - mass_origin_v2_trial3_two_component_distinct_w_z_anchor_split_source_inventory_metrics.json")
    print(" - mass_origin_v2_trial3_two_component_distinct_w_z_anchor_split_audit_metrics.json")
    print(" - mass_origin_v2_trial3_two_component_declaration_second_gate_metrics.json")
    print(" - mass_origin_v2_trial3_two_component_paper_sync_trial4_disposition_twenty_seventh_refresh_metrics.json")


# 関数: CLI から branch を実行する。

def run_cli() -> None:
    """CLI entry point for the Trial-3 two-component anchor-split branch."""
    main()


if __name__ == "__main__":
    run_cli()
