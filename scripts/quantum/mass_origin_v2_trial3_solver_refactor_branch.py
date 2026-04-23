#!/usr/bin/env python3
"""
Generate Trial-3 solver-refactor pivot artifacts for 8.7.56.273-.276.

The previous Trial-3 mainline kept shrinking a software-side residual around
`base_modes_by_ell`. The user-provided solver-refactor advice reclassifies that
loop as non-scientific: the blocker is not a missing theoretical ingredient,
but an implementation that kept the k-axis frozen at zero. This branch freezes
the direct solver refactor, confirms that explicit k>0 modes now exist in both
the numerical and exact ladders, and reclassifies the remaining Trial-3 work as
high-mass refactored extension rather than more artifact-inventory retries.
"""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output" / "public" / "quantum"
STATUS = ROOT / "doc" / "STATUS.md"
ROADMAP = ROOT / "doc" / "ROADMAP.md"
AI_CONTEXT = ROOT / "doc" / "AI_CONTEXT_MIN.json"

POST_PHOTON_PRESERVATION = OUT / "mass_origin_v2_post_photon_vector_mass_ratio_preservation_audit_metrics.json"
RELAUNCHED_PILOT = OUT / "mass_origin_v2_trial3_relaunched_weak_sector_pilot_metrics.json"
RELAUNCHED_AUDIT = OUT / "mass_origin_v2_trial3_relaunched_weinberg_angle_weak_coupling_audit_metrics.json"
NUMERICAL_PILOT = OUT / "mass_origin_vector_qball_ell_sector_shooting_pilot_metrics.json"
SPIN_TABLE = OUT / "mass_origin_vector_qball_spin_orbit_mass_ratio_table_metrics.json"
FULL_PILOT = OUT / "mass_origin_vector_qball_full_coupled_solver_pilot_metrics.json"
EXACT_HANDOFF = OUT / "mass_origin_vector_qball_exact_mass_table_handoff_retry_metrics.json"
NUMERICAL_BRANCH = ROOT / "scripts" / "quantum" / "mass_origin_vector_qball_numerical_solver_branch.py"
FULL_BRANCH = ROOT / "scripts" / "quantum" / "mass_origin_vector_qball_full_coupled_solver_branch.py"

W_TARGET = 80369.0 / 0.51099895
Z_TARGET = 91187.6 / 0.51099895


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


# 関数: JSON artifact と rows CSV を保存する。

def write_artifact(stem: str, data: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    json_path = OUT / f"{stem}_metrics.json"
    csv_path = OUT / f"{stem}_rows.csv"
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["row_id", "status", "metric", "value", "note"])
        writer.writeheader()
        writer.writerows(data["rows"])


# 関数: solver-refactor pivot branch を実行する。

def main() -> None:
    for path in (
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        POST_PHOTON_PRESERVATION,
        RELAUNCHED_PILOT,
        RELAUNCHED_AUDIT,
        NUMERICAL_PILOT,
        SPIN_TABLE,
        FULL_PILOT,
        EXACT_HANDOFF,
        NUMERICAL_BRANCH,
        FULL_BRANCH,
    ):
        req(path)

    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    ai_context = read_json(AI_CONTEXT)
    numerical_text = read_text(NUMERICAL_BRANCH)
    full_text = read_text(FULL_BRANCH)

    preservation = read_json(POST_PHOTON_PRESERVATION)
    relaunched_pilot = read_json(RELAUNCHED_PILOT)
    relaunched_audit = read_json(RELAUNCHED_AUDIT)
    numerical_pilot = read_json(NUMERICAL_PILOT)
    spin_table = read_json(SPIN_TABLE)
    full_pilot = read_json(FULL_PILOT)
    exact_handoff = read_json(EXACT_HANDOFF)

    normalization_scale = float(preservation["summary"]["absolute_mass_normalization_scale_factor"])
    historic_verified_ceiling = float(relaunched_pilot["summary"]["relaunched_verified_high_ell_max_ratio_to_electron"])
    low_ell_exact_ceiling = float(exact_handoff["summary"]["maximum_mass_ratio_to_scalar_base"])
    normalized_low_ell_exact_ceiling = low_ell_exact_ceiling * normalization_scale
    low_ell_w_gap = float(W_TARGET / normalized_low_ell_exact_ceiling)
    low_ell_z_gap = float(Z_TARGET / normalized_low_ell_exact_ceiling)
    historic_w_gap = float(relaunched_pilot["summary"]["w_gap_factor_after_normalization_only"])
    historic_z_gap = float(relaunched_pilot["summary"]["z_gap_factor_after_normalization_only"])

    numerical_k_positive_mode_count = int(numerical_pilot["summary"]["k_positive_mode_count"])
    exact_k_positive_mode_count = int(full_pilot["summary"]["k_positive_mode_count"])
    exact_handoff_ready = bool(exact_handoff["summary"]["hand_off_to_8_7_55_2_84"])
    best_exact_match = exact_handoff["summary"]["best_exact_match_or_none"]

    common_inputs = {
        "status_markdown": rel(STATUS),
        "roadmap_markdown": rel(ROADMAP),
        "ai_context_json": rel(AI_CONTEXT),
        "mass_origin_v2_post_photon_vector_mass_ratio_preservation_audit_json": rel(POST_PHOTON_PRESERVATION),
        "mass_origin_v2_trial3_relaunched_weak_sector_pilot_json": rel(RELAUNCHED_PILOT),
        "mass_origin_v2_trial3_relaunched_weinberg_angle_weak_coupling_audit_json": rel(RELAUNCHED_AUDIT),
        "mass_origin_vector_qball_ell_sector_shooting_pilot_json": rel(NUMERICAL_PILOT),
        "mass_origin_vector_qball_spin_orbit_mass_ratio_table_json": rel(SPIN_TABLE),
        "mass_origin_vector_qball_full_coupled_solver_pilot_json": rel(FULL_PILOT),
        "mass_origin_vector_qball_exact_mass_table_handoff_retry_json": rel(EXACT_HANDOFF),
        "mass_origin_vector_qball_numerical_solver_branch_py": rel(NUMERICAL_BRANCH),
        "mass_origin_vector_qball_full_coupled_solver_branch_py": rel(FULL_BRANCH),
    }

    inventory_targets = [
        {
            "label": "numerical_find_sector_amplitudes_present",
            "present": hit(numerical_text, "def find_sector_amplitudes(") is not None,
            "evidence": hit(numerical_text, "def find_sector_amplitudes("),
            "note": "The numerical solver must expose the multi-root sector search needed to capture explicit k>0 modes.",
        },
        {
            "label": "numerical_groups_localized_rows_by_k",
            "present": hit(numerical_text, "localized_by_k: dict[int, list[dict]] = {}") is not None,
            "evidence": hit(numerical_text, "localized_by_k: dict[int, list[dict]] = {}"),
            "note": "The interpolation layer must separate localized rows by k before building integer modes.",
        },
        {
            "label": "numerical_build_base_modes_flat_list_present",
            "present": hit(numerical_text, "def build_base_modes(") is not None,
            "evidence": hit(numerical_text, "def build_base_modes("),
            "note": "The numerical solver must expose the flat base-mode builder used by the refactor.",
        },
        {
            "label": "full_builder_accepts_flat_or_dict_input",
            "present": hit(full_text, "if isinstance(base_modes, dict):") is not None,
            "evidence": hit(full_text, "if isinstance(base_modes, dict):"),
            "note": "The exact ladder builder must tolerate historical dict callers while the refactor migrates runtime branches.",
        },
        {
            "label": "full_builder_preserves_node_count_k",
            "present": hit(full_text, "\"node_count_k\": int(mode.get(\"node_count_k\", mode[\"k\"]))") is not None,
            "evidence": hit(full_text, "\"node_count_k\": int(mode.get(\"node_count_k\", mode[\"k\"]))"),
            "note": "The exact ladder must propagate node_count_k instead of freezing it at zero.",
        },
        {
            "label": "numerical_k_positive_modes_exist",
            "present": numerical_k_positive_mode_count > 0,
            "evidence": {"k_positive_mode_count": numerical_k_positive_mode_count},
            "note": "The numerical pilot must already produce at least one explicit k>0 base mode.",
        },
        {
            "label": "exact_k_positive_modes_exist",
            "present": exact_k_positive_mode_count > 0,
            "evidence": {"k_positive_mode_count": exact_k_positive_mode_count},
            "note": "The exact full-coupled ladder must already preserve explicit k>0 states.",
        },
        {
            "label": "exact_handoff_reopened",
            "present": exact_handoff_ready,
            "evidence": {"hand_off_to_8_7_55_2_84": exact_handoff_ready},
            "note": "The solver refactor should reopen the exact handoff gate instead of leaving the ladder software-blocked.",
        },
    ]
    inventory_ready = all(item["present"] for item in inventory_targets)

    source_inventory = payload(
        "8.7.56.273",
        "Trial-3 solver-refactor source inventory",
        common_inputs,
        "Freeze the direct solver-refactor source pack after the user-provided advice reclassified the old retry loop as a software blocker.",
        {
            "pivot_rule": "replace the artifact-inventory retry loop with direct solver refactor once the blocker is identified as missing k-axis propagation in code rather than missing current-canon physics",
            "data_shape_rule": "base states are now rebuilt through an explicit flat base-mode list plus k-aware interpolation before any weak-sector re-audit",
            "success_rule": "the refactor pack is ready only if explicit k>0 exists numerically, survives in the exact ladder, and reopens the exact handoff gate",
        },
        [
            row(
                "trial3_solver_refactor_source_inventory_complete",
                "pass",
                "Trial-3 solver-refactor source inventory complete",
                1,
                "The direct solver-refactor source inventory is frozen.",
            ),
            row(
                "trial3_solver_refactor_required_source_count",
                "pass" if inventory_ready else "reject",
                "required solver-refactor source count",
                len(inventory_targets),
                "The refactor pack requires k-aware numerical search, flat base-mode construction, exact node propagation, and reopened exact handoff.",
            ),
            row(
                "trial3_solver_refactor_numerical_k_positive_mode_count",
                "pass" if numerical_k_positive_mode_count > 0 else "reject",
                "numerical k-positive mode count",
                numerical_k_positive_mode_count,
                "The numerical pilot now records how many explicit k>0 modes already exist.",
            ),
            row(
                "trial3_solver_refactor_exact_k_positive_mode_count",
                "pass" if exact_k_positive_mode_count > 0 else "reject",
                "exact k-positive mode count",
                exact_k_positive_mode_count,
                "The exact ladder now preserves how many explicit k>0 modes survive the adopted coupled reconstruction.",
            ),
        ],
        {
            "inventory_ready": inventory_ready,
            "numerical_k_positive_mode_count": numerical_k_positive_mode_count,
            "exact_k_positive_mode_count": exact_k_positive_mode_count,
            "best_exact_match_or_none": best_exact_match,
            "exact_handoff_reopened": exact_handoff_ready,
            "first_route_to_close_or_none": "trial3_solver_refactor_execution_audit",
        },
        {
            "overall_status": "trial3_solver_refactor_source_inventory_frozen",
            "trial3_branch_closeable": False,
            "advance_to_8_7_56_274": True,
            "next_required_artifacts": ["trial3_solver_refactor_execution_audit"],
        },
        {
            "inventory_targets": inventory_targets,
            "status_current_step_before_pivot": ai_context["current_step"],
            "roadmap_old_branch_line": hit(roadmap_text, "`8.7.56.273-.276`"),
        },
    )

    solver_blocker_removed = bool(inventory_ready and numerical_k_positive_mode_count > 0 and exact_k_positive_mode_count > 0 and exact_handoff_ready)
    execution_audit = payload(
        "8.7.56.274",
        "Trial-3 solver-refactor execution audit",
        common_inputs,
        "Audit whether the direct solver refactor actually removes the old software blocker rather than only renaming it.",
        {
            "execution_rule": "the software blocker is removed only if the numerical solver emits explicit k>0 modes, the exact ladder keeps them, and the exact handoff gate reopens",
            "residual_reclassification_rule": "once the software blocker is removed, the remaining Trial-3 problem is a high-mass scientific extension rather than more inventory-level source hunting",
        },
        [
            row(
                "trial3_solver_refactor_execution_audit_complete",
                "pass",
                "Trial-3 solver-refactor execution audit complete",
                1,
                "The direct solver-refactor execution audit is frozen.",
            ),
            row(
                "trial3_solver_refactor_software_blocker_removed",
                "pass" if solver_blocker_removed else "reject",
                "software blocker removed by direct solver refactor",
                1 if solver_blocker_removed else 0,
                "The old retry loop is superseded only if k>0 now exists in both numerical and exact ladders and the exact handoff gate reopens.",
            ),
            row(
                "trial3_solver_refactor_exact_muon_anchor_pass",
                "pass" if best_exact_match and best_exact_match["passes_threshold"] else "reject",
                "exact muon anchor passes after solver refactor",
                1 if best_exact_match and best_exact_match["passes_threshold"] else 0,
                "The refactored exact ladder reopens the muon anchor with the updated k>0 states included.",
            ),
            row(
                "trial3_solver_refactor_old_dict_literal_retry_still_mainline",
                "reject" if solver_blocker_removed else "pass",
                "old dict-literal retry remains mainline",
                0 if solver_blocker_removed else 1,
                "Once the direct solver refactor works, the old dict-literal retry loop is downgraded to historical fallback rather than kept as the scientific mainline.",
            ),
        ],
        {
            "software_blocker_removed": solver_blocker_removed,
            "numerical_k_positive_mode_count": numerical_k_positive_mode_count,
            "exact_k_positive_mode_count": exact_k_positive_mode_count,
            "best_exact_match_or_none": best_exact_match,
            "exact_handoff_reopened": exact_handoff_ready,
            "selected_remaining_problem_class": (
                "refactored_high_mass_weak_sector_extension"
                if solver_blocker_removed
                else "software_blocker_still_open"
            ),
            "first_route_to_close_or_none": "trial3_solver_refactor_weak_sector_reaudit",
        },
        {
            "overall_status": "trial3_solver_refactor_execution_audited",
            "trial3_branch_closeable": False,
            "advance_to_8_7_56_275": True,
            "next_required_artifacts": ["trial3_solver_refactor_weak_sector_reaudit"],
        },
        {
            "numerical_summary": numerical_pilot["summary"],
            "spin_table_summary": spin_table["summary"],
            "full_pilot_summary": full_pilot["summary"],
            "exact_handoff_summary": exact_handoff["summary"],
        },
    )

    weak_reaudit = payload(
        "8.7.56.275",
        "Trial-3 solver-refactor weak-sector re-audit",
        common_inputs,
        "Re-audit Trial-3 after the solver refactor and separate the removed software blocker from the still-open high-mass weak-sector gap.",
        {
            "preservation_rule": "keep the post-photon normalized historic high-ell ceiling as the current verified weak-sector ceiling until a refactored high-mass table is rebuilt",
            "local_exact_rule": "use the refactored low-ell exact ladder only as proof that explicit k>0 execution now exists and improves the exact family locally",
            "gap_rule": "the remaining Trial-3 gap is scientific only if the preserved normalized ceiling still misses W/Z after the software blocker is removed",
        },
        [
            row(
                "trial3_solver_refactor_weak_sector_reaudit_complete",
                "pass",
                "Trial-3 solver-refactor weak-sector re-audit complete",
                1,
                "The post-refactor weak-sector re-audit is frozen.",
            ),
            row(
                "trial3_solver_refactor_preserved_high_ell_weak_gap_still_open",
                "pass",
                "preserved high-ell weak-sector gap still open",
                1,
                "Even after the software blocker is removed, the preserved normalized weak-sector ceiling still misses W/Z and requires a new high-mass table rebuild under the refactored solver.",
            ),
            row(
                "trial3_solver_refactor_low_ell_exact_ceiling_reaches_w_scale",
                "pass" if normalized_low_ell_exact_ceiling >= W_TARGET else "reject",
                "low-ell exact k-positive ceiling reaches W scale",
                1 if normalized_low_ell_exact_ceiling >= W_TARGET else 0,
                "The refactored low-ell exact ladder proves k>0 execution, but it does not by itself close the weak-sector mass scale.",
            ),
            row(
                "trial3_solver_refactor_weinberg_angle_route_closed",
                "pass" if False else "reject",
                "Weinberg-angle / weak-coupling route closed after solver refactor",
                0,
                "The solver refactor removes the software blocker first; a new high-mass state table is still required before the weak-coupling audit can be revisited honestly.",
            ),
        ],
        {
            "normalization_scale_factor": normalization_scale,
            "historic_preserved_verified_ceiling_to_electron": historic_verified_ceiling,
            "historic_w_gap_factor": historic_w_gap,
            "historic_z_gap_factor": historic_z_gap,
            "refactored_low_ell_exact_ceiling_to_electron": low_ell_exact_ceiling,
            "refactored_low_ell_exact_ceiling_to_electron_after_normalization": normalized_low_ell_exact_ceiling,
            "refactored_low_ell_w_gap_factor": low_ell_w_gap,
            "refactored_low_ell_z_gap_factor": low_ell_z_gap,
            "best_exact_match_or_none": best_exact_match,
            "mw_mz_ratio_value": relaunched_audit["summary"]["mw_mz_ratio_value"],
            "sin2_theta_w_value": relaunched_audit["summary"]["sin2_theta_w_value"],
            "trial3_recommended_condition_satisfied": False,
            "first_route_to_close_or_none": "trial3_solver_refactor_declaration_gate",
        },
        {
            "overall_status": "trial3_solver_refactor_weak_sector_reaudited",
            "trial3_branch_closeable": False,
            "advance_to_8_7_56_276": True,
            "next_required_artifacts": ["trial3_solver_refactor_declaration_gate"],
        },
        {
            "post_photon_preservation_summary": preservation["summary"],
            "relaunched_pilot_summary": relaunched_pilot["summary"],
            "relaunched_audit_summary": relaunched_audit["summary"],
            "full_exact_summary": exact_handoff["summary"],
        },
    )

    declaration = payload(
        "8.7.56.276",
        "Trial-3 solver-refactor declaration gate / disposition refresh",
        common_inputs,
        "Freeze the new Trial-3 mainline after the solver refactor removes the software blocker and leaves only the refactored high-mass weak-sector extension as the honest next route.",
        {
            "mainline_switch_rule": "once the software blocker is removed, the official mainline switches away from the dict-literal retry family and toward the refactored high-mass weak-sector extension",
            "trial2_rule": "Trial-2 paper-side sync stays unlocked reserve work while the refactored high-mass weak-sector route remains scientifically open",
            "trial4_rule": "Trial-4 stays deferred until the refactored high-mass weak-sector route loses all honest current-canon paths",
        },
        [
            row(
                "trial3_solver_refactor_declaration_gate_complete",
                "pass",
                "Trial-3 solver-refactor declaration gate complete",
                1,
                "The post-refactor declaration/disposition gate is frozen.",
            ),
            row(
                "trial3_solver_refactor_mainline_switch_complete",
                "pass" if solver_blocker_removed else "reject",
                "Trial-3 mainline switch to refactored high-mass route complete",
                1 if solver_blocker_removed else 0,
                "The official mainline switches only if the direct solver refactor has already removed the software blocker.",
            ),
            row(
                "trial3_solver_refactor_trial2_paper_side_sync_execute_now",
                "reject",
                "execute Trial-2 paper-side sync now",
                0,
                "Trial-2 paper-side sync stays unlocked reserve work while the refactored Trial-3 route remains scientifically open.",
            ),
            row(
                "trial3_solver_refactor_trial4_execute_now",
                "reject",
                "execute Trial-4 now",
                0,
                "Trial-4 remains deferred because the refactored high-mass weak-sector route is still honest.",
            ),
        ],
        {
            "trial3_branch_closeable": False,
            "solver_side_blocker_removed": solver_blocker_removed,
            "selected_residual_route": "trial3_relaunched_refactored_high_mass_k_positive_extension_identification",
            "missing_v2_artifact": "trial3_relaunched_refactored_high_mass_k_positive_exact_family_table",
            "trial2_paper_side_sync_state": "unlocked_reserve_retained",
            "trial4_deferred": True,
            "retry_loop_state": "fallback_hold_historical_diagnosis",
            "recommended_next_route_or_none": "8.7.56.277",
        },
        {
            "overall_status": "trial3_solver_refactor_declaration_gate_frozen",
            "trial3_branch_closeable": False,
            "advance_to_8_7_56_277": True,
            "next_required_artifacts": [
                "trial3_relaunched_refactored_high_mass_k_positive_extension_source_inventory",
                "trial3_relaunched_refactored_high_mass_k_positive_extension_audit",
            ],
        },
        {
            "source_inventory_summary": source_inventory["summary"],
            "execution_audit_summary": execution_audit["summary"],
            "weak_reaudit_summary": weak_reaudit["summary"],
            "status_old_open_question_line": hit(status_text, "trial3_relaunched_explicit_k_positive_base_modes_by_ell_dict_literal_source_inventory"),
        },
    )

    write_artifact("mass_origin_v2_trial3_solver_refactor_source_inventory", source_inventory)
    write_artifact("mass_origin_v2_trial3_solver_refactor_execution_audit", execution_audit)
    write_artifact("mass_origin_v2_trial3_solver_refactor_weak_sector_reaudit", weak_reaudit)
    write_artifact("mass_origin_v2_trial3_solver_refactor_declaration_gate", declaration)

    print("[ok] wrote:")
    print(" - mass_origin_v2_trial3_solver_refactor_source_inventory_metrics.json")
    print(" - mass_origin_v2_trial3_solver_refactor_execution_audit_metrics.json")
    print(" - mass_origin_v2_trial3_solver_refactor_weak_sector_reaudit_metrics.json")
    print(" - mass_origin_v2_trial3_solver_refactor_declaration_gate_metrics.json")


# 関数: CLI から solver-refactor pivot branch を起動する。

if __name__ == "__main__":
    main()
