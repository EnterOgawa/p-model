#!/usr/bin/env python3
"""
Generate relaunched Trial-3 weak-sector artifacts for 8.7.56.224-.228.

The post-photon unlock pivot reclassified the vector-Q-ball ladder as a
preserved current physical claim under a common `sqrt(2)` mass normalization.
That reopened Trial-3 on the mainline. This branch freezes the relaunch pack,
re-audits the weak-sector pilot under the preserved ladder, and decides whether
the next blocker is still the explicit `k>0` ladder itself or only its current
numerical/executable realization.
"""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output" / "public" / "quantum"
PART1 = ROOT / "doc" / "paper" / "10_part1_core_theory.md"
PART3A = ROOT / "doc" / "paper" / "12_part3a_quantum_foundations.md"
STATUS = ROOT / "doc" / "STATUS.md"
ROADMAP = ROOT / "doc" / "ROADMAP.md"
AI_CONTEXT = ROOT / "doc" / "AI_CONTEXT_MIN.json"

POST_PHOTON_PRESERVATION = OUT / "mass_origin_v2_post_photon_vector_mass_ratio_preservation_audit_metrics.json"
POST_PHOTON_UNLOCK = OUT / "mass_origin_v2_post_photon_dependency_unlock_gate_metrics.json"
POST_PHOTON_RELAUNCH = OUT / "mass_origin_v2_post_photon_trial3_relaunch_route_contract_metrics.json"
TRIAL3_OLD_SOURCE = OUT / "mass_origin_v2_trial3_wz_sector_source_inventory_metrics.json"
TRIAL3_OLD_PILOT = OUT / "mass_origin_v2_trial3_wz_vector_mode_pilot_metrics.json"
TRIAL3_OLD_AUDIT = OUT / "mass_origin_v2_trial3_weinberg_angle_weak_coupling_audit_metrics.json"
TRIAL3_HIGH_MASS = OUT / "mass_origin_v2_trial3_high_mass_scale_extension_inventory_metrics.json"
TRIAL3_HIGH_MASS_ROUTE = OUT / "mass_origin_v2_trial3_explicit_k_positive_extension_route_contract_metrics.json"
VECTOR_SOLVER_SPEC = OUT / "mass_origin_vector_qball_solver_spec_metrics.json"
VECTOR_CONSTRAINT = OUT / "mass_origin_vector_qball_coupled_constraint_freeze_audit_metrics.json"
VECTOR_EXACT = OUT / "mass_origin_vector_qball_exact_mass_table_handoff_retry_metrics.json"
VECTOR_HEAVY = OUT / "mass_origin_vector_qball_baryon_tau_neutron_fit_table_metrics.json"

NUMERICAL_BRANCH = ROOT / "scripts" / "quantum" / "mass_origin_vector_qball_numerical_solver_branch.py"
FULL_COUPLED_BRANCH = ROOT / "scripts" / "quantum" / "mass_origin_vector_qball_full_coupled_solver_branch.py"

ELECTRON_MASS_MEV = 0.51099895
W_MASS_MEV = 80369.0
Z_MASS_MEV = 91187.6
W_TARGET = W_MASS_MEV / ELECTRON_MASS_MEV
Z_TARGET = Z_MASS_MEV / ELECTRON_MASS_MEV
WZ_RATIO_TARGET = W_MASS_MEV / Z_MASS_MEV
SIN2_THETA_W_TARGET = 1.0 - WZ_RATIO_TARGET * WZ_RATIO_TARGET
PASS_THRESHOLD = 0.10


# 関数: 現在の UTC 時刻を ISO 8601 形式で返す。
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: 必須入力が欠けている場合に即時停止する。

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


# 関数: 絶対パスを repo 相対文字列へ変換する。

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


# 関数: source inventory 用の target record を組み立てる。

def target_record(file_key: str, path: Path, text: str, pattern: str, note: str) -> dict:
    target_hit = hit(text, pattern)
    return {
        "file_key": file_key,
        "file": rel(path),
        "pattern": pattern,
        "present": target_hit is not None,
        "note": note,
        "evidence": target_hit,
    }


# 関数: relaunched Trial-3 branch を実行する。

def main() -> None:
    for path in (
        PART1,
        PART3A,
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        POST_PHOTON_PRESERVATION,
        POST_PHOTON_UNLOCK,
        POST_PHOTON_RELAUNCH,
        TRIAL3_OLD_SOURCE,
        TRIAL3_OLD_PILOT,
        TRIAL3_OLD_AUDIT,
        TRIAL3_HIGH_MASS,
        TRIAL3_HIGH_MASS_ROUTE,
        VECTOR_SOLVER_SPEC,
        VECTOR_CONSTRAINT,
        VECTOR_EXACT,
        VECTOR_HEAVY,
        NUMERICAL_BRANCH,
        FULL_COUPLED_BRANCH,
    ):
        req(path)

    part1_text = read_text(PART1)
    part3a_text = read_text(PART3A)
    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    numerical_text = read_text(NUMERICAL_BRANCH)
    full_text = read_text(FULL_COUPLED_BRANCH)
    ai_context = read_json(AI_CONTEXT)

    preservation = read_json(POST_PHOTON_PRESERVATION)
    unlock = read_json(POST_PHOTON_UNLOCK)
    relaunch = read_json(POST_PHOTON_RELAUNCH)
    trial3_old_source = read_json(TRIAL3_OLD_SOURCE)
    trial3_old_pilot = read_json(TRIAL3_OLD_PILOT)
    trial3_old_audit = read_json(TRIAL3_OLD_AUDIT)
    trial3_high_mass = read_json(TRIAL3_HIGH_MASS)
    trial3_high_mass_route = read_json(TRIAL3_HIGH_MASS_ROUTE)
    vector_solver_spec = read_json(VECTOR_SOLVER_SPEC)
    vector_constraint = read_json(VECTOR_CONSTRAINT)
    vector_exact = read_json(VECTOR_EXACT)
    vector_heavy = read_json(VECTOR_HEAVY)

    normalization_scale = float(preservation["summary"]["absolute_mass_normalization_scale_factor"])
    radius_scale = float(preservation["summary"]["radius_normalization_scale_factor"])
    old_pilot_max_ratio = float(trial3_old_pilot["summary"]["maximum_mass_ratio_to_electron"])
    old_verified_max_ratio = float(trial3_high_mass["summary"]["current_verified_max_ratio_to_electron"])
    relaunched_pilot_max_ratio = old_pilot_max_ratio * normalization_scale
    relaunched_verified_max_ratio = old_verified_max_ratio * normalization_scale
    w_gap_after_normalization = float(W_TARGET / relaunched_verified_max_ratio)
    z_gap_after_normalization = float(Z_TARGET / relaunched_verified_max_ratio)
    w_relative_error_after_normalization = abs(relaunched_verified_max_ratio - W_TARGET) / W_TARGET
    z_relative_error_after_normalization = abs(relaunched_verified_max_ratio - Z_TARGET) / Z_TARGET
    mw_mz_ratio_value = float(trial3_old_audit["summary"]["best_pair_or_none"]["mw_mz_ratio_value"])
    sin2_theta_w_value = float(trial3_old_audit["summary"]["best_pair_or_none"]["sin2_theta_w_value"])
    mw_mz_ratio_relative_error = abs(mw_mz_ratio_value - WZ_RATIO_TARGET) / WZ_RATIO_TARGET
    sin2_theta_w_relative_error = abs(sin2_theta_w_value - SIN2_THETA_W_TARGET) / SIN2_THETA_W_TARGET

    explicit_k_positive_candidate_available = bool(trial3_high_mass["summary"]["explicit_k_positive_candidate_available"])
    explicit_k_positive_solver_axis_present = "k>0 after the base sectors are stable" in str(
        vector_solver_spec["formulas"]["pilot_sector_rule"]
    )
    explicit_k_positive_bookkeeping_available = bool(vector_constraint["summary"]["k_node_bookkeeping_available"])
    explicit_k_positive_integer_mode_interpolation_available = False
    explicit_k_positive_exact_ladder_available = False

    common_inputs = {
        "part1_markdown": rel(PART1),
        "part3a_markdown": rel(PART3A),
        "status_markdown": rel(STATUS),
        "roadmap_markdown": rel(ROADMAP),
        "ai_context_json": rel(AI_CONTEXT),
        "mass_origin_v2_post_photon_vector_mass_ratio_preservation_audit_json": rel(POST_PHOTON_PRESERVATION),
        "mass_origin_v2_post_photon_dependency_unlock_gate_json": rel(POST_PHOTON_UNLOCK),
        "mass_origin_v2_post_photon_trial3_relaunch_route_contract_json": rel(POST_PHOTON_RELAUNCH),
        "mass_origin_v2_trial3_wz_sector_source_inventory_json": rel(TRIAL3_OLD_SOURCE),
        "mass_origin_v2_trial3_wz_vector_mode_pilot_json": rel(TRIAL3_OLD_PILOT),
        "mass_origin_v2_trial3_weinberg_angle_weak_coupling_audit_json": rel(TRIAL3_OLD_AUDIT),
        "mass_origin_v2_trial3_high_mass_scale_extension_inventory_json": rel(TRIAL3_HIGH_MASS),
        "mass_origin_v2_trial3_explicit_k_positive_extension_route_contract_json": rel(TRIAL3_HIGH_MASS_ROUTE),
        "mass_origin_vector_qball_solver_spec_json": rel(VECTOR_SOLVER_SPEC),
        "mass_origin_vector_qball_coupled_constraint_freeze_audit_json": rel(VECTOR_CONSTRAINT),
        "mass_origin_vector_qball_exact_mass_table_handoff_retry_json": rel(VECTOR_EXACT),
        "mass_origin_vector_qball_baryon_tau_neutron_fit_table_json": rel(VECTOR_HEAVY),
        "mass_origin_vector_qball_numerical_solver_branch_py": rel(NUMERICAL_BRANCH),
        "mass_origin_vector_qball_full_coupled_solver_branch_py": rel(FULL_COUPLED_BRANCH),
    }

    inventory_targets = [
        target_record(
            "status_relaunched_trial3_next_step",
            STATUS,
            status_text,
            "current official next step は `8.7.56.224`",
            "STATUS must already point to the relaunched weak-sector source inventory.",
        ),
        target_record(
            "roadmap_relaunched_trial3_branch",
            ROADMAP,
            roadmap_text,
            "`8.7.56.224-.228` 試練3 relaunched explicit `k>0` weak-sector extension branch",
            "ROADMAP must already freeze the relaunched Trial-3 branch.",
        ),
        target_record(
            "part1_micro_chiral_line",
            PART1,
            part1_text,
            "左手系カイラル流",
            "Part I must still expose the micro chiral-current line that motivates Trial-3.",
        ),
        target_record(
            "part1_weak_va_line",
            PART1,
            part1_text,
            "V-A 演算子構造",
            "Part I must still state the weak V-A structural hint.",
        ),
        target_record(
            "part3a_exact_vector_hierarchy_line",
            PART3A,
            part3a_text,
            "exact vector hierarchy",
            "Part III-A must still expose the exact vector hierarchy as the weak-sector source pack.",
        ),
        target_record(
            "numerical_solver_k_zero_hardcode",
            NUMERICAL_BRANCH,
            numerical_text,
            "\"k\": 0,",
            "The current numerical interpolation path still hardcodes k=0.",
        ),
        target_record(
            "full_solver_node_count_k_zero_hardcode",
            FULL_COUPLED_BRANCH,
            full_text,
            "\"node_count_k\": 0,",
            "The current exact builder still writes node_count_k=0 in the frozen ladder rows.",
        ),
    ]
    inventory_ready = all(item["present"] for item in inventory_targets)

    source_inventory = payload(
        "8.7.56.224",
        "Trial-3 relaunched weak-sector source inventory",
        common_inputs,
        "Inventory the preserved vector ladder, the sqrt(2) normalization update, the W/Z target pack, the explicit k>0 continuation candidate, and the old weak-sector pilot evidence under the post-photon relaunch.",
        {
            "relaunch_rule": "reuse the preserved vector ladder as a current physical claim and apply the common sqrt(2) mass normalization before rerunning weak-sector bookkeeping",
            "target_pack": {
                "M_W/m_e": W_TARGET,
                "M_Z/m_e": Z_TARGET,
                "M_W/M_Z": WZ_RATIO_TARGET,
                "sin2_theta_W": SIN2_THETA_W_TARGET,
            },
            "k_positive_rule": "explicit k>0 remains admissible only if the frozen k-axis and k-node bookkeeping survive the post-photon unlock pivot",
        },
        [
            row(
                "trial3_relaunched_source_inventory_complete",
                "pass",
                "Trial-3 relaunched weak-sector source inventory complete",
                1,
                "The relaunched weak-sector source inventory is frozen.",
            ),
            row(
                "trial3_relaunched_required_source_count",
                "pass" if inventory_ready else "reject",
                "required relaunched source count",
                len(inventory_targets),
                "The relaunched Trial-3 pack needs the post-photon unlock outputs, the old weak-sector evidence, and the k-axis bookkeeping.",
            ),
            row(
                "trial3_relaunched_vector_ladder_preserved",
                "pass" if preservation["summary"]["working_action_vector_mass_spectrum_physical_claim_preserved"] else "reject",
                "preserved vector ladder available as current physical claim",
                1 if preservation["summary"]["working_action_vector_mass_spectrum_physical_claim_preserved"] else 0,
                "The relaunch assumes the vector ladder is preserved under the working action.",
            ),
            row(
                "trial3_relaunched_explicit_k_positive_candidate_available",
                "pass" if explicit_k_positive_candidate_available else "reject",
                "explicit k-positive continuation candidate available",
                1 if explicit_k_positive_candidate_available else 0,
                "The relaunch keeps the explicit k>0 axis as the next same-family extension candidate.",
            ),
        ],
        {
            "inventory_ready": inventory_ready,
            "normalization_scale_factor": normalization_scale,
            "radius_scale_factor": radius_scale,
            "trial3_relaunch_ready": bool(unlock["summary"]["trial3_explicit_k_positive_branch_relaunch_ready"]),
            "trial2_paper_side_sync_unlock_ready": bool(unlock["summary"]["trial2_paper_side_sync_unlock_ready"]),
            "explicit_k_positive_candidate_available": explicit_k_positive_candidate_available,
            "explicit_k_positive_solver_axis_present": explicit_k_positive_solver_axis_present,
            "explicit_k_positive_bookkeeping_available": explicit_k_positive_bookkeeping_available,
            "first_route_to_close_or_none": "trial3_relaunched_weak_sector_pilot",
        },
        {
            "overall_status": "trial3_relaunched_source_inventory_frozen",
            "trial3_branch_closeable": False,
            "advance_to_8_7_56_225": True,
            "next_required_artifacts": ["trial3_relaunched_weak_sector_pilot"],
        },
        {
            "inventory_targets": inventory_targets,
            "post_photon_relaunch_summary": relaunch["summary"],
            "post_photon_unlock_summary": unlock["summary"],
            "old_trial3_source_summary": trial3_old_source["summary"],
            "high_mass_inventory_summary": trial3_high_mass["summary"],
        },
    )

    relaunched_pilot = payload(
        "8.7.56.225",
        "Trial-3 relaunched weak-sector pilot",
        common_inputs,
        "Rerun the weak-sector pilot under the preserved ladder and the sqrt(2) normalization update, then separate the normalization-only improvement from the still-missing executable explicit k>0 ladder.",
        {
            "normalization_update_rule": "M^(relaunch) = sqrt(2) * M^(historic exact family)",
            "baseline_rule": "use the old verified high-ell k=0 ceiling as the preserved pre-k>0 baseline under the new working action",
            "pilot_gap_rule": "compare the normalized verified ceiling against the W/Z target scales before claiming an executable explicit k>0 extension",
            "executable_ladder_rule": "a genuine relaunch pilot needs a node-resolved k>0 integer-mode table rather than only the frozen k-axis bookkeeping",
        },
        [
            row(
                "trial3_relaunched_weak_sector_pilot_complete",
                "pass",
                "Trial-3 relaunched weak-sector pilot complete",
                1,
                "The relaunched weak-sector pilot baseline is frozen.",
            ),
            row(
                "trial3_relaunched_normalization_only_w_anchor_pass",
                "pass" if w_relative_error_after_normalization <= PASS_THRESHOLD else "reject",
                "W/electron anchor passes after normalization-only update",
                1 if w_relative_error_after_normalization <= PASS_THRESHOLD else 0,
                "The preserved ladder plus sqrt(2) update alone still does not reach the W scale.",
            ),
            row(
                "trial3_relaunched_normalization_only_z_anchor_pass",
                "pass" if z_relative_error_after_normalization <= PASS_THRESHOLD else "reject",
                "Z/electron anchor passes after normalization-only update",
                1 if z_relative_error_after_normalization <= PASS_THRESHOLD else 0,
                "The preserved ladder plus sqrt(2) update alone still does not reach the Z scale.",
            ),
            row(
                "trial3_relaunched_explicit_k_positive_integer_mode_interpolation_available",
                "pass" if explicit_k_positive_integer_mode_interpolation_available else "reject",
                "explicit k-positive integer-mode interpolation available",
                1 if explicit_k_positive_integer_mode_interpolation_available else 0,
                "The current numerical solver still hardcodes k=0, so the explicit k>0 ladder is not yet executable.",
            ),
            row(
                "trial3_relaunched_high_mass_gap_still_present",
                "pass",
                "high-mass weak-sector gap still present after normalization-only update",
                1,
                "Normalization-only improvement helps but does not close the weak-sector scale gap.",
            ),
        ],
        {
            "historic_k0_pilot_max_ratio_to_electron": old_pilot_max_ratio,
            "historic_verified_high_ell_max_ratio_to_electron": old_verified_max_ratio,
            "relaunched_k0_pilot_max_ratio_to_electron": relaunched_pilot_max_ratio,
            "relaunched_verified_high_ell_max_ratio_to_electron": relaunched_verified_max_ratio,
            "w_gap_factor_after_normalization_only": w_gap_after_normalization,
            "z_gap_factor_after_normalization_only": z_gap_after_normalization,
            "w_relative_error_after_normalization_only": w_relative_error_after_normalization,
            "z_relative_error_after_normalization_only": z_relative_error_after_normalization,
            "explicit_k_positive_candidate_available": explicit_k_positive_candidate_available,
            "explicit_k_positive_integer_mode_interpolation_available": explicit_k_positive_integer_mode_interpolation_available,
            "explicit_k_positive_exact_ladder_available": explicit_k_positive_exact_ladder_available,
            "first_route_to_close_or_none": "trial3_relaunched_weinberg_angle_weak_coupling_audit",
        },
        {
            "overall_status": "trial3_relaunched_weak_sector_pilot_frozen",
            "trial3_branch_closeable": False,
            "advance_to_8_7_56_226": True,
            "next_required_artifacts": ["trial3_relaunched_weinberg_angle_weak_coupling_audit"],
        },
        {
            "old_trial3_pilot_summary": trial3_old_pilot["summary"],
            "high_mass_inventory_summary": trial3_high_mass["summary"],
            "preservation_summary": preservation["summary"],
            "numerical_solver_k_zero_line": hit(numerical_text, "\"k\": 0,"),
            "full_solver_node_count_line": hit(full_text, "\"node_count_k\": 0,"),
        },
    )

    weak_audit = payload(
        "8.7.56.226",
        "Trial-3 relaunched Weinberg-angle / weak-coupling audit",
        common_inputs,
        "Re-audit the W/Z ratio structure and weak-coupling admissibility after the post-photon normalization update and the weak-sector relaunch.",
        {
            "ratio_invariance_rule": "a common mass rescaling leaves M_W/M_Z unchanged, so the old ratio audit is reused unless an explicit k>0 ladder changes the state pairing",
            "weinberg_rule": "sin^2(theta_W) = 1 - (M_W/M_Z)^2 is re-evaluated on the relaunched branch",
            "coupling_rule": "weak-coupling closure still requires an executable explicit k>0 exact ladder or another first-principles weak coupling map",
        },
        [
            row(
                "trial3_relaunched_mw_mz_ratio_pass",
                "pass" if mw_mz_ratio_relative_error <= PASS_THRESHOLD else "reject",
                "relaunched M_W/M_Z ratio passes threshold",
                1 if mw_mz_ratio_relative_error <= PASS_THRESHOLD else 0,
                "The common normalization update alone does not improve the dimensionless W/Z ratio.",
            ),
            row(
                "trial3_relaunched_sin2_theta_w_pass",
                "pass" if sin2_theta_w_relative_error <= PASS_THRESHOLD else "reject",
                "relaunched sin^2(theta_W) passes threshold",
                1 if sin2_theta_w_relative_error <= PASS_THRESHOLD else 0,
                "The Weinberg-angle proxy remains unchanged until a new state pairing appears.",
            ),
            row(
                "trial3_relaunched_weak_coupling_first_principles_route_available",
                "pass" if False else "reject",
                "relaunched weak-coupling first-principles route available",
                0,
                "The relaunch preserves the structural candidate but does not yet close a numeric weak-coupling route.",
            ),
        ],
        {
            "mw_mz_ratio_target": WZ_RATIO_TARGET,
            "sin2_theta_w_target": SIN2_THETA_W_TARGET,
            "mw_mz_ratio_value": mw_mz_ratio_value,
            "mw_mz_ratio_relative_error": mw_mz_ratio_relative_error,
            "sin2_theta_w_value": sin2_theta_w_value,
            "sin2_theta_w_relative_error": sin2_theta_w_relative_error,
            "common_mass_rescaling_changes_mw_mz_ratio": False,
            "weak_coupling_first_principles_route_available": False,
            "first_route_to_close_or_none": "trial3_relaunched_declaration_gate",
        },
        {
            "overall_status": "trial3_relaunched_weinberg_and_coupling_audited",
            "trial3_branch_closeable": False,
            "advance_to_8_7_56_227": True,
            "next_required_artifacts": ["trial3_relaunched_declaration_gate"],
        },
        {
            "old_trial3_audit_summary": trial3_old_audit["summary"],
            "old_trial3_best_pair": trial3_old_audit["summary"]["best_pair_or_none"],
        },
    )

    declaration_gate = payload(
        "8.7.56.227",
        "Trial-3 relaunched declaration gate",
        common_inputs,
        "Freeze whether the relaunched weak-sector branch is already closeable under the preserved ladder or whether a deeper explicit k>0 executable-ladder residual route is still required.",
        {
            "closeout_rule": "the relaunched branch closes only if the normalized ladder plus an executable explicit k>0 continuation reaches the W/Z anchors and the ratio audit",
            "residual_rule": "if the k-axis is only a bookkeeping candidate, the next residual route is the missing executable integer-mode table for k>0 states",
        },
        [
            row(
                "trial3_relaunched_declaration_gate_complete",
                "pass",
                "Trial-3 relaunched declaration gate complete",
                1,
                "The relaunched weak-sector declaration gate is frozen.",
            ),
            row(
                "trial3_relaunched_branch_closeable",
                "pass" if False else "reject",
                "relaunched weak-sector branch closeable",
                0,
                "The branch does not close while the explicit k>0 ladder is still non-executable.",
            ),
            row(
                "trial3_relaunched_residual_route_required",
                "pass",
                "relaunched weak-sector residual route required",
                1,
                "A deeper residual route is still required after the relaunch audit.",
            ),
            row(
                "trial3_relaunched_trial2_paper_sync_execute_now",
                "pass" if False else "reject",
                "execute Trial-2 paper-side sync now",
                0,
                "Trial-2 paper sync remains unlocked reserve work, but the scientific mainline still points to the weak-sector residual.",
            ),
        ],
        {
            "trial3_relaunched_branch_closeable": False,
            "trial3_recommended_condition_satisfied": False,
            "trial3_relaunched_residual_route_required": True,
            "trial2_paper_side_sync_execute_now": False,
            "trial4_deferred": True,
            "recommended_next_route_or_none": "8.7.56.229",
        },
        {
            "overall_status": "trial3_relaunched_declaration_gate_frozen",
            "trial3_branch_closeable": False,
            "advance_to_8_7_56_228": True,
            "next_required_artifacts": ["trial3_relaunched_paper_sync_trial4_disposition_gate"],
        },
        {
            "source_inventory_summary": source_inventory["summary"],
            "relaunched_pilot_summary": relaunched_pilot["summary"],
            "weak_audit_summary": weak_audit["summary"],
        },
    )

    disposition_gate = payload(
        "8.7.56.228",
        "Trial-2 paper-side sync / Trial-4 disposition refresh gate",
        common_inputs,
        "Refresh the reserve/deferred ordering after the relaunched weak-sector audit and freeze the next residual route for the mainline.",
        {
            "trial2_rule": "retain Trial-2 paper-side sync as unlocked reserve work while the scientific weak-sector route is still open",
            "trial4_rule": "keep Trial-4 deferred until the relaunched Trial-3 branch loses all honest current-canon routes",
            "selected_residual_route": "trial3_relaunched_explicit_k_positive_integer_mode_table_identification",
            "missing_v2_artifact": "trial3_relaunched_explicit_k_positive_integer_mode_table",
        },
        [
            row(
                "trial3_relaunched_disposition_gate_complete",
                "pass",
                "Trial-3 relaunched disposition gate complete",
                1,
                "The post-relaunch reserve/deferred ordering is frozen.",
            ),
            row(
                "trial3_relaunched_trial2_paper_side_sync_reserve_retained",
                "pass",
                "Trial-2 paper-side sync retained as unlocked reserve",
                1,
                "Trial-2 paper-side sync stays available but not yet promoted to the main scientific route.",
            ),
            row(
                "trial3_relaunched_trial4_deferred_retained",
                "pass",
                "Trial-4 deferred disposition retained",
                1,
                "Trial-4 remains deferred while the relaunched Trial-3 residual route is still honest.",
            ),
            row(
                "trial3_relaunched_next_residual_route_frozen",
                "pass",
                "relaunched explicit k-positive integer-mode-table residual route frozen",
                1,
                "The next residual route is the missing executable node-resolved integer-mode table for k>0 states.",
            ),
        ],
        {
            "selected_residual_route": "trial3_relaunched_explicit_k_positive_integer_mode_table_identification",
            "missing_v2_artifact": "trial3_relaunched_explicit_k_positive_integer_mode_table",
            "trial2_paper_side_sync_state": "unlocked_reserve_retained",
            "trial4_deferred": True,
            "split_contract_ready": True,
            "recommended_next_route_or_none": "8.7.56.229",
        },
        {
            "overall_status": "trial3_relaunched_disposition_gate_frozen",
            "trial3_branch_closeable": False,
            "advance_to_8_7_56_229": True,
            "next_required_artifacts": [
                "trial3_relaunched_explicit_k_positive_integer_mode_source_inventory",
                "trial3_relaunched_explicit_k_positive_integer_mode_audit",
            ],
        },
        {
            "post_photon_relaunch_summary": relaunch["summary"],
            "trial3_old_route_summary": trial3_high_mass_route["summary"],
            "declaration_summary": declaration_gate["summary"],
            "ai_context_current_step": ai_context["current_step"],
            "vector_exact_summary": vector_exact["summary"],
            "vector_heavy_summary": vector_heavy["summary"],
        },
    )

    write_artifact("mass_origin_v2_trial3_relaunched_weak_sector_source_inventory", source_inventory)
    write_artifact("mass_origin_v2_trial3_relaunched_weak_sector_pilot", relaunched_pilot)
    write_artifact("mass_origin_v2_trial3_relaunched_weinberg_angle_weak_coupling_audit", weak_audit)
    write_artifact("mass_origin_v2_trial3_relaunched_declaration_gate", declaration_gate)
    write_artifact("mass_origin_v2_trial3_relaunched_paper_sync_trial4_disposition_gate", disposition_gate)

    print("[ok] wrote:")
    print(" - mass_origin_v2_trial3_relaunched_weak_sector_source_inventory_metrics.json")
    print(" - mass_origin_v2_trial3_relaunched_weak_sector_pilot_metrics.json")
    print(" - mass_origin_v2_trial3_relaunched_weinberg_angle_weak_coupling_audit_metrics.json")
    print(" - mass_origin_v2_trial3_relaunched_declaration_gate_metrics.json")
    print(" - mass_origin_v2_trial3_relaunched_paper_sync_trial4_disposition_gate_metrics.json")


# 関数: CLI から relaunched Trial-3 weak-sector branch を起動する。

if __name__ == "__main__":
    main()
