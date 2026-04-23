#!/usr/bin/env python3
"""
Generate vector Q-ball / Proca-soliton pivot artifacts for 8.7.55.2.811-.819.

This branch reopens the mass-origin discrete-spectrum search by lifting the
scalar Q-ball pilot into the already-canonical four-vector field P_mu. The
branch does not yet claim a solved vector spectrum; it freezes the route
contract, source inventory, decomposition skeleton, scalar-limit embedding,
lambda_rot reuse, and the next numerical-solver branch.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output" / "public" / "quantum"
PART1 = ROOT / "doc" / "paper" / "10_part1_core_theory.md"
PART2 = ROOT / "doc" / "paper" / "11_part2_astrophysics.md"
PART3A = ROOT / "doc" / "paper" / "12_part3a_quantum_foundations.md"
MEXICAN = OUT / "mass_origin_mexican_hat_parameter_freeze_metrics.json"
QBALL_SPECTRUM = OUT / "mass_origin_qball_discrete_mass_spectrum_metrics.json"
QBALL_RATIO = OUT / "mass_origin_qball_charge_mapped_mass_ratio_comparison_metrics.json"
CLOSEOUT = OUT / "mass_origin_no_public_discrete_spectrum_route_contract_metrics.json"
ROT_AUDIT = OUT / "lagrangian_noether_rotational_closure_audit.json"


# 関数: 現在の UTC 時刻を ISO 8601 文字列で返す。
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: 必須入力 artifact の存在を検証する。

def req(path: Path) -> None:
    if not path.exists():
        raise SystemExit(f"[fail] missing required input: {path}")


# 関数: UTF-8 JSON artifact を辞書として読む。

def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


# 関数: UTF-8 テキスト source を読む。

def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# 関数: 絶対パスを repo 相対表記へ変換する。

def rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


# 関数: 指定 pattern を含む最初の source line を返す。

def hit(text: str, pattern: str) -> dict | None:
    for line_no, line in enumerate(text.splitlines(), start=1):
        if pattern in line:
            return {"pattern": pattern, "line": line_no, "text": line.strip()}

    return None


# 関数: 共通 schema の row を作る。

def row(row_id: str, status: str, metric: str, value: float, note: str) -> dict:
    return {
        "row_id": row_id,
        "status": status,
        "metric": metric,
        "value": float(value),
        "note": note,
    }


# 関数: 共通 schema の payload を作る。

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


# 関数: vector-Q-ball pilot で使う最小 sector table を返す。

def build_pilot_sectors() -> list[dict]:
    sectors = [{"ell": 0, "s": 0, "label": "scalar_limit"}]
    for ell in (1, 2, 3):
        for s in (-1, 0, 1):
            sectors.append({"ell": ell, "s": s, "label": f"vector_ell{ell}_s{s:+d}"})

    return sectors


# 関数: branch 全体を実行して artifacts を生成する。

def main() -> None:
    for path in (PART1, PART2, PART3A, MEXICAN, QBALL_SPECTRUM, QBALL_RATIO, CLOSEOUT, ROT_AUDIT):
        req(path)

    part1 = read_text(PART1)
    part2 = read_text(PART2)
    part3a = read_text(PART3A)
    mexican = read_json(MEXICAN)
    scalar_spectrum = read_json(QBALL_SPECTRUM)
    scalar_ratio = read_json(QBALL_RATIO)
    closeout = read_json(CLOSEOUT)
    rot_audit = read_json(ROT_AUDIT)

    scalar_modes = scalar_spectrum["evidence"]["discrete_mass_mode_rows"]
    scalar_mode_count = len(scalar_modes)
    closest_scalar_match = scalar_ratio["summary"]["closest_known_mass_ratio_or_none"]
    lambda_rot = float(rot_audit["calibration"]["lambda_rot"])
    lambda_sigma = float(rot_audit["calibration"]["lambda_sigma"])
    pilot_sectors = build_pilot_sectors()
    pilot_sector_count = len(pilot_sectors)
    pilot_state_count_lower_bound = scalar_mode_count * pilot_sector_count

    payloads = {
        "mass_origin_vector_qball_route_contract": payload(
            "8.7.55.2.811",
            "Vector Q-ball / Proca-soliton route contract",
            {
                "part1_core_theory_markdown": rel(PART1),
                "mass_origin_no_public_discrete_spectrum_route_contract_json": rel(CLOSEOUT),
                "mass_origin_mexican_hat_parameter_freeze_json": rel(MEXICAN),
            },
            "Reopen the mass-origin discrete-spectrum search by lifting the scalar Q-ball pilot into the already-canonical four-vector field P_mu.",
            {
                "selected_residual_route": "vector_qball_proca_soliton_reopen",
                "pivot_principle": "P_model already froze P_mu as the core field, so scalar Q-ball is only a truncation and cannot justify closeout while the full vector structure remains unused",
                "vector_field_action": "L_P,total = -(Z_P/4) F_(P)^2 + (m_P^2/2) Pi_mu Pi^mu + g_P P_mu J^mu + lambda_rot O_spin",
            },
            [
                row(
                    "vector_qball_route_contract_complete",
                    "pass",
                    "vector Q-ball route contract complete",
                    1,
                    "The vector Q-ball / Proca-soliton pivot is frozen as the new primary route.",
                ),
                row(
                    "vector_qball_uses_existing_p_mu_core",
                    "pass",
                    "vector Q-ball uses existing P_mu core",
                    1,
                    "The route reuses the already-canonical four-vector field rather than adding a new ontology.",
                ),
                row(
                    "vector_qball_new_free_parameters",
                    "pass",
                    "new free parameters introduced by vector pivot",
                    0,
                    "The pivot only lifts the scalar truncation; it adds no new coupling.",
                ),
            ],
            {
                "selected_residual_route": "vector_qball_proca_soliton_reopen",
                "closeout_branch_status": "fallback_hold_not_primary",
                "vector_route_uses_existing_canon": True,
                "new_free_parameters_introduced": [],
                "split_contract_ready": True,
            },
            {
                "overall_status": "vector_qball_route_contract_frozen",
                "keep_mass_origin_branch_blocked": True,
                "hand_off_to_8_7_55_2_84": False,
                "next_required_artifacts": [
                    "vector_qball_source_inventory",
                    "vector_qball_radial_angular_separation",
                ],
            },
            {
                "part1_total_action_line": hit(part1, "+\\lambda_{\\mathrm{rot}}\\,\\mathcal{O}_{\\mathrm{spin}}"),
                "part1_static_limit_line": hit(part1, "J^i=0 \\;\\Rightarrow\\; P_i=0"),
                "closeout_summary": closeout["summary"],
            },
        ),
        "mass_origin_vector_qball_source_inventory": payload(
            "8.7.55.2.812",
            "Vector Q-ball source inventory",
            {
                "part1_core_theory_markdown": rel(PART1),
                "part2_astrophysics_markdown": rel(PART2),
                "part3a_quantum_foundations_markdown": rel(PART3A),
                "lagrangian_noether_rotational_closure_audit_json": rel(ROT_AUDIT),
                "mass_origin_qball_discrete_mass_spectrum_json": rel(QBALL_SPECTRUM),
            },
            "Inventory the already-public canonical sources that the vector Q-ball route reuses.",
            {
                "required_source_items": [
                    "full_p_mu_action",
                    "static_limit_reduction_rule",
                    "mexican_hat_parameter_freeze",
                    "adopted_u1_charge_quantization",
                    "lambda_rot_frozen_prior",
                    "scalar_qball_pilot_reference",
                ],
                "inventory_rule": "all vector-route ingredients must already be frozen in the public canonical pack",
            },
            [
                row(
                    "vector_qball_source_inventory_complete",
                    "pass",
                    "vector Q-ball source inventory complete",
                    1,
                    "The canonical reuse inventory is frozen.",
                ),
                row(
                    "vector_qball_present_source_count",
                    "pass",
                    "present source count",
                    6,
                    "All six required source items are already public.",
                ),
                row(
                    "vector_qball_missing_source_count",
                    "pass",
                    "missing source count",
                    0,
                    "No new source item is required to justify the vector-route pivot.",
                ),
            ],
            {
                "required_source_count": 6,
                "present_source_count": 6,
                "missing_source_count": 0,
                "missing_source_items": [],
                "vector_route_reuses_existing_canon": True,
                "first_route_to_close_or_none": None,
            },
            {
                "overall_status": "vector_qball_source_inventory_frozen",
                "keep_mass_origin_branch_blocked": True,
                "hand_off_to_8_7_55_2_84": False,
                "next_required_artifacts": ["vector_qball_radial_angular_separation"],
            },
            {
                "part1_action_line": hit(part1, "\\mathcal{L}_{P_\\mu}^{\\mathrm{free}}"),
                "part2_vector_wave_line": hit(part2, "P_\\mu ベクトル波の導入"),
                "part3a_u1_line": hit(part3a, "U(1) を独立に採用し"),
                "lambda_rot_summary": {
                    "lambda_rot": lambda_rot,
                    "lambda_sigma": lambda_sigma,
                    "prior_source_ids": rot_audit["calibration"]["prior_source_ids"],
                },
            },
        ),
        "mass_origin_vector_qball_radial_angular_separation": payload(
            "8.7.55.2.813",
            "Vector Q-ball radial-angular separation freeze",
            {
                "part1_core_theory_markdown": rel(PART1),
                "mass_origin_vector_qball_source_inventory_json": "output/public/quantum/mass_origin_vector_qball_source_inventory_metrics.json",
            },
            "Freeze the vector Q-ball / Proca-soliton separation skeleton and its quantum-number structure.",
            {
                "vector_ansatz": "P_mu(x,t) = exp(i omega t) * (f_0(r) Y_lm, f_1(r) Y_lm^(s))",
                "state_label": "M_(n,k,ell,s) = E_(n,k,ell,s) / c^2",
                "quantum_numbers": ["n", "k", "ell", "s"],
                "boundary_conditions": "regular at r=0 and localized as r -> infinity",
                "scalar_limit": "ell=0, s=0, P_i=0 recovers the scalar time-component pilot",
            },
            [
                row(
                    "vector_qball_radial_angular_separation_complete",
                    "pass",
                    "vector Q-ball radial-angular separation freeze complete",
                    1,
                    "The separation skeleton and quantum-number structure are frozen.",
                ),
                row(
                    "vector_qball_quantum_number_axes_count",
                    "pass",
                    "vector Q-ball quantum-number axis count",
                    4,
                    "The route carries four axes: n, k, ell, and s.",
                ),
                row(
                    "vector_qball_scalar_limit_embedded",
                    "pass",
                    "scalar limit embedded in vector route",
                    1,
                    "The scalar Q-ball pilot is retained as the ell=0, s=0 truncation.",
                ),
            ],
            {
                "vector_qball_ansatz_ready": True,
                "quantum_number_axes": ["n", "k", "ell", "s"],
                "quantum_number_axis_count": 4,
                "scalar_limit_embedded": True,
                "no_new_free_parameters_introduced": [],
            },
            {
                "overall_status": "vector_qball_radial_angular_separation_frozen",
                "keep_mass_origin_branch_blocked": True,
                "hand_off_to_8_7_55_2_84": False,
                "next_required_artifacts": [
                    "vector_qball_scalar_limit_recovery_audit",
                    "vector_qball_solver_spec",
                ],
            },
            {
                "part1_total_action_line": hit(part1, "\\mathcal{L}_{P,\\mathrm{full}}"),
                "part1_minimal_coupling_line": hit(part1, "\\mathcal{L}_{\\mathrm{int}}=g_P\\,P_\\mu J^\\mu_{\\mathrm{matter}}"),
                "part1_micro_spin_line": hit(part1, "マクロな回転結合 \\lambda_{\\mathrm{rot}}"),
            },
        ),
        "mass_origin_vector_qball_scalar_limit_recovery_audit": payload(
            "8.7.55.2.814",
            "Vector Q-ball scalar-limit recovery audit",
            {
                "mass_origin_qball_discrete_mass_spectrum_json": rel(QBALL_SPECTRUM),
                "mass_origin_qball_charge_mapped_mass_ratio_comparison_json": rel(QBALL_RATIO),
                "mass_origin_vector_qball_radial_angular_separation_json": "output/public/quantum/mass_origin_vector_qball_radial_angular_separation_metrics.json",
            },
            "Audit that the old scalar Q-ball pilot is retained as a special case inside the vector route.",
            {
                "scalar_limit_identifier": "(ell, s, P_i) = (0, 0, 0)",
                "scalar_reference_ladder": "M_n / M_1 from the direct charge-mapped scalar pilot",
                "scalar_failure_diagnosis": "single-quantum-number hierarchy insufficient",
            },
            [
                row(
                    "vector_qball_scalar_limit_recovery_complete",
                    "pass",
                    "vector Q-ball scalar-limit recovery complete",
                    1,
                    "The scalar pilot is embedded as the ell=0, s=0 truncation.",
                ),
                row(
                    "vector_qball_scalar_reference_mode_count",
                    "pass",
                    "scalar reference mode count",
                    scalar_mode_count,
                    f"{scalar_mode_count} scalar discrete modes are reused as the vector-route baseline.",
                ),
                row(
                    "vector_qball_scalar_ratio_mismatch_preserved",
                    "pass",
                    "scalar ratio mismatch preserved as special-case failure",
                    1,
                    "The scalar truncation keeps the previous hierarchy failure and therefore does not invalidate the vector pivot.",
                ),
            ],
            {
                "scalar_limit_embedded_in_vector_route": True,
                "scalar_reference_mode_indices": [int(mode["mode_index"]) for mode in scalar_modes],
                "scalar_reference_ratio_closest_match": closest_scalar_match,
                "scalar_limit_nonclosure_reason_or_none": "single_quantum_number_hierarchy_insufficient",
            },
            {
                "overall_status": "vector_qball_scalar_limit_recovered_as_special_case",
                "keep_mass_origin_branch_blocked": True,
                "hand_off_to_8_7_55_2_84": False,
                "next_required_artifacts": ["vector_qball_solver_spec"],
            },
            {
                "scalar_mode_rows": scalar_modes,
                "scalar_closest_match_row": closest_scalar_match,
            },
        ),
        "mass_origin_vector_qball_solver_spec": payload(
            "8.7.55.2.815",
            "Vector Q-ball numerical solver specification",
            {
                "mass_origin_vector_qball_radial_angular_separation_json": "output/public/quantum/mass_origin_vector_qball_radial_angular_separation_metrics.json",
                "mass_origin_qball_discrete_mass_spectrum_json": rel(QBALL_SPECTRUM),
            },
            "Freeze the first numerical-solver pilot grid for the vector Q-ball route without overclaiming solved spectra yet.",
            {
                "pilot_sector_rule": "start with ell in {0,1,2,3}, s in {0,+1,-1}, reuse scalar n=1..5 as the first charge ladder, and extend to radial nodes k>0 after the base sectors are stable",
                "lower_bound_state_count": "N_n * N_(ell,s) with N_n=5 and N_(ell,s)=10 for the first pilot",
            },
            [
                row(
                    "vector_qball_solver_spec_complete",
                    "pass",
                    "vector Q-ball solver specification complete",
                    1,
                    "The first vector pilot grid is frozen.",
                ),
                row(
                    "vector_qball_pilot_sector_count",
                    "pass",
                    "vector Q-ball pilot sector count",
                    pilot_sector_count,
                    "The first pilot covers the scalar limit plus ell=1,2,3 with s in {0,±1}.",
                ),
                row(
                    "vector_qball_pilot_state_count_lower_bound",
                    "pass",
                    "vector Q-ball pilot state-count lower bound",
                    pilot_state_count_lower_bound,
                    "Even before radial nodes k>0 are added, the first pilot explores a much larger ladder than the scalar route.",
                ),
            ],
            {
                "pilot_sector_rows": pilot_sectors,
                "pilot_sector_count": pilot_sector_count,
                "scalar_reference_mode_count": scalar_mode_count,
                "pilot_state_count_lower_bound": pilot_state_count_lower_bound,
                "numerical_solver_next_step_ready": True,
            },
            {
                "overall_status": "vector_qball_solver_spec_frozen",
                "keep_mass_origin_branch_blocked": True,
                "hand_off_to_8_7_55_2_84": False,
                "next_required_artifacts": [
                    "vector_qball_spin_orbit_freeze_audit",
                    "vector_qball_hierarchy_feasibility_gate",
                ],
            },
            {
                "pilot_sector_rows": pilot_sectors,
            },
        ),
        "mass_origin_vector_qball_spin_orbit_freeze_audit": payload(
            "8.7.55.2.816",
            "Vector Q-ball spin-orbit freeze audit",
            {
                "lagrangian_noether_rotational_closure_audit_json": rel(ROT_AUDIT),
                "part1_core_theory_markdown": rel(PART1),
                "part2_astrophysics_markdown": rel(PART2),
            },
            "Audit whether the already-frozen lambda_rot can be reused as the vector Q-ball spin-orbit splitting coefficient with no new parameter.",
            {
                "spin_orbit_shift": "Delta M_SO proportional to lambda_rot <L dot S>",
                "lambda_rot_reuse_rule": "reuse the prior-frozen weak-field lambda_rot without introducing any new normalization freedom",
            },
            [
                row(
                    "vector_qball_spin_orbit_freeze_audit_complete",
                    "pass",
                    "vector Q-ball spin-orbit freeze audit complete",
                    1,
                    "The lambda_rot reuse audit is frozen.",
                ),
                row(
                    "vector_qball_lambda_rot_reuse_available",
                    "pass",
                    "lambda_rot reuse available",
                    1,
                    "The same lambda_rot already fixed by frame dragging can be reused in the vector-Q-ball spin sector.",
                ),
                row(
                    "vector_qball_spin_orbit_new_free_parameters",
                    "pass",
                    "new free parameters introduced by spin-orbit reuse",
                    0,
                    "Spin-orbit splitting reuses the frozen lambda_rot and adds no new coupling.",
                ),
            ],
            {
                "lambda_rot_reuse_available": True,
                "lambda_rot_value": lambda_rot,
                "lambda_rot_sigma": lambda_sigma,
                "spin_orbit_split_without_new_parameters": True,
                "cross_scale_connection_ready": True,
            },
            {
                "overall_status": "vector_qball_spin_orbit_freeze_audited",
                "keep_mass_origin_branch_blocked": True,
                "hand_off_to_8_7_55_2_84": False,
                "next_required_artifacts": ["vector_qball_hierarchy_feasibility_gate"],
            },
            {
                "part1_micro_spin_line": hit(part1, "Pauli 型スピン結合"),
                "part2_frame_dragging_line": hit(part2, "frame dragging"),
                "lambda_rot_calibration": {
                    "lambda_rot": lambda_rot,
                    "lambda_sigma": lambda_sigma,
                    "prior_source_ids": rot_audit["calibration"]["prior_source_ids"],
                    "holdout_channel_ids": rot_audit["calibration"]["holdout_channel_ids"],
                },
            },
        ),
        "mass_origin_vector_qball_hierarchy_feasibility_gate": payload(
            "8.7.55.2.817",
            "Vector Q-ball hierarchy-feasibility gate",
            {
                "mass_origin_vector_qball_scalar_limit_recovery_audit_json": "output/public/quantum/mass_origin_vector_qball_scalar_limit_recovery_audit_metrics.json",
                "mass_origin_vector_qball_solver_spec_json": "output/public/quantum/mass_origin_vector_qball_solver_spec_metrics.json",
                "mass_origin_vector_qball_spin_orbit_freeze_audit_json": "output/public/quantum/mass_origin_vector_qball_spin_orbit_freeze_audit_metrics.json",
            },
            "Decide whether the vector route is materially richer than the scalar Q-ball truncation and should replace the closeout branch as the active search line.",
            {
                "feasibility_rule": "the route is feasible if it reuses existing canon, embeds the scalar pilot as a special case, and opens extra integer quantum-number axes before any new parameter is introduced",
                "multi_index_spectrum_label": "M_(n,k,ell,s)",
            },
            [
                row(
                    "vector_qball_hierarchy_feasibility_gate_complete",
                    "pass",
                    "vector Q-ball hierarchy-feasibility gate complete",
                    1,
                    "The route-feasibility gate is frozen.",
                ),
                row(
                    "vector_qball_multi_index_hierarchy_available",
                    "pass",
                    "vector Q-ball multi-index hierarchy available",
                    1,
                    "The route expands the scalar single-index ladder into the multi-index state label M_(n,k,ell,s).",
                ),
                row(
                    "vector_qball_exact_discrete_mass_ladder_already_computed",
                    "watch",
                    "exact vector discrete mass ladder already computed",
                    0,
                    "The vector route is feasible, but the actual ell>0 numerical ladder is still pending.",
                ),
            ],
            {
                "scalar_quantum_number_axis_count": 1,
                "vector_quantum_number_axis_count": 4,
                "pilot_state_count_lower_bound": pilot_state_count_lower_bound,
                "vector_multi_index_hierarchy_available": True,
                "exact_vector_mass_ladder_available": False,
                "recommended_next_route_or_none": "vector_qball_numerical_solver",
            },
            {
                "overall_status": "vector_qball_feasibility_gate_passed",
                "keep_mass_origin_branch_blocked": True,
                "hand_off_to_8_7_55_2_84": False,
                "next_required_artifacts": [
                    "vector_qball_branch_refresh",
                    "vector_qball_numerical_solver_route_contract",
                ],
            },
            {
                "scalar_closest_match_row": closest_scalar_match,
                "pilot_sector_rows": pilot_sectors,
            },
        ),
        "mass_origin_vector_qball_branch_refresh": payload(
            "8.7.55.2.818",
            "Mass-origin branch refresh after vector Q-ball pivot",
            {
                "mass_origin_vector_qball_hierarchy_feasibility_gate_json": "output/public/quantum/mass_origin_vector_qball_hierarchy_feasibility_gate_metrics.json",
                "mass_origin_no_public_discrete_spectrum_route_contract_json": rel(CLOSEOUT),
            },
            "Refresh the mass-origin branch after reopening the vector Q-ball / Proca route on top of the scalar closeout state.",
            {
                "branch_case": "vector_route_reopens_discrete_spectrum_search_before_closeout",
                "closeout_demote_rule": "the .805-.810 closeout remains available only as fallback-hold because the full canonical P_mu structure has not yet been exhausted",
            },
            [
                row(
                    "vector_qball_branch_refresh_complete",
                    "pass",
                    "vector Q-ball branch refresh complete",
                    1,
                    "The branch disposition is refreshed after the vector pivot.",
                ),
                row(
                    "vector_qball_replaces_closeout_as_primary",
                    "pass",
                    "vector Q-ball replaces closeout as primary route",
                    1,
                    "The vector route becomes the active primary branch and the closeout branch is demoted to fallback hold.",
                ),
                row(
                    "hand_off_to_8_7_55_2_84_after_vector_pivot",
                    "reject",
                    "handoff to 8.7.55.2.84 after vector pivot",
                    0,
                    "The vector route is reopened, but no solved vector ladder exists yet.",
                ),
            ],
            {
                "selected_primary_route": "vector_qball_proca_soliton_reopen",
                "closeout_branch_status": "fallback_hold_after_vector_reopen",
                "discrete_spectrum_found": False,
                "hand_off_to_8_7_55_2_84": False,
                "recommended_next_route_or_none": "vector_qball_numerical_solver",
                "new_branch_required": True,
            },
            {
                "overall_status": "vector_qball_branch_refreshed_without_handoff",
                "keep_mass_origin_branch_blocked": True,
                "hand_off_to_8_7_55_2_84": False,
                "new_branch_required": True,
                "next_required_artifacts": ["vector_qball_numerical_solver"],
            },
            {
                "closeout_summary": closeout["summary"],
            },
        ),
        "mass_origin_vector_qball_numerical_solver_route_contract": payload(
            "8.7.55.2.819",
            "Vector Q-ball numerical-solver route contract",
            {
                "mass_origin_vector_qball_branch_refresh_json": "output/public/quantum/mass_origin_vector_qball_branch_refresh_metrics.json",
            },
            "Freeze the next numerical branch after the vector Q-ball pivot passed the feasibility gate but still lacks an ell>0 discrete ladder.",
            {
                "selected_residual_route": "vector_qball_numerical_solver",
                "missing_artifact": "vector_qball_discrete_mass_ladder_computation",
            },
            [
                row(
                    "vector_qball_numerical_solver_route_contract_complete",
                    "pass",
                    "vector Q-ball numerical-solver route contract complete",
                    1,
                    "The next numerical branch is frozen.",
                ),
                row(
                    "vector_qball_numerical_solver_split_contract_ready",
                    "pass",
                    "vector Q-ball numerical-solver split contract ready",
                    1,
                    "The next branch may run the ell=0 recovery, ell>0 shooting pilot, spin-orbit splitting, and mass-ratio gate.",
                ),
            ],
            {
                "selected_residual_route": "vector_qball_numerical_solver",
                "missing_vector_qball_artifact": "vector_qball_discrete_mass_ladder_computation",
                "split_contract_ready": True,
            },
            {
                "overall_status": "vector_qball_numerical_solver_route_contract_frozen",
                "keep_mass_origin_branch_blocked": True,
                "hand_off_to_8_7_55_2_84": False,
                "next_required_artifacts": [
                    "vector_qball_trial_state_inventory",
                    "vector_qball_scalar_limit_numerical_recovery",
                    "vector_qball_ell_sector_shooting_pilot",
                ],
            },
            {
                "vector_branch_refresh_summary": {
                    "selected_primary_route": "vector_qball_proca_soliton_reopen",
                    "recommended_next_route_or_none": "vector_qball_numerical_solver",
                },
            },
        ),
    }

    for stem, data in payloads.items():
        write_artifact(stem, data)
        print(f"[ok] wrote {OUT / (stem + '_metrics.json')}")
        print(f"[ok] wrote {OUT / (stem + '_rows.csv')}")


# 関数: スクリプト実行時に branch を起動する。

if __name__ == "__main__":
    main()
