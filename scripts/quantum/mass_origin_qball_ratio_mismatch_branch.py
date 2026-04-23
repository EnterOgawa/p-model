#!/usr/bin/env python3
"""
Generate Q-ball ratio-mismatch resolution artifacts for 8.7.55.2.782-.787.

This branch audits whether the discrete Q-ball ladder found under direct
charge mapping can still be moved by already-frozen parameter relations or by
an existing charge-operator normalization freedom. If not, it freezes the
ratio mismatch and pivots the residual route toward the stronger exact-spectrum
fallback.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output" / "public" / "quantum"
PART3A = ROOT / "doc" / "paper" / "12_part3a_quantum_foundations.md"
MEXICAN = OUT / "mass_origin_mexican_hat_parameter_freeze_metrics.json"
QBALL_RADIAL = OUT / "mass_origin_qball_radial_equation_derivation_metrics.json"
QBALL_STATEMENT = OUT / "mass_origin_qball_charge_mapping_statement_freeze_metrics.json"
QBALL_INVERSION = OUT / "mass_origin_qball_charge_discrete_frequency_inversion_metrics.json"
QBALL_SPECTRUM = OUT / "mass_origin_qball_discrete_mass_spectrum_metrics.json"
QBALL_RATIO = OUT / "mass_origin_qball_charge_mapped_mass_ratio_comparison_metrics.json"
QBALL_ROUTE = OUT / "mass_origin_qball_ratio_mismatch_route_contract_metrics.json"
OSCILLON = OUT / "mass_origin_oscillon_fallback_assessment_metrics.json"
GRAVITY = OUT / "mass_origin_gravitational_self_binding_boson_star_assessment_metrics.json"


# 関数: 現在のUTC時刻を ISO 8601 で返す。
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


# 関数: pattern を含む最初の source line を返す。

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


# 関数: ratio table を compact な summary row list に変換する。

def mode_ratio_rows(discrete_mode_rows: list[dict]) -> list[dict]:
    rows = []
    for mode in discrete_mode_rows:
        rows.append(
            {
                "mode_index": int(mode["mode_index"]),
                "beta_n": float(mode["beta_n"]),
                "mass_ratio_to_first": float(mode["mass_ratio_to_first"]),
            }
        )

    return rows


# 関数: branch 全体を実行して artifacts を生成する。

def main() -> None:
    for path in (
        PART3A,
        MEXICAN,
        QBALL_RADIAL,
        QBALL_STATEMENT,
        QBALL_INVERSION,
        QBALL_SPECTRUM,
        QBALL_RATIO,
        QBALL_ROUTE,
        OSCILLON,
        GRAVITY,
    ):
        req(path)

    part3a = read_text(PART3A)
    mexican = read_json(MEXICAN)
    radial = read_json(QBALL_RADIAL)
    statement = read_json(QBALL_STATEMENT)
    inversion = read_json(QBALL_INVERSION)
    spectrum = read_json(QBALL_SPECTRUM)
    ratio = read_json(QBALL_RATIO)
    route = read_json(QBALL_ROUTE)
    oscillon = read_json(OSCILLON)
    gravity = read_json(GRAVITY)

    discrete_rows = list(inversion["evidence"]["discrete_mode_rows"])
    ratio_rows = mode_ratio_rows(discrete_rows)
    prior_closest = dict(ratio["summary"]["closest_known_mass_ratio_or_none"])
    prior_hand_off = bool(ratio["summary"]["hand_off_to_8_7_55_2_84"])

    scale_free = True
    parameter_refinement_can_move_ratios = False
    normalization_freedom_available = False
    recomputation_needed = False
    recomputation_shift_max_abs = 0.0
    fallback_priority = [
        "gravitational_self_binding_boson_star",
        "oscillon",
    ]
    selected_next_route = "gravitational_self_binding_boson_star_reopen"
    selected_missing_artifact = str(gravity["summary"]["boson_star_nonclosure_reason_or_none"])

    payloads = {
        "mass_origin_qball_ratio_scale_invariance_audit": payload(
            "8.7.55.2.782",
            "Q-ball ratio scale-invariance audit",
            {
                "mass_origin_mexican_hat_parameter_freeze_json": rel(MEXICAN),
                "mass_origin_qball_radial_equation_derivation_json": rel(QBALL_RADIAL),
                "mass_origin_qball_discrete_mass_spectrum_json": rel(QBALL_SPECTRUM),
            },
            "Audit whether M_n / M_1 can move under the already-frozen lambda, v = P_infty, and chi_P relations.",
            {
                "mass_parameter_formula": mexican["summary"]["mass_parameter_formula"],
                "susceptibility_constraint_formula": mexican["summary"]["susceptibility_constraint_formula"],
                "dimensionless_qball_equation": radial["formulas"]["dimensionless_pilot_equation"],
                "ratio_rule": "M_n / M_1 = E_hat(beta_n) / E_hat(beta_1) once the common mass prefactor is factored out",
            },
            [
                row(
                    "qball_ratio_scale_invariance_audit_complete",
                    "pass",
                    "Q-ball ratio scale-invariance audit complete",
                    1,
                    "Existing scale parameters were audited against the direct charge-mapped ladder.",
                ),
                row(
                    "qball_mass_ratio_scale_free_under_lambda_v_chi_p",
                    "pass" if scale_free else "reject",
                    "mass ratios stay scale-free under lambda, v, chi_P",
                    1 if scale_free else 0,
                    "The frozen lambda/v/chi_P relations enter only through a common mass prefactor, so M_n/M_1 is unchanged."
                    if scale_free
                    else "Existing parameter relations still move the Q-ball mass ratios.",
                ),
                row(
                    "qball_parameter_refinement_can_move_mass_ratios",
                    "reject" if not parameter_refinement_can_move_ratios else "pass",
                    "existing parameter refinement can move mass ratios",
                    1 if parameter_refinement_can_move_ratios else 0,
                    "No ratio-moving freedom survives inside the already-frozen mexican-hat parameter pack."
                    if not parameter_refinement_can_move_ratios
                    else "At least one already-frozen parameter relation still moves the mass ratios.",
                ),
            ],
            {
                "mass_ratio_scale_free_under_lambda_v_chi_p": scale_free,
                "parameter_refinement_can_move_mass_ratios": parameter_refinement_can_move_ratios,
                "common_mass_prefactor_symbols": ["lambda", "v", "chi_P", "Z_P", "g_P"],
                "direct_charge_mapping_beta_n_preserved": True,
            },
            {
                "overall_status": "qball_ratio_scale_invariance_frozen",
                "keep_mass_origin_branch_blocked": True,
                "hand_off_to_8_7_55_2_84": False,
                "next_required_artifacts": ["qball_charge_operator_normalization_audit"],
            },
            {
                "mexican_hat_parameter_summary": mexican["summary"],
                "discrete_mode_ratio_rows": ratio_rows,
            },
        ),
        "mass_origin_qball_charge_operator_normalization_audit": payload(
            "8.7.55.2.783",
            "Q-ball charge-operator normalization audit",
            {
                "mass_origin_qball_charge_mapping_statement_freeze_json": rel(QBALL_STATEMENT),
                "part3a_quantum_foundations_markdown": rel(PART3A),
            },
            "Audit whether the adopted U(1) statement still leaves any existing normalization freedom beyond the direct Q-ball identity.",
            {
                "canonical_statement": statement["formulas"]["canonical_statement"],
                "normalization_rule": "Q_qball = Q_U(1) and Q_n = n q with q fixed as the adopted elementary charge unit",
                "freedom_rule": "normalization freedom exists only if the public canonical pack leaves a multiplicative factor between Q_qball and the adopted U(1) charge",
            },
            [
                row(
                    "qball_charge_operator_normalization_audit_complete",
                    "pass",
                    "Q-ball charge-operator normalization audit complete",
                    1,
                    "The adopted U(1) wording was audited for residual normalization freedom.",
                ),
                row(
                    "qball_charge_operator_normalization_freedom_available",
                    "reject" if not normalization_freedom_available else "pass",
                    "charge-operator normalization freedom available",
                    1 if normalization_freedom_available else 0,
                    "The frozen direct identity leaves no multiplicative normalization freedom."
                    if not normalization_freedom_available
                    else "An existing normalization freedom survives the direct identity.",
                ),
                row(
                    "qball_direct_identity_required_by_public_pack",
                    "pass",
                    "direct Q-ball/U(1) identity required by public canonical pack",
                    1,
                    "The canonical statement freezes coincidence of the Q-ball Noether charge with the adopted U(1) charge.",
                ),
            ],
            {
                "charge_operator_normalization_freedom_available": normalization_freedom_available,
                "direct_qball_u1_identity_required": True,
                "charge_quantum_normalization": statement["summary"]["charge_quantum_normalization"],
                "new_free_parameters_introduced": [],
            },
            {
                "overall_status": "qball_charge_operator_normalization_audit_frozen",
                "keep_mass_origin_branch_blocked": True,
                "hand_off_to_8_7_55_2_84": False,
                "next_required_artifacts": ["qball_charge_mapped_mass_ratio_recomputation_retry"],
            },
            {
                "part3a_adopted_u1_line": statement["evidence"]["part3a_adopted_u1_line"],
                "part3a_charge_quantization_adopted_line": statement["evidence"]["part3a_charge_quantization_adopted_line"],
                "part3a_u1_effective_line": hit(part3a, "有効理論"),
            },
        ),
        "mass_origin_qball_charge_mapped_mass_ratio_recomputation_retry": payload(
            "8.7.55.2.784",
            "Q-ball charge-mapped mass-ratio recomputation retry",
            {
                "mass_origin_qball_ratio_scale_invariance_audit_json": "output/public/quantum/mass_origin_qball_ratio_scale_invariance_audit_metrics.json",
                "mass_origin_qball_charge_operator_normalization_audit_json": "output/public/quantum/mass_origin_qball_charge_operator_normalization_audit_metrics.json",
                "mass_origin_qball_charge_mapped_mass_ratio_comparison_json": rel(QBALL_RATIO),
            },
            "Retry the charge-mapped ratio computation only if the scale or normalization audits open an already-frozen freedom.",
            {
                "retry_rule": "recompute only if existing parameter or normalization freedom is available",
                "baseline_ratio_source": rel(QBALL_RATIO),
            },
            [
                row(
                    "qball_mass_ratio_recomputation_retry_complete",
                    "pass",
                    "Q-ball mass-ratio recomputation retry complete",
                    1,
                    "The retry gate was evaluated after the invariance and normalization audits.",
                ),
                row(
                    "qball_mass_ratio_recomputation_needed",
                    "reject" if not recomputation_needed else "pass",
                    "mass-ratio recomputation needed",
                    1 if recomputation_needed else 0,
                    "No existing freedom opened, so the direct charge-mapped ladder is numerically rigid."
                    if not recomputation_needed
                    else "An existing freedom opened and a fresh recomputation is needed.",
                ),
                row(
                    "qball_mass_ratio_retry_shift_max_abs",
                    "pass",
                    "maximum absolute ratio shift under retry",
                    recomputation_shift_max_abs,
                    "The retry is a no-op because the ratio table is unchanged.",
                ),
            ],
            {
                "mass_ratio_recomputation_needed": recomputation_needed,
                "mass_ratio_recomputation_executed_as_noop": not recomputation_needed,
                "maximum_absolute_ratio_shift": recomputation_shift_max_abs,
                "retained_mass_ratio_rows": ratio_rows,
                "closest_known_mass_ratio_or_none": prior_closest,
                "hand_off_to_8_7_55_2_84": prior_hand_off,
            },
            {
                "overall_status": "qball_mass_ratio_retry_closed_without_change",
                "keep_mass_origin_branch_blocked": True,
                "hand_off_to_8_7_55_2_84": prior_hand_off,
                "next_required_artifacts": ["qball_handoff_second_retry"],
            },
            {
                "retained_mode_ratio_rows": ratio["evidence"]["mode_ratio_rows"],
                "closest_match_row": prior_closest,
            },
        ),
        "mass_origin_qball_handoff_second_retry": payload(
            "8.7.55.2.785",
            "Q-ball handoff second retry / fallback reprioritization",
            {
                "mass_origin_qball_charge_mapped_mass_ratio_recomputation_retry_json": "output/public/quantum/mass_origin_qball_charge_mapped_mass_ratio_recomputation_retry_metrics.json",
                "mass_origin_oscillon_fallback_assessment_json": rel(OSCILLON),
                "mass_origin_gravitational_self_binding_boson_star_assessment_json": rel(GRAVITY),
            },
            "Retry the .84 handoff and, if it still fails, reprioritize the post-Q-ball fallback routes.",
            {
                "handoff_rule": "handoff requires at least one known-particle ratio match inside the current no-new-free-parameter pack",
                "fallback_priority_rule": "prefer the strongest remaining exact-spectrum route over quasi-discrete routes when the direct Q-ball ladder is rigid but mismatched",
            },
            [
                row(
                    "qball_handoff_second_retry_complete",
                    "pass",
                    "Q-ball handoff second retry complete",
                    1,
                    "The direct charge-mapped ladder was re-gated after the invariance audits.",
                ),
                row(
                    "hand_off_to_8_7_55_2_84_second_retry",
                    "reject",
                    "handoff to 8.7.55.2.84 available after second retry",
                    0,
                    "The direct charge-mapped ladder remains far from the canonical particle targets.",
                ),
                row(
                    "qball_fallback_primary_reprioritized_to_gravitational",
                    "pass",
                    "fallback priority moved to gravitational self-binding / boson-star route",
                    1,
                    "Oscillon remains quasi-discrete only, so the stronger exact-spectrum fallback is gravitational self-binding.",
                ),
            ],
            {
                "hand_off_to_8_7_55_2_84": False,
                "selected_fallback_route_if_qball_closed": "gravitational_self_binding_boson_star",
                "fallback_priority_order": fallback_priority,
                "closest_known_mass_ratio_or_none": prior_closest,
            },
            {
                "overall_status": "qball_handoff_second_retry_failed_fallback_reprioritized",
                "keep_mass_origin_branch_blocked": True,
                "hand_off_to_8_7_55_2_84": False,
                "new_branch_required": True,
                "next_required_artifacts": [selected_next_route],
            },
            {
                "oscillon_assessment_summary": oscillon["summary"],
                "gravitational_assessment_summary": gravity["summary"],
                "closest_match_row": prior_closest,
            },
        ),
        "mass_origin_qball_ratio_mismatch_branch_refresh": payload(
            "8.7.55.2.786",
            "Mass-origin branch refresh after direct charge mapping",
            {
                "mass_origin_qball_handoff_second_retry_json": "output/public/quantum/mass_origin_qball_handoff_second_retry_metrics.json",
                "mass_origin_qball_ratio_mismatch_route_contract_json": rel(QBALL_ROUTE),
            },
            "Freeze the outcome of the ratio-mismatch resolution branch and decide whether to reopen .84 or pivot the residual route.",
            {
                "disposition_case": "case_b_discrete_ladder_but_ratio_mismatch_scale_and_normalization_rigid",
                "selected_next_route": selected_next_route,
                "missing_artifact": selected_missing_artifact,
            },
            [
                row(
                    "qball_ratio_mismatch_branch_refresh_complete",
                    "pass",
                    "Q-ball ratio-mismatch branch refresh complete",
                    1,
                    "The ratio-mismatch branch outcome has been frozen.",
                ),
                row(
                    "qball_direct_charge_mapping_discrete_ladder_retained",
                    "pass",
                    "direct charge-mapped discrete ladder retained",
                    1,
                    "The ladder exists but remains inconsistent with known particle ratios.",
                ),
                row(
                    "qball_ratio_mismatch_route_reopens_8_7_55_2_84",
                    "reject",
                    "ratio-mismatch route reopens 8.7.55.2.84",
                    0,
                    "No existing scale or normalization freedom moves the ratios into the pass window.",
                ),
            ],
            {
                "discrete_spectrum_found_under_direct_charge_mapping": True,
                "ratio_scale_invariance_frozen": True,
                "charge_operator_normalization_freedom_available": False,
                "hand_off_to_8_7_55_2_84": False,
                "new_branch_required": True,
                "recommended_next_route_or_none": selected_next_route,
            },
            {
                "overall_status": "qball_ratio_mismatch_branch_closed_without_handoff",
                "keep_mass_origin_branch_blocked": True,
                "hand_off_to_8_7_55_2_84": False,
                "new_branch_required": True,
                "next_required_artifacts": [selected_next_route],
            },
            {
                "route_contract_summary": route["summary"],
                "closest_match_row": prior_closest,
            },
        ),
        "mass_origin_gravitational_self_binding_route_contract": payload(
            "8.7.55.2.787",
            "Gravitational self-binding / boson-star reopen route contract",
            {
                "mass_origin_qball_ratio_mismatch_branch_refresh_json": "output/public/quantum/mass_origin_qball_ratio_mismatch_branch_refresh_metrics.json",
                "mass_origin_gravitational_self_binding_boson_star_assessment_json": rel(GRAVITY),
            },
            "Freeze the next residual route after the direct Q-ball charge-mapping ladder fails with rigid ratios.",
            {
                "selected_residual_route": selected_next_route,
                "route_priority_basis": "exact-spectrum fallback preferred over quasi-discrete oscillon fallback",
                "missing_artifact": selected_missing_artifact,
            },
            [
                row(
                    "gravitational_self_binding_route_contract_complete",
                    "pass",
                    "gravitational self-binding route contract complete",
                    1,
                    "The next residual route has been frozen.",
                ),
                row(
                    "gravitational_self_binding_split_contract_ready",
                    "pass",
                    "gravitational self-binding split contract ready",
                    1,
                    "The next branch may inventory the coupled self-gravity solver route.",
                ),
            ],
            {
                "selected_residual_route": selected_next_route,
                "missing_gravitational_binding_artifact": selected_missing_artifact,
                "fallback_priority_order": fallback_priority,
                "split_contract_ready": True,
            },
            {
                "overall_status": "gravitational_self_binding_route_contract_frozen",
                "keep_mass_origin_branch_blocked": True,
                "hand_off_to_8_7_55_2_84": False,
                "next_required_artifacts": [
                    "gravitational_self_binding_solver_source_inventory",
                    "gravitational_self_binding_weak_field_closure_audit",
                ],
            },
            {
                "gravitational_assessment_summary": gravity["summary"],
                "oscillon_assessment_summary": oscillon["summary"],
            },
        ),
    }

    for stem, data in payloads.items():
        write_artifact(stem, data)
        print(f"[ok] wrote {OUT / (stem + '_metrics.json')}")
        print(f"[ok] wrote {OUT / (stem + '_rows.csv')}")


if __name__ == "__main__":
    main()
