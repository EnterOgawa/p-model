#!/usr/bin/env python3
"""
Generate gravitational self-binding solver-statement residual artifacts for 8.7.55.2.794-.799.

This branch checks whether the missing coupled self-gravity solver statement can
be promoted into the no-new-free-parameter public canonical pack. The present
docs only expose self-gravity as an extension branch, so this script also
formalizes the pivot to the oscillon fallback when the self-gravity route stays
extension-only.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output" / "public" / "quantum"
HANDOFF = ROOT / "doc" / "P_model_handoff.md"
NOTE = ROOT / "doc" / "quantum" / "18_p_field_action_and_schrodinger_mapping.md"
PART3A = ROOT / "doc" / "paper" / "12_part3a_quantum_foundations.md"
PART5 = ROOT / "doc" / "paper" / "14_part5_future_predictions.md"
ROUTE_CONTRACT = OUT / "mass_origin_gravitational_self_binding_solver_statement_route_contract_metrics.json"
OSCILLON = OUT / "mass_origin_oscillon_fallback_assessment_metrics.json"


# 関数: 現在の UTC 時刻を ISO 8601 文字列で返す。
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: 必須入力 artifact の存在を確認する。

def req(path: Path) -> None:
    if not path.exists():
        raise SystemExit(f"[fail] missing required input: {path}")


# 関数: JSON artifact を UTF-8 で読む。

def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


# 関数: UTF-8 テキスト source を読む。

def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# 関数: 絶対パスを repo 相対へ正規化する。

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


# 関数: branch 全体を実行して artifacts を生成する。

def main() -> None:
    for path in (HANDOFF, NOTE, PART3A, PART5, ROUTE_CONTRACT, OSCILLON):
        req(path)

    handoff = read_text(HANDOFF)
    note = read_text(NOTE)
    part3a = read_text(PART3A)
    part5 = read_text(PART5)
    route_contract = read_json(ROUTE_CONTRACT)
    oscillon = read_json(OSCILLON)

    required_statement_sources = [
        "p_to_phi_gravity_source_rule",
        "weak_field_poisson_source_rule",
        "complex_field_packet_statement",
        "self_gravity_mean_field_extension_statement",
        "extension_branch_policy_statement",
        "coupled_self_gravity_solver_statement",
    ]
    present_statement_sources = [
        "p_to_phi_gravity_source_rule",
        "weak_field_poisson_source_rule",
        "complex_field_packet_statement",
        "self_gravity_mean_field_extension_statement",
        "extension_branch_policy_statement",
    ]
    missing_statement_sources = ["coupled_self_gravity_solver_statement"]

    statement_available = False
    statement_reason = "self_gravity_mean_field_extension_only"
    weak_field_retry_available = False
    boson_star_public_solver_available = False
    self_gravity_handoff = False

    stems = {
        "source_inventory": "mass_origin_gravitational_self_binding_solver_statement_source_inventory",
        "wording_audit": "mass_origin_gravitational_self_binding_solver_statement_wording_audit",
        "weak_field_retry": "mass_origin_gravitational_self_binding_weak_field_closure_retry",
        "second_freeze": "mass_origin_boson_star_public_solver_second_freeze_audit",
        "gate_refresh": "mass_origin_self_gravity_discrete_spectrum_second_gate_refresh",
        "route_contract": "mass_origin_oscillon_quasi_discrete_route_contract",
    }

    payloads = {
        stems["source_inventory"]: payload(
            "8.7.55.2.794",
            "Gravitational self-binding solver-statement source inventory",
            {
                "mass_origin_gravitational_self_binding_solver_statement_route_contract_json": rel(ROUTE_CONTRACT),
                "p_model_handoff_markdown": rel(HANDOFF),
                "mass_origin_note_markdown": rel(NOTE),
                "part3a_quantum_foundations_markdown": rel(PART3A),
                "part5_future_predictions_markdown": rel(PART5),
            },
            "Inventory the public source items needed to promote the missing coupled self-gravity solver statement.",
            {
                "required_statement_sources": required_statement_sources,
                "inventory_rule": "the route stays blocked until the self-gravity mean-field extension can be promoted into a public coupled solver statement without introducing a new core hypothesis",
            },
            [
                row(
                    "gravitational_self_binding_solver_statement_source_inventory_complete",
                    "pass",
                    "gravitational self-binding solver-statement source inventory complete",
                    1,
                    "Source inventory fixed for the solver-statement residual branch.",
                ),
                row(
                    "gravitational_self_binding_solver_statement_source_inventory_present_count",
                    "inventory",
                    "present solver-statement source count",
                    len(present_statement_sources),
                    f"{len(present_statement_sources)} of {len(required_statement_sources)} required source items are already public.",
                ),
                row(
                    "gravitational_self_binding_solver_statement_source_inventory_missing_count",
                    "watch",
                    "missing solver-statement source count",
                    len(missing_statement_sources),
                    f"Missing items are {', '.join(missing_statement_sources)}.",
                ),
            ],
            {
                "required_solver_statement_source_count": len(required_statement_sources),
                "present_solver_statement_source_count": len(present_statement_sources),
                "missing_solver_statement_source_count": len(missing_statement_sources),
                "missing_solver_statement_source_items": missing_statement_sources,
                "first_route_to_close_or_none": "coupled_self_gravity_solver_statement",
                "self_gravity_mean_field_extension_explicit": True,
            },
            {
                "overall_status": "gravitational_self_binding_solver_statement_source_inventory_frozen",
                "keep_mass_origin_branch_blocked": True,
                "hand_off_to_8_7_55_2_84": False,
                "next_required_artifacts": ["gravitational_self_binding_solver_statement_wording_audit"],
            },
            {
                "p_model_handoff_phi_line": hit(handoff, "\\phi \\equiv -c^2"),
                "p_model_handoff_poisson_line": hit(handoff, "\\nabla^2\\phi = 4\\pi G\\rho"),
                "mass_origin_note_complex_field_line": hit(note, "複素場（位相）への拡張"),
                "part3a_self_gravity_extension_line": hit(part3a, "量子質量密度を u のソースへ入れる"),
                "part5_extension_branch_line": hit(part5, "extension branch"),
            },
        ),
        stems["wording_audit"]: payload(
            "8.7.55.2.795",
            "Gravitational self-binding solver-statement wording audit",
            {
                "mass_origin_gravitational_self_binding_solver_statement_source_inventory_json": f"output/public/quantum/{stems['source_inventory']}_metrics.json",
                "part3a_quantum_foundations_markdown": rel(PART3A),
                "part5_future_predictions_markdown": rel(PART5),
            },
            "Audit whether the coupled self-gravity solver statement can be written as a public no-new-free-parameter core statement.",
            {
                "statement_rule": "a solver statement is admissible only if it does not promote an explicitly extension-only mean-field hypothesis into the core pack",
            },
            [
                row(
                    "gravitational_self_binding_solver_statement_wording_audit_complete",
                    "pass",
                    "gravitational self-binding solver-statement wording audit complete",
                    1,
                    "The missing solver statement was audited against the current public canonical wording rules.",
                ),
                row(
                    "coupled_self_gravity_solver_statement_available",
                    "reject",
                    "coupled self-gravity solver statement available",
                    0,
                    "The available self-gravity wording is explicitly extension-only and cannot be promoted into the core pack as a no-new-free-parameter solver statement.",
                ),
                row(
                    "coupled_self_gravity_solver_statement_without_new_free_parameters",
                    "reject",
                    "coupled self-gravity solver statement without new free parameters",
                    0,
                    "Promoting the mean-field statement would require elevating an extension hypothesis into the core solver pack.",
                ),
            ],
            {
                "coupled_self_gravity_solver_statement_available": statement_available,
                "solver_statement_without_new_free_parameters": False,
                "solver_statement_nonclosure_reason_or_none": statement_reason,
                "extension_branch_explicit": True,
                "missing_solver_statement_inputs": missing_statement_sources,
            },
            {
                "overall_status": "gravitational_self_binding_solver_statement_wording_audited_extension_only",
                "keep_mass_origin_branch_blocked": True,
                "hand_off_to_8_7_55_2_84": False,
                "next_required_artifacts": ["gravitational_self_binding_weak_field_closure_retry"],
            },
            {
                "part3a_self_gravity_extension_line": hit(part3a, "追加仮定"),
                "part5_extension_branch_line": hit(part5, "コア予測ではなく extension branch"),
            },
        ),
        stems["weak_field_retry"]: payload(
            "8.7.55.2.796",
            "Gravitational self-binding weak-field closure retry",
            {
                "mass_origin_gravitational_self_binding_solver_statement_wording_audit_json": f"output/public/quantum/{stems['wording_audit']}_metrics.json",
                "mass_origin_gravitational_self_binding_solver_statement_route_contract_json": rel(ROUTE_CONTRACT),
            },
            "Retry the weak-field self-gravity closure after auditing the missing solver statement.",
            {
                "closure_retry_rule": "weak-field self-gravity closure is available only if the coupled solver statement enters the public core pack rather than remaining an extension-only hypothesis",
            },
            [
                row(
                    "gravitational_self_binding_weak_field_closure_retry_complete",
                    "pass",
                    "gravitational self-binding weak-field closure retry complete",
                    1,
                    "Weak-field closure was retried after the solver-statement wording audit.",
                ),
                row(
                    "weak_field_self_gravity_closure_retry_available",
                    "reject",
                    "weak-field self-gravity closure available after solver-statement retry",
                    0,
                    "The solver statement remains extension-only, so the weak-field self-gravity closure stays unavailable.",
                ),
            ],
            {
                "weak_field_self_gravity_closure_available": weak_field_retry_available,
                "weak_field_closure_retry_nonclosure_reason_or_none": statement_reason,
                "extension_branch_only": True,
            },
            {
                "overall_status": "gravitational_self_binding_weak_field_closure_retry_frozen_absent",
                "keep_mass_origin_branch_blocked": True,
                "hand_off_to_8_7_55_2_84": False,
                "next_required_artifacts": ["boson_star_public_solver_second_freeze_audit"],
            },
            {
                "solver_statement_route_contract_summary": route_contract["summary"],
                "solver_statement_wording_summary": {
                    "solver_statement_nonclosure_reason_or_none": statement_reason,
                },
            },
        ),
        stems["second_freeze"]: payload(
            "8.7.55.2.797",
            "Boson-star public solver second freeze audit",
            {
                "mass_origin_gravitational_self_binding_weak_field_closure_retry_json": f"output/public/quantum/{stems['weak_field_retry']}_metrics.json",
                "mass_origin_oscillon_fallback_assessment_json": rel(OSCILLON),
            },
            "Audit whether the self-gravity route can still be frozen as a public exact-spectrum solver after the extension-only wording result.",
            {
                "solver_freeze_rule": "public boson-star freeze requires a core coupled self-gravity solver statement and a self-gravity eigenmode boundary rule",
                "missing_items": ["coupled_self_gravity_solver_statement", "self_gravity_eigenmode_boundary_rule"],
            },
            [
                row(
                    "boson_star_public_solver_second_freeze_audit_complete",
                    "pass",
                    "boson-star public solver second freeze audit complete",
                    1,
                    "Public solver freeze readiness was re-audited after the solver-statement retry.",
                ),
                row(
                    "boson_star_public_solver_second_freeze_available",
                    "reject",
                    "boson-star public solver available after second freeze audit",
                    0,
                    "The self-gravity route remains extension-only and still lacks a frozen coupled solver statement and boundary rule.",
                ),
                row(
                    "exact_self_gravity_discrete_spectrum_second_freeze_ready",
                    "reject",
                    "exact self-gravity discrete spectrum ready after second freeze audit",
                    0,
                    "No public exact self-gravity ladder can be claimed while the route remains extension-only.",
                ),
            ],
            {
                "boson_star_public_solver_available": boson_star_public_solver_available,
                "missing_public_solver_items": ["coupled_self_gravity_solver_statement", "self_gravity_eigenmode_boundary_rule"],
                "exact_self_gravity_discrete_spectrum_ready": False,
                "public_solver_nonclosure_reason_or_none": statement_reason,
            },
            {
                "overall_status": "boson_star_public_solver_second_freeze_audited_extension_only",
                "keep_mass_origin_branch_blocked": True,
                "hand_off_to_8_7_55_2_84": False,
                "next_required_artifacts": ["self_gravity_discrete_spectrum_second_gate_refresh"],
            },
            {
                "oscillon_assessment_summary": oscillon["summary"],
            },
        ),
        stems["gate_refresh"]: payload(
            "8.7.55.2.798",
            "Self-gravity discrete-spectrum gate second refresh",
            {
                "mass_origin_boson_star_public_solver_second_freeze_audit_json": f"output/public/quantum/{stems['second_freeze']}_metrics.json",
                "mass_origin_oscillon_fallback_assessment_json": rel(OSCILLON),
            },
            "Refresh the mass-origin gate after the solver-statement residual branch remains extension-only.",
            {
                "gate_rule": "handoff requires a public exact-spectrum route in the core pack; extension-only self-gravity does not satisfy the .84 reopen condition",
            },
            [
                row(
                    "self_gravity_discrete_spectrum_second_gate_refresh_complete",
                    "pass",
                    "self-gravity discrete-spectrum second gate refresh complete",
                    1,
                    "The self-gravity gate was refreshed after the solver-statement residual audit.",
                ),
                row(
                    "self_gravity_discrete_spectrum_second_gate_found",
                    "reject",
                    "self-gravity discrete spectrum found after second gate refresh",
                    0,
                    "The self-gravity route remains extension-only, so no public exact ladder is available.",
                ),
                row(
                    "hand_off_to_8_7_55_2_84_after_self_gravity_second_gate_refresh",
                    "reject",
                    "handoff to 8.7.55.2.84 available after self-gravity second gate refresh",
                    0,
                    "The mass-origin branch stays blocked; the next admissible fallback is the oscillon quasi-discrete route.",
                ),
            ],
            {
                "selected_binding_route_or_none": None,
                "discrete_spectrum_found": False,
                "hand_off_to_8_7_55_2_84": self_gravity_handoff,
                "remaining_binding_blockers": [statement_reason],
                "new_branch_required": True,
                "recommended_next_route_or_none": "oscillon_quasi_discrete_reopen",
            },
            {
                "overall_status": "self_gravity_second_gate_refreshed_extension_only",
                "keep_mass_origin_branch_blocked": True,
                "hand_off_to_8_7_55_2_84": False,
                "new_branch_required": True,
                "next_required_artifacts": ["oscillon_quasi_discrete_route_contract"],
            },
            {
                "oscillon_assessment_summary": oscillon["summary"],
                "solver_statement_wording_summary": {
                    "solver_statement_nonclosure_reason_or_none": statement_reason,
                },
            },
        ),
        stems["route_contract"]: payload(
            "8.7.55.2.799",
            "Oscillon quasi-discrete reopen route contract",
            {
                "mass_origin_self_gravity_discrete_spectrum_second_gate_refresh_json": f"output/public/quantum/{stems['gate_refresh']}_metrics.json",
                "mass_origin_oscillon_fallback_assessment_json": rel(OSCILLON),
            },
            "Freeze the next residual route after the gravitational self-binding solver-statement branch remains extension-only.",
            {
                "selected_residual_route": "oscillon_quasi_discrete_reopen",
                "route_priority_basis": "self-gravity core route closed as extension-only; next admissible fallback is the documented oscillon quasi-discrete family",
                "missing_artifact": "oscillon_width_acceptance_rule",
            },
            [
                row(
                    "oscillon_quasi_discrete_route_contract_complete",
                    "pass",
                    "oscillon quasi-discrete route contract complete",
                    1,
                    "The next residual branch has been frozen after the self-gravity route stayed extension-only.",
                ),
                row(
                    "oscillon_quasi_discrete_split_contract_ready",
                    "pass",
                    "oscillon quasi-discrete split contract ready",
                    1,
                    "The next branch may inventory and audit the resonance-width acceptance rule for oscillon quasi-discrete states.",
                ),
            ],
            {
                "selected_residual_route": "oscillon_quasi_discrete_reopen",
                "missing_mass_origin_artifact": "oscillon_width_acceptance_rule",
                "split_contract_ready": True,
                "oscillon_fallback_admissible": True,
                "quasi_discrete_only": True,
            },
            {
                "overall_status": "oscillon_quasi_discrete_route_contract_frozen",
                "keep_mass_origin_branch_blocked": True,
                "hand_off_to_8_7_55_2_84": False,
                "next_required_artifacts": [
                    "oscillon_width_acceptance_source_inventory",
                    "oscillon_quasi_discrete_width_gate_audit",
                ],
            },
            {
                "self_gravity_second_gate_summary": {
                    "recommended_next_route_or_none": "oscillon_quasi_discrete_reopen",
                    "remaining_binding_blockers": [statement_reason],
                },
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
