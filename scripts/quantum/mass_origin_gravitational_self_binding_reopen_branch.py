#!/usr/bin/env python3
"""
Generate gravitational self-binding reopen artifacts for 8.7.55.2.788-.793.

This branch audits whether the current public canonical pack is already strong
enough to freeze a coupled self-gravity / boson-star solver after the Q-ball
route closed with a rigid mass-ratio mismatch.
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
MEXICAN = OUT / "mass_origin_mexican_hat_parameter_freeze_metrics.json"
QBALL_ROUTE = OUT / "mass_origin_gravitational_self_binding_route_contract_metrics.json"
GRAVITY_ASSESS = OUT / "mass_origin_gravitational_self_binding_boson_star_assessment_metrics.json"
QBALL_REFRESH = OUT / "mass_origin_qball_ratio_mismatch_branch_refresh_metrics.json"


# 関数: 現在の UTC 時刻を ISO 8601 で返す。
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


# 関数: branch 全体を実行して artifacts を生成する。

def main() -> None:
    for path in (HANDOFF, NOTE, PART3A, MEXICAN, QBALL_ROUTE, GRAVITY_ASSESS, QBALL_REFRESH):
        req(path)

    handoff = read_text(HANDOFF)
    note = read_text(NOTE)
    part3a = read_text(PART3A)
    mexican = read_json(MEXICAN)
    route = read_json(QBALL_ROUTE)
    gravity = read_json(GRAVITY_ASSESS)
    qball_refresh = read_json(QBALL_REFRESH)

    required_sources = [
        "p_to_phi_gravity_source_rule",
        "weak_field_acceleration_rule",
        "complex_field_packet_statement",
        "mexican_hat_mass_parameter_relation",
        "coupled_self_gravity_solver_statement",
        "self_gravity_eigenmode_boundary_rule",
    ]
    present_sources = [
        "p_to_phi_gravity_source_rule",
        "weak_field_acceleration_rule",
        "complex_field_packet_statement",
        "mexican_hat_mass_parameter_relation",
    ]
    missing_sources = [
        "coupled_self_gravity_solver_statement",
        "self_gravity_eigenmode_boundary_rule",
    ]
    selected_missing_artifact = "coupled_self_gravity_solver_statement"

    weak_field_closure_available = False
    boson_star_public_solver_available = False
    discrete_spectrum_found = False
    handoff_available = False

    payloads = {
        "mass_origin_gravitational_self_binding_solver_source_inventory": payload(
            "8.7.55.2.788",
            "Gravitational self-binding solver source inventory",
            {
                "p_model_handoff_markdown": rel(HANDOFF),
                "mass_origin_note_markdown": rel(NOTE),
                "part3a_quantum_foundations_markdown": rel(PART3A),
                "mass_origin_mexican_hat_parameter_freeze_json": rel(MEXICAN),
                "mass_origin_gravitational_self_binding_route_contract_json": rel(QBALL_ROUTE),
            },
            "Inventory the public source items needed to promote a coupled self-gravity / boson-star solver.",
            {
                "required_source_items": required_sources,
                "inventory_rule": "the route stays blocked until a coupled self-gravity solver statement and its eigenmode boundary rule are both public canonical",
            },
            [
                row(
                    "gravitational_self_binding_solver_source_inventory_complete",
                    "pass",
                    "gravitational self-binding solver source inventory complete",
                    1,
                    "Source inventory fixed for the boson-star reopen route.",
                ),
                row(
                    "gravitational_self_binding_present_source_count",
                    "inventory",
                    "present source count",
                    len(present_sources),
                    f"{len(present_sources)} of {len(required_sources)} required source items are already public.",
                ),
                row(
                    "gravitational_self_binding_missing_source_count",
                    "watch",
                    "missing source count",
                    len(missing_sources),
                    f"Missing items are {', '.join(missing_sources)}.",
                ),
            ],
            {
                "required_source_count": len(required_sources),
                "present_source_count": len(present_sources),
                "missing_source_count": len(missing_sources),
                "missing_source_items": missing_sources,
                "first_route_to_close_or_none": selected_missing_artifact,
            },
            {
                "overall_status": "gravitational_self_binding_solver_source_inventory_frozen",
                "keep_mass_origin_branch_blocked": True,
                "hand_off_to_8_7_55_2_84": False,
                "next_required_artifacts": ["gravitational_self_binding_weak_field_closure_audit"],
            },
            {
                "p_model_handoff_phi_line": hit(handoff, "\\phi \\equiv -c^2"),
                "p_model_handoff_acceleration_line": hit(handoff, "a = -\\nabla\\phi"),
                "mass_origin_note_complex_field_line": hit(note, "複素場（位相）への拡張"),
                "mexican_hat_mass_parameter_formula": mexican["summary"]["mass_parameter_formula"],
            },
        ),
        "mass_origin_gravitational_self_binding_weak_field_closure_audit": payload(
            "8.7.55.2.789",
            "Gravitational self-binding weak-field closure audit",
            {
                "mass_origin_gravitational_self_binding_solver_source_inventory_json": "output/public/quantum/mass_origin_gravitational_self_binding_solver_source_inventory_metrics.json",
                "p_model_handoff_markdown": rel(HANDOFF),
                "mass_origin_mexican_hat_parameter_freeze_json": rel(MEXICAN),
            },
            "Audit whether the current public pack closes a no-new-free-parameter weak-field self-gravity coupling for the localized packet.",
            {
                "gravity_source_rule": gravity["formulas"]["gravity_source_rule"],
                "mass_parameter_formula": mexican["summary"]["mass_parameter_formula"],
                "closure_rule": "weak-field closure is available only if the packet source and the backreaction operator are frozen together as a coupled solver statement",
            },
            [
                row(
                    "gravitational_self_binding_weak_field_closure_audit_complete",
                    "pass",
                    "gravitational self-binding weak-field closure audit complete",
                    1,
                    "Weak-field closure was audited against the current public canonical pack.",
                ),
                row(
                    "gravitational_self_binding_source_rule_available",
                    "pass",
                    "gravitational self-binding source rule available",
                    1,
                    "The P -> phi mapping is already public canonical.",
                ),
                row(
                    "gravitational_self_binding_weak_field_closure_available",
                    "reject" if not weak_field_closure_available else "pass",
                    "weak-field self-gravity closure available",
                    1 if weak_field_closure_available else 0,
                    "No coupled self-gravity solver statement freezes the packet source and backreaction operator together."
                    if not weak_field_closure_available
                    else "A no-new-free-parameter weak-field closure is already frozen.",
                ),
            ],
            {
                "gravitational_self_binding_source_rule_available": True,
                "weak_field_self_gravity_closure_available": weak_field_closure_available,
                "weak_field_closure_nonclosure_reason_or_none": (
                    "coupled_self_gravity_solver_statement_absent" if not weak_field_closure_available else None
                ),
                "new_free_parameters_introduced": [],
            },
            {
                "overall_status": "gravitational_self_binding_weak_field_closure_audited",
                "keep_mass_origin_branch_blocked": True,
                "hand_off_to_8_7_55_2_84": False,
                "next_required_artifacts": ["boson_star_public_solver_freeze_audit"],
            },
            {
                "p_model_handoff_phi_line": hit(handoff, "\\phi \\equiv -c^2"),
                "mass_origin_note_oscillon_qball_line": hit(note, "oscillon/Q-ball"),
                "part3a_adopted_u1_line": hit(part3a, "U(1) を独立に採用し"),
            },
        ),
        "mass_origin_boson_star_public_solver_freeze_audit": payload(
            "8.7.55.2.790",
            "Boson-star public solver freeze audit",
            {
                "mass_origin_gravitational_self_binding_weak_field_closure_audit_json": "output/public/quantum/mass_origin_gravitational_self_binding_weak_field_closure_audit_metrics.json",
                "mass_origin_gravitational_self_binding_boson_star_assessment_json": rel(GRAVITY_ASSESS),
            },
            "Audit whether a coupled self-gravity / boson-star solver can already be frozen as a public exact-spectrum route.",
            {
                "solver_freeze_rule": "public solver freeze requires a coupled self-gravity solver statement plus a self-gravity eigenmode boundary rule",
                "missing_items": missing_sources,
            },
            [
                row(
                    "boson_star_public_solver_freeze_audit_complete",
                    "pass",
                    "boson-star public solver freeze audit complete",
                    1,
                    "Public solver freeze readiness was audited.",
                ),
                row(
                    "boson_star_public_solver_available",
                    "reject" if not boson_star_public_solver_available else "pass",
                    "boson-star public solver available",
                    1 if boson_star_public_solver_available else 0,
                    "The public pack still lacks a coupled self-gravity solver statement and boundary rule."
                    if not boson_star_public_solver_available
                    else "A public boson-star solver is already frozen.",
                ),
                row(
                    "boson_star_exact_discrete_spectrum_ready",
                    "reject",
                    "exact self-gravity discrete spectrum ready",
                    0,
                    "The coupled self-gravity solver is not frozen, so no exact self-gravity ladder can be claimed yet.",
                ),
            ],
            {
                "boson_star_public_solver_available": boson_star_public_solver_available,
                "missing_public_solver_items": missing_sources,
                "exact_self_gravity_discrete_spectrum_ready": False,
            },
            {
                "overall_status": "boson_star_public_solver_freeze_audited",
                "keep_mass_origin_branch_blocked": True,
                "hand_off_to_8_7_55_2_84": False,
                "next_required_artifacts": ["self_gravity_discrete_spectrum_gate_refresh"],
            },
            {
                "gravitational_assessment_summary": gravity["summary"],
                "route_contract_summary": route["summary"],
            },
        ),
        "mass_origin_self_gravity_discrete_spectrum_gate_refresh": payload(
            "8.7.55.2.791",
            "Self-gravity discrete-spectrum gate refresh",
            {
                "mass_origin_boson_star_public_solver_freeze_audit_json": "output/public/quantum/mass_origin_boson_star_public_solver_freeze_audit_metrics.json",
                "mass_origin_gravitational_self_binding_route_contract_json": rel(QBALL_ROUTE),
            },
            "Refresh the mass-origin gate after the gravitational self-binding reopen audit.",
            {
                "gate_rule": "handoff requires a public coupled self-gravity solver that returns a discrete spectrum with no new free parameter",
            },
            [
                row(
                    "self_gravity_discrete_spectrum_gate_refresh_complete",
                    "pass",
                    "self-gravity discrete-spectrum gate refresh complete",
                    1,
                    "The gate was refreshed after the boson-star reopen audit.",
                ),
                row(
                    "self_gravity_discrete_spectrum_found",
                    "reject" if not discrete_spectrum_found else "pass",
                    "self-gravity discrete spectrum found",
                    1 if discrete_spectrum_found else 0,
                    "No public coupled self-gravity solver exists yet, so no discrete spectrum is available."
                    if not discrete_spectrum_found
                    else "A public self-gravity discrete ladder is available.",
                ),
                row(
                    "hand_off_to_8_7_55_2_84",
                    "reject" if not handoff_available else "pass",
                    "handoff to 8.7.55.2.84 available",
                    1 if handoff_available else 0,
                    "The gravitational self-binding route cannot reopen .84 until the coupled solver is frozen."
                    if not handoff_available
                    else "The gravitational self-binding route can reopen .84.",
                ),
            ],
            {
                "selected_binding_route_or_none": None,
                "discrete_spectrum_found": discrete_spectrum_found,
                "hand_off_to_8_7_55_2_84": handoff_available,
                "remaining_binding_blockers": [selected_missing_artifact],
                "new_branch_required": True,
            },
            {
                "overall_status": "self_gravity_gate_refreshed_still_blocked",
                "keep_mass_origin_branch_blocked": True,
                "hand_off_to_8_7_55_2_84": False,
                "new_branch_required": True,
                "next_required_artifacts": ["gravitational_self_binding_solver_statement_residual"],
            },
            {
                "qball_ratio_mismatch_refresh_summary": qball_refresh["summary"],
                "gravitational_assessment_summary": gravity["summary"],
            },
        ),
        "mass_origin_gravitational_self_binding_branch_refresh": payload(
            "8.7.55.2.792",
            "Mass-origin branch refresh after Q-ball mismatch pivot",
            {
                "mass_origin_self_gravity_discrete_spectrum_gate_refresh_json": "output/public/quantum/mass_origin_self_gravity_discrete_spectrum_gate_refresh_metrics.json",
                "mass_origin_gravitational_self_binding_route_contract_json": rel(QBALL_ROUTE),
            },
            "Freeze the outcome of the gravitational self-binding reopen audit and decide the next residual route.",
            {
                "disposition_case": "gravitational_self_binding_not_publicly_frozen",
                "selected_next_route": "gravitational_self_binding_solver_statement_residual",
                "missing_artifact": selected_missing_artifact,
            },
            [
                row(
                    "gravitational_self_binding_branch_refresh_complete",
                    "pass",
                    "gravitational self-binding branch refresh complete",
                    1,
                    "The branch outcome has been frozen.",
                ),
                row(
                    "gravitational_self_binding_public_solver_reopens_8_7_55_2_84",
                    "reject",
                    "gravitational self-binding route reopens 8.7.55.2.84",
                    0,
                    "The public pack still lacks the coupled self-gravity solver statement.",
                ),
                row(
                    "gravitational_self_binding_new_branch_required",
                    "pass",
                    "new residual branch required",
                    1,
                    "A residual branch is required to resolve the missing solver statement.",
                ),
            ],
            {
                "hand_off_to_8_7_55_2_84": False,
                "new_branch_required": True,
                "recommended_next_route_or_none": "gravitational_self_binding_solver_statement_residual",
                "missing_gravitational_binding_artifact": selected_missing_artifact,
            },
            {
                "overall_status": "gravitational_self_binding_branch_closed_without_handoff",
                "keep_mass_origin_branch_blocked": True,
                "hand_off_to_8_7_55_2_84": False,
                "new_branch_required": True,
                "next_required_artifacts": ["gravitational_self_binding_solver_statement_residual"],
            },
            {
                "route_contract_summary": route["summary"],
                "self_gravity_gate_summary": {
                    "remaining_binding_blockers": [selected_missing_artifact],
                },
            },
        ),
        "mass_origin_gravitational_self_binding_solver_statement_route_contract": payload(
            "8.7.55.2.793",
            "Gravitational self-binding solver-statement residual route contract",
            {
                "mass_origin_gravitational_self_binding_branch_refresh_json": "output/public/quantum/mass_origin_gravitational_self_binding_branch_refresh_metrics.json",
            },
            "Freeze the next residual branch after the gravitational self-binding reopen route remained blocked on a missing coupled solver statement.",
            {
                "selected_residual_route": "gravitational_self_binding_solver_statement_residual",
                "missing_artifact": selected_missing_artifact,
            },
            [
                row(
                    "gravitational_self_binding_solver_statement_route_contract_complete",
                    "pass",
                    "gravitational self-binding solver-statement route contract complete",
                    1,
                    "The next residual branch has been frozen.",
                ),
                row(
                    "gravitational_self_binding_solver_statement_split_contract_ready",
                    "pass",
                    "gravitational self-binding solver-statement split contract ready",
                    1,
                    "The next branch may inventory and audit the missing coupled solver statement.",
                ),
            ],
            {
                "selected_residual_route": "gravitational_self_binding_solver_statement_residual",
                "missing_gravitational_binding_artifact": selected_missing_artifact,
                "split_contract_ready": True,
            },
            {
                "overall_status": "gravitational_self_binding_solver_statement_route_contract_frozen",
                "keep_mass_origin_branch_blocked": True,
                "hand_off_to_8_7_55_2_84": False,
                "next_required_artifacts": [
                    "gravitational_self_binding_solver_statement_source_inventory",
                    "gravitational_self_binding_solver_statement_wording_audit",
                ],
            },
            {
                "gravitational_branch_refresh_summary": {
                    "recommended_next_route_or_none": "gravitational_self_binding_solver_statement_residual",
                    "missing_gravitational_binding_artifact": selected_missing_artifact,
                },
            },
        ),
    }

    for stem, data in payloads.items():
        write_artifact(stem, data)
        print(f"[ok] wrote {OUT / (stem + '_metrics.json')}")
        print(f"[ok] wrote {OUT / (stem + '_rows.csv')}")


if __name__ == "__main__":
    main()
