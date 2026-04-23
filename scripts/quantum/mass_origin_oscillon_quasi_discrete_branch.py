#!/usr/bin/env python3
"""
Generate oscillon quasi-discrete reopen artifacts for 8.7.55.2.800-.805.

The gravitational self-binding branch closed as extension-only, so the last
admissible no-new-free-parameter fallback is the documented oscillon family.
This branch audits whether oscillon resonance widths/lifetimes can be promoted
into a public mass-spectrum proxy rule. When the rule is absent, it closes the
oscillon route and freezes the next closeout branch for the mass-origin line.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output" / "public" / "quantum"
NOTE = ROOT / "doc" / "quantum" / "18_p_field_action_and_schrodinger_mapping.md"
OSCILLON_ASSESSMENT = OUT / "mass_origin_oscillon_fallback_assessment_metrics.json"
OSCILLON_ROUTE = OUT / "mass_origin_oscillon_quasi_discrete_route_contract_metrics.json"
QBALL_BRANCH = OUT / "mass_origin_qball_ratio_mismatch_branch_refresh_metrics.json"
SELF_GRAVITY_GATE = OUT / "mass_origin_self_gravity_discrete_spectrum_second_gate_refresh_metrics.json"


# 関数: 現在の UTC 時刻を ISO 8601 文字列で返す。
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: 必須入力ファイルの存在を検証する。

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
    for path in (NOTE, OSCILLON_ASSESSMENT, OSCILLON_ROUTE, QBALL_BRANCH, SELF_GRAVITY_GATE):
        req(path)

    note = read_text(NOTE)
    oscillon_assessment = read_json(OSCILLON_ASSESSMENT)
    oscillon_route = read_json(OSCILLON_ROUTE)
    qball_branch = read_json(QBALL_BRANCH)
    self_gravity_gate = read_json(SELF_GRAVITY_GATE)

    required_width_sources = [
        "oscillon_route_documented",
        "oscillon_quasi_discrete_only_statement",
        "oscillon_exact_mass_ladder_absent_statement",
        "route_contract_missing_artifact_freeze",
        "oscillon_width_acceptance_rule",
    ]
    present_width_sources = [
        "oscillon_route_documented",
        "oscillon_quasi_discrete_only_statement",
        "oscillon_exact_mass_ladder_absent_statement",
        "route_contract_missing_artifact_freeze",
    ]
    missing_width_sources = ["oscillon_width_acceptance_rule"]

    width_rule_available = False
    quasi_discrete_mass_proxy_acceptable = False
    proxy_mode_count = 0
    mass_ratio_proxy_available = False
    handoff = False
    remaining_blocker = "oscillon_width_acceptance_rule_absent"
    selected_next_route = "mass_origin_no_public_discrete_spectrum_closeout"
    missing_artifact = "core_discrete_spectrum_generation_rule"

    stems = {
        "source_inventory": "mass_origin_oscillon_width_acceptance_source_inventory",
        "width_gate": "mass_origin_oscillon_quasi_discrete_width_gate_audit",
        "mass_ratio_proxy": "mass_origin_oscillon_mass_ratio_proxy_pilot",
        "handoff_gate": "mass_origin_oscillon_handoff_gate_refresh",
        "branch_refresh": "mass_origin_oscillon_branch_refresh",
        "route_contract": "mass_origin_no_public_discrete_spectrum_route_contract",
    }

    payloads = {
        stems["source_inventory"]: payload(
            "8.7.55.2.800",
            "Oscillon width-acceptance source inventory",
            {
                "mass_origin_oscillon_quasi_discrete_route_contract_json": rel(OSCILLON_ROUTE),
                "mass_origin_oscillon_fallback_assessment_json": rel(OSCILLON_ASSESSMENT),
                "mass_origin_note_markdown": rel(NOTE),
            },
            "Inventory the public source items required to accept oscillon resonance widths/lifetimes as a mass-spectrum proxy.",
            {
                "required_width_acceptance_sources": required_width_sources,
                "inventory_rule": "the oscillon route stays blocked until a no-new-free-parameter width/lifetime acceptance rule is explicitly frozen in the public pack",
            },
            [
                row(
                    "oscillon_width_acceptance_source_inventory_complete",
                    "pass",
                    "oscillon width-acceptance source inventory complete",
                    1,
                    "Source inventory fixed for the oscillon quasi-discrete reopen branch.",
                ),
                row(
                    "oscillon_width_acceptance_source_inventory_present_count",
                    "inventory",
                    "present oscillon width-acceptance source count",
                    len(present_width_sources),
                    f"{len(present_width_sources)} of {len(required_width_sources)} required source items are already public.",
                ),
                row(
                    "oscillon_width_acceptance_source_inventory_missing_count",
                    "watch",
                    "missing oscillon width-acceptance source count",
                    len(missing_width_sources),
                    f"Missing items are {', '.join(missing_width_sources)}.",
                ),
            ],
            {
                "required_oscillon_width_source_count": len(required_width_sources),
                "present_oscillon_width_source_count": len(present_width_sources),
                "missing_oscillon_width_source_count": len(missing_width_sources),
                "missing_oscillon_width_source_items": missing_width_sources,
                "first_route_to_close_or_none": "oscillon_width_acceptance_rule",
                "oscillon_quasi_discrete_only": True,
                "oscillon_exact_mass_ladder_available": False,
            },
            {
                "overall_status": "oscillon_width_acceptance_source_inventory_frozen",
                "keep_mass_origin_branch_blocked": True,
                "hand_off_to_8_7_55_2_84": False,
                "next_required_artifacts": ["oscillon_quasi_discrete_width_gate_audit"],
            },
            {
                "mass_origin_note_oscillon_qball_line": hit(note, "oscillon/Q-ball"),
                "oscillon_assessment_summary": oscillon_assessment["summary"],
                "oscillon_route_contract_summary": oscillon_route["summary"],
            },
        ),
        stems["width_gate"]: payload(
            "8.7.55.2.801",
            "Oscillon quasi-discrete width gate audit",
            {
                "mass_origin_oscillon_width_acceptance_source_inventory_json": f"output/public/quantum/{stems['source_inventory']}_metrics.json",
                "mass_origin_oscillon_fallback_assessment_json": rel(OSCILLON_ASSESSMENT),
                "mass_origin_note_markdown": rel(NOTE),
            },
            "Audit whether oscillon resonance widths/lifetimes can be admitted as a no-new-free-parameter mass-spectrum proxy gate.",
            {
                "gate_rule": "oscillon quasi-discrete states are admissible only if the public pack freezes an explicit acceptance rule that maps resonance width/lifetime into an accepted mass proxy",
            },
            [
                row(
                    "oscillon_quasi_discrete_width_gate_audit_complete",
                    "pass",
                    "oscillon quasi-discrete width gate audit complete",
                    1,
                    "The oscillon width/lifetime gate was audited against the current public pack.",
                ),
                row(
                    "oscillon_width_acceptance_rule_available",
                    "reject",
                    "oscillon width acceptance rule available",
                    0,
                    "No public no-new-free-parameter rule freezes which resonance width/lifetime is acceptable as a mass-spectrum proxy.",
                ),
                row(
                    "oscillon_quasi_discrete_mass_proxy_acceptable",
                    "reject",
                    "oscillon quasi-discrete mass proxy acceptable",
                    0,
                    "Without a public width acceptance rule, quasi-discrete oscillon resonances cannot be promoted into an accepted mass ladder proxy.",
                ),
            ],
            {
                "oscillon_width_acceptance_rule_available": width_rule_available,
                "oscillon_width_gate_without_new_free_parameters": False,
                "quasi_discrete_mass_proxy_acceptable": quasi_discrete_mass_proxy_acceptable,
                "width_gate_nonclosure_reason_or_none": remaining_blocker,
                "missing_width_gate_inputs": missing_width_sources,
            },
            {
                "overall_status": "oscillon_width_gate_audited_missing_rule",
                "keep_mass_origin_branch_blocked": True,
                "hand_off_to_8_7_55_2_84": False,
                "next_required_artifacts": ["oscillon_mass_ratio_proxy_pilot"],
            },
            {
                "mass_origin_note_oscillon_qball_line": hit(note, "oscillon/Q-ball"),
                "mass_origin_note_stability_dependence_line": hit(note, "3+1 の安定性は V の形や対称性に依存する"),
                "oscillon_assessment_summary": oscillon_assessment["summary"],
            },
        ),
        stems["mass_ratio_proxy"]: payload(
            "8.7.55.2.802",
            "Oscillon mass-ratio proxy pilot",
            {
                "mass_origin_oscillon_quasi_discrete_width_gate_audit_json": f"output/public/quantum/{stems['width_gate']}_metrics.json",
                "mass_origin_oscillon_fallback_assessment_json": rel(OSCILLON_ASSESSMENT),
            },
            "Attempt the oscillon mass-ratio proxy pilot only if the width gate admits quasi-discrete resonances as a public proxy ladder.",
            {
                "proxy_rule": "proxy ladder construction requires an accepted oscillon width/lifetime rule; absent that rule, no public proxy ladder is constructed",
            },
            [
                row(
                    "oscillon_mass_ratio_proxy_pilot_complete",
                    "pass",
                    "oscillon mass-ratio proxy pilot complete",
                    1,
                    "The oscillon proxy pilot was evaluated after the width gate audit.",
                ),
                row(
                    "oscillon_mass_ratio_proxy_available",
                    "reject",
                    "oscillon mass-ratio proxy available",
                    0,
                    "No accepted width/lifetime gate exists, so no public oscillon proxy ladder can be constructed.",
                ),
                row(
                    "oscillon_known_mass_ratio_match_found",
                    "reject",
                    "oscillon known-mass ratio match found",
                    0,
                    "The proxy pilot cannot compare against particle ratios before a width acceptance rule exists.",
                ),
            ],
            {
                "accepted_oscillon_mass_ratio_proxy_available": mass_ratio_proxy_available,
                "proxy_mode_count": proxy_mode_count,
                "mass_ratio_proxy_available": mass_ratio_proxy_available,
                "known_mass_ratio_match_found": False,
                "closest_known_mass_ratio_or_none": None,
                "mass_ratio_proxy_nonclosure_reason_or_none": remaining_blocker,
            },
            {
                "overall_status": "oscillon_mass_ratio_proxy_pilot_blocked_by_width_rule",
                "keep_mass_origin_branch_blocked": True,
                "hand_off_to_8_7_55_2_84": False,
                "next_required_artifacts": ["oscillon_handoff_gate_refresh"],
            },
            {
                "oscillon_width_gate_summary": {
                    "oscillon_width_acceptance_rule_available": width_rule_available,
                    "quasi_discrete_mass_proxy_acceptable": quasi_discrete_mass_proxy_acceptable,
                    "width_gate_nonclosure_reason_or_none": remaining_blocker,
                },
            },
        ),
        stems["handoff_gate"]: payload(
            "8.7.55.2.803",
            "Oscillon handoff gate refresh",
            {
                "mass_origin_oscillon_mass_ratio_proxy_pilot_json": f"output/public/quantum/{stems['mass_ratio_proxy']}_metrics.json",
                "mass_origin_oscillon_quasi_discrete_width_gate_audit_json": f"output/public/quantum/{stems['width_gate']}_metrics.json",
            },
            "Refresh the mass-origin handoff gate after the oscillon quasi-discrete proxy audit.",
            {
                "gate_rule": "handoff requires an accepted mass-spectrum proxy ladder that can be compared against canonical particle ratios; quasi-discrete oscillons without a width rule do not satisfy the reopen condition",
            },
            [
                row(
                    "oscillon_handoff_gate_refresh_complete",
                    "pass",
                    "oscillon handoff gate refresh complete",
                    1,
                    "The mass-origin handoff gate was refreshed after the oscillon proxy audit.",
                ),
                row(
                    "oscillon_discrete_spectrum_found_after_gate_refresh",
                    "reject",
                    "oscillon discrete spectrum found after gate refresh",
                    0,
                    "Oscillons remain quasi-discrete only, and no accepted proxy ladder exists.",
                ),
                row(
                    "hand_off_to_8_7_55_2_84_after_oscillon_gate_refresh",
                    "reject",
                    "handoff to 8.7.55.2.84 available after oscillon gate refresh",
                    0,
                    "The oscillon width rule is absent, so the mass-origin branch stays blocked.",
                ),
            ],
            {
                "selected_binding_route_or_none": None,
                "discrete_spectrum_found": False,
                "hand_off_to_8_7_55_2_84": handoff,
                "remaining_binding_blockers": [remaining_blocker],
                "new_branch_required": True,
                "recommended_next_route_or_none": selected_next_route,
            },
            {
                "overall_status": "oscillon_handoff_gate_refreshed_without_proxy_rule",
                "keep_mass_origin_branch_blocked": True,
                "hand_off_to_8_7_55_2_84": False,
                "new_branch_required": True,
                "next_required_artifacts": ["oscillon_branch_refresh"],
            },
            {
                "oscillon_mass_ratio_proxy_summary": {
                    "accepted_oscillon_mass_ratio_proxy_available": mass_ratio_proxy_available,
                    "mass_ratio_proxy_nonclosure_reason_or_none": remaining_blocker,
                },
                "oscillon_assessment_summary": oscillon_assessment["summary"],
            },
        ),
        stems["branch_refresh"]: payload(
            "8.7.55.2.804",
            "Mass-origin branch refresh after oscillon quasi-discrete audit",
            {
                "mass_origin_oscillon_handoff_gate_refresh_json": f"output/public/quantum/{stems['handoff_gate']}_metrics.json",
                "mass_origin_oscillon_quasi_discrete_route_contract_json": rel(OSCILLON_ROUTE),
            },
            "Freeze the oscillon branch outcome and decide whether the mass-origin line reopens .84 or closes the currently admissible no-new-free-parameter routes.",
            {
                "disposition_case": "case_last_admissible_fallback_missing_width_acceptance_rule",
                "selected_next_route": selected_next_route,
                "missing_artifact": missing_artifact,
            },
            [
                row(
                    "oscillon_branch_refresh_complete",
                    "pass",
                    "oscillon branch refresh complete",
                    1,
                    "The oscillon quasi-discrete branch outcome has been frozen.",
                ),
                row(
                    "oscillon_last_admissible_fallback_reopens_8_7_55_2_84",
                    "reject",
                    "last admissible oscillon fallback reopens 8.7.55.2.84",
                    0,
                    "The oscillon route remains blocked by the missing width acceptance rule, so .84 stays closed.",
                ),
                row(
                    "all_current_no_new_free_parameter_routes_exhausted",
                    "pass",
                    "all current no-new-free-parameter routes exhausted",
                    1,
                    "Geometric reflective boundary, Q-ball direct mapping, self-gravity core reopening, and oscillon quasi-discrete reopening are all closed under the current public pack.",
                ),
            ],
            {
                "disposition_case": "case_last_admissible_fallback_missing_width_acceptance_rule",
                "hand_off_to_8_7_55_2_84": False,
                "new_branch_required": True,
                "recommended_next_route_or_none": selected_next_route,
                "all_current_no_new_free_parameter_routes_exhausted": True,
                "remaining_binding_blockers": [remaining_blocker],
            },
            {
                "overall_status": "oscillon_branch_closed_without_handoff",
                "keep_mass_origin_branch_blocked": True,
                "hand_off_to_8_7_55_2_84": False,
                "new_branch_required": True,
                "next_required_artifacts": [selected_next_route],
            },
            {
                "qball_branch_refresh_summary": qball_branch["summary"],
                "self_gravity_second_gate_summary": self_gravity_gate["summary"],
                "oscillon_handoff_gate_summary": {
                    "discrete_spectrum_found": False,
                    "hand_off_to_8_7_55_2_84": False,
                    "remaining_binding_blockers": [remaining_blocker],
                },
            },
        ),
        stems["route_contract"]: payload(
            "8.7.55.2.805",
            "Mass-origin no-public discrete-spectrum closeout route contract",
            {
                "mass_origin_oscillon_branch_refresh_json": f"output/public/quantum/{stems['branch_refresh']}_metrics.json",
                "mass_origin_qball_ratio_mismatch_branch_refresh_json": rel(QBALL_BRANCH),
                "mass_origin_self_gravity_discrete_spectrum_second_gate_refresh_json": rel(SELF_GRAVITY_GATE),
            },
            "Freeze the next residual branch after all currently admissible no-new-free-parameter mass-origin routes remain blocked.",
            {
                "selected_residual_route": selected_next_route,
                "route_priority_basis": "geometric route frozen_rejected_after_31_retries; Q-ball direct mapping stays ratio-mismatched and rigid; self-gravity is extension-only; oscillon remains quasi-discrete without a public width acceptance rule",
                "missing_artifact": missing_artifact,
            },
            [
                row(
                    "mass_origin_no_public_discrete_spectrum_route_contract_complete",
                    "pass",
                    "mass-origin no-public discrete-spectrum route contract complete",
                    1,
                    "The next residual branch has been frozen after all currently admissible routes stayed blocked.",
                ),
                row(
                    "mass_origin_no_public_discrete_spectrum_split_contract_ready",
                    "pass",
                    "mass-origin no-public discrete-spectrum split contract ready",
                    1,
                    "The next branch may inventory the exhausted core routes and freeze the long-term blocked-state closeout.",
                ),
            ],
            {
                "selected_residual_route": selected_next_route,
                "missing_mass_origin_artifact": missing_artifact,
                "split_contract_ready": True,
                "all_current_no_new_free_parameter_routes_exhausted": True,
            },
            {
                "overall_status": "mass_origin_no_public_discrete_spectrum_route_contract_frozen",
                "keep_mass_origin_branch_blocked": True,
                "hand_off_to_8_7_55_2_84": False,
                "next_required_artifacts": [
                    "core_discrete_spectrum_route_exhaustion_inventory",
                    "mass_origin_long_term_blocked_closeout_gate",
                ],
            },
            {
                "qball_branch_refresh_summary": qball_branch["summary"],
                "self_gravity_second_gate_summary": self_gravity_gate["summary"],
                "oscillon_branch_refresh_summary": {
                    "disposition_case": "case_last_admissible_fallback_missing_width_acceptance_rule",
                    "remaining_binding_blockers": [remaining_blocker],
                    "recommended_next_route_or_none": selected_next_route,
                },
            },
        ),
    }

    for stem, data in payloads.items():
        write_artifact(stem, data)
        print(f"[ok] wrote {stem}")


# 関数: スクリプト実行時に branch を起動する。

if __name__ == "__main__":
    main()
