#!/usr/bin/env python3
"""Generate 8.7.56.407-.410 integrated closeout / v3 hold artifacts.

After Trial-1 breakthrough, Trial-2 structural + paper-side sync closure,
Trial-3 coupled-localization closeout, and Trial-4 exploratory strong-side
classification, the remaining work is no longer a single trial residual.
Instead this branch freezes:

1. the integrated v2.0 closeout source inventory,
2. the integrated checkpoint audit,
3. the v2.0 integrated declaration gate, and
4. the v3.0 hold carry-over contract.
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
PART1 = ROOT / "doc" / "paper" / "10_part1_core_theory.md"
PART3A = ROOT / "doc" / "paper" / "12_part3a_quantum_foundations.md"
PART5 = ROOT / "doc" / "paper" / "14_part5_future_predictions.md"

TRIAL1_DECL = OUT / "mass_origin_v2_trial1_breakthrough_declaration_gate_metrics.json"
TRIAL2_DECL = OUT / "mass_origin_v2_trial2_declaration_gate_metrics.json"
TRIAL2_PAPER_SYNC = OUT / "mass_origin_v2_trial2_paper_side_sync_reopened_declaration_gate_metrics.json"
TRIAL3_GATE = OUT / "mass_origin_v2_t3_t2_coupled_localization_closeout_declaration_gate_metrics.json"
TRIAL4_STRUCT = OUT / "mass_origin_v2_trial4_su3_analogy_structural_audit_metrics.json"
TRIAL4_PILOT = OUT / "mass_origin_v2_trial4_running_confinement_qualitative_pilot_metrics.json"
TRIAL4_GATE = OUT / "mass_origin_v2_trial4_exploratory_declaration_v3_hold_gate_metrics.json"

NEXT_ROUTE = "8.7.56.411"
NEXT_ROUTE_LABEL = "v3_hold_carryover_registry_after_v2_integrated_closeout"

PART1_PHOTON = "$A_\\mu=\\delta P_\\mu^T/\\sqrt{Z_P}$"
PART1_COUPLED = "\\kappa_{\\mathrm{coupled}}^2 = m_0^2 - \\beta_n^2"
PART3A_STRUCTURAL_PASS = "foundational / structural pass (numeric α open)"
PART3A_COUPLED_REFERENCE = "coupled-localization statement fixed in Part I 2.7.0"
PART3A_STRONG_IF = "強い相互作用側の核I/F"
PART5_TRIAL4_HOLD = "exploratory foothold retained / v3.0 hold recommended"
PART5_INTEGRATED = "v2.0 integrated closeout"
PART5_NEXT_STEP = "8.7.56.407-.410"


# 関数: UTC 現在時刻を ISO 8601 形式で返す。
def now_iso() -> str:
    """Return the current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


# 関数: 必須入力 path の存在を確認する。

def req(path: Path) -> None:
    """Abort immediately when a required input path is missing."""
    if not path.exists():
        raise SystemExit(f"[fail] missing required input: {path}")


# 関数: UTF-8 JSON artifact を読む。

def read_json(path: Path) -> dict:
    """Load a UTF-8 JSON artifact into a dictionary."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


# 関数: UTF-8 text source を読む。

def read_text(path: Path) -> str:
    """Read a UTF-8 text source into memory."""
    return path.read_text(encoding="utf-8")


# 関数: repo 相対 POSIX path を返す。

def rel(path: Path) -> str:
    """Return a repository-relative POSIX path string."""
    return str(path.relative_to(ROOT)).replace("\\", "/")


# 関数: 指定した部分文字列の最初の hit 行を返す。

def hit(text: str, pattern: str) -> dict | None:
    """Return the first line hit for a substring pattern, if any."""
    for line_no, line in enumerate(text.splitlines(), start=1):
        if pattern in line:
            return {"pattern": pattern, "line": line_no, "text": line.strip()}

    return None


# 関数: 共通 schema の row を組み立てる。

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


# 関数: wording target の present/absent を監査する。

def audit_target(
    file_key: str,
    path: Path,
    text: str,
    pattern: str,
    note: str,
    expected_present: bool = True,
) -> dict:
    """Audit whether a wording target is present or intentionally absent."""
    target_hit = hit(text, pattern)
    present = target_hit is not None
    return {
        "file_key": file_key,
        "file": rel(path),
        "pattern": pattern,
        "expected_present": expected_present,
        "present": present,
        "matched_expectation": present is expected_present,
        "note": note,
        "evidence": target_hit,
    }


# 関数: integrated closeout source inventory を構築する。

def build_inventory(
    common_inputs: dict,
    trial1_decl: dict,
    trial2_decl: dict,
    trial2_sync: dict,
    trial3_gate: dict,
    trial4_gate: dict,
    part1_text: str,
    part3a_text: str,
    part5_text: str,
) -> dict:
    """Freeze the integrated v2.0 closeout source pack."""
    inventory_targets = [
        audit_target(
            "part1_photon_definition",
            PART1,
            part1_text,
            PART1_PHOTON,
            "Part I must keep the Trial-1 photon definition on the current canon surface.",
        ),
        audit_target(
            "part1_coupled_localization_rule",
            PART1,
            part1_text,
            PART1_COUPLED,
            "Part I must keep the coupled-localization rule that honestly closed Trial-3.",
        ),
        audit_target(
            "part3a_structural_pass",
            PART3A,
            part3a_text,
            PART3A_STRUCTURAL_PASS,
            "Part III-A must preserve the Trial-2 structural-pass wording.",
        ),
        audit_target(
            "part3a_coupled_reference",
            PART3A,
            part3a_text,
            PART3A_COUPLED_REFERENCE,
            "Part III-A must keep the Trial-3 rule as reference-only after the Part I closeout.",
        ),
        audit_target(
            "part3a_strong_if_reference",
            PART3A,
            part3a_text,
            PART3A_STRONG_IF,
            "Part III-A must still classify the strong side as a nuclear-interface reference rather than a closed first-principles QCD derivation.",
        ),
        audit_target(
            "part5_trial4_hold",
            PART5,
            part5_text,
            PART5_TRIAL4_HOLD,
            "Part V must record the Trial-4 exploratory closeout and the v3.0 hold recommendation.",
        ),
        audit_target(
            "part5_integrated_closeout",
            PART5,
            part5_text,
            PART5_INTEGRATED,
            "Part V must state that the mainline has moved to the integrated v2.0 closeout stage.",
        ),
        audit_target(
            "part5_next_step",
            PART5,
            part5_text,
            PART5_NEXT_STEP,
            "Part V must point to 8.7.56.407-.410 as the current official next route.",
        ),
    ]
    inventory_ready = all(item["matched_expectation"] for item in inventory_targets)
    upstream_ready = bool(
        trial1_decl["summary"]["trial1_breakthrough_pass_under_working_action"]
        and trial2_decl["summary"]["v2_minimum_condition_satisfied_under_breakthrough_working_action"]
        and trial2_sync["summary"]["trial2_current_paper_state_synced"]
        and trial3_gate["summary"]["trial3_two_component_closeout_pass_under_coupled_localization"]
        and trial4_gate["summary"]["trial4_exploratory_branch_closeable"]
    )
    present_target_count = sum(1 for item in inventory_targets if item["present"])
    required_target_count = len(inventory_targets)

    return payload(
        "8.7.56.407",
        "v2_integrated_closeout_source_inventory",
        common_inputs,
        "Freeze the integrated v2.0 closeout source pack spanning the Trial-1 breakthrough, the Trial-2 structural + paper-side pass, the Trial-3 coupled-localization closeout, the Trial-4 exploratory hold classification, and the synchronized Part I / Part III-A / Part V surfaces.",
        {
            "inventory_rule": "inventory passes only if the current canon simultaneously exposes the breakthrough photon route, the Trial-2 structural EM wording, the Trial-3 coupled-localization statement, and the Trial-4 exploratory hold wording",
            "integration_scope": "integrated closeout is a checkpoint-level freeze for v2.0; it does not require numeric alpha precision closure or a first-principles QCD derivation",
        },
        [
            row(
                "integrated_closeout_inventory_complete",
                "pass" if inventory_ready and upstream_ready else "reject",
                "integrated closeout inventory complete",
                1 if inventory_ready and upstream_ready else 0,
                "The integrated branch starts only after Trial-1/2/3/4 all have an honest current-canon status and the paper/control surfaces reflect that status.",
            ),
            row(
                "integrated_upstream_trial_pack_ready",
                "pass" if upstream_ready else "reject",
                "integrated upstream trial pack ready",
                1 if upstream_ready else 0,
                "Trial-1 breakthrough, Trial-2 structural pass + paper sync, Trial-3 closeout, and Trial-4 exploratory closeout must all be fixed first.",
            ),
            row(
                "integrated_required_target_count",
                "pass",
                "required source targets",
                required_target_count,
                "Integrated inventory counts the shared Part I / Part III-A / Part V wording surfaces that summarize the current v2.0 checkpoint.",
            ),
            row(
                "integrated_present_target_count",
                "pass" if present_target_count == required_target_count else "reject",
                "present source targets",
                present_target_count,
                "All integrated checkpoint wording surfaces should already be present before the declaration branch starts.",
            ),
        ],
        {
            "inventory_ready": inventory_ready and upstream_ready,
            "upstream_ready": upstream_ready,
            "required_target_count": required_target_count,
            "present_target_count": present_target_count,
            "trial1_breakthrough_fixed": trial1_decl["summary"]["trial1_breakthrough_pass_under_working_action"],
            "trial2_structural_pass_fixed": trial2_decl["summary"]["v2_minimum_condition_satisfied_under_breakthrough_working_action"],
            "trial2_paper_sync_complete": trial2_sync["summary"]["trial2_current_paper_state_synced"],
            "trial3_closeout_fixed": trial3_gate["summary"]["trial3_two_component_closeout_pass_under_coupled_localization"],
            "trial4_exploratory_hold_fixed": trial4_gate["summary"]["trial4_v3_hold_recommended"],
            "first_route_to_close_or_none": "v2_integrated_closeout_audit",
        },
        {
            "overall_status": "v2_integrated_closeout_inventory_frozen",
            "advance_to_8_7_56_408": inventory_ready and upstream_ready,
            "next_required_artifacts": [] if inventory_ready and upstream_ready else ["v2_integrated_closeout_source_inventory"],
        },
        {
            "inventory_targets": inventory_targets,
            "trial1_summary": trial1_decl["summary"],
            "trial2_summary": trial2_decl["summary"],
            "trial2_paper_sync_summary": trial2_sync["summary"],
            "trial3_summary": trial3_gate["summary"],
            "trial4_summary": trial4_gate["summary"],
        },
    )


# 関数: integrated closeout audit を構築する。

def build_audit(
    common_inputs: dict,
    inventory: dict,
    trial1_decl: dict,
    trial2_decl: dict,
    trial2_sync: dict,
    trial3_gate: dict,
    trial4_struct: dict,
    trial4_pilot: dict,
    trial4_gate: dict,
) -> dict:
    """Audit whether the current v2.0 checkpoint closes as one integrated state."""
    inventory_ready = bool(inventory["summary"]["inventory_ready"])
    checkpoint_consistent = bool(
        trial1_decl["summary"]["trial1_breakthrough_pass_under_working_action"]
        and trial2_decl["summary"]["v2_minimum_condition_satisfied_under_breakthrough_working_action"]
        and trial2_sync["summary"]["trial2_current_paper_state_synced"]
        and trial3_gate["summary"]["trial3_two_component_closeout_pass_under_coupled_localization"]
        and trial4_gate["summary"]["trial4_exploratory_branch_closeable"]
    )
    numeric_alpha_open = not bool(trial2_decl["summary"]["trial2_alpha_numeric_precision_ready"])
    trial4_exploratory_only = bool(
        trial4_gate["summary"]["trial4_v3_hold_recommended"]
        and not trial4_gate["summary"]["trial4_v3_mainline_promotion_ready"]
    )
    carryover_classified = bool(
        numeric_alpha_open
        and not trial4_struct["summary"]["su3_analogy_structural_pass"]
        and not trial4_pilot["summary"]["running_qualitative_foothold_available"]
        and not trial4_pilot["summary"]["confinement_qualitative_foothold_available"]
        and trial4_exploratory_only
    )
    integrated_closeout_ready = bool(inventory_ready and checkpoint_consistent and carryover_classified)
    v3_hold_carryover_required = bool(numeric_alpha_open or trial4_exploratory_only)

    return payload(
        "8.7.56.408",
        "v2_integrated_closeout_audit",
        common_inputs,
        "Audit whether the current Trial-1/2/3/4 checkpoint is internally consistent as one integrated v2.0 closeout while explicitly classifying the remaining precision and strong-side issues as carry-over rather than blockers.",
        {
            "integrated_closeout_rule": "v2.0 integrated closeout is ready once Trial-1 breakthrough, Trial-2 structural + paper-side pass, Trial-3 coupled-localization closeout, and Trial-4 exploratory hold all coexist on the same current canon surfaces",
            "carryover_rule": "numeric alpha precision and the missing strong-side non-Abelian / running / confinement structures are carry-over items, not blockers to the integrated checkpoint freeze",
        },
        [
            row(
                "v2_integrated_checkpoint_consistent",
                "pass" if checkpoint_consistent else "reject",
                "v2 integrated checkpoint consistent",
                1 if checkpoint_consistent else 0,
                "All completed trial branches must remain mutually consistent on the current canon.",
            ),
            row(
                "v2_numeric_alpha_precision_carryover_classified",
                "pass" if numeric_alpha_open else "watch",
                "numeric alpha precision carry-over classified",
                1 if numeric_alpha_open else 0,
                "Numeric alpha remains open, but it is already classified as a non-blocking carry-over beyond the structural EM pass.",
            ),
            row(
                "v2_trial4_exploratory_hold_classified",
                "pass" if trial4_exploratory_only else "reject",
                "Trial-4 exploratory hold classified",
                1 if trial4_exploratory_only else 0,
                "Trial-4 closes only at exploratory level, so its missing non-Abelian / running / confinement structures must stay on the v3 side.",
            ),
            row(
                "v2_integrated_closeout_ready",
                "pass" if integrated_closeout_ready else "reject",
                "v2 integrated closeout ready",
                1 if integrated_closeout_ready else 0,
                "The integrated checkpoint is ready only once the carry-over classification is explicit and non-blocking.",
            ),
        ],
        {
            "checkpoint_consistent": checkpoint_consistent,
            "numeric_alpha_precision_open": numeric_alpha_open,
            "trial4_exploratory_only": trial4_exploratory_only,
            "carryover_classification_ready": carryover_classified,
            "v2_integrated_closeout_ready": integrated_closeout_ready,
            "v3_hold_carryover_required": v3_hold_carryover_required,
            "first_route_to_close_or_none": "v2_integrated_declaration_gate",
        },
        {
            "overall_status": "v2_integrated_closeout_audit_complete",
            "advance_to_8_7_56_409": integrated_closeout_ready,
            "next_required_artifacts": [] if integrated_closeout_ready else ["v2_integrated_closeout_audit"],
        },
        {
            "inventory_summary": inventory["summary"],
            "trial1_summary": trial1_decl["summary"],
            "trial2_summary": trial2_decl["summary"],
            "trial2_paper_sync_summary": trial2_sync["summary"],
            "trial3_summary": trial3_gate["summary"],
            "trial4_structural_summary": trial4_struct["summary"],
            "trial4_pilot_summary": trial4_pilot["summary"],
            "trial4_gate_summary": trial4_gate["summary"],
        },
    )


# 関数: integrated declaration gate を構築する。

def build_gate(common_inputs: dict, audit: dict, trial2_decl: dict, trial4_gate: dict) -> dict:
    """Freeze the official declaration for the integrated v2.0 checkpoint."""
    integrated_ready = bool(audit["summary"]["v2_integrated_closeout_ready"])
    numeric_alpha_carryover = bool(audit["summary"]["numeric_alpha_precision_open"])
    v3_hold_required = bool(audit["summary"]["v3_hold_carryover_required"])

    return payload(
        "8.7.56.409",
        "v2_integrated_declaration_gate",
        common_inputs,
        "Freeze the official declaration once the current v2.0 program is coherent as one integrated checkpoint and the remaining precision / strong-side items are explicitly pushed into the carry-over contract.",
        {
            "gate_rule": "close the integrated v2.0 declaration once the checkpoint is coherent and all surviving open items are classified as carry-over rather than current blockers",
            "carryover_rule": "numeric alpha precision and Trial-4 strong-side upgrades remain active but non-blocking after the integrated declaration closes",
        },
        [
            row(
                "v2_integrated_declaration_gate_complete",
                "pass",
                "v2 integrated declaration gate complete",
                1,
                "The integrated declaration gate is now frozen.",
            ),
            row(
                "v2_integrated_closeout_complete",
                "pass" if integrated_ready else "reject",
                "v2 integrated closeout complete",
                1 if integrated_ready else 0,
                "The v2.0 program closes only if the integrated checkpoint audit passes.",
            ),
            row(
                "v2_numeric_alpha_precision_carryover_retained",
                "pass" if numeric_alpha_carryover else "watch",
                "numeric alpha precision carry-over retained",
                1 if numeric_alpha_carryover else 0,
                "Numeric alpha remains open but is already downstream of the structural v2.0 pass and therefore does not block the integrated declaration.",
            ),
            row(
                "v3_hold_contract_required",
                "pass" if v3_hold_required else "reject",
                "v3 hold contract required",
                1 if v3_hold_required else 0,
                "The next official move after the v2 declaration is to freeze the v3 hold contract.",
            ),
        ],
        {
            "v2_integrated_closeout_complete": integrated_ready,
            "v2_program_mainline_closeable": integrated_ready,
            "numeric_alpha_precision_carryover_retained": numeric_alpha_carryover,
            "trial4_v3_hold_recommended": trial4_gate["summary"]["trial4_v3_hold_recommended"],
            "recommended_next_route_or_none": "8.7.56.410",
        },
        {
            "overall_status": "v2_integrated_declaration_closed" if integrated_ready else "v2_integrated_declaration_open",
            "advance_to_8_7_56_410": integrated_ready,
            "next_required_artifacts": [] if integrated_ready else ["v2_integrated_declaration_gate"],
        },
        {
            "audit_summary": audit["summary"],
            "trial2_summary": trial2_decl["summary"],
            "trial4_gate_summary": trial4_gate["summary"],
        },
    )


# 関数: v3 hold carry-over contract を構築する。

def build_contract(
    common_inputs: dict,
    audit: dict,
    gate: dict,
    trial4_struct: dict,
    trial4_pilot: dict,
) -> dict:
    """Freeze the v3.0 hold carry-over contract after the integrated closeout."""
    gate_closed = bool(gate["summary"]["v2_integrated_closeout_complete"])
    trial2_alpha_carryover = bool(audit["summary"]["numeric_alpha_precision_open"])
    trial4_strong_carryover = bool(
        not trial4_struct["summary"]["su3_analogy_structural_pass"]
        or not trial4_pilot["summary"]["running_qualitative_foothold_available"]
        or not trial4_pilot["summary"]["confinement_qualitative_foothold_available"]
    )

    return payload(
        "8.7.56.410",
        "v3_hold_route_contract",
        common_inputs,
        "Formalize the post-v2 carry-over contract: numeric alpha precision remains open, Trial-4 strong-side work remains on the v3 hold side, and the next official branch is the hold registry rather than another v2 residual loop.",
        {
            "contract_rule": "once v2.0 integrated closeout is frozen, remaining precision and strong-side items move into a v3 hold registry rather than reopening the current checkpoint",
            "next_route_rule": "the next official branch inventories and audits the carry-over pack before any new-generation mainline promotion is considered",
        },
        [
            row(
                "v2_integrated_branch_closed",
                "pass" if gate_closed else "reject",
                "v2 integrated branch closed",
                1 if gate_closed else 0,
                "The v3 hold contract becomes official only after the integrated v2 declaration closes.",
            ),
            row(
                "v3_hold_trial2_numeric_alpha_carryover_required",
                "pass" if trial2_alpha_carryover else "watch",
                "Trial-2 numeric alpha carry-over required",
                1 if trial2_alpha_carryover else 0,
                "Numeric alpha precision remains an explicit carry-over item beyond the structural EM closeout.",
            ),
            row(
                "v3_hold_trial4_strong_side_carryover_required",
                "pass" if trial4_strong_carryover else "watch",
                "Trial-4 strong-side carry-over required",
                1 if trial4_strong_carryover else 0,
                "Explicit non-Abelian closure plus honest running/confinement remain on the v3 side.",
            ),
            row(
                "v3_hold_registry_route_selected",
                "pass" if gate_closed else "reject",
                "v3 hold registry route selected",
                1 if gate_closed else 0,
                "The next official route is the v3 hold carry-over registry rather than another residual rewrite of the integrated v2 checkpoint.",
            ),
        ],
        {
            "selected_v3_hold_route": NEXT_ROUTE_LABEL,
            "trial2_numeric_alpha_carryover_required": trial2_alpha_carryover,
            "trial4_strong_side_carryover_required": trial4_strong_carryover,
            "v2_integrated_checkpoint_frozen": gate_closed,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "v3_hold_contract_frozen" if gate_closed else "v3_hold_contract_pending",
            "advance_to_next_route": gate_closed,
            "next_required_artifacts": [
                "trial2_numeric_alpha_precision_carryover_registry",
                "trial4_strong_side_v3_hold_registry",
            ]
            if gate_closed
            else ["v3_hold_route_contract"],
        },
        {
            "audit_summary": audit["summary"],
            "gate_summary": gate["summary"],
            "trial4_structural_summary": trial4_struct["summary"],
            "trial4_pilot_summary": trial4_pilot["summary"],
        },
    )


# 関数: current branch を実行する。

def main() -> None:
    """Execute the integrated closeout / v3 hold contract branch."""
    for path in (
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        PART1,
        PART3A,
        PART5,
        TRIAL1_DECL,
        TRIAL2_DECL,
        TRIAL2_PAPER_SYNC,
        TRIAL3_GATE,
        TRIAL4_STRUCT,
        TRIAL4_PILOT,
        TRIAL4_GATE,
    ):
        req(path)

    common_inputs = {
        "status_markdown": rel(STATUS),
        "roadmap_markdown": rel(ROADMAP),
        "ai_context_json": rel(AI_CONTEXT),
        "part1_core_theory_markdown": rel(PART1),
        "part3a_quantum_foundations_markdown": rel(PART3A),
        "part5_future_predictions_markdown": rel(PART5),
        "mass_origin_v2_trial1_breakthrough_declaration_gate_json": rel(TRIAL1_DECL),
        "mass_origin_v2_trial2_declaration_gate_json": rel(TRIAL2_DECL),
        "mass_origin_v2_trial2_paper_side_sync_reopened_declaration_gate_json": rel(TRIAL2_PAPER_SYNC),
        "mass_origin_v2_t3_t2_coupled_localization_closeout_declaration_gate_json": rel(TRIAL3_GATE),
        "mass_origin_v2_trial4_su3_analogy_structural_audit_json": rel(TRIAL4_STRUCT),
        "mass_origin_v2_trial4_running_confinement_qualitative_pilot_json": rel(TRIAL4_PILOT),
        "mass_origin_v2_trial4_exploratory_declaration_v3_hold_gate_json": rel(TRIAL4_GATE),
    }

    trial1_decl = read_json(TRIAL1_DECL)
    trial2_decl = read_json(TRIAL2_DECL)
    trial2_sync = read_json(TRIAL2_PAPER_SYNC)
    trial3_gate = read_json(TRIAL3_GATE)
    trial4_struct = read_json(TRIAL4_STRUCT)
    trial4_pilot = read_json(TRIAL4_PILOT)
    trial4_gate = read_json(TRIAL4_GATE)

    part1_text = read_text(PART1)
    part3a_text = read_text(PART3A)
    part5_text = read_text(PART5)

    inventory = build_inventory(
        common_inputs,
        trial1_decl,
        trial2_decl,
        trial2_sync,
        trial3_gate,
        trial4_gate,
        part1_text,
        part3a_text,
        part5_text,
    )
    audit = build_audit(
        common_inputs,
        inventory,
        trial1_decl,
        trial2_decl,
        trial2_sync,
        trial3_gate,
        trial4_struct,
        trial4_pilot,
        trial4_gate,
    )
    gate = build_gate(common_inputs, audit, trial2_decl, trial4_gate)
    contract = build_contract(common_inputs, audit, gate, trial4_struct, trial4_pilot)

    write_artifact("mass_origin_v2_integrated_closeout_source_inventory", inventory)
    write_artifact("mass_origin_v2_integrated_closeout_audit", audit)
    write_artifact("mass_origin_v2_integrated_declaration_gate", gate)
    write_artifact("mass_origin_v2_v3_hold_route_contract", contract)

    print("[ok] generated integrated closeout / v3 hold artifacts:")
    print(" - mass_origin_v2_integrated_closeout_source_inventory_metrics.json")
    print(" - mass_origin_v2_integrated_closeout_audit_metrics.json")
    print(" - mass_origin_v2_integrated_declaration_gate_metrics.json")
    print(" - mass_origin_v2_v3_hold_route_contract_metrics.json")


# 関数: CLI 直実行時に branch main を起動する。

if __name__ == "__main__":
    main()
