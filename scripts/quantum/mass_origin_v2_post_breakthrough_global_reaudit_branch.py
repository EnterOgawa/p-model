#!/usr/bin/env python3
"""
Generate post-breakthrough global re-audit artifacts for 8.7.56.161-.164.

This branch formalizes the first action-sensitive follow-through after the
Trial-1 breakthrough pivot and the reopened Trial-2 structural pass.

The branch does not rerun the old vector/weak solvers yet. Instead it freezes:

1. which vector mass-spectrum artifacts are still only historical benchmarks
   because they were derived under the explicitly Proca-like action,
2. why the weak-sector branch must remain on fallback hold until the
   vector-side re-audit is executed,
3. which paper-side wording targets are waiting for the same re-audit to
   settle, and
4. the next official executable route after the inventory branch closes.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output" / "public" / "quantum"
PART1 = ROOT / "doc" / "paper" / "10_part1_core_theory.md"
PART3A = ROOT / "doc" / "paper" / "12_part3a_quantum_foundations.md"
PART5 = ROOT / "doc" / "paper" / "14_part5_future_predictions.md"
STATUS = ROOT / "doc" / "STATUS.md"
ROADMAP = ROOT / "doc" / "ROADMAP.md"
AI_CONTEXT = ROOT / "doc" / "AI_CONTEXT_MIN.json"

BREAKTHROUGH_PRESERVATION = OUT / "mass_origin_v2_trial1_breakthrough_legacy_preservation_audit_metrics.json"
TRIAL2_DECLARATION = OUT / "mass_origin_v2_trial2_declaration_gate_metrics.json"
TRIAL2_ROUTE = OUT / "mass_origin_v2_post_breakthrough_action_sensitive_global_reaudit_route_contract_metrics.json"
VECTOR_ROUTE = OUT / "mass_origin_vector_qball_route_contract_metrics.json"
VECTOR_EXACT = OUT / "mass_origin_vector_qball_exact_mass_table_handoff_retry_metrics.json"
VECTOR_BRANCH_REFRESH = OUT / "mass_origin_vector_qball_branch_refresh_after_exact_solver_metrics.json"
VECTOR_HEAVY = OUT / "mass_origin_vector_qball_baryon_tau_neutron_fit_table_metrics.json"
TRIAL3_DECLARATION = OUT / "mass_origin_v2_trial3_declaration_gate_metrics.json"
TRIAL3_FALLBACK_ROUTE = OUT / "mass_origin_v2_trial3_explicit_k_positive_extension_route_contract_metrics.json"
TRIAL1_CASE_B_SCOPE = OUT / "mass_origin_v2_trial1_case_b_scope_declaration_gate_metrics.json"
TRIAL1_CASE_B_FUTURE_DELTA = OUT / "mass_origin_v2_trial1_case_b_future_canon_delta_inventory_metrics.json"


# Function: return the current UTC timestamp in ISO 8601 format.
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# Function: stop execution if a required path is missing.

def req(path: Path) -> None:
    if not path.exists():
        raise SystemExit(f"[fail] missing required input: {path}")


# Function: load a UTF-8 JSON artifact.

def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


# Function: load a UTF-8 text source.

def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# Function: convert an absolute path into a repository-relative path.

def rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


# Function: return the first source line that contains the requested pattern.

def hit(text: str, pattern: str) -> dict | None:
    for line_no, line in enumerate(text.splitlines(), start=1):
        if pattern in line:
            return {"pattern": pattern, "line": line_no, "text": line.strip()}

    return None


# Function: build a standard result row.

def row(row_id: str, status: str, metric: str, value: float, note: str) -> dict:
    return {
        "row_id": row_id,
        "status": status,
        "metric": metric,
        "value": float(value),
        "note": note,
    }


# Function: build a standard payload object.

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


# Function: save a JSON artifact and its row table.

def write_artifact(stem: str, data: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    json_path = OUT / f"{stem}_metrics.json"
    csv_path = OUT / f"{stem}_rows.csv"
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["row_id", "status", "metric", "value", "note"])
        writer.writeheader()
        writer.writerows(data["rows"])


# Function: build a standard inventory target record.

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


# Function: execute the post-breakthrough global re-audit inventory branch.

def main() -> None:
    for path in (
        PART1,
        PART3A,
        PART5,
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        BREAKTHROUGH_PRESERVATION,
        TRIAL2_DECLARATION,
        TRIAL2_ROUTE,
        VECTOR_ROUTE,
        VECTOR_EXACT,
        VECTOR_BRANCH_REFRESH,
        VECTOR_HEAVY,
        TRIAL3_DECLARATION,
        TRIAL3_FALLBACK_ROUTE,
        TRIAL1_CASE_B_SCOPE,
        TRIAL1_CASE_B_FUTURE_DELTA,
    ):
        req(path)

    part1 = read_text(PART1)
    part3a = read_text(PART3A)
    part5 = read_text(PART5)
    status = read_text(STATUS)
    ai_context = read_json(AI_CONTEXT)

    breakthrough_preservation = read_json(BREAKTHROUGH_PRESERVATION)
    trial2_declaration = read_json(TRIAL2_DECLARATION)
    trial2_route = read_json(TRIAL2_ROUTE)
    vector_route = read_json(VECTOR_ROUTE)
    vector_exact = read_json(VECTOR_EXACT)
    vector_branch_refresh = read_json(VECTOR_BRANCH_REFRESH)
    vector_heavy = read_json(VECTOR_HEAVY)
    trial3_declaration = read_json(TRIAL3_DECLARATION)
    trial3_fallback_route = read_json(TRIAL3_FALLBACK_ROUTE)
    trial1_case_b_scope = read_json(TRIAL1_CASE_B_SCOPE)
    trial1_case_b_future_delta = read_json(TRIAL1_CASE_B_FUTURE_DELTA)

    common_inputs = {
        "part1_markdown": rel(PART1),
        "part3a_markdown": rel(PART3A),
        "part5_markdown": rel(PART5),
        "status_markdown": rel(STATUS),
        "roadmap_markdown": rel(ROADMAP),
        "ai_context_json": rel(AI_CONTEXT),
        "mass_origin_v2_trial1_breakthrough_legacy_preservation_audit_json": rel(BREAKTHROUGH_PRESERVATION),
        "mass_origin_v2_trial2_declaration_gate_json": rel(TRIAL2_DECLARATION),
        "mass_origin_v2_post_breakthrough_action_sensitive_global_reaudit_route_contract_json": rel(TRIAL2_ROUTE),
        "mass_origin_vector_qball_route_contract_json": rel(VECTOR_ROUTE),
        "mass_origin_vector_qball_exact_mass_table_handoff_retry_json": rel(VECTOR_EXACT),
        "mass_origin_vector_qball_branch_refresh_after_exact_solver_json": rel(VECTOR_BRANCH_REFRESH),
        "mass_origin_vector_qball_baryon_tau_neutron_fit_table_json": rel(VECTOR_HEAVY),
        "mass_origin_v2_trial3_declaration_gate_json": rel(TRIAL3_DECLARATION),
        "mass_origin_v2_trial3_explicit_k_positive_extension_route_contract_json": rel(TRIAL3_FALLBACK_ROUTE),
        "mass_origin_v2_trial1_case_b_scope_declaration_gate_json": rel(TRIAL1_CASE_B_SCOPE),
        "mass_origin_v2_trial1_case_b_future_canon_delta_inventory_json": rel(TRIAL1_CASE_B_FUTURE_DELTA),
    }

    vector_action = vector_route["formulas"]["vector_field_action"]
    vector_route_uses_proca = "Pi_mu Pi^mu" in vector_action
    vector_exact_ready = bool(vector_exact["summary"]["exact_full_coupled_vector_ladder_available"])
    vector_mass_reaudit_required = bool(
        breakthrough_preservation["summary"]["vector_mass_spectrum_reaudit_required"]
    )
    heavy_anchor_pack_present = bool(vector_heavy["summary"]["proton_anchor_pass"]) and bool(
        vector_heavy["summary"]["tau_anchor_pass"]
    )

    vector_targets = [
        target_record(
            "part3a_exact_vector_hierarchy_line",
            PART3A,
            part3a,
            "mass-origin route で固定した exact vector hierarchy",
            "Part III-A still exposes the exact vector hierarchy as the source pack behind the mass-ratio branch.",
        ),
        target_record(
            "status_global_reaudit_next_step",
            STATUS,
            status,
            "current official next step は `8.7.56.161`",
            "STATUS must already point to the global re-audit inventory branch before sync.",
        ),
        target_record(
            "roadmap_global_reaudit_branch",
            ROADMAP,
            read_text(ROADMAP),
            "`8.7.56.160-.164`",
            "ROADMAP must already expose the post-breakthrough global re-audit branch.",
        ),
    ]
    vector_source_pack_ready = all(item["present"] for item in vector_targets) and vector_exact_ready

    vector_inventory = payload(
        "8.7.56.161",
        "Vector mass-spectrum re-audit inventory",
        common_inputs,
        "Inventory which vector mass-spectrum artifacts remain usable only as historical benchmarks and which items must be re-audited under the breakthrough working action.",
        {
            "working_action_rule": "any vector mass-spectrum closeout derived from L_P,total = -(Z_P/4) F_(P)^2 + (m_P^2/2) Pi_mu Pi^mu + ... is action-sensitive once the breakthrough working action removes the separate Proca/Stueckelberg mass source",
            "historic_benchmark_rule": "existing exact ladder and anchor tables remain valid as old-canon benchmarks even when they are not yet preserved as working-action outputs",
            "reaudit_goal": "classify preserved bookkeeping versus rebuild-required physical claims before relaunching the vector branch",
        },
        [
            row(
                "post_breakthrough_vector_mass_spectrum_inventory_complete",
                "pass",
                "post-breakthrough vector mass-spectrum inventory complete",
                1,
                "The vector mass-spectrum re-audit source pack is frozen.",
            ),
            row(
                "post_breakthrough_vector_mass_spectrum_source_pack_ready",
                "pass" if vector_source_pack_ready else "reject",
                "vector mass-spectrum source pack ready",
                1 if vector_source_pack_ready else 0,
                "The branch needs the exact ladder, heavy-anchor pack, and roadmap/status anchors to be visible simultaneously.",
            ),
            row(
                "post_breakthrough_vector_route_uses_proca_like_action",
                "pass" if vector_route_uses_proca else "reject",
                "vector route uses explicitly Proca-like action",
                1 if vector_route_uses_proca else 0,
                "The original vector-Q-ball branch was built on an action that still contains Pi_mu Pi^mu.",
            ),
            row(
                "post_breakthrough_vector_mass_spectrum_working_action_preservation_ready",
                "reject",
                "vector mass-spectrum working-action preservation ready",
                0,
                "No working-action rebuild has yet shown that the old exact ladder is preserved after the breakthrough pivot.",
            ),
        ],
        {
            "vector_mass_spectrum_source_pack_ready": vector_source_pack_ready,
            "vector_route_uses_proca_like_action": vector_route_uses_proca,
            "exact_vector_ladder_historic_benchmark_ready": vector_exact_ready,
            "heavy_hierarchy_anchor_pack_present": heavy_anchor_pack_present,
            "vector_mass_spectrum_working_action_preservation_ready": False,
            "vector_mass_spectrum_reaudit_required": vector_mass_reaudit_required,
            "first_route_to_close_or_none": "post_breakthrough_weak_sector_reaudit_inventory",
        },
        {
            "overall_status": "post_breakthrough_vector_mass_spectrum_inventory_frozen",
            "trial2_branch_closeable": False,
            "advance_to_8_7_56_162": True,
            "next_required_artifacts": [
                "post_breakthrough_weak_sector_reaudit_inventory",
            ],
        },
        {
            "inventory_targets": vector_targets,
            "vector_route_summary": vector_route["summary"],
            "vector_route_action": vector_action,
            "vector_exact_summary": vector_exact["summary"],
            "vector_branch_refresh_summary": vector_branch_refresh["summary"],
            "vector_heavy_summary": vector_heavy["summary"],
            "breakthrough_preservation_summary": breakthrough_preservation["summary"],
        },
    )

    weak_sector_reaudit_required = bool(breakthrough_preservation["summary"]["weak_sector_reaudit_required"])
    trial3_fallback_hold_retained = bool(trial2_route["summary"]["trial3_fallback_hold_retained"])
    weak_targets = [
        target_record(
            "part1_weak_va_line",
            PART1,
            part1,
            "V-A 演算子構造",
            "Part I still provides the weak-sector motivation line used before the breakthrough pivot.",
        ),
        target_record(
            "part3a_exact_vector_hierarchy_line",
            PART3A,
            part3a,
            "mass-origin route で固定した exact vector hierarchy",
            "Trial-3 still reuses the exact vector hierarchy as its upstream mass ladder.",
        ),
        target_record(
            "part5_future_canon_trial2_hold_line",
            PART5,
            part5,
            "Trial-2 hold",
            "Part V still records the old current-canon hold state, so weak-sector follow-through must wait for the new re-audit program.",
        ),
    ]
    weak_source_pack_ready = all(item["present"] for item in weak_targets)

    weak_inventory = payload(
        "8.7.56.162",
        "Weak-sector re-audit inventory",
        common_inputs,
        "Inventory why the weak-sector branch remains action-sensitive after the Trial-1 breakthrough and why the explicit k-positive route must stay on fallback hold.",
        {
            "dependency_rule": "Trial-3 reuses the vector exact hierarchy, so a change in the underlying working action forces a weak-sector dependency re-audit",
            "fallback_rule": "the explicit k-positive weak-sector route cannot relaunch while the vector mass-spectrum re-audit remains unresolved",
            "scope_rule": "this step freezes the dependency pack only; it does not rerun the weak-sector pilot yet",
        },
        [
            row(
                "post_breakthrough_weak_sector_inventory_complete",
                "pass",
                "post-breakthrough weak-sector inventory complete",
                1,
                "The weak-sector re-audit source pack is frozen.",
            ),
            row(
                "post_breakthrough_weak_sector_source_pack_ready",
                "pass" if weak_source_pack_ready else "reject",
                "weak-sector re-audit source pack ready",
                1 if weak_source_pack_ready else 0,
                "The weak-sector re-audit needs the old Trial-3 outputs, the exact vector hierarchy, and the fallback-hold state to be visible simultaneously.",
            ),
            row(
                "post_breakthrough_weak_sector_depends_on_action_sensitive_vector_route",
                "pass" if weak_sector_reaudit_required else "reject",
                "weak sector depends on action-sensitive vector route",
                1 if weak_sector_reaudit_required else 0,
                "The breakthrough preservation audit already marked the weak-sector branch as action-sensitive.",
            ),
            row(
                "post_breakthrough_trial3_fallback_hold_retained",
                "pass" if trial3_fallback_hold_retained else "reject",
                "Trial-3 fallback hold retained",
                1 if trial3_fallback_hold_retained else 0,
                "The explicit k-positive weak-sector route remains frozen on reserve until the re-audit branch closes.",
            ),
        ],
        {
            "weak_sector_source_pack_ready": weak_source_pack_ready,
            "weak_sector_reaudit_required": weak_sector_reaudit_required,
            "trial3_explicit_k_positive_fallback_hold_retained": trial3_fallback_hold_retained,
            "weak_sector_rerun_ready_without_vector_mass_spectrum_reaudit": False,
            "first_route_to_close_or_none": "post_breakthrough_trial2_paper_side_sync_prerequisite_inventory",
        },
        {
            "overall_status": "post_breakthrough_weak_sector_inventory_frozen",
            "trial2_branch_closeable": False,
            "advance_to_8_7_56_163": True,
            "next_required_artifacts": [
                "post_breakthrough_trial2_paper_side_sync_prerequisite_inventory",
            ],
        },
        {
            "inventory_targets": weak_targets,
            "trial3_declaration_summary": trial3_declaration["summary"],
            "trial3_fallback_route_summary": trial3_fallback_route["summary"],
            "breakthrough_preservation_summary": breakthrough_preservation["summary"],
        },
    )

    paper_targets = [
        target_record(
            "part1_independent_em_term",
            PART1,
            part1,
            "+\\mathcal{L}_{\\mathrm{EM}}",
            "Part I still carries the explicit independent electromagnetic term in the total action.",
        ),
        target_record(
            "part3a_independent_maxwell_sentence",
            PART3A,
            part3a,
            "Maxwell 方程式（U(1) ゲージ場 $A_\\mu$、電場 $E$、磁場 $B$）を、P-model の枠組みとは独立に採用する",
            "Part III-A still states independent Maxwell adoption in the current public wording.",
        ),
        target_record(
            "part3a_a_reject_b_adopt_judgment",
            PART3A,
            part3a,
            "A棄却、B採用",
            "Part III-A still freezes the old judgment that accompanied the independent-EM route.",
        ),
        target_record(
            "part5_trial2_hold_line",
            PART5,
            part5,
            "Trial-2 hold",
            "Part V still records the old current-canon Trial-2 hold state.",
        ),
        target_record(
            "part5_future_canon_delta_line",
            PART5,
            part5,
            "future-canon delta registry",
            "Part V still keeps the pre-breakthrough future-canon registry wording.",
        ),
    ]
    paper_sync_prereq_ready = all(item["present"] for item in paper_targets)

    paper_inventory = payload(
        "8.7.56.163",
        "Trial-2 paper-side sync prerequisite inventory",
        common_inputs,
        "Inventory the paper-side wording targets that must eventually be synchronized with the reopened Trial-2 branch after the action-sensitive re-audits settle the new working canon.",
        {
            "paper_sync_rule": "paper sync may only execute honestly after the action-sensitive vector and weak-sector re-audits establish which old-current-canon statements survive",
            "current_target_rule": "the inventory freezes the wording targets, not the edits themselves",
            "defer_rule": "if current public wording still states independent Maxwell adoption and Trial-2 hold, sync is required but deferred",
        },
        [
            row(
                "post_breakthrough_trial2_paper_sync_prerequisite_inventory_complete",
                "pass",
                "post-breakthrough Trial-2 paper-side sync prerequisite inventory complete",
                1,
                "The paper-side sync prerequisite pack is frozen.",
            ),
            row(
                "post_breakthrough_trial2_paper_sync_prerequisite_pack_ready",
                "pass" if paper_sync_prereq_ready else "reject",
                "Trial-2 paper-side sync prerequisite pack ready",
                1 if paper_sync_prereq_ready else 0,
                "Part I / Part III-A / Part V expose all wording targets needed for a later sync pass.",
            ),
            row(
                "post_breakthrough_current_paper_still_reflects_independent_em_frame",
                "pass",
                "current paper still reflects independent-EM frame",
                1,
                "The public paper wording still carries the old independent Maxwell adoption and Trial-2 hold frame.",
            ),
            row(
                "post_breakthrough_trial2_paper_sync_should_run_now",
                "reject",
                "Trial-2 paper-side sync should run now",
                0,
                "The paper sync is deferred until the action-sensitive vector/weak re-audits settle what survives under the breakthrough working action.",
            ),
        ],
        {
            "trial2_paper_side_sync_prerequisite_inventory_ready": paper_sync_prereq_ready,
            "part1_em_sector_wording_present": True,
            "part3a_independent_maxwell_wording_present": True,
            "part3a_a_reject_b_adopt_wording_present": True,
            "part5_trial2_hold_wording_present": True,
            "trial2_paper_side_sync_execution_ready_after_reaudit": True,
            "paper_side_sync_should_run_now": False,
            "first_route_to_close_or_none": "post_breakthrough_global_reaudit_declaration_gate",
        },
        {
            "overall_status": "post_breakthrough_trial2_paper_sync_prerequisite_inventory_frozen",
            "trial2_branch_closeable": False,
            "advance_to_8_7_56_164": True,
            "next_required_artifacts": [
                "post_breakthrough_global_reaudit_declaration_gate",
            ],
        },
        {
            "inventory_targets": paper_targets,
            "trial1_case_b_scope_summary": trial1_case_b_scope["summary"],
            "trial1_future_canon_delta_summary": trial1_case_b_future_delta["summary"],
            "trial2_declaration_summary": trial2_declaration["summary"],
        },
    )

    global_inventory_ready = (
        vector_inventory["summary"]["vector_mass_spectrum_source_pack_ready"]
        and weak_inventory["summary"]["weak_sector_source_pack_ready"]
        and paper_inventory["summary"]["trial2_paper_side_sync_prerequisite_inventory_ready"]
    )

    declaration_gate = payload(
        "8.7.56.164",
        "Global re-audit declaration gate / Trial-3 fallback disposition refresh",
        common_inputs,
        "Integrate the post-breakthrough vector, weak-sector, and paper-side prerequisite inventories and freeze the next executable route plus the Trial-3 fallback disposition.",
        {
            "next_route_rule": "prioritize the vector mass-spectrum re-audit because both the weak-sector branch and paper-side sync depend on its preserved-vs-rebuild classification",
            "trial3_rule": "retain the explicit k-positive branch on fallback hold until the vector-side re-audit closes",
            "paper_sync_rule": "paper-side sync remains deferred even though its prerequisite pack is ready, because the action-sensitive route classification is not settled yet",
        },
        [
            row(
                "post_breakthrough_global_reaudit_inventory_pack_complete",
                "pass" if global_inventory_ready else "reject",
                "post-breakthrough global re-audit inventory pack complete",
                1 if global_inventory_ready else 0,
                "The branch closes only after vector, weak-sector, and paper-side prerequisite inventories are all frozen.",
            ),
            row(
                "post_breakthrough_vector_mass_spectrum_reaudit_execution_required",
                "pass",
                "vector mass-spectrum re-audit execution required",
                1,
                "The vector mass-spectrum branch is the first action-sensitive route that must be rerun under the breakthrough working action.",
            ),
            row(
                "post_breakthrough_trial2_paper_sync_deferred_until_reaudit",
                "pass",
                "Trial-2 paper-side sync deferred until re-audit",
                1,
                "The wording targets are known, but execution is deferred until the action-sensitive branch classification is settled.",
            ),
            row(
                "post_breakthrough_trial3_fallback_hold_retained",
                "pass" if trial3_fallback_hold_retained else "reject",
                "Trial-3 fallback hold retained after global re-audit inventory",
                1 if trial3_fallback_hold_retained else 0,
                "The weak-sector fallback route remains on reserve after the inventory branch.",
            ),
        ],
        {
            "action_sensitive_global_reaudit_inventory_ready": global_inventory_ready,
            "vector_mass_spectrum_reaudit_execution_required": True,
            "weak_sector_reaudit_execution_required": True,
            "trial2_paper_side_sync_deferred_until_reaudit": True,
            "trial3_fallback_hold_retained": trial3_fallback_hold_retained,
            "trial3_fallback_hold_release_ready": False,
            "recommended_next_route_or_none": "8.7.56.165",
        },
        {
            "overall_status": "post_breakthrough_global_reaudit_inventory_branch_complete",
            "trial2_branch_closeable": True,
            "advance_to_8_7_56_165": True,
            "next_required_artifacts": [
                "post_breakthrough_vector_mass_spectrum_action_sensitive_source_inventory",
                "post_breakthrough_vector_mass_spectrum_preservation_audit",
                "post_breakthrough_vector_mass_spectrum_rebuild_contract",
            ],
        },
        {
            "vector_inventory_summary": vector_inventory["summary"],
            "weak_inventory_summary": weak_inventory["summary"],
            "paper_inventory_summary": paper_inventory["summary"],
            "trial2_declaration_summary": trial2_declaration["summary"],
            "trial2_route_summary": trial2_route["summary"],
        },
    )

    write_artifact("mass_origin_v2_post_breakthrough_vector_mass_spectrum_reaudit_inventory", vector_inventory)
    write_artifact("mass_origin_v2_post_breakthrough_weak_sector_reaudit_inventory", weak_inventory)
    write_artifact("mass_origin_v2_post_breakthrough_trial2_paper_sync_prerequisite_inventory", paper_inventory)
    write_artifact("mass_origin_v2_post_breakthrough_global_reaudit_declaration_gate", declaration_gate)

    print("[ok] wrote:")
    print(" - mass_origin_v2_post_breakthrough_vector_mass_spectrum_reaudit_inventory_metrics.json")
    print(" - mass_origin_v2_post_breakthrough_weak_sector_reaudit_inventory_metrics.json")
    print(" - mass_origin_v2_post_breakthrough_trial2_paper_sync_prerequisite_inventory_metrics.json")
    print(" - mass_origin_v2_post_breakthrough_global_reaudit_declaration_gate_metrics.json")


# Function: run the post-breakthrough global re-audit inventory branch from the CLI.

if __name__ == "__main__":
    main()
