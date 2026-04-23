#!/usr/bin/env python3
"""
Generate vector mass-spectrum re-audit execution artifacts for 8.7.56.165-.168.

This branch is the first executable follow-through after the post-breakthrough
global inventory. It freezes:

1. the full source pack behind the old exact vector mass-spectrum closure,
2. which parts of that old closure survive only as historical bookkeeping
   benchmarks under the breakthrough working action,
3. the new rebuild-required route that must replace the old current-canon
   physical claim, and
4. the updated dependency state for Trial-3 and the deferred Trial-2 paper sync.
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
GLOBAL_REAUDIT_DECLARATION = OUT / "mass_origin_v2_post_breakthrough_global_reaudit_declaration_gate_metrics.json"
VECTOR_ROUTE = OUT / "mass_origin_vector_qball_route_contract_metrics.json"
VECTOR_SOLVER = OUT / "mass_origin_vector_qball_full_coupled_solver_pilot_metrics.json"
VECTOR_EXACT = OUT / "mass_origin_vector_qball_exact_mass_table_handoff_retry_metrics.json"
VECTOR_HEAVY = OUT / "mass_origin_vector_qball_baryon_tau_neutron_fit_table_metrics.json"
VECTOR_BRANCH_REFRESH = OUT / "mass_origin_vector_qball_branch_refresh_after_exact_solver_metrics.json"
TRIAL3_FALLBACK_ROUTE = OUT / "mass_origin_v2_trial3_explicit_k_positive_extension_route_contract_metrics.json"


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


# Function: execute the post-breakthrough vector mass-spectrum re-audit branch.

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
        GLOBAL_REAUDIT_DECLARATION,
        VECTOR_ROUTE,
        VECTOR_SOLVER,
        VECTOR_EXACT,
        VECTOR_HEAVY,
        VECTOR_BRANCH_REFRESH,
        TRIAL3_FALLBACK_ROUTE,
    ):
        req(path)

    part1 = read_text(PART1)
    part3a = read_text(PART3A)
    part5 = read_text(PART5)
    status = read_text(STATUS)
    roadmap = read_text(ROADMAP)
    ai_context = read_json(AI_CONTEXT)

    breakthrough_preservation = read_json(BREAKTHROUGH_PRESERVATION)
    trial2_declaration = read_json(TRIAL2_DECLARATION)
    global_reaudit_declaration = read_json(GLOBAL_REAUDIT_DECLARATION)
    vector_route = read_json(VECTOR_ROUTE)
    vector_solver = read_json(VECTOR_SOLVER)
    vector_exact = read_json(VECTOR_EXACT)
    vector_heavy = read_json(VECTOR_HEAVY)
    vector_branch_refresh = read_json(VECTOR_BRANCH_REFRESH)
    trial3_fallback_route = read_json(TRIAL3_FALLBACK_ROUTE)

    common_inputs = {
        "part1_markdown": rel(PART1),
        "part3a_markdown": rel(PART3A),
        "part5_markdown": rel(PART5),
        "status_markdown": rel(STATUS),
        "roadmap_markdown": rel(ROADMAP),
        "ai_context_json": rel(AI_CONTEXT),
        "mass_origin_v2_trial1_breakthrough_legacy_preservation_audit_json": rel(BREAKTHROUGH_PRESERVATION),
        "mass_origin_v2_trial2_declaration_gate_json": rel(TRIAL2_DECLARATION),
        "mass_origin_v2_post_breakthrough_global_reaudit_declaration_gate_json": rel(GLOBAL_REAUDIT_DECLARATION),
        "mass_origin_vector_qball_route_contract_json": rel(VECTOR_ROUTE),
        "mass_origin_vector_qball_full_coupled_solver_pilot_json": rel(VECTOR_SOLVER),
        "mass_origin_vector_qball_exact_mass_table_handoff_retry_json": rel(VECTOR_EXACT),
        "mass_origin_vector_qball_baryon_tau_neutron_fit_table_json": rel(VECTOR_HEAVY),
        "mass_origin_vector_qball_branch_refresh_after_exact_solver_json": rel(VECTOR_BRANCH_REFRESH),
        "mass_origin_v2_trial3_explicit_k_positive_extension_route_contract_json": rel(TRIAL3_FALLBACK_ROUTE),
    }

    vector_action = vector_route["formulas"]["vector_field_action"]
    uses_old_proca_like_action = "Pi_mu Pi^mu" in vector_action
    exact_state_count = int(vector_solver["summary"]["exact_state_count"])
    exact_integer_mode_count = int(vector_solver["summary"]["exact_integer_mode_count"])
    best_exact = vector_exact["summary"]["best_exact_match_or_none"]
    best_muon = vector_heavy["summary"]["best_muon_row_or_none"]
    best_proton = vector_heavy["summary"]["best_proton_row_or_none"]
    best_tau = vector_heavy["summary"]["best_tau_row_or_none"]
    best_np_pair = vector_heavy["summary"]["best_neutron_proton_pair_or_none"]
    trial3_fallback_hold_retained = bool(
        global_reaudit_declaration["summary"]["trial3_fallback_hold_retained"]
    )

    source_targets = [
        target_record(
            "part1_explicit_stueckelberg_mass_line",
            PART1,
            part1,
            "+\\frac{m_P^2}{2}\\left(P_\\mu-\\frac{1}{m_P}\\partial_\\mu\\pi\\right)",
            "Part I still exposes the explicit Proca/Stueckelberg mass source that makes the old vector route action-sensitive.",
        ),
        target_record(
            "part1_pi_mu_definition_line",
            PART1,
            part1,
            "Pi_\\mu:=P_\\mu-\\partial_\\mu\\pi/m_P",
            "Part I still defines the Stückelberg-completed vector combination used by the old action.",
        ),
        target_record(
            "part3a_exact_vector_hierarchy_line",
            PART3A,
            part3a,
            "mass-origin route で固定した exact vector hierarchy",
            "Part III-A still states that the exact vector hierarchy underlies the mass-ratio branch.",
        ),
        target_record(
            "part3a_case_b_line",
            PART3A,
            part3a,
            "Part I の explicit Proca/Stückelberg term が残るため",
            "Part III-A already records the Case-B interpretation that motivated the breakthrough follow-through.",
        ),
        target_record(
            "part5_trial2_hold_line",
            PART5,
            part5,
            "Trial-2 hold",
            "Part V still carries the pre-breakthrough hold wording that depends on the re-audit outcome.",
        ),
        target_record(
            "status_next_step_anchor",
            STATUS,
            status,
            "current official next step は `8.7.56.165`",
            "STATUS must already point to the vector mass-spectrum re-audit execution branch.",
        ),
        target_record(
            "roadmap_vector_reaudit_branch",
            ROADMAP,
            roadmap,
            "`8.7.56.165-.168`",
            "ROADMAP must expose the vector mass-spectrum re-audit execution branch.",
        ),
    ]
    source_pack_ready = all(item["present"] for item in source_targets)

    source_inventory = payload(
        "8.7.56.165",
        "Vector mass-spectrum action-sensitive source inventory",
        common_inputs,
        "Freeze the exact ladder, heavy-anchor fit table, solver pack, and old action dependency in one machine-readable source inventory before classifying preserved bookkeeping versus rebuild-required claims.",
        {
            "source_inventory_rule": "the execution branch needs the old action formula, exact ladder pack, heavy-anchor fit table, and current breakthrough/global-reaudit gates visible at the same time",
            "action_sensitive_rule": "if the old vector route still depends on the explicit Pi_mu Pi^mu mass term, physical preservation under the breakthrough working action cannot be assumed",
            "benchmark_rule": "old exact ladder and heavy-anchor outputs may remain as historical benchmarks even when working-action preservation is not yet proved",
        },
        [
            row(
                "post_breakthrough_vector_mass_spectrum_action_sensitive_source_inventory_complete",
                "pass",
                "vector mass-spectrum action-sensitive source inventory complete",
                1,
                "The execution branch source pack is frozen.",
            ),
            row(
                "post_breakthrough_vector_mass_spectrum_action_sensitive_source_pack_ready",
                "pass" if source_pack_ready else "reject",
                "vector mass-spectrum action-sensitive source pack ready",
                1 if source_pack_ready else 0,
                "The branch needs the old action dependency, exact ladder, heavy anchors, and roadmap/status anchors to be visible simultaneously.",
            ),
            row(
                "post_breakthrough_vector_exact_state_count_historic",
                "pass",
                "historic exact vector state count",
                exact_state_count,
                "The adopted exact multicomponent ladder remains available as an old-canon benchmark pack.",
            ),
            row(
                "post_breakthrough_vector_exact_integer_mode_count_historic",
                "pass",
                "historic exact integer mode count",
                exact_integer_mode_count,
                "The integer-mode bookkeeping behind the old exact ladder remains available for re-audit.",
            ),
        ],
        {
            "vector_mass_spectrum_action_sensitive_source_pack_ready": source_pack_ready,
            "vector_route_uses_old_proca_like_action": uses_old_proca_like_action,
            "exact_ladder_historic_benchmark_pack_present": True,
            "heavy_anchor_historic_benchmark_pack_present": True,
            "solver_spec_historic_pack_present": True,
            "first_route_to_close_or_none": "post_breakthrough_vector_mass_spectrum_preservation_rebuild_audit",
        },
        {
            "overall_status": "post_breakthrough_vector_mass_spectrum_action_sensitive_source_inventory_frozen",
            "trial2_branch_closeable": False,
            "advance_to_8_7_56_166": True,
            "next_required_artifacts": [
                "post_breakthrough_vector_mass_spectrum_preservation_rebuild_audit",
            ],
        },
        {
            "inventory_targets": source_targets,
            "vector_route_summary": vector_route["summary"],
            "vector_route_action": vector_action,
            "vector_solver_summary": vector_solver["summary"],
            "vector_exact_summary": vector_exact["summary"],
            "vector_heavy_summary": vector_heavy["summary"],
            "vector_branch_refresh_summary": vector_branch_refresh["summary"],
            "breakthrough_preservation_summary": breakthrough_preservation["summary"],
            "global_reaudit_declaration_summary": global_reaudit_declaration["summary"],
            "ai_context_current_step": ai_context["current_step"],
        },
    )

    historic_benchmark_pack_retained = (
        bool(vector_exact["summary"]["exact_full_coupled_vector_ladder_available"])
        and bool(vector_heavy["summary"]["muon_anchor_pass"])
        and bool(vector_heavy["summary"]["proton_anchor_pass"])
        and bool(vector_heavy["summary"]["tau_anchor_pass"])
        and bool(vector_heavy["summary"]["neutron_proton_pair_pass"])
    )
    working_action_preserved_physical_claim = False
    vector_mass_spectrum_rebuild_required = uses_old_proca_like_action and not working_action_preserved_physical_claim

    preservation_audit = payload(
        "8.7.56.166",
        "Vector mass-spectrum preservation / rebuild-required audit",
        common_inputs,
        "Classify the old exact ladder closure into historical bookkeeping that may be retained as benchmark data and physical claims that must be rebuilt under the breakthrough working action.",
        {
            "benchmark_retention_rule": "historic tables may survive as benchmark bookkeeping if they remain explicitly labeled as old-canon outputs",
            "physical_preservation_rule": "physical preservation under the breakthrough working action would require a new derivation that no longer depends on the explicit Pi_mu Pi^mu mass source",
            "rebuild_rule": "if the old exact ladder and heavy anchors still depend on the old action, the current-canon physical claim must be rebuilt before it can drive Trial-3 or paper-side sync",
        },
        [
            row(
                "post_breakthrough_vector_historic_exact_ladder_benchmark_pack_retained",
                "pass" if historic_benchmark_pack_retained else "reject",
                "historic exact ladder benchmark pack retained",
                1 if historic_benchmark_pack_retained else 0,
                "The old exact ladder remains usable as benchmark bookkeeping for the re-audit.",
            ),
            row(
                "post_breakthrough_vector_best_exact_muon_relative_error_historic",
                "pass",
                "historic best exact muon relative error",
                float(best_exact["relative_error"]),
                "The old-canon exact ladder still records the muon anchor quality that motivated the original handoff.",
            ),
            row(
                "post_breakthrough_vector_best_proton_relative_error_historic",
                "pass",
                "historic best proton relative error",
                float(best_proton["relative_error"]),
                "The heavy-anchor table still records the proton benchmark quality as old-canon evidence.",
            ),
            row(
                "post_breakthrough_vector_working_action_physical_preservation_available",
                "reject",
                "working-action physical preservation available",
                0,
                "No working-action rebuild currently proves that the exact ladder and anchor claims survive as current-canon physical outputs.",
            ),
            row(
                "post_breakthrough_vector_mass_spectrum_rebuild_required",
                "pass" if vector_mass_spectrum_rebuild_required else "reject",
                "vector mass-spectrum rebuild required",
                1 if vector_mass_spectrum_rebuild_required else 0,
                "The physical vector mass-spectrum claim remains action-sensitive and must be rebuilt under the breakthrough working action.",
            ),
        ],
        {
            "historic_benchmark_pack_retained": historic_benchmark_pack_retained,
            "historic_exact_ladder_benchmark_pack_retained": bool(
                vector_exact["summary"]["exact_full_coupled_vector_ladder_available"]
            ),
            "historic_heavy_anchor_benchmark_pack_retained": True,
            "working_action_vector_mass_spectrum_physical_claim_preserved": working_action_preserved_physical_claim,
            "vector_mass_spectrum_rebuild_required": vector_mass_spectrum_rebuild_required,
            "trial3_dependency_release_ready": False,
            "first_route_to_close_or_none": "post_breakthrough_vector_mass_spectrum_reaudit_route_contract",
        },
        {
            "overall_status": "post_breakthrough_vector_mass_spectrum_preservation_rebuild_classification_frozen",
            "trial2_branch_closeable": False,
            "advance_to_8_7_56_167": True,
            "next_required_artifacts": [
                "post_breakthrough_vector_mass_spectrum_reaudit_route_contract",
            ],
        },
        {
            "best_exact_match_row": best_exact,
            "best_muon_row": best_muon,
            "best_proton_row": best_proton,
            "best_tau_row": best_tau,
            "best_neutron_proton_pair": best_np_pair,
            "vector_branch_refresh_summary": vector_branch_refresh["summary"],
            "breakthrough_preservation_summary": breakthrough_preservation["summary"],
        },
    )

    route_contract = payload(
        "8.7.56.167",
        "Vector mass-spectrum re-audit route contract / weak-sector dependency refresh",
        common_inputs,
        "Formalize the preserved-vs-rebuild classification as a new executable route and refresh the Trial-3 dependency state accordingly.",
        {
            "selected_residual_route": "post_breakthrough_vector_mass_spectrum_working_action_rebuild",
            "dependency_rule": "Trial-3 remains blocked until the vector mass-spectrum branch is rebuilt or explicitly reclassified under the breakthrough working action",
            "paper_sync_rule": "Trial-2 paper-side sync remains deferred until the same vector rebuild settles which old-canon statements survive",
        },
        [
            row(
                "post_breakthrough_vector_mass_spectrum_reaudit_route_contract_complete",
                "pass",
                "vector mass-spectrum re-audit route contract complete",
                1,
                "The next executable vector route is frozen after the preservation audit.",
            ),
            row(
                "post_breakthrough_vector_historic_benchmark_pack_retained",
                "pass" if preservation_audit["summary"]["historic_benchmark_pack_retained"] else "reject",
                "historic benchmark pack retained",
                1 if preservation_audit["summary"]["historic_benchmark_pack_retained"] else 0,
                "The old exact ladder remains on hand as benchmark evidence while the working-action rebuild is prepared.",
            ),
            row(
                "post_breakthrough_vector_rebuild_route_selected",
                "pass" if vector_mass_spectrum_rebuild_required else "reject",
                "vector rebuild route selected",
                1 if vector_mass_spectrum_rebuild_required else 0,
                "The route contract selects working-action rebuild rather than wording-only preservation.",
            ),
            row(
                "post_breakthrough_trial3_dependency_still_blocked_by_vector_rebuild",
                "pass" if trial3_fallback_hold_retained else "reject",
                "Trial-3 dependency still blocked by vector rebuild",
                1 if trial3_fallback_hold_retained else 0,
                "The weak-sector branch still depends on the vector rebuild outcome.",
            ),
        ],
        {
            "selected_residual_route": "post_breakthrough_vector_mass_spectrum_working_action_rebuild",
            "missing_v2_artifact": "working_action_vector_mass_spectrum_rebuild_pack",
            "historic_benchmark_pack_retained": preservation_audit["summary"]["historic_benchmark_pack_retained"],
            "trial3_dependency_state": "blocked_by_vector_mass_spectrum_rebuild",
            "trial2_paper_side_sync_state": "deferred_until_vector_mass_spectrum_rebuild",
            "split_contract_ready": True,
            "recommended_next_route_or_none": "8.7.56.169",
        },
        {
            "overall_status": "post_breakthrough_vector_mass_spectrum_reaudit_route_contract_frozen",
            "trial2_branch_closeable": True,
            "advance_to_8_7_56_168": True,
            "next_required_artifacts": [
                "working_action_vector_mass_spectrum_rebuild_source_inventory",
                "working_action_vector_mass_spectrum_reduced_solver_pilot",
                "working_action_vector_mass_spectrum_anchor_refresh",
            ],
        },
        {
            "preservation_audit_summary": preservation_audit["summary"],
            "trial3_fallback_route_summary": trial3_fallback_route["summary"],
            "global_reaudit_declaration_summary": global_reaudit_declaration["summary"],
            "trial2_declaration_summary": trial2_declaration["summary"],
        },
    )

    declaration_gate = payload(
        "8.7.56.168",
        "Vector mass-spectrum declaration gate / Trial-3 fallback disposition refresh",
        common_inputs,
        "Integrate the vector re-audit execution results, confirm that the historic benchmark pack is retained only as old-canon evidence, and refresh the Trial-3 fallback disposition plus paper-side sync defer state.",
        {
            "gate_rule": "the execution branch closes once the source inventory, preservation audit, and rebuild route contract are all frozen",
            "trial3_rule": "retain the explicit k-positive weak-sector route on fallback hold until the vector rebuild route closes",
            "paper_sync_rule": "defer Trial-2 paper-side sync until the vector rebuild route settles current-canon preservation versus rebuild",
        },
        [
            row(
                "post_breakthrough_vector_mass_spectrum_reaudit_execution_branch_complete",
                "pass",
                "vector mass-spectrum re-audit execution branch complete",
                1,
                "The source inventory, preservation audit, and rebuild route contract are now integrated.",
            ),
            row(
                "post_breakthrough_vector_historic_benchmark_pack_retained_gate",
                "pass" if route_contract["summary"]["historic_benchmark_pack_retained"] else "reject",
                "historic benchmark pack retained at gate",
                1 if route_contract["summary"]["historic_benchmark_pack_retained"] else 0,
                "The old vector ladder remains available only as benchmark support while the rebuild route is prepared.",
            ),
            row(
                "post_breakthrough_vector_working_action_rebuild_required_gate",
                "pass" if vector_mass_spectrum_rebuild_required else "reject",
                "working-action vector rebuild required at gate",
                1 if vector_mass_spectrum_rebuild_required else 0,
                "The vector branch cannot yet be claimed preserved under the breakthrough working action.",
            ),
            row(
                "post_breakthrough_trial3_fallback_hold_retained_gate",
                "pass" if trial3_fallback_hold_retained else "reject",
                "Trial-3 fallback hold retained at gate",
                1 if trial3_fallback_hold_retained else 0,
                "The weak-sector route remains on hold pending the vector rebuild.",
            ),
        ],
        {
            "action_sensitive_vector_mass_spectrum_reaudit_execution_ready": True,
            "historic_benchmark_pack_retained": route_contract["summary"]["historic_benchmark_pack_retained"],
            "working_action_vector_mass_spectrum_rebuild_required": vector_mass_spectrum_rebuild_required,
            "trial2_paper_side_sync_deferred_until_vector_rebuild": True,
            "trial3_fallback_hold_retained": trial3_fallback_hold_retained,
            "trial3_fallback_hold_release_ready": False,
            "recommended_next_route_or_none": "8.7.56.169",
        },
        {
            "overall_status": "post_breakthrough_vector_mass_spectrum_reaudit_execution_branch_complete",
            "trial2_branch_closeable": True,
            "advance_to_8_7_56_169": True,
            "next_required_artifacts": [
                "working_action_vector_mass_spectrum_rebuild_source_inventory",
                "working_action_vector_mass_spectrum_reduced_solver_pilot",
                "working_action_vector_mass_spectrum_anchor_refresh",
            ],
        },
        {
            "source_inventory_summary": source_inventory["summary"],
            "preservation_audit_summary": preservation_audit["summary"],
            "route_contract_summary": route_contract["summary"],
            "global_reaudit_declaration_summary": global_reaudit_declaration["summary"],
        },
    )

    write_artifact(
        "mass_origin_v2_post_breakthrough_vector_mass_spectrum_action_sensitive_source_inventory",
        source_inventory,
    )
    write_artifact(
        "mass_origin_v2_post_breakthrough_vector_mass_spectrum_preservation_rebuild_audit",
        preservation_audit,
    )
    write_artifact(
        "mass_origin_v2_post_breakthrough_vector_mass_spectrum_reaudit_route_contract",
        route_contract,
    )
    write_artifact(
        "mass_origin_v2_post_breakthrough_vector_mass_spectrum_declaration_gate",
        declaration_gate,
    )

    print("[ok] wrote:")
    print(" - mass_origin_v2_post_breakthrough_vector_mass_spectrum_action_sensitive_source_inventory_metrics.json")
    print(" - mass_origin_v2_post_breakthrough_vector_mass_spectrum_preservation_rebuild_audit_metrics.json")
    print(" - mass_origin_v2_post_breakthrough_vector_mass_spectrum_reaudit_route_contract_metrics.json")
    print(" - mass_origin_v2_post_breakthrough_vector_mass_spectrum_declaration_gate_metrics.json")


# Function: run the vector mass-spectrum re-audit execution branch from the CLI.

if __name__ == "__main__":
    main()
