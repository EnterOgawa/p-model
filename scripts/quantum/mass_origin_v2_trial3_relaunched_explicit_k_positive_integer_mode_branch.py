#!/usr/bin/env python3
"""
Generate relaunched Trial-3 explicit k-positive integer-mode residual artifacts.

This branch executes roadmap steps 8.7.56.229-.232.

The post-photon unlock reopened the weak-sector route, but the relaunched pilot
showed that the remaining blocker is now solver-side: the current numerical
interpolation and full-coupled exact-ladder builder still freeze the k-axis at
zero. This script freezes that blocker as machine-readable inventory, audit,
declaration, and disposition artifacts.
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
STATUS = ROOT / "doc" / "STATUS.md"
ROADMAP = ROOT / "doc" / "ROADMAP.md"
AI_CONTEXT = ROOT / "doc" / "AI_CONTEXT_MIN.json"

TRIAL3_RELAUNCHED_SOURCE = OUT / "mass_origin_v2_trial3_relaunched_weak_sector_source_inventory_metrics.json"
TRIAL3_RELAUNCHED_PILOT = OUT / "mass_origin_v2_trial3_relaunched_weak_sector_pilot_metrics.json"
TRIAL3_RELAUNCHED_AUDIT = OUT / "mass_origin_v2_trial3_relaunched_weinberg_angle_weak_coupling_audit_metrics.json"
TRIAL3_RELAUNCHED_DECLARATION = OUT / "mass_origin_v2_trial3_relaunched_declaration_gate_metrics.json"
TRIAL3_RELAUNCHED_DISPOSITION = OUT / "mass_origin_v2_trial3_relaunched_paper_sync_trial4_disposition_gate_metrics.json"
TRIAL3_HIGH_MASS_ROUTE = OUT / "mass_origin_v2_trial3_explicit_k_positive_extension_route_contract_metrics.json"
VECTOR_SOLVER_SPEC = OUT / "mass_origin_vector_qball_solver_spec_metrics.json"
VECTOR_CONSTRAINT = OUT / "mass_origin_vector_qball_coupled_constraint_freeze_audit_metrics.json"
VECTOR_NUMERICAL_PILOT = OUT / "mass_origin_vector_qball_ell_sector_shooting_pilot_metrics.json"
VECTOR_EXACT = OUT / "mass_origin_vector_qball_exact_mass_table_handoff_retry_metrics.json"
VECTOR_HEAVY = OUT / "mass_origin_vector_qball_baryon_tau_neutron_fit_table_metrics.json"

NUMERICAL_BRANCH = ROOT / "scripts" / "quantum" / "mass_origin_vector_qball_numerical_solver_branch.py"
FULL_COUPLED_BRANCH = ROOT / "scripts" / "quantum" / "mass_origin_vector_qball_full_coupled_solver_branch.py"

NEXT_ROUTE = "8.7.56.233"
RESIDUAL_ROUTE = "trial3_relaunched_explicit_k_positive_integer_mode_builder_identification"
MISSING_ARTIFACT = "trial3_relaunched_explicit_k_positive_integer_mode_builder"


# Function: Return the current UTC timestamp in ISO 8601 format.
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# Function: Abort immediately when a required input path is missing.

def req(path: Path) -> None:
    if not path.exists():
        raise SystemExit(f"[fail] missing required input: {path}")


# Function: Read a UTF-8 JSON artifact.

def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


# Function: Read a UTF-8 text source.

def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# Function: Convert an absolute path into a repo-relative string.

def rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


# Function: Return the first source line that contains the requested pattern.

def hit(text: str, pattern: str) -> dict | None:
    for line_no, line in enumerate(text.splitlines(), start=1):
        if pattern in line:
            return {"pattern": pattern, "line": line_no, "text": line.strip()}

    return None


# Function: Return all source lines that contain the requested pattern.

def hits(text: str, pattern: str) -> list[dict]:
    found = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        if pattern in line:
            found.append({"pattern": pattern, "line": line_no, "text": line.strip()})

    return found


# Function: Build a standard result row.

def row(row_id: str, status: str, metric: str, value: float, note: str) -> dict:
    return {
        "row_id": row_id,
        "status": status,
        "metric": metric,
        "value": float(value),
        "note": note,
    }


# Function: Build a standard payload object.

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


# Function: Save a JSON artifact and its row CSV.

def write_artifact(stem: str, data: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    json_path = OUT / f"{stem}_metrics.json"
    csv_path = OUT / f"{stem}_rows.csv"
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["row_id", "status", "metric", "value", "note"])
        writer.writeheader()
        writer.writerows(data["rows"])


# Function: Build a source-inventory target record.

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


# Function: Execute the relaunched Trial-3 integer-mode residual branch.

def main() -> None:
    for path in (
        PART1,
        PART3A,
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        TRIAL3_RELAUNCHED_SOURCE,
        TRIAL3_RELAUNCHED_PILOT,
        TRIAL3_RELAUNCHED_AUDIT,
        TRIAL3_RELAUNCHED_DECLARATION,
        TRIAL3_RELAUNCHED_DISPOSITION,
        TRIAL3_HIGH_MASS_ROUTE,
        VECTOR_SOLVER_SPEC,
        VECTOR_CONSTRAINT,
        VECTOR_NUMERICAL_PILOT,
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
    ai_context = read_json(AI_CONTEXT)
    numerical_text = read_text(NUMERICAL_BRANCH)
    full_text = read_text(FULL_COUPLED_BRANCH)

    trial3_relaunched_source = read_json(TRIAL3_RELAUNCHED_SOURCE)
    trial3_relaunched_pilot = read_json(TRIAL3_RELAUNCHED_PILOT)
    trial3_relaunched_audit = read_json(TRIAL3_RELAUNCHED_AUDIT)
    trial3_relaunched_declaration = read_json(TRIAL3_RELAUNCHED_DECLARATION)
    trial3_relaunched_disposition = read_json(TRIAL3_RELAUNCHED_DISPOSITION)
    trial3_high_mass_route = read_json(TRIAL3_HIGH_MASS_ROUTE)
    vector_solver_spec = read_json(VECTOR_SOLVER_SPEC)
    vector_constraint = read_json(VECTOR_CONSTRAINT)
    vector_numerical_pilot = read_json(VECTOR_NUMERICAL_PILOT)
    vector_exact = read_json(VECTOR_EXACT)
    vector_heavy = read_json(VECTOR_HEAVY)

    numerical_k_zero_hits = hits(numerical_text, '"k": 0,')
    numerical_k_zero_interpolation_hits = [
        entry for entry in numerical_k_zero_hits if 276 <= int(entry["line"]) <= 320
    ]
    full_node_count_zero_hits = hits(full_text, '"node_count_k": 0,')
    solver_axis_pattern = "k>0 after the base sectors are stable"
    solver_axis_present = solver_axis_pattern in str(vector_solver_spec["formulas"]["pilot_sector_rule"])
    bookkeeping_available = bool(vector_constraint["summary"]["k_node_bookkeeping_available"])
    interpolation_signature_present = hit(numerical_text, "def interpolate_integer_modes(scan_rows: list[dict], ell: int)")
    exact_builder_signature_present = hit(full_text, "def build_exact_ladder(")
    exact_builder_uses_mode_k = hit(full_text, '"k": int(mode["k"]),')
    integer_mode_rule_mentions_k0 = "k=0" in str(vector_numerical_pilot["formulas"]["integer_mode_rule"])

    inventory_targets = [
        target_record(
            "status_relaunched_trial3_next_step",
            STATUS,
            status_text,
            "current official next step は `8.7.56.229`",
            "STATUS must already point to the solver-side integer-mode residual.",
        ),
        target_record(
            "roadmap_relaunched_integer_mode_branch",
            ROADMAP,
            roadmap_text,
            "`8.7.56.229-.232` 試練3 relaunched explicit `k>0` integer-mode-table residual branch",
            "ROADMAP must already freeze the relaunched integer-mode residual branch.",
        ),
        {
            "file_key": "vector_solver_spec_k_positive_axis",
            "file": rel(VECTOR_SOLVER_SPEC),
            "pattern": solver_axis_pattern,
            "present": solver_axis_present,
            "note": "The solver specification must still reserve k>0 as the next same-family extension axis.",
            "evidence": {
                "formula_key": "pilot_sector_rule",
                "text": vector_solver_spec["formulas"]["pilot_sector_rule"],
            },
        },
        {
            "file_key": "vector_constraint_k_node_bookkeeping",
            "file": rel(VECTOR_CONSTRAINT),
            "pattern": "k_node_bookkeeping_available",
            "present": bookkeeping_available,
            "note": "The coupled constraint audit must still freeze k-node bookkeeping before any k>0 activation.",
            "evidence": {
                "summary_key": "k_node_bookkeeping_available",
                "value": vector_constraint["summary"]["k_node_bookkeeping_available"],
            },
        },
        {
            "file_key": "numerical_interpolation_signature",
            "file": rel(NUMERICAL_BRANCH),
            "pattern": "def interpolate_integer_modes(scan_rows: list[dict], ell: int)",
            "present": interpolation_signature_present is not None,
            "note": "The current numerical branch must still expose the integer-mode interpolation entry point.",
            "evidence": interpolation_signature_present,
        },
        {
            "file_key": "numerical_interpolation_k_zero_hardcode",
            "file": rel(NUMERICAL_BRANCH),
            "pattern": "\"k\": 0,",
            "present": len(numerical_k_zero_interpolation_hits) > 0,
            "note": "The interpolation builder currently freezes every generated integer mode at k=0.",
            "evidence": numerical_k_zero_interpolation_hits,
        },
        {
            "file_key": "full_coupled_exact_builder_signature",
            "file": rel(FULL_COUPLED_BRANCH),
            "pattern": "def build_exact_ladder(",
            "present": exact_builder_signature_present is not None,
            "note": "The full-coupled branch must still expose the exact-ladder builder entry point.",
            "evidence": exact_builder_signature_present,
        },
        {
            "file_key": "full_coupled_exact_builder_uses_mode_k",
            "file": rel(FULL_COUPLED_BRANCH),
            "pattern": "\"k\": int(mode[\"k\"]),",
            "present": exact_builder_uses_mode_k is not None,
            "note": "The exact builder reads mode['k'], so the missing artifact is upstream table generation and node bookkeeping, not the row label alone.",
            "evidence": exact_builder_uses_mode_k,
        },
        {
            "file_key": "full_coupled_zero_node_rows",
            "file": rel(FULL_COUPLED_BRANCH),
            "pattern": "\"node_count_k\": 0,",
            "present": len(full_node_count_zero_hits) > 0,
            "note": "The current exact-ladder rows still freeze node_count_k at zero.",
            "evidence": full_node_count_zero_hits,
        },
        {
            "file_key": "numerical_integer_mode_rule_mentions_k0",
            "file": rel(VECTOR_NUMERICAL_PILOT),
            "pattern": "k=0",
            "present": integer_mode_rule_mentions_k0,
            "note": "The existing numerical metrics still define the integer-mode rule as a first k=0 ladder.",
            "evidence": {
                "formula_key": "integer_mode_rule",
                "text": vector_numerical_pilot["formulas"]["integer_mode_rule"],
            },
        },
    ]
    inventory_ready = all(bool(item["present"]) for item in inventory_targets)

    integer_mode_table_available = bool(
        solver_axis_present
        and bookkeeping_available
        and not numerical_k_zero_interpolation_hits
        and not full_node_count_zero_hits
    )
    integer_mode_interpolation_available = bool(
        interpolation_signature_present is not None and not numerical_k_zero_interpolation_hits
    )
    exact_ladder_node_axis_available = bool(
        exact_builder_signature_present is not None
        and exact_builder_uses_mode_k is not None
        and not full_node_count_zero_hits
    )
    nonclosure_reason = (
        "interpolate_integer_modes_still_hardcodes_k_zero_and_full_coupled_rows_still_freeze_node_count_k_zero"
        if not integer_mode_table_available
        else None
    )

    common_inputs = {
        "part1_markdown": rel(PART1),
        "part3a_markdown": rel(PART3A),
        "status_markdown": rel(STATUS),
        "roadmap_markdown": rel(ROADMAP),
        "ai_context_json": rel(AI_CONTEXT),
        "mass_origin_v2_trial3_relaunched_weak_sector_source_inventory_json": rel(TRIAL3_RELAUNCHED_SOURCE),
        "mass_origin_v2_trial3_relaunched_weak_sector_pilot_json": rel(TRIAL3_RELAUNCHED_PILOT),
        "mass_origin_v2_trial3_relaunched_weinberg_angle_weak_coupling_audit_json": rel(TRIAL3_RELAUNCHED_AUDIT),
        "mass_origin_v2_trial3_relaunched_declaration_gate_json": rel(TRIAL3_RELAUNCHED_DECLARATION),
        "mass_origin_v2_trial3_relaunched_paper_sync_trial4_disposition_gate_json": rel(TRIAL3_RELAUNCHED_DISPOSITION),
        "mass_origin_v2_trial3_explicit_k_positive_extension_route_contract_json": rel(TRIAL3_HIGH_MASS_ROUTE),
        "mass_origin_vector_qball_solver_spec_json": rel(VECTOR_SOLVER_SPEC),
        "mass_origin_vector_qball_coupled_constraint_freeze_audit_json": rel(VECTOR_CONSTRAINT),
        "mass_origin_vector_qball_ell_sector_shooting_pilot_json": rel(VECTOR_NUMERICAL_PILOT),
        "mass_origin_vector_qball_exact_mass_table_handoff_retry_json": rel(VECTOR_EXACT),
        "mass_origin_vector_qball_baryon_tau_neutron_fit_table_json": rel(VECTOR_HEAVY),
        "mass_origin_vector_qball_numerical_solver_branch_py": rel(NUMERICAL_BRANCH),
        "mass_origin_vector_qball_full_coupled_solver_branch_py": rel(FULL_COUPLED_BRANCH),
    }

    inventory = payload(
        "8.7.56.229",
        "Trial-3 relaunched explicit k-positive integer-mode source inventory",
        common_inputs,
        "Inventory the solver-side source pack that controls whether the relaunched weak-sector branch can activate an executable node-resolved k>0 integer-mode table.",
        {
            "inventory_rule": "freeze the solver specification, k-node bookkeeping, interpolation hardcode, and full-coupled zero-node rows in one machine-readable pack",
            "solver_axis_rule": "the current canon admits k>0 only at the solver level when the base sectors are stable",
            "blocking_rule": "if interpolation still emits k=0-only modes and the exact builder still freezes node_count_k=0, the executable integer-mode table is still absent",
        },
        [
            row(
                "trial3_relaunched_integer_mode_source_inventory_complete",
                "pass",
                "Trial-3 relaunched explicit k-positive integer-mode source inventory complete",
                1,
                "The solver-side residual source inventory is frozen.",
            ),
            row(
                "trial3_relaunched_integer_mode_required_source_count",
                "pass",
                "required solver-side source count",
                len(inventory_targets),
                "The residual pack needs the relaunch outputs plus the two current solver implementations.",
            ),
            row(
                "trial3_relaunched_integer_mode_solver_axis_present",
                "pass" if solver_axis_present else "reject",
                "solver specification keeps the explicit k-positive axis",
                1 if solver_axis_present else 0,
                "The solver spec must explicitly reserve k>0 before the residual can be pursued honestly.",
            ),
            row(
                "trial3_relaunched_integer_mode_bookkeeping_present",
                "pass" if bookkeeping_available else "reject",
                "k-node bookkeeping available",
                1 if bookkeeping_available else 0,
                "The coupled constraint audit must already define what k counts.",
            ),
        ],
        {
            "inventory_ready": inventory_ready,
            "explicit_k_positive_solver_axis_present": solver_axis_present,
            "explicit_k_positive_k_node_bookkeeping_available": bookkeeping_available,
            "interpolate_integer_modes_k_zero_hardcode_present": bool(numerical_k_zero_interpolation_hits),
            "full_coupled_zero_node_row_present": bool(full_node_count_zero_hits),
            "first_route_to_close_or_none": "trial3_relaunched_explicit_k_positive_node_interpolation_audit",
        },
        {
            "overall_status": "trial3_relaunched_integer_mode_source_inventory_frozen",
            "trial3_branch_closeable": False,
            "advance_to_8_7_56_230": True,
            "next_required_artifacts": ["trial3_relaunched_explicit_k_positive_node_interpolation_audit"],
        },
        {
            "inventory_targets": inventory_targets,
            "relaunch_source_summary": trial3_relaunched_source["summary"],
            "relaunch_declaration_summary": trial3_relaunched_declaration["summary"],
            "relaunch_disposition_summary": trial3_relaunched_disposition["summary"],
            "part1_micro_chiral_line": hit(part1_text, "左手系カイラル流"),
            "part3a_exact_vector_hierarchy_line": hit(part3a_text, "exact vector hierarchy"),
        },
    )

    audit = payload(
        "8.7.56.230",
        "Trial-3 relaunched explicit k-positive node/interpolation audit",
        common_inputs,
        "Audit exactly where the current solver chain stops the relaunched weak-sector branch from emitting an executable node-resolved k>0 exact ladder.",
        {
            "interpolation_rule": "the integer-mode interpolation must expose k as an active node label instead of emitting a frozen k=0 row family",
            "exact_builder_rule": "the full-coupled builder must preserve node_count_k beyond zero so the exact ladder can carry the activated k-axis",
            "closeout_rule": "the explicit k-positive integer-mode table exists only when both interpolation and exact-ladder bookkeeping are executable under the current canon",
        },
        [
            row(
                "trial3_relaunched_integer_mode_interpolation_available",
                "pass" if integer_mode_interpolation_available else "reject",
                "explicit k-positive integer-mode interpolation available",
                1 if integer_mode_interpolation_available else 0,
                "Interpolation remains blocked while interpolate_integer_modes still emits k=0-only rows.",
            ),
            row(
                "trial3_relaunched_exact_ladder_node_axis_available",
                "pass" if exact_ladder_node_axis_available else "reject",
                "explicit k-positive exact-ladder node axis available",
                1 if exact_ladder_node_axis_available else 0,
                "The exact builder remains blocked while node_count_k stays frozen at zero.",
            ),
            row(
                "trial3_relaunched_explicit_k_positive_integer_mode_table_available",
                "pass" if integer_mode_table_available else "reject",
                "relaunched explicit k-positive integer-mode table available",
                1 if integer_mode_table_available else 0,
                "The solver chain does not yet expose an executable node-resolved k>0 ladder.",
            ),
        ],
        {
            "trial3_relaunched_explicit_k_positive_integer_mode_table_available": integer_mode_table_available,
            "trial3_relaunched_explicit_k_positive_integer_mode_interpolation_available": integer_mode_interpolation_available,
            "trial3_relaunched_explicit_k_positive_exact_ladder_node_axis_available": exact_ladder_node_axis_available,
            "identification_nonclosure_reason_or_none": nonclosure_reason,
            "first_route_to_close_or_none": "trial3_relaunched_declaration_second_gate",
        },
        {
            "overall_status": "trial3_relaunched_integer_mode_node_interpolation_audited",
            "trial3_branch_closeable": False,
            "advance_to_8_7_56_231": True,
            "next_required_artifacts": ["trial3_relaunched_declaration_second_gate"],
        },
        {
            "relaunch_pilot_summary": trial3_relaunched_pilot["summary"],
            "relaunch_weak_audit_summary": trial3_relaunched_audit["summary"],
            "vector_solver_spec_formulas": vector_solver_spec["formulas"],
            "vector_constraint_summary": vector_constraint["summary"],
            "numerical_interpolation_k_zero_hits": numerical_k_zero_interpolation_hits,
            "full_coupled_zero_node_hits": full_node_count_zero_hits,
            "vector_exact_summary": vector_exact["summary"],
            "vector_heavy_summary": vector_heavy["summary"],
        },
    )

    declaration = payload(
        "8.7.56.231",
        "Trial-3 relaunched declaration second gate",
        common_inputs,
        "Freeze whether the solver-side explicit k-positive integer-mode residual is already closeable or whether a deeper builder-specific route is required.",
        {
            "gate_rule": "the relaunched Trial-3 branch closes only if the executable integer-mode table exists and can be carried into the exact ladder",
            "reserve_rule": "Trial-2 paper-side sync stays unlocked reserve work while the scientific weak-sector solver residual remains open",
        },
        [
            row(
                "trial3_relaunched_second_declaration_gate_complete",
                "pass",
                "Trial-3 relaunched second declaration gate complete",
                1,
                "The second declaration gate is frozen.",
            ),
            row(
                "trial3_relaunched_second_branch_closeable",
                "pass" if integer_mode_table_available else "reject",
                "relaunched explicit k-positive integer-mode branch closeable",
                1 if integer_mode_table_available else 0,
                "The branch does not close while the executable integer-mode table is still absent.",
            ),
            row(
                "trial3_relaunched_second_residual_route_required",
                "pass" if not integer_mode_table_available else "reject",
                "relaunched solver-side residual route required",
                1 if not integer_mode_table_available else 0,
                "A deeper builder-specific residual route remains required after the node/interpolation audit.",
            ),
            row(
                "trial3_relaunched_trial2_paper_sync_execute_now",
                "reject",
                "execute Trial-2 paper-side sync now",
                0,
                "Trial-2 paper sync stays unlocked reserve work while the scientific Trial-3 route remains open.",
            ),
        ],
        {
            "trial3_relaunched_branch_closeable": integer_mode_table_available,
            "trial3_relaunched_residual_route_required": not integer_mode_table_available,
            "trial2_paper_side_sync_execute_now": False,
            "trial4_deferred": True,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial3_relaunched_second_declaration_gate_frozen",
            "trial3_branch_closeable": integer_mode_table_available,
            "advance_to_8_7_56_232": True,
            "next_required_artifacts": ["trial3_relaunched_paper_sync_trial4_disposition_second_refresh"],
        },
        {
            "source_inventory_summary": inventory["summary"],
            "node_interpolation_audit_summary": audit["summary"],
            "prior_declaration_summary": trial3_relaunched_declaration["summary"],
        },
    )

    disposition = payload(
        "8.7.56.232",
        "Trial-2 paper-side sync / Trial-4 disposition second refresh",
        common_inputs,
        "Refresh the reserve/deferred ordering after the solver-side integer-mode audit and freeze the next residual route for the relaunched weak-sector mainline.",
        {
            "trial2_rule": "retain Trial-2 paper-side sync as unlocked reserve work while the scientific solver-side Trial-3 route is still open",
            "trial4_rule": "keep Trial-4 deferred until the relaunched Trial-3 branch loses all honest current-canon solver routes",
            "selected_residual_route": RESIDUAL_ROUTE,
            "missing_v2_artifact": MISSING_ARTIFACT,
        },
        [
            row(
                "trial3_relaunched_second_disposition_gate_complete",
                "pass",
                "Trial-3 relaunched second disposition gate complete",
                1,
                "The post-audit reserve/deferred ordering is frozen.",
            ),
            row(
                "trial3_relaunched_second_trial2_paper_side_sync_reserve_retained",
                "pass",
                "Trial-2 paper-side sync retained as unlocked reserve",
                1,
                "Trial-2 paper-side sync remains available but not yet promoted to the main scientific route.",
            ),
            row(
                "trial3_relaunched_second_trial4_deferred_retained",
                "pass",
                "Trial-4 deferred disposition retained",
                1,
                "Trial-4 remains deferred while the solver-side Trial-3 residual route is still honest.",
            ),
            row(
                "trial3_relaunched_second_next_residual_route_frozen",
                "pass",
                "relaunched explicit k-positive integer-mode builder residual route frozen",
                1,
                "The next residual route is the missing executable builder for node-resolved k>0 integer-mode rows.",
            ),
        ],
        {
            "selected_residual_route": RESIDUAL_ROUTE,
            "missing_v2_artifact": MISSING_ARTIFACT,
            "trial2_paper_side_sync_state": "unlocked_reserve_retained",
            "trial4_deferred": True,
            "split_contract_ready": True,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial3_relaunched_second_disposition_gate_frozen",
            "trial3_branch_closeable": integer_mode_table_available,
            "advance_to_8_7_56_233": True,
            "next_required_artifacts": [
                "trial3_relaunched_explicit_k_positive_integer_mode_builder_source_inventory",
                "trial3_relaunched_explicit_k_positive_integer_mode_builder_audit",
            ],
        },
        {
            "source_inventory_summary": inventory["summary"],
            "node_interpolation_audit_summary": audit["summary"],
            "declaration_summary": declaration["summary"],
            "prior_disposition_summary": trial3_relaunched_disposition["summary"],
            "ai_context_current_step": ai_context["current_step"],
        },
    )

    write_artifact("mass_origin_v2_trial3_relaunched_explicit_k_positive_integer_mode_source_inventory", inventory)
    write_artifact("mass_origin_v2_trial3_relaunched_explicit_k_positive_node_interpolation_audit", audit)
    write_artifact("mass_origin_v2_trial3_relaunched_declaration_second_gate", declaration)
    write_artifact("mass_origin_v2_trial3_relaunched_paper_sync_trial4_disposition_second_refresh", disposition)

    print("[ok] wrote:")
    print(" - mass_origin_v2_trial3_relaunched_explicit_k_positive_integer_mode_source_inventory_metrics.json")
    print(" - mass_origin_v2_trial3_relaunched_explicit_k_positive_node_interpolation_audit_metrics.json")
    print(" - mass_origin_v2_trial3_relaunched_declaration_second_gate_metrics.json")
    print(" - mass_origin_v2_trial3_relaunched_paper_sync_trial4_disposition_second_refresh_metrics.json")


# Function: Run the relaunched Trial-3 integer-mode residual branch from the command line.

if __name__ == "__main__":
    main()
