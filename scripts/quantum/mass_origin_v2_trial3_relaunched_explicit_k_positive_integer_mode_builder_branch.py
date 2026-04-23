#!/usr/bin/env python3
"""
Generate relaunched Trial-3 explicit k-positive integer-mode-builder artifacts.

This branch executes roadmap steps 8.7.56.233-.236.

The previous residual formalized that the weak-sector relaunch is blocked not by
W/Z wording but by the absence of an executable k-positive integer-mode table.
This branch narrows that solver-side blocker to the concrete builder surfaces
that still freeze k=0 in the current numerical and full-coupled pipelines.
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

TRIAL3_INTEGER_MODE_SOURCE = OUT / "mass_origin_v2_trial3_relaunched_explicit_k_positive_integer_mode_source_inventory_metrics.json"
TRIAL3_INTEGER_MODE_AUDIT = OUT / "mass_origin_v2_trial3_relaunched_explicit_k_positive_node_interpolation_audit_metrics.json"
TRIAL3_DECLARATION_SECOND = OUT / "mass_origin_v2_trial3_relaunched_declaration_second_gate_metrics.json"
TRIAL3_DISPOSITION_SECOND = OUT / "mass_origin_v2_trial3_relaunched_paper_sync_trial4_disposition_second_refresh_metrics.json"
TRIAL3_RELAUNCHED_PILOT = OUT / "mass_origin_v2_trial3_relaunched_weak_sector_pilot_metrics.json"
VECTOR_SOLVER_SPEC = OUT / "mass_origin_vector_qball_solver_spec_metrics.json"
VECTOR_CONSTRAINT = OUT / "mass_origin_vector_qball_coupled_constraint_freeze_audit_metrics.json"
VECTOR_NUMERICAL_PILOT = OUT / "mass_origin_vector_qball_ell_sector_shooting_pilot_metrics.json"
VECTOR_EXACT = OUT / "mass_origin_vector_qball_exact_mass_table_handoff_retry_metrics.json"
VECTOR_HEAVY = OUT / "mass_origin_vector_qball_baryon_tau_neutron_fit_table_metrics.json"

NUMERICAL_BRANCH = ROOT / "scripts" / "quantum" / "mass_origin_vector_qball_numerical_solver_branch.py"
FULL_COUPLED_BRANCH = ROOT / "scripts" / "quantum" / "mass_origin_vector_qball_full_coupled_solver_branch.py"

NEXT_ROUTE = "8.7.56.237"
RESIDUAL_ROUTE = "trial3_relaunched_explicit_k_positive_node_resolved_interpolation_builder_identification"
MISSING_ARTIFACT = "trial3_relaunched_explicit_k_positive_node_resolved_interpolation_builder"


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


# Function: Execute the relaunched Trial-3 integer-mode-builder residual branch.

def main() -> None:
    for path in (
        PART1,
        PART3A,
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        TRIAL3_INTEGER_MODE_SOURCE,
        TRIAL3_INTEGER_MODE_AUDIT,
        TRIAL3_DECLARATION_SECOND,
        TRIAL3_DISPOSITION_SECOND,
        TRIAL3_RELAUNCHED_PILOT,
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

    trial3_integer_mode_source = read_json(TRIAL3_INTEGER_MODE_SOURCE)
    trial3_integer_mode_audit = read_json(TRIAL3_INTEGER_MODE_AUDIT)
    trial3_declaration_second = read_json(TRIAL3_DECLARATION_SECOND)
    trial3_disposition_second = read_json(TRIAL3_DISPOSITION_SECOND)
    trial3_relaunched_pilot = read_json(TRIAL3_RELAUNCHED_PILOT)
    vector_solver_spec = read_json(VECTOR_SOLVER_SPEC)
    vector_constraint = read_json(VECTOR_CONSTRAINT)
    vector_numerical_pilot = read_json(VECTOR_NUMERICAL_PILOT)
    vector_exact = read_json(VECTOR_EXACT)
    vector_heavy = read_json(VECTOR_HEAVY)

    interpolation_signature = hit(
        numerical_text, "def interpolate_integer_modes(scan_rows: list[dict], ell: int)"
    )
    interpolation_signature_text = interpolation_signature["text"] if interpolation_signature else ""
    interpolation_has_k_argument = ", k" in interpolation_signature_text
    interpolation_k_zero_hits = [
        entry for entry in hits(numerical_text, '"k": 0,') if 276 <= int(entry["line"]) <= 320
    ]
    trial_state_signature = hit(
        numerical_text, "def build_trial_state_rows(scalar_modes: list[dict], sector_rows: list[dict])"
    )
    trial_state_k_zero_hits = [
        entry for entry in hits(numerical_text, '"k": 0,') if 323 <= int(entry["line"]) <= 345
    ]
    trial_state_id_zero = hit(numerical_text, 'trial_state_id": f"M_({n},0,{ell},{s})"')
    exact_builder_signature = hit(full_text, "def build_exact_ladder(")
    exact_builder_mode_k_passthrough = hit(full_text, '"k": int(mode["k"]),')
    exact_builder_node_count_zero_hits = hits(full_text, '"node_count_k": 0,')
    solver_axis_present = "k>0 after the base sectors are stable" in str(
        vector_solver_spec["formulas"]["pilot_sector_rule"]
    )
    bookkeeping_available = bool(vector_constraint["summary"]["k_node_bookkeeping_available"])
    integer_mode_rule_text = str(vector_numerical_pilot["formulas"]["integer_mode_rule"])
    integer_mode_rule_mentions_first_k0_ladder = "first `k=0` base ladder" in integer_mode_rule_text

    inventory_targets = [
        target_record(
            "status_relaunched_trial3_next_step",
            STATUS,
            status_text,
            "current official next step は `8.7.56.233`",
            "STATUS must already point to the builder residual branch.",
        ),
        target_record(
            "roadmap_relaunched_builder_branch",
            ROADMAP,
            roadmap_text,
            "`8.7.56.233-.236` 試練3 relaunched explicit `k>0` integer-mode-builder residual branch",
            "ROADMAP must already freeze the builder residual branch.",
        ),
        {
            "file_key": "numerical_interpolation_signature",
            "file": rel(NUMERICAL_BRANCH),
            "pattern": "def interpolate_integer_modes(scan_rows: list[dict], ell: int)",
            "present": interpolation_signature is not None,
            "note": "The numerical builder must still expose the current interpolation entry point.",
            "evidence": interpolation_signature,
        },
        {
            "file_key": "numerical_interpolation_signature_lacks_k_argument",
            "file": rel(NUMERICAL_BRANCH),
            "pattern": "no explicit k argument",
            "present": not interpolation_has_k_argument,
            "note": "The interpolation signature still lacks a dedicated k or node-axis argument.",
            "evidence": interpolation_signature,
        },
        {
            "file_key": "numerical_interpolation_k_zero_hardcode",
            "file": rel(NUMERICAL_BRANCH),
            "pattern": "\"k\": 0,",
            "present": len(interpolation_k_zero_hits) > 0,
            "note": "The interpolation builder still emits k=0 rows only.",
            "evidence": interpolation_k_zero_hits,
        },
        {
            "file_key": "trial_state_builder_signature",
            "file": rel(NUMERICAL_BRANCH),
            "pattern": "def build_trial_state_rows(scalar_modes: list[dict], sector_rows: list[dict])",
            "present": trial_state_signature is not None,
            "note": "The downstream trial-state row builder must still expose the current state-row surface.",
            "evidence": trial_state_signature,
        },
        {
            "file_key": "trial_state_builder_k_zero_hardcode",
            "file": rel(NUMERICAL_BRANCH),
            "pattern": "\"k\": 0,",
            "present": len(trial_state_k_zero_hits) > 0,
            "note": "The trial-state inventory still freezes every generated state row at k=0.",
            "evidence": trial_state_k_zero_hits,
        },
        {
            "file_key": "trial_state_id_k_zero_pattern",
            "file": rel(NUMERICAL_BRANCH),
            "pattern": "M_(n,0,ell,s)",
            "present": trial_state_id_zero is not None,
            "note": "The trial-state identifier still bakes the zero-node label into the row id.",
            "evidence": trial_state_id_zero,
        },
        {
            "file_key": "full_coupled_exact_builder_mode_k_passthrough",
            "file": rel(FULL_COUPLED_BRANCH),
            "pattern": "\"k\": int(mode[\"k\"]),",
            "present": exact_builder_mode_k_passthrough is not None,
            "note": "The exact builder can already pass through mode['k'] once upstream rows exist.",
            "evidence": exact_builder_mode_k_passthrough,
        },
        {
            "file_key": "full_coupled_exact_builder_node_count_zero",
            "file": rel(FULL_COUPLED_BRANCH),
            "pattern": "\"node_count_k\": 0,",
            "present": len(exact_builder_node_count_zero_hits) > 0,
            "note": "The exact builder still freezes node_count_k at zero even when mode['k'] is passed through.",
            "evidence": exact_builder_node_count_zero_hits,
        },
    ]
    inventory_ready = all(bool(item["present"]) for item in inventory_targets)

    node_resolved_interpolation_builder_available = bool(
        interpolation_signature is not None
        and interpolation_has_k_argument
        and not interpolation_k_zero_hits
        and not trial_state_k_zero_hits
        and trial_state_id_zero is None
    )
    node_count_row_builder_available = bool(
        exact_builder_signature is not None
        and exact_builder_mode_k_passthrough is not None
        and not exact_builder_node_count_zero_hits
    )
    integer_mode_builder_available = bool(
        solver_axis_present
        and bookkeeping_available
        and node_resolved_interpolation_builder_available
        and node_count_row_builder_available
    )
    nonclosure_reason = (
        "interpolate_integer_modes_signature_still_excludes_k_and_emits_only_k_zero_mode_rows"
        if not node_resolved_interpolation_builder_available
        else "full_coupled_exact_rows_still_freeze_node_count_k_zero"
        if not node_count_row_builder_available
        else None
    )

    common_inputs = {
        "part1_markdown": rel(PART1),
        "part3a_markdown": rel(PART3A),
        "status_markdown": rel(STATUS),
        "roadmap_markdown": rel(ROADMAP),
        "ai_context_json": rel(AI_CONTEXT),
        "mass_origin_v2_trial3_relaunched_explicit_k_positive_integer_mode_source_inventory_json": rel(TRIAL3_INTEGER_MODE_SOURCE),
        "mass_origin_v2_trial3_relaunched_explicit_k_positive_node_interpolation_audit_json": rel(TRIAL3_INTEGER_MODE_AUDIT),
        "mass_origin_v2_trial3_relaunched_declaration_second_gate_json": rel(TRIAL3_DECLARATION_SECOND),
        "mass_origin_v2_trial3_relaunched_paper_sync_trial4_disposition_second_refresh_json": rel(TRIAL3_DISPOSITION_SECOND),
        "mass_origin_v2_trial3_relaunched_weak_sector_pilot_json": rel(TRIAL3_RELAUNCHED_PILOT),
        "mass_origin_vector_qball_solver_spec_json": rel(VECTOR_SOLVER_SPEC),
        "mass_origin_vector_qball_coupled_constraint_freeze_audit_json": rel(VECTOR_CONSTRAINT),
        "mass_origin_vector_qball_ell_sector_shooting_pilot_json": rel(VECTOR_NUMERICAL_PILOT),
        "mass_origin_vector_qball_exact_mass_table_handoff_retry_json": rel(VECTOR_EXACT),
        "mass_origin_vector_qball_baryon_tau_neutron_fit_table_json": rel(VECTOR_HEAVY),
        "mass_origin_vector_qball_numerical_solver_branch_py": rel(NUMERICAL_BRANCH),
        "mass_origin_vector_qball_full_coupled_solver_branch_py": rel(FULL_COUPLED_BRANCH),
    }

    inventory = payload(
        "8.7.56.233",
        "Trial-3 relaunched explicit k-positive integer-mode-builder source inventory",
        common_inputs,
        "Inventory the concrete builder surfaces that still stop the weak-sector relaunch from emitting node-resolved k-positive mode rows.",
        {
            "inventory_rule": "freeze the interpolation signature, the numerical row builders, and the exact-row node bookkeeping in one machine-readable pack",
            "builder_rule": "a builder-level residual exists when the interpolation signature and downstream row emitters still hardcode k=0 or omit the node axis entirely",
            "exact_row_rule": "the full-coupled ladder is downstream evidence only; it becomes closeable after upstream k-positive rows and node counts are emitted honestly",
        },
        [
            row(
                "trial3_relaunched_integer_mode_builder_source_inventory_complete",
                "pass",
                "Trial-3 relaunched explicit k-positive integer-mode-builder source inventory complete",
                1,
                "The builder residual source inventory is frozen.",
            ),
            row(
                "trial3_relaunched_integer_mode_builder_required_source_count",
                "pass",
                "required builder-side source count",
                len(inventory_targets),
                "The builder residual pack needs the prior integer-mode artifacts plus the two solver implementations.",
            ),
            row(
                "trial3_relaunched_interpolation_signature_lacks_k_argument",
                "pass" if not interpolation_has_k_argument else "reject",
                "interpolation signature still lacks an explicit k argument",
                1 if not interpolation_has_k_argument else 0,
                "The interpolation entry point still exposes only scan_rows and ell.",
            ),
            row(
                "trial3_relaunched_trial_state_rows_still_bake_k_zero",
                "pass" if len(trial_state_k_zero_hits) > 0 and trial_state_id_zero is not None else "reject",
                "trial-state rows still bake the zero-node label",
                1 if len(trial_state_k_zero_hits) > 0 and trial_state_id_zero is not None else 0,
                "The downstream trial-state row builder still emits k=0-only state ids and rows.",
            ),
        ],
        {
            "inventory_ready": inventory_ready,
            "interpolation_signature_present": interpolation_signature is not None,
            "interpolation_signature_has_k_argument": interpolation_has_k_argument,
            "interpolation_k_zero_hardcode_present": bool(interpolation_k_zero_hits),
            "trial_state_row_k_zero_hardcode_present": bool(trial_state_k_zero_hits),
            "full_coupled_node_count_k_zero_present": bool(exact_builder_node_count_zero_hits),
            "first_route_to_close_or_none": "trial3_relaunched_explicit_k_positive_integer_mode_builder_identification_audit",
        },
        {
            "overall_status": "trial3_relaunched_integer_mode_builder_source_inventory_frozen",
            "trial3_branch_closeable": False,
            "advance_to_8_7_56_234": True,
            "next_required_artifacts": ["trial3_relaunched_explicit_k_positive_integer_mode_builder_identification_audit"],
        },
        {
            "inventory_targets": inventory_targets,
            "prior_integer_mode_source_summary": trial3_integer_mode_source["summary"],
            "prior_integer_mode_audit_summary": trial3_integer_mode_audit["summary"],
            "prior_disposition_summary": trial3_disposition_second["summary"],
            "part1_micro_chiral_line": hit(part1_text, "左手系カイラル流"),
            "part3a_exact_vector_hierarchy_line": hit(part3a_text, "exact vector hierarchy"),
        },
    )

    audit = payload(
        "8.7.56.234",
        "Trial-3 relaunched explicit k-positive integer-mode-builder identification audit",
        common_inputs,
        "Audit whether the current solver already has a node-resolved k-positive interpolation builder and a non-frozen exact-row node-count builder.",
        {
            "interpolation_builder_rule": "the interpolation builder is available only if its signature exposes the node axis and it stops emitting k=0-only mode rows",
            "trial_state_rule": "the trial-state builder must stop hardcoding k=0 in both state rows and trial_state_id labels",
            "exact_row_builder_rule": "the full-coupled exact-row builder must stop freezing node_count_k at zero once k-positive rows are propagated",
        },
        [
            row(
                "trial3_relaunched_node_resolved_interpolation_builder_available",
                "pass" if node_resolved_interpolation_builder_available else "reject",
                "node-resolved k-positive interpolation builder available",
                1 if node_resolved_interpolation_builder_available else 0,
                "The interpolation builder remains blocked while its signature excludes k and it emits k=0-only rows.",
            ),
            row(
                "trial3_relaunched_node_count_row_builder_available",
                "pass" if node_count_row_builder_available else "reject",
                "node-count exact-row builder available",
                1 if node_count_row_builder_available else 0,
                "The exact-row builder remains blocked while node_count_k stays frozen at zero.",
            ),
            row(
                "trial3_relaunched_explicit_k_positive_integer_mode_builder_available",
                "pass" if integer_mode_builder_available else "reject",
                "relaunched explicit k-positive integer-mode builder available",
                1 if integer_mode_builder_available else 0,
                "The current solver chain still lacks the builder that emits honest node-resolved k-positive rows.",
            ),
        ],
        {
            "trial3_relaunched_explicit_k_positive_integer_mode_builder_available": integer_mode_builder_available,
            "trial3_relaunched_explicit_k_positive_node_resolved_interpolation_builder_available": node_resolved_interpolation_builder_available,
            "trial3_relaunched_explicit_k_positive_node_count_row_builder_available": node_count_row_builder_available,
            "identification_nonclosure_reason_or_none": nonclosure_reason,
            "first_route_to_close_or_none": "trial3_relaunched_declaration_third_gate",
        },
        {
            "overall_status": "trial3_relaunched_integer_mode_builder_identification_audited",
            "trial3_branch_closeable": False,
            "advance_to_8_7_56_235": True,
            "next_required_artifacts": ["trial3_relaunched_declaration_third_gate"],
        },
        {
            "prior_relaunch_pilot_summary": trial3_relaunched_pilot["summary"],
            "prior_integer_mode_audit_summary": trial3_integer_mode_audit["summary"],
            "vector_solver_spec_formulas": vector_solver_spec["formulas"],
            "vector_constraint_summary": vector_constraint["summary"],
            "vector_numerical_integer_mode_rule": vector_numerical_pilot["formulas"]["integer_mode_rule"],
            "interpolation_signature": interpolation_signature,
            "trial_state_signature": trial_state_signature,
            "interpolation_k_zero_hits": interpolation_k_zero_hits,
            "trial_state_k_zero_hits": trial_state_k_zero_hits,
            "trial_state_id_zero": trial_state_id_zero,
            "exact_builder_signature": exact_builder_signature,
            "exact_builder_mode_k_passthrough": exact_builder_mode_k_passthrough,
            "exact_builder_node_count_zero_hits": exact_builder_node_count_zero_hits,
            "vector_exact_summary": vector_exact["summary"],
            "vector_heavy_summary": vector_heavy["summary"],
        },
    )

    declaration = payload(
        "8.7.56.235",
        "Trial-3 relaunched declaration third gate",
        common_inputs,
        "Freeze whether the builder residual is already closeable or whether the interpolation-builder sub-route must become the next official blocker.",
        {
            "gate_rule": "the relaunched Trial-3 builder branch closes only if both the node-resolved interpolation builder and the node-count row builder are available",
            "reserve_rule": "Trial-2 paper-side sync stays unlocked reserve work while the scientific builder residual remains open",
        },
        [
            row(
                "trial3_relaunched_third_declaration_gate_complete",
                "pass",
                "Trial-3 relaunched third declaration gate complete",
                1,
                "The third declaration gate is frozen.",
            ),
            row(
                "trial3_relaunched_builder_branch_closeable",
                "pass" if integer_mode_builder_available else "reject",
                "relaunched explicit k-positive integer-mode-builder branch closeable",
                1 if integer_mode_builder_available else 0,
                "The branch does not close while the node-resolved interpolation builder is still absent.",
            ),
            row(
                "trial3_relaunched_builder_residual_route_required",
                "pass" if not integer_mode_builder_available else "reject",
                "relaunched builder residual route required",
                1 if not integer_mode_builder_available else 0,
                "A deeper interpolation-builder residual route remains required after the builder audit.",
            ),
            row(
                "trial3_relaunched_trial2_paper_sync_execute_now",
                "reject",
                "execute Trial-2 paper-side sync now",
                0,
                "Trial-2 paper sync stays unlocked reserve work while the scientific Trial-3 builder route remains open.",
            ),
        ],
        {
            "trial3_relaunched_branch_closeable": integer_mode_builder_available,
            "trial3_relaunched_residual_route_required": not integer_mode_builder_available,
            "trial2_paper_side_sync_execute_now": False,
            "trial4_deferred": True,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial3_relaunched_third_declaration_gate_frozen",
            "trial3_branch_closeable": integer_mode_builder_available,
            "advance_to_8_7_56_236": True,
            "next_required_artifacts": ["trial3_relaunched_paper_sync_trial4_disposition_third_refresh"],
        },
        {
            "source_inventory_summary": inventory["summary"],
            "builder_identification_summary": audit["summary"],
            "prior_declaration_summary": trial3_declaration_second["summary"],
        },
    )

    disposition = payload(
        "8.7.56.236",
        "Trial-2 paper-side sync / Trial-4 disposition third refresh",
        common_inputs,
        "Refresh the reserve/deferred ordering after the builder audit and freeze the next residual route for the relaunched weak-sector mainline.",
        {
            "trial2_rule": "retain Trial-2 paper-side sync as unlocked reserve work while the scientific solver-side Trial-3 builder route is still open",
            "trial4_rule": "keep Trial-4 deferred until the relaunched Trial-3 branch loses all honest current-canon builder routes",
            "selected_residual_route": RESIDUAL_ROUTE,
            "missing_v2_artifact": MISSING_ARTIFACT,
        },
        [
            row(
                "trial3_relaunched_third_disposition_gate_complete",
                "pass",
                "Trial-3 relaunched third disposition gate complete",
                1,
                "The post-audit reserve/deferred ordering is frozen.",
            ),
            row(
                "trial3_relaunched_third_trial2_paper_side_sync_reserve_retained",
                "pass",
                "Trial-2 paper-side sync retained as unlocked reserve",
                1,
                "Trial-2 paper-side sync remains available but not yet promoted to the main scientific route.",
            ),
            row(
                "trial3_relaunched_third_trial4_deferred_retained",
                "pass",
                "Trial-4 deferred disposition retained",
                1,
                "Trial-4 remains deferred while the builder residual route is still honest.",
            ),
            row(
                "trial3_relaunched_third_next_residual_route_frozen",
                "pass",
                "relaunched explicit k-positive node-resolved interpolation-builder residual route frozen",
                1,
                "The next residual route is the missing interpolation builder that should emit node-resolved k-positive rows.",
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
            "overall_status": "trial3_relaunched_third_disposition_gate_frozen",
            "trial3_branch_closeable": integer_mode_builder_available,
            "advance_to_8_7_56_237": True,
            "next_required_artifacts": [
                "trial3_relaunched_explicit_k_positive_node_resolved_interpolation_builder_source_inventory",
                "trial3_relaunched_explicit_k_positive_node_resolved_interpolation_builder_audit",
            ],
        },
        {
            "source_inventory_summary": inventory["summary"],
            "builder_identification_summary": audit["summary"],
            "declaration_summary": declaration["summary"],
            "prior_disposition_summary": trial3_disposition_second["summary"],
            "ai_context_current_step": ai_context["current_step"],
        },
    )

    write_artifact("mass_origin_v2_trial3_relaunched_explicit_k_positive_integer_mode_builder_source_inventory", inventory)
    write_artifact("mass_origin_v2_trial3_relaunched_explicit_k_positive_integer_mode_builder_identification_audit", audit)
    write_artifact("mass_origin_v2_trial3_relaunched_declaration_third_gate", declaration)
    write_artifact("mass_origin_v2_trial3_relaunched_paper_sync_trial4_disposition_third_refresh", disposition)

    print("[ok] wrote:")
    print(" - mass_origin_v2_trial3_relaunched_explicit_k_positive_integer_mode_builder_source_inventory_metrics.json")
    print(" - mass_origin_v2_trial3_relaunched_explicit_k_positive_integer_mode_builder_identification_audit_metrics.json")
    print(" - mass_origin_v2_trial3_relaunched_declaration_third_gate_metrics.json")
    print(" - mass_origin_v2_trial3_relaunched_paper_sync_trial4_disposition_third_refresh_metrics.json")


# Function: Run the relaunched Trial-3 integer-mode-builder residual branch from the command line.

if __name__ == "__main__":
    main()
