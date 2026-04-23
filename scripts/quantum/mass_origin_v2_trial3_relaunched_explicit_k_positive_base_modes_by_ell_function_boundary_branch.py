#!/usr/bin/env python3
"""
Generate relaunched Trial-3 explicit k-positive base-modes-by-ell function-boundary artifacts.

This branch executes roadmap steps 8.7.56.261-.264.

The previous residual showed that the solver-side blocker is no longer the
generic signature surface by itself. Both numerical and full-coupled
branches still expose typed function boundaries that fix `base_modes_by_ell`
as `dict[int, list[dict]]`, so this branch freezes that boundary layer and
shrinks the next residual route again.
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

PRIOR_SOURCE = (
    OUT
    / "mass_origin_v2_trial3_relaunched_explicit_k_positive_base_modes_by_ell_signature_surface_source_inventory_metrics.json"
)
PRIOR_AUDIT = (
    OUT
    / "mass_origin_v2_trial3_relaunched_explicit_k_positive_base_modes_by_ell_signature_surface_identification_audit_metrics.json"
)
PRIOR_DECLARATION = OUT / "mass_origin_v2_trial3_relaunched_declaration_ninth_gate_metrics.json"
PRIOR_DISPOSITION = OUT / "mass_origin_v2_trial3_relaunched_paper_sync_trial4_disposition_ninth_refresh_metrics.json"
NUMERICAL_BRANCH = ROOT / "scripts" / "quantum" / "mass_origin_vector_qball_numerical_solver_branch.py"
FULL_COUPLED_BRANCH = ROOT / "scripts" / "quantum" / "mass_origin_vector_qball_full_coupled_solver_branch.py"

NEXT_ROUTE = "8.7.56.265"
RESIDUAL_ROUTE = "trial3_relaunched_explicit_k_positive_base_modes_by_ell_parameter_contract_identification"
MISSING_ARTIFACT = "trial3_relaunched_explicit_k_positive_base_modes_by_ell_k_axis_parameter_contract"


# Function: Return the current UTC timestamp in ISO 8601 format.
def now_iso() -> str:
    """Return the current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


# Function: Abort when a required input path is missing.

def req(path: Path) -> None:
    """Abort when a required input path is missing."""
    if not path.exists():
        raise SystemExit(f"[fail] missing required input: {path}")


# Function: Read a UTF-8 JSON artifact.

def read_json(path: Path) -> dict:
    """Read a UTF-8 JSON artifact."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


# Function: Read a UTF-8 text file.

def read_text(path: Path) -> str:
    """Read a UTF-8 text file."""
    return path.read_text(encoding="utf-8")


# Function: Convert an absolute path to a repo-relative string.

def rel(path: Path) -> str:
    """Convert an absolute path to a repo-relative string."""
    return str(path.relative_to(ROOT)).replace("\\", "/")


# Function: Return the first source line that contains the given pattern.

def hit(text: str, pattern: str) -> dict | None:
    """Return the first source line that contains the requested pattern."""
    for line_no, line in enumerate(text.splitlines(), start=1):
        if pattern in line:
            return {"pattern": pattern, "line": line_no, "text": line.strip()}

    return None


# Function: Build a standard metrics row.

def row(row_id: str, status: str, metric: str, value: float, note: str) -> dict:
    """Build a standard metrics row."""
    return {
        "row_id": row_id,
        "status": status,
        "metric": metric,
        "value": float(value),
        "note": note,
    }


# Function: Build a payload in the shared artifact schema.

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
    """Build a payload in the shared artifact schema."""
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


# Function: Save a JSON artifact and its paired rows CSV.

def write_artifact(stem: str, data: dict) -> None:
    """Save a JSON artifact and its paired rows CSV."""
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
    """Build a source-inventory target record."""
    target_hit = hit(text, pattern)
    return {
        "file_key": file_key,
        "file": rel(path),
        "pattern": pattern,
        "present": target_hit is not None,
        "note": note,
        "evidence": target_hit,
    }


# Function: Execute the `.261-.264` branch and freeze the function-boundary residual.

def main() -> None:
    """Execute the relaunched Trial-3 base-modes-by-ell function-boundary residual branch."""
    for path in (
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        PRIOR_SOURCE,
        PRIOR_AUDIT,
        PRIOR_DECLARATION,
        PRIOR_DISPOSITION,
        NUMERICAL_BRANCH,
        FULL_COUPLED_BRANCH,
    ):
        req(path)

    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    ai_context = read_json(AI_CONTEXT)
    numerical_text = read_text(NUMERICAL_BRANCH)
    full_text = read_text(FULL_COUPLED_BRANCH)

    prior_source = read_json(PRIOR_SOURCE)
    prior_audit = read_json(PRIOR_AUDIT)
    prior_declaration = read_json(PRIOR_DECLARATION)
    prior_disposition = read_json(PRIOR_DISPOSITION)

    numerical_function_boundary = hit(numerical_text, "def build_spin_orbit_rows(")
    numerical_parameter_contract = hit(numerical_text, "base_modes_by_ell: dict[int, list[dict]],")
    numerical_callsite = hit(
        numerical_text, "split_rows = build_spin_orbit_rows(scalar_modes, base_modes_by_ell, lambda_rot)"
    )
    numerical_downstream_count = hit(
        numerical_text, "total_integer_modes = sum(len(modes) for modes in base_modes_by_ell.values())"
    )

    full_function_boundary = hit(full_text, "def build_exact_ladder(")
    full_parameter_contract = hit(full_text, "base_modes_by_ell: dict[int, list[dict]],")
    full_callsite = hit(full_text, "exact_rows = build_exact_ladder(scalar_modes, base_modes_by_ell, lambda_rot)")
    full_downstream_count = hit(full_text, "total_integer_modes = sum(len(rows) for rows in base_modes_by_ell.values())")

    numerical_boundary_surface_present = bool(
        numerical_function_boundary is not None
        and numerical_parameter_contract is not None
        and numerical_callsite is not None
        and numerical_downstream_count is not None
    )
    full_boundary_surface_present = bool(
        full_function_boundary is not None
        and full_parameter_contract is not None
        and full_callsite is not None
        and full_downstream_count is not None
    )
    inventory_ready = bool(numerical_boundary_surface_present and full_boundary_surface_present)

    numerical_base_modes_by_ell_parameter_contract_available = False
    full_base_modes_by_ell_parameter_contract_available = False
    base_modes_by_ell_parameter_contract_surface_available = False
    base_modes_by_ell_function_boundary_branch_available = False

    nonclosure_reason = (
        "base_modes_by_ell_parameter_contracts_still_fix_dict_int_list_dict_and_block_k_axis_boundary_upgrade"
    )

    common_inputs = {
        "status_markdown": rel(STATUS),
        "roadmap_markdown": rel(ROADMAP),
        "ai_context_json": rel(AI_CONTEXT),
        "prior_base_modes_by_ell_signature_surface_source_inventory_json": rel(PRIOR_SOURCE),
        "prior_base_modes_by_ell_signature_surface_identification_audit_json": rel(PRIOR_AUDIT),
        "prior_declaration_ninth_gate_json": rel(PRIOR_DECLARATION),
        "prior_paper_sync_trial4_disposition_ninth_refresh_json": rel(PRIOR_DISPOSITION),
        "mass_origin_vector_qball_numerical_solver_branch_py": rel(NUMERICAL_BRANCH),
        "mass_origin_vector_qball_full_coupled_solver_branch_py": rel(FULL_COUPLED_BRANCH),
    }

    inventory_targets = [
        target_record(
            "status_next_step_8_7_56_261",
            STATUS,
            status_text,
            "current official next step は `8.7.56.261`",
            "STATUS must already point to the base-modes-by-ell function-boundary branch.",
        ),
        target_record(
            "roadmap_branch_8_7_56_261_264",
            ROADMAP,
            roadmap_text,
            "`8.7.56.261-.264` 試練3 relaunched explicit `k>0` base-modes-by-ell function-boundary residual branch",
            "ROADMAP must already freeze the base-modes-by-ell function-boundary branch.",
        ),
        {
            "file_key": "numerical_function_boundary",
            "file": rel(NUMERICAL_BRANCH),
            "pattern": "def build_spin_orbit_rows(",
            "present": numerical_function_boundary is not None,
            "note": "The numerical function boundary that consumes base_modes_by_ell is present.",
            "evidence": numerical_function_boundary,
        },
        {
            "file_key": "numerical_parameter_contract",
            "file": rel(NUMERICAL_BRANCH),
            "pattern": "base_modes_by_ell: dict[int, list[dict]],",
            "present": numerical_parameter_contract is not None,
            "note": "The numerical function boundary still fixes base_modes_by_ell as dict[int, list[dict]].",
            "evidence": numerical_parameter_contract,
        },
        {
            "file_key": "numerical_callsite_pass_through",
            "file": rel(NUMERICAL_BRANCH),
            "pattern": "split_rows = build_spin_orbit_rows(scalar_modes, base_modes_by_ell, lambda_rot)",
            "present": numerical_callsite is not None,
            "note": "The numerical branch still passes the ell-only container directly across the function boundary.",
            "evidence": numerical_callsite,
        },
        {
            "file_key": "numerical_downstream_count_surface",
            "file": rel(NUMERICAL_BRANCH),
            "pattern": "total_integer_modes = sum(len(modes) for modes in base_modes_by_ell.values())",
            "present": numerical_downstream_count is not None,
            "note": "The numerical downstream count still inherits the ell-only function-boundary contract.",
            "evidence": numerical_downstream_count,
        },
        {
            "file_key": "full_function_boundary",
            "file": rel(FULL_COUPLED_BRANCH),
            "pattern": "def build_exact_ladder(",
            "present": full_function_boundary is not None,
            "note": "The full-coupled function boundary that consumes base_modes_by_ell is present.",
            "evidence": full_function_boundary,
        },
        {
            "file_key": "full_parameter_contract",
            "file": rel(FULL_COUPLED_BRANCH),
            "pattern": "base_modes_by_ell: dict[int, list[dict]],",
            "present": full_parameter_contract is not None,
            "note": "The full-coupled function boundary still fixes base_modes_by_ell as dict[int, list[dict]].",
            "evidence": full_parameter_contract,
        },
        {
            "file_key": "full_callsite_pass_through",
            "file": rel(FULL_COUPLED_BRANCH),
            "pattern": "exact_rows = build_exact_ladder(scalar_modes, base_modes_by_ell, lambda_rot)",
            "present": full_callsite is not None,
            "note": "The full-coupled branch still passes the ell-only container directly across the function boundary.",
            "evidence": full_callsite,
        },
        {
            "file_key": "full_downstream_count_surface",
            "file": rel(FULL_COUPLED_BRANCH),
            "pattern": "total_integer_modes = sum(len(rows) for rows in base_modes_by_ell.values())",
            "present": full_downstream_count is not None,
            "note": "The full-coupled downstream count still inherits the ell-only function-boundary contract.",
            "evidence": full_downstream_count,
        },
    ]

    inventory = payload(
        "8.7.56.261",
        "Trial-3 relaunched explicit k-positive base-modes-by-ell function-boundary source inventory",
        common_inputs,
        "Freeze the typed function-boundary blocker by inventorying the parameter contracts and callsites that still force base_modes_by_ell to behave as dict[int, list[dict]].",
        {
            "inventory_rule": "collect both numerical and full-coupled base_modes_by_ell function boundaries in one machine-readable pack",
            "parameter_contract_rule": "the source pack must say explicitly whether each function boundary still types base_modes_by_ell as dict[int, list[dict]]",
            "boundary_rule": "the source pack must say explicitly whether those typed contracts are passed through current callsites and inherited by downstream counts",
        },
        [
            row(
                "trial3_relaunched_base_modes_by_ell_function_boundary_source_inventory_complete",
                "pass",
                "Trial-3 relaunched base-modes-by-ell function-boundary source inventory complete",
                1,
                "The source inventory for the function-boundary residual branch is frozen.",
            ),
            row(
                "trial3_relaunched_numerical_base_modes_by_ell_function_boundary_surface_present",
                "pass" if numerical_boundary_surface_present else "reject",
                "numerical base-modes-by-ell function-boundary surfaces present",
                1 if numerical_boundary_surface_present else 0,
                "The numerical branch still exposes the boundary, parameter contract, callsite, and downstream count surface.",
            ),
            row(
                "trial3_relaunched_full_base_modes_by_ell_function_boundary_surface_present",
                "pass" if full_boundary_surface_present else "reject",
                "full-coupled base-modes-by-ell function-boundary surfaces present",
                1 if full_boundary_surface_present else 0,
                "The full-coupled branch still exposes the boundary, parameter contract, callsite, and downstream count surface.",
            ),
            row(
                "trial3_relaunched_base_modes_by_ell_parameter_contract_still_dict_int_list_dict",
                "pass",
                "base-modes-by-ell parameter contract still fixes dict[int, list[dict]]",
                1,
                "Both branches still force base_modes_by_ell through a dict[int, list[dict]] function-boundary contract.",
            ),
        ],
        {
            "inventory_ready": inventory_ready,
            "numerical_boundary_surface_present": numerical_boundary_surface_present,
            "full_boundary_surface_present": full_boundary_surface_present,
            "numerical_base_modes_by_ell_parameter_contract_available": numerical_base_modes_by_ell_parameter_contract_available,
            "full_base_modes_by_ell_parameter_contract_available": full_base_modes_by_ell_parameter_contract_available,
            "base_modes_by_ell_parameter_contract_surface_available": base_modes_by_ell_parameter_contract_surface_available,
            "first_route_to_close_or_none": "trial3_relaunched_explicit_k_positive_base_modes_by_ell_function_boundary_identification_audit",
        },
        {
            "overall_status": "trial3_relaunched_base_modes_by_ell_function_boundary_source_inventory_frozen",
            "trial3_branch_closeable": False,
            "advance_to_8_7_56_262": True,
            "next_required_artifacts": [
                "trial3_relaunched_explicit_k_positive_base_modes_by_ell_function_boundary_identification_audit"
            ],
        },
        {
            "inventory_targets": inventory_targets,
            "prior_signature_surface_source_summary": prior_source["summary"],
            "prior_signature_surface_audit_summary": prior_audit["summary"],
            "prior_declaration_summary": prior_declaration["summary"],
            "prior_disposition_summary": prior_disposition["summary"],
        },
    )

    audit = payload(
        "8.7.56.262",
        "Trial-3 relaunched explicit k-positive base-modes-by-ell function-boundary identification audit",
        common_inputs,
        "Audit whether the current solver can already lift the typed base_modes_by_ell function boundaries beyond dict[int, list[dict]] or whether those parameter contracts still propagate ell-only grouping through the downstream code.",
        {
            "numerical_rule": "the numerical branch passes only if its function boundary stops typing base_modes_by_ell as dict[int, list[dict]]",
            "full_rule": "the full-coupled branch passes only if its function boundary stops typing base_modes_by_ell as dict[int, list[dict]]",
            "boundary_rule": "the shared dependency passes only if the typed parameter contracts themselves admit a k-axis-aware surface",
        },
        [
            row(
                "trial3_relaunched_numerical_base_modes_by_ell_parameter_contract_available",
                "pass" if numerical_base_modes_by_ell_parameter_contract_available else "reject",
                "numerical base-modes-by-ell parameter contract available",
                1 if numerical_base_modes_by_ell_parameter_contract_available else 0,
                "The numerical function boundary still fixes base_modes_by_ell as dict[int, list[dict]] and propagates that contract downstream.",
            ),
            row(
                "trial3_relaunched_full_base_modes_by_ell_parameter_contract_available",
                "pass" if full_base_modes_by_ell_parameter_contract_available else "reject",
                "full-coupled base-modes-by-ell parameter contract available",
                1 if full_base_modes_by_ell_parameter_contract_available else 0,
                "The full-coupled function boundary still fixes the same dict[int, list[dict]] contract.",
            ),
            row(
                "trial3_relaunched_base_modes_by_ell_parameter_contract_surface_available",
                "pass" if base_modes_by_ell_parameter_contract_surface_available else "reject",
                "base-modes-by-ell parameter contract surface available",
                1 if base_modes_by_ell_parameter_contract_surface_available else 0,
                "The shared typed parameter contracts still block any k-axis-aware downstream surface.",
            ),
        ],
        {
            "trial3_relaunched_explicit_k_positive_numerical_base_modes_by_ell_parameter_contract_available": numerical_base_modes_by_ell_parameter_contract_available,
            "trial3_relaunched_explicit_k_positive_full_base_modes_by_ell_parameter_contract_available": full_base_modes_by_ell_parameter_contract_available,
            "trial3_relaunched_explicit_k_positive_base_modes_by_ell_parameter_contract_surface_available": base_modes_by_ell_parameter_contract_surface_available,
            "trial3_relaunched_explicit_k_positive_base_modes_by_ell_function_boundary_branch_available": base_modes_by_ell_function_boundary_branch_available,
            "identification_nonclosure_reason_or_none": nonclosure_reason,
            "first_route_to_close_or_none": "trial3_relaunched_declaration_tenth_gate",
        },
        {
            "overall_status": "trial3_relaunched_base_modes_by_ell_function_boundary_identification_audited",
            "trial3_branch_closeable": False,
            "advance_to_8_7_56_263": True,
            "next_required_artifacts": ["trial3_relaunched_declaration_tenth_gate"],
        },
        {
            "prior_signature_surface_audit_summary": prior_audit["summary"],
            "numerical_function_boundary": numerical_function_boundary,
            "numerical_parameter_contract": numerical_parameter_contract,
            "numerical_callsite": numerical_callsite,
            "numerical_downstream_count": numerical_downstream_count,
            "full_function_boundary": full_function_boundary,
            "full_parameter_contract": full_parameter_contract,
            "full_callsite": full_callsite,
            "full_downstream_count": full_downstream_count,
        },
    )

    declaration = payload(
        "8.7.56.263",
        "Trial-3 relaunched declaration tenth gate",
        common_inputs,
        "Freeze whether the base-modes-by-ell function-boundary residual is already closeable or whether the next official route must shrink again to the typed parameter-contract artifact that still fixes ell-only grouping.",
        {
            "gate_rule": "the branch closes only if both numerical and full-coupled function boundaries stop forcing dict[int, list[dict]]",
            "reserve_rule": "Trial-2 paper-side sync remains unlocked reserve work while the scientific Trial-3 residual stays open",
        },
        [
            row(
                "trial3_relaunched_tenth_declaration_gate_complete",
                "pass",
                "Trial-3 relaunched tenth declaration gate complete",
                1,
                "The tenth declaration gate is frozen.",
            ),
            row(
                "trial3_relaunched_base_modes_by_ell_function_boundary_branch_closeable",
                "pass" if base_modes_by_ell_function_boundary_branch_available else "reject",
                "base-modes-by-ell function-boundary branch closeable",
                1 if base_modes_by_ell_function_boundary_branch_available else 0,
                "The branch remains non-closeable while typed function boundaries still hard-code ell-only grouping.",
            ),
            row(
                "trial3_relaunched_base_modes_by_ell_function_boundary_residual_route_required",
                "pass" if not base_modes_by_ell_function_boundary_branch_available else "reject",
                "base-modes-by-ell function-boundary residual route required",
                1 if not base_modes_by_ell_function_boundary_branch_available else 0,
                "A narrower residual route remains necessary after the function-boundary audit.",
            ),
            row(
                "trial3_relaunched_trial2_paper_side_sync_execute_now",
                "reject",
                "execute Trial-2 paper-side sync now",
                0,
                "Trial-2 paper sync stays unlocked reserve work while the Trial-3 solver-side residual remains open.",
            ),
        ],
        {
            "trial3_relaunched_branch_closeable": base_modes_by_ell_function_boundary_branch_available,
            "trial3_relaunched_residual_route_required": not base_modes_by_ell_function_boundary_branch_available,
            "trial2_paper_side_sync_execute_now": False,
            "trial4_deferred": True,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial3_relaunched_tenth_declaration_gate_frozen",
            "trial3_branch_closeable": base_modes_by_ell_function_boundary_branch_available,
            "advance_to_8_7_56_264": True,
            "next_required_artifacts": ["trial3_relaunched_paper_sync_trial4_disposition_tenth_refresh"],
        },
        {
            "source_inventory_summary": inventory["summary"],
            "function_boundary_identification_summary": audit["summary"],
            "prior_declaration_summary": prior_declaration["summary"],
        },
    )

    disposition = payload(
        "8.7.56.264",
        "Trial-2 paper-side sync / Trial-4 disposition tenth refresh",
        common_inputs,
        "Refresh the reserve/deferred ordering after the base-modes-by-ell function-boundary audit and freeze the next residual route for the relaunched weak-sector mainline.",
        {
            "trial2_rule": "retain Trial-2 paper-side sync as unlocked reserve work while the scientific Trial-3 residual is still open",
            "trial4_rule": "keep Trial-4 deferred until the relaunched Trial-3 branch loses all honest current-canon solver routes",
            "selected_residual_route": RESIDUAL_ROUTE,
            "missing_v2_artifact": MISSING_ARTIFACT,
        },
        [
            row(
                "trial3_relaunched_tenth_disposition_gate_complete",
                "pass",
                "Trial-3 relaunched tenth disposition gate complete",
                1,
                "The reserve/deferred ordering after the function-boundary audit is frozen.",
            ),
            row(
                "trial3_relaunched_tenth_trial2_paper_side_sync_reserve_retained",
                "pass",
                "Trial-2 paper-side sync retained as unlocked reserve",
                1,
                "Trial-2 paper-side sync remains available but not promoted ahead of the solver-side residual.",
            ),
            row(
                "trial3_relaunched_tenth_trial4_deferred_retained",
                "pass",
                "Trial-4 deferred disposition retained",
                1,
                "Trial-4 remains deferred while the Trial-3 solver residual remains honest.",
            ),
            row(
                "trial3_relaunched_tenth_next_residual_route_frozen",
                "pass",
                "next residual route frozen",
                1,
                "The next blocker is the missing parameter-contract artifact that would let base_modes_by_ell admit a k axis downstream.",
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
            "overall_status": "trial3_relaunched_tenth_disposition_gate_frozen",
            "trial3_branch_closeable": base_modes_by_ell_function_boundary_branch_available,
            "advance_to_8_7_56_265": True,
            "next_required_artifacts": [
                "trial3_relaunched_explicit_k_positive_base_modes_by_ell_parameter_contract_source_inventory",
                "trial3_relaunched_explicit_k_positive_base_modes_by_ell_parameter_contract_identification_audit",
            ],
        },
        {
            "source_inventory_summary": inventory["summary"],
            "function_boundary_identification_summary": audit["summary"],
            "declaration_summary": declaration["summary"],
            "prior_disposition_summary": prior_disposition["summary"],
            "ai_context_current_step": ai_context["current_step"],
        },
    )

    write_artifact(
        "mass_origin_v2_trial3_relaunched_explicit_k_positive_base_modes_by_ell_function_boundary_source_inventory",
        inventory,
    )
    write_artifact(
        "mass_origin_v2_trial3_relaunched_explicit_k_positive_base_modes_by_ell_function_boundary_identification_audit",
        audit,
    )
    write_artifact("mass_origin_v2_trial3_relaunched_declaration_tenth_gate", declaration)
    write_artifact(
        "mass_origin_v2_trial3_relaunched_paper_sync_trial4_disposition_tenth_refresh",
        disposition,
    )

    print("[ok] wrote:")
    print(" - mass_origin_v2_trial3_relaunched_explicit_k_positive_base_modes_by_ell_function_boundary_source_inventory_metrics.json")
    print(" - mass_origin_v2_trial3_relaunched_explicit_k_positive_base_modes_by_ell_function_boundary_identification_audit_metrics.json")
    print(" - mass_origin_v2_trial3_relaunched_declaration_tenth_gate_metrics.json")
    print(" - mass_origin_v2_trial3_relaunched_paper_sync_trial4_disposition_tenth_refresh_metrics.json")


# Function: Execute the `.261-.264` branch from the CLI.

if __name__ == "__main__":
    main()
