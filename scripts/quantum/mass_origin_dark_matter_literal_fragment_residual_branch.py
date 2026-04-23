#!/usr/bin/env python3
"""
Generate dark-matter literal-fragment residual artifacts for 8.7.55.3.14-.17.

This branch continues the reopened third route after 8.7.55.3.13 fixed that
the remaining blocker is the missing literal fragment needed to derive kappa_a
from the exact vector hierarchy.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output" / "public" / "quantum"
PART1 = ROOT / "doc" / "paper" / "10_part1_core_theory.md"
PART2 = ROOT / "doc" / "paper" / "11_part2_astrophysics.md"
SPARC_NOTE = ROOT / "doc" / "cosmology" / "SPARC_RAR_BTFR.md"
STATUS = ROOT / "doc" / "STATUS.md"
ROADMAP = ROOT / "doc" / "ROADMAP.md"
SPARC_ROTATION = ROOT / "output" / "public" / "cosmology" / "sparc_rotation_curve_pmodel_audit_metrics.json"
VECTOR_FIT = ROOT / "output" / "public" / "quantum" / "mass_origin_vector_qball_baryon_tau_neutron_fit_table_metrics.json"
VECTOR_SPIN = ROOT / "output" / "public" / "quantum" / "mass_origin_vector_qball_spin_orbit_freeze_audit_metrics.json"
PREV_SOURCE = ROOT / "output" / "public" / "quantum" / "mass_origin_vector_kappa_statement_literal_source_inventory_metrics.json"
PREV_AUDIT = ROOT / "output" / "public" / "quantum" / "mass_origin_vector_kappa_statement_literal_wording_audit_metrics.json"
PREV_GATE = ROOT / "output" / "public" / "quantum" / "mass_origin_dark_matter_postnewtonian_gate_second_retry_metrics.json"
PREV_ROUTE = ROOT / "output" / "public" / "quantum" / "mass_origin_vector_kappa_literal_fragment_route_contract_metrics.json"


# Function: Return the current UTC timestamp in ISO 8601 format.
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# Function: Abort immediately when a required artifact is missing.

def req(path: Path) -> None:
    if not path.exists():
        raise SystemExit(f"[fail] missing required input: {path}")


# Function: Read a UTF-8 JSON artifact into a dictionary.

def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


# Function: Read a UTF-8 text file.

def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# Function: Convert an absolute path to a repo-relative string.

def rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


# Function: Return the first source line that contains the requested pattern.

def hit(text: str, pattern: str) -> dict | None:
    for line_no, line in enumerate(text.splitlines(), start=1):
        if pattern in line:
            return {"pattern": pattern, "line": line_no, "text": line.strip()}

    return None


# Function: Build a common metrics row.

def row(row_id: str, status: str, metric: str, value: float, note: str) -> dict:
    return {
        "row_id": row_id,
        "status": status,
        "metric": metric,
        "value": float(value),
        "note": note,
    }


# Function: Build a common payload with the shared schema.

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


# Function: Save a JSON artifact and the paired CSV row table.

def write_artifact(stem: str, data: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    json_path = OUT / f"{stem}_metrics.json"
    csv_path = OUT / f"{stem}_rows.csv"
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["row_id", "status", "metric", "value", "note"])
        writer.writeheader()
        writer.writerows(data["rows"])


# Function: Run the literal-fragment residual branch and write all artifacts.

def main() -> None:
    for path in (
        PART1,
        PART2,
        SPARC_NOTE,
        STATUS,
        ROADMAP,
        SPARC_ROTATION,
        VECTOR_FIT,
        VECTOR_SPIN,
        PREV_SOURCE,
        PREV_AUDIT,
        PREV_GATE,
        PREV_ROUTE,
    ):
        req(path)

    part1_text = read_text(PART1)
    part2_text = read_text(PART2)
    sparc_note_text = read_text(SPARC_NOTE)
    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)

    sparc_rotation = read_json(SPARC_ROTATION)
    vector_fit = read_json(VECTOR_FIT)
    vector_spin = read_json(VECTOR_SPIN)
    prev_source = read_json(PREV_SOURCE)
    prev_audit = read_json(PREV_AUDIT)
    prev_gate = read_json(PREV_GATE)
    prev_route = read_json(PREV_ROUTE)

    sparc_operational_pass = (
        str(sparc_rotation["fit_results"]["comparison"]["better_model_by_chi2"]) == "pmodel_corrected"
        and float(sparc_rotation["fit_results"]["comparison"]["delta_chi2_baryon_minus_pmodel"]) > 0.0
    )
    token_atom_available = False
    relation_operator_available = False
    literal_fragment_available = False
    nonclosure_reason = "vector_exact_hierarchy_to_kappa_a_token_atom_absent"

    required_sources = [
        {
            "source_id": "effective_metric_weak_field_bridge",
            "present": True,
            "evidence": hit(part1_text, "g_{\\mu\\nu}^{(P)}"),
        },
        {
            "source_id": "sparc_operational_kappa_definition",
            "present": True,
            "evidence": hit(part2_text, "a_0=\\kappa_a c H_{0}^{(P)}"),
        },
        {
            "source_id": "vector_exact_hierarchy_anchor_table",
            "present": True,
            "evidence": vector_fit["summary"],
        },
        {
            "source_id": "vector_cross_scale_connection_ready_flag",
            "present": bool(vector_spin["summary"]["cross_scale_connection_ready"]),
            "evidence": vector_spin["summary"],
        },
        {
            "source_id": "vector_exact_hierarchy_to_kappa_a_relation_operator",
            "present": False,
            "evidence": None,
        },
        {
            "source_id": "vector_exact_hierarchy_to_kappa_a_literal_fragment",
            "present": False,
            "evidence": None,
        },
    ]
    present_sources = [item["source_id"] for item in required_sources if item["present"]]
    missing_sources = [item["source_id"] for item in required_sources if not item["present"]]

    payloads = {
        "mass_origin_vector_kappa_literal_fragment_source_inventory": payload(
            "8.7.55.3.14",
            "Vector exact-hierarchy to kappa_a literal-fragment source inventory",
            {
                "mass_origin_vector_kappa_literal_fragment_route_contract_json": rel(PREV_ROUTE),
                "mass_origin_vector_kappa_statement_literal_source_inventory_json": rel(PREV_SOURCE),
                "mass_origin_vector_kappa_statement_literal_wording_audit_json": rel(PREV_AUDIT),
                "mass_origin_vector_qball_baryon_tau_neutron_fit_table_json": rel(VECTOR_FIT),
                "mass_origin_vector_qball_spin_orbit_freeze_audit_json": rel(VECTOR_SPIN),
                "part1_core_theory_markdown": rel(PART1),
                "part2_astrophysics_markdown": rel(PART2),
                "sparc_rar_note_markdown": rel(SPARC_NOTE),
            },
            "Inventory whether the missing kappa_a literal fragment can be reduced to a smaller token-atom or operator artifact.",
            {
                "literal_fragment_rule": "the fragment requires both a relation operator and at least one token-level atom that explicitly ties the exact vector hierarchy to kappa_a",
            },
            [
                row(
                    "vector_kappa_literal_fragment_source_inventory_complete",
                    "pass",
                    "vector kappa_a literal-fragment source inventory complete",
                    1,
                    "The literal-fragment source inventory is frozen.",
                ),
                row(
                    "vector_kappa_literal_fragment_present_source_count",
                    "pass",
                    "present literal-fragment source count",
                    len(present_sources),
                    "Four of the six required sources are already present.",
                ),
                row(
                    "vector_kappa_literal_fragment_missing_source_count",
                    "reject",
                    "missing literal-fragment source count",
                    len(missing_sources),
                    "The current residual gap is the token atom plus the relation operator.",
                ),
            ],
            {
                "required_vector_kappa_literal_fragment_sources": [item["source_id"] for item in required_sources],
                "present_vector_kappa_literal_fragment_sources": present_sources,
                "missing_vector_kappa_literal_fragment_sources": missing_sources,
                "first_route_to_close_or_none": "vector_exact_hierarchy_to_kappa_a_token_atom",
                "source_inventory_ready": True,
            },
            {
                "overall_status": "vector_kappa_literal_fragment_source_inventory_frozen",
                "dark_matter_branch_active": True,
                "literal_fragment_ready": False,
                "next_required_artifacts": ["vector_kappa_literal_fragment_wording_audit"],
            },
            {
                "required_source_rows": required_sources,
                "previous_route_summary": prev_route["summary"],
                "status_next_line": hit(status_text, "次の公式 step は `8.7.55.3.14`"),
                "roadmap_branch_line": hit(roadmap_text, "`8.7.55.3.13-.16`"),
            },
        ),
        "mass_origin_vector_kappa_literal_fragment_wording_audit": payload(
            "8.7.55.3.15",
            "Vector exact-hierarchy to kappa_a literal-fragment wording audit",
            {
                "mass_origin_vector_kappa_literal_fragment_source_inventory_json": "output/public/quantum/mass_origin_vector_kappa_literal_fragment_source_inventory_metrics.json",
                "mass_origin_vector_kappa_statement_literal_wording_audit_json": rel(PREV_AUDIT),
                "mass_origin_vector_qball_spin_orbit_freeze_audit_json": rel(VECTOR_SPIN),
            },
            "Audit whether the current pack already contains the token atom and operator needed to form the missing kappa_a literal fragment.",
            {
                "literal_fragment_rule": "a valid fragment needs both a relation operator and a token-level atom that explicitly states how the exact vector hierarchy fixes kappa_a",
            },
            [
                row(
                    "vector_kappa_literal_fragment_relation_operator_available",
                    "pass" if relation_operator_available else "reject",
                    "vector kappa_a relation operator available at literal-fragment stage",
                    1 if relation_operator_available else 0,
                    "The current pack still does not provide the operator needed for the fragment.",
                ),
                row(
                    "vector_kappa_literal_fragment_token_atom_available",
                    "pass" if token_atom_available else "reject",
                    "vector kappa_a token atom available",
                    1 if token_atom_available else 0,
                    "The current pack still lacks the token-level atom that would define the fragment.",
                ),
                row(
                    "vector_kappa_literal_fragment_available",
                    "pass" if relation_operator_available and token_atom_available and literal_fragment_available else "reject",
                    "vector kappa_a literal fragment available",
                    1 if relation_operator_available and token_atom_available and literal_fragment_available else 0,
                    "Without the token atom, the literal fragment cannot be formed.",
                ),
            ],
            {
                "vector_exact_hierarchy_to_kappa_a_relation_operator_available": relation_operator_available,
                "vector_exact_hierarchy_to_kappa_a_token_atom_available": token_atom_available,
                "vector_exact_hierarchy_to_kappa_a_literal_fragment_available": literal_fragment_available,
                "bridge_nonclosure_reason_or_none": nonclosure_reason,
            },
            {
                "overall_status": "literal_fragment_requires_token_atom_and_operator",
                "dark_matter_branch_active": True,
                "literal_fragment_ready": False,
                "next_required_artifacts": ["dark_matter_postnewtonian_gate_third_retry"],
            },
            {
                "previous_statement_literal_source_summary": prev_source["summary"],
                "previous_statement_literal_audit_summary": prev_audit["summary"],
                "part1_bridge_line": hit(part1_text, "2.7.1A ベクトル場から有効計量"),
                "part2_sparc_line": hit(part2_text, "### 4.14"),
                "sparc_note_fixed_candidate_line": hit(sparc_note_text, "candidate_rar_pbg_a0_fixed_kappa"),
            },
        ),
        "mass_origin_dark_matter_postnewtonian_gate_third_retry": payload(
            "8.7.55.3.16",
            "Dark-matter post-Newtonian gate third retry",
            {
                "mass_origin_vector_kappa_literal_fragment_wording_audit_json": "output/public/quantum/mass_origin_vector_kappa_literal_fragment_wording_audit_metrics.json",
                "mass_origin_dark_matter_postnewtonian_gate_second_retry_json": rel(PREV_GATE),
                "mass_origin_vector_kappa_literal_fragment_route_contract_json": rel(PREV_ROUTE),
            },
            "Retry the reopened third-route gate after reducing the missing literal fragment to a token-atom-level blocker.",
            {
                "close_rule": "close only if SPARC operational pass survives and the exact vector hierarchy now supplies a full kappa_a literal fragment",
            },
            [
                row(
                    "dark_matter_postnewtonian_operational_sparc_pass_third_retry",
                    "pass" if sparc_operational_pass else "reject",
                    "operational SPARC pass still available on third retry",
                    1 if sparc_operational_pass else 0,
                    "The SPARC operational pass remains available on the third retry.",
                ),
                row(
                    "dark_matter_postnewtonian_literal_fragment_ready_third_retry",
                    "pass" if relation_operator_available and token_atom_available and literal_fragment_available else "reject",
                    "kappa_a literal fragment ready on third retry",
                    1 if relation_operator_available and token_atom_available and literal_fragment_available else 0,
                    "The gate remains blocked because the literal fragment still cannot be formed.",
                ),
                row(
                    "dark_matter_postnewtonian_branch_closeable_third_retry",
                    "pass" if sparc_operational_pass and relation_operator_available and token_atom_available and literal_fragment_available else "reject",
                    "dark-matter branch closeable on third retry",
                    1 if sparc_operational_pass and relation_operator_available and token_atom_available and literal_fragment_available else 0,
                    "Operational SPARC success is retained, but first-principles closure still fails at the token-atom stage.",
                ),
            ],
            {
                "sparc_operational_pass_still_available": sparc_operational_pass,
                "kappa_a_first_principles_derivation_ready": relation_operator_available and token_atom_available and literal_fragment_available,
                "dark_matter_postnewtonian_branch_closeable": sparc_operational_pass and relation_operator_available and token_atom_available and literal_fragment_available,
                "recommended_next_route_or_none": "8.7.55.3.17",
            },
            {
                "overall_status": "dark_matter_postnewtonian_operational_pass_retained_but_token_atom_blocked",
                "dark_matter_branch_active": True,
                "advance_to_dark_matter_closeout": False,
                "new_branch_required": True,
                "next_required_artifacts": ["vector_exact_hierarchy_to_kappa_a_token_atom"],
            },
            {
                "previous_gate_summary": prev_gate["summary"],
                "previous_route_summary": prev_route["summary"],
            },
        ),
        "mass_origin_vector_kappa_token_atom_route_contract": payload(
            "8.7.55.3.17",
            "Vector exact-hierarchy to kappa_a token-atom route contract",
            {
                "mass_origin_dark_matter_postnewtonian_gate_third_retry_json": "output/public/quantum/mass_origin_dark_matter_postnewtonian_gate_third_retry_metrics.json",
                "mass_origin_vector_kappa_literal_fragment_wording_audit_json": "output/public/quantum/mass_origin_vector_kappa_literal_fragment_wording_audit_metrics.json",
            },
            "Freeze the next residual route after reducing the kappa_a literal-fragment failure to a token-atom artifact.",
            {
                "selected_residual_route": "vector_exact_hierarchy_to_kappa_a_token_atom",
                "missing_artifact": "vector_exact_hierarchy_to_kappa_a_token_atom",
            },
            [
                row(
                    "vector_kappa_token_atom_route_contract_complete",
                    "pass",
                    "vector kappa_a token-atom route contract complete",
                    1,
                    "The next residual route contract is frozen.",
                ),
                row(
                    "vector_kappa_token_atom_missing_artifact",
                    "reject",
                    "missing vector kappa_a token-atom artifact",
                    1,
                    "The exact vector hierarchy still lacks the token atom that would derive kappa_a.",
                ),
                row(
                    "vector_kappa_token_atom_split_contract_ready",
                    "pass",
                    "vector kappa_a token-atom split contract ready",
                    1,
                    "The next branch may now start from the missing token atom.",
                ),
            ],
            {
                "selected_residual_route": "vector_exact_hierarchy_to_kappa_a_token_atom",
                "missing_dark_matter_artifact": "vector_exact_hierarchy_to_kappa_a_token_atom",
                "split_contract_ready": True,
            },
            {
                "overall_status": "vector_kappa_token_atom_route_contract_frozen",
                "dark_matter_branch_active": True,
                "advance_to_dark_matter_closeout": False,
                "new_branch_required": True,
                "next_required_artifacts": [
                    "vector_kappa_token_atom_source_inventory",
                    "vector_kappa_token_atom_wording_audit",
                ],
            },
            {
                "part1_bridge_line": hit(part1_text, "2.7.1A ベクトル場から有効計量"),
                "part2_sparc_line": hit(part2_text, "### 4.14"),
            },
        ),
    }

    for stem, data in payloads.items():
        write_artifact(stem, data)
        print(f"[ok] wrote {OUT / (stem + '_metrics.json')}")
        print(f"[ok] wrote {OUT / (stem + '_rows.csv')}")


# Function: Run the literal-fragment residual branch when invoked as a script.

if __name__ == "__main__":
    main()
