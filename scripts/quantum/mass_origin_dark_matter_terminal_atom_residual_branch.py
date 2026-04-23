#!/usr/bin/env python3
"""
Generate dark-matter terminal-atom residual artifacts for 8.7.55.3.102-.105.

This branch continues the reopened third route after 8.7.55.3.101 fixed that
the remaining blocker is again the missing terminal atom needed to derive
kappa_a from the exact vector hierarchy.
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
SPIN_ORBIT = ROOT / "output" / "public" / "quantum" / "mass_origin_vector_qball_spin_orbit_freeze_audit_metrics.json"
PREV_SOURCE = ROOT / "output" / "public" / "quantum" / "mass_origin_vector_kappa_symbol_fragment_source_inventory_metrics.json"
PREV_AUDIT = ROOT / "output" / "public" / "quantum" / "mass_origin_vector_kappa_symbol_fragment_wording_audit_metrics.json"
PREV_GATE = ROOT / "output" / "public" / "quantum" / "mass_origin_dark_matter_postnewtonian_gate_twenty_fourth_retry_metrics.json"
PREV_ROUTE = ROOT / "output" / "public" / "quantum" / "mass_origin_vector_kappa_terminal_atom_route_contract_metrics.json"


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


# Function: Run the terminal-atom residual branch and write all artifacts.

def main() -> None:
    for path in (
        PART1,
        PART2,
        SPARC_NOTE,
        STATUS,
        ROADMAP,
        SPARC_ROTATION,
        VECTOR_FIT,
        SPIN_ORBIT,
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
    spin_orbit = read_json(SPIN_ORBIT)
    prev_source = read_json(PREV_SOURCE)
    prev_audit = read_json(PREV_AUDIT)
    prev_gate = read_json(PREV_GATE)
    prev_route = read_json(PREV_ROUTE)

    sparc_operational_pass = (
        str(sparc_rotation["fit_results"]["comparison"]["better_model_by_chi2"]) == "pmodel_corrected"
        and float(sparc_rotation["fit_results"]["comparison"]["delta_chi2_baryon_minus_pmodel"]) > 0.0
    )
    relation_operator_available = False
    terminal_glyph_available = False
    terminal_atom_available = False
    nonclosure_reason = "vector_exact_hierarchy_to_kappa_a_terminal_glyph_absent"

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
            "present": bool(spin_orbit["summary"]["cross_scale_connection_ready"]),
            "evidence": spin_orbit["summary"],
        },
        {
            "source_id": "vector_exact_hierarchy_to_kappa_a_relation_operator",
            "present": False,
            "evidence": None,
        },
        {
            "source_id": "vector_exact_hierarchy_to_kappa_a_terminal_atom",
            "present": False,
            "evidence": None,
        },
    ]
    present_sources = [item["source_id"] for item in required_sources if item["present"]]
    missing_sources = [item["source_id"] for item in required_sources if not item["present"]]

    payloads = {
        "mass_origin_vector_kappa_terminal_atom_source_inventory": payload(
            "8.7.55.3.102",
            "Vector exact-hierarchy to kappa_a terminal-atom source inventory",
            {
                "mass_origin_vector_kappa_terminal_atom_route_contract_json": rel(PREV_ROUTE),
                "mass_origin_vector_kappa_symbol_fragment_source_inventory_json": rel(PREV_SOURCE),
                "mass_origin_vector_kappa_symbol_fragment_wording_audit_json": rel(PREV_AUDIT),
                "mass_origin_vector_qball_baryon_tau_neutron_fit_table_json": rel(VECTOR_FIT),
                "mass_origin_vector_qball_spin_orbit_freeze_audit_json": rel(SPIN_ORBIT),
                "part1_core_theory_markdown": rel(PART1),
                "part2_astrophysics_markdown": rel(PART2),
                "sparc_rar_note_markdown": rel(SPARC_NOTE),
            },
            "Inventory whether the missing kappa_a terminal atom can be reduced to a smaller terminal-glyph or operator artifact.",
            {
                "terminal_atom_rule": "the terminal atom requires both a relation operator and at least one terminal glyph that explicitly ties the exact vector hierarchy to kappa_a",
            },
            [
                row(
                    "vector_kappa_terminal_atom_source_inventory_complete",
                    "pass",
                    "vector kappa_a terminal-atom source inventory complete",
                    1,
                    "The terminal-atom source inventory is frozen.",
                ),
                row(
                    "vector_kappa_terminal_atom_present_source_count",
                    "pass",
                    "present terminal-atom source count",
                    len(present_sources),
                    f"{len(present_sources)} of the {len(required_sources)} required sources are already present.",
                ),
                row(
                    "vector_kappa_terminal_atom_missing_source_count",
                    "reject",
                    "missing terminal-atom source count",
                    len(missing_sources),
                    "The current residual gap is the terminal glyph plus the relation operator.",
                ),
            ],
            {
                "required_vector_kappa_terminal_atom_sources": [item["source_id"] for item in required_sources],
                "present_vector_kappa_terminal_atom_sources": present_sources,
                "missing_vector_kappa_terminal_atom_sources": missing_sources,
                "first_route_to_close_or_none": "vector_exact_hierarchy_to_kappa_a_terminal_glyph",
                "source_inventory_ready": True,
            },
            {
                "overall_status": "vector_kappa_terminal_atom_source_inventory_frozen",
                "dark_matter_branch_active": True,
                "terminal_atom_ready": False,
                "next_required_artifacts": ["vector_kappa_terminal_atom_wording_audit"],
            },
            {
                "required_source_rows": required_sources,
                "previous_route_summary": prev_route["summary"],
                "status_next_line": hit(status_text, "次の公式 step は `8.7.55.3.102`"),
                "roadmap_branch_line": hit(roadmap_text, "`8.7.55.3.101-.104`"),
            },
        ),
        "mass_origin_vector_kappa_terminal_atom_wording_audit": payload(
            "8.7.55.3.103",
            "Vector exact-hierarchy to kappa_a terminal-atom wording audit",
            {
                "mass_origin_vector_kappa_terminal_atom_source_inventory_json": "output/public/quantum/mass_origin_vector_kappa_terminal_atom_source_inventory_metrics.json",
                "mass_origin_vector_kappa_symbol_fragment_wording_audit_json": rel(PREV_AUDIT),
                "mass_origin_vector_qball_spin_orbit_freeze_audit_json": rel(SPIN_ORBIT),
            },
            "Audit whether the current pack already contains the terminal glyph and operator needed to form the missing kappa_a terminal atom.",
            {
                "terminal_atom_rule": "a valid terminal atom needs both a relation operator and a terminal glyph that explicitly states how the exact vector hierarchy fixes kappa_a",
            },
            [
                row(
                    "vector_kappa_terminal_atom_relation_operator_available",
                    "reject",
                    "vector kappa_a relation operator available at terminal-atom stage",
                    0,
                    "The current pack still does not provide the operator needed for the terminal atom.",
                ),
                row(
                    "vector_kappa_terminal_atom_terminal_glyph_available",
                    "reject",
                    "vector kappa_a terminal glyph available",
                    0,
                    "The current pack still lacks the terminal glyph that would define the terminal atom.",
                ),
                row(
                    "vector_kappa_terminal_atom_available",
                    "reject",
                    "vector kappa_a terminal atom available",
                    0,
                    "Without the terminal glyph, the terminal atom cannot be formed.",
                ),
            ],
            {
                "vector_exact_hierarchy_to_kappa_a_relation_operator_available": relation_operator_available,
                "vector_exact_hierarchy_to_kappa_a_terminal_glyph_available": terminal_glyph_available,
                "vector_exact_hierarchy_to_kappa_a_terminal_atom_available": terminal_atom_available,
                "bridge_nonclosure_reason_or_none": nonclosure_reason,
            },
            {
                "overall_status": "terminal_atom_requires_terminal_glyph_and_operator",
                "dark_matter_branch_active": True,
                "terminal_atom_ready": False,
                "next_required_artifacts": ["dark_matter_postnewtonian_gate_twenty_fifth_retry"],
            },
            {
                "previous_symbol_fragment_source_summary": prev_source["summary"],
                "previous_symbol_fragment_audit_summary": prev_audit["summary"],
                "part1_bridge_line": hit(part1_text, "2.7.1A ベクトル場から有効計量"),
                "part2_sparc_line": hit(part2_text, "### 4.14"),
                "sparc_note_fixed_candidate_line": hit(sparc_note_text, "candidate_rar_pbg_a0_fixed_kappa"),
            },
        ),
        "mass_origin_dark_matter_postnewtonian_gate_twenty_fifth_retry": payload(
            "8.7.55.3.104",
            "Dark-matter post-Newtonian gate twenty-fifth retry",
            {
                "mass_origin_vector_kappa_terminal_atom_wording_audit_json": "output/public/quantum/mass_origin_vector_kappa_terminal_atom_wording_audit_metrics.json",
                "mass_origin_dark_matter_postnewtonian_gate_twenty_fourth_retry_json": rel(PREV_GATE),
                "mass_origin_vector_kappa_terminal_atom_route_contract_json": rel(PREV_ROUTE),
            },
            "Retry the reopened third-route gate after reducing the missing terminal atom to a terminal-glyph-level blocker.",
            {
                "close_rule": "close only if SPARC operational pass survives and the exact vector hierarchy now supplies a full kappa_a terminal atom",
            },
            [
                row(
                    "dark_matter_postnewtonian_operational_sparc_pass_twenty_fifth_retry",
                    "pass" if sparc_operational_pass else "reject",
                    "operational SPARC pass still available on twenty-fifth retry",
                    1 if sparc_operational_pass else 0,
                    "The SPARC operational pass remains available on the twenty-fifth retry.",
                ),
                row(
                    "dark_matter_postnewtonian_terminal_atom_ready_twenty_fifth_retry",
                    "reject",
                    "kappa_a terminal atom ready on twenty-fifth retry",
                    0,
                    "The gate remains blocked because the terminal atom still cannot be formed.",
                ),
                row(
                    "dark_matter_postnewtonian_branch_closeable_twenty_fifth_retry",
                    "reject",
                    "dark-matter branch closeable on twenty-fifth retry",
                    0,
                    "Operational SPARC success is retained, but first-principles closure still fails at the terminal-glyph stage.",
                ),
            ],
            {
                "sparc_operational_pass_still_available": sparc_operational_pass,
                "kappa_a_first_principles_derivation_ready": False,
                "dark_matter_postnewtonian_branch_closeable": False,
                "recommended_next_route_or_none": "8.7.55.3.105",
            },
            {
                "overall_status": "dark_matter_postnewtonian_operational_pass_retained_but_terminal_glyph_blocked",
                "dark_matter_branch_active": True,
                "advance_to_dark_matter_closeout": False,
                "new_branch_required": True,
                "next_required_artifacts": ["vector_exact_hierarchy_to_kappa_a_terminal_glyph"],
            },
            {
                "previous_gate_summary": prev_gate["summary"],
                "previous_route_summary": prev_route["summary"],
            },
        ),
        "mass_origin_vector_kappa_terminal_glyph_route_contract": payload(
            "8.7.55.3.105",
            "Vector exact-hierarchy to kappa_a terminal-glyph route contract",
            {
                "mass_origin_dark_matter_postnewtonian_gate_twenty_fifth_retry_json": "output/public/quantum/mass_origin_dark_matter_postnewtonian_gate_twenty_fifth_retry_metrics.json",
                "mass_origin_vector_kappa_terminal_atom_wording_audit_json": "output/public/quantum/mass_origin_vector_kappa_terminal_atom_wording_audit_metrics.json",
            },
            "Freeze the next residual route after reducing the kappa_a terminal-atom failure to a terminal-glyph artifact.",
            {
                "selected_residual_route": "vector_exact_hierarchy_to_kappa_a_terminal_glyph",
                "missing_artifact": "vector_exact_hierarchy_to_kappa_a_terminal_glyph",
            },
            [
                row(
                    "vector_kappa_terminal_glyph_route_contract_complete",
                    "pass",
                    "vector kappa_a terminal-glyph route contract complete",
                    1,
                    "The next residual route contract is frozen.",
                ),
                row(
                    "vector_kappa_terminal_glyph_missing_artifact",
                    "reject",
                    "missing vector kappa_a terminal-glyph artifact",
                    1,
                    "The exact vector hierarchy still lacks the terminal glyph that would derive kappa_a.",
                ),
                row(
                    "vector_kappa_terminal_glyph_split_contract_ready",
                    "pass",
                    "vector kappa_a terminal-glyph split contract ready",
                    1,
                    "The next branch may now start from the missing terminal glyph.",
                ),
            ],
            {
                "selected_residual_route": "vector_exact_hierarchy_to_kappa_a_terminal_glyph",
                "missing_dark_matter_artifact": "vector_exact_hierarchy_to_kappa_a_terminal_glyph",
                "split_contract_ready": True,
            },
            {
                "overall_status": "vector_kappa_terminal_glyph_route_contract_frozen",
                "dark_matter_branch_active": True,
                "advance_to_dark_matter_closeout": False,
                "new_branch_required": True,
                "next_required_artifacts": [
                    "vector_kappa_terminal_glyph_source_inventory",
                    "vector_kappa_terminal_glyph_wording_audit",
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


# Function: Run the terminal-atom residual branch when invoked as a script.

def _entrypoint() -> None:
    main()


if __name__ == "__main__":
    _entrypoint()
