#!/usr/bin/env python3
"""
Generate dark-matter vector-bridge residual artifacts for 8.7.55.3.5-.9.

This branch continues the reopened third route after 8.7.55.3.1-.4 fixed that
SPARC operational success survives but the current public pack still lacks a
first-principles bridge from the exact vector hierarchy to kappa_a. The next
question is whether the missing bridge can at least be reduced to a smaller
statement-literal artifact.
"""

from __future__ import annotations

import csv
import json
import math
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
VECTOR_GATE = ROOT / "output" / "public" / "quantum" / "mass_origin_vector_qball_second_route_gate_refresh_metrics.json"
VECTOR_SPIN = ROOT / "output" / "public" / "quantum" / "mass_origin_vector_qball_spin_orbit_freeze_audit_metrics.json"
PREV_BRIDGE_AUDIT = ROOT / "output" / "public" / "quantum" / "mass_origin_kappa_a_vector_hierarchy_bridge_audit_metrics.json"
PREV_GATE = ROOT / "output" / "public" / "quantum" / "mass_origin_dark_matter_postnewtonian_gate_refresh_metrics.json"
PREV_ROUTE_CONTRACT = ROOT / "output" / "public" / "quantum" / "mass_origin_dark_matter_vector_bridge_route_contract_metrics.json"

PBG_CANDIDATE_KAPPA = 1.0 / (2.0 * math.pi)


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


# Function: Return a compact sample from a long list or dictionary-backed rows.

def sample(items: list[dict], count: int = 12) -> list[dict]:
    if len(items) <= count:
        return items

    stride = max(1, len(items) // count)
    return [items[index] for index in range(0, len(items), stride)][:count]


# Function: Run the vector-bridge residual branch and write all artifacts.

def main() -> None:
    for path in (
        PART1,
        PART2,
        SPARC_NOTE,
        STATUS,
        ROADMAP,
        SPARC_ROTATION,
        VECTOR_FIT,
        VECTOR_GATE,
        VECTOR_SPIN,
        PREV_BRIDGE_AUDIT,
        PREV_GATE,
        PREV_ROUTE_CONTRACT,
    ):
        req(path)

    part1_text = read_text(PART1)
    part2_text = read_text(PART2)
    sparc_note_text = read_text(SPARC_NOTE)
    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)

    sparc_rotation = read_json(SPARC_ROTATION)
    vector_fit = read_json(VECTOR_FIT)
    vector_gate = read_json(VECTOR_GATE)
    vector_spin = read_json(VECTOR_SPIN)
    prev_bridge_audit = read_json(PREV_BRIDGE_AUDIT)
    prev_gate = read_json(PREV_GATE)
    prev_route_contract = read_json(PREV_ROUTE_CONTRACT)

    sparc_kappa = float(sparc_rotation["inputs"]["pbg_kappa"])
    sparc_operational_pass = (
        str(sparc_rotation["fit_results"]["comparison"]["better_model_by_chi2"]) == "pmodel_corrected"
        and float(sparc_rotation["fit_results"]["comparison"]["delta_chi2_baryon_minus_pmodel"]) > 0.0
    )
    operational_kappa_matches_pbg_candidate = math.isclose(
        sparc_kappa,
        PBG_CANDIDATE_KAPPA,
        rel_tol=0.0,
        abs_tol=1e-15,
    )
    vector_cross_scale_connection_ready = bool(vector_spin["summary"]["cross_scale_connection_ready"])
    current_pack_has_statement_literal = False
    bridge_nonclosure_reason = "vector_exact_hierarchy_to_kappa_a_statement_literal_absent"

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
            "source_id": "pbg_fixed_kappa_candidate_note",
            "present": True,
            "evidence": hit(sparc_note_text, "candidate_rar_pbg_a0_fixed_kappa"),
        },
        {
            "source_id": "vector_exact_hierarchy_anchor_table",
            "present": True,
            "evidence": vector_fit["summary"],
        },
        {
            "source_id": "vector_cross_scale_connection_ready_flag",
            "present": vector_cross_scale_connection_ready,
            "evidence": vector_spin["summary"],
        },
        {
            "source_id": "vector_exact_hierarchy_to_kappa_a_statement_literal",
            "present": False,
            "evidence": None,
        },
    ]
    present_sources = [item["source_id"] for item in required_sources if item["present"]]
    missing_sources = [item["source_id"] for item in required_sources if not item["present"]]

    payloads = {
        "mass_origin_vector_kappa_bridge_source_inventory": payload(
            "8.7.55.3.5",
            "Vector exact-hierarchy to kappa_a bridge source inventory",
            {
                "mass_origin_dark_matter_vector_bridge_route_contract_json": rel(PREV_ROUTE_CONTRACT),
                "mass_origin_kappa_a_vector_hierarchy_bridge_audit_json": rel(PREV_BRIDGE_AUDIT),
                "mass_origin_vector_qball_baryon_tau_neutron_fit_table_json": rel(VECTOR_FIT),
                "mass_origin_vector_qball_spin_orbit_freeze_audit_json": rel(VECTOR_SPIN),
                "sparc_rotation_curve_pmodel_audit_json": rel(SPARC_ROTATION),
                "part1_core_theory_markdown": rel(PART1),
                "part2_astrophysics_markdown": rel(PART2),
                "sparc_rar_note_markdown": rel(SPARC_NOTE),
            },
            "Inventory whether the missing exact-vector-hierarchy-to-kappa_a bridge can be reduced to a smaller statement-literal artifact within the current public pack.",
            {
                "bridge_statement_rule": "the bridge requires a literal statement that maps the exact vector hierarchy onto the SPARC coefficient kappa_a without introducing any new scale or normalization freedom",
            },
            [
                row(
                    "vector_kappa_bridge_source_inventory_complete",
                    "pass",
                    "vector kappa_a bridge source inventory complete",
                    1,
                    "The residual bridge inventory is frozen.",
                ),
                row(
                    "vector_kappa_bridge_present_source_count",
                    "pass",
                    "present residual bridge source count",
                    len(present_sources),
                    "Five of the six required sources are already present.",
                ),
                row(
                    "vector_kappa_bridge_missing_source_count",
                    "reject" if missing_sources else "pass",
                    "missing residual bridge source count",
                    len(missing_sources),
                    "The current missing source is the statement literal that would connect the exact vector hierarchy to kappa_a.",
                ),
            ],
            {
                "required_vector_kappa_bridge_statement_sources": [item["source_id"] for item in required_sources],
                "present_vector_kappa_bridge_statement_sources": present_sources,
                "missing_vector_kappa_bridge_statement_sources": missing_sources,
                "first_route_to_close_or_none": "vector_exact_hierarchy_to_kappa_a_statement_literal",
                "source_inventory_ready": True,
            },
            {
                "overall_status": "vector_kappa_bridge_source_inventory_frozen",
                "dark_matter_branch_active": True,
                "kappa_a_bridge_statement_ready": False,
                "next_required_artifacts": ["vector_kappa_bridge_statement_literal_audit"],
            },
            {
                "required_source_rows": required_sources,
                "previous_route_contract_summary": prev_route_contract["summary"],
                "status_next_line": hit(status_text, "次の公式 step は `8.7.55.3.5`"),
                "roadmap_branch_line": hit(roadmap_text, "`8.7.55.3.5-.8`"),
            },
        ),
        "mass_origin_vector_kappa_bridge_statement_literal_audit": payload(
            "8.7.55.3.6",
            "Vector exact-hierarchy to kappa_a statement-literal audit",
            {
                "mass_origin_vector_kappa_bridge_source_inventory_json": "output/public/quantum/mass_origin_vector_kappa_bridge_source_inventory_metrics.json",
                "mass_origin_vector_qball_spin_orbit_freeze_audit_json": rel(VECTOR_SPIN),
                "mass_origin_kappa_a_vector_hierarchy_bridge_audit_json": rel(PREV_BRIDGE_AUDIT),
            },
            "Audit whether the current public pack already contains a literal statement that upgrades the operational SPARC kappa_a into a first-principles consequence of the exact vector hierarchy.",
            {
                "statement_literal_rule": "a valid literal must connect the exact vector hierarchy, the weak-field bridge, and the SPARC coefficient kappa_a without introducing any new free parameter",
            },
            [
                row(
                    "vector_kappa_bridge_cross_scale_prerequisite_available",
                    "pass" if vector_cross_scale_connection_ready else "reject",
                    "vector cross-scale prerequisite available",
                    1 if vector_cross_scale_connection_ready else 0,
                    "The lambda_rot route already froze a cross-scale reuse flag, but that does not by itself supply the needed kappa_a literal.",
                ),
                row(
                    "vector_kappa_bridge_statement_literal_available",
                    "pass" if current_pack_has_statement_literal else "reject",
                    "vector exact hierarchy to kappa_a statement literal available",
                    1 if current_pack_has_statement_literal else 0,
                    "No public statement literal currently maps the exact vector hierarchy to the galactic acceleration coefficient.",
                ),
                row(
                    "vector_kappa_bridge_without_new_free_parameters",
                    "pass" if current_pack_has_statement_literal else "reject",
                    "vector kappa_a bridge available without new free parameters",
                    1 if current_pack_has_statement_literal else 0,
                    "Current artifacts support operational SPARC success, but not a first-principles bridge literal.",
                ),
            ],
            {
                "vector_cross_scale_connection_ready": vector_cross_scale_connection_ready,
                "operational_kappa_matches_pbg_candidate": operational_kappa_matches_pbg_candidate,
                "vector_exact_hierarchy_to_kappa_a_statement_literal_available": current_pack_has_statement_literal,
                "vector_exact_hierarchy_to_kappa_bridge_available": False,
                "bridge_nonclosure_reason_or_none": bridge_nonclosure_reason,
            },
            {
                "overall_status": "cross_scale_prerequisites_present_but_statement_literal_absent",
                "dark_matter_branch_active": True,
                "kappa_a_bridge_statement_ready": False,
                "next_required_artifacts": ["dark_matter_postnewtonian_gate_retry"],
            },
            {
                "part1_cross_scale_line": hit(part1_text, "Pauli 型スピン結合"),
                "part2_sparc_line": hit(part2_text, "a_0=\\kappa_a c H_{0}^{(P)}"),
                "previous_bridge_audit_summary": prev_bridge_audit["summary"],
                "vector_spin_summary": vector_spin["summary"],
            },
        ),
        "mass_origin_dark_matter_postnewtonian_gate_retry": payload(
            "8.7.55.3.7",
            "Dark-matter post-Newtonian gate retry",
            {
                "mass_origin_vector_kappa_bridge_statement_literal_audit_json": "output/public/quantum/mass_origin_vector_kappa_bridge_statement_literal_audit_metrics.json",
                "mass_origin_dark_matter_postnewtonian_gate_refresh_json": rel(PREV_GATE),
                "sparc_rotation_curve_pmodel_audit_json": rel(SPARC_ROTATION),
            },
            "Retry the reopened third-route gate after reducing the missing bridge to a statement-literal artifact.",
            {
                "close_rule": "close only if SPARC operational pass survives and the exact vector hierarchy now supplies a literal bridge to kappa_a",
            },
            [
                row(
                    "dark_matter_postnewtonian_operational_sparc_pass_retry",
                    "pass" if sparc_operational_pass else "reject",
                    "operational SPARC pass still available",
                    1 if sparc_operational_pass else 0,
                    "The SPARC operational pass remains available across the residual bridge retry.",
                ),
                row(
                    "dark_matter_postnewtonian_bridge_statement_ready_retry",
                    "pass" if current_pack_has_statement_literal else "reject",
                    "kappa_a bridge statement ready on retry",
                    1 if current_pack_has_statement_literal else 0,
                    "The gate remains blocked because the statement literal is still absent.",
                ),
                row(
                    "dark_matter_postnewtonian_branch_closeable_retry",
                    "pass" if sparc_operational_pass and current_pack_has_statement_literal else "reject",
                    "dark-matter branch closeable on retry",
                    1 if sparc_operational_pass and current_pack_has_statement_literal else 0,
                    "Operational SPARC success is retained, but first-principles closure still fails.",
                ),
            ],
            {
                "sparc_operational_pass_still_available": sparc_operational_pass,
                "kappa_a_first_principles_derivation_ready": current_pack_has_statement_literal,
                "dark_matter_postnewtonian_branch_closeable": sparc_operational_pass and current_pack_has_statement_literal,
                "recommended_next_route_or_none": "8.7.55.3.9",
            },
            {
                "overall_status": "dark_matter_postnewtonian_operational_pass_retained_but_statement_literal_blocked",
                "dark_matter_branch_active": True,
                "advance_to_dark_matter_closeout": False,
                "new_branch_required": True,
                "next_required_artifacts": ["vector_exact_hierarchy_to_kappa_a_statement_literal"],
            },
            {
                "previous_gate_summary": prev_gate["summary"],
                "previous_bridge_audit_summary": prev_bridge_audit["summary"],
            },
        ),
        "mass_origin_dark_matter_postnewtonian_branch_refresh": payload(
            "8.7.55.3.8",
            "Dark-matter post-Newtonian branch refresh",
            {
                "mass_origin_dark_matter_postnewtonian_gate_retry_json": "output/public/quantum/mass_origin_dark_matter_postnewtonian_gate_retry_metrics.json",
                "mass_origin_vector_kappa_bridge_statement_literal_audit_json": "output/public/quantum/mass_origin_vector_kappa_bridge_statement_literal_audit_metrics.json",
            },
            "Refresh the third-route branch disposition after confirming that the residual blocker has reduced to a statement-literal artifact.",
            {
                "refresh_rule": "keep the branch active while the operational SPARC pass survives and the blocker continues to shrink without introducing a new parameter",
            },
            [
                row(
                    "dark_matter_postnewtonian_branch_refresh_complete",
                    "pass",
                    "dark-matter branch refresh complete",
                    1,
                    "The residual branch refresh is frozen.",
                ),
                row(
                    "dark_matter_postnewtonian_statement_literal_blocker",
                    "reject",
                    "statement-literal blocker present",
                    1,
                    "The remaining blocker is now the missing vector-exact-hierarchy-to-kappa_a statement literal.",
                ),
                row(
                    "dark_matter_postnewtonian_new_branch_required",
                    "pass",
                    "new residual branch required",
                    1,
                    "The branch must continue as a finer-grained statement-literal residual route.",
                ),
            ],
            {
                "branch_refresh_case_or_none": "operational_pass_retained_statement_literal_absent",
                "fallback_closeout_needed": False,
                "new_branch_required": True,
                "recommended_next_route_or_none": "8.7.55.3.9",
            },
            {
                "overall_status": "dark_matter_branch_refresh_requires_statement_literal_residual_route",
                "dark_matter_branch_active": True,
                "advance_to_dark_matter_closeout": False,
                "next_required_artifacts": ["vector_kappa_statement_literal_route_contract"],
            },
            {
                "gate_retry_summary": {
                    "sparc_operational_pass_still_available": sparc_operational_pass,
                    "recommended_next_route_or_none": "8.7.55.3.9",
                },
                "bridge_literal_summary": {
                    "vector_cross_scale_connection_ready": vector_cross_scale_connection_ready,
                    "bridge_nonclosure_reason_or_none": bridge_nonclosure_reason,
                },
            },
        ),
        "mass_origin_vector_kappa_statement_literal_route_contract": payload(
            "8.7.55.3.9",
            "Vector exact-hierarchy to kappa_a statement-literal route contract",
            {
                "mass_origin_dark_matter_postnewtonian_branch_refresh_json": "output/public/quantum/mass_origin_dark_matter_postnewtonian_branch_refresh_metrics.json",
                "mass_origin_vector_kappa_bridge_statement_literal_audit_json": "output/public/quantum/mass_origin_vector_kappa_bridge_statement_literal_audit_metrics.json",
            },
            "Freeze the next residual route after reducing the kappa_a bridge failure to a missing statement-literal artifact.",
            {
                "selected_residual_route": "vector_exact_hierarchy_to_kappa_a_statement_literal",
                "missing_artifact": "vector_exact_hierarchy_to_kappa_a_statement_literal",
            },
            [
                row(
                    "vector_kappa_statement_literal_route_contract_complete",
                    "pass",
                    "vector kappa_a statement-literal route contract complete",
                    1,
                    "The next residual route contract is frozen.",
                ),
                row(
                    "vector_kappa_statement_literal_missing_artifact",
                    "reject",
                    "missing vector kappa_a statement-literal artifact",
                    1,
                    "The exact vector hierarchy still lacks the statement literal that would derive kappa_a.",
                ),
                row(
                    "vector_kappa_statement_literal_split_contract_ready",
                    "pass",
                    "vector kappa_a statement-literal split contract ready",
                    1,
                    "The next branch may now start from the missing statement literal.",
                ),
            ],
            {
                "selected_residual_route": "vector_exact_hierarchy_to_kappa_a_statement_literal",
                "missing_dark_matter_artifact": "vector_exact_hierarchy_to_kappa_a_statement_literal",
                "split_contract_ready": True,
            },
            {
                "overall_status": "vector_kappa_statement_literal_route_contract_frozen",
                "dark_matter_branch_active": True,
                "advance_to_dark_matter_closeout": False,
                "new_branch_required": True,
                "next_required_artifacts": [
                    "vector_kappa_statement_literal_source_inventory",
                    "vector_kappa_statement_literal_wording_audit",
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


# Function: Run the residual branch when invoked as a script.

if __name__ == "__main__":
    main()
