#!/usr/bin/env python3
"""
Generate dark-matter-elimination post-Newtonian artifacts for 8.7.55.3.1-.4.

This branch starts the reopened third route after the vector-Q-ball extended
hierarchy succeeded in 8.7.55.2.836. The question is no longer whether SPARC
has an operational P-model pass; that already exists. The question is whether
the reopened exact vector hierarchy now supplies a no-new-free-parameter bridge
from the mass-origin route to the SPARC coefficient kappa_a in
`a0 = kappa_a c H0^(P)`.
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
SPARC_FREEZE = ROOT / "output" / "public" / "cosmology" / "sparc_rar_freeze_test_metrics.json"
VECTOR_FIT = ROOT / "output" / "public" / "quantum" / "mass_origin_vector_qball_baryon_tau_neutron_fit_table_metrics.json"
VECTOR_GATE = ROOT / "output" / "public" / "quantum" / "mass_origin_vector_qball_second_route_gate_refresh_metrics.json"
VECTOR_SPIN = ROOT / "output" / "public" / "quantum" / "mass_origin_vector_qball_spin_orbit_freeze_audit_metrics.json"

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


# Function: Return a compact sample from a long list.

def sample(rows: list[dict], count: int = 12) -> list[dict]:
    if len(rows) <= count:
        return rows

    step = max(1, len(rows) // count)
    return [rows[index] for index in range(0, len(rows), step)][:count]


# Function: Run the dark-matter post-Newtonian opening branch and write artifacts.

def main() -> None:
    for path in (
        PART1,
        PART2,
        SPARC_NOTE,
        STATUS,
        ROADMAP,
        SPARC_ROTATION,
        SPARC_FREEZE,
        VECTOR_FIT,
        VECTOR_GATE,
        VECTOR_SPIN,
    ):
        req(path)

    part1_text = read_text(PART1)
    part2_text = read_text(PART2)
    sparc_note_text = read_text(SPARC_NOTE)
    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)

    sparc_rotation = read_json(SPARC_ROTATION)
    sparc_freeze = read_json(SPARC_FREEZE)
    vector_fit = read_json(VECTOR_FIT)
    vector_gate = read_json(VECTOR_GATE)
    vector_spin = read_json(VECTOR_SPIN)

    sparc_kappa = float(sparc_rotation["inputs"]["pbg_kappa"])
    sparc_a0 = float(sparc_rotation["pmodel_fixed"]["a0_m_s2"])
    sparc_h0p = float(sparc_rotation["pmodel_fixed"]["h0p_si_s^-1"])
    sparc_formula = str(sparc_rotation["pmodel_fixed"]["formula"])
    chi2_better_model = str(sparc_rotation["fit_results"]["comparison"]["better_model_by_chi2"])
    delta_chi2 = float(sparc_rotation["fit_results"]["comparison"]["delta_chi2_baryon_minus_pmodel"])
    sparc_operational_pass = chi2_better_model == "pmodel_corrected" and delta_chi2 > 0.0

    vector_summary = vector_fit["summary"]
    vector_gate_summary = vector_gate["summary"]
    lambda_rot_value = float(vector_spin["summary"]["lambda_rot_value"])
    lambda_rot_sigma = float(vector_spin["summary"]["lambda_rot_sigma"])

    operational_kappa_matches_pbg_candidate = math.isclose(
        sparc_kappa,
        PBG_CANDIDATE_KAPPA,
        rel_tol=0.0,
        abs_tol=1e-15,
    )
    current_pack_has_vector_to_kappa_bridge = False
    nonclosure_reason = "vector_exact_hierarchy_to_kappa_a_bridge_statement_absent"

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
            "source_id": "sparc_operational_pass_metrics",
            "present": True,
            "evidence": {"better_model_by_chi2": chi2_better_model, "delta_chi2_baryon_minus_pmodel": delta_chi2},
        },
        {
            "source_id": "pbg_fixed_kappa_candidate_note",
            "present": True,
            "evidence": hit(sparc_note_text, "candidate_rar_pbg_a0_fixed_kappa"),
        },
        {
            "source_id": "vector_exact_hierarchy_anchor_table",
            "present": True,
            "evidence": vector_summary,
        },
        {
            "source_id": "vector_exact_hierarchy_to_kappa_a_bridge_statement",
            "present": False,
            "evidence": None,
        },
    ]
    present_sources = [item["source_id"] for item in required_sources if item["present"]]
    missing_sources = [item["source_id"] for item in required_sources if not item["present"]]

    rar_models = sparc_freeze.get("models", [])
    fixed_kappa_rows = [item for item in rar_models if item.get("name") == "candidate_rar_pbg_a0_fixed_kappa"]
    fit_kappa_rows = [item for item in rar_models if item.get("name") == "candidate_rar_pbg_fit_kappa"]

    payloads = {
        "mass_origin_dark_matter_postnewtonian_source_inventory": payload(
            "8.7.55.3.1",
            "Dark-matter elimination source inventory",
            {
                "mass_origin_vector_qball_baryon_tau_neutron_fit_table_json": rel(VECTOR_FIT),
                "mass_origin_vector_qball_second_route_gate_refresh_json": rel(VECTOR_GATE),
                "sparc_rotation_curve_pmodel_audit_json": rel(SPARC_ROTATION),
                "sparc_rar_freeze_test_json": rel(SPARC_FREEZE),
                "part1_core_theory_markdown": rel(PART1),
                "part2_astrophysics_markdown": rel(PART2),
                "sparc_rar_note_markdown": rel(SPARC_NOTE),
            },
            "Inventory whether the reopened exact vector hierarchy already has the public source pack needed to derive the SPARC coefficient kappa_a from P dynamics.",
            {
                "operational_relation": "a0 = kappa_a c H0^(P)",
                "bridge_rule": "a first-principles derivation requires a public statement that maps the exact vector hierarchy to the galactic weak-field/post-Newtonian acceleration coefficient",
            },
            [
                row(
                    "dark_matter_postnewtonian_source_inventory_complete",
                    "pass",
                    "dark-matter post-Newtonian source inventory complete",
                    1,
                    "The reopened third-route source inventory is frozen.",
                ),
                row(
                    "dark_matter_postnewtonian_present_source_count",
                    "pass",
                    "present required source count",
                    len(present_sources),
                    "Five of the six required sources are already in the public pack.",
                ),
                row(
                    "dark_matter_postnewtonian_missing_source_count",
                    "reject" if missing_sources else "pass",
                    "missing required source count",
                    len(missing_sources),
                    "The remaining gap is the explicit vector-hierarchy-to-kappa_a bridge statement.",
                ),
            ],
            {
                "required_postnewtonian_bridge_sources": [item["source_id"] for item in required_sources],
                "present_postnewtonian_bridge_sources": present_sources,
                "missing_postnewtonian_bridge_sources": missing_sources,
                "first_route_to_close_or_none": "vector_exact_hierarchy_to_kappa_a_bridge_statement",
                "source_inventory_ready": True,
            },
            {
                "overall_status": "dark_matter_postnewtonian_source_inventory_frozen",
                "dark_matter_branch_active": True,
                "kappa_a_derivation_ready": False,
                "next_required_artifacts": ["kappa_a_vector_hierarchy_bridge_audit"],
            },
            {
                "required_source_rows": required_sources,
                "roadmap_step_line": hit(roadmap_text, "`8.7.55.3` 第三の攻め口"),
                "status_next_line": hit(status_text, "次の公式 step は `8.7.55.3`"),
            },
        ),
        "mass_origin_kappa_a_vector_hierarchy_bridge_audit": payload(
            "8.7.55.3.2",
            "kappa_a vector-hierarchy bridge audit",
            {
                "mass_origin_dark_matter_postnewtonian_source_inventory_json": "output/public/quantum/mass_origin_dark_matter_postnewtonian_source_inventory_metrics.json",
                "sparc_rotation_curve_pmodel_audit_json": rel(SPARC_ROTATION),
                "sparc_rar_freeze_test_json": rel(SPARC_FREEZE),
                "mass_origin_vector_qball_spin_orbit_freeze_audit_json": rel(VECTOR_SPIN),
            },
            "Audit whether the exact vector hierarchy plus the already-frozen weak-field/vector canon gives a no-new-free-parameter derivation of kappa_a, rather than only an operational SPARC candidate.",
            {
                "operational_kappa_rule": "kappa_a = a0 / (c H0^(P))",
                "candidate_background_value": PBG_CANDIDATE_KAPPA,
                "non_derivation_rule": "operational equality to the background candidate is not sufficient unless a public bridge maps the exact vector hierarchy to kappa_a",
            },
            [
                row(
                    "kappa_a_operational_value_matches_background_candidate",
                    "pass" if operational_kappa_matches_pbg_candidate else "reject",
                    "operational kappa_a matches background candidate",
                    1 if operational_kappa_matches_pbg_candidate else 0,
                    "The current SPARC operational coefficient equals the frozen background candidate 1/(2π).",
                ),
                row(
                    "kappa_a_vector_hierarchy_bridge_available",
                    "pass" if current_pack_has_vector_to_kappa_bridge else "reject",
                    "vector hierarchy to kappa_a bridge available",
                    1 if current_pack_has_vector_to_kappa_bridge else 0,
                    "Current public artifacts do not yet map the exact vector hierarchy to a galactic acceleration coefficient.",
                ),
                row(
                    "kappa_a_without_new_free_parameters_from_vector_hierarchy",
                    "pass" if current_pack_has_vector_to_kappa_bridge else "reject",
                    "kappa_a derivable from vector hierarchy without new parameters",
                    1 if current_pack_has_vector_to_kappa_bridge else 0,
                    "The reopened mass-origin branch currently supplies mass ratios and lambda_rot reuse, but not a no-new-free-parameter kappa_a derivation.",
                ),
            ],
            {
                "sparc_operational_kappa_a_value": sparc_kappa,
                "sparc_operational_a0_m_s2": sparc_a0,
                "sparc_operational_h0p_si_s^-1": sparc_h0p,
                "sparc_operational_formula": sparc_formula,
                "pbg_candidate_kappa_value": PBG_CANDIDATE_KAPPA,
                "operational_kappa_matches_pbg_candidate": operational_kappa_matches_pbg_candidate,
                "candidate_fit_kappa_is_only_reparameterized_baseline": True,
                "vector_exact_hierarchy_to_kappa_a_bridge_available": current_pack_has_vector_to_kappa_bridge,
                "bridge_nonclosure_reason_or_none": None if current_pack_has_vector_to_kappa_bridge else nonclosure_reason,
            },
            {
                "overall_status": "operational_kappa_present_but_vector_bridge_absent",
                "dark_matter_branch_active": True,
                "kappa_a_derivation_ready": False,
                "next_required_artifacts": ["dark_matter_postnewtonian_gate_refresh"],
            },
            {
                "vector_fit_summary": vector_summary,
                "vector_gate_summary": vector_gate_summary,
                "lambda_rot_reuse_summary": {
                    "lambda_rot_value": lambda_rot_value,
                    "lambda_rot_sigma": lambda_rot_sigma,
                    "cross_scale_connection_ready": vector_spin["summary"]["cross_scale_connection_ready"],
                },
                "sparc_note_fixed_candidate_line": hit(sparc_note_text, "candidate_rar_pbg_a0_fixed_kappa"),
                "sparc_note_fit_candidate_line": hit(sparc_note_text, "candidate_rar_pbg_fit_kappa"),
                "freeze_test_fixed_kappa_rows_sample": sample(fixed_kappa_rows),
                "freeze_test_fit_kappa_rows_sample": sample(fit_kappa_rows),
            },
        ),
        "mass_origin_dark_matter_postnewtonian_gate_refresh": payload(
            "8.7.55.3.3",
            "Dark-matter post-Newtonian gate refresh",
            {
                "mass_origin_kappa_a_vector_hierarchy_bridge_audit_json": "output/public/quantum/mass_origin_kappa_a_vector_hierarchy_bridge_audit_metrics.json",
                "sparc_rotation_curve_pmodel_audit_json": rel(SPARC_ROTATION),
                "mass_origin_vector_qball_second_route_gate_refresh_json": rel(VECTOR_GATE),
            },
            "Refresh the reopened third-route gate and decide whether the current pack already supports a first-principles dark-matter-elimination claim for SPARC.",
            {
                "close_rule": "close only if SPARC operational pass exists and kappa_a is derivable from the reopened exact vector hierarchy without a new parameter",
            },
            [
                row(
                    "dark_matter_postnewtonian_operational_sparc_pass",
                    "pass" if sparc_operational_pass else "reject",
                    "operational SPARC pass available",
                    1 if sparc_operational_pass else 0,
                    "The SPARC operational audit still prefers the P-model corrected curve over baryon-only.",
                ),
                row(
                    "dark_matter_postnewtonian_first_principles_bridge_ready",
                    "pass" if current_pack_has_vector_to_kappa_bridge else "reject",
                    "first-principles kappa_a bridge ready",
                    1 if current_pack_has_vector_to_kappa_bridge else 0,
                    "Operational SPARC success is not yet backed by a vector-hierarchy derivation of kappa_a.",
                ),
                row(
                    "dark_matter_postnewtonian_branch_closeable",
                    "pass" if sparc_operational_pass and current_pack_has_vector_to_kappa_bridge else "reject",
                    "dark-matter post-Newtonian branch closeable",
                    1 if sparc_operational_pass and current_pack_has_vector_to_kappa_bridge else 0,
                    "The branch remains open because the bridge artifact is missing.",
                ),
            ],
            {
                "sparc_operational_pass_still_available": sparc_operational_pass,
                "mass_origin_branch_reopen_ready": bool(vector_gate_summary["mass_origin_branch_reopen_ready"]),
                "kappa_a_first_principles_derivation_ready": current_pack_has_vector_to_kappa_bridge,
                "dark_matter_postnewtonian_branch_closeable": sparc_operational_pass and current_pack_has_vector_to_kappa_bridge,
                "recommended_next_route_or_none": "8.7.55.3.5",
            },
            {
                "overall_status": "dark_matter_postnewtonian_operational_pass_retained_but_derivation_blocked",
                "dark_matter_branch_active": True,
                "advance_to_dark_matter_closeout": False,
                "new_branch_required": True,
                "next_required_artifacts": ["vector_exact_hierarchy_to_kappa_a_bridge_statement"],
            },
            {
                "sparc_rotation_summary": sparc_rotation["fit_results"],
                "vector_gate_summary": vector_gate_summary,
            },
        ),
        "mass_origin_dark_matter_vector_bridge_route_contract": payload(
            "8.7.55.3.4",
            "Dark-matter vector bridge residual route contract",
            {
                "mass_origin_dark_matter_postnewtonian_gate_refresh_json": "output/public/quantum/mass_origin_dark_matter_postnewtonian_gate_refresh_metrics.json",
                "mass_origin_kappa_a_vector_hierarchy_bridge_audit_json": "output/public/quantum/mass_origin_kappa_a_vector_hierarchy_bridge_audit_metrics.json",
            },
            "Freeze the residual route after confirming that SPARC operational success persists but the exact vector hierarchy still lacks a public bridge to kappa_a.",
            {
                "selected_residual_route": "vector_exact_hierarchy_to_kappa_a_bridge",
                "missing_artifact": "vector_exact_hierarchy_to_kappa_a_bridge_statement",
            },
            [
                row(
                    "dark_matter_vector_bridge_route_contract_complete",
                    "pass",
                    "dark-matter vector bridge route contract complete",
                    1,
                    "The residual third-route contract is frozen.",
                ),
                row(
                    "dark_matter_vector_bridge_missing_artifact",
                    "reject",
                    "missing dark-matter bridge artifact",
                    1,
                    "The exact vector hierarchy still lacks a public bridge statement to kappa_a.",
                ),
                row(
                    "dark_matter_vector_bridge_split_contract_ready",
                    "pass",
                    "dark-matter vector bridge split contract ready",
                    1,
                    "The next residual branch may start from the missing bridge statement.",
                ),
            ],
            {
                "selected_residual_route": "vector_exact_hierarchy_to_kappa_a_bridge",
                "missing_dark_matter_artifact": "vector_exact_hierarchy_to_kappa_a_bridge_statement",
                "split_contract_ready": True,
            },
            {
                "overall_status": "dark_matter_vector_bridge_route_contract_frozen",
                "dark_matter_branch_active": True,
                "advance_to_dark_matter_closeout": False,
                "new_branch_required": True,
                "next_required_artifacts": [
                    "vector_exact_hierarchy_to_kappa_a_bridge_source_inventory",
                    "vector_exact_hierarchy_to_kappa_a_bridge_audit",
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


# Function: Run the dark-matter post-Newtonian branch when invoked as a script.

if __name__ == "__main__":
    main()
