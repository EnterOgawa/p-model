#!/usr/bin/env python3
"""
Freeze the paper-side sync pack for 8.7.55.3.124-.128.

The direct kappa_a bridge already closed the third route on the theory side.
This branch does not run a paper build. Instead, it audits whether the
canonical paper sources now contain the wording needed to synchronize the
dark-matter-elimination declaration across Part II, Part III-A, Part IV,
and Part V, and then freezes the user-build handoff gate.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output" / "public" / "quantum"
DIRECT_BRIDGE = OUT / "mass_origin_direct_kappa_bridge_statement_freeze_metrics.json"
DIRECT_RETRY = OUT / "mass_origin_dark_matter_postnewtonian_direct_bridge_retry_metrics.json"
DECLARATION_GATE = OUT / "mass_origin_dark_matter_elimination_declaration_gate_metrics.json"
PART2 = ROOT / "doc" / "paper" / "11_part2_astrophysics.md"
PART3A = ROOT / "doc" / "paper" / "12_part3a_quantum_foundations.md"
PART4 = ROOT / "doc" / "paper" / "13_part4_verification.md"
PART5 = ROOT / "doc" / "paper" / "14_part5_future_predictions.md"


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


# Function: Audit a wording target in a paper source.

def audit_target(file_key: str, path: Path, text: str, pattern: str, note: str) -> dict:
    target_hit = hit(text, pattern)
    return {
        "file_key": file_key,
        "file": rel(path),
        "pattern": pattern,
        "present": target_hit is not None,
        "note": note,
        "evidence": target_hit,
    }


# Function: Run the paper-side sync audit branch and write artifacts.

def main() -> None:
    for path in (DIRECT_BRIDGE, DIRECT_RETRY, DECLARATION_GATE, PART2, PART3A, PART4, PART5):
        req(path)

    direct_bridge = read_json(DIRECT_BRIDGE)
    direct_retry = read_json(DIRECT_RETRY)
    declaration_gate = read_json(DECLARATION_GATE)
    part2_text = read_text(PART2)
    part3a_text = read_text(PART3A)
    part4_text = read_text(PART4)
    part5_text = read_text(PART5)

    theory_side_ready = bool(declaration_gate["summary"]["dark_matter_elimination_declaration_ready"])
    paper_side_allowed = bool(declaration_gate["decision"]["advance_to_paper_side_sync"])
    direct_bridge_ready = bool(direct_bridge["summary"]["vector_exact_hierarchy_to_kappa_a_bridge_statement_available"])
    kappa_value = float(direct_bridge["summary"]["kappa_a_value"])

    inventory_targets = [
        audit_target(
            "part2_section_4_14",
            PART2,
            part2_text,
            "### 4.14 銀河回転曲線（SPARC",
            "Part II must still expose the SPARC section that carries the direct-kappa wording.",
        ),
        audit_target(
            "part2_direct_kappa_input",
            PART2,
            part2_text,
            "derived scale を使用",
            "Part II must describe a0 as the derived late-time background scale, not as a free SPARC coefficient.",
        ),
        audit_target(
            "part2_direct_kappa_equation",
            PART2,
            part2_text,
            r"\kappa_a \equiv \frac{a_0}{cH_{0}^{(P)}} = \frac{1}{2\pi}",
            "Part II must freeze the explicit derived-kappa equation.",
        ),
        audit_target(
            "part3a_bridge_connection",
            PART3A,
            part3a_text,
            "mass-origin route で固定した exact vector hierarchy",
            "Part III-A must explicitly connect the mass-origin route to the direct kappa_a bridge.",
        ),
        audit_target(
            "part3a_direct_bridge_quantity",
            PART3A,
            part3a_text,
            "direct bridge quantity",
            "Part III-A must state that kappa_a is the direct bridge quantity rather than a phenomenological fit parameter.",
        ),
        audit_target(
            "part4_direct_bridge_artifact_list",
            PART4,
            part4_text,
            "mass_origin_direct_kappa_bridge_statement_freeze_metrics.json",
            "Part IV must expose the direct-bridge public artifacts.",
        ),
        audit_target(
            "part4_independent_galaxy_reject_condition",
            PART4,
            part4_text,
            r"\kappa_a\neq1/(2\pi)",
            "Part IV must state the independent-galaxy falsification condition.",
        ),
        audit_target(
            "part5_independent_galaxy_section",
            PART5,
            part5_text,
            "### 3.1 独立銀河回転曲線による direct",
            "Part V must expose an independent-galaxy future-test section for the direct kappa_a bridge.",
        ),
        audit_target(
            "part5_independent_galaxy_reject_condition",
            PART5,
            part5_text,
            "同一 baryon I/F のまま有意に外れれば dark-matter-elimination branch を棄却する。",
            "Part V must state the rejection condition for an independent galaxy sample.",
        ),
    ]
    present_targets = [item for item in inventory_targets if item["present"]]
    missing_targets = [item for item in inventory_targets if not item["present"]]

    part2_ready = all(
        item["present"]
        for item in inventory_targets
        if item["file_key"] in {"part2_section_4_14", "part2_direct_kappa_input", "part2_direct_kappa_equation"}
    )
    part3a_ready = all(
        item["present"]
        for item in inventory_targets
        if item["file_key"] in {"part3a_bridge_connection", "part3a_direct_bridge_quantity"}
    )
    part4_ready = all(
        item["present"]
        for item in inventory_targets
        if item["file_key"] in {"part4_direct_bridge_artifact_list", "part4_independent_galaxy_reject_condition"}
    )
    part5_ready = all(
        item["present"]
        for item in inventory_targets
        if item["file_key"] in {"part5_independent_galaxy_section", "part5_independent_galaxy_reject_condition"}
    )
    paper_sync_ready = theory_side_ready and paper_side_allowed and part2_ready and part3a_ready and part4_ready and part5_ready

    payloads = {
        "mass_origin_dark_matter_paper_sync_inventory": payload(
            "8.7.55.3.124",
            "Dark-matter elimination paper-sync inventory",
            {
                "mass_origin_direct_kappa_bridge_statement_freeze_json": rel(DIRECT_BRIDGE),
                "mass_origin_dark_matter_elimination_declaration_gate_json": rel(DECLARATION_GATE),
                "part2_markdown": rel(PART2),
                "part3a_markdown": rel(PART3A),
                "part4_markdown": rel(PART4),
                "part5_markdown": rel(PART5),
            },
            "Inventory the canonical paper-side wording targets that must reflect the direct kappa_a bridge and the dark-matter-elimination declaration.",
            {
                "inventory_rule": "inventory passes only if Part II, Part III-A, Part IV, and Part V each expose the direct bridge wording target required by the declaration gate"
            },
            [
                row(
                    "dark_matter_paper_sync_inventory_complete",
                    "pass",
                    "dark-matter paper-side sync inventory complete",
                    1,
                    "The paper-side sync inventory was executed against the updated paper sources.",
                ),
                row(
                    "dark_matter_paper_sync_present_target_count",
                    "pass" if not missing_targets else "reject",
                    "present paper-side sync target count",
                    len(present_targets),
                    "All required wording targets must be present before the handoff gate can be opened.",
                ),
                row(
                    "dark_matter_paper_sync_missing_target_count",
                    "pass" if not missing_targets else "reject",
                    "missing paper-side sync target count",
                    len(missing_targets),
                    "The paper-side sync inventory closes only when the missing count stays zero.",
                ),
            ],
            {
                "required_paper_sync_targets": [item["file_key"] for item in inventory_targets],
                "present_paper_sync_targets": [item["file_key"] for item in present_targets],
                "missing_paper_sync_targets": [item["file_key"] for item in missing_targets],
                "source_inventory_ready": not missing_targets,
                "first_route_to_close_or_none": None if not missing_targets else "paper_wording_target_missing",
            },
            {
                "overall_status": "dark_matter_paper_sync_inventory_frozen",
                "paper_side_sync_ready": not missing_targets,
                "next_required_artifacts": [
                    "mass_origin_part2_direct_kappa_wording_freeze",
                    "mass_origin_part3a_direct_kappa_bridge_wording_freeze",
                    "mass_origin_part4_part5_dark_matter_wording_pack_freeze",
                ],
            },
            {"inventory_targets": inventory_targets},
        ),
        "mass_origin_part2_direct_kappa_wording_freeze": payload(
            "8.7.55.3.125",
            "Part II direct-kappa wording freeze",
            {
                "part2_markdown": rel(PART2),
                "mass_origin_direct_kappa_bridge_statement_freeze_json": rel(DIRECT_BRIDGE),
            },
            "Freeze the Part II wording that upgrades kappa_a from an operational SPARC coefficient to the derived background-wave quantity 1/(2*pi).",
            {
                "part2_wording_rule": "Part II passes only if Section 4.14 states that a0 is derived from the late-time background wave and explicitly fixes kappa_a = 1/(2*pi)"
            },
            [
                row(
                    "part2_direct_kappa_wording_freeze_complete",
                    "pass",
                    "Part II direct-kappa wording freeze complete",
                    1,
                    "The Part II SPARC section was audited after the direct-bridge wording update.",
                ),
                row(
                    "part2_direct_kappa_wording_ready",
                    "pass" if part2_ready else "reject",
                    "Part II direct-kappa wording ready",
                    1 if part2_ready else 0,
                    "Part II is ready only if the input sentence, derived-scale wording, and explicit kappa equation are all present.",
                ),
                row(
                    "part2_direct_kappa_matches_theory_bridge",
                    "pass" if direct_bridge_ready and part2_ready else "reject",
                    "Part II wording matches the theory-side direct bridge",
                    1 if direct_bridge_ready and part2_ready else 0,
                    "The paper wording must track the already-frozen theory-side bridge statement.",
                ),
            ],
            {
                "part2_direct_kappa_wording_ready": part2_ready,
                "kappa_a_value": kappa_value,
                "part2_target_section": "4.14",
                "part2_updates_are_source_only_no_build_run": True,
            },
            {
                "overall_status": "part2_direct_kappa_wording_frozen" if part2_ready else "part2_direct_kappa_wording_missing",
                "next_required_artifacts": ["mass_origin_part3a_direct_kappa_bridge_wording_freeze"],
            },
            {
                "target_hits": [
                    item for item in inventory_targets if item["file_key"].startswith("part2_")
                ]
            },
        ),
        "mass_origin_part3a_direct_kappa_bridge_wording_freeze": payload(
            "8.7.55.3.126",
            "Part III-A bridge wording freeze",
            {
                "part3a_markdown": rel(PART3A),
                "mass_origin_direct_kappa_bridge_statement_freeze_json": rel(DIRECT_BRIDGE),
            },
            "Freeze the Part III-A wording that connects the exact vector mass-origin hierarchy to the direct background-wave kappa_a bridge.",
            {
                "part3a_wording_rule": "Part III-A passes only if it explicitly connects the mass-origin route to the direct bridge quantity kappa_a = 1/(2*pi)"
            },
            [
                row(
                    "part3a_direct_bridge_wording_freeze_complete",
                    "pass",
                    "Part III-A direct bridge wording freeze complete",
                    1,
                    "The Part III-A bridge wording was audited after the new connection paragraph was added.",
                ),
                row(
                    "part3a_direct_bridge_wording_ready",
                    "pass" if part3a_ready else "reject",
                    "Part III-A direct bridge wording ready",
                    1 if part3a_ready else 0,
                    "Part III-A is ready only if the mass-origin connection and the direct bridge quantity both appear in the paper source.",
                ),
                row(
                    "part3a_direct_bridge_matches_theory_bridge",
                    "pass" if direct_bridge_ready and part3a_ready else "reject",
                    "Part III-A bridge wording matches the theory-side direct bridge",
                    1 if direct_bridge_ready and part3a_ready else 0,
                    "The Part III-A connection must not drift away from the already-frozen theory bridge.",
                ),
            ],
            {
                "part3a_direct_bridge_wording_ready": part3a_ready,
                "kappa_a_value": kappa_value,
                "bridge_connection_kind": "mass_origin_exact_vector_hierarchy_to_background_wave",
                "part3a_updates_are_source_only_no_build_run": True,
            },
            {
                "overall_status": "part3a_direct_bridge_wording_frozen" if part3a_ready else "part3a_direct_bridge_wording_missing",
                "next_required_artifacts": ["mass_origin_part4_part5_dark_matter_wording_pack_freeze"],
            },
            {
                "target_hits": [
                    item for item in inventory_targets if item["file_key"].startswith("part3a_")
                ]
            },
        ),
        "mass_origin_part4_part5_dark_matter_wording_pack_freeze": payload(
            "8.7.55.3.127",
            "Part IV / Part V falsification-and-future-test wording pack freeze",
            {
                "part4_markdown": rel(PART4),
                "part5_markdown": rel(PART5),
                "mass_origin_dark_matter_elimination_declaration_gate_json": rel(DECLARATION_GATE),
            },
            "Freeze the Part IV falsification wording and the Part V future-test wording that follow from the direct kappa_a bridge declaration.",
            {
                "part4_rule": "Part IV must expose the direct-bridge public artifacts and the independent-galaxy falsification condition",
                "part5_rule": "Part V must expose the independent-galaxy future-test section and the rejection condition for kappa_a != 1/(2*pi)",
            },
            [
                row(
                    "part4_part5_wording_pack_freeze_complete",
                    "pass",
                    "Part IV / Part V wording pack freeze complete",
                    1,
                    "The falsification and future-test wording pack was audited against the updated paper sources.",
                ),
                row(
                    "part4_dark_matter_falsification_wording_ready",
                    "pass" if part4_ready else "reject",
                    "Part IV dark-matter falsification wording ready",
                    1 if part4_ready else 0,
                    "Part IV is ready only if it exposes the direct-bridge artifacts and the independent-galaxy reject condition.",
                ),
                row(
                    "part5_direct_kappa_future_test_wording_ready",
                    "pass" if part5_ready else "reject",
                    "Part V direct-kappa future-test wording ready",
                    1 if part5_ready else 0,
                    "Part V is ready only if the independent-galaxy future-test section and reject condition are present.",
                ),
            ],
            {
                "part4_dark_matter_falsification_wording_ready": part4_ready,
                "part5_direct_kappa_future_test_wording_ready": part5_ready,
                "independent_galaxy_reject_rule": "reject if an independent galaxy sample requires kappa_a != 1/(2*pi) under the same baryon interface",
                "part4_part5_updates_are_source_only_no_build_run": True,
            },
            {
                "overall_status": "part4_part5_dark_matter_wording_pack_frozen" if part4_ready and part5_ready else "part4_part5_dark_matter_wording_pack_missing",
                "next_required_artifacts": ["mass_origin_dark_matter_paper_side_user_build_handoff_gate"],
            },
            {
                "target_hits": [
                    item
                    for item in inventory_targets
                    if item["file_key"].startswith("part4_") or item["file_key"].startswith("part5_")
                ]
            },
        ),
        "mass_origin_dark_matter_paper_side_user_build_handoff_gate": payload(
            "8.7.55.3.128",
            "Paper-side user-build handoff gate",
            {
                "mass_origin_dark_matter_elimination_declaration_gate_json": rel(DECLARATION_GATE),
                "mass_origin_dark_matter_paper_sync_inventory_json": "output/public/quantum/mass_origin_dark_matter_paper_sync_inventory_metrics.json",
                "part2_markdown": rel(PART2),
                "part3a_markdown": rel(PART3A),
                "part4_markdown": rel(PART4),
                "part5_markdown": rel(PART5),
            },
            "Confirm that the paper-side wording pack is complete and freeze the gate that lets the user trigger the next build explicitly, without running the build in this branch.",
            {
                "handoff_rule": "handoff passes only if the theory-side declaration gate passed and every paper-side wording target is present; paper build itself remains user-driven"
            },
            [
                row(
                    "dark_matter_paper_side_handoff_gate_complete",
                    "pass",
                    "dark-matter paper-side handoff gate complete",
                    1,
                    "The paper-side handoff gate was evaluated after the wording pack freeze.",
                ),
                row(
                    "paper_side_wording_pack_ready",
                    "pass" if paper_sync_ready else "reject",
                    "paper-side wording pack ready",
                    1 if paper_sync_ready else 0,
                    "The handoff gate opens only if all paper-side wording targets are present and the theory-side declaration gate already passed.",
                ),
                row(
                    "paper_build_remains_user_driven",
                    "pass",
                    "paper build remains user driven",
                    1,
                    "No paper build was executed in this branch because the operating policy keeps paper builds behind explicit user instruction.",
                ),
            ],
            {
                "theory_side_dark_matter_elimination_ready": theory_side_ready,
                "paper_side_wording_pack_ready": paper_sync_ready,
                "paper_side_user_build_handoff_ready": paper_sync_ready,
                "run_paper_build_now": False,
                "next_route_or_none": "8.7.55.3.129" if paper_sync_ready else None,
                "next_route_requires_explicit_user_instruction": True if paper_sync_ready else False,
            },
            {
                "overall_status": "paper_side_user_build_handoff_ready" if paper_sync_ready else "paper_side_user_build_handoff_blocked",
                "third_route_completed": theory_side_ready,
                "paper_side_sync_completed": paper_sync_ready,
                "await_explicit_user_build_instruction": paper_sync_ready,
                "next_required_artifacts": [] if paper_sync_ready else ["paper_wording_pack_completion"],
            },
            {
                "declaration_gate_summary": declaration_gate["summary"],
                "direct_retry_summary": direct_retry["summary"],
            },
        ),
    }

    for stem, data in payloads.items():
        write_artifact(stem, data)
        print(f"[ok] wrote {OUT / (stem + '_metrics.json')}")
        print(f"[ok] wrote {OUT / (stem + '_rows.csv')}")


# Function: Run the paper-side sync branch when invoked as a script.

if __name__ == "__main__":
    main()
