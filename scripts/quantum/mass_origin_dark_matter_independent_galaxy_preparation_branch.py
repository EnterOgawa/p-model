#!/usr/bin/env python3
"""
Freeze the independent-galaxy intake preparation branch for 8.7.55.3.137-.140.

This branch starts after the direct kappa_a bridge and the paper-side sync are
already closed. Its job is to turn the follow-through checklist into a launchable
dataset-intake route by:

1. freezing a non-SPARC public-source registry,
2. freezing the same-baryon-interface rule shared by SPARC and the future
   independent-galaxy comparison,
3. inventorying the baryonic-decomposition candidate packs needed for that
   comparison, and
4. deciding whether the independent-galaxy dataset-intake branch can launch now.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output" / "public" / "quantum"
PRIMARY_SOURCES = ROOT / "doc" / "PRIMARY_SOURCES.md"
PART2 = ROOT / "doc" / "paper" / "11_part2_astrophysics.md"
PART4 = ROOT / "doc" / "paper" / "13_part4_verification.md"
PART5 = ROOT / "doc" / "paper" / "14_part5_future_predictions.md"
MANIFEST = ROOT / "data" / "cosmology" / "sources" / "independent_galaxy_registry" / "manifest.json"
PREVIOUS_GATE = OUT / "mass_origin_dark_matter_external_feedback_dataset_handoff_gate_metrics.json"


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


# Function: Return a manifest entry by key, or None if it is absent.

def manifest_entry(entries: dict[str, dict], key: str) -> dict | None:
    return entries.get(key)


# Function: Build a source-registry evidence record.

def source_record(entries: dict[str, dict], key: str, note: str) -> dict:
    entry = manifest_entry(entries, key)
    return {
        "item_key": key,
        "present": entry is not None,
        "note": note,
        "evidence": entry,
    }


# Function: Run the independent-galaxy preparation branch and write artifacts.

def main() -> None:
    for path in (PRIMARY_SOURCES, PART2, PART4, PART5, MANIFEST, PREVIOUS_GATE):
        req(path)

    primary_sources_text = read_text(PRIMARY_SOURCES)
    part2_text = read_text(PART2)
    part4_text = read_text(PART4)
    part5_text = read_text(PART5)
    manifest = read_json(MANIFEST)
    previous_gate = read_json(PREVIOUS_GATE)

    manifest_entries = {entry["key"]: entry for entry in manifest["entries"]}
    external_share_ready = bool(previous_gate["summary"]["external_share_ready"])

    registry_keys = [
        "things_data_products",
        "things_survey_overview",
        "things_mass_models",
        "sings_overview",
        "little_things_sample",
        "little_things_data",
        "little_things_survey_overview",
        "little_things_mass_models",
    ]
    registry_records = [
        source_record(
            manifest_entries,
            key,
            "This public-source candidate must exist in the independent-galaxy registry manifest."
        )
        for key in registry_keys
    ]
    present_registry_records = [item for item in registry_records if item["present"]]
    missing_registry_records = [item for item in registry_records if not item["present"]]
    public_source_registry_ready = not missing_registry_records

    baryon_formula = r"V_{\rm bar}^2(R)=V_{\rm gas}^2+\Upsilon V_{\rm disk}^2+(f_{\rm bulge}\Upsilon)V_{\rm bulge}^2"
    part2_formula_hit = hit(part2_text, baryon_formula)
    part4_rule_hit = hit(part4_text, "同一の baryon I/F")
    part4_formula_hit = hit(part4_text, baryon_formula)
    part5_rule_hit = hit(part5_text, "同一 baryon I/F")
    part5_formula_hit = hit(part5_text, baryon_formula)
    same_baryon_interface_rule_ready = all(
        item is not None for item in (part2_formula_hit, part4_rule_hit, part4_formula_hit, part5_rule_hit, part5_formula_hit)
    )

    decomposition_keys = [
        "things_mass_models",
        "sings_overview",
        "little_things_data",
        "little_things_mass_models",
    ]
    decomposition_records = [
        source_record(
            manifest_entries,
            key,
            "This baryonic-decomposition candidate must exist in the registry manifest before intake can launch."
        )
        for key in decomposition_keys
    ]
    present_decomposition_records = [item for item in decomposition_records if item["present"]]
    missing_decomposition_records = [item for item in decomposition_records if not item["present"]]
    independent_baryonic_pack_ready = not missing_decomposition_records

    launch_dataset_intake_now = (
        external_share_ready
        and public_source_registry_ready
        and same_baryon_interface_rule_ready
        and independent_baryonic_pack_ready
    )

    payloads = {
        "mass_origin_dark_matter_independent_galaxy_public_source_inventory": payload(
            "8.7.55.3.137",
            "Independent-galaxy public-source inventory",
            {
                "primary_sources_markdown": rel(PRIMARY_SOURCES),
                "independent_galaxy_registry_manifest": rel(MANIFEST),
            },
            "Freeze the first non-SPARC public-source registry that will support the direct-kappa independent-galaxy intake.",
            {
                "source_inventory_rule": "the source registry is ready only if the THINGS, SINGS, and LITTLE THINGS public-source entries are all cached in the canonical manifest"
            },
            [
                row("independent_galaxy_public_source_inventory_complete", "pass", "independent-galaxy public-source inventory complete", 1, "The non-SPARC source registry was inventoried against the canonical manifest."),
                row("independent_galaxy_public_source_present_count", "pass" if public_source_registry_ready else "reject", "present non-SPARC public-source count", len(present_registry_records), "The public-source registry is ready only when every required candidate source is present."),
                row("independent_galaxy_public_source_missing_count", "pass" if public_source_registry_ready else "reject", "missing non-SPARC public-source count", len(missing_registry_records), "The missing count identifies the next blocking source entries."),
            ],
            {
                "required_public_source_entries": registry_keys,
                "present_public_source_entries": [item["item_key"] for item in present_registry_records],
                "missing_public_source_entries": [item["item_key"] for item in missing_registry_records],
                "public_source_registry_ready": public_source_registry_ready,
                "candidate_sample_families": ["THINGS+SINGS", "LITTLE THINGS"],
            },
            {
                "overall_status": "independent_public_source_registry_frozen",
                "public_source_registry_ready": public_source_registry_ready,
                "next_required_artifacts": [
                    "mass_origin_dark_matter_same_baryon_interface_rule_freeze",
                    "mass_origin_dark_matter_independent_baryonic_decomposition_pack_inventory",
                ],
            },
            {
                "source_registry_records": registry_records,
                "registry_manifest": manifest["entries"],
                "primary_sources_section_hit": hit(primary_sources_text, "SPARC 以外の公開 rotation-curve sample"),
            },
        ),
        "mass_origin_dark_matter_same_baryon_interface_rule_freeze": payload(
            "8.7.55.3.138",
            "Same-baryon-interface rule freeze",
            {
                "part2_markdown": rel(PART2),
                "part4_markdown": rel(PART4),
                "part5_markdown": rel(PART5),
            },
            "Freeze the common baryon-interface rule that must remain identical between SPARC and the future independent-galaxy sample.",
            {
                "same_baryon_rule": "the rule is ready only if Part II defines the baryon interface formula and both Part IV and Part V restate that independent galaxies must use the same interface"
            },
            [
                row("same_baryon_interface_part2_formula_available", "pass" if part2_formula_hit else "reject", "Part II baryon-interface formula available", 1 if part2_formula_hit else 0, "Part II must remain the canonical source for the baryon-interface formula."),
                row("same_baryon_interface_part4_rule_available", "pass" if part4_rule_hit and part4_formula_hit else "reject", "Part IV same-baryon-interface rule available", 1 if part4_rule_hit and part4_formula_hit else 0, "Part IV must state that independent galaxies use the same baryon interface."),
                row("same_baryon_interface_part5_rule_available", "pass" if part5_rule_hit and part5_formula_hit else "reject", "Part V same-baryon-interface rule available", 1 if part5_rule_hit and part5_formula_hit else 0, "Part V must carry the same canonical wording into the future-test section."),
            ],
            {
                "same_baryon_interface_rule_ready": same_baryon_interface_rule_ready,
                "canonical_baryon_interface_formula": baryon_formula,
            },
            {
                "overall_status": "same_baryon_interface_rule_frozen",
                "same_baryon_interface_rule_ready": same_baryon_interface_rule_ready,
                "next_required_artifacts": ["mass_origin_dark_matter_independent_baryonic_decomposition_pack_inventory"],
            },
            {
                "part2_formula_hit": part2_formula_hit,
                "part4_rule_hit": part4_rule_hit,
                "part4_formula_hit": part4_formula_hit,
                "part5_rule_hit": part5_rule_hit,
                "part5_formula_hit": part5_formula_hit,
            },
        ),
        "mass_origin_dark_matter_independent_baryonic_decomposition_pack_inventory": payload(
            "8.7.55.3.139",
            "Independent baryonic-decomposition pack inventory",
            {
                "primary_sources_markdown": rel(PRIMARY_SOURCES),
                "independent_galaxy_registry_manifest": rel(MANIFEST),
            },
            "Inventory the candidate baryonic-decomposition packs that let the future independent-galaxy comparison stay on the same baryon interface as SPARC.",
            {
                "baryonic_pack_rule": "the decomposition inventory is ready only if at least one spiral-family pack and one dwarf-family pack have both rotation-curve and baryonic-side public entries"
            },
            [
                row("independent_baryonic_pack_inventory_complete", "pass", "independent baryonic-decomposition inventory complete", 1, "The decomposition candidates were inventoried against the manifest and the source registry section."),
                row("independent_baryonic_pack_present_count", "pass" if independent_baryonic_pack_ready else "reject", "present independent baryonic-decomposition pack count", len(present_decomposition_records), "The intake can open only when both spiral and dwarf decomposition packs are publicly registered."),
                row("independent_baryonic_pack_missing_count", "pass" if independent_baryonic_pack_ready else "reject", "missing independent baryonic-decomposition pack count", len(missing_decomposition_records), "The missing count identifies the decomposition-side blockers."),
            ],
            {
                "required_baryonic_pack_entries": decomposition_keys,
                "present_baryonic_pack_entries": [item["item_key"] for item in present_decomposition_records],
                "missing_baryonic_pack_entries": [item["item_key"] for item in missing_decomposition_records],
                "independent_baryonic_decomposition_pack_ready": independent_baryonic_pack_ready,
                "candidate_pack_families": {
                    "spiral_family": ["things_mass_models", "sings_overview"],
                    "dwarf_family": ["little_things_data", "little_things_mass_models"],
                },
            },
            {
                "overall_status": "independent_baryonic_pack_inventory_frozen",
                "independent_baryonic_decomposition_pack_ready": independent_baryonic_pack_ready,
                "next_required_artifacts": ["mass_origin_dark_matter_independent_galaxy_dataset_intake_kickoff_gate"],
            },
            {
                "decomposition_records": decomposition_records,
                "primary_sources_decomposition_hit": hit(primary_sources_text, "baryonic decomposition"),
            },
        ),
        "mass_origin_dark_matter_independent_galaxy_dataset_intake_kickoff_gate": payload(
            "8.7.55.3.140",
            "External feedback / dataset-intake kickoff gate",
            {
                "previous_external_feedback_gate_json": rel(PREVIOUS_GATE),
                "source_registry_manifest": rel(MANIFEST),
                "primary_sources_markdown": rel(PRIMARY_SOURCES),
                "part4_markdown": rel(PART4),
                "part5_markdown": rel(PART5),
            },
            "Decide whether the non-SPARC independent-galaxy dataset-intake branch can launch immediately after the source-registry and baryon-interface preparation work.",
            {
                "kickoff_rule": "the kickoff is ready only if external share is already ready, the non-SPARC public-source registry is ready, the same-baryon-interface rule is frozen, and the baryonic-decomposition packs are all registered"
            },
            [
                row("external_share_ready_for_parallel_feedback", "pass" if external_share_ready else "reject", "external share ready for parallel feedback", 1 if external_share_ready else 0, "The declaration pack must already be externally shareable before the dataset-intake branch opens."),
                row("independent_galaxy_dataset_intake_launch_ready", "pass" if launch_dataset_intake_now else "reject", "independent-galaxy dataset intake launch ready", 1 if launch_dataset_intake_now else 0, "The intake launches only when registry, interface, and decomposition prerequisites are all ready."),
                row("independent_galaxy_dataset_intake_kickoff_gate_complete", "pass", "independent-galaxy dataset-intake kickoff gate complete", 1, "The post-closeout kickoff gate is now frozen."),
            ],
            {
                "external_share_ready": external_share_ready,
                "public_source_registry_ready": public_source_registry_ready,
                "same_baryon_interface_rule_ready": same_baryon_interface_rule_ready,
                "independent_baryonic_decomposition_pack_ready": independent_baryonic_pack_ready,
                "await_external_feedback": external_share_ready,
                "launch_dataset_intake_now": launch_dataset_intake_now,
                "recommended_next_route_or_none": "8.7.55.3.141" if launch_dataset_intake_now else None,
                "selected_next_route": "independent_galaxy_dataset_intake_execution" if launch_dataset_intake_now else "independent_galaxy_public_source_inventory",
            },
            {
                "overall_status": "independent_galaxy_dataset_intake_ready" if launch_dataset_intake_now else "independent_galaxy_dataset_intake_prerequisites_missing",
                "third_route_fully_closed": True,
                "independent_galaxy_intake_preparation_complete": True,
                "next_required_artifacts": ["independent_galaxy_dataset_intake_execution_branch"] if launch_dataset_intake_now else [],
            },
            {
                "previous_external_feedback_gate": previous_gate["summary"],
                "source_registry_missing": [item["item_key"] for item in missing_registry_records],
                "decomposition_missing": [item["item_key"] for item in missing_decomposition_records],
                "same_baryon_rule_hits": {
                    "part2_formula_hit": part2_formula_hit,
                    "part4_rule_hit": part4_rule_hit,
                    "part4_formula_hit": part4_formula_hit,
                    "part5_rule_hit": part5_rule_hit,
                    "part5_formula_hit": part5_formula_hit,
                },
            },
        ),
    }

    for stem, data in payloads.items():
        write_artifact(stem, data)
        print(f"[ok] wrote {OUT / (stem + '_metrics.json')}")
        print(f"[ok] wrote {OUT / (stem + '_rows.csv')}")


# Function: Run the independent-galaxy preparation branch when invoked as a script.

if __name__ == "__main__":
    main()
