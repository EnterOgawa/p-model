#!/usr/bin/env python3
"""
Freeze the independent-galaxy radial-table reconstruction branch for 8.7.55.3.147-.153.

This branch continues after the pilot-intake execution branch has already frozen
the first independent spiral and dwarf pilot subsets. Its job is to:

1. inventory the THINGS-side spiral source candidates needed for same-baryon
   radial-table reconstruction,
2. inventory the LITTLE THINGS-side dwarf source candidates needed for the same
   reconstruction,
3. freeze the current spiral and dwarf reconstruction state,
4. retry the independent direct-kappa comparison with the new source evidence,
5. decide whether the intake branch can close now, and
6. if it cannot, formalize the next residual route.
"""

from __future__ import annotations

import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output" / "public" / "quantum"
REGISTRY_DIR = ROOT / "data" / "cosmology" / "sources" / "independent_galaxy_registry"
NON_SPARC_DIR = ROOT / "data" / "cosmology" / "non_sparc_rotation_curves"
PILOT_MANIFEST = REGISTRY_DIR / "pilot_intake_manifest.json"
PREVIOUS_GATE = OUT / "mass_origin_dark_matter_independent_galaxy_dataset_intake_declaration_gate_metrics.json"
THINGS_PAGE = REGISTRY_DIR / "things_data_products_20260320.html"
THINGS_MASS_MODELS = REGISTRY_DIR / "things_mass_models_0810.2100_abs.html"
SINGS_PAGE = REGISTRY_DIR / "sings_overview_20260320.html"
LITTLE_THINGS_SAMPLE = REGISTRY_DIR / "little_things_sample_20260320.html"
LITTLE_THINGS_DATA = REGISTRY_DIR / "little_things_pubdata_20260320.html"
LITTLE_THINGS_MASS_MODELS = REGISTRY_DIR / "little_things_mass_models_1502.01281_abs.html"
RECONSTRUCTION_MANIFEST = NON_SPARC_DIR / "pilot_reconstruction_manifest.json"


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


# Function: Read a UTF-8 text file while tolerating legacy encodings.

def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


# Function: Convert an absolute path to a repo-relative string.

def rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


# Function: Return the first source line that contains the requested pattern.

def hit(text: str, pattern: str) -> dict | None:
    for line_no, line in enumerate(text.splitlines(), start=1):
        if pattern in line:
            return {"pattern": pattern, "line": line_no, "text": line.strip()}

    return None


# Function: Extract a matching href from cached HTML, if present.

def href(text: str, fragment: str) -> str | None:
    match = re.search(rf'href="([^"]*{re.escape(fragment)}[^"]*)"', text, flags=re.IGNORECASE)
    if not match:
        return None

    return match.group(1)


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


# Function: Save the canonical reconstruction manifest for the next residual branch.

def write_manifest(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


# Function: Return the selected family records from the pilot manifest.

def selected_family_records(pilot_manifest: dict, family_key: str) -> list[dict]:
    selected = set(pilot_manifest["selected_independent_pilot_subset"][family_key])
    return [record for record in pilot_manifest[family_key] if record["name"] in selected]


# Function: Return a THINGS robust-product URL for the requested galaxy and suffix.

def things_product_url(things_html: str, galaxy_name: str, suffix: str) -> str | None:
    relative = href(things_html, f"Data_files/{galaxy_name}_{suffix}_THINGS.FITS")
    if relative is None:
        return None

    return urljoin("https://www2.mpia-hd.mpg.de/THINGS/Data.html", relative)


# Function: Build a selected THINGS spiral reconstruction-source record.

def spiral_reconstruction_record(things_html: str, record: dict) -> dict:
    moment0_url = things_product_url(things_html, record["name"], "RO_MOM0")
    moment1_url = things_product_url(things_html, record["name"], "RO_MOM1")
    moment2_url = things_product_url(things_html, record["name"], "RO_MOM2")
    cube_url = things_product_url(things_html, record["name"], "RO_CUBE")
    rotation_primitives_ready = all(item is not None for item in (moment0_url, moment1_url, moment2_url, cube_url))
    baryon_primitives_ready = bool(record["baryon_side_family_urls"]["things_mass_models"]) and bool(record["baryon_side_family_urls"]["sings_overview"])

    return {
        "name": record["name"],
        "display_name": record["display_name"],
        "survey_family": record["survey_family"],
        "rotation_primitives_ready": rotation_primitives_ready,
        "baryon_primitives_ready": baryon_primitives_ready,
        "local_raw_fits_cached": False,
        "machine_readable_survey_native_radial_table_ready": False,
        "machine_readable_same_baryon_radial_table_ready": False,
        "rotation_source_urls": {
            "robust_moment0_fits": moment0_url,
            "robust_moment1_fits": moment1_url,
            "robust_moment2_fits": moment2_url,
            "robust_cube_fits": cube_url,
        },
        "baryon_source_urls": record["baryon_side_family_urls"],
        "local_cache": record["local_cache"],
        "reconstruction_blocker_or_none": "spiral_survey_native_radial_profile_table_absent",
    }


# Function: Return a LITTLE THINGS file URL for the requested cached page and fragment.

def page_url(base_url: str, text: str, fragment: str) -> str | None:
    relative = href(text, fragment)
    if relative is None:
        return None

    return urljoin(base_url, relative)


# Function: Build a selected LITTLE THINGS dwarf reconstruction-source record.

def dwarf_reconstruction_record(record: dict) -> dict:
    page_path = ROOT / record["local_cache"]["little_things_page_html"]
    hi_path = ROOT / record["local_cache"]["little_things_hi_directory_html"]
    page_text = read_text(page_path)
    hi_text = read_text(hi_path)
    hi_base_url = record["rotation_side_urls"]["robust_moment1_fits"].rsplit("/", 1)[0] + "/"

    x0_url = page_url(hi_base_url, hi_text, "_R_X0_P_R.FITS")
    moment1_url = page_url(hi_base_url, hi_text, "_R_XMOM1.FITS")
    moment2_url = page_url(hi_base_url, hi_text, "_R_XMOM2.FITS")
    cube_url = page_url(hi_base_url, hi_text, "_R_ICL001.FITS")
    ubv_url = record["baryon_side_urls"]["ubv_calibration"]
    halpha_url = record["baryon_side_urls"]["halpha_image"]
    fuv_url = record["baryon_side_urls"]["fuv_image"]
    nuv_url = record["baryon_side_urls"]["nuv_image"]
    irac_url = record["baryon_side_urls"]["lvl_irac_directory"]

    rotation_primitives_ready = all(item is not None for item in (x0_url, moment1_url, moment2_url, cube_url))
    baryon_primitives_ready = all(item is not None for item in (ubv_url, halpha_url, fuv_url, nuv_url, irac_url))

    return {
        "name": record["name"],
        "display_name": record["display_name"],
        "survey_family": record["survey_family"],
        "rotation_primitives_ready": rotation_primitives_ready,
        "baryon_primitives_ready": baryon_primitives_ready,
        "local_raw_fits_cached": False,
        "machine_readable_survey_native_radial_table_ready": False,
        "machine_readable_same_baryon_radial_table_ready": False,
        "rotation_source_urls": {
            "robust_x0_p_r_fits": x0_url,
            "robust_moment1_fits": moment1_url,
            "robust_moment2_fits": moment2_url,
            "robust_cube_fits": cube_url,
        },
        "baryon_source_urls": {
            "ubv_calibration": ubv_url,
            "halpha_image": halpha_url,
            "fuv_image": fuv_url,
            "nuv_image": nuv_url,
            "lvl_irac_directory": irac_url,
        },
        "local_cache": record["local_cache"],
        "reconstruction_blocker_or_none": "dwarf_survey_native_radial_profile_table_absent",
    }


# Function: Count the number of records that satisfy the requested boolean field.

def ready_count(records: list[dict], field: str) -> int:
    return sum(1 for record in records if record[field])


# Function: Run the radial-table reconstruction branch and write its artifacts.

def main() -> None:
    for path in (
        PILOT_MANIFEST,
        PREVIOUS_GATE,
        THINGS_PAGE,
        THINGS_MASS_MODELS,
        SINGS_PAGE,
        LITTLE_THINGS_SAMPLE,
        LITTLE_THINGS_DATA,
        LITTLE_THINGS_MASS_MODELS,
    ):
        req(path)

    pilot_manifest = read_json(PILOT_MANIFEST)
    previous_gate = read_json(PREVIOUS_GATE)
    things_html = read_text(THINGS_PAGE)
    things_mass_models_text = read_text(THINGS_MASS_MODELS)
    sings_text = read_text(SINGS_PAGE)
    little_things_sample_text = read_text(LITTLE_THINGS_SAMPLE)
    little_things_data_text = read_text(LITTLE_THINGS_DATA)
    little_things_mass_models_text = read_text(LITTLE_THINGS_MASS_MODELS)

    selected_spiral_records = selected_family_records(pilot_manifest, "spiral_family")
    selected_dwarf_records = selected_family_records(pilot_manifest, "dwarf_family")
    spiral_reconstruction_records = [spiral_reconstruction_record(things_html, record) for record in selected_spiral_records]
    dwarf_reconstruction_records = [dwarf_reconstruction_record(record) for record in selected_dwarf_records]

    spiral_rotation_ready_count = ready_count(spiral_reconstruction_records, "rotation_primitives_ready")
    spiral_baryon_ready_count = ready_count(spiral_reconstruction_records, "baryon_primitives_ready")
    dwarf_rotation_ready_count = ready_count(dwarf_reconstruction_records, "rotation_primitives_ready")
    dwarf_baryon_ready_count = ready_count(dwarf_reconstruction_records, "baryon_primitives_ready")

    spiral_same_baryon_radial_table_ready = all(
        record["machine_readable_same_baryon_radial_table_ready"] for record in spiral_reconstruction_records
    )
    dwarf_same_baryon_radial_table_ready = all(
        record["machine_readable_same_baryon_radial_table_ready"] for record in dwarf_reconstruction_records
    )

    narrowed_blocker = "survey_native_spiral_and_dwarf_radial_profile_tables_absent"
    numeric_comparison_executed = False
    dataset_intake_branch_closeable = False

    reconstruction_manifest = {
        "generated_utc": now_iso(),
        "registry_name": "independent_galaxy_radial_table_reconstruction_manifest",
        "derived_from": {
            "pilot_intake_manifest": rel(PILOT_MANIFEST),
            "previous_dataset_intake_gate": rel(PREVIOUS_GATE),
        },
        "spiral_family": spiral_reconstruction_records,
        "dwarf_family": dwarf_reconstruction_records,
        "comparison_blocker": narrowed_blocker,
        "notes": [
            "Selected THINGS and LITTLE THINGS pilot galaxies expose raw survey primitives, but no machine-readable survey-native radial tables are cached locally yet.",
            "Same-baryon radial-table harmonization remains blocked until survey-native radial profiles are extracted or otherwise frozen into canonical tables.",
        ],
    }
    write_manifest(RECONSTRUCTION_MANIFEST, reconstruction_manifest)

    payloads = {
        "mass_origin_dark_matter_things_spiral_radial_table_source_inventory": payload(
            "8.7.55.3.147",
            "THINGS spiral radial-table source inventory",
            {
                "pilot_intake_manifest_json": rel(PILOT_MANIFEST),
                "things_data_products_html": rel(THINGS_PAGE),
                "things_mass_models_abs_html": rel(THINGS_MASS_MODELS),
                "sings_overview_html": rel(SINGS_PAGE),
            },
            "Inventory the THINGS-side source candidates needed to reconstruct same-baryon spiral radial tables for the selected independent pilot subset.",
            {
                "spiral_inventory_rule": "the source inventory is complete once each selected spiral pilot has its THINGS raw moment-map and cube candidates enumerated alongside the SINGS-side baryon support sources"
            },
            [
                row("things_spiral_source_inventory_complete", "pass", "THINGS spiral radial-table source inventory complete", 1, "The selected spiral pilot subset was inventoried against the THINGS and SINGS cached source pages."),
                row("things_spiral_rotation_primitives_ready_count", "pass" if spiral_rotation_ready_count == len(spiral_reconstruction_records) else "reject", "selected spiral pilots with THINGS rotation primitives ready", spiral_rotation_ready_count, "Each selected spiral pilot should expose robust moment-0/1/2 maps and a robust cube as reconstruction primitives."),
                row("things_spiral_machine_readable_table_ready_count", "reject", "selected spiral pilots with machine-readable same-baryon radial tables ready", ready_count(spiral_reconstruction_records, "machine_readable_same_baryon_radial_table_ready"), "No selected spiral pilot currently exposes a machine-readable same-baryon radial table inside the canonical cache."),
            ],
            {
                "selected_spiral_pilot_subset": [record["name"] for record in spiral_reconstruction_records],
                "things_spiral_source_inventory_ready": True,
                "spiral_rotation_primitives_ready_count": spiral_rotation_ready_count,
                "spiral_baryon_support_ready_count": spiral_baryon_ready_count,
                "spiral_machine_readable_same_baryon_radial_table_ready_count": ready_count(spiral_reconstruction_records, "machine_readable_same_baryon_radial_table_ready"),
                "first_missing_spiral_artifact_or_none": "spiral_survey_native_radial_profile_tables",
            },
            {
                "overall_status": "things_spiral_source_inventory_frozen",
                "spiral_source_inventory_ready": True,
                "next_required_artifacts": [
                    "mass_origin_dark_matter_little_things_dwarf_radial_table_source_inventory",
                    "mass_origin_dark_matter_spiral_same_baryon_radial_table_reconstruction_freeze",
                ],
            },
            {
                "spiral_reconstruction_records": spiral_reconstruction_records,
                "things_mass_models_hit": hit(things_mass_models_text, "We present rotation curves of 19 galaxies from THINGS"),
                "sings_overview_hit": hit(sings_text, "Spitzer/IRAC 3.6"),
            },
        ),
        "mass_origin_dark_matter_little_things_dwarf_radial_table_source_inventory": payload(
            "8.7.55.3.148",
            "LITTLE THINGS dwarf radial-table source inventory",
            {
                "pilot_intake_manifest_json": rel(PILOT_MANIFEST),
                "little_things_sample_html": rel(LITTLE_THINGS_SAMPLE),
                "little_things_pubdata_html": rel(LITTLE_THINGS_DATA),
                "little_things_mass_models_abs_html": rel(LITTLE_THINGS_MASS_MODELS),
                "pilot_reconstruction_manifest_json": rel(RECONSTRUCTION_MANIFEST),
            },
            "Inventory the LITTLE THINGS-side source candidates needed to reconstruct same-baryon dwarf radial tables for the selected independent pilot subset.",
            {
                "dwarf_inventory_rule": "the source inventory is complete once each selected dwarf pilot has its LITTLE THINGS HI raw products and baryonic helper-source candidates enumerated"
            },
            [
                row("little_things_dwarf_source_inventory_complete", "pass", "LITTLE THINGS dwarf radial-table source inventory complete", 1, "The selected dwarf pilot subset was inventoried against the LITTLE THINGS helper pages and HI directory caches."),
                row("little_things_dwarf_rotation_primitives_ready_count", "pass" if dwarf_rotation_ready_count == len(dwarf_reconstruction_records) else "reject", "selected dwarf pilots with LITTLE THINGS rotation primitives ready", dwarf_rotation_ready_count, "Each selected dwarf pilot should expose robust HI cube and moment-map candidates."),
                row("little_things_dwarf_machine_readable_table_ready_count", "reject", "selected dwarf pilots with machine-readable same-baryon radial tables ready", ready_count(dwarf_reconstruction_records, "machine_readable_same_baryon_radial_table_ready"), "No selected dwarf pilot currently exposes a machine-readable same-baryon radial table inside the canonical cache."),
            ],
            {
                "selected_dwarf_pilot_subset": [record["name"] for record in dwarf_reconstruction_records],
                "little_things_dwarf_source_inventory_ready": True,
                "dwarf_rotation_primitives_ready_count": dwarf_rotation_ready_count,
                "dwarf_baryon_support_ready_count": dwarf_baryon_ready_count,
                "dwarf_machine_readable_same_baryon_radial_table_ready_count": ready_count(dwarf_reconstruction_records, "machine_readable_same_baryon_radial_table_ready"),
                "first_missing_dwarf_artifact_or_none": "dwarf_survey_native_radial_profile_tables",
            },
            {
                "overall_status": "little_things_dwarf_source_inventory_frozen",
                "dwarf_source_inventory_ready": True,
                "next_required_artifacts": [
                    "mass_origin_dark_matter_spiral_same_baryon_radial_table_reconstruction_freeze",
                    "mass_origin_dark_matter_dwarf_same_baryon_radial_table_reconstruction_freeze",
                ],
            },
            {
                "dwarf_reconstruction_records": dwarf_reconstruction_records,
                "little_things_sample_hit": hit(little_things_sample_text, "42 dwarf irregular"),
                "little_things_pubdata_hit": hit(little_things_data_text, "public data"),
                "little_things_mass_models_hit": hit(little_things_mass_models_text, "High-Resolution Mass Models of Dwarf Galaxies from LITTLE THINGS"),
            },
        ),
        "mass_origin_dark_matter_spiral_same_baryon_radial_table_reconstruction_freeze": payload(
            "8.7.55.3.149",
            "Spiral same-baryon radial-table reconstruction freeze",
            {
                "pilot_reconstruction_manifest_json": rel(RECONSTRUCTION_MANIFEST),
                "things_data_products_html": rel(THINGS_PAGE),
                "sings_overview_html": rel(SINGS_PAGE),
            },
            "Freeze the current spiral-side same-baryon radial-table reconstruction state for the selected THINGS pilot subset.",
            {
                "spiral_reconstruction_rule": "spiral reconstruction is ready only after the selected THINGS pilot subset has machine-readable survey-native radial tables that can be harmonized onto the SPARC-side baryon interface"
            },
            [
                row("spiral_rotation_primitives_available_for_reconstruction", "pass" if spiral_rotation_ready_count == len(spiral_reconstruction_records) else "reject", "selected spiral pilots with raw rotation primitives available", spiral_rotation_ready_count, "The selected THINGS spiral pilots already expose raw moment-map and cube candidates."),
                row("spiral_baryon_support_available_for_reconstruction", "pass" if spiral_baryon_ready_count == len(spiral_reconstruction_records) else "reject", "selected spiral pilots with baryon support available", spiral_baryon_ready_count, "The selected THINGS spiral pilots already retain the family-level SINGS-side baryon support sources."),
                row("spiral_same_baryon_radial_table_ready", "pass" if spiral_same_baryon_radial_table_ready else "reject", "spiral same-baryon radial-table ready", 1 if spiral_same_baryon_radial_table_ready else 0, "The branch is still blocked because no selected THINGS spiral pilot has a machine-readable survey-native radial table in the canonical cache."),
            ],
            {
                "selected_spiral_pilot_subset": [record["name"] for record in spiral_reconstruction_records],
                "spiral_rotation_primitives_ready": spiral_rotation_ready_count == len(spiral_reconstruction_records),
                "spiral_baryon_support_ready": spiral_baryon_ready_count == len(spiral_reconstruction_records),
                "spiral_same_baryon_radial_table_ready": spiral_same_baryon_radial_table_ready,
                "reconstruction_nonclosure_reason_or_none": "spiral_survey_native_radial_profile_table_absent" if not spiral_same_baryon_radial_table_ready else None,
                "first_route_to_close_or_none": "survey_native_spiral_radial_profile_extraction" if not spiral_same_baryon_radial_table_ready else None,
            },
            {
                "overall_status": "spiral_same_baryon_radial_table_reconstruction_blocked_at_survey_native_profile_extraction",
                "spiral_same_baryon_radial_table_ready": spiral_same_baryon_radial_table_ready,
                "next_required_artifacts": [
                    "mass_origin_dark_matter_dwarf_same_baryon_radial_table_reconstruction_freeze",
                    "mass_origin_dark_matter_independent_direct_kappa_comparison_retry",
                ],
            },
            {
                "spiral_reconstruction_records": spiral_reconstruction_records,
            },
        ),
        "mass_origin_dark_matter_dwarf_same_baryon_radial_table_reconstruction_freeze": payload(
            "8.7.55.3.150",
            "Dwarf same-baryon radial-table reconstruction freeze",
            {
                "pilot_reconstruction_manifest_json": rel(RECONSTRUCTION_MANIFEST),
                "little_things_sample_html": rel(LITTLE_THINGS_SAMPLE),
                "little_things_pubdata_html": rel(LITTLE_THINGS_DATA),
            },
            "Freeze the current dwarf-side same-baryon radial-table reconstruction state for the selected LITTLE THINGS pilot subset.",
            {
                "dwarf_reconstruction_rule": "dwarf reconstruction is ready only after the selected LITTLE THINGS pilot subset has machine-readable survey-native radial tables that can be harmonized onto the SPARC-side baryon interface"
            },
            [
                row("dwarf_rotation_primitives_available_for_reconstruction", "pass" if dwarf_rotation_ready_count == len(dwarf_reconstruction_records) else "reject", "selected dwarf pilots with raw rotation primitives available", dwarf_rotation_ready_count, "The selected LITTLE THINGS dwarf pilots already expose raw HI cube and moment-map candidates."),
                row("dwarf_baryon_support_available_for_reconstruction", "pass" if dwarf_baryon_ready_count == len(dwarf_reconstruction_records) else "reject", "selected dwarf pilots with baryon support available", dwarf_baryon_ready_count, "The selected LITTLE THINGS dwarf pilots already retain the helper-source baryon inputs needed for a same-baryon mapping."),
                row("dwarf_same_baryon_radial_table_ready", "pass" if dwarf_same_baryon_radial_table_ready else "reject", "dwarf same-baryon radial-table ready", 1 if dwarf_same_baryon_radial_table_ready else 0, "The branch is still blocked because no selected LITTLE THINGS dwarf pilot has a machine-readable survey-native radial table in the canonical cache."),
            ],
            {
                "selected_dwarf_pilot_subset": [record["name"] for record in dwarf_reconstruction_records],
                "dwarf_rotation_primitives_ready": dwarf_rotation_ready_count == len(dwarf_reconstruction_records),
                "dwarf_baryon_support_ready": dwarf_baryon_ready_count == len(dwarf_reconstruction_records),
                "dwarf_same_baryon_radial_table_ready": dwarf_same_baryon_radial_table_ready,
                "reconstruction_nonclosure_reason_or_none": "dwarf_survey_native_radial_profile_table_absent" if not dwarf_same_baryon_radial_table_ready else None,
                "first_route_to_close_or_none": "survey_native_dwarf_radial_profile_extraction" if not dwarf_same_baryon_radial_table_ready else None,
            },
            {
                "overall_status": "dwarf_same_baryon_radial_table_reconstruction_blocked_at_survey_native_profile_extraction",
                "dwarf_same_baryon_radial_table_ready": dwarf_same_baryon_radial_table_ready,
                "next_required_artifacts": [
                    "mass_origin_dark_matter_independent_direct_kappa_comparison_retry",
                    "mass_origin_dark_matter_independent_galaxy_dataset_intake_second_gate",
                ],
            },
            {
                "dwarf_reconstruction_records": dwarf_reconstruction_records,
            },
        ),
        "mass_origin_dark_matter_independent_direct_kappa_comparison_retry": payload(
            "8.7.55.3.151",
            "Independent direct-kappa numeric comparison retry",
            {
                "pilot_reconstruction_manifest_json": rel(RECONSTRUCTION_MANIFEST),
                "previous_dataset_intake_gate_json": rel(PREVIOUS_GATE),
            },
            "Retry the independent direct-kappa comparison after freezing the spiral and dwarf radial-table reconstruction state.",
            {
                "comparison_retry_rule": "the retry can execute only after both the spiral and dwarf pilot subsets expose machine-readable same-baryon radial tables"
            },
            [
                row("spiral_same_baryon_radial_table_ready_for_retry", "pass" if spiral_same_baryon_radial_table_ready else "reject", "spiral same-baryon radial tables ready for retry", 1 if spiral_same_baryon_radial_table_ready else 0, "The direct-kappa retry still depends on spiral-side same-baryon radial tables."),
                row("dwarf_same_baryon_radial_table_ready_for_retry", "pass" if dwarf_same_baryon_radial_table_ready else "reject", "dwarf same-baryon radial tables ready for retry", 1 if dwarf_same_baryon_radial_table_ready else 0, "The direct-kappa retry still depends on dwarf-side same-baryon radial tables."),
                row("independent_direct_kappa_numeric_comparison_retry_executed", "pass" if numeric_comparison_executed else "reject", "independent direct-kappa numeric comparison retry executed", 1 if numeric_comparison_executed else 0, "The retry remains blocked because the selected pilots still lack machine-readable survey-native radial tables."),
            ],
            {
                "spiral_same_baryon_radial_table_ready": spiral_same_baryon_radial_table_ready,
                "dwarf_same_baryon_radial_table_ready": dwarf_same_baryon_radial_table_ready,
                "independent_direct_kappa_numeric_comparison_ready": False,
                "independent_direct_kappa_numeric_comparison_executed": numeric_comparison_executed,
                "comparison_blocker_or_none": narrowed_blocker,
                "pass_condition_or_none": None,
            },
            {
                "overall_status": "independent_direct_kappa_retry_blocked_at_survey_native_radial_profile_extraction",
                "independent_direct_kappa_numeric_comparison_ready": False,
                "recommended_next_route_or_none": "8.7.55.3.152",
                "next_required_artifacts": ["mass_origin_dark_matter_independent_galaxy_dataset_intake_second_gate"],
            },
            {
                "spiral_reconstruction_records": spiral_reconstruction_records,
                "dwarf_reconstruction_records": dwarf_reconstruction_records,
            },
        ),
        "mass_origin_dark_matter_independent_galaxy_dataset_intake_second_gate": payload(
            "8.7.55.3.152",
            "Dataset-intake declaration second gate",
            {
                "pilot_reconstruction_manifest_json": rel(RECONSTRUCTION_MANIFEST),
                "previous_dataset_intake_gate_json": rel(PREVIOUS_GATE),
            },
            "Decide whether the independent-galaxy dataset-intake branch can close after the first radial-table reconstruction audit.",
            {
                "second_gate_rule": "the branch can close only after the spiral and dwarf pilot subsets both expose machine-readable same-baryon radial tables and the independent direct-kappa numeric retry has executed"
            },
            [
                row("independent_spiral_reconstruction_inventory_ready", "pass", "independent spiral reconstruction inventory ready", 1, "The selected spiral pilot subset has a frozen source inventory."),
                row("independent_dwarf_reconstruction_inventory_ready", "pass", "independent dwarf reconstruction inventory ready", 1, "The selected dwarf pilot subset has a frozen source inventory."),
                row("independent_dataset_intake_branch_closeable", "pass" if dataset_intake_branch_closeable else "reject", "independent dataset-intake branch closeable", 1 if dataset_intake_branch_closeable else 0, "The branch remains open because machine-readable same-baryon radial tables are still absent."),
            ],
            {
                "spiral_source_inventory_ready": True,
                "dwarf_source_inventory_ready": True,
                "spiral_same_baryon_radial_table_ready": spiral_same_baryon_radial_table_ready,
                "dwarf_same_baryon_radial_table_ready": dwarf_same_baryon_radial_table_ready,
                "independent_direct_kappa_numeric_comparison_executed": numeric_comparison_executed,
                "dataset_intake_branch_closeable": dataset_intake_branch_closeable,
                "recommended_next_route_or_none": "8.7.55.3.153",
                "selected_next_route": "independent_galaxy_survey_native_radial_profile_extraction",
            },
            {
                "overall_status": "independent_dataset_intake_blocked_at_survey_native_radial_profile_extraction",
                "dataset_intake_branch_closeable": dataset_intake_branch_closeable,
                "recommended_next_route_or_none": "8.7.55.3.153",
                "next_required_artifacts": ["mass_origin_dark_matter_survey_native_radial_profile_route_contract"],
            },
            {
                "previous_dataset_intake_gate": previous_gate["summary"],
                "reconstruction_manifest": reconstruction_manifest,
            },
        ),
        "mass_origin_dark_matter_survey_native_radial_profile_route_contract": payload(
            "8.7.55.3.153",
            "Survey-native radial-profile route contract",
            {
                "pilot_reconstruction_manifest_json": rel(RECONSTRUCTION_MANIFEST),
                "dataset_intake_second_gate_json": "output/public/quantum/mass_origin_dark_matter_independent_galaxy_dataset_intake_second_gate_metrics.json",
            },
            "Freeze the next residual route after the first radial-table reconstruction branch fails to close the independent-galaxy intake branch.",
            {
                "route_contract_rule": "the next residual route must target the first missing machine-readable artifact that blocks both the spiral and dwarf pilot subsets at once"
            },
            [
                row("survey_native_radial_profile_route_selected", "pass", "survey-native radial-profile route selected", 1, "The next residual route is now frozen."),
                row("survey_native_radial_profile_common_blocker_present", "pass", "common spiral+dwarf blocker present", 1, "Both selected pilot families are blocked by the absence of survey-native radial-profile tables."),
                row("survey_native_radial_profile_split_contract_ready", "pass", "survey-native radial-profile split contract ready", 1, "The next branch can now decompose the spiral and dwarf extraction work without ambiguity."),
            ],
            {
                "selected_residual_route": "independent_galaxy_survey_native_radial_profile_extraction",
                "missing_dark_matter_artifact": "survey_native_spiral_and_dwarf_radial_profile_tables",
                "split_contract_ready": True,
                "defer_v2_mainline_until_route_close": True,
            },
            {
                "overall_status": "independent_galaxy_survey_native_radial_profile_route_frozen",
                "next_required_artifacts": [
                    "8.7.55.3.154",
                    "8.7.55.3.155",
                    "8.7.55.3.156",
                    "8.7.55.3.157",
                    "8.7.55.3.158",
                ],
            },
            {
                "spiral_reconstruction_records": spiral_reconstruction_records,
                "dwarf_reconstruction_records": dwarf_reconstruction_records,
                "common_blocker": narrowed_blocker,
            },
        ),
    }

    for stem, data in payloads.items():
        write_artifact(stem, data)
        print(f"[ok] wrote {OUT / (stem + '_metrics.json')}")
        print(f"[ok] wrote {OUT / (stem + '_rows.csv')}")

    print(f"[ok] wrote {RECONSTRUCTION_MANIFEST}")


# Function: Run the radial-table reconstruction branch when invoked as a script.

if __name__ == "__main__":
    main()
