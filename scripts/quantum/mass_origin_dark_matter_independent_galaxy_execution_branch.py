#!/usr/bin/env python3
"""
Freeze the independent-galaxy dataset-intake execution branch for 8.7.55.3.141-.146.

This branch executes the first non-SPARC intake contract after the direct
dark-matter-elimination bridge has already closed. Its job is to:

1. sync an intake-ready raw-source manifest for independent spiral and dwarf
   pilot families,
2. freeze the same-baryon preprocessing contract that reuses the SPARC-side
   interface,
3. pass spiral and dwarf pilot subsets through an independence-aware quality
   gate, and
4. decide whether an actual direct-kappa numeric pilot can launch now or must
   hand off to a radial-table reconstruction branch.
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
PRIMARY_SOURCES = ROOT / "doc" / "PRIMARY_SOURCES.md"
PART2 = ROOT / "doc" / "paper" / "11_part2_astrophysics.md"
SPARC_AUDIT = ROOT / "output" / "public" / "cosmology" / "sparc_rotation_curve_pmodel_audit_metrics.json"
SPARC_GALAXY_SUMMARY = ROOT / "output" / "public" / "cosmology" / "sparc_rotation_curve_pmodel_audit_galaxy_summary.csv"
PREVIOUS_GATE = OUT / "mass_origin_dark_matter_independent_galaxy_dataset_intake_kickoff_gate_metrics.json"
REGISTRY_MANIFEST = REGISTRY_DIR / "manifest.json"
THINGS_PAGE = REGISTRY_DIR / "things_data_products_20260320.html"
SINGS_PAGE = REGISTRY_DIR / "sings_overview_20260320.html"
LITTLE_THINGS_SAMPLE = REGISTRY_DIR / "little_things_sample_20260320.html"
INTAKE_MANIFEST = REGISTRY_DIR / "pilot_intake_manifest.json"

SPIRAL_PILOTS = [
    {"name": "NGC_3031", "display_name": "NGC 3031", "survey_family": "THINGS+SINGS"},
    {"name": "NGC_3621", "display_name": "NGC 3621", "survey_family": "THINGS+SINGS"},
    {"name": "NGC_5194", "display_name": "NGC 5194", "survey_family": "THINGS+SINGS"},
    {"name": "NGC_628", "display_name": "NGC 628", "survey_family": "THINGS+SINGS"},
    {"name": "NGC_925", "display_name": "NGC 925", "survey_family": "THINGS+SINGS"},
]

DWARF_PILOTS = [
    {
        "name": "DDO154",
        "display_name": "DDO 154",
        "page_path": REGISTRY_DIR / "d154_20260320.html",
        "page_url": "https://www2.lowell.edu/users/dah/littlethings/data/d154.html",
        "hi_page_path": REGISTRY_DIR / "ddo154_hiobs_20260320.html",
        "hi_base_url": "http://things.cv.nrao.edu/littlethings/ddo154/HI/",
        "hi_prefix": "DDO154",
        "halpha_pattern": "d154ha.fits",
        "fuv_pattern": "d154fcut.fit",
        "nuv_pattern": "d154ncut.fit",
    },
    {
        "name": "IC1613",
        "display_name": "IC 1613",
        "page_path": REGISTRY_DIR / "ic1613_20260320.html",
        "page_url": "https://www2.lowell.edu/users/dah/littlethings/data/ic1613.html",
        "hi_page_path": REGISTRY_DIR / "ic1613_hiobs_20260320.html",
        "hi_base_url": "http://things.cv.nrao.edu/littlethings/ic1613/HI/",
        "hi_prefix": "IC1613",
        "halpha_pattern": "ic1613hmrms.fits",
        "fuv_pattern": "ic1613fcut.fit",
        "nuv_pattern": "ic1613ncut.fit",
    },
    {
        "name": "WLM",
        "display_name": "WLM",
        "page_path": REGISTRY_DIR / "wlm_20260320.html",
        "page_url": "https://www2.lowell.edu/users/dah/littlethings/data/wlm.html",
        "hi_page_path": REGISTRY_DIR / "wlm_hiobs_20260320.html",
        "hi_base_url": "http://things.cv.nrao.edu/littlethings/wlm/HI/",
        "hi_prefix": "WLM",
        "halpha_pattern": "wlmha.fits",
        "fuv_pattern": "wlmfcut.fit",
        "nuv_pattern": "wlmncut.fit",
    },
]


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


# Function: Return the first line hit for a literal substring search.

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


# Function: Save the canonical intake manifest for the independent-galaxy pilot.

def write_manifest(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


# Function: Extract a matching href from cached HTML, if present.

def href(text: str, fragment: str) -> str | None:
    match = re.search(rf'href="([^"]*{re.escape(fragment)}[^"]*)"', text, flags=re.IGNORECASE)
    if not match:
        return None

    return match.group(1)


# Function: Normalize a galaxy label for overlap checks against SPARC.

def normalize_galaxy_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", name).upper()


# Function: Load the SPARC galaxy-name set from the public audit summary.

def read_sparc_names(path: Path) -> set[str]:
    names: set[str] = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        for entry in csv.DictReader(handle):
            names.add(normalize_galaxy_name(entry["galaxy"]))

    return names


# Function: Build a THINGS spiral pilot record from the cached data-products page.

def spiral_record(things_html: str, sparc_names: set[str], entry: dict, sings_ready: bool) -> dict:
    raw_mom1_rel = href(things_html, f"Data_files/{entry['name']}_RO_MOM1_THINGS.FITS")
    raw_cube_rel = href(things_html, f"Data_files/{entry['name']}_RO_CUBE_THINGS.FITS")
    overlap_with_sparc = normalize_galaxy_name(entry["name"]) in sparc_names

    return {
        "name": entry["name"],
        "display_name": entry["display_name"],
        "survey_family": entry["survey_family"],
        "overlaps_sparc": overlap_with_sparc,
        "selected_for_independent_pilot": (raw_mom1_rel is not None and raw_cube_rel is not None and not overlap_with_sparc),
        "rotation_side_ready": raw_mom1_rel is not None and raw_cube_rel is not None,
        "baryon_side_family_ready": sings_ready,
        "machine_readable_same_baryon_radial_table_ready": False,
        "rotation_side_urls": {
            "robust_moment1_fits": urljoin("https://www2.mpia-hd.mpg.de/THINGS/Data.html", raw_mom1_rel) if raw_mom1_rel else None,
            "robust_cube_fits": urljoin("https://www2.mpia-hd.mpg.de/THINGS/Data.html", raw_cube_rel) if raw_cube_rel else None,
        },
        "baryon_side_family_urls": {
            "things_mass_models": "https://arxiv.org/abs/0810.2100",
            "sings_overview": "https://irsa.ipac.caltech.edu/data/SPITZER/SINGS/overview.html",
        },
        "local_cache": {
            "things_data_products_html": rel(THINGS_PAGE),
            "sings_overview_html": rel(SINGS_PAGE),
        },
        "exclusion_reason_or_none": "overlaps_sparc_audit_sample" if overlap_with_sparc else None,
    }


# Function: Build a LITTLE THINGS dwarf pilot record from cached per-galaxy helper pages.

def dwarf_record(sparc_names: set[str], entry: dict) -> dict:
    page_text = read_text(entry["page_path"])
    hi_text = read_text(entry["hi_page_path"])

    ubv_rel = href(page_text, "_ubvcalib.txt")
    halpha_rel = href(page_text, entry["halpha_pattern"])
    fuv_rel = href(page_text, entry["fuv_pattern"])
    nuv_rel = href(page_text, entry["nuv_pattern"])
    irac_url = href(page_text, "lvl/20090731_enhanced/irac/")
    raw_mom1_rel = href(hi_text, f"{entry['hi_prefix']}_R_XMOM1.FITS")
    raw_cube_rel = href(hi_text, f"{entry['hi_prefix']}_R_ICL001.FITS")
    overlap_with_sparc = normalize_galaxy_name(entry["name"]) in sparc_names

    baryon_ready = all(item is not None for item in (ubv_rel, halpha_rel, fuv_rel, nuv_rel, irac_url))
    rotation_ready = raw_mom1_rel is not None and raw_cube_rel is not None

    return {
        "name": entry["name"],
        "display_name": entry["display_name"],
        "survey_family": "LITTLE THINGS",
        "overlaps_sparc": overlap_with_sparc,
        "selected_for_independent_pilot": rotation_ready and baryon_ready and not overlap_with_sparc,
        "rotation_side_ready": rotation_ready,
        "baryon_side_ready": baryon_ready,
        "machine_readable_same_baryon_radial_table_ready": False,
        "rotation_side_urls": {
            "robust_moment1_fits": urljoin(entry["hi_base_url"], raw_mom1_rel) if raw_mom1_rel else None,
            "robust_cube_fits": urljoin(entry["hi_base_url"], raw_cube_rel) if raw_cube_rel else None,
        },
        "baryon_side_urls": {
            "ubv_calibration": urljoin(entry["page_url"], ubv_rel) if ubv_rel else None,
            "halpha_image": urljoin(entry["page_url"], halpha_rel) if halpha_rel else None,
            "fuv_image": urljoin(entry["page_url"], fuv_rel) if fuv_rel else None,
            "nuv_image": urljoin(entry["page_url"], nuv_rel) if nuv_rel else None,
            "lvl_irac_directory": irac_url,
        },
        "local_cache": {
            "little_things_page_html": rel(entry["page_path"]),
            "little_things_hi_directory_html": rel(entry["hi_page_path"]),
        },
        "exclusion_reason_or_none": "overlaps_sparc_audit_sample" if overlap_with_sparc else None,
    }


# Function: Return the selected pilot names from a family record list.

def selected_names(records: list[dict]) -> list[str]:
    return [record["name"] for record in records if record["selected_for_independent_pilot"]]


# Function: Return the excluded overlap names from a family record list.

def excluded_overlap_names(records: list[dict]) -> list[str]:
    return [record["name"] for record in records if record["overlaps_sparc"]]


# Function: Run the independent-galaxy dataset-intake execution branch and write artifacts.

def main() -> None:
    for path in (
        PRIMARY_SOURCES,
        PART2,
        SPARC_AUDIT,
        SPARC_GALAXY_SUMMARY,
        PREVIOUS_GATE,
        REGISTRY_MANIFEST,
        THINGS_PAGE,
        SINGS_PAGE,
        LITTLE_THINGS_SAMPLE,
    ):
        req(path)

    for entry in DWARF_PILOTS:
        req(entry["page_path"])
        req(entry["hi_page_path"])

    primary_sources_text = read_text(PRIMARY_SOURCES)
    part2_text = read_text(PART2)
    sparc_audit = read_json(SPARC_AUDIT)
    previous_gate = read_json(PREVIOUS_GATE)
    registry_manifest = read_json(REGISTRY_MANIFEST)
    things_html = read_text(THINGS_PAGE)
    little_things_sample_text = read_text(LITTLE_THINGS_SAMPLE)
    sparc_names = read_sparc_names(SPARC_GALAXY_SUMMARY)

    same_baryon_formula = r"V_{\rm bar}^2(R)=V_{\rm gas}^2+\Upsilon V_{\rm disk}^2+(f_{\rm bulge}\Upsilon)V_{\rm bulge}^2"
    part2_formula_hit = hit(part2_text, same_baryon_formula)
    primary_sources_section_hit = hit(primary_sources_text, "8.1) SPARC 以外の公開 rotation-curve sample")
    previous_launch_ready = bool(previous_gate["summary"]["launch_dataset_intake_now"])
    sings_text = read_text(SINGS_PAGE)
    sings_ready = href(sings_text, "galaxies/") is not None and hit(sings_text, "Spitzer/IRAC 3.6") is not None

    spiral_records = [spiral_record(things_html, sparc_names, entry, sings_ready) for entry in SPIRAL_PILOTS]
    dwarf_records = [dwarf_record(sparc_names, entry) for entry in DWARF_PILOTS]

    spiral_selected = selected_names(spiral_records)
    dwarf_selected = selected_names(dwarf_records)
    spiral_excluded = excluded_overlap_names(spiral_records)
    dwarf_excluded = excluded_overlap_names(dwarf_records)

    raw_manifest_sync_ready = previous_launch_ready and bool(spiral_records) and bool(dwarf_records)
    same_baryon_preprocessing_ready = (
        previous_launch_ready
        and part2_formula_hit is not None
        and abs(float(sparc_audit["inputs"]["bulge_to_disk_ml_ratio"]) - 1.4) < 1e-12
    )
    spiral_subset_ready = len(spiral_selected) >= 3
    dwarf_subset_ready = len(dwarf_selected) >= 2

    comparison_blocker = "machine_readable_same_baryon_radial_tables_absent"
    direct_kappa_target = float(sparc_audit["inputs"]["pbg_kappa"])
    direct_a0_target = float(sparc_audit["pmodel_fixed"]["a0_m_s2"])
    numeric_comparison_ready = False
    numeric_comparison_executed = False

    intake_manifest = {
        "generated_utc": now_iso(),
        "registry_name": "independent_galaxy_dataset_intake_manifest",
        "derived_from": {
            "registry_manifest": rel(REGISTRY_MANIFEST),
            "previous_kickoff_gate": rel(PREVIOUS_GATE),
            "sparc_reference_audit": rel(SPARC_AUDIT),
        },
        "same_baryon_interface_formula": same_baryon_formula,
        "same_baryon_preprocessing": {
            "bulge_to_disk_ml_ratio": float(sparc_audit["inputs"]["bulge_to_disk_ml_ratio"]),
            "direct_kappa_target": direct_kappa_target,
            "direct_a0_target_m_s2": direct_a0_target,
            "harmonization_rule": "convert survey-native radial profiles onto the SPARC-side same-baryon interface before any independent direct-kappa comparison",
        },
        "spiral_family": spiral_records,
        "dwarf_family": dwarf_records,
        "selected_independent_pilot_subset": {
            "spiral_family": spiral_selected,
            "dwarf_family": dwarf_selected,
        },
        "excluded_due_to_sparc_overlap": {
            "spiral_family": spiral_excluded,
            "dwarf_family": dwarf_excluded,
        },
        "numeric_comparison_blocker": comparison_blocker,
        "notes": [
            "THINGS and LITTLE THINGS raw URLs are synced here as intake-ready inputs, but harmonized radial tables are still absent.",
            "DDO154 remains cached as a helper page but is excluded from the independent dwarf pilot because it overlaps the SPARC audit sample.",
        ],
    }
    write_manifest(INTAKE_MANIFEST, intake_manifest)

    payloads = {
        "mass_origin_dark_matter_independent_galaxy_raw_source_manifest_sync": payload(
            "8.7.55.3.141",
            "Independent-galaxy raw-source fetch / manifest sync freeze",
            {
                "registry_manifest_json": rel(REGISTRY_MANIFEST),
                "things_data_products_html": rel(THINGS_PAGE),
                "little_things_sample_html": rel(LITTLE_THINGS_SAMPLE),
                "previous_kickoff_gate_json": rel(PREVIOUS_GATE),
            },
            "Sync the THINGS and LITTLE THINGS raw-source side into an intake-ready manifest for the first truly independent spiral and dwarf pilot subsets.",
            {
                "manifest_sync_rule": "the raw-source manifest is ready only if the preparation gate is already open and both spiral and dwarf families have cached raw-source URLs that can be written into a canonical intake manifest"
            },
            [
                row("independent_raw_source_manifest_written", "pass" if raw_manifest_sync_ready else "reject", "independent-galaxy raw-source manifest written", 1 if raw_manifest_sync_ready else 0, "The execution branch must first freeze a canonical manifest for the raw THINGS and LITTLE THINGS intake inputs."),
                row("spiral_raw_candidate_count", "pass" if spiral_records else "reject", "spiral raw candidate count", len(spiral_records), "The spiral intake family is defined by THINGS raw products plus the SINGS-side baryon pack family."),
                row("dwarf_raw_candidate_count", "pass" if dwarf_records else "reject", "dwarf raw candidate count", len(dwarf_records), "The dwarf intake family is defined by LITTLE THINGS HI raw products plus the per-galaxy baryonic helper pages."),
            ],
            {
                "previous_launch_ready": previous_launch_ready,
                "raw_source_manifest_sync_ready": raw_manifest_sync_ready,
                "pilot_intake_manifest_path": rel(INTAKE_MANIFEST),
                "spiral_candidate_names": [record["name"] for record in spiral_records],
                "dwarf_candidate_names": [record["name"] for record in dwarf_records],
            },
            {
                "overall_status": "independent_raw_source_manifest_frozen" if raw_manifest_sync_ready else "independent_raw_source_manifest_blocked",
                "raw_source_manifest_sync_ready": raw_manifest_sync_ready,
                "next_required_artifacts": [
                    "mass_origin_dark_matter_independent_galaxy_same_baryon_preprocessing_freeze",
                    "mass_origin_dark_matter_spiral_family_pilot_subset_quality_gate",
                    "mass_origin_dark_matter_dwarf_family_pilot_subset_quality_gate",
                ],
            },
            {
                "intake_manifest": intake_manifest,
                "primary_sources_section_hit": primary_sources_section_hit,
                "registry_entries": registry_manifest["entries"],
            },
        ),
        "mass_origin_dark_matter_independent_galaxy_same_baryon_preprocessing_freeze": payload(
            "8.7.55.3.142",
            "Same-baryon preprocessing freeze",
            {
                "part2_markdown": rel(PART2),
                "sparc_rotation_curve_audit_json": rel(SPARC_AUDIT),
                "pilot_intake_manifest_json": rel(INTAKE_MANIFEST),
            },
            "Freeze the preprocessing contract that lets the independent galaxies reuse the SPARC-side baryon interface and direct-kappa target without retuning.",
            {
                "same_baryon_preprocessing_rule": "the preprocessing contract is ready only if Part II still fixes the baryon-interface formula and the SPARC-side operational audit still fixes kappa=1/(2pi) with bulge_to_disk_ml_ratio=1.4"
            },
            [
                row("same_baryon_formula_available", "pass" if part2_formula_hit else "reject", "same-baryon formula available", 1 if part2_formula_hit else 0, "Part II remains the canonical source for the baryon-interface formula."),
                row("sparc_bulge_factor_available", "pass" if abs(float(sparc_audit["inputs"]["bulge_to_disk_ml_ratio"]) - 1.4) < 1e-12 else "reject", "SPARC bulge-to-disk mass-to-light factor available", 1 if abs(float(sparc_audit["inputs"]["bulge_to_disk_ml_ratio"]) - 1.4) < 1e-12 else 0, "The independent sample must retain the same bulge factor used by the SPARC-side direct-kappa audit."),
                row("same_baryon_preprocessing_ready", "pass" if same_baryon_preprocessing_ready else "reject", "same-baryon preprocessing ready", 1 if same_baryon_preprocessing_ready else 0, "The independent preprocessing contract is ready only if the same formula, bulge factor, and kappa target remain frozen."),
            ],
            {
                "same_baryon_preprocessing_ready": same_baryon_preprocessing_ready,
                "same_baryon_interface_formula": same_baryon_formula,
                "bulge_to_disk_ml_ratio": float(sparc_audit["inputs"]["bulge_to_disk_ml_ratio"]),
                "direct_kappa_target": direct_kappa_target,
                "direct_a0_target_m_s2": direct_a0_target,
                "harmonization_rule": intake_manifest["same_baryon_preprocessing"]["harmonization_rule"],
            },
            {
                "overall_status": "independent_same_baryon_preprocessing_frozen",
                "same_baryon_preprocessing_ready": same_baryon_preprocessing_ready,
                "next_required_artifacts": [
                    "mass_origin_dark_matter_spiral_family_pilot_subset_quality_gate",
                    "mass_origin_dark_matter_dwarf_family_pilot_subset_quality_gate",
                ],
            },
            {
                "part2_formula_hit": part2_formula_hit,
                "sparc_operational_inputs": sparc_audit["inputs"],
                "sparc_operational_fixed": sparc_audit["pmodel_fixed"],
            },
        ),
        "mass_origin_dark_matter_spiral_family_pilot_subset_quality_gate": payload(
            "8.7.55.3.143",
            "Spiral-family pilot subset quality gate",
            {
                "things_data_products_html": rel(THINGS_PAGE),
                "sings_overview_html": rel(SINGS_PAGE),
                "sparc_galaxy_summary_csv": rel(SPARC_GALAXY_SUMMARY),
                "pilot_intake_manifest_json": rel(INTAKE_MANIFEST),
            },
            "Choose the first independent spiral-family pilot subset from THINGS entries that do not overlap the SPARC audit sample.",
            {
                "spiral_quality_rule": "a spiral candidate passes only if the THINGS raw rotation products are cached, the SINGS-side baryon pack family stays available, and the galaxy does not overlap the SPARC audit sample"
            },
            [
                row("spiral_family_candidates_present", "pass" if spiral_records else "reject", "spiral-family candidates present", len(spiral_records), "The THINGS-side intake family must provide explicit raw candidates before a spiral pilot subset can be frozen."),
                row("spiral_family_selected_count", "pass" if spiral_subset_ready else "reject", "selected independent spiral pilot count", len(spiral_selected), "The first spiral pilot subset should contain at least three non-SPARC galaxies with cached THINGS rotation products."),
                row("spiral_family_excluded_overlap_count", "pass", "excluded spiral overlap count", len(spiral_excluded), "SPARC-overlap galaxies are explicitly removed from the independent pilot subset."),
            ],
            {
                "spiral_family_pilot_subset_ready": spiral_subset_ready,
                "selected_spiral_pilot_subset": spiral_selected,
                "excluded_spiral_overlap_subset": spiral_excluded,
                "remaining_numeric_blocker_or_none": comparison_blocker if spiral_subset_ready else "independent_spiral_subset_not_ready",
            },
            {
                "overall_status": "spiral_pilot_subset_frozen" if spiral_subset_ready else "spiral_pilot_subset_blocked",
                "spiral_family_pilot_subset_ready": spiral_subset_ready,
                "next_required_artifacts": ["mass_origin_dark_matter_independent_direct_kappa_comparison_pilot"],
            },
            {
                "spiral_records": spiral_records,
                "sings_overview_hit": hit(sings_text, "Spitzer/IRAC 3.6"),
            },
        ),
        "mass_origin_dark_matter_dwarf_family_pilot_subset_quality_gate": payload(
            "8.7.55.3.144",
            "Dwarf-family pilot subset quality gate",
            {
                "little_things_sample_html": rel(LITTLE_THINGS_SAMPLE),
                "pilot_intake_manifest_json": rel(INTAKE_MANIFEST),
                "sparc_galaxy_summary_csv": rel(SPARC_GALAXY_SUMMARY),
            },
            "Choose the first independent dwarf-family pilot subset from LITTLE THINGS helper pages that do not overlap the SPARC audit sample.",
            {
                "dwarf_quality_rule": "a dwarf candidate passes only if the LITTLE THINGS raw HI products and baryonic helper links are cached and the galaxy does not overlap the SPARC audit sample"
            },
            [
                row("dwarf_family_candidates_present", "pass" if dwarf_records else "reject", "dwarf-family candidates present", len(dwarf_records), "The LITTLE THINGS-side intake family must provide explicit helper pages before a dwarf pilot subset can be frozen."),
                row("dwarf_family_selected_count", "pass" if dwarf_subset_ready else "reject", "selected independent dwarf pilot count", len(dwarf_selected), "The first dwarf pilot subset should contain at least two non-SPARC galaxies with cached HI and baryonic helper products."),
                row("dwarf_family_excluded_overlap_count", "pass", "excluded dwarf overlap count", len(dwarf_excluded), "SPARC-overlap dwarfs are explicitly removed from the independent pilot subset."),
            ],
            {
                "dwarf_family_pilot_subset_ready": dwarf_subset_ready,
                "selected_dwarf_pilot_subset": dwarf_selected,
                "excluded_dwarf_overlap_subset": dwarf_excluded,
                "remaining_numeric_blocker_or_none": comparison_blocker if dwarf_subset_ready else "independent_dwarf_subset_not_ready",
            },
            {
                "overall_status": "dwarf_pilot_subset_frozen" if dwarf_subset_ready else "dwarf_pilot_subset_blocked",
                "dwarf_family_pilot_subset_ready": dwarf_subset_ready,
                "next_required_artifacts": ["mass_origin_dark_matter_independent_direct_kappa_comparison_pilot"],
            },
            {
                "dwarf_records": dwarf_records,
                "little_things_sample_hit": hit(little_things_sample_text, "42 dwarf irregular"),
            },
        ),
        "mass_origin_dark_matter_independent_direct_kappa_comparison_pilot": payload(
            "8.7.55.3.145",
            "Independent direct-kappa comparison pilot",
            {
                "pilot_intake_manifest_json": rel(INTAKE_MANIFEST),
                "sparc_rotation_curve_audit_json": rel(SPARC_AUDIT),
            },
            "Attempt the first independent direct-kappa comparison using the selected spiral and dwarf pilot subsets.",
            {
                "comparison_rule": "the independent direct-kappa comparison can execute only after each selected pilot galaxy has a machine-readable rotation curve table and a machine-readable baryonic decomposition table on the same baryon interface"
            },
            [
                row("independent_spiral_pilot_ready", "pass" if spiral_subset_ready else "reject", "independent spiral pilot subset ready", 1 if spiral_subset_ready else 0, "The direct-kappa pilot depends on the spiral-family quality gate."),
                row("independent_dwarf_pilot_ready", "pass" if dwarf_subset_ready else "reject", "independent dwarf pilot subset ready", 1 if dwarf_subset_ready else 0, "The direct-kappa pilot depends on the dwarf-family quality gate."),
                row("independent_direct_kappa_numeric_comparison_executed", "pass" if numeric_comparison_executed else "reject", "independent direct-kappa numeric comparison executed", 1 if numeric_comparison_executed else 0, "The current intake contract still lacks harmonized radial tables, so the numeric comparison cannot run yet."),
            ],
            {
                "direct_kappa_target": direct_kappa_target,
                "direct_a0_target_m_s2": direct_a0_target,
                "selected_spiral_pilot_subset": spiral_selected,
                "selected_dwarf_pilot_subset": dwarf_selected,
                "independent_direct_kappa_numeric_comparison_ready": numeric_comparison_ready,
                "independent_direct_kappa_numeric_comparison_executed": numeric_comparison_executed,
                "comparison_blocker_or_none": comparison_blocker,
                "pass_condition_or_none": None,
            },
            {
                "overall_status": "independent_direct_kappa_numeric_pilot_blocked_at_radial_table_reconstruction",
                "independent_direct_kappa_numeric_comparison_ready": numeric_comparison_ready,
                "recommended_next_route_or_none": "8.7.55.3.147",
                "next_required_artifacts": [
                    "spiral_same_baryon_radial_table_reconstruction",
                    "dwarf_same_baryon_radial_table_reconstruction",
                ],
            },
            {
                "spiral_records": spiral_records,
                "dwarf_records": dwarf_records,
                "comparison_blocker": comparison_blocker,
            },
        ),
        "mass_origin_dark_matter_independent_galaxy_dataset_intake_declaration_gate": payload(
            "8.7.55.3.146",
            "Dataset-intake declaration gate / share-pack refresh",
            {
                "previous_kickoff_gate_json": rel(PREVIOUS_GATE),
                "pilot_intake_manifest_json": rel(INTAKE_MANIFEST),
                "direct_kappa_comparison_pilot": "mass_origin_dark_matter_independent_direct_kappa_comparison_pilot_metrics.json",
            },
            "Close the first intake-execution branch by deciding whether the independent-galaxy direct-kappa route can declare a numeric pilot now or must hand off to radial-table reconstruction.",
            {
                "declaration_gate_rule": "the branch is closeable only if the intake manifest is frozen, both pilot families are ready, and the first independent direct-kappa numeric comparison has actually executed"
            },
            [
                row("independent_dataset_intake_manifest_ready", "pass" if raw_manifest_sync_ready else "reject", "independent dataset-intake manifest ready", 1 if raw_manifest_sync_ready else 0, "The execution branch first needs an intake-ready manifest."),
                row("independent_dataset_intake_family_gates_ready", "pass" if spiral_subset_ready and dwarf_subset_ready else "reject", "independent dataset-intake family gates ready", 1 if spiral_subset_ready and dwarf_subset_ready else 0, "Both spiral and dwarf pilot families must pass their quality gates before the declaration gate can close."),
                row("independent_dataset_intake_declaration_ready", "pass" if numeric_comparison_executed else "reject", "independent dataset-intake declaration ready", 1 if numeric_comparison_executed else 0, "The declaration gate cannot close before the first numeric direct-kappa comparison exists."),
            ],
            {
                "external_share_ready_still_available": bool(previous_gate["summary"]["external_share_ready"]),
                "independent_dataset_intake_manifest_ready": raw_manifest_sync_ready,
                "spiral_family_pilot_subset_ready": spiral_subset_ready,
                "dwarf_family_pilot_subset_ready": dwarf_subset_ready,
                "direct_kappa_numeric_pilot_executed": numeric_comparison_executed,
                "dataset_intake_declaration_ready": numeric_comparison_executed,
                "launch_numeric_comparison_now": numeric_comparison_executed,
                "share_pack_refresh_required_now": False,
                "recommended_next_route_or_none": "8.7.55.3.147",
                "selected_next_route": "independent_galaxy_radial_table_reconstruction",
            },
            {
                "overall_status": "independent_dataset_intake_manifest_ready_numeric_pilot_pending",
                "dataset_intake_branch_closeable": False,
                "external_share_ready": bool(previous_gate["summary"]["external_share_ready"]),
                "recommended_next_route_or_none": "8.7.55.3.147",
                "next_required_artifacts": [
                    "independent_spiral_radial_table_reconstruction",
                    "independent_dwarf_radial_table_reconstruction",
                    "independent_direct_kappa_numeric_pilot_refresh",
                ],
            },
            {
                "previous_kickoff_gate": previous_gate["summary"],
                "comparison_blocker": comparison_blocker,
                "pilot_intake_manifest": intake_manifest,
            },
        ),
    }

    for stem, data in payloads.items():
        write_artifact(stem, data)
        print(f"[ok] wrote {OUT / (stem + '_metrics.json')}")
        print(f"[ok] wrote {OUT / (stem + '_rows.csv')}")

    print(f"[ok] wrote {INTAKE_MANIFEST}")


# Function: Run the independent-galaxy dataset-intake execution branch when invoked as a script.

if __name__ == "__main__":
    main()
