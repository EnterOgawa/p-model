#!/usr/bin/env python3
"""
Freeze the survey-native radial-profile extraction branch for 8.7.55.3.154-.159.

This branch follows the first radial-table reconstruction audit, which already
froze the THINGS spiral and LITTLE THINGS dwarf pilot subsets as source-ready
but still lacking machine-readable survey-native radial-profile tables.

Its job is to:

1. inventory the THINGS-side extraction inputs needed to turn remote raw FITS
   products into canonical survey-native radial-profile tables,
2. inventory the LITTLE THINGS-side extraction inputs needed for the same,
3. freeze the current spiral and dwarf extraction contracts,
4. decide whether a reopened independent direct-kappa comparison can launch
   now, and
5. formalize the next residual route if the branch still cannot close.
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
RAW_CACHE_ROOT = NON_SPARC_DIR / "survey_native_raw"
PILOT_RECONSTRUCTION_MANIFEST = NON_SPARC_DIR / "pilot_reconstruction_manifest.json"
PREVIOUS_ROUTE_CONTRACT = OUT / "mass_origin_dark_matter_survey_native_radial_profile_route_contract_metrics.json"
THINGS_PAGE = REGISTRY_DIR / "things_data_products_20260320.html"
THINGS_SURVEY = REGISTRY_DIR / "things_survey_0810.2125_abs.html"
THINGS_MASS_MODELS = REGISTRY_DIR / "things_mass_models_0810.2100_abs.html"
LITTLE_THINGS_DATA = REGISTRY_DIR / "little_things_pubdata_20260320.html"
LITTLE_THINGS_SURVEY = REGISTRY_DIR / "little_things_survey_1208.5834_abs.html"
LITTLE_THINGS_MASS_MODELS = REGISTRY_DIR / "little_things_mass_models_1502.01281_abs.html"
EXTRACTION_MANIFEST = NON_SPARC_DIR / "pilot_survey_native_extraction_manifest.json"


# 関数: 現在UTCを ISO 8601 形式で返す。
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: 必須入力が無い場合は即座に終了する。

def req(path: Path) -> None:
    if not path.exists():
        raise SystemExit(f"[fail] missing required input: {path}")


# 関数: UTF-8 JSON を辞書として読む。

def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


# 関数: UTF-8 テキストを読む。

def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


# 関数: リポジトリ相対パス文字列へ変換する。

def rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


# 関数: テキスト内で最初に一致した行を返す。

def hit(text: str, pattern: str) -> dict | None:
    for line_no, line in enumerate(text.splitlines(), start=1):
        if pattern in line:
            return {"pattern": pattern, "line": line_no, "text": line.strip()}

    return None


# 関数: HTML から指定 fragment を含む href を抜き出す。

def href(text: str, fragment: str) -> str | None:
    match = re.search(rf'href="([^"]*{re.escape(fragment)}[^"]*)"', text, flags=re.IGNORECASE)
    if not match:
        return None

    return match.group(1)


# 関数: 共通 row 形式を返す。

def row(row_id: str, status: str, metric: str, value: float, note: str) -> dict:
    return {
        "row_id": row_id,
        "status": status,
        "metric": metric,
        "value": float(value),
        "note": note,
    }


# 関数: 共通 payload 形式を返す。

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


# 関数: metrics JSON と対応する rows CSV を保存する。

def write_artifact(stem: str, data: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    json_path = OUT / f"{stem}_metrics.json"
    csv_path = OUT / f"{stem}_rows.csv"
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["row_id", "status", "metric", "value", "note"])
        writer.writeheader()
        writer.writerows(data["rows"])


# 関数: canonical manifest を保存する。

def write_manifest(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


# 関数: canonical survey-native radial-profile table schema を返す。

def canonical_schema() -> list[dict]:
    return [
        {"name": "galaxy_name", "unit": "", "description": "Survey-native galaxy identifier."},
        {"name": "survey_family", "unit": "", "description": "Origin survey family such as THINGS+SINGS or LITTLE THINGS."},
        {"name": "radial_bin_index", "unit": "", "description": "Monotonic radial bin index starting from the innermost retained point."},
        {"name": "radius_arcsec", "unit": "arcsec", "description": "Native survey radius before SPARC-side harmonization."},
        {"name": "radius_kpc", "unit": "kpc", "description": "Physical radius after the survey-native distance is applied."},
        {"name": "v_rot_km_s", "unit": "km/s", "description": "Survey-native rotation-speed estimate per radial bin."},
        {"name": "v_rot_err_km_s", "unit": "km/s", "description": "Survey-native rotation-speed uncertainty per radial bin."},
        {"name": "velocity_dispersion_km_s", "unit": "km/s", "description": "Native moment-2 or equivalent dispersion summary when available."},
        {"name": "inclination_deg", "unit": "deg", "description": "Adopted inclination for the survey-native kinematic solution."},
        {"name": "position_angle_deg", "unit": "deg", "description": "Adopted position angle for the survey-native kinematic solution."},
        {"name": "source_weighting", "unit": "", "description": "Native weighting choice such as robust or natural."},
        {"name": "source_files", "unit": "", "description": "Semicolon-separated source FITS filenames used for the extracted row."},
    ]


# 関数: remote URL から local raw cache path を組み立てる。

def local_raw_target(family_dir: str, galaxy_name: str, url: str | None) -> Path | None:
    if not url:
        return None

    filename = url.rstrip("/").rsplit("/", 1)[-1]
    return RAW_CACHE_ROOT / family_dir / galaxy_name.lower() / filename


# 関数: THINGS spiral record を extraction contract 用に整形する。

def spiral_record(record: dict, things_mass_models_text: str) -> dict:
    remote_urls = record["rotation_source_urls"]
    local_targets = {
        key: local_raw_target("things_spiral", record["name"], value)
        for key, value in remote_urls.items()
    }
    local_cached_count = sum(1 for value in local_targets.values() if value is not None and value.exists())
    remote_ready = all(value is not None for value in remote_urls.values())
    local_ready = all(value is not None and value.exists() for value in local_targets.values())
    extraction_rule_hit = hit(things_mass_models_text, "derive the geometrical and dynamical parameters using HI data alone")
    extraction_rule_ready = extraction_rule_hit is not None
    schema_ready = True
    extraction_ready = remote_ready and local_ready and extraction_rule_ready and schema_ready

    return {
        "name": record["name"],
        "display_name": record["display_name"],
        "survey_family": record["survey_family"],
        "remote_raw_urls_ready": remote_ready,
        "local_raw_fits_cached": local_ready,
        "local_raw_fits_cached_count": local_cached_count,
        "extraction_rule_source_available": extraction_rule_ready,
        "canonical_table_schema_candidate_ready": schema_ready,
        "survey_native_radial_profile_extraction_ready": extraction_ready,
        "rotation_source_urls": remote_urls,
        "local_raw_target_paths": {key: rel(value) if value is not None else None for key, value in local_targets.items()},
        "reconstruction_blocker_or_none": None if extraction_ready else "things_spiral_local_raw_fits_cache_absent",
        "extraction_rule_source_hit": extraction_rule_hit,
    }


# 関数: LITTLE THINGS dwarf record を extraction contract 用に整形する。

def dwarf_record(record: dict, little_things_data_text: str, little_things_mass_models_text: str) -> dict:
    remote_urls = record["rotation_source_urls"]
    local_targets = {
        key: local_raw_target("little_things_dwarf", record["name"], value)
        for key, value in remote_urls.items()
    }
    local_cached_count = sum(1 for value in local_targets.values() if value is not None and value.exists())
    remote_ready = all(value is not None for value in remote_urls.values())
    local_ready = all(value is not None and value.exists() for value in local_targets.values())
    calibration_recipe_rel = href(little_things_data_text, "recipe_calibration_public.txt")
    mapping_recipe_rel = href(little_things_data_text, "recipe_mapping_public.txt")
    extraction_rule_ready = (
        calibration_recipe_rel is not None
        and mapping_recipe_rel is not None
        and hit(little_things_mass_models_text, "derived in a homogeneous and consistent manner") is not None
    )
    schema_ready = True
    extraction_ready = remote_ready and local_ready and extraction_rule_ready and schema_ready

    return {
        "name": record["name"],
        "display_name": record["display_name"],
        "survey_family": record["survey_family"],
        "remote_raw_urls_ready": remote_ready,
        "local_raw_fits_cached": local_ready,
        "local_raw_fits_cached_count": local_cached_count,
        "extraction_rule_source_available": extraction_rule_ready,
        "canonical_table_schema_candidate_ready": schema_ready,
        "survey_native_radial_profile_extraction_ready": extraction_ready,
        "rotation_source_urls": remote_urls,
        "local_raw_target_paths": {key: rel(value) if value is not None else None for key, value in local_targets.items()},
        "reconstruction_blocker_or_none": None if extraction_ready else "little_things_dwarf_local_raw_fits_cache_absent",
        "calibration_recipe_url": urljoin("https://www2.lowell.edu/users/dah/littlethings/", calibration_recipe_rel) if calibration_recipe_rel else None,
        "mapping_recipe_url": urljoin("https://www2.lowell.edu/users/dah/littlethings/", mapping_recipe_rel) if mapping_recipe_rel else None,
        "extraction_rule_source_hit": hit(little_things_mass_models_text, "derived in a homogeneous and consistent manner"),
    }


# 関数: bool field が true の record 数を数える。

def ready_count(records: list[dict], field: str) -> int:
    return sum(1 for record in records if record[field])


# 関数: bool を 0/1 の float へ変換する。

def as_float(value: bool) -> float:
    return 1.0 if value else 0.0


# 関数: branch 全体を実行して artifact を生成する。

def main() -> None:
    for path in (
        PILOT_RECONSTRUCTION_MANIFEST,
        PREVIOUS_ROUTE_CONTRACT,
        THINGS_PAGE,
        THINGS_SURVEY,
        THINGS_MASS_MODELS,
        LITTLE_THINGS_DATA,
        LITTLE_THINGS_SURVEY,
        LITTLE_THINGS_MASS_MODELS,
    ):
        req(path)

    pilot_manifest = read_json(PILOT_RECONSTRUCTION_MANIFEST)
    previous_route_contract = read_json(PREVIOUS_ROUTE_CONTRACT)
    things_page_text = read_text(THINGS_PAGE)
    things_survey_text = read_text(THINGS_SURVEY)
    things_mass_models_text = read_text(THINGS_MASS_MODELS)
    little_things_data_text = read_text(LITTLE_THINGS_DATA)
    little_things_survey_text = read_text(LITTLE_THINGS_SURVEY)
    little_things_mass_models_text = read_text(LITTLE_THINGS_MASS_MODELS)

    schema = canonical_schema()
    spiral_records = [spiral_record(record, things_mass_models_text) for record in pilot_manifest["spiral_family"]]
    dwarf_records = [dwarf_record(record, little_things_data_text, little_things_mass_models_text) for record in pilot_manifest["dwarf_family"]]

    spiral_remote_ready = ready_count(spiral_records, "remote_raw_urls_ready") == len(spiral_records)
    spiral_local_ready = ready_count(spiral_records, "local_raw_fits_cached") == len(spiral_records)
    spiral_rule_ready = ready_count(spiral_records, "extraction_rule_source_available") == len(spiral_records)
    spiral_schema_ready = True
    spiral_extraction_ready = ready_count(spiral_records, "survey_native_radial_profile_extraction_ready") == len(spiral_records)

    dwarf_remote_ready = ready_count(dwarf_records, "remote_raw_urls_ready") == len(dwarf_records)
    dwarf_local_ready = ready_count(dwarf_records, "local_raw_fits_cached") == len(dwarf_records)
    dwarf_rule_ready = ready_count(dwarf_records, "extraction_rule_source_available") == len(dwarf_records)
    dwarf_schema_ready = True
    dwarf_extraction_ready = ready_count(dwarf_records, "survey_native_radial_profile_extraction_ready") == len(dwarf_records)

    direct_kappa_second_retry_ready = spiral_extraction_ready and dwarf_extraction_ready
    direct_kappa_second_retry_executed = False
    dataset_intake_branch_closeable = False
    launch_v2_mainline_now = False
    next_common_blocker = "survey_native_spiral_and_dwarf_local_raw_fits_cache_absent"

    extraction_manifest = {
        "generated_utc": now_iso(),
        "registry_name": "independent_galaxy_survey_native_extraction_manifest",
        "derived_from": {
            "pilot_reconstruction_manifest": rel(PILOT_RECONSTRUCTION_MANIFEST),
            "previous_route_contract": rel(PREVIOUS_ROUTE_CONTRACT),
        },
        "spiral_family": spiral_records,
        "dwarf_family": dwarf_records,
        "canonical_survey_native_table_schema": schema,
        "notes": [
            "Remote THINGS and LITTLE THINGS raw FITS URLs remain available, but no local survey-native raw FITS cache exists under data/cosmology/non_sparc_rotation_curves/survey_native_raw/ yet.",
            "The next direct-kappa retry remains blocked until both spiral and dwarf families have local raw FITS caches and can emit machine-readable survey-native radial-profile tables on the canonical schema.",
        ],
    }
    write_manifest(EXTRACTION_MANIFEST, extraction_manifest)

    spiral_present_count = sum(int(value) for value in (spiral_remote_ready, spiral_local_ready, spiral_rule_ready, spiral_schema_ready))
    dwarf_present_count = sum(int(value) for value in (dwarf_remote_ready, dwarf_local_ready, dwarf_rule_ready, dwarf_schema_ready))

    payloads = {
        "mass_origin_dark_matter_things_spiral_survey_native_radial_profile_extraction_inventory": payload(
            "8.7.55.3.154",
            "THINGS spiral survey-native radial-profile extraction inventory",
            {
                "pilot_reconstruction_manifest_json": rel(PILOT_RECONSTRUCTION_MANIFEST),
                "things_data_products_html": rel(THINGS_PAGE),
                "things_survey_abs_html": rel(THINGS_SURVEY),
                "things_mass_models_abs_html": rel(THINGS_MASS_MODELS),
            },
            "Inventory the local raw FITS, extraction-rule evidence, and canonical schema candidates needed to freeze THINGS spiral survey-native radial-profile extraction.",
            {
                "spiral_extraction_inventory_rule": "the THINGS-side extraction inventory is complete only after remote raw FITS URLs, local raw FITS cache paths, extraction-rule evidence, and the canonical survey-native table schema have all been enumerated"
            },
            [
                row("things_spiral_remote_raw_url_family_ready", "pass" if spiral_remote_ready else "reject", "THINGS spiral remote raw URL family ready", as_float(spiral_remote_ready), "Each selected spiral pilot already exposes THINGS robust moment-map and cube URLs."),
                row("things_spiral_local_raw_fits_cache_ready", "pass" if spiral_local_ready else "reject", "THINGS spiral local raw FITS cache ready", as_float(spiral_local_ready), "The canonical non-SPARC raw cache does not yet contain the THINGS spiral FITS products."),
                row("things_spiral_extraction_rule_source_ready", "pass" if spiral_rule_ready else "reject", "THINGS spiral extraction-rule source ready", as_float(spiral_rule_ready), "The THINGS mass-model paper abstract still provides the minimal source statement that the rotation curves are derived from HI data alone."),
                row("things_spiral_canonical_table_schema_candidate_ready", "pass", "THINGS spiral canonical table-schema candidate ready", 1.0, "The branch froze a canonical survey-native radial-profile table schema for later extraction output."),
            ],
            {
                "selected_spiral_pilot_subset": [record["name"] for record in spiral_records],
                "required_spiral_extraction_source_count": 4,
                "present_spiral_extraction_source_count": spiral_present_count,
                "spiral_remote_raw_url_family_ready": spiral_remote_ready,
                "spiral_local_raw_fits_cache_ready": spiral_local_ready,
                "spiral_extraction_rule_source_ready": spiral_rule_ready,
                "spiral_canonical_table_schema_candidate_ready": spiral_schema_ready,
                "missing_source_components_or_none": [component for component, ready in (
                    ("things_spiral_local_raw_fits_cache", spiral_local_ready),
                ) if not ready],
                "first_route_to_close_or_none": "things_spiral_local_raw_fits_fetch" if not spiral_local_ready else None,
            },
            {
                "overall_status": "things_spiral_survey_native_extraction_inventory_frozen",
                "spiral_extraction_inventory_ready": True,
                "next_required_artifacts": [
                    "mass_origin_dark_matter_little_things_dwarf_survey_native_radial_profile_extraction_inventory",
                    "mass_origin_dark_matter_spiral_survey_native_radial_profile_extraction_freeze",
                ],
            },
            {
                "spiral_records": spiral_records,
                "things_data_products_hit": hit(things_page_text, "RO_MOM1_THINGS.FITS"),
                "things_survey_hit": hit(things_survey_text, "The high spatial and velocity resolution"),
                "things_mass_models_hit": hit(things_mass_models_text, "derive the geometrical and dynamical parameters using HI data alone"),
            },
        ),
        "mass_origin_dark_matter_little_things_dwarf_survey_native_radial_profile_extraction_inventory": payload(
            "8.7.55.3.155",
            "LITTLE THINGS dwarf survey-native radial-profile extraction inventory",
            {
                "pilot_reconstruction_manifest_json": rel(PILOT_RECONSTRUCTION_MANIFEST),
                "little_things_pubdata_html": rel(LITTLE_THINGS_DATA),
                "little_things_survey_abs_html": rel(LITTLE_THINGS_SURVEY),
                "little_things_mass_models_abs_html": rel(LITTLE_THINGS_MASS_MODELS),
                "survey_native_extraction_manifest_json": rel(EXTRACTION_MANIFEST),
            },
            "Inventory the local raw FITS, extraction-rule evidence, and canonical schema candidates needed to freeze LITTLE THINGS dwarf survey-native radial-profile extraction.",
            {
                "dwarf_extraction_inventory_rule": "the LITTLE THINGS-side extraction inventory is complete only after remote raw FITS URLs, local raw FITS cache paths, extraction recipes, and the canonical survey-native table schema have all been enumerated"
            },
            [
                row("little_things_dwarf_remote_raw_url_family_ready", "pass" if dwarf_remote_ready else "reject", "LITTLE THINGS dwarf remote raw URL family ready", as_float(dwarf_remote_ready), "Each selected dwarf pilot already exposes LITTLE THINGS cube and moment-map URLs."),
                row("little_things_dwarf_local_raw_fits_cache_ready", "pass" if dwarf_local_ready else "reject", "LITTLE THINGS dwarf local raw FITS cache ready", as_float(dwarf_local_ready), "The canonical non-SPARC raw cache does not yet contain the LITTLE THINGS dwarf FITS products."),
                row("little_things_dwarf_extraction_rule_source_ready", "pass" if dwarf_rule_ready else "reject", "LITTLE THINGS dwarf extraction-rule source ready", as_float(dwarf_rule_ready), "The public LITTLE THINGS data page still exposes calibration and mapping recipes, and the mass-model paper states that the rotation curves are derived in a homogeneous manner."),
                row("little_things_dwarf_canonical_table_schema_candidate_ready", "pass", "LITTLE THINGS dwarf canonical table-schema candidate ready", 1.0, "The branch froze a canonical survey-native radial-profile table schema for later extraction output."),
            ],
            {
                "selected_dwarf_pilot_subset": [record["name"] for record in dwarf_records],
                "required_dwarf_extraction_source_count": 4,
                "present_dwarf_extraction_source_count": dwarf_present_count,
                "dwarf_remote_raw_url_family_ready": dwarf_remote_ready,
                "dwarf_local_raw_fits_cache_ready": dwarf_local_ready,
                "dwarf_extraction_rule_source_ready": dwarf_rule_ready,
                "dwarf_canonical_table_schema_candidate_ready": dwarf_schema_ready,
                "missing_source_components_or_none": [component for component, ready in (
                    ("little_things_dwarf_local_raw_fits_cache", dwarf_local_ready),
                ) if not ready],
                "first_route_to_close_or_none": "little_things_dwarf_local_raw_fits_fetch" if not dwarf_local_ready else None,
            },
            {
                "overall_status": "little_things_dwarf_survey_native_extraction_inventory_frozen",
                "dwarf_extraction_inventory_ready": True,
                "next_required_artifacts": [
                    "mass_origin_dark_matter_spiral_survey_native_radial_profile_extraction_freeze",
                    "mass_origin_dark_matter_dwarf_survey_native_radial_profile_extraction_freeze",
                ],
            },
            {
                "dwarf_records": dwarf_records,
                "little_things_data_hit": hit(little_things_data_text, "intensity-weighted moment 1"),
                "little_things_survey_hit": hit(little_things_survey_text, "high-resolution Very Large Array HI survey"),
                "little_things_mass_models_hit": hit(little_things_mass_models_text, "derived in a homogeneous and consistent manner"),
            },
        ),
        "mass_origin_dark_matter_spiral_survey_native_radial_profile_extraction_freeze": payload(
            "8.7.55.3.156",
            "Spiral survey-native radial-profile extraction freeze",
            {
                "survey_native_extraction_manifest_json": rel(EXTRACTION_MANIFEST),
                "things_data_products_html": rel(THINGS_PAGE),
                "things_mass_models_abs_html": rel(THINGS_MASS_MODELS),
            },
            "Freeze the current THINGS spiral extraction contract after inventorying its local raw FITS, extraction rule, and canonical schema candidates.",
            {
                "spiral_extraction_freeze_rule": "the THINGS spiral extraction contract is ready only after the selected spiral pilot subset has remote raw FITS URLs, a populated local raw FITS cache, extraction-rule evidence, and a canonical survey-native table schema"
            },
            [
                row("things_spiral_remote_raw_urls_available_for_extraction", "pass" if spiral_remote_ready else "reject", "THINGS spiral remote raw URLs available for extraction", as_float(spiral_remote_ready), "The remote THINGS robust products are already enumerated."),
                row("things_spiral_local_raw_fits_available_for_extraction", "pass" if spiral_local_ready else "reject", "THINGS spiral local raw FITS available for extraction", as_float(spiral_local_ready), "The local THINGS raw FITS cache remains empty, so extraction cannot start yet."),
                row("things_spiral_extraction_rule_available_for_extraction", "pass" if spiral_rule_ready else "reject", "THINGS spiral extraction-rule source available for extraction", as_float(spiral_rule_ready), "The THINGS mass-model paper remains the minimal public source statement for HI-only rotation-curve derivation."),
                row("things_spiral_survey_native_extraction_ready", "pass" if spiral_extraction_ready else "reject", "THINGS spiral survey-native extraction ready", as_float(spiral_extraction_ready), "The contract remains blocked until the local raw FITS cache is actually populated."),
            ],
            {
                "selected_spiral_pilot_subset": [record["name"] for record in spiral_records],
                "spiral_remote_raw_urls_ready": spiral_remote_ready,
                "spiral_local_raw_fits_cache_ready": spiral_local_ready,
                "spiral_extraction_rule_source_ready": spiral_rule_ready,
                "spiral_canonical_table_schema_candidate_ready": spiral_schema_ready,
                "spiral_survey_native_radial_profile_extraction_ready": spiral_extraction_ready,
                "extraction_nonclosure_reason_or_none": None if spiral_extraction_ready else "things_spiral_local_raw_fits_cache_absent",
                "first_route_to_close_or_none": "things_spiral_local_raw_fits_fetch" if not spiral_extraction_ready else None,
            },
            {
                "overall_status": "things_spiral_survey_native_extraction_blocked_at_local_raw_fits_fetch",
                "spiral_survey_native_radial_profile_extraction_ready": spiral_extraction_ready,
                "next_required_artifacts": [
                    "mass_origin_dark_matter_dwarf_survey_native_radial_profile_extraction_freeze",
                    "mass_origin_dark_matter_independent_galaxy_dataset_intake_third_gate",
                ],
            },
            {
                "spiral_records": spiral_records,
                "canonical_schema": schema,
            },
        ),
        "mass_origin_dark_matter_dwarf_survey_native_radial_profile_extraction_freeze": payload(
            "8.7.55.3.157",
            "Dwarf survey-native radial-profile extraction freeze / direct-kappa reopen readiness",
            {
                "survey_native_extraction_manifest_json": rel(EXTRACTION_MANIFEST),
                "little_things_pubdata_html": rel(LITTLE_THINGS_DATA),
                "little_things_mass_models_abs_html": rel(LITTLE_THINGS_MASS_MODELS),
            },
            "Freeze the current LITTLE THINGS dwarf extraction contract and decide whether a second independent direct-kappa retry can reopen now.",
            {
                "dwarf_extraction_freeze_rule": "the LITTLE THINGS dwarf extraction contract is ready only after the selected dwarf pilot subset has remote raw FITS URLs, a populated local raw FITS cache, extraction-rule evidence, and a canonical survey-native table schema"
            },
            [
                row("little_things_dwarf_remote_raw_urls_available_for_extraction", "pass" if dwarf_remote_ready else "reject", "LITTLE THINGS dwarf remote raw URLs available for extraction", as_float(dwarf_remote_ready), "The remote LITTLE THINGS cube and moment-map URLs are already enumerated."),
                row("little_things_dwarf_local_raw_fits_available_for_extraction", "pass" if dwarf_local_ready else "reject", "LITTLE THINGS dwarf local raw FITS available for extraction", as_float(dwarf_local_ready), "The local LITTLE THINGS raw FITS cache remains empty, so extraction cannot start yet."),
                row("little_things_dwarf_extraction_rule_available_for_extraction", "pass" if dwarf_rule_ready else "reject", "LITTLE THINGS dwarf extraction-rule source available for extraction", as_float(dwarf_rule_ready), "The public LITTLE THINGS recipes and homogeneous-derivation statement remain available."),
                row("independent_direct_kappa_second_retry_reopen_ready", "pass" if direct_kappa_second_retry_ready else "reject", "independent direct-kappa second retry reopen ready", as_float(direct_kappa_second_retry_ready), "The second direct-kappa retry cannot reopen before both spiral and dwarf survey-native extraction contracts are ready."),
            ],
            {
                "selected_dwarf_pilot_subset": [record["name"] for record in dwarf_records],
                "dwarf_remote_raw_urls_ready": dwarf_remote_ready,
                "dwarf_local_raw_fits_cache_ready": dwarf_local_ready,
                "dwarf_extraction_rule_source_ready": dwarf_rule_ready,
                "dwarf_canonical_table_schema_candidate_ready": dwarf_schema_ready,
                "dwarf_survey_native_radial_profile_extraction_ready": dwarf_extraction_ready,
                "independent_direct_kappa_second_retry_ready": direct_kappa_second_retry_ready,
                "extraction_nonclosure_reason_or_none": None if dwarf_extraction_ready else "little_things_dwarf_local_raw_fits_cache_absent",
                "first_route_to_close_or_none": "little_things_dwarf_local_raw_fits_fetch" if not dwarf_extraction_ready else None,
            },
            {
                "overall_status": "little_things_dwarf_survey_native_extraction_blocked_at_local_raw_fits_fetch",
                "dwarf_survey_native_radial_profile_extraction_ready": dwarf_extraction_ready,
                "direct_kappa_second_retry_reopen_ready": direct_kappa_second_retry_ready,
                "next_required_artifacts": [
                    "mass_origin_dark_matter_independent_galaxy_dataset_intake_third_gate",
                    "mass_origin_dark_matter_local_raw_fits_fetch_route_contract",
                ],
            },
            {
                "dwarf_records": dwarf_records,
                "canonical_schema": schema,
            },
        ),
        "mass_origin_dark_matter_independent_galaxy_dataset_intake_third_gate": payload(
            "8.7.55.3.158",
            "Dataset-intake declaration third gate / v2.0 defer-or-launch audit",
            {
                "survey_native_extraction_manifest_json": rel(EXTRACTION_MANIFEST),
                "previous_route_contract_json": rel(PREVIOUS_ROUTE_CONTRACT),
            },
            "Connect the survey-native extraction results to the declaration gate and decide whether the independent-galaxy branch can close now or must continue with a new residual route.",
            {
                "third_gate_rule": "the branch can close only after both spiral and dwarf survey-native extraction contracts are ready and the independent direct-kappa second retry can execute"
            },
            [
                row("independent_spiral_survey_native_extraction_ready", "pass" if spiral_extraction_ready else "reject", "independent spiral survey-native extraction ready", as_float(spiral_extraction_ready), "The THINGS spiral extraction contract still lacks populated local raw FITS caches."),
                row("independent_dwarf_survey_native_extraction_ready", "pass" if dwarf_extraction_ready else "reject", "independent dwarf survey-native extraction ready", as_float(dwarf_extraction_ready), "The LITTLE THINGS dwarf extraction contract still lacks populated local raw FITS caches."),
                row("independent_dataset_intake_third_gate_closeable", "pass" if dataset_intake_branch_closeable else "reject", "independent dataset-intake third gate closeable", as_float(dataset_intake_branch_closeable), "The independent-galaxy branch remains open because survey-native extraction cannot run without local raw FITS caches."),
                row("launch_v2_mainline_now", "pass" if launch_v2_mainline_now else "reject", "launch v2.0 mainline now", as_float(launch_v2_mainline_now), "The queued v2.0 mainline stays deferred until the independent-galaxy intake route closes."),
            ],
            {
                "previous_common_blocker": previous_route_contract["summary"]["missing_dark_matter_artifact"],
                "spiral_survey_native_extraction_ready": spiral_extraction_ready,
                "dwarf_survey_native_extraction_ready": dwarf_extraction_ready,
                "independent_direct_kappa_second_retry_executed": direct_kappa_second_retry_executed,
                "dataset_intake_branch_closeable": dataset_intake_branch_closeable,
                "launch_v2_mainline_now": launch_v2_mainline_now,
                "defer_v2_mainline_now": not launch_v2_mainline_now,
                "recommended_next_route_or_none": "8.7.55.3.159",
                "selected_next_route": "independent_galaxy_local_raw_fits_fetch",
            },
            {
                "overall_status": "independent_dataset_intake_blocked_at_local_raw_fits_fetch",
                "dataset_intake_branch_closeable": dataset_intake_branch_closeable,
                "launch_v2_mainline_now": launch_v2_mainline_now,
                "recommended_next_route_or_none": "8.7.55.3.159",
                "next_required_artifacts": ["mass_origin_dark_matter_local_raw_fits_fetch_route_contract"],
            },
            {
                "previous_route_contract": previous_route_contract["summary"],
                "extraction_manifest": extraction_manifest,
                "common_blocker": next_common_blocker,
            },
        ),
        "mass_origin_dark_matter_local_raw_fits_fetch_route_contract": payload(
            "8.7.55.3.159",
            "Local raw FITS fetch route contract",
            {
                "survey_native_extraction_manifest_json": rel(EXTRACTION_MANIFEST),
                "dataset_intake_third_gate_json": "output/public/quantum/mass_origin_dark_matter_independent_galaxy_dataset_intake_third_gate_metrics.json",
            },
            "Freeze the next residual route after the survey-native extraction branch fails to close because no local raw FITS cache exists yet.",
            {
                "route_contract_rule": "the next residual route must target the earliest common missing artifact that blocks both the THINGS spiral and LITTLE THINGS dwarf extraction contracts"
            },
            [
                row("local_raw_fits_fetch_route_selected", "pass", "local raw FITS fetch route selected", 1.0, "The next residual route is now frozen."),
                row("local_raw_fits_fetch_common_blocker_present", "pass", "common spiral+dwarf local raw FITS blocker present", 1.0, "Both selected pilot families are blocked by the absence of populated local raw FITS caches."),
                row("local_raw_fits_fetch_split_contract_ready", "pass", "local raw FITS fetch split contract ready", 1.0, "The next branch can now decompose THINGS spiral and LITTLE THINGS dwarf raw-fetch work without ambiguity."),
            ],
            {
                "selected_residual_route": "independent_galaxy_local_raw_fits_fetch",
                "missing_dark_matter_artifact": "survey_native_spiral_and_dwarf_local_raw_fits_cache",
                "split_contract_ready": True,
                "defer_v2_mainline_until_route_close": True,
            },
            {
                "overall_status": "independent_galaxy_local_raw_fits_fetch_route_frozen",
                "next_required_artifacts": [
                    "8.7.55.3.160",
                    "8.7.55.3.161",
                    "8.7.55.3.162",
                    "8.7.55.3.163",
                    "8.7.55.3.164",
                ],
            },
            {
                "spiral_records": spiral_records,
                "dwarf_records": dwarf_records,
                "common_blocker": next_common_blocker,
            },
        ),
    }

    for stem, data in payloads.items():
        write_artifact(stem, data)
        print(f"[ok] wrote {OUT / (stem + '_metrics.json')}")
        print(f"[ok] wrote {OUT / (stem + '_rows.csv')}")

    print(f"[ok] wrote {EXTRACTION_MANIFEST}")


# 関数: script 実行時に branch 本体を起動する。

if __name__ == "__main__":
    main()
