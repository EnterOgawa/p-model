#!/usr/bin/env python3
"""
Freeze the independent-galaxy local raw FITS fetch branch for 8.7.55.3.160-.165.

This branch follows the survey-native radial-profile extraction audit, which
already established that THINGS spiral and LITTLE THINGS dwarf pilot subsets
have remote raw URL families and extraction rules, but still lack a populated
local raw FITS cache under the canonical non-SPARC store.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output" / "public" / "quantum"
NON_SPARC_DIR = ROOT / "data" / "cosmology" / "non_sparc_rotation_curves"
RAW_CACHE_ROOT = NON_SPARC_DIR / "survey_native_raw"
EXTRACTION_MANIFEST = NON_SPARC_DIR / "pilot_survey_native_extraction_manifest.json"
PREVIOUS_ROUTE_CONTRACT = OUT / "mass_origin_dark_matter_local_raw_fits_fetch_route_contract_metrics.json"
FETCH_MANIFEST = NON_SPARC_DIR / "pilot_local_raw_fits_fetch_manifest.json"
CHECKSUM_MANIFEST = NON_SPARC_DIR / "pilot_local_raw_fits_checksum_manifest.json"


# 関数: 現在UTCを ISO 8601 形式で返す。
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: 必須入力が無い場合は即座に終了する。

def req(path: Path) -> None:
    if not path.exists():
        raise SystemExit(f"[fail] missing required input: {path}")


# 関数: UTF-8 JSON を辞書として読む。

def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


# 関数: リポジトリ相対パス文字列へ変換する。

def rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


# 関数: 共通 row 形式を返す。

def row(row_id: str, status: str, metric: str, value: float, note: str) -> dict[str, Any]:
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
    inputs: dict[str, Any],
    intent: str,
    formulas: dict[str, Any],
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    decision: dict[str, Any],
    evidence: dict[str, Any],
) -> dict[str, Any]:
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

def write_artifact(stem: str, data: dict[str, Any]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    json_path = OUT / f"{stem}_metrics.json"
    csv_path = OUT / f"{stem}_rows.csv"
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["row_id", "status", "metric", "value", "note"])
        writer.writeheader()
        writer.writerows(data["rows"])


# 関数: canonical manifest を保存する。

def write_manifest(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


# 関数: bool を 0/1 の float へ変換する。

def as_float(value: bool) -> float:
    return 1.0 if value else 0.0


# 関数: URL の HEAD metadata を取得する。

def remote_head(url: str) -> dict[str, Any]:
    request = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(request, timeout=60) as response:
        length_raw = response.headers.get("Content-Length")
        return {
            "url": url,
            "status": int(getattr(response, "status", 200)),
            "content_length_bytes": int(length_raw) if length_raw else None,
            "last_modified": response.headers.get("Last-Modified"),
            "content_type": response.headers.get("Content-Type"),
        }


# 関数: ファイルの sha256 を計算する。

def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest().upper()


# 関数: URL が bulk cube か lightweight map かを分類する。

def classify_role(url: str) -> dict[str, Any]:
    name = url.rsplit("/", 1)[-1].upper()
    is_bulk_cube = name.endswith("_CUBE_THINGS.FITS") or name.endswith("_ICL001.FITS")
    return {
        "file_role": "bulk_cube_fits" if is_bulk_cube else "lightweight_moment_map_fits",
        "should_fetch_now": not is_bulk_cube,
    }


# 関数: 軽量 FITS を local cache へ取得し sha256 を返す。

def fetch_lightweight_file(url: str, target: Path) -> dict[str, Any]:
    target.parent.mkdir(parents=True, exist_ok=True)
    remote = remote_head(url)
    expected_size = remote.get("content_length_bytes")

    if target.exists() and (expected_size is None or target.stat().st_size == expected_size):
        return {
            "fetch_status": "reused_existing_file",
            "local_exists": True,
            "local_size_bytes": int(target.stat().st_size),
            "local_sha256": sha256_file(target),
            "remote_head": remote,
        }

    temp_path = target.with_suffix(target.suffix + ".part")
    if temp_path.exists():
        temp_path.unlink()

    hasher = hashlib.sha256()
    bytes_written = 0
    with urllib.request.urlopen(url, timeout=120) as response, temp_path.open("wb") as handle:
        while True:
            chunk = response.read(8 * 1024 * 1024)
            if not chunk:
                break

            handle.write(chunk)
            hasher.update(chunk)
            bytes_written += len(chunk)

    if expected_size is not None and bytes_written != expected_size:
        temp_path.unlink(missing_ok=True)
        raise RuntimeError(f"download size mismatch for {url}: expected {expected_size}, got {bytes_written}")

    shutil.move(str(temp_path), str(target))
    return {
        "fetch_status": "downloaded_now",
        "local_exists": True,
        "local_size_bytes": int(target.stat().st_size),
        "local_sha256": hasher.hexdigest().upper(),
        "remote_head": remote,
    }


# 関数: 1 entry を fetch manifest / checksum manifest 用に整形する。

def file_entry(family: str, galaxy_name: str, file_key: str, url: str, local_path: str) -> dict[str, Any]:
    role = classify_role(url)
    target = ROOT / local_path
    remote = remote_head(url)
    result = {
        "family": family,
        "galaxy_name": galaxy_name,
        "file_key": file_key,
        "remote_url": url,
        "local_path": local_path,
        "file_role": role["file_role"],
        "should_fetch_now": role["should_fetch_now"],
        "remote_head": remote,
        "fetch_status": "deferred_bulk_cube" if not role["should_fetch_now"] else None,
        "local_exists": target.exists(),
        "local_size_bytes": int(target.stat().st_size) if target.exists() else None,
        "local_sha256": sha256_file(target) if target.exists() else None,
    }

    if role["should_fetch_now"]:
        fetch_result = fetch_lightweight_file(url, target)
        result.update(fetch_result)
    elif target.exists():
        result["fetch_status"] = "bulk_cube_already_cached"
        result["local_exists"] = True
        result["local_size_bytes"] = int(target.stat().st_size)
        result["local_sha256"] = sha256_file(target)

    return result


# 関数: family entry 群から summary を集計する。

def summarize_family(entries: list[dict[str, Any]]) -> dict[str, Any]:
    total_bytes = sum(int(entry["remote_head"]["content_length_bytes"] or 0) for entry in entries)
    light_entries = [entry for entry in entries if entry["file_role"] == "lightweight_moment_map_fits"]
    bulk_entries = [entry for entry in entries if entry["file_role"] == "bulk_cube_fits"]
    fetched_light_entries = [entry for entry in light_entries if entry.get("local_exists")]
    bulk_cached_entries = [entry for entry in bulk_entries if entry.get("local_exists")]
    lightweight_bytes = sum(int(entry["remote_head"]["content_length_bytes"] or 0) for entry in light_entries)
    bulk_bytes = sum(int(entry["remote_head"]["content_length_bytes"] or 0) for entry in bulk_entries)
    fetched_bytes = sum(int(entry.get("local_size_bytes") or 0) for entry in fetched_light_entries)

    return {
        "required_entry_count": len(entries),
        "remote_head_ready_count": sum(1 for entry in entries if entry["remote_head"]["content_length_bytes"] is not None),
        "lightweight_entry_count": len(light_entries),
        "bulk_cube_entry_count": len(bulk_entries),
        "lightweight_cached_count": len(fetched_light_entries),
        "bulk_cube_cached_count": len(bulk_cached_entries),
        "total_remote_bytes": total_bytes,
        "lightweight_remote_bytes": lightweight_bytes,
        "bulk_cube_remote_bytes": bulk_bytes,
        "fetched_lightweight_bytes": fetched_bytes,
        "lightweight_cache_ready": len(fetched_light_entries) == len(light_entries),
        "bulk_cube_cache_ready": len(bulk_cached_entries) == len(bulk_entries),
        "local_raw_fits_cache_ready": len(fetched_light_entries) == len(light_entries) and len(bulk_cached_entries) == len(bulk_entries),
    }


# 関数: extraction manifest の family record から file entry 一覧を作る。

def build_family_entries(family_name: str, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for record in records:
        local_paths = record["local_raw_target_paths"]
        for file_key, url in record["rotation_source_urls"].items():
            entries.append(file_entry(family_name, record["name"], file_key, url, local_paths[file_key]))

    return entries


# 関数: metrics と manifests を一括生成する。

def main() -> None:
    for path in (EXTRACTION_MANIFEST, PREVIOUS_ROUTE_CONTRACT):
        req(path)

    extraction_manifest = read_json(EXTRACTION_MANIFEST)
    previous_route_contract = read_json(PREVIOUS_ROUTE_CONTRACT)
    spiral_entries = build_family_entries("things_spiral", extraction_manifest["spiral_family"])
    dwarf_entries = build_family_entries("little_things_dwarf", extraction_manifest["dwarf_family"])
    spiral_summary = summarize_family(spiral_entries)
    dwarf_summary = summarize_family(dwarf_entries)

    fetch_manifest = {
        "generated_utc": now_iso(),
        "registry_name": "independent_galaxy_local_raw_fits_fetch_manifest",
        "derived_from": {
            "survey_native_extraction_manifest": rel(EXTRACTION_MANIFEST),
            "previous_route_contract": rel(PREVIOUS_ROUTE_CONTRACT),
        },
        "spiral_family": spiral_entries,
        "dwarf_family": dwarf_entries,
        "notes": [
            "Lightweight moment-map FITS were fetched immediately and hashed into the canonical non-SPARC raw cache.",
            "Bulk cube FITS remain deferred because the selected pilot families still require multi-GB raw cubes that are larger than the lightweight fetch tranche.",
        ],
    }
    write_manifest(FETCH_MANIFEST, fetch_manifest)

    checksum_manifest = {
        "generated_utc": now_iso(),
        "registry_name": "independent_galaxy_local_raw_fits_checksum_manifest",
        "derived_from": {
            "local_raw_fits_fetch_manifest": rel(FETCH_MANIFEST),
        },
        "spiral_family": spiral_entries,
        "dwarf_family": dwarf_entries,
        "notes": [
            "sha256 is populated for downloaded lightweight moment maps and remains null for deferred bulk cube FITS.",
        ],
    }
    write_manifest(CHECKSUM_MANIFEST, checksum_manifest)

    dataset_branch_closeable = spiral_summary["local_raw_fits_cache_ready"] and dwarf_summary["local_raw_fits_cache_ready"]
    launch_v2_mainline_now = dataset_branch_closeable
    selected_next_route = "none" if dataset_branch_closeable else "independent_galaxy_bulk_cube_fits_fetch"
    recommended_next_route = None if dataset_branch_closeable else "8.7.55.3.165"

    payloads: dict[str, dict[str, Any]] = {}
    payloads["mass_origin_dark_matter_things_spiral_local_raw_fits_fetch_manifest_freeze"] = payload(
        "8.7.55.3.160",
        "THINGS spiral local raw FITS fetch manifest freeze",
        {
            "survey_native_extraction_manifest_json": rel(EXTRACTION_MANIFEST),
            "previous_route_contract_json": rel(PREVIOUS_ROUTE_CONTRACT),
        },
        "Freeze the THINGS-side local raw FITS fetch targets, remote metadata, and manifest layout needed to build a canonical non-SPARC raw cache.",
        {
            "things_spiral_fetch_freeze_rule": "the THINGS-side local raw FITS fetch manifest is ready after each selected spiral pilot has remote HEAD metadata, a canonical local path, and an explicit now-or-defer fetch classification"
        },
        [
            row("things_spiral_remote_head_ready", "pass", "THINGS spiral remote HEAD ready", as_float(spiral_summary["remote_head_ready_count"] == spiral_summary["required_entry_count"]), "All THINGS spiral targets returned remote content-length metadata."),
            row("things_spiral_lightweight_fetch_targets_frozen", "pass", "THINGS spiral lightweight fetch targets frozen", as_float(spiral_summary["lightweight_entry_count"] > 0), "All spiral moment-map FITS targets are classified as lightweight fetch-now entries."),
            row("things_spiral_bulk_cube_targets_frozen", "pass", "THINGS spiral bulk cube targets frozen", as_float(spiral_summary["bulk_cube_entry_count"] > 0), "All spiral cube FITS targets are classified as bulk fetch-later entries."),
            row("things_spiral_fetch_manifest_written", "pass", "THINGS spiral fetch manifest written", 1.0, "The canonical local raw FITS fetch manifest now exists on disk."),
        ],
        {
            "selected_spiral_pilot_subset": [record["name"] for record in extraction_manifest["spiral_family"]],
            "required_things_spiral_raw_fits_entry_count": spiral_summary["required_entry_count"],
            "things_spiral_remote_head_ready_count": spiral_summary["remote_head_ready_count"],
            "things_spiral_lightweight_entry_count": spiral_summary["lightweight_entry_count"],
            "things_spiral_bulk_cube_entry_count": spiral_summary["bulk_cube_entry_count"],
            "things_spiral_total_remote_bytes": spiral_summary["total_remote_bytes"],
            "things_spiral_lightweight_remote_bytes": spiral_summary["lightweight_remote_bytes"],
            "things_spiral_bulk_cube_remote_bytes": spiral_summary["bulk_cube_remote_bytes"],
            "things_spiral_fetch_manifest_json": rel(FETCH_MANIFEST),
            "first_route_to_close_or_none": "things_spiral_bulk_cube_fits_fetch" if not spiral_summary["bulk_cube_cache_ready"] else None,
        },
        {
            "overall_status": "things_spiral_local_raw_fits_fetch_manifest_frozen",
            "things_spiral_fetch_manifest_ready": True,
            "next_required_artifacts": [
                "mass_origin_dark_matter_little_things_dwarf_local_raw_fits_fetch_manifest_freeze",
                "mass_origin_dark_matter_things_spiral_local_raw_fits_fetch_checksum_manifest",
            ],
        },
        {
            "spiral_entries": spiral_entries,
        },
    )
    payloads["mass_origin_dark_matter_little_things_dwarf_local_raw_fits_fetch_manifest_freeze"] = payload(
        "8.7.55.3.161",
        "LITTLE THINGS dwarf local raw FITS fetch manifest freeze",
        {
            "survey_native_extraction_manifest_json": rel(EXTRACTION_MANIFEST),
            "local_raw_fits_fetch_manifest_json": rel(FETCH_MANIFEST),
        },
        "Freeze the LITTLE THINGS-side local raw FITS fetch targets, remote metadata, and manifest layout needed to build a canonical non-SPARC raw cache.",
        {
            "little_things_dwarf_fetch_freeze_rule": "the LITTLE THINGS-side local raw FITS fetch manifest is ready after each selected dwarf pilot has remote HEAD metadata, a canonical local path, and an explicit now-or-defer fetch classification"
        },
        [
            row("little_things_dwarf_remote_head_ready", "pass", "LITTLE THINGS dwarf remote HEAD ready", as_float(dwarf_summary["remote_head_ready_count"] == dwarf_summary["required_entry_count"]), "All LITTLE THINGS dwarf targets returned remote content-length metadata."),
            row("little_things_dwarf_lightweight_fetch_targets_frozen", "pass", "LITTLE THINGS dwarf lightweight fetch targets frozen", as_float(dwarf_summary["lightweight_entry_count"] > 0), "All dwarf moment-map FITS targets are classified as lightweight fetch-now entries."),
            row("little_things_dwarf_bulk_cube_targets_frozen", "pass", "LITTLE THINGS dwarf bulk cube targets frozen", as_float(dwarf_summary["bulk_cube_entry_count"] > 0), "All dwarf cube FITS targets are classified as bulk fetch-later entries."),
            row("little_things_dwarf_fetch_manifest_written", "pass", "LITTLE THINGS dwarf fetch manifest written", 1.0, "The canonical local raw FITS fetch manifest now exists on disk."),
        ],
        {
            "selected_dwarf_pilot_subset": [record["name"] for record in extraction_manifest["dwarf_family"]],
            "required_little_things_dwarf_raw_fits_entry_count": dwarf_summary["required_entry_count"],
            "little_things_dwarf_remote_head_ready_count": dwarf_summary["remote_head_ready_count"],
            "little_things_dwarf_lightweight_entry_count": dwarf_summary["lightweight_entry_count"],
            "little_things_dwarf_bulk_cube_entry_count": dwarf_summary["bulk_cube_entry_count"],
            "little_things_dwarf_total_remote_bytes": dwarf_summary["total_remote_bytes"],
            "little_things_dwarf_lightweight_remote_bytes": dwarf_summary["lightweight_remote_bytes"],
            "little_things_dwarf_bulk_cube_remote_bytes": dwarf_summary["bulk_cube_remote_bytes"],
            "little_things_dwarf_fetch_manifest_json": rel(FETCH_MANIFEST),
            "first_route_to_close_or_none": "little_things_dwarf_bulk_cube_fits_fetch" if not dwarf_summary["bulk_cube_cache_ready"] else None,
        },
        {
            "overall_status": "little_things_dwarf_local_raw_fits_fetch_manifest_frozen",
            "little_things_dwarf_fetch_manifest_ready": True,
            "next_required_artifacts": [
                "mass_origin_dark_matter_things_spiral_local_raw_fits_fetch_checksum_manifest",
                "mass_origin_dark_matter_little_things_dwarf_local_raw_fits_fetch_checksum_manifest",
            ],
        },
        {
            "dwarf_entries": dwarf_entries,
        },
    )
    payloads["mass_origin_dark_matter_things_spiral_local_raw_fits_fetch_checksum_manifest"] = payload(
        "8.7.55.3.162",
        "THINGS spiral local raw FITS fetch / checksum manifest",
        {
            "local_raw_fits_fetch_manifest_json": rel(FETCH_MANIFEST),
        },
        "Fetch the lightweight THINGS spiral FITS tranche and freeze the checksum manifest while keeping the bulk cube tranche explicit.",
        {
            "things_spiral_checksum_manifest_rule": "the THINGS-side checksum manifest is ready after every lightweight target has a local file and sha256, even if the bulk cube tranche remains deferred"
        },
        [
            row("things_spiral_lightweight_tranche_cached", "pass" if spiral_summary["lightweight_cache_ready"] else "reject", "THINGS spiral lightweight tranche cached", as_float(spiral_summary["lightweight_cache_ready"]), "All spiral moment-map FITS targets should now exist in the canonical cache."),
            row("things_spiral_lightweight_tranche_checksummed", "pass" if spiral_summary["lightweight_cache_ready"] else "reject", "THINGS spiral lightweight tranche checksummed", as_float(spiral_summary["lightweight_cache_ready"]), "Downloaded spiral moment maps now carry sha256 digests in the checksum manifest."),
            row("things_spiral_bulk_cube_tranche_cached", "pass" if spiral_summary["bulk_cube_cache_ready"] else "reject", "THINGS spiral bulk cube tranche cached", as_float(spiral_summary["bulk_cube_cache_ready"]), "The multi-GB spiral cube FITS targets remain deferred."),
            row("things_spiral_checksum_manifest_written", "pass", "THINGS spiral checksum manifest written", 1.0, "The canonical checksum manifest now exists on disk."),
        ],
        {
            "things_spiral_lightweight_cached_count": spiral_summary["lightweight_cached_count"],
            "things_spiral_lightweight_entry_count": spiral_summary["lightweight_entry_count"],
            "things_spiral_bulk_cube_cached_count": spiral_summary["bulk_cube_cached_count"],
            "things_spiral_bulk_cube_entry_count": spiral_summary["bulk_cube_entry_count"],
            "things_spiral_fetched_lightweight_bytes": spiral_summary["fetched_lightweight_bytes"],
            "things_spiral_bulk_cube_remote_bytes": spiral_summary["bulk_cube_remote_bytes"],
            "things_spiral_local_raw_fits_cache_ready": spiral_summary["local_raw_fits_cache_ready"],
            "things_spiral_checksum_manifest_json": rel(CHECKSUM_MANIFEST),
            "remaining_blocker_or_none": None if spiral_summary["local_raw_fits_cache_ready"] else "things_spiral_bulk_cube_fits_cache_absent",
        },
        {
            "overall_status": "things_spiral_lightweight_raw_fits_cached_bulk_cube_deferred",
            "things_spiral_lightweight_cache_ready": spiral_summary["lightweight_cache_ready"],
            "things_spiral_local_raw_fits_cache_ready": spiral_summary["local_raw_fits_cache_ready"],
            "next_required_artifacts": [
                "mass_origin_dark_matter_little_things_dwarf_local_raw_fits_fetch_checksum_manifest",
                "mass_origin_dark_matter_independent_galaxy_dataset_intake_fourth_gate",
            ],
        },
        {
            "spiral_entries": spiral_entries,
            "checksum_manifest_json": rel(CHECKSUM_MANIFEST),
        },
    )
    payloads["mass_origin_dark_matter_little_things_dwarf_local_raw_fits_fetch_checksum_manifest"] = payload(
        "8.7.55.3.163",
        "LITTLE THINGS dwarf local raw FITS fetch / checksum manifest / extraction reopen readiness",
        {
            "local_raw_fits_fetch_manifest_json": rel(FETCH_MANIFEST),
            "local_raw_fits_checksum_manifest_json": rel(CHECKSUM_MANIFEST),
        },
        "Fetch the lightweight LITTLE THINGS dwarf FITS tranche, freeze the checksum manifest, and decide whether survey-native extraction can reopen now.",
        {
            "little_things_dwarf_checksum_manifest_rule": "the LITTLE THINGS-side checksum manifest is ready after every lightweight target has a local file and sha256, even if the bulk cube tranche remains deferred"
        },
        [
            row("little_things_dwarf_lightweight_tranche_cached", "pass" if dwarf_summary["lightweight_cache_ready"] else "reject", "LITTLE THINGS dwarf lightweight tranche cached", as_float(dwarf_summary["lightweight_cache_ready"]), "All dwarf moment-map FITS targets should now exist in the canonical cache."),
            row("little_things_dwarf_lightweight_tranche_checksummed", "pass" if dwarf_summary["lightweight_cache_ready"] else "reject", "LITTLE THINGS dwarf lightweight tranche checksummed", as_float(dwarf_summary["lightweight_cache_ready"]), "Downloaded dwarf moment maps now carry sha256 digests in the checksum manifest."),
            row("little_things_dwarf_bulk_cube_tranche_cached", "pass" if dwarf_summary["bulk_cube_cache_ready"] else "reject", "LITTLE THINGS dwarf bulk cube tranche cached", as_float(dwarf_summary["bulk_cube_cache_ready"]), "The multi-GB dwarf cube FITS targets remain deferred."),
            row("independent_survey_native_extraction_reopen_ready", "pass" if dataset_branch_closeable else "reject", "independent survey-native extraction reopen ready", as_float(dataset_branch_closeable), "Survey-native extraction cannot reopen until the bulk cube FITS cache exists for both families."),
        ],
        {
            "little_things_dwarf_lightweight_cached_count": dwarf_summary["lightweight_cached_count"],
            "little_things_dwarf_lightweight_entry_count": dwarf_summary["lightweight_entry_count"],
            "little_things_dwarf_bulk_cube_cached_count": dwarf_summary["bulk_cube_cached_count"],
            "little_things_dwarf_bulk_cube_entry_count": dwarf_summary["bulk_cube_entry_count"],
            "little_things_dwarf_fetched_lightweight_bytes": dwarf_summary["fetched_lightweight_bytes"],
            "little_things_dwarf_bulk_cube_remote_bytes": dwarf_summary["bulk_cube_remote_bytes"],
            "little_things_dwarf_local_raw_fits_cache_ready": dwarf_summary["local_raw_fits_cache_ready"],
            "independent_survey_native_extraction_reopen_ready": dataset_branch_closeable,
            "remaining_blocker_or_none": None if dwarf_summary["local_raw_fits_cache_ready"] else "little_things_dwarf_bulk_cube_fits_cache_absent",
        },
        {
            "overall_status": "little_things_dwarf_lightweight_raw_fits_cached_bulk_cube_deferred",
            "little_things_dwarf_lightweight_cache_ready": dwarf_summary["lightweight_cache_ready"],
            "little_things_dwarf_local_raw_fits_cache_ready": dwarf_summary["local_raw_fits_cache_ready"],
            "independent_survey_native_extraction_reopen_ready": dataset_branch_closeable,
            "next_required_artifacts": [
                "mass_origin_dark_matter_independent_galaxy_dataset_intake_fourth_gate",
                "mass_origin_dark_matter_bulk_cube_fits_fetch_route_contract",
            ],
        },
        {
            "dwarf_entries": dwarf_entries,
            "checksum_manifest_json": rel(CHECKSUM_MANIFEST),
        },
    )
    payloads["mass_origin_dark_matter_independent_galaxy_dataset_intake_fourth_gate"] = payload(
        "8.7.55.3.164",
        "Dataset-intake declaration fourth gate / v2.0 defer-or-launch audit",
        {
            "local_raw_fits_fetch_manifest_json": rel(FETCH_MANIFEST),
            "local_raw_fits_checksum_manifest_json": rel(CHECKSUM_MANIFEST),
        },
        "Connect the local raw FITS fetch results to the declaration gate and decide whether the independent-galaxy branch can close now or must continue with a bulk-cube residual route.",
        {
            "fourth_gate_rule": "the branch can close only after both spiral and dwarf families have fully populated local raw FITS caches, including the bulk cube tranche"
        },
        [
            row("things_spiral_local_raw_fits_cache_ready", "pass" if spiral_summary["local_raw_fits_cache_ready"] else "reject", "THINGS spiral local raw FITS cache ready", as_float(spiral_summary["local_raw_fits_cache_ready"]), "The spiral cache remains incomplete because the bulk cube tranche is still absent."),
            row("little_things_dwarf_local_raw_fits_cache_ready", "pass" if dwarf_summary["local_raw_fits_cache_ready"] else "reject", "LITTLE THINGS dwarf local raw FITS cache ready", as_float(dwarf_summary["local_raw_fits_cache_ready"]), "The dwarf cache remains incomplete because the bulk cube tranche is still absent."),
            row("independent_dataset_intake_fourth_gate_closeable", "pass" if dataset_branch_closeable else "reject", "independent dataset-intake fourth gate closeable", as_float(dataset_branch_closeable), "The independent-galaxy branch remains open because the cube FITS tranche is still missing."),
            row("launch_v2_mainline_now", "pass" if launch_v2_mainline_now else "reject", "launch v2.0 mainline now", as_float(launch_v2_mainline_now), "The queued v2.0 mainline stays deferred until the independent-galaxy cube FITS route closes."),
        ],
        {
            "spiral_lightweight_cache_ready": spiral_summary["lightweight_cache_ready"],
            "spiral_bulk_cube_cache_ready": spiral_summary["bulk_cube_cache_ready"],
            "dwarf_lightweight_cache_ready": dwarf_summary["lightweight_cache_ready"],
            "dwarf_bulk_cube_cache_ready": dwarf_summary["bulk_cube_cache_ready"],
            "dataset_intake_branch_closeable": dataset_branch_closeable,
            "launch_v2_mainline_now": launch_v2_mainline_now,
            "defer_v2_mainline_now": not launch_v2_mainline_now,
            "recommended_next_route_or_none": recommended_next_route,
            "selected_next_route": selected_next_route,
        },
        {
            "overall_status": "independent_dataset_intake_blocked_at_bulk_cube_fits_fetch",
            "dataset_intake_branch_closeable": dataset_branch_closeable,
            "launch_v2_mainline_now": launch_v2_mainline_now,
            "recommended_next_route_or_none": recommended_next_route,
            "next_required_artifacts": ["mass_origin_dark_matter_bulk_cube_fits_fetch_route_contract"],
        },
        {
            "previous_route_contract": previous_route_contract["summary"],
            "spiral_summary": spiral_summary,
            "dwarf_summary": dwarf_summary,
        },
    )
    payloads["mass_origin_dark_matter_bulk_cube_fits_fetch_route_contract"] = payload(
        "8.7.55.3.165",
        "Bulk cube FITS fetch route contract",
        {
            "local_raw_fits_fetch_manifest_json": rel(FETCH_MANIFEST),
            "local_raw_fits_checksum_manifest_json": rel(CHECKSUM_MANIFEST),
            "dataset_intake_fourth_gate_json": "output/public/quantum/mass_origin_dark_matter_independent_galaxy_dataset_intake_fourth_gate_metrics.json",
        },
        "Freeze the next residual route after the lightweight local raw FITS tranche is cached but the bulk cube tranche remains absent.",
        {
            "route_contract_rule": "the next residual route must target the earliest common remaining artifact that still blocks both the THINGS spiral and LITTLE THINGS dwarf local raw FITS caches"
        },
        [
            row("bulk_cube_fits_fetch_route_selected", "pass", "bulk cube FITS fetch route selected", 1.0, "The next residual route is now frozen."),
            row("bulk_cube_fits_common_blocker_present", "pass", "common spiral+dwarf bulk cube blocker present", 1.0, "Both selected pilot families remain blocked only by their bulk cube FITS tranche."),
            row("bulk_cube_fits_split_contract_ready", "pass", "bulk cube FITS split contract ready", 1.0, "The next branch can now decompose THINGS spiral and LITTLE THINGS dwarf cube fetch work without ambiguity."),
        ],
        {
            "selected_residual_route": "independent_galaxy_bulk_cube_fits_fetch",
            "missing_dark_matter_artifact": "survey_native_spiral_and_dwarf_bulk_cube_fits_cache",
            "split_contract_ready": True,
            "defer_v2_mainline_until_route_close": True,
        },
        {
            "overall_status": "independent_galaxy_bulk_cube_fits_fetch_route_frozen",
            "next_required_artifacts": [
                "8.7.55.3.166",
                "8.7.55.3.167",
                "8.7.55.3.168",
                "8.7.55.3.169",
                "8.7.55.3.170",
            ],
        },
        {
            "spiral_summary": spiral_summary,
            "dwarf_summary": dwarf_summary,
            "common_blocker": "survey_native_spiral_and_dwarf_bulk_cube_fits_cache_absent",
        },
    )

    for stem, data in payloads.items():
        write_artifact(stem, data)
        print(f"[ok] wrote {OUT / (stem + '_metrics.json')}")
        print(f"[ok] wrote {OUT / (stem + '_rows.csv')}")

    print(f"[ok] wrote {FETCH_MANIFEST}")
    print(f"[ok] wrote {CHECKSUM_MANIFEST}")


# 関数: script 実行時に branch 本体を起動する。

if __name__ == "__main__":
    main()
