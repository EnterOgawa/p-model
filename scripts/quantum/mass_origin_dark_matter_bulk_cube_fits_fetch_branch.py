#!/usr/bin/env python3
"""
Freeze the independent-galaxy bulk cube FITS fetch branch for 8.7.55.3.166-.170.

This branch follows the lightweight local-raw FITS fetch audit, which already
cached the moment-map tranche for the selected THINGS spiral and LITTLE THINGS
dwarf pilot subsets but left every multi-GB cube in deferred state.

Its job is to:

1. freeze the THINGS-side bulk cube fetch manifest, staged order, and capacity
   contract,
2. freeze the LITTLE THINGS-side bulk cube fetch manifest, staged order, and
   capacity contract,
3. fetch and checksum the THINGS-side bulk cube tranche into the canonical
   survey-native raw cache,
4. fetch and checksum the LITTLE THINGS-side bulk cube tranche into the same
   canonical cache while updating extraction-reopen readiness, and
5. decide whether the independent-galaxy follow-through can now stand down so
   that the queued v2.0 mainline may start.
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
FETCH_MANIFEST = NON_SPARC_DIR / "pilot_local_raw_fits_fetch_manifest.json"
CHECKSUM_MANIFEST = NON_SPARC_DIR / "pilot_local_raw_fits_checksum_manifest.json"
BULK_FETCH_MANIFEST = NON_SPARC_DIR / "pilot_bulk_cube_fits_fetch_manifest.json"
BULK_CHECKSUM_MANIFEST = NON_SPARC_DIR / "pilot_bulk_cube_fits_checksum_manifest.json"
PREVIOUS_ROUTE_CONTRACT = OUT / "mass_origin_dark_matter_bulk_cube_fits_fetch_route_contract_metrics.json"


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


# 関数: bool を 0/1 の float へ変換する。

def as_float(value: bool) -> float:
    return 1.0 if value else 0.0


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


# 関数: HEAD metadata を取得する。

def remote_head(url: str) -> dict[str, Any]:
    request = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(request, timeout=120) as response:
        length_raw = response.headers.get("Content-Length")
        return {
            "url": url,
            "status": int(getattr(response, "status", 200)),
            "content_length_bytes": int(length_raw) if length_raw else None,
            "last_modified": response.headers.get("Last-Modified"),
            "content_type": response.headers.get("Content-Type"),
            "accept_ranges": response.headers.get("Accept-Ranges"),
        }


# 関数: ファイル名から bulk cube family を判定する。

def is_bulk_cube(entry: dict[str, Any]) -> bool:
    return entry.get("file_role") == "bulk_cube_fits"


# 関数: family entries を size 昇順に並べた staged fetch list へ変換する。

def build_bulk_stage_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sorted_entries = sorted(entries, key=lambda item: int(item["remote_head"]["content_length_bytes"] or 0))
    total_bytes = sum(int(item["remote_head"]["content_length_bytes"] or 0) for item in sorted_entries)
    staged_entries: list[dict[str, Any]] = []
    cumulative = 0
    for index, entry in enumerate(sorted_entries, start=1):
        entry_bytes = int(entry["remote_head"]["content_length_bytes"] or 0)
        cumulative += entry_bytes
        staged = dict(entry)
        staged["stage_order_index"] = index
        staged["expected_bytes"] = entry_bytes
        staged["capacity_fraction_of_family"] = (entry_bytes / total_bytes) if total_bytes else 0.0
        staged["cumulative_bytes_after_stage"] = cumulative
        staged_entries.append(staged)

    return staged_entries


# 関数: family staged entries の summary を集計する。

def summarize_stage(entries: list[dict[str, Any]]) -> dict[str, Any]:
    total_bytes = sum(int(entry["expected_bytes"]) for entry in entries)
    cached_count = sum(1 for entry in entries if entry.get("local_exists"))
    cached_bytes = sum(int(entry.get("local_size_bytes") or 0) for entry in entries if entry.get("local_exists"))
    return {
        "entry_count": len(entries),
        "total_remote_bytes": total_bytes,
        "cached_count": cached_count,
        "cached_bytes": cached_bytes,
        "cache_ready": cached_count == len(entries),
    }


# 関数: Range 再開付きで remote file を local cache へ取得する。

def fetch_with_resume(url: str, target: Path, expected_size: int | None) -> dict[str, Any]:
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target.with_suffix(target.suffix + ".part")
    existing_target_size = target.stat().st_size if target.exists() else 0

    if target.exists() and expected_size is not None and existing_target_size == expected_size:
        return {
            "fetch_status": "reused_existing_file",
            "local_exists": True,
            "local_size_bytes": int(existing_target_size),
            "local_sha256": sha256_file(target),
        }

    resume_from = temp_path.stat().st_size if temp_path.exists() else 0
    if expected_size is not None and resume_from > expected_size:
        temp_path.unlink()
        resume_from = 0

    if resume_from == 0 and target.exists():
        target.unlink()

    headers: dict[str, str] = {}
    if resume_from > 0:
        headers["Range"] = f"bytes={resume_from}-"

    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=600) as response:
        response_status = int(getattr(response, "status", 200))
        if response_status == 200 and resume_from > 0:
            temp_path.unlink(missing_ok=True)
            resume_from = 0

        write_mode = "ab" if resume_from > 0 else "wb"
        bytes_written = resume_from
        with temp_path.open(write_mode) as handle:
            while True:
                chunk = response.read(8 * 1024 * 1024)
                if not chunk:
                    break

                handle.write(chunk)
                bytes_written += len(chunk)

    if expected_size is not None and bytes_written != expected_size:
        raise RuntimeError(f"download size mismatch for {url}: expected {expected_size}, got {bytes_written}")

    shutil.move(str(temp_path), str(target))
    return {
        "fetch_status": "downloaded_now" if resume_from == 0 else "resumed_and_downloaded_now",
        "local_exists": True,
        "local_size_bytes": int(target.stat().st_size),
        "local_sha256": sha256_file(target),
    }


# 関数: bulk cube staged entries を順に取得し checksum を埋める。

def fetch_stage(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fetched_entries: list[dict[str, Any]] = []
    for entry in entries:
        updated = dict(entry)
        target = ROOT / entry["local_path"]
        expected_size = int(entry["expected_bytes"])
        fetch_result = fetch_with_resume(entry["remote_url"], target, expected_size)
        updated.update(fetch_result)
        fetched_entries.append(updated)

    return fetched_entries


# 関数: family 名で fetch-manifest entries を抽出する。

def family_entries(fetch_manifest: dict[str, Any], family_key: str) -> list[dict[str, Any]]:
    return [entry for entry in fetch_manifest[family_key] if is_bulk_cube(entry)]


# 関数: canonical local raw manifests を bulk cube 取得後の状態へ同期する。

def sync_canonical_local_manifests(
    fetch_manifest: dict[str, Any],
    checksum_manifest: dict[str, Any],
    fetched_entries: dict[tuple[str, str, str], dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    # 関数: `sync_entry` の入出力契約と処理意図を定義する。
    def sync_entry(entry: dict[str, Any]) -> dict[str, Any]:
        key = (entry["family"], entry["galaxy_name"], entry["file_key"])
        if key not in fetched_entries:
            return entry

        updated = dict(entry)
        fetched = fetched_entries[key]
        updated["fetch_status"] = fetched["fetch_status"]
        updated["local_exists"] = fetched["local_exists"]
        updated["local_size_bytes"] = fetched["local_size_bytes"]
        updated["local_sha256"] = fetched["local_sha256"]
        return updated

    updated_fetch_manifest = dict(fetch_manifest)
    updated_fetch_manifest["generated_utc"] = now_iso()
    updated_fetch_manifest["spiral_family"] = [sync_entry(entry) for entry in fetch_manifest["spiral_family"]]
    updated_fetch_manifest["dwarf_family"] = [sync_entry(entry) for entry in fetch_manifest["dwarf_family"]]
    updated_fetch_manifest["notes"] = [
        "Lightweight moment-map FITS and bulk cube FITS are now tracked together in the canonical non-SPARC raw cache manifest.",
        "Every selected pilot family now carries remote metadata, local path, fetch status, and sha256-ready local cache state for both moment maps and cubes.",
    ]

    updated_checksum_manifest = dict(checksum_manifest)
    updated_checksum_manifest["generated_utc"] = now_iso()
    updated_checksum_manifest["spiral_family"] = [sync_entry(entry) for entry in checksum_manifest["spiral_family"]]
    updated_checksum_manifest["dwarf_family"] = [sync_entry(entry) for entry in checksum_manifest["dwarf_family"]]
    updated_checksum_manifest["notes"] = [
        "sha256 is now populated for both the lightweight moment-map tranche and the bulk cube tranche of the selected pilot subset.",
    ]
    return updated_fetch_manifest, updated_checksum_manifest


# 関数: branch 全体を実行して artifact を生成する。

def main() -> None:
    for path in (FETCH_MANIFEST, CHECKSUM_MANIFEST, PREVIOUS_ROUTE_CONTRACT):
        req(path)

    fetch_manifest = read_json(FETCH_MANIFEST)
    checksum_manifest = read_json(CHECKSUM_MANIFEST)
    previous_route_contract = read_json(PREVIOUS_ROUTE_CONTRACT)

    spiral_stage_entries = build_bulk_stage_entries(family_entries(fetch_manifest, "spiral_family"))
    dwarf_stage_entries = build_bulk_stage_entries(family_entries(fetch_manifest, "dwarf_family"))
    spiral_stage_summary = summarize_stage(spiral_stage_entries)
    dwarf_stage_summary = summarize_stage(dwarf_stage_entries)

    bulk_fetch_manifest = {
        "generated_utc": now_iso(),
        "registry_name": "independent_galaxy_bulk_cube_fits_fetch_manifest",
        "derived_from": {
            "local_raw_fits_fetch_manifest": rel(FETCH_MANIFEST),
            "previous_route_contract": rel(PREVIOUS_ROUTE_CONTRACT),
        },
        "spiral_family": spiral_stage_entries,
        "dwarf_family": dwarf_stage_entries,
        "notes": [
            "The staged order sorts by remote cube size so that smaller cubes close first while the largest THINGS and LITTLE THINGS cubes remain explicit.",
            "This manifest is the canonical capacity and fetch-order contract for the non-SPARC bulk cube tranche.",
        ],
    }
    write_manifest(BULK_FETCH_MANIFEST, bulk_fetch_manifest)

    spiral_fetched_entries = fetch_stage(spiral_stage_entries)
    dwarf_fetched_entries = fetch_stage(dwarf_stage_entries)
    fetched_lookup = {
        (entry["family"], entry["galaxy_name"], entry["file_key"]): entry
        for entry in spiral_fetched_entries + dwarf_fetched_entries
    }

    updated_fetch_manifest, updated_checksum_manifest = sync_canonical_local_manifests(
        fetch_manifest,
        checksum_manifest,
        fetched_lookup,
    )
    write_manifest(FETCH_MANIFEST, updated_fetch_manifest)
    write_manifest(CHECKSUM_MANIFEST, updated_checksum_manifest)

    bulk_checksum_manifest = {
        "generated_utc": now_iso(),
        "registry_name": "independent_galaxy_bulk_cube_fits_checksum_manifest",
        "derived_from": {
            "bulk_cube_fits_fetch_manifest": rel(BULK_FETCH_MANIFEST),
            "local_raw_fits_checksum_manifest": rel(CHECKSUM_MANIFEST),
        },
        "spiral_family": spiral_fetched_entries,
        "dwarf_family": dwarf_fetched_entries,
        "notes": [
            "sha256 is populated for every selected THINGS spiral and LITTLE THINGS dwarf bulk cube FITS target.",
        ],
    }
    write_manifest(BULK_CHECKSUM_MANIFEST, bulk_checksum_manifest)

    spiral_fetched_summary = summarize_stage(spiral_fetched_entries)
    dwarf_fetched_summary = summarize_stage(dwarf_fetched_entries)
    extraction_reopen_ready = spiral_fetched_summary["cache_ready"] and dwarf_fetched_summary["cache_ready"]
    dataset_intake_branch_closeable = extraction_reopen_ready
    launch_v2_mainline_now = extraction_reopen_ready
    selected_next_route = "8.7.56.1" if launch_v2_mainline_now else "independent_galaxy_bulk_cube_fits_fetch"
    recommended_next_route = "8.7.56.1" if launch_v2_mainline_now else "8.7.55.3.165"

    payloads: dict[str, dict[str, Any]] = {}
    payloads["mass_origin_dark_matter_things_spiral_bulk_cube_fits_fetch_manifest_freeze"] = payload(
        "8.7.55.3.166",
        "THINGS spiral bulk cube FITS fetch manifest freeze",
        {
            "local_raw_fits_fetch_manifest_json": rel(FETCH_MANIFEST),
            "previous_route_contract_json": rel(PREVIOUS_ROUTE_CONTRACT),
        },
        "Freeze the THINGS-side bulk cube fetch order, expected byte counts, and staged capacity contract for the selected spiral pilot subset.",
        {
            "things_spiral_bulk_cube_manifest_rule": "the THINGS-side bulk cube manifest is ready after every selected spiral cube has remote metadata, staged order, expected bytes, and a canonical local cache target"
        },
        [
            row("things_spiral_bulk_cube_entries_ready", "pass", "THINGS spiral bulk cube entries ready", as_float(spiral_stage_summary["entry_count"] > 0), "The selected spiral pilot subset has a complete bulk cube tranche manifest."),
            row("things_spiral_bulk_cube_capacity_contract_ready", "pass", "THINGS spiral bulk cube capacity contract ready", as_float(spiral_stage_summary["total_remote_bytes"] > 0), "Every THINGS spiral cube now contributes an expected byte count to the staged fetch plan."),
            row("things_spiral_bulk_cube_manifest_written", "pass", "THINGS spiral bulk cube manifest written", 1.0, "The bulk cube fetch manifest was saved to the canonical non-SPARC store."),
        ],
        {
            "selected_spiral_pilot_subset": sorted({entry["galaxy_name"] for entry in spiral_stage_entries}),
            "required_things_spiral_bulk_cube_entry_count": spiral_stage_summary["entry_count"],
            "things_spiral_bulk_cube_total_remote_bytes": spiral_stage_summary["total_remote_bytes"],
            "things_spiral_bulk_cube_manifest_json": rel(BULK_FETCH_MANIFEST),
            "largest_things_spiral_cube_galaxy": max(spiral_stage_entries, key=lambda item: int(item["expected_bytes"]))["galaxy_name"] if spiral_stage_entries else None,
            "largest_things_spiral_cube_bytes": max((int(item["expected_bytes"]) for item in spiral_stage_entries), default=0),
            "first_route_to_close_or_none": "things_spiral_bulk_cube_staged_fetch",
        },
        {
            "overall_status": "things_spiral_bulk_cube_fetch_manifest_frozen",
            "things_spiral_bulk_cube_manifest_ready": True,
            "next_required_artifacts": [
                "mass_origin_dark_matter_little_things_dwarf_bulk_cube_fits_fetch_manifest_freeze",
                "mass_origin_dark_matter_things_spiral_bulk_cube_staged_fetch_checksum_manifest",
            ],
        },
        {
            "spiral_stage_entries": spiral_stage_entries,
            "previous_route_contract": previous_route_contract["summary"],
        },
    )
    payloads["mass_origin_dark_matter_little_things_dwarf_bulk_cube_fits_fetch_manifest_freeze"] = payload(
        "8.7.55.3.167",
        "LITTLE THINGS dwarf bulk cube FITS fetch manifest freeze",
        {
            "local_raw_fits_fetch_manifest_json": rel(FETCH_MANIFEST),
            "bulk_cube_fits_fetch_manifest_json": rel(BULK_FETCH_MANIFEST),
        },
        "Freeze the LITTLE THINGS-side bulk cube fetch order, expected byte counts, and staged capacity contract for the selected dwarf pilot subset.",
        {
            "little_things_dwarf_bulk_cube_manifest_rule": "the LITTLE THINGS-side bulk cube manifest is ready after every selected dwarf cube has remote metadata, staged order, expected bytes, and a canonical local cache target"
        },
        [
            row("little_things_dwarf_bulk_cube_entries_ready", "pass", "LITTLE THINGS dwarf bulk cube entries ready", as_float(dwarf_stage_summary["entry_count"] > 0), "The selected dwarf pilot subset has a complete bulk cube tranche manifest."),
            row("little_things_dwarf_bulk_cube_capacity_contract_ready", "pass", "LITTLE THINGS dwarf bulk cube capacity contract ready", as_float(dwarf_stage_summary["total_remote_bytes"] > 0), "Every LITTLE THINGS dwarf cube now contributes an expected byte count to the staged fetch plan."),
            row("little_things_dwarf_bulk_cube_manifest_written", "pass", "LITTLE THINGS dwarf bulk cube manifest written", 1.0, "The bulk cube fetch manifest was saved to the canonical non-SPARC store."),
        ],
        {
            "selected_dwarf_pilot_subset": sorted({entry["galaxy_name"] for entry in dwarf_stage_entries}),
            "required_little_things_dwarf_bulk_cube_entry_count": dwarf_stage_summary["entry_count"],
            "little_things_dwarf_bulk_cube_total_remote_bytes": dwarf_stage_summary["total_remote_bytes"],
            "little_things_dwarf_bulk_cube_manifest_json": rel(BULK_FETCH_MANIFEST),
            "largest_little_things_dwarf_cube_galaxy": max(dwarf_stage_entries, key=lambda item: int(item["expected_bytes"]))["galaxy_name"] if dwarf_stage_entries else None,
            "largest_little_things_dwarf_cube_bytes": max((int(item["expected_bytes"]) for item in dwarf_stage_entries), default=0),
            "first_route_to_close_or_none": "little_things_dwarf_bulk_cube_staged_fetch",
        },
        {
            "overall_status": "little_things_dwarf_bulk_cube_fetch_manifest_frozen",
            "little_things_dwarf_bulk_cube_manifest_ready": True,
            "next_required_artifacts": [
                "mass_origin_dark_matter_things_spiral_bulk_cube_staged_fetch_checksum_manifest",
                "mass_origin_dark_matter_little_things_dwarf_bulk_cube_staged_fetch_checksum_manifest",
            ],
        },
        {
            "dwarf_stage_entries": dwarf_stage_entries,
        },
    )
    payloads["mass_origin_dark_matter_things_spiral_bulk_cube_staged_fetch_checksum_manifest"] = payload(
        "8.7.55.3.168",
        "THINGS spiral bulk cube staged fetch / checksum manifest",
        {
            "bulk_cube_fits_fetch_manifest_json": rel(BULK_FETCH_MANIFEST),
            "local_raw_fits_fetch_manifest_json": rel(FETCH_MANIFEST),
        },
        "Fetch the staged THINGS spiral bulk cube tranche into the canonical cache and freeze the corresponding checksum manifest.",
        {
            "things_spiral_bulk_cube_fetch_rule": "the THINGS-side bulk cube tranche is ready after every staged cube has a local file, byte match, and sha256 digest in the canonical checksum manifest"
        },
        [
            row("things_spiral_bulk_cube_fetch_complete", "pass" if spiral_fetched_summary["cache_ready"] else "reject", "THINGS spiral bulk cube fetch complete", as_float(spiral_fetched_summary["cache_ready"]), "Every selected THINGS spiral bulk cube should now exist in the canonical non-SPARC raw cache."),
            row("things_spiral_bulk_cube_checksum_complete", "pass" if spiral_fetched_summary["cache_ready"] else "reject", "THINGS spiral bulk cube checksum complete", as_float(spiral_fetched_summary["cache_ready"]), "Every selected THINGS spiral bulk cube now carries a sha256 digest."),
            row("things_spiral_local_raw_fits_cache_ready", "pass" if spiral_fetched_summary["cache_ready"] else "reject", "THINGS spiral full local raw FITS cache ready", as_float(spiral_fetched_summary["cache_ready"]), "The spiral family remains blocked until every THINGS cube is local and hashed."),
        ],
        {
            "things_spiral_bulk_cube_cached_count": spiral_fetched_summary["cached_count"],
            "things_spiral_bulk_cube_entry_count": spiral_fetched_summary["entry_count"],
            "things_spiral_bulk_cube_cached_bytes": spiral_fetched_summary["cached_bytes"],
            "things_spiral_bulk_cube_total_remote_bytes": spiral_fetched_summary["total_remote_bytes"],
            "things_spiral_bulk_cube_checksum_manifest_json": rel(BULK_CHECKSUM_MANIFEST),
            "things_spiral_full_local_raw_fits_cache_ready": spiral_fetched_summary["cache_ready"],
            "remaining_blocker_or_none": None if spiral_fetched_summary["cache_ready"] else "things_spiral_bulk_cube_fits_cache_absent",
        },
        {
            "overall_status": "things_spiral_bulk_cube_tranche_cached",
            "things_spiral_full_local_raw_fits_cache_ready": spiral_fetched_summary["cache_ready"],
            "next_required_artifacts": [
                "mass_origin_dark_matter_little_things_dwarf_bulk_cube_staged_fetch_checksum_manifest",
                "mass_origin_dark_matter_independent_galaxy_dataset_intake_fifth_gate",
            ],
        },
        {
            "spiral_fetched_entries": spiral_fetched_entries,
        },
    )
    payloads["mass_origin_dark_matter_little_things_dwarf_bulk_cube_staged_fetch_checksum_manifest"] = payload(
        "8.7.55.3.169",
        "LITTLE THINGS dwarf bulk cube staged fetch / checksum manifest / extraction reopen readiness",
        {
            "bulk_cube_fits_fetch_manifest_json": rel(BULK_FETCH_MANIFEST),
            "bulk_cube_fits_checksum_manifest_json": rel(BULK_CHECKSUM_MANIFEST),
        },
        "Fetch the staged LITTLE THINGS dwarf bulk cube tranche into the canonical cache, freeze the checksum manifest, and update extraction-reopen readiness.",
        {
            "little_things_dwarf_bulk_cube_fetch_rule": "the LITTLE THINGS-side bulk cube tranche is ready after every staged cube has a local file, byte match, and sha256 digest in the canonical checksum manifest"
        },
        [
            row("little_things_dwarf_bulk_cube_fetch_complete", "pass" if dwarf_fetched_summary["cache_ready"] else "reject", "LITTLE THINGS dwarf bulk cube fetch complete", as_float(dwarf_fetched_summary["cache_ready"]), "Every selected LITTLE THINGS dwarf bulk cube should now exist in the canonical non-SPARC raw cache."),
            row("little_things_dwarf_bulk_cube_checksum_complete", "pass" if dwarf_fetched_summary["cache_ready"] else "reject", "LITTLE THINGS dwarf bulk cube checksum complete", as_float(dwarf_fetched_summary["cache_ready"]), "Every selected LITTLE THINGS dwarf bulk cube now carries a sha256 digest."),
            row("independent_survey_native_extraction_reopen_ready", "pass" if extraction_reopen_ready else "reject", "independent survey-native extraction reopen ready", as_float(extraction_reopen_ready), "Survey-native extraction can reopen once both spiral and dwarf bulk cube caches are fully populated."),
        ],
        {
            "little_things_dwarf_bulk_cube_cached_count": dwarf_fetched_summary["cached_count"],
            "little_things_dwarf_bulk_cube_entry_count": dwarf_fetched_summary["entry_count"],
            "little_things_dwarf_bulk_cube_cached_bytes": dwarf_fetched_summary["cached_bytes"],
            "little_things_dwarf_bulk_cube_total_remote_bytes": dwarf_fetched_summary["total_remote_bytes"],
            "little_things_dwarf_bulk_cube_checksum_manifest_json": rel(BULK_CHECKSUM_MANIFEST),
            "little_things_dwarf_full_local_raw_fits_cache_ready": dwarf_fetched_summary["cache_ready"],
            "independent_survey_native_extraction_reopen_ready": extraction_reopen_ready,
            "remaining_blocker_or_none": None if dwarf_fetched_summary["cache_ready"] else "little_things_dwarf_bulk_cube_fits_cache_absent",
        },
        {
            "overall_status": "little_things_dwarf_bulk_cube_tranche_cached",
            "little_things_dwarf_full_local_raw_fits_cache_ready": dwarf_fetched_summary["cache_ready"],
            "independent_survey_native_extraction_reopen_ready": extraction_reopen_ready,
            "next_required_artifacts": [
                "mass_origin_dark_matter_independent_galaxy_dataset_intake_fifth_gate",
            ],
        },
        {
            "dwarf_fetched_entries": dwarf_fetched_entries,
        },
    )
    payloads["mass_origin_dark_matter_independent_galaxy_dataset_intake_fifth_gate"] = payload(
        "8.7.55.3.170",
        "Dataset-intake declaration fifth gate / v2.0 defer-or-launch audit",
        {
            "bulk_cube_fits_fetch_manifest_json": rel(BULK_FETCH_MANIFEST),
            "bulk_cube_fits_checksum_manifest_json": rel(BULK_CHECKSUM_MANIFEST),
            "local_raw_fits_fetch_manifest_json": rel(FETCH_MANIFEST),
            "local_raw_fits_checksum_manifest_json": rel(CHECKSUM_MANIFEST),
        },
        "Connect the bulk cube fetch results to the declaration gate and decide whether the independent-galaxy follow-through can stand down so that the queued v2.0 mainline may start.",
        {
            "fifth_gate_rule": "the follow-through can stand down once the selected THINGS spiral and LITTLE THINGS dwarf pilot subsets both have fully populated local raw FITS caches, including their bulk cube tranche"
        },
        [
            row("things_spiral_full_local_raw_fits_cache_ready", "pass" if spiral_fetched_summary["cache_ready"] else "reject", "THINGS spiral full local raw FITS cache ready", as_float(spiral_fetched_summary["cache_ready"]), "The spiral side is ready only after all selected THINGS cubes are local and hashed."),
            row("little_things_dwarf_full_local_raw_fits_cache_ready", "pass" if dwarf_fetched_summary["cache_ready"] else "reject", "LITTLE THINGS dwarf full local raw FITS cache ready", as_float(dwarf_fetched_summary["cache_ready"]), "The dwarf side is ready only after all selected LITTLE THINGS cubes are local and hashed."),
            row("independent_dataset_intake_fifth_gate_closeable", "pass" if dataset_intake_branch_closeable else "reject", "independent dataset-intake fifth gate closeable", as_float(dataset_intake_branch_closeable), "The follow-through can stand down only after both pilot families have their full local raw FITS cache."),
            row("launch_v2_mainline_now", "pass" if launch_v2_mainline_now else "reject", "launch v2.0 mainline now", as_float(launch_v2_mainline_now), "The queued v2.0 mainline may start once the independent-galaxy follow-through no longer has an open raw-cache blocker."),
        ],
        {
            "spiral_full_local_raw_fits_cache_ready": spiral_fetched_summary["cache_ready"],
            "dwarf_full_local_raw_fits_cache_ready": dwarf_fetched_summary["cache_ready"],
            "survey_native_extraction_reopen_ready": extraction_reopen_ready,
            "dataset_intake_branch_closeable": dataset_intake_branch_closeable,
            "launch_v2_mainline_now": launch_v2_mainline_now,
            "defer_v2_mainline_now": not launch_v2_mainline_now,
            "recommended_next_route_or_none": recommended_next_route,
            "selected_next_route": selected_next_route,
        },
        {
            "overall_status": "independent_dataset_intake_bulk_cube_blocker_closed" if dataset_intake_branch_closeable else "independent_dataset_intake_still_blocked_at_bulk_cube_fits_fetch",
            "dataset_intake_branch_closeable": dataset_intake_branch_closeable,
            "launch_v2_mainline_now": launch_v2_mainline_now,
            "recommended_next_route_or_none": recommended_next_route,
            "next_required_artifacts": ["8.7.56.1"] if launch_v2_mainline_now else ["8.7.55.3.165"],
        },
        {
            "previous_route_contract": previous_route_contract["summary"],
            "spiral_fetched_summary": spiral_fetched_summary,
            "dwarf_fetched_summary": dwarf_fetched_summary,
        },
    )

    for stem, data in payloads.items():
        write_artifact(stem, data)
        print(f"[ok] wrote {OUT / (stem + '_metrics.json')}")
        print(f"[ok] wrote {OUT / (stem + '_rows.csv')}")

    print(f"[ok] wrote {FETCH_MANIFEST}")
    print(f"[ok] wrote {CHECKSUM_MANIFEST}")
    print(f"[ok] wrote {BULK_FETCH_MANIFEST}")
    print(f"[ok] wrote {BULK_CHECKSUM_MANIFEST}")


# 関数: script 実行時に branch 本体を起動する。

if __name__ == "__main__":
    main()
