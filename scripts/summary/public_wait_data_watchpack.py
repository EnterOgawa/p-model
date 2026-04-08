"""
public_wait_data_watchpack.py

Phase 8 / Step 8.7.17:
Bundle low-priority public-wait or blocked-primary-data items into one
machine-readable watchpack. The script only reads canonical local artifacts
and emits JSON/CSV snapshots for roadmap tracking.

Inputs:
- output/private/cosmology/jwst_spectra_release_waitlist.json
- output/public/quantum/bell_selection_sensitivity_summary.json
- data/quantum/sources/giustina2015_prl115_250401/manifest.json

Outputs:
- output/public/summary/public_wait_data_watchpack.json
- output/public/summary/public_wait_data_watchpack.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence


ROOT = Path(__file__).resolve().parents[2]


# Function: Return the current UTC timestamp in ISO 8601 form.
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# Function: Read a UTF-8 JSON file into a dictionary payload.

def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


# Function: Write a dictionary payload as stable UTF-8 JSON.

def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


# Function: Compute an uppercase SHA256 digest for raw bytes.

def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


# Function: Describe file existence and change signatures for watchpack inputs.

def _file_signature(path: Path) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "path": str(path.resolve()).replace("\\", "/"),
        "exists": path.exists(),
        "size_bytes": None,
        "mtime_utc": None,
        "sha256": None,
    }
    if not path.exists():
        return payload

    stat = path.stat()
    payload["size_bytes"] = int(stat.st_size)
    payload["mtime_utc"] = datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat()
    payload["sha256"] = _sha256_bytes(path.read_bytes())
    return payload


# Function: Load the previous watchpack diagnostics to support no-change tracking.

def _load_previous_watchpack(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}

    try:
        payload = _read_json(path)
    except Exception:
        return {}

    diagnostics = payload.get("diagnostics")
    if not isinstance(diagnostics, dict):
        return {}

    watchpack = diagnostics.get("public_wait_update_watchpack")
    if not isinstance(watchpack, dict):
        return {}

    return watchpack


# Function: Parse an ISO 8601 UTC string into a datetime when possible.

def _parse_utc_iso(text: str | None) -> Optional[datetime]:
    if not text:
        return None

    normalized = str(text).strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"

    try:
        return datetime.fromisoformat(normalized).astimezone(timezone.utc)
    except Exception:
        return None


# Function: Compute days remaining until a UTC target time.

def _days_until(now_utc: datetime, target_utc: str | None) -> Optional[float]:
    target = _parse_utc_iso(target_utc)
    if target is None:
        return None

    return float((target - now_utc).total_seconds() / 86400.0)


# Class: Represent one blocked or waiting public-data item in the watchpack.

@dataclass(frozen=True)
class WaitRow:
    item_id: str
    domain: str
    item_label: str
    wait_kind: str
    status: str
    ready_now: bool
    blocked_priority_class: str
    detail: str
    next_action: str
    next_release_utc: str | None
    days_until_release: float | None
    source_artifact: str
    source_manifest: str | None
    expected_location: str | None
    integration_target: str


# Function: Convert the JWST release waitlist into normalized watchpack rows.

def _load_jwst_rows(waitlist_path: Path, now_utc: datetime) -> list[WaitRow]:
    if not waitlist_path.exists():
        return []

    payload = _read_json(waitlist_path)
    blocked_targets = payload.get("blocked_targets")
    if not isinstance(blocked_targets, list):
        return []

    rows: list[WaitRow] = []
    for target in blocked_targets:
        if not isinstance(target, dict):
            continue

        slug = str(target.get("target_slug") or "").strip()
        label = str(target.get("target") or slug or "unknown_target").strip()
        next_release_utc = str(target.get("next_release_utc") or "").strip() or None
        days_until_release = _days_until(now_utc, next_release_utc)
        rows.append(
            WaitRow(
                item_id=f"jwst::{slug or label.lower().replace(' ', '_')}",
                domain="cosmology",
                item_label=label,
                wait_kind="future_release_date",
                status="blocked",
                ready_now=False,
                blocked_priority_class="low_blocked_not_released_yet",
                detail=(
                    "JWST/MAST spectroscopy is still under proprietary hold or has a future release date "
                    "in the cached manifest."
                ),
                next_action="rerun_jwst_release_waitlist_after_manifest_or_release_update",
                next_release_utc=next_release_utc,
                days_until_release=days_until_release,
                source_artifact=str(waitlist_path.resolve()).replace("\\", "/"),
                source_manifest=None,
                expected_location=None,
                integration_target="Phase 4 / Step 4.6 JWST spectra follow-up",
            )
        )

    return rows


# Function: Convert the Giustina blocked-primary-data state into one normalized row.

def _load_giustina_rows(summary_path: Path, manifest_path: Path) -> list[WaitRow]:
    summary_payload = _read_json(summary_path) if summary_path.exists() else {}
    manifest_payload = _read_json(manifest_path) if manifest_path.exists() else {}

    giustina_summary = summary_payload.get("giustina2015")
    if not isinstance(giustina_summary, dict):
        giustina_summary = {}

    primary_requirements = manifest_payload.get("primary_data_requirements")
    if not isinstance(primary_requirements, dict):
        primary_requirements = {}

    expected_location = str(primary_requirements.get("expected_location") or "").strip() or None
    expected_path = Path(expected_location) if expected_location else None
    click_log_ready = bool(expected_path and expected_path.exists())

    status = "ready" if click_log_ready else str(giustina_summary.get("status") or primary_requirements.get("status") or "blocked")
    ready_now = status == "ready"
    blocked_priority_class = "ready" if ready_now else "low_blocked_missing_primary_data"
    reason = str(
        giustina_summary.get("reason")
        or primary_requirements.get("fallback_if_unavailable", {}).get("rationale")
        or "Primary click logs are not available in the local canonical source set."
    ).strip()

    next_action = (
        "run_bell_primary_products_then_refresh_steps_8_7_12_to_8_7_16"
        if ready_now
        else "wait_for_public_click_logs_or_manual_primary_delivery_then_rerun_step_8_7_17"
    )

    return [
        WaitRow(
            item_id="quantum::giustina2015_click_logs",
            domain="quantum",
            item_label="Giustina 2015 click logs",
            wait_kind="missing_primary_data",
            status=status,
            ready_now=ready_now,
            blocked_priority_class=blocked_priority_class,
            detail=reason,
            next_action=next_action,
            next_release_utc=None,
            days_until_release=None,
            source_artifact=str(summary_path.resolve()).replace("\\", "/"),
            source_manifest=str(manifest_path.resolve()).replace("\\", "/") if manifest_path.exists() else None,
            expected_location=str(expected_path.resolve()).replace("\\", "/") if expected_path else expected_location,
            integration_target="Phase 7 / Step 7.4 and Phase 8 / Steps 8.7.12-8.7.16 Bell reanalysis",
        )
    ]


# Function: Reduce rows to the state fields that matter for change detection.

def _normalized_state(rows: Sequence[WaitRow]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        normalized.append(
            {
                "item_id": row.item_id,
                "status": row.status,
                "ready_now": row.ready_now,
                "wait_kind": row.wait_kind,
                "next_release_utc": row.next_release_utc,
                "blocked_priority_class": row.blocked_priority_class,
            }
        )

    return sorted(normalized, key=lambda item: str(item["item_id"]))


# Function: Compare current and previous states and derive a stable watch/update event.

def _derive_update_watchpack(
    *,
    current_input_signatures: dict[str, dict[str, Any]],
    previous_watchpack: dict[str, Any],
    rows: Sequence[WaitRow],
) -> dict[str, Any]:
    previous_input_signatures = (
        previous_watchpack.get("input_signatures")
        if isinstance(previous_watchpack.get("input_signatures"), dict)
        else {}
    )
    previous_state_signature = str(previous_watchpack.get("blocked_state_signature") or "").strip().upper()

    normalized_rows = _normalized_state(rows)
    current_state_signature = _sha256_bytes(
        json.dumps(normalized_rows, ensure_ascii=True, sort_keys=True).encode("utf-8")
    )

    input_hash_changed = False
    input_metadata_changed_without_hash_change = False
    baseline_initialized_now = not previous_input_signatures
    for key, current_signature in current_input_signatures.items():
        previous_signature = previous_input_signatures.get(key, {}) if isinstance(previous_input_signatures, dict) else {}
        current_exists = bool(current_signature.get("exists"))
        previous_exists = bool(previous_signature.get("exists"))
        current_sha = str(current_signature.get("sha256") or "").strip().upper()
        previous_sha = str(previous_signature.get("sha256") or "").strip().upper()

        if current_exists and previous_exists and current_sha and previous_sha and current_sha != previous_sha:
            input_hash_changed = True

        current_mtime = str(current_signature.get("mtime_utc") or "").strip()
        previous_mtime = str(previous_signature.get("mtime_utc") or "").strip()
        current_size = current_signature.get("size_bytes")
        previous_size = previous_signature.get("size_bytes")
        if current_exists and previous_exists and current_sha == previous_sha:
            if (current_mtime and previous_mtime and current_mtime != previous_mtime) or (current_size != previous_size):
                input_metadata_changed_without_hash_change = True

    blocked_state_changed = bool(previous_state_signature) and current_state_signature != previous_state_signature
    current_rows_by_id = {row.item_id: row for row in rows}
    previous_rows_by_id = {
        str(item.get("item_id")): item
        for item in (previous_watchpack.get("rows_state") or [])
        if isinstance(item, dict) and item.get("item_id") is not None
    }

    newly_ready_item_ids: list[str] = []
    newly_blocked_item_ids: list[str] = []
    for item_id, row in current_rows_by_id.items():
        previous_item = previous_rows_by_id.get(item_id, {})
        previous_ready = bool(previous_item.get("ready_now")) if previous_item else False
        previous_status = str(previous_item.get("status") or "").strip()
        if row.ready_now and ((not previous_ready) or previous_status != "ready"):
            newly_ready_item_ids.append(item_id)

        if (not row.ready_now) and previous_item and previous_ready:
            newly_blocked_item_ids.append(item_id)

    update_event_detected = blocked_state_changed
    if baseline_initialized_now:
        update_event_type = "baseline_initialized"
    elif newly_ready_item_ids:
        update_event_type = "blocked_item_released_or_received"
    elif newly_blocked_item_ids:
        update_event_type = "item_returned_to_blocked"
    elif blocked_state_changed:
        update_event_type = "blocked_state_changed"
    elif input_hash_changed:
        update_event_type = "input_hash_changed_state_same"
    elif input_metadata_changed_without_hash_change:
        update_event_type = "metadata_changed_hash_same"
    else:
        update_event_type = "no_change"

    event_counter_prev = int(previous_watchpack.get("event_counter", 0)) if previous_watchpack else 0
    event_counter = event_counter_prev + 1 if update_event_detected else event_counter_prev
    blocked_items_n = sum(not row.ready_now for row in rows)
    ready_items_n = sum(row.ready_now for row in rows)

    if newly_ready_item_ids:
        next_action = "integrate_newly_available_primary_data_now"
    elif blocked_items_n > 0:
        next_action = "keep_low_priority_watch_and_rerun_on_release_or_primary_data_update"
    else:
        next_action = "none"

    return {
        "input_signatures": current_input_signatures,
        "previous_input_signatures": previous_input_signatures,
        "rows_state": normalized_rows,
        "blocked_state_signature": current_state_signature,
        "previous_blocked_state_signature": previous_state_signature or None,
        "baseline_initialized_now": baseline_initialized_now,
        "input_hash_changed": input_hash_changed,
        "input_metadata_changed_without_hash_change": input_metadata_changed_without_hash_change,
        "blocked_state_changed": blocked_state_changed,
        "update_event_detected": update_event_detected,
        "update_event_type": update_event_type,
        "event_counter": event_counter,
        "blocked_items_n": blocked_items_n,
        "ready_items_n": ready_items_n,
        "newly_ready_item_ids": newly_ready_item_ids,
        "newly_blocked_item_ids": newly_blocked_item_ids,
        "next_action": next_action,
        "note": (
            "Event counter increments only when the logical blocked-state set changes. "
            "Hash-only refreshes are logged without incrementing the counter."
        ),
    }


# Function: Write normalized watch rows into a flat CSV file.

def _write_csv(path: Path, rows: Sequence[WaitRow]) -> None:
    fieldnames = [
        "item_id",
        "domain",
        "item_label",
        "wait_kind",
        "status",
        "ready_now",
        "blocked_priority_class",
        "detail",
        "next_action",
        "next_release_utc",
        "days_until_release",
        "source_artifact",
        "source_manifest",
        "expected_location",
        "integration_target",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


# Function: Parse CLI arguments, build the watchpack, and write outputs.

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Build the Phase 8 / Step 8.7.17 public-wait data watchpack.")
    ap.add_argument(
        "--step-tag",
        default="8.7.17.1",
        help="Step tag recorded in the watchpack output (default: 8.7.17.1).",
    )
    ap.add_argument(
        "--jwst-waitlist",
        default=str(ROOT / "output" / "private" / "cosmology" / "jwst_spectra_release_waitlist.json"),
        help="Canonical JWST release-waitlist JSON.",
    )
    ap.add_argument(
        "--bell-summary",
        default=str(ROOT / "output" / "public" / "quantum" / "bell_selection_sensitivity_summary.json"),
        help="Canonical Bell selection-sensitivity summary JSON.",
    )
    ap.add_argument(
        "--giustina-manifest",
        default=str(ROOT / "data" / "quantum" / "sources" / "giustina2015_prl115_250401" / "manifest.json"),
        help="Giustina 2015 local source manifest.",
    )
    ap.add_argument(
        "--outdir",
        default=str(ROOT / "output" / "public" / "summary"),
        help="Output directory for the watchpack JSON/CSV.",
    )
    args = ap.parse_args(list(argv) if argv is not None else None)

    jwst_waitlist = Path(args.jwst_waitlist).resolve()
    bell_summary = Path(args.bell_summary).resolve()
    giustina_manifest = Path(args.giustina_manifest).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    out_json = outdir / "public_wait_data_watchpack.json"
    out_csv = outdir / "public_wait_data_watchpack.csv"

    now_utc = datetime.now(timezone.utc)
    rows = _load_jwst_rows(jwst_waitlist, now_utc) + _load_giustina_rows(bell_summary, giustina_manifest)
    rows = sorted(rows, key=lambda row: (row.ready_now, row.domain, row.item_id))

    input_signatures = {
        "jwst_waitlist": _file_signature(jwst_waitlist),
        "bell_summary": _file_signature(bell_summary),
        "giustina_manifest": _file_signature(giustina_manifest),
    }
    previous_watchpack = _load_previous_watchpack(out_json)
    update_watchpack = _derive_update_watchpack(
        current_input_signatures=input_signatures,
        previous_watchpack=previous_watchpack,
        rows=rows,
    )

    blocked_rows = [row for row in rows if not row.ready_now]
    ready_rows = [row for row in rows if row.ready_now]
    if update_watchpack["update_event_type"] == "baseline_initialized":
        decision = "baseline_initialized"
    elif ready_rows and update_watchpack["update_event_detected"]:
        decision = "ready_items_detected"
    elif not blocked_rows:
        decision = "all_clear"
    elif update_watchpack["update_event_type"] in {"input_hash_changed_state_same", "metadata_changed_hash_same"}:
        decision = "upstream_refresh_status_same"
    else:
        decision = "no_change_hold"

    payload: dict[str, Any] = {
        "generated_utc": _utc_now_iso(),
        "schema": "wavep.summary.public_wait_data_watchpack.v1",
        "phase": 8,
        "step": str(args.step_tag),
        "inputs": {
            "jwst_waitlist": str(jwst_waitlist).replace("\\", "/"),
            "bell_summary": str(bell_summary).replace("\\", "/"),
            "giustina_manifest": str(giustina_manifest).replace("\\", "/"),
        },
        "summary": {
            "overall_status": "watch" if blocked_rows else "pass",
            "decision": decision,
            "blocked_items_n": len(blocked_rows),
            "ready_items_n": len(ready_rows),
            "blocked_item_ids": [row.item_id for row in blocked_rows],
            "ready_item_ids": [row.item_id for row in ready_rows],
            "next_action": update_watchpack["next_action"],
            "update_event_type": update_watchpack["update_event_type"],
            "update_event_detected": bool(update_watchpack["update_event_detected"]),
            "event_counter": int(update_watchpack["event_counter"]),
        },
        "rows": [asdict(row) for row in rows],
        "diagnostics": {
            "public_wait_update_watchpack": update_watchpack,
        },
        "outputs": {
            "json": str(out_json).replace("\\", "/"),
            "csv": str(out_csv).replace("\\", "/"),
        },
        "notes": [
            "This watchpack is a low-priority queue for public-wait or blocked-primary-data items.",
            "It does not fetch new network data; it only summarizes canonical local artifacts.",
        ],
    }

    _write_json(out_json, payload)
    _write_csv(out_csv, rows)

    print(f"[ok] wrote: {out_json}")
    print(f"[ok] wrote: {out_csv}")
    print(
        "[info] "
        f"decision={payload['summary']['decision']} "
        f"blocked_items_n={payload['summary']['blocked_items_n']} "
        f"update_event_type={payload['summary']['update_event_type']}"
    )
    return 0


# Function: Run the CLI entrypoint when executed as a script.

def _entrypoint() -> int:
    return main()


if __name__ == "__main__":
    raise SystemExit(_entrypoint())
