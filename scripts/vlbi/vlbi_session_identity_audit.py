#!/usr/bin/env python3
"""
vlbi_session_identity_audit.py

Audit vgosDb session identity aliases from primary netCDF metadata.

Purpose:
- Confirm whether an external archive label (for example, "17MAY01XA")
  maps to an internal session code (for example, "AUA020") without using
  any derived physics model.

Input:
- Extracted vgosDb directory containing .nc files.

Output:
- output/vlbi/vlbi_<session>_session_identity_audit.json
- output/vlbi/vlbi_<session>_session_identity_audit.csv
- synced copies under output/public/vlbi/
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.io import netcdf_file

try:
    from netCDF4 import Dataset as _NC4Dataset  # type: ignore
except Exception:
    _NC4Dataset = None


# Function: Resolve repository root from this script location.

def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


# Function: Normalize labels for stable output filenames.

def _slugify(text: str) -> str:
    value = "".join(ch if ch.isalnum() else "_" for ch in str(text).strip().lower())
    return value or "session"


# Function: Decode netCDF char arrays into a stripped text value.

def _decode_char_array(arr: np.ndarray) -> str:
    a = np.asarray(arr)
    if a.size == 0:
        return ""

    if a.dtype.kind == "S":
        if a.ndim == 0:
            raw = a.item()
            if isinstance(raw, (bytes, bytearray)):
                return raw.decode("ascii", "ignore").strip()

            return str(raw).strip()

        raw = b"".join(np.asarray(a).reshape(-1).tolist())
        return raw.decode("ascii", "ignore").strip()

    if a.dtype.kind == "U":
        if a.ndim == 0:
            return str(a.item()).strip()

        return "".join([str(x) for x in a.reshape(-1).tolist()]).strip()

    if a.ndim == 0:
        return str(a.item()).strip()

    return str(a.reshape(-1)[0]).strip()


# Function: Read a variable from netCDF using scipy first and netCDF4 fallback.

def _read_variable(path: Path, var_name: str) -> np.ndarray:
    key = str(var_name).lower()
    try:
        with netcdf_file(str(path), mode="r", mmap=False) as nc:
            names = {str(k).lower(): str(k) for k in nc.variables.keys()}
            if key not in names:
                raise KeyError(f"variable not found: {var_name} in {path}")

            return np.array(nc.variables[names[key]][:])
    except Exception:
        if _NC4Dataset is None:
            raise

    with _NC4Dataset(str(path), mode="r") as nc:  # type: ignore[misc]
        names = {str(k).lower(): str(k) for k in nc.variables.keys()}
        if key not in names:
            raise KeyError(f"variable not found: {var_name} in {path}")

        return np.array(nc.variables[names[key]][:])


# Function: Extract Session variable text from one netCDF file when available.

def _read_session_value(path: Path) -> Optional[str]:
    try:
        arr = _read_variable(path, "Session")
    except Exception:
        return None

    txt = _decode_char_array(arr)
    txt = txt.strip()
    return txt or None


# Function: Scan all netCDF files and aggregate discovered Session values.

def _collect_session_values(input_root: Path) -> Tuple[Dict[str, int], Dict[str, List[str]], int]:
    counts: Dict[str, int] = {}
    examples: Dict[str, List[str]] = {}
    n_scanned = 0
    for nc_path in sorted(input_root.rglob("*.nc")):
        n_scanned += 1
        session_value = _read_session_value(nc_path)
        if not session_value:
            continue

        if session_value not in counts:
            counts[session_value] = 0

        counts[session_value] += 1
        if session_value not in examples:
            examples[session_value] = []

        if len(examples[session_value]) < 5:
            examples[session_value].append(str(nc_path))

    return counts, examples, n_scanned


# Function: Detect wrapper files for traceability of external labels.

def _collect_wrapper_files(input_root: Path, max_items: int = 20) -> List[str]:
    paths = sorted(input_root.rglob("*.wrp"))
    out = [str(p) for p in paths[: max(0, int(max_items))]]
    return out


# Function: Copy generated artifacts to output/public/vlbi.

def _sync_public(root: Path, outputs: Sequence[Path]) -> None:
    dst = root / "output" / "public" / "vlbi"
    dst.mkdir(parents=True, exist_ok=True)
    for path in outputs:
        if path.exists():
            shutil.copy2(path, dst / path.name)


# Function: Main entrypoint for session identity audit.

def main() -> int:
    root = _repo_root()
    ap = argparse.ArgumentParser(description="Audit vgosDb session identity aliases from primary metadata.")
    ap.add_argument("--session-label", type=str, default="17MAY01XA", help="External session label for output filenames.")
    ap.add_argument(
        "--input-root",
        type=Path,
        default=root / "data" / "vlbi" / "sources" / "vgosdb" / "17MAY01XA" / "extracted",
        help="Extracted vgosDb directory containing netCDF files.",
    )
    ap.add_argument("--expected-id", type=str, default="AUA020", help="Expected internal session id to verify.")
    args = ap.parse_args()

    session_label = str(args.session_label).strip()
    session_slug = _slugify(session_label)
    input_root = args.input_root.resolve()
    expected_id = str(args.expected_id).strip()
    if not input_root.exists():
        raise FileNotFoundError(f"input root not found: {input_root}")

    counts, examples, n_scanned = _collect_session_values(input_root)
    wrappers = _collect_wrapper_files(input_root=input_root)
    observed_values = sorted(counts.keys())
    alias_confirmed = expected_id in counts
    status = "pass" if alias_confirmed else "watch"
    rationale = (
        "expected internal id found in Session metadata across netCDF files"
        if alias_confirmed
        else "expected internal id was not found in scanned Session metadata"
    )

    out_dir = root / "output" / "vlbi"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"vlbi_{session_slug}_session_identity_audit.json"
    csv_path = out_dir / f"vlbi_{session_slug}_session_identity_audit.csv"

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "session_label": session_label,
        "expected_internal_id": expected_id,
        "input_root": str(input_root),
        "scan_summary": {
            "n_netcdf_scanned": int(n_scanned),
            "session_value_counts": counts,
            "observed_session_values": observed_values,
            "status": status,
            "alias_confirmed": bool(alias_confirmed),
            "rationale": rationale,
        },
        "session_value_examples": examples,
        "wrapper_files_sample": wrappers,
        "outputs": {
            "json": str(json_path),
            "csv": str(csv_path),
        },
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["key", "value"])
        w.writerow(["session_label", session_label])
        w.writerow(["expected_internal_id", expected_id])
        w.writerow(["status", status])
        w.writerow(["alias_confirmed", str(bool(alias_confirmed)).lower()])
        w.writerow(["n_netcdf_scanned", str(int(n_scanned))])
        w.writerow(["observed_session_values", ";".join(observed_values)])
        for key in observed_values:
            w.writerow([f"count[{key}]", str(int(counts.get(key, 0)))])

        w.writerow(["rationale", rationale])

    _sync_public(root, [json_path, csv_path])
    print("Wrote:", json_path)
    print("Wrote:", csv_path)
    print("Synced:", root / "output" / "public" / "vlbi")
    return 0


# Branch: Execute CLI entrypoint when this file is invoked directly.

if __name__ == "__main__":
    raise SystemExit(main())

