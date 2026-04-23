"""Strictly search the repo for v2 W/Z Trial-3 source remnants.

Purpose:
    Determine whether the W/Z heavy-scan source scripts and source metrics still
    exist anywhere under C:/develop/waveP after the environment copy.
Inputs:
    The full repository tree, including ignored private outputs, __pycache__,
    zip bundles, and git history metadata.
Outputs:
    A JSON/CSV evidence report under output/public/quantum.
Assumptions:
    The search is intentionally limited to this repository root because the user
    confirmed the old system was copied as C:/develop/waveP.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import struct
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PUBLIC_OUT = ROOT / "output" / "public" / "quantum"

SOURCE_STEMS = [
    "mass_origin_vector_qball_full_coupled_solver_branch",
    "mass_origin_vector_qball_numerical_solver_branch",
    "mass_origin_vector_qball_extended_hierarchy_branch",
    "mass_origin_vector_qball_route_branch",
    "mass_origin_v2_trial3_two_component_spectrum_branch",
    "mass_origin_v2_trial3_two_component_family_bridge_branch",
    "mass_origin_v2_trial3_two_component_pivot_branch",
    "mass_origin_v2_trial3_two_component_anchor_split_branch",
    "mass_origin_v2_trial3_two_component_anchor_family_absolute_support_branch",
    "mass_origin_v2_trial3_two_component_anchor_family_floor_lowering_branch",
    "mass_origin_v2_trial3_two_component_anchor_family_upper_charge_window_extension_branch",
    "mass_origin_v2_trial3_weak_sector_branch",
    "mass_origin_v2_trial3_relaunched_weak_sector_branch",
]

METRIC_FILENAMES = [
    "mass_origin_v2_t3_t2_coupled_localization_closeout_audit_metrics.json",
    "mass_origin_v2_trial3_two_component_shooting_solver_implementation_metrics.json",
    "mass_origin_v2_trial3_two_component_coupled_radial_ode_derivation_metrics.json",
    "mass_origin_v2_trial3_two_component_spectrum_computation_metrics.json",
]

TARGET_SOURCE_FILENAMES = [f"{stem}.py" for stem in SOURCE_STEMS]
TARGET_FILENAMES = set(TARGET_SOURCE_FILENAMES + METRIC_FILENAMES)
TARGET_SUBSTRINGS = SOURCE_STEMS + [name.removesuffix(".json") for name in METRIC_FILENAMES]

SKIP_DIR_NAMES = {".git", ".venv_wsl"}


# Function: Return an ISO timestamp for the current audit run.
def utc_now_iso() -> str:
    """Return the current UTC timestamp as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


# Function: Compute SHA256 for a file.

def sha256_file(path: Path) -> str:
    """Return a SHA256 digest for an existing file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


# Function: Convert one path to a repo-relative display path when possible.

def display_path(path: Path) -> str:
    """Return a path string relative to ROOT when the path is inside ROOT."""
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


# Function: Iterate repository files while skipping only irrelevant system dirs.

def iter_repo_files() -> list[Path]:
    """Return all regular files under ROOT except git/venv/cache traversal targets."""
    files: list[Path] = []
    stack = [ROOT]

    while stack:
        current = stack.pop()
        try:
            for child in current.iterdir():
                if child.is_dir():
                    if child.name in SKIP_DIR_NAMES:
                        continue

                    stack.append(child)
                elif child.is_file():
                    files.append(child)

        except (OSError, PermissionError):
            continue

    return files


# Function: Build one normal file record.

def file_record(path: Path) -> dict[str, Any]:
    """Return metadata for an existing file."""
    stat = path.stat()

    return {
        "path": display_path(path),
        "size_bytes": stat.st_size,
        "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
        "sha256": sha256_file(path),
    }


# Function: Parse useful metadata from timestamp-based pyc headers.

def pyc_record(path: Path) -> dict[str, Any]:
    """Return metadata and header details for a CPython bytecode file."""
    record = file_record(path)
    try:
        data = path.read_bytes()[:16]
        flags = struct.unpack("<I", data[4:8])[0]
        record["pyc_flags"] = flags
        record["pyc_hash_based"] = bool(flags & 1)
        if not (flags & 1):
            source_mtime, source_size = struct.unpack("<II", data[8:16])
            record["source_mtime_utc"] = datetime.fromtimestamp(source_mtime, timezone.utc).isoformat()
            record["source_size_bytes_from_header"] = source_size

    except (OSError, struct.error):
        record["pyc_header_error"] = True

    return record


# Function: Search zip bundles for exact target entries.

def search_zip_entries(files: list[Path]) -> list[dict[str, Any]]:
    """Return zip entries whose leaf filename matches a target source or metric."""
    hits: list[dict[str, Any]] = []

    for path in files:
        if path.suffix.lower() != ".zip":
            continue

        try:
            with zipfile.ZipFile(path) as archive:
                for entry in archive.infolist():
                    leaf = Path(entry.filename).name
                    if leaf in TARGET_FILENAMES:
                        hits.append(
                            {
                                "zip": display_path(path),
                                "entry": entry.filename,
                                "entry_size_bytes": entry.file_size,
                                "zip_size_bytes": path.stat().st_size,
                            }
                        )

        except (OSError, zipfile.BadZipFile):
            continue

    return hits


# Function: Run rg for target text references across non-cache repo files.

def search_text_references() -> dict[str, Any]:
    """Return text-reference hits produced by ripgrep."""
    pattern = "|".join(re.escape(item) for item in TARGET_SUBSTRINGS)
    command = [
        "rg",
        "--hidden",
        "--no-ignore",
        "-n",
        "--glob",
        "!/.git/**",
        "--glob",
        "!.venv_wsl/**",
        "--glob",
        "!scripts/quantum/__pycache__/**",
        pattern,
        str(ROOT),
    ]
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    lines = result.stdout.splitlines()

    return {
        "returncode": result.returncode,
        "hit_count": len(lines),
        "first_hits": lines[:200],
    }


# Function: Query git for tracked history for all canonical target paths.

def search_git_history() -> list[dict[str, Any]]:
    """Return git-log hits for target source and public metric paths."""
    paths = [f"scripts/quantum/{name}" for name in TARGET_SOURCE_FILENAMES]
    paths.extend(f"output/public/quantum/{name}" for name in METRIC_FILENAMES)
    hits: list[dict[str, Any]] = []

    for rel_path in paths:
        result = subprocess.run(
            ["git", "log", "--all", "--oneline", "--", rel_path],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        lines = result.stdout.splitlines()
        if lines:
            hits.append({"path": rel_path, "commits": lines})

    return hits


# Function: Search stash untracked-file parents for exact target entries.

def search_stash_untracked_entries() -> list[dict[str, Any]]:
    """Return exact target entries present in stash third-parent trees."""
    stash_list = subprocess.run(
        ["git", "stash", "list", "--format=%gd %gs"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    hits: list[dict[str, Any]] = []

    for line in stash_list.stdout.splitlines():
        if not line.strip():
            continue

        ref = line.split(maxsplit=1)[0]
        subject = line.partition(" ")[2]
        treeish = f"{ref}^3"
        result = subprocess.run(
            ["git", "ls-tree", "-r", "--name-only", treeish],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if result.returncode != 0:
            continue

        for entry in result.stdout.splitlines():
            if Path(entry).name in TARGET_FILENAMES:
                hits.append(
                    {
                        "stash": ref,
                        "subject": subject,
                        "treeish": treeish,
                        "entry": entry,
                        "kind": "source_py" if entry.endswith(".py") else "metric_json",
                    }
                )

    return hits


# Function: Write summary CSV rows.

def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write strict-search summary rows to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["category", "count", "status", "note"])
        writer.writeheader()
        writer.writerows(rows)


# Function: Run the strict source search.

def main() -> int:
    """Search the repository and write strict-search evidence."""
    PUBLIC_OUT.mkdir(parents=True, exist_ok=True)
    files = iter_repo_files()

    exact_file_hits = [file_record(path) for path in files if path.name in TARGET_FILENAMES]
    source_py_hits = [row for row in exact_file_hits if row["path"].endswith(".py")]
    metric_hits = [row for row in exact_file_hits if row["path"].endswith(".json")]
    pyc_hits = [
        pyc_record(path)
        for path in files
        if path.suffix == ".pyc" and any(path.name.startswith(f"{stem}.cpython-") for stem in SOURCE_STEMS)
    ]
    zip_hits = search_zip_entries(files)
    text_refs = search_text_references()
    git_history_hits = search_git_history()
    stash_untracked_hits = search_stash_untracked_entries()
    stash_source_hits = [row for row in stash_untracked_hits if row["kind"] == "source_py"]
    stash_metric_hits = [row for row in stash_untracked_hits if row["kind"] == "metric_json"]

    conclusion = "source_py_absent_but_bytecode_and_private_metric_traces_remain"
    if source_py_hits:
        conclusion = "source_py_present"
    elif stash_source_hits:
        conclusion = "source_py_absent_from_worktree_but_present_in_stash_untracked_parent"

    rows = [
        {
            "category": "exact_source_py_files",
            "count": len(source_py_hits),
            "status": "present" if source_py_hits else "absent",
            "note": "Canonical .py source files under scripts/quantum.",
        },
        {
            "category": "exact_metric_json_files",
            "count": len(metric_hits),
            "status": "present" if metric_hits else "absent",
            "note": "Exact metric filenames under the copied repo tree.",
        },
        {
            "category": "source_pyc_files",
            "count": len(pyc_hits),
            "status": "present" if pyc_hits else "absent",
            "note": "Bytecode remnants under scripts/quantum/__pycache__.",
        },
        {
            "category": "zip_entries",
            "count": len(zip_hits),
            "status": "present" if zip_hits else "absent",
            "note": "Exact target entries inside repo zip bundles.",
        },
        {
            "category": "text_references",
            "count": text_refs["hit_count"],
            "status": "present" if text_refs["hit_count"] else "absent",
            "note": "References in logs/manifests/docs/scripts.",
        },
        {
            "category": "git_history",
            "count": len(git_history_hits),
            "status": "present" if git_history_hits else "absent",
            "note": "Tracked git history for canonical target paths.",
        },
        {
            "category": "stash_untracked_entries",
            "count": len(stash_untracked_hits),
            "status": "present" if stash_untracked_hits else "absent",
            "note": "Exact target entries in stash third-parent untracked trees.",
        },
    ]

    report = {
        "generated_utc": utc_now_iso(),
        "search_root": str(ROOT),
        "scope_note": "Repo-only strict search requested by user; .git and .venv_wsl traversal are skipped, while __pycache__ is included for pyc remnants.",
        "target_source_filenames": TARGET_SOURCE_FILENAMES,
        "target_metric_filenames": METRIC_FILENAMES,
        "summary": {
            "conclusion": conclusion,
            "source_py_exact_count": len(source_py_hits),
            "metric_json_exact_count": len(metric_hits),
            "source_pyc_count": len(pyc_hits),
            "zip_entry_count": len(zip_hits),
            "text_reference_count": text_refs["hit_count"],
            "git_history_hit_count": len(git_history_hits),
            "stash_untracked_entry_count": len(stash_untracked_hits),
            "stash_untracked_source_py_count": len(stash_source_hits),
            "stash_untracked_metric_json_count": len(stash_metric_hits),
        },
        "exact_source_py_hits": source_py_hits,
        "exact_metric_json_hits": metric_hits,
        "source_pyc_hits": pyc_hits,
        "zip_entry_hits": zip_hits,
        "text_references": text_refs,
        "git_history_hits": git_history_hits,
        "stash_untracked_hits": stash_untracked_hits,
        "csv_rows": rows,
    }

    json_path = PUBLIC_OUT / "v3_trial1_wz_source_strict_search.json"
    csv_path = PUBLIC_OUT / "v3_trial1_wz_source_strict_search.csv"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(csv_path, rows)

    print(f"[write] {json_path.relative_to(ROOT)}")
    print(f"[write] {csv_path.relative_to(ROOT)}")
    print(f"[summary] {conclusion}")
    print(f"[summary] source_py={len(source_py_hits)} pyc={len(pyc_hits)} metrics={len(metric_hits)} zip_entries={len(zip_hits)} text_refs={text_refs['hit_count']} stash_entries={len(stash_untracked_hits)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
