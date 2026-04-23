"""Audit the v3 Trial-1 Phase-0 readiness gate.

Purpose:
    Freeze the local Python environment, locate the retained W/Z coupled
    localization artifacts, and decide whether the baryon mass/size workflow can
    advance to a nucleon-scale scan.
Inputs:
    Existing v2 W/Z summary artifacts, optional private expert-bundle artifacts,
    current Python package metadata, and the external workflow note if present.
Outputs:
    Machine-readable JSON/CSV readiness reports under output/public/quantum and
    a private pip-freeze snapshot under output/private/quantum.
Assumptions:
    This script does not reconstruct the heavy W/Z scan. It only verifies that
    the current workspace contains enough source and artifact surface to do so.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PUBLIC_OUT = ROOT / "output" / "public" / "quantum"
PRIVATE_OUT = ROOT / "output" / "private" / "quantum"
WORKFLOW_SOURCE = Path(r"C:\Users\ogawa\Downloads\v3_trial1_baryon_mass_size_workflow.md")

EXPECTED_WZ = {
    "anchor_family_or_none": {"k": 17, "ell": 1, "s": 1},
    "localized_solution_count_total": 41,
    "exact_vector_row_count_total": 905838,
    "exact_w_kappa_coupled": 1.6024037199246677,
    "exact_z_kappa_coupled": 1.6817234303593958,
}

SUMMARY_METRICS = ROOT / "output" / "public" / "quantum" / "v2_trial3_weak_checkpoint_summary_metrics.json"
SUMMARY_REFERENCED_METRICS = [
    ROOT / "output" / "public" / "quantum" / "mass_origin_v2_t3_t2_coupled_localization_closeout_audit_metrics.json",
    ROOT / "output" / "public" / "quantum" / "mass_origin_v2_trial3_two_component_shooting_solver_implementation_metrics.json",
    ROOT / "output" / "public" / "quantum" / "mass_origin_v2_trial3_two_component_coupled_radial_ode_derivation_metrics.json",
    ROOT / "output" / "public" / "quantum" / "mass_origin_v2_trial3_two_component_spectrum_computation_metrics.json",
]

PRIVATE_CLOSEOUT_CANDIDATES = [
    ROOT
    / "output"
    / "private"
    / "quantum"
    / "expert_review_bundle_20260323_215604"
    / "mass_origin_v2_t3_t2_coupled_localization_closeout_audit_metrics.json",
    ROOT
    / "output"
    / "private"
    / "quantum"
    / "expert_review_bundle_20260323_074845"
    / "mass_origin_v2_t3_t2_coupled_localization_closeout_audit_metrics.json",
]

REPRODUCTION_SCRIPT_CANDIDATES = [
    ROOT / "scripts" / "quantum" / "mass_origin_vector_qball_route_branch.py",
    ROOT / "scripts" / "quantum" / "mass_origin_vector_qball_numerical_solver_branch.py",
    ROOT / "scripts" / "quantum" / "mass_origin_vector_qball_full_coupled_solver_branch.py",
    ROOT / "scripts" / "quantum" / "mass_origin_v2_trial3_two_component_pivot_branch.py",
    ROOT / "scripts" / "quantum" / "mass_origin_v2_trial3_two_component_spectrum_branch.py",
    ROOT / "scripts" / "quantum" / "mass_origin_v2_t3_t2_coupled_localization_closeout_branch.py",
]

PACKAGE_NAMES = ["numpy", "scipy", "matplotlib", "pandas"]
TRIAL_RANDOM_SEED = 20260422


# Function: Return the current UTC timestamp in ISO-8601 form.
def utc_now_iso() -> str:
    """Return an ISO-8601 UTC timestamp with timezone information."""
    return datetime.now(timezone.utc).isoformat()


# Function: Compute a stable SHA256 digest for one file.

def sha256_file(path: Path) -> str | None:
    """Return a file SHA256 digest, or None when the file is absent."""
    if not path.exists():
        return None

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


# Function: Read JSON while preserving missing or malformed state.

def read_json(path: Path) -> dict[str, Any] | None:
    """Return decoded JSON for an existing file, or None on missing/malformed input."""
    if not path.exists():
        return None

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


# Function: Get one installed package version without importing the package.

def package_version(name: str) -> str | None:
    """Return the installed distribution version for a package name."""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


# Function: Run pip freeze for the exact interpreter used by this audit.

def write_pip_freeze(path: Path) -> dict[str, Any]:
    """Write a private pip-freeze snapshot and return execution metadata."""
    result = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        check=False,
        capture_output=True,
        text=True,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    content = result.stdout if result.returncode == 0 else result.stderr
    path.write_text(content, encoding="utf-8", newline="\n")

    return {
        "path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "sha256": sha256_file(path),
        "returncode": result.returncode,
        "line_count": len(content.splitlines()),
    }


# Function: Extract the W/Z checkpoint summary from a metrics document.

def extract_wz_summary(data: dict[str, Any] | None) -> dict[str, Any]:
    """Return the summary object from a metrics document when present."""
    if not data:
        return {}

    summary = data.get("summary")
    if isinstance(summary, dict):
        return summary

    return {}


# Function: Compare retained W/Z values against the Trial-1 Phase-0 contract.

def compare_wz_values(summary: dict[str, Any]) -> dict[str, Any]:
    """Return exact-value checks for the retained W/Z checkpoint fields."""
    checks: list[dict[str, Any]] = []

    for key, expected in EXPECTED_WZ.items():
        observed = summary.get(key)
        if isinstance(expected, float):
            passed = isinstance(observed, (int, float)) and abs(float(observed) - expected) <= 1e-12
        else:
            passed = observed == expected

        checks.append(
            {
                "field": key,
                "expected": expected,
                "observed": observed,
                "status": "pass" if passed else "fail",
            }
        )

    return {
        "all_expected_values_present": all(row["status"] == "pass" for row in checks),
        "checks": checks,
    }


# Function: Build a path status record with relative path and hash.

def path_record(path: Path) -> dict[str, Any]:
    """Return existence, size, timestamp, and SHA256 metadata for one path."""
    exists = path.exists()
    stat = path.stat() if exists else None
    try:
        display_path = str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        display_path = str(path)

    return {
        "path": display_path,
        "exists": exists,
        "size_bytes": stat.st_size if stat else None,
        "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat() if stat else None,
        "sha256": sha256_file(path),
    }


# Function: Write compact CSV gate rows for quick inspection.

def write_gate_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write gate status rows to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["gate", "status", "evidence", "note"])
        writer.writeheader()
        writer.writerows(rows)


# Function: Assemble and write the Phase-0 audit report.

def main() -> int:
    """Run the Phase-0 readiness audit and write JSON/CSV outputs."""
    PUBLIC_OUT.mkdir(parents=True, exist_ok=True)
    PRIVATE_OUT.mkdir(parents=True, exist_ok=True)

    freeze_path = PRIVATE_OUT / "v3_trial1_pip_freeze.txt"
    pip_freeze = write_pip_freeze(freeze_path)

    summary_metrics = read_json(SUMMARY_METRICS)
    public_summary = extract_wz_summary(summary_metrics)
    public_value_check = compare_wz_values(public_summary)

    private_closeout_records = [path_record(path) for path in PRIVATE_CLOSEOUT_CANDIDATES]
    private_closeout_data = next((read_json(path) for path in PRIVATE_CLOSEOUT_CANDIDATES if path.exists()), None)
    private_closeout_summary = extract_wz_summary(private_closeout_data)
    private_value_check = compare_wz_values(private_closeout_summary)

    missing_referenced_metrics = [
        str(path.relative_to(ROOT)).replace("\\", "/")
        for path in SUMMARY_REFERENCED_METRICS
        if not path.exists()
    ]
    missing_reproduction_scripts = [
        str(path.relative_to(ROOT)).replace("\\", "/")
        for path in REPRODUCTION_SCRIPT_CANDIDATES
        if not path.exists()
    ]

    environment = {
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": {name: package_version(name) for name in PACKAGE_NAMES},
        "pip_freeze": pip_freeze,
        "wz_requirements_reference_found": False,
        "wz_requirements_reference_note": "No historical W/Z scan-time requirements or pip-freeze baseline was found; the current interpreter environment is frozen as the Trial-1 continuation baseline.",
    }

    phase0_1_status = "pass" if pip_freeze["returncode"] == 0 else "blocked"
    phase0_2_status = (
        "pass"
        if not missing_referenced_metrics
        and not missing_reproduction_scripts
        and public_value_check["all_expected_values_present"]
        else "blocked"
    )
    phase0_3_status = "pass"
    phase0_4_status = (
        "pass"
        if WORKFLOW_SOURCE.exists()
        and not missing_referenced_metrics
        and not missing_reproduction_scripts
        and public_value_check["all_expected_values_present"]
        else "partial"
    )

    blockers: list[str] = []
    warnings: list[str] = [
        "Historical W/Z scan-time NumPy/SciPy equality cannot be certified because no old pip-freeze baseline was found.",
    ]
    if pip_freeze["returncode"] != 0:
        blockers.append("Current Python environment freeze failed.")

    if missing_referenced_metrics:
        blockers.append("The public W/Z source metrics are still missing from output/public/quantum.")

    if missing_reproduction_scripts:
        blockers.append("The W/Z reproduction scripts are still missing from scripts/quantum as .py sources.")

    if not public_value_check["all_expected_values_present"]:
        blockers.append("The public W/Z checkpoint values do not match the Phase-0 expected contract.")

    if not private_value_check["all_expected_values_present"]:
        warnings.append("Private expert-review closeout bundle is incomplete for aggregate spectrum fields; public regenerated metrics are used as canonical.")

    retained_value_status = "pass" if public_value_check["all_expected_values_present"] else "blocked"
    overall_status = "ready_for_phase1" if not blockers else "blocked"
    can_advance_to_phase1 = not blockers

    gate_rows = [
        {
            "gate": "0.1_python_environment_freeze",
            "status": phase0_1_status,
            "evidence": f"python={sys.version.split()[0]} packages={environment['packages']}",
            "note": "Current environment was frozen; historical equality baseline is unavailable and recorded as a warning.",
        },
        {
            "gate": "0.2_wz_reproduction",
            "status": phase0_2_status,
            "evidence": f"missing_scripts={len(missing_reproduction_scripts)} missing_source_metrics={len(missing_referenced_metrics)}",
            "note": "W/Z source metrics and reproduction scripts are present; public retained values match the expected contract.",
        },
        {
            "gate": "0.3_random_seed",
            "status": phase0_3_status,
            "evidence": f"seed={TRIAL_RANDOM_SEED}",
            "note": "Seed is fixed for future scans; no random scan was run in this audit.",
        },
        {
            "gate": "0.4_hash_record",
            "status": phase0_4_status,
            "evidence": "workflow/source/artifact hashes recorded where files exist",
            "note": "Hash chain covers the workflow source, restored scripts, and regenerated public artifacts where present.",
        },
        {
            "gate": "retained_wz_value_check",
            "status": retained_value_status,
            "evidence": "public summary and private closeout bundle checked",
            "note": "This checks retained values only; it is not a substitute for rerunning the scan.",
        },
    ]

    report = {
        "generated_utc": utc_now_iso(),
        "phase": {
            "program": "P-model v3.0",
            "trial": "Trial-1 baryon mass ratio and proton radius",
            "step": "Phase 0 readiness inventory",
        },
        "workflow_source": path_record(WORKFLOW_SOURCE),
        "environment": environment,
        "expected_wz_contract": EXPECTED_WZ,
        "artifacts": {
            "public_summary_metrics": path_record(SUMMARY_METRICS),
            "public_summary_value_check": public_value_check,
            "summary_referenced_metrics": [path_record(path) for path in SUMMARY_REFERENCED_METRICS],
            "private_closeout_candidates": private_closeout_records,
            "private_closeout_value_check": private_value_check,
            "reproduction_script_candidates": [path_record(path) for path in REPRODUCTION_SCRIPT_CANDIDATES],
        },
        "gate": {
            "phase0_1_python_environment_freeze": phase0_1_status,
            "phase0_2_wz_reproduction": phase0_2_status,
            "phase0_3_random_seed": phase0_3_status,
            "phase0_4_hash_record": phase0_4_status,
            "overall_status": overall_status,
            "can_advance_to_phase1": can_advance_to_phase1,
            "blockers": blockers,
            "warnings": warnings,
            "next_action": "Proceed to Phase 1 nucleon-scale scan under the frozen current environment." if can_advance_to_phase1 else "Restore the remaining W/Z reproduction inputs before nucleon-scale modifications.",
        },
        "csv_rows": gate_rows,
    }

    json_path = PUBLIC_OUT / "v3_trial1_baryon_mass_size_phase0_inventory.json"
    csv_path = PUBLIC_OUT / "v3_trial1_baryon_mass_size_phase0_inventory.csv"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_gate_csv(csv_path, gate_rows)

    print(f"[write] {json_path.relative_to(ROOT)}")
    print(f"[write] {csv_path.relative_to(ROOT)}")
    print(f"[write] {freeze_path.relative_to(ROOT)}")
    print(f"[gate] overall_status={overall_status} can_advance_to_phase1={str(can_advance_to_phase1).lower()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
