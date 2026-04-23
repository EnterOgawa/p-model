"""Inventory v3 Trial-1 Phase-1 nucleon-scale dependencies.

Purpose:
    Freeze the minimal paper-critical dependency map before the baryon
    mass-ratio and proton-radius scan. This script checks that the restored
    W/Z solver assets expose the reusable functions and artifacts required by
    the workflow, and records exactly which files may be staged for the next
    paper-facing calculation.
Inputs:
    The Trial-1 workflow note, Phase-0 readiness output, restored W/Z solver
    scripts, and public W/Z / two-component metrics.
Outputs:
    JSON and CSV Phase-1 inventory reports under output/public/quantum.
Assumptions:
    This branch does not run the heavy nucleon scan. It fixes the adapter
    contract and the allowed dependency surface before Phase 2 implementation.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PUBLIC_OUT = ROOT / "output" / "public" / "quantum"
WORKFLOW_SOURCE = Path(r"C:\Users\ogawa\Downloads\v3_trial1_baryon_mass_size_workflow.md")
PHASE0_INVENTORY = PUBLIC_OUT / "v3_trial1_baryon_mass_size_phase0_inventory.json"
PUSH_SCOPE_DOC = ROOT / "doc" / "quantum" / "187_v3_trial1_push_scope_cleanup.md"
UNTRACKED_METRIC_MANIFEST = (
    ROOT
    / "output"
    / "private"
    / "summary"
    / "windows_worksets"
    / "v3_trial1_untracked_public_quantum_metrics_20260423.csv"
)

SOURCE_SCRIPTS = {
    "route_contract": ROOT / "scripts" / "quantum" / "mass_origin_vector_qball_route_branch.py",
    "effective_numerical_solver": ROOT / "scripts" / "quantum" / "mass_origin_vector_qball_numerical_solver_branch.py",
    "full_coupled_solver": ROOT / "scripts" / "quantum" / "mass_origin_vector_qball_full_coupled_solver_branch.py",
    "two_component_pivot": ROOT / "scripts" / "quantum" / "mass_origin_v2_trial3_two_component_pivot_branch.py",
    "two_component_spectrum": ROOT / "scripts" / "quantum" / "mass_origin_v2_trial3_two_component_spectrum_branch.py",
    "coupled_localization_closeout": ROOT
    / "scripts"
    / "quantum"
    / "mass_origin_v2_t3_t2_coupled_localization_closeout_branch.py",
    "post_ell18_helper": ROOT / "scripts" / "quantum" / "mass_origin_v2_t3_post_ell18_amplitude_branch.py",
    "phase0_inventory": ROOT / "scripts" / "quantum" / "v3_trial1_baryon_mass_size_phase0_inventory.py",
    "strict_search": ROOT / "scripts" / "quantum" / "v3_trial1_wz_source_strict_search.py",
    "phase1_inventory": ROOT / "scripts" / "quantum" / "v3_trial1_nucleon_scale_phase1_inventory.py",
}

PUBLIC_METRICS = {
    "weak_checkpoint_summary": PUBLIC_OUT / "v2_trial3_weak_checkpoint_summary_metrics.json",
    "phase0_inventory": PUBLIC_OUT / "v3_trial1_baryon_mass_size_phase0_inventory.json",
    "strict_search": PUBLIC_OUT / "v3_trial1_wz_source_strict_search.json",
    "two_component_radial_ode": PUBLIC_OUT / "mass_origin_v2_trial3_two_component_coupled_radial_ode_derivation_metrics.json",
    "two_component_shooting": PUBLIC_OUT / "mass_origin_v2_trial3_two_component_shooting_solver_implementation_metrics.json",
    "two_component_spectrum": PUBLIC_OUT / "mass_origin_v2_trial3_two_component_spectrum_computation_metrics.json",
    "coupled_localization_closeout": PUBLIC_OUT
    / "mass_origin_v2_t3_t2_coupled_localization_closeout_audit_metrics.json",
}

REQUIRED_FUNCTIONS = {
    "effective_numerical_solver": [
        "solve_sector_profile",
        "cached_profile",
        "scan_ell_sector",
        "interpolate_integer_modes",
        "build_base_modes",
    ],
    "full_coupled_solver": [
        "polarization_weight",
        "coupled_charge_factor",
        "coupled_mass_factor",
        "build_exact_ladder",
    ],
    "two_component_pivot": ["solve_two_component_profile"],
    "two_component_spectrum": [
        "solve_two_component_profile",
        "interpolate_two_component_modes",
        "best_ratio_pair_fast",
    ],
    "post_ell18_helper": ["normalize_vector_rows", "best_ratio_pair"],
}

REQUIRED_METRIC_CHECKS = {
    "weak_checkpoint_summary": {
        "summary.localized_solution_count_total": 41,
        "summary.exact_vector_row_count_total": 905838,
        "summary.exact_w_kappa_coupled": 1.6024037199246677,
        "summary.exact_z_kappa_coupled": 1.6817234303593958,
    },
    "two_component_spectrum": {
        "summary.localized_solution_count_total": 41,
        "summary.base_mode_count_total": 324937,
        "summary.exact_vector_row_count_total": 905838,
    },
    "two_component_radial_ode": {
        "summary.two_component_coupled_radial_ode_template_ready": True,
        "summary.new_free_parameters_introduced": [],
    },
    "two_component_shooting": {
        "summary.two_component_shooting_solver_implemented": True,
        "summary.full_spectrum_scan_ready": True,
    },
    "coupled_localization_closeout": {
        "summary.anchor_family_or_none": {"k": 17, "ell": 1, "s": 1},
    },
}

PROHIBITED_MODIFICATIONS = [
    "new independent model constants",
    "new coupling term",
    "new potential term",
    "boundary-condition intermediate target",
    "relaxed mass-ratio or radius target",
    "invasive rewrite of the restored W/Z solver",
]

ALLOWED_MINIMAL_MODIFICATIONS = [
    "retarget absolute energy scale from W/Z to m_N ~= 0.94 GeV",
    "expand family labels n0/nL/nT for the nucleon-scale search",
    "keep or refine spatial grid so 0.84 fm is resolved",
]

PROTON_MASS_MEV = 938.272
W_MASS_MEV = 80369.0
RADIUS_TARGET_FM = 0.84
HBARC_MEV_FM = 197.3269804
CURRENT_TWO_COMPONENT_R_MAX = 25.0
CURRENT_TWO_COMPONENT_MAX_STEP = 0.10


# Function: Return the current UTC timestamp in ISO-8601 form.
def utc_now_iso() -> str:
    """Return an ISO-8601 UTC timestamp with timezone information."""
    return datetime.now(timezone.utc).isoformat()


# Function: Convert an absolute path to a repo-relative POSIX string.

def rel(path: Path) -> str:
    """Return a repository-relative path when possible."""
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


# Function: Compute a stable SHA256 digest for an existing file.

def sha256_file(path: Path) -> str | None:
    """Return the SHA256 hex digest for a file, or None when absent."""
    if not path.exists():
        return None

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


# Function: Read JSON as a dictionary and return None on missing/malformed input.

def read_json(path: Path) -> dict[str, Any] | None:
    """Decode a JSON object when available."""
    if not path.exists():
        return None

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None

    return data if isinstance(data, dict) else None


# Function: Return file existence and hash metadata.

def path_record(path: Path) -> dict[str, Any]:
    """Build path metadata used by the inventory report."""
    exists = path.exists()
    stat = path.stat() if exists else None
    return {
        "path": rel(path),
        "exists": exists,
        "size_bytes": stat.st_size if stat else None,
        "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat() if stat else None,
        "sha256": sha256_file(path),
    }


# Function: Convert parsed literals into JSON-safe structures.

def json_safe(value: Any) -> Any:
    """Return a JSON-serializable representation of a parsed literal."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]

    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}

    return repr(value)


# Function: Parse one Python script without importing it.

def parse_python_surface(path: Path) -> dict[str, Any]:
    """Return function, import, and simple constant surfaces for a script."""
    record = path_record(path)
    if not path.exists():
        record.update({"parse_ok": False, "functions": [], "imports": [], "constants": {}})
        return record

    source = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        record.update({"parse_ok": False, "parse_error": str(exc), "functions": [], "imports": [], "constants": {}})
        return record

    functions: list[str] = []
    imports: list[str] = []
    constants: dict[str, Any] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            functions.append(node.name)
        elif isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)

    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.isupper():
                    try:
                        constants[target.id] = json_safe(ast.literal_eval(node.value))
                    except (ValueError, TypeError):
                        constants[target.id] = "<non-literal>"

    record.update(
        {
            "parse_ok": True,
            "functions": sorted(set(functions)),
            "imports": sorted(set(imports)),
            "constants": constants,
        }
    )
    return record


# Function: Resolve dotted keys inside nested dictionaries.

def get_dotted(data: dict[str, Any] | None, dotted_key: str) -> Any:
    """Return a nested value selected by a dotted key."""
    current: Any = data
    for key in dotted_key.split("."):
        if not isinstance(current, dict) or key not in current:
            return None

        current = current[key]

    return current


# Function: Compare values with tight numeric tolerance for inventory checks.

def value_matches(observed: Any, expected: Any) -> bool:
    """Return True when observed matches expected for gate purposes."""
    if isinstance(expected, float):
        return isinstance(observed, (float, int)) and math.isclose(float(observed), expected, rel_tol=0.0, abs_tol=1.0e-12)

    return observed == expected


# Function: Check required functions against parsed Python surfaces.

def function_gate(script_surfaces: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    """Return reusable-function checks and missing function labels."""
    checks: list[dict[str, Any]] = []
    missing: list[str] = []
    for script_key, function_names in REQUIRED_FUNCTIONS.items():
        surface = script_surfaces.get(script_key, {})
        present_functions = set(surface.get("functions", []))
        for function_name in function_names:
            present = function_name in present_functions
            label = f"{script_key}.{function_name}"
            if not present:
                missing.append(label)

            checks.append(
                {
                    "dependency": label,
                    "script": rel(SOURCE_SCRIPTS[script_key]),
                    "function": function_name,
                    "status": "pass" if present else "blocked",
                }
            )

    return checks, missing


# Function: Check public metrics against retained W/Z / Trial-3 contracts.

def metric_gate(metric_data: dict[str, dict[str, Any] | None]) -> tuple[list[dict[str, Any]], list[str]]:
    """Return public-metric checks and missing/failed labels."""
    checks: list[dict[str, Any]] = []
    failed: list[str] = []
    for metric_key, dotted_checks in REQUIRED_METRIC_CHECKS.items():
        data = metric_data.get(metric_key)
        if data is None:
            failed.append(f"{metric_key}.missing")
            checks.append(
                {
                    "dependency": metric_key,
                    "field": "<file>",
                    "expected": "exists",
                    "observed": None,
                    "status": "blocked",
                }
            )
            continue

        for dotted_key, expected in dotted_checks.items():
            observed = get_dotted(data, dotted_key)
            passed = value_matches(observed, expected)
            if not passed:
                failed.append(f"{metric_key}.{dotted_key}")

            checks.append(
                {
                    "dependency": metric_key,
                    "field": dotted_key,
                    "expected": expected,
                    "observed": observed,
                    "status": "pass" if passed else "blocked",
                }
            )

    return checks, failed


# Function: Build physical scale diagnostics for the nucleon retargeting.

def nucleon_scale_diagnostics() -> dict[str, Any]:
    """Return unit-conversion diagnostics for the nucleon-scale grid."""
    nucleon_length_fm = HBARC_MEV_FM / PROTON_MASS_MEV
    current_r_max_fm = CURRENT_TWO_COMPONENT_R_MAX * nucleon_length_fm
    current_max_step_fm = CURRENT_TWO_COMPONENT_MAX_STEP * nucleon_length_fm
    radius_target_x = RADIUS_TARGET_FM / nucleon_length_fm
    return {
        "proton_mass_mev_target_for_scoring": PROTON_MASS_MEV,
        "w_mass_mev_previous_absolute_scale": W_MASS_MEV,
        "nucleon_to_w_absolute_scale_factor": PROTON_MASS_MEV / W_MASS_MEV,
        "hbarc_mev_fm_unit_conversion": HBARC_MEV_FM,
        "nucleon_length_fm": nucleon_length_fm,
        "radius_target_fm": RADIUS_TARGET_FM,
        "radius_target_dimensionless_x": radius_target_x,
        "current_two_component_r_max_dimensionless": CURRENT_TWO_COMPONENT_R_MAX,
        "current_two_component_r_max_fm_at_nucleon_scale": current_r_max_fm,
        "current_two_component_max_step_dimensionless": CURRENT_TWO_COMPONENT_MAX_STEP,
        "current_two_component_max_step_fm_at_nucleon_scale": current_max_step_fm,
        "grid_resolves_0p84fm_target": current_r_max_fm >= 1.0 and current_max_step_fm <= 0.05,
    }


# Function: Build the paper-critical dependency list for the next Trial-1 commit.

def build_dependency_list() -> dict[str, Any]:
    """Return the staged/deferred dependency list required by cleanup policy."""
    return {
        "paper_critical_scripts": [
            rel(SOURCE_SCRIPTS["phase1_inventory"]),
            rel(SOURCE_SCRIPTS["phase0_inventory"]),
            rel(SOURCE_SCRIPTS["strict_search"]),
            rel(SOURCE_SCRIPTS["effective_numerical_solver"]),
            rel(SOURCE_SCRIPTS["full_coupled_solver"]),
            rel(SOURCE_SCRIPTS["two_component_pivot"]),
            rel(SOURCE_SCRIPTS["two_component_spectrum"]),
            rel(SOURCE_SCRIPTS["coupled_localization_closeout"]),
            rel(SOURCE_SCRIPTS["post_ell18_helper"]),
        ],
        "paper_critical_public_outputs": [
            "output/public/quantum/v3_trial1_nucleon_scale_phase1_inventory.json",
            "output/public/quantum/v3_trial1_nucleon_scale_phase1_inventory.csv",
            rel(PUBLIC_METRICS["phase0_inventory"]),
            "output/public/quantum/v3_trial1_baryon_mass_size_phase0_inventory.csv",
            rel(PUBLIC_METRICS["strict_search"]),
            "output/public/quantum/v3_trial1_wz_source_strict_search.csv",
            rel(PUBLIC_METRICS["weak_checkpoint_summary"]),
            rel(PUBLIC_METRICS["two_component_radial_ode"]),
            rel(PUBLIC_METRICS["two_component_shooting"]),
            rel(PUBLIC_METRICS["two_component_spectrum"]),
            rel(PUBLIC_METRICS["coupled_localization_closeout"]),
        ],
        "holding_local_only": [
            rel(UNTRACKED_METRIC_MANIFEST),
            "80 untracked output/public/quantum metrics recorded in the manifest above",
        ],
        "excluded_by_default": [
            "fb7ed024 broad restored quantum-script set unless a later dependency map promotes individual files",
            "b8e78baa Cassini/Mercury set for this Trial-1 branch unless cited by an active paper surface",
            "failed-route / archive-only / exploratory calculation scripts",
        ],
        "reason": "Phase 1 needs only the restored W/Z solver chain and this inventory; broad existence-based staging is prohibited.",
    }


# Function: Build one CSV status row.

def csv_row(gate: str, status: str, evidence: str, note: str) -> dict[str, str]:
    """Return a normalized CSV row."""
    return {"gate": gate, "status": status, "evidence": evidence, "note": note}


# Function: Write compact CSV gate rows for quick inspection.

def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    """Write gate rows to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["gate", "status", "evidence", "note"])
        writer.writeheader()
        writer.writerows(rows)


# Function: Assemble and write the Phase-1 dependency inventory.

def main() -> int:
    """Run the Phase-1 inventory and write JSON/CSV outputs."""
    PUBLIC_OUT.mkdir(parents=True, exist_ok=True)

    script_surfaces = {key: parse_python_surface(path) for key, path in SOURCE_SCRIPTS.items()}
    metric_data = {key: read_json(path) for key, path in PUBLIC_METRICS.items()}
    function_checks, missing_functions = function_gate(script_surfaces)
    metric_checks, failed_metrics = metric_gate(metric_data)
    scale = nucleon_scale_diagnostics()

    missing_scripts = [rel(path) for path in SOURCE_SCRIPTS.values() if not path.exists()]
    missing_metrics = [rel(path) for path in PUBLIC_METRICS.values() if not path.exists()]
    blockers = []
    if missing_scripts:
        blockers.append("Required Phase-1 source scripts are missing.")

    if missing_metrics:
        blockers.append("Required public W/Z metrics are missing.")

    if missing_functions:
        blockers.append("Required reusable solver functions are absent.")

    if failed_metrics:
        blockers.append("Required public metric values failed the retained contract.")

    if not scale["grid_resolves_0p84fm_target"]:
        blockers.append("Current two-component grid does not resolve the 0.84 fm radius target at nucleon scale.")

    warnings = [
        "This inventory does not run the nucleon scan; it freezes the dependency and adapter contract only.",
        "The 0.94 GeV mass and 0.84 fm radius are scoring targets / unit scales, not new P-model free parameters.",
    ]
    overall_status = "ready_for_phase2_scan_implementation" if not blockers else "blocked"
    can_advance = not blockers

    gate_rows = [
        csv_row(
            "1.1_wz_archive_assets",
            "pass" if not missing_metrics and not failed_metrics else "blocked",
            f"missing_metrics={len(missing_metrics)} failed_metric_checks={len(failed_metrics)}",
            "41 localized sectors and 905838 exact vector rows are required before nucleon scaling.",
        ),
        csv_row(
            "1.1_reusable_solver_functions",
            "pass" if not missing_scripts and not missing_functions else "blocked",
            f"missing_scripts={len(missing_scripts)} missing_functions={len(missing_functions)}",
            "Reusable y_beta/profile, interpolation, four-state shooting, and exact-ladder surfaces are present.",
        ),
        csv_row(
            "1.2_minimal_modification_contract",
            "pass",
            "allowed=energy_scale, family_range, spatial_grid prohibited=new_constants,new_terms,target_relaxation",
            "Phase 2 must retarget only scale/range/grid and must not alter the physical model.",
        ),
        csv_row(
            "1.2_radius_grid_resolution",
            "pass" if scale["grid_resolves_0p84fm_target"] else "blocked",
            f"r_max_fm={scale['current_two_component_r_max_fm_at_nucleon_scale']:.6g} step_fm={scale['current_two_component_max_step_fm_at_nucleon_scale']:.6g}",
            "The existing dimensionless grid is sufficient at nucleon scale for an initial 0.84 fm RMS audit.",
        ),
        csv_row(
            "paper_critical_dependency_list",
            "pass",
            "dependency_list_present=true",
            "Only listed paper-critical scripts and outputs may be staged for the next Trial-1 commit.",
        ),
    ]

    report = {
        "generated_utc": utc_now_iso(),
        "phase": {
            "program": "P-model v3.0",
            "trial": "Trial-1 baryon mass ratio and proton radius",
            "step": "Phase 1 nucleon-scale dependency inventory",
        },
        "workflow_source": path_record(WORKFLOW_SOURCE),
        "phase0_inventory": path_record(PHASE0_INVENTORY),
        "push_scope_doc": path_record(PUSH_SCOPE_DOC),
        "source_scripts": script_surfaces,
        "public_metrics": {key: path_record(path) for key, path in PUBLIC_METRICS.items()},
        "function_checks": function_checks,
        "metric_checks": metric_checks,
        "minimal_modification_contract": {
            "allowed_modifications": ALLOWED_MINIMAL_MODIFICATIONS,
            "prohibited_modifications": PROHIBITED_MODIFICATIONS,
            "no_new_independent_constants": True,
            "no_bc_intermediate_phase": True,
            "do_not_relax_targets": True,
        },
        "nucleon_scale_diagnostics": scale,
        "dependency_list": build_dependency_list(),
        "gate": {
            "overall_status": overall_status,
            "can_advance_to_phase2_scan_implementation": can_advance,
            "blockers": blockers,
            "warnings": warnings,
            "next_action": (
                "Implement the Phase-2 broad nucleon-scale scan using only the listed paper-critical solver chain."
                if can_advance
                else "Resolve the missing Phase-1 dependencies before implementing the nucleon scan."
            ),
        },
        "csv_rows": gate_rows,
    }

    json_path = PUBLIC_OUT / "v3_trial1_nucleon_scale_phase1_inventory.json"
    csv_path = PUBLIC_OUT / "v3_trial1_nucleon_scale_phase1_inventory.csv"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(csv_path, gate_rows)

    print(f"[write] {rel(json_path)}")
    print(f"[write] {rel(csv_path)}")
    print(f"[gate] overall_status={overall_status} can_advance_to_phase2_scan_implementation={str(can_advance).lower()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
