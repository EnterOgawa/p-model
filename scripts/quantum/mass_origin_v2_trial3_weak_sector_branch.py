#!/usr/bin/env python3
"""
Generate Trial-3 weak-sector artifacts for 8.7.56.9-.12 and 8.7.56.149.

Trial-1 has already closed on an honest Case-B partial closeout and Trial-2 is
on hold. The next executable v2.0 branch is therefore the recommended
weak-sector challenge: can the already-frozen vector hierarchy and lambda_rot
reuse extend naturally to the W/Z sector?

This branch:

1. inventories the canonical source pack for a weak-sector pilot,
2. extends the adopted exact vector ladder to a first weak-sector search window
   and compares it to M_W / m_e and M_Z / m_e,
3. audits whether the same pilot can recover M_W / M_Z and sin^2(theta_W), and
4. freezes the Trial-3 declaration gate. If the pilot window is insufficient,
   it also freezes the next residual route contract.
"""

from __future__ import annotations

import csv
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output" / "public" / "quantum"
PART1 = ROOT / "doc" / "paper" / "10_part1_core_theory.md"
PART3A = ROOT / "doc" / "paper" / "12_part3a_quantum_foundations.md"
STATUS = ROOT / "doc" / "STATUS.md"
ROADMAP = ROOT / "doc" / "ROADMAP.md"
AI_CONTEXT = ROOT / "doc" / "AI_CONTEXT_MIN.json"

VECTOR_ROUTE = OUT / "mass_origin_vector_qball_route_contract_metrics.json"
VECTOR_SPIN = OUT / "mass_origin_vector_qball_spin_orbit_freeze_audit_metrics.json"
VECTOR_HEAVY = OUT / "mass_origin_vector_qball_baryon_tau_neutron_fit_table_metrics.json"
SCALAR_SPECTRUM = OUT / "mass_origin_qball_discrete_mass_spectrum_metrics.json"

NUMERICAL_BRANCH = ROOT / "scripts" / "quantum" / "mass_origin_vector_qball_numerical_solver_branch.py"
FULL_COUPLED_BRANCH = ROOT / "scripts" / "quantum" / "mass_origin_vector_qball_full_coupled_solver_branch.py"
EXTENDED_BRANCH = ROOT / "scripts" / "quantum" / "mass_origin_vector_qball_extended_hierarchy_branch.py"

ELECTRON_MASS_MEV = 0.51099895
W_MASS_MEV = 80369.0
Z_MASS_MEV = 91187.6
W_TARGET = W_MASS_MEV / ELECTRON_MASS_MEV
Z_TARGET = Z_MASS_MEV / ELECTRON_MASS_MEV
WZ_RATIO_TARGET = W_MASS_MEV / Z_MASS_MEV
SIN2_THETA_W_TARGET = 1.0 - WZ_RATIO_TARGET * WZ_RATIO_TARGET
PASS_THRESHOLD = 0.10
TRIAL3_ELL_MAX = 14
PAIR_TOP_COUNT = 300


# Function: return the current UTC timestamp in ISO 8601 format.
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# Function: abort if a required path is missing.

def req(path: Path) -> None:
    if not path.exists():
        raise SystemExit(f"[fail] missing required input: {path}")


# Function: load a UTF-8 JSON artifact.

def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


# Function: load a UTF-8 text source.

def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# Function: convert an absolute path into a repository-relative path.

def rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


# Function: return the first source line that contains the requested pattern.

def hit(text: str, pattern: str) -> dict | None:
    for line_no, line in enumerate(text.splitlines(), start=1):
        if pattern in line:
            return {"pattern": pattern, "line": line_no, "text": line.strip()}

    return None


# Function: build a standard result row.

def row(row_id: str, status: str, metric: str, value: float, note: str) -> dict:
    return {
        "row_id": row_id,
        "status": status,
        "metric": metric,
        "value": float(value),
        "note": note,
    }


# Function: build a standard payload object.

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


# Function: save a JSON artifact and its row table.

def write_artifact(stem: str, data: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    json_path = OUT / f"{stem}_metrics.json"
    csv_path = OUT / f"{stem}_rows.csv"
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["row_id", "status", "metric", "value", "note"])
        writer.writeheader()
        writer.writerows(data["rows"])


# Function: dynamically load a local Python module.

def load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"[fail] unable to import module: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Function: build a source-inventory record.

def target_record(file_key: str, path: Path, text: str, pattern: str, note: str) -> dict:
    target_hit = hit(text, pattern)
    return {
        "file_key": file_key,
        "file": rel(path),
        "pattern": pattern,
        "present": target_hit is not None,
        "note": note,
        "evidence": target_hit,
    }


# Function: find the closest ladder state to a single weak target.

def closest_state(rows: list[dict], target_value: float) -> dict | None:
    best = None
    for item in rows:
        ratio = float(item["mass_ratio_to_scalar_base"])
        relative_error = abs(ratio - target_value) / target_value
        record = {
            "n": int(item["n"]),
            "k": int(item["k"]),
            "ell": int(item["ell"]),
            "s": int(item["s"]),
            "ratio_value": ratio,
            "relative_error": float(relative_error),
            "passes_threshold": bool(relative_error <= PASS_THRESHOLD),
        }
        if best is None or record["relative_error"] < best["relative_error"]:
            best = record

    return best


# Function: search the high-mass tail for the best W/Z ratio pair.

def best_ratio_pair(rows: list[dict]) -> dict | None:
    candidates = sorted(rows, key=lambda item: float(item["mass_ratio_to_scalar_base"]), reverse=True)[:PAIR_TOP_COUNT]
    best = None
    for index, left in enumerate(candidates):
        left_ratio = float(left["mass_ratio_to_scalar_base"])
        for right in candidates[index + 1 :]:
            right_ratio = float(right["mass_ratio_to_scalar_base"])
            heavier = max(left_ratio, right_ratio)
            lighter = min(left_ratio, right_ratio)
            ratio_value = lighter / heavier
            ratio_error = abs(ratio_value - WZ_RATIO_TARGET) / WZ_RATIO_TARGET
            sin2_value = 1.0 - ratio_value * ratio_value
            sin2_error = abs(sin2_value - SIN2_THETA_W_TARGET) / SIN2_THETA_W_TARGET
            record = {
                "lighter_state": {
                    "n": int(left["n"]) if left_ratio <= right_ratio else int(right["n"]),
                    "k": int(left["k"]) if left_ratio <= right_ratio else int(right["k"]),
                    "ell": int(left["ell"]) if left_ratio <= right_ratio else int(right["ell"]),
                    "s": int(left["s"]) if left_ratio <= right_ratio else int(right["s"]),
                    "mass_ratio_to_electron": float(lighter),
                },
                "heavier_state": {
                    "n": int(left["n"]) if left_ratio > right_ratio else int(right["n"]),
                    "k": int(left["k"]) if left_ratio > right_ratio else int(right["k"]),
                    "ell": int(left["ell"]) if left_ratio > right_ratio else int(right["ell"]),
                    "s": int(left["s"]) if left_ratio > right_ratio else int(right["s"]),
                    "mass_ratio_to_electron": float(heavier),
                },
                "mw_mz_ratio_value": float(ratio_value),
                "mw_mz_ratio_relative_error": float(ratio_error),
                "sin2_theta_w_value": float(sin2_value),
                "sin2_theta_w_relative_error": float(sin2_error),
                "passes_threshold": bool(ratio_error <= PASS_THRESHOLD and sin2_error <= PASS_THRESHOLD),
            }
            key = (record["mw_mz_ratio_relative_error"], record["sin2_theta_w_relative_error"])
            if best is None or key < (
                best["mw_mz_ratio_relative_error"],
                best["sin2_theta_w_relative_error"],
            ):
                best = record

    return best


# Function: execute the Trial-3 weak-sector branch.

def main() -> None:
    for path in (
        PART1,
        PART3A,
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        VECTOR_ROUTE,
        VECTOR_SPIN,
        VECTOR_HEAVY,
        SCALAR_SPECTRUM,
        NUMERICAL_BRANCH,
        FULL_COUPLED_BRANCH,
        EXTENDED_BRANCH,
    ):
        req(path)

    part1_text = read_text(PART1)
    part3a_text = read_text(PART3A)
    status_text = read_text(STATUS)
    ai_context = read_json(AI_CONTEXT)
    vector_route = read_json(VECTOR_ROUTE)
    vector_spin = read_json(VECTOR_SPIN)
    vector_heavy = read_json(VECTOR_HEAVY)
    scalar_spectrum = read_json(SCALAR_SPECTRUM)

    numerical = load_module(NUMERICAL_BRANCH, "trial3_num")
    full = load_module(FULL_COUPLED_BRANCH, "trial3_full")
    extended = load_module(EXTENDED_BRANCH, "trial3_ext")

    scalar_modes = list(scalar_spectrum["evidence"]["discrete_mass_mode_rows"])
    lambda_rot = float(vector_spin["summary"]["lambda_rot_value"])

    common_inputs = {
        "part1_markdown": rel(PART1),
        "part3a_markdown": rel(PART3A),
        "status_markdown": rel(STATUS),
        "roadmap_markdown": rel(ROADMAP),
        "ai_context_json": rel(AI_CONTEXT),
        "mass_origin_vector_qball_route_contract_json": rel(VECTOR_ROUTE),
        "mass_origin_vector_qball_spin_orbit_freeze_audit_json": rel(VECTOR_SPIN),
        "mass_origin_vector_qball_baryon_tau_neutron_fit_table_json": rel(VECTOR_HEAVY),
        "mass_origin_qball_discrete_mass_spectrum_json": rel(SCALAR_SPECTRUM),
    }

    inventory_targets = [
        target_record(
            "part1_lambda_rot_micro_chiral_line",
            PART1,
            part1_text,
            "左手系カイラル流",
            "Part I must still expose the micro chiral-current interpretation of lambda_rot reuse.",
        ),
        target_record(
            "part1_weak_va_line",
            PART1,
            part1_text,
            "V-A 演算子構造",
            "Part I must still state that the vector-P coupling structurally points toward weak-sector V-A behavior.",
        ),
        target_record(
            "part3a_exact_vector_hierarchy_line",
            PART3A,
            part3a_text,
            "mass-origin route で固定した exact vector hierarchy",
            "Part III-A must still expose the exact vector hierarchy as a canonical source pack.",
        ),
        target_record(
            "status_trial3_next_step",
            STATUS,
            status_text,
            "current official next step は `8.7.56.9`",
            "STATUS must already point to Trial-3 as the next executable route.",
        ),
    ]
    inventory_ready = all(item["present"] for item in inventory_targets) and ("8.7.56.9" in ai_context["current_step"])

    inventory = payload(
        "8.7.56.9",
        "W/Z sector source inventory",
        common_inputs,
        "Inventory the canonical source pack that can be reused immediately for a weak-sector pilot based on the exact vector hierarchy and lambda_rot reuse.",
        {
            "target_ratios": {
                "M_W/m_e": W_TARGET,
                "M_Z/m_e": Z_TARGET,
                "M_W/M_Z": WZ_RATIO_TARGET,
                "sin2_theta_W": SIN2_THETA_W_TARGET,
            },
            "pilot_window_rule": f"reuse the adopted exact vector ladder and extend the first weak pilot to k=0, ell<= {TRIAL3_ELL_MAX}",
            "reuse_rule": "reuse the already-frozen lambda_rot and exact vector hierarchy without introducing a new coupling",
        },
        [
            row(
                "trial3_wz_source_inventory_complete",
                "pass",
                "Trial-3 W/Z source inventory complete",
                1,
                "The weak-sector source inventory was executed against the current canon.",
            ),
            row(
                "trial3_wz_required_source_count",
                "pass" if inventory_ready else "reject",
                "required weak-sector source count",
                len(inventory_targets) + (1 if "8.7.56.9" in ai_context["current_step"] else 0),
                "All weak-sector source targets must be present before the pilot window can be frozen.",
            ),
            row(
                "trial3_lambda_rot_reuse_available",
                "pass" if bool(vector_spin["summary"]["lambda_rot_reuse_available"]) else "reject",
                "lambda_rot reuse available for Trial-3",
                1 if bool(vector_spin["summary"]["lambda_rot_reuse_available"]) else 0,
                "Trial-3 reuses the same lambda_rot already frozen by the vector hierarchy route.",
            ),
        ],
        {
            "w_target_ratio": W_TARGET,
            "z_target_ratio": Z_TARGET,
            "weinberg_angle_target": SIN2_THETA_W_TARGET,
            "pilot_ell_window_max": TRIAL3_ELL_MAX,
            "inventory_ready": inventory_ready,
            "first_route_to_close_or_none": None if inventory_ready else "trial3_wz_source_missing",
        },
        {
            "overall_status": "trial3_wz_source_inventory_frozen",
            "trial3_branch_closeable": False,
            "advance_to_8_7_56_10": True,
            "next_required_artifacts": ["trial3_wz_vector_mode_pilot"],
        },
        {
            "inventory_targets": inventory_targets,
            "vector_route_summary": vector_route["summary"],
            "vector_spin_summary": vector_spin["summary"],
            "vector_heavy_summary": vector_heavy["summary"],
        },
    )

    base_modes_by_ell = {}
    ell_mode_counts = []
    for ell in range(1, TRIAL3_ELL_MAX + 1):
        if ell <= 3:
            scan_rows = numerical.scan_ell_sector(ell)
        else:
            scan_rows = extended.scan_extended_sector(numerical, ell)

        modes = numerical.interpolate_integer_modes(scan_rows, ell)
        base_modes_by_ell[ell] = modes
        ell_mode_counts.append({"ell": int(ell), "integer_mode_count": len(modes)})

    exact_rows = full.build_exact_ladder(scalar_modes, base_modes_by_ell, lambda_rot)
    vector_rows = [row_data for row_data in exact_rows if int(row_data["ell"]) > 0]
    best_w = closest_state(vector_rows, W_TARGET)
    best_z = closest_state(vector_rows, Z_TARGET)
    max_ratio = max(float(row_data["mass_ratio_to_scalar_base"]) for row_data in exact_rows)
    total_integer_modes = sum(len(rows) for rows in base_modes_by_ell.values())
    wz_anchor_pass = bool(best_w and best_z and best_w["passes_threshold"] and best_z["passes_threshold"])

    pilot = payload(
        "8.7.56.10",
        "W-boson / Z-boson vector-mode pilot",
        common_inputs,
        "Extend the adopted exact vector ladder to the first weak-sector pilot window and compare it against the W/Z mass-scale targets.",
        {
            "weak_target_rule": "compare the exact vector ladder against M_W/m_e and M_Z/m_e using the electron anchor already frozen in the scalar base mode",
            "pilot_window": f"k=0, ell=1..{TRIAL3_ELL_MAX}, s=-1,0,1 with the adopted exact full-coupled reconstruction",
            "pass_threshold": PASS_THRESHOLD,
        },
        [
            row(
                "trial3_wz_vector_mode_pilot_complete",
                "pass",
                "Trial-3 W/Z vector-mode pilot complete",
                1,
                "The first weak-sector exact-ladder pilot has been rebuilt.",
            ),
            row(
                "trial3_w_anchor_pass",
                "pass" if best_w and best_w["passes_threshold"] else "reject",
                "W/electron anchor passes threshold",
                1 if best_w and best_w["passes_threshold"] else 0,
                "The pilot checks whether the current exact family reaches the W scale within threshold.",
            ),
            row(
                "trial3_z_anchor_pass",
                "pass" if best_z and best_z["passes_threshold"] else "reject",
                "Z/electron anchor passes threshold",
                1 if best_z and best_z["passes_threshold"] else 0,
                "The pilot checks whether the current exact family reaches the Z scale within threshold.",
            ),
            row(
                "trial3_high_mass_scale_gap_present",
                "reject" if wz_anchor_pass else "pass",
                "high-mass weak-sector scale gap present",
                0 if wz_anchor_pass else 1,
                "The current exact family is still far below the weak boson mass scale when the anchors do not pass.",
            ),
        ],
        {
            "pilot_ell_window_max": TRIAL3_ELL_MAX,
            "total_integer_mode_count": total_integer_modes,
            "exact_state_count": len(exact_rows),
            "maximum_mass_ratio_to_electron": max_ratio,
            "best_w_row_or_none": best_w,
            "best_z_row_or_none": best_z,
            "w_scale_gap_factor_or_none": None if best_w is None else W_TARGET / float(best_w["ratio_value"]),
            "z_scale_gap_factor_or_none": None if best_z is None else Z_TARGET / float(best_z["ratio_value"]),
            "wz_anchor_pass": wz_anchor_pass,
        },
        {
            "overall_status": "trial3_wz_vector_mode_pilot_complete",
            "trial3_branch_closeable": False,
            "advance_to_8_7_56_11": True,
            "next_required_artifacts": ["trial3_weinberg_angle_weak_coupling_audit"],
        },
        {
            "ell_mode_counts": ell_mode_counts,
            "best_w_row": best_w,
            "best_z_row": best_z,
        },
    )

    best_pair = best_ratio_pair(vector_rows)
    ratio_pass = bool(best_pair and best_pair["passes_threshold"])

    audit = payload(
        "8.7.56.11",
        "Weinberg-angle / weak-coupling audit",
        common_inputs,
        "Audit whether the first weak-sector pilot can recover the W/Z ratio structure and whether the current canon exposes a first-principles weak-coupling map.",
        {
            "ratio_rule": "use the high-mass tail of the exact vector ladder to search for the closest M_W/M_Z candidate pair",
            "weinberg_rule": "derive sin^2(theta_W) = 1 - (M_W/M_Z)^2 from the same pair",
            "coupling_rule": "a valid weak-coupling route needs more than lambda_rot reuse; it must map a weak coupling constant from existing canon without a new parameter",
            "pass_threshold": PASS_THRESHOLD,
        },
        [
            row(
                "trial3_weinberg_ratio_pass",
                "pass" if ratio_pass else "reject",
                "M_W/M_Z ratio passes threshold",
                1 if ratio_pass else 0,
                "The pilot checks whether the current exact family can reproduce the W/Z ratio structure in the scanned window.",
            ),
            row(
                "trial3_sin2_theta_w_pass",
                "pass" if ratio_pass else "reject",
                "sin^2(theta_W) passes threshold",
                1 if ratio_pass else 0,
                "The same candidate pair is checked against the Weinberg-angle proxy.",
            ),
            row(
                "trial3_weak_coupling_first_principles_route_available",
                "reject",
                "weak-coupling first-principles route available",
                0,
                "The current canon reuses lambda_rot structurally but does not yet map a weak coupling constant without adding a new parameter.",
            ),
        ],
        {
            "mw_mz_ratio_target": WZ_RATIO_TARGET,
            "sin2_theta_w_target": SIN2_THETA_W_TARGET,
            "best_pair_or_none": best_pair,
            "mw_mz_ratio_pass": ratio_pass,
            "sin2_theta_w_pass": ratio_pass,
            "weak_coupling_first_principles_route_available": False,
            "first_route_to_close_or_none": "trial3_declaration_gate",
        },
        {
            "overall_status": "trial3_weinberg_angle_weak_coupling_audited",
            "trial3_branch_closeable": False,
            "advance_to_8_7_56_12": True,
            "next_required_artifacts": ["trial3_declaration_gate"],
        },
        {
            "best_pair": best_pair,
            "part1_weak_va_line": hit(part1_text, "V-A 演算子構造"),
            "part1_micro_chiral_line": hit(part1_text, "左手系カイラル流"),
        },
    )

    trial3_recommended_condition = bool(wz_anchor_pass and ratio_pass)

    declaration = payload(
        "8.7.56.12",
        "Trial-3 declaration gate / v2.0 recommended-condition audit",
        common_inputs,
        "Freeze whether the current weak-sector pilot is strong enough to count as a recommended-condition success for v2.0.",
        {
            "recommended_condition_rule": "Trial-3 recommended condition requires W and Z anchor recovery plus an admissible M_W/M_Z / sin^2(theta_W) structure in the scanned exact family",
            "failure_rule": "If the current pilot window cannot reach the weak mass scale, Trial-3 remains open and moves to a high-mass extension residual route",
        },
        [
            row(
                "trial3_recommended_condition_satisfied",
                "pass" if trial3_recommended_condition else "reject",
                "Trial-3 recommended condition satisfied",
                1 if trial3_recommended_condition else 0,
                "The recommended-condition gate opens only if the weak-sector anchors and ratio structure both close.",
            ),
            row(
                "trial3_current_window_reaches_weak_mass_scale",
                "pass" if wz_anchor_pass else "reject",
                "current weak-sector pilot window reaches weak mass scale",
                1 if wz_anchor_pass else 0,
                "The current k=0, ell<=14 pilot must actually reach the W/Z scale to close Trial-3.",
            ),
            row(
                "trial3_residual_route_required",
                "reject" if trial3_recommended_condition else "pass",
                "Trial-3 residual route required",
                0 if trial3_recommended_condition else 1,
                "A residual route is required when the current pilot window does not close the weak-sector challenge.",
            ),
        ],
        {
            "trial3_recommended_condition_satisfied": trial3_recommended_condition,
            "wz_anchor_pass": wz_anchor_pass,
            "mw_mz_ratio_pass": ratio_pass,
            "advance_to_8_7_56_13": False,
            "recommended_next_route_or_none": "8.7.56.149" if not trial3_recommended_condition else "8.7.56.13",
        },
        {
            "overall_status": "trial3_declaration_gate_frozen",
            "trial3_branch_closeable": trial3_recommended_condition,
            "advance_to_8_7_56_13": False,
            "next_required_artifacts": [] if trial3_recommended_condition else ["trial3_high_mass_scale_gap_route_contract"],
        },
        {
            "pilot_summary": pilot["summary"],
            "audit_summary": audit["summary"],
            "vector_heavy_summary": vector_heavy["summary"],
        },
    )

    route_contract = payload(
        "8.7.56.149",
        "Trial-3 high-mass scale-gap residual route contract",
        common_inputs,
        "Freeze the next official residual route after the first weak-sector pilot fails to reach the W/Z mass scale inside the current exact window.",
        {
            "selected_residual_route": "trial3_high_mass_weak_sector_scale_extension",
            "missing_v2_artifact": "weak_sector_high_mass_exact_family_extension",
            "pivot_principle": "The current exact vector family already supplies heavy lepton and baryon anchors, but the weak boson mass scale remains structurally out of range in the present pilot window.",
        },
        [
            row(
                "trial3_high_mass_scale_gap_route_contract_complete",
                "pass",
                "Trial-3 high-mass scale-gap route contract complete",
                1,
                "The next residual route is frozen after the weak-sector pilot window failed to close.",
            ),
            row(
                "trial3_high_mass_scale_gap_requires_extension",
                "pass",
                "weak-sector high-mass scale extension required",
                1,
                "The present ladder window must be extended before W/Z targets can be re-audited honestly.",
            ),
            row(
                "trial3_trial4_handoff_ready_now",
                "reject",
                "Trial-4 handoff ready now",
                0,
                "Trial-4 remains deferred while the Trial-3 residual route is still open.",
            ),
        ],
        {
            "selected_residual_route": "trial3_high_mass_weak_sector_scale_extension",
            "missing_v2_artifact": "weak_sector_high_mass_exact_family_extension",
            "split_contract_ready": True,
            "trial3_recommended_condition_satisfied": trial3_recommended_condition,
            "recommended_next_route_or_none": "8.7.56.150",
        },
        {
            "overall_status": "trial3_high_mass_scale_gap_route_contract_frozen",
            "trial3_branch_closeable": False,
            "advance_to_8_7_56_13": False,
            "next_required_artifacts": [
                "trial3_high_mass_scale_extension_inventory",
                "trial3_high_mass_scale_extension_admissibility_audit",
            ],
        },
        {
            "declaration_summary": declaration["summary"],
            "pilot_summary": pilot["summary"],
            "audit_summary": audit["summary"],
        },
    )

    write_artifact("mass_origin_v2_trial3_wz_sector_source_inventory", inventory)
    write_artifact("mass_origin_v2_trial3_wz_vector_mode_pilot", pilot)
    write_artifact("mass_origin_v2_trial3_weinberg_angle_weak_coupling_audit", audit)
    write_artifact("mass_origin_v2_trial3_declaration_gate", declaration)
    write_artifact("mass_origin_v2_trial3_high_mass_scale_gap_route_contract", route_contract)

    print("[ok] wrote:")
    print(" - mass_origin_v2_trial3_wz_sector_source_inventory_metrics.json")
    print(" - mass_origin_v2_trial3_wz_vector_mode_pilot_metrics.json")
    print(" - mass_origin_v2_trial3_weinberg_angle_weak_coupling_audit_metrics.json")
    print(" - mass_origin_v2_trial3_declaration_gate_metrics.json")
    print(" - mass_origin_v2_trial3_high_mass_scale_gap_route_contract_metrics.json")


# Function: run the Trial-3 weak-sector branch from the command line.

if __name__ == "__main__":
    main()
