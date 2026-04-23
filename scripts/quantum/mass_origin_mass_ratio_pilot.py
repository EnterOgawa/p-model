#!/usr/bin/env python3
"""
Generate mass-ratio pilot artifacts for 8.7.55.2.84 and the follow-up
extended-hierarchy route contract for 8.7.55.2.831.

The script reuses the adopted exact vector-Q-ball ladder from the current
full-coupled solver branch, evaluates known-particle mass-ratio anchors, and
decides whether the second route may advance to 8.7.55.3. When only the
leptonic anchor is recovered, the script freezes a new vector-Q-ball
extended-hierarchy branch instead of advancing immediately.
"""

from __future__ import annotations

import csv
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output" / "public" / "quantum"
STATUS = ROOT / "doc" / "STATUS.md"
ROADMAP = ROOT / "doc" / "ROADMAP.md"
PART1 = ROOT / "doc" / "paper" / "10_part1_core_theory.md"
VECTOR_HOLD = OUT / "mass_origin_vector_qball_branch_refresh_after_exact_solver_metrics.json"
VECTOR_EXACT = OUT / "mass_origin_vector_qball_exact_mass_table_handoff_retry_metrics.json"
VECTOR_SPIN = OUT / "mass_origin_vector_qball_spin_orbit_freeze_audit_metrics.json"
SCALAR_SPECTRUM = OUT / "mass_origin_qball_discrete_mass_spectrum_metrics.json"
NUMERICAL_BRANCH = ROOT / "scripts" / "quantum" / "mass_origin_vector_qball_numerical_solver_branch.py"
FULL_COUPLED_BRANCH = ROOT / "scripts" / "quantum" / "mass_origin_vector_qball_full_coupled_solver_branch.py"


# Function: Return the current UTC timestamp in ISO 8601 format.
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# Function: Abort immediately if a required artifact is missing.

def req(path: Path) -> None:
    if not path.exists():
        raise SystemExit(f"[fail] missing required input: {path}")


# Function: Read a UTF-8 JSON artifact into a dictionary.

def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


# Function: Read a UTF-8 text file.

def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# Function: Convert an absolute path to a repo-relative string.

def rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


# Function: Return the first line that contains the requested pattern.

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


# Function: Save a JSON artifact and the paired CSV table.

def write_artifact(stem: str, data: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    json_path = OUT / f"{stem}_metrics.json"
    csv_path = OUT / f"{stem}_rows.csv"
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["row_id", "status", "metric", "value", "note"])
        writer.writeheader()
        writer.writerows(data["rows"])


# Function: Dynamically load a local Python module.

def load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"[fail] unable to import module: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Function: Build the exact vector ladder by reusing the adopted full-coupled branch helpers.

def rebuild_exact_ladder() -> tuple[list[dict], list[dict], dict, list[dict]]:
    prev = load_module(NUMERICAL_BRANCH, "vector_qball_effective")
    full = load_module(FULL_COUPLED_BRANCH, "vector_qball_full")
    scalar_modes = read_json(SCALAR_SPECTRUM)["evidence"]["discrete_mass_mode_rows"]
    lambda_rot = float(read_json(VECTOR_SPIN)["summary"]["lambda_rot_value"])
    ell_scan_rows = {ell: prev.scan_ell_sector(ell) for ell in (1, 2, 3)}
    base_modes_by_ell = {ell: prev.interpolate_integer_modes(rows, ell) for ell, rows in ell_scan_rows.items()}
    exact_rows = full.build_exact_ladder(scalar_modes, base_modes_by_ell, lambda_rot)
    comparisons, closest = full.compare_known_targets(exact_rows)
    return exact_rows, comparisons, closest, scalar_modes


# Function: Return the best candidate for each named target.

def best_rows_by_target(comparisons: list[dict]) -> dict[str, dict]:
    labels = ["m_mu/m_e", "m_p/m_e", "m_tau/m_e", "m_n/m_p"]
    best_rows: dict[str, dict] = {}
    for label in labels:
        rows = [record for record in comparisons if record["target_label"] == label]
        best_rows[label] = min(rows, key=lambda item: float(item["relative_error"]))

    return best_rows


# Function: Keep a readable sample from a long row table.

def sample(rows: list[dict], count: int = 20) -> list[dict]:
    if len(rows) <= count:
        return rows

    step = max(1, len(rows) // count)
    sampled = [rows[index] for index in range(0, len(rows), step)]
    return sampled[:count]


# Function: Freeze the second-route decision from the per-target anchor pattern.

def second_route_decision(best_by_target: dict[str, dict]) -> tuple[str, bool]:
    mu_pass = bool(best_by_target["m_mu/m_e"]["passes_threshold"])
    proton_pass = bool(best_by_target["m_p/m_e"]["passes_threshold"])
    tau_pass = bool(best_by_target["m_tau/m_e"]["passes_threshold"])
    neutron_pass = bool(best_by_target["m_n/m_p"]["passes_threshold"])

    if mu_pass and proton_pass and (tau_pass or neutron_pass):
        return "advance_to_8_7_55_3_after_vector_mass_ratio_pilot", True

    if mu_pass:
        return "continue_vector_qball_extended_hierarchy_without_8_7_55_3", False

    return "hold_mass_origin_branch_vector_qball_ratio_insufficient", False


# Function: Run the .84 mass-ratio pilot and the .831 residual route contract.

def main() -> None:
    for path in (STATUS, ROADMAP, PART1, VECTOR_HOLD, VECTOR_EXACT, VECTOR_SPIN, SCALAR_SPECTRUM, NUMERICAL_BRANCH, FULL_COUPLED_BRANCH):
        req(path)

    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    part1_text = read_text(PART1)
    vector_hold = read_json(VECTOR_HOLD)
    vector_exact = read_json(VECTOR_EXACT)

    exact_rows, comparisons, closest, scalar_modes = rebuild_exact_ladder()
    best_by_target = best_rows_by_target(comparisons)
    threshold_rows = [record for record in comparisons if record["passes_threshold"]]
    decision_label, advance_to_8_7_55_3 = second_route_decision(best_by_target)

    discrete_spectrum_found = bool(vector_exact["summary"]["exact_full_coupled_vector_ladder_available"])
    proton_rel = float(best_by_target["m_p/m_e"]["relative_error"])
    tau_rel = float(best_by_target["m_tau/m_e"]["relative_error"])
    neutron_rel = float(best_by_target["m_n/m_p"]["relative_error"])
    muon_rel = float(best_by_target["m_mu/m_e"]["relative_error"])
    extended_route = "vector_qball_extended_hierarchy_reopen"
    missing_artifact = "high_ell_or_high_k_vector_qball_mass_hierarchy_table"

    payloads = {
        "mass_origin_mass_ratio_pilot": payload(
            "8.7.55.2.84",
            "Mass-ratio comparison and second-route decision",
            {
                "mass_origin_vector_qball_exact_mass_table_handoff_retry_json": rel(VECTOR_EXACT),
                "mass_origin_vector_qball_branch_refresh_after_exact_solver_json": rel(VECTOR_HOLD),
                "mass_origin_qball_discrete_mass_spectrum_json": rel(SCALAR_SPECTRUM),
            },
            "Compare the adopted exact vector-Q-ball ladder against known-particle mass-ratio anchors and decide whether the second route may advance to 8.7.55.3.",
            {
                "reference_state": "M_(1,0,0,0)",
                "targets": {
                    "m_mu/m_e": 206.768283,
                    "m_p/m_e": 1836.15267343,
                    "m_tau/m_e": 3477.48,
                    "m_n/m_p": 1.00137841925,
                },
                "advance_rule": "advance requires a leptonic anchor plus at least one heavy-hierarchy anchor; otherwise continue the mass-origin branch without 8.7.55.3",
            },
            [
                row(
                    "mass_origin_mass_ratio_pilot_complete",
                    "pass",
                    "mass-origin mass-ratio pilot complete",
                    1,
                    "The exact vector ladder has been compared against known-particle mass-ratio anchors.",
                ),
                row(
                    "mass_origin_discrete_spectrum_found",
                    "pass" if discrete_spectrum_found else "reject",
                    "discrete spectrum found",
                    1 if discrete_spectrum_found else 0,
                    "The adopted exact vector ladder exists and is eligible for mass-ratio comparison." if discrete_spectrum_found else "No exact vector ladder is available for comparison.",
                ),
                row(
                    "mass_origin_muon_electron_anchor_pass",
                    "pass" if best_by_target["m_mu/m_e"]["passes_threshold"] else "reject",
                    "muon/electron anchor passes threshold",
                    1 if best_by_target["m_mu/m_e"]["passes_threshold"] else 0,
                    "The best exact vector candidate for m_mu/m_e is inside the 10% threshold." if best_by_target["m_mu/m_e"]["passes_threshold"] else "No exact vector candidate matches m_mu/m_e within threshold.",
                ),
                row(
                    "mass_origin_proton_electron_anchor_pass",
                    "pass" if best_by_target["m_p/m_e"]["passes_threshold"] else "reject",
                    "proton/electron anchor passes threshold",
                    1 if best_by_target["m_p/m_e"]["passes_threshold"] else 0,
                    "A proton/electron-scale anchor exists within threshold." if best_by_target["m_p/m_e"]["passes_threshold"] else "No proton/electron-scale anchor is yet recovered within threshold.",
                ),
                row(
                    "advance_to_8_7_55_3",
                    "pass" if advance_to_8_7_55_3 else "reject",
                    "advance to 8.7.55.3",
                    1 if advance_to_8_7_55_3 else 0,
                    "The second route may advance to 8.7.55.3." if advance_to_8_7_55_3 else "The second route continues inside 8.7.55.2 and does not yet advance to 8.7.55.3.",
                ),
            ],
            {
                "discrete_spectrum_found": discrete_spectrum_found,
                "candidate_ratio_count": len(threshold_rows),
                "all_comparison_row_count": len(comparisons),
                "closest_known_mass_ratio_or_none": closest,
                "muon_electron_ratio_relative_error": muon_rel,
                "proton_electron_ratio_relative_error": proton_rel,
                "tau_electron_ratio_relative_error": tau_rel,
                "neutron_proton_ratio_relative_error": neutron_rel,
                "second_route_decision": decision_label,
                "advance_to_8_7_55_3": advance_to_8_7_55_3,
            },
            {
                "overall_status": "mass_origin_mass_ratio_pilot_advance_ready" if advance_to_8_7_55_3 else "mass_origin_mass_ratio_pilot_partial_vector_success",
                "keep_mass_origin_branch_blocked": not advance_to_8_7_55_3,
                "hand_off_to_8_7_55_2_84": True,
                "advance_to_8_7_55_3": advance_to_8_7_55_3,
                "next_required_artifacts": [] if advance_to_8_7_55_3 else [extended_route],
            },
            {
                "candidate_ratio_rows": threshold_rows[:64],
                "per_target_best_rows": best_by_target,
                "closest_match_row": closest,
                "vector_exact_handoff_summary": vector_exact["summary"],
                "vector_branch_refresh_summary": vector_hold["summary"],
                "roadmap_step_line": hit(roadmap_text, "`8.7.55.2.84`"),
                "status_next_line": hit(status_text, "次の公式 step は `8.7.55.2.84`"),
            },
        ),
        "mass_origin_vector_qball_extended_hierarchy_route_contract": payload(
            "8.7.55.2.831",
            "Vector Q-ball extended hierarchy route contract",
            {
                "mass_origin_mass_ratio_pilot_json": "output/public/quantum/mass_origin_mass_ratio_pilot_metrics.json",
                "mass_origin_vector_qball_exact_mass_table_handoff_retry_json": rel(VECTOR_EXACT),
            },
            "Freeze the next residual route after the exact vector ladder recovers the mu/e anchor but still misses the proton/tau/neutron hierarchy anchors.",
            {
                "selected_residual_route": extended_route,
                "missing_artifact": missing_artifact,
                "extension_principle": "reuse the same exact vector solver and extend the hierarchy search in ell and k rather than adding a new coupling or abandoning the vector route",
            },
            [
                row(
                    "vector_qball_extended_hierarchy_route_contract_complete",
                    "pass",
                    "vector Q-ball extended hierarchy route contract complete",
                    1,
                    "The next residual hierarchy branch is frozen.",
                ),
                row(
                    "vector_qball_muon_anchor_recovered",
                    "pass" if best_by_target["m_mu/m_e"]["passes_threshold"] else "reject",
                    "muon/electron anchor recovered",
                    1 if best_by_target["m_mu/m_e"]["passes_threshold"] else 0,
                    "The route keeps the recovered mu/e anchor as a proven checkpoint.",
                ),
                row(
                    "vector_qball_heavy_hierarchy_anchor_missing",
                    "reject" if not (best_by_target["m_p/m_e"]["passes_threshold"] or best_by_target["m_tau/m_e"]["passes_threshold"]) else "pass",
                    "heavy hierarchy anchor recovered",
                    1 if (best_by_target["m_p/m_e"]["passes_threshold"] or best_by_target["m_tau/m_e"]["passes_threshold"]) else 0,
                    "Proton/tau-scale anchors remain unresolved and force an extended-hierarchy branch.",
                ),
            ],
            {
                "selected_residual_route": extended_route,
                "missing_vector_qball_artifact": missing_artifact,
                "proven_anchor_targets": [label for label, row_data in best_by_target.items() if row_data["passes_threshold"]],
                "unresolved_anchor_targets": [label for label, row_data in best_by_target.items() if not row_data["passes_threshold"]],
                "split_contract_ready": True,
            },
            {
                "overall_status": "vector_qball_extended_hierarchy_route_contract_frozen",
                "keep_mass_origin_branch_blocked": True,
                "advance_to_8_7_55_3": False,
                "new_branch_required": True,
                "next_required_artifacts": [
                    "vector_qball_extended_hierarchy_sector_inventory",
                    "vector_qball_high_ell_pilot",
                    "vector_qball_k_node_pilot",
                ],
            },
            {
                "closest_match_row": closest,
                "per_target_best_rows": best_by_target,
                "part1_vector_line": hit(part1_text, "P_\\mu=(P_t,P_1,P_2,P_3)"),
            },
        ),
    }

    for stem, data in payloads.items():
        write_artifact(stem, data)
        print(f"[ok] wrote {OUT / (stem + '_metrics.json')}")
        print(f"[ok] wrote {OUT / (stem + '_rows.csv')}")


# Function: Run the mass-ratio pilot when invoked as a script.

if __name__ == "__main__":
    main()
